# This file is part of Lisien, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import inspect
import sys
from collections import OrderedDict
from dataclasses import KW_ONLY, dataclass, field
from functools import cached_property, partial, partialmethod
from queue import Queue
from threading import Thread
from typing import Iterator, Union, get_args

from sqlalchemy import (
	BLOB,
	BOOLEAN,
	FLOAT,
	INT,
	TEXT,
	Column,
	ColumnElement,
	MetaData,
	Select,
	Table,
	and_,
	bindparam,
	create_engine,
	func,
	null,
	or_,
	select,
)
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.sql.ddl import CreateTable

from . import types
from .db import (
	SCHEMA_VERSION_B,
	SCHEMAVER_B,
	Batch,
	ConnectionLooper,
	GlobalKeyValueStore,
	ThreadedDatabaseConnector,
	mutexed,
)
from .exc import KeyframeError
from .types import (
	Branch,
	BranchRowType,
	CharacterRulesHandledRowType,
	CharName,
	EdgeKeyframe,
	EdgeRowType,
	EdgeValRowType,
	GraphValRowType,
	Key,
	KeyframeExtensionRowType,
	KeyframeGraphRowType,
	NodeKeyframe,
	NodeName,
	NodeRowType,
	NodeRulesHandledRowType,
	NodeValRowType,
	PortalRulesHandledRowType,
	RulebookName,
	RuleName,
	StatDict,
	Tick,
	Time,
	Turn,
	UnitRulesHandledRowType,
	Value,
	root_type,
)

meta = MetaData()
py2sql: dict[type, type] = {
	bytes: BLOB,
	Key: BLOB,
	Value: BLOB,
	int: INT,
	str: TEXT,
	bool: BOOLEAN,
	float: FLOAT,
}
for table, serializer in Batch.serializers.items():
	cached_prop: cached_property = Batch.cached_properties[table]
	batch = cached_prop.func(None)
	spec = inspect.getfullargspec(serializer)
	ret_annot = spec.annotations["return"]
	if isinstance(ret_annot, str):
		ret_annot = eval(ret_annot, types.__dict__)
	if hasattr(ret_annot, "evaluate_value"):
		ret_annot = ret_annot.evaluate_value()
	if hasattr(ret_annot, "__value__"):
		ret_annot = ret_annot.__value__
	columns = []
	with_rowid = batch.key_len == 0
	for n, (arg, ret_typ) in enumerate(
		zip(spec.args[1:], get_args(ret_annot)), start=1
	):
		if hasattr(ret_typ, "evaluate_value"):
			ret_typ = ret_typ.evaluate_value()
		args = get_args(ret_typ)
		nullable = type(None) in args
		orig = root_type(ret_typ)
		if orig is Union:
			for orig in args:
				if orig is not None:
					break
			else:
				raise TypeError(
					"Too many types for column", table, arg, orig, table
				)
		orig2 = root_type(orig)
		if isinstance(orig2, tuple):
			nullable = type(None) in orig2
			for orig3 in orig2:
				if orig3 is not None:
					break
			else:
				raise TypeError("No actual type for column", table, arg, orig2)
			orig2 = orig3
		if orig2 not in py2sql:
			raise TypeError("Unknown type for column", table, arg, orig)
		col = Column(
			arg,
			py2sql[orig2],
			primary_key=n <= batch.key_len,
			nullable=nullable,
		)
		columns.append(col)
	Table(table, meta, *columns, sqlite_with_rowid=with_rowid)


@dataclass
class SQLAlchemyDatabaseConnector(ThreadedDatabaseConnector):
	connect_string: str = "sqlite:///:memory:"
	connect_args: dict[str, str] = field(default_factory=dict)
	_: KW_ONLY
	clear: bool = False

	@dataclass
	class Looper(ConnectionLooper):
		connector: SQLAlchemyDatabaseConnector

		@cached_property
		def dbstring(self) -> str:
			return self.connector.connect_string

		@cached_property
		def connect_args(self) -> dict[str, str]:
			return self.connector.connect_args

		@cached_property
		def inq(self) -> Queue:
			return self.connector._inq

		@cached_property
		def outq(self) -> Queue:
			return self.connector._outq

		def __post_init__(self):
			self.existence_lock.acquire(timeout=1)

		def commit(self):
			self.transaction.commit()
			self.transaction = self.connection.begin()

		def init_table(self, tbl):
			return self.call("create_{}".format(tbl))

		def call(self, k, *largs, **kwargs):
			from sqlalchemy import CursorResult

			statement = self.sql[k].compile(dialect=self.engine.dialect)
			if hasattr(statement, "positiontup"):
				kwargs.update(dict(zip(statement.positiontup, largs)))
				repositioned = [
					kwargs[param] for param in statement.positiontup
				]
				self.logger.debug(
					f"SQLAlchemyConnectionHolder: calling {k}; {statement}  %  {repositioned}"
				)
				ret: CursorResult = self.connection.execute(statement, kwargs)
				self.logger.debug(
					f"SQLAlchemyConnectionHolder: {k} got {ret.rowcount} rows"
				)
				return ret
			elif largs:
				raise TypeError("{} is a DDL query, I think".format(k))
			self.logger.debug(
				f"SQLAlchemyConnectionHolder: calling {k}; {statement}"
			)
			ret: CursorResult = self.connection.execute(self.sql[k], kwargs)
			self.logger.debug(
				f"SQLAlchemyConnectionHolder: {k} got {ret.rowcount} rows"
			)
			return ret

		def call_many(self, k, largs):
			statement = self.sql[k].compile(dialect=self.engine.dialect)
			aargs = []
			for larg in largs:
				if isinstance(larg, dict):
					aargs.append(larg)
				else:
					aargs.append(dict(zip(statement.positiontup, larg)))
			return self.connection.execute(
				statement,
				aargs,
			)

		@cached_property
		def sql(self) -> dict[str, Select]:
			def update_where(updcols, wherecols):
				"""Return an ``UPDATE`` statement that updates the columns ``updcols``
				when the ``wherecols`` match. Every column has a bound parameter of
				the same name.

				updcols are strings, wherecols are column objects

				"""
				vmap = OrderedDict()
				for col in updcols:
					vmap[col] = bindparam(col)
				wheres = [c == bindparam(c.name) for c in wherecols]
				tab = wherecols[0].table
				return tab.update().values(**vmap).where(and_(*wheres))

			def tick_to_end_clause(tab: Table) -> ColumnElement[bool]:
				return and_(
					tab.c.branch == bindparam("branch"),
					or_(
						tab.c.turn > bindparam("turn_from"),
						and_(
							tab.c.turn == bindparam("turn_from"),
							tab.c.tick >= bindparam("tick_from"),
						),
					),
				)

			def tick_to_tick_clause(tab: Table) -> ColumnElement[bool]:
				return and_(
					tick_to_end_clause(tab),
					or_(
						tab.c.turn < bindparam("turn_to"),
						and_(
							tab.c.turn == bindparam("turn_to"),
							tab.c.tick <= bindparam("tick_to"),
						),
					),
				)

			def load_something_tick_to_end(tab: Table) -> Select:
				time_cols = [tab.c.branch, tab.c.turn, tab.c.tick]
				other_stuff = [
					col for col in tab.primary_key.c if col not in time_cols
				]
				order_cols = time_cols + other_stuff
				return (
					select(*(col for col in tab.c if col is not tab.c.branch))
					.select_from(tab)
					.where(tick_to_end_clause(tab))
					.order_by(*order_cols)
				)

			def load_something_tick_to_tick(tab: Table) -> Select:
				time_cols = [tab.c.branch, tab.c.turn, tab.c.tick]
				other_stuff = [
					col for col in tab.primary_key.c if col not in time_cols
				]
				order_cols = time_cols + other_stuff
				return (
					select(*(col for col in tab.c if col is not tab.c.branch))
					.select_from(tab)
					.where(tick_to_tick_clause(tab))
					.order_by(*order_cols)
				)

			def after_clause(tab: Table) -> list[ColumnElement[bool]]:
				return [
					tab.c.branch == bindparam("branch"),
					or_(
						tab.c.turn > bindparam("turn"),
						and_(
							tab.c.turn == bindparam("turn"),
							tab.c.tick >= bindparam("tick"),
						),
					),
				]

			table = meta.tables

			graphs = table["graphs"]
			globtab = table["global"]
			univtab = table["universals"]
			edge_val = table["edge_val"]
			edges = table["edges"]
			nodes = table["nodes"]
			node_val = table["node_val"]
			graph_val = table["graph_val"]
			branches = table["branches"]
			turns = table["turns"]
			keyframes_graphs = table["keyframes_graphs"]
			keyframes = table["keyframes"]
			r = {
				"universal_get": select(univtab.c.value)
				.where(
					and_(
						univtab.c.key == bindparam("key"),
						univtab.c.branch == bindparam("branch"),
						or_(
							univtab.c.turn < bindparam("turn"),
							and_(
								univtab.c.turn == bindparam("turn"),
								univtab.c.tick <= bindparam("tick"),
							),
						),
					)
				)
				.order_by(univtab.c.turn, univtab.c.tick),
				"global_get": select(globtab.c.value).where(
					globtab.c.key == bindparam("key")
				),
				"global_update": globtab.update()
				.values(value=bindparam("value"))
				.where(globtab.c.key == bindparam("key")),
				"graph_type": select(graphs.c.type).where(
					graphs.c.graph == bindparam("graph")
				),
				"del_edge_val_after": edge_val.delete().where(
					and_(
						edge_val.c.graph == bindparam("graph"),
						edge_val.c.orig == bindparam("orig"),
						edge_val.c.dest == bindparam("dest"),
						edge_val.c.key == bindparam("key"),
						*after_clause(edge_val),
					)
				),
				"del_edges_graph": edges.delete().where(
					edges.c.graph == bindparam("graph")
				),
				"del_edges_after": edges.delete().where(
					and_(
						edges.c.graph == bindparam("graph"),
						edges.c.orig == bindparam("orig"),
						edges.c.dest == bindparam("dest"),
						*after_clause(edges),
					)
				),
				"del_nodes_after": nodes.delete().where(
					and_(
						nodes.c.graph == bindparam("graph"),
						nodes.c.node == bindparam("node"),
						*after_clause(nodes),
					)
				),
				"del_node_val_after": node_val.delete().where(
					and_(
						node_val.c.graph == bindparam("graph"),
						node_val.c.node == bindparam("node"),
						node_val.c.key == bindparam("key"),
						*after_clause(node_val),
					)
				),
				"del_graph_val_after": graph_val.delete().where(
					and_(
						graph_val.c.graph == bindparam("graph"),
						graph_val.c.key == bindparam("key"),
						*after_clause(graph_val),
					)
				),
				"global_delete": globtab.delete().where(
					globtab.c.key == bindparam("key")
				),
				"graphs_types": select(graphs.c.graph, graphs.c.type),
				"graphs_delete": graphs.delete().where(
					and_(
						graphs.c.graph == bindparam("graph"),
						graphs.c.branch == bindparam("branch"),
						graphs.c.turn == bindparam("turn"),
						graphs.c.tick == bindparam("tick"),
					)
				),
				"graphs_named": select(func.COUNT())
				.select_from(graphs)
				.where(graphs.c.graph == bindparam("graph")),
				"graphs_between": select(
					graphs.c.graph,
					graphs.c.turn,
					graphs.c.tick,
					graphs.c.type,
				).where(
					and_(
						graphs.c.branch == bindparam("branch"),
						or_(
							graphs.c.turn > bindparam("turn_from_a"),
							and_(
								graphs.c.turn == bindparam("turn_from_b"),
								graphs.c.tick >= bindparam("tick_from"),
							),
						),
						or_(
							graphs.c.turn < bindparam("turn_to_a"),
							and_(
								graphs.c.turn == bindparam("turn_to_b"),
								graphs.c.tick <= bindparam("tick_to"),
							),
						),
					)
				),
				"graphs_after": select(
					graphs.c.graph,
					graphs.c.turn,
					graphs.c.tick,
					graphs.c.type,
				).where(*after_clause(graphs)),
				"main_branch_ends": select(
					branches.c.branch,
					branches.c.end_turn,
					branches.c.end_tick,
				).where(branches.c.parent == null()),
				"update_branches": branches.update()
				.values(
					parent=bindparam("parent"),
					parent_turn=bindparam("parent_turn"),
					parent_tick=bindparam("parent_tick"),
					end_turn=bindparam("end_turn"),
					end_tick=bindparam("end_tick"),
				)
				.where(branches.c.branch == bindparam("branch")),
				"update_turns": turns.update()
				.values(
					end_tick=bindparam("end_tick"),
					plan_end_tick=bindparam("plan_end_tick"),
				)
				.where(
					and_(
						turns.c.branch == bindparam("branch"),
						turns.c.turn == bindparam("turn"),
					)
				),
				"keyframes_graphs_list": select(
					keyframes_graphs.c.graph,
					keyframes_graphs.c.branch,
					keyframes_graphs.c.turn,
					keyframes_graphs.c.tick,
				),
				"all_graphs_in_keyframe": select(
					keyframes_graphs.c.graph,
					keyframes_graphs.c.nodes,
					keyframes_graphs.c.edges,
					keyframes_graphs.c.graph_val,
				)
				.where(
					and_(
						keyframes_graphs.c.branch == bindparam("branch"),
						keyframes_graphs.c.turn == bindparam("turn"),
						keyframes_graphs.c.tick == bindparam("tick"),
					)
				)
				.order_by(keyframes_graphs.c.graph),
				"get_keyframe_graph": select(
					keyframes_graphs.c.nodes,
					keyframes_graphs.c.edges,
					keyframes_graphs.c.graph_val,
				).where(
					and_(
						keyframes_graphs.c.graph == bindparam("graph"),
						keyframes_graphs.c.branch == bindparam("branch"),
						keyframes_graphs.c.turn == bindparam("turn"),
						keyframes_graphs.c.tick == bindparam("tick"),
					)
				),
				"delete_keyframe": keyframes.delete().where(
					and_(
						keyframes.c.branch == bindparam("branch"),
						keyframes.c.turn == bindparam("turn"),
						keyframes.c.tick == bindparam("tick"),
					)
				),
				"delete_keyframe_graph": keyframes_graphs.delete().where(
					and_(
						keyframes_graphs.c.graph == bindparam("graph"),
						keyframes_graphs.c.branch == bindparam("branch"),
						keyframes_graphs.c.turn == bindparam("turn"),
						keyframes_graphs.c.tick == bindparam("tick"),
					)
				),
			}

			rulebooks = table["rulebooks"]
			r["rulebooks_update"] = update_where(
				["rules"],
				[
					rulebooks.c.rulebook,
					rulebooks.c.branch,
					rulebooks.c.turn,
					rulebooks.c.tick,
				],
			)

			for t in table.values():
				key = ord_key = list(t.primary_key.c)
				if not key:
					key = ord_key = list(t.c)
					assert all(isinstance(k, Column) for k in key)
				r["create_" + t.name] = CreateTable(t)
				r["truncate_" + t.name] = t.delete()
				r[t.name + "_del"] = t.delete().where(
					and_(*[c == bindparam(c.name) for c in key])
				)
				if (
					"branch" in t.columns
					and "turn" in t.columns
					and "tick" in t.columns
				):
					branch = t.columns["branch"]
					turn = t.columns["turn"]
					tick = t.columns["tick"]
					ord_key = [branch, turn, tick] + [
						c for c in t.c if c not in (branch, turn, tick)
					]
					r[t.name + "_del_time"] = t.delete().where(
						and_(
							t.c.branch == bindparam("branch"),
							t.c.turn == bindparam("turn"),
							t.c.tick == bindparam("tick"),
						)
					)
					if branch is t.c[0] and turn is t.c[1]:
						r[f"load_{t.name}_tick_to_end"] = (
							load_something_tick_to_end(t)
						)
						r[f"load_{t.name}_tick_to_tick"] = (
							load_something_tick_to_tick(t)
						)
				r[t.name + "_dump"] = select(*t.c.values()).order_by(*ord_key)
				r[t.name + "_insert"] = t.insert().values(
					tuple(bindparam(cname) for cname in t.c.keys())
				)
				r[t.name + "_count"] = select(func.COUNT("*")).select_from(t)
			things = table["things"]
			r["del_things_after"] = things.delete().where(
				and_(
					things.c.character == bindparam("character"),
					things.c.thing == bindparam("thing"),
					things.c.branch == bindparam("branch"),
					or_(
						things.c.turn > bindparam("turn"),
						and_(
							things.c.turn == bindparam("turn"),
							things.c.tick >= bindparam("tick"),
						),
					),
				)
			)
			units = table["units"]
			r["del_units_after"] = units.delete().where(
				and_(
					units.c.character_graph == bindparam("character"),
					units.c.unit_graph == bindparam("graph"),
					units.c.unit_node == bindparam("unit"),
					units.c.branch == bindparam("branch"),
					or_(
						units.c.turn > bindparam("turn"),
						and_(
							units.c.turn == bindparam("turn"),
							units.c.tick >= bindparam("tick"),
						),
					),
				)
			)
			bookmarks = table["bookmarks"]
			r["update_bookmark"] = (
				bookmarks.update()
				.where(bookmarks.c.key == bindparam("key"))
				.values(
					branch=bindparam("branch"),
					turn=bindparam("turn"),
					tick=bindparam("tick"),
				)
			)
			r["delete_bookmark"] = bookmarks.delete().where(
				bookmarks.c.key == bindparam("key")
			)

			for name in (
				"character_rulebook",
				"unit_rulebook",
				"character_thing_rulebook",
				"character_place_rulebook",
				"character_portal_rulebook",
			):
				tab = table[name]
				r[f"{name}_delete"] = tab.delete().where(
					and_(
						tab.c.character == bindparam("character"),
						tab.c.branch == bindparam("branch"),
						tab.c.turn == bindparam("turn"),
						tab.c.tick == bindparam("tick"),
					)
				)

			def rule_update_cond(t: Table) -> and_:
				return and_(
					t.c.rule == bindparam("rule"),
					t.c.branch == bindparam("branch"),
					t.c.turn == bindparam("turn"),
					t.c.tick == bindparam("tick"),
				)

			hood = table["rule_neighborhood"]
			r["rule_neighborhood_update"] = (
				hood.update()
				.where(rule_update_cond(hood))
				.values(neighborhood=bindparam("neighborhood"))
			)
			big = table["rule_big"]
			r["rule_big_update"] = (
				big.update()
				.where(rule_update_cond(big))
				.values(big=bindparam("big"))
			)
			trig = table["rule_triggers"]
			r["rule_triggers_update"] = (
				trig.update()
				.where(rule_update_cond(trig))
				.values(triggers=bindparam("triggers"))
			)
			preq = table["rule_prereqs"]
			r["rule_prereqs_update"] = (
				preq.update()
				.where(rule_update_cond(preq))
				.values(prereqs=bindparam("prereqs"))
			)
			act = table["rule_actions"]
			r["rule_actions_update"] = (
				act.update()
				.where(rule_update_cond(act))
				.values(actions=bindparam("actions"))
			)
			kf = keyframes

			def time_clause(tab):
				return and_(
					tab.c.branch == bindparam("branch"),
					tab.c.turn == bindparam("turn"),
					tab.c.tick == bindparam("tick"),
				)

			r["delete_from_keyframes"] = kf.delete().where(time_clause(kf))
			kfg = keyframes_graphs
			r["delete_from_keyframes_graphs"] = kfg.delete().where(
				time_clause(kfg)
			)
			kfx = table["keyframe_extensions"]
			r["delete_from_keyframe_extensions"] = kfx.delete().where(
				time_clause(kfx)
			)
			r["get_keyframe_extensions"] = select(
				kfx.c.universal,
				kfx.c.rule,
				kfx.c.rulebook,
			).where(time_clause(kfx))

			for handledtab in (
				"character_rules_handled",
				"unit_rules_handled",
				"character_thing_rules_handled",
				"character_place_rules_handled",
				"character_portal_rules_handled",
				"node_rules_handled",
				"portal_rules_handled",
			):
				ht = table[handledtab]
				r["del_{}_turn".format(handledtab)] = ht.delete().where(
					and_(
						ht.c.branch == bindparam("branch"),
						ht.c.turn == bindparam("turn"),
					)
				)

			branches = branches

			r["branch_children"] = select(branches.c.branch).where(
				branches.c.parent == bindparam("branch")
			)

			tc = table["turns_completed"]
			r["turns_completed_update"] = update_where(["turn"], [tc.c.branch])

			return r

		def run(self):
			dbstring = self.dbstring
			connect_args = self.connect_args
			self.logger.debug("about to connect " + dbstring)
			self.engine = create_engine(dbstring, connect_args=connect_args)
			self.connection = self.engine.connect()
			self.transaction = self.connection.begin()
			self.logger.debug("transaction started")
			while True:
				inst = self.inq.get()
				if inst == "shutdown":
					self.transaction.close()
					self.connection.close()
					self.engine.dispose()
					self.existence_lock.release()
					self.inq.task_done()
					return
				if inst == "commit":
					self.commit()
					self.inq.task_done()
					continue
				if inst == "initdb":
					self.outq.put(self.initdb())
					self.inq.task_done()
					continue
				silent = False
				if inst[0] == "silent":
					inst = inst[1:]
					silent = True
				self.logger.debug(inst[:2])

				def _call_n(mth, cmd, *args, silent=False, **kwargs):
					try:
						res = mth(cmd, *args, **kwargs)
						if silent:
							return ...
						else:
							if (
								hasattr(res, "returns_rows")
								and res.returns_rows
							):
								return list(res)
							return None
					except Exception as ex:
						self.logger.error(repr(ex))
						if silent:
							print(
								f"Got exception while silenced: {repr(ex)}",
								file=sys.stderr,
							)
							sys.exit(repr(ex))
						return ex

				call_one = partial(_call_n, self.call)
				call_many = partial(_call_n, self.call_many)
				call_select = partial(_call_n, self.connection.execute)
				match inst:
					case ("echo", msg):
						self.outq.put(msg)
						self.inq.task_done()
					case ("echo", msg, _):
						self.outq.put(msg)
						self.inq.task_done()
					case ("select", qry, args):
						o = call_select(qry, args, silent=silent)
						if not silent:
							self.outq.put(o)
						self.inq.task_done()
					case ("one", cmd, args, kwargs):
						o = call_one(cmd, *args, silent=silent, **kwargs)
						if not silent:
							self.outq.put(o)
						self.inq.task_done()
					case ("many", cmd, several):
						o = call_many(cmd, several, silent=silent)
						if not silent:
							self.outq.put(o)
						self.inq.task_done()

		def initdb(self) -> dict[bytes, bytes] | Exception:
			"""Set up the database schema, both for allegedb and the special
			extensions for lisien

			"""
			for table in meta.tables:
				try:
					self.init_table(table)
				except OperationalError as ex:
					msg = ex.args[0]
					if not (
						msg.startswith("(sqlite3.OperationalError) table ")
						and msg.endswith(" already exists")
					):
						raise
				except Exception as ex:
					return ex
			glob_d: dict[bytes, bytes] = dict(
				self.call("global_dump").fetchall()
			)
			if SCHEMAVER_B not in glob_d:
				self.call("global_insert", SCHEMAVER_B, SCHEMA_VERSION_B)
				glob_d[SCHEMAVER_B] = SCHEMA_VERSION_B
			elif glob_d[SCHEMAVER_B] != SCHEMA_VERSION_B:
				return ValueError(
					"Unsupported database schema version", glob_d[SCHEMAVER_B]
				)
			return glob_d

		def close(self):
			self.transaction.close()
			self.connection.close()

	def __post_init__(self):
		self._t = Thread(target=self._looper.run)
		self._t.start()
		if self.clear:
			self.truncate_all()

	@mutexed
	def call(self, string, *args, **kwargs):
		self._inq.put(("one", string, args, kwargs))
		ret = self._outq.get()
		self._outq.task_done()
		if isinstance(ret, Exception):
			raise ret
		return ret

	def call_silent(self, string, *args, **kwargs):
		self._inq.put(("one", string, args, kwargs))

	def call_many(self, string, args):
		with self.mutex():
			self._inq.put(("many", string, args))
			ret = self._outq.get()
			self._outq.task_done()
		if isinstance(ret, Exception):
			raise ret
		return ret

	def call_many_silent(self, string, args):
		self._inq.put(("silent", "many", string, args))

	def delete_many_silent(self, table, args):
		self.call_many_silent(table + "_del", args)

	@mutexed
	def insert_many(self, table_name: str, args: list[dict]):
		with self.mutex():
			self._inq.put(("many", table_name + "_insert", args))
			ret = self._outq.get()
			self._outq.task_done()
		if isinstance(ret, Exception):
			raise ret
		return ret

	def insert_many_silent(self, table_name: str, args: list[dict]) -> None:
		self._inq.put(("silent", "many", table_name + "_insert", args))

	def execute(self, stmt, *args):
		if not isinstance(stmt, Select):
			raise TypeError("Only select statements should be executed")
		self.flush()
		with self.mutex():
			self._inq.put(("select", stmt, args))
			ret = self._outq.get()
			self._outq.task_done()
			return ret

	def universal_get(
		self, key: Key, branch: Branch, turn: Turn, tick: Tick
	) -> Value:
		for (result,) in self.call(
			"universal_get", self.pack(key), branch, turn, turn, tick
		):
			ret = self.unpack(result)
			if ret is ...:
				raise KeyError(
					"No value for that universal key now",
					key,
					branch,
					turn,
					tick,
				)
			return ret
		raise KeyError(
			"No value for that universal key ever", key, branch, turn, tick
		)

	def bookmarks_dump(self) -> Iterator[tuple[Key, Time]]:
		self.flush()
		unpack = self.unpack
		for key, branch, turn, tick in self.call("bookmarks_dump"):
			yield unpack(key), (branch, turn, tick)

	def keyframes_dump(self) -> Iterator[Time]:
		self.flush()
		return self.call("keyframes_dump")

	def keyframes_graphs(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick]]:
		self._new_keyframes_graphs()
		unpack = self.unpack
		for graph, branch, turn, tick in self.call("keyframes_graphs_list"):
			yield unpack(graph), branch, turn, tick

	def get_all_keyframe_graphs(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> Iterator[tuple[CharName, NodeKeyframe, EdgeKeyframe, StatDict]]:
		if (branch, turn, tick) not in self._all_keyframe_times:
			raise KeyframeError(branch, turn, tick)
		unpack_key = self.unpack_key
		for graph, nodes, edges, graph_val in self.call(
			"all_graphs_in_keyframe", branch, turn, tick
		):
			yield (
				CharName(unpack_key(graph)),
				self._unpack_node_keyframe(nodes),
				self._unpack_edge_keyframe(edges),
				self._unpack_graph_val_keyframe(graph_val),
			)

	def keyframes_graphs_dump(
		self,
	) -> Iterator[KeyframeGraphRowType]:
		self.flush()
		unpack_key = self.unpack_key
		for (
			branch,
			turn,
			tick,
			graph,
			nodes,
			edges,
			graph_val,
		) in self.call("keyframes_graphs_dump"):
			yield (
				branch,
				turn,
				tick,
				CharName(unpack_key(graph)),
				self._unpack_node_keyframe(nodes),
				self._unpack_edge_keyframe(edges),
				self._unpack_graph_val_keyframe(graph_val),
			)

	def keyframe_extensions_dump(
		self,
	) -> Iterator[KeyframeExtensionRowType]:
		self.flush()
		for branch, turn, tick, universal, rule, rulebook in self.call(
			"keyframe_extensions_dump"
		):
			yield (
				branch,
				turn,
				tick,
				self._unpack_universal_keyframe(universal),
				self._unpack_rules_keyframe(rule),
				self._unpack_rulebooks_keyframe(rulebook),
			)

	def delete_keyframe(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		def keyframe_filter(tup: tuple):
			_, kfbranch, kfturn, kftick, __, ___, ____ = tup
			return (kfbranch, kfturn, kftick) != (branch, turn, tick)

		def keyframe_extension_filter(tup: tuple):
			kfbranch, kfturn, kftick, _, __, ___ = tup
			return (kfbranch, kfturn, kftick) != (branch, turn, tick)

		new_keyframes = list(filter(keyframe_filter, self._new_keyframes))
		self._new_keyframes.clear()
		self._new_keyframes.extend(new_keyframes)
		self._new_keyframe_times.discard((branch, turn, tick))
		new_keyframe_extensions = self._new_keyframe_extensions.copy()
		self._new_keyframe_extensions.clear()
		self._new_keyframe_extensions.extend(
			filter(keyframe_extension_filter, new_keyframe_extensions)
		)
		with self._looper.lock:
			self._inq.put(
				(
					"silent",
					"one",
					"delete_from_keyframes",
					(branch, turn, tick),
					{},
				)
			)
			self._inq.put(
				(
					"silent",
					"one",
					"delete_from_keyframes_graphs",
					(branch, turn, tick),
					{},
				)
			)
			self._inq.put(
				(
					"silent",
					"one",
					"delete_from_keyframe_extensions",
					(branch, turn, tick),
					{},
				)
			)
			self._inq.put(("echo", "done deleting keyframe"))
			if (got := self._outq.get()) != "done deleting keyframe":
				raise RuntimeError("Didn't delete keyframe right", got)
			self._outq.task_done()

	def have_branch(self, branch):
		"""Return whether the branch thus named exists in the database."""
		return bool(self.call("ctbranch", branch)[0][0])

	def branches_dump(
		self,
	) -> Iterator[BranchRowType]:
		"""Return all the branch data in tuples of (branch, parent,
		start_turn, start_tick, end_turn, end_tick).

		"""
		self.flush()
		return self.call("branches_dump")

	def global_get(self, key: Key) -> Value:
		"""Return the value for the given key in the ``globals`` table."""
		key = self.pack(key)
		r = self.call("global_get", key)[0]
		if r is None:
			raise KeyError("Not set")
		return self.unpack(r[0])

	def global_dump(self) -> Iterator[tuple[Key, Value]]:
		"""Iterate over (key, value) pairs in the ``globals`` table."""
		self.flush()
		unpack = self.unpack
		dumped = self.call("global_dump")
		for k, v in dumped:
			yield (unpack(k), unpack(v))

	def get_branch(self) -> Branch:
		v = self.call("global_get", self.pack("branch"))[0]
		if v is None:
			return self.eternal["trunk"]
		return self.unpack(v[0])

	def get_turn(self) -> Turn:
		v = self.call("global_get", self.pack("turn"))[0]
		if v is None:
			return Turn(0)
		return self.unpack(v[0])

	def get_tick(self) -> Tick:
		v = self.call("global_get", self.pack("tick"))[0]
		if v is None:
			return Tick(0)
		return self.unpack(v[0])

	def turns_dump(self) -> Iterator[TurnRowType]:
		self._turns2set()
		return self.call("turns_dump")

	def graph_val_dump(self) -> Iterator[GraphValRowType]:
		"""Yield the entire contents of the graph_val table."""
		self._graphvals2set()
		unpack = self.unpack
		for branch, turn, tick, graph, key, value in self.call(
			"graph_val_dump"
		):
			yield (
				unpack(graph),
				unpack(key),
				branch,
				turn,
				tick,
				unpack(value),
			)

	def graph_val_del_time(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		super().graph_val_del_time(branch, turn, tick)
		self.call("graph_val_del_time", branch, turn, tick)

	def graphs_types(
		self,
		branch,
		turn_from,
		tick_from,
		turn_to=None,
		tick_to=None,
	):
		unpack = self.unpack
		if turn_to is None:
			if tick_to is not None:
				raise ValueError("Need both or neither of turn_to and tick_to")
			for graph, turn, tick, typ in self.call(
				"graphs_after", branch, turn_from, turn_from, tick_from
			):
				yield unpack(graph), branch, turn, tick, typ
			return
		else:
			if tick_to is None:
				raise ValueError("Need both or neither of turn_to and tick_to")
		for graph, turn, tick, typ in self.call(
			"graphs_between",
			branch,
			turn_from,
			turn_from,
			tick_from,
			turn_to,
			turn_to,
			tick_to,
		):
			yield unpack(graph), branch, turn, tick, typ

	def graphs_dump(self):
		self.flush()
		unpack = self.unpack
		for branch, turn, tick, graph, typ in self.call("graphs_dump"):
			yield unpack(graph), branch, turn, tick, typ

	def nodes_del_time(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		super().nodes_del_time(branch, turn, tick)
		self.call("nodes_del_time", branch, turn, tick)

	def nodes_dump(self) -> Iterator[NodeRowType]:
		"""Dump the entire contents of the nodes table."""
		self._nodes2set()
		unpack = self.unpack
		for branch, turn, tick, graph, node, extant in self.call("nodes_dump"):
			yield (
				unpack(graph),
				unpack(node),
				branch,
				turn,
				tick,
				bool(extant),
			)

	def _iter_nodes(
		self, graph, branch, turn_from, tick_from, turn_to=None, tick_to=None
	) -> Iterator[NodeRowType]:
		if (turn_to is None) ^ (tick_to is None):
			raise TypeError("I need both or neither of turn_to and tick_to")
		self._nodes2set()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.call(
				"load_nodes_tick_to_end",
				pack(graph),
				branch,
				turn_from,
				turn_from,
				tick_from,
			)
		else:
			it = self.call(
				"load_nodes_tick_to_tick",
				pack(graph),
				branch,
				turn_from,
				turn_from,
				tick_from,
				turn_to,
				turn_to,
				tick_to,
			)
		for node, turn, tick, extant in it:
			yield graph, unpack(node), branch, turn, tick, extant

	def node_val_dump(self) -> Iterator[NodeValRowType]:
		"""Yield the entire contents of the node_val table."""
		self._nodevals2set()
		unpack = self.unpack
		for branch, turn, tick, graph, node, key, value in self.call(
			"node_val_dump"
		):
			yield (
				unpack(graph),
				unpack(node),
				unpack(key),
				branch,
				turn,
				tick,
				unpack(value),
			)

	def _iter_node_val(
		self, graph, branch, turn_from, tick_from, turn_to=None, tick_to=None
	) -> Iterator[NodeValRowType]:
		if (turn_to is None) ^ (tick_to is None):
			raise TypeError("I need both or neither of turn_to and tick_to")
		self._nodevals2set()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.call(
				"load_node_val_tick_to_end",
				pack(graph),
				branch,
				turn_from,
				turn_from,
				tick_from,
			)
		else:
			it = self.call(
				"load_node_val_tick_to_tick",
				pack(graph),
				branch,
				turn_from,
				turn_from,
				tick_from,
				turn_to,
				turn_to,
				tick_to,
			)
		for node, key, turn, tick, value in it:
			yield (
				graph,
				unpack(node),
				unpack(key),
				branch,
				turn,
				tick,
				unpack(value),
			)

	def node_val_del_time(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		super().node_val_del_time(branch, turn, tick)
		self.call("node_val_del_time", branch, turn, tick)

	def edges_dump(self) -> Iterator[EdgeRowType]:
		"""Dump the entire contents of the edges table."""
		self._edges2set()
		unpack = self.unpack
		for (
			graph,
			orig,
			dest,
			branch,
			turn,
			tick,
			extant,
		) in self.call("edges_dump"):
			yield (
				branch,
				turn,
				tick,
				unpack(graph),
				unpack(orig),
				unpack(dest),
				bool(extant),
			)

	def edges_del_time(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		super().edges_del_time(branch, turn, tick)
		self.call("edges_del_time", branch, turn, tick)

	def edge_val_dump(self) -> Iterator[EdgeValRowType]:
		"""Yield the entire contents of the edge_val table."""
		self._edgevals2set()
		unpack = self.unpack
		for (
			branch,
			turn,
			tick,
			graph,
			orig,
			dest,
			key,
			value,
		) in self.call("edge_val_dump"):
			yield (
				unpack(graph),
				unpack(orig),
				unpack(dest),
				unpack(key),
				branch,
				turn,
				tick,
				unpack(value),
			)

	def _iter_edge_val(
		self, graph, branch, turn_from, tick_from, turn_to=None, tick_to=None
	) -> Iterator[EdgeValRowType]:
		if (turn_to is None) ^ (tick_to is None):
			raise TypeError("I need both or neither of turn_to and tick_to")
		self._edgevals2set()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.call(
				"load_edge_val_tick_to_end",
				pack(graph),
				branch,
				turn_from,
				turn_from,
				tick_from,
			)
		else:
			it = self.call(
				"load_edge_val_tick_to_tick",
				pack(graph),
				branch,
				turn_from,
				turn_from,
				tick_from,
				turn_to,
				turn_to,
				tick_to,
			)
		for orig, dest, key, turn, tick, value in it:
			yield (
				branch,
				turn,
				tick,
				graph,
				unpack(orig),
				unpack(dest),
				unpack(key),
				unpack(value),
			)

	def edge_val_del_time(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		super().edge_val_del_time(branch, turn, tick)
		self.call("edge_val_del_time", branch, turn, tick)

	def plan_ticks_dump(self):
		self._planticks2set()
		for rec in (data := self.call("plan_ticks_dump")):
			yield rec

	def commit(self):
		"""Commit the transaction"""
		self.flush()
		self._inq.put("commit")
		self._inq.join()
		if (got := self.echo("committed")) != "committed":
			raise RuntimeError("Failed commit", got)

	def close(self):
		"""Commit the transaction, then close the connection"""
		self._inq.put("shutdown")
		self._looper.existence_lock.acquire()
		self._looper.existence_lock.release()
		self._t.join()

	def _init_db(self) -> dict:
		if hasattr(self, "_initialized"):
			raise RuntimeError("Tried to initialize database twice")
		self._initialized = True
		with self.mutex():
			self._inq.put("initdb")
			got = self._outq.get()
			if isinstance(got, Exception):
				raise got
			elif not isinstance(got, dict):
				raise TypeError("initdb didn't return a dictionary", got)
			globals = {
				self.unpack(k): self.unpack(v) for (k, v) in got.items()
			}
			self._outq.task_done()
			if isinstance(globals, Exception):
				raise globals
			self._inq.put(("one", "keyframes_dump", (), {}))
			x = self._outq.get()
			self._outq.task_done()
			if isinstance(x, Exception):
				raise x
		if "trunk" not in globals:
			self._eternal2set.append(("trunk", "trunk"))
			globals["trunk"] = "trunk"
		if "branch" not in globals:
			self._eternal2set.append(("branch", "trunk"))
			globals["branch"] = "trunk"
		if "turn" not in globals:
			self._eternal2set.append(("turn", 0))
			globals["turn"] = 0
		if "tick" not in globals:
			self._eternal2set.append(("tick", 0))
			globals["tick"] = 0
		self.eternal = GlobalKeyValueStore(self, globals)
		self.all_rules.clear()
		self.all_rules.update(self.rules_dump())
		self._all_keyframe_times.clear()
		self._all_keyframe_times.update(x)
		return globals

	def truncate_all(self):
		"""Delete all data from every table"""
		for table in meta.tables.keys():
			try:
				self.call("truncate_" + table)
			except OperationalError:
				pass  # table wasn't created yet
		self.commit()

	def get_keyframe_extensions(self, branch: Branch, turn: Turn, tick: Tick):
		if (branch, turn, tick) not in self._all_keyframe_times:
			raise KeyframeError(branch, turn, tick)
		self.flush()
		unpack = self.unpack
		exts = self.call("get_keyframe_extensions", branch, turn, tick)
		if not exts:
			raise KeyframeError(branch, turn, tick)
		assert len(exts) == 1, f"Incoherent keyframe {branch, turn, tick}"
		universal, rule, rulebook = exts[0]
		return (
			unpack(universal),
			unpack(rule),
			unpack(rulebook),
		)

	def universals_dump(self):
		self.flush()
		unpack = self.unpack
		for branch, turn, tick, key, value in self.call("universals_dump"):
			yield unpack(key), branch, turn, tick, unpack(value)

	def rulebooks_dump(self):
		self.flush()
		unpack = self.unpack
		for branch, turn, tick, rulebook, rules, prio in self.call(
			"rulebooks_dump"
		):
			yield unpack(rulebook), branch, turn, tick, (unpack(rules), prio)

	def _rule_dump(self, typ):
		self.flush()
		unpack = self.unpack
		for branch, turn, tick, rule, lst in self.call(
			"rule_{}_dump".format(typ)
		):
			yield rule, branch, turn, tick, unpack(lst)

	def rule_triggers_dump(self):
		return self._rule_dump("triggers")

	def rule_prereqs_dump(self):
		return self._rule_dump("prereqs")

	def rule_actions_dump(self):
		return self._rule_dump("actions")

	def rule_neighborhood_dump(self):
		self.flush()
		return self.call("rule_neighborhood_dump")

	def rule_big_dump(self):
		self.flush()
		return self.call("rule_big_dump")

	def node_rulebook_dump(self):
		self.flush()
		unpack = self.unpack
		for branch, turn, tick, character, node, rulebook in self.call(
			"node_rulebook_dump"
		):
			yield (
				unpack(character),
				unpack(node),
				branch,
				turn,
				tick,
				unpack(rulebook),
			)

	def portal_rulebook_dump(self):
		self.flush()
		unpack = self.unpack
		for (
			branch,
			turn,
			tick,
			character,
			orig,
			dest,
			rulebook,
		) in self.call("portal_rulebook_dump"):
			yield (
				unpack(character),
				unpack(orig),
				unpack(dest),
				branch,
				turn,
				tick,
				unpack(rulebook),
			)

	def _charactery_rulebook_dump(self, qry):
		self.flush()
		unpack = self.unpack
		for branch, turn, tick, character, rulebook in self.call(
			qry + "_rulebook_dump"
		):
			yield unpack(character), branch, turn, tick, unpack(rulebook)

	character_rulebook_dump = partialmethod(
		_charactery_rulebook_dump, "character"
	)
	unit_rulebook_dump = partialmethod(_charactery_rulebook_dump, "unit")
	character_thing_rulebook_dump = partialmethod(
		_charactery_rulebook_dump, "character_thing"
	)
	character_place_rulebook_dump = partialmethod(
		_charactery_rulebook_dump, "character_place"
	)
	character_portal_rulebook_dump = partialmethod(
		_charactery_rulebook_dump, "character_portal"
	)

	def character_rules_handled_dump(
		self,
	) -> Iterator[CharacterRulesHandledRowType]:
		self.flush()
		unpack_key = self.unpack_key
		for branch, turn, character, rulebook, rule, tick in self.call(
			"character_rules_handled_dump"
		):
			yield (
				Branch(branch),
				Turn(turn),
				CharName(unpack_key(character)),
				RulebookName(unpack_key(rulebook)),
				RuleName(rule),
				Tick(tick),
			)

	def unit_rules_handled_dump(self) -> Iterator[UnitRulesHandledRowType]:
		self._unit_rules_handled_to_set()
		unpack_key = self.unpack_key
		for (
			branch,
			turn,
			character,
			graph,
			unit,
			rulebook,
			rule,
			tick,
		) in self.call("unit_rules_handled_dump"):
			yield (
				Branch(branch),
				Turn(turn),
				CharName(unpack_key(character)),
				CharName(unpack_key(graph)),
				NodeName(unpack_key(unit)),
				RulebookName(unpack_key(rulebook)),
				RuleName(rule),
				Tick(tick),
			)

	def character_thing_rules_handled_dump(
		self,
	) -> Iterator[NodeRulesHandledRowType]:
		self.flush()
		unpack_key = self.unpack_key
		for (
			branch,
			turn,
			character,
			thing,
			rulebook,
			rule,
			tick,
		) in self.call("character_thing_rules_handled_dump"):
			yield (
				Branch(branch),
				Turn(turn),
				CharName(unpack_key(character)),
				NodeName(unpack_key(thing)),
				RulebookName(unpack_key(rulebook)),
				RuleName(rule),
				Tick(tick),
			)

	def character_place_rules_handled_dump(
		self,
	) -> Iterator[NodeRulesHandledRowType]:
		self.flush()
		unpack_key = self.unpack_key
		for (
			branch,
			turn,
			character,
			place,
			rulebook,
			rule,
			tick,
		) in self.call("character_place_rules_handled_dump"):
			yield (
				Branch(branch),
				Turn(turn),
				CharName(unpack_key(character)),
				NodeName(unpack_key(place)),
				RulebookName(unpack_key(rulebook)),
				RuleName(rule),
				Tick(tick),
			)

	def character_portal_rules_handled_dump(
		self,
	) -> Iterator[PortalRulesHandledRowType]:
		self.flush()
		unpack_key = self.unpack_key
		for (
			branch,
			turn,
			character,
			orig,
			dest,
			rulebook,
			rule,
			tick,
		) in self.call("character_portal_rules_handled_dump"):
			yield (
				Branch(branch),
				Turn(turn),
				CharName(unpack_key(character)),
				NodeName(unpack_key(orig)),
				NodeName(unpack_key(dest)),
				RulebookName(unpack_key(rulebook)),
				RuleName(rule),
				Tick(tick),
			)

	def node_rules_handled_dump(self) -> Iterator[NodeRulesHandledRowType]:
		self.flush()
		unpack_key = self.unpack_key
		for (
			branch,
			turn,
			character,
			node,
			rulebook,
			rule,
			tick,
		) in self.call("node_rules_handled_dump"):
			yield (
				Branch(branch),
				Turn(turn),
				CharName(unpack_key(character)),
				NodeName(unpack_key(node)),
				RulebookName(unpack_key(rulebook)),
				RuleName(rule),
				Tick(tick),
			)

	def portal_rules_handled_dump(self) -> Iterator[PortalRulesHandledRowType]:
		self.flush()
		unpack_key = self.unpack_key
		for (
			branch,
			turn,
			character,
			orig,
			dest,
			rulebook,
			rule,
			tick,
		) in self.call("portal_rules_handled_dump"):
			yield (
				Branch(branch),
				Turn(turn),
				CharName(unpack_key(character)),
				NodeName(unpack_key(orig)),
				NodeName(unpack_key(dest)),
				RulebookName(unpack_key(rulebook)),
				RuleName(rule),
				Tick(tick),
			)

	def things_dump(self) -> Iterator[ThingRowType]:
		self.flush()
		unpack_key = self.unpack_key
		for branch, turn, tick, character, thing, location in self.call(
			"things_dump"
		):
			yield (
				Branch(branch),
				Turn(turn),
				Tick(tick),
				CharName(unpack_key(character)),
				NodeName(unpack_key(thing)),
				NodeName(unpack_key(location)),
			)

	def units_dump(
		self,
	) -> Iterator[UnitRowType]:
		self.flush()
		unpack_key = self.unpack_key
		for (
			branch,
			turn,
			tick,
			character_graph,
			unit_graph,
			unit_node,
			is_unit,
		) in self.call("units_dump"):
			yield (
				Branch(branch),
				Turn(turn),
				Tick(tick),
				CharName(unpack_key(character_graph)),
				CharName(unpack_key(unit_graph)),
				NodeName(unpack_key(unit_node)),
				is_unit,
			)

	def count_all_table(self, tbl):
		return self.call("{}_count".format(tbl)).fetchone()[0]

	def rules_dump(self):
		self.flush()
		for (name,) in self.call("rules_dump"):
			yield name

	def things_del_time(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		super().things_del_time(branch, turn, tick)
		self.call("things_del_time", branch, turn, tick)

	def rulebook_set(
		self,
		rulebook: RulebookName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rules: list[RuleName],
	) -> None:
		# what if the rulebook has other values set afterward? wipe them out, right?
		# should that happen in the query engine or elsewhere?
		rulebook, rules = map(self.pack, (rulebook, rules))
		try:
			self.call("rulebooks_insert", rulebook, branch, turn, tick, rules)
			self._increc()
		except IntegrityError:
			try:
				self.call(
					"rulebooks_update", rules, rulebook, branch, turn, tick
				)
			except IntegrityError:
				self.commit()
				self.call(
					"rulebooks_update", rules, rulebook, branch, turn, tick
				)

	def turns_completed_dump(self) -> Iterator[tuple[Branch, Turn]]:
		self._turns_completed_to_set()
		return self.call("turns_completed_dump")

	def rules_insert(self, rule: RuleName):
		self.call("rules_insert", rule)

	def del_bookmark(self, key: Key) -> None:
		self._bookmarks2set.cull(lambda keey, _: key == keey)
		self.call("bookmarks_del", key)
