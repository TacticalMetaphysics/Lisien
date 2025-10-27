from __future__ import annotations

import sys
from dataclasses import dataclass, field, KW_ONLY
from functools import cached_property, partial, partialmethod
from queue import Queue
from threading import Thread
from typing import ClassVar, Iterator

from sqlalchemy import create_engine, Select
from sqlalchemy.exc import IntegrityError, OperationalError

from .alchemy import meta, queries
from .db import (
	ThreadedDatabaseConnector,
	ConnectionLooper,
	SCHEMAVER_B,
	SCHEMA_VERSION_B,
	mutexed,
	GlobalKeyValueStore,
)
from .exc import KeyframeError, ExceptionGroup
from .types import (
	Key,
	Time,
	Branch,
	Turn,
	Tick,
	CharName,
	NodeKeyframe,
	EdgeKeyframe,
	GraphValKeyframe,
	CharDict,
	UniversalKeyframe,
	RuleKeyframe,
	RulebookKeyframe,
	Value,
	GraphValRowType,
	NodeRowType,
	NodeValRowType,
	EdgeRowType,
	EdgeValRowType,
	NodeName,
	RulebookName,
	RuleName,
)


@dataclass
class SQLAlchemyDatabaseConnector(ThreadedDatabaseConnector):
	connect_string: str = "sqlite:///:memory:"
	connect_args: dict[str, str] = field(default_factory=dict)
	_: KW_ONLY
	clear: bool = False

	@dataclass
	class Looper(ConnectionLooper):
		connector: SQLAlchemyDatabaseConnector
		tables: ClassVar[set[str]] = meta.tables.keys()

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

		def run(self):
			dbstring = self.dbstring
			connect_args = self.connect_args
			self.logger.debug("about to connect " + dbstring)
			self.engine = create_engine(dbstring, connect_args=connect_args)
			self.sql = queries(meta)
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
			for table in self.tables:
				try:
					self.init_table(table)
				except OperationalError:
					pass
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
		if self._outq.unfinished_tasks != 0:
			excs = []
			unfinished_tasks = self._outq.unfinished_tasks
			while not self._outq.empty():
				got = self._outq.get()
				if isinstance(got, Exception):
					excs.append(got)
				else:
					excs.append(ValueError("Unconsumed output", got))
			if excs:
				if len(excs) == 1:
					raise excs[-1]
				raise ExceptionGroup(
					f"{unfinished_tasks} unfinished tasks in output queue "
					"before call_one",
					excs,
				)
			else:
				raise RuntimeError(
					f"{unfinished_tasks} unfinished tasks in output queue "
					"before call_one"
				)
		self._inq.put(("one", string, args, kwargs))
		ret = self._outq.get()
		self._outq.task_done()
		if self._outq.unfinished_tasks != 0:
			raise RuntimeError(
				f"{self._outq.unfinished_tasks} unfinished tasks in output "
				"queue after call_one",
			)
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

	def bookmarks_dump(self) -> Iterator[tuple[Key, Time]]:
		self.flush()
		unpack = self.unpack
		for key, branch, turn, tick in self.call("bookmarks_dump"):
			yield unpack(key), (branch, turn, tick)

	def keyframes_dump(self) -> Iterator[tuple[Branch, Turn, Tick]]:
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
	) -> Iterator[
		tuple[CharName, NodeKeyframe, EdgeKeyframe, GraphValKeyframe]
	]:
		if (branch, turn, tick) not in self._all_keyframe_times:
			raise KeyframeError(branch, turn, tick)
		unpack = self.unpack
		for graph, nodes, edges, graph_val in self.call(
			"all_graphs_in_keyframe", branch, turn, tick
		):
			yield (
				unpack(graph),
				unpack(nodes),
				unpack(edges),
				unpack(graph_val),
			)

	def keyframes_graphs_dump(
		self,
	) -> Iterator[
		tuple[
			CharName,
			Branch,
			Turn,
			Tick,
			NodeKeyframe,
			EdgeKeyframe,
			CharDict,
		]
	]:
		self.flush()
		unpack = self.unpack
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
				unpack(graph),
				branch,
				turn,
				tick,
				unpack(nodes),
				unpack(edges),
				unpack(graph_val),
			)

	def keyframe_extensions_dump(
		self,
	) -> Iterator[
		tuple[
			Branch,
			Turn,
			Tick,
			UniversalKeyframe,
			RuleKeyframe,
			RulebookKeyframe,
		]
	]:
		self.flush()
		unpack = self.unpack
		for branch, turn, tick, universal, rule, rulebook in self.call(
			"keyframe_extensions_dump"
		):
			yield (
				branch,
				turn,
				tick,
				unpack(universal),
				unpack(rule),
				unpack(rulebook),
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
	) -> Iterator[tuple[Branch, Branch, Turn, Tick, Turn, Tick]]:
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

	def turns_dump(self) -> Iterator[tuple[Branch, Turn, Tick, Tick]]:
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

	def graph_val_del_time(self, branch, turn, tick):
		self._graphvals2set.cull(
			lambda g, k, b, r, t, v: (b, r, t) == (branch, turn, tick)
		)
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

	def nodes_del_time(self, branch, turn, tick):
		self._nodes2set.cull(
			lambda g, n, b, r, t, x: (b, r, t) == (branch, turn, tick)
		)
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

	def node_val_del_time(self, branch, turn, tick):
		self._nodevals2set.cull(
			lambda g, n, k, b, r, t, v: (b, r, t) == (branch, turn, tick)
		)
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

	def edges_del_time(self, branch, turn, tick):
		self._edges2set.cull(
			lambda g, o, d, b, r, t, x: (b, r, t) == (branch, turn, tick)
		)
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
				graph,
				unpack(orig),
				unpack(dest),
				unpack(key),
				branch,
				turn,
				tick,
				unpack(value),
			)

	def edge_val_del_time(self, branch, turn, tick):
		self._edgevals2set.cull(
			lambda g, o, d, k, b, r, t, v: (b, r, t) == (branch, turn, tick)
		)
		self.call("edge_val_del_time", branch, turn, tick)

	def plan_ticks_dump(self):
		self._planticks2set()
		return self.call("plan_ticks_dump")

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

	def character_rules_handled_dump(self):
		self.flush()
		unpack = self.unpack
		for branch, turn, character, rulebook, rule, tick in self.call(
			"character_rules_handled_dump"
		):
			yield unpack(character), unpack(rulebook), rule, branch, turn, tick

	def unit_rules_handled_dump(self):
		self._unit_rules_handled_to_set()
		unpack = self.unpack
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
				unpack(character),
				unpack(graph),
				unpack(unit),
				unpack(rulebook),
				rule,
				branch,
				turn,
				tick,
			)

	def character_thing_rules_handled_dump(self):
		self.flush()
		unpack = self.unpack
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
				unpack(character),
				unpack(thing),
				unpack(rulebook),
				rule,
				branch,
				turn,
				tick,
			)

	def character_place_rules_handled_dump(self):
		self.flush()
		unpack = self.unpack
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
				unpack(character),
				unpack(place),
				unpack(rulebook),
				rule,
				branch,
				turn,
				tick,
			)

	def character_portal_rules_handled_dump(self):
		self.flush()
		unpack = self.unpack
		for (
			branch,
			turn,
			character,
			rulebook,
			rule,
			orig,
			dest,
			tick,
		) in self.call("character_portal_rules_handled_dump"):
			yield (
				unpack(character),
				unpack(rulebook),
				unpack(orig),
				unpack(dest),
				rule,
				branch,
				turn,
				tick,
			)

	def node_rules_handled_dump(self):
		self.flush()
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
				self.unpack(character),
				self.unpack(node),
				self.unpack(rulebook),
				rule,
				branch,
				turn,
				tick,
			)

	def portal_rules_handled_dump(self):
		self.flush()
		unpack = self.unpack
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
				unpack(character),
				unpack(orig),
				unpack(dest),
				unpack(rulebook),
				rule,
				branch,
				turn,
				tick,
			)

	def things_dump(self):
		self.flush()
		unpack = self.unpack
		for branch, turn, tick, character, thing, location in self.call(
			"things_dump"
		):
			yield (
				unpack(character),
				unpack(thing),
				branch,
				turn,
				tick,
				unpack(location),
			)

	def units_dump(
		self,
	) -> Iterator[
		tuple[CharName, CharName, NodeName, Branch, Turn, Tick, bool]
	]:
		self.flush()
		unpack = self.unpack
		for (
			branch,
			turn,
			tick,
			character_graph,
			unit_graph,
			unit_node,
			is_av,
		) in self.call("units_dump"):
			yield (
				unpack(character_graph),
				unpack(unit_graph),
				unpack(unit_node),
				branch,
				turn,
				tick,
				is_av,
			)

	def count_all_table(self, tbl):
		return self.call("{}_count".format(tbl)).fetchone()[0]

	def rules_dump(self):
		self.flush()
		for (name,) in self.call("rules_dump"):
			yield name

	def things_del_time(self, branch: Branch, turn: Turn, tick: Tick):
		self._things2set.cull(
			lambda c, th, b, r, t, l: (b, r, t) == (branch, turn, tick)
		)
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
