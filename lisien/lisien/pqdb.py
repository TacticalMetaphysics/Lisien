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
import os
import sys
from dataclasses import KW_ONLY, dataclass
from functools import cached_property, partial
from operator import itemgetter
from typing import (
	Annotated,
	Any,
	ClassVar,
	Iterator,
	Literal,
	Optional,
	TypeAliasType,
	Union,
	get_args,
	get_origin,
)

import pyarrow as pa
from _operator import itemgetter
from pyarrow import compute as pc

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
	ActionRowType,
	Branch,
	CharacterRulesHandledRowType,
	CharName,
	CharRulebookRowType,
	EdgeKeyframe,
	EdgeRowType,
	EdgeValRowType,
	GraphRowType,
	GraphTypeStr,
	GraphValKeyframe,
	GraphValRowType,
	Key,
	KeyframeExtensionRowType,
	KeyframeGraphRowType,
	NodeKeyframe,
	NodeName,
	NodeRowType,
	NodeRulebookRowType,
	NodeRulesHandledRowType,
	NodeValRowType,
	Plan,
	PortalRulebookRowType,
	PortalRulesHandledRowType,
	PrereqRowType,
	RuleBigRowType,
	RulebookName,
	RulebookPriority,
	RulebooksKeyframe,
	RulebookTypeStr,
	RuleFuncName,
	RuleKeyframe,
	RuleName,
	RuleNeighborhoodRowType,
	Stat,
	StatDict,
	ThingRowType,
	Tick,
	Time,
	TriggerRowType,
	Turn,
	UnitRowType,
	UnitRulesHandledRowType,
	UniversalKey,
	UniversalKeyframe,
	Value,
)
from .types import __dict__ as types_dict
from .util import ELLIPSIS, EMPTY


@dataclass
class ParquetDatabaseConnector(ThreadedDatabaseConnector):
	path: str
	_: KW_ONLY
	clear: bool = False

	@dataclass
	class Looper(ConnectionLooper):
		def __post_init__(self):
			self.existence_lock.acquire(timeout=1)

		@cached_property
		def schema(self):
			import pyarrow as pa

			def origif(typ):
				if isinstance(typ, TypeAliasType):
					typ = typ.__value__
				if hasattr(typ, "evaluate_value"):
					typ = typ.evaluate_value()
				if hasattr(typ, "__supertype__"):
					return typ.__supertype__
				ret = get_origin(typ)
				if ret is Annotated:
					return get_args(typ)[0]
				return ret

			def argeval(typ: type) -> tuple[type, ...]:
				if hasattr(typ, "evaluate_value"):
					typ = typ.evaluate_value()
				if isinstance(typ, TypeAliasType):
					typ = typ.__value__
				return get_args(typ)

			def original(typ):
				prev = origif(typ)
				ret = origif(prev)
				if prev is None:
					return typ
				while ret is not None:
					prev = ret
					ret = origif(ret)
				return prev

			py2pq_typ = {
				bytes: pa.binary,
				float: pa.float64,
				str: pa.string,
				int: pa.int64,
				bool: pa.bool_,
			}
			ret = {}
			for table, serializer in Batch.serializers.items():
				argspec = inspect.getfullargspec(serializer)
				serialized_tuple_type = argspec.annotations["return"]
				if isinstance(serialized_tuple_type, str):
					serialized_tuple_type = eval(
						serialized_tuple_type, dict(types_dict)
					)
				columns = ret[table] = []
				for column, serialized_type in zip(
					argspec.args[1:], argeval(serialized_tuple_type)
				):
					origin = original(serialized_type)
					if origin is Union:
						options = argeval(serialized_type)
						if len(options) != 2 or type(None) not in options:
							raise TypeError(
								"Too many options for union type",
								column,
								serialized_type,
							)
						if type(None) is options[0]:
							origin = options[1]
						else:
							origin = options[0]
					elif origin is Literal:
						options = argeval(serialized_type)
						origin = type(options[0])
						if not all(isinstance(opt, origin) for opt in options):
							raise TypeError(
								"Literals not all of the same type",
								column,
								serialized_type,
							)
					columns.append((column, py2pq_typ[original(origin)]()))
			return ret

		@cached_property
		def _schema(self):
			return {}

		initial: ClassVar[dict] = {
			"global": [
				{
					"key": SCHEMAVER_B,
					"value": SCHEMA_VERSION_B,
				},
				{"key": b"\xa5trunk", "value": b"\xa5trunk"},
				{"key": b"\xa6branch", "value": b"\xa5trunk"},
				{"key": b"\xa4turn", "value": b"\x00"},
				{"key": b"\xa4tick", "value": b"\x00"},
				{"key": b"\xa8language", "value": b"\xa3eng"},
			],
			"branches": [
				{
					"branch": "trunk",
					"parent": None,
					"parent_turn": 0,
					"parent_tick": 0,
					"end_turn": 0,
					"end_tick": 0,
				}
			],
		}

		@staticmethod
		def echo(*args, **_):
			return args

		def commit(self):
			pass

		def close(self):
			if not self.outq.empty():
				self.outq.join()
			self.existence_lock.release()

		def initdb(self):
			if hasattr(self, "_initialized"):
				return RuntimeError("Already initialized the database")
			self._initialized = True
			initial = self.initial
			for table, schema in self.schema.items():
				schema = self._get_schema(table)
				db = self._get_db(table)
				if db.is_empty() and table in initial:
					db.create(
						initial[table],
						schema=schema,
					)
			glob_d = {}
			for d in self.dump("global"):
				if d["key"] in glob_d:
					return KeyError(
						"Initialization resulted in duplicate eternal record",
						d["key"],
					)
				glob_d[d["key"]] = d["value"]
			if SCHEMAVER_B not in glob_d:
				return ValueError("Not a Lisien database")
			elif glob_d[SCHEMAVER_B] != SCHEMA_VERSION_B:
				return ValueError(
					"Unsupported database schema version", glob_d[SCHEMAVER_B]
				)
			return glob_d

		def _get_db(self, table: str):
			from parquetdb import ParquetDB

			table_path = os.path.join(self.connector.path, table)
			try:
				return ParquetDB(
					table_path,
					schema=self._get_schema(table),
				)
			except TypeError:  # old parquetdb
				return ParquetDB(table_path, initial_fields=self.schema[table])

		def insert(self, table: str, data: list) -> None:
			self._get_db(table).create(data, schema=self._schema[table])

		def keyframes_graphs_delete(self, data: list[dict]):
			from pyarrow import compute as pc

			db = self._get_db("keyframes")
			todel = []
			for d in data:
				found: pa.Table = db.read(
					columns=["id"],
					filters=[
						pc.field("graph") == d["graph"],
						pc.field("branch") == d["branch"],
						pc.field("turn") == d["turn"],
						pc.field("tick") == d["tick"],
					],
				)
				if found.num_rows > 0:
					todel.extend(id_.as_py() for id_ in found["id"])
			if todel:
				db.delete(todel)

		def delete_keyframe(self, branch: Branch, turn: Turn, tick: Tick):
			from pyarrow import compute as pc

			filters = [
				pc.field("branch") == branch,
				pc.field("turn") == turn,
				pc.field("tick") == tick,
			]

			self._get_db("keyframes").delete(filters=filters)
			self._get_db("keyframes_graphs").delete(filters=filters)
			self._get_db("keyframe_extensions").delete(filters=filters)

		def delete(self, table: str, data: list[dict]):
			from pyarrow import compute as pc

			db = self._get_db(table)
			for datum in data:
				db.delete(
					filters=[pc.field(k) == v for (k, v) in datum.items()]
				)

		def all_keyframe_times(self):
			return {
				(d["branch"], d["turn"], d["tick"])
				for d in self._get_db("keyframes")
				.read(columns=["branch", "turn", "tick"])
				.to_pylist()
			}

		def truncate_all(self):
			for table in self.schema:
				db = self._get_db(table)
				if db.dataset_exists():
					db.drop_dataset()

		def del_units_after(self, many):
			from pyarrow import compute as pc

			db = self._get_db("units")
			ids = []
			for character, graph, node, branch, turn, tick in many:
				for d in db.read(
					filters=[
						pc.field("character_graph") == character,
						pc.field("unit_graph") == graph,
						pc.field("unit_node") == node,
						pc.field("branch") == branch,
						pc.field("turn") >= turn,
					],
					columns=["id", "turn", "tick"],
				).to_pylist():
					if d["turn"] == turn:
						if d["tick"] >= tick:
							ids.append(d["id"])
					else:
						ids.append(d["id"])
			if ids:
				db.delete(ids)

		def del_things_after(self, many):
			from pyarrow import compute as pc

			db = self._get_db("things")
			ids = []
			for character, thing, branch, turn, tick in many:
				for d in db.read(
					filters=[
						pc.field("character") == character,
						pc.field("thing") == thing,
						pc.field("branch") == branch,
						pc.field("turn") >= turn,
					],
					columns=["id", "turn", "tick"],
				).to_pylist():
					if d["turn"] == turn:
						if d["tick"] >= tick:
							ids.append(d["id"])
					else:
						ids.append(d["id"])
			if ids:
				db.delete(ids)

		def dump(self, table: str) -> list:
			data = (
				self._get_db(table).read().sort_by(self._sort_columns(table))
			)
			return data.to_pylist()

		def rowcount(self, table: str) -> int:
			return self._get_db(table).read().num_rows

		def bookmark_items(self) -> list[tuple[Key, Time]]:
			return [
				(d["name"], (d["branch"], d["turn"], d["tick"]))
				for d in self.dump("bookmarks")
			]

		def set_bookmark(
			self, key: bytes, branch: Branch, turn: Turn, tick: Tick
		):
			import pyarrow.compute as pc

			db = self._get_db("bookmarks")
			schema = self._get_schema("bookmarks")
			try:
				id_ = db.read(
					filters=[pc.field("key") == pc.scalar(key)], columns=["id"]
				)["id"][0]
			except IndexError:
				db.create(
					[
						{
							"key": key,
							"branch": branch,
							"turn": turn,
							"tick": tick,
						}
					],
					schema=schema,
				)
				return
			db.update(
				[
					{
						"id": id_,
						"key": key,
						"branch": branch,
						"turn": turn,
						"tick": tick,
					}
				],
				schema=schema,
			)

		def del_bookmark(self, key: bytes):
			import pyarrow.compute as pc

			self._get_db("bookmarks").delete(
				filters=[pc.field("key") == pc.scalar(key)]
			)

		def graphs(self) -> set[CharName]:
			return set(
				name.as_py()
				for name in self._get_db("graphs").read(columns=["graph"])[
					"graph"
				]
			)

		def load_tick_to_end(
			self, table: str, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[dict]:
			branch_data = self._get_db(table).read(
				filters=[pc.field("branch") == branch]
			)
			if branch_data.num_rows == 0:
				return []
			data0 = branch_data.filter(pc.field("turn") == turn_from).filter(
				pc.field("tick") >= tick_from
			)
			data1 = branch_data.filter(pc.field("turn") > turn_from)
			data = pa.concat_tables([data0, data1]).sort_by(
				self._sort_columns(table)
			)
			return data.to_pylist()

		def load_tick_to_tick(
			self,
			table: str,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[dict]:
			branch_data = self._get_db(table).read(
				filters=[pc.field("branch") == branch]
			)
			if branch_data.num_rows == 0:
				return []
			if turn_from == turn_to:
				datas = [
					branch_data.filter(pc.field("turn") == turn_from)
					.filter(pc.field("tick") >= tick_from)
					.filter(pc.field("tick") <= tick_to)
				]
			elif turn_to == turn_from + 1:
				datas = [
					branch_data.filter(pc.field("turn") == turn_from).filter(
						pc.field("tick") >= tick_from
					),
					branch_data.filter(pc.field("turn") == turn_to).filter(
						pc.field("tick") <= tick_to
					),
				]
			else:
				datas = [
					branch_data.filter(pc.field("turn") == turn_from).filter(
						pc.field("tick") >= tick_from
					),
					branch_data.filter(pc.field("turn") > turn_from).filter(
						pc.field("turn") < turn_to
					),
					branch_data.filter(pc.field("turn") == turn_to).filter(
						pc.field("tick") <= tick_to
					),
				]
			data = pa.concat_tables(datas).sort_by(self._sort_columns(table))
			return data.to_pylist()

		def _sort_columns(self, table: str) -> list[tuple[str, str]]:
			schema = self.schema[table]
			columns = ["branch", "turn", "tick"]
			timecols = set(columns)
			for column in map(itemgetter(0), schema):
				if column in timecols:
					timecols.remove(column)
					continue
				columns.append(column)
			if timecols:
				return [
					(col, "ascending") for col in map(itemgetter(0), schema)
				]
			return [(col, "ascending") for col in columns]

		def list_keyframes(self) -> list:
			return sorted(
				(
					self._get_db("keyframes")
					.read(
						columns=["graph", "branch", "turn", "tick"],
					)
					.to_pylist()
				),
				key=lambda d: (d["branch"], d["turn"], d["tick"], d["graph"]),
			)

		def get_keyframe(
			self, graph: bytes, branch: Branch, turn: Turn, tick: Tick
		) -> tuple[bytes, bytes, bytes] | None:
			from pyarrow import compute as pc

			rec = self._get_db("keyframes_graphs").read(
				filters=[
					pc.field("graph") == pc.scalar(graph),
					pc.field("branch") == pc.scalar(branch),
					pc.field("turn") == pc.scalar(turn),
					pc.field("tick") == pc.scalar(tick),
				],
				columns=["nodes", "edges", "graph_val"],
			)
			if not rec.num_rows:
				return None
			if rec.num_rows > 1:
				raise ValueError("Ambiguous keyframe, probably corrupt table")
			return (
				rec["nodes"][0].as_py(),
				rec["edges"][0].as_py(),
				rec["graph_val"][0].as_py(),
			)

		def universal_get(
			self, key: bytes, branch: Branch, turn: Turn, tick: Tick
		) -> bytes | type(...):
			db = self._get_db("universals")
			data = db.read(
				filters=[
					pc.field("branch") == branch,
					pc.field("key") == key,
					pc.field("turn") <= turn,
				]
			).sort_by([("turn", "descending"), ("tick", "descending")])
			for d in data.to_pylist():
				if (d["turn"], d["tick"]) <= (turn, tick):
					return d["value"]
			return ...

		def insert1(self, table: str, data: dict):
			try:
				return self.insert(table, [data])
			except Exception as ex:
				return ex

		def _set_rulebook_on_character(
			self,
			rbtyp: RulebookTypeStr,
			char: CharName,
			branch: Branch,
			turn: Turn,
			tick: Tick,
			rb: RulebookName,
		):
			self.insert1(
				f"{rbtyp}_rulebook",
				{
					"character": char,
					"branch": branch,
					"turn": turn,
					"tick": tick,
					"rulebook": rb,
				},
			)

		def graph_exists(self, graph: bytes) -> bool:
			from pyarrow import compute as pc

			return bool(
				self._get_db("graphs")
				.read(
					filters=[pc.field("graph") == pc.scalar(graph)],
					columns=["id"],
				)
				.num_rows
			)

		def get_global(self, key: bytes) -> bytes:
			from pyarrow import compute as pc

			ret = self._get_db("global").read(
				filters=[pc.field("key") == key],
			)
			if ret:
				return ret["value"][0].as_py()
			return ELLIPSIS

		def _get_schema(self, table) -> pa.schema:
			import pyarrow as pa

			if table in self._schema:
				return self._schema[table]
			ret = self._schema[table] = pa.schema(self.schema[table])
			return ret

		def global_keys(self):
			return [
				d["key"]
				for d in self._get_db("global")
				.read("global", columns=["key"])
				.to_pylist()
			]

		def field_get_id(self, table, keyfield, value):
			from pyarrow import compute as pc

			return self.filter_get_id(
				table, filters=[pc.field(keyfield) == value]
			)

		def filter_get_id(self, table, filters):
			ret = self._get_db(table).read(filters=filters, columns=["id"])
			if ret:
				return ret["id"][0].as_py()

		def have_branch(self, branch: Branch) -> bool:
			from pyarrow import compute as pc

			return bool(
				self._get_db("branches")
				.read("branches", filters=[pc.field("branch") == branch])
				.rowcount
			)

		def update_turn(
			self,
			branch: Branch,
			turn: Turn,
			end_tick: Tick,
			plan_end_tick: Tick,
		):
			from pyarrow import compute as pc

			id_ = self.filter_get_id(
				"turns",
				[pc.field("branch") == branch, pc.field("turn") == turn],
			)
			if id_ is None:
				return self._get_db("turns").create(
					[
						{
							"branch": branch,
							"turn": turn,
							"end_tick": end_tick,
							"plan_end_tick": plan_end_tick,
						}
					],
				)
			return self._get_db("turns").update(
				[
					{
						"id": id_,
						"end_tick": end_tick,
						"plan_end_tick": plan_end_tick,
					}
				]
			)

		def _del_time(
			self, table: str, branch: Branch, turn: Turn, tick: Tick
		):
			from pyarrow import compute as pc

			id_ = self.filter_get_id(
				table,
				filters=[
					pc.field("branch") == branch,
					pc.field("turn") == turn,
					pc.field("tick") == tick,
				],
			)
			if id_ is None:
				return
			self._get_db(table).delete([id_])

		def nodes_del_time(self, branch: Branch, turn: Turn, tick: Tick):
			self._del_time("nodes", branch, turn, tick)

		def edges_del_time(self, branch: Branch, turn: Turn, tick: Tick):
			self._del_time("edges", branch, turn, tick)

		def graph_val_del_time(self, branch: Branch, turn: Turn, tick: Tick):
			self._del_time("graph_val", branch, turn, tick)

		def node_val_del_time(self, branch: Branch, turn: Turn, tick: Tick):
			self._del_time("node_val", branch, turn, tick)

		def edge_val_del_time(self, branch: Branch, turn: Turn, tick: Tick):
			self._del_time("edge_val", branch, turn, tick)

		def get_keyframe_extensions(
			self, branch: Branch, turn: Turn, tick: Tick
		) -> tuple[bytes, bytes, bytes] | None:
			from pyarrow import compute as pc

			db = self._get_db("keyframe_extensions")
			data = db.read(
				filters=[
					pc.field("branch") == branch,
					pc.field("turn") == turn,
					pc.field("tick") == tick,
				]
			)
			if not data:
				return EMPTY, EMPTY, EMPTY
			return (
				data["universal"][0].as_py(),
				data["rule"][0].as_py(),
				data["rulebook"][0].as_py(),
			)

		def all_keyframe_graphs(self, branch: Branch, turn: Turn, tick: Tick):
			from pyarrow import compute as pc

			db = self._get_db("keyframes_graphs")
			data = db.read(
				filters=[
					pc.field("branch") == branch,
					pc.field("turn") == turn,
					pc.field("tick") == tick,
				]
			)
			return sorted(
				[
					(d["graph"], d["nodes"], d["edges"], d["graph_val"])
					for d in data.to_pylist()
				]
			)

		def set_rulebook(
			self,
			rulebook: bytes,
			branch: Branch,
			turn: Turn,
			tick: Tick,
			rules: bytes,
			priority: RulebookPriority,
		) -> bool:
			import pyarrow.compute as pc

			db = self._get_db("rulebooks")
			named_data = {
				"rulebook": rulebook,
				"branch": branch,
				"turn": turn,
				"tick": tick,
			}
			extant = db.read(
				filters=[
					pc.field(key) == value
					for (key, value) in named_data.items()
				]
			)
			create = not bool(extant.num_rows)
			named_data["rules"] = rules
			named_data["priority"] = priority
			if create:
				db.create([named_data])
			else:
				named_data["id"] = extant["id"][0].as_py()
				db.update([named_data])
			return create

		def run(self):
			def loud_exit(inst, ex):
				try:
					msg = (
						f"While calling {inst[0]}"
						f"({', '.join(map(repr, inst[1]))}{', ' if inst[2] else ''}"
						f"{', '.join('='.join(pair) for pair in inst[2].items())})"
						f"silenced, ParquetDBHolder got the exception: {repr(ex)}"
					)
				except Exception as ex2:
					grp = ExceptionGroup(
						"Multiple exceptions in ParquetDB connector", [ex, ex2]
					)
					msg = f"While calling {inst[0]}: {grp}"
				print(msg, file=sys.stderr)
				sys.exit(msg)

			inq = self.inq
			outq = self.outq

			def call_method(name, *args, silent=False, **kwargs):
				if callable(name):
					mth = name
				elif hasattr(self, name):
					mth = getattr(self, name)
				elif name.startswith("load_") and name.endswith(
					"_tick_to_end"
				):
					table = name.removeprefix("load_").removesuffix(
						"_tick_to_end"
					)
					mth = partial(self.load_tick_to_end, table)
				elif name.startswith("load_") and name.endswith(
					"_tick_to_tick"
				):
					table = name.removeprefix("load_").removesuffix(
						"_tick_to_tick"
					)
					mth = partial(self.load_tick_to_tick, table)
				else:
					raise ValueError("No method", name)
				try:
					res = mth(*args, **kwargs)
				except Exception as ex:
					if silent:
						loud_exit(inst, ex)
					res = ex
				if not silent:
					outq.put(res)
				inq.task_done()

			while True:
				inst = inq.get()
				if inst == "close":
					self.close()
					inq.task_done()
					return
				if inst == "commit":
					inq.task_done()
					continue
				if not isinstance(inst, (str, tuple)):
					raise TypeError("Can't use SQLAlchemy with ParquetDB")
				silent = False
				if inst[0] == "silent":
					silent = True
					inst = inst[1:]
				match inst:
					case ("echo", msg):
						outq.put(msg)
						inq.task_done()
					case ("echo", args, _):
						outq.put(args)
						inq.task_done()
					case ("one", cmd):
						call_method(cmd, silent=silent)
					case ("one", cmd, args):
						call_method(cmd, *args, silent=silent)
					case ("one", cmd, args, kwargs):
						call_method(cmd, *args, silent=silent, **kwargs)
					case ("many", cmd, several):
						for args, kwargs in several:
							try:
								res = getattr(self, cmd)(*args, **kwargs)
							except Exception as ex:
								if silent:
									loud_exit(("many", cmd, several), ex)
								res = ex
							if not silent:
								outq.put(res)
							if isinstance(res, Exception):
								break
						inq.task_done()
					case (cmd, args, kwargs):
						call_method(cmd, *args, silent=silent, **kwargs)
					case (cmd, args):
						call_method(cmd, *args, silent=silent)
					case cmd:
						call_method(cmd)

	@mutexed
	def call(self, method, *args, **kwargs):
		self._inq.put((method, args, kwargs))
		ret = self._outq.get()
		self._outq.task_done()
		if isinstance(ret, Exception):
			raise ret
		return ret

	def call_silent(self, method, *args, **kwargs):
		self._inq.put(("silent", method, args, kwargs))

	@mutexed
	def call_many(self, query_name: str, args: list):
		self._inq.put(("many", query_name, args))
		ret = self._outq.get()
		self._outq.task_done()
		if isinstance(ret, Exception):
			raise ret
		return ret

	def call_many_silent(self, query_name: str, args: list):
		self._inq.put(("silent", "many", query_name, args))

	@mutexed
	def insert_many(self, table_name: str, args: list[dict]):
		self.call("insert", table_name, args)

	def insert_many_silent(self, table_name: str, args: list[dict]):
		self.call_silent("insert", table_name, args)

	def delete_many_silent(self, table_name: str, args: list[dict]):
		self.call_silent("delete", table_name, args)

	def global_keys(self):
		unpack = self.unpack
		for key in self.call("global_keys"):
			yield unpack(key)

	def keyframes_dump(self) -> Iterator[tuple[Branch, Turn, Tick]]:
		self.flush()
		for d in self.call("dump", "keyframes"):
			yield d["branch"], d["turn"], d["tick"]

	def get_keyframe_extensions(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> tuple[UniversalKeyframe, RuleKeyframe, RulebooksKeyframe]:
		unpack = self.unpack
		univ, rule, rulebook = self.call(
			"get_keyframe_extensions", branch, turn, tick
		)
		return unpack(univ), unpack(rule), unpack(rulebook)

	def keyframes_graphs(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick]]:
		unpack = self.unpack
		for d in self.call("list_keyframes"):
			yield unpack(d["graph"]), d["branch"], d["turn"], d["tick"]

	def delete_keyframe(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		self.call("delete_keyframe", branch, turn, tick)

	def universal_get(
		self, key: UniversalKey, branch: Branch, turn: Turn, tick: Tick
	) -> Value:
		b = self.call("universal_get", self.pack(key), branch, turn, tick)
		if b is ...:
			raise KeyError(
				"No value for that universal key now", key, branch, turn, tick
			)
		return self.unpack(b)

	def graphs_types(
		self,
		branch: Branch,
		turn_from: Turn,
		tick_from: Tick,
		turn_to: Optional[Turn] = None,
		tick_to: Optional[Tick] = None,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, str]]:
		unpack = self.unpack_key
		if turn_to is None:
			if tick_to is not None:
				raise TypeError("Need both or neither of turn_to, tick_to")
			data = self.call(
				"load_tick_to_end", "graphs", branch, turn_from, tick_from
			)
		else:
			if tick_to is None:
				raise TypeError("Need both or neither of turn_to, tick_to")
			data = self.call(
				"load_tick_to_tick",
				"graphs",
				branch,
				turn_from,
				tick_from,
				turn_to,
				tick_to,
			)
		for d in data:
			yield (
				CharName(unpack(d["graph"])),
				branch,
				Turn(d["turn"]),
				Tick(d["tick"]),
				str(d["type"]),
			)

	def have_branch(self, branch: Branch) -> bool:
		return self.call("have_branch", branch)

	def branches_dump(
		self,
	) -> Iterator[tuple[Branch, Branch, Turn, Tick, Turn, Tick]]:
		for d in self.call("dump", "branches"):
			yield (
				d["branch"],
				d["parent"],
				d["parent_turn"],
				d["parent_tick"],
				d["end_turn"],
				d["end_tick"],
			)

	def global_get(self, key: Key) -> Any:
		try:
			return self.unpack(self.call("get_global", self.pack(key)))
		except KeyError:
			return ...

	def global_dump(self) -> Iterator[tuple[Key, Value]]:
		unpack = self.unpack
		unpack_key = self.unpack_key
		yield from (
			(unpack_key(d["key"]), unpack(d["value"]))
			for d in self.call("dump", "global")
		)

	def get_branch(self) -> Branch:
		v = self.unpack(self.call("get_global", b"\xa6branch"))
		if v is ...:
			mainbranch = self.unpack_key(self.call("get_global", b"\xa5trunk"))
			if not isinstance(mainbranch, str):
				raise TypeError("Invalid trunk", mainbranch)
			if mainbranch is None:
				return Branch("trunk")
			return Branch(mainbranch)
		return v

	def get_turn(self) -> Turn:
		v = self.unpack(self.call("get_global", b"\xa4turn"))
		if v is ...:
			return Turn(0)
		if not isinstance(v, int):
			raise TypeError("Invalid turn", v)
		return Turn(v)

	def get_tick(self) -> Tick:
		v = self.unpack(self.call("get_global", b"\xa4tick"))
		if v is ...:
			return Tick(0)
		if not isinstance(v, int):
			raise TypeError("Invalid tick", v)
		return Tick(v)

	def turns_dump(self) -> Iterator[tuple[Branch, Turn, Tick, Tick]]:
		for d in self.call("dump", "turns"):
			yield d["branch"], d["turn"], d["end_tick"], d["plan_end_tick"]

	def _extract_time(self, d: dict) -> Time:
		branch = d["branch"]
		if not isinstance(branch, str):
			raise TypeError("Invalid branch", branch)
		turn = d["turn"]
		if not isinstance(turn, int):
			raise TypeError("Invalid turn", turn)
		tick = d["tick"]
		if not isinstance(tick, int):
			raise TypeError("Invalid tick")
		return Branch(branch), Turn(turn), Tick(tick)

	def universals_dump(
		self,
	) -> Iterator[tuple[Branch, Turn, Tick, UniversalKey, Value]]:
		self.flush()
		unpack = self.unpack
		unpack_key = self.unpack_key
		for d in self.call("dump", "universals"):
			branch, turn, tick = self._extract_time(d)
			yield (
				branch,
				turn,
				tick,
				UniversalKey(unpack_key(d["key"])),
				unpack(d["value"]),
			)

	def rulebooks_dump(
		self,
	) -> Iterator[
		tuple[Branch, Turn, Tick, RulebookName, tuple[list[RuleName], float]]
	]:
		self.flush()
		unpack = self.unpack
		unpack_key = self.unpack_key
		for d in self.call("dump", "rulebooks"):
			branch, turn, tick = self._extract_time(d)
			_rules = unpack(d["rules"])
			if not isinstance(_rules, list) or not all(
				isinstance(rule, str) for rule in _rules
			):
				raise TypeError("Invalid rules", _rules)
			rules: list[RuleName] = _rules
			priority = d["priority"]
			if not isinstance(priority, float):
				raise TypeError("Invalid priority", priority)
			yield (
				branch,
				turn,
				tick,
				unpack_key(d["rulebook"]),
				(rules, priority),
			)

	def rules_dump(self) -> Iterator[RuleName]:
		for d in sorted(self.call("dump", "rules"), key=itemgetter("rule")):
			yield d["rule"]

	def _rule_dump(
		self, typ: Literal["triggers", "prereqs", "actions"]
	) -> Iterator[tuple[Branch, Turn, Tick, RuleName, list[RuleFuncName]]]:
		getattr(self, f"_{typ}2set")()
		unpack = self.unpack
		unpacked: dict[
			tuple[Branch, Turn, Tick, RuleName], list[RuleFuncName]
		] = {}
		for d in self.call("dump", "rule_" + typ):
			an_rule = unpack(d[typ])
			if not isinstance(an_rule, list) or not all(
				isinstance(f, str) for f in an_rule
			):
				raise TypeError("Invalid func list", an_rule)
			funx: list[RuleFuncName] = an_rule
			unpacked[d["rule"], d["branch"], d["turn"], d["tick"]] = funx
		for branch, turn, tick, rule in sorted(unpacked):
			yield branch, turn, tick, rule, unpacked[branch, turn, tick, rule]

	def rule_triggers_dump(
		self,
	) -> Iterator[TriggerRowType]:
		return self._rule_dump("triggers")

	def rule_prereqs_dump(
		self,
	) -> Iterator[PrereqRowType]:
		return self._rule_dump("prereqs")

	def rule_actions_dump(
		self,
	) -> Iterator[ActionRowType]:
		return self._rule_dump("actions")

	def rule_neighborhood_dump(
		self,
	) -> Iterator[RuleNeighborhoodRowType]:
		self._neighbors2set()
		return iter(
			sorted(
				(
					d["rule"],
					d["branch"],
					d["turn"],
					d["tick"],
					d["neighborhood"],
				)
				for d in self.call("dump", "rule_neighborhood")
			)
		)

	def rule_big_dump(
		self,
	) -> Iterator[RuleBigRowType]:
		self._big2set()
		return iter(
			sorted(
				(d["rule"], d["branch"], d["turn"], d["tick"], d["big"])
				for d in self.call("dump", "rule_big")
			)
		)

	def node_rulebook_dump(
		self,
	) -> Iterator[NodeRulebookRowType]:
		self._noderb2set()
		unpack_key = self.unpack_key
		for d in self.call("dump", "node_rulebook"):
			charn = CharName(unpack_key(d["character"]))
			nn = NodeName(unpack_key(d["node"]))
			branch, turn, tick = self._extract_time(d)
			rb = RulebookName(unpack_key(d["rulebook"]))
			yield branch, turn, tick, charn, nn, rb

	def portal_rulebook_dump(
		self,
	) -> Iterator[PortalRulebookRowType]:
		self._portrb2set()
		unpack_key = self.unpack_key
		for d in self.call("dump", "portal_rulebook"):
			charn = CharName(unpack_key(d["character"]))
			orig = NodeName(unpack_key(d["orig"]))
			dest = NodeName(unpack_key(d["dest"]))
			branch, turn, tick = self._extract_time(d)
			rb = RulebookName(unpack_key(d["rulebook"]))
			yield branch, turn, tick, charn, orig, dest, rb

	def rules_insert(self, rule: RuleName) -> None:
		self.call("insert1", "rule", {"rule": rule})

	def _character_rulebook_dump(
		self, typ: RulebookTypeStr
	) -> Iterator[CharRulebookRowType]:
		getattr(self, f"_{typ}_rulebook_to_set")()
		unpack_key = self.unpack_key
		for d in self.call("dump", f"{typ}_rulebook"):
			charn = CharName(unpack_key(d["character"]))
			branch, turn, tick = self._extract_time(d)
			rb = RulebookName(unpack_key(d["rulebook"]))
			yield branch, turn, tick, charn, rb

	def character_rulebook_dump(
		self,
	) -> Iterator[CharRulebookRowType]:
		return self._character_rulebook_dump("character")

	def unit_rulebook_dump(
		self,
	) -> Iterator[CharRulebookRowType]:
		return self._character_rulebook_dump("unit")

	def character_thing_rulebook_dump(
		self,
	) -> Iterator[CharRulebookRowType]:
		return self._character_rulebook_dump("character_thing")

	def character_place_rulebook_dump(
		self,
	) -> Iterator[CharRulebookRowType]:
		return self._character_rulebook_dump("character_place")

	def character_portal_rulebook_dump(
		self,
	) -> Iterator[CharRulebookRowType]:
		return self._character_rulebook_dump("character_portal")

	def character_rules_handled_dump(
		self,
	) -> Iterator[CharacterRulesHandledRowType]:
		self._char_rules_handled()
		unpack_key = self.unpack_key
		for d in self.call("dump", "character_rules_handled"):
			charn = CharName(unpack_key(d["character"]))
			rb = RulebookName(unpack_key(d["rulebook"]))
			rule = RuleName(d["rule"])
			branch, turn, tick = self._extract_time(d)
			yield branch, turn, charn, rb, rule, tick

	def unit_rules_handled_dump(
		self,
	) -> Iterator[UnitRulesHandledRowType]:
		self._unit_rules_handled_to_set()
		unpack_key = self.unpack_key
		for d in self.call("dump", "unit_rules_handled"):
			charn = CharName(unpack_key(d["character"]))
			graph = CharName(unpack_key(d["graph"]))
			unit = NodeName(unpack_key(d["unit"]))
			rb = RulebookName(unpack_key(d["rulebook"]))
			rule = RuleName(d["rule"])
			branch, turn, tick = self._extract_time(d)
			yield branch, turn, charn, graph, unit, rb, rule, tick

	def character_thing_rules_handled_dump(
		self,
	) -> Iterator[NodeRulesHandledRowType]:
		self._char_thing_rules_handled()
		unpack_key = self.unpack_key
		for d in self.call("dump", "character_thing_rules_handled"):
			charn = CharName(unpack_key(d["character"]))
			thing = NodeName(unpack_key(d["thing"]))
			rb = RulebookName(unpack_key(d["rulebook"]))
			rule = RuleName(d["rule"])
			branch, turn, tick = self._extract_time(d)
			yield branch, turn, charn, thing, rb, rule, tick

	def character_place_rules_handled_dump(
		self,
	) -> Iterator[NodeRulesHandledRowType]:
		self._char_place_rules_handled()
		unpack_key = self.unpack_key
		for d in self.call("dump", "character_place_rules_handled"):
			charn = CharName(unpack_key(d["character"]))
			place = NodeName(unpack_key(d["place"]))
			rb = RulebookName(unpack_key(d["rulebook"]))
			rule = RuleName(d["rule"])
			branch, turn, tick = self._extract_time(d)
			yield branch, turn, charn, place, rb, rule, tick

	def character_portal_rules_handled_dump(
		self,
	) -> Iterator[PortalRulesHandledRowType]:
		self.flush()
		unpack_key = self.unpack_key
		for d in self.call("dump", "character_portal_rules_handled"):
			charn = CharName(unpack_key(d["character"]))
			orig = NodeName(unpack_key(d["orig"]))
			dest = NodeName(unpack_key(d["dest"]))
			rb = RulebookName(unpack_key(d["rulebook"]))
			rule = RuleName(d["rule"])
			branch, turn, tick = self._extract_time(d)
			yield branch, turn, charn, orig, dest, rb, rule, tick

	def node_rules_handled_dump(
		self,
	) -> Iterator[NodeRulesHandledRowType]:
		self._node_rules_handled_to_set()
		unpack_key = self.unpack_key
		for d in self.call("dump", "node_rules_handled"):
			charn = CharName(unpack_key(d["character"]))
			node = NodeName(unpack_key(d["node"]))
			rb = RulebookName(unpack_key(d["rulebook"]))
			rule = RuleName(d["rule"])
			branch, turn, tick = self._extract_time(d)
			yield branch, turn, charn, node, rb, rule, tick

	def portal_rules_handled_dump(
		self,
	) -> Iterator[PortalRulesHandledRowType]:
		self._portal_rules_handled_to_set()
		unpack_key = self.unpack_key
		for d in self.call("dump", "portal_rules_handled"):
			charn = CharName(unpack_key(d["character"]))
			orig = NodeName(unpack_key(d["orig"]))
			dest = NodeName(unpack_key(d["dest"]))
			rb = RulebookName(unpack_key(d["rulebook"]))
			rule = RuleName(d["rule"])
			branch, turn, tick = self._extract_time(d)
			yield branch, turn, charn, orig, dest, rb, rule, tick

	def things_dump(
		self,
	) -> Iterator[ThingRowType]:
		self._things2set()
		unpack_key = self.unpack_key
		for d in self.call("dump", "things"):
			charn = CharName(unpack_key(d["character"]))
			thing = NodeName(unpack_key(d["thing"]))
			loc = NodeName(unpack_key(d["location"]))
			branch, turn, tick = self._extract_time(d)
			yield branch, turn, tick, charn, thing, loc

	def units_dump(
		self,
	) -> Iterator[UnitRowType]:
		self._unitness()
		unpack_key = self.unpack_key
		for d in self.call("dump", "units"):
			charn = CharName(unpack_key(d["character_graph"]))
			graph = CharName(unpack_key(d["unit_graph"]))
			unit = NodeName(unpack_key(d["unit_node"]))
			branch, turn, tick = self._extract_time(d)
			is_unit = d["is_unit"]
			if not isinstance(is_unit, bool):
				raise TypeError("Not boolean", is_unit)
			yield branch, turn, tick, charn, graph, unit, is_unit

	def count_all_table(self, tbl: str) -> int:
		self.flush()
		return self.call("rowcount", tbl)

	def things_del_time(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		super().things_del_time(branch, turn, tick)
		self.call(
			"delete",
			"things",
			[{"branch": branch, "turn": turn, "tick": tick}],
		)

	def unit_set(
		self,
		character: CharName,
		graph: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		is_unit: bool,
	) -> None:
		self._unitness.append(
			(branch, turn, tick, character, graph, node, is_unit)
		)

	def rulebook_set(
		self,
		rulebook: RulebookName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rules: list[RuleName],
	) -> None:
		pack = self.pack
		self.call(
			"insert1",
			"rulebooks",
			dict(
				rulebook=pack(rulebook),
				branch=branch,
				turn=turn,
				tick=tick,
				rules=pack(rules),
			),
		)

	def turns_completed_dump(self) -> Iterator[tuple[Branch, Turn]]:
		self.flush()
		for d in self.call("dump", "turns_completed"):
			yield d["branch"], d["turn"]

	def graph_val_dump(self) -> Iterator[GraphValRowType]:
		self.flush()
		unpack = self.unpack
		unpack_key = self.unpack_key
		extract_time = self._extract_time
		for d in self.call("dump", "graph_val"):
			graph = CharName(unpack_key(d["graph"]))
			key = Stat(unpack_key(d["key"]))
			value = unpack(d["value"])
			branch, turn, tick = extract_time(d)
			yield branch, turn, tick, graph, key, value

	def graph_val_del_time(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		super().graph_val_del_time(branch, turn, tick)
		self.call("graph_val_del_time", branch, turn, tick)

	def graphs_dump(
		self,
	) -> Iterator[GraphRowType]:
		self.flush()
		unpack_key = self.unpack_key
		extract_time = self._extract_time
		for d in self.call("dump", "graphs"):
			graph = CharName(unpack_key(d["graph"]))
			type: GraphTypeStr = d["type"]
			branch, turn, tick = extract_time(d)
			yield branch, turn, tick, graph, type

	def nodes_del_time(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		super().nodes_del_time(branch, turn, tick)
		self.call("nodes_del_time", branch, turn, tick)

	def nodes_dump(self) -> Iterator[NodeRowType]:
		self.flush()
		unpack_key = self.unpack_key
		for d in self.call("dump", "nodes"):
			graph = CharName(unpack_key(d["graph"]))
			node = NodeName(unpack_key(d["node"]))
			extant = d["extant"]
			if not isinstance(extant, bool):
				raise TypeError("Not boolean", extant)
			branch, turn, tick = self._extract_time(d)
			yield branch, turn, tick, graph, node, extant

	def node_val_dump(self) -> Iterator[NodeValRowType]:
		self.flush()
		unpack = self.unpack
		unpack_key = self.unpack_key
		for d in self.call("dump", "node_val"):
			graph = CharName(unpack_key(d["graph"]))
			node = NodeName(unpack_key(d["node"]))
			key = Stat(unpack_key(d["key"]))
			value = unpack(d["value"])
			branch, turn, tick = self._extract_time(d)
			yield branch, turn, tick, graph, node, key, value

	def node_val_del_time(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		super().node_val_del_time(branch, turn, tick)
		self.call("node_val_del_time", branch, turn, tick)

	def edges_dump(self) -> Iterator[EdgeRowType]:
		self._edges2set()
		unpack_key = self.unpack_key
		extract_time = self._extract_time
		for d in self.call("dump", "edges"):
			graph = CharName(unpack_key(d["graph"]))
			orig = NodeName(unpack_key(d["orig"]))
			dest = NodeName(unpack_key(d["dest"]))
			extant = d["extant"]
			if not isinstance(extant, bool):
				raise TypeError("Not boolean", extant)
			branch, turn, tick = extract_time(d)
			yield branch, turn, tick, graph, orig, dest, extant

	def edges_del_time(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		super().edges_del_time(branch, turn, tick)
		self.call("edges_del_time", branch, turn, tick)

	def edge_val_dump(self) -> Iterator[EdgeValRowType]:
		self.flush()
		unpack = self.unpack
		unpack_key = self.unpack_key
		extract_time = self._extract_time
		for d in self.call("dump", "edge_val"):
			char = CharName(unpack_key(d["character"]))
			orig = NodeName(unpack_key(d["orig"]))
			dest = NodeName(unpack_key(d["dest"]))
			key = Stat(unpack_key(d["key"]))
			value = unpack(d["value"])
			branch, turn, tick = extract_time(d)
			yield branch, turn, tick, char, orig, dest, key, value

	def edge_val_del_time(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		super().edge_val_del_time(branch, turn, tick)
		self.call("edge_val_del_time", branch, turn, tick)

	def plan_ticks_dump(self) -> Iterator[tuple[Plan, Branch, Turn, Tick]]:
		self._planticks2set()
		for d in self.call("dump", "plan_ticks"):
			yield d["plan_id"], d["branch"], d["turn"], d["tick"]

	def get_all_keyframe_graphs(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> Iterator[
		tuple[CharName, NodeKeyframe, EdgeKeyframe, GraphValKeyframe]
	]:
		if (branch, turn, tick) not in self._all_keyframe_times:
			raise KeyframeError(branch, turn, tick)
		unpack = self.unpack
		unpack_key = self.unpack_key
		for graph, nodes, edges, graph_val in self.call(
			"all_keyframe_graphs", branch, turn, tick
		):
			yield (
				CharName(unpack_key(graph)),
				unpack(nodes),
				unpack(edges),
				unpack(graph_val),
			)

	def keyframes_graphs_dump(
		self,
	) -> Iterator[KeyframeGraphRowType]:
		self._new_keyframes_graphs()
		unpack = self.unpack
		unpack_key = self.unpack_key
		extract_time = self._extract_time
		for d in self.call("dump", "keyframes_graphs"):
			graph = CharName(unpack_key(d["graph"]))
			nodes: NodeKeyframe = unpack(d["nodes"])
			edges: EdgeKeyframe = unpack(d["edges"])
			graph_val: StatDict = unpack(d["graph_val"])
			branch, turn, tick = extract_time(d)
			yield branch, turn, tick, graph, nodes, edges, graph_val

	def keyframe_extensions_dump(
		self,
	) -> Iterator[KeyframeExtensionRowType]:
		self._new_keyframe_extensions()
		extract_time = self._extract_time
		for d in self.call("dump", "keyframe_extensions"):
			branch, turn, tick = extract_time(d)
			universal = self._unpack_universal_keyframe(d["universal"])
			rule = self._unpack_rules_keyframe(d["rule"])
			rulebook = self._unpack_rulebooks_keyframe(d["rulebook"])
			yield branch, turn, tick, universal, rule, rulebook

	def truncate_all(self) -> None:
		self.call("truncate_all")

	def close(self) -> None:
		self._inq.put("close")
		self._looper.existence_lock.acquire()
		self._looper.existence_lock.release()
		self._t.join()

	def commit(self) -> None:
		self.flush()
		self.call("commit")

	def _init_db(self) -> dict:
		if hasattr(self, "_initialized"):
			raise RuntimeError("Initialized the database twice")
		ret = self.call("initdb")
		if isinstance(ret, Exception):
			raise ret
		elif not isinstance(ret, dict):
			raise TypeError("initdb didn't return a dictionary", ret)
		unpack = self.unpack
		self.eternal = GlobalKeyValueStore(
			self, {unpack(k): unpack(v) for (k, v) in ret.items()}
		)
		self.all_rules.clear()
		self.all_rules.update(d["rule"] for d in self.call("dump", "rules"))
		self._all_keyframe_times.clear()
		self._all_keyframe_times.update(self.keyframes_dump())
		self._initialized = True
		return ret

	def bookmarks_dump(self) -> Iterator[tuple[Key, Time]]:
		return iter(self.call("bookmark_items"))

	def del_bookmark(self, key: Key) -> None:
		self.call("del_bookmark", key)
