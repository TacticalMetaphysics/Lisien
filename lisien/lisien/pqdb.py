from __future__ import annotations

import inspect
import os
import sys
from _operator import itemgetter
from dataclasses import dataclass, KW_ONLY
from functools import cached_property
from typing import (
	get_origin,
	Annotated,
	get_args,
	Union,
	Literal,
	ClassVar,
	Iterator,
	Optional,
	Any,
	Iterable,
)

import pyarrow as pa

from .db import (
	ThreadedDatabaseConnector,
	ConnectionLooper,
	batched,
	SCHEMAVER_B,
	SCHEMA_VERSION_B,
	mutexed,
	GlobalKeyValueStore,
)
from .exc import KeyframeError
from .types import (
	Branch,
	Turn,
	Tick,
	Key,
	Time,
	CharName,
	RulebookTypeStr,
	RulebookName,
	RulebookPriority,
	RuleName,
	RuleNeighborhood,
	RuleBig,
	UniversalKeyframe,
	RuleKeyframe,
	RulebookKeyframe,
	RuleFuncName,
	TriggerFuncName,
	PrereqFuncName,
	ActionFuncName,
	NodeName,
	GraphValRowType,
	GraphTypeStr,
	NodeRowType,
	NodeValRowType,
	EdgeRowType,
	EdgeValRowType,
	Plan,
	NodeKeyframe,
	EdgeKeyframe,
	GraphValKeyframe,
	CharDict,
)
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
				if hasattr(typ, "__supertype__"):
					return typ.__supertype__
				ret = get_origin(typ)
				if ret is Annotated:
					return get_args(typ)[0]
				return ret

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
			for table, serializer in batched.serializers.items():
				argspec = inspect.getfullargspec(serializer)
				serialized_tuple_type = argspec.annotations["return"]
				if isinstance(serialized_tuple_type, str):
					serialized_tuple_type = eval(serialized_tuple_type)
				columns = ret[table] = []
				for column, serialized_type in zip(
					argspec.args[1:], get_args(serialized_tuple_type)
				):
					origin = original(serialized_type)
					if origin is Union:
						options = get_args(serialized_type)
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
						options = get_args(serialized_type)
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
					f"Unsupported database schema version", glob_d[SCHEMAVER_B]
				)
			return glob_d

		def _get_db(self, table: str):
			from parquetdb import ParquetDB

			return ParquetDB(os.path.join(self.connector.path, table))

		def insert(self, table: str, data: list) -> None:
			self._get_db(table).create(data, schema=self._schema[table])

		def keyframes_graphs_delete(self, data: list[dict]):
			import pyarrow as pa
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
			data = [
				d
				for d in self._get_db(table).read().to_pylist()
				if d.keys() - {"id"}
			]
			schema = self._get_schema(table)
			data.sort(key=lambda d: tuple(d[name] for name in schema.names))
			return data

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

		def load_graphs_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		):
			from pyarrow import compute as pc

			data = (
				self._get_db("graphs").read(
					filters=[
						pc.field("branch") == branch,
						pc.field("turn") >= turn_from,
					],
				)
			).to_pylist()
			return sorted(
				[
					(d["graph"], d["turn"], d["tick"], d["type"])
					for d in data
					if (turn_from, tick_from) <= (d["turn"], d["tick"])
				],
				key=lambda d: (d[1], d[2], d[0]),
			)

		def load_graphs_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		):
			from pyarrow import compute as pc

			data = (
				self._get_db("graphs").read(
					filters=[
						pc.field("branch") == branch,
						pc.field("turn") >= turn_from,
						pc.field("turn") <= turn_to,
					]
				)
			).to_pylist()
			return sorted(
				[
					(d["graph"], d["turn"], d["tick"], d["type"])
					for d in data
					if (turn_from, tick_from)
					<= (d["turn"], d["tick"])
					<= (turn_to, tick_to)
				],
				key=lambda d: (d[2], d[3], d[0]),
			)

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

		def load_universals_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_universals_tick_to_end(
					branch, turn_from, tick_from
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def _table_columns(self, table: str) -> list[str]:
			return list(map(itemgetter(0), self.schema[table]))

		def _iter_part_tick_to_end(
			self, table: str, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> Iterator[dict]:
			from pyarrow import compute as pc

			db = self._get_db(table)
			for d in filter(
				None,
				db.read(
					filters=[
						pc.field("branch") == branch,
						pc.field("turn") >= turn_from,
					],
					columns=self._table_columns(table),
				).to_pylist(),
			):
				if d["turn"] == turn_from:
					if d["tick"] >= tick_from:
						yield d
				else:
					yield d

		def _iter_universals_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> Iterator[tuple[bytes, Turn, Tick, bytes]]:
			for d in self._iter_part_tick_to_end(
				"universals", branch, turn_from, tick_from
			):
				try:
					yield d["key"], d["turn"], d["tick"], d["value"]
				except KeyError:
					continue

		def _list_part_tick_to_tick(
			self,
			table: str,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> Iterator[dict]:
			from pyarrow import compute as pc

			db = self._get_db(table)
			if turn_from == turn_to:
				return db.read(
					filters=[
						pc.field("branch") == branch,
						pc.field("turn") == turn_from,
						pc.field("tick") >= tick_from,
						pc.field("tick") <= tick_to,
					],
					columns=self._table_columns(table),
				).to_pylist()
			else:
				ret = []
				for d in db.read(
					filters=[
						pc.field("branch") == branch,
						pc.field("turn") >= turn_from,
						pc.field("turn") <= turn_to,
					],
					columns=self._table_columns(table),
				).to_pylist():
					if (
						(turn_from, tick_from)
						<= (d["turn"], d["tick"])
						<= (turn_to, tick_to)
					):
						ret.append(d)
				return ret

		def load_universals_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, Turn, Tick, bytes]]:
			return [
				(d["key"], d["turn"], d["tick"], d["value"])
				for d in sorted(
					self._list_part_tick_to_tick(
						"universals",
						branch,
						turn_from,
						tick_from,
						turn_to,
						tick_to,
					),
					key=lambda dee: (dee["turn"], dee["tick"], dee["key"]),
				)
			]

		def load_things_tick_to_end(self, *args, **kwargs):
			if len(args) + len(kwargs) == 4:
				return self._load_things_tick_to_end_character(*args, **kwargs)
			else:
				return self._load_things_tick_to_end_all(*args, **kwargs)

		def _load_things_tick_to_end_all(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, bytes, Turn, Tick, bytes]]:
			return [
				(
					d["character"],
					d["thing"],
					d["turn"],
					d["tick"],
					d["location"],
				)
				for d in sorted(
					self._iter_part_tick_to_end(
						"things", branch, turn_from, tick_from
					),
					key=lambda d: (
						d["turn"],
						d["tick"],
						d["character"],
						d["thing"],
					),
				)
			]

		def _load_things_tick_to_end_character(
			self,
			character: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
		) -> list[tuple[bytes, Turn, Tick, bytes]]:
			import pyarrow.compute as pc

			return [
				(d["thing"], d["turn"], d["tick"], d["location"])
				for d in sorted(
					self._get_db("things")
					.read(
						filters=[
							pc.field("character") == character,
							pc.field("branch") == branch,
							pc.field("turn") >= turn_from,
						],
					)
					.to_pylist(),
					key=lambda d: (d["turn"], d["tick"], d["thing"]),
				)
				if (turn_from, tick_from) <= (d["turn"], d["tick"])
			]

		def load_things_tick_to_tick(self, *args, **kwargs):
			if len(args) + len(kwargs) == 6:
				return self._load_things_tick_to_tick_character(
					*args, **kwargs
				)
			else:
				return self._load_things_tick_to_tick_all(*args, **kwargs)

		def _load_things_tick_to_tick_all(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, Turn, Tick, bytes]]:
			def sort_key(d: dict) -> tuple[int, int, bytes, bytes]:
				return d["turn"], d["tick"], d["character"], d["thing"]

			data = self._list_part_tick_to_tick(
				"things", branch, turn_from, tick_from, turn_to, tick_to
			)
			data.sort(key=sort_key)
			return [
				(
					d["character"],
					d["thing"],
					d["turn"],
					d["tick"],
					d["location"],
				)
				for d in data
			]

		def _load_things_tick_to_tick_character(
			self,
			character: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_things_tick_to_tick_character(
					character, branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def _iter_things_tick_to_tick_character(
			self,
			character: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		):
			from pyarrow import compute as pc

			db = self._get_db("things")
			if turn_from == turn_to:
				for d in db.read(
					filters=[
						pc.field("character") == character,
						pc.field("branch") == branch,
						pc.field("turn") == turn_from,
						pc.field("tick") >= tick_from,
						pc.field("tick") <= tick_to,
					],
				).to_pylist():
					yield d["thing"], d["turn"], d["tick"], d["location"]
			else:
				for d in db.read(
					filters=[
						pc.field("character") == character,
						pc.field("branch") == branch,
						pc.field("turn_from") >= turn_from,
						pc.field("turn_to") <= turn_to,
					],
				).to_pylist():
					if d["turn"] == turn_from:
						if d["tick"] >= tick_from:
							yield (
								d["thing"],
								d["turn"],
								d["tick"],
								d["location"],
							)
					elif d["turn"] == turn_to:
						if d["tick"] <= tick_to:
							yield (
								d["thing"],
								d["turn"],
								d["tick"],
								d["location"],
							)
					else:
						yield d["thing"], d["turn"], d["tick"], d["location"]

		def load_graph_val_tick_to_end(self, *args, **kwargs):
			if len(args) + len(kwargs) == 4:
				return self._load_graph_val_tick_to_end_graph(*args, **kwargs)
			else:
				return self._load_graph_val_tick_to_end_all(*args, **kwargs)

		def _load_graph_val_tick_to_end_all(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_graph_val_tick_to_end_all(
					branch, turn_from, tick_from
				),
				key=lambda t: (t[2], t[3], t[0], t[1]),
			)

		def _iter_graph_val_tick_to_end_all(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> Iterator[tuple[bytes, bytes, Turn, Tick, bytes]]:
			for d in self._iter_part_tick_to_end(
				"graph_val", branch, turn_from, tick_from
			):
				yield (
					d["graph"],
					d["key"],
					d["turn"],
					d["tick"],
					d["value"],
				)

		def _load_graph_val_tick_to_end_graph(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
		) -> list[tuple[bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_graph_val_tick_to_end_graph(
					graph, branch, turn_from, tick_from
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def _iter_graph_val_tick_to_end_graph(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
		) -> Iterator[tuple[bytes, Turn, Tick, bytes]]:
			from pyarrow import compute as pc

			for d in (
				self._get_db("graph_val")
				.read(
					filters=[
						pc.field("graph") == graph,
						pc.field("branch") == branch,
						pc.field("turn") >= turn_from,
					],
				)
				.to_pylist()
			):
				if d["turn"] == turn_from:
					if d["tick"] >= tick_from:
						yield d["key"], d["turn"], d["tick"], d["value"]
				else:
					yield d["key"], d["turn"], d["tick"], d["value"]

		def load_graph_val_tick_to_tick(self, *args, **kwargs):
			if len(args) + len(kwargs) == 6:
				return self._load_graph_val_tick_to_tick_graph(*args, **kwargs)
			else:
				return self._load_graph_val_tick_to_tick_all(*args, **kwargs)

		def _load_graph_val_tick_to_tick_all(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_graph_val_tick_to_tick_all(
					branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda t: (t[2], t[3], t[0], t[1]),
			)

		def _iter_graph_val_tick_to_tick_all(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> Iterator[tuple[bytes, bytes, Turn, Tick, bytes]]:
			for d in self._list_part_tick_to_tick(
				"graph_val", branch, turn_from, tick_from, turn_to, tick_to
			):
				yield d["graph"], d["key"], d["turn"], d["tick"], d["value"]

		def _load_graph_val_tick_to_tick_graph(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_graph_val_tick_to_tick(
					graph, branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def _iter_graph_val_tick_to_tick(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> Iterator[tuple[bytes, Turn, Tick, bytes]]:
			for d in self._list_part_tick_to_tick(
				"graph_val", branch, turn_from, tick_from, turn_to, tick_to
			):
				yield d["key"], d["turn"], d["tick"], d["value"]

		def _load_nodes_tick_to_end_graph(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
		) -> list[tuple[bytes, Turn, Tick, bool]]:
			return sorted(
				self._iter_nodes_tick_to_end_graph(
					graph, branch, turn_from, tick_from
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def _load_nodes_tick_to_end_all(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, bytes, Turn, Tick, bool]]:
			return sorted(
				self._iter_nodes_tick_to_end_all(branch, turn_from, tick_from),
				key=lambda t: (t[2], t[3], t[0], t[1]),
			)

		def load_nodes_tick_to_end(self, *args, **kwargs):
			if len(args) + len(kwargs) == 4:
				return self._load_nodes_tick_to_end_graph(*args, **kwargs)
			else:
				return self._load_nodes_tick_to_end_all(*args, **kwargs)

		def _iter_nodes_tick_to_end_graph(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
		) -> Iterator[tuple[bytes, Turn, Tick, bool]]:
			from pyarrow import compute as pc

			for d in (
				self._get_db("nodes")
				.read(
					filters=[
						pc.field("graph") == graph,
						pc.field("branch") == branch,
						pc.field("turn") >= turn_from,
					],
				)
				.to_pylist()
			):
				if d["turn"] == turn_from:
					if d["tick"] >= tick_from:
						yield (
							d["node"],
							d["turn"],
							d["tick"],
							d["extant"],
						)
				else:
					yield (
						d["node"],
						d["turn"],
						d["tick"],
						d["extant"],
					)

		def _iter_nodes_tick_to_end_all(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> Iterator[tuple[bytes, bytes, Turn, Tick, bool]]:
			for d in self._iter_part_tick_to_end(
				"nodes", branch, turn_from, tick_from
			):
				yield (
					d["graph"],
					d["node"],
					d["turn"],
					d["tick"],
					d["extant"],
				)

		def load_nodes_tick_to_tick(self, *args, **kwargs):
			if len(args) + len(kwargs) == 6:
				return self.load_nodes_tick_to_tick_graph(*args, **kwargs)
			else:
				return self.load_nodes_tick_to_tick_all(*args, **kwargs)

		def load_nodes_tick_to_tick_all(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, Turn, Tick, bool]]:
			return sorted(
				self._iter_nodes_tick_to_tick_all(
					branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda t: (t[2], t[3], t[0], t[1]),
			)

		def load_nodes_tick_to_tick_graph(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, Turn, Tick, bool]]:
			return sorted(
				self._iter_nodes_tick_to_tick_graph(
					graph, branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def _iter_nodes_tick_to_tick_all(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> Iterator[tuple[bytes, bytes, Turn, Tick, bool]]:
			for d in self._list_part_tick_to_tick(
				"nodes", branch, turn_from, tick_from, turn_to, tick_to
			):
				yield d["graph"], d["node"], d["turn"], d["tick"], d["extant"]

		def _iter_nodes_tick_to_tick_graph(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> Iterator[tuple[bytes, Turn, Tick, bool]]:
			from pyarrow import compute as pc

			db = self._get_db("nodes")
			if turn_from == turn_to:
				for d in db.read(
					filters=[
						pc.field("graph") == graph,
						pc.field("branch") == branch,
						pc.field("turn") == turn_from,
						pc.field("tick") >= tick_from,
						pc.field("tick") <= tick_to,
					],
				).to_pylist():
					yield (
						d["node"],
						d["turn"],
						d["tick"],
						d["extant"],
					)
			else:
				for d in db.read(
					filters=[
						pc.field("graph") == graph,
						pc.field("branch") == branch,
						pc.field("turn") >= turn_from,
						pc.field("turn") <= turn_to,
					],
				).to_pylist():
					if d["turn"] == turn_from:
						if d["tick"] >= tick_from:
							yield (
								d["node"],
								d["turn"],
								d["tick"],
								d["extant"],
							)
					elif d["turn"] == turn_to:
						if d["tick"] <= tick_to:
							yield (
								d["node"],
								d["turn"],
								d["tick"],
								d["extant"],
							)
					else:
						yield (
							d["node"],
							d["turn"],
							d["tick"],
							d["extant"],
						)

		def load_node_val_tick_to_end(self, *args, **kwargs):
			if len(args) + len(kwargs) == 4:
				return self._load_node_val_tick_to_end_graph(*args, **kwargs)
			else:
				return self._load_node_val_tick_to_end_all(*args, **kwargs)

		def _load_node_val_tick_to_end_graph(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
		) -> list[tuple[bytes, bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_node_val_tick_to_end_graph(
					graph, branch, turn_from, tick_from
				),
				key=lambda t: (t[2], t[3], t[0], t[1]),
			)

		def _load_node_val_tick_to_end_all(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, bytes, bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_node_val_tick_to_end_all(
					branch, turn_from, tick_from
				),
				key=lambda t: (t[3], t[4], t[0], t[1], t[2]),
			)

		def _iter_node_val_tick_to_end_all(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> Iterator[tuple[bytes, bytes, bytes, Turn, Tick, bytes]]:
			for d in self._iter_part_tick_to_end(
				"node_val", branch, turn_from, tick_from
			):
				yield (
					d["graph"],
					d["node"],
					d["key"],
					d["turn"],
					d["tick"],
					d["value"],
				)

		def _iter_node_val_tick_to_end_graph(
			self, graph: bytes, branch: str, turn_from: Turn, tick_from: Tick
		) -> Iterator[tuple[bytes, bytes, int, int, bytes]]:
			from pyarrow import compute as pc

			for d in (
				self._get_db("node_val")
				.read(
					filters=[
						pc.field("graph") == graph,
						pc.field("branch") == branch,
						pc.field("turn") >= turn_from,
					],
				)
				.to_pylist()
			):
				if d["turn"] == turn_from:
					if d["tick"] >= tick_from:
						yield (
							d["node"],
							d["key"],
							d["turn"],
							d["tick"],
							d["value"],
						)
				else:
					yield d["node"], d["key"], d["turn"], d["tick"], d["value"]

		def load_node_val_tick_to_tick(self, *args, **kwargs):
			if len(args) + len(kwargs) == 6:
				return self._load_node_val_tick_to_tick_graph(*args, **kwargs)
			else:
				return self._load_node_val_tick_to_tick_all(*args, **kwargs)

		def _load_node_val_tick_to_tick_all(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_node_val_tick_to_tick_all(
					branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda t: (t[3], t[4], t[0], t[1], t[2]),
			)

		def _iter_node_val_tick_to_tick_all(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> Iterator[tuple[bytes, bytes, bytes, Turn, Tick, bytes]]:
			for d in self._list_part_tick_to_tick(
				"node_val", branch, turn_from, tick_from, turn_to, tick_to
			):
				yield (
					d["graph"],
					d["node"],
					d["key"],
					d["turn"],
					d["tick"],
					d["value"],
				)

		def _load_node_val_tick_to_tick_graph(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_node_val_tick_to_tick_graph(
					graph, branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda t: (t[2], t[3], t[0], t[1]),
			)

		def _iter_node_val_tick_to_tick_graph(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> Iterator[tuple[bytes, bytes, Turn, Tick, bytes]]:
			from pyarrow import compute as pc

			db = self._get_db("node_val")
			if turn_from == turn_to:
				for d in db.read(
					filters=[
						pc.field("graph") == graph,
						pc.field("branch") == branch,
						pc.field("turn") == turn_from,
						pc.field("tick") >= tick_from,
						pc.field("tick") <= tick_to,
					],
				).to_pylist():
					yield (
						d["node"],
						d["key"],
						d["turn"],
						d["tick"],
						d["value"],
					)
			else:
				for d in db.read(
					filters=[
						pc.field("graph") == graph,
						pc.field("branch") == branch,
						pc.field("turn") >= turn_from,
						pc.field("turn") <= turn_to,
					],
				).to_pylist():
					if d["turn"] == turn_from:
						if d["tick"] >= tick_from:
							yield (
								d["node"],
								d["key"],
								d["turn"],
								d["tick"],
								d["value"],
							)
					elif d["turn"] == turn_to:
						if d["tick"] <= tick_to:
							yield (
								d["node"],
								d["key"],
								d["turn"],
								d["tick"],
								d["value"],
							)
					else:
						yield (
							d["node"],
							d["key"],
							d["turn"],
							d["tick"],
							d["value"],
						)

		def load_edges_tick_to_end(self, *args, **kwargs):
			if len(args) + len(kwargs) == 4:
				return self._load_edges_tick_to_end_graph(*args, **kwargs)
			else:
				return self._load_edges_tick_to_end_all(*args, **kwargs)

		def _load_edges_tick_to_end_all(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, bytes, bytes, Turn, Tick, bool]]:
			return sorted(
				self._iter_edges_tick_to_end_all(branch, turn_from, tick_from),
				key=lambda t: (t[3], t[4], t[0], t[1], t[2]),
			)

		def _iter_edges_tick_to_end_all(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> Iterator[tuple[bytes, bytes, bytes, Turn, Tick, bool]]:
			for d in self._iter_part_tick_to_end(
				"edges", branch, turn_from, tick_from
			):
				yield (
					d["graph"],
					d["orig"],
					d["dest"],
					d["turn"],
					d["tick"],
					d["extant"],
				)

		def _load_edges_tick_to_end_graph(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
		) -> list[tuple[bytes, bytes, Turn, Tick, bool]]:
			return sorted(
				self._iter_edges_tick_to_end_graph(
					graph, branch, turn_from, tick_from
				),
				key=lambda t: (t[2], t[3], t[0], t[1]),
			)

		def _iter_edges_tick_to_end_graph(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
		) -> Iterator[tuple[bytes, bytes, Turn, Tick, bool]]:
			from pyarrow import compute as pc

			for d in (
				self._get_db("edges")
				.read(
					filters=[
						pc.field("graph") == graph,
						pc.field("branch") == branch,
						pc.field("turn") >= turn_from,
					],
				)
				.to_pylist()
			):
				if d["turn"] == turn_from:
					if d["tick"] >= tick_from:
						yield (
							d["orig"],
							d["dest"],
							d["turn"],
							d["tick"],
							d["extant"],
						)
				else:
					yield (
						d["orig"],
						d["dest"],
						d["turn"],
						d["tick"],
						d["extant"],
					)

		def load_edges_tick_to_tick(self, *args, **kwargs):
			if len(args) + len(kwargs) == 6:
				return self._load_edges_tick_to_tick_graph(*args, **kwargs)
			else:
				return self._load_edges_tick_to_tick_all(*args, **kwargs)

		def _load_edges_tick_to_tick_all(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, bytes, Turn, Tick, bool]]:
			return sorted(
				self._iter_edges_tick_to_tick_all(
					branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda t: (t[3], t[4], t[0], t[1], t[2]),
			)

		def _iter_edges_tick_to_tick_all(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> Iterator[tuple[bytes, bytes, bytes, Turn, Tick, bool]]:
			for d in self._list_part_tick_to_tick(
				"edges", branch, turn_from, tick_from, turn_to, tick_to
			):
				yield (
					d["graph"],
					d["orig"],
					d["dest"],
					d["turn"],
					d["tick"],
					d["extant"],
				)

		def _load_edges_tick_to_tick_graph(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, bytes, Turn, Tick, bool]]:
			return sorted(
				self._iter_edges_tick_to_tick_graph(
					graph, branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda t: (t[3], t[4], t[0], t[1], t[2]),
			)

		def _iter_edges_tick_to_tick_graph(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> Iterator[tuple[bytes, bytes, bytes, Turn, Tick, bool]]:
			from pyarrow import compute as pc

			db = self._get_db("edges")
			if turn_from == turn_to:
				for d in db.read(
					filters=[
						pc.field("graph") == graph,
						pc.field("branch") == branch,
						pc.field("turn") == turn_from,
						pc.field("tick") >= tick_from,
						pc.field("tick") <= tick_to,
					],
				).to_pylist():
					yield (
						d["orig"],
						d["dest"],
						d["turn"],
						d["tick"],
						d["extant"],
					)
			else:
				for d in db.read(
					filters=[
						pc.field("graph") == graph,
						pc.field("branch") == branch,
						pc.field("turn") >= turn_from,
						pc.field("turn") <= turn_to,
					],
				).to_pylist():
					if d["turn"] == turn_from:
						if d["tick"] >= tick_from:
							yield (
								d["orig"],
								d["dest"],
								d["turn"],
								d["tick"],
								d["extant"],
							)
					elif d["turn"] == turn_to:
						if d["tick"] <= tick_to:
							yield (
								d["orig"],
								d["dest"],
								d["turn"],
								d["tick"],
								d["extant"],
							)
					else:
						yield (
							d["orig"],
							d["dest"],
							d["turn"],
							d["tick"],
							d["extant"],
						)

		def load_edge_val_tick_to_end(self, *args, **kwargs):
			if len(args) + len(kwargs) == 6:
				return self._load_edge_val_tick_to_end_graph(*args, **kwargs)
			else:
				return self._load_edge_val_tick_to_end_all(*args, **kwargs)

		def _load_edge_val_tick_to_end_all(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, bytes, bytes, bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_edge_val_tick_to_end_all(
					branch, turn_from, tick_from
				),
				key=lambda t: (t[4], t[5], t[0], t[1], t[2], t[3]),
			)

		def _iter_edge_val_tick_to_end_all(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> Iterator[tuple[bytes, bytes, bytes, bytes, Turn, Tick, bytes]]:
			for d in self._iter_part_tick_to_end(
				"edge_val", branch, turn_from, tick_from
			):
				yield (
					d["graph"],
					d["orig"],
					d["dest"],
					d["key"],
					d["turn"],
					d["tick"],
					d["value"],
				)

		def _load_edge_val_tick_to_end_graph(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
		) -> list[tuple[bytes, bytes, bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_edge_val_tick_to_end_graph(
					graph, branch, turn_from, tick_from
				),
				key=lambda t: (t[3], t[4], t[0], t[1], t[2]),
			)

		def _iter_edge_val_tick_to_end_graph(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
		) -> Iterator[tuple[bytes, bytes, bytes, Turn, Tick, bytes]]:
			from pyarrow import compute as pc

			for d in (
				self._get_db("edge_val")
				.read(
					filters=[
						pc.field("graph") == graph,
						pc.field("branch") == branch,
						pc.field("turn") >= turn_from,
					],
				)
				.to_pylist()
			):
				if d["turn"] == turn_from:
					if d["tick"] >= tick_from:
						yield (
							d["orig"],
							d["dest"],
							d["key"],
							d["turn"],
							d["tick"],
							d["value"],
						)
				else:
					yield (
						d["orig"],
						d["dest"],
						d["key"],
						d["turn"],
						d["tick"],
						d["value"],
					)

		def load_edge_val_tick_to_tick(self, *args, **kwargs):
			if len(args) + len(kwargs) == 6:
				return self._load_edge_val_tick_to_tick_graph(*args, **kwargs)
			else:
				return self._load_edge_val_tick_to_tick_all(*args, **kwargs)

		def _load_edge_val_tick_to_tick_all(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, bytes, bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_edge_val_tick_to_tick_all(
					branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda t: (t[4], t[5], t[0], t[1], t[2], t[3]),
			)

		def _iter_edge_val_tick_to_tick_all(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> Iterator[tuple[bytes, bytes, bytes, bytes, Turn, Tick, bytes]]:
			for d in self._list_part_tick_to_tick(
				"edge_val", branch, turn_from, tick_from, turn_to, tick_to
			):
				yield (
					d["graph"],
					d["orig"],
					d["dest"],
					d["key"],
					d["turn"],
					d["tick"],
					d["value"],
				)

		def _load_edge_val_tick_to_tick_graph(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_edge_val_tick_to_tick_graph(
					graph, branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda t: (t[3], t[4], t[0], t[1], t[2]),
			)

		def _iter_edge_val_tick_to_tick_graph(
			self,
			graph: bytes,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> Iterator[tuple[bytes, bytes, bytes, Turn, Tick, bytes]]:
			from pyarrow import compute as pc

			db = self._get_db("edge_val")
			if turn_from == turn_to:
				for d in db.read(
					filters=[
						pc.field("graph") == graph,
						pc.field("branch") == branch,
						pc.field("turn") == turn_from,
						pc.field("tick") >= tick_from,
						pc.field("tick") <= tick_to,
					],
				).to_pylist():
					yield (
						d["orig"],
						d["dest"],
						d["key"],
						d["turn"],
						d["tick"],
						d["value"],
					)
			else:
				for d in db.read(
					filters=[
						pc.field("graph") == graph,
						pc.field("branch") == branch,
						pc.field("turn") >= turn_from,
						pc.field("turn") <= turn_to,
					],
				).to_pylist():
					if d["turn"] == turn_from:
						if d["tick"] >= tick_from:
							yield (
								d["orig"],
								d["dest"],
								d["key"],
								d["turn"],
								d["tick"],
								d["value"],
							)
					elif d["turn"] == turn_to:
						if d["tick"] <= tick_to:
							yield (
								d["orig"],
								d["dest"],
								d["key"],
								d["turn"],
								d["tick"],
								d["value"],
							)
					else:
						yield (
							d["orig"],
							d["dest"],
							d["key"],
							d["turn"],
							d["tick"],
							d["value"],
						)

		def load_character_rulebook_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_character_rulebook_tick_to_end_part(
					"character", branch, turn_from, tick_from
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def _iter_character_rulebook_tick_to_end_part(
			self, part: str, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> Iterator[tuple[bytes, Turn, Tick, bytes]]:
			for d in self._iter_part_tick_to_end(
				f"{part}_rulebook", branch, turn_from, tick_from
			):
				yield d["character"], d["turn"], d["tick"], d["rulebook"]

		def load_character_rulebook_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_character_rulebook_tick_to_tick_part(
					"character", branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def _iter_character_rulebook_tick_to_tick_part(
			self,
			part: str,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> Iterator[tuple[bytes, Turn, Tick, bytes]]:
			for d in self._list_part_tick_to_tick(
				f"{part}_rulebook",
				branch,
				turn_from,
				tick_from,
				turn_to,
				tick_to,
			):
				yield d["character"], d["turn"], d["tick"], d["rulebook"]

		def load_unit_rulebook_tick_to_end(
			self, branch: str, turn_from: int, tick_from: int
		) -> list[tuple[bytes, int, int, bytes]]:
			return sorted(
				self._iter_character_rulebook_tick_to_end_part(
					"unit", branch, turn_from, tick_from
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def load_unit_rulebook_tick_to_tick(
			self,
			branch: str,
			turn_from: int,
			tick_from: int,
			turn_to: int,
			tick_to: int,
		) -> list[tuple[bytes, int, int, bytes]]:
			return sorted(
				self._iter_character_rulebook_tick_to_tick_part(
					"unit", branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def load_character_thing_rulebook_tick_to_end(
			self, branch: str, turn_from: int, tick_from: int
		) -> list[tuple[bytes, int, int, bytes]]:
			return sorted(
				self._iter_character_rulebook_tick_to_end_part(
					"character_thing", branch, turn_from, tick_from
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def load_character_thing_rulebook_tick_to_tick(
			self,
			branch: str,
			turn_from: int,
			tick_from: int,
			turn_to: int,
			tick_to: int,
		) -> list[tuple[bytes, int, int, bytes]]:
			return sorted(
				self._iter_character_rulebook_tick_to_tick_part(
					"character_thing",
					branch,
					turn_from,
					tick_from,
					turn_to,
					tick_to,
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def load_character_place_rulebook_tick_to_end(
			self, branch: str, turn_from: int, tick_from: int
		) -> list[tuple[bytes, int, int, bytes]]:
			return sorted(
				self._iter_character_rulebook_tick_to_end_part(
					"character_place", branch, turn_from, tick_from
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def load_character_place_rulebook_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_character_rulebook_tick_to_tick_part(
					"character_place",
					branch,
					turn_from,
					tick_from,
					turn_to,
					tick_to,
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def load_character_portal_rulebook_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_character_rulebook_tick_to_end_part(
					"character_portal", branch, turn_from, tick_from
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def load_character_portal_rulebook_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_character_rulebook_tick_to_tick_part(
					"character_portal",
					branch,
					turn_from,
					tick_from,
					turn_to,
					tick_to,
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def load_node_rulebook_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, bytes, int, int, bytes]]:
			return sorted(
				self._iter_node_rulebook_tick_to_end(
					branch, turn_from, tick_from
				),
				key=lambda t: (t[2], t[3], t[0], t[1]),
			)

		def _iter_node_rulebook_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> Iterator[tuple[bytes, bytes, Turn, Tick, bytes]]:
			for d in self._iter_part_tick_to_end(
				"node_rulebook", branch, turn_from, tick_from
			):
				yield (
					d["character"],
					d["node"],
					d["turn"],
					d["tick"],
					d["rulebook"],
				)

		def load_node_rulebook_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_node_rulebook_tick_to_tick(
					branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda t: (t[2], t[3], t[0], t[1]),
			)

		def _iter_node_rulebook_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> Iterator[tuple[bytes, bytes, Turn, Tick, bytes]]:
			for d in self._list_part_tick_to_tick(
				"node_rulebook", branch, turn_from, tick_from, turn_to, tick_to
			):
				yield (
					d["character"],
					d["node"],
					d["turn"],
					d["tick"],
					d["rulebook"],
				)

		def load_portal_rulebook_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, bytes, bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_portal_rulebook_tick_to_end(
					branch, turn_from, tick_from
				),
				key=lambda t: (t[3], t[4], t[0], t[1], t[2]),
			)

		def _iter_portal_rulebook_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> Iterator[tuple[bytes, bytes, bytes, Turn, Tick, bytes]]:
			for d in self._iter_part_tick_to_end(
				"portal_rulebook", branch, turn_from, tick_from
			):
				yield (
					d["character"],
					d["orig"],
					d["dest"],
					d["turn"],
					d["tick"],
					d["rulebook"],
				)

		def load_portal_rulebook_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, bytes, Turn, Tick, bytes]]:
			return sorted(
				self._iter_portal_rulebook_tick_to_tick(
					branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda t: (t[3], t[4], t[0], t[1], t[2]),
			)

		def _iter_portal_rulebook_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> Iterator[tuple[bytes, bytes, bytes, Turn, Tick, bytes]]:
			for d in self._list_part_tick_to_tick(
				"portal_rulebook",
				branch,
				turn_from,
				tick_from,
				turn_to,
				tick_to,
			):
				yield (
					d["character"],
					d["orig"],
					d["dest"],
					d["turn"],
					d["tick"],
					d["rulebook"],
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

		def load_rulebooks_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, Turn, Tick, bytes, RulebookPriority]]:
			return sorted(
				self._iter_rulebooks_tick_to_end(branch, turn_from, tick_from),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def _iter_rulebooks_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> Iterator[tuple[bytes, Turn, Tick, bytes, RulebookPriority]]:
			for d in self._iter_part_tick_to_end(
				"rulebooks", branch, turn_from, tick_from
			):
				yield (
					d["rulebook"],
					d["turn"],
					d["tick"],
					d["rules"],
					d["priority"],
				)

		def load_rulebooks_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, Turn, Tick, bytes, RulebookPriority]]:
			return sorted(
				self._iter_rulebooks_tick_to_tick(
					branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def _iter_rulebooks_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> Iterator[tuple[bytes, Turn, Tick, bytes, RulebookPriority]]:
			for d in self._list_part_tick_to_tick(
				"rulebooks", branch, turn_from, tick_from, turn_to, tick_to
			):
				yield (
					d["rulebook"],
					d["turn"],
					d["tick"],
					d["rules"],
					d["priority"],
				)

		def _load_rule_part_tick_to_end(
			self, part, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[
			tuple[RuleName, Turn, Tick, bytes | RuleNeighborhood | RuleBig]
		]:
			return sorted(
				self._iter_rule_part_tick_to_end(
					part, branch, turn_from, tick_from
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def _iter_rule_part_tick_to_end(
			self, part, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> Iterator[tuple[RuleName, Turn, Tick, bytes | RuleBig]]:
			for d in sorted(
				self._iter_part_tick_to_end(
					f"rule_{part}", branch, turn_from, tick_from
				),
				key=lambda d: (d["turn"], d["tick"], d["rule"]),
			):
				yield d["rule"], d["turn"], d["tick"], d[part]

		def _load_rule_part_tick_to_tick(
			self,
			part: str,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[
			tuple[Branch, Turn, Tick, bytes | RuleNeighborhood | RuleBig]
		]:
			return sorted(
				self._iter_rule_part_tick_to_tick(
					part, branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda t: (t[1], t[2], t[0]),
			)

		def _iter_rule_part_tick_to_tick(
			self,
			part,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> Iterator[tuple[Branch, Turn, Tick, bytes]]:
			for d in self._list_part_tick_to_tick(
				f"rule_{part}", branch, turn_from, tick_from, turn_to, tick_to
			):
				yield d["rule"], d["turn"], d["tick"], d[part]

		def load_rule_triggers_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[Branch, Turn, Tick, bytes]]:
			return self._load_rule_part_tick_to_end(
				"triggers", branch, turn_from, tick_from
			)

		def load_rule_triggers_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[Branch, Turn, Tick, bytes]]:
			return self._load_rule_part_tick_to_tick(
				"triggers", branch, turn_from, tick_from, turn_to, tick_to
			)

		def load_rule_prereqs_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[Branch, Turn, Tick, bytes]]:
			return self._load_rule_part_tick_to_end(
				"prereqs", branch, turn_from, tick_from
			)

		def load_rule_prereqs_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[Branch, Turn, Tick, bytes]]:
			return self._load_rule_part_tick_to_tick(
				"prereqs", branch, turn_from, tick_from, turn_to, tick_to
			)

		def load_rule_actions_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[Branch, Turn, Tick, bytes]]:
			return self._load_rule_part_tick_to_end(
				"actions", branch, turn_from, tick_from
			)

		def load_rule_actions_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[Branch, Turn, Tick, bytes]]:
			return self._load_rule_part_tick_to_tick(
				"actions", branch, turn_from, tick_from, turn_to, tick_to
			)

		def load_rule_neighborhood_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[Branch, Turn, Tick, bytes]]:
			return self._load_rule_part_tick_to_end(
				"neighborhood", branch, turn_from, tick_from
			)

		def load_rule_neighborhood_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[RuleName, Turn, Tick, bytes]]:
			return self._load_rule_part_tick_to_tick(
				"neighborhood", branch, turn_from, tick_from, turn_to, tick_to
			)

		def load_rule_big_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[RuleName, Turn, Tick, RuleBig]]:
			return self._load_rule_part_tick_to_end(
				"big", branch, turn_from, tick_from
			)

		def load_rule_big_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[RuleName, Turn, Tick, RuleBig]]:
			return self._load_rule_part_tick_to_tick(
				"big", branch, turn_from, tick_from, turn_to, tick_to
			)

		def load_character_rules_handled_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, bytes, RuleName, Turn, Tick]]:
			return sorted(
				self._iter_character_rules_handled_tick_to_end(
					branch, turn_from, tick_from
				),
				key=lambda t: (t[3], t[4], t[0], t[1], t[2]),
			)

		def _iter_character_rules_handled_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> Iterator[tuple[bytes, bytes, RuleName, Turn, Tick]]:
			for d in self._iter_part_tick_to_end(
				"character_rules_handled", branch, turn_from, tick_from
			):
				yield (
					d["character"],
					d["rulebook"],
					d["rule"],
					d["turn"],
					d["tick"],
				)

		def load_character_rules_handled_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, RuleName, Turn, Tick]]:
			return sorted(
				self._iter_character_rules_handled_tick_to_tick(
					branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda t: (t[3], t[4], t[0], t[1], t[2]),
			)

		def _iter_character_rules_handled_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> Iterator[tuple[bytes, bytes, RuleName, Turn, Tick]]:
			for d in self._list_part_tick_to_tick(
				"character_rules_handled",
				branch,
				turn_from,
				tick_from,
				turn_to,
				tick_to,
			):
				yield (
					d["character"],
					d["rulebook"],
					d["rule"],
					d["turn"],
					d["tick"],
				)

		def load_unit_rules_handled_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, bytes, bytes, bytes, RuleName, Turn, Tick]]:
			return [
				(
					d["character"],
					d["graph"],
					d["unit"],
					d["rulebook"],
					d["rule"],
					d["turn"],
					d["tick"],
				)
				for d in sorted(
					self._iter_part_tick_to_end(
						"unit_rules_handled", branch, turn_from, tick_from
					),
					key=lambda d: (
						d["turn"],
						d["tick"],
						d["character"],
						d["graph"],
						d["unit"],
						d["rulebook"],
						d["rule"],
					),
				)
			]

		def load_unit_rules_handled_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, bytes, bytes, RuleName, Turn, Tick]]:
			return [
				(
					d["character"],
					d["graph"],
					d["unit"],
					d["rulebook"],
					d["rule"],
					d["turn"],
					d["tick"],
				)
				for d in sorted(
					self._list_part_tick_to_tick(
						"unit_rules_handled",
						branch,
						turn_from,
						tick_from,
						turn_to,
						tick_to,
					),
					key=lambda d: (
						d["turn"],
						d["tick"],
						d["character"],
						d["graph"],
						d["unit"],
						d["rulebook"],
						d["rule"],
					),
				)
			]

		def load_character_thing_rules_handled_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, bytes, bytes, CharName, Turn, Tick]]:
			return [
				(
					d["character"],
					d["thing"],
					d["rulebook"],
					d["rule"],
					d["turn"],
					d["tick"],
				)
				for d in sorted(
					self._iter_part_tick_to_end(
						"character_thing_rules_handled",
						branch,
						turn_from,
						tick_from,
					),
					key=lambda d: (
						d["turn"],
						d["tick"],
						d["character"],
						d["thing"],
						d["rulebook"],
						d["rule"],
					),
				)
			]

		def load_character_thing_rules_handled_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, bytes, RuleName, Turn, Tick]]:
			return [
				(
					d["character"],
					d["thing"],
					d["rulebook"],
					d["rule"],
					d["turn"],
					d["tick"],
				)
				for d in sorted(
					self._list_part_tick_to_tick(
						"character_thing_rules_handled",
						branch,
						turn_from,
						tick_from,
						turn_to,
						tick_to,
					),
					key=lambda d: (
						d["turn"],
						d["tick"],
						d["character"],
						d["thing"],
						d["rulebook"],
						d["rule"],
					),
				)
			]

		def load_character_place_rules_handled_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, bytes, bytes, RuleName, Turn, Tick]]:
			return [
				(
					d["character"],
					d["place"],
					d["rulebook"],
					d["rule"],
					d["turn"],
					d["tick"],
				)
				for d in sorted(
					self._iter_part_tick_to_end(
						"character_place_rules_handled",
						branch,
						turn_from,
						tick_from,
					),
					key=lambda d: (
						d["turn"],
						d["tick"],
						d["character"],
						d["place"],
						d["rulebook"],
						d["rule"],
					),
				)
			]

		def load_character_place_rules_handled_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, bytes, RuleName, Turn, Tick]]:
			return [
				(
					d["character"],
					d["place"],
					d["rulebook"],
					d["rule"],
					d["turn"],
					d["tick"],
				)
				for d in sorted(
					self._list_part_tick_to_tick(
						"character_place_rules_handled",
						branch,
						turn_from,
						tick_from,
						turn_to,
						tick_to,
					),
					key=lambda d: (
						d["turn"],
						d["tick"],
						d["character"],
						d["place"],
						d["rulebook"],
						d["rule"],
					),
				)
			]

		def load_character_portal_rules_handled_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, bytes, bytes, bytes, RuleName, Turn, Tick]]:
			return [
				(
					d["character"],
					d["orig"],
					d["dest"],
					d["rulebook"],
					d["rule"],
					d["turn"],
					d["tick"],
				)
				for d in sorted(
					self._iter_part_tick_to_end(
						"character_portal_rules_handled",
						branch,
						turn_from,
						tick_from,
					),
					key=lambda d: (
						d["turn"],
						d["tick"],
						d["character"],
						d["orig"],
						d["dest"],
						d["rulebook"],
						d["rule"],
					),
				)
			]

		def load_character_portal_rules_handled_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, bytes, bytes, RuleName, Turn, Tick]]:
			return [
				(
					d["character"],
					d["orig"],
					d["dest"],
					d["rulebook"],
					d["rule"],
					d["turn"],
					d["tick"],
				)
				for d in sorted(
					self._list_part_tick_to_tick(
						"character_portal_rules_handled",
						branch,
						turn_from,
						tick_from,
						turn_to,
						tick_to,
					),
					key=lambda d: (
						d["turn"],
						d["tick"],
						d["character"],
						d["orig"],
						d["dest"],
						d["rlulebook"],
						d["rule"],
					),
				)
			]

		def load_node_rules_handled_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, bytes, bytes, RuleName, Turn, Tick]]:
			return [
				(
					d["character"],
					d["node"],
					d["rulebook"],
					d["rule"],
					d["turn"],
					d["tick"],
				)
				for d in sorted(
					self._iter_part_tick_to_end(
						"node_rules_handled", branch, turn_from, tick_from
					),
					key=lambda d: (
						d["turn"],
						d["tick"],
						d["character"],
						d["node"],
						d["rulebook"],
						d["rule"],
					),
				)
			]

		def load_node_rules_handled_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, bytes, RuleName, Turn, Tick]]:
			return [
				(
					d["character"],
					d["node"],
					d["rulebook"],
					d["rule"],
					d["turn"],
					d["tick"],
				)
				for d in sorted(
					self._list_part_tick_to_tick(
						"node_rules_handled",
						branch,
						turn_from,
						tick_from,
						turn_to,
						tick_to,
					),
					key=lambda d: (
						d["turn"],
						d["tick"],
						d["character"],
						d["node"],
						d["rulebook"],
						d["rule"],
					),
				)
			]

		def load_portal_rules_handled_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, bytes, bytes, bytes, RuleName, Turn, Tick]]:
			return [
				(
					d["character"],
					d["orig"],
					d["dest"],
					d["rulebook"],
					d["rule"],
					d["turn"],
					d["tick"],
				)
				for d in sorted(
					self._iter_part_tick_to_end(
						"portal_rules_handled", branch, turn_from, tick_from
					),
					key=lambda d: (
						d["turn"],
						d["tick"],
						d["character"],
						d["orig"],
						d["dest"],
						d["rulebook"],
						d["rule"],
					),
				)
			]

		def load_portal_rules_handled_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, bytes, bytes, RuleName, Turn, Tick]]:
			return [
				(
					d["character"],
					d["orig"],
					d["dest"],
					d["rulebook"],
					d["rule"],
					d["turn"],
					d["tick"],
				)
				for d in sorted(
					self._list_part_tick_to_tick(
						"portal_rules_handled",
						branch,
						turn_from,
						tick_from,
						turn_to,
						tick_to,
					),
					key=lambda d: (
						d["turn"],
						d["tick"],
						d["character"],
						d["orig"],
						d["dest"],
						d["rulebook"],
						d["rule"],
					),
				)
			]

		def load_units_tick_to_end(
			self, branch: Branch, turn_from: Turn, tick_from: Tick
		) -> list[tuple[bytes, bytes, bytes, int, int, bool]]:
			return [
				(
					d["character_graph"],
					d["unit_graph"],
					d["unit_node"],
					d["turn"],
					d["tick"],
					d["is_unit"],
				)
				for d in sorted(
					self._iter_part_tick_to_end(
						"units", branch, turn_from, tick_from
					),
					key=lambda d: (
						d["turn"],
						d["tick"],
						d["character_graph"],
						d["unit_graph"],
						d["unit_node"],
					),
				)
			]

		def load_units_tick_to_tick(
			self,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn,
			tick_to: Tick,
		) -> list[tuple[bytes, bytes, bytes, Turn, Tick, bool]]:
			return [
				(
					d["character_graph"],
					d["unit_graph"],
					d["unit_node"],
					d["turn"],
					d["tick"],
					d["is_unit"],
				)
				for d in self._list_part_tick_to_tick(
					"units", branch, turn_from, tick_from, turn_to, tick_to
				)
			]

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

		def create_rule(self, rule: RuleName) -> bool:
			import pyarrow.compute as pc

			db = self._get_db("rules")
			create = not bool(
				db.read(filters=[pc.field("rule") == rule]).num_rows
			)
			if create:
				db.create([{"rule": rule}])
			return create

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
				except:
					msg = f"called {inst}; got exception {repr(ex)}"
				print(msg, file=sys.stderr)
				sys.exit(msg)

			inq = self.inq
			outq = self.outq

			def call_method(name, *args, silent=False, **kwargs):
				if callable(name):
					mth = name
				else:
					mth = getattr(self, name)
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
	) -> tuple[UniversalKeyframe, RuleKeyframe, RulebookKeyframe]:
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

	def graphs_types(
		self,
		branch: Branch,
		turn_from: Turn,
		tick_from: Tick,
		turn_to: Optional[Turn] = None,
		tick_to: Optional[Tick] = None,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, str]]:
		unpack = self.unpack
		if turn_to is None:
			if tick_to is not None:
				raise TypeError("Need both or neither of turn_to, tick_to")
			data = self.call(
				"load_graphs_tick_to_end", branch, turn_from, tick_from
			)
		else:
			if tick_to is None:
				raise TypeError("Need both or neither of turn_to, tick_to")
			data = self.call(
				"load_graphs_tick_to_tick",
				branch,
				turn_from,
				tick_from,
				turn_to,
				tick_to,
			)
		for graph, turn, tick, typ in data:
			yield (
				unpack(graph),
				branch,
				turn,
				tick,
				typ,
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

	def global_dump(self) -> Iterator[tuple[Key, Any]]:
		unpack = self.unpack
		yield from (
			(unpack(d["key"]), unpack(d["value"]))
			for d in self.call("dump", "global")
		)

	def get_branch(self) -> Branch:
		v = self.unpack(self.call("get_global", b"\xa6branch"))
		if v is ...:
			mainbranch = Branch(
				self.unpack(self.call("get_global", b"\xa5trunk"))
			)
			if mainbranch is None:
				return Branch("trunk")
			return mainbranch
		return v

	def get_turn(self) -> Turn:
		v = self.unpack(self.call("get_global", b"\xa4turn"))
		if v is ...:
			return Turn(0)
		return v

	def get_tick(self) -> Tick:
		v = self.unpack(self.call("get_global", b"\xa4tick"))
		if v is ...:
			return Tick(0)
		return v

	def turns_dump(self) -> Iterator[tuple[Branch, Turn, Tick, Tick]]:
		for d in self.call("dump", "turns"):
			yield d["branch"], d["turn"], d["end_tick"], d["plan_end_tick"]

	def universals_dump(self) -> Iterator[tuple[Key, Branch, Turn, Tick, Any]]:
		self.flush()
		unpack = self.unpack
		for d in self.call("dump", "universals"):
			yield (
				unpack(d["key"]),
				d["branch"],
				d["turn"],
				d["tick"],
				unpack(d["value"]),
			)

	def rulebooks_dump(
		self,
	) -> Iterator[
		tuple[RulebookName, Branch, Turn, Tick, tuple[list[RuleName], float]]
	]:
		self.flush()
		unpack = self.unpack
		for d in self.call("dump", "rulebooks"):
			yield (
				unpack(d["rulebook"]),
				d["branch"],
				d["turn"],
				d["tick"],
				(unpack(d["rules"]), d["priority"]),
			)

	def rules_dump(self) -> Iterator[RuleName]:
		for d in sorted(self.call("dump", "rules"), key=itemgetter("rule")):
			yield d["rule"]

	def _rule_dump(
		self, typ: Literal["triggers", "prereqs", "actions"]
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, list[RuleFuncName]]]:
		getattr(self, f"_{typ}2set")()
		unpack = self.unpack
		unpacked: dict[
			tuple[RuleName, Branch, Turn, Tick], list[RuleFuncName]
		] = {}
		for d in self.call("dump", "rule_" + typ):
			unpacked[d["rule"], d["branch"], d["turn"], d["tick"]] = unpack(
				d[typ]
			)
		for rule, branch, turn, tick in sorted(unpacked):
			yield rule, branch, turn, tick, unpacked[rule, branch, turn, tick]

	def rule_triggers_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, list[TriggerFuncName]]]:
		return self._rule_dump("triggers")

	def rule_prereqs_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, list[PrereqFuncName]]]:
		return self._rule_dump("prereqs")

	def rule_actions_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, list[ActionFuncName]]]:
		return self._rule_dump("actions")

	def rule_neighborhood_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, RuleNeighborhood]]:
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
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, RuleBig]]:
		self._big2set()
		return iter(
			sorted(
				(d["rule"], d["branch"], d["turn"], d["tick"], d["big"])
				for d in self.call("dump", "rule_big")
			)
		)

	def node_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, NodeName, Branch, Turn, Tick, RulebookName]]:
		self._noderb2set()
		unpack = self.unpack
		return iter(
			sorted(
				(
					unpack(d["character"]),
					unpack(d["node"]),
					d["branch"],
					d["turn"],
					d["tick"],
					unpack(d["rulebook"]),
				)
				for d in self.call("dump", "node_rulebook")
			)
		)

	def portal_rulebook_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, NodeName, Branch, Turn, Tick, RulebookName]
	]:
		self._portrb2set()
		unpack = self.unpack
		return iter(
			sorted(
				(
					unpack(d["character"]),
					unpack(d["orig"]),
					unpack(d["dest"]),
					d["branch"],
					d["turn"],
					d["tick"],
					unpack(d["rulebook"]),
				)
				for d in self.call("dump", "portal_rulebook")
			)
		)

	def rules_insert(self, rule):
		self.call("insert1", "rule", {"rule": rule})

	def _character_rulebook_dump(self, typ: RulebookTypeStr):
		getattr(self, f"_{typ}_rulebook_to_set")()
		unpack = self.unpack
		return iter(
			sorted(
				(
					unpack(d["character"]),
					d["branch"],
					d["turn"],
					d["tick"],
					unpack(d["rulebook"]),
				)
				for d in self.call("dump", f"{typ}_rulebook")
			)
		)

	def character_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		return self._character_rulebook_dump("character")

	def unit_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		return self._character_rulebook_dump("unit")

	def character_thing_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		return self._character_rulebook_dump("character_thing")

	def character_place_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		return self._character_rulebook_dump("character_place")

	def character_portal_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		return self._character_rulebook_dump("character_portal")

	def character_rules_handled_dump(
		self,
	) -> Iterator[tuple[CharName, RulebookName, RuleName, Branch, Turn, Tick]]:
		self._char_rules_handled()
		unpack = self.unpack
		return (
			(
				unpack(d["character"]),
				unpack(d["rulebook"]),
				d["rule"],
				d["branch"],
				d["turn"],
				d["tick"],
			)
			for d in sorted(
				self.call("dump", "character_rules_handled"),
				key=lambda d: (d["branch"], d["turn"], d["tick"]),
			)
		)

	def unit_rules_handled_dump(
		self,
	) -> Iterator[
		tuple[
			CharName,
			CharName,
			NodeName,
			RulebookName,
			RuleName,
			Branch,
			Turn,
			Tick,
		]
	]:
		self._unit_rules_handled_to_set()
		unpack = self.unpack
		return (
			(
				unpack(d["character"]),
				unpack(d["graph"]),
				unpack(d["unit"]),
				unpack(d["rulebook"]),
				d["rule"],
				d["branch"],
				d["turn"],
				d["tick"],
			)
			for d in sorted(
				self.call("dump", "unit_rules_handled"),
				key=lambda d: (d["branch"], d["turn"], d["tick"]),
			)
		)

	def character_thing_rules_handled_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, RulebookName, RuleName, Branch, Turn, Tick]
	]:
		self._char_thing_rules_handled()
		unpack = self.unpack
		return (
			(
				unpack(d["character"]),
				unpack(d["thing"]),
				unpack(d["rulebook"]),
				d["rule"],
				d["branch"],
				d["turn"],
				d["tick"],
			)
			for d in sorted(
				self.call("dump", "character_thing_rules_handled"),
				key=lambda d: (d["branch"], d["turn"], d["tick"]),
			)
		)

	def character_place_rules_handled_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, RulebookName, RuleName, Branch, Turn, Tick]
	]:
		self._char_place_rules_handled()
		unpack = self.unpack
		return (
			(
				unpack(d["character"]),
				unpack(d["place"]),
				unpack(d["rulebook"]),
				d["rule"],
				d["branch"],
				d["turn"],
				d["tick"],
			)
			for d in self.call("dump", "character_place_rules_handled")
		)

	def character_portal_rules_handled_dump(
		self,
	) -> Iterator[
		tuple[
			CharName,
			NodeName,
			NodeName,
			RulebookName,
			RuleName,
			Branch,
			Turn,
			Tick,
		]
	]:
		self.flush()
		unpack = self.unpack
		return (
			(
				unpack(d["character"]),
				unpack(d["orig"]),
				unpack(d["dest"]),
				unpack(d["rulebook"]),
				d["rule"],
				d["branch"],
				d["turn"],
				d["tick"],
			)
			for d in self.call("dump", "character_portal_rules_handled")
		)

	def node_rules_handled_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, RulebookName, RuleName, Branch, Turn, Tick]
	]:
		self._node_rules_handled_to_set()
		unpack = self.unpack
		return (
			(
				unpack(d["character"]),
				unpack(d["node"]),
				unpack(d["rulebook"]),
				d["rule"],
				d["branch"],
				d["turn"],
				d["tick"],
			)
			for d in self.call("dump", "node_rules_handled")
		)

	def portal_rules_handled_dump(
		self,
	) -> Iterator[
		tuple[
			CharName,
			NodeName,
			NodeName,
			RulebookName,
			RuleName,
			Branch,
			Turn,
			Tick,
		]
	]:
		self._portal_rules_handled_to_set()
		unpack = self.unpack
		return (
			(
				unpack(d["character"]),
				unpack(d["orig"]),
				unpack(d["dest"]),
				unpack(d["rulebook"]),
				d["rule"],
				d["branch"],
				d["turn"],
				d["tick"],
			)
			for d in self.call("dump", "portal_rules_handled")
		)

	def things_dump(
		self,
	) -> Iterator[tuple[CharName, NodeName, Branch, Turn, Tick, NodeName]]:
		self._location()
		unpack = self.unpack
		return (
			(
				unpack(d["character"]),
				unpack(d["thing"]),
				d["branch"],
				d["turn"],
				d["tick"],
				unpack(d["location"]),
			)
			for d in self.call("dump", "things")
		)

	def units_dump(
		self,
	) -> Iterator[
		tuple[CharName, CharName, NodeName, Branch, Turn, Tick, bool]
	]:
		self._unitness()
		unpack = self.unpack
		return (
			(
				unpack(d["character_graph"]),
				unpack(d["unit_graph"]),
				unpack(d["unit_node"]),
				d["branch"],
				d["turn"],
				d["tick"],
				d["is_unit"],
			)
			for d in self.call("dump", "units")
		)

	def count_all_table(self, tbl: str) -> int:
		self.flush()
		return self.call("rowcount", tbl)

	def set_rule_triggers(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		triggers: list[TriggerFuncName],
	):
		if not self.create_rule(rule, branch, turn, tick, triggers):
			self._triggers2set.append((rule, branch, turn, tick, triggers))

	def set_rule_prereqs(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		prereqs: list[PrereqFuncName],
	):
		if not self.create_rule(rule, branch, turn, tick, prereqs=prereqs):
			self._prereqs2set.append((rule, branch, turn, tick, prereqs))

	def set_rule_actions(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		actions: list[ActionFuncName],
	):
		if not self.create_rule(rule, branch, turn, tick, actions=actions):
			self._actions2set.append((rule, branch, turn, tick, actions))

	def set_rule_neighborhood(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		neighborhood: RuleNeighborhood,
	):
		if not self.create_rule(
			rule, branch, turn, tick, neighborhood=neighborhood
		):
			self._neighbors2set.append(
				(rule, branch, turn, tick, neighborhood)
			)

	def set_rule_big(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		big: RuleBig,
	) -> None:
		if not self.create_rule(rule, branch, turn, tick, big=big):
			self._big2set.append((rule, branch, turn, tick, big))

	def create_rule(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		triggers: Iterable[TriggerFuncName] = (),
		prereqs: Iterable[PrereqFuncName] = (),
		actions: Iterable[ActionFuncName] = (),
		neighborhood: RuleNeighborhood = None,
		big: RuleBig = False,
	) -> bool:
		if self.call(
			"create_rule",
			rule=rule,
		):
			self._triggers2set.append(
				(rule, branch, turn, tick, list(triggers))
			)
			self._prereqs2set.append((rule, branch, turn, tick, list(prereqs)))
			self._actions2set.append((rule, branch, turn, tick, list(actions)))
			self._neighbors2set.append(
				(rule, branch, turn, tick, neighborhood)
			)
			self._big2set.append((rule, branch, turn, tick, big))
			return True
		return False

	def things_del_time(self, branch: Branch, turn: Turn, tick: Tick):
		self._location.cull(
			lambda c, th, b, r, t, l: (b, r, t) == (branch, turn, tick)
		)
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
			(character, graph, node, branch, turn, tick, is_unit)
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
		for d in self.call("dump", "graph_val"):
			yield (
				unpack(d["graph"]),
				unpack(d["key"]),
				d["branch"],
				d["turn"],
				d["tick"],
				unpack(d["value"]),
			)

	def graph_val_del_time(self, branch: Branch, turn: Turn, tick: Tick):
		self._graphvals2set.cull(
			lambda g, k, b, r, t, v: (b, r, t) == (branch, turn, tick)
		)
		self.call("graph_val_del_time", branch, turn, tick)

	def graphs_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, str]]:
		self.flush()
		unpack = self.unpack
		for d in self.call("dump", "graphs"):
			yield (
				unpack(d["graph"]),
				d["branch"],
				d["turn"],
				d["tick"],
				d["type"],
			)

	def nodes_del_time(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		self._nodes2set.cull(
			lambda g, n, b, r, t, x: (b, r, t) == (branch, turn, tick)
		)
		self.call("nodes_del_time", branch, turn, tick)

	def nodes_dump(self) -> Iterator[NodeRowType]:
		self.flush()
		unpack = self.unpack
		for d in self.call("dump", "nodes"):
			yield (
				unpack(d["graph"]),
				unpack(d["node"]),
				d["branch"],
				d["turn"],
				d["tick"],
				d["extant"],
			)

	def node_val_dump(self) -> Iterator[NodeValRowType]:
		self.flush()
		unpack = self.unpack
		for d in self.call("dump", "node_val"):
			yield (
				unpack(d["graph"]),
				unpack(d["node"]),
				unpack(d["key"]),
				d["branch"],
				d["turn"],
				d["tick"],
				unpack(d["value"]),
			)

	def node_val_del_time(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		self._nodevals2set.cull(
			lambda g, n, k, b, r, t, v: (b, r, t) == (branch, turn, tick)
		)
		self.call("node_val_del_time", branch, turn, tick)

	def edges_dump(self) -> Iterator[EdgeRowType]:
		self._edges2set()
		unpack = self.unpack
		for d in self.call("dump", "edges"):
			yield (
				unpack(d["graph"]),
				unpack(d["orig"]),
				unpack(d["dest"]),
				d["branch"],
				d["turn"],
				d["tick"],
				d["extant"],
			)

	def edges_del_time(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		self._edges2set.cull(
			lambda g, o, d, b, r, t, x: (b, r, t) == (branch, turn, tick)
		)
		self.call("edges_del_time", branch, turn, tick)

	def edge_val_dump(self) -> Iterator[EdgeValRowType]:
		self.flush()
		unpack = self.unpack
		for d in self.call("dump", "edge_val"):
			yield (
				unpack(d["character"]),
				unpack(d["orig"]),
				unpack(d["dest"]),
				d["branch"],
				d["turn"],
				d["tick"],
				unpack(d["value"]),
			)

	def edge_val_del_time(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		self._edgevals2set.cull(
			lambda g, o, d, k, b, r, t, v: (b, r, t) == (branch, turn, tick)
		)
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
		for graph, nodes, edges, graph_val in self.call(
			"all_keyframe_graphs", branch, turn, tick
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
		self._new_keyframes_graphs()
		unpack = self.unpack
		for d in self.call("dump", "keyframes_graphs"):
			yield (
				unpack(d["graph"]),
				d["branch"],
				d["turn"],
				d["tick"],
				unpack(d["nodes"]),
				unpack(d["edges"]),
				unpack(d["graph_val"]),
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
		self._new_keyframe_extensions()
		unpack = self.unpack
		for d in self.call("dump", "keyframe_extensions"):
			yield (
				d["branch"],
				d["turn"],
				d["tick"],
				unpack(d["universal"]),
				unpack(d["rule"]),
				unpack(d["rulebook"]),
			)

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
