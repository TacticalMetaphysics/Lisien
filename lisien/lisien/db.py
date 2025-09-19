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

import builtins
import inspect
import os
import sys
from abc import ABC, abstractmethod
from collections import UserDict, defaultdict
from contextlib import contextmanager
from functools import cached_property, partial, partialmethod, wraps
from itertools import filterfalse, starmap
from operator import itemgetter
from queue import Queue
from sqlite3 import IntegrityError as LiteIntegrityError
from sqlite3 import OperationalError as LiteOperationalError
from threading import Lock, Thread
from types import MethodType
from typing import (
	Any,
	Callable,
	Iterable,
	Iterator,
	Literal,
	MutableMapping,
	Optional,
	TypeAlias,
	Union,
	get_type_hints,
)

from sqlalchemy import (
	BLOB,
	BOOLEAN,
	FLOAT,
	INT,
	TEXT,
	MetaData,
	Select,
	create_engine,
)
from sqlalchemy.exc import IntegrityError as AlchemyIntegrityError
from sqlalchemy.exc import OperationalError as AlchemyOperationalError

import lisien.types

from .alchemy import meta, queries
from .exc import KeyframeError
from .types import (
	ActionFuncName,
	Branch,
	CharDict,
	CharName,
	CharRulebookRowType,
	EdgeKeyframe,
	EdgeRowType,
	EdgeValRowType,
	EternalKey,
	GraphTypeStr,
	GraphValKeyframe,
	GraphValRowType,
	Key,
	Keyframe,
	NodeKeyframe,
	NodeName,
	NodeRowType,
	NodeRulebookRowType,
	NodeValRowType,
	Plan,
	PortalRulebookRowType,
	PrereqFuncName,
	RuleBig,
	RulebookKeyframe,
	RulebookName,
	RulebookPriority,
	RulebookRowType,
	RulebookTypeStr,
	RuleFuncName,
	RuleKeyframe,
	RuleName,
	RuleNeighborhood,
	RuleRowType,
	Stat,
	ThingRowType,
	Tick,
	Time,
	TimeWindow,
	TriggerFuncName,
	Turn,
	UnitRowType,
	UniversalKey,
	UniversalKeyframe,
	UniversalRowType,
	Value,
)
from .util import ELLIPSIS, EMPTY, garbage
from .wrap import DictWrapper, ListWrapper, SetWrapper

if sys.version_info.minor < 11:

	class ExceptionGroup(Exception):
		pass


IntegrityError = (LiteIntegrityError, AlchemyIntegrityError)
OperationalError = (LiteOperationalError, AlchemyOperationalError)


SCHEMAVER_B = b"\xb6_lisien_schema_version"
SCHEMA_VERSION = 2
SCHEMA_VERSION_B = SCHEMA_VERSION.to_bytes(1, "little")


class GlobalKeyValueStore(UserDict):
	"""A dict-like object that keeps its contents in a table.

	Mostly this is for holding the current branch and revision.

	"""

	def __init__(self, qe: AbstractDatabaseConnector, data: dict):
		self.qe = qe
		super().__init__()
		self.data = data

	def __getitem__(self, k: Key) -> Value:
		ret = super().__getitem__(k)
		if ret is ...:
			raise KeyError(k)
		if isinstance(ret, dict):
			return DictWrapper(
				lambda: super().__getitem__(k),
				lambda v: self.__setitem__(k, v),
				self,
				k,
			)
		elif isinstance(ret, list):
			return ListWrapper(
				lambda: super().__getitem__(k),
				lambda v: self.__setitem__(k, v),
				self,
				k,
			)
		elif isinstance(ret, set):
			return SetWrapper(
				lambda: super().__getitem__(k),
				lambda v: self.__setitem__(k, v),
				self,
				k,
			)
		return ret

	def __setitem__(self, k: Key, v: Value):
		if hasattr(v, "unwrap"):
			v = v.unwrap()
		self.qe.global_set(k, v)
		super().__setitem__(k, v)

	def __delitem__(self, k: Key):
		super().__delitem__(k)
		self.qe.global_del(k)


class ConnectionLooper:
	strings: dict
	lock: Lock

	@cached_property
	def existence_lock(self):
		return Lock()

	@cached_property
	def logger(self):
		from logging import getLogger

		return getLogger("lisien." + self.__class__.__name__)

	@abstractmethod
	def run(self):
		pass

	@abstractmethod
	def initdb(self):
		pass

	@abstractmethod
	def commit(self):
		pass

	@abstractmethod
	def close(self):
		pass


class ParquetDBLooper(ConnectionLooper):
	@cached_property
	def schema(self):
		import pyarrow as pa

		sql2parquetdb_type = {
			BLOB: pa.binary,
			FLOAT: pa.float64,
			TEXT: pa.string,
			INT: pa.int64,
			BOOLEAN: pa.bool_,
		}

		return {
			name: [
				(column.name, sql2parquetdb_type[type(column.type)]())
				for column in table.columns
			]
			for (name, table) in meta.tables.items()
		}

	initial = {
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
	_inq: Queue
	_outq: Queue

	def __init__(self, path: str | os.PathLike, inq: Queue, outq: Queue):
		self._inq = inq
		self._outq = outq
		self._schema = {}
		self._path = path
		self.lock = Lock()
		self.existence_lock.acquire(timeout=1)

	@staticmethod
	def echo(*args, **_):
		return args

	def commit(self):
		pass

	def close(self):
		if not self._outq.empty():
			self._outq.join()
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

		return ParquetDB(os.path.join(self._path, table))

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
			db.delete(filters=[pc.field(k) == v for (k, v) in datum.items()])

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

	def set_bookmark(self, key: bytes, branch: Branch, turn: Turn, tick: Tick):
		import pyarrow.compute as pc

		db = self._get_db("bookmarks")
		schema = self._get_schema("bookmarks")
		try:
			id_ = db.read(
				filters=[pc.field("key") == pc.scalar(key)], columns=["id"]
			)["id"][0]
		except IndexError:
			db.create(
				[{"key": key, "branch": branch, "turn": turn, "tick": tick}],
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

	def rulebooks(self) -> set[RulebookName]:
		return set(
			self._get_db("rulebooks").read(columns=["rulebook"])["rulebook"]
		)

	def graphs(self) -> set[CharName]:
		return set(
			name.as_py()
			for name in self._get_db("graphs").read(columns=["graph"])["graph"]
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
				filters=[pc.field("graph") == pc.scalar(graph)], columns=["id"]
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

		return self.filter_get_id(table, filters=[pc.field(keyfield) == value])

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
		self, branch: Branch, turn: Turn, end_tick: Tick, plan_end_tick: Tick
	):
		from pyarrow import compute as pc

		id_ = self.filter_get_id(
			"turns", [pc.field("branch") == branch, pc.field("turn") == turn]
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
			self._iter_universals_tick_to_end(branch, turn_from, tick_from),
			key=lambda t: (t[1], t[2], t[0]),
		)

	def _table_columns(self, table: str) -> list[str]:
		return ["id"] + list(map(itemgetter(0), self.schema[table]))

	def _iter_part_tick_to_end(
		self, table: str, branch: Branch, turn_from: Turn, tick_from: Tick
	) -> Iterator[dict]:
		from pyarrow import compute as pc

		db = self._get_db(table)
		for d in db.read(
			filters=[
				pc.field("branch") == branch,
				pc.field("turn") >= turn_from,
			],
			columns=self._table_columns(table),
		).to_pylist():
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

	def _iter_part_tick_to_tick(
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
			return iter(
				db.read(
					filters=[
						pc.field("branch") == branch,
						pc.field("turn") == turn_from,
						pc.field("tick") >= tick_from,
						pc.field("tick") < tick_to,
					],
					columns=self._table_columns(table),
				).to_pylist()
			)
		else:
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
					yield d

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
				self._iter_part_tick_to_tick(
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
			return self._load_things_tick_to_tick_character(*args, **kwargs)
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
		return [
			(
				d["character"],
				d["thing"],
				d["turn"],
				d["tick"],
				d["location"],
			)
			for d in sorted(
				self._iter_part_tick_to_tick(
					"things", branch, turn_from, tick_from, turn_to, tick_to
				),
				key=lambda d: (
					d["turn"],
					d["tick"],
					d["character"],
					d["thing"],
				),
			)
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
						yield d["thing"], d["turn"], d["tick"], d["location"]
				elif d["turn"] == turn_to:
					if d["tick"] <= tick_to:
						yield d["thing"], d["turn"], d["tick"], d["location"]
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
			self._iter_graph_val_tick_to_end_all(branch, turn_from, tick_from),
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
		self, graph: bytes, branch: Branch, turn_from: Turn, tick_from: Tick
	) -> list[tuple[bytes, Turn, Tick, bytes]]:
		return sorted(
			self._iter_graph_val_tick_to_end_graph(
				graph, branch, turn_from, tick_from
			),
			key=lambda t: (t[1], t[2], t[0]),
		)

	def _iter_graph_val_tick_to_end_graph(
		self, graph: bytes, branch: Branch, turn_from: Turn, tick_from: Tick
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
		for d in self._iter_part_tick_to_tick(
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
		for d in self._iter_part_tick_to_tick(
			"graph_val", branch, turn_from, tick_from, turn_to, tick_to
		):
			yield d["key"], d["turn"], d["tick"], d["value"]

	def _load_nodes_tick_to_end_graph(
		self, graph: bytes, branch: Branch, turn_from: Turn, tick_from: Tick
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
		self, graph: bytes, branch: Branch, turn_from: Turn, tick_from: Tick
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
		for d in self._iter_part_tick_to_tick(
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
		self, graph: bytes, branch: Branch, turn_from: Turn, tick_from: Tick
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
			self._iter_node_val_tick_to_end_all(branch, turn_from, tick_from),
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
		for d in self._iter_part_tick_to_tick(
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
		self, graph: bytes, branch: Branch, turn_from: Turn, tick_from: Tick
	) -> list[tuple[bytes, bytes, Turn, Tick, bool]]:
		return sorted(
			self._iter_edges_tick_to_end_graph(
				graph, branch, turn_from, tick_from
			),
			key=lambda t: (t[2], t[3], t[0], t[1]),
		)

	def _iter_edges_tick_to_end_graph(
		self, graph: bytes, branch: Branch, turn_from: Turn, tick_from: Tick
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
		for d in self._iter_part_tick_to_tick(
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
			self._iter_edge_val_tick_to_end_all(branch, turn_from, tick_from),
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
		self, graph: bytes, branch: Branch, turn_from: Turn, tick_from: Tick
	) -> list[tuple[bytes, bytes, bytes, Turn, Tick, bytes]]:
		return sorted(
			self._iter_edge_val_tick_to_end_graph(
				graph, branch, turn_from, tick_from
			),
			key=lambda t: (t[3], t[4], t[0], t[1], t[2]),
		)

	def _iter_edge_val_tick_to_end_graph(
		self, graph: bytes, branch: Branch, turn_from: Turn, tick_from: Tick
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
		for d in self._iter_part_tick_to_tick(
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
		for d in self._iter_part_tick_to_tick(
			f"{part}_rulebook", branch, turn_from, tick_from, turn_to, tick_to
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
			self._iter_node_rulebook_tick_to_end(branch, turn_from, tick_from),
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
		for d in self._iter_part_tick_to_tick(
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
		for d in self._iter_part_tick_to_tick(
			"portal_rulebook", branch, turn_from, tick_from, turn_to, tick_to
		):
			yield (
				d["character"],
				d["orig"],
				d["dest"],
				d["turn"],
				d["tick"],
				d["rulebook"],
			)

	def _del_time(self, table: str, branch: Branch, turn: Turn, tick: Tick):
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
		for d in self._iter_part_tick_to_tick(
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
	) -> list[tuple[RuleName, Turn, Tick, bytes | RuleBig]]:
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
	) -> list[tuple[str, int, int, bytes | RuleBig]]:
		return sorted(
			self._iter_rule_part_tick_to_tick(
				part, branch, turn_from, tick_from, turn_to, tick_to
			),
			key=lambda t: (t[1], t[2], t[0]),
		)

	def _iter_rule_part_tick_to_tick(
		self,
		part,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> Iterator[tuple[str, int, int, bytes]]:
		for d in self._iter_part_tick_to_tick(
			f"rule_{part}", branch, turn_from, tick_from, turn_to, tick_to
		):
			yield d["rule"], d["turn"], d["tick"], d[part]

	def load_rule_triggers_tick_to_end(
		self, branch, turn_from, tick_from
	) -> list[tuple[str, int, int, bytes]]:
		return self._load_rule_part_tick_to_end(
			"triggers", branch, turn_from, tick_from
		)

	def load_rule_triggers_tick_to_tick(
		self,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> list[tuple[str, int, int, bytes]]:
		return self._load_rule_part_tick_to_tick(
			"triggers", branch, turn_from, tick_from, turn_to, tick_to
		)

	def load_rule_prereqs_tick_to_end(
		self, branch: str, turn_from: int, tick_from: int
	) -> list[tuple[str, int, int, bytes]]:
		return self._load_rule_part_tick_to_end(
			"prereqs", branch, turn_from, tick_from
		)

	def load_rule_prereqs_tick_to_tick(
		self,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> list[tuple[str, int, int, bytes]]:
		return self._load_rule_part_tick_to_tick(
			"prereqs", branch, turn_from, tick_from, turn_to, tick_to
		)

	def load_rule_actions_tick_to_end(
		self, branch: str, turn_from: int, tick_from: int
	) -> list[tuple[str, int, int, bytes]]:
		return self._load_rule_part_tick_to_end(
			"actions", branch, turn_from, tick_from
		)

	def load_rule_actions_tick_to_tick(
		self,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> list[tuple[str, int, int, bytes]]:
		return self._load_rule_part_tick_to_tick(
			"actions", branch, turn_from, tick_from, turn_to, tick_to
		)

	def load_rule_neighborhoods_tick_to_end(
		self, branch: str, turn_from: int, tick_from: int
	) -> list[tuple[str, int, int, bytes]]:
		return self._load_rule_part_tick_to_end(
			"neighborhood", branch, turn_from, tick_from
		)

	def load_rule_neighborhoods_tick_to_tick(
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
		for d in self._iter_part_tick_to_tick(
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
				self._iter_part_tick_to_tick(
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
				self._iter_part_tick_to_tick(
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
				self._iter_part_tick_to_tick(
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
				self._iter_part_tick_to_tick(
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
				self._iter_part_tick_to_tick(
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
				self._iter_part_tick_to_tick(
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
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
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
			for d in self._iter_part_tick_to_tick(
				"units", branch, turn_from, tick_from, turn_to, tick_to
			)
		]

	def get_keyframe_extensions(
		self, branch: str, turn: int, tick: int
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

	def all_keyframe_graphs(self, branch: str, turn: int, tick: int):
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
		create = not bool(db.read(filters=[pc.field("rule") == rule]).num_rows)
		if create:
			db.create([{"rule": rule}])
		return create

	def set_rulebook(
		self,
		rulebook: bytes,
		branch: str,
		turn: int,
		tick: int,
		rules: bytes,
		priority: float,
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
				pc.field(key) == value for (key, value) in named_data.items()
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

		inq = self._inq
		outq = self._outq

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


def mutexed(func):
	"""Decorator for when an entire method's body holds a mutex lock"""

	@wraps(func)
	def mutexy(self, *args, **kwargs):
		with self.mutex():
			return func(self, *args, **kwargs)

	return mutexy


LoadedCharWindow: TypeAlias = dict[
	Literal[
		"nodes",
		"edges",
		"graph_val",
		"node_val",
		"edge_val",
		"things",
		"units",
		"character_rulebook",
		"unit_rulebook",
		"character_thing_rulebook",
		"character_place_rulebook",
		"character_portal_rulebook",
		"node_rulebook",
		"portal_rulebook",
	],
	list[NodeRowType]
	| list[EdgeRowType]
	| list[GraphValRowType]
	| list[NodeValRowType]
	| list[EdgeValRowType]
	| list[ThingRowType]
	| list[UnitRowType]
	| list[CharRulebookRowType]
	| list[NodeRulebookRowType]
	| list[PortalRulebookRowType],
]


class Batch(list):
	validate: bool = True
	"""Whether to check that records added to the batch are correctly typed tuples"""

	_hint2type = {}

	def __init__(
		self,
		qe: AbstractDatabaseConnector,
		table: str,
		key_len: int,
		inc_rec_counter: bool,
		serialize_record: callable,
	):
		super().__init__()
		self._qe = qe
		self.table = table
		self.key_len = key_len
		self.inc_rec_counter = inc_rec_counter
		self.serialize_record = serialize_record
		self.argspec = inspect.getfullargspec(self.serialize_record)

	def cull(self, condition: Callable[..., bool]) -> None:
		"""Remove records matching a condition from the batch

		Records are unpacked before being passed into the condition function.

		"""
		datta = list(self)
		self.clear()
		self.extend(
			filterfalse(
				partial(self._call_with_unpacked_tuple, condition), datta
			)
		)

	@staticmethod
	def _call_with_unpacked_tuple(func, tup):
		return func(*tup)

	def _validate(self, t: tuple):
		def deannotate(annotation):
			if "|" in annotation:
				for a in annotation.split("|"):
					yield from deannotate(a.strip())
				return
			if "Literal" == annotation[:7]:
				for a in annotation[7:].strip("[]").split(", "):
					yield from deannotate(a)
				return
			elif "[" in annotation:
				annotation = annotation[: annotation.index("[")]
			if hasattr(builtins, annotation):
				typ = getattr(builtins, annotation)
				if not isinstance(typ, type):
					typ = type(typ)
			elif annotation in ("type(...)", "..."):
				yield type(...)
				return
			else:
				typ = getattr(lisien.types, annotation)
			if hasattr(typ, "__supertype__"):
				typ = typ.__supertype__
			if hasattr(typ, "__origin__"):
				if typ.__origin__ is Union:
					for arg in typ.__args__:
						yield getattr(arg, "__origin__", arg)
				elif typ.__origin__ is Literal:
					yield from map(type, typ.__args__)
				else:
					yield typ.__origin__
			else:
				yield typ

		if not isinstance(t, tuple):
			raise TypeError("Can only batch tuples")
		if len(t) != len(self.argspec.args) - 1:  # exclude self
			raise TypeError(
				f"Need a tuple of length {len(self.argspec.args) - 1}, not {len(t)}"
			)
		for i, (name, value) in enumerate(zip(self.argspec.args[1:], t)):
			annot = self.argspec.annotations[name]

			if not isinstance(value, tuple(deannotate(annot))):
				raise TypeError(
					f"Tuple element {i} is of type {type(value)};"
					f" should be {self.argspec.annotations[name]}"
				)

	def __setitem__(self, i: int, v):
		if self.validate:
			self._validate(v)
		super().__setitem__(i, v)

	def insert(self, i: int, v):
		if self.validate:
			self._validate(v)
		super().insert(i, v)

	def append(self, v):
		if self.validate:
			self._validate(v)
		super().append(v)

	def __call__(self):
		if not self:
			return 0
		if self.key_len:
			deduplicated = {
				rec[: self.key_len]: rec[self.key_len :] for rec in self
			}
			records = starmap(
				self.serialize_record,
				((*key, *value) for (key, value) in deduplicated.items()),
			)
		else:
			records = starmap(self.serialize_record, self)
		data = list(records)
		argnames = self.argspec.args[1:]
		if self.key_len:
			self._qe.delete_many_silent(
				self.table,
				[
					dict(zip(argnames[: self.key_len], datum))
					for datum in {rec[: self.key_len] for rec in data}
				],
			)
		self._qe.insert_many_silent(
			self.table, [dict(zip(argnames, datum)) for datum in data]
		)
		n = len(data)
		self.clear()
		if self.inc_rec_counter:
			self._qe._increc(n)
		return n


def batched(
	table: str,
	serialize_record: Callable | None = None,
	*,
	key_len: int = 0,
	inc_rec_counter: bool = True,
) -> partial | cached_property:
	if serialize_record is None:
		return partial(
			batched,
			table,
			key_len=key_len,
			inc_rec_counter=inc_rec_counter,
		)
	serialized_tuple_type = get_type_hints(serialize_record)["return"]

	def the_batch(
		self,
	) -> Batch[serialized_tuple_type]:
		return Batch(
			self,
			table,
			key_len,
			inc_rec_counter,
			MethodType(serialize_record, self),
		)

	return cached_property(the_batch)


class AbstractDatabaseConnector(ABC):
	pack: callable
	unpack: callable
	looper_cls: type[ConnectionLooper]
	eternal: MutableMapping
	kf_interval_override: Callable[[Any], bool | None] = lambda _: None
	keyframe_interval: int | None
	snap_keyframe: callable
	all_rules: set[RuleName]
	_inq: Queue
	_outq: Queue
	_looper: looper_cls
	_records: int

	@abstractmethod
	def __init__(
		self, dbstring, connect_args, pack=None, unpack=None, *, clear=False
	): ...

	@batched(
		"global",
		key_len=1,
		inc_rec_counter=False,
	)
	def _eternal2set(
		self, key: EternalKey, value: Value
	) -> tuple[bytes, bytes]:
		pack = self.pack
		return pack(key), pack(value)

	@batched(
		"branches",
		key_len=1,
		inc_rec_counter=False,
	)
	def _branches2set(
		self,
		branch: Branch,
		parent: Branch | None,
		parent_turn: Turn,
		parent_tick: Tick,
		end_turn: Turn,
		end_tick: Tick,
	) -> tuple[Branch, Branch | None, Turn, Tick, Turn, Tick]:
		return branch, parent, parent_turn, parent_tick, end_turn, end_tick

	@batched("turns", key_len=2)
	def _turns2set(
		self, branch: Branch, turn: Turn, end_tick: Tick, plan_end_tick: Tick
	) -> tuple[Branch, Turn, Tick, Tick]:
		return (branch, turn, end_tick, plan_end_tick)

	@batched(
		"turns_completed",
		key_len=1,
	)
	def _turns_completed_to_set(
		self, branch: Branch, turn: Turn
	) -> tuple[Branch, Turn]:
		return (branch, turn)

	def complete_turn(
		self, branch: Branch, turn: Turn, discard_rules: bool = False
	) -> None:
		self._turns_completed_to_set.append((branch, turn))
		if discard_rules:
			self._char_rules_handled.clear()
			self._unit_rules_handled.clear()
			self._char_thing_rules_handled.clear()
			self._char_place_rules_handled.clear()
			self._char_portal_rules_handled.clear()
			self._node_rules_handled.clear()
			self._portal_rules_handled.clear()

	@batched("plan_ticks", inc_rec_counter=False)
	def _planticks2set(
		self, plan_id: Plan, branch: Branch, turn: Turn, tick: Tick
	) -> tuple[Plan, Branch, Turn, Tick]:
		return plan_id, branch, turn, tick

	@batched("bookmarks", key_len=1, inc_rec_counter=False)
	def _bookmarks2set(
		self, key: str, branch: Branch, turn: Turn, tick: Tick
	) -> tuple[str, Branch, Turn, Tick]:
		return (key, branch, turn, tick)

	def set_bookmark(
		self, key: str, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		self._bookmarks2set.append((key, branch, turn, tick))

	@abstractmethod
	def del_bookmark(self, key: str) -> None: ...

	@batched("universals", key_len=4)
	def _universals2set(
		self,
		key: UniversalKey,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
	) -> tuple[bytes, Branch, Turn, Tick, bytes]:
		pack = self.pack
		return pack(key), branch, turn, tick, pack(value)

	@batched("rule_triggers", key_len=4)
	def _triggers2set(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		triggers: list[TriggerFuncName],
	) -> tuple[RuleName, Branch, Turn, Tick, bytes]:
		return (rule, branch, turn, tick, self.pack(triggers))

	@batched("rule_prereqs", key_len=4)
	def _prereqs2set(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		prereqs: list[PrereqFuncName],
	) -> tuple[RuleName, Branch, Turn, Tick, bytes]:
		return (rule, branch, turn, tick, self.pack(prereqs))

	@batched("rule_actions", key_len=4)
	def _actions2set(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		actions: list[ActionFuncName],
	) -> tuple[RuleName, Branch, Turn, Tick, bytes]:
		return (rule, branch, turn, tick, self.pack(actions))

	@batched(
		"rule_neighborhood",
		key_len=4,
	)
	def _neighbors2set(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		neighborhood: RuleNeighborhood,
	) -> tuple[RuleName, Branch, Turn, Tick, RuleNeighborhood]:
		return (rule, branch, turn, tick, neighborhood)

	@batched("rule_big", key_len=4)
	def _big2set(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		big: RuleBig,
	) -> tuple[RuleName, Branch, Turn, Tick, RuleBig]:
		return (rule, branch, turn, tick, big)

	@batched("rulebooks", key_len=4)
	def _rulebooks2set(
		self,
		rulebook: RulebookName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rules: Iterable[RuleName] = (),
		priority: RulebookPriority = 0.0,
	) -> tuple[bytes, Branch, Turn, Tick, bytes, RulebookPriority]:
		return (
			self.pack(rulebook),
			branch,
			turn,
			tick,
			self.pack(rules),
			priority,
		)

	@batched("graphs", key_len=4)
	def _graphs2set(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		type: GraphTypeStr,
	) -> tuple[bytes, Branch, Turn, Tick, GraphTypeStr]:
		return self.pack(graph), branch, turn, tick, type

	@batched(
		"character_rulebook",
		key_len=4,
	)
	def _character_rulebooks_to_set(
		self,
		character: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rulebook: RulebookName,
	) -> tuple[bytes, Branch, Turn, Tick, bytes]:
		pack = self.pack
		return pack(character), branch, turn, tick, pack(rulebook)

	@batched("unit_rulebook", key_len=4)
	def _unit_rulebooks_to_set(
		self,
		character: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rulebook: RulebookName,
	) -> tuple[bytes, Branch, Turn, Tick, bytes]:
		pack = self.pack
		return pack(character), branch, turn, tick, pack(rulebook)

	@batched(
		"character_thing_rulebook",
		key_len=4,
	)
	def _character_thing_rulebooks_to_set(
		self,
		character: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rulebook: RulebookName,
	) -> tuple[bytes, Branch, Turn, Tick, bytes]:
		pack = self.pack
		return pack(character), branch, turn, tick, pack(rulebook)

	@batched(
		"character_place_rulebook",
		key_len=4,
	)
	def _character_place_rulebooks_to_set(
		self,
		character: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rulebook: RulebookName,
	) -> tuple[bytes, Branch, Turn, Tick, bytes]:
		pack = self.pack
		return pack(character), branch, turn, tick, pack(rulebook)

	@batched(
		"character_portal_rulebook",
		key_len=4,
	)
	def _character_portal_rulebooks_to_set(
		self,
		character: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rulebook: RulebookName,
	) -> tuple[bytes, Branch, Turn, Tick, bytes]:
		pack = self.pack
		return pack(character), branch, turn, tick, pack(rulebook)

	@batched("node_rulebook", key_len=5)
	def _noderb2set(
		self,
		character: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rulebook: RulebookName,
	) -> tuple[bytes, bytes, Branch, Turn, Tick, bytes]:
		pack = self.pack
		return pack(character), pack(node), branch, turn, tick, pack(rulebook)

	@batched(
		"portal_rulebook",
		key_len=6,
	)
	def _portrb2set(
		self,
		character: CharName,
		orig: NodeName,
		dest: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rulebook: RulebookName,
	) -> tuple[bytes, bytes, bytes, Branch, Turn, Tick, bytes]:
		pack = self.pack
		return (
			pack(character),
			pack(orig),
			pack(dest),
			branch,
			turn,
			tick,
			pack(rulebook),
		)

	@batched("nodes", key_len=5)
	def _nodes2set(
		self,
		graph: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		extant: bool,
	) -> tuple[bytes, bytes, Branch, Turn, Tick, bool]:
		pack = self.pack
		return pack(graph), pack(node), branch, turn, tick, bool(extant)

	@batched("edges", key_len=6)
	def _edges2set(
		self,
		graph: CharName,
		orig: NodeName,
		dest: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		extant: bool,
	) -> tuple[bytes, bytes, bytes, Branch, Turn, Tick, bool]:
		pack = self.pack
		return (
			pack(graph),
			pack(orig),
			pack(dest),
			branch,
			turn,
			tick,
			bool(extant),
		)

	@batched("node_val", key_len=6)
	def _nodevals2set(
		self,
		graph: CharName,
		node: NodeName,
		key: Stat,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
	) -> tuple[bytes, bytes, bytes, Branch, Turn, Tick, bytes]:
		pack = self.pack
		return (
			pack(graph),
			pack(node),
			pack(key),
			branch,
			turn,
			tick,
			pack(value),
		)

	@batched("edge_val", key_len=7)
	def _edgevals2set(
		self,
		graph: CharName,
		orig: NodeName,
		dest: NodeName,
		key: Stat,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
	) -> tuple[bytes, bytes, bytes, bytes, Branch, Turn, Tick, bytes]:
		pack = self.pack
		return (
			pack(graph),
			pack(orig),
			pack(dest),
			pack(key),
			branch,
			turn,
			tick,
			pack(value),
		)

	@batched("graph_val", key_len=5)
	def _graphvals2set(
		self,
		graph: CharName,
		key: Stat,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
	) -> tuple[bytes, bytes, Branch, Turn, Tick, bytes]:
		pack = self.pack
		return pack(graph), pack(key), branch, turn, tick, pack(value)

	@batched(
		"keyframes",
		key_len=3,
		inc_rec_counter=False,
	)
	def _new_keyframes(self, branch: Branch, turn: Turn, tick: Tick) -> Time:
		return branch, turn, tick

	@batched(
		"keyframes_graphs",
		key_len=4,
		inc_rec_counter=False,
	)
	def _new_keyframes_graphs(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		nodes: NodeKeyframe,
		edges: EdgeKeyframe,
		graph_val: GraphValKeyframe,
	) -> tuple[bytes, Branch, Turn, Tick, bytes, bytes, bytes]:
		pack = self.pack
		return (
			pack(graph),
			branch,
			turn,
			tick,
			pack(nodes),
			pack(edges),
			pack(graph_val),
		)

	@batched(
		"keyframe_extensions",
		key_len=3,
		inc_rec_counter=False,
	)
	def _new_keyframe_extensions(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		universal: UniversalKeyframe,
		rule: RuleKeyframe,
		rulebook: RulebookKeyframe,
	) -> tuple[Branch, Turn, Tick, bytes, bytes, bytes]:
		pack = self.pack
		return branch, turn, tick, pack(universal), pack(rule), pack(rulebook)

	@batched("character_rules_handled", inc_rec_counter=False)
	def _char_rules_handled(
		self,
		character: CharName,
		rulebook: RulebookName,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> tuple[bytes, bytes, RuleName, Branch, Turn, Tick]:
		(character, rulebook) = map(self.pack, (character, rulebook))
		return (character, rulebook, rule, branch, turn, tick)

	@batched("unit_rules_handled", inc_rec_counter=False)
	def _unit_rules_handled(
		self,
		character: CharName,
		rulebook: RulebookName,
		rule: RuleName,
		graph: CharName,
		unit: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> tuple[bytes, bytes, bytes, bytes, RuleName, Branch, Turn, Tick]:
		character, graph, unit, rulebook = map(
			self.pack, (character, graph, unit, rulebook)
		)
		return character, rulebook, rule, graph, unit, branch, turn, tick

	@batched("character_thing_rules_handled", inc_rec_counter=False)
	def _char_thing_rules_handled(
		self,
		character: CharName,
		rulebook: RulebookName,
		rule: RuleName,
		thing: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> tuple[bytes, bytes, RuleName, bytes, Branch, Turn, Tick]:
		character, thing, rulebook = map(
			self.pack, (character, thing, rulebook)
		)
		return (character, rulebook, rule, thing, branch, turn, tick)

	@batched("character_place_rules_handled", inc_rec_counter=False)
	def _char_place_rules_handled(
		self,
		character: CharName,
		place: NodeName,
		rulebook: RulebookName,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> tuple[bytes, bytes, bytes, RuleName, Branch, Turn, Tick]:
		character, rulebook, place = map(
			self.pack, (character, rulebook, place)
		)
		return (character, place, rulebook, rule, branch, turn, tick)

	@batched("character_portal_rules_handled", inc_rec_counter=False)
	def _char_portal_rules_handled(
		self,
		character: CharName,
		orig: NodeName,
		dest: NodeName,
		rulebook: RulebookName,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> tuple[bytes, bytes, bytes, bytes, RuleName, Branch, Turn, Tick]:
		character, rulebook, orig, dest = map(
			self.pack, (character, rulebook, orig, dest)
		)
		return character, orig, dest, rulebook, rule, branch, turn, tick

	@batched("node_rules_handled", inc_rec_counter=False)
	def _node_rules_handled(
		self,
		character: CharName,
		node: NodeName,
		rulebook: RulebookName,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> tuple[bytes, bytes, bytes, RuleName, Branch, Turn, Tick]:
		character, rulebook, node = map(self.pack, (character, rulebook, node))
		return character, node, rulebook, rule, branch, turn, tick

	@batched("portal_rules_handled", inc_rec_counter=False)
	def _portal_rules_handled(
		self,
		character: CharName,
		orig: NodeName,
		dest: NodeName,
		rulebook: RulebookName,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> tuple[bytes, bytes, bytes, bytes, RuleName, Branch, Turn, Tick]:
		(character, orig, dest, rulebook) = map(
			self.pack, (character, orig, dest, rulebook)
		)
		return character, orig, dest, rulebook, rule, branch, turn, tick

	@batched("units", key_len=6)
	def _unitness(
		self,
		character_graph: CharName,
		unit_graph: CharName,
		unit_node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		is_unit: bool,
	) -> tuple[bytes, bytes, bytes, Branch, Turn, Tick, bool]:
		(character_graph, unit_graph, unit_node) = map(
			self.pack, (character_graph, unit_graph, unit_node)
		)
		return (
			character_graph,
			unit_graph,
			unit_node,
			branch,
			turn,
			tick,
			is_unit,
		)

	@batched("things", key_len=5)
	def _location(
		self,
		character: CharName,
		thing: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		location: NodeName | type(...),
	) -> tuple[bytes, bytes, Branch, Turn, Tick, bytes]:
		(character, thing, location) = map(
			self.pack, (character, thing, location)
		)
		return character, thing, branch, turn, tick, location

	def universal_set(
		self,
		key: UniversalKey,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
	) -> None:
		self._universals2set.append((key, branch, turn, tick, value))

	def universal_del(
		self, key: UniversalKey, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		self.universal_set(key, branch, turn, tick, None)

	def exist_node(
		self,
		graph: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		extant: bool,
	) -> None:
		self._nodes2set.append((graph, node, branch, turn, tick, extant))

	@cached_property
	def _all_keyframe_times(self):
		return set(self.keyframes_dump())

	def keyframe_insert(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		self._new_keyframes.append((branch, turn, tick))
		self._all_keyframe_times.add((branch, turn, tick))

	def keyframe_graph_insert(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		nodes: NodeKeyframe,
		edges: EdgeKeyframe,
		graph_val: CharDict,
	) -> None:
		self._new_keyframes_graphs.append(
			(graph, branch, turn, tick, nodes, edges, graph_val)
		)

	def keyframe_extension_insert(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		universal: UniversalKeyframe,
		rule: RuleKeyframe,
		rulebook: RulebookKeyframe,
	):
		self._new_keyframe_extensions.append(
			(
				branch,
				turn,
				tick,
				universal,
				rule,
				rulebook,
			)
		)

	def node_val_set(
		self,
		graph: CharName,
		node: NodeName,
		key: Key,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
	):
		self._nodevals2set.append(
			(graph, node, key, branch, turn, tick, value)
		)

	def edge_val_set(
		self,
		graph: CharName,
		orig: NodeName,
		dest: NodeName,
		key: Key,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
	) -> None:
		self._edgevals2set.append(
			(graph, orig, dest, key, branch, turn, tick, value)
		)

	def plans_insert(
		self, plan_id: Plan, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		self._planticks2set.append((plan_id, branch, turn, tick))

	def plans_insert_many(
		self, many: list[tuple[Plan, Branch, Turn, Tick]]
	) -> None:
		self._planticks2set.extend(many)

	@garbage
	def flush(self):
		"""Put all pending changes into the SQL transaction, or write to disk."""
		if (wat := self.echo("ready")) != "ready":
			raise RuntimeError("Not ready to flush", wat)
		self._flush()
		if (wat := self.echo("flushed")) != "flushed":
			raise RuntimeError("Failed flush", wat)

	@mutexed
	def _flush(self):
		for att in dir(self.__class__):
			attr = getattr(self.__class__, att)
			if not isinstance(attr, cached_property):
				continue
			batch = getattr(self, att)
			if isinstance(batch, Batch):
				batch()

	@cached_property
	def logger(self):
		from logging import getLogger

		return getLogger("lisien." + self.__class__.__name__)

	def log(self, level, msg, *args):
		self.logger.log(level, msg, *args)

	def debug(self, msg, *args):
		self.logger.debug(msg, *args)

	def info(self, msg, *args):
		self.logger.info(msg, *args)

	def warning(self, msg, *args):
		self.logger.warning(msg, *args)

	def error(self, msg, *args):
		self.logger.error(msg, *args)

	def critical(self, msg, *args):
		self.logger.critical(msg, *args)

	def echo(self, string: str) -> str:
		with self.mutex():
			self._inq.put(("echo", string))
			ret = self._outq.get()
			self._outq.task_done()
			return ret

	@abstractmethod
	def call(self, query_name: str, *args, **kwargs): ...

	@abstractmethod
	def call_silent(self, query_name: str, *args, **kwargs): ...

	@abstractmethod
	def call_many(self, query_name: str, args: list) -> None: ...

	@abstractmethod
	def call_many_silent(self, query_name: str, args: list) -> None: ...

	@abstractmethod
	def insert_many(self, table_name: str, args: list[dict]) -> None: ...

	@abstractmethod
	def insert_many_silent(
		self, table_name: str, args: list[dict]
	) -> None: ...

	@abstractmethod
	def delete_many_silent(
		self, table_name: str, args: list[dict]
	) -> None: ...

	@abstractmethod
	def get_keyframe_extensions(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> tuple[UniversalKeyframe, RuleKeyframe, RulebookKeyframe]:
		pass

	@abstractmethod
	def keyframes_dump(self) -> Iterator[tuple[Branch, Turn, Tick]]:
		pass

	@abstractmethod
	def delete_keyframe(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		pass

	def graphs_insert(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		type: GraphTypeStr,
	) -> None:
		self._graphs2set.append((graph, branch, turn, tick, type))

	def graph_val_set(
		self,
		graph: CharName,
		key: Key,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		val: Value,
	) -> None:
		self._graphvals2set.append((graph, key, branch, turn, tick, val))

	def exist_edge(
		self,
		graph: CharName,
		orig: NodeName,
		dest: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		extant: bool,
	) -> None:
		self._edges2set.append((graph, orig, dest, branch, turn, tick, extant))

	@abstractmethod
	def keyframes_graphs(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick]]:
		pass

	@abstractmethod
	def have_branch(self, branch: Branch) -> bool:
		pass

	@abstractmethod
	def branches_dump(
		self,
	) -> Iterator[tuple[Branch, Branch, Turn, Tick, Turn, Tick]]:
		pass

	@abstractmethod
	def global_get(self, key: Key) -> Value:
		pass

	@abstractmethod
	def global_dump(self) -> Iterator[tuple[Key, Value]]:
		pass

	@abstractmethod
	def get_branch(self) -> Branch:
		pass

	@abstractmethod
	def get_turn(self) -> Turn:
		pass

	@abstractmethod
	def get_tick(self) -> Tick:
		pass

	def global_set(self, key: EternalKey, value: Value):
		self._eternal2set.append((key, value))

	def global_del(self, key: Key) -> None:
		self._eternal2set.append((key, ...))

	def set_branch(
		self,
		branch: Branch,
		parent: Branch,
		parent_turn: Turn,
		parent_tick: Tick,
		end_turn: Turn,
		end_tick: Tick,
	) -> None:
		self._branches2set.append(
			(branch, parent, parent_turn, parent_tick, end_turn, end_tick)
		)

	def set_turn(
		self, branch: Branch, turn: Turn, end_tick: Tick, plan_end_tick: Tick
	) -> None:
		self._turns2set.append((branch, turn, end_tick, plan_end_tick))

	@abstractmethod
	def turns_dump(self) -> Iterator[tuple[Branch, Turn, Tick, Tick]]:
		pass

	@abstractmethod
	def graph_val_dump(self) -> Iterator[GraphValRowType]:
		pass

	@abstractmethod
	def graphs_types(
		self,
		branch: Branch,
		turn_from: Turn,
		tick_from: Tick,
		turn_to: Optional[Turn] = None,
		tick_to: Optional[Tick] = None,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, str]]:
		pass

	@abstractmethod
	def characters(self) -> Iterator[tuple[CharName, Branch, Turn, Tick, str]]:
		pass

	@abstractmethod
	def nodes_del_time(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		pass

	@abstractmethod
	def nodes_dump(self) -> Iterator[NodeRowType]:
		pass

	@abstractmethod
	def node_val_dump(self) -> Iterator[NodeValRowType]:
		pass

	@abstractmethod
	def node_val_del_time(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		pass

	@abstractmethod
	def edges_dump(self) -> Iterator[EdgeRowType]:
		pass

	@abstractmethod
	def edge_val_dump(self) -> Iterator[EdgeValRowType]:
		pass

	@abstractmethod
	def edge_val_del_time(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		pass

	@abstractmethod
	def plan_ticks_dump(self) -> Iterator[tuple[Plan, Branch, Turn, Tick]]:
		pass

	@abstractmethod
	def commit(self) -> None:
		pass

	@abstractmethod
	def close(self) -> None:
		pass

	@abstractmethod
	def _init_db(self) -> None:
		pass

	@abstractmethod
	def truncate_all(self) -> None:
		pass

	_infixes2load = [
		"graphs",
		"nodes",
		"edges",
		"graph_val",
		"node_val",
		"edge_val",
		"things",
		"units",
		"character_rulebook",
		"unit_rulebook",
		"character_thing_rulebook",
		"character_place_rulebook",
		"character_portal_rulebook",
		"node_rulebook",
		"portal_rulebook",
		"universals",
		"rulebooks",
		"rule_triggers",
		"rule_prereqs",
		"rule_actions",
		"rule_neighborhoods",
		"rule_big",
	]

	def _put_window_tick_to_end(
		self, branch: Branch, turn_from: Turn, tick_from: Tick
	):
		putkwargs = {
			"branch": branch,
			"turn_from": turn_from,
			"tick_from": tick_from,
		}
		for i, infix in enumerate(self._infixes2load):
			self._inq.put(
				(
					"echo",
					(
						"begin",
						infix,
						branch,
						turn_from,
						tick_from,
						None,
						None,
					),
					{},
				)
			)
			self._inq.put(("one", f"load_{infix}_tick_to_end", (), putkwargs))
			self._inq.put(
				(
					"echo",
					("end", infix, branch, turn_from, tick_from, None, None),
					{},
				)
			)

	def _put_window_tick_to_tick(
		self,
		branch: Branch,
		turn_from: Turn,
		tick_from: Tick,
		turn_to: Turn,
		tick_to: Tick,
	):
		putkwargs = {
			"branch": branch,
			"turn_from": turn_from,
			"tick_from": tick_from,
			"turn_to": turn_to,
			"tick_to": tick_to,
		}
		for i, infix in enumerate(self._infixes2load):
			self._inq.put(
				(
					"echo",
					(
						"begin",
						infix,
						branch,
						turn_from,
						tick_from,
						turn_to,
						tick_to,
					),
					{},
				)
			)
			self._inq.put(("one", f"load_{infix}_tick_to_tick", (), putkwargs))
			self._inq.put(
				(
					"echo",
					(
						"end",
						infix,
						branch,
						turn_from,
						tick_from,
						turn_to,
						tick_to,
					),
					{},
				)
			)

	@contextmanager
	def mutex(self):
		with self._looper.lock:
			yield
			if self._outq.qsize() != 0:
				raise RuntimeError("Unhandled items in output queue")

	@mutexed
	def _load_windows_into(self, ret: dict, windows: list[TimeWindow]) -> None:
		for branch, turn_from, tick_from, turn_to, tick_to in windows:
			if turn_to is None:
				self._put_window_tick_to_end(branch, turn_from, tick_from)
			else:
				self._put_window_tick_to_tick(
					branch, turn_from, tick_from, turn_to, tick_to
				)
		self._inq.join()
		for window in windows:
			self._get_one_window(ret, *window)

	def _increc(self, n: int = 1):
		"""Snap a keyframe, if the keyframe interval has passed.

		But the engine can override this behavior when it'd be impractical,
		such as during a rule's execution. This defers the keyframe snap
		until next we get a falsy result from the override function.

		Not to be called directly. Instead, use a batch, likely created via
		the ``@batch`` decorator.

		"""
		if n == 0:
			return
		if n < 0:
			raise ValueError("Don't reduce the count of written records")
		self._records += n
		override: bool | None = self.kf_interval_override()
		if override:
			self._kf_interval_overridden = True
			return
		elif getattr(self, "_kf_interval_overridden", False) or (
			self.keyframe_interval is not None
			and self._records % self.keyframe_interval == 0
		):
			self.snap_keyframe()
			self._kf_interval_overridden = False

	def _get_one_window(
		self,
		ret,
		branch: Branch,
		turn_from: Turn,
		tick_from: Tick,
		turn_to: Turn,
		tick_to: Tick,
	):
		self.debug(
			f"_get_one_window({branch}, {turn_from}, {tick_from}, {turn_to}, {tick_to})"
		)
		unpack = self.unpack
		outq = self._outq
		got = outq.get()
		if got != (
			"begin",
			"graphs",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError("Expected beginning of graphs", got)
		outq.task_done()
		if "graphs" not in ret:
			ret["graphs"] = []
		while isinstance(got := outq.get(), list):
			got: list[tuple[bytes, Turn, Tick, str]]
			for graph, turn, tick, typ_str in got:
				graph: CharName = unpack(graph)
				ret["graphs"].append((graph, branch, turn, tick, typ_str))
			outq.task_done()
		if got != (
			"end",
			"graphs",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError("Expected end of graphs", got)
		outq.task_done()
		got = outq.get()
		if got != (
			"begin",
			"nodes",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError("Expected beginning of nodes", got)
		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[bytes, bytes, Turn, Tick, bool]]
			for graph, node, turn, tick, ex in got:
				(graph, node) = map(unpack, (graph, node))
				graph: CharName
				node: NodeName
				ret[graph]["nodes"].append(
					(graph, node, branch, turn, tick, ex)
				)
			outq.task_done()
		if got != (
			"end",
			"nodes",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				f"Expected {('end', 'nodes', branch, turn_from, tick_from, turn_to, tick_to)}",
				got,
			)
		outq.task_done()
		got = outq.get()
		if got != (
			"begin",
			"edges",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError("Expected beginning of edges", got)
		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[bytes, bytes, bytes, Turn, Tick, bool]]
			for graph, orig, dest, turn, tick, ex in got:
				(graph, orig, dest) = map(unpack, (graph, orig, dest))
				graph: CharName
				orig: NodeName
				dest: NodeName
				ret[graph]["edges"].append(
					(
						graph,
						orig,
						dest,
						branch,
						turn,
						tick,
						ex,
					)
				)
			outq.task_done()
		if got != (
			"end",
			"edges",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError("Expected end of edges", got)

		outq.task_done()
		got = outq.get()
		if got != (
			"begin",
			"graph_val",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError("Expected beginning of graph_val", got)

		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[bytes, bytes, Turn, Tick, bytes]]
			for graph, key, turn, tick, val in got:
				(graph, key, val) = map(unpack, (graph, key, val))
				graph: CharName
				key: Stat
				val: Value
				ret[graph]["graph_val"].append(
					(graph, key, branch, turn, tick, val)
				)
			outq.task_done()
		if got != (
			"end",
			"graph_val",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected end of graph_val",
				got,
			)
		outq.task_done()
		if (got := outq.get()) != (
			"begin",
			"node_val",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected beginning of node_val",
				got,
			)
		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[bytes, bytes, bytes, Turn, Tick, bytes]]
			for graph, node, key, turn, tick, val in got:
				(graph, node, key, val) = map(unpack, (graph, node, key, val))
				graph: CharName
				node: NodeName
				key: Stat
				val: Value
				ret[graph]["node_val"].append(
					(graph, node, key, branch, turn, tick, val)
				)
			outq.task_done()
		if got != (
			"end",
			"node_val",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected end of node_val",
				got,
			)
		outq.task_done()
		if (got := outq.get()) != (
			"begin",
			"edge_val",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected beginning of edge_val",
				got,
			)
		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[bytes, bytes, bytes, bytes, Turn, Tick, bytes]]
			for graph, orig, dest, key, turn, tick, val in got:
				(graph, orig, dest, key, val) = map(
					unpack, (graph, orig, dest, key, val)
				)
				graph: CharName
				orig: NodeName
				dest: NodeName
				key: Stat
				val: Value
				ret[graph]["edge_val"].append(
					(
						graph,
						orig,
						dest,
						key,
						branch,
						turn,
						tick,
						val,
					)
				)
			outq.task_done()
		if got != (
			"end",
			"edge_val",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected end of edge_val",
				got,
			)
		outq.task_done()
		if (got := outq.get()) != (
			"begin",
			"things",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected beginning of things",
				got,
			)
		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[bytes, bytes, Turn, Tick, bytes]]
			for graph, node, turn, tick, loc in got:
				(graph, node, loc) = map(unpack, (graph, node, loc))
				graph: CharName
				node: NodeName
				loc: NodeName
				ret[graph]["things"].append(
					(graph, node, branch, turn, tick, loc)
				)
			outq.task_done()
		if got != (
			"end",
			"things",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected end of things",
				got,
			)
		outq.task_done()
		if (got := outq.get()) != (
			"begin",
			"units",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError("Expected beginning of units", got)
		outq.task_done()
		while isinstance((got := outq.get()), list):
			got: list[tuple[bytes, bytes, bytes, Branch, Turn, Tick, bool]]
			for char, graph, node, turn, tick, is_unit in got:
				(char, graph, node) = map(unpack, (char, graph, node))
				ret[graph]["units"].append(
					(char, graph, node, branch, turn, tick, is_unit)
				)
			outq.task_done()
		if got != (
			"end",
			"units",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError("Expected end of units", got)
		outq.task_done()
		if (got := outq.get()) != (
			"begin",
			"character_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected beginning of character_rulebook",
				got,
			)
		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[bytes, Turn, Tick, bytes]]
			for graph, turn, tick, rb in got:
				(graph, rb) = map(unpack, (graph, rb))
				graph: CharName
				rb: RulebookName
				ret[graph]["character_rulebook"].append(
					(graph, branch, turn, tick, rb)
				)
			outq.task_done()
		if got != (
			"end",
			"character_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected end of character_rulebook",
				got,
			)
		outq.task_done()
		if (got := outq.get()) != (
			"begin",
			"unit_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected beginning of unit_rulebook",
				got,
			)
		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[bytes, Turn, Tick, bytes]]
			for graph, turn, tick, rb in got:
				(graph, rb) = map(unpack, (graph, rb))
				graph: CharName
				rb: RulebookName
				ret[graph]["unit_rulebook"].append(
					(graph, branch, turn, tick, rb)
				)
			outq.task_done()
		if got != (
			"end",
			"unit_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected end of unit_rulebook",
				got,
			)
		outq.task_done()
		if (got := outq.get()) != (
			"begin",
			"character_thing_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected beginning of character_thing_rulebook",
				got,
			)
		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[bytes, Turn, Tick, bytes]]
			for graph, turn, tick, rb in got:
				(graph, rb) = map(unpack, (graph, rb))
				graph: CharName
				rb: RulebookName
				ret[graph]["character_thing_rulebook"].append(
					(graph, branch, turn, tick, rb)
				)
			outq.task_done()
		if got != (
			"end",
			"character_thing_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected end of character_thing_rulebook",
				got,
			)
		outq.task_done()
		if (got := outq.get()) != (
			"begin",
			"character_place_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected beginning of character_place_rulebook",
				got,
			)
		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[bytes, Turn, Tick, bytes]]
			for graph, turn, tick, rb in got:
				(graph, rb) = map(unpack, (graph, rb))
				graph: CharName
				rb: RulebookName
				ret[graph]["character_place_rulebook"].append(
					(graph, branch, turn, tick, rb)
				)
			outq.task_done()
		if got != (
			"end",
			"character_place_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected end of character_place_rulebook",
				got,
			)
		outq.task_done()
		if (got := outq.get()) != (
			"begin",
			"character_portal_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected beginning of character_portal_rulebook",
				got,
			)
		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[bytes, Turn, Tick, bytes]]
			for graph, turn, tick, rb in got:
				(graph, rb) = map(unpack, (graph, rb))
				graph: CharName
				rb: RulebookName
				ret[graph]["character_portal_rulebook"].append(
					(graph, branch, turn, tick, rb)
				)
			outq.task_done()
		if got != (
			"end",
			"character_portal_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected end of character_portal_rulebook",
				got,
			)
		outq.task_done()
		if (got := outq.get()) != (
			"begin",
			"node_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError("Expected beginning of node_rulebook", got)

		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[bytes, bytes, Turn, Tick, bytes]]
			for graph, node, turn, tick, rb in got:
				(graph, node, rb) = map(unpack, (graph, node, rb))
				graph: CharName
				node: NodeName
				rb: RulebookName
				ret[graph]["node_rulebook"].append(
					(graph, node, branch, turn, tick, rb)
				)
			outq.task_done()
		if got != (
			"end",
			"node_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError("Expected end of node_rulebook", got)

		outq.task_done()
		if (got := outq.get()) != (
			"begin",
			"portal_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected beginning of portal_rulebook",
				got,
			)
		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[bytes, bytes, bytes, Turn, Tick, bytes]]
			for graph, orig, dest, turn, tick, rb in got:
				(graph, orig, dest, rb) = map(unpack, (graph, orig, dest, rb))
				graph: CharName
				orig: NodeName
				dest: NodeName
				rb: RulebookName
				ret[graph]["portal_rulebook"].append(
					(graph, orig, dest, branch, turn, tick, rb)
				)
			outq.task_done()
		if got != (
			"end",
			"portal_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected end of portal_rulebook",
				got,
			)
		outq.task_done()
		if (got := outq.get()) != (
			"begin",
			"universals",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected beginning of universals",
				got,
			)
		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[bytes, Turn, Tick, bytes]]
			for key, turn, tick, val in got:
				(key, val) = map(unpack, (key, val))
				key: UniversalKey
				val: Value
				ret["universals"].append((key, branch, turn, tick, val))
			outq.task_done()
		if got != (
			"end",
			"universals",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected end of universals",
				got,
			)
		outq.task_done()
		if (got := outq.get()) != (
			"begin",
			"rulebooks",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected beginning of rulebooks",
				got,
			)
		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[bytes, Turn, Tick, bytes, RulebookPriority]]
			for rulebook, turn, tick, rules, priority in got:
				(rulebook, rules) = map(unpack, (rulebook, rules))
				rulebook: RulebookName
				rules: list[RuleName]
				if "rulebooks" in ret:
					ret["rulebooks"].append(
						(rulebook, branch, turn, tick, (rules, priority))
					)
				else:
					ret["rulebooks"] = [
						(rulebook, branch, turn, tick, (rules, priority))
					]
			outq.task_done()
		if got != (
			"end",
			"rulebooks",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected end of rulebooks",
				got,
			)
		outq.task_done()
		if (got := outq.get()) != (
			"begin",
			"rule_triggers",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected beginning of rule_triggers",
				got,
			)
		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[RuleName, Turn, Tick, bytes]]
			for rule, turn, tick, triggers in got:
				triggers: list[TriggerFuncName] = unpack(triggers)
				if "rule_triggers" in ret:
					ret["rule_triggers"].append(
						(rule, branch, turn, tick, triggers)
					)
				else:
					ret["rule_triggers"] = [
						(rule, branch, turn, tick, triggers)
					]
			outq.task_done()
		if got != (
			"end",
			"rule_triggers",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected end of rule_triggers",
				got,
			)
		outq.task_done()
		if (got := outq.get()) != (
			"begin",
			"rule_prereqs",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected beginning of rule_prereqs",
				got,
			)
		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[RuleName, Turn, Tick, bytes]]
			for rule, turn, tick, prereqs in got:
				prereqs: list[PrereqFuncName] = unpack(prereqs)
				if "rule_prereqs" in ret:
					ret["rule_prereqs"].append(
						(rule, branch, turn, tick, prereqs)
					)
				else:
					ret["rule_prereqs"] = [(rule, branch, turn, tick, prereqs)]
			outq.task_done()
		if got != (
			"end",
			"rule_prereqs",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected end of rule_prereqs",
				got,
			)
		outq.task_done()
		if (got := outq.get()) != (
			"begin",
			"rule_actions",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected beginning of rule_actions",
				got,
			)
		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[RuleName, Turn, Tick, bytes]]
			for rule, turn, tick, actions in got:
				actions: list[ActionFuncName] = unpack(actions)
				if "rule_actions" in ret:
					ret["rule_actions"].append(
						(rule, branch, turn, tick, actions)
					)
				else:
					ret["rule_actions"] = [(rule, branch, turn, tick, actions)]
			outq.task_done()
		if got != (
			"end",
			"rule_actions",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected end of rule_actions",
				got,
			)
		outq.task_done()
		if (got := outq.get()) != (
			"begin",
			"rule_neighborhoods",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected beginning of rule_neighborhoods",
				got,
			)
		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[RuleName, Turn, Tick, RuleNeighborhood]]
			for rule, turn, tick, neighbors in got:
				if "rule_neighborhood" in ret:
					ret["rule_neighborhood"].append(
						(rule, branch, turn, tick, neighbors)
					)
				else:
					ret["rule_neighborhood"] = [
						(rule, branch, turn, tick, neighbors)
					]
			outq.task_done()
		if got != (
			"end",
			"rule_neighborhoods",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected end of rule_neighborhoods",
				got,
			)
		outq.task_done()
		if (got := outq.get()) != (
			"begin",
			"rule_big",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected beginning of rule_big",
				got,
			)
		outq.task_done()
		while isinstance(got := outq.get(), list):
			got: list[tuple[RuleName, Turn, Tick, RuleBig]]
			for rule, turn, tick, big in got:
				if "rule_big" in ret:
					ret["rule_big"].append((rule, branch, turn, tick, big))
				else:
					ret["rule_big"] = [(rule, branch, turn, tick, big)]
			outq.task_done()
		if got != (
			"end",
			"rule_big",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError(
				"Expected end of rule_big",
				got,
			)
		outq.task_done()

	@abstractmethod
	def get_all_keyframe_graphs(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> Iterator[tuple[CharName, NodeKeyframe, EdgeKeyframe, CharDict]]:
		pass

	def get_keyframe(self, branch: Branch, turn: Turn, tick: Tick) -> Keyframe:
		universal_kf, rule_kf, rulebook_kf = self.get_keyframe_extensions(
			branch, turn, tick
		)
		kf: Keyframe = {
			"universal": universal_kf,
			"rulebook": rulebook_kf,
		} | rule_kf
		for (
			char,
			node_val,
			edge_val,
			graph_val,
		) in self.get_all_keyframe_graphs(branch, turn, tick):
			if "node_val" in kf:
				kf["node_val"][char] = node_val
			else:
				kf["node_val"] = {char: node_val}
			if "edge_val" in kf:
				kf["edge_val"][char] = edge_val
			else:
				kf["edge_val"] = {char: edge_val}
			if "graph_val" in kf:
				kf["graph_val"][char] = graph_val
			else:
				kf["graph_val"] = {char: graph_val}
		return kf

	@abstractmethod
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
	]: ...

	@abstractmethod
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
	]: ...

	@abstractmethod
	def universals_dump(
		self,
	) -> Iterator[tuple[Key, Branch, Turn, Tick, Value]]:
		pass

	@abstractmethod
	def rulebooks_dump(
		self,
	) -> Iterator[
		tuple[RulebookName, Branch, Turn, Tick, tuple[list[RuleName], float]]
	]:
		pass

	@abstractmethod
	def rules_dump(self) -> Iterator[RuleName]:
		pass

	@abstractmethod
	def rule_triggers_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, list[TriggerFuncName]]]:
		pass

	@abstractmethod
	def rule_prereqs_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, list[PrereqFuncName]]]:
		pass

	@abstractmethod
	def rule_actions_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, list[ActionFuncName]]]:
		pass

	@abstractmethod
	def rule_neighborhood_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, RuleNeighborhood]]:
		pass

	@abstractmethod
	def rule_big_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, RuleBig]]: ...

	@abstractmethod
	def node_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, NodeName, Branch, Turn, Tick, RulebookName]]:
		pass

	@abstractmethod
	def portal_rulebook_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, NodeName, Branch, Turn, Tick, RulebookName]
	]:
		pass

	@abstractmethod
	def character_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		pass

	@abstractmethod
	def unit_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		pass

	@abstractmethod
	def character_thing_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		pass

	@abstractmethod
	def character_place_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		pass

	@abstractmethod
	def character_portal_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		pass

	@abstractmethod
	def character_rules_handled_dump(
		self,
	) -> Iterator[tuple[CharName, RulebookName, RuleName, Branch, Turn, Tick]]:
		pass

	@abstractmethod
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
		pass

	@abstractmethod
	def character_thing_rules_handled_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, RulebookName, RuleName, Branch, Turn, Tick]
	]:
		pass

	@abstractmethod
	def character_place_rules_handled_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, RulebookName, RuleName, Branch, Turn, Tick]
	]:
		pass

	@abstractmethod
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
		pass

	@abstractmethod
	def node_rules_handled_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, RulebookName, RuleName, Branch, Turn, Tick]
	]:
		pass

	@abstractmethod
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
		pass

	@abstractmethod
	def things_dump(
		self,
	) -> Iterator[tuple[CharName, NodeName, Branch, Turn, Tick, NodeName]]:
		pass

	@abstractmethod
	def units_dump(
		self,
	) -> Iterator[
		tuple[CharName, CharName, NodeName, Branch, Turn, Tick, bool]
	]:
		pass

	@abstractmethod
	def count_all_table(self, tbl: str) -> int:
		pass

	def set_rule_triggers(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		triggers: list[TriggerFuncName],
	):
		if rule in self.all_rules:
			self._triggers2set.append((rule, branch, turn, tick, triggers))
		else:
			self.create_rule(rule, branch, turn, tick, triggers=triggers)

	def set_rule_prereqs(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		prereqs: list[PrereqFuncName],
	):
		if rule in self.all_rules:
			self._prereqs2set.append((rule, branch, turn, tick, prereqs))
		else:
			self.create_rule(rule, branch, turn, tick, prereqs=prereqs)

	def set_rule_actions(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		actions: list[ActionFuncName],
	):
		if rule in self.all_rules:
			self._actions2set.append((rule, branch, turn, tick, actions))
		else:
			self.create_rule(rule, branch, turn, tick, actions=actions)

	def set_rule_neighborhood(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		neighborhood: RuleNeighborhood,
	):
		if rule in self.all_rules:
			self._neighbors2set.append(
				(rule, branch, turn, tick, neighborhood)
			)
		else:
			self.create_rule(
				rule, branch, turn, tick, neighborhood=neighborhood
			)

	def set_rule_big(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		big: RuleBig,
	) -> None:
		if rule in self.all_rules:
			self._big2set.append((rule, branch, turn, tick, big))
		else:
			self.create_rule(rule, branch, turn, tick, big=big)

	@abstractmethod
	def rules_insert(self, rule: RuleName): ...

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
	) -> None:
		self._triggers2set.append((rule, branch, turn, tick, list(triggers)))
		self._prereqs2set.append((rule, branch, turn, tick, list(prereqs)))
		self._actions2set.append((rule, branch, turn, tick, list(actions)))
		self._neighbors2set.append((rule, branch, turn, tick, neighborhood))
		self._big2set.append((rule, branch, turn, tick, big))
		self.all_rules.add(rule)
		self.rules_insert(rule)

	def set_rulebook(
		self,
		name: RulebookName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rules: Optional[list[RuleName]] = None,
		prio: RulebookPriority = 0.0,
	):
		self._rulebooks2set.append(
			(name, branch, turn, tick, rules or [], prio)
		)

	def set_character_rulebook(
		self,
		char: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rb: RulebookName,
	):
		self._character_rulebooks_to_set.append((char, branch, turn, tick, rb))

	def set_unit_rulebook(
		self,
		char: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rb: RulebookName,
	):
		self._unit_rulebooks_to_set.append((char, branch, turn, tick, rb))

	def set_character_thing_rulebook(
		self,
		char: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rb: RulebookName,
	):
		self._character_thing_rulebooks_to_set.append(
			(char, branch, turn, tick, rb)
		)

	def set_character_place_rulebook(
		self,
		char: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rb: RulebookName,
	):
		self._character_place_rulebooks_to_set.append(
			(char, branch, turn, tick, rb)
		)

	def set_character_portal_rulebook(
		self,
		char: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rb: RulebookName,
	):
		self._character_portal_rulebooks_to_set.append(
			(char, branch, turn, tick, rb)
		)

	@abstractmethod
	def rulebooks(self) -> Iterator[RulebookName]:
		pass

	def set_node_rulebook(
		self,
		character: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rulebook: RulebookName,
	):
		self._noderb2set.append(
			(character, node, branch, turn, tick, rulebook)
		)

	def set_portal_rulebook(
		self,
		character: CharName,
		orig: NodeName,
		dest: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rulebook: RulebookName,
	):
		self._portrb2set.append(
			(character, orig, dest, branch, turn, tick, rulebook)
		)

	def handled_character_rule(
		self,
		character: CharName,
		rulebook: RulebookName,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	):
		self._char_rules_handled.append(
			(character, rulebook, rule, branch, turn, tick)
		)

	def handled_unit_rule(
		self,
		character: CharName,
		rulebook: RulebookName,
		rule: RuleName,
		graph: CharName,
		unit: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	):
		self._unit_rules_handled.append(
			(character, rulebook, rule, graph, unit, branch, turn, tick)
		)

	def handled_character_thing_rule(
		self,
		character: CharName,
		rulebook: RulebookName,
		rule: RuleName,
		thing: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	):
		self._char_thing_rules_handled.append(
			(character, rulebook, rule, thing, branch, turn, tick)
		)

	def handled_character_place_rule(
		self,
		character: CharName,
		rulebook: RulebookName,
		rule: RuleName,
		place: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	):
		self._char_place_rules_handled.append(
			(character, place, rulebook, rule, branch, turn, tick)
		)

	def handled_character_portal_rule(
		self,
		character: CharName,
		rulebook: RulebookName,
		rule: RuleName,
		orig: NodeName,
		dest: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	):
		self._char_portal_rules_handled.append(
			(character, orig, dest, rulebook, rule, branch, turn, tick)
		)

	def handled_node_rule(
		self,
		character: CharName,
		node: NodeName,
		rulebook: RulebookName,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	):
		self._node_rules_handled.append(
			(character, node, rulebook, rule, branch, turn, tick)
		)

	def handled_portal_rule(
		self,
		character: CharName,
		orig: NodeName,
		dest: NodeName,
		rulebook: RulebookName,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	):
		self._portal_rules_handled.append(
			(character, orig, dest, rulebook, rule, branch, turn, tick)
		)

	def set_thing_loc(
		self,
		character: CharName,
		thing: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		loc: NodeName,
	):
		self._location.append((character, thing, branch, turn, tick, loc))

	@abstractmethod
	def things_del_time(self, branch: Branch, turn: Turn, tick: Tick): ...

	def unit_set(
		self,
		character: CharName,
		graph: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		is_unit: bool,
	):
		self._unitness.append(
			(character, graph, node, branch, turn, tick, is_unit)
		)

	@abstractmethod
	def turns_completed_dump(self) -> Iterator[tuple[Branch, Turn]]:
		pass

	@abstractmethod
	def bookmark_items(self) -> Iterator[tuple[Key, Time]]: ...

	def load_windows(
		self, windows: list[TimeWindow]
	) -> dict[
		Literal[
			"universals",
			"rulebooks",
			"rule_triggers",
			"rule_prereqs",
			"rule_actions",
			"rule_neighborhood",
			"rule_big",
			"graphs",
		]
		| CharName,
		list[UniversalRowType]
		| list[RulebookRowType]
		| list[RuleRowType]
		| LoadedCharWindow,
	]:
		def empty_char() -> LoadedCharWindow:
			nodes_l: list[NodeRowType] = []
			edges_l: list[EdgeRowType] = []
			graph_val_l: list[GraphValRowType] = []
			node_val_l: list[NodeValRowType] = []
			edge_val_l: list[EdgeValRowType] = []
			things_l: list[ThingRowType] = []
			units_l: list[UnitRowType] = []
			character_rulebook_l: list[CharRulebookRowType] = []
			unit_rulebook_l: list[CharRulebookRowType] = []
			char_thing_rulebook_l: list[CharRulebookRowType] = []
			char_place_rulebook_l: list[CharRulebookRowType] = []
			char_portal_rulebook_l: list[CharRulebookRowType] = []
			node_rulebook_l: list[NodeRulebookRowType] = []
			portal_rulebook_l: list[PortalRulebookRowType] = []
			return {
				"nodes": nodes_l,
				"edges": edges_l,
				"graph_val": graph_val_l,
				"node_val": node_val_l,
				"edge_val": edge_val_l,
				"things": things_l,
				"units": units_l,
				"character_rulebook": character_rulebook_l,
				"unit_rulebook": unit_rulebook_l,
				"character_thing_rulebook": char_thing_rulebook_l,
				"character_place_rulebook": char_place_rulebook_l,
				"character_portal_rulebook": char_portal_rulebook_l,
				"node_rulebook": node_rulebook_l,
				"portal_rulebook": portal_rulebook_l,
			}

		self.debug(f"load_windows({windows})")

		ret: dict[
			Literal[
				"universals",
				"rulebooks",
				"rule_triggers",
				"rule_prereqs",
				"rule_actions",
				"rule_neighborhood",
				"rule_big",
				"graphs",
			]
			| CharName,
			list[UniversalRowType]
			| list[RulebookRowType]
			| list[RuleRowType]
			| LoadedCharWindow,
		] = defaultdict(empty_char)
		ret["universals"]: list[UniversalRowType] = []
		ret["rule_triggers"]: list[RuleRowType] = []
		ret["rule_prereqs"]: list[RuleRowType] = []
		ret["rule_actions"]: list[RuleRowType] = []
		ret["rule_neighborhood"]: list[RuleRowType] = []
		ret["rule_big"]: list[RuleRowType] = []
		ret["rulebooks"]: list[RulebookRowType] = []
		self.flush()
		self._load_windows_into(ret, windows)
		self.debug(f"finished loading windows {windows}")
		return dict(ret)


class NullDatabaseConnector(AbstractDatabaseConnector):
	"""Query engine that does nothing, connects to no database

	For tests, mainly. If you want to run Lisien in-memory,
	:class:`SQLAlchemyQueryEngine` is more appropriate, with
	``connect_str='sqlite:///:memory:'``

	"""

	@cached_property
	def eternal(self) -> dict:
		return {
			"branch": "trunk",
			"turn": 0,
			"tick": 0,
			"language": "eng",
			"trunk": "trunk",
			"_lisien_schema_version": SCHEMA_VERSION,
		}

	def __init__(self):
		pass

	def call(self, query_name: str, *args, **kwargs):
		pass

	def call_silent(self, query_name: str, *args, **kwargs):
		pass

	def call_many(self, query_name: str, args: list) -> None:
		pass

	def call_many_silent(self, query_name: str, args: list) -> None:
		pass

	def delete_many_silent(self, table_name: str, args: list[dict]) -> None:
		pass

	def insert_many(self, table_name: str, args: list[dict]) -> None:
		pass

	def insert_many_silent(self, table_name: str, args: list[dict]) -> None:
		pass

	def rules_insert(self, rule: RuleName):
		pass

	def get_keyframe_extensions(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> tuple[UniversalKeyframe, RuleKeyframe, RulebookKeyframe]:
		return {}, {}, {}

	def keyframes_dump(self) -> Iterator[tuple[Branch, Turn, Tick]]:
		return iter(())

	def new_graph(
		self, graph: CharName, branch: Branch, turn: Turn, tick: Tick, typ: str
	) -> None:
		pass

	def get_all_keyframe_graphs(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> Iterator[
		tuple[CharName, NodeKeyframe, EdgeKeyframe, GraphValKeyframe]
	]:
		return iter(())

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
			GraphValKeyframe,
		]
	]:
		return iter(())

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
		return iter(())

	def graphs_insert(
		self, graph: CharName, branch: Branch, turn: Turn, tick: Tick, typ: str
	) -> None:
		pass

	def keyframes_graphs(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick]]:
		return iter(())

	def delete_keyframe(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		pass

	def have_branch(self, branch: Branch) -> bool:
		pass

	def branches_dump(
		self,
	) -> Iterator[tuple[Branch, Branch, Turn, Tick, Turn, Tick]]:
		return iter(())

	def global_get(self, key: Key) -> Any:
		return self.eternal[key]

	def global_dump(self) -> Iterator[tuple[Key, Any]]:
		return iter(self.eternal.items())

	def get_branch(self) -> Branch:
		return self.eternal["branch"]

	def get_turn(self) -> Turn:
		return self.eternal["turn"]

	def get_tick(self) -> Tick:
		return self.eternal["tick"]

	def global_set(self, key: Key, value: Any):
		self.eternal[key] = value

	def global_del(self, key: Key):
		del self.eternal[key]

	def set_branch(
		self,
		branch: Branch,
		parent: Branch,
		parent_turn: Turn,
		parent_tick: Tick,
		end_turn: Turn,
		end_tick: Tick,
	):
		pass

	def set_turn(
		self, branch: Branch, turn: Turn, end_tick: Tick, plan_end_tick: Tick
	):
		pass

	def turns_dump(self):
		return iter(())

	def graph_val_dump(self) -> Iterator[GraphValRowType]:
		return iter(())

	def graph_val_set(
		self,
		graph: CharName,
		key: Key,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		val: Any,
	):
		pass

	def graph_val_del_time(self, branch: Branch, turn: Turn, tick: Tick):
		pass

	def graphs_types(
		self,
		branch: Branch,
		turn_from: Turn,
		tick_from: Tick,
		turn_to: Optional[Turn] = None,
		tick_to: Optional[Tick] = None,
	) -> Iterator[tuple[Key, str, int, int, str]]:
		return iter(())

	def characters(self) -> Iterator[tuple[CharName, Branch, Turn, Tick, str]]:
		return iter(())

	def exist_node(
		self,
		graph: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		extant: bool,
	):
		pass

	def nodes_del_time(self, branch: Branch, turn: Turn, tick: Tick):
		pass

	def nodes_dump(self) -> Iterator[NodeRowType]:
		return iter(())

	def node_val_dump(self) -> Iterator[NodeValRowType]:
		return iter(())

	def node_val_del_time(self, branch: Branch, turn: Turn, tick: Tick):
		pass

	def edges_dump(self) -> Iterator[EdgeRowType]:
		return iter(())

	def edges_del_time(self, branch: Branch, turn: Turn, tick: Tick):
		pass

	def edge_val_dump(self) -> Iterator[EdgeValRowType]:
		return iter(())

	def edge_val_del_time(self, branch: Branch, turn: Turn, tick: Tick):
		pass

	def plan_ticks_dump(self) -> Iterator:
		return iter(())

	def flush(self):
		pass

	def commit(self):
		pass

	def close(self):
		pass

	def _init_db(self):
		pass

	def truncate_all(self):
		pass

	def universals_dump(self) -> Iterator[tuple[Key, Branch, Turn, Tick, Any]]:
		return iter(())

	def rulebooks_dump(
		self,
	) -> Iterator[
		tuple[RulebookName, Branch, Turn, Tick, tuple[list[RuleName], float]]
	]:
		return iter(())

	@cached_property
	def all_rules(self) -> set[RuleName]:
		return set()

	def rules_dump(self) -> Iterator[str]:
		return iter(())

	def rule_triggers_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, list[TriggerFuncName]]]:
		return iter(())

	def rule_prereqs_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, list[PrereqFuncName]]]:
		return iter(())

	def rule_actions_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, list[ActionFuncName]]]:
		return iter(())

	def rule_neighborhood_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, RuleNeighborhood]]:
		return iter(())

	def rule_big_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, RuleBig]]:
		return iter(())

	def node_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, NodeName, Branch, Turn, Tick, RulebookName]]:
		return iter(())

	def portal_rulebook_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, NodeName, Branch, Turn, Tick, RulebookName]
	]:
		return iter(())

	def character_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		return iter(())

	def unit_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		return iter(())

	def character_thing_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		return iter(())

	def character_place_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		return iter(())

	def character_portal_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		return iter(())

	def character_rules_handled_dump(
		self,
	) -> Iterator[tuple[CharName, RulebookName, RuleName, Branch, Turn, Tick]]:
		return iter(())

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
		return iter(())

	def character_thing_rules_handled_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, RulebookName, RuleName, Branch, Turn, Tick]
	]:
		return iter(())

	def character_place_rules_handled_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, RulebookName, RuleName, Branch, Turn, Tick]
	]:
		return iter(())

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
		return iter(())

	def node_rules_handled_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, RulebookName, RuleName, Branch, Turn, Tick]
	]:
		return iter(())

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
		return iter(())

	def things_dump(
		self,
	) -> Iterator[tuple[CharName, NodeName, Branch, Turn, Tick, NodeName]]:
		return iter(())

	def units_dump(
		self,
	) -> Iterator[
		tuple[CharName, CharName, NodeName, Branch, Turn, Tick, bool]
	]:
		return iter(())

	def universal_set(
		self, key: Key, branch: Branch, turn: Turn, tick: Tick, val: Any
	):
		pass

	def count_all_table(self, tbl: str) -> int:
		return 0

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
		return False

	def set_rule_triggers(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		flist: list[TriggerFuncName],
	):
		pass

	def set_rule_prereqs(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		flist: list[PrereqFuncName],
	):
		pass

	def set_rule_actions(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		flist: list[ActionFuncName],
	):
		pass

	def set_rule_neighborhood(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		neighborhood: RuleNeighborhood,
	):
		pass

	def set_rule_big(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		big: RuleBig,
	) -> None:
		pass

	def set_rulebook(
		self,
		name: RulebookName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rules: Optional[list[RuleName]] = None,
		prio: RulebookPriority = 0.0,
	):
		pass

	def set_character_rulebook(
		self,
		char: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rb: RulebookName,
	):
		pass

	def set_unit_rulebook(
		self,
		char: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rb: RulebookName,
	):
		pass

	def set_character_thing_rulebook(
		self,
		char: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rb: RulebookName,
	):
		pass

	def set_character_place_rulebook(
		self,
		char: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rb: RulebookName,
	):
		pass

	def set_character_portal_rulebook(
		self,
		char: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rb: RulebookName,
	):
		pass

	def rulebooks(self) -> Iterator[Key]:
		return iter(())

	def set_node_rulebook(
		self,
		character: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rulebook: RulebookName,
	):
		pass

	def set_portal_rulebook(
		self,
		character: CharName,
		orig: NodeName,
		dest: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rulebook: RulebookName,
	):
		pass

	def handled_character_rule(
		self,
		character: CharName,
		rulebook: RulebookName,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	):
		pass

	def handled_unit_rule(
		self,
		character: CharName,
		rulebook: RulebookName,
		rule: RuleName,
		graph: CharName,
		unit: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	):
		pass

	def handled_character_thing_rule(
		self,
		character: CharName,
		rulebook: RulebookName,
		rule: RuleName,
		thing: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	):
		pass

	def handled_character_place_rule(
		self,
		character: CharName,
		rulebook: RulebookName,
		rule: RuleName,
		place: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	):
		pass

	def handled_character_portal_rule(
		self,
		character: CharName,
		rulebook: RulebookName,
		rule: RuleName,
		orig: NodeName,
		dest: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	):
		pass

	def handled_node_rule(
		self,
		character: CharName,
		node: NodeName,
		rulebook: RulebookName,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	):
		pass

	def handled_portal_rule(
		self,
		character: CharName,
		orig: NodeName,
		dest: NodeName,
		rulebook: RulebookName,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	):
		pass

	def set_thing_loc(
		self,
		character: CharName,
		thing: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		loc: NodeName,
	):
		pass

	def things_del_time(self, branch: Branch, turn: Turn, tick: Tick):
		pass

	def unit_set(
		self,
		character: CharName,
		graph: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		is_unit: bool,
	):
		pass

	def rulebook_set(
		self,
		rulebook: RulebookName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rules: list[RuleName],
	):
		pass

	def turns_completed_dump(self) -> Iterator[tuple[Branch, Turn]]:
		return iter(())

	def complete_turn(
		self, branch: Branch, turn: Turn, discard_rules: bool = False
	):
		pass

	def _put_window_tick_to_end(
		self, branch: Branch, turn_from: Turn, tick_from: Tick
	):
		pass

	def _put_window_tick_to_tick(
		self,
		branch: Branch,
		turn_from: Turn,
		tick_from: Tick,
		turn_to: Turn,
		tick_to: Tick,
	):
		pass

	def _load_windows_into(self, ret: dict, windows: list[TimeWindow]) -> None:
		pass

	def _increc(self):
		pass

	def _get_one_window(
		self,
		ret,
		branch: Branch,
		turn_from: Turn,
		tick_from: Tick,
		turn_to: Turn,
		tick_to: Tick,
	):
		pass

	def bookmark_items(self) -> Iterator[tuple[Key, Time]]:
		return iter(())

	def set_bookmark(
		self, key: Key, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		pass

	def del_bookmark(self, key: Key) -> None:
		pass

	def load_windows(self, windows: list[TimeWindow]) -> dict:
		return {}


class ParquetDatabaseConnector(AbstractDatabaseConnector):
	looper_cls = ParquetDBLooper

	def __init__(self, path, pack=None, unpack=None, *, clear=False):
		self._inq = Queue()
		self._outq = Queue()
		self._looper = self.looper_cls(path, self._inq, self._outq)
		self._records = 0
		self.keyframe_interval = None
		self.snap_keyframe = lambda: None
		self._new_keyframe_times = set()

		if pack is None:

			def pack(s: Any) -> bytes:
				return repr(s).encode()

		if unpack is None:
			from ast import literal_eval

			def unpack(b: bytes) -> Any:
				return literal_eval(b.decode())

		self.pack = pack
		self.unpack = unpack
		self._branches = {}
		self._btts = set()
		self._t = Thread(target=self._looper.run, daemon=True)
		self._t.start()
		if clear:
			self.truncate_all()
		self._init_db()

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
		return iter(
			sorted(
				(
					unpack(d["character"]),
					unpack(d["rulebook"]),
					d["rule"],
					d["branch"],
					d["turn"],
					d["tick"],
				)
				for d in self.call("dump", "character_rules_handled")
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
		self._unit_rules_handled()
		unpack = self.unpack
		return iter(
			sorted(
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
				for d in self.call("dump", "unit_rules_handled")
			)
		)

	def character_thing_rules_handled_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, RulebookName, RuleName, Branch, Turn, Tick]
	]:
		self._char_thing_rules_handled()
		unpack = self.unpack
		return iter(
			sorted(
				(
					unpack(d["character"]),
					unpack(d["thing"]),
					unpack(d["rulebook"]),
					d["rule"],
					d["branch"],
					d["turn"],
					d["tick"],
				)
				for d in self.call("dump", "character_thing_rules_handled")
			)
		)

	def character_place_rules_handled_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, RulebookName, RuleName, Branch, Turn, Tick]
	]:
		self._char_place_rules_handled()
		unpack = self.unpack
		return iter(
			sorted(
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
		return iter(
			sorted(
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
		)

	def node_rules_handled_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, RulebookName, RuleName, Branch, Turn, Tick]
	]:
		self._node_rules_handled()
		unpack = self.unpack
		return iter(
			sorted(
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
		self._portal_rules_handled()
		unpack = self.unpack
		return iter(
			sorted(
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
		)

	def things_dump(
		self,
	) -> Iterator[tuple[CharName, NodeName, Branch, Turn, Tick, NodeName]]:
		self._location()
		unpack = self.unpack
		return iter(
			sorted(
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
		)

	def units_dump(
		self,
	) -> Iterator[
		tuple[CharName, CharName, NodeName, Branch, Turn, Tick, bool]
	]:
		self._unitness()
		unpack = self.unpack
		return iter(
			sorted(
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

	def rulebooks(self) -> Iterator[RulebookName]:
		return map(self.pack, self.call("rulebooks"))

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

	def characters(self) -> Iterator[tuple[CharName, Branch, Turn, Tick, str]]:
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
		ret = self.call("initdb")
		if isinstance(ret, Exception):
			raise ret
		elif not isinstance(ret, dict):
			raise TypeError("initdb didn't return a dictionary", ret)
		unpack = self.unpack
		self.eternal = GlobalKeyValueStore(
			self, {unpack(k): unpack(v) for (k, v) in ret.items()}
		)
		self._all_keyframe_times = self.call("all_keyframe_times")
		self.all_rules = set(d["rule"] for d in self.call("dump", "rules"))
		return ret

	def bookmark_items(self) -> Iterator[tuple[Key, Time]]:
		return iter(self.call("bookmark_items"))

	def del_bookmark(self, key: Key) -> None:
		self.call("del_bookmark", key)


class SQLAlchemyConnectionLooper(ConnectionLooper):
	def __init__(
		self,
		dbstring: str,
		connect_args: dict,
		inq: Queue,
		outq: Queue,
		tables: list[str],
	):
		self.lock = Lock()
		self.existence_lock.acquire(timeout=1)
		self._dbstring = dbstring
		self._connect_args = connect_args
		self.inq = inq
		self.outq = outq
		self.tables = tables

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
			repositioned = [kwargs[param] for param in statement.positiontup]
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
		dbstring = self._dbstring
		connect_args = self._connect_args
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
						if hasattr(res, "returns_rows") and res.returns_rows:
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
		glob_d = dict(self.call("global_dump").fetchall())
		if SCHEMAVER_B not in glob_d:
			self.call("global_insert", SCHEMAVER_B, SCHEMA_VERSION_B)
			glob_d[SCHEMAVER_B] = SCHEMA_VERSION_B
		elif glob_d[SCHEMAVER_B] != SCHEMA_VERSION_B:
			return ValueError(
				"Unsupported database schema version", glob_d[SCHEMAVER_B]
			)
		return glob_d


class SQLAlchemyDatabaseConnector(AbstractDatabaseConnector):
	IntegrityError = IntegrityError
	OperationalError = OperationalError
	looper_cls = SQLAlchemyConnectionLooper
	kf_interval_override: callable

	def __init__(
		self, dbstring, connect_args, pack=None, unpack=None, *, clear=False
	):
		dbstring = dbstring or "sqlite:///:memory:"
		self._inq = Queue()
		self._outq = Queue()
		self._looper = self.looper_cls(
			dbstring,
			connect_args,
			self._inq,
			self._outq,
			list(meta.tables.keys()),
		)

		if pack is None:

			def pack(s: Any) -> bytes:
				return repr(s).encode()

		if unpack is None:
			from ast import literal_eval

			def unpack(b: bytes) -> Any:
				return literal_eval(b.decode())

		self.pack = pack
		self.unpack = unpack
		self._branches = {}
		self._new_keyframe_times: set[Time] = set()
		self._records = 0
		self.keyframe_interval = None
		self.snap_keyframe = lambda: None
		self._t = Thread(target=self._looper.run, daemon=True)
		self._t.start()
		if clear:
			self.truncate_all()
		self._init_db()

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
			raise ExceptionGroup(
				f"{unfinished_tasks} unfinished tasks in output queue "
				"before call_one",
				excs,
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

	def bookmark_items(self) -> Iterator[tuple[Key, Time]]:
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
			return 0
		return self.unpack(v[0])

	def get_tick(self) -> Tick:
		v = self.call("global_get", self.pack("tick"))[0]
		if v is None:
			return 0
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

	def characters(self):
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

	def iter_edges(
		self, graph, branch, turn_from, tick_from, turn_to=None, tick_to=None
	) -> Iterator[EdgeRowType]:
		if (turn_to is None) ^ (tick_to is None):
			raise ValueError("I need both or neither of turn_to and tick_to")
		self._edgevals2set()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.call(
				"load_edges_tick_to_end",
				pack(graph),
				branch,
				turn_from,
				turn_from,
				tick_from,
			)
		else:
			it = self.call(
				"load_edges_tick_to_tick",
				pack(graph),
				branch,
				turn_from,
				turn_from,
				tick_from,
				turn_to,
				turn_to,
				tick_to,
			)
		for orig, dest, turn, tick, extant in it:
			yield (
				graph,
				unpack(orig),
				unpack(dest),
				branch,
				turn,
				tick,
				extant,
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
			self._all_keyframe_times = set(x)
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
		self.all_rules = set(self.rules_dump())
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
		self._unit_rules_handled()
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

	def rulebooks(self):
		for book in self.call("rulebooks"):
			yield self.unpack(book)

	def things_del_time(self, branch: Branch, turn: Turn, tick: Tick):
		self._location.cull(
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
