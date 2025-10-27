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
from dataclasses import dataclass, field, KW_ONLY
from functools import cached_property, partial, partialmethod, wraps
from itertools import filterfalse, starmap
from operator import itemgetter
from queue import Queue
from sqlite3 import IntegrityError as LiteIntegrityError
from sqlite3 import OperationalError as LiteOperationalError
from threading import Lock, Thread
from types import MethodType
from typing import (
	Annotated,
	Any,
	Callable,
	Iterable,
	Iterator,
	Literal,
	MutableMapping,
	Optional,
	Union,
	get_args,
	get_origin,
	get_type_hints,
	TypeVar,
	TYPE_CHECKING,
	ClassVar,
)

from sqlalchemy import Select, create_engine
from sqlalchemy.exc import IntegrityError as AlchemyIntegrityError
from sqlalchemy.exc import OperationalError as AlchemyOperationalError

import lisien.types
from .alchemy import meta, queries
from .exc import KeyframeError
from .types import (
	ActionFuncName,
	actfuncn,
	ActionRowType,
	Branch,
	CharDict,
	CharName,
	charn,
	CharRulebookRowType,
	EdgeKeyframe,
	EdgeRowType,
	EdgeValRowType,
	EternalKey,
	GraphRowType,
	GraphTypeStr,
	GraphValKeyframe,
	GraphValRowType,
	Key,
	Keyframe,
	NodeKeyframe,
	NodeName,
	nodename,
	NodeRowType,
	NodeRulebookRowType,
	NodeValRowType,
	Plan,
	PortalRulebookRowType,
	PrereqFuncName,
	preqfuncn,
	PrereqRowType,
	RuleBig,
	RuleBigRowType,
	RulebookKeyframe,
	RulebookName,
	rbname,
	RulebookPriority,
	RulebookRowType,
	RulebookTypeStr,
	RuleFuncName,
	RuleKeyframe,
	RuleName,
	rulename,
	RuleNeighborhood,
	RuleNeighborhoodRowType,
	RuleRowType,
	Stat,
	StatDict,
	ThingRowType,
	Tick,
	Time,
	TimeWindow,
	TriggerFuncName,
	trigfuncn,
	TriggerRowType,
	Turn,
	UnitRowType,
	UniversalKey,
	UniversalKeyframe,
	UniversalRowType,
	Value,
	PackSignature,
	UnpackSignature,
	ekey,
	LoadedDict,
	LoadedCharWindow,
	CharacterRulesHandledRowType,
	PortalRulesHandledRowType,
	NodeRulesHandledRowType,
	UnitRulesHandledRowType,
)

from .util import ELLIPSIS, EMPTY, garbage, sort_set
from .wrap import DictWrapper, ListWrapper, SetWrapper

if sys.version_info.minor < 11:

	class ExceptionGroup(Exception):
		pass


if TYPE_CHECKING:
	import pyarrow as pa


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

	def __getitem__(
		self, k: Key
	) -> Value | DictWrapper | ListWrapper | SetWrapper:
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


@dataclass
class ConnectionLooper(ABC):
	connector: AbstractDatabaseConnector

	@cached_property
	def inq(self) -> Queue:
		return self.connector._inq

	@cached_property
	def outq(self) -> Queue:
		return self.connector._outq

	@cached_property
	def lock(self):
		return Lock()

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


_ARGS = TypeVar("_ARGS")
_RET = TypeVar("_RET")


def mutexed(
	func: Callable[[_ARGS, ...], _RET],
) -> Callable[[_ARGS, ...], _RET]:
	"""Decorator for when an entire method's body holds a mutex lock"""

	@wraps(func)
	def mutexy(self, *args, **kwargs):
		with self.mutex():
			return func(self, *args, **kwargs)

	return mutexy


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
		serialize_record: Callable[[_ARGS, ...], tuple[bytes, ...]],
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
	"""Decorator for serializers that operate on batches of records

	Needs at least the name of the ``table`` the batch will be inserted into.

	:param key_len: How long the primary key is. Used to delete records matching
		those in the batch.
	:param inc_rec_counter: Whether to count these records toward the number
		needed to trigger an automatic keyframe snap.

	"""
	if serialize_record is None:
		return partial(
			batched,
			table,
			key_len=key_len,
			inc_rec_counter=inc_rec_counter,
		)
	batched.serializers[table] = serialize_record
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

	return batched.tables.setdefault(table, cached_property(the_batch))


batched.tables = {}
batched.serializers = {}


class AbstractDatabaseConnector(ABC):
	_: KW_ONLY
	kf_interval_override: Callable[[], bool | None] = lambda _: None
	keyframe_interval: int | None = 1000
	snap_keyframe: Callable[[], None] = lambda: None

	@cached_property
	def _records(self) -> int:
		return 0

	@cached_property
	def _new_keyframe_times(self) -> set[Time]:
		return set()

	@cached_property
	def all_rules(self) -> set[RuleName]:
		return set()

	@cached_property
	def eternal(self) -> MutableMapping:
		return {
			"branch": "trunk",
			"turn": 0,
			"tick": 0,
			"language": "eng",
			"trunk": "trunk",
			"_lisien_schema_version": SCHEMA_VERSION,
		}

	@cached_property
	def _lock(self) -> Lock:
		return Lock()

	@contextmanager
	def mutex(self):
		with self._lock:
			yield

	@property
	def pack(self) -> PackSignature:
		return self._pack

	@pack.setter
	def pack(self, v: PackSignature) -> None:
		self._pack = v
		if hasattr(self, "_unpack") and not hasattr(self, "_initialized"):
			self._init_db()

	@property
	def unpack(self) -> UnpackSignature:
		return self._unpack

	@unpack.setter
	def unpack(self, v: UnpackSignature) -> None:
		self._unpack = v
		if hasattr(self, "_pack") and not hasattr(self, "_initialized"):
			self._init_db()

	def dump_everything(self) -> dict[str, list[tuple]]:
		self.flush()
		return {
			table: list(getattr(self, f"{table}_dump")())
			for table in batched.tables
		}

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
			self._unit_rules_handled_to_set.clear()
			self._char_thing_rules_handled.clear()
			self._char_place_rules_handled.clear()
			self._char_portal_rules_handled.clear()
			self._node_rules_handled_to_set.clear()
			self._portal_rules_handled_to_set.clear()

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

	@batched("rules", key_len=1)
	def _rules2set(self, rule: RuleName) -> tuple[bytes]:
		return (self.pack(rule),)

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
	def _character_rulebook_to_set(
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
	def _unit_rulebook_to_set(
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
	def _character_thing_rulebook_to_set(
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
	def _character_place_rulebook_to_set(
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
	def _character_portal_rulebook_to_set(
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

	@batched("character_rules_handled", key_len=5, inc_rec_counter=False)
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

	@batched("unit_rules_handled", key_len=7, inc_rec_counter=False)
	def _unit_rules_handled_to_set(
		self,
		character: CharName,
		rulebook: RulebookName,
		rule: RuleName,
		graph: CharName,
		unit: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> tuple[bytes, bytes, RuleName, bytes, bytes, Branch, Turn, Tick]:
		character, graph, unit, rulebook = map(
			self.pack, (character, graph, unit, rulebook)
		)
		return character, rulebook, rule, graph, unit, branch, turn, tick

	@batched("character_thing_rules_handled", key_len=6, inc_rec_counter=False)
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

	@batched("character_place_rules_handled", key_len=6, inc_rec_counter=False)
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

	@batched(
		"character_portal_rules_handled", key_len=7, inc_rec_counter=False
	)
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

	@batched("node_rules_handled", key_len=6, inc_rec_counter=False)
	def _node_rules_handled_to_set(
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
	def _portal_rules_handled_to_set(
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
		value: Value | type(...),
	) -> None:
		self._universals2set.append((key, branch, turn, tick, value))

	def universal_del(
		self, key: UniversalKey, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		self.universal_set(key, branch, turn, tick, ...)

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
		return set()

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
	def graph_val_del_time(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> None: ...

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
	def graphs_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, str]]:
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
	def edges_del_time(self, branch: Branch, turn: Turn, tick: Tick) -> None:
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
		"rule_neighborhood",
		"rule_big",
		"character_rules_handled",
		"unit_rules_handled",
		"character_thing_rules_handled",
		"character_place_rules_handled",
		"character_portal_rules_handled",
		"node_rules_handled",
		"portal_rules_handled",
	]

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

	@abstractmethod
	def get_all_keyframe_graphs(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> Iterator[tuple[CharName, NodeKeyframe, EdgeKeyframe, StatDict]]:
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
			StatDict,
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
	) -> Iterator[CharacterRulesHandledRowType]:
		pass

	@abstractmethod
	def unit_rules_handled_dump(
		self,
	) -> Iterator[UnitRulesHandledRowType]:
		pass

	@abstractmethod
	def character_thing_rules_handled_dump(
		self,
	) -> Iterator[NodeRulesHandledRowType]:
		pass

	@abstractmethod
	def character_place_rules_handled_dump(
		self,
	) -> Iterator[NodeRulesHandledRowType]:
		pass

	@abstractmethod
	def character_portal_rules_handled_dump(
		self,
	) -> Iterator[PortalRulesHandledRowType]:
		pass

	@abstractmethod
	def node_rules_handled_dump(
		self,
	) -> Iterator[NodeRulesHandledRowType]:
		pass

	@abstractmethod
	def portal_rules_handled_dump(
		self,
	) -> Iterator[PortalRulesHandledRowType]:
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
		self._character_rulebook_to_set.append((char, branch, turn, tick, rb))

	def set_unit_rulebook(
		self,
		char: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rb: RulebookName,
	):
		self._unit_rulebook_to_set.append((char, branch, turn, tick, rb))

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

	def rulebook_set(
		self,
		rulebook: RulebookName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rules: list[RuleName],
		priority: RulebookPriority,
	):
		self._rulebooks2set.append(
			(rulebook, branch, turn, tick, rules, priority)
		)

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
		self._unit_rules_handled_to_set.append(
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
		self._node_rules_handled_to_set.append(
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
		self._portal_rules_handled_to_set.append(
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
	def bookmarks_dump(self) -> Iterator[tuple[Key, Time]]: ...

	@abstractmethod
	def _load_windows_into(self, ret: dict, windows: list[TimeWindow]): ...

	@staticmethod
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

	def load_windows(self, windows: list[TimeWindow]) -> LoadedDict:
		self.debug(f"load_windows({windows})")

		ret: LoadedDict = defaultdict(self.empty_char)
		ret["universals"]: list[UniversalRowType] = []
		ret["rule_triggers"]: list[RuleRowType] = []
		ret["rule_prereqs"]: list[RuleRowType] = []
		ret["rule_actions"]: list[RuleRowType] = []
		ret["rule_neighborhood"]: list[RuleRowType] = []
		ret["rule_big"]: list[RuleRowType] = []
		ret["rulebooks"]: list[RulebookRowType] = []
		ret["character_rules_handled"]: list[CharacterRulesHandledRowType] = []
		ret["unit_rules_handled"]: list[UnitRulesHandledRowType] = []
		ret["character_thing_rules_handled"]: list[
			NodeRulesHandledRowType
		] = []
		ret["character_place_rules_handled"]: list[
			NodeRulesHandledRowType
		] = []
		ret["character_portal_rules_handled"]: list[
			PortalRulesHandledRowType
		] = []
		ret["node_rules_handled"]: list[NodeRulesHandledRowType] = []
		ret["portal_rules_handled"]: list[PortalRulesHandledRowType] = []
		ret["graphs"]: list[GraphRowType] = []
		self.flush()
		self._load_windows_into(ret, windows)
		self.debug(f"finished loading windows {windows}")
		return dict(ret)


_T = TypeVar("_T")


@dataclass
class PythonDatabaseConnector(AbstractDatabaseConnector):
	"""Database connector that holds all data in memory

	You'll have to write it to disk yourself, somehow. Use `dump_everything`
	to get all the data in a dictionary.

	This does not start any threads, unlike the connectors that really
	connect to databases, making it an appropriate choice if running in
	an environment that lacks threading, such as WASI.

	"""

	@cached_property
	def _bookmarks(self) -> dict[str, Time]:
		return {}

	@cached_property
	def _keyframe_extensions(
		self,
	) -> dict[Time, tuple[UniversalKeyframe, RuleKeyframe, RulebookKeyframe]]:
		return {}

	@cached_property
	def _keyframes(self) -> set[Time]:
		return set()

	@property
	def _all_keyframe_times(self) -> set[Time]:
		return self._keyframes.copy()

	@cached_property
	def _keyframes_graphs(
		self,
	) -> dict[
		tuple[Branch, Turn, Tick, CharName],
		tuple[NodeKeyframe, EdgeKeyframe, StatDict],
	]:
		return {}

	@cached_property
	def _branches(self) -> dict[Branch, tuple[Branch, Turn, Tick, Turn, Tick]]:
		return {}

	@cached_property
	def _global(self) -> list[tuple[EternalKey, Value]]:
		return []

	@cached_property
	def eternal(self) -> GlobalKeyValueStore:
		initial = {
			ekey(k): Value(v)
			for (k, v) in {
				"branch": "trunk",
				"turn": 0,
				"tick": 0,
				"language": "eng",
				"trunk": "trunk",
				"_lisien_schema_version": SCHEMA_VERSION,
			}.items()
		}
		initial.update(self._global)
		return GlobalKeyValueStore(self, initial)

	@cached_property
	def _turns(self) -> dict[tuple[Branch, Turn], tuple[Tick, Tick]]:
		return {}

	@cached_property
	def _graphs(
		self,
	) -> dict[tuple[Branch, Turn, Tick, CharName], GraphTypeStr]:
		return {}

	@cached_property
	def _graph_val(
		self,
	) -> dict[tuple[Branch, Turn, Tick, CharName, Stat], Value]:
		return {}

	@cached_property
	def _nodes(
		self,
	) -> dict[tuple[Branch, Turn, Tick, CharName, NodeName], bool]:
		return {}

	@cached_property
	def _node_val(
		self,
	) -> dict[tuple[Branch, Turn, Tick, CharName, NodeName, Stat], Value]:
		return {}

	@cached_property
	def _edges(
		self,
	) -> dict[tuple[Branch, Turn, Tick, CharName, NodeName, NodeName], bool]:
		return {}

	@cached_property
	def _edge_val(
		self,
	) -> dict[
		tuple[Branch, Turn, Tick, CharName, NodeName, NodeName, Stat], Value
	]:
		return {}

	@cached_property
	def _plans(self) -> dict[Plan, tuple[Branch, Turn, Tick]]:
		return {}

	@cached_property
	def _plan_ticks(self) -> set[tuple[Plan, Branch, Turn, Tick]]:
		return set()

	@cached_property
	def _universals(
		self,
	) -> dict[tuple[Branch, Turn, Tick, UniversalKey], Value]:
		return {}

	@cached_property
	def _rules(self) -> set[RuleName]:
		return set()

	@cached_property
	def _rulebooks(
		self,
	) -> dict[
		tuple[
			Branch,
			Turn,
			Tick,
			RulebookName,
		],
		tuple[list[RuleName], RulebookPriority],
	]:
		return {}

	@cached_property
	def _rule_triggers(
		self,
	) -> dict[tuple[Branch, Turn, Tick, RuleName], list[TriggerFuncName]]:
		return {}

	@cached_property
	def _rule_neighborhood(
		self,
	) -> dict[tuple[Branch, Turn, Tick, RuleName], RuleNeighborhood]:
		return {}

	@cached_property
	def _rule_prereqs(
		self,
	) -> dict[tuple[Branch, Turn, Tick, RuleName], list[PrereqFuncName]]:
		return {}

	@cached_property
	def _rule_actions(
		self,
	) -> dict[tuple[Branch, Turn, Tick, RuleName], list[ActionFuncName]]:
		return {}

	@cached_property
	def _rule_big(self) -> dict[tuple[Branch, Turn, Tick, RuleName], RuleBig]:
		return {}

	@cached_property
	def _character_rulebook(
		self,
	) -> dict[tuple[Branch, Turn, Tick, CharName], RulebookName]:
		return {}

	@cached_property
	def _unit_rulebook(
		self,
	) -> dict[tuple[Branch, Turn, Tick, CharName], RulebookName]:
		return {}

	@cached_property
	def _character_thing_rulebook(
		self,
	) -> dict[tuple[Branch, Turn, Tick, CharName], RulebookName]:
		return {}

	@cached_property
	def _character_place_rulebook(
		self,
	) -> dict[tuple[Branch, Turn, Tick, CharName], RulebookName]:
		return {}

	@cached_property
	def _character_portal_rulebook(
		self,
	) -> dict[tuple[Branch, Turn, Tick, CharName], RulebookName]:
		return {}

	@cached_property
	def _node_rules_handled(
		self,
	) -> dict[
		tuple[Branch, Turn, CharName, NodeName, RulebookName, RuleName], Tick
	]:
		return {}

	@cached_property
	def _portal_rules_handled(
		self,
	) -> dict[
		tuple[
			Branch, Turn, CharName, NodeName, NodeName, RulebookName, RuleName
		],
		Tick,
	]:
		return {}

	@cached_property
	def _things(
		self,
	) -> dict[tuple[Branch, Turn, Tick, CharName, NodeName], NodeName]:
		return {}

	@cached_property
	def _node_rulebook(
		self,
	) -> dict[tuple[Branch, Turn, Tick, CharName, NodeName], RulebookName]:
		return {}

	@cached_property
	def _portal_rulebook(
		self,
	) -> dict[
		tuple[Branch, Turn, Tick, CharName, NodeName, NodeName], RulebookName
	]:
		return {}

	@cached_property
	def _units(
		self,
	) -> dict[tuple[Branch, Turn, Tick, CharName, CharName, NodeName], bool]:
		return {}

	@cached_property
	def _character_rules_handled(
		self,
	) -> dict[tuple[Branch, Turn, CharName, RulebookName, RuleName], Tick]:
		return {}

	@cached_property
	def _unit_rules_handled(
		self,
	) -> dict[
		tuple[
			Branch, Turn, CharName, CharName, NodeName, RulebookName, RuleName
		],
		Tick,
	]:
		return {}

	@cached_property
	def _character_thing_rules_handled(
		self,
	) -> dict[
		tuple[Branch, Turn, CharName, RulebookName, RuleName, NodeName], Tick
	]:
		return {}

	@cached_property
	def _character_place_rules_handled(
		self,
	) -> dict[
		tuple[Branch, Turn, CharName, NodeName, RulebookName, RuleName], Tick
	]:
		return {}

	@cached_property
	def _character_portal_rules_handled(
		self,
	) -> dict[
		tuple[
			Branch, Turn, CharName, NodeName, NodeName, RulebookName, RuleName
		],
		Tick,
	]:
		return {}

	@cached_property
	def _turns_completed(self) -> dict[Branch, Turn]:
		return {}

	_table_names = [
		"_bookmarks",
		"_global",
		"_branches",
		"_turns",
		"_graphs",
		"_keyframes",
		"_keyframes_graphs",
		"_keyframe_extensions",
		"_graph_val",
		"_nodes",
		"_node_val",
		"_edges",
		"_edge_val",
		"_plans",
		"_plan_ticks",
		"_universals",
		"_rules",
		"_rulebooks",
		"_rule_triggers",
		"_rule_neighborhood",
		"_rule_prereqs",
		"_rule_actions",
		"_rule_big",
		"_character_rulebook",
		"_unit_rulebook",
		"_character_thing_rulebook",
		"_character_place_rulebook",
		"_character_portal_rulebook",
		"_node_rules_handled",
		"_portal_rules_handled",
		"_things",
		"_node_rulebook",
		"_portal_rulebook",
		"_units",
		"_character_rules_handled",
		"_unit_rules_handled",
		"_character_thing_rules_handled",
		"_character_place_rules_handled",
		"_character_portal_rules_handled",
		"_turns_completed",
	]

	@cached_property
	def _lock(self) -> Lock:
		return Lock()

	@property
	def pack(self):
		return self._pack

	@pack.setter
	def pack(self, v):
		pass

	@staticmethod
	def _pack(a: _T) -> _T:
		return a

	@property
	def unpack(self):
		return self._unpack

	@unpack.setter
	def unpack(self, v):
		pass

	@staticmethod
	def _unpack(a: _T) -> _T:
		return a

	def _load_window(
		self,
		ret: LoadedDict,
		branch: Branch,
		turn_from: Turn,
		tick_from: Tick,
		turn_to: Turn | None = None,
		tick_to: Tick | None = None,
	) -> None:
		if turn_to is None:
			turn_to = float("inf")
		if tick_to is None:
			tick_to = float("inf")
		universals: list[UniversalRowType] = ret.setdefault("universals", [])
		rulebooks: list[RulebookRowType] = ret.setdefault("rulebooks", [])
		rule_triggers: list[TriggerRowType] = ret.setdefault(
			"rule_triggers", []
		)
		rule_prereqs: list[PrereqRowType] = ret.setdefault("rule_prereqs", [])
		rule_actions: list[ActionRowType] = ret.setdefault("rule_actions", [])
		rule_neighborhood: list[RuleNeighborhoodRowType] = ret.setdefault(
			"rule_neighborhood", []
		)
		rule_big: list[RuleBigRowType] = ret.setdefault("rule_big", [])
		character_rules_handled: list[CharacterRulesHandledRowType] = (
			ret.setdefault("character_rules_handled", [])
		)
		unit_rules_handled: list[UnitRulesHandledRowType] = ret.setdefault(
			"unit_rules_handled", []
		)
		character_thing_rules_handled: list[NodeRulesHandledRowType] = (
			ret.setdefault("character_thing_rules_handled", [])
		)
		character_place_rules_handled: list[NodeRulesHandledRowType] = (
			ret.setdefault("character_place_rules_handled", [])
		)
		character_portal_rules_handled: list[PortalRulesHandledRowType] = (
			ret.setdefault("character_portal_rules_handled", [])
		)
		node_rules_handled: list[NodeRulesHandledRowType] = ret.setdefault(
			"node_rules_handled", []
		)
		portal_rules_handled: list[PortalRulesHandledRowType] = ret.setdefault(
			"portal_rules_handled", []
		)
		graphs: list[GraphRowType] = ret.setdefault("graphs", [])
		for b, turn, tick, key in sort_set(self._universals.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			universals.append(
				(key, b, turn, tick, self._universals[b, turn, tick, key])
			)
		for b, turn, tick, rb in sort_set(self._rulebooks.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			(rules_l, prio) = self._rulebooks[b, turn, tick, rb]
			rulebooks.append((rb, b, turn, tick, (rules_l.copy(), prio)))
		for b, turn, tick, rb in sort_set(self._rule_triggers.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			trigs_l = self._rule_triggers[b, turn, tick, rb]
			rule_triggers.append((rb, b, turn, tick, trigs_l.copy()))
		for b, turn, tick, rb in sort_set(self._rule_prereqs.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			preqs_l = self._rule_prereqs[b, turn, tick, rb]
			rule_prereqs.append((rb, b, turn, tick, preqs_l.copy()))
		for b, turn, tick, rb in sort_set(self._rule_actions.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			acts_l = self._rule_actions[b, turn, tick, rb]
			rule_actions.append((rb, b, turn, tick, acts_l.copy()))
		for b, turn, tick, rb in sort_set(self._rule_neighborhood.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			rule_neighborhood.append(
				(rb, b, turn, tick, self._rule_neighborhood[b, turn, tick, rb])
			)
		for b, turn, tick, rb in sort_set(self._rule_big.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			rule_big.append(
				(rb, b, turn, tick, self._rule_big[b, turn, tick, rb])
			)
		for b, turn, tick, g in sort_set(self._graphs.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			graphs.append((g, b, turn, tick, self._graphs[b, turn, tick, g]))

		for b, turn, tick, g, n in sort_set(self._nodes.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			char_d: LoadedCharWindow = ret[g]
			nodes: list[NodeRowType] = char_d["nodes"]
			nodes.append(
				(g, n, b, turn, tick, self._nodes[b, turn, tick, g, n])
			)
		for b, turn, tick, g, n, k in sort_set(self._node_val.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			char_d: LoadedCharWindow = ret[g]
			node_val: list[NodeValRowType] = char_d["node_val"]
			node_val.append(
				(
					g,
					n,
					k,
					b,
					turn,
					tick,
					self._node_val[b, turn, tick, g, n, k],
				)
			)
		for b, turn, tick, g, o, d in sort_set(self._edges.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			char_d: LoadedCharWindow = ret[g]
			edges: list[EdgeRowType] = char_d["edges"]
			edges.append(
				(g, o, d, b, turn, tick, self._edges[b, turn, tick, g, o, d])
			)
		for b, turn, tick, g, o, d, k in sort_set(self._edge_val.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			char_d: LoadedCharWindow = ret[g]
			edge_val: list[EdgeValRowType] = char_d["edge_val"]
			edge_val.append(
				(
					g,
					o,
					d,
					k,
					b,
					turn,
					tick,
					self._edge_val[b, turn, tick, g, o, d, k],
				)
			)
		for b, turn, tick, g, k in sort_set(self._graph_val.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			char_d: LoadedCharWindow = ret[g]
			graph_val: list[GraphValRowType] = char_d["graph_val"]
			graph_val.append(
				(g, k, b, turn, tick, self._graph_val[b, turn, tick, g, k])
			)
		for b, turn, tick, g, n in sort_set(self._things.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			char_d: LoadedCharWindow = ret[g]
			things: list[ThingRowType] = char_d["things"]
			things.append(
				(g, n, b, turn, tick, self._things[b, turn, tick, g, n])
			)
		for b, turn, tick, g, n in sort_set(self._node_rulebook.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			char_d: LoadedCharWindow = ret[g]
			node_rulebook: list[NodeRulebookRowType] = char_d["node_rulebook"]
			node_rulebook.append(
				(g, n, b, turn, tick, self._node_rulebook[b, turn, tick, g, n])
			)
		for b, turn, tick, g, o, d in sort_set(self._portal_rulebook.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			char_d: LoadedCharWindow = ret[g]
			portal_rb: list[PortalRulebookRowType] = char_d["portal_rulebook"]
			portal_rb.append(
				(
					g,
					o,
					d,
					b,
					turn,
					tick,
					self._portal_rulebook[b, turn, tick, g, o, d],
				)
			)
		for b, turn, tick, g in sort_set(self._character_rulebook.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			char_d: LoadedCharWindow = ret[g]
			char_rb: list[CharRulebookRowType] = char_d["character_rulebook"]
			char_rb.append(
				(g, b, turn, tick, self._character_rulebook[b, turn, tick, g])
			)
		for b, turn, tick, g in sort_set(self._unit_rulebook.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			char_d: LoadedCharWindow = ret[g]
			unit_rb: list[CharRulebookRowType] = char_d["unit_rulebook"]
			unit_rb.append(
				(g, b, turn, tick, self._unit_rulebook[b, turn, tick, g])
			)
		cthrb = self._character_thing_rulebook
		for b, turn, tick, g in sort_set(cthrb.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			char_d: LoadedCharWindow = ret[g]
			char_thing_rb: list[NodeRulebookRowType] = char_d[
				"character_thing_rulebook"
			]
			char_thing_rb.append((g, b, turn, tick, cthrb[b, turn, tick, g]))
		cplrb = self._character_place_rulebook
		for b, turn, tick, g in sort_set(cplrb.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			char_d: LoadedCharWindow = ret[g]
			cprb: list[CharRulebookRowType] = char_d[
				"character_place_rulebook"
			]
			cprb.append((g, b, turn, tick, cplrb[b, turn, tick, g]))
		cporb = self._character_portal_rulebook
		for b, turn, tick, g in sort_set(cporb.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			char_d: LoadedCharWindow = ret[g]
			cporb_l: list[PortalRulebookRowType] = char_d[
				"character_portal_rulebook"
			]
			cporb_l.append((g, b, turn, tick, cporb[b, turn, tick, g]))
		nrh = self._node_rules_handled
		for b, turn, g, n, rb, rn in sort_set(nrh.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			node_rules_handled.append(
				(g, n, rb, rn, b, turn, nrh[b, turn, g, n, rb, rn])
			)
		porh = self._portal_rules_handled
		for b, r, g, o, d, rb, rn in sort_set(porh.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			portal_rules_handled.append(
				(g, o, d, rb, rn, b, r, porh[b, r, g, o, d, rb, rn])
			)
		crh = self._character_rules_handled
		for b, r, g, rb, rn in sort_set(crh.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			t = crh[b, r, g, rb, rn]
			character_rules_handled.append((g, rb, rn, b, r, t))
		urh = self._unit_rules_handled
		for (
			b,
			r,
			char,
			graph,
			node,
			rb,
			rn,
		) in sort_set(urh.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			t = urh[b, r, char, graph, node, rb, rn]
			unit_rules_handled.append((char, graph, node, rb, rn, b, r, t))
		cthrh = self._character_thing_rules_handled
		for b, r, g, rb, rn, n in sort_set(cthrh.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			t = cthrh[b, r, g, rb, rn, n]
			character_thing_rules_handled.append((g, n, rb, rn, b, r, t))
		cplrh = self._character_place_rules_handled
		for b, r, g, n, rb, rn in sort_set(cplrh.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			t = cplrh[b, r, g, n, rb, rn]
			character_place_rules_handled.append((g, n, rb, rn, b, r, t))
		cporh = self._character_portal_rules_handled
		for b, r, g, o, d, rb, rn in sort_set(cporh.keys()):
			if b != branch or not (
				(turn_from, tick_from) <= (turn, tick) <= (turn_to, tick_to)
			):
				continue
			t = cporh[b, r, g, o, d, rb, rn]
			character_portal_rules_handled.append((g, o, d, rb, rn, b, r, t))

	def _load_windows_into(
		self, ret: LoadedDict, windows: list[TimeWindow]
	) -> None:
		for branch, turn_from, tick_from, turn_to, tick_to in windows:
			self._load_window(
				ret, branch, turn_from, tick_from, turn_to, tick_to
			)

	def del_bookmark(self, key: str) -> None:
		self._bookmarks2set.cull(lambda k, _: k == key)
		if key in self._bookmarks:
			del self._bookmarks[key]

	def echo(self, string: str) -> str:
		return string

	def call(self, query_name: str, *args, **kwargs):
		raise NotImplementedError("Not a real database, so can't call it")

	def call_silent(self, query_name: str, *args, **kwargs):
		raise NotImplementedError("Not a real database, so can't call it")

	def call_many(self, query_name: str, args: list) -> None:
		raise NotImplementedError("Not a real database, so can't call it")

	def call_many_silent(self, query_name: str, args: list) -> None:
		raise NotImplementedError("Not a real database, so can't call it")

	def insert_many(self, table_name: str, args: list[dict]) -> None:
		tab_serializer = batched.serializers[table_name]
		key_len = getattr(self, batched.tables[table_name].attrname).key_len
		if key_len < 1:
			key_len = ...
		tab_spec = inspect.getfullargspec(tab_serializer)
		tab = getattr(self, "_" + table_name)
		if isinstance(tab, list):
			tab.extend(
				tuple(d[arg] for arg in tab_spec.args[1:]) for d in args
			)
		elif isinstance(tab, set):
			tab.update(
				tuple(d[arg] for arg in tab_spec.args[1:]) for d in args
			)
		elif isinstance(tab, dict):
			if key_len is ...:
				raise TypeError("dict table without key_len")
			elif key_len == 1:
				key_name = tab_spec.args[1]
				if len(tab_spec.args) == 3:
					val_name = tab_spec.args[-1]
					for d in args:
						key = d[key_name]
						tab[key] = d[val_name]
				else:
					for d in args:
						key = d[key_name]
						tab[key] = tuple(d[k] for k in tab_spec.args[2:])
			elif (
				key_len == len(tab_spec.args) - 2
			):  # the self argument, and the value
				for d in args:
					key = tuple(d[k] for k in tab_spec.args[1:-1])
					tab[key] = d[tab_spec.args[-1]]
			else:
				for d in args:
					key = tuple(d[k] for k in tab_spec.args[1 : key_len + 1])
					tab[key] = tuple(
						d[k] for k in tab_spec.args[key_len + 1 :]
					)
		else:
			raise TypeError("Don't know how to insert here", tab)

	insert_many_silent = insert_many

	def delete_many_silent(self, table_name: str, args: list[dict]) -> None:
		cached: cached_property = batched.tables[table_name]
		the_batch: Batch = getattr(self, cached.attrname)
		tab_serializer = batched.serializers[table_name]
		tab_spec = inspect.getfullargspec(tab_serializer)
		tab = getattr(self, "_" + table_name)
		if the_batch.key_len >= 1:
			key_len = the_batch.key_len
			key_args = tab_spec.args[1 : the_batch.key_len]
		else:
			key_args = tab_spec.args[1:]
			key_len = len(key_args)
		keys2del = set(tuple(d[arg] for arg in key_args) for d in args)
		if isinstance(tab, list):
			setattr(
				self,
				"_" + table_name,
				list(
					filterfalse(
						lambda t: t[:key_len] in keys2del,
						tab,
					)
				),
			)
		elif isinstance(tab, dict):
			for key in keys2del & tab.keys():
				del tab[key]
		elif isinstance(tab, set):
			tab.difference_update(keys2del)
		else:
			raise TypeError("Don't know how to delete from this table", tab)

	def get_keyframe_extensions(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> tuple[UniversalKeyframe, RuleKeyframe, RulebookKeyframe]:
		return self._keyframe_extensions[branch, turn, tick]

	def keyframes_dump(self) -> Iterator[tuple[Branch, Turn, Tick]]:
		with self._lock:
			yield from sorted(self._keyframes)

	def delete_keyframe(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		with self._lock:
			self._keyframes.remove((branch, turn, tick))
			del self._keyframe_extensions[branch, turn, tick]

	def keyframes_graphs(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick]]:
		with self._lock:
			for b, r, t, g in sort_set(self._keyframes):
				yield g, b, r, t

	def have_branch(self, branch: Branch) -> bool:
		return branch in self._branches

	def branches_dump(
		self,
	) -> Iterator[tuple[Branch, Branch, Turn, Tick, Turn, Tick]]:
		with self._lock:
			for branch in sort_set(self._branches.keys()):
				parent, r0, t0, r1, t1 = self._branches[branch]
				yield branch, parent, r0, t0, r1, t1

	def global_get(self, key: EternalKey) -> Value:
		return self.eternal[key]

	def global_dump(self) -> Iterator[tuple[Key, Value]]:
		with self._lock:
			yield from self.eternal.items()

	def get_branch(self) -> Branch:
		b = self.eternal[EternalKey(Key("branch"))]
		assert isinstance(b, str)
		return Branch(b)

	def get_turn(self) -> Turn:
		r = self.eternal[EternalKey(Key("turn"))]
		assert isinstance(r, int)
		return Turn(r)

	def get_tick(self) -> Tick:
		t = self.eternal[EternalKey(Key("tick"))]
		assert isinstance(t, int)
		return Tick(t)

	def turns_dump(self) -> Iterator[tuple[Branch, Turn, Tick, Tick]]:
		with self._lock:
			for (branch, turn), (
				end_tick,
				plan_end_tick,
			) in sorted(self._turns.items()):
				yield branch, turn, end_tick, plan_end_tick

	def graph_val_dump(self) -> Iterator[GraphValRowType]:
		with self._lock:
			gv = self._graph_val
			for b, r, t, g, k in sort_set(gv.keys()):
				yield g, k, b, r, t, gv[b, r, t, g, k]

	def graph_val_del_time(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		todel = set()
		for b, r, t, g in self._graph_val:
			if (b, r, t) == (branch, turn, tick):
				todel.add((b, r, t, g))
		for k in todel:
			del self._graph_val[k]

	def edges_del_time(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		todel = set()
		for b, r, t, g, o, d in self._edges:
			if (b, r, t) == (branch, turn, tick):
				todel.add((b, r, t, g, o, d))
		for k in todel:
			del self._edges[k]

	def graphs_types(
		self,
		branch: Branch,
		turn_from: Turn,
		tick_from: Tick,
		turn_to: Optional[Turn] = None,
		tick_to: Optional[Tick] = None,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, str]]:
		if (turn_to is None) ^ (tick_to is None):
			raise TypeError(
				"Need both or neither of 'turn_to' and 'tick_to'",
				turn_to,
				tick_to,
			)
		if turn_to is None:
			with self._lock:
				for (b, r, t, g), v in self._graphs.items():
					if b != branch or not ((turn_from, tick_from) <= (r, t)):
						continue
					yield g, b, r, t, v
			return
		with self._lock:
			for (b, r, t, g), v in self._graphs.items():
				if b != branch or not (
					(turn_from, tick_from) <= (r, t) < (turn_to, tick_to)
				):
					continue
				yield g, b, r, t, v

	def graphs_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, str]]:
		with self._lock:
			for b, r, t, g in sort_set(self._graphs.keys()):
				yield g, b, r, t, self._graphs[b, r, t, g]

	def nodes_del_time(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		with self._lock:
			for k in {
				(b, r, t, g, n)
				for (b, r, t, g, n) in self._nodes
				if (b, r, t) == (branch, turn, tick)
			}:
				del self._nodes[k]

	def nodes_dump(self) -> Iterator[NodeRowType]:
		with self._lock:
			for b, r, t, g, n in sort_set(self._nodes.keys()):
				yield g, n, b, r, t, self._nodes[b, r, t, g, n]

	def node_val_dump(self) -> Iterator[NodeValRowType]:
		with self._lock:
			for b, r, t, g, n, k in sort_set(self._node_val.keys()):
				yield g, n, k, b, r, t, self._node_val[b, r, t, g, n, k]

	def node_val_del_time(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		with self._lock:
			for key in {
				(b, r, t, g, n, k)
				for (b, r, t, g, n, k) in self._node_val
				if (b, r, t) == (branch, turn, tick)
			}:
				del self._node_val[key]

	def edges_dump(self) -> Iterator[EdgeRowType]:
		with self._lock:
			for b, r, t, g, o, d in sort_set(self._edges.keys()):
				yield g, o, d, b, r, t, self._edges[b, r, t, g, o, d]

	def edge_val_dump(self) -> Iterator[EdgeValRowType]:
		with self._lock:
			for b, r, t, g, o, d, k in sort_set(self._edge_val.keys()):
				yield g, o, d, k, b, r, t, self._edge_val[b, r, t, g, o, d, k]

	def edge_val_del_time(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		with self._lock:
			for key in {
				(b, r, t, g, o, d, k)
				for (b, r, t, g, o, d, k) in self._edge_val
				if (b, r, t) == (branch, turn, tick)
			}:
				del self._edge_val[key]

	def plan_ticks_dump(self) -> Iterator[tuple[Plan, Branch, Turn, Tick]]:
		with self._lock:
			for plan_id in sort_set(self._plans.keys()):
				yield plan_id, *self._plans[plan_id]

	commit = close = AbstractDatabaseConnector.flush

	def _init_db(self) -> None:
		pass

	def truncate_all(self) -> None:
		for table in batched.tables:
			getattr(self, "_" + table).clear()

	def get_all_keyframe_graphs(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> Iterator[tuple[CharName, NodeKeyframe, EdgeKeyframe, StatDict]]:
		with self._lock:
			for (b, r, t, g), (
				nkf,
				ekf,
				gvkf,
			) in self._keyframes_graphs.items():
				if (b, r, t) != (branch, turn, tick):
					continue
				yield g, nkf, ekf, gvkf

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
			StatDict,
		]
	]:
		with self._lock:
			for b, r, t, g in sort_set(self._keyframes_graphs.keys()):
				yield g, b, r, t, *self._keyframes_graphs[b, r, t, g]

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
		with self._lock:
			for b, r, t in sort_set(self._keyframe_extensions.keys()):
				yield b, r, t, *self._keyframe_extensions[b, r, t]

	def universals_dump(
		self,
	) -> Iterator[tuple[Key, Branch, Turn, Tick, Value]]:
		with self._lock:
			for b, r, t, k in sort_set(self._universals.keys()):
				yield k, b, r, t, *self._universals[b, r, t, k]

	def rulebooks_dump(
		self,
	) -> Iterator[
		tuple[RulebookName, Branch, Turn, Tick, tuple[list[RuleName], float]]
	]:
		with self._lock:
			for b, r, t, rb in sorted(self._rulebooks.keys()):
				(rs, prio) = self._rulebooks[b, r, t, rb]
				yield rb, b, r, t, (rs.copy(), prio)

	def rules_dump(self) -> Iterator[RuleName]:
		with self._lock:
			yield from sort_set(self._rules)

	def rule_triggers_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, list[TriggerFuncName]]]:
		with self._lock:
			for b, r, t, rn in sort_set(self._rule_triggers.keys()):
				yield rn, b, r, t, self._rule_triggers[b, r, t, rn].copy()

	def rule_prereqs_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, list[PrereqFuncName]]]:
		with self._lock:
			for b, r, t, rn in sort_set(self._rule_prereqs.keys()):
				yield rn, b, r, t, self._rule_prereqs[b, r, t, rn].copy()

	def rule_actions_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, list[ActionFuncName]]]:
		with self._lock:
			for b, r, t, rn in sort_set(self._rule_actions.keys()):
				yield rn, b, r, t, self._rule_actions[b, r, t, rn].copy()

	def rule_neighborhood_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, RuleNeighborhood]]:
		with self._lock:
			for b, r, t, rn in sort_set(self._rule_neighborhood.keys()):
				yield rn, b, r, t, self._rule_neighborhood[b, r, t, rn]

	def rule_big_dump(
		self,
	) -> Iterator[tuple[RuleName, Branch, Turn, Tick, RuleBig]]:
		with self._lock:
			for b, r, t, rn in sort_set(self._rule_big.keys()):
				yield rn, b, r, t, self._rule_big[b, r, t, rn]

	def node_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, NodeName, Branch, Turn, Tick, RulebookName]]:
		with self._lock:
			for b, r, t, g, n in sort_set(self._node_rulebook.keys()):
				yield g, n, b, r, t, self._node_rulebook[b, r, t, g, n]

	def portal_rulebook_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, NodeName, Branch, Turn, Tick, RulebookName]
	]:
		with self._lock:
			for b, r, t, g, o, d in sort_set(self._portal_rulebook.keys()):
				yield g, o, d, b, r, t, self._portal_rulebook[b, r, t, g, o, d]

	def character_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		with self._lock:
			for b, r, t, g in sort_set(self._character_rulebook.keys()):
				yield g, b, r, t, self._character_rulebook[b, r, t, g]

	def unit_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		with self._lock:
			for b, r, t, g in sort_set(self._unit_rulebook.keys()):
				yield g, b, r, t, self._unit_rulebook[b, r, t, g]

	def character_thing_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		with self._lock:
			for b, r, t, g in sort_set(self._character_thing_rulebook.keys()):
				yield g, b, r, t, self._character_thing_rulebook[b, r, t, g]

	def character_place_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		with self._lock:
			for b, r, t, g in sort_set(self._character_place_rulebook.keys()):
				yield g, b, r, t, self._character_place_rulebook[b, r, t, g]

	def character_portal_rulebook_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, RulebookName]]:
		with self._lock:
			for b, r, t, g in sort_set(self._character_portal_rulebook.keys()):
				yield g, b, r, t, self._character_portal_rulebook[b, r, t, g]

	def character_rules_handled_dump(
		self,
	) -> Iterator[tuple[CharName, RulebookName, RuleName, Branch, Turn, Tick]]:
		with self._lock:
			crh = self._character_rules_handled
			for b, r, g, rb, rn in sort_set(crh.keys()):
				t = crh[b, r, g, rb, rn]
				yield g, rb, rn, b, r, t

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
		with self._lock:
			urh = self._unit_rules_handled
			for b, r, char, graph, node, rb, rn in sort_set(urh.keys()):
				t = urh[b, r, char, graph, node, rb, rn]
				yield char, graph, node, rb, rn, b, r, t

	def character_thing_rules_handled_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, RulebookName, RuleName, Branch, Turn, Tick]
	]:
		with self._lock:
			ctrh = self._character_thing_rules_handled
			for b, r, g, rb, rn, n in sort_set(ctrh.keys()):
				t = ctrh[b, r, g, rb, rn, n]
				yield g, n, rb, rn, b, r, t

	def character_place_rules_handled_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, RulebookName, RuleName, Branch, Turn, Tick]
	]:
		with self._lock:
			cprh = self._character_place_rules_handled
			for b, r, g, n, rb, rn in sort_set(cprh.keys()):
				t = cprh[b, r, g, n, rb, rn]
				yield g, n, rb, rn, b, r, t

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
		with self._lock:
			cporh = self._character_portal_rules_handled
			for b, r, g, o, d, rb, rn in sort_set(cporh.keys()):
				t = cporh[b, r, g, o, d, rb, rn]
				yield g, o, d, rb, rn, b, r, t

	def node_rules_handled_dump(
		self,
	) -> Iterator[
		tuple[CharName, NodeName, RulebookName, RuleName, Branch, Turn, Tick]
	]:
		with self._lock:
			nrh = self._node_rules_handled
			for b, r, g, n, rb, rn in sort_set(nrh.keys()):
				t = nrh[b, r, g, n, rb, rn]
				yield g, n, rb, rn, b, r, t

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
		with self._lock:
			porh = self._portal_rules_handled
			for b, r, g, o, d, rb, rn in sort_set(porh.keys()):
				t = porh[b, r, g, o, d, rb, rn]
				yield g, o, d, rb, rn, b, r, t

	def things_dump(
		self,
	) -> Iterator[tuple[CharName, NodeName, Branch, Turn, Tick, NodeName]]:
		with self._lock:
			for b, r, t, g, n in sort_set(self._things.keys()):
				yield g, n, b, r, t, self._things[b, r, t, g, n]

	def units_dump(
		self,
	) -> Iterator[
		tuple[CharName, CharName, NodeName, Branch, Turn, Tick, bool]
	]:
		with self._lock:
			for b, r, t, char, graph, node in sort_set(self._units.keys()):
				units = self._units[b, r, t, char, graph, node]
				yield char, graph, node, b, r, t, units

	def count_all_table(self, tbl: str) -> int:
		return len(getattr(self, "_" + tbl))

	def rules_insert(self, rule: RuleName):
		with self._lock:
			self._rules.add(rule)

	def things_del_time(self, branch: Branch, turn: Turn, tick: Tick):
		with self._lock:
			for key in {
				(b, r, t, g, n)
				for (b, r, t, g, n) in self._things
				if (b, r, t) == (branch, turn, tick)
			}:
				del self._things[key]

	def turns_completed_dump(self) -> Iterator[tuple[Branch, Turn]]:
		with self._lock:
			yield from sorted(self._turns_completed.items())

	def bookmarks_dump(self) -> Iterator[tuple[Key, Time]]:
		with self._lock:
			yield from sort_set(self._bookmarks.items())


@dataclass
class NullDatabaseConnector(AbstractDatabaseConnector):
	"""Database connector that does nothing, connects to no database

	This will never return any data, either. If you want it to hold data
	you put into it, instead use :class:`PythonDatabaseConnector`.

	"""

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

	def graphs_dump(
		self,
	) -> Iterator[tuple[CharName, Branch, Turn, Tick, str]]:
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

	def bookmarks_dump(self) -> Iterator[tuple[Key, Time]]:
		return iter(())

	def set_bookmark(
		self, key: Key, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		pass

	def del_bookmark(self, key: Key) -> None:
		pass

	def load_windows(self, windows: list[TimeWindow]) -> dict:
		return {}


def window_getter(
	table: str,
	f: Callable[[Branch, ...], None] | partial | None = None,
	per_character: bool = False,
):
	"""Decorator for functions that get a window of time from the output queue"""
	if f is None:
		return partial(window_getter, table, per_character=per_character)

	if per_character:
		if isinstance(f, partial):
			argspec = inspect.getfullargspec(f.func)
		else:
			argspec = inspect.getfullargspec(f)
		if "return" not in argspec.annotations:
			raise TypeError("No character in return annotation", f)
		ret_sig = argspec.annotations["return"]
		if isinstance(ret_sig, str):
			ret_sig = eval(ret_sig)
		char_index = get_args(ret_sig).index(CharName)
		if char_index is None:
			raise TypeError(
				"per_character window getter needs CharName in its return signature"
			)

		def get_a_window(
			self,
			ret: dict,
			branch: Branch,
			turn_from: Turn,
			tick_from: Tick,
			turn_to: Turn | None,
			tick_to: Tick | None,
		) -> None:
			if (got := self._outq.get()) != (
				"begin",
				table,
				branch,
				turn_from,
				tick_from,
				turn_to,
				tick_to,
			):
				raise RuntimeError("Expected beginning of " + table, got)
			self._outq.task_done()
			while isinstance(got := self._outq.get(), list):
				for rec in got:
					charn = rec[char_index]
					ret[charn][table].append(f(self, branch, *rec))
				self._outq.task_done()
			if got != (
				"end",
				table,
				branch,
				turn_from,
				tick_from,
				turn_to,
				tick_to,
			):
				raise RuntimeError("Expected end of " + table, got)
			self._outq.task_done()

		window_getter.tables[table] = get_a_window
		return get_a_window

	def get_a_window(
		self,
		ret: dict,
		branch: Branch,
		turn_from: Turn,
		tick_from: Tick,
		turn_to: Turn | None,
		tick_to: Tick | None,
	) -> None:
		if (got := self._outq.get()) != (
			"begin",
			table,
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError("Expected beginning of " + table, got)
		self._outq.task_done()
		while isinstance(got := self._outq.get(), list):
			ret[table].extend(starmap(partial(f, self, branch), got))
			self._outq.task_done()
		if got != (
			"end",
			table,
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		):
			raise RuntimeError("Expected end of " + table, got)
		self._outq.task_done()

	window_getter.tables[table] = get_a_window
	return get_a_window


window_getter.tables = {}


class ThreadedDatabaseConnector(AbstractDatabaseConnector):
	Looper: ClassVar[type[ConnectionLooper]]
	clear: bool

	def __post_init__(self):
		self._t = Thread(target=self._looper.run)
		self._t.start()
		if self.clear:
			self.truncate_all()

	@cached_property
	def _looper(self) -> ConnectionLooper:
		return self.Looper(self)

	@cached_property
	def _lock(self):
		return self._looper.lock

	@cached_property
	def _inq(self) -> Queue:
		return Queue()

	@cached_property
	def _outq(self) -> Queue:
		return Queue()

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
		with self._lock:
			yield
			if self._outq.qsize() != 0:
				raise RuntimeError("Unhandled items in output queue")

	@mutexed
	def _load_windows_into(self, ret: dict, windows: list[TimeWindow]) -> None:
		assert "graphs" in ret
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

	def unpack_key(self, k: bytes) -> Key:
		unpacked = self.unpack(k)
		if not isinstance(unpacked, Key):
			raise TypeError("Invalid key", unpacked)
		return unpacked

	@window_getter("graphs")
	def _unpack_graph_rec(
		self,
		branch: Branch,
		graph_b: bytes,
		turn: Turn,
		tick: Tick,
		typ_s: GraphTypeStr,
	) -> tuple[CharName, Branch, Turn, Tick, GraphTypeStr]:
		graph = CharName(self.unpack_key(graph_b))
		if typ_s not in get_args(GraphTypeStr):
			raise TypeError("Unknown graph type", typ_s)
		return graph, branch, turn, tick, typ_s

	@window_getter("nodes", per_character=True)
	def _unpack_node_rec(
		self,
		branch: Branch,
		graph_b: bytes,
		node_b: bytes,
		turn: Turn,
		tick: Tick,
		ex: bool,
	) -> tuple[CharName, NodeName, Branch, Turn, Tick, bool]:
		graph = CharName(self.unpack_key(graph_b))
		node = NodeName(self.unpack_key(node_b))
		return graph, node, branch, turn, tick, ex

	@window_getter("edges", per_character=True)
	def _unpack_edge_rec(
		self,
		branch: Branch,
		graph_b: bytes,
		orig_b: bytes,
		dest_b: bytes,
		turn: Turn,
		tick: Tick,
		ex: bool,
	) -> tuple[CharName, NodeName, NodeName, Branch, Turn, Tick, bool]:
		graph = CharName(self.unpack_key(graph_b))
		orig = NodeName(self.unpack_key(orig_b))
		dest = NodeName(self.unpack_key(dest_b))
		return graph, orig, dest, branch, turn, tick, ex

	@window_getter("graph_val", per_character=True)
	def _unpack_graph_val_rec(
		self,
		branch: Branch,
		graph_b: bytes,
		key_b: bytes,
		turn: Turn,
		tick: Tick,
		val_b: bytes,
	) -> tuple[CharName, Stat, Branch, Turn, Tick, Value]:
		graph = CharName(self.unpack_key(graph_b))
		stat = Stat(self.unpack_key(key_b))
		val = self.unpack(val_b)
		return graph, stat, branch, turn, tick, val

	@window_getter("node_val", per_character=True)
	def _unpack_node_val_rec(
		self,
		branch: Branch,
		graph_b: bytes,
		node_b: bytes,
		key_b: bytes,
		turn: Turn,
		tick: Tick,
		val_b: bytes,
	) -> tuple[CharName, NodeName, Stat, Branch, Turn, Tick, Value]:
		graph = CharName(self.unpack_key(graph_b))
		node = NodeName(self.unpack_key(node_b))
		key = Stat(self.unpack_key(key_b))
		val = self.unpack(val_b)
		return graph, node, key, branch, turn, tick, val

	@window_getter("edge_val", per_character=True)
	def _unpack_edge_val_rec(
		self,
		branch: Branch,
		graph_b: bytes,
		orig_b: bytes,
		dest_b: bytes,
		key_b: bytes,
		turn: Turn,
		tick: Tick,
		val_b: bytes,
	) -> tuple[CharName, NodeName, NodeName, Stat, Branch, Turn, Tick, Value]:
		graph = CharName(self.unpack_key(graph_b))
		orig = NodeName(self.unpack_key(orig_b))
		dest = NodeName(self.unpack_key(dest_b))
		key = Stat(self.unpack_key(key_b))
		val = self.unpack(val_b)
		return graph, orig, dest, key, branch, turn, tick, val

	@window_getter("things", per_character=True)
	def _unpack_thing_rec(
		self,
		branch: Branch,
		graph_b: bytes,
		node_b: bytes,
		turn: Turn,
		tick: Tick,
		loc_b: bytes,
	) -> tuple[CharName, NodeName, Branch, Turn, Tick, NodeName | type(...)]:
		graph = CharName(self.unpack_key(graph_b))
		thing = NodeName(self.unpack_key(node_b))
		loc = self.unpack(loc_b)
		if loc is not ...:
			if not isinstance(loc, Key):
				raise TypeError("Invalid location", loc)
			loc = NodeName(loc)
		return graph, thing, branch, turn, tick, loc

	@window_getter("units", per_character=True)
	def _unpack_unit_rec(
		self,
		branch: Branch,
		char_b: bytes,
		graph_b: bytes,
		node_b: bytes,
		turn: Turn,
		tick: Tick,
		is_unit: bool,
	) -> tuple[CharName, CharName, NodeName, Branch, Turn, Tick, bool]:
		character = CharName(self.unpack_key(char_b))
		graph = CharName(self.unpack_key(graph_b))
		unit = NodeName(self.unpack_key(node_b))
		return character, graph, unit, branch, turn, tick, is_unit

	def _unpack_char_rb_rec(
		self,
		branch: Branch,
		char_b: bytes,
		turn: Turn,
		tick: Tick,
		rb_b: bytes,
	) -> tuple[CharName, Branch, Turn, Tick, RulebookName]:
		character = CharName(self.unpack_key(char_b))
		rulebook = RulebookName(self.unpack_key(rb_b))
		return character, branch, turn, tick, rulebook

	_unpack_character_rulebook_rec = window_getter(
		"character_rulebook", _unpack_char_rb_rec, True
	)
	_unpack_unit_rb_rec = window_getter(
		"unit_rulebook", _unpack_char_rb_rec, True
	)
	_unpack_char_thing_rb_rec = window_getter(
		"character_thing_rulebook", _unpack_char_rb_rec, True
	)
	_unpack_char_place_rb_rec = window_getter(
		"character_place_rulebook", _unpack_char_rb_rec, True
	)
	_unpack_char_portal_rb_rec = window_getter(
		"character_portal_rulebook", _unpack_char_rb_rec, True
	)

	@window_getter("node_rulebook", per_character=True)
	def _unpack_node_rb_rec(
		self,
		branch: Branch,
		graph_b: bytes,
		node_b: bytes,
		turn: Turn,
		tick: Tick,
		rb_b: bytes,
	) -> tuple[CharName, NodeName, Branch, Turn, Tick, RulebookName]:
		graph = CharName(self.unpack_key(graph_b))
		node = NodeName(self.unpack_key(node_b))
		rulebook = RulebookName(self.unpack_key(rb_b))
		return graph, node, branch, turn, tick, rulebook

	@window_getter("portal_rulebook", per_character=True)
	def _unpack_port_rb_rec(
		self,
		branch: Branch,
		graph_b: bytes,
		orig_b: bytes,
		dest_b: bytes,
		turn: Turn,
		tick: Tick,
		rb_b: bytes,
	) -> tuple[CharName, NodeName, NodeName, Branch, Turn, Tick, RulebookName]:
		graph = CharName(self.unpack_key(graph_b))
		orig = NodeName(self.unpack_key(orig_b))
		dest = NodeName(self.unpack_key(dest_b))
		rulebook = RulebookName(self.unpack_key(rb_b))
		return graph, orig, dest, branch, turn, tick, rulebook

	@window_getter("universals")
	def _unpack_universal_rec(
		self,
		branch: Branch,
		key_b: bytes,
		turn: Turn,
		tick: Tick,
		value_b: bytes,
	) -> tuple[UniversalKey, Branch, Turn, Tick, Value]:
		key = UniversalKey(self.unpack_key(key_b))
		value = self.unpack(value_b)
		return key, branch, turn, tick, value

	@window_getter("rulebooks")
	def _unpack_rulebook_rec(
		self,
		branch: Branch,
		rb_b: bytes,
		turn: Turn,
		tick: Tick,
		rules_b: bytes,
		prio: RulebookPriority,
	) -> tuple[
		RulebookName,
		Branch,
		Turn,
		Tick,
		tuple[list[RuleName], RulebookPriority],
	]:
		rulebook = RulebookName(self.unpack_key(rb_b))
		rules = self.unpack(rules_b)
		if not isinstance(rules, list):
			raise TypeError("Invalid rules list", rules)
		return rulebook, branch, turn, tick, (list(map(rulename, rules)), prio)

	@window_getter("rule_triggers")
	def _unpack_rule_trigs_rec(
		self,
		branch: Branch,
		rule: RuleName,
		turn: Turn,
		tick: Tick,
		trigs_b: bytes,
	) -> tuple[RuleName, Branch, Turn, Tick, list[TriggerFuncName]]:
		trigs = self.unpack(trigs_b)
		if not isinstance(trigs, list):
			raise TypeError("Invalid triggers list", trigs)
		return rule, branch, turn, tick, list(map(trigfuncn, trigs))

	@window_getter("rule_prereqs")
	def _unpack_rule_preqs_rec(
		self,
		branch: Branch,
		rule: RuleName,
		turn: Turn,
		tick: Tick,
		preqs_b: bytes,
	) -> tuple[RuleName, Branch, Turn, Tick, list[PrereqFuncName]]:
		preqs = self.unpack(preqs_b)
		if not isinstance(preqs, list):
			raise TypeError("Invalid prereqs list", preqs)
		return rule, branch, turn, tick, list(map(preqfuncn, preqs))

	@window_getter("rule_actions")
	def _unpack_rule_acts_rec(
		self,
		branch: Branch,
		rule: RuleName,
		turn: Turn,
		tick: Tick,
		acts_b: bytes,
	) -> tuple[RuleName, Branch, Turn, Tick, list[ActionFuncName]]:
		acts = self.unpack(acts_b)
		if not isinstance(acts, list):
			raise TypeError("Invalid actions list", acts)
		return rule, branch, turn, tick, list(map(actfuncn, acts))

	@window_getter("rule_neighborhood")
	def _unpack_rule_nbrs_rec(
		self,
		branch: Branch,
		rule: RuleName,
		turn: Turn,
		tick: Tick,
		neighbors: RuleNeighborhood,
	) -> tuple[RuleName, Branch, Turn, Tick, RuleNeighborhood]:
		if neighbors is not None and not (
			isinstance(neighbors, int) and neighbors >= 0
		):
			raise TypeError("Invalid rule neighborhood", neighbors)
		return rule, branch, turn, tick, neighbors

	@window_getter("rule_big")
	def _unpack_rule_big_rec(
		self,
		branch: Branch,
		rule: RuleName,
		turn: Turn,
		tick: Tick,
		big: RuleBig,
	) -> tuple[RuleName, Branch, Turn, Tick, RuleBig]:
		return rule, branch, turn, tick, big

	@window_getter("character_rules_handled")
	def _unpack_char_rule_handled_rec(
		self,
		branch: Branch,
		char_b: bytes,
		rb_b: bytes,
		rule: RuleName,
		turn: Turn,
		tick: Tick,
	) -> CharacterRulesHandledRowType:
		char = CharName(self.unpack_key(char_b))
		rb = RulebookName(self.unpack_key(rb_b))
		return char, rb, rule, branch, turn, tick

	@window_getter("unit_rules_handled")
	def _unpack_unit_rule_handled_rec(
		self,
		branch: Branch,
		char_b: bytes,
		graph_b: bytes,
		unit_b: bytes,
		rb_b: bytes,
		rule: RuleName,
		turn: Turn,
		tick: Tick,
	) -> UnitRulesHandledRowType:
		char = CharName(self.unpack_key(char_b))
		graph = CharName(self.unpack_key(graph_b))
		unit = NodeName(self.unpack_key(unit_b))
		rb = RulebookName(self.unpack_key(rb_b))
		return char, graph, unit, rb, rule, branch, turn, tick

	@window_getter("character_thing_rules_handled")
	def _unpack_char_thing_rule_handled_rec(
		self,
		branch: Branch,
		char_b: bytes,
		thing_b: bytes,
		rb_b: bytes,
		rule: RuleName,
		turn: Turn,
		tick: Tick,
	) -> NodeRulesHandledRowType:
		char = CharName(self.unpack_key(char_b))
		thing = NodeName(self.unpack_key(thing_b))
		rulebook = RulebookName(self.unpack_key(rb_b))
		return char, thing, rulebook, rule, branch, turn, tick

	@window_getter("character_place_rules_handled")
	def _unpack_char_place_rule_handled_rec(
		self,
		branch: Branch,
		char_b: bytes,
		place_b: bytes,
		rb_b: bytes,
		rule: RuleName,
		turn: Turn,
		tick: Tick,
	) -> NodeRulesHandledRowType:
		char = CharName(self.unpack_key(char_b))
		place = NodeName(self.unpack_key(place_b))
		rulebook = RulebookName(self.unpack_key(rb_b))
		return char, place, rulebook, rule, branch, turn, tick

	@window_getter("character_portal_rules_handled")
	def _unpack_char_portal_rule_handled_rec(
		self,
		branch: Branch,
		char_b: bytes,
		orig_b: bytes,
		dest_b: bytes,
		rb_b: bytes,
		rule: RuleName,
		turn: Turn,
		tick: Tick,
	) -> PortalRulesHandledRowType:
		char = CharName(self.unpack_key(char_b))
		orig = NodeName(self.unpack_key(orig_b))
		dest = NodeName(self.unpack_key(dest_b))
		rb = RulebookName(self.unpack_key(rb_b))
		return char, orig, dest, rb, rule, branch, turn, tick

	@window_getter("node_rules_handled")
	def _unpack_node_rule_handled_rec(
		self,
		branch: Branch,
		char_b: bytes,
		node_b: bytes,
		rb_b: bytes,
		rule: RuleName,
		turn: Turn,
		tick: Tick,
	) -> NodeRulesHandledRowType:
		char = CharName(self.unpack_key(char_b))
		node = NodeName(self.unpack_key(node_b))
		rb = RulebookName(self.unpack_key(rb_b))
		return char, node, rb, rule, branch, turn, tick

	@window_getter("portal_rules_handled")
	def _unpack_portal_rule_handled_rec(
		self,
		branch: Branch,
		char_b: bytes,
		orig_b: bytes,
		dest_b: bytes,
		rb_b: RuleName,
		turn: Turn,
		tick: Tick,
	) -> PortalRulesHandledRowType:
		char = CharName(self.unpack_key(char_b))
		orig = NodeName(self.unpack_key(orig_b))
		dest = NodeName(self.unpack_key(dest_b))
		rb = RulebookName(self.unpack_key(rb_b))
		return char, orig, dest, rb, rule, branch, turn, tick

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
		for table in self._infixes2load:
			window_getter.tables[table](
				self, ret, branch, turn_from, tick_from, turn_to, tick_to
			)


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
