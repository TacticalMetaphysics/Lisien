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
from ast import literal_eval
from collections import UserDict, defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, KW_ONLY
from functools import cached_property, partial, wraps, partialmethod
from io import IOBase, StringIO
from itertools import filterfalse, starmap
from pathlib import Path
from queue import Queue
from threading import Lock, Thread
from types import MethodType, FunctionType
from typing import (
	Any,
	Callable,
	Iterable,
	Iterator,
	Literal,
	MutableMapping,
	Optional,
	Union,
	get_args,
	get_type_hints,
	TypeVar,
	TYPE_CHECKING,
	ClassVar,
	Set,
	MutableSet,
	Mapping,
)

import networkx as nx
from tblib import Traceback


if TYPE_CHECKING:
	from xml.etree.ElementTree import (
		ElementTree,
		Element,
		indent as indent_tree,
		parse,
	)
else:
	try:
		from lxml.etree import (
			ElementTree,
			Element,
			indent as indent_tree,
			parse,
		)
	except ImportError:
		from xml.etree.ElementTree import (
			ElementTree,
			Element,
			indent as indent_tree,
			parse,
		)

from .window import SettingsTurnDict, WindowDict
import lisien.types
from .types import (
	ActionFuncName,
	actfuncn,
	ActionRowType,
	Branch,
	CharDict,
	CharName,
	CharRulebookRowType,
	EdgeKeyframe,
	EdgeRowType,
	EdgeValRowType,
	EternalKey,
	FuncName,
	GraphRowType,
	GraphTypeStr,
	GraphNodeValKeyframe,
	GraphEdgeValKeyframe,
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
	preqfuncn,
	PrereqRowType,
	RuleBig,
	RuleBigRowType,
	RulebookKeyframe,
	RulebookName,
	RulebookPriority,
	RulebookRowType,
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
from .facade import EngineFacade
from .util import garbage, sort_set, AbstractEngine
from .wrap import DictWrapper, ListWrapper, SetWrapper

if sys.version_info.minor < 11:

	class ExceptionGroup(Exception):
		pass


SCHEMAVER_B = b"\xb6_lisien_schema_version"
SCHEMA_VERSION = 2
SCHEMA_VERSION_B = SCHEMA_VERSION.to_bytes(1, "little")
XML_SCHEMA_VERSION = 2


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
				self,
				k,
			)
		elif isinstance(ret, list):
			return ListWrapper(
				lambda: super().__getitem__(k),
				self,
				k,
			)
		elif isinstance(ret, set):
			return SetWrapper(
				lambda: super().__getitem__(k),
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
	def engine(self) -> AbstractEngine:
		return EngineFacade(None)

	@cached_property
	def tree(self) -> ElementTree:
		return ElementTree(Element("lisien"))

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
		"""Return the whole database in a Python dictionary.

		You should probably use ``to_xml`` instead, but this could be helpful
		for debugging, or if you have your own ideas about serialization.

		"""
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
	def _things2set(
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
		for table, serializer in batched.serializers.items():
			batch = getattr(self, serializer.__name__)
			if not isinstance(batch, Batch):
				raise TypeError(
					"Batch was overwritten", table, serializer.__name__, batch
				)
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

	@abstractmethod
	def echo(self, *args): ...

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
		self._things2set.append((character, thing, branch, turn, tick, loc))

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

	def to_etree(self, name: str) -> ElementTree:
		root = self.tree.getroot()
		self.commit()
		eternals = dict(self.eternal.items())
		root.set(
			"db-schema-version", str(eternals.pop("_lisien_schema_version"))
		)
		root.set("xml-schema-version", str(XML_SCHEMA_VERSION))
		root.set("trunk", str(eternals.pop("trunk")))
		root.set("branch", str(eternals.pop("branch")))
		root.set("turn", str(eternals.pop("turn")))
		root.set("tick", str(eternals.pop("tick")))
		if "language" in eternals:
			root.set("language", str(eternals.pop("language")))
		for k in sort_set(eternals.keys()):
			el = Element("dict-item", key=repr(k))
			root.append(el)
			el.append(self._value_to_xml_el(eternals[k]))
		plan_ticks: dict[Branch, dict[Turn, dict[Plan, set[Tick]]]] = {}
		for plan, branch, turn, tick in self.plan_ticks_dump():
			if branch in plan_ticks:
				if turn in plan_ticks[branch]:
					if plan in plan_ticks[branch][turn]:
						plan_ticks[branch][turn][plan].add(tick)
					else:
						plan_ticks[branch][turn][plan] = {tick}
				else:
					plan_ticks[branch][turn] = {plan: {tick}}
			else:
				plan_ticks[branch] = {turn: {plan: {tick}}}
		self._plan_ticks = plan_ticks
		trunks = set()
		branches_d = {}
		branch_descendants = {}
		turn_end_plan_d: dict[Branch, dict[Turn, tuple[Tick, Tick]]] = {}
		branch_elements = {}
		playtrees: dict[Branch, Element] = {}
		turns_completed_d: dict[Branch, Turn] = dict(
			self.turns_completed_dump()
		)
		keyframe_times: set[Time] = set(self.keyframes_dump())
		for (
			branch,
			turn,
			last_real_tick,
			last_planned_tick,
		) in self.turns_dump():
			if branch in turn_end_plan_d:
				turn_end_plan_d[branch][turn] = (
					last_real_tick,
					last_planned_tick,
				)
			else:
				turn_end_plan_d[branch] = {
					turn: (last_real_tick, last_planned_tick)
				}
		branch2do = deque(sorted(self.branches_dump()))
		while branch2do:
			(
				branch,
				parent,
				parent_turn,
				parent_tick,
				end_turn,
				end_tick,
			) = branch2do.popleft()
			branches_d[branch] = (
				parent,
				parent_turn,
				parent_tick,
				end_turn,
				end_tick,
			)
			if parent is None:
				trunks.add(branch)
				playtree = Element("playtree", game=name, trunk=branch)
				playtrees[branch] = playtree
				branch_element = branch_elements[branch] = Element(
					"branch",
					{
						"name": branch,
						"start-turn": "0",
						"start-tick": "0",
						"end-turn": str(end_turn),
						"end-tick": str(end_tick),
					},
				)
				if branch in turns_completed_d:
					branch_element.set(
						"last-turn-completed", str(turns_completed_d[branch])
					)
				root.append(playtree)
				playtree.append(branch_element)
			else:
				if parent in branch_descendants:
					branch_descendants[parent].add(branch)
				else:
					branch_descendants[parent] = {branch}
				if parent in branch_elements:
					branch_el = Element(
						"branch",
						{
							"name": branch,
							"parent": parent,
							"start-turn": str(parent_turn),
							"start-tick": str(parent_tick),
							"end-turn": str(end_turn),
							"end-tick": str(end_tick),
						},
					)
					if branch in turns_completed_d:
						branch_el.set(
							"last-turn-completed",
							str(turns_completed_d[branch]),
						)
					branch_elements[parent].append(branch_el)
				else:
					branch2do.append(
						(
							branch,
							parent,
							parent_turn,
							parent_tick,
							end_turn,
							end_tick,
						)
					)

		def recurse_branch(b: Branch):
			parent, turn_from, tick_from, turn_to, tick_to = branches_d[b]
			if b in turn_end_plan_d:
				turn_to, tick_to = max(
					[
						(turn_to, tick_to),
						*(
							(r, t)
							for r, (_, t) in turn_end_plan_d[branch].items()
						),
					]
				)
			data = self.load_windows(
				[(b, turn_from, tick_from, turn_to, tick_to)]
			)
			self._fill_branch_element(
				branch_elements[b],
				turn_end_plan_d[b],
				keyframe_times,
				data,
			)
			if b in branch_descendants:
				for desc in sorted(branch_descendants[b], key=branches_d.get):
					recurse_branch(desc)

		for trunk in trunks:
			recurse_branch(trunk)
		return self.tree

	def to_xml(
		self,
		xml_file_path: str | os.PathLike | IOBase,
		indent: bool = True,
		name: str | None = None,
	) -> None:
		if not isinstance(xml_file_path, (os.PathLike, IOBase)):
			xml_file_path = Path(xml_file_path)

		tree = self.to_etree(name)
		if indent:
			indent_tree(tree)
		tree.write(xml_file_path, encoding="utf-8")

	@classmethod
	def _value_to_xml_el(cls, value: Value | dict[Key, Value]) -> Element:
		if value is ...:
			return Element("Ellipsis")
		elif value is None:
			return Element("None")
		elif isinstance(value, bool):
			return Element("bool", value="true" if value else "false")
		elif isinstance(value, int):
			return Element("int", value=str(value))
		elif isinstance(value, float):
			return Element("float", value=str(value))
		elif isinstance(value, str):
			return Element("str", value=value)
		elif isinstance(value, lisien.types.DiGraph):
			# Since entity names are restricted to what we can use for dict
			# keys and also serialize to msgpack, I don't think there's any name
			# an entity can have that can't be repr'd
			return Element("character", name=repr(value.name))
		elif isinstance(value, lisien.types.Node):
			return Element(
				"node", character=repr(value.graph.name), name=repr(value.name)
			)
		elif isinstance(value, lisien.types.Edge):
			return Element(
				"portal",
				character=repr(value.graph.name),
				origin=repr(value.orig),
				destination=repr(value.dest),
			)
		elif isinstance(value, nx.Graph):
			return nx.readwrite.GraphMLWriter(value).myElement
		elif isinstance(value, FunctionType) or isinstance(value, MethodType):
			if value.__module__ not in (
				"trigger",
				"prereq",
				"action",
				"function",
				"method",
			):
				raise ValueError(
					"Callable is not stored in the Lisien engine", value
				)
			return Element(value.__module__, name=value.__name__)
		elif isinstance(value, Exception):
			# weird but ok
			el = Element("exception", pyclass=value.__class__.__name__)
			if hasattr(value, "__traceback__"):
				el.set("traceback", str(Traceback(value.__traceback__)))
			for arg in value.args:
				el.append(cls._value_to_xml_el(arg))
			return el
		elif isinstance(value, list):
			el = Element("list")
			for v in value:
				el.append(cls._value_to_xml_el(v))
			return el
		elif isinstance(value, tuple):
			el = Element("tuple")
			for v in value:
				el.append(cls._value_to_xml_el(v))
			return el
		elif isinstance(value, Set):
			if isinstance(value, (set, MutableSet)):
				el = Element("set")
				for v in value:
					el.append(cls._value_to_xml_el(v))
				return el
			else:
				el = Element("frozenset")
				for v in value:
					el.append(cls._value_to_xml_el(v))
				return el
		elif isinstance(value, Mapping):
			el = Element("dict")
			for k, v in value.items():
				dict_item = Element("dict-item", key=repr(k))
				dict_item.append(cls._value_to_xml_el(v))
				el.append(dict_item)
			return el
		else:
			raise TypeError("Can't convert to XML", value)

	@classmethod
	def _add_keyframe_to_turn_el(
		cls,
		turn_el: Element,
		tick: Tick,
		keyframe: Keyframe,
	) -> None:
		kfel = Element("keyframe", tick=str(tick))
		turn_el.append(kfel)
		universal_d: dict[Key, Value] = keyframe.get("universal", {})
		univel = cls._value_to_xml_el(universal_d)
		univel.tag = "universal"
		kfel.append(univel)
		triggers_kf: dict[RuleName, list[TriggerFuncName]] = keyframe.get(
			"triggers", {}
		)
		prereqs_kf: dict[RuleName, list[PrereqFuncName]] = keyframe.get(
			"prereqs", {}
		)
		actions_kf: dict[RuleName, list[ActionFuncName]] = keyframe.get(
			"actions", {}
		)
		neighborhoods_kf: dict[RuleName, RuleNeighborhood] = keyframe.get(
			"neighborhood", {}
		)
		bigs_kf: dict[RuleName, RuleBig] = keyframe.get("big", {})
		for rule_name in sorted(
			triggers_kf.keys() | prereqs_kf.keys() | actions_kf.keys()
		):
			rule_name: RuleName
			rule_el = Element(
				"rule",
				name=rule_name,
			)
			kfel.append(rule_el)
			if rule_name in bigs_kf and bigs_kf[rule_name]:
				rule_el.set("big", "true")
			if (
				rule_name in neighborhoods_kf
				and neighborhoods_kf[rule_name] is not None
			):
				rule_el.set(
					"neighborhood",
					str(neighborhoods_kf[rule_name]),
				)
			if trigs := triggers_kf.get(rule_name):
				for trig in trigs:
					rule_el.append(Element("trigger", name=trig))
			if preqs := prereqs_kf.get(rule_name):
				for preq in preqs:
					rule_el.append(Element("prereq", name=preq))
			if acts := actions_kf.get(rule_name):
				for act in acts:
					rule_el.append(Element("action", name=act))
		rulebook_kf: dict[
			RulebookName, tuple[list[RuleName], RulebookPriority]
		] = keyframe.get("rulebook", {})
		for rulebook_name, (rule_list, priority) in rulebook_kf.items():
			rulebook_el = Element(
				"rulebook", name=repr(rulebook_name), priority=repr(priority)
			)
			kfel.append(rulebook_el)
			for rule_name in rule_list:
				rulebook_el.append(Element("rule", name=rule_name))
		char_els: dict[CharName, Element] = {}
		graph_val_kf: GraphValKeyframe = keyframe.get("graph_val", {})
		for char_name, vals in sorted(graph_val_kf.items()):
			graph_el = char_els[char_name] = Element(
				"character", name=repr(char_name)
			)
			kfel.append(graph_el)
			if units_kf := vals.pop("units", {}):
				units_el = Element("units")
				any_unit_graphs = False
				for graph, nodes in units_kf.items():
					unit_graphs_el = Element("graph", character=repr(graph))
					any_unit_nodes = False
					for node, is_unit in nodes.items():
						if is_unit:
							any_unit_nodes = True
							unit_graphs_el.append(
								Element("unit", node=repr(node))
							)
					if any_unit_nodes:
						units_el.append(unit_graphs_el)
						any_unit_graphs = True
				if any_unit_graphs:
					graph_el.append(units_el)
			if "character_rulebook" in vals:
				graph_el.set(
					"character-rulebook",
					repr(
						vals.pop(
							"character_rulebook",
							("character_rulebook", char_name),
						)
					),
				)
			if "unit_rulebook" in vals:
				graph_el.set(
					"unit-rulebook",
					repr(
						vals.pop("unit_rulebook", ("unit_rulebook", char_name))
					),
				)
			if "character_thing_rulebook" in vals:
				graph_el.set(
					"character-thing-rulebook",
					repr(
						vals.pop(
							"character_thing_rulebook",
							("character_thing_rulebook", char_name),
						)
					),
				)
			if "character_place_rulebook" in vals:
				graph_el.set(
					"character-place-rulebook",
					repr(
						vals.pop(
							"character_place_rulebook",
							("character_place_rulebook", char_name),
						)
					),
				)
			if "character_portal_rulebook" in vals:
				graph_el.set(
					"character-portal-rulebook",
					repr(
						vals.pop(
							"character_portal_rulebook",
							("character_portal_rulebook", char_name),
						)
					),
				)
			for k, v in vals.items():
				item_el = Element("dict-item", key=repr(k))
				graph_el.append(item_el)
				item_el.append(cls._value_to_xml_el(v))
		node_val_kf: GraphNodeValKeyframe = keyframe.get("node_val", {})
		for char_name, node_vals in node_val_kf.items():
			if char_name in char_els:
				char_el = char_els[char_name]
			else:
				char_el = char_els[char_name] = Element(
					"character", name=repr(char_name)
				)
				kfel.append(char_el)
			for node, val in node_vals.items():
				node_el = Element(
					"node",
					name=repr(node),
				)
				if "rulebook" in val:
					node_el.set("rulebook", repr(val.pop("rulebook")))
				char_el.append(node_el)
				for k, v in val.items():
					item_el = Element("dict-item", key=repr(k))
					node_el.append(item_el)
					item_el.append(cls._value_to_xml_el(v))
		edge_val_kf: GraphEdgeValKeyframe = keyframe.get("edge_val", {})
		for char_name, edge_vals in edge_val_kf.items():
			if char_name in char_els:
				char_el = char_els[char_name]
			else:
				char_el = char_els[char_name] = Element(
					"character", name=repr(char_name)
				)
				kfel.append(char_el)
			for orig, dests in edge_vals.items():
				for dest, val in dests.items():
					edge_el = Element(
						"edge",
						orig=repr(orig),
						dest=repr(dest),
					)
					if "rulebook" in val:
						edge_el.set("rulebook", repr(val.pop("rulebook")))
					char_el.append(edge_el)
					for k, v in val.items():
						item_el = Element("dict-item", key=repr(k))
						edge_el.append(item_el)
						item_el.append(cls._value_to_xml_el(v))

	def to_xml(self, name: str, indent: bool = True) -> str:
		"""Return a string XML representation of the whole database

		Use ``load_xml`` to load it later.

		:param name: What to call the game in the XML.
		:param indent: Whether to format the XML for human readers. Default
			``True``.

		"""
		file = StringIO()
		self.write_xml(file, name, indent)
		return file.getvalue()

	def write_xml(
		self,
		file: str | os.PathLike | IOBase,
		name: str | None = None,
		indent: bool = True,
	) -> None:
		"""Serialize the whole database to an XML file

		:param file: A file name, or a file object.
		:param name: Optional name to give to the game in the XML. Defaults
			to the name of the file, minus .xml extension.
		:param indent: Whether to format the XML for a human reader. Default
			``True``.

		"""
		if not isinstance(file, os.PathLike) and not isinstance(file, IOBase):
			if file is None:
				if name is None:
					raise ValueError("Need a name or a path")
				file = os.path.join(os.getcwd(), name + ".xml")
			file = Path(file)
		name = name.removesuffix(".xml")

		tree = self.to_etree(name)

		if indent:
			indent_tree(tree)
		tree.write(file, encoding="utf-8")

	def _set_plans(
		self, el: Element, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		plans = []
		if branch in self._plan_ticks and turn in self._plan_ticks[branch]:
			for plan, ticks in self._plan_ticks[branch][turn].items():
				if tick in ticks:
					plans.append(plan)
		if plans:
			el.set("plans", ",".join(map(str, plans)))

	def _univ_el(self, universal_rec: UniversalRowType) -> Element:
		key, b, r, t, val = universal_rec
		univ_el = Element(
			"universal",
			key=repr(key),
			tick=str(t),
		)
		univ_el.append(self._value_to_xml_el(val))
		self._set_plans(univ_el, b, r, t)
		return univ_el

	def _rulebook_el(self, rulebook_rec: RulebookRowType) -> Element:
		rb, b, r, t, (rules, prio) = rulebook_rec
		rb_el = Element(
			"rulebook",
			name=repr(rb),
			priority=repr(prio),
			tick=str(t),
		)
		for i, rule in enumerate(rules):
			rb_el.append(Element("rule", name=rule))
		self._set_plans(rb_el, b, r, t)
		return rb_el

	def _rule_flist_el(
		self,
		typ: str,
		rec: TriggerRowType | PrereqRowType | ActionRowType,
	) -> Element:
		rule, branch, turn, tick, funcs = rec
		func_el = Element(f"{typ}s", rule=rule, tick=str(tick))
		for func in funcs:
			func_el.append(Element(typ[5:], name=func))
		self._set_plans(func_el, branch, turn, tick)
		return func_el

	_rule_triggers_el = partialmethod(_rule_flist_el, "rule-trigger")
	_rule_prereqs_el = partialmethod(_rule_flist_el, "rule-prereq")
	_rule_actions_el = partialmethod(_rule_flist_el, "rule-action")

	def _rule_neighborhood_el(
		self, nbr_rec: RuleNeighborhoodRowType
	) -> Element:
		rule, branch, turn, tick, nbr = nbr_rec
		if nbr is not None:
			nbr_el = Element(
				"rule-neighborhood",
				rule=rule,
				tick=str(tick),
				neighbors=str(nbr),
			)
		else:
			nbr_el = Element("rule-neighborhood", rule=rule, tick=str(tick))
		self._set_plans(nbr_el, branch, turn, tick)
		return nbr_el

	def _rule_big_el(self, big_rec: RuleBigRowType) -> Element:
		rule, branch, turn, tick, big = big_rec
		el = Element(
			"rule-big",
			rule=rule,
			tick=str(tick),
			big="true" if big else "false",
		)
		self._set_plans(el, branch, turn, tick)
		return el

	def _graph_el(self, graph: GraphRowType) -> Element:
		char, b, r, t, typ_str = graph
		graph_el = Element(
			"graph",
			character=repr(char),
			tick=str(t),
			type=typ_str,
		)
		self._set_plans(graph_el, b, r, t)
		return graph_el

	def _graph_val_el(self, graph_val: GraphValRowType) -> Element:
		char, stat, b, r, t, val = graph_val
		graph_val_el = Element(
			"graph-val",
			character=repr(char),
			key=repr(stat),
			tick=str(t),
		)
		graph_val_el.append(self._value_to_xml_el(val))
		self._set_plans(graph_val_el, b, r, t)
		return graph_val_el

	def _nodes_el(self, nodes: NodeRowType) -> Element:
		char, node, b, r, t, ex = nodes
		node_el = Element(
			"node",
			character=repr(char),
			name=repr(node),
			tick=str(t),
			exists="true" if ex else "false",
		)
		self._set_plans(node_el, b, r, t)
		return node_el

	def _node_val_el(self, node_val: NodeValRowType) -> Element:
		char, node, stat, b, r, t, val = node_val
		node_val_el = Element(
			"node-val",
			character=repr(char),
			node=repr(node),
			key=repr(stat),
			tick=str(t),
		)
		node_val_el.append(self._value_to_xml_el(val))
		self._set_plans(node_val_el, b, r, t)
		return node_val_el

	def _edge_el(self, edges: EdgeRowType) -> Element:
		char, orig, dest, b, r, t, ex = edges
		edge_el = Element(
			"edge",
			character=repr(char),
			orig=repr(orig),
			dest=repr(dest),
			tick=str(t),
			exists="true" if ex else "false",
		)
		self._set_plans(edge_el, b, r, t)
		return edge_el

	def _edge_val_el(self, edge_val: EdgeValRowType) -> Element:
		char, orig, dest, stat, b, r, t, val = edge_val
		edge_val_el = Element(
			"edge-val",
			character=repr(char),
			orig=repr(orig),
			dest=repr(dest),
			key=repr(stat),
			tick=str(t),
		)
		edge_val_el.append(self._value_to_xml_el(val))
		self._set_plans(edge_val_el, b, r, t)
		return edge_val_el

	def _thing_el(self, thing: ThingRowType) -> Element:
		char, thing, b, r, t, loc = thing
		loc_el = Element(
			"location",
			character=repr(char),
			thing=repr(thing),
			tick=str(t),
			location=repr(loc),
		)
		self._set_plans(loc_el, b, r, t)
		return loc_el

	def _unit_el(self, unit: UnitRowType) -> Element:
		char, graph, node, b, r, t, is_unit = unit
		unit_el = Element(
			"unit",
			{
				"character-graph": repr(char),
				"unit-graph": repr(graph),
				"unit-node": repr(node),
				"tick": str(t),
			},
		)
		unit_el.set("is-unit", "true" if is_unit else "false")
		self._set_plans(unit_el, b, r, t)
		return unit_el

	def _char_rb_el(self, rbtyp: str, rbrow: CharRulebookRowType) -> Element:
		char, b, r, t, rb = rbrow
		chrbel = Element(
			rbtyp,
			character=repr(char),
			tick=str(t),
			rulebook=repr(rb),
		)
		self._set_plans(chrbel, b, r, t)
		return chrbel

	def _node_rb_el(self, nrb_row: NodeRulebookRowType) -> Element:
		char, node, b, r, t, rb = nrb_row
		nrb_el = Element(
			"node-rulebook",
			character=repr(char),
			node=repr(node),
			tick=str(t),
			rulebook=repr(rb),
		)
		self._set_plans(nrb_el, b, r, t)
		return nrb_el

	def _portal_rb_el(self, port_rb_row: PortalRulebookRowType) -> Element:
		char, orig, dest, b, r, t, rb = port_rb_row
		porb_el = Element(
			"portal-rulebook",
			character=repr(char),
			orig=repr(orig),
			dest=repr(dest),
			tick=str(t),
			rulebook=repr(rb),
		)
		self._set_plans(porb_el, b, r, t)
		return porb_el

	@staticmethod
	def _get_current_rule_el(data: dict) -> Element | None:
		# Rules record the time they finished running, not the time they
		# started. So we need to peek ahead to find out what the current
		# rule is.
		earliest = (float("inf"), float("inf"))
		earliest_key = None
		if data["character_rules_handled"]:
			handled_at = data["character_rules_handled"][0][-2:]
			if handled_at < earliest:
				earliest = handled_at
				earliest_key = "character_rules_handled"
		if data["unit_rules_handled"]:
			handled_at = data["unit_rules_handled"][0][-2:]
			if handled_at < earliest:
				earliest = handled_at
				earliest_key = "unit_rules_handled"
		if data["character_thing_rules_handled"]:
			handled_at = data["character_thing_rules_handled"][0][-2:]
			if handled_at < earliest:
				earliest = handled_at
				earliest_key = "character_thing_rules_handled"
		if data["character_place_rules_handled"]:
			handled_at = data["character_place_rules_handled"][0][-2:]
			if handled_at < earliest:
				earliest = handled_at
				earliest_key = "character_place_rules_handled"
		if data["character_portal_rules_handled"]:
			handled_at = data["character_portal_rules_handled"][0][-2:]
			if handled_at < earliest:
				earliest = handled_at
				earliest_key = "character_portal_rules_handled"
		if data["node_rules_handled"]:
			handled_at = data["node_rules_handled"][0][-2:]
			if handled_at < earliest:
				earliest = handled_at
				earliest_key = "node_rules_handled"
		if data["portal_rules_handled"]:
			handled_at = data["portal_rules_handled"][0][-2:]
			if handled_at < earliest:
				earliest = handled_at
				earliest_key = "portal_rules_handled"
		if earliest_key is None:
			return None
		ret = Element(
			"rule",
			{"end-tick": str(earliest[1])},
		)
		match earliest_key:
			case "character_rules_handled":
				char, rb, rule, b, r, t = data["character_rules_handled"].pop(
					0
				)
				ret.set("type", "character")
				ret.set("character", repr(char))
				ret.set("rulebook", repr(rb))
				ret.set("name", rule)
			case "unit_rules_handled":
				char, graph, unit, rb, rule, b, r, t = data[
					"unit_rules_handled"
				].pop(0)
				ret.set("type", "unit")
				ret.set("character", repr(char))
				ret.set("graph", repr(graph))
				ret.set("unit", repr(unit))
				ret.set("rulebook", repr(rb))
				ret.set("rule", rule)
			case "character_thing_rules_handled":
				char, thing, rb, rule, b, r, t = data[
					"character_thing_rules_handled"
				].pop(0)
				ret.set("type", "character-thing")
				ret.set("character", repr(char))
				ret.set("thing", repr(thing))
				ret.set("rulebook", repr(rb))
				ret.set("rule", rule)
			case "character_place_rules_handled":
				char, place, rb, rule, b, r, t = data[
					"character_place_rules_handled"
				].pop(0)
				ret.set("type", "character-place")
				ret.set("character", repr(char))
				ret.set("place", repr(place))
				ret.set("rulebook", repr(rb))
				ret.set("rule", rule)
			case "character_portal_rules_handled":
				char, orig, dest, rb, rule, b, r, t = data[
					"character_portal_rules_handled"
				].pop(0)
				ret.set("type", "character-portal")
				ret.set("character", repr(char))
				ret.set("origin", repr(orig))
				ret.set("destination", repr(dest))
				ret.set("rulebook", repr(rb))
				ret.set("rule", rule)
			case "node_rules_handled":
				char, node, rb, rule, b, r, t = data["node_rules_handled"].pop(
					0
				)
				ret.set("type", "node")
				ret.set("character", repr(char))
				ret.set("node", repr(node))
				ret.set("rulebook", repr(rb))
				ret.set("rule", rule)
		return ret

	def _fill_branch_element(
		self,
		branch_el: Element,
		turn_ends: dict[Turn, tuple[Tick, Tick]],
		keyframe_times: set[Time],
		data: LoadedDict,
	):
		branch_ = branch_el.get("name")
		if branch_ is None:
			raise TypeError("branch missing")
		branch = Branch(branch_)

		uncharacterized = {
			"graphs",
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
		}
		turn: Turn
		for turn, (ending_tick, plan_ending_tick) in sorted(turn_ends.items()):
			turn_el = Element(
				"turn",
				{
					"number": str(turn),
					"end-tick": str(ending_tick),
					"plan-end-tick": str(plan_ending_tick),
				},
			)
			branch_el.append(turn_el)
			tick: Tick
			current_rule_el: Element | None = self._get_current_rule_el(data)
			if current_rule_el is not None:
				turn_el.append(current_rule_el)

			def get_current_el() -> Element:
				if current_rule_el is None:
					return turn_el
				return current_rule_el

			for tick in range(plan_ending_tick + 1):
				if current_rule_el is None:
					current_rule_el = self._get_current_rule_el(data)
					if current_rule_el is not None:
						turn_el.append(current_rule_el)
				# Loop over rules that didn't result in any changes, if needed
				while current_rule_el is not None and tick >= (
					int(current_rule_el.get("end-tick"))
				):
					current_rule_el = self._get_current_rule_el(data)
					if current_rule_el is not None:
						turn_el.append(current_rule_el)
				if (branch, turn, tick) in keyframe_times:
					kf = self.get_keyframe(branch, turn, tick)
					self._add_keyframe_to_turn_el(turn_el, tick, kf)
					keyframe_times.remove((branch, turn, tick))
				if data["universals"]:
					universal_rec: UniversalRowType = data["universals"][0]
					key, branch_now, turn_now, tick_now, _ = universal_rec
					if (branch_now, turn_now, tick_now) == (
						branch,
						turn,
						tick,
					):
						univ_el = self._univ_el(universal_rec)
						get_current_el().append(univ_el)
						del data["universals"][0]
				if data["rulebooks"]:
					rulebook_rec: RulebookRowType = data["rulebooks"][0]
					rulebook, branch_now, turn_now, tick_now, _ = rulebook_rec
					if (branch_now, turn_now, tick_now) == (
						branch,
						turn,
						tick,
					):
						rulebook_el = self._rulebook_el(rulebook_rec)
						get_current_el().append(rulebook_el)
						del data["rulebooks"][0]
				if data["graphs"]:
					graph_rec: GraphRowType = data["graphs"][0]
					_, branch_now, turn_now, tick_now, _ = graph_rec
					if (branch_now, turn_now, tick_now) == (
						branch,
						turn,
						tick,
					):
						graph_el = self._graph_el(graph_rec)
						get_current_el().append(graph_el)
						del data["graphs"][0]
				if (
					"rule_triggers" in data
					and data["rule_triggers"]
					and data["rule_triggers"][0][1:4]
					== (
						branch,
						turn,
						tick,
					)
				):
					trel = self._rule_triggers_el(data["rule_triggers"].pop(0))
					get_current_el().append(trel)
				if (
					"rule_prereqs" in data
					and data["rule_prereqs"]
					and data["rule_prereqs"][0][1:4]
					== (
						branch,
						turn,
						tick,
					)
				):
					prel = self._rule_prereqs_el(data["rule_prereqs"].pop(0))
					get_current_el().append(prel)
				if (
					"rule_actions" in data
					and data["rule_actions"]
					and data["rule_actions"][0][1:4]
					== (
						branch,
						turn,
						tick,
					)
				):
					actel = self._rule_actions_el(data["rule_actions"].pop(0))
					get_current_el().append(actel)
				if (
					"rule_neighborhood" in data
					and data["rule_neighborhood"]
					and data["rule_neighborhood"][0][1:4]
					== (branch, turn, tick)
				):
					nbrel = self._rule_neighborhood_el(
						data["rule_neighborhood"].pop(0)
					)
					get_current_el().append(nbrel)
				if (
					"rule_big" in data
					and data["rule_big"]
					and data["rule_big"][0][1:4]
					== (
						branch,
						turn,
						tick,
					)
				):
					big_el = self._rule_big_el(data["rule_big"].pop(0))
					get_current_el().append(big_el)
				for char_name in data.keys() - uncharacterized:
					char_data: LoadedCharWindow = data[char_name]
					if char_data["graph_val"]:
						graph_val_row: GraphValRowType = char_data[
							"graph_val"
						][0]
						_, __, b, r, t, ___ = graph_val_row
						if (b, r, t) == (branch, turn, tick):
							gvel = self._graph_val_el(graph_val_row)
							get_current_el().append(gvel)
							del char_data["graph_val"][0]
					if char_data["nodes"]:
						nodes_row: NodeRowType = char_data["nodes"][0]
						char, node, branch_now, turn_now, tick_now, ex = (
							nodes_row
						)
						if (branch_now, turn_now, tick_now) == (
							branch,
							turn,
							tick,
						):
							nel = self._nodes_el(nodes_row)
							get_current_el().append(nel)
							del char_data["nodes"][0]
					if char_data["node_val"]:
						node_val_row: NodeValRowType = char_data["node_val"][0]
						(
							char,
							node,
							stat,
							branch_now,
							turn_now,
							tick_now,
							val,
						) = node_val_row
						if (branch_now, turn_now, tick_now) == (
							branch,
							turn,
							tick,
						):
							nvel = self._node_val_el(node_val_row)
							get_current_el().append(nvel)
							del char_data["node_val"][0]
					if char_data["edges"]:
						edges_row: EdgeRowType = char_data["edges"][0]
						_, __, ___, b, r, t, ____ = edges_row
						if (b, r, t) == (branch, turn, tick):
							edgel = self._edge_el(edges_row)
							get_current_el().append(edgel)
							del char_data["edges"][0]
					if char_data["edge_val"]:
						edge_val_row: EdgeValRowType = char_data["edge_val"][0]
						_, __, ___, ____, b, r, t, _____ = edge_val_row
						if (b, r, t) == (branch, turn, tick):
							evel = self._edge_val_el(edge_val_row)
							get_current_el().append(evel)
							del char_data["edge_val"][0]
					if char_data["things"]:
						thing_row: ThingRowType = char_data["things"][0]
						_, __, b, r, t, ___ = thing_row
						if (b, r, t) == (branch, turn, tick):
							thel = self._thing_el(thing_row)
							get_current_el().append(thel)
							del char_data["things"][0]
					if char_data["units"]:
						units_row: UnitRowType = char_data["units"][0]
						_, __, ___, b, r, t, ____ = units_row
						if (b, r, t) == (branch, turn, tick):
							unit_el = self._unit_el(units_row)
							get_current_el().append(unit_el)
							del char_data["units"][0]
					for char_rb_typ in (
						"character_rulebook",
						"unit_rulebook",
						"character_thing_rulebook",
						"character_place_rulebook",
						"character_portal_rulebook",
					):
						if char_data[char_rb_typ]:
							char_rb_row: CharRulebookRowType = char_data[
								char_rb_typ
							][0]
							_, b, r, t, __ = char_rb_row
							if (b, r, t) == (branch, turn, tick):
								crbel = self._char_rb_el(
									char_rb_typ.replace("_", "-"),
									char_rb_row,
								)
								get_current_el().append(crbel)
								del char_data[char_rb_typ][0]
					if char_data["node_rulebook"]:
						node_rb_row: NodeRulebookRowType = char_data[
							"node_rulebook"
						][0]
						_, __, b, r, t, ___ = node_rb_row
						if (b, r, t) == (branch, turn, tick):
							nrbel = self._node_rb_el(node_rb_row)
							get_current_el().append(nrbel)
							del char_data["node_rulebook"][0]
					if char_data["portal_rulebook"]:
						port_rb_row: PortalRulebookRowType = char_data[
							"portal_rulebook"
						][0]
						_, __, ___, b, r, t, ____ = port_rb_row
						if (b, r, t) == (branch, turn, tick):
							porbel = self._portal_rb_el(port_rb_row)
							get_current_el().append(porbel)
							del char_data["portal_rulebook"][0]
		for k in uncharacterized:
			if k in data:
				assert not data[k]
		for char_name in data.keys() - uncharacterized:
			for k, v in data[char_name].items():
				assert not v, f"Leftover data in {char_name}'s {k}: {v}"
		assert not keyframe_times, keyframe_times

	@cached_property
	def _known_triggers(
		self,
	) -> dict[
		RuleName,
		dict[
			Branch,
			SettingsTurnDict[Turn, WindowDict[Tick, list[TriggerFuncName]]],
		],
	]:
		return {}

	@cached_property
	def _known_prereqs(
		self,
	) -> dict[
		RuleName,
		dict[
			Branch,
			SettingsTurnDict[Turn, WindowDict[Tick, list[PrereqFuncName]]],
		],
	]:
		return {}

	@cached_property
	def _known_actions(
		self,
	) -> dict[
		RuleName,
		dict[
			Branch,
			SettingsTurnDict[Turn, WindowDict[Tick, list[ActionFuncName]]],
		],
	]:
		return {}

	@cached_property
	def _known_neighborhoods(
		self,
	) -> dict[
		RuleName,
		dict[
			Branch, SettingsTurnDict[Turn, WindowDict[Tick, RuleNeighborhood]]
		],
	]:
		return {}

	@cached_property
	def _known_big(
		self,
	) -> dict[
		RuleName,
		dict[Branch, SettingsTurnDict[Turn, WindowDict[Tick, RuleBig]]],
	]:
		return {}

	@cached_property
	def _plan_times(self) -> dict[Plan, set[Time]]:
		return {}

	def _element_to_value(
		self, el: Element
	) -> (
		Value
		| list[Value]
		| tuple[Value, ...]
		| set[Key | type(...)]
		| frozenset[Key | type(...)]
		| dict[Key, Value | type(...)]
		| type(...)
	):
		eng = self.engine
		match el.tag:
			case "Ellipsis":
				return ...
			case "None":
				return Value(None)
			case "int":
				return Value(int(el.get("value")))
			case "float":
				return Value(float(el.get("value")))
			case "str":
				return Value(el.get("value"))
			case "bool":
				return Value(el.get("value") in {"T", "true"})
			case "character":
				name = CharName(literal_eval(el.get("name")))
				return eng.character[name]
			case "node":
				char_name = CharName(literal_eval(el.get("character")))
				place_name = NodeName(literal_eval(el.get("name")))
				return eng.character[char_name].node[place_name]
			case "portal":
				char_name = CharName(literal_eval(el.get("character")))
				orig = NodeName(literal_eval(el.get("origin")))
				dest = NodeName(literal_eval(el.get("destination")))
				return eng.character[char_name].portal[orig][dest]
			case "list":
				return [self._element_to_value(listel) for listel in el]
			case "tuple":
				return tuple(self._element_to_value(tupel) for tupel in el)
			case "set":
				return {self._element_to_value(setel) for setel in el}
			case "frozenset":
				return frozenset(self._element_to_value(setel) for setel in el)
			case "dict":
				ret = {}
				for dict_item_el in el:
					ret[literal_eval(dict_item_el.get("key"))] = (
						self._element_to_value(dict_item_el[0])
					)
				return ret
			case "exception":
				raise NotImplementedError(
					"Deserializing exceptions from XML not implemented"
				)
			case s if s in {
				"trigger",
				"prereq",
				"action",
				"function",
				"method",
			}:
				return getattr(getattr(eng, s), el.get("name"))
			case default:
				raise ValueError("Can't deserialize the element", default)

	@staticmethod
	def _get_time(branch_el: Element, turn_el: Element, el: Element) -> Time:
		ret = (
			Branch(branch_el.get("name")),
			Turn(int(turn_el.get("number"))),
			Tick(int(el.get("tick"))),
		)
		if not isinstance(ret[0], str):
			raise TypeError("nonstring branch", ret[0])
		return ret

	def _keyframe(self, branch_el: Element, turn_el: Element, kf_el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, kf_el)
		self.keyframe_insert(branch, turn, tick)
		universal_kf: UniversalKeyframe = {}
		triggers_kf: dict[RuleName, list[TriggerFuncName]] = {}
		prereqs_kf: dict[RuleName, list[PrereqFuncName]] = {}
		actions_kf: dict[RuleName, list[ActionFuncName]] = {}
		neighborhoods_kf: dict[RuleName, RuleNeighborhood] = {}
		bigs_kf: dict[RuleName, RuleBig] = {}
		rule_kf: RuleKeyframe = {
			"triggers": triggers_kf,
			"prereqs": prereqs_kf,
			"actions": actions_kf,
			"neighborhood": neighborhoods_kf,
			"big": bigs_kf,
		}
		rulebook_kf: dict[
			RulebookName, tuple[list[RuleName], RulebookPriority]
		] = {}
		graph_val_kf: GraphValKeyframe = {}
		node_val_kf: GraphNodeValKeyframe = {}
		edge_val_kf: GraphEdgeValKeyframe = {}
		for subel in kf_el:
			if subel.tag == "universal":
				for univel in subel:
					k = literal_eval(univel.get("key"))
					v = self._element_to_value(univel[0])
					universal_kf[k] = v
			elif subel.tag == "rule":
				rule = RuleName(subel.get("name"))
				if rule is None:
					raise TypeError("Rules need names")
				if "big" in subel.keys():
					bigs_kf[rule] = RuleBig(subel.get("big") in {"T", "true"})
				if "neighborhood" in subel.keys():
					neighborhoods_kf[rule] = int(subel.get("neighborhood"))
				else:
					neighborhoods_kf[rule] = None
				for funcl_el in subel:
					name = FuncName(funcl_el.get("name"))
					if not isinstance(name, str):
						raise TypeError("Function name must be str", name)
					if funcl_el.tag == "trigger":
						if rule in triggers_kf:
							triggers_kf[rule].append(TriggerFuncName(name))
						else:
							triggers_kf[rule] = [TriggerFuncName(name)]
					elif funcl_el.tag == "prereq":
						if rule in prereqs_kf:
							prereqs_kf[rule].append(PrereqFuncName(name))
						else:
							prereqs_kf[rule] = [PrereqFuncName(name)]
					elif funcl_el.tag == "action":
						if rule in actions_kf:
							actions_kf[rule].append(ActionFuncName(name))
						else:
							actions_kf[rule] = [ActionFuncName(name)]
					else:
						raise ValueError("Unknown rule tag", funcl_el.tag)
			elif subel.tag == "rulebook":
				name = subel.get("name")
				if name is None:
					raise TypeError("rulebook tag missing name")
				name = literal_eval(name)
				if not isinstance(name, Key):
					raise TypeError("Rulebook name must be Key", name)
				name = RulebookName(name)
				prio = subel.get("priority")
				if prio is None:
					raise TypeError("rulebook tag missing priority")
				prio = RulebookPriority(float(prio))
				rules: list[RuleName] = []
				for rule_el in subel:
					if rule_el.tag != "rule":
						raise ValueError("Expected a rule tag", rule_el.tag)
					rules.append(RuleName(rule_el.get("name")))
				rulebook_kf[name] = (rules, prio)
			elif subel.tag == "character":
				name = subel.get("name")
				if name is None:
					raise TypeError("character tag missing name")
				name = literal_eval(name)
				if not isinstance(name, Key):
					raise TypeError("character names must be Key", name)
				char_name = CharName(name)
				if isinstance(self.engine, EngineFacade):
					# Only needed for deserializing later entities.
					# Real Engines don't require this, because they have the
					# keyframe to work with.
					self.engine.add_character(char_name)
				graph_vals = graph_val_kf[char_name] = {}
				for k in (
					"character-rulebook",
					"unit-rulebook",
					"character-thing-rulebook",
					"character-place-rulebook",
					"character-portal-rulebook",
				):
					if k in subel.keys():
						graph_vals[k.replace("-", "_")] = literal_eval(
							subel.get(k)
						)
				node_vals = node_val_kf[char_name] = {}
				edge_vals = edge_val_kf[char_name] = {}
				for key_el in subel:
					if key_el.tag == "dict-item":
						key = literal_eval(key_el.get("key"))
						graph_vals[key] = self._element_to_value(key_el[0])
					elif key_el.tag == "node":
						name = literal_eval(key_el.get("name"))
						if isinstance(self.engine, EngineFacade):
							self.engine.character[char_name].add_node(name)
						if name in node_vals:
							val = node_vals[name]
						else:
							val = node_vals[name] = {}
						if "rulebook" in key_el.keys():
							val["rulebook"] = literal_eval(
								key_el.get("rulebook")
							)
						for item_el in key_el:
							val[literal_eval(item_el.get("key"))] = (
								self._element_to_value(item_el[0])
							)
					elif key_el.tag == "edge":
						orig = literal_eval(key_el.get("orig"))
						dest = literal_eval(key_el.get("dest"))
						if isinstance(self.engine, EngineFacade):
							self.engine.character[char_name].add_edge(
								orig, dest
							)
						if orig not in edge_vals:
							edge_vals[orig] = {dest: {}}
						if dest not in edge_vals[orig]:
							edge_vals[orig][dest] = {}
						val = edge_vals[orig][dest]
						if "rulebook" in key_el.keys():
							val["rulebook"] = literal_eval(
								key_el.get("rulebook")
							)
						for item_el in key_el:
							val[literal_eval(item_el.get("key"))] = (
								self._element_to_value(item_el[0])
							)
					elif key_el.tag == "units":
						graph_vals["units"] = {}
						for unit_graph_el in key_el:
							unit_graph_name = literal_eval(
								unit_graph_el.get("character")
							)
							unit_graph_nodes_d = graph_vals["units"][
								unit_graph_name
							] = {}
							for unit_node_el in unit_graph_el:
								unit_graph_nodes_d[
									literal_eval(unit_node_el.get("node"))
								] = True
					else:
						raise ValueError(
							"Don't know how to deal with tag", key_el.tag
						)
			else:
				raise ValueError("Don't know how to deal with tag", subel.tag)
		self.keyframe_insert(branch, turn, tick)
		self.keyframe_extension_insert(
			branch, turn, tick, universal_kf, rule_kf, rulebook_kf
		)
		for graph in (
			graph_val_kf.keys() | node_val_kf.keys() | edge_val_kf.keys()
		):
			self.keyframe_graph_insert(
				graph,
				branch,
				turn,
				tick,
				node_val_kf.get(graph, {}),
				edge_val_kf.get(graph, {}),
				graph_val_kf.get(graph, {}),
			)

	def _universal(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		self._get_plans(el, branch, turn, tick)
		key = UniversalKey(literal_eval(el.get("key")))
		value = self._element_to_value(el[0])
		self.universal_set(key, branch, turn, tick, value)

	def _get_plans(
		self, el: Element, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		if "plans" in el.keys():
			plan_id: Plan
			for plan_id in map(int, el.get("plans").split(",")):
				if plan_id in self._plan_times:
					self._plan_times[plan_id].add((branch, turn, tick))
				else:
					self._plan_times[plan_id] = {(branch, turn, tick)}

	def _rulebook(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		self._get_plans(el, branch, turn, tick)
		rulebook = RulebookName(literal_eval(el.get("name")))
		priority = RulebookPriority(float(el.get("priority")))
		rules: list[RuleName] = []
		for subel in el:
			if subel.tag != "rule":
				raise ValueError("Don't know what to do with tag", subel.tag)
			rules.append(RuleName(subel.get("name")))
		self.set_rulebook(rulebook, branch, turn, tick, rules, priority)

	def _rule_func_list(
		self,
		what: Literal["triggers", "prereqs", "actions"],
		branch_el: Element,
		turn_el: Element,
		el: Element,
	):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		self._get_plans(el, branch, turn, tick)
		rule = RuleName(el.get("rule"))
		funcs: list[RuleFuncName] = [func_el.get("name") for func_el in el]
		self._memorize_rule(what, rule, branch, turn, tick, funcs)

	def _memorize_rule(
		self,
		what: Literal["triggers", "prereqs", "actions", "neighborhood", "big"],
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		datum: list[TriggerFuncName]
		| list[PrereqFuncName]
		| list[ActionFuncName]
		| RuleNeighborhood
		| RuleBig,
	):
		if what == "triggers":
			d = self._known_triggers
		elif what == "prereqs":
			d = self._known_prereqs
		elif what == "actions":
			d = self._known_actions
		elif what == "neighborhood":
			d = self._known_neighborhoods
		elif what == "big":
			d = self._known_big
		else:
			raise ValueError(what)
		if rule in d:
			if branch in d[rule]:
				if turn in d[rule][branch]:
					d[rule][branch][turn][tick] = datum
				else:
					d[rule][branch][turn] = {tick: datum}
			else:
				d[rule][branch] = SettingsTurnDict({turn: {tick: datum}})
		else:
			d[rule] = {branch: SettingsTurnDict({turn: {tick: datum}})}

	_rule_triggers = partialmethod(_rule_func_list, "triggers")
	_rule_prereqs = partialmethod(_rule_func_list, "prereqs")
	_rule_actions = partialmethod(_rule_func_list, "actions")

	def _rule_neighborhood(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		self._get_plans(el, branch, turn, tick)
		rule = RuleName(el.get("rule"))
		neighborhood = el.get("neighbors")
		if neighborhood is not None:
			neighborhood = int(neighborhood)
		self._memorize_rule(
			"neighborhood",
			rule,
			branch,
			turn,
			tick,
			neighborhood,
		)

	def _rule_big(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		self._get_plans(el, branch, turn, tick)
		big = RuleBig(el.get("big") in {"T", "true"})
		rule = RuleName(el.get("rule"))
		self._memorize_rule("big", rule, branch, turn, tick, big)

	def _graph(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		self._get_plans(el, branch, turn, tick)
		graph = CharName(literal_eval(el.get("character")))
		typ_str = el.get("type")
		if typ_str is None:
			raise TypeError("Missing graph type", el)
		if typ_str not in get_args(GraphTypeStr):
			raise TypeError("Unknown graph type", typ_str)
		typ_str: GraphTypeStr
		self.graphs_insert(graph, branch, turn, tick, typ_str)

	def _graph_val(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		self._get_plans(el, branch, turn, tick)
		graph = CharName(literal_eval(el.get("character")))
		key = Stat(literal_eval(el.get("key")))
		value = self._element_to_value(el[0])
		self.graph_val_set(graph, key, branch, turn, tick, value)

	def _node(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		self._get_plans(el, branch, turn, tick)
		char = CharName(literal_eval(el.get("character")))
		node = NodeName(literal_eval(el.get("name")))
		ex = el.get("exists") in {"T", "true"}
		self.exist_node(char, node, branch, turn, tick, ex)

	def _node_val(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		self._get_plans(el, branch, turn, tick)
		char = CharName(literal_eval(el.get("character")))
		node = NodeName(literal_eval(el.get("node")))
		key = Stat(literal_eval(el.get("key")))
		val = self._element_to_value(el[0])
		self.node_val_set(char, node, key, branch, turn, tick, val)

	def _edge(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		self._get_plans(el, branch, turn, tick)
		char = CharName(literal_eval(el.get("character")))
		orig = NodeName(literal_eval(el.get("orig")))
		dest = NodeName(literal_eval(el.get("dest")))
		ex = el.get("exists") in {"T", "true"}
		self.exist_edge(char, orig, dest, branch, turn, tick, ex)

	def _edge_val(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		self._get_plans(el, branch, turn, tick)
		char = CharName(literal_eval(el.get("character")))
		orig = NodeName(literal_eval(el.get("orig")))
		dest = NodeName(literal_eval(el.get("dest")))
		key = Stat(literal_eval(el.get("key")))
		val = self._element_to_value(el[0])
		self.edge_val_set(char, orig, dest, key, branch, turn, tick, val)

	def _location(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		self._get_plans(el, branch, turn, tick)
		char = CharName(literal_eval(el.get("character")))
		thing = NodeName(literal_eval(el.get("thing")))
		location = NodeName(literal_eval(el.get("location")))
		self.set_thing_loc(char, thing, branch, turn, tick, location)

	def _unit(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		self._get_plans(el, branch, turn, tick)
		char = CharName(literal_eval(el.get("character-graph")))
		graph = CharName(literal_eval(el.get("unit-graph")))
		node = NodeName(literal_eval(el.get("unit-node")))
		self.unit_set(
			char,
			graph,
			node,
			branch,
			turn,
			tick,
			el.get("is-unit", "false") in {"T", "true"},
		)

	def _some_character_rulebook(
		self, branch_el: Element, turn_el: Element, rbtyp: str, el: Element
	):
		meth = getattr(self, f"set_{rbtyp}")
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		self._get_plans(el, branch, turn, tick)
		char = CharName(literal_eval(el.get("character")))
		rb = RulebookName(literal_eval(el.get("rulebook")))
		meth(char, branch, turn, tick, rb)

	def _character_rulebook(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		self._some_character_rulebook(
			branch_el, turn_el, "character_rulebook", el
		)

	def _unit_rulebook(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		self._some_character_rulebook(branch_el, turn_el, "unit_rulebook", el)

	def _character_thing_rulebook(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		self._some_character_rulebook(
			branch_el, turn_el, "character_thing_rulebook", el
		)

	def _character_place_rulebook(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		self._some_character_rulebook(
			branch_el, turn_el, "character_place_rulebook", el
		)

	def _character_portal_rulebook(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		self._some_character_rulebook(
			branch_el, turn_el, "character_portal_rulebook", el
		)

	def _node_rulebook(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		self._get_plans(el, branch, turn, tick)
		char = CharName(literal_eval(el.get("character")))
		node = NodeName(literal_eval(el.get("node")))
		rb = RulebookName(literal_eval(el.get("rulebook")))
		self.set_node_rulebook(char, node, branch, turn, tick, rb)

	def _portal_rulebook(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		self._get_plans(el, branch, turn, tick)
		char = CharName(literal_eval(el.get("character")))
		orig = NodeName(literal_eval(el.get("orig")))
		dest = NodeName(literal_eval(el.get("dest")))
		rb = RulebookName(literal_eval(el.get("rulebook")))
		self.set_portal_rulebook(char, orig, dest, branch, turn, tick, rb)

	@classmethod
	def _iter_descendants(
		cls,
		branch_descendants: dict[Branch, list[Branch]],
		branch: Branch = "trunk",
		stop=lambda obj: False,
		key=None,
	):
		branch_descendants[branch].sort(key=key)
		for desc in branch_descendants[branch]:
			yield desc
			if stop(desc):
				continue
			if desc in branch_descendants:
				yield from cls._iter_descendants(branch_descendants, desc)

	def _create_rule_from_etree(
		self, rule: RuleName, branch: Branch, turn: Turn, tick: Tick
	):
		kwargs = {}
		for mapping, kwarg in [
			(self._known_triggers, "triggers"),
			(self._known_prereqs, "prereqs"),
			(self._known_actions, "actions"),
			(self._known_neighborhoods, "neighborhood"),
			(self._known_big, "big"),
		]:
			if (
				rule in mapping
				and branch in mapping[rule]
				and turn in mapping[rule][branch]
				and tick in mapping[rule][branch][turn]
			):
				kwargs[kwarg] = mapping[rule][branch][turn].pop(tick)
		self.create_rule(rule, branch, turn, tick, **kwargs)

	def _rule(
		self, branch_el: Element, turn_el: Element, rule_el: Element
	) -> None:
		branch = Branch(branch_el.get("name"))
		turn = Turn(int(turn_el.get("number")))
		tick = Tick(int(rule_el.get("end-tick")))
		character = CharName(literal_eval(rule_el.get("character")))
		rulebook = RulebookName(literal_eval(rule_el.get("rulebook")))
		rule = RuleName(rule_el.get("name"))
		match rule_el.get("type"):
			case "character":
				self.handled_character_rule(
					character,
					rulebook,
					rule,
					branch,
					turn,
					tick,
				)
			case "unit":
				graph = CharName(literal_eval(rule_el.get("graph")))
				unit = NodeName(literal_eval(rule_el.get("unit")))
				self.handled_unit_rule(
					character,
					graph,
					unit,
					rulebook,
					rule,
					branch,
					turn,
					tick,
				)
			case "character-thing":
				thing = NodeName(literal_eval(rule_el.get("thing")))
				self.handled_character_thing_rule(
					character,
					thing,
					rulebook,
					rule,
					branch,
					turn,
					tick,
				)
			case "character-place":
				place = NodeName(literal_eval(rule_el.get("place")))
				self.handled_character_place_rule(
					character,
					place,
					rulebook,
					rule,
					branch,
					turn,
					tick,
				)
			case "character-portal":
				orig = NodeName(literal_eval(rule_el.get("origin")))
				dest = NodeName(literal_eval(rule_el.get("destination")))
				self.handled_character_portal_rule(
					character,
					orig,
					dest,
					rulebook,
					rule,
					branch,
					turn,
					tick,
				)
			case "node":
				node = NodeName(literal_eval(rule_el.get("node")))
				self.handled_node_rule(
					character,
					node,
					rulebook,
					rule,
					branch,
					turn,
					tick,
				)
			case "portal":
				orig = NodeName(literal_eval(rule_el.get("origin")))
				dest = NodeName(literal_eval(rule_el.get("destination")))
				self.handled_portal_rule(
					character,
					orig,
					dest,
					rulebook,
					rule,
					branch,
					turn,
					tick,
				)

	def load_etree(
		self,
		tree: ElementTree,
	) -> None:
		root = tree.getroot()
		branch_descendants: dict[Branch, list[Branch]] = {Branch("trunk"): []}
		branch_starts: dict[Branch, tuple[Turn, Tick]] = {}
		if "_lisien_schema_version" in self.eternal:
			if self.eternal["_lisien_schema_version"] != int(
				root.get("db-schema-version")
			):
				raise RuntimeError("Incompatible database versions")
		else:
			self.eternal["_lisien_schema_version"] = int(
				root.get("db-schema-version")
			)
		if "xml-schema-version" in root.keys():
			ver = int(root.get("xml_schema_version"))
			if ver > XML_SCHEMA_VERSION:
				raise RuntimeError("Incompatible XML schema version", ver)
		self.eternal["trunk"] = root.get("trunk")
		self.eternal["branch"] = root.get("branch")
		self.eternal["turn"] = int(root.get("turn"))
		self.eternal["tick"] = int(root.get("tick"))
		for el in root:
			if el.tag == "language":
				continue
			if el.tag == "playtree":
				for branch_el in el:
					parent: Branch | None = branch_el.get("parent")
					branch = Branch(branch_el.get("name"))
					if parent is not None:
						if parent in branch_descendants:
							branch_descendants[parent].append(branch)
						else:
							branch_descendants[parent] = [branch]
					start_turn = Turn(int(branch_el.get("start-turn")))
					start_tick = Tick(int(branch_el.get("start-tick")))
					branch_starts[branch] = (start_turn, start_tick)
					end_turn = Turn(int(branch_el.get("end-turn")))
					end_tick = Tick(int(branch_el.get("end-tick")))
					self.set_branch(
						branch,
						parent,
						start_turn,
						start_tick,
						end_turn,
						end_tick,
					)
					if "last-turn-completed" in branch_el.keys():
						last_completed_turn = Turn(
							int(branch_el.get("last-turn-completed"))
						)
						self.complete_turn(branch, last_completed_turn, False)

					for turn_el in branch_el:
						turn = Turn(int(turn_el.get("number")))
						end_tick = Tick(int(turn_el.get("end-tick")))
						plan_end_tick = Tick(int(turn_el.get("plan-end-tick")))
						self.set_turn(branch, turn, end_tick, plan_end_tick)
						for elem in turn_el:
							if elem.tag == "rule":
								self._rule(branch_el, turn_el, elem)
								for ellem in elem:
									getattr(
										self, "_" + ellem.tag.replace("-", "_")
									)(branch_el, turn_el, ellem)
							else:
								getattr(
									self, "_" + elem.tag.replace("-", "_")
								)(branch_el, turn_el, elem)
				known_rules = (
					self._known_triggers.keys()
					| self._known_prereqs.keys()
					| self._known_actions.keys()
					| self._known_neighborhoods.keys()
					| self._known_big.keys()
				)
				trunk = Branch(el.get("trunk"))
				rules_created = set(self.rules_dump())
				for rule in known_rules:
					for mapp in [
						self._known_triggers,
						self._known_prereqs,
						self._known_actions,
						self._known_neighborhoods,
						self._known_big,
					]:
						if rule not in mapp:
							continue
						if rule not in rules_created:
							# Iterate depth first down the timestream, but no
							# deeper than when the rule is first set.
							# The game may have a rule by the same name
							# created in many branches independently.
							for branch in (
								trunk,
								*self._iter_descendants(
									branch_descendants,
									trunk,
									mapp[rule].__contains__,
									branch_starts.get,
								),
							):
								turn, tick = mapp[rule][branch].start_time()
								self._create_rule_from_etree(
									rule, branch, turn, tick
								)
								rules_created.add(rule)
				for mapp, setter in [
					(
						self._known_triggers,
						self.set_rule_triggers,
					),
					(self._known_prereqs, self.set_rule_prereqs),
					(self._known_actions, self.set_rule_actions),
					(
						self._known_neighborhoods,
						self.set_rule_neighborhood,
					),
					(self._known_big, self.set_rule_big),
				]:
					for rule in mapp:
						for branch in mapp[rule]:
							for turn in mapp[rule][branch]:
								# Turn and tick are guaranteed to be in
								# chronological order here, because that's what
								# a SettingsTurnDict does.
								for tick, datum in mapp[rule][branch][
									turn
								].items():
									setter(rule, branch, turn, tick, datum)
				for plan, times in self._plan_times.items():
					for branch, turn, tick in times:
						self.plans_insert(plan, branch, turn, tick)
			else:
				k = literal_eval(el.get("key"))
				v = self._element_to_value(el[0])
				self.eternal[k] = v
		self.commit()

	def load_xml(self, xml_or_file_path: str | os.PathLike | IOBase):
		"""Restore data from an XML export

		Supports a string with the XML in it, a path to an XML file, or
		a file object.

		"""
		self.load_etree(parse(xml_or_file_path))


_T = TypeVar("_T")


@dataclass
class PythonDatabaseConnector(AbstractDatabaseConnector):
	"""Database connector that holds all data in memory

	You'll have to write it to disk yourself. Use the ``write_xml`` method
	for that, or

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

	def echo(self, *args):
		if len(args) == 0:
			return
		elif len(args) == 1:
			return args[0]
		return args

	def call(self, query_name: str, *args, **kwargs):
		raise TypeError("Not a real database, so can't call it")

	def call_silent(self, query_name: str, *args, **kwargs):
		raise TypeError("Not a real database, so can't call it")

	def call_many(self, query_name: str, args: list) -> None:
		raise TypeError("Not a real database, so can't call it")

	def call_many_silent(self, query_name: str, args: list) -> None:
		raise TypeError("Not a real database, so can't call it")

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
			for b, r, t, g, n in sort_set(self._things2set.keys()):
				yield g, n, b, r, t, self._things2set[b, r, t, g, n]

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
				del self._things2set[key]

	def turns_completed_dump(self) -> Iterator[tuple[Branch, Turn]]:
		with self._lock:
			yield from sorted(self._turns_completed.items())

	def bookmarks_dump(self) -> Iterator[tuple[str, Time]]:
		with self._lock:
			yield from sort_set(self._bookmarks.items())


@dataclass
class NullDatabaseConnector(AbstractDatabaseConnector):
	"""Database connector that does nothing, connects to no database

	This will never return any data, either. If you want it to hold data
	you put into it, instead use :class:`PythonDatabaseConnector`.

	"""

	def echo(self, *args):
		if len(args) == 1:
			return args[0]
		return args

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
			try:
				ret[table].extend(starmap(partial(f, self, branch), got))
			except TypeError as err:
				raise TypeError(*err.args, table, got[0]) from err
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

	def echo(self, string: str) -> str:
		with self.mutex():
			self._inq.put(("echo", string))
			ret = self._outq.get()
			self._outq.task_done()
			return ret

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
		turn: Turn,
		char_b: bytes,
		rb_b: bytes,
		rule: RuleName,
		tick: Tick,
	) -> CharacterRulesHandledRowType:
		char = CharName(self.unpack_key(char_b))
		rb = RulebookName(self.unpack_key(rb_b))
		return char, rb, rule, branch, turn, tick

	@window_getter("unit_rules_handled")
	def _unpack_unit_rule_handled_rec(
		self,
		branch: Branch,
		turn: Turn,
		char_b: bytes,
		graph_b: bytes,
		unit_b: bytes,
		rb_b: bytes,
		rule: RuleName,
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
		turn: Turn,
		char_b: bytes,
		thing_b: bytes,
		rb_b: bytes,
		rule: RuleName,
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
		turn: Turn,
		char_b: bytes,
		place_b: bytes,
		rb_b: bytes,
		rule: RuleName,
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
		turn: Turn,
		char_b: bytes,
		orig_b: bytes,
		dest_b: bytes,
		rb_b: bytes,
		rule: RuleName,
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
		turn: Turn,
		char_b: bytes,
		node_b: bytes,
		rb_b: bytes,
		rule: RuleName,
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
		turn: Turn,
		char_b: bytes,
		orig_b: bytes,
		dest_b: bytes,
		rb_b: bytes,
		rule: RuleName,
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
