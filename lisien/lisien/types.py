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
"""Base classes and type hints for Lisien"""

from __future__ import annotations

import builtins
import inspect
import operator
import os
import typing
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from collections.abc import Sequence, Set
from concurrent.futures import Future
from enum import Enum
from functools import cached_property, partial, wraps
from itertools import chain
from logging import Logger
from operator import (
	add,
	attrgetter,
	eq,
	floordiv,
	ge,
	gt,
	le,
	lt,
	mod,
	mul,
	ne,
	pow,
	sub,
	truediv,
)
from pathlib import Path, PosixPath, WindowsPath
from random import Random
from threading import RLock
from types import (
	EllipsisType,
	FunctionType,
	GenericAlias,
	GetSetDescriptorType,
	MethodType,
)
from typing import (
	TYPE_CHECKING,
	Annotated,
	Any,
	Callable,
	ClassVar,
	ContextManager,
	Iterable,
	Iterator,
	KeysView,
	Literal,
	Mapping,
	MutableMapping,
	NewType,
	Optional,
	Protocol,
	Self,
	TypedDict,
	TypeGuard,
	TypeVar,
	Union,
	get_args,
	get_origin,
	overload,
	override,
)

import networkx as nx
from annotated_types import Ge, Le
from attrs import Factory, define, field, validators
from blinker import Signal, ANY
from networkx import NetworkXError
from tblib import Traceback
from zict import LRU

from . import enum, exc
from .enum import MsgpackExtensionType
from .exc import TimeError, TravelException, WorkerProcessReadOnlyError
from .util import concat_d, concat_list
from .wrap import (
	DictWrapper,
	ListWrapper,
	MappingUnwrapper,
	OrderlyFrozenSet,
	OrderlySet,
	SetWrapper,
	unwrap_items,
	wrapval,
)

if TYPE_CHECKING:
	from .character import Character
	from .db import AbstractDatabaseConnector
	from .engine import Engine
	from .facade import EngineFacade
	from .node import Thing
	from .portal import Portal
	from .rule import Rule, RuleBook


type KeyHint = (
	str | int | float | None | tuple[KeyHint, ...] | frozenset[KeyHint]
)
"""This can be serialized as a key in a mapping"""

type ValueHint = (
	str
	| int
	| float
	| None
	| EllipsisType
	| dict[KeyHint, ValueHint]
	| list[ValueHint]
	| tuple[ValueHint, ...]
	| set[ValueHint]
	| frozenset[ValueHint]
	| FunctionType
	| MethodType
)
"""This can be serialized, generally"""


def is_valid_value(obj: Any) -> TypeGuard[Value]:
	"""Is this an object that Lisien can serialize as a value?"""
	return (
		obj is ...
		or obj is None
		or isinstance(obj, (str, int, float))
		or (
			isinstance(obj, (list, tuple, set, frozenset))
			and all(is_valid_value(elem) for elem in obj)
		)
		or isinstance(obj, Node)
		or isinstance(obj, Edge)
		or isinstance(obj, DiGraph)
		or (
			isinstance(obj, (list, ListWrapper))
			and all(map(is_valid_value, obj))
		)
		or (
			isinstance(obj, (dict, DictWrapper))
			and all(map(is_valid_key, obj.keys()))
			and all(map(is_valid_value, obj.values()))
		)
		or (
			isinstance(obj, (Set, Sequence, SetWrapper))
			and isinstance(obj, Iterable)
			and all(map(is_valid_value, obj))
		)
		or (
			isinstance(obj, nx.DiGraph)
			and all(map(is_valid_key, obj.graph.keys()))
			and all(map(is_valid_value, obj.graph.values()))
			and all(
				is_valid_key(k) and is_valid_value(v)
				for node in obj.nodes().values()
				for (k, v) in node.items()
			)
			and all(
				is_valid_key(orig)
				and is_valid_key(dest)
				and is_valid_key(k)
				and is_valid_value(v)
				for orig in obj.adj
				for dest in obj.adj[orig]
				for (k, v) in obj.adj[orig][dest].items()
			)
		)
		or (
			isinstance(obj, (FunctionType, MethodType))
			and obj.__module__
			in {"function", "method", "trigger", "prereq", "action"}
		)
	)


class _ValueMeta(type):
	def __instancecheck__(self, instance) -> TypeGuard[Value]:
		return is_valid_value(instance)

	@staticmethod
	def __call__(obj) -> Value:
		if is_valid_value(obj):
			return obj
		raise TypeError("Not a valid value", obj)

	def __class_getitem__(cls, item):
		return GenericAlias(cls, item)


class Value(metaclass=_ValueMeta):
	"""Serializable object. Used for dynamic typing."""

	def __new__(cls, obj: ValueHint) -> Value:
		if not is_valid_value(obj):
			raise TypeError("Invalid value")
		return obj


def is_valid_key(obj: KeyHint) -> TypeGuard[Key]:
	"""Is this an object that Lisien can serialize as a key?"""
	return (
		obj is None
		or isinstance(obj, (str, int, float))
		or (
			isinstance(obj, (tuple, frozenset))
			and all(is_valid_key(elem) for elem in obj)
		)
	)


class _KeyMeta(_ValueMeta):
	def __instancecheck__(self, instance) -> TypeGuard[Key]:
		return is_valid_key(instance)

	@staticmethod
	def __call__(obj: KeyHint) -> Key:
		if is_valid_key(obj):
			return obj
		raise TypeError(f"Not a valid key: {obj}", obj)

	def __class_getitem__(cls, item):
		return GenericAlias(cls, item)


class Key(Value, metaclass=_KeyMeta):
	"""Serializable key for mappings. Used in dynamic typing."""

	def __new__(cls, obj: KeyHint) -> Key:
		if not is_valid_key(obj):
			raise TypeError("Invalid key")
		return obj


def keyval(pair: tuple[KeyHint, ValueHint]) -> tuple[Key, Value]:
	"""Convenience function to serialize both items of a pair"""
	k, v = pair
	return Key(k), Value(v)


Stat = NewType("Stat", Key)
"""Key of a Lisien entity's variable"""
EternalKey = NewType("EternalKey", Key)
"""Key in the ``eternal`` mapping, not subject to time travel"""
UniversalKey = NewType("UniversalKey", Key)
"""Key in the ``universal`` mapping, subject to time travel"""
Branch = NewType("Branch", str)
"""Name of a branch of time"""
Turn = NewType("Turn", Annotated[int, Ge(0)])
"""Turn number. Starts at zero and increases monotonically."""
Tick = NewType("Tick", Annotated[int, Ge(0)])
"""Tick number. Starts at zero and increases one by one, but resets every turn."""


class Time(tuple[Branch, Turn, Tick]):
	"""A tuple of branch, turn, and tick

	If you access this from :attr:`lisien.engine.Engine.time`, you can register
	a function to be called whenever the time changes::

		from lisien import Engine

		with Engine() as engine:
			@engine.time.connect
			def time_changed(engine, *, then: tuple[str, int, int], now: tuple[str, int, int]):
				print(f'{then} was then, {now} is now')

	"""

	__slots__ = ()

	def __new__(
		cls,
		branch: Branch
		| tuple[Branch, Turn, Tick]
		| tuple[Branch, Turn, Tick, AbstractEngine],
		turn: Turn | None = None,
		tick: Tick | None = None,
		engine: AbstractEngine | None = None,
	) -> Time:
		if turn is None or tick is None:
			if not isinstance(branch, tuple):
				raise TypeError(
					f"Can't make Time out of {type(branch)}", branch
				)
			if len(branch) == 4:
				return tuple.__new__(cls, branch)
			elif len(branch) == 3:
				return tuple.__new__(cls, (*branch, None))
			else:
				raise TypeError(f"Wrong arity", branch)
		return tuple.__new__(cls, (branch, turn, tick, engine))

	@property
	def engine(self) -> AbstractEngine | None:
		return tuple.__getitem__(self, 3)

	@property
	def branch(self) -> Branch:
		return tuple.__getitem__(self, 0)

	@property
	def turn(self):
		return tuple.__getitem__(self, 1)

	@property
	def tick(self):
		return tuple.__getitem__(self, 2)

	def __iter__(self):
		yield super().__getitem__(0)
		yield super().__getitem__(1)
		yield super().__getitem__(2)

	def __len__(self):
		return 3

	def __hash__(self):
		return hash(
			(
				super().__getitem__(0),
				super().__getitem__(1),
				super().__getitem__(2),
			)
		)

	def __getitem__(self, i: str | int) -> Branch | Turn | Tick:
		if i in ("branch", 0):
			return super().__getitem__(0)
		if i in ("turn", 1):
			return super().__getitem__(1)
		if i in ("tick", 2):
			return super().__getitem__(2)
		if isinstance(i, int):
			raise IndexError(i)
		else:
			raise KeyError(i)

	@wraps(Signal.connect)
	def connect(
		self,
		receiver: Callable[..., Any],
		sender: Any = ANY,
		weak: bool = True,
	) -> None:
		self.engine._time_signal.connect(receiver)

	@wraps(Signal.disconnect)
	def disconnect(
		self, receiver: Callable[..., Any], sender: Any = ANY
	) -> None:
		self.engine._time_signal.disconnect(receiver, sender)

	@wraps(Signal.connect_via)
	def connect_via(
		self, sender, weak: bool = False
	) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
		return self.engine._time_signal.connect_via(sender, weak)

	@wraps(Signal.connected_to)
	def connected_to(
		self, receiver: Callable[..., Any], sender: Any = ANY
	) -> Generator[None, None, None]:
		return self.engine._time_signal.connected_to(receiver, sender)

	@wraps(Signal.muted)
	def muted(self) -> Generator[None, None, None]:
		return self.engine._time_signal.muted()

	@wraps(Signal.has_receivers_for)
	def has_receivers_for(self, sender):
		return self.engine._time_signal.has_receivers_for(sender)

	@wraps(Signal.receivers_for)
	def receivers_for(self, sender):
		return self.engine._time_signal.receivers_for(sender)

	@wraps(Signal.send)
	def send(self, sender, *, _async_wrapper=None, **kwargs):
		return self.engine._time_signal.send(
			sender, _async_wrapper=_async_wrapper, **kwargs
		)

	@wraps(Signal.send_async)
	def send_async(self, sender, *, _sync_wrapper=None, **kwargs):
		return self.engine._time_signal.send_async(
			sender, _sync_wrapper=_sync_wrapper, **kwargs
		)


class ValidateTimeProtocol(Protocol):
	@override
	@staticmethod
	def __call__(time) -> Time: ...

	@override
	@staticmethod
	def __call__(time: LinearTime) -> LinearTime: ...

	@override
	@staticmethod
	def __call__(time: tuple[int, int]) -> LinearTime: ...

	@override
	@staticmethod
	def __call__(time: tuple[str, int, int]) -> Time: ...

	@override
	@staticmethod
	def __call__(time: tuple[Turn, Tick]) -> LinearTime: ...

	@override
	@staticmethod
	def __call__(time: tuple[Branch, Turn, Tick]) -> Time: ...


def _validate_time(time):
	if not isinstance(time, tuple):
		raise TypeError("Invalid time", time)
	match len(time):
		case 3:
			if not isinstance(time[0], str):
				raise TypeError("Invalid branch", time[0])
			if not isinstance(time[1], int):
				raise TypeError("Invalid turn", time[1])
			if not isinstance(time[2], int):
				raise TypeError("Invalid tick", time[2])
			if time[1] < 0:
				raise ValueError("Negative turn", time[1])
			if time[2] < 0:
				raise ValueError("Negative tick", time[2])
			return time
		case 2:
			if not isinstance(time[0], int):
				raise TypeError("Invalid turn", time[0])
			if not isinstance(time[1], int):
				raise TypeError("Invalid tick", time[1])
			if time[0] < 0:
				raise ValueError("Negative turn")
			if time[1] < 0:
				raise ValueError("Negative tick")
			return LinearTime(*Time)
		case _:
			raise TypeError("Invalid time", time)


validate_time: ValidateTimeProtocol = _validate_time


class LinearTime(tuple[Turn, Tick]):
	"""A point in time within some branch"""

	def __new__(
		cls, tup: tuple[Turn, Tick] | Turn, tick: Tick | None = None
	) -> LinearTime:
		if tick is not None:
			if not isinstance(tick, int):
				raise TypeError("Invalid tick", tick)
			turn: Turn = tup
			if not isinstance(turn, int):
				raise TypeError("Invalid turn", turn)
			return tuple.__new__(cls, (turn, tick))
		turn, tick = tup
		if not isinstance(turn, int):
			raise TypeError("Invalid turn")
		if not isinstance(tick, int):
			raise TypeError("Invalid tick")
		return tuple.__new__(cls, tup)


type TimeWindow = tuple[Branch, Turn, Tick, Turn, Tick]
"""The span of time between two points in the same branch, inclusive"""
Plan = NewType("Plan", Annotated[int, Ge(0)])
"""Identifier for a plan, corresponding to some state changes that may be canceled"""
type PlanTicksRowType = tuple[Plan, Branch, Turn, Tick]
"""Row in the ``plan_ticks`` table

Indicates that the given point in time is planned, and hasn't 'really happened'.

"""
CharName = NewType("CharName", Key)
"""Name of a :class:`lisien.character.Character`"""
NodeName = NewType("NodeName", Key)
"""Name of a :class:`lisien.node.Place` or :class:`lisien.node.Thing`"""


type EntityKey = (
	tuple[CharName]
	| tuple[CharName, NodeName]
	| tuple[CharName, NodeName, NodeName]
)
"""Key identifying a Lisien entity

Possible entity types are :class:`lisien.character.Character`, identified by
its name; :class:`lisien.node.Place` and :class:`lisien.node.Thing`,
identified by the name of the character they're in and their own name;
and :class:`lisien.portal.Portal`, identified by the name of the character
it's in, and the names of its two endpoints (origin and destination).

"""
RulebookName = NewType("RulebookName", Key)
"""Key identifying a rulebook

Every Lisien entity gets its own rulebook by default, named the same as its
:type:`EntityKey`, but the user may assign it another rulebook if they like.

"""


RulebookPriority = NewType("RulebookPriority", float)
"""Priority of a rulebook, used to sort rulebooks before evaluation

Infinite values are permitted, though not recommended. Defaults to ``0.0``.

"""
RuleName = NewType("RuleName", str)
"""Name of a rule

Rules are made of three function lists: triggers, prereqs, and actions.
They can be assigned to arbitrarily many rulebooks, which in turn can be
assigned to as many Lisien entities as the user wants.

"""


type RuleNeighborhood = Annotated[int, Ge(0)] | None
"""Size of a rule's neighborhood

:type:`RuleNeighborhood` is an integer specifying how far away the rules engine
will look to determine whether to evaluate the trigger functions for a rule.
It only has effects on rules applied to nodes. If not ``None``, it
is how many edges (:class:`lisien.portal.Portal` objects) the rules engine
will traverse to build a collection of nodes, and only evaluate the rule's
triggers if something changed in that collection since the previous turn.
If ``None`` (the default), triggers will be evaluated each and every turn.

"""
RuleBig = NewType("RuleBig", bool)
"""Whether a rule makes a lot of changes to the world

Big rules won't be run on regular Lisien entities; instead, they will be run
on their facades. The changes to those facades will then be applied to the
real entities in a batch, which is faster when the number of changes is big,
but slower when it's small.

"""
RuleFunc = NewType("RuleFunc", FunctionType)
"""A function that's in one of the lists that make up a rule

A trigger, prereq, or action.

"""
FuncName = NewType("FuncName", str)
"""The name of a function in one of Lisien's function stores"""
type FuncStoreName = Literal[
	"trigger", "prereq", "action", "function", "method"
]
"""Name of one of Lisien's function stores"""
TriggerFuncName = NewType("TriggerFuncName", FuncName)
"""Name of a Boolean function to trigger a rule"""
TriggerFunc = NewType("TriggerFunc", RuleFunc)
"""Boolean function to trigger a rule"""


PrereqFuncName = NewType("PrereqFuncName", FuncName)
"""Name of a Boolean function required to return ``True`` before a rule runs"""
PrereqFunc = NewType("PrereqFunc", RuleFunc)
"""Boolean function required to return ``True`` before a rule runs"""

ActionFuncName = NewType("ActionFuncName", FuncName)
"""Name of a function that alters the world state for a rule"""
ActionFunc = NewType("ActionFunc", RuleFunc)
"""Function that alters the world state as a result of a rule"""


type RuleFuncName = TriggerFuncName | PrereqFuncName | ActionFuncName
"""Name of a function that can be used in a rule somehow"""
type UniversalKeyframe = dict[UniversalKey, Value]
"""The state of the ``universal`` mapping at a point in the timestream"""


class RuleKeyframe(TypedDict):
	"""The state of all rules at a point in the timestream"""

	triggers: dict[RuleName, list[TriggerFuncName]]
	prereqs: dict[RuleName, list[PrereqFuncName]]
	actions: dict[RuleName, list[ActionFuncName]]
	neighborhood: dict[RuleName, RuleNeighborhood]
	big: dict[RuleName, RuleBig]


type RulebooksKeyframe = dict[
	RulebookName, tuple[list[RuleName], RulebookPriority]
]
"""The state of all rulebooks at a point in the timestream"""
type UniversalRowType = tuple[Branch, Turn, Tick, UniversalKey, Value]
"""A row from the ``universals`` table, declaring a key's value at a point in the timestream"""
type RulebookRowType = tuple[
	Branch,
	Turn,
	Tick,
	RulebookName,
	list[RuleName],
	RulebookPriority,
]
"""A row from the ``rulebooks`` table, declaring a rulebook's rules and priority at a point in the timestream"""
type RuleRowType = tuple[
	Branch,
	Turn,
	Tick,
	RuleName,
	list[TriggerFuncName]
	| list[PrereqFuncName]
	| list[ActionFuncName]
	| RuleNeighborhood
	| RuleBig,
]
"""A row from one of the rule tables"""
type TriggerRowType = tuple[
	Branch, Turn, Tick, RuleName, list[TriggerFuncName]
]
"""A row from the ``rule_triggers`` table

Gives the names of a rule's trigger functions at a point in the timestream.

"""
type PrereqRowType = tuple[Branch, Turn, Tick, RuleName, list[PrereqFuncName]]
"""A row from the ``rule_prereqs`` table

Gives the names of a rule's prereq functions at a point in the timestream.

"""
type ActionRowType = tuple[Branch, Turn, Tick, RuleName, list[ActionFuncName]]
"""A row from the ``rule_actions`` table

Gives the names of a rule's action functions at a point in the timestream.

"""
type RuleNeighborhoodRowType = tuple[
	Branch, Turn, Tick, RuleName, RuleNeighborhood
]
"""A row from the ``rule_neighborhood`` table

Gives the number of edges to traverse to build the neighborhood of nodes to
consider in deciding whether to evaluate the rule's triggers. If it's ``None``,
the triggers will be evaluated every turn.

"""
type RuleBigRowType = tuple[Branch, Turn, Tick, RuleName, RuleBig]
"""A row from the ``rule_big`` table

Has a Boolean value that determines whether all of a rule's changes to the
world should be batched.

"""
type BranchRowType = tuple[Branch, Branch | None, Turn, Tick, Turn, Tick]
"""A row from the ``branches`` table

Has the parent branch, if any, and the linear span of time contained in the
branch.

"""
type TurnRowType = tuple[Branch, Turn, Tick, Tick]
"""A row from the ``turns`` table

Has the 'real' last tick of each turn that's been simulated, as well as the
'planned' last tick, which may be higher, if there are planned changes left.

"""
type GraphTypeStr = Literal["DiGraph", "Deleted"]
"""Permissible graph types

Currently, Lisien only supports the 'DiGraph' type, and the type 'Deleted' for
graphs that don't exist anymore.

Graphs are most often called 'characters' in Lisien.

"""
type GraphRowType = tuple[Branch, Turn, Tick, CharName, GraphTypeStr]
"""A row from the ``graphs`` table

Marks the point in the timestream when a character is created or deleted.

The last element is a string identifying the type of graph, but only 'DiGraph'
and 'Deleted' are accepted at the moment.

"""
type NodeRowType = tuple[Branch, Turn, Tick, CharName, NodeName, bool]
"""A row from the ``nodes`` table

Marks the point in the timestream when a node is created or deleted.

The last element is ``True`` when the node is created, and ``False`` when
it is deleted.

In Lisien, nodes are categorized as places or things, but that distinction
doesn't matter in the ``nodes`` table.

"""
type EdgeRowType = tuple[
	Branch, Turn, Tick, CharName, NodeName, NodeName, bool
]
"""A row from the ``edges`` table

Marks the point in the timestream when an edge is created or deleted.

The last element is ``True`` when the edge is created, and ``False`` when it
is deleted.

In Lisien, edges are usually called portals.

"""
type GraphValRowType = tuple[Branch, Turn, Tick, CharName, Stat, Value]
"""A row from the ``graph_val`` table

The last element is the value that the given stat of the given character takes
at this point in the timestream.

"""
type NodeValRowType = tuple[
	Branch, Turn, Tick, CharName, NodeName, Stat, Value
]
"""A row from the ``node_val`` table

The last element is the value that the given stat of a node takes at this
point in the timestream.

In Lisien, nodes are either places or things, but that distinction doesn't
matter in the ``node_val`` table.

The stat cannot be ``'location'``. Things' locations are stored in the
``things`` table--see :type:`ThingRowType`, below.

"""
type EdgeValRowType = tuple[
	Branch, Turn, Tick, CharName, NodeName, NodeName, Stat, Value
]
"""A row from the ``edge_val`` table

The last element is the value that the given stat of an edge takes at this
point in the timestream.

Edges are usually called portals in Lisien.

"""
type ThingRowType = tuple[
	Branch, Turn, Tick, CharName, NodeName, NodeName | EllipsisType
]
"""A row from the ``things`` table

The last element is the location that the given thing is in at this point in
the timestream. It must be the name of a node in the same character, or the
ellipsis object, if the thing becomes a place.

It doesn't matter whether the location is a thing or a place.

"""
type UnitRowType = tuple[
	Branch, Turn, Tick, CharName, CharName, NodeName, bool
]
"""A row from the ``units`` table

Indicates whether a node in the latter character should be considered a unit
of the former character. If the last element is ``True``, then the former
character is the leader of the node.

The latter character is most commonly one representing the physical world,
and the former is usually a fictional person.

"""
type CharRulebookRowType = tuple[Branch, Turn, Tick, CharName, RulebookName]
"""A row from one of the character-rulebook tables

Has a rulebook for a character to follow, starting at this point in the
timestream.

Depending on which table it's in, the rule might act on the
:class:`lisien.character.Character` object itself, or on a set of entities
in the character.

The tables in question are:
* ``character_rulebook``
* ``unit_rulebook``
* ``character_thing_rulebook``
* ``character_place_rulebook``
* ``character_portal_rulebook``

"""
type NodeRulebookRowType = tuple[
	Branch, Turn, Tick, CharName, NodeName, RulebookName
]
"""A row from the ``node_rulebook`` table

Has a rulebook for a node in a character to follow, starting at this point in
the timestream.

The node may be a place or a thing.

"""
type PortalRulebookRowType = tuple[
	Branch, Turn, Tick, CharName, NodeName, NodeName, RulebookName
]
"""A row from the ``portal_rulebook`` table

Has a rulebook for a portal in a character to follow, starting at this point
in the timestream.

Portals are edges in graphs, which are also called characters.

"""
type AssignmentRowType = (
	NodeRowType
	| NodeValRowType
	| EdgeRowType
	| EdgeValRowType
	| GraphValRowType
	| ThingRowType
	| UnitRowType
	| CharRulebookRowType
	| NodeRulebookRowType
	| PortalRulebookRowType
)
"""A row from one of the tables that stores entity state

Includes rows in tables that don't exactly reflect 'stats' but are stored
similarly.

"""
type AssignmentRowListType = (
	list[NodeRowType]
	| list[NodeValRowType]
	| list[EdgeRowType]
	| list[EdgeValRowType]
	| list[GraphValRowType]
	| list[ThingRowType]
	| list[UnitRowType]
	| list[CharRulebookRowType]
	| list[NodeRulebookRowType]
	| list[PortalRulebookRowType]
)
"""A list of rows reflecting entity state, which might be loaded in bulk"""
type CharacterRulesHandledRowType = tuple[
	Branch,
	Turn,
	CharName,
	RulebookName,
	RuleName,
	Tick,
]
"""A row from the ``character_rules_handled`` table

Has the point in the timestream at which a particular rule in the rulebook
finished running on a given character this turn.

"""
type PortalRulesHandledRowType = tuple[
	Branch,
	Turn,
	CharName,
	NodeName,
	NodeName,
	RulebookName,
	RuleName,
	Tick,
]
"""A row from the ``portal_rules_handled`` table

Has the point in the timestream at which a particular rule in the rulebook
finished running on a given portal this turn.

"""
type NodeRulesHandledRowType = tuple[
	Branch,
	Turn,
	CharName,
	NodeName,
	RulebookName,
	RuleName,
	Tick,
]
"""A row from the ``node_rules_handled`` table

Has the point in the timestream at which a particular rule in the rulebook
finished running on a given node this turn.

"""
type UnitRulesHandledRowType = tuple[
	Branch,
	Turn,
	CharName,
	CharName,
	NodeName,
	RulebookName,
	RuleName,
	Tick,
]
"""A row from the ``unit_rules_handled`` table

Has the point in the timestream at which a particular rule in the rulebook
finished running on a given unit this turn.

"""
type StatDict = dict[Stat | Literal["rulebook"], Value | RulebookName]
type ThingDict = dict[
	Stat | Literal["rulebook", "location"], Value | RulebookName
]
type CharDict = dict[
	Stat
	| Literal[
		"units",
		"character_rulebook",
		"unit_rulebook",
		"character_thing_rulebook",
		"character_place_rulebook",
		"character_portal_rulebook",
	],
	Value | dict[CharName, dict[NodeName, bool]] | RulebookName,
]
type GraphValKeyframe = dict[CharName, CharDict]
type NodeValDict = dict[NodeName, StatDict]
NodeKeyframe = NodeValDict
type GraphNodeValKeyframe = dict[CharName, NodeValDict]
type EdgeValDict = dict[NodeName, dict[NodeName, StatDict]]
EdgeKeyframe = EdgeValDict
type GraphEdgeValKeyframe = dict[CharName, EdgeValDict]
type NodesDict = dict[NodeName, bool]
type GraphNodesKeyframe = dict[CharName, NodesDict]
type EdgesDict = dict[NodeName, dict[NodeName, bool]]
type GraphEdgesKeyframe = dict[CharName, EdgesDict]
type UnitsDict = dict[CharName, dict[NodeName, bool]]
type CharDelta = dict[
	Stat
	| Literal[
		"character_rulebook",
		"unit_rulebook",
		"character_thing_rulebook",
		"character_place_rulebook",
		"character_portal_rulebook",
		"nodes",
		"node_val",
		"edges",
		"edge_val",
		"rulebooks",
		"rules",
		"units",
	],
	NodesDict
	| NodeValDict
	| EdgesDict
	| EdgeValDict
	| RulebookName
	| UnitsDict
	| dict[
		RuleName,
		dict[
			Literal["triggers", "prereqs", "actions"],
			list[TriggerFuncName]
			| list[PrereqFuncName]
			| list[ActionFuncName],
		],
	]
	| Value,
]
type DeltaDict = dict[
	CharName,
	CharDelta | None,
]
type KeyframeGraphRowType = tuple[
	Branch,
	Turn,
	Tick,
	CharName,
	NodeKeyframe,
	EdgeKeyframe,
	CharDict,
]
type KeyframeExtensionRowType = tuple[
	Branch, Turn, Tick, UniversalKeyframe, RuleKeyframe, RulebooksKeyframe
]


class Keyframe(TypedDict):
	universal: dict[UniversalKey, Value]
	triggers: dict[RuleName, list[TriggerFuncName]]
	prereqs: dict[RuleName, list[PrereqFuncName]]
	actions: dict[RuleName, list[ActionFuncName]]
	neighborhood: dict[RuleName, RuleNeighborhood]
	big: dict[RuleName, RuleBig]
	rulebook: dict[RulebookName, tuple[list[RuleName], RulebookPriority]]
	graph_val: GraphValKeyframe
	node_val: GraphNodeValKeyframe
	edge_val: GraphEdgeValKeyframe
	nodes: GraphNodesKeyframe
	edges: GraphEdgesKeyframe


type SlightlyPackedDeltaType = dict[
	bytes,
	dict[
		bytes,
		bytes
		| dict[
			bytes,
			bytes | dict[bytes, bytes | dict[bytes, bytes]],
		],
	],
]
type RulebookTypeStr = Literal[
	"character",
	"unit",
	"character_thing",
	"character_place",
	"character_portal",
]
type CharacterRulebookTypeStr = Literal[
	"character_rulebook",
	"unit_rulebook",
	"character_thing_rulebook",
	"character_place_rulebook",
	"character_portal_rulebook",
]


@define(eq=False)
class CharacterMapping[_K, _V](MappingUnwrapper[_K, _V], ABC):
	"""Common amenities for mappings in :class:`Character`"""

	__slots__ = ()

	character: AbstractCharacter

	@cached_property
	def engine(self) -> AbstractEngine:
		return self.character.engine

	@cached_property
	def name(self):
		return self.character.name


class AbstractEntityMapping[_K, _V](
	MappingUnwrapper, MutableMapping[_K, _V], ABC
):
	__slots__ = ()

	@abstractmethod
	def _get_cache(
		self, key: Key, branch: Branch, turn: Turn, tick: Tick
	) -> dict: ...

	def _get_cache_now(self, key: Key):
		return self._get_cache(key, *self.engine.time)

	@abstractmethod
	def _cache_contains(
		self, key: Key, branch: Branch, turn: Turn, tick: Tick
	): ...

	@abstractmethod
	def _set_db(
		self,
		key: Key,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
	):
		"""Set a value for a key in the database (not the cache)."""

	@abstractmethod
	def _set_cache(
		self,
		key: Key,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
	): ...

	def _del_db(self, key: Key, branch: Branch, turn: Turn, tick: Tick):
		"""Delete a key from the database (not the cache)."""
		self._set_db(key, branch, turn, tick, Value(...))

	def _del_cache(self, key: Key, branch: Branch, turn: Turn, tick: Tick):
		self._set_cache(key, branch, turn, tick, Value(...))

	def __getitem__(self, key: Key | KeyHint):
		"""If key is 'graph', return myself as a dict, else get the present
		value of the key and return that

		"""
		if not isinstance(key, Key):
			raise TypeError("Invalid key", key)
		return wrapval(self, key, self._get_cache_now(key))

	def __contains__(self, item: Key | KeyHint):
		if not isinstance(item, Key):
			raise TypeError("Invalid key", item)
		return item == "name" or self._cache_contains(item, *self.engine.time)

	def __setitem__(self, key: Key | KeyHint, value: Value | ValueHint):
		"""Set key=value at the present branch and revision"""
		if not isinstance(key, Key):
			raise TypeError("Invalid key", key)
		if value is ...:
			raise ValueError(
				"Lisien uses the ellipsis to indicate that a key's been deleted"
			)
		if not isinstance(value, Value):
			raise TypeError("Invalid value", value)
		try:
			if self._get_cache_now(key) == value:
				return
		except KeyError:
			pass
		branch, turn, tick = self.engine._nbtt()
		self._set_cache(key, branch, turn, tick, value)
		self._set_db(key, branch, turn, tick, value)

	def __delitem__(self, key: Key | KeyHint):
		if not isinstance(key, Key):
			raise TypeError("Invalid key", key)
		self._get_cache_now(key)  # deliberately raise KeyError if unset
		branch, turn, tick = self.engine._nbtt()
		self._del_cache(key, branch, turn, tick)
		self._del_db(key, branch, turn, tick)


@define(eq=False)
class GraphMapping(AbstractEntityMapping[Stat, Value], ABC):
	"""Mapping for graph attributes"""

	__slots__ = ()
	character: DiGraph

	@cached_property
	def engine(self):
		return self.character.engine

	def __iter__(self) -> Iterator[Stat]:
		yield from self.engine._graph_val_cache.iter_keys(
			self.character.name, *self.engine.time
		)

	def _cache_contains(
		self, key: Stat, branch: Branch, turn: Turn, tick: Tick
	) -> bool:
		return self.engine._graph_val_cache.contains_key(
			self.character.name, key, branch, turn, tick
		)

	def __len__(self):
		branch, turn, tick = btt
		return self.engine._graph_val_cache.count_keys(
			self.character.name, branch, turn, tick
		)

	def __getitem__(self, item: Key | KeyHint) -> Value:
		if item == "name":
			return self.character.name
		return super().__getitem__(item)

	def __setitem__(
		self, key: Stat | KeyHint, value: Value | ValueHint
	) -> None:
		if key == "name":
			raise KeyError("name cannot be changed after creation")
		super().__setitem__(key, value)

	def _get_cache(
		self, key: Stat, branch: Branch, turn: Turn, tick: Tick
	) -> Value:
		return self.engine._graph_val_cache.retrieve(
			self.character.name, key, branch, turn, tick
		)

	def _get(self, key: Stat) -> Value:
		return self._get_cache(key, *self.engine.time)

	def _set_db(
		self,
		key: Stat,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
	) -> None:
		self.engine.database.graph_val_set(
			self.character.name, key, branch, turn, tick, value
		)

	def _set_cache(
		self,
		key: Stat,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
	) -> None:
		self.engine._graph_val_cache.store(
			self.character.name, key, branch, turn, tick, value
		)

	def _del_db(
		self, key: Stat, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		self.engine.database.graph_val_set(
			self.character.name, key, branch, turn, tick, Value(...)
		)

	def clear(self) -> None:
		keys = set(self.keys())
		for k in keys:
			del self[k]

	def unwrap(self) -> dict[Stat, Value]:
		return unwrap_items(self.items())

	def update(self, upd: dict[Stat, Value] | dict[KeyHint, ValueHint]):
		realupd: dict[Stat, Value] = {}
		for k, v in upd.items():
			if not isinstance(k, Key):
				raise TypeError("Invalid graph stat", k)
			k = Stat(k)
			if not isinstance(v, Value):
				raise TypeError("Invalid stat value", v)
			realupd[k] = v
		for k, v in realupd.items():
			if v is ...:
				del self[k]
			else:
				self[k] = v

	def __eq__(
		self, other: Mapping[Stat | KeyHint, Value | ValueHint]
	) -> bool:
		if hasattr(other, "unwrap"):
			other = other.unwrap()
		other = other.copy()
		me = self.unwrap().copy()
		return me == other


@define(eq=False)
class Node(AbstractEntityMapping, ABC):
	__slots__ = ()

	character: AbstractCharacter
	name: NodeName

	def _validate_node_type(self):
		return True

	@override
	def __getitem__(self, item: Literal["name"]) -> NodeName: ...

	@override
	def __getitem__(self, item: Key | KeyHint) -> Value: ...

	def __getitem__(
		self, item: Literal["name"] | Stat | KeyHint
	) -> NodeName | Value:
		if item == "name":
			return self.name
		return Value(super().__getitem__(item))

	@cached_property
	def engine(self):
		return self.character.engine

	def __iter__(self):
		branch, turn, tick = self.engine.time
		return self.engine._node_val_cache.iter_keys(
			self.character.name, self.name, branch, turn, tick
		)

	def _cache_contains(
		self, key: Stat, branch: Branch, turn: Turn, tick: Tick
	) -> bool:
		return self.engine._node_val_cache.contains_key(
			self.character.name, self.name, key, branch, turn, tick
		)

	def __len__(self):
		return self.engine._node_val_cache.count_keys(
			self.character.name, self.name, *self.engine.time
		)

	def _get_cache(
		self, key: Stat, branch: Branch, turn: Turn, tick: Tick
	) -> Value:
		return self.engine._node_val_cache.retrieve(
			self.character.name, self.name, key, branch, turn, tick
		)

	def _set_db(
		self,
		key: Stat,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
	) -> None:
		self.engine.database.node_val_set(
			self.character.name, self.name, key, branch, turn, tick, value
		)

	def _set_cache(
		self,
		key: Stat,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
	) -> None:
		self.engine._node_val_cache.store(
			self.character.name, self.name, key, branch, turn, tick, value
		)

	def add_portal(
		self,
		other: NodeName | KeyHint | Node,
		**stats: dict[Stat | KeyHint, Value | ValueHint],
	) -> None:
		"""Connect a portal from here to another node"""
		if not isinstance(other, (Key, Node)):
			raise TypeError("Invalid node", other)
		name: NodeName
		if isinstance(other, Node):
			name = other.name
		else:
			name = NodeName(other)
		self.character.add_portal(self.name, name, **stats)

	def new_portal(
		self,
		other: KeyHint | NodeName | Node,
		**stats: dict[Stat | KeyHint, Value | ValueHint],
	) -> Portal:
		"""Connect a portal from here to another node, and return it."""
		if not isinstance(other, (Key, Node)):
			raise TypeError("Invalid node", other)
		name: NodeName
		if isinstance(other, Node):
			name = other.name
		else:
			name = NodeName(other)
		return self.character.new_portal(self.name, name, **stats)

	def add_thing(
		self,
		name: NodeName | KeyHint,
		**stats: dict[KeyHint, ValueHint] | dict[Stat, Value],
	) -> None:
		"""Make a new Thing here"""
		self.character.add_thing(name, self.name, **stats)

	def new_thing(
		self,
		name: NodeName | KeyHint,
		**stats: dict[KeyHint, ValueHint] | dict[Stat, Value],
	) -> Thing:
		"""Create a new thing, located here, and return it."""
		return self.character.new_thing(name, self.name, **stats)

	def historical(self, stat: Stat | KeyHint) -> EntityStatAlias:
		"""Return a reference to the values that a stat has had in the past.

		You can use the reference in comparisons to make a history
		query, and execute the query by calling it, or passing it to
		``self.engine.ticks_when``.

		"""
		if not isinstance(stat, Key):
			raise TypeError("Invalid stat", stat)
		return EntityStatAlias(entity=self, stat=Stat(stat))

	@property
	@abstractmethod
	def leader(self) -> Mapping: ...

	@abstractmethod
	def leaders(self) -> Iterator[AbstractCharacter]: ...


@define(eq=False)
class Edge(AbstractEntityMapping, ABC):
	"""Mapping for edge attributes"""

	__slots__ = ()

	character: DiGraph
	orig: NodeName
	dest: NodeName

	@cached_property
	def origin(self) -> Node:
		return self.character.node[self.orig]

	@cached_property
	def destination(self):
		return self.character.node[self.dest]

	@property
	def engine(self):
		return self.character.engine

	def __str__(self):
		return str(dict(self))

	def __iter__(self) -> Iterator[Stat]:
		return self.character.engine._edge_val_cache.iter_keys(
			self.character.name, self.orig, self.dest, *self.engine.time
		)

	def _cache_contains(
		self, key: Stat, branch: Branch, turn: Turn, tick: Tick
	) -> bool:
		return self.character.engine._edge_val_cache.contains_key(
			self.character.name, self.orig, self.dest, key, branch, turn, tick
		)

	def __len__(self) -> int:
		return self.engine._edge_val_cache.count_keys(
			self.character.name, self.orig, self.dest, *self.engine.time
		)

	def _get_cache(
		self, key: Stat, branch: Branch, turn: Turn, tick: Tick
	) -> Value:
		return self.engine._edge_val_cache.retrieve(
			self.character.name, self.orig, self.dest, key, branch, turn, tick
		)

	def _set_db(
		self, key: Stat, branch: Branch, turn: Turn, tick: Tick, value: Value
	) -> None:
		self.engine.database.edge_val_set(
			self.character.name,
			self.orig,
			self.dest,
			key,
			branch,
			turn,
			tick,
			value,
		)

	def _set_cache(
		self, key: Stat, branch: Branch, turn: Turn, tick: Tick, value: Value
	) -> None:
		self.engine._edge_val_cache.store(
			self.character.name,
			self.orig,
			self.dest,
			key,
			branch,
			turn,
			tick,
			value,
		)


@define(slots=True)
class AttrSignal(Signal):
	def __attrs_pre_init__(self):
		super().__init__()


@define(eq=False)
class GraphNodeMapping[_K, _V](
	CharacterMapping[_K, _V], MutableMapping[_K, _V], AttrSignal, ABC
):
	"""Mapping for nodes in a graph"""

	character: DiGraph

	@cached_property
	def engine(self) -> Engine:
		return self.character.engine

	def __iter__(self) -> Iterator[NodeName]:
		"""Iterate over the names of the nodes"""
		now = tuple(self.engine.time)
		gn = self.character.name
		nc = self.engine._nodes_cache
		for entity in nc.iter_entities(gn, *now):
			if entity in self:
				yield entity

	def __eq__(self, other: Mapping[NodeName, Node]) -> bool:
		if not isinstance(other, Mapping):
			return NotImplemented
		if self.keys() != other.keys():
			return False
		for k in self.keys():
			me = self[k]
			you = other[k]
			if hasattr(me, "unwrap") and not hasattr(me, "no_unwrap"):
				me = me.unwrap()
			if hasattr(you, "unwrap") and not hasattr(you, "no_unwrap"):
				you = you.unwrap()
			if me != you:
				return False
		else:
			return True

	def __contains__(self, node: NodeName | KeyHint) -> bool:
		"""Return whether the node exists presently"""
		return self.engine._nodes_cache.contains_entity(
			self.character.name, node, *self.engine.time
		)

	def __len__(self) -> int:
		"""How many nodes exist right now?"""
		return self.engine._nodes_cache.count_entities(
			self.character.name, *self.engine.time
		)

	def __getitem__(self, node: NodeName | KeyHint) -> Node:
		"""If the node exists at present, return it, else throw KeyError"""
		if not isinstance(node, Key):
			raise TypeError("Invalid node", node)
		node = NodeName(node)
		if node not in self:
			raise KeyError("No node", node)
		return self.engine._get_node(self.character, node)

	def __setitem__(
		self,
		node: NodeName | KeyHint,
		data: dict[Stat, Value] | dict[KeyHint, ValueHint],
	) -> None:
		"""Only accept dict-like values for assignment. These are taken to be
		dicts of node attributes, and so, a new GraphNodeMapping.Node
		is made with them, perhaps clearing out the one already there.

		"""
		if not isinstance(node, Key):
			raise TypeError("Invalid node", node)
		dikt: dict[NodeName, dict[Stat, Value]] = {}
		for node, stats in data.items():
			if not isinstance(node, Key):
				raise TypeError("Invalid node", node)
			if not isinstance(stats, dict):
				raise TypeError("Invalid stats", stats)
			diikt = dikt.setdefault(NodeName(node), {})
			for k, v in stats.items():
				if not isinstance(k, Key):
					raise TypeError("Invalid stat", k)
				stat = Stat(k)
				if v is ...:
					raise ValueError(
						"Lisien uses the ellipsis to indicate deleted items"
					)
				elif not isinstance(v, Value):
					raise TypeError("Invalid value", v)
				diikt[stat] = v
		node = NodeName(node)
		db = self.engine
		graph = self.character
		gname = graph.name
		if not db._node_exists(gname, node):
			db._exist_node(gname, node, True)
		n = db._get_node(graph, node)
		n.clear()
		n.update(dikt)

	def __delitem__(self, node: NodeName | KeyHint) -> None:
		"""Indicate that the given node no longer exists"""
		if not isinstance(node, Key):
			raise TypeError("Invalid node name", node)
		node = NodeName(node)
		if node not in self:
			raise KeyError("No such node")
		for succ in self.character.adj[node]:
			del self.character.adj[node][succ]
		for pred in self.character.pred[node]:
			del self.character.pred[node][pred]
		branch, turn, tick = self.engine._nbtt()
		self.engine.database.exist_node(
			self.character.name, node, branch, turn, tick, False
		)
		self.engine._nodes_cache.store(
			self.character.name, node, branch, turn, tick, False
		)
		key = (self.character.name, node)
		if node in self.engine._node_objs:
			del self.engine._node_objs[key]

	type _m_typ = dict[KeyHint, EllipsisType | dict[KeyHint, ValueHint]]

	@staticmethod
	def _combo(
		m: _m_typ,
		kwargs: _m_typ,
	) -> Iterator[tuple[NodeName, Value]]:
		for node, value in chain(m.items(), kwargs.items()):
			if not isinstance(node, Key):
				raise TypeError("Invalid node", node)
			noded = NodeName(node)
			if not isinstance(value, Value):
				raise TypeError("Invalid value", value)
			yield noded, value

	def update(
		self,
		m: _m_typ,
		/,
		**kwargs: _m_typ,
	):
		for node, value in self._combo(m, kwargs):
			if value is ...:
				del self[node]
			elif not isinstance(value, dict):
				raise TypeError("Invalid node update", value)
			elif node not in self:
				self[node] = value
			else:
				self[node].update(value)


@define(eq=False)
class GraphEdgeMapping[
	_ORIG: NodeName,
	_DEST: MutableMapping[_DEST, Edge] | bool,
](
	MutableMapping[_ORIG, _DEST],
	AttrSignal,
	CharacterMapping,
	ABC,
):
	"""Provides an adjacency mapping and possibly a predecessor mapping
	for a graph.

	"""

	character: DiGraph

	@cached_property
	def engine(self) -> Engine:
		return self.character.engine

	def __eq__(self, other: Mapping):
		"""Compare dictified versions of the edge mappings within me.

		As I serve custom Predecessor or Successor classes, which
		themselves serve the custom Edge class, I wouldn't normally be
		comparable to a networkx adjacency dictionary. Converting
		myself and the other argument to dicts allows the comparison
		to work anyway.

		"""
		if not isinstance(other, Mapping):
			return False
		if self.keys() != other.keys():
			return False
		for k in self.keys():
			if dict(self[k]) != dict(other[k]):
				return False
		return True

	def __iter__(self):
		return iter(self.character.node)


@define
class AbstractSuccessors(
	MutableMapping[NodeName, MutableMapping[NodeName, Edge]], ABC
):
	__slots__ = ()

	container: GraphEdgeMapping
	orig: NodeName

	@cached_property
	def character(self):
		return self.container.character

	@cached_property
	def engine(self):
		return self.container.character.engine

	@cached_property
	def _cache(self):
		return {}

	@abstractmethod
	def _order_nodes(self, node: NodeName) -> tuple[NodeName, NodeName]: ...

	def __iter__(self) -> Iterator[NodeName]:
		"""Iterate over node IDs that have an edge with my orig"""
		for that in self.engine._edges_cache.iter_successors(
			self.character.name, self.orig, *self.engine.time
		):
			if that in self:
				yield that

	def __contains__(self, dest: KeyHint | NodeName) -> bool:
		"""Is there an edge leading to ``dest`` at the moment?"""
		if not isinstance(dest, Key):
			raise TypeError("Invalid node", dest)
		orig, dest = self._order_nodes(NodeName(dest))
		return self.engine._edges_cache.has_successor(
			self.character.name, orig, dest, *self.engine.time
		)

	def __len__(self):
		"""How many nodes touch an edge shared with my orig?"""
		n = 0
		for n, _ in enumerate(self, start=1):
			pass
		return n

	def _make_edge(self, dest: NodeName) -> Edge:
		return Edge(self.character, *self._order_nodes(dest))

	def __getitem__(self, dest: KeyHint | NodeName) -> Edge:
		"""Get the edge between my orig and the given node"""
		if not isinstance(dest, Key):
			raise TypeError("Invalid node", dest)
		dest = NodeName(dest)
		if dest not in self:
			raise KeyError("No edge {}->{}".format(self.orig, dest))
		orig, dest = self._order_nodes(dest)
		return self.engine._get_edge(self.character, orig, dest)

	def __setitem__(
		self,
		dest: KeyHint | NodeName,
		value: dict[KeyHint, ValueHint] | dict[Stat, Value],
	):
		"""Set the edge between my orig and the given dest to the given
		value, a mapping.

		"""
		if not isinstance(dest, Key):
			raise TypeError("Invalid node", dest)
		real_dest = dest = NodeName(dest)
		orig, dest = self._order_nodes(dest)
		if orig not in self.character.node:
			self.character.add_node(orig)
		if dest not in self.character.node:
			self.character.add_node(dest)
		branch, turn, tick = self.engine._nbtt()
		self.engine.database.exist_edge(
			self.character.name, orig, dest, branch, turn, tick, True
		)
		self.engine._edges_cache.store(
			self.character.name, orig, dest, branch, turn, tick, True
		)
		e = self[real_dest]
		e.clear()
		e.update(value)

	def __delitem__(self, dest: KeyHint | NodeName) -> None:
		"""Remove the edge between my orig and the given dest"""
		if not isinstance(dest, Key):
			raise TypeError("Invalid node", dest)
		dest = NodeName(dest)
		branch, turn, tick = self.engine._nbtt()
		orig, dest = self._order_nodes(dest)
		self.engine.database.exist_edge(
			self.character.name, orig, dest, branch, turn, tick, False
		)
		self.engine._edges_cache.store(
			self.character.name, orig, dest, branch, turn, tick, False
		)

	def clear(self) -> None:
		"""Delete every edge with origin at my orig"""
		for dest in list(self):
			del self[dest]

	def update(
		self,
		m: dict[
			NodeName,
			EllipsisType | dict[NodeName, EllipsisType | dict[Stat, Value]],
		]
		| dict[
			KeyHint,
			EllipsisType
			| dict[KeyHint, EllipsisType | dict[KeyHint, ValueHint]],
		],
	) -> None:
		realupd: dict[
			NodeName,
			EllipsisType | dict[NodeName, EllipsisType | dict[Stat, Value]],
		] = {}
		for k0, v0 in m.items():
			if not isinstance(k0, Key):
				raise TypeError("Invalid node", k0)
			orig = NodeName(k0)  # could be dest if we're a pred map, whatev
			if v0 is ...:
				realupd[orig] = ...
				continue
			elif not isinstance(v0, dict):
				raise TypeError("Invalid update dict", v0)
			for k1, v1 in v0.items():
				if not isinstance(k1, Key):
					raise TypeError("Invalid node", k1)
				dest = NodeName(k1)
				if v1 is ...:
					if orig in realupd:
						realupd[orig][dest] = ...
					else:
						realupd[orig] = {dest: ...}
					continue
				elif not isinstance(v1, dict):
					raise TypeError("Invalid edge stats", v1)
				for k2, v2 in v1.items():
					if not isinstance(k2, Key):
						raise TypeError("Invalid stat", k2)
					stat = Stat(k2)
					if not isinstance(v2, Value):  # could be ..., that's fine
						raise TypeError("Invalid stat value", v2)
					if orig in realupd:
						if dest in realupd[orig]:
							realupd[orig][dest][stat] = v2
						else:
							realupd[orig][dest] = {stat: v2}
					else:
						realupd[orig] = {dest: {stat: v2}}
		for orig, dests in realupd.items():
			if dests is ...:
				del self[orig]
				continue
			for dest, stats in dests.items():
				if stats is ...:
					del self[orig][dest]
					continue
				for stat, val in stats.items():
					if val is ...:
						del self[orig][dest][stat]
						continue
					self[orig][dest][stat] = val


@define(repr=False)
class GraphSuccessorsMapping(
	GraphEdgeMapping[NodeName, AbstractSuccessors], ABC
):
	"""Mapping for Successors (itself a MutableMapping)"""

	__slots__ = ()

	character: DiGraph

	@define
	class Successors(AbstractSuccessors):
		__slots__ = ()

		@cached_property
		def _cache(self) -> dict[NodeName, Edge]:
			return {}

		def _order_nodes(self, dest: NodeName):
			if dest < self.orig:
				return (dest, self.orig)
			else:
				return (self.orig, dest)

		def update(
			self,
			m: dict[NodeName, EllipsisType | dict[Stat, Value]]
			| dict[KeyHint, EllipsisType | dict[KeyHint, ValueHint]],
		) -> None:
			realupd: dict[NodeName, EllipsisType | dict[Stat, Value]] = {}
			for k, v in m.items():
				if not isinstance(k, Key):
					raise TypeError("Invalid node", k)
				k = NodeName(k)
				if v is ...:
					realupd[k] = ...
					continue
				elif not isinstance(v, dict):
					raise TypeError("Invalid update dict", v)
				realv: dict[Stat, Value] = {}
				for k_, v_ in v.items():
					if not isinstance(k_, Key):
						raise TypeError("Invalid stat", k_)
					if not isinstance(v_, Value):
						raise TypeError("Invalid value", v_)
					realv[Stat(k_)] = v_
				realupd[k] = realv
			for k, v in realupd.items():
				if v is ...:
					del self[k]
				else:
					for k2, v2 in v.items():
						if v2 is ...:
							del self[k][k2]
						else:
							self[k][k2] = v2

	@cached_property
	def _cache(self) -> dict[NodeName, Successors]:
		return {}

	def __getitem__(self, orig: KeyHint | NodeName) -> Successors:
		if not isinstance(orig, Key):
			raise TypeError("Invalid node", orig)
		orig = NodeName(orig)
		if orig not in self._cache:
			self._cache[orig] = self.Successors(self, orig)
		return self._cache[orig]

	def __setitem__(
		self,
		key: KeyHint | NodeName,
		val: dict[KeyHint, dict[KeyHint, ValueHint]]
		| dict[NodeName, dict[Stat, Value]],
	):
		"""Wipe out any edges presently emanating from orig and replace them
		with those described by val

		"""
		if not isinstance(key, Key):
			raise TypeError("Invalid node", key)
		key = NodeName(key)
		if key in self:
			sucs = self[key]
			sucs.clear()
		else:
			sucs = self._cache[key] = self.Successors(self, key)
		if val:
			sucs.update(val)

	def __delitem__(self, key: KeyHint | NodeName):
		"""Wipe out edges emanating from orig"""
		self[key].clear()
		del self._cache[key]

	def __iter__(self) -> Iterator[NodeName]:
		for node in self.character.node:
			if node in self:
				yield node

	def __len__(self):
		n = 0
		for node in self.character.node:
			if node in self:
				n += 1
		return n

	def __contains__(self, key: NodeName | KeyHint) -> bool:
		return key in self.character.node

	def __repr__(self):
		cls = self.__class__
		return "<{}.{} object containing {}>".format(
			cls.__module__,
			cls.__name__,
			{
				k: {k2: dict(v2) for (k2, v2) in v.items()}
				for (k, v) in self.items()
			},
		)


@define
class DiGraphSuccessorsMapping(GraphSuccessorsMapping, ABC):
	__slots__ = ()

	@define
	class Successors(AbstractSuccessors, ABC):
		__slots__ = ()

		def _order_nodes(self, dest: NodeName) -> tuple[NodeName, NodeName]:
			return (self.orig, dest)


@define
class DiGraphPredecessorsMapping(
	MutableMapping[NodeName, MutableMapping[NodeName, Edge]], ABC
):
	"""Mapping for Predecessors instances, which map to Edges that end at
	the dest provided to this

	"""

	__slots__ = ()
	character: DiGraph

	@cached_property
	def engine(self):
		return self.character.engine

	def __contains__(self, dest: KeyHint | NodeName) -> bool:
		if not isinstance(dest, Key):
			raise TypeError("Invalid node", dest)
		dest = NodeName(dest)
		for orig in self.engine._edges_cache.iter_predecessors(
			self.character.name, dest, *self.engine.time
		):
			try:
				if self.engine._edges_cache.retrieve(
					self.character.name, orig, dest, *self.engine.time
				):
					return True
			except KeyError:
				continue
		return False

	def __getitem__(self, dest: KeyHint | NodeName) -> Predecessors:
		"""Return a Predecessors instance for edges ending at the given
		node

		"""
		if not isinstance(dest, Key):
			raise TypeError("Invalid node", dest)
		dest = NodeName(dest)
		if dest not in self.character.node:
			raise KeyError("No such node", dest)
		if dest not in self._cache:
			self._cache[dest] = self.Predecessors(self, dest)
		return self._cache[dest]

	def __setitem__(
		self,
		key: KeyHint | NodeName,
		val: dict[NodeName, EllipsisType | dict[Stat, Value]]
		| dict[KeyHint, EllipsisType | dict[KeyHint, ValueHint]],
	):
		"""Interpret ``val`` as a mapping of edges that end at ``dest``"""
		if not isinstance(key, Key):
			raise TypeError("Invalid node", key)
		dest = NodeName(key)
		if dest not in self._cache:
			self._cache[dest] = self.Predecessors(self, dest)
		preds = self._cache[key]
		preds.clear()
		preds.update(val)

	def __delitem__(self, key: KeyHint | NodeName):
		"""Delete all edges ending at ``dest``"""
		if not isinstance(key, Key):
			raise TypeError("Invalid node", key)
		key = NodeName(key)
		it = self[key]
		it.clear()
		del self._cache[key]

	def __iter__(self) -> Iterator[NodeName]:
		return iter(self.character.node)

	def __len__(self) -> int:
		return len(self.character.node)

	@define
	class Predecessors(MutableMapping[NodeName, Edge]):
		"""Mapping of Edges that end at a particular node"""

		__slots__ = ()

		container: DiGraphPredecessorsMapping
		dest: NodeName

		@cached_property
		def character(self):
			return self.container.character

		@cached_property
		def engine(self):
			return self.container.character.engine

		def __iter__(self) -> Iterator[NodeName]:
			"""Iterate over the edges that exist at the present (branch, rev)"""
			for orig in self.engine._edges_cache.iter_predecessors(
				self.character.name, self.dest, *self.engine.time
			):
				if orig in self:
					yield orig

		def __contains__(self, orig: NodeName | KeyHint) -> bool:
			"""Is there an edge from ``orig`` at the moment?"""
			if not isinstance(orig, Key):
				raise TypeError("Invalid node", orig)
			orig = NodeName(orig)
			return self.engine._edges_cache.has_predecessor(
				self.character.name, self.dest, orig, *self.engine.time
			)

		def __len__(self):
			"""How many edges exist at this rev of this branch?"""
			n = 0
			for n, _ in enumerate(self, start=1):
				pass
			return n

		def _make_edge(self, orig: NodeName) -> Edge:
			return Edge(self.character, orig, self.dest)

		def __getitem__(self, orig: KeyHint | NodeName) -> Edge:
			"""Get the edge from the given node to mine"""
			if not isinstance(orig, Key):
				raise TypeError("Invalid node", orig)
			orig = NodeName(orig)
			if orig not in self:
				raise KeyError(orig)
			return self.character.adj[orig][self.dest]

		def __setitem__(
			self,
			orig: NodeName | KeyHint,
			value: dict[Stat, Value] | dict[KeyHint, ValueHint],
		):
			"""Use ``value`` as a mapping of edge attributes, set an edge from the
			given node to mine.

			"""
			if not isinstance(orig, Key):
				raise TypeError("Invalid node", orig)
			orig = NodeName(orig)
			branch, turn, tick = self.engine._nbtt()
			try:
				e = self[orig]
				e.clear()
			except KeyError:
				self.engine.database.exist_edge(
					self.character.name,
					orig,
					self.dest,
					branch,
					turn,
					tick,
					True,
				)
				e = self._make_edge(orig)
			realupd: dict[Stat, Value] = {}
			for k, v in value.items():
				if not isinstance(k, Key):
					raise TypeError("Invalid stat", k)
				if v is ...:
					raise ValueError(
						"Lisien uses the ellipsis to indicate deleted items"
					)
				if not isinstance(v, Value):
					raise TypeError("Invalid stat value", v)
				realupd[Stat(k)] = v
			e.update(value)
			self.engine._edges_cache.store(
				self.character.name, orig, self.dest, branch, turn, tick, True
			)

		def __delitem__(self, orig: NodeName | KeyHint):
			"""Unset the existence of the edge from the given node to mine"""
			if not isinstance(orig, Key):
				raise TypeError("Invalid node", orig)
			orig = NodeName(orig)
			branch, turn, tick = self.engine._nbtt()
			self.engine.database.exist_edge(
				self.character.name, orig, self.dest, branch, turn, tick, False
			)
			self.engine._edges_cache.store(
				self.character.name, orig, self.dest, branch, turn, tick, False
			)

	@cached_property
	def _cache(self) -> dict[NodeName, Predecessors]:
		return {}


def unwrapped_dict(d) -> dict:
	ret = {}
	for k, v in d.items():
		if hasattr(v, "unwrap") and not getattr(v, "no_unwrap", False):
			ret[k] = v.unwrap()
		else:
			ret[k] = v
	return ret


class DiGraph(nx.DiGraph, ABC):
	"""A version of the networkx.DiGraph class that stores its state in a
	database.

	"""

	adj_cls: ClassVar = DiGraphSuccessorsMapping
	pred_cls: ClassVar = DiGraphPredecessorsMapping
	graph_map_cls: ClassVar = GraphMapping
	node_map_cls: ClassVar = GraphNodeMapping

	def __new__(
		cls,
		engine: AbstractEngine | None = None,
		name: CharName | None = None,
		*,
		init_rulebooks: bool = False,
	):
		return super().__new__(cls)

	def _nodes_state(self) -> dict[NodeName, dict[Stat, Value]]:
		return {
			noden: {
				k: v for (k, v) in unwrapped_dict(node).items() if k != "name"
			}
			for noden, node in self._node.items()
		}

	def _edges_state(
		self,
	) -> dict[NodeName, dict[NodeName, dict[Stat, Value]]]:
		ret = {}
		ismul = self.is_multigraph()
		orig: NodeName
		for orig, dests in self.adj.items():
			if orig not in ret:
				ret[orig] = {}
			origd = ret[orig]
			dest: NodeName
			edge: Edge
			for dest, edge in dests.items():
				if ismul:
					if dest not in origd:
						origd[dest] = edges = {}
					else:
						edges = origd[dest]
					for i, val in edge.items():
						edges[i] = unwrapped_dict(val)
				else:
					origd[dest] = unwrapped_dict(edge)
		return ret

	def _val_state(self) -> dict[Stat, Value]:
		return {
			k: v
			for (k, v) in unwrapped_dict(self.graph).items()
			if k != "name"
		}

	def __bool__(self):
		return self._name in self.engine._graph_objs

	@cached_property
	def _statmap(self):
		return self.graph_map_cls(self)

	@property
	def graph(self) -> graph_map_cls:
		return self._statmap

	@graph.setter
	def graph(self, v: dict[Stat, Value] | dict[KeyHint, ValueHint]):
		self._statmap.clear()
		self._statmap.update(v)

	@cached_property
	def _node(self) -> node_map_cls:
		return self.node_map_cls(self)

	@property
	def node(self) -> node_map_cls:
		return self._node

	@cached_property
	def _adj(self) -> adj_cls:
		return self.adj_cls(self)

	@property
	def adj(self) -> adj_cls:
		return self._adj

	edge = succ = _succ = adj

	@cached_property
	def _pred(self) -> pred_cls:
		return self.pred_cls(self)

	@property
	def pred(self) -> pred_cls:
		return self._pred

	@property
	def name(self) -> CharName:
		return self._name

	@name.setter
	def name(self, v):
		raise TypeError("graphs can't be renamed")

	def remove_node(self, n: NodeName | KeyHint) -> None:
		"""Version of remove_node that minimizes writes"""
		if n not in self._node:
			raise NetworkXError("The node %s is not in the digraph." % (n,))
		nbrs = list(self._succ[n])
		for u in nbrs:
			del self._pred[u][n]  # remove all edges n-u in digraph
		pred = list(self._pred[n])
		for u in pred:
			del self._succ[u][n]  # remove all edges n-u in digraph
		del self._node[n]

	def remove_edge(
		self, u: NodeName | KeyHint, v: NodeName | KeyHint
	) -> None:
		"""Version of remove_edge that's much like normal networkx but only
		deletes once, since the database doesn't keep separate adj and
		succ mappings

		"""
		try:
			del self.succ[u][v]
		except KeyError:
			raise NetworkXError(
				"The edge {}-{} is not in the graph.".format(u, v)
			)

	def remove_edges_from(
		self, ebunch: Iterable[tuple[NodeName | KeyHint, NodeName | KeyHint]]
	):
		"""Version of remove_edges_from that's much like normal networkx but only
		deletes once, since the database doesn't keep separate adj and
		succ mappings

		"""
		for e in ebunch:
			(u, v) = e[:2]
			if u in self.succ and v in self.succ[u]:
				del self.succ[u][v]

	@abstractmethod
	def add_edge(
		self,
		u: NodeName | KeyHint,
		v: NodeName | KeyHint,
		attr_dict: dict[Stat, Value] | dict[KeyHint, ValueHint] | None = None,
		**attr: dict[Stat, Value] | dict[KeyHint, ValueHint],
	): ...

	def add_edges_from(
		self,
		ebunch: Iterable[
			tuple[NodeName | KeyHint, NodeName | KeyHint]
			| tuple[
				NodeName | KeyHint,
				NodeName | KeyHint,
				dict[Stat | KeyHint, Value | ValueHint],
			]
		],
		attr_dict: dict[Stat | KeyHint, Value | ValueHint] | None = None,
		**attr: dict[Stat | KeyHint, Value | ValueHint],
	):
		"""Version of add_edges_from that only writes to the database once"""
		if attr_dict is None:
			attr_dict = attr
		else:
			try:
				attr_dict.update(attr)
			except AttributeError:
				raise NetworkXError("The attr_dict argument must be a dict.")
		for e in ebunch:
			ne = len(e)
			if ne == 3:
				u, v, dd = e
				assert hasattr(dd, "update")
			elif ne == 2:
				u, v = e
				dd = {}
			else:
				raise NetworkXError(
					"Edge tupse {} must be a 2-tuple or 3-tuple.".format(e)
				)
			if u not in self.node:
				self.node[u] = {}
			if v not in self.node:
				self.node[v] = {}
			datadict = self.adj.get(u, {}).get(v, {})
			datadict.update(attr_dict)
			datadict.update(dd)
			self.succ[u][v] = datadict
			assert u in self.succ
			assert v in self.succ[u]

	def clear(self):
		"""Remove all nodes and edges from the graph.

		Unlike the regular networkx implementation, this does *not*
		remove the graph's name. But all the other graph, node, and
		edge attributes go away.

		"""
		self.adj.clear()
		self.node.clear()
		self.graph.clear()

	def adjlist_inner_dict_factory(self) -> dict[NodeName, dict[Stat, Value]]:
		return {}

	def node_dict_factory(self) -> dict[Stat, Value]:
		return {}

	def add_node(
		self,
		node_for_adding: KeyHint | NodeName,
		**attr: dict[Stat, Value] | dict[KeyHint, ValueHint],
	):
		"""Version of add_node that minimizes writes"""
		if not isinstance(node_for_adding, Key):
			raise TypeError("Invalid node", node_for_adding)
		node_for_adding = NodeName(node_for_adding)
		if node_for_adding not in self._succ:
			self._succ[node_for_adding] = self.adjlist_inner_dict_factory()
			self._pred[node_for_adding] = self.adjlist_inner_dict_factory()
			self._node[node_for_adding] = self.node_dict_factory()
		self._node[node_for_adding].update(attr)


type PackSignature = Callable[
	[Key | KeyHint | EternalKey | UniversalKey | Stat | ValueHint | Value],
	bytes,
]
type UnpackSignature = Callable[[bytes], Value]
type LoadedCharWindow = dict[
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
type LoadedDict = dict[
	Literal[
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
		"graphs",
	]
	| CharName,
	list[UniversalRowType]
	| list[RulebookRowType]
	| list[RuleRowType]
	| list[CharacterRulesHandledRowType]
	| list[UnitRulesHandledRowType]
	| list[NodeRulesHandledRowType]
	| list[PortalRulesHandledRowType]
	| list[GraphRowType]
	| LoadedCharWindow,
]


class get_rando:
	"""Attribute getter for randomization functions

	Aliases functions of a randomizer, wrapped so that they won't run in
	planning mode, and will save the randomizer's state after every call.

	"""

	__slots__ = ("_getter", "_wrapfun", "_instance")
	_getter: attrgetter

	def __init__(self, attr, *attrs):
		self._getter = attrgetter(attr, *attrs)

	def __get__(self, instance, owner) -> Callable:
		if hasattr(self, "_wrapfun") and self._instance is instance:
			return self._wrapfun
		retfun: Callable = self._getter(instance)

		@wraps(retfun)
		def remembering_rando_state(*args, **kwargs):
			if instance._planning:
				raise exc.PlanError("Don't use randomization in a plan")
			ret = retfun(*args, **kwargs)
			instance.universal["rando_state"] = instance._rando.getstate()
			return ret

		self._wrapfun = remembering_rando_state
		self._instance = instance
		return remembering_rando_state


class SignalDict[_K, _V](Signal, dict[_K, _V]):
	def __setitem__(self, __key, __value):
		super().__setitem__(__key, __value)
		self.send(self, key=__key, value=__value)

	def __delitem__(self, __key):
		super().__delitem__(__key)
		self.send(self, key=__key, value=None)


class EntityAccessor(ABC):
	__slots__ = (
		"engine",
		"entity",
		"branch",
		"turn",
		"tick",
		"stat",
		"current",
		"mungers",
	)

	def __init__(
		self,
		entity: GraphMapping | Node | Edge,
		stat: Stat,
		engine: AbstractEngine | None = None,
		branch: Branch | None = None,
		turn: Turn | None = None,
		tick: Tick | None = None,
		current: bool = False,
		mungers: list[Callable] | None = None,
	):
		if engine is None:
			engine = entity.engine
		if branch is None:
			branch = engine.branch
		if turn is None:
			turn = engine.turn
		if mungers is None:
			mungers = []
		self.current = current
		self.engine = engine
		self.entity = entity
		self.stat = stat
		self.branch = branch
		self.turn = turn
		self.tick = tick
		self.mungers = mungers

	def __ne__(self, other):
		return self() != other

	def __str__(self):
		return str(self())

	def __repr__(self):
		return "EntityStatAccessor({}[{}]{}), {} mungers".format(
			self.entity,
			self.stat,
			""
			if self.current
			else ", branch={}, turn={}, tick={}".format(
				self.branch, self.turn, self.tick
			),
			len(self.mungers),
		)

	def __gt__(self, other):
		return self() > other

	def __ge__(self, other):
		return self() >= other

	def __lt__(self, other):
		return self() < other

	def __le__(self, other):
		return self() <= other

	def __eq__(self, other):
		return self() == other

	def munge(self, munger: Callable):
		return EntityStatAccessor(
			self.entity,
			self.stat,
			self.engine,
			self.branch,
			self.turn,
			self.tick,
			self.current,
			self.mungers + [munger],
		)

	def __add__(self, other):
		return self.munge(partial(add, other))

	def __sub__(self, other):
		return self.munge(partial(sub, other))

	def __mul__(self, other):
		return self.munge(partial(mul, other))

	def __rpow__(self, other, modulo=None):
		return self.munge(partial(pow, other, modulo=modulo))

	def __rdiv__(self, other):
		return self.munge(partial(truediv, other))

	def __rfloordiv__(self, other):
		return self.munge(partial(floordiv, other))

	def __rmod__(self, other):
		return self.munge(partial(mod, other))

	def __contains__(self, item):
		return item in self()

	def __getitem__(self, k):
		return self.munge(lambda x: x[k])

	@abstractmethod
	def _get_value_now(self) -> Value: ...

	def __call__(
		self,
		branch: Branch | None = None,
		turn: Turn | None = None,
		tick: Tick | None = None,
	):
		if self.current:
			res = self._get_value_now()
		else:
			time_was = tuple(self.engine.time)
			self.engine.branch = branch or self.branch
			self.engine.turn = turn if turn is not None else self.turn
			if tick is not None:
				self.engine.tick = tick
			elif self.tick is not None:
				self.engine.tick = self.tick
			res = self._get_value_now()
			self.engine.time = time_was
		for munger in self.mungers:
			res = munger(res)
		return res

	def iter_history(self, beginning: Turn, end: Turn) -> Iterator[Value]:
		"""Iterate over all the values this stat has had in the given window, inclusive."""
		# It might be useful to do this in a way that doesn't change the
		# engine's time, perhaps for thread safety
		engine = self.engine
		oldturn = engine.turn
		oldtick = engine.tick
		for turn in range(beginning, end + 1):
			engine.turn = turn
			try:
				y = self._get_value_now()
			except KeyError:
				yield Value(None)
				continue
			if hasattr(y, "unwrap"):
				y = y.unwrap()
			yield y
		engine.turn = oldturn
		engine.tick = oldtick


class UnitsAccessor(EntityAccessor):
	entity: AbstractCharacter

	def _get_value_now(self) -> dict[CharName, list[NodeName]]:
		ret = {}
		for graph in self.entity.unit:
			ret[graph] = []
			for node in self.entity.unit[graph]:
				ret[graph].append(node)
		return ret


class CharacterStatAccessor(EntityAccessor):
	entity: AbstractCharacter

	def _get_value_now(self) -> Value:
		return self.entity.stat[self.stat]


class EntityStatAccessor(EntityAccessor):
	def _get_value_now(self) -> Value:
		return self.entity[self.stat]


class FakeFuture(Future):
	"""A 'Future' that calls its function immediately and sets the result"""

	def __init__(self, func: Callable, *args, **kwargs):
		super().__init__()
		self.set_result(func(*args, **kwargs))


class AbstractBookmarkMapping(MutableMapping[KeyHint, Time], Callable):
	@abstractmethod
	def __call__(self, key: KeyHint) -> None: ...


_SEQT = TypeVar("_SEQT", bound=Sequence[ValueHint])


_EXT = TypeVar("_EXT")
_TYP = TypeVar("_TYP")


@define
class AbstractEngine(ABC):
	"""Parent class to the real Engine as well as EngineProxy.

	Implements serialization and the __getattr__ for stored methods.

	"""

	thing_cls: ClassVar[type[AbstractThing]]
	place_cls: ClassVar[type[Node]]
	portal_cls: ClassVar[type[Edge]]
	char_cls: ClassVar[type[AbstractCharacter]]

	@property
	def time(self) -> tuple[Branch, Turn, Tick]:
		return Time(self.branch, self.turn, self.tick, self)

	@time.setter
	def time(self, time: tuple[Branch, Turn, Tick]):
		branch_then, turn_then, tick_then = self.time
		branch_now, turn_now, tick_now = time
		if (branch_then, turn_then, tick_then) == (
			branch_now,
			turn_now,
			tick_now,
		):
			return
		# enforce the arrow of time, if it's in effect
		if self._forward and not self._planning:
			if branch_now != branch_then:
				raise TimeError("Can't change branches in a forward context")
			if turn_now < turn_then:
				raise TimeError(
					"Can't time travel backward in a forward context"
				)
			if turn_now > turn_then + 1:
				raise TimeError("Can't skip turns in a forward context")
		# make sure I'll end up within the revision range of the
		# destination branch

		if branch_now in self.branches():
			self._extend_branch(branch_now, turn_now, tick_now)
			self.load_at(branch_now, turn_now, tick_now)
		else:
			self._start_branch(branch_then, branch_now, turn_now, tick_now)
		self._time_warp(branch_now, turn_now, tick_now)
		self._time_signal.send(
			self,
			branch_then=branch_then,
			turn_then=turn_then,
			tick_then=tick_then,
			branch_now=branch_now,
			turn_now=turn_now,
			tick_now=tick_now,
		)

	@cached_property
	def _time_signal(self) -> Signal:
		return Signal()

	illegal_node_names: ClassVar = {
		"nodes",
		"node_val",
		"edges",
		"edge_val",
		"things",
	}
	is_proxy: ClassVar[bool]

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		try:
			self.close()
		except Exception as ex:
			if exc_val:
				raise ExceptionGroup(
					f"Multiple exceptions during {self.__class__.__name__}.__exit__",
					(
						ex,
						exc_val,
					),
				)
			raise
		finally:
			if exc_val:
				raise exc_val

	def facade(self) -> EngineFacade:
		from .facade import EngineFacade

		return EngineFacade(self)

	@abstractmethod
	def close(self) -> None: ...

	@property
	@abstractmethod
	def bookmark(self) -> AbstractBookmarkMapping: ...

	@property
	@abstractmethod
	def character(self) -> Mapping[KeyHint | CharName, char_cls]: ...

	@property
	@abstractmethod
	def eternal(self) -> MutableMapping[EternalKey, Value]: ...

	@property
	@abstractmethod
	def universal(
		self,
	) -> MutableMapping[KeyHint | UniversalKey, ValueHint]: ...

	@property
	@abstractmethod
	def rulebook(
		self,
	) -> MutableMapping[KeyHint | RulebookName, "RuleBook"]: ...

	@property
	@abstractmethod
	def rule(self) -> MutableMapping[KeyHint | RuleName, "Rule"]: ...

	@property
	@abstractmethod
	def trunk(self) -> Branch: ...

	@trunk.setter
	@abstractmethod
	def trunk(self, branch: str | Branch) -> None: ...

	@property
	@abstractmethod
	def branch(self) -> Branch: ...

	@branch.setter
	@abstractmethod
	def branch(self, branch: str | Branch) -> None: ...

	@property
	@abstractmethod
	def turn(self) -> Turn: ...

	@turn.setter
	@abstractmethod
	def turn(self, turn: int | Turn): ...

	@property
	@abstractmethod
	def tick(self) -> Tick: ...

	@tick.setter
	@abstractmethod
	def tick(self, tick: int | Tick) -> None: ...

	@property
	@abstractmethod
	def _rando(self) -> Random: ...

	@property
	@abstractmethod
	def _branches_d(
		self,
	) -> dict[Branch, tuple[Optional[Branch], Turn, Tick, Turn, Tick]]: ...

	@property
	@abstractmethod
	def logger(self) -> Logger: ...

	@abstractmethod
	def _char_exists(self, character: CharName) -> bool: ...

	@abstractmethod
	def _node_exists(self, character: CharName, node: NodeName) -> bool: ...

	@abstractmethod
	def _edge_exists(
		self, character: CharName, orig: NodeName, dest: NodeName
	) -> bool: ...

	@abstractmethod
	def _detect_kf_interval_override(self) -> bool | None: ...

	def log(self, level, msg, *args, **kwargs):
		self.logger.log(level, msg, *args, **kwargs)

	def debug(self, msg, *args, **kwargs):
		self.log(10, msg, *args, **kwargs)

	def info(self, msg, *args, **kwargs):
		self.log(20, msg, *args, **kwargs)

	def warning(self, msg, *args, **kwargs):
		self.log(30, msg, *args, **kwargs)

	def error(self, msg, *args, **kwargs):
		self.log(40, msg, *args, **kwargs)

	def critical(self, msg, *args, **kwargs):
		self.log(50, msg, *args, **kwargs)

	def is_ancestor_of(self, parent: Branch, child: Branch) -> bool:
		"""Return whether ``child`` is a branch descended from ``parent``

		At any remove.

		"""
		branches = self.branches()
		if parent not in branches:
			raise ValueError("Not a branch", parent)
		if child not in branches:
			raise ValueError("Not a branch", child)
		if parent is None or parent == child or parent == self.trunk:
			return True
		if child == self.trunk:
			return False
		if self.branch_parent(child) == parent:
			return True
		return self.is_ancestor_of(parent, self.branch_parent(child))

	@cached_property
	def pack(self) -> Callable[[ValueHint | Value], bytes]:
		try:
			from msgpack import Packer

			if Packer.__module__.endswith("cmsgpack"):
				return self._msgpack_packer()
			else:
				return self._pack_with_umsgpack
		except ImportError:
			return self._pack_with_umsgpack

	def _pack_with_umsgpack(self, obj: ValueHint | Value) -> bytes:
		from umsgpack import Ext, packb

		from .db import AbstractDatabaseConnector, PythonDatabaseConnector

		if inspect.isclass(obj) and issubclass(obj, AbstractDatabaseConnector):
			return packb(
				Ext(
					MsgpackExtensionType.database.value,
					self._pack_with_umsgpack({"class": obj.__name__}),
				)
			)
		elif (
			isinstance(obj, partial)
			and inspect.isclass(obj.func)
			and issubclass(obj.func, AbstractDatabaseConnector)
		):
			return packb(
				Ext(
					MsgpackExtensionType.database.value,
					self._pack_with_umsgpack(
						{"class": obj.func.__name__, **obj.keywords}
					),
				)
			)
		elif isinstance(obj, Time):
			return packb(tuple(obj))
		elif isinstance(obj, (list, ListWrapper)):
			return concat_list(list(map(self._pack_with_umsgpack, obj)))
		elif isinstance(obj, tuple):
			return packb(
				Ext(
					MsgpackExtensionType.tuple.value,
					concat_list(list(map(self._pack_with_umsgpack, obj))),
				)
			)
		elif (
			isinstance(obj, Mapping)
			and type(obj) not in self._umsgpack_pack_handlers
		):
			return concat_d(
				{
					self._pack_with_umsgpack(k): self._pack_with_umsgpack(v)
					for (k, v) in obj.items()
				}
			)
		return packb(obj, ext_handlers=self._umsgpack_pack_handlers)

	def _msgpack_packer(self) -> Callable[[ValueHint | value], bytes]:
		import msgpack

		from .db import AbstractDatabaseConnector
		from .facade import (
			CharacterFacade,
			FacadePlace,
			FacadePortal,
			FacadeThing,
		)

		handlers = self._msgpack_pack_handlers

		def pack_handler(obj):
			if isinstance(obj, Exception):
				typ = Exception
			elif isinstance(obj, Enum):
				typ = Enum
			elif isinstance(obj, type) and issubclass(
				obj, AbstractDatabaseConnector
			):
				typ = obj
			elif isinstance(obj, partial) and issubclass(
				obj.func, AbstractDatabaseConnector
			):
				typ = obj.func
			else:
				typ = type(obj)
			if typ in handlers:
				return handlers[typ](obj)
			elif isinstance(obj, DiGraph):
				return msgpack.ExtType(
					MsgpackExtensionType.graph.value, packer(obj.name)
				)
			elif isinstance(obj, AbstractThing):
				return msgpack.ExtType(
					MsgpackExtensionType.thing.value,
					packer([obj.character.name, obj.name]),
				)
			elif isinstance(obj, Node):
				return msgpack.ExtType(
					MsgpackExtensionType.place.value,
					packer([obj.character.name, obj.name]),
				)
			elif isinstance(obj, Edge):
				return msgpack.ExtType(
					MsgpackExtensionType.portal.value,
					packer(
						[
							obj.character.name,
							obj.orig,
							obj.dest,
						]
					),
				)
			elif isinstance(obj, Set):
				return self._pack_set(msgpack.ExtType, obj)
			elif isinstance(obj, Mapping):
				return dict(obj)
			elif isinstance(obj, (list, ListWrapper)):
				return list(obj)
			elif isinstance(obj, Time):
				return obj()
			raise TypeError("Can't pack {}".format(typ))

		packer = partial(
			msgpack.packb,
			default=pack_handler,
			strict_types=True,
			use_bin_type=True,
		)
		return packer

	@cached_property
	def _unpack_handlers(self):
		from .facade import (
			CharacterFacade,
			FacadePlace,
			FacadePortal,
			FacadePortalSuccessors,
			FacadeThing,
		)

		char_cls = self.char_cls
		place_cls = self.place_cls
		portal_cls = self.portal_cls
		thing_cls = self.thing_cls
		excs = {}
		for module in (builtins, nx.exception, exc):
			for name, cls in inspect.getmembers(module):
				if inspect.isclass(cls) and issubclass(cls, Exception):
					excs[name] = cls

		def unpack_graph(ext: bytes) -> nx.Graph:
			if hasattr(ext, "data"):  # umsgpack.Ext
				ext = ext.data
			cls, node, adj, graph = self.unpack(ext)
			blank = {
				"Graph": nx.Graph,
				"DiGraph": nx.DiGraph,
				"MultiGraph": nx.MultiGraph,
				"MultiDiGraph": nx.MultiDiGraph,
			}[cls]()
			blank._node = node
			blank._adj = adj
			blank.graph = graph
			return blank

		def unpack_exception(ext: bytes) -> Exception:
			exctyp, excargs, tb = self.unpack(getattr(ext, "data", ext))
			if exctyp not in excs:
				return Exception(exctyp, *excargs)
			ret = excs[exctyp](*excargs)
			if tb is not None:
				ret.__traceback__ = Traceback.from_dict(tb).to_traceback()
			return ret

		def unpack_database_connector(ext: bytes) -> AbstractDatabaseConnector:
			data = self.unpack(getattr(ext, "data", ext))
			if not isinstance(data, dict):
				raise TypeError("Invalid database connector data", data)
			clsn = data.pop("class")
			if clsn == "PythonDatabaseConnector":
				from .db import PythonDatabaseConnector

				if "load_me" in data:
					connector = PythonDatabaseConnector(self)
					connector.load_everything(data["load_me"])
				else:
					connector = PythonDatabaseConnector
				return connector
			elif clsn == "SQLAlchemyDatabaseConnector":
				from .sql import SQLAlchemyDatabaseConnector

				return partial(SQLAlchemyDatabaseConnector, **data)
			elif clsn == "ParquetDatabaseConnector":
				from .pqdb import ParquetDatabaseConnector

				return partial(ParquetDatabaseConnector, **data)
			else:
				raise ValueError("Unsupported database", clsn)

		def unpack_char(ext: bytes) -> char_cls:
			charn = self.unpack(getattr(ext, "data", ext))
			if self._char_exists(charn):
				return self.character[charn]
			return char_cls(self, CharName(Key(charn)), init_rulebooks=False)

		def unpack_place(ext: bytes) -> place_cls:
			charn, placen = self.unpack(getattr(ext, "data", ext))
			if self._node_exists(charn, placen):
				return self.character[charn].place[placen]
			return place_cls(
				char_cls(self, CharName(Key(charn)), init_rulebooks=False),
				NodeName(Key(placen)),
			)

		def unpack_thing(ext: bytes) -> thing_cls:
			charn, thingn = self.unpack(getattr(ext, "data", ext))
			if self._node_exists(charn, thingn):
				return self.character[charn].thing[thingn]
			return thing_cls(
				char_cls(self, CharName(Key(charn)), init_rulebooks=False),
				NodeName(Key(thingn)),
			)

		def unpack_portal(ext: bytes) -> portal_cls:
			charn, orign, destn = self.unpack(getattr(ext, "data", ext))
			if self._edge_exists(charn, orign, destn):
				return self.character[charn].portal[orign][destn]
			return portal_cls(
				char_cls(self, CharName(Key(charn)), init_rulebooks=False),
				NodeName(Key(orign)),
				NodeName(Key(destn)),
			)

		def unpack_seq(t: type[_SEQT], ext: bytes) -> _SEQT:
			unpacked = self.unpack(getattr(ext, "data", ext))
			if not isinstance(unpacked, list):
				raise TypeError("Tried to unpack", type(unpacked), t)
			return t(unpacked)

		def unpack_func(store: AbstractFunctionStore, ext: bytes) -> Callable:
			unpacked = self.unpack(getattr(ext, "data", ext))
			if not isinstance(unpacked, str):
				raise TypeError("Tried to unpack as func", type(unpacked))
			return getattr(store, unpacked)

		def unpack_path(ext: bytes):
			unpacked = self.unpack(getattr(ext, "data", ext))
			return Path(unpacked)

		def unpack_enum(ext: bytes) -> Enum:
			enum_typ_str, enum_name = self.unpack(getattr(ext, "data", ext))
			return getattr(getattr(enum, enum_typ_str), enum_name)

		return {
			MsgpackExtensionType.enum.value: unpack_enum,
			MsgpackExtensionType.path.value: unpack_path,
			MsgpackExtensionType.database.value: unpack_database_connector,
			MsgpackExtensionType.ellipsis.value: lambda _: ...,
			MsgpackExtensionType.graph.value: unpack_graph,
			MsgpackExtensionType.character.value: unpack_char,
			MsgpackExtensionType.place.value: unpack_place,
			MsgpackExtensionType.thing.value: unpack_thing,
			MsgpackExtensionType.portal.value: unpack_portal,
			MsgpackExtensionType.tuple.value: partial(unpack_seq, tuple),
			MsgpackExtensionType.frozenset.value: partial(
				unpack_seq, OrderlyFrozenSet
			),
			MsgpackExtensionType.set.value: partial(unpack_seq, OrderlySet),
			MsgpackExtensionType.function.value: partial(
				unpack_func, self.function
			),
			MsgpackExtensionType.method.value: partial(
				unpack_func, self.method
			),
			MsgpackExtensionType.trigger.value: partial(
				unpack_func, self.trigger
			),
			MsgpackExtensionType.prereq.value: partial(
				unpack_func, self.prereq
			),
			MsgpackExtensionType.action.value: partial(
				unpack_func, self.action
			),
			MsgpackExtensionType.exception.value: unpack_exception,
		}

	@cached_property
	def unpack(
		self,
	) -> Callable[[bytes], Value]:
		try:
			import msgpack

			if not msgpack.unpackb.__module__.endswith("cmsgpack"):
				from umsgpack import unpackb

				return partial(unpackb, ext_handlers=self._unpack_handlers)
		except ImportError:
			from umsgpack import unpackb

			return partial(unpackb, ext_handlers=self._unpack_handlers)

		def unpack_handler(
			code: MsgpackExtensionType, data: bytes
		) -> Value | Exception | msgpack.ExtType:
			if code in self._unpack_handlers:
				return self._unpack_handlers[code](data)
			return msgpack.ExtType(code, data)

		def unpacker(b: bytes):
			the_unpacker = msgpack.Unpacker(
				ext_hook=unpack_handler, raw=False, strict_map_key=False
			)
			the_unpacker.feed(b)
			# Deliberately only returning the initial item;
			# others are likely to be null bytes as a result of the
			# way browsers work, and anyway if you really want more
			# you can just pack a list
			return the_unpacker.unpack()

		return unpacker

	def _pack_facade_thing(self, cls, thing: FacadeThing):
		patch = dict(thing._patch)
		if "location" in thing._patch:
			locn = patch.pop("location")
		else:
			locn = thing["location"]
		return cls(
			MsgpackExtensionType.facade_thing.value,
			self.pack([thing.character.name, thing.name, locn, patch]),
		)

	def _pack_set(self, ext: type[_EXT], s: Set) -> _EXT:
		return ext(MsgpackExtensionType.set.value, self.pack(list(s)))

	def _generate_pack_handlers(
		self, ext: type[_EXT]
	) -> dict[type[_TYP], Callable[[_TYP], _EXT]]:
		from .collections import CompositeDict
		from .db import AbstractDatabaseConnector, PythonDatabaseConnector
		from .facade import (
			CharacterFacade,
			FacadePlace,
			FacadePortal,
			FacadeThing,
		)
		from .wrap import (
			DictWrapper,
			ListWrapper,
			SetWrapper,
			SubDictWrapper,
			SubListWrapper,
			SubSetWrapper,
		)

		pack_set = self._pack_set

		def pack_path(p: Path) -> ext:
			return ext(MsgpackExtensionType.path.value, self.pack(str(p)))

		def pack_char(obj: self.char_cls) -> ext:
			return ext(
				MsgpackExtensionType.character.value, self.pack(obj.name)
			)

		def pack_node(
			code: MsgpackExtensionType, obj: self.thing_cls | self.place_cls
		) -> ext:
			return ext(code.value, self.pack([obj.character.name, obj.name]))

		pack_thing = partial(pack_node, MsgpackExtensionType.thing)
		pack_place = partial(pack_node, MsgpackExtensionType.place)

		def pack_portal(obj: self.portal_cls) -> ext:
			return ext(
				MsgpackExtensionType.portal.value,
				self.pack(
					[obj.character.name, obj.origin.name, obj.destination.name]
				),
			)

		def pack_graph(typstr: str, obj: nx.Graph) -> ext:
			return ext(
				MsgpackExtensionType.graph.value,
				self.pack(
					[
						typstr,
						{k: dict(v) for (k, v) in obj.nodes.items()},
						{uv: dict(w) for (uv, w) in obj.adj.items()},
						{k: v for (k, v) in obj.graph.items()},
					]
				),
			)

		def pack_database_connector(
			db: AbstractDatabaseConnector | partial[AbstractDatabaseConnector],
		) -> ext:
			if isinstance(db, partial):
				data = {
					"class": db.func.__name__,
				}
				data.update(db.keywords)
			elif isinstance(db, PythonDatabaseConnector):
				data = {
					"class": "PythonDatabaseConnector",
					"load_me": db.dump_everything(),
				}
			else:
				data = {"class": db.__name__}
			return ext(
				MsgpackExtensionType.database.value,
				self.pack(data),
			)

		def pack_exception(exc: Exception) -> ext:
			return ext(
				MsgpackExtensionType.exception.value,
				self.pack(
					[
						exc.__class__.__name__,
						exc.args,
						Traceback(exc.__traceback__).to_dict()
						if getattr(exc, "__traceback__", None)
						else None,
					]
				),
			)

		def pack_enum(en: Enum) -> ext:
			return ext(
				MsgpackExtensionType.enum.value,
				self.pack([en.__class__.__name__, en.name]),
			)

		ret = {
			Enum: pack_enum,
			PythonDatabaseConnector: pack_database_connector,
			Path: pack_path,
			WindowsPath: pack_path,
			PosixPath: pack_path,
			type(...): lambda _: ext(MsgpackExtensionType.ellipsis.value, b""),
			self.char_cls: pack_char,
			CharacterFacade: pack_char,
			self.place_cls: pack_place,
			FacadePlace: pack_place,
			self.thing_cls: pack_thing,
			FacadeThing: pack_thing,
			self.portal_cls: pack_portal,
			FacadePortal: pack_portal,
			nx.Graph: partial(pack_graph, "Graph"),
			nx.DiGraph: partial(pack_graph, "DiGraph"),
			nx.MultiGraph: partial(pack_graph, "MultiGraph"),
			nx.MultiDiGraph: partial(pack_graph, "MultiDiGraph"),
			tuple: lambda tup: ext(
				MsgpackExtensionType.tuple.value, self.pack(list(tup))
			),
			frozenset: lambda frozs: ext(
				MsgpackExtensionType.frozenset.value, self.pack(list(frozs))
			),
			set: partial(self._pack_set, ext),
			FunctionType: lambda func: ext(
				getattr(MsgpackExtensionType, func.__module__).value,
				self.pack(func.__name__),
			),
			MethodType: lambda meth: ext(
				MsgpackExtensionType.method.value, self.pack(meth.__name__)
			),
		}
		for module in (builtins, nx.exception, exc):
			for name, cls in inspect.getmembers(module):
				if inspect.isclass(cls) and issubclass(cls, Exception):
					ret[cls] = pack_exception
		try:
			from .sql import SQLAlchemyDatabaseConnector

			ret[SQLAlchemyDatabaseConnector] = pack_database_connector
		except ImportError:
			pass
		try:
			from .pqdb import ParquetDatabaseConnector

			ret[ParquetDatabaseConnector] = pack_database_connector
		except ImportError:
			pass
		return ret

	@cached_property
	def _msgpack_pack_handlers(self):
		from msgpack import ExtType

		return self._generate_pack_handlers(ExtType)

	@cached_property
	def _umsgpack_pack_handlers(self):
		from umsgpack import Ext

		return self._generate_pack_handlers(Ext)

	@abstractmethod
	def _get_node(self, char: DiGraph | CharName, node: NodeName) -> Node: ...

	def branches(self) -> KeysView[Branch]:
		return self._branches_d.keys()

	def branch_parent(self, branch: Branch | None) -> Branch | None:
		if branch is None or branch not in self._branches_d:
			return None
		return self._branches_d[branch][0]

	def _branch_start(self, branch: Branch | None = None) -> LinearTime:
		if branch is None:
			branch = self.branch
		_, turn, tick, _, _ = self._branches_d[branch]
		return LinearTime(turn, tick)

	def _branch_end(self, branch: Branch | None = None) -> LinearTime:
		if branch is None:
			branch = self.branch
		_, _, _, turn, tick = self._branches_d[branch]
		return LinearTime(turn, tick)

	@abstractmethod
	def _start_branch(
		self, parent: Branch, branch: Branch, turn: Turn, tick: Tick
	) -> None: ...

	@abstractmethod
	def _set_btt(self, branch: Branch, turn: Turn, tick: Tick) -> None: ...

	@abstractmethod
	def _time_warp(self, branch: Branch, turn: Turn, tick: Tick) -> None: ...

	@abstractmethod
	def _extend_branch(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> None: ...

	@abstractmethod
	def load_at(
		self, branch: str | Branch, turn: Turn, tick: Tick
	) -> None: ...

	def branch_start_turn(self, branch: str | Branch | None = None) -> Turn:
		return self._branch_start(Branch(branch))[0]

	def branch_start_tick(self, branch: str | Branch | None = None) -> Tick:
		return self._branch_start(Branch(branch))[1]

	def branch_end_turn(self, branch: str | Branch | None = None) -> Turn:
		return self._branch_end(Branch(branch))[0]

	def branch_end_tick(self, branch: str | Branch | None = None) -> Tick:
		return self._branch_end(Branch(branch))[1]

	@abstractmethod
	def turn_end(
		self,
		branch: str | Branch | None = None,
		turn: int | Turn | None = None,
	) -> Tick: ...

	@abstractmethod
	def turn_end_plan(
		self,
		branch: str | Branch | None = None,
		turn: int | Turn | None = None,
	) -> Tick: ...

	@abstractmethod
	def add_character(
		self,
		name: KeyHint | CharName,
		data: nx.Graph | DiGraph | None = None,
		layout: bool = False,
		node: NodeValDict | None = None,
		edge: EdgeValDict | None = None,
		**kwargs,
	): ...

	def new_character(
		self,
		name: KeyHint | CharName,
		data: nx.Graph | DiGraph | None = None,
		layout: bool = False,
		node: NodeValDict | None = None,
		edge: EdgeValDict | None = None,
		**kwargs,
	):
		self.add_character(name, data)
		return self.character[name]

	@abstractmethod
	def export(
		self,
		name: str | None,
		path: Path | os.PathLike[str] | None = None,
		indent: bool = True,
	) -> Path: ...

	@classmethod
	@abstractmethod
	def from_archive(
		cls,
		path: Path | os.PathLike[str],
		prefix: Path | os.PathLike[str] | None = ".",
		**kwargs,
	) -> AbstractEngine: ...

	def coin_flip(self) -> bool:
		"""Return True or False with equal probability."""
		return self.choice((True, False))

	def die_roll(self, d: Annotated[int, Ge(1)]) -> int:
		"""Roll a die with ``d`` faces. Return the result."""
		return self.randint(1, d)

	def dice(
		self, n: Annotated[int, Ge(1)], d: Annotated[int, Ge(1)]
	) -> Iterable[int]:
		"""Roll ``n`` dice with ``d`` faces, and yield the results.

		This is an iterator. You'll get the result of each die in
		succession.

		"""
		for i in range(0, n):
			yield self.die_roll(d)

	def dice_check(
		self,
		n: Annotated[int, Ge(1)],
		d: Annotated[int, Ge(1)],
		target: Annotated[int, Ge(0)],
		comparator: str | Callable[[int, int], bool] = "<=",
	) -> bool:
		"""Roll ``n`` dice with ``d`` sides, sum them, and compare

		If ``comparator`` is provided, use it instead of the default <=.
		You may use a string like '<' or '>='.

		"""
		from operator import eq, ge, gt, le, lt, ne

		comps: dict[str, Callable] = {
			">": gt,
			"<": lt,
			">=": ge,
			"<=": le,
			"=": eq,
			"==": eq,
			"!=": ne,
		}
		if not callable(comparator):
			comparator = comps[comparator]
		return comparator(sum(self.dice(n, d)), target)

	def chance(self, f: Annotated[float, Ge(0.0), Le(1.0)]) -> bool:
		"""Return True or False with a given unit probability

		Supply a float between 0.0 and 1.0 to express the probability--
		or use `percent_chance`

		"""
		if f <= 0.0:
			return False
		if f >= 1.0:
			return True
		return f > self._rando.random()

	def percent_chance(
		self,
		pct: Annotated[int, Ge(0), Le(100)]
		| Annotated[float, Ge(0.0), Le(100.0)],
	) -> bool:
		"""Return True or False with a given percentile probability

		Values not between 0 and 100 are treated as though they
		were 0 or 100, whichever is nearer.

		"""
		return self.chance(pct / 100)

	@abstractmethod
	def plan(self) -> ContextManager: ...

	betavariate = get_rando("_rando.betavariate")
	choice = get_rando("_rando.choice")
	expovariate = get_rando("_rando.expovariate")
	gammavariate = get_rando("_rando.gammavariate")
	gauss = get_rando("_rando.gauss")
	getrandbits = get_rando("_rando.getrandbits")
	lognormvariate = get_rando("_rando.lognormvariate")
	normalvariate = get_rando("_rando.normalvariate")
	paretovariate = get_rando("_rando.paretovariate")
	randint = get_rando("_rando.randint")
	random = get_rando("_rando.random")
	randrange = get_rando("_rando.randrange")
	sample = get_rando("_rando.sample")
	shuffle = get_rando("_rando.shuffle")
	triangular = get_rando("_rando.triangular")
	uniform = get_rando("_rando.uniform")
	vonmisesvariate = get_rando("_rando.vonmisesvariate")
	weibullvariate = get_rando("_rando.weibullvariate")


class BaseMutableCharacterMapping[_KT, _VT](
	MutableMapping[_KT, _VT], CharacterMapping, ABC
): ...


@define(eq=False)
class AbstractCharacter(DiGraph, ABC):
	"""The Character API, with all requisite mappings and graph generators.

	Mappings resemble those of a NetworkX digraph:

	* ``thing`` and ``place`` are subsets of ``node``
	* ``edge``, ``adj``, and ``succ`` are aliases of ``portal``
	* ``pred`` is an alias to ``preportal``
	* ``stat`` is a dict-like mapping of data that changes over game-time,
	to be used in place of graph attributes

	"""

	engine: AbstractEngine
	_name: CharName
	_init_rulebooks: bool = field(default=False, kw_only=True)

	no_unwrap: ClassVar = True

	@property
	def name(self):
		return self._name

	@staticmethod
	def is_directed():
		return True

	@staticmethod
	def is_multigraph():
		return False

	@abstractmethod
	def add_place(self, name: KeyHint | NodeName, **kwargs):
		pass

	def add_node(self, name: KeyHint | NodeName, **kwargs):
		self.add_place(name, **kwargs)

	@abstractmethod
	def add_places_from(self, seq: Iterable, **attrs):
		pass

	def add_nodes_from(self, seq: Iterable, **attrs):
		self.add_places_from(seq, **attrs)

	def new_place(
		self,
		name: KeyHint | NodeName,
		**kwargs: dict[Stat, Value] | dict[KeyHint, ValueHint],
	):
		"""Add a Place and return it.

		If there's already a Place by that name, put a number on the end.

		"""
		if not isinstance(name, Key):
			raise TypeError("Invalid node name", name)
		name = NodeName(name)
		if name not in self.node:
			self.add_place(name, **kwargs)
			return self.place[name]
		if isinstance(name, str):
			n = 0
			while name + str(n) in self.node:
				n += 1
			self.add_place(name + str(n), **kwargs)
			return self.place[name]
		raise KeyError("Already have a node named {}".format(name))

	def new_node(
		self,
		name: KeyHint | NodeName,
		**kwargs: dict[Stat, Value] | dict[KeyHint, ValueHint],
	):
		return self.new_place(name, **kwargs)

	@abstractmethod
	def add_thing(
		self, name: KeyHint | NodeName, location: KeyHint | NodeName, **kwargs
	):
		pass

	@abstractmethod
	def add_things_from(self, seq: Iterable, **attrs):
		pass

	def new_thing(
		self,
		name: KeyHint | NodeName,
		location: KeyHint | NodeName,
		**kwargs: dict[Stat, Value] | dict[KeyHint, ValueHint],
	):
		"""Add a Thing and return it.

		If there's already a Thing by that name, put a number on the end.

		"""
		if not isinstance(name, Key):
			raise TypeError("Invalid thing name", name)
		name = NodeName(name)
		if not isinstance(location, Key):
			raise TypeError("Invalid location name", location)
		location = NodeName(location)
		if name not in self.node:
			self.add_thing(name, location, **kwargs)
			return self.thing[name]
		if isinstance(name, str):
			if name in self.node:
				n = 0
				while name + str(n) in self.node:
					n += 1
				name = name + str(n)
			self.add_thing(name, location, **kwargs)
			return self.thing[name]
		raise KeyError("Already have a thing named {}".format(name))

	@abstractmethod
	def place2thing(
		self, place: KeyHint | NodeName, location: KeyHint | NodeName
	) -> None: ...

	@abstractmethod
	def thing2place(self, thing: KeyHint | NodeName) -> None: ...

	def remove_node(self, node: KeyHint | NodeName):
		if not isinstance(node, Key):
			raise TypeError("Invalid node", node)
		node = NodeName(node)
		if node in self.node:
			self.node[node].delete()

	def remove_nodes_from(self, nodes: Iterable[KeyHint | NodeName]):
		for node in nodes:
			if not isinstance(node, Key):
				raise TypeError("Invalid node", node)
			node = NodeName(node)
			if node in self.node:
				self.node[node].delete()

	@abstractmethod
	def add_portal(
		self,
		orig: KeyHint | NodeName,
		dest: KeyHint | NodeName,
		**kwargs: dict[Stat, Value] | dict[KeyHint, ValueHint],
	):
		pass

	def add_edge(
		self,
		orig: KeyHint | NodeName,
		dest: KeyHint | NodeName,
		**kwargs: dict[Stat, Value] | dict[KeyHint, ValueHint],
	):
		self.add_portal(orig, dest, **kwargs)

	def new_portal(
		self,
		orig: KeyHint | NodeName,
		dest: KeyHint | NodeName,
		**kwargs: dict[Stat, Value] | dict[KeyHint, ValueHint],
	):
		self.add_portal(orig, dest, **kwargs)
		return self.portal[orig][dest]

	@abstractmethod
	def add_portals_from(
		self,
		seq: Iterable,
		**attrs: dict[Stat, Value] | dict[KeyHint, ValueHint],
	):
		pass

	def add_edges_from(self, seq: Iterable, **attrs):
		self.add_portals_from(seq, **attrs)

	@abstractmethod
	def remove_portal(
		self, origin: KeyHint | NodeName, destination: KeyHint | NodeName
	):
		pass

	def remove_portals_from(
		self, seq: Iterable[tuple[KeyHint | NodeName, KeyHint | NodeName]]
	):
		for orig, dest in seq:
			if not isinstance(orig, Key):
				raise TypeError("Invalid node", orig)
			orig = NodeName(orig)
			if not isinstance(dest, Key):
				raise TypeError("Invalid node", dest)
			dest = NodeName(dest)
			del self.portal[orig][dest]

	def remove_edges_from(
		self, seq: Iterable[tuple[KeyHint | NodeName, KeyHint | NodeName]]
	):
		self.remove_portals_from(seq)

	@abstractmethod
	def remove_place(self, place: KeyHint | NodeName):
		pass

	def remove_places_from(self, seq: Iterable[KeyHint | NodeName]):
		for place in seq:
			if not isinstance(place, Key):
				raise TypeError("Invalid node", place)
			self.remove_place(place)

	@abstractmethod
	def remove_thing(self, thing: KeyHint | NodeName) -> None:
		pass

	def remove_things_from(self, seq: Iterable[KeyHint | NodeName]) -> None:
		for thing in seq:
			self.remove_thing(thing)

	@abstractmethod
	def add_unit(
		self,
		a: KeyHint | CharName | Node,
		b: Optional[KeyHint | NodeName] = None,
	) -> None:
		pass

	@abstractmethod
	def remove_unit(
		self,
		a: KeyHint | CharName | Node,
		b: Optional[KeyHint | NodeName] = None,
	) -> None:
		pass

	def __iter__(self):
		return iter(self.node)

	def __len__(self):
		return len(self.node)

	def __eq__(self, other):
		return isinstance(other, AbstractCharacter) and self.name == other.name

	def __bool__(self):
		try:
			return self.name in self.engine.character
		except AttributeError:
			return False  # we can't "really exist" when we've no engine

	def __contains__(self, k: KeyHint | NodeName):
		return k in self.node

	def __getitem__(self, k: KeyHint | NodeName):
		return self.adj[k]

	ThingMapping: ClassVar[type[BaseMutableCharacterMapping[NodeName, Node]]]

	@cached_property
	def thing(self) -> ThingMapping:
		return self.ThingMapping(self)

	PlaceMapping: ClassVar[type[BaseMutableCharacterMapping[NodeName, Node]]]

	@cached_property
	def place(self) -> PlaceMapping:
		return self.PlaceMapping(self)

	ThingPlaceMapping: ClassVar[
		type[BaseMutableCharacterMapping[NodeName, Node]]
	]

	@cached_property
	def _node(self) -> ThingPlaceMapping:
		return self.ThingPlaceMapping(self)

	@property
	def node(self) -> ThingPlaceMapping:
		return self._node

	@property
	def nodes(self) -> ThingPlaceMapping:
		return self._node

	PortalSuccessorsMapping: ClassVar[
		type[
			BaseMutableCharacterMapping[
				NodeName, MutableMapping[NodeName, Edge]
			]
		]
	]

	@cached_property
	def _succ(self) -> PortalSuccessorsMapping:
		return self.PortalSuccessorsMapping(self)

	@property
	def portal(self):
		return self._succ

	@property
	def _adj(self):
		return self._succ

	@property
	def adj(self):
		return self._succ

	@property
	def edge(self):
		return self._succ

	PortalPredecessorsMapping: ClassVar[
		type[
			BaseMutableCharacterMapping[
				NodeName, MutableMapping[NodeName, Edge]
			]
		]
	]

	@cached_property
	def _pred(self) -> PortalPredecessorsMapping:
		return self.PortalPredecessorsMapping(self)

	@property
	def preportal(self):
		return self._pred

	@property
	def pred(self):
		return self._pred

	UnitGraphMapping: ClassVar[
		type[
			BaseMutableCharacterMapping[
				CharName, MutableMapping[NodeName, Node]
			]
		]
	]

	@cached_property
	def unit(self) -> UnitGraphMapping:
		return self.UnitGraphMapping(self)

	@property
	def stat(self):
		return self.graph

	def units(self):
		for units in self.unit.values():
			yield from units.values()

	def historical(self, stat: Stat):
		return EntityStatAlias(entity=self.stat, stat=stat)

	def do(self, func: Callable | str, *args, **kwargs) -> Self:
		"""Apply the function to myself, and return myself.

		Look up the function in the method store if needed. Pass it any
		arguments given, keyword or positional.

		Useful chiefly when chaining.

		"""
		if not callable(func):
			func = getattr(self.engine.method, func)
		func(self, *args, **kwargs)
		return self

	def copy_from(self, g: AbstractCharacter) -> Self:
		"""Copy all nodes and edges from the given graph into this.

		Return myself.

		"""
		renamed = {}
		for k in g.nodes:
			ok = k
			if k in self.place:
				n = 0
				while k in self.place:
					k = ok + (n,) if isinstance(ok, tuple) else (ok, n)
					n += 1
			renamed[ok] = k
			self.place[k] = g.nodes[k]
		if type(g) is nx.MultiDiGraph:
			g = nx.DiGraph(g)
		elif type(g) is nx.MultiGraph:
			g = nx.Graph(g)
		if type(g) is nx.DiGraph:
			for u, v in g.edges:
				self.edge[renamed[u]][renamed[v]] = g.adj[u][v]
		else:
			assert type(g) is nx.Graph
			for u, v, d in g.edges.data():
				self.add_portal(renamed[u], renamed[v], **d)
				self.add_portal(renamed[v], renamed[u], **d)
		return self

	def become(self, g: AbstractCharacter) -> Self:
		"""Erase all my nodes and edges. Replace them with a copy of the graph
		provided.

		Return myself.

		"""
		self.clear()
		self.place.update(g.nodes)
		self.adj.update(g.adj)
		return self

	def clear(self) -> None:
		self.node.clear()
		self.portal.clear()
		self.stat.clear()

	def _lookup_comparator(self, comparator: Callable | str) -> Callable:
		if callable(comparator):
			return comparator
		ops = {"ge": ge, "gt": gt, "le": le, "lt": lt, "eq": eq}
		if comparator in ops:
			return ops[comparator]
		return getattr(self.engine.function, comparator)

	def cull_nodes(
		self,
		stat: Stat,
		threshold: float = 0.5,
		comparator: Callable | str = ge,
	) -> Self:
		"""Delete nodes whose stat >= ``threshold`` (default 0.5).

		Optional argument ``comparator`` will replace >= as the test
		for whether to cull. You can use the name of a stored function.

		"""
		comparator = self._lookup_comparator(comparator)
		dead = [
			name
			for name, node in self.node.items()
			if stat in node and comparator(node[stat], threshold)
		]
		self.remove_nodes_from(dead)
		return self

	def cull_portals(
		self,
		stat: Stat,
		threshold: float = 0.5,
		comparator: Callable | str = ge,
	) -> Self:
		"""Delete portals whose stat >= ``threshold`` (default 0.5).

		Optional argument ``comparator`` will replace >= as the test
		for whether to cull. You can use the name of a stored function.

		"""
		comparator = self._lookup_comparator(comparator)
		dead = []
		for u in self.portal:
			for v in self.portal[u]:
				if stat in self.portal[u][v] and comparator(
					self.portal[u][v][stat], threshold
				):
					dead.append((u, v))
		self.remove_edges_from(dead)
		return self

	cull_edges = cull_portals

	def portals(self) -> Iterator[Edge]:
		"""Iterate over all portals."""
		for o in self.adj.values():
			yield from o.values()


class AbstractThing(ABC):
	__slots__ = ()
	character: AbstractCharacter
	name: NodeName

	@property
	def engine(self) -> AbstractEngine:
		return self.character.engine

	@property
	def location(self) -> Node:
		"""The ``Thing`` or ``Place`` I'm in."""
		locn = self["location"]
		if locn is None:
			raise AttributeError("Not really a Thing")
		try:
			return self.engine._get_node(self.character, locn)
		except KeyError as ex:
			raise AttributeError("Doesn't really exist") from ex

	@location.setter
	def location(self, v: KeyHint | Node | NodeName):
		if isinstance(v, Node):
			v = v.name
		elif not isinstance(v, Key):
			raise TypeError("Invalid location", v)
		self["location"] = NodeName(v)

	@abstractmethod
	def to_place(self) -> Node: ...

	def go_to_place(
		self,
		place: KeyHint | Node | NodeName,
		weight: KeyHint | Stat | EllipsisType = ...,
	) -> int:
		"""Assuming I'm in a node that has a :class:`Portal` direct
		to the given node, schedule myself to travel to the
		given :class:`Place`, taking an amount of time indicated by
		the ``weight`` stat on the :class:`Portal`, if given; else 1
		turn.

		Return the number of turns the travel will take.

		"""
		if isinstance(place, Node):
			placen = place.name
		elif not isinstance(place, Key):
			raise TypeError("Invalid node", place)
		else:
			placen = NodeName(place)
		curloc = self["location"]
		orm = self.character.engine
		turns = (
			1
			if weight is ...
			else self.character.portal[curloc][placen][weight]
		)
		with self.engine.plan():
			orm.turn += turns
			self["location"] = placen
		return turns

	def follow_path(
		self,
		path: list[KeyHint | NodeName],
		weight: KeyHint | Stat | EllipsisType = ...,
		check: bool = True,
	) -> int:
		"""Go to several nodes in succession, deciding how long to
		spend in each by consulting the ``weight`` stat of the
		:class:`Portal` connecting the one node to the next,
		default 1 turn.

		Return the total number of turns the travel will take. Raise
		:class:`TravelException` if I can't follow the whole path,
		either because some of its nodes don't exist, or because I'm
		scheduled to be somewhere else. Set ``check=False`` if
		you're really sure the path is correct, and this function
		will be faster.

		"""
		if len(path) < 2:
			raise ValueError("Paths need at least 2 nodes")
		eng = self.character.engine
		subpath: list[NodeName] = []
		if check:
			prevplace = path.pop(0)
			if not isinstance(prevplace, Key):
				raise TypeError("Invalid node", prevplace)
			prevplace = NodeName(prevplace)
			if prevplace != self["location"]:
				raise ValueError("Path does not start at my present location")
			subpath.append(prevplace)
			for place in path:
				if not isinstance(place, Key):
					raise TypeError("Invalid node", place)
				place = NodeName(place)
				if (
					prevplace not in self.character.portal
					or place not in self.character.portal[prevplace]
				):
					raise exc.TravelException(
						"Couldn't follow portal from {} to {}".format(
							prevplace, place
						),
						path=subpath,
						traveller=self,
					)
				subpath.append(place)
				prevplace = place
		else:
			for node in path:
				if not isinstance(node, Key):
					raise TypeError("Invalid node", node)
				subpath.append(NodeName(node))
		turns_total = 0
		prevsubplace = subpath.pop(0)
		turn_incs = []
		branch, turn, tick = eng.time
		for subplace in subpath:
			if weight is not ...:
				turn_incs.append(
					self.engine._edge_val_cache.retrieve(
						self.character.name,
						prevsubplace,
						subplace,
						weight,
						branch,
						turn,
						tick,
					)
				)
			else:
				turn_incs.append(1)
			turns_total += turn_incs[-1]
			turn += turn_incs[-1]
			tick = eng._turn_end_plan.get((branch, turn), 0)
		with eng.plan():
			for subplace, turn_inc in zip(subpath, turn_incs):
				eng.turn += turn_inc
				self["location"] = subplace
		return turns_total

	def travel_to(
		self,
		dest: Node | KeyHint,
		weight: Stat | KeyHint | EllipsisType = ...,
		graph: nx.DiGraph | EllipsisType = ...,
	) -> int:
		"""Find the shortest path to the given node from where I am
		now, and follow it.

		If supplied, the ``weight`` stat of each :class:`Portal` along
		the path will be used in pathfinding, and for deciding how
		long to stay in each Place along the way. Otherwise, I will stay
		in each :class:`Place` for 1 turn.

		The ``graph`` argument may be any NetworkX-style graph. It
		will be used for pathfinding if supplied, otherwise I'll use
		my :class:`Character`. In either case, however, I will attempt
		to actually follow the path using my :class:`Character`, which
		might not be possible if the supplied ``graph`` and my
		:class:`Character` are too different. If it's not possible,
		I'll raise a :class:`TravelException`, whose ``subpath``
		attribute holds the part of the path that I *can* follow. To
		make me follow it, pass it to my ``follow_path`` method.

		Return value is the number of turns the travel will take.

		"""
		if isinstance(dest, Node):
			destn = dest.name
		elif not isinstance(dest, Key):
			raise TypeError("Invalid node", dest)
		else:
			destn = dest
		if destn == self.location.name:
			raise TravelException("I'm already there", self.name, destn)
		if graph is ...:
			graph = self.character
		elif isinstance(graph, nx.DiGraph):
			graph = graph
		elif not isinstance(graph, Key):
			raise TypeError("Invalid character name", graph)
		else:
			graph = self.engine.character[CharName(graph)]
		orign: NodeName = self["location"]
		if weight is ...:
			path = nx.shortest_path(graph, orign, destn)
		elif not isinstance(weight, str):  # networkx limitation
			raise TypeError("Invalid weight", weight)
		else:
			path = nx.shortest_path(graph, orign, destn, weight)
		return self.follow_path(path, weight)


_T = TypeVar("_T")


def sort_set(s: Set[_T]) -> list[_T]:
	"""Return a sorted list of the contents of a set

	This is intended to be used to iterate over world state, so it doesn't
	support anything you can't use as a key in a dictionary.

	Works by converting everything to bytes before comparison. Tuples get
	their contents converted and joined with 0x1e, the record separator.
	Floats get converted to integer ratios, which are joined with 0x1f,
	the unit separator.

	This is memoized.

	"""

	def sort_set_key(v) -> bytes:
		if isinstance(v, bytes):
			return v
		elif isinstance(v, tuple):
			return b"\x1e".join(map(sort_set_key, v))
		elif isinstance(v, str):
			return v.encode()
		elif isinstance(v, int):
			return v.to_bytes(8)
		elif isinstance(v, float):
			return b"\x1f".join(i.to_bytes(8) for i in v.as_integer_ratio())
		else:
			raise TypeError(v)

	if not isinstance(s, Set):
		raise TypeError("sets only", type(s))
	s = frozenset(s)
	if s not in sort_set.memo:
		sort_set.memo[s] = sorted(s, key=sort_set_key)
	return sort_set.memo[s].copy()


sort_set.memo = LRU(1000, {})


def root_type(t: type) -> type | tuple[type, ...]:
	if hasattr(t, "evaluate_value"):
		t = t.evaluate_value()
	if hasattr(t, "__value__"):
		t = t.__value__
	if t is Key or t is Value:
		return t
	elif t is Time:
		return tuple
	elif hasattr(t, "__supertype__"):
		return root_type(t.__supertype__)
	elif hasattr(t, "__origin__"):
		orig = get_origin(t)
		if orig is None:
			return t
		elif orig is Annotated:
			return get_args(t)[0]
		ret = root_type(orig)
		if ret is Literal:
			for arg in get_args(orig):
				if not isinstance(arg, str):
					raise TypeError("Literal not storeable", arg)
			return str
		elif ret is Union:
			return tuple(map(root_type, get_args(t)))
		return ret
	elif isinstance(t, tuple):
		assert all(isinstance(tt, type) for tt in t)
		return t
	elif t is Literal:
		return t
	elif isinstance(t, type) and issubclass(t, dict):
		return dict
	return t


def deannotate(annotation: str) -> Iterator[type]:
	"""Yield all the types in an annotation

	For when you don't know if you're dealing with a union or not.

	"""
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
		typ = eval(annotation)
	typ = root_type(typ)
	if isinstance(typ, tuple):
		yield from typ
	else:
		yield typ


@define
class AbstractStringStore(MutableMapping[str, str], ABC):
	language: ClassVar[AbstractLanguageDescriptor]

	@abstractmethod
	def save(self, reimport: bool = False) -> None: ...

	@abstractmethod
	def _switch_language(self, lang: str) -> None: ...

	@abstractmethod
	def lang_items(
		self, lang: str | None = None
	) -> Iterator[tuple[str, str]]: ...

	@abstractmethod
	def blake2b(self) -> bytes: ...


@define
class AbstractFunctionStore[_K: str, _V: FunctionType | MethodType](ABC):
	@abstractmethod
	def save(self, reimport: bool = True) -> None: ...

	@abstractmethod
	def reimport(self) -> None: ...

	@abstractmethod
	def iterplain(self) -> Iterator[tuple[str, str]]: ...

	@abstractmethod
	def store_source(self, v: str, name: _K | None = None) -> None: ...

	@abstractmethod
	def get_source(self, name: _K) -> str: ...


class StatAlias:
	engine: AbstractEngine

	def __eq__(self, other):
		return EqQuery(self.engine, self, other)

	def __ne__(self, other):
		return NeQuery(self.engine, self, other)

	def __gt__(self, other):
		return GtQuery(self.engine, self, other)

	def __lt__(self, other):
		return LtQuery(self.engine, self, other)

	def __ge__(self, other):
		return GeQuery(self.engine, self, other)

	def __le__(self, other):
		return LeQuery(self.engine, self, other)

	def __contains__(self, item):
		return ContainsQuery(self.engine, self, item)


class EntityStatAlias(StatAlias, EntityStatAccessor): ...


class CharacterStatAlias(StatAlias, CharacterStatAccessor): ...


class UnitsAlias(StatAlias, UnitsAccessor): ...


class Query(ABC):
	oper: Callable[[Any, Any], Any] = lambda x, y: NotImplemented

	def __new__(cls, engine, leftside, rightside=None, **kwargs):
		if rightside is None:
			if not isinstance(leftside, cls):
				raise TypeError("You can't make a query with only one side")
			me = leftside
		else:
			me = super().__new__(cls)
			me.leftside = leftside
			me.rightside = rightside
		me.engine = engine
		return me

	def __eq__(self, other):
		return EqQuery(self.engine, self, other)

	def __gt__(self, other):
		return GtQuery(self.engine, self, other)

	def __ge__(self, other):
		return GeQuery(self.engine, self, other)

	def __lt__(self, other):
		return LtQuery(self.engine, self, other)

	def __le__(self, other):
		return LeQuery(self.engine, self, other)

	def __ne__(self, other):
		return NeQuery(self.engine, self, other)


class ComparisonQuery(Query):
	oper: Callable[[Any, Any], bool] = lambda x, y: NotImplemented

	def _list_times(self):
		with self.engine.wayback():
			times = list(
				slow_iter_turns_eval_cmp(self, self.oper, engine=self.engine)
			)
		return times

	def __and__(self, other):
		return IntersectionQuery(self.engine, self, other)

	def __or__(self, other):
		return UnionQuery(self.engine, self, other)

	def __sub__(self, other):
		return MinusQuery(self.engine, self, other)


class EqQuery(ComparisonQuery):
	oper = eq


class NeQuery(ComparisonQuery):
	oper = ne


class GtQuery(ComparisonQuery):
	oper = gt


class LtQuery(ComparisonQuery):
	oper = lt


class GeQuery(ComparisonQuery):
	oper = ge


class LeQuery(ComparisonQuery):
	oper = le


class ContainsQuery(ComparisonQuery):
	@staticmethod
	def oper(a, b):
		return b in a


class CompoundQuery(Query):
	oper: Callable[[Any, Any], set] = lambda x, y: NotImplemented


class UnionQuery(CompoundQuery):
	oper = operator.or_


class IntersectionQuery(CompoundQuery):
	oper = operator.and_


class MinusQuery(CompoundQuery):
	oper = operator.sub


class QueryResult(Sequence, Set, ABC):
	"""A slightly lazy tuple-like object holding a history query's results

	Testing for membership of a turn number in a QueryResult only evaluates
	the predicate for that turn number, and testing for membership of nearby
	turns is fast. Accessing the start or the end of the QueryResult only
	evaluates the initial or final item. Other forms of access cause the whole
	query to be evaluated in parallel.

	"""

	def __init__(self, windows_l, windows_r, oper, end_of_time):
		self._past_l = windows_l
		self._future_l = []
		self._past_r = windows_r
		self._future_r = []
		self._oper = oper
		self._list = None
		self._trues = set()
		self._falses = set()
		self._end_of_time = end_of_time

	def __iter__(self):
		if self._list is None:
			self._generate()
		return iter(self._list)

	def __reversed__(self):
		if self._list is None:
			self._generate()
		return reversed(self._list)

	def __len__(self):
		if not self._list:
			self._generate()
		return len(self._list)

	def __getitem__(self, item):
		if not self._list:
			if item == 0:
				return self._first()
			elif item == -1:
				return self._last()
			self._generate()
		return self._list[item]

	@abstractmethod
	def __contains__(self, turn: int | Turn) -> bool: ...

	@abstractmethod
	def _generate(self): ...

	@abstractmethod
	def _first(self): ...

	@abstractmethod
	def _last(self): ...

	def __str__(self):
		return f"<{self.__class__.__name__} containing {list(self)}>"

	def __repr__(self):
		return (
			f"<{self.__class__.__name__}({self._past_l}, {self._past_r},"
			f"{self._oper}, {self._end_of_time})>"
		)


class QueryResultEndTurn(QueryResult):
	def _generate(self):
		spans: list[tuple[int, int]] = []
		left = []
		right = []
		for turn_from, turn_to, l_v, r_v in _yield_intersections(
			chain(iter(self._past_l), reversed(self._future_l)),
			chain(iter(self._past_r), reversed(self._future_r)),
			until=self._end_of_time,
		):
			spans.append((turn_from, turn_to))
			left.append(l_v)
			right.append(r_v)
		try:
			import numpy as np

			bools = self._oper(np.array(left), np.array(right))
		except ImportError:
			bools = [self._oper(l, r) for (l, r) in zip(left, right)]
		self._list = _list = []
		append = _list.append
		add = self._trues.add
		for (start, stop), buul in zip(spans, bools):
			if buul:
				for turn in range(start, stop):
					append(turn)
					add(turn)

	def __contains__(self, item):
		if self._list is not None:
			return item in self._trues
		elif item in self._trues:
			return True
		elif item in self._falses:
			return False
		future_l = self._future_l
		past_l = self._past_l
		future_r = self._future_r
		past_r = self._past_r
		if not past_l:
			if not future_l:
				return False
			past_l.append((future_l.pop()))
		if not past_r:
			if not future_r:
				return False
			past_r.append((future_r.pop()))
		while past_l and past_l[-1][0] > item:
			future_l.append(past_l.pop())
		while future_l and future_l[-1][0] <= item:
			past_l.append(future_l.pop())
		while past_r and past_r[-1][0] > item:
			future_r.append(past_r.pop())
		while future_r and future_r[-1][0] <= item:
			past_r.append(future_r.pop())
		ret = self._oper(past_l[-1][2], past_r[-1][2])
		if ret:
			self._trues.add(item)
		else:
			self._falses.add(item)
		return ret

	def _last(self):
		"""Get the last turn on which the predicate held true"""
		past_l = self._past_l
		future_l = self._future_l
		while future_l:
			past_l.append(future_l.pop())
		past_r = self._past_r
		future_r = self._future_r
		while future_r:
			past_r.append(future_r)
		oper = self._oper
		while past_l and past_r:
			l_from, l_to, l_v = past_l[-1]
			r_from, r_to, r_v = past_r[-1]
			inter = intersect2((l_from, l_to), (r_from, r_to))
			if not inter:
				if l_from < r_from:
					future_r.append(past_r.pop())
				else:
					future_l.append(past_l.pop())
				continue
			if oper(l_v, r_v):
				# SQL results are exclusive on the right
				if inter[1] is None:
					return self._end_of_time - 1
				return inter[1] - 1

	def _first(self):
		"""Get the first turn on which the predicate held true"""
		if self._list is not None:
			if not self._list:
				return
			return self._list[0]
		oper = self._oper
		for turn_from, turn_to, l_v, r_v in _yield_intersections(
			chain(iter(self._past_l), reversed(self._future_l)),
			chain(iter(self._past_r), reversed(self._future_r)),
			until=self._end_of_time,
		):
			if oper(l_v, r_v):
				return turn_from


class QueryResultMidTurn(QueryResult):
	def _generate(self):
		spans = []
		left = []
		right = []
		for time_from, time_to, l_v, r_v in _yield_intersections(
			chain(iter(self._past_l), reversed(self._future_l)),
			chain(iter(self._past_r), reversed(self._future_r)),
			until=(self._end_of_time, 0),
		):
			spans.append((time_from, time_to))
			left.append(l_v)
			right.append(r_v)
		try:
			import numpy as np

			bools = self._oper(np.array(left), np.array(right))
		except ImportError:
			bools = [self._oper(l, r) for (l, r) in zip(left, right)]
		trues = self._trues
		_list = self._list = []
		for span, buul in zip(spans, bools):
			if buul:
				for turn in range(
					span[0][0], span[1][0] + (1 if span[1][1] else 0)
				):
					if turn in trues:
						continue
					trues.add(turn)
					_list.append(turn)

	def __contains__(self, item):
		if self._list is not None:
			return item in self._trues
		if item in self._trues:
			return True
		if item in self._falses:
			return False
		future_l = self._future_l
		past_l = self._past_l
		future_r = self._future_r
		past_r = self._past_r
		if not past_l:
			if not future_l:
				return False
			past_l.append(future_l.pop())
		if not past_r:
			if not future_r:
				return False
			past_r.append(future_r.pop())
		while past_l and past_l[-1][0][0] >= item:
			future_l.append(past_l.pop())
		while future_l and not (
			past_l
			and past_l[-1][0][0] <= item
			and (past_l[-1][1][0] is None or item <= past_l[-1][1][0])
		):
			past_l.append(future_l.pop())
		left_candidates = [past_l[-1]]
		while (
			future_l
			and future_l[-1][0][0] <= item
			and (future_l[-1][1][0] is None or item <= future_l[-1][1][0])
		):
			past_l.append(future_l.pop())
			left_candidates.append(past_l[-1])
		while past_r and past_r[-1][0][0] >= item:
			future_r.append(past_r.pop())
		while future_r and not (
			past_r and past_r[-1][0][0] <= item <= past_r[-1][1][0]
		):
			past_r.append(future_r.pop())
		right_candidates = [past_r[-1]]
		while (
			future_r
			and future_r[-1][0][0] <= item
			and (future_r[-1][1][0] is None or item <= future_r[-1][1][0])
		):
			past_r.append(future_r.pop())
			right_candidates.append(past_r[-1])
		oper = self._oper
		while left_candidates and right_candidates:
			if intersect2(left_candidates[-1][:2], right_candidates[-1][:2]):
				if oper(left_candidates[-1][2], right_candidates[-1][2]):
					return True
			if left_candidates[-1][0] < right_candidates[-1][0]:
				right_candidates.pop()
			else:
				left_candidates.pop()
		return False

	def _last(self):
		"""Get the last turn on which the predicate held true"""
		past_l = self._past_l
		future_l = self._future_l
		while future_l:
			past_l.append(future_l.pop())
		past_r = self._past_r
		future_r = self._future_r
		while future_r:
			past_r.append(future_r)
		oper = self._oper
		while past_l and past_r:
			l_from, l_to, l_v = past_l[-1]
			r_from, r_to, r_v = past_r[-1]
			inter = intersect2((l_from, l_to), (r_from, r_to))
			if not inter:
				if l_from < r_from:
					future_r.append(past_r.pop())
				else:
					future_l.append(past_l.pop())
				continue
			if oper(l_v, r_v):
				if inter[1] == (None, None):
					return self._end_of_time - 1
				return inter[1][0] - (0 if inter[1][1] else 1)

	def _first(self):
		"""Get the first turn on which the predicate held true"""
		oper = self._oper
		for time_from, time_to, l_v, r_v in _yield_intersections(
			chain(iter(self._past_l), reversed(self._future_l)),
			chain(iter(self._past_r), reversed(self._future_r)),
			until=(self._end_of_time, 0),
		):
			if oper(l_v, r_v):
				return time_from[0]


class CombinedQueryResult(QueryResult):
	def __init__(self, left: QueryResult, right: QueryResult, oper):
		self._left = left
		self._right = right
		self._oper = oper

	def _generate(self):
		if not hasattr(self, "_set"):
			self._set = self._oper(set(self._left), set(self._right))

	def __iter__(self):
		self._generate()
		return iter(self._set)

	def __len__(self):
		self._generate()
		return len(self._set)

	def __contains__(self, item):
		if hasattr(self, "_set"):
			return item in self._set
		return self._oper(item in self._left, item in self._right)

	def _first(self):
		return self._left._first()

	def _last(self):
		return self._right._last()


def _mungeside(side):
	if isinstance(side, Query):
		return side._iter_times
	elif isinstance(side, EntityStatAlias):
		return EntityStatAccessor(
			side.entity,
			side.stat,
			side.engine,
			side.branch,
			side.turn,
			side.tick,
			side.current,
			side.mungers,
		)
	elif isinstance(side, EntityStatAccessor):
		return side
	else:
		return lambda: side


def slow_iter_turns_eval_cmp(qry, oper, start_branch=None, engine=None):
	"""Iterate over all turns on which a comparison holds.

	This is expensive. It evaluates the query for every turn in history.

	"""
	leftside = _mungeside(qry.leftside)
	rightside = _mungeside(qry.rightside)
	engine = engine or leftside.engine or rightside.engine

	for branch, fork_turn, fork_tick in engine._iter_parent_btt(
		start_branch or engine.branch
	):
		if branch is None:
			return
		turn_start, tick_start = engine._branch_start(branch)
		for turn in map(Turn, range(turn_start, fork_turn + 1)):
			if oper(leftside(branch, turn), rightside(branch, turn)):
				yield branch, turn


def slow_iter_btts_eval_cmp(qry, oper, start_branch=None, engine=None):
	leftside = _mungeside(qry.leftside)
	rightside = _mungeside(qry.rightside)
	engine = engine or leftside.engine or rightside.engine
	assert engine is not None

	for branch, fork_turn, fork_tick in engine._iter_parent_btt(
		start_branch or engine.branch
	):
		if branch is None:
			return
		turn_start = engine._branch_start(branch)[0]
		for turn in map(Turn, range(turn_start, fork_turn + 1)):
			if turn == fork_turn:
				local_turn_end = fork_tick
			else:
				local_turn_end = engine._turn_end_plan[branch, turn]
			for tick in map(Tick, range(0, local_turn_end + 1)):
				try:
					val = oper(
						leftside(branch, turn, tick),
						rightside(branch, turn, tick),
					)
				except KeyError:
					continue
				if val:
					yield branch, turn, tick


def intersect2(left, right):
	"""Return intersection of 2 windows of time"""
	if left == right:
		return left
	elif left == (None, None) or left == ((None, None), (None, None)):
		return right
	elif right == (None, None) or right == ((None, None), (None, None)):
		return left
	elif left[0] is None or left[0] == (None, None):
		if right[0] is None or right[0] == (None, None):
			return None, min((left[1], right[1]))
		elif right[1] is None or right[1] == (None, None):
			if left[1] <= right[0]:
				return left[1], right[0]
			else:
				return None
		elif right[0] <= left[1]:
			return right[0], left[1]
		else:
			return None
	elif left[1] is None or left[1] == (None, None):
		if right[0] is None or right[0] == (None, None):
			return left[0], right[1]
		elif left[0] <= right[0]:
			return right
		elif right[1] is None or right[1] == (None, None):
			return max((left[0], right[0])), (None, None) if isinstance(
				left[0], tuple
			) else None
		elif left[0] <= right[1]:
			return left[0], right[1]
		else:
			return None
	# None not in left
	elif right[0] is None or right[0] == (None, None):
		return left[0], min((left[1], right[1]))
	elif right[1] is None or right[1] == (None, None):
		if left[1] >= right[0]:
			return right[0], left[1]
		else:
			return None
	if left > right:
		(left, right) = (right, left)
	if left[1] >= right[0]:
		if right[1] > left[1]:
			return right[0], left[1]
		else:
			return right
	return None


def _yield_intersections(iter_l, iter_r, until=None):
	try:
		l_from, l_to, l_v = next(iter_l)
	except StopIteration:
		return
	try:
		r_from, r_to, r_v = next(iter_r)
	except StopIteration:
		return
	while True:
		if l_to in (None, (None, None)):
			l_to = until
		if r_to in (None, (None, None)):
			r_to = until
		intersection = intersect2((l_from, l_to), (r_from, r_to))
		if intersection and intersection[0] != intersection[1]:
			yield intersection + (l_v, r_v)
			if intersection[1] is None or (
				isinstance(intersection[1], tuple) and intersection[1] is None
			):
				return
		if (
			l_to is None
			or r_to is None
			or (isinstance(l_to, tuple) and l_to[1] is None)
			or (isinstance(r_to, tuple) and r_to[1] is None)
		):
			break
		elif l_to <= r_to:
			try:
				l_from, l_to, l_v = next(iter_l)
			except StopIteration:
				break
		else:
			try:
				r_from, r_to, r_v = next(iter_r)
			except StopIteration:
				break
	if l_to is None:
		while True:
			try:
				r_from, r_to, r_v = next(iter_r)
			except StopIteration:
				if until:
					yield intersect2((l_from, l_to), (r_to, until)) + (
						l_v,
						r_v,
					)
				return
			yield intersect2((l_from, l_to), (r_from, r_to)) + (l_v, r_v)
	if r_to is None:
		while True:
			try:
				l_from, l_to, l_v = next(iter_l)
			except StopIteration:
				if until:
					yield intersect2((l_to, until), (r_from, r_to)) + (
						l_v,
						r_v,
					)
				return
			yield intersect2((l_from, l_to), (r_from, r_to)) + (l_v, r_v)


def _default_args_munger(self, k):
	"""By default, `PickyDefaultDict`'s ``type`` takes no positional arguments."""
	return tuple()


def _default_kwargs_munger(self, k):
	"""By default, `PickyDefaultDict`'s ``type`` takes no keyword arguments."""
	return {}


@define
class AbstractPickyDefaultDict[_K, _V](dict[_K, _V]):
	__slots__ = ()

	@property
	@abstractmethod
	def value_type(self) -> type: ...

	def args_munger(self, key: _K) -> tuple[_K, ...]:
		return tuple()

	def kwargs_munger(self, key: _K) -> dict[_K, _V]:
		return {}

	@cached_property
	def _lock(self):
		return RLock()

	def __getitem__(self, k):
		with self._lock:
			if k in self:
				return super().__getitem__(k)
			if isinstance(self.value_type, GenericAlias):
				value_type = get_origin(self.value_type)
				if hasattr(value_type, "key_type"):
					value_type = partial(
						value_type, *get_args(self.value_type)
					)
				else:
					value_type = partial(
						value_type, get_args(self.value_type)[-1]
					)
			else:
				value_type = self.value_type
			ret = value_type(
				*self.args_munger(self, k), **self.kwargs_munger(self, k)
			)
			super().__setitem__(k, ret)
			return ret

	@classmethod
	def _validate(cls, d: dict) -> bool:
		"""Does ``d`` have the right types in it to be one of me?"""
		keytyp, valtyp = inspect.getargs(cls)
		if issubclass(valtyp, AbstractPickyDefaultDict):
			for v in d.values():
				if not valtyp._validate(v):
					return False
		else:
			for v in d.values():
				if not isinstance(v, valtyp):
					return False
		return True

	@staticmethod
	def _validate_generic_alias(key_type, value_type, key, value):
		if isinstance(value_type, GenericAlias):
			if not isinstance(value, dict):
				raise TypeError("Dictionary required", value, type(value))
			for k, v in value.items():
				self._validate_generic_alias(*value_type.__args__, k, v)
		else:
			if not isinstance(value, value_type):
				raise TypeError("Incorrect value type", value, value_type)

	def __setitem__(self, k, v):
		with self._lock:
			if isinstance(self.value_type, GenericAlias):
				if not isinstance(v, dict):
					raise TypeError("Must store dictionary", v)
				self._validate_generic_alias(*self.value_type.__args__, k, v)
				super().__setitem__(k, v)
			elif not isinstance(v, root_type(self.value_type)):
				v = self.value_type(v)
			super().__setitem__(k, v)


def convert_subtype(subtyp: type) -> type:
	return getattr(subtyp, "__supertype__", subtyp)


@define
class PickyDefaultDict[_K, _V](AbstractPickyDefaultDict[_K, _V]):
	"""A ``defaultdict`` alternative that requires values of a specific type.

	Pass some type object (such as a class) to the constructor to
	specify what type to use by default, which is the only type I will
	accept.

	Default values are constructed with no arguments by default;
	supply ``args_munger`` and/or ``kwargs_munger`` to override this.
	They take arguments ``self`` and the unused key being looked up.
	If they're set to ``None``, we can't construct default arguments.

	"""

	__slots__ = ()
	value_type: type[_V] = field(converter=convert_subtype)
	args_munger: Callable[[Self, _K], tuple[_K, ...]] | None = (
		_default_args_munger
	)
	kwargs_munger: Callable[[Self, _K], dict[_K, _V]] | None = (
		_default_kwargs_munger
	)
	parent: dict | None = None
	key: _K | None = None


@define
class PickierDefaultDict[_K, _V](AbstractPickyDefaultDict[_K, _V]):
	"""So picky, even the keys need to be a certain type"""

	__slots__ = ()
	key_type: type[_K] = field(converter=convert_subtype)
	value_type: type[_V] = field(converter=convert_subtype)
	args_munger: Callable[[Self, _K], tuple[_K, ...]] | None = (
		_default_args_munger
	)
	kwargs_munger: Callable[[Self, _K], dict[_K, _V]] | None = (
		_default_kwargs_munger
	)
	parent: dict | None = None
	key: _K | None = None

	def __getitem__(self, item):
		if not isinstance(item, self.key_type):
			raise TypeError("Illegal key type", item, self.key_type)
		return super().__getitem__(item)

	def __setitem__(self, key: _K, value: _V):
		if not isinstance(key, self.key_type):
			raise TypeError("Wrong type of key", key, self.key_type)
		super().__setitem__(key, value)

	@staticmethod
	def _validate_generic_alias(key_type, value_type, key, value):
		if not isinstance(key, key_type):
			raise TypeError("Wrong key type", key, key_type)
		super()._validate_generic_alias(key_type, value_type, key, value)


@define
class StructuredDefaultDict[_K, _V](dict[_K, _V]):
	"""A `defaultdict`-like class with values stored at a specific depth.

	Requires an integer to tell it how many layers deep to go.
	The innermost layer will be ``PickyDefaultDict``, which will take the
	``type``, ``args_munger``, and ``kwargs_munger`` arguments supplied
	to my constructor.

	"""

	__slots__ = ()

	layer: int = field(validator=validators.ge(1))
	type: type = type(None)
	args_munger: Callable[[Self, _K], tuple[_K, ...]] | None = (
		_default_args_munger
	)
	kwargs_munger: Callable[[Self, _K], dict[_K, _V]] | None = (
		_default_kwargs_munger
	)
	gettest: Callable[[_K], None] = lambda k: None
	settest: Callable[[_K, _V], None] = lambda k, v: None
	parent: StructuredDefaultDict | None = None
	key: _K | None = None

	@cached_property
	def _lock(self):
		return RLock()

	def __getitem__(self, k: _K) -> _V:
		with self._lock:
			self.gettest(k)
			if k in self:
				return dict.__getitem__(self, k)
			layer, typ, args_munger, kwargs_munger = (
				self.layer,
				self.type,
				self.args_munger,
				self.kwargs_munger,
			)
			if layer == 1:
				if typ is type(None):
					ret = {}
				else:
					ret = PickyDefaultDict(typ, args_munger, kwargs_munger)
					ret.parent = self
					ret.key = k
			elif layer < 1:
				raise ValueError("Invalid layer")
			else:
				ret = StructuredDefaultDict(
					layer - 1, typ, args_munger, kwargs_munger
				)
				ret.parent = self
				ret.key = k
			dict.__setitem__(self, k, ret)
			return ret

	def __setitem__(self, k: _K, v: _V) -> None:
		with self._lock:
			self.settest(k, v)
			layer, typ, args_munger, kwargs_munger = (
				self.layer,
				self.type,
				self.args_munger,
				self.kwargs_munger,
			)
			if type(v) is StructuredDefaultDict:
				if (
					v.layer == layer - 1
					and (typ is type(None) or v.type is typ)
					and v.args_munger is args_munger
					and v.kwargs_munger is kwargs_munger
				):
					super().__setitem__(k, v)
					return
			elif type(v) is PickyDefaultDict:
				if (
					layer == 1
					and v.value_type is typ
					and v.args_munger is args_munger
					and v.kwargs_munger is kwargs_munger
				):
					super().__setitem__(k, v)
					return
			raise TypeError("Can't set layer {}".format(self.layer))


class AbstractLanguageDescriptor(Signal, ABC):
	@abstractmethod
	def _get_language(self, inst: AbstractStringStore) -> str:
		pass

	@abstractmethod
	def _set_language(self, inst: AbstractStringStore, val: str) -> None:
		pass

	def __get__(self, instance: AbstractStringStore, owner=None):
		return self._get_language(instance)

	def __set__(self, inst: AbstractStringStore, val: str):
		self._set_language(inst, val)
		self.send(inst, language=val)
