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
"""Common utility functions and data structures."""

from __future__ import annotations

import gc
import os
import sys
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping, Set
from concurrent.futures import Future
from contextlib import contextmanager
from enum import Enum
from functools import cached_property, partial, wraps
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
	pow,
	sub,
	truediv,
)
from random import Random
from textwrap import dedent
from time import monotonic
from types import FunctionType, MethodType, ModuleType
from typing import (
	TYPE_CHECKING,
	Annotated,
	Callable,
	Hashable,
	Iterable,
	Iterator,
	KeysView,
	MutableMapping,
	Optional,
	Sequence,
	Type,
	TypeVar,
)

import msgpack
import networkx as nx
import numpy as np
from annotated_types import Ge, Le
from blinker import Signal
from tblib import Traceback

from . import exc
from .exc import TimeError, WorkerProcessReadOnlyError
from .types import (
	Branch,
	CharName,
	DiGraph,
	Edge,
	EdgeValDict,
	EternalKey,
	GraphMapping,
	Key,
	LinearTime,
	Node,
	NodeName,
	NodeValDict,
	RulebookName,
	RuleName,
	Stat,
	Tick,
	Time,
	Turn,
	UniversalKey,
	Value,
)
from .wrap import SpecialMapping

if TYPE_CHECKING:
	from .collections import FunctionStore
	from .rule import Rule, RuleBook

TRUE: bytes = msgpack.packb(True)
FALSE: bytes = msgpack.packb(False)
NONE: bytes = msgpack.packb(None)
EMPTY: bytes = msgpack.packb({})
EMPTY_LIST: bytes = msgpack.packb([])
ELLIPSIS: bytes = b"\xc7\x00{"
NAME: bytes = msgpack.packb("name")
NODES: bytes = msgpack.packb("nodes")
EDGES: bytes = msgpack.packb("edges")
UNITS: bytes = msgpack.packb("units")
RULEBOOK: bytes = msgpack.packb("rulebook")
RULEBOOKS: bytes = msgpack.packb("rulebooks")
NODE_VAL: bytes = msgpack.packb("node_val")
EDGE_VAL: bytes = msgpack.packb("edge_val")
ETERNAL: bytes = msgpack.packb("eternal")
UNIVERSAL: bytes = msgpack.packb("universal")
STRINGS: bytes = msgpack.packb("strings")
RULES: bytes = msgpack.packb("rules")
TRIGGERS: bytes = msgpack.packb("triggers")
PREREQS: bytes = msgpack.packb("prereqs")
ACTIONS: bytes = msgpack.packb("actions")
NEIGHBORHOOD: bytes = msgpack.packb("neighborhood")
BIG: bytes = msgpack.packb("big")
LOCATION: bytes = msgpack.packb("location")
BRANCH: bytes = msgpack.packb("branch")
EMPTY_MAPPING: bytes = msgpack.packb({})


class SignalDict(Signal, dict):
	def __setitem__(self, __key, __value):
		super().__setitem__(__key, __value)
		self.send(self, key=__key, value=__value)

	def __delitem__(self, __key):
		super().__delitem__(__key)
		self.send(self, key=__key, value=None)


class BadTimeException(Exception):
	"""You tried to do something that would make sense at a different game-time

	But doesn't make sense now

	"""


class MsgpackExtensionType(Enum):
	"""Type codes for packing special lisien types into msgpack"""

	tuple = 0x00
	frozenset = 0x01
	set = 0x02
	exception = 0x03
	graph = 0x04
	character = 0x7F
	place = 0x7E
	thing = 0x7D
	portal = 0x7C
	ellipsis = 0x7B
	function = 0x7A
	method = 0x79
	trigger = 0x78
	prereq = 0x77
	action = 0x76


class get_rando:
	"""Attribute getter for randomization functions

	Aliases functions of a randomizer, wrapped so that they won't run in
	planning mode, and will save the randomizer's state after every call.

	"""

	__slots__ = ("_getter", "_wrapfun", "_instance")
	_getter: callable

	def __init__(self, attr, *attrs):
		self._getter = attrgetter(attr, *attrs)

	def __get__(self, instance, owner) -> callable:
		if hasattr(self, "_wrapfun") and self._instance is instance:
			return self._wrapfun
		retfun: callable = self._getter(instance)

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


@contextmanager
def timer(msg="", logfun: callable = None):
	if logfun is None:
		logfun = print
	start = monotonic()
	yield
	logfun("{:,.3f} {}".format(monotonic() - start, msg))


def getatt(attribute_name):
	"""An easy way to make an alias"""
	return property(attrgetter(attribute_name))


_T = TypeVar("_T")


def singleton_get(s: Iterable[_T]) -> _T:
	"""Take an iterable and return its only item if possible, else None."""
	it = None
	for that in s:
		if it is not None:
			return None
		it = that
	return it


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
			engine = entity.db
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

	def munge(self, munger: callable):
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
			time_was = self.engine.time
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
				yield None
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


def dedent_source(source):
	nlidx = source.index("\n")
	if nlidx is None:
		raise ValueError("Invalid source")
	while source[:nlidx].strip().startswith("@"):
		source = source[nlidx + 1 :]
		nlidx = source.index("\n")
	return dedent(source)


def _sort_set_key(v):
	if isinstance(v, tuple):
		return (2,) + tuple(map(repr, v))
	if isinstance(v, str):
		return 1, v
	return 0, repr(v)


class SizedDict(OrderedDict):
	"""A dictionary that discards old entries when it gets too big."""

	def __init__(self, max_entries: Annotated[int, Ge(0)] = 1000):
		self._n = max_entries
		super().__init__()

	def __setitem__(self, key, value):
		while len(self) > self._n:
			self.popitem(last=False)
		super().__setitem__(key, value)


_sort_set_memo = SizedDict()


def sort_set(s: Set[_T]) -> list[_T]:
	"""Return a sorted list of the contents of a set

	This is intended to be used to iterate over world state.

	Non-strings come before strings and then tuples. Tuples compare
	element-wise as normal.

	This is memoized.

	"""
	if not isinstance(s, Set):
		raise TypeError("sets only")
	s = frozenset(s)
	if s not in _sort_set_memo:
		_sort_set_memo[s] = sorted(s, key=_sort_set_key)
	return _sort_set_memo[s].copy()


class FakeFuture(Future):
	"""A 'Future' that calls its function immediately and sets the result"""

	def __init__(self, func: callable, *args, **kwargs):
		super().__init__()
		self.set_result(func(*args, **kwargs))


def fake_submit(func: callable, *args, **kwargs) -> FakeFuture:
	"""A replacement for `concurrent.futures.Executor.submit` that works in serial

	In testing, you may use, eg.,
	``@patch.object(executor, 'submit', new=fake_submit)``
	to make normally parallel operations serial.

	"""
	return FakeFuture(func, *args, **kwargs)


class AbstractBookmarkMapping(MutableMapping, Callable):
	@abstractmethod
	def __call__(self, key: Key) -> None: ...


class AbstractEngine(ABC):
	"""Parent class to the real Engine as well as EngineProxy.

	Implements serialization and the __getattr__ for stored methods.

	By default, the deserializers will refuse to create lisien entities.
	If you want them to, use my ``loading`` property to open a ``with``
	block, in which deserialized entities will be created as needed.

	"""

	thing_cls: type
	place_cls: type
	portal_cls: type
	char_cls: type
	character: Mapping[CharName, Type[char_cls]]
	eternal: MutableMapping[EternalKey, Value]
	universal: MutableMapping[UniversalKey, Value]
	rulebook: MutableMapping[RulebookName, "RuleBook"]
	rule: MutableMapping[RuleName, "Rule"]
	trunk: Branch
	branch: Branch
	turn: Turn
	tick: Tick
	time: Time
	function: ModuleType | "FunctionStore"
	method: ModuleType | "FunctionStore"
	trigger: ModuleType | "FunctionStore"
	prereq: ModuleType | "FunctionStore"
	action: ModuleType | "FunctionStore"
	bookmark: AbstractBookmarkMapping
	_rando: Random
	_branches_d: dict[
		Optional[Branch], tuple[Optional[Branch], Turn, Tick, Turn, Tick]
	]

	@cached_property
	def logger(self):
		if hasattr(self, "_logger"):
			return self._logger
		from logging import getLogger

		return getLogger("lisien")

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
	def pack(self) -> Callable[[Value], bytes]:
		try:
			from lise_ormsgpack import packb

			return packb
		except ImportError:
			pass

		def pack_set(s):
			return msgpack.ExtType(
				MsgpackExtensionType.set.value, packer(list(s))
			)

		handlers = {
			type(...): lambda _: msgpack.ExtType(
				MsgpackExtensionType.ellipsis.value, b""
			),
			nx.Graph: lambda graf: msgpack.ExtType(
				MsgpackExtensionType.graph.value,
				packer(
					[
						"Graph",
						graf._node,
						graf._adj,
						graf.graph,
					]
				),
			),
			nx.DiGraph: lambda graf: msgpack.ExtType(
				MsgpackExtensionType.graph.value,
				packer(["DiGraph", graf._node, graf._adj, graf.graph]),
			),
			nx.MultiGraph: lambda graf: msgpack.ExtType(
				MsgpackExtensionType.graph.value,
				packer(["MultiGraph", graf._node, graf._adj, graf.graph]),
			),
			nx.MultiDiGraph: lambda graf: msgpack.ExtType(
				MsgpackExtensionType.graph.value,
				packer(["MultiDiGraph", graf._node, graf._adj, graf.graph]),
			),
			tuple: lambda tup: msgpack.ExtType(
				MsgpackExtensionType.tuple.value, packer(list(tup))
			),
			frozenset: lambda frozs: msgpack.ExtType(
				MsgpackExtensionType.frozenset.value, packer(list(frozs))
			),
			set: pack_set,
			FunctionType: lambda func: msgpack.ExtType(
				getattr(MsgpackExtensionType, func.__module__).value,
				packer(func.__name__),
			),
			MethodType: lambda meth: msgpack.ExtType(
				MsgpackExtensionType.method.value, packer(meth.__name__)
			),
			Exception: lambda exc: msgpack.ExtType(
				MsgpackExtensionType.exception.value,
				packer(
					[
						exc.__class__.__name__,
						Traceback(exc.__traceback__).to_dict()
						if hasattr(exc, "__traceback__")
						else None,
					]
					+ list(exc.args)
				),
			),
		}

		def pack_handler(obj):
			if isinstance(obj, Exception):
				typ = Exception
			else:
				typ = type(obj)
			if typ in handlers:
				return handlers[typ](obj)
			elif isinstance(obj, DiGraph):
				return msgpack.ExtType(
					MsgpackExtensionType.character.value, packer(obj.name)
				)
			elif isinstance(obj, AbstractThing):
				return msgpack.ExtType(
					MsgpackExtensionType.thing.value,
					packer([obj.character.name, obj.name]),
				)
			elif isinstance(obj, Node):
				return msgpack.ExtType(
					MsgpackExtensionType.place.value,
					packer([obj.graph.name, obj.name]),
				)
			elif isinstance(obj, Edge):
				return msgpack.ExtType(
					MsgpackExtensionType.portal.value,
					packer(
						[
							obj.graph.name,
							obj.orig,
							obj.dest,
						]
					),
				)
			elif isinstance(obj, Set):
				return pack_set(obj)
			elif isinstance(obj, Mapping):
				return dict(obj)
			elif isinstance(obj, list):
				return list(obj)
			raise TypeError("Can't pack {}".format(typ))

		packer = partial(
			msgpack.packb,
			default=pack_handler,
			strict_types=True,
			use_bin_type=True,
		)
		return packer

	@cached_property
	def unpack(
		self,
	) -> Callable[
		[bytes],
		Value
		| nx.DiGraph
		| char_cls
		| place_cls
		| thing_cls
		| portal_cls
		| Exception
		| callable,
	]:
		char_cls = self.char_cls
		place_cls = self.place_cls
		portal_cls = self.portal_cls
		thing_cls = self.thing_cls
		excs = {
			# builtin exceptions
			"AssertionError": AssertionError,
			"AttributeError": AttributeError,
			"EOFError": EOFError,
			"FloatingPointError": FloatingPointError,
			"GeneratorExit": GeneratorExit,
			"ImportError": ImportError,
			"IndexError": IndexError,
			"KeyError": KeyError,
			"KeyboardInterrupt": KeyboardInterrupt,
			"MemoryError": MemoryError,
			"NameError": NameError,
			"NotImplementedError": NotImplementedError,
			"OSError": OSError,
			"OverflowError": OverflowError,
			"RecursionError": RecursionError,
			"ReferenceError": ReferenceError,
			"RuntimeError": RuntimeError,
			"StopIteration": StopIteration,
			"IndentationError": IndentationError,
			"TabError": TabError,
			"SystemError": SystemError,
			"SystemExit": SystemExit,
			"TypeError": TypeError,
			"UnboundLocalError": UnboundLocalError,
			"UnicodeError": UnicodeError,
			"UnicodeEncodeError": UnicodeEncodeError,
			"UnicodeDecodeError": UnicodeDecodeError,
			"UnicodeTranslateError": UnicodeTranslateError,
			"ValueError": ValueError,
			"ZeroDivisionError": ZeroDivisionError,
			# networkx exceptions
			"HasACycle": nx.exception.HasACycle,
			"NodeNotFound": nx.exception.NodeNotFound,
			"PowerIterationFailedConvergence": nx.exception.PowerIterationFailedConvergence,
			"ExceededMaxIterations": nx.exception.ExceededMaxIterations,
			"AmbiguousSolution": nx.exception.AmbiguousSolution,
			"NetworkXAlgorithmError": nx.exception.NetworkXAlgorithmError,
			"NetworkXException": nx.exception.NetworkXException,
			"NetworkXError": nx.exception.NetworkXError,
			"NetworkXNoCycle": nx.exception.NetworkXNoCycle,
			"NetworkXNoPath": nx.exception.NetworkXNoPath,
			"NetworkXNotImplemented": nx.exception.NetworkXNotImplemented,
			"NetworkXPointlessConcept": nx.exception.NetworkXPointlessConcept,
			"NetworkXUnbounded": nx.exception.NetworkXUnbounded,
			"NetworkXUnfeasible": nx.exception.NetworkXUnfeasible,
			# lisien exceptions
			"NonUniqueError": exc.NonUniqueError,
			"AmbiguousUserError": exc.AmbiguousLeaderError,
			"RulesEngineError": exc.RulesEngineError,
			"RuleError": exc.RuleError,
			"RedundantRuleError": exc.RedundantRuleError,
			"UserFunctionError": exc.UserFunctionError,
			"WorldIntegrityError": exc.WorldIntegrityError,
			"CacheError": exc.CacheError,
			"TravelException": exc.TravelException,
			"OutOfTimelineError": exc.OutOfTimelineError,
			"HistoricKeyError": exc.HistoricKeyError,
			"NotInKeyframeError": exc.NotInKeyframeError,
			"WorkerProcessReadOnlyError": exc.WorkerProcessReadOnlyError,
		}

		def unpack_graph(ext: bytes) -> nx.Graph:
			cls, node, adj, graph = unpacker(ext)
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
			data: tuple[str, dict | None] = unpacker(ext)
			if data[0] not in excs:
				return Exception(*data)
			ret = excs[data[0]](*data[2:])
			if data[1] is not None:
				ret.__traceback__ = Traceback.from_dict(data[1]).to_traceback()
			return ret

		def unpack_char(ext: bytes) -> char_cls:
			charn = unpacker(ext)
			return char_cls(self, charn, init_rulebooks=False)

		def unpack_place(ext: bytes) -> place_cls:
			charn, placen = unpacker(ext)
			return place_cls(
				char_cls(self, charn, init_rulebooks=False), placen
			)

		def unpack_thing(ext: bytes) -> thing_cls:
			charn, thingn = unpacker(ext)
			# Breaks if the thing hasn't been instantiated yet, not great
			return self.character[charn].thing[thingn]

		def unpack_portal(ext: bytes) -> portal_cls:
			charn, orign, destn = unpacker(ext)
			return portal_cls(
				char_cls(self, charn, init_rulebooks=False), orign, destn
			)

		def unpack_seq(t: type[_T], ext: bytes) -> _T:
			unpacked = unpacker(ext)
			if not isinstance(unpacked, list):
				raise TypeError("Tried to unpack", type(unpacked), t)
			return t(unpacked)

		def unpack_func(store: FunctionStore, ext: bytes) -> callable:
			unpacked = unpacker(ext)
			if not isinstance(unpacked, str):
				raise TypeError("Tried to unpack as func", type(unpacked))
			return getattr(store, unpacked)

		handlers = {
			MsgpackExtensionType.ellipsis.value: lambda _: ...,
			MsgpackExtensionType.graph.value: unpack_graph,
			MsgpackExtensionType.character.value: unpack_char,
			MsgpackExtensionType.place.value: unpack_place,
			MsgpackExtensionType.thing.value: unpack_thing,
			MsgpackExtensionType.portal.value: unpack_portal,
			MsgpackExtensionType.tuple.value: partial(unpack_seq, tuple),
			MsgpackExtensionType.frozenset.value: partial(
				unpack_seq, frozenset
			),
			MsgpackExtensionType.set.value: partial(unpack_seq, set),
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

		def unpack_handler(
			code: MsgpackExtensionType, data: bytes
		) -> Value | Exception | msgpack.ExtType:
			if code in handlers:
				return handlers[code](data)
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

	@abstractmethod
	def _get_node(
		self, char: AbstractCharacter | CharName, node: NodeName
	) -> Node: ...

	@abstractmethod
	def _btt(self) -> tuple[Branch, Turn, Tick]: ...

	def branches(self) -> KeysView:
		return self._branches_d.keys()

	def branch_parent(self, branch: Branch | None) -> Branch | None:
		if branch is None or branch not in self._branches_d:
			return None
		return self._branches_d[branch][0]

	def _branch_start(self, branch: Branch | None = None) -> LinearTime:
		if branch is None:
			branch = self.branch
		_, turn, tick, _, _ = self._branches_d[branch]
		return turn, tick

	def _branch_end(self, branch: Branch | None = None) -> LinearTime:
		if branch is None:
			branch = self.branch
		_, _, _, turn, tick = self._branches_d[branch]
		return turn, tick

	@abstractmethod
	def _start_branch(
		self, parent: Branch, branch: Branch, turn: Turn, tick: Tick
	) -> None: ...

	@abstractmethod
	def _set_btt(self, branch: Branch, turn: Turn, tick: Tick) -> None: ...

	@abstractmethod
	def _extend_branch(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> None: ...

	@abstractmethod
	def load_at(self, branch: Branch, turn: Turn, tick: Tick) -> None: ...

	def branch_start_turn(self, branch: Branch | None = None) -> Turn:
		return self._branch_start(branch)[0]

	def branch_start_tick(self, branch: Branch | None = None) -> Tick:
		return self._branch_start(branch)[1]

	def branch_end_turn(self, branch: Branch | None = None) -> Turn:
		return self._branch_end(branch)[0]

	def branch_end_tick(self, branch: Branch | None = None) -> Tick:
		return self._branch_end(branch)[1]

	@abstractmethod
	def turn_end(
		self, branch: Branch | None = None, turn: Turn | None = None
	) -> Tick: ...

	@abstractmethod
	def turn_end_plan(
		self, branch: Branch | None = None, turn: Turn | None = None
	) -> Tick: ...

	@abstractmethod
	def add_character(
		self,
		name: Key,
		data: nx.Graph | DiGraph | None = None,
		layout: bool = False,
		node: NodeValDict | None = None,
		edge: EdgeValDict | None = None,
		**kwargs,
	): ...

	def new_character(
		self,
		name: Key,
		data: nx.Graph | DiGraph | None = None,
		layout: bool = False,
		node: NodeValDict | None = None,
		edge: EdgeValDict | None = None,
		**kwargs,
	):
		name = CharName(name)
		self.add_character(name, data)
		return self.character[name]

	@abstractmethod
	def export(
		self,
		name: str | None,
		path: str | os.PathLike | None = None,
		indent: bool = True,
	) -> str | os.PathLike: ...

	@classmethod
	@abstractmethod
	def from_archive(
		cls,
		path: str | os.PathLike,
		prefix: str | os.PathLike | None = ".",
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


class AbstractCharacter(DiGraph):
	"""The Character API, with all requisite mappings and graph generators.

	Mappings resemble those of a NetworkX digraph:

	* ``thing`` and ``place`` are subsets of ``node``
	* ``edge``, ``adj``, and ``succ`` are aliases of ``portal``
	* ``pred`` is an alias to ``preportal``
	* ``stat`` is a dict-like mapping of data that changes over game-time,
	to be used in place of graph attributes

	"""

	engine = getatt("db")
	no_unwrap = True
	name: Hashable
	db: AbstractEngine

	@staticmethod
	def is_directed():
		return True

	@staticmethod
	def is_multigraph():
		return False

	@abstractmethod
	def add_place(self, name: NodeName, **kwargs):
		pass

	def add_node(self, name: NodeName, **kwargs):
		self.add_place(name, **kwargs)

	@abstractmethod
	def add_places_from(self, seq: Iterable, **attrs):
		pass

	def add_nodes_from(self, seq: Iterable, **attrs):
		self.add_places_from(seq, **attrs)

	def new_place(self, name: NodeName, **kwargs):
		"""Add a Place and return it.

		If there's already a Place by that name, put a number on the end.

		"""
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

	def new_node(self, name: NodeName, **kwargs):
		return self.new_place(name, **kwargs)

	@abstractmethod
	def add_thing(self, name: NodeName, location: NodeName, **kwargs):
		pass

	@abstractmethod
	def add_things_from(self, seq: Iterable, **attrs):
		pass

	def new_thing(self, name: NodeName, location: NodeName, **kwargs):
		"""Add a Thing and return it.

		If there's already a Thing by that name, put a number on the end.

		"""
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
	def place2thing(self, place: NodeName, location: NodeName) -> None: ...

	@abstractmethod
	def thing2place(self, thing: NodeName) -> None: ...

	def remove_node(self, node):
		if node in self.node:
			self.node[node].delete()

	def remove_nodes_from(self, nodes: Iterable[NodeName]):
		for node in nodes:
			if node in self.node:
				self.node[node].delete()

	@abstractmethod
	def add_portal(self, orig: NodeName, dest: NodeName, **kwargs):
		pass

	def add_edge(self, orig: NodeName, dest: NodeName, **kwargs):
		self.add_portal(orig, dest, **kwargs)

	def new_portal(self, orig: NodeName, dest: NodeName, **kwargs):
		self.add_portal(orig, dest, **kwargs)
		return self.portal[orig][dest]

	@abstractmethod
	def add_portals_from(self, seq: Iterable, **attrs):
		pass

	def add_edges_from(self, seq: Iterable, **attrs):
		self.add_portals_from(seq, **attrs)

	@abstractmethod
	def remove_portal(self, origin: NodeName, destination: NodeName):
		pass

	def remove_portals_from(self, seq: Iterable[tuple[NodeName, NodeName]]):
		for orig, dest in seq:
			del self.portal[orig][dest]

	def remove_edges_from(self, seq: Iterable[tuple[NodeName, NodeName]]):
		self.remove_portals_from(seq)

	@abstractmethod
	def remove_place(self, place: NodeName):
		pass

	def remove_places_from(self, seq: Iterable[NodeName]):
		for place in seq:
			self.remove_place(place)

	@abstractmethod
	def remove_thing(self, thing: NodeName) -> None:
		pass

	def remove_things_from(self, seq: Iterable[NodeName]) -> None:
		for thing in seq:
			self.remove_thing(thing)

	@abstractmethod
	def add_unit(
		self, a: CharName | Node, b: Optional[NodeName] = None
	) -> None:
		pass

	@abstractmethod
	def remove_unit(
		self, a: CharName | Node, b: Optional[NodeName] = None
	) -> None:
		pass

	def __eq__(self, other: AbstractCharacter):
		return isinstance(other, AbstractCharacter) and self.name == other.name

	def __iter__(self):
		return iter(self.node)

	def __len__(self):
		return len(self.node)

	def __bool__(self):
		try:
			return self.name in self.db.character
		except AttributeError:
			return False  # we can't "really exist" when we've no engine

	def __contains__(self, k: NodeName):
		return k in self.node

	def __getitem__(self, k: NodeName):
		return self.adj[k]

	ThingMapping: type[SpecialMapping]

	@cached_property
	def thing(self) -> ThingMapping:
		return self.ThingMapping(self)

	PlaceMapping: type[SpecialMapping]

	@cached_property
	def place(self) -> PlaceMapping:
		return self.PlaceMapping(self)

	ThingPlaceMapping: type[SpecialMapping]

	@cached_property
	def _node(self) -> ThingPlaceMapping:
		return self.ThingPlaceMapping(self)

	node: ThingPlaceMapping = getatt("_node")
	nodes: ThingPlaceMapping = getatt("_node")

	PortalSuccessorsMapping: type[SpecialMapping]

	@cached_property
	def _succ(self) -> PortalSuccessorsMapping:
		return self.PortalSuccessorsMapping(self)

	portal: PortalSuccessorsMapping = getatt("_succ")
	adj: PortalSuccessorsMapping = getatt("_succ")
	succ: PortalSuccessorsMapping = getatt("_succ")
	edge: PortalSuccessorsMapping = getatt("_succ")
	_adj: PortalSuccessorsMapping = getatt("_succ")

	PortalPredecessorsMapping: type[SpecialMapping]

	@cached_property
	def _pred(self) -> PortalPredecessorsMapping:
		return self.PortalPredecessorsMapping(self)

	preportal: PortalPredecessorsMapping = getatt("_pred")
	pred: PortalPredecessorsMapping = getatt("_pred")

	UnitGraphMapping: type[SpecialMapping]

	@cached_property
	def unit(self) -> UnitGraphMapping:
		return self.UnitGraphMapping(self)

	stat: GraphMapping = getatt("graph")

	def units(self):
		for units in self.unit.values():
			yield from units.values()

	def historical(self, stat: Stat):
		from .query import EntityStatAlias

		return EntityStatAlias(entity=self.stat, stat=stat)

	def do(self, func: callable, *args, **kwargs) -> AbstractCharacter:
		"""Apply the function to myself, and return myself.

		Look up the function in the method store if needed. Pass it any
		arguments given, keyword or positional.

		Useful chiefly when chaining.

		"""
		if not callable(func):
			func = getattr(self.engine.method, func)
		func(self, *args, **kwargs)
		return self

	def copy_from(self, g: AbstractCharacter) -> None:
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
				self.add_portal(renamed[u], renamed[v], symmetrical=True, **d)
		return self

	def become(self, g: AbstractCharacter) -> AbstractCharacter:
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

	def _lookup_comparator(self, comparator: callable | str) -> callable:
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
		comparator: callable | str = ge,
	) -> None:
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
		comparator: callable | str = ge,
	):
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


DiGraph.register(AbstractCharacter)


def normalize_layout(l: dict) -> dict:
	"""Make sure all the spots in a layout are where you can click.

	Returns a copy of the layout with all spot coordinates are
	normalized to within (0.0, 0.98).

	"""
	xs = []
	ys = []
	ks = []
	for k, (x, y) in l.items():
		xs.append(x)
		ys.append(y)
		ks.append(k)
	minx = np.min(xs)
	maxx = np.max(xs)
	if maxx == minx:
		xnorm = np.array([0.5] * len(xs))
	else:
		xco = 0.98 / (maxx - minx)
		xnorm = np.multiply(np.subtract(xs, [minx] * len(xs)), xco)
	miny = np.min(ys)
	maxy = np.max(ys)
	if miny == maxy:
		ynorm = np.array([0.5] * len(ys))
	else:
		yco = 0.98 / (maxy - miny)
		ynorm = np.multiply(np.subtract(ys, [miny] * len(ys)), yco)
	return dict(zip(ks, zip(map(float, xnorm), map(float, ynorm))))


class AbstractThing(MutableMapping):
	character: AbstractCharacter
	engine: AbstractEngine
	name: NodeName

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
	def location(self, v: Node | NodeName):
		if hasattr(v, "name"):
			v = v.name
		self["location"] = v

	def go_to_place(
		self, place: Node | NodeName, weight: Optional[Stat] = None
	) -> int:
		"""Assuming I'm in a node that has a :class:`Portal` direct
		to the given node, schedule myself to travel to the
		given :class:`Place`, taking an amount of time indicated by
		the ``weight`` stat on the :class:`Portal`, if given; else 1
		turn.

		Return the number of turns the travel will take.

		"""
		if hasattr(place, "name"):
			placen = place.name
		else:
			placen = place
		curloc = self["location"]
		orm = self.character.engine
		turns = (
			1
			if weight is None
			else self.engine._portal_objs[
				(self.character.name, curloc, place)
			].get(weight, 1)
		)
		with self.engine.plan():
			orm.turn += turns
			self["location"] = placen
		return turns

	def follow_path(
		self,
		path: list[NodeName],
		weight: Optional[Stat] = None,
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
		if check:
			prevplace = path.pop(0)
			if prevplace != self["location"]:
				raise ValueError("Path does not start at my present location")
			subpath = [prevplace]
			for place in path:
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
			subpath = path.copy()
		turns_total = 0
		prevsubplace = subpath.pop(0)
		turn_incs = []
		branch, turn, tick = eng._btt()
		for subplace in subpath:
			if weight is not None:
				turn_incs.append(
					self.engine._edge_val_cache.retrieve(
						self.character.name,
						prevsubplace,
						subplace,
						0,
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
		dest: Node | NodeName,
		weight: Optional[Stat] = None,
		graph: nx.DiGraph = None,
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
		destn = dest.name if hasattr(dest, "name") else dest
		if destn == self.location.name:
			raise ValueError("I'm already there", self.name, destn)
		graph = self.character if graph is None else graph
		path = nx.shortest_path(graph, self["location"], destn, weight)
		return self.follow_path(path, weight)


def repr_call_sig(func: callable | str, *args, **kwargs):
	if not isinstance(func, (str, bytes)):
		func = func.__name__
	return (
		f"{func}({', '.join(map(repr, args))}"
		f"{', ' if args and kwargs else ''}"
		f"{', '.join(f'{arg}={repr(item)}' for (arg, item) in kwargs.items())})"
	)


def print_call_sig(
	func: callable | str, *args, file=sys.stdout, end="\n", **kwargs
):
	print(repr_call_sig(func, *args, **kwargs), file=file, end=end)


@contextmanager
def _garbage_ctx(collect=True):
	"""Context manager to disable the garbage collector

	:param collect: Whether to immediately collect garbage upon context exit

	"""
	gc_was_active = gc.isenabled()
	if gc_was_active:
		gc.disable()
	yield
	if gc_was_active:
		gc.enable()
	if collect:
		gc.collect()


def _garbage_dec(fn: callable, collect=True) -> callable:
	"""Decorator to disable the garbage collector for a function

	:param collect: Whether to immediately collect garbage when the function returns

	"""

	@wraps(fn)
	def garbage(*args, **kwargs):
		with _garbage_ctx(collect=collect):
			return fn(*args, **kwargs)

	return garbage


def garbage(arg: callable = None, collect: bool = False):
	"""Disable the garbage collector, then re-enable it when done.

	May be used as a context manager or a decorator.

	:param collect: Whether to immediately run a collection after re-enabling
		the garbage collector. Default ``False``.

	"""

	if arg is None:
		return _garbage_ctx(collect=collect)
	else:
		return _garbage_dec(arg, collect=collect)


def world_locked(fn: callable) -> callable:
	"""Decorator for functions that alter the world state

	They will hold a reentrant lock, preventing more than one function
	from mutating the world at a time.

	"""

	@wraps(fn)
	def lockedy(*args, **kwargs):
		with args[0].world_lock:
			return fn(*args, **kwargs)

	return lockedy


class TimeSignal(Signal, Sequence):
	"""Acts like a tuple of ``(branch, turn, tick)`` for the most part.

	This is a ``Signal``. To set a function to be called whenever the
	branch or turn changes, pass it to my ``connect`` method.

	"""

	engine: AbstractEngine

	def __init__(self, engine: AbstractEngine):
		super().__init__()
		self.engine = engine

	def __iter__(self):
		yield self.engine.branch
		yield self.engine.turn
		yield self.engine.tick

	def __len__(self):
		return 3

	def __getitem__(self, i: str | int) -> Branch | Turn | Tick:
		if i in ("branch", 0):
			return self.engine.branch
		if i in ("turn", 1):
			return self.engine.turn
		if i in ("tick", 2):
			return self.engine.tick
		if isinstance(i, int):
			raise IndexError(i)
		else:
			raise KeyError(i)

	def __setitem__(self, i: str | int, v: str | int) -> None:
		if i in ("branch", 0):
			self.engine.branch = Branch(v)
		elif i in ("turn", 1):
			self.engine.turn = Turn(v)
		elif i in ("tick", 2):
			self.engine.tick = Tick(v)
		else:
			exctyp = KeyError if isinstance(i, str) else IndexError
			raise exctyp(i)

	def __str__(self):
		return str(tuple(self))

	def __eq__(self, other):
		return tuple(self) == other

	def __ne__(self, other):
		return tuple(self) != other

	def __gt__(self, other):
		return tuple(self) > other

	def __ge__(self, other):
		return tuple(self) >= other

	def __lt__(self, other):
		return tuple(self) < other

	def __le__(self, other):
		return tuple(self) <= other


class TimeSignalDescriptor:
	__doc__ = TimeSignal.__doc__

	def __get__(self, inst, cls):
		if not hasattr(inst, "_time_signal"):
			inst._time_signal = TimeSignal(inst)
		return inst._time_signal

	def __set__(self, inst: AbstractEngine, val: Time):
		if getattr(inst, "_worker", False):
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		if not hasattr(inst, "_time_signal"):
			inst._time_signal = TimeSignal(inst)
		sig = inst._time_signal
		branch_then, turn_then, tick_then = inst._btt()
		branch_now, turn_now, tick_now = val
		if (branch_then, turn_then, tick_then) == (
			branch_now,
			turn_now,
			tick_now,
		):
			return
		e = inst
		# enforce the arrow of time, if it's in effect
		if (
			hasattr(e, "_forward")
			and e._forward
			and hasattr(e, "_planning")
			and not e._planning
		):
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

		if branch_now in e.branches():
			e._extend_branch(branch_now, turn_now, tick_now)
			e.load_at(branch_now, turn_now, tick_now)
		else:
			e._start_branch(branch_then, branch_now, turn_now, tick_now)
		e._set_btt(branch_now, turn_now, tick_now)
		sig.send(
			e,
			branch_then=branch_then,
			turn_then=turn_then,
			tick_then=tick_then,
			branch_now=branch_now,
			turn_now=turn_now,
			tick_now=tick_now,
		)
