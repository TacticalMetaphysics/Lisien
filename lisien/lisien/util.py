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

from abc import ABC, abstractmethod
from collections.abc import Set
from concurrent.futures import Future
from enum import Enum
from operator import (
	ge,
	gt,
	le,
	lt,
	eq,
	attrgetter,
	add,
	sub,
	mul,
	pow,
	truediv,
	floordiv,
	mod,
)
from functools import partial, wraps, cached_property
from contextlib import contextmanager
from random import Random
from textwrap import dedent
from time import monotonic
from types import MethodType, FunctionType
from typing import (
	Any,
	Mapping,
	Iterable,
	Union,
	Callable,
	Dict,
	Hashable,
	Tuple,
	FrozenSet,
)

import msgpack
import networkx as nx
import numpy as np
from tblib import Traceback
from blinker import Signal

from . import exc, allegedb
from .exc import TravelException
from .allegedb.graph import Node, Edge, DiGraph

Key = Union[str, int, float, Tuple["Key", ...], FrozenSet["Key"]]


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


class FinalRule:
	"""A singleton sentinel for the rule iterator"""

	__slots__ = []

	def __hash__(self):
		# completely random integer
		return 6448962173793096248


final_rule = FinalRule()


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
	final_rule = 0x7B
	function = 0x7A
	method = 0x79


class get_rando:
	"""Attribute getter for randomization functions

	Aliases functions of a randomizer, wrapped so that they won't run in
	planning mode, and will save the randomizer's state after every call.

	"""

	__slots__ = ("_getter", "_wrapfun", "_instance")
	_getter: Callable

	def __init__(self, attr, *attrs):
		self._getter = attrgetter(attr, *attrs)

	def __get__(self, instance, owner) -> Callable:
		if hasattr(self, "_wrapfun") and self._instance is instance:
			return self._wrapfun
		retfun = self._getter(instance)

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


def singleton_get(s):
	"""Take an iterable and return its only item if possible, else None."""
	it = None
	for that in s:
		if it is not None:
			return None
		it = that
	return it


class EntityStatAccessor(object):
	__slots__ = [
		"engine",
		"entity",
		"branch",
		"turn",
		"tick",
		"stat",
		"current",
		"mungers",
	]

	def __init__(
		self,
		entity,
		stat,
		engine=None,
		branch=None,
		turn=None,
		tick=None,
		current=False,
		mungers: list = None,
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

	def __call__(self, branch=None, turn=None, tick=None):
		if self.current:
			res = self.entity[self.stat]
		else:
			branc, trn, tck = self.engine._btt()
			self.engine.branch = branch or self.branch
			self.engine.turn = turn if turn is not None else self.turn
			if tick is not None:
				self.engine.tick = tick
			elif self.tick is not None:
				self.engine.tick = self.tick
			if hasattr(self.entity, "stat"):
				res = self.entity.stat[self.stat]
			else:
				res = self.entity[self.stat]
			self.engine._set_btt(branc, trn, tck)
		for munger in self.mungers:
			res = munger(res)
		return res

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

	def munge(self, munger):
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

	def __getitem__(self, k):
		return self.munge(lambda x: x[k])

	def iter_history(self, beginning, end):
		"""Iterate over all the values this stat has had in the given window, inclusive."""
		# It might be useful to do this in a way that doesn't change the
		# engine's time, perhaps for thread safety
		engine = self.engine
		entity = self.entity
		oldturn = engine.turn
		oldtick = engine.tick
		stat = self.stat
		for turn in range(beginning, end + 1):
			engine.turn = turn
			try:
				y = entity[stat]
			except KeyError:
				yield None
				continue
			if hasattr(y, "unwrap"):
				y = y.unwrap()
			yield y
		engine.turn = oldturn
		engine.tick = oldtick


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


_sort_set_memo = {}


def sort_set(s):
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
	return _sort_set_memo[s]


def fake_submit(func, *args, **kwargs):
	"""A replacement for `concurrent.futures.Executor.submit` that works in serial

	This is for testing. Use, eg.,
	``@patch.object(executor, 'submit', new=fake_submit)``
	to make normally parallel operations serial.

	"""

	class FakeFuture(Future):
		def __init__(self, func, *args, **kwargs):
			super().__init__()
			self.set_result(func(*args, **kwargs))

	return FakeFuture(func, *args, **kwargs)


class AbstractEngine(ABC):
	"""Parent class to the real Engine as well as EngineProxy.

	Implements serialization and the __getattr__ for stored methods.

	By default, the deserializers will refuse to create lisien entities.
	If you want them to, use my ``loading`` property to open a ``with``
	block, in which deserialized entities will be created as needed.

	"""

	portal_cls: type
	thing_cls: type
	place_cls: type
	portal_cls: type
	char_cls: type
	character: Mapping[Any, "char_cls"]
	_rando: Random

	@cached_property
	def pack(self):
		try:
			from lise_ormsgpack import packb

			return packb
		except ImportError:
			pass
		handlers = {
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
			set: lambda s: msgpack.ExtType(
				MsgpackExtensionType.set.value, packer(list(s))
			),
			FinalRule: lambda obj: msgpack.ExtType(
				MsgpackExtensionType.final_rule.value, b""
			),
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
			elif isinstance(obj, Node):
				if hasattr(obj, "location"):
					return msgpack.ExtType(
						MsgpackExtensionType.thing.value,
						packer([obj.character.name, obj.name]),
					)
				else:
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
							obj.origin.name,
							obj.destination.name,
						]
					),
				)
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
	def unpack(self):
		charmap = self.character
		char_cls = self.char_cls
		place_cls = self.place_cls
		thing_cls = self.thing_cls
		portal_cls = self.portal_cls
		function = self.function
		method = self.method
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
			"AmbiguousAvatarError": exc.AmbiguousAvatarError,
			"AmbiguousUserError": exc.AmbiguousUserError,
			"RulesEngineError": exc.RulesEngineError,
			"RuleError": exc.RuleError,
			"RedundantRuleError": exc.RedundantRuleError,
			"UserFunctionError": exc.UserFunctionError,
			"WorldIntegrityError": exc.WorldIntegrityError,
			"CacheError": exc.CacheError,
			"TravelException": exc.TravelException,
			"OutOfTimelineError": allegedb.OutOfTimelineError,
			"HistoricKeyError": HistoricKeyError,
			"NotInKeyframeError": allegedb.cache.NotInKeyframeError,
			"WorkerProcessReadOnlyError": exc.WorkerProcessReadOnlyError,
		}

		def unpack_graph(ext):
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

		def unpack_exception(ext):
			data = unpacker(ext)
			if data[0] not in excs:
				return Exception(*data)
			ret = excs[data[0]](*data[2:])
			if data[1] is not None:
				ret.__traceback__ = Traceback.from_dict(data[1]).to_traceback()
			return ret

		def unpack_char(ext):
			charn = unpacker(ext)
			return char_cls(self, charn, init_rulebooks=False)

		def unpack_place(ext):
			charn, placen = unpacker(ext)
			return place_cls(
				char_cls(self, charn, init_rulebooks=False), placen
			)

		def unpack_thing(ext):
			charn, thingn = unpacker(ext)
			# Breaks if the thing hasn't been instantiated yet, not great
			return self.character[charn].thing[thingn]

		def unpack_portal(ext):
			charn, orign, destn = unpacker(ext)
			return portal_cls(
				char_cls(self, charn, init_rulebooks=False), orign, destn
			)

		handlers = {
			MsgpackExtensionType.graph.value: unpack_graph,
			MsgpackExtensionType.character.value: unpack_char,
			MsgpackExtensionType.place.value: unpack_place,
			MsgpackExtensionType.thing.value: unpack_thing,
			MsgpackExtensionType.portal.value: unpack_portal,
			MsgpackExtensionType.final_rule.value: lambda obj: final_rule,
			MsgpackExtensionType.tuple.value: lambda ext: tuple(unpacker(ext)),
			MsgpackExtensionType.frozenset.value: lambda ext: frozenset(
				unpacker(ext)
			),
			MsgpackExtensionType.set.value: lambda ext: set(unpacker(ext)),
			MsgpackExtensionType.function.value: lambda ext: getattr(
				function, unpacker(ext)
			),
			MsgpackExtensionType.method.value: lambda ext: getattr(
				method, unpacker(ext)
			),
			MsgpackExtensionType.exception.value: unpack_exception,
		}

		def unpack_handler(code, data):
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

	@property
	def initial_turn(self):
		"""The first turn of the current branch"""
		return self._branches[self.branch][1]

	@property
	def final_turn(self):
		"""The last turn of the current branch"""
		return self._branches[self.branch][3]

	def coin_flip(self) -> bool:
		"""Return True or False with equal probability."""
		return self.choice((True, False))

	def die_roll(self, d) -> int:
		"""Roll a die with ``d`` faces. Return the result."""
		return self.randint(1, d)

	def dice(self, n, d) -> Iterable[int]:
		"""Roll ``n`` dice with ``d`` faces, and yield the results.

		This is an iterator. You'll get the result of each die in
		succession.

		"""
		for i in range(0, n):
			yield self.die_roll(d)

	def dice_check(
		self,
		n: int,
		d: int,
		target: int,
		comparator: Union[str, Callable] = "<=",
	) -> bool:
		"""Roll ``n`` dice with ``d`` sides, sum them, and compare

		If ``comparator`` is provided, use it instead of the default <=.
		You may use a string like '<' or '>='.

		"""
		from operator import gt, lt, ge, le, eq, ne

		comps: Dict[str, Callable] = {
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

	def percent_chance(self, pct: int) -> bool:
		"""Return True or False with a given percentile probability

		Values not between 0 and 100 are treated as though they
		were 0 or 100, whichever is nearer.

		"""
		if pct <= 0:
			return False
		if pct >= 100:
			return True
		return pct > self.randint(0, 99)

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


class SpecialMappingDescriptor:
	def __init__(self, mapclsname):
		self.mapps = {}
		self.mapclsname = mapclsname

	def __get__(self, instance, owner):
		attname = "_" + self.mapclsname
		if not hasattr(instance, attname):
			mappcls = getattr(instance, self.mapclsname)
			setattr(instance, attname, mappcls(instance))
		return getattr(instance, attname)

	def __set__(self, instance, value):
		attname = "_" + self.mapclsname
		if not hasattr(instance, attname):
			mappcls = getattr(instance, self.mapclsname)
			setattr(instance, attname, mappcls(instance))
		it = getattr(instance, attname)
		it.clear()
		it.update(value)


class AbstractCharacter(Mapping):
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
	def add_place(self, name, **kwargs):
		pass

	def add_node(self, name, **kwargs):
		self.add_place(name, **kwargs)

	@abstractmethod
	def add_places_from(self, seq, **attrs):
		pass

	def add_nodes_from(self, seq, **attrs):
		self.add_places_from(seq, **attrs)

	def new_place(self, name, **kwargs):
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

	def new_node(self, name, **kwargs):
		return self.new_place(name, **kwargs)

	@abstractmethod
	def add_thing(self, name, location, **kwargs):
		pass

	@abstractmethod
	def add_things_from(self, seq, **attrs):
		pass

	def new_thing(self, name, location, **kwargs):
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
	def add_portal(self, orig, dest, **kwargs):
		pass

	def add_edge(self, orig, dest, **kwargs):
		self.add_portal(orig, dest, **kwargs)

	def new_portal(self, orig, dest, **kwargs):
		self.add_portal(orig, dest, **kwargs)
		return self.portal[orig][dest]

	@abstractmethod
	def add_portals_from(self, seq, **attrs):
		pass

	def add_edges_from(self, seq, **attrs):
		self.add_portals_from(seq, **attrs)

	@abstractmethod
	def remove_portal(self, origin, destination):
		pass

	def remove_portals_from(self, seq):
		for orig, dest in seq:
			del self.portal[orig][dest]

	def remove_edges_from(self, seq):
		self.remove_portals_from(seq)

	@abstractmethod
	def remove_place(self, place):
		pass

	def remove_places_from(self, seq):
		for place in seq:
			self.remove_place(place)

	@abstractmethod
	def remove_thing(self, thing):
		pass

	def remove_things_from(self, seq):
		for thing in seq:
			self.remove_thing(thing)

	@abstractmethod
	def remove_node(self, node):
		pass

	def remove_nodes_from(self, seq):
		for node in seq:
			self.remove_node(node)

	@abstractmethod
	def add_unit(self, a, b=None):
		pass

	@abstractmethod
	def remove_unit(self, a, b=None):
		pass

	def __eq__(self, other):
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

	def __contains__(self, k):
		return k in self.node

	def __getitem__(self, k):
		return self.adj[k]

	thing = SpecialMappingDescriptor("ThingMapping")
	place = SpecialMappingDescriptor("PlaceMapping")
	node = nodes = _node = SpecialMappingDescriptor("ThingPlaceMapping")
	portal = adj = succ = edge = _adj = _succ = SpecialMappingDescriptor(
		"PortalSuccessorsMapping"
	)
	preportal = pred = _pred = SpecialMappingDescriptor(
		"PortalPredecessorsMapping"
	)
	unit = SpecialMappingDescriptor("UnitGraphMapping")
	stat = getatt("graph")

	def units(self):
		for units in self.unit.values():
			yield from units.values()

	def historical(self, stat):
		from .query import StatusAlias

		return StatusAlias(entity=self.stat, stat=stat)

	def do(self, func, *args, **kwargs):
		"""Apply the function to myself, and return myself.

		Look up the function in the method store if needed. Pass it any
		arguments given, keyword or positional.

		Useful chiefly when chaining.

		"""
		if not callable(func):
			func = getattr(self.engine.method, func)
		func(self, *args, **kwargs)
		return self

	def copy_from(self, g):
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

	def become(self, g):
		"""Erase all my nodes and edges. Replace them with a copy of the graph
		provided.

		Return myself.

		"""
		self.clear()
		self.place.update(g.nodes)
		self.adj.update(g.adj)
		return self

	def clear(self):
		self.node.clear()
		self.portal.clear()
		self.stat.clear()

	def _lookup_comparator(self, comparator):
		if callable(comparator):
			return comparator
		ops = {"ge": ge, "gt": gt, "le": le, "lt": lt, "eq": eq}
		if comparator in ops:
			return ops[comparator]
		return getattr(self.engine.function, comparator)

	def cull_nodes(self, stat, threshold=0.5, comparator=ge):
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

	def cull_portals(self, stat, threshold=0.5, comparator=ge):
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


def normalize_layout(l):
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


class AbstractThing(ABC):
	@property
	def location(self) -> "allegedb.Node":
		"""The ``Thing`` or ``Place`` I'm in."""
		locn = self["location"]
		if locn is None:
			raise AttributeError("Not really a Thing")
		return self.engine._get_node(self.character, locn)

	@location.setter
	def location(self, v: Union["allegedb.Node", Key]):
		if hasattr(v, "name"):
			v = v.name
		self["location"] = v

	def go_to_place(
		self, place: Union["allegedb.Node", Key], weight: Key = None
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
		self, path: list, weight: Key = None, check: bool = True
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
					raise TravelException(
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
			tick = eng._turn_end_plan.get(turn, 0)
			_, start_turn, start_tick, end_turn, end_tick = eng._branches[
				branch
			]
			if (
				(start_turn < turn < end_turn)
				or (
					start_turn == end_turn == turn
					and start_tick <= tick < end_tick
				)
				or (start_turn == turn and start_tick <= tick)
				or (end_turn == turn and tick < end_tick)
			):
				eng.load_at(branch, turn, tick)
		with eng.plan():
			for subplace, turn_inc in zip(subpath, turn_incs):
				eng.turn += turn_inc
				self["location"] = subplace
		return turns_total

	def travel_to(
		self,
		dest: Union["allegedb.Node", Key],
		weight: Key = None,
		graph: nx.DiGraph = None,
	) -> int:
		"""Find the shortest path to the given node from where I am
		now, and follow it.

		If supplied, the ``weight`` stat of each :class:`Portal` along
		the path will be used in pathfinding, and for deciding how
		long to stay in each Place along the way.

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
			raise ValueError("I'm already at {}".format(destn))
		graph = self.character if graph is None else graph
		path = nx.shortest_path(graph, self["location"], destn, weight)
		return self.follow_path(path, weight)


class HistoricKeyError(KeyError):
	"""Distinguishes deleted keys from those that were never set"""

	def __init__(self, *args, deleted=False):
		super().__init__(*args)
		self.deleted = deleted
