# This file is part of LiSE, a framework for life simulation games.
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
"""Common utility functions and data structures.

"""
from _operator import ge, gt, le, lt, eq
from abc import ABC, abstractmethod
from collections.abc import Set
from operator import attrgetter, add, sub, mul, pow, truediv, floordiv, mod
from functools import partial
from contextlib import contextmanager
from textwrap import dedent
from time import monotonic
from types import MethodType, FunctionType
from typing import Mapping, Iterable, Union, Callable, Dict, Hashable
from weakref import WeakValueDictionary

import msgpack
import networkx as nx
from .reify import reify

from LiSE import exc


class FinalRule:
	"""A singleton sentinel for the rule iterator"""
	__slots__ = []

	def __hash__(self):
		# completely random integer
		return 6448962173793096248


final_rule = FinalRule()

MSGPACK_TUPLE = 0x00
MSGPACK_FROZENSET = 0x01
MSGPACK_SET = 0x02
MSGPACK_EXCEPTION = 0x03
MSGPACK_CHARACTER = 0x7f
MSGPACK_PLACE = 0x7e
MSGPACK_THING = 0x7d
MSGPACK_PORTAL = 0x7c
MSGPACK_FINAL_RULE = 0x7b
MSGPACK_FUNCTION = 0x7a
MSGPACK_METHOD = 0x79
MSGPACK_TRIGGER = 0x78
MSGPACK_PREREQ = 0x77
MSGPACK_ACTION = 0x76


class getnoplan:
	"""Attribute getter that raises an exception if in planning mode"""
	__slots__ = ('_getter', )
	_getter: Callable

	def __init__(self, attr, *attrs):
		self._getter = attrgetter(attr, *attrs)

	def __get__(self, instance, owner) -> Callable:
		if instance._planning:
			raise exc.PlanError("Don't use randomization in a plan")
		return self._getter(instance)


@contextmanager
def timer(msg=''):
	start = monotonic()
	yield
	print("{:,.3f} {}".format(monotonic() - start, msg))


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
		'engine', 'entity', 'branch', 'turn', 'tick', 'stat', 'current',
		'mungers'
	]

	def __init__(self,
					entity,
					stat,
					engine=None,
					branch=None,
					turn=None,
					tick=None,
					current=False,
					mungers: list = None):
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
			res = self.entity[self.stat]
			self.engine.branch = branc
			self.engine.turn = trn
			self.engine.tick = tck
		for munger in self.mungers:
			res = munger(res)
		return res

	def __ne__(self, other):
		return self() != other

	def __str__(self):
		return str(self())

	def __repr__(self):
		return "EntityStatAccessor({}[{}]{}), {} mungers".format(
			self.entity, self.stat,
			"" if self.current else ", branch={}, turn={}, tick={}".format(
				self.branch, self.turn, self.tick), len(self.mungers))

	def __gt__(self, other):
		return self() > other

	def __ge__(self, other):
		return self >= other

	def __lt__(self, other):
		return self < other

	def __le__(self, other):
		return self <= other

	def __eq__(self, other):
		return self == other

	def munge(self, munger):
		return EntityStatAccessor(self.entity, self.stat, self.engine,
									self.branch, self.turn, self.tick,
									self.current, self.mungers + [munger])

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
		"""Iterate over all the values this stat has had in the given window, inclusive.

        """
		# It might be useful to do this in a way that doesn't change the engine's time, perhaps for thread safety
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
			if hasattr(y, 'unwrap'):
				y = y.unwrap()
			yield y
		engine.turn = oldturn
		engine.tick = oldtick


def dedent_source(source):
	nlidx = source.index('\n')
	if nlidx is None:
		raise ValueError("Invalid source")
	while source[:nlidx].strip().startswith('@'):
		source = source[nlidx + 1:]
		nlidx = source.index('\n')
	return dedent(source)


def _sort_set_key(v):
	if isinstance(v, tuple):
		return (2, ) + tuple(map(repr, v))
	if isinstance(v, str):
		return 1, repr(v)
	return 0, repr(v)


_sort_set_memo = {}


def sort_set(s):
	"""Return a sorted list of the contents of a set

    This is intended to be used to iterate over world state, where you just need keys
    to be in some deterministic order, but the sort order should be obvious from the key.

    Non-strings come before strings and then tuples. Tuples compare element-wise as normal.
    But ultimately all comparisons are between values' ``repr``.

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

    This is for testing. Use, eg., `@patch.object(executor, 'submit', new=fake_submit)`
    to make normally parallel operations serial.

    """

	class FakeFuture:

		def __init__(self, func, *args, **kwargs):
			self._func = func
			self._args = args
			self._kwargs = kwargs

		def result(self):
			return self._func(*self._args, **self._kwargs)

	return FakeFuture(func, *args, **kwargs)


class AbstractEngine(ABC):
	"""Parent class to the real Engine as well as EngineProxy.

    Implements serialization and the __getattr__ for stored methods.

    By default, the deserializers will refuse to create LiSE entities.
    If you want them to, use my ``loading`` property to open a ``with``
    block, in which deserialized entities will be created as needed.

    """

	def __getattr__(self, item):
		meth = super().__getattribute__('method').__getattr__(item)
		return MethodType(meth, self)

	@reify
	def pack(self):
		handlers = {
			self.char_cls:
			lambda char: msgpack.ExtType(MSGPACK_CHARACTER, packer(char.name)),
			self.place_cls:
			lambda place: msgpack.ExtType(
				MSGPACK_PLACE, packer([place.character.name, place.name])),
			self.thing_cls:
			lambda thing: msgpack.ExtType(
				MSGPACK_THING, packer([thing.character.name, thing.name])),
			self.portal_cls:
			lambda port: msgpack.ExtType(
				MSGPACK_PORTAL,
				packer([
					port.character.name, port.origin.name, port.destination.
					name
				])),
			tuple:
			lambda tup: msgpack.ExtType(MSGPACK_TUPLE, packer(list(tup))),
			frozenset:
			lambda frozs: msgpack.ExtType(MSGPACK_FROZENSET, packer(list(frozs)
																	)),
			set:
			lambda s: msgpack.ExtType(MSGPACK_SET, packer(list(s))),
			FinalRule:
			lambda obj: msgpack.ExtType(MSGPACK_FINAL_RULE, b""),
			FunctionType:
			lambda func: msgpack.ExtType({
				'method': MSGPACK_METHOD,
				'function': MSGPACK_FUNCTION,
				'trigger': MSGPACK_TRIGGER,
				'prereq': MSGPACK_PREREQ,
				'action': MSGPACK_ACTION
			}[func.__module__], packer(func.__name__)),
			MethodType:
			lambda meth: msgpack.ExtType(MSGPACK_METHOD, packer(meth.__name__)
											),
			Exception:
			lambda exc: msgpack.ExtType(
				MSGPACK_EXCEPTION,
				packer([exc.__class__.__name__] + list(exc.args)))
		}

		def pack_handler(obj):
			if isinstance(obj, Exception):
				typ = Exception
			else:
				typ = type(obj)
			if typ in handlers:
				return handlers[typ](obj)
			elif isinstance(obj, Mapping):
				return dict(obj)
			raise TypeError("Can't pack {}".format(typ))

		packer = partial(msgpack.packb,
							default=pack_handler,
							strict_types=True,
							use_bin_type=True)
		return packer

	@reify
	def unpack(self):
		charmap = self.character
		char_cls = self.char_cls
		place_cls = self.place_cls
		thing_cls = self.thing_cls
		portal_cls = self.portal_cls
		trigger = self.trigger
		prereq = self.prereq
		action = self.action
		function = self.function
		method = self.method
		excs = {
			# builtin exceptions
			'AssertionError': AssertionError,
			'AttributeError': AttributeError,
			'EOFError': EOFError,
			'FloatingPointError': FloatingPointError,
			'GeneratorExit': GeneratorExit,
			'ImportError': ImportError,
			'IndexError': IndexError,
			'KeyError': KeyError,
			'KeyboardInterrupt': KeyboardInterrupt,
			'MemoryError': MemoryError,
			'NameError': NameError,
			'NotImplementedError': NotImplementedError,
			'OSError': OSError,
			'OverflowError': OverflowError,
			'RecursionError': RecursionError,
			'ReferenceError': ReferenceError,
			'RuntimeError': RuntimeError,
			'StopIteration': StopIteration,
			'IndentationError': IndentationError,
			'TabError': TabError,
			'SystemError': SystemError,
			'SystemExit': SystemExit,
			'TypeError': TypeError,
			'UnboundLocalError': UnboundLocalError,
			'UnicodeError': UnicodeError,
			'UnicodeEncodeError': UnicodeEncodeError,
			'UnicodeDecodeError': UnicodeDecodeError,
			'UnicodeTranslateError': UnicodeTranslateError,
			'ValueError': ValueError,
			'ZeroDivisionError': ZeroDivisionError,
			# LiSE exceptions
			'NonUniqueError': exc.NonUniqueError,
			'AmbiguousAvatarError': exc.AmbiguousAvatarError,
			'AmbiguousUserError': exc.AmbiguousUserError,
			'RulesEngineError': exc.RulesEngineError,
			'RuleError': exc.RuleError,
			'RedundantRuleError': exc.RedundantRuleError,
			'UserFunctionError': exc.UserFunctionError,
			'WorldIntegrityError': exc.WorldIntegrityError,
			'CacheError': exc.CacheError,
			'TravelException': exc.TravelException,
			'OutOfTimelineError': exc.OutOfTimelineError,
			'HistoricKeyError': exc.HistoricKeyError
		}

		def unpack_exception(ext):
			data = unpacker(ext)
			if data[0] not in excs:
				return Exception(*data)
			return excs[data[0]](*data[1:])

		def unpack_char(ext):
			charn = unpacker(ext)
			try:
				return charmap[charn]
			except KeyError:
				return char_cls(self, charn, init_rulebooks=False)

		def unpack_place(ext):
			charn, placen = unpacker(ext)
			try:
				char = charmap[charn]
			except KeyError:
				return place_cls(char_cls(self, charn), placen)
			try:
				return char.place[placen]
			except KeyError:
				return place_cls(char, placen)

		def unpack_thing(ext):
			charn, thingn = unpacker(ext)
			try:
				char = charmap[charn]
			except KeyError:
				return thing_cls(char_cls(self, charn), thingn)
			try:
				return char.thing[thingn]
			except KeyError:
				return thing_cls(char, thingn)

		def unpack_portal(ext):
			charn, orign, destn = unpacker(ext)
			try:
				char = charmap[charn]
			except KeyError:
				char = char_cls(self, charn)
			try:
				return char.portal[orign][destn]
			except KeyError:
				return portal_cls(char, orign, destn)

		handlers = {
			MSGPACK_CHARACTER: unpack_char,
			MSGPACK_PLACE: unpack_place,
			MSGPACK_THING: unpack_thing,
			MSGPACK_PORTAL: unpack_portal,
			MSGPACK_FINAL_RULE: lambda obj: final_rule,
			MSGPACK_TUPLE: lambda ext: tuple(unpacker(ext)),
			MSGPACK_FROZENSET: lambda ext: frozenset(unpacker(ext)),
			MSGPACK_SET: lambda ext: set(unpacker(ext)),
			MSGPACK_TRIGGER: lambda ext: getattr(trigger, unpacker(ext)),
			MSGPACK_PREREQ: lambda ext: getattr(prereq, unpacker(ext)),
			MSGPACK_ACTION: lambda ext: getattr(action, unpacker(ext)),
			MSGPACK_FUNCTION: lambda ext: getattr(function, unpacker(ext)),
			MSGPACK_METHOD: lambda ext: getattr(method, unpacker(ext)),
			MSGPACK_EXCEPTION: unpack_exception
		}

		def unpack_handler(code, data):
			if code in handlers:
				return handlers[code](data)
			return msgpack.ExtType(code, data)

		def unpacker(b: bytes):
			the_unpacker = msgpack.Unpacker(ext_hook=unpack_handler,
											raw=False,
											strict_map_key=False)
			the_unpacker.feed(b)
			# Deliberately only returning the initial item;
			# others are likely to be null bytes as a result of the
			# way browsers work, and anyway if you really want more
			# you can just pack a list
			return the_unpacker.unpack()

		return unpacker

	def coinflip(self) -> bool:
		"""Return True or False with equal probability."""
		return self.choice((True, False))

	def dieroll(self, d) -> int:
		"""Roll a die with ``d`` faces. Return the result."""
		return self.randint(1, d)

	def dice(self, n, d) -> Iterable[int]:
		"""Roll ``n`` dice with ``d`` faces, and yield the results.

        This is an iterator. You'll get the result of each die in
        succession.

        """
		for i in range(0, n):
			yield self.dieroll(d)

	def dice_check(self,
					n: int,
					d: int,
					target: int,
					comparator: Union[str, Callable] = '<=') -> bool:
		"""Roll ``n`` dice with ``d`` sides, sum them, and compare

        If ``comparator`` is provided, use it instead of the default <=.
        You may use a string like '<' or '>='.

        """
		from operator import gt, lt, ge, le, eq, ne

		comps: Dict[str, Callable] = {
			'>': gt,
			'<': lt,
			'>=': ge,
			'<=': le,
			'=': eq,
			'==': eq,
			'!=': ne
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
		return pct / 100 < self.random()

	betavariate = getnoplan('_rando.betavariate')
	choice = getnoplan('_rando.choice')
	expovariate = getnoplan('_rando.expovariate')
	gammavariate = getnoplan('_rando.gammavariate')
	gauss = getnoplan('_rando.gauss')
	getrandbits = getnoplan('_rando.getrandbits')
	lognormvariate = getnoplan('_rando.lognormvariate')
	normalvariate = getnoplan('_rando.normalvariate')
	paretovariate = getnoplan('_rando.paretovariate')
	randint = getnoplan('_rando.randint')
	random = getnoplan('_rando.random')
	randrange = getnoplan('_rando.randrange')
	sample = getnoplan('_rando.sample')
	shuffle = getnoplan('_rando.shuffle')
	triangular = getnoplan('_rando.triangular')
	uniform = getnoplan('_rando.uniform')
	vonmisesvariate = getnoplan('_rando.vonmisesvariate')
	weibullvariate = getnoplan('_rando.weibullvariate')


class SpecialMappingDescriptor:

	def __init__(self, mapclsname):
		self.insts = WeakValueDictionary()
		self.mapps = {}
		self.mapclsname = mapclsname

	def __get__(self, instance, owner):
		if id(instance) in self.mapps:
			if id(instance) not in self.insts:
				del self.mapps[id(instance)]
			else:
				return self.mapps[id(instance)]
		self.insts[id(instance)] = instance
		mappcls = getattr(instance, self.mapclsname)
		ret = self.mapps[id(instance)] = mappcls(instance)
		return ret

	def __set__(self, instance, value):
		if id(instance) not in self.mapps:
			self.insts[id(instance)] = instance
			self.mapps[id(instance)] = getattr(instance,
												self.mapclsname)(instance)
		it = self.mapps[id(instance)]
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
	engine = getatt('db')
	no_unwrap = True
	name: Hashable
	db: AbstractEngine

	@staticmethod
	def is_directed():
		return True

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
	def thing2place(self, name):
		pass

	@abstractmethod
	def place2thing(self, name, location):
		pass

	@abstractmethod
	def add_portal(self, orig, dest, symmetrical=False, **kwargs):
		pass

	def add_edge(self, orig, dest, **kwargs):
		self.add_portal(orig, dest, **kwargs)

	def new_portal(self, orig, dest, symmetrical=False, **kwargs):
		self.add_portal(orig, dest, symmetrical, **kwargs)
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
		return isinstance(other, AbstractCharacter) \
                           and self.name == other.name

	def __iter__(self):
		return iter(self.node)

	def __len__(self):
		return len(self.node)

	def __bool__(self):
		return self.name in self.db.character

	def __contains__(self, k):
		return k in self.node

	def __getitem__(self, k):
		return self.adj[k]

	thing = SpecialMappingDescriptor('ThingMapping')
	place = SpecialMappingDescriptor('PlaceMapping')
	node = nodes = _node = SpecialMappingDescriptor('ThingPlaceMapping')
	portal = adj = succ = edge = _adj = _succ = SpecialMappingDescriptor(
		'PortalSuccessorsMapping')
	preportal = pred = _pred = SpecialMappingDescriptor(
		'PortalPredecessorsMapping')
	unit = SpecialMappingDescriptor('UnitGraphMapping')
	stat = getatt('graph')

	def historical(self, stat):
		from .query import StatusAlias
		return StatusAlias(entity=self.stat, stat=stat)

	def do(self, func, *args, **kwargs):
		"""Apply the function to myself, and return myself.

        Look up the function in the database if needed. Pass it any
        arguments given, keyword or positional.

        Useful chiefly when chaining.

        """
		if not callable(func):
			func = getattr(self.engine.function, func)
		func(self, *args, **kwargs)
		return self

	def perlin(self, stat='perlin'):
		"""Apply Perlin noise to my nodes, and return myself.

        I'll try to use the name of the node as its spatial position
        for this purpose, or use its stats 'x', 'y', and 'z', or skip
        the node if neither are available. z is assumed 0 if not
        provided for a node.

        Result will be stored in a node stat named 'perlin' by default.
        Supply the name of another stat to use it instead.

        """
		from math import floor
		p = self.engine.shuffle([
			151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7,
			225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6,
			148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35,
			11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171,
			168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158,
			231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55,
			46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73,
			209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188,
			159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250,
			124, 123, 5, 202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206,
			59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119,
			248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9,
			129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185,
			112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12,
			191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49, 192,
			214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45,
			127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243,
			141, 128, 195, 78, 66, 215, 61, 156, 180
		]) * 2

		def fade(t):
			return t * t * t * (t * (t * 6 - 15) + 10)

		def lerp(t, a, b):
			return a + t * (b - a)

		def grad(hsh, x, y, z):
			"""CONVERT LO 4 BITS OF HASH CODE INTO 12 GRADIENT DIRECTIONS."""
			h = hsh & 15
			u = x if h < 8 else y
			v = y if h < 4 else x if h == 12 or h == 14 else z
			return (u if h & 1 == 0 else -u) + (v if h & 2 == 0 else -v)

		def noise(x, y, z):
			# FIND UNIT CUBE THAT CONTAINS POINT.
			X = int(x) & 255
			Y = int(y) & 255
			Z = int(z) & 255
			# FIND RELATIVE X, Y, Z OF POINT IN CUBE.
			x -= floor(x)
			y -= floor(y)
			z -= floor(z)
			# COMPUTE FADE CURVES FOR EACH OF X, Y, Z.
			u = fade(x)
			v = fade(y)
			w = fade(z)
			# HASH COORDINATES OF THE 8 CUBE CORNERS,
			A = p[X] + Y
			AA = p[A] + Z
			AB = p[A + 1] + Z
			B = p[X + 1] + y
			BA = p[B] + Z
			BB = p[B + 1] + Z
			# AND ADD BLENDED RESULTS FROM 8 CORNERS OF CUBE
			return lerp(
				w,
				lerp(
					v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)),
					lerp(u, grad(p[AB], x, y - 1, z),
							grad(p[BB], x - 1, y - 1, z))),
				lerp(
					v,
					lerp(u, grad(p[AA + 1], x, y, z - 1),
							grad(p[BA + 1], x - 1, y, z - 1)),
					lerp(u, grad(p[AB + 1], x, y - 1, z - 1),
							grad(p[BB + 1], x - 1, y - 1, z - 1))))

		for node in self.node.values():
			try:
				(x, y, z) = node.name
			except ValueError:
				try:
					(x, y) = node.name
					z = 0.0
				except ValueError:
					try:
						x = node['x']
						y = node['y']
						z = node.get('z', 0.0)
					except KeyError:
						continue
			x, y, z = map(float, (x, y, z))
			node[stat] = noise(x, y, z)

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
					k = ok + (n, ) if isinstance(ok, tuple) else (ok, n)
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
		ops = {'ge': ge, 'gt': gt, 'le': le, 'lt': lt, 'eq': eq}
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
			name for name, node in self.node.items()
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
					self.portal[u][v][stat], threshold):
					dead.append((u, v))
		self.remove_edges_from(dead)
		return self

	def cull_edges(self, stat, threshold=0.5, comparator=ge):
		"""Delete edges whose stat >= ``threshold`` (default 0.5).

        Optional argument ``comparator`` will replace >= as the test
        for whether to cull. You can use the name of a stored function.

        """
		return self.cull_portals(stat, threshold, comparator)
