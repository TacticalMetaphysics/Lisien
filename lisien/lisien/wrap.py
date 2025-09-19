# This file is part of allegedb, an object relational mapper for versioned graphs.
# Copyright (C) Zachary Spector. public@zacharyspector.com
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
"""Wrapper classes to let you store mutable data types in the allegedb ORM

The wrapper objects act like regular mutable objects, but write a new copy
of themselves to allegedb every time they are changed.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import (
	Container,
	Iterable,
	Mapping,
	MutableMapping,
	MutableSequence,
	Sequence,
	Sized,
)
from functools import partial
from itertools import chain, zip_longest
from typing import TYPE_CHECKING, Any, Callable, Hashable, Set

from blinker import Signal

if TYPE_CHECKING:
	from .util import AbstractCharacter


class OrderlySet(set):
	"""A set with deterministic order of iteration

	Order is not regarded as significant for the purposes of equality.

	"""

	def __init__(self, data: Iterable[Hashable] = ()):
		super().__init__()
		self._data = {}
		for k in data:
			self._data[k] = True

	def copy(self):
		return OrderlySet(self._data.copy())

	def difference(self, *s):
		dat = self._data.copy()
		for subtracted in dat.keys() - s:
			del dat[subtracted]
		ret = OrderlySet()
		ret._data = dat
		return ret

	def difference_update(self, *s):
		for k in s:
			if k in self._data:
				del self._data[k]

	def intersection(self, *s):
		return OrderlySet(self._data.keys() & s)

	def intersection_update(self, *s):
		for k in list(self._data.keys()):
			if k not in s:
				del self._data[k]

	def issubset(self, __s):
		if not isinstance(__s, Set):
			__s = set(__s)
		return self._data.keys() <= __s

	def issuperset(self, __s):
		if not isinstance(__s, Set):
			__s = set(__s)
		return self._data.keys() >= __s

	def symmetric_difference(self, __s):
		if not isinstance(__s, Set):
			__s = set(__s)
		return OrderlySet(self._data.keys() ^ __s)

	def symmetric_difference_update(self, __s):
		if not isinstance(__s, Set):
			__s = set(__s)
		self._data = {k: True for k in self._data.keys() ^ __s}

	def union(self, *s):
		for k in s:
			self._data[k] = True

	def __repr__(self):
		return repr(set(self))

	def __iter__(self):
		return iter(self._data.keys())

	def __len__(self):
		return len(self._data)

	def __contains__(self, item):
		return item in self._data

	def __eq__(self, other):
		if not isinstance(other, Set):
			return False
		if len(self) != len(other):
			return False
		return self._data.keys() == other

	def __ne__(self, other):
		if not isinstance(other, Set):
			return True
		return self._data.keys() != other

	def __isub__(self, it):
		for item in it:
			self.discard(item)

	def __ixor__(self, it):
		for item in list(self):
			if item in it:
				self.remove(item)
		for item in it:
			if item in self:
				self.remove(item)
			else:
				self.add(item)

	def __iand__(self, it):
		for item in list(self):
			if item not in it:
				self.remove(item)

	def __ior__(self, it):
		self.update(it)

	def remove(self, value):
		del self._data[value]

	def pop(self):
		k, _ = self._data.popitem()
		return k

	def clear(self):
		self._data.clear()

	def add(self, item):
		self._data[item] = True

	def discard(self, value):
		if value in self._data:
			del self._data[value]

	def update(self, it):
		for item in it:
			self.add(item)

	def __copy__(self):
		return self.copy()

	def __le__(self, other):
		if not isinstance(other, Set):
			return False
		if isinstance(other, OrderlySet):
			return self._data.keys() <= other._data.keys()
		return self._data.keys() <= other

	def __lt__(self, other):
		if not isinstance(other, Set):
			return False
		if isinstance(other, OrderlySet):
			return self._data.keys() < other._data.keys()
		return self._data.keys() < other

	def __gt__(self, other):
		if not isinstance(other, Set):
			return False
		if isinstance(other, OrderlySet):
			return self._data.keys() > other._data.keys()
		return self._data.keys() > other

	def __ge__(self, other):
		if not isinstance(other, Set):
			return False
		if isinstance(other, OrderlySet):
			return self._data.keys() >= other._data.keys()
		return self._data.keys() >= other

	def __and__(self, other):
		if isinstance(other, Set):
			if isinstance(other, OrderlySet):
				other = other._data.keys()
			intersection = self._data.keys() & other
			return OrderlySet(
				datum for datum in self._data.keys() if datum in intersection
			)
		return OrderlySet(
			datum for datum in self._data.keys() if datum in other
		)

	def __or__(self, other):
		if isinstance(other, OrderlySet):
			ret = OrderlySet()
			ret._data = {**self._data, **other._data}
			return ret
		return OrderlySet(chain(self, other))

	def __sub__(self, other):
		return OrderlySet(k for k in self if k not in other)

	def __xor__(self, other):
		if isinstance(other, Set):
			if isinstance(other, OrderlySet):
				other = other._data.keys()
			split = self._data.keys() ^ other
			return OrderlySet(
				chain(
					(k for k in self._data if k in split),
					(k for k in other if k in split),
				)
			)
		return OrderlySet(
			chain(
				(k for k in self if k not in other),
				(k for k in other if k not in self),
			)
		)

	def isdisjoint(self, other):
		if isinstance(other, OrderlySet):
			return self._data.keys().isdisjoint(other._data.keys())
		return super().isdisjoint(other)


class OrderlyFrozenSet(frozenset):
	"""A frozenset with deterministic order of iteration

	Order is not considered significant for the purpose of determining
	equality.

	"""

	def __init__(self, data):
		self._data = tuple(data)
		super().__init__(data)

	def __iter__(self):
		return iter(self._data)

	def __repr__(self):
		return repr(frozenset(self))

	def copy(self):
		return OrderlyFrozenSet(self._data)

	def difference(self, *s):
		diffed = super().difference(s)
		return OrderlyFrozenSet(
			datum for datum in self._data if datum in diffed
		)

	def intersection(self, *s):
		intersected = super().intersection(s)
		return OrderlyFrozenSet(
			datum for datum in self._data if datum in intersected
		)

	def union(self, *s):
		unified = super().union(*s)
		return OrderlyFrozenSet(
			datum for datum in self._data if datum in unified
		)

	def __xor__(self, __value):
		return OrderlyFrozenSet(
			*(datum for datum in self._data if datum not in __value),
			*(datum for datum in __value if datum not in self),
		)

	def __and__(self, __value):
		intersected = super().__and__(__value)
		return OrderlyFrozenSet(
			datum for datum in self._data if datum in intersected
		)

	def __or__(self, __value):
		return OrderlyFrozenSet(
			(
				*self._data,
				*(datum for datum in __value if datum not in self),
			)
		)

	def __sub__(self, __value):
		subtracted = super().__sub__(__value)
		return OrderlyFrozenSet(
			datum for datum in self._data if datum in subtracted
		)

	def symmetric_difference(self, __s):
		return OrderlyFrozenSet(
			*(datum for datum in self._data if datum not in __s),
			*(datum for datum in __s if datum not in self),
		)


class SpecialMapping(Mapping, Signal, ABC):
	@abstractmethod
	def __init__(self, character: AbstractCharacter): ...


class MutableWrapper(ABC):
	__slots__ = ()

	def __iter__(self):
		return iter(self._getter())

	def __len__(self):
		return len(self._getter())

	def __contains__(self, item):
		return item in self._getter()

	def __repr__(self):
		return "<{} instance at {}, wrapping {}>".format(
			self.__class__.__name__, id(self), self._getter()
		)

	def __str__(self):
		return str(self._getter())

	@abstractmethod
	def _getter(self):
		raise NotImplementedError

	@abstractmethod
	def _copy(self):
		raise NotImplementedError

	@abstractmethod
	def _set(self, v):
		raise NotImplementedError

	@abstractmethod
	def unwrap(self):
		raise NotImplementedError


Iterable.register(MutableWrapper)
Sized.register(MutableWrapper)
Container.register(MutableWrapper)


class MutableWrapperDictList(MutableWrapper, ABC):
	__slots__ = ()

	def _subset(self, k, v):
		new = self._copy()
		new[k] = v
		self._set(new)

	def __getitem__(self, k):
		ret = self._getter()[k]
		if isinstance(ret, dict):
			return SubDictWrapper(
				lambda: self._getter()[k], partial(self._subset, k)
			)
		if isinstance(ret, list):
			return SubListWrapper(
				lambda: self._getter()[k], partial(self._subset, k)
			)
		if isinstance(ret, set):
			return SubSetWrapper(
				lambda: self._getter()[k], partial(self._subset, k)
			)
		return ret

	def __setitem__(self, key, value):
		me = self._copy()
		me[key] = value
		self._set(me)

	def __delitem__(self, key):
		me = self._copy()
		del me[key]
		self._set(me)


class MutableMappingUnwrapper(MutableMapping, ABC):
	__slots__ = ()

	def __eq__(self, other):
		if self is other:
			return True
		if not isinstance(other, Mapping):
			return False
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

	def __repr__(self):
		return f"<{type(self).__name__} {unwrap_items(self.items())}>"


class MutableMappingWrapper(
	MutableWrapperDictList, MutableMappingUnwrapper, ABC
):
	def __eq__(self, other):
		return MutableMappingUnwrapper.__eq__(self, other)

	def unwrap(self):
		return unwrap_items(self.items())


class SubDictWrapper(MutableMappingWrapper, dict):
	__slots__ = ("_getter", "_set")
	_getter: Callable
	_set: Callable

	def __init__(self, getter, setter):
		super().__init__()
		self._getter = getter
		self._set = setter

	def _copy(self):
		return dict(self._getter())

	def _subset(self, k, v):
		new = dict(self._getter())
		new[k] = v
		self._set(new)


class MutableSequenceWrapper(MutableWrapperDictList, MutableSequence, ABC):
	def __eq__(self, other):
		if self is other:
			return True
		if not isinstance(other, Sequence):
			return NotImplemented
		for me, you in zip_longest(self, other):
			if hasattr(me, "unwrap"):
				me = me.unwrap()
			if hasattr(you, "unwrap"):
				you = you.unwrap()
			if me != you:
				return False
		else:
			return True

	def unwrap(self):
		"""Deep copy myself as a list, all contents unwrapped"""
		return [v.unwrap() if hasattr(v, "unwrap") else v for v in self]


class SubListWrapper(MutableSequenceWrapper, list):
	__slots__ = ("_getter", "_set")
	_getter: Callable
	_set: Callable

	def __init__(self, getter, setter):
		super().__init__()
		self._getter = getter
		self._set = setter

	def _copy(self):
		return list(self._getter())

	def insert(self, index, object):
		me = self._copy()
		me.insert(index, object)
		self._set(me)

	def append(self, object):
		me = self._copy()
		me.append(object)
		self._set(me)


class MutableWrapperSet(MutableWrapper, set):
	__slots__ = ()
	_getter: Callable
	_set: Callable

	def _copy(self):
		return OrderlySet(self._getter())

	def pop(self):
		me = self._copy()
		yours = me.pop()
		self._set(me)
		return yours

	def discard(self, element):
		me = self._copy()
		me.discard(element)
		self._set(me)

	def remove(self, element):
		me = self._copy()
		me.remove(element)
		self._set(me)

	def add(self, element):
		me = self._copy()
		me.add(element)
		self._set(me)

	def unwrap(self):
		"""Deep copy myself as a set, all contents unwrapped"""
		unwrapped = OrderlySet()
		for v in self:
			if hasattr(v, "unwrap") and not hasattr(v, "no_unwrap"):
				unwrapped.add(v.unwrap())
			else:
				unwrapped.add(v)
		return unwrapped

	def clear(self):
		self._set(OrderlySet())

	def __repr__(self):
		return f"<{type(self).__name__} containing {set(self._getter())}>"

	def __ior__(self, it):
		me = self._copy()
		me |= it
		self._set(me)

	def __iand__(self, it):
		me = self._copy()
		me &= it
		self._set(me)

	def __ixor__(self, it):
		me = self._copy()
		me ^= it
		self._set(me)

	def __isub__(self, it):
		me = self._copy()
		me -= it
		self._set(me)

	def __le__(self, other):
		return self._getter() <= other

	def __lt__(self, other):
		return self._getter() < other

	def __gt__(self, other):
		return self._getter() > other

	def __ge__(self, other):
		return self._getter() >= other

	def __and__(self, other):
		return OrderlySet(self._getter() & other)

	def __or__(self, other):
		return OrderlySet(self._getter() | other)

	def __sub__(self, other):
		return OrderlySet(self._getter() - other)

	def __xor__(self, other):
		return OrderlySet(self._getter() ^ other)

	def __eq__(self, other):
		return self._getter() == other

	def isdisjoint(self, other):
		return self._getter().isdisjoint(other)


class SubSetWrapper(MutableWrapperSet):
	__slots__ = ("_getter", "_set")
	_getter: Callable
	_set: Callable

	def __init__(self, getter, setter):
		super().__init__()
		self._getter = getter
		self._set = setter

	def _copy(self):
		return OrderlySet(self._getter())


def unwrap_items(it: Iterable[tuple[Any, Any]]) -> dict:
	ret = {}
	for k, v in it:
		if hasattr(v, "unwrap") and not hasattr(v, "no_unwrap"):
			ret[k] = v.unwrap()
		else:
			ret[k] = v
	return ret


class DictWrapper(MutableMappingWrapper, dict):
	"""A dictionary synchronized with a serialized field.

	This is meant to be used in allegedb entities (graph, node, or
	edge), for when the user stores a dictionary in them.

	"""

	__slots__ = ("_getter", "_outer", "_key")
	_getter: Callable

	def __init__(self, getter, outer, key):
		super().__init__()
		self._getter = getter
		self._outer = outer
		self._key = key

	def _copy(self):
		return dict(self._getter())

	def _set(self, v):
		self._outer[self._key] = v


class ListWrapper(MutableWrapperDictList, MutableSequence):
	"""A list synchronized with a serialized field.

	This is meant to be used in allegedb entities (graph, node, or
	edge), for when the user stores a list in them.

	"""

	__slots__ = ("_getter", "_outer", "_key")

	def __init__(self, getter, outer, key):
		self._outer = outer
		self._key = key
		self._getter = getter

	def __eq__(self, other):
		if self is other:
			return True
		if not isinstance(other, Sequence):
			return NotImplemented
		for me, you in zip_longest(self, other):
			if hasattr(me, "unwrap"):
				me = me.unwrap()
			if hasattr(you, "unwrap"):
				you = you.unwrap()
			if me != you:
				return False
		else:
			return True

	def _copy(self):
		return list(self._getter())

	def _set(self, v):
		self._outer[self._key] = v

	def insert(self, i, v):
		new = self._copy()
		new.insert(i, v)
		self._set(new)

	def append(self, v):
		new = self._copy()
		new.append(v)
		self._set(new)

	def unwrap(self):
		"""Deep copy myself as a list, with all contents unwrapped"""
		return [
			v.unwrap()
			if hasattr(v, "unwrap") and not hasattr(v, "no_unwrap")
			else v
			for v in self
		]


class SetWrapper(MutableWrapperSet):
	"""A set synchronized with a serialized field.

	This is meant to be used in allegedb entities (graph, node, or
	edge), for when the user stores a set in them.

	"""

	__slots__ = ("_getter", "_outer", "_key")
	_getter: Callable

	def __init__(self, getter, outer, key):
		super().__init__()
		self._getter = getter
		self._outer = outer
		self._key = key

	def _set(self, v):
		self._outer[self._key] = v


class UnwrappingDict(dict):
	"""Dict that stores the data from the wrapper classes

	Won't store those objects themselves.

	"""

	def __setitem__(self, key, value):
		if isinstance(value, MutableWrapper):
			value = value.unwrap()
		super(UnwrappingDict, self).__setitem__(key, value)


def wrapval(self, key, v):
	if isinstance(v, list):
		return ListWrapper(
			partial(self._get_cache_now, key),
			self,
			key,
		)
	elif isinstance(v, dict):
		return DictWrapper(
			partial(self._get_cache_now, key),
			self,
			key,
		)
	elif isinstance(v, set):
		return SetWrapper(
			partial(self._get_cache_now, key),
			self,
			key,
		)
	else:
		return v
