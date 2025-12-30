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
"""Wrapper classes to let you store mutable data types in Lisien

The wrapper objects act like regular mutable objects, but write a new copy
of themselves to Lisien every time they are changed.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import (
	Iterable,
	Mapping,
	MutableMapping,
	MutableSequence,
	Sequence,
	Collection,
	Container,
	MutableSet,
	Set,
)
from functools import partial, wraps
from itertools import chain, zip_longest, filterfalse
from threading import RLock
from typing import Callable, Hashable, TypeVar, Self, Iterator

from attrs import define, field
from more_itertools import unique_everseen


def rlocked(f):
	@wraps(f)
	def locked(self, *args, **kwargs):
		with self._lock:
			return f(self, *args, **kwargs)

	return locked


class AbstractOrderlySet[_K](Set[_K]):
	__slots__ = ()

	@abstractmethod
	def _get(self) -> tuple[_K, ...]: ...

	@abstractmethod
	def _set(self, data: tuple[_K, ...]) -> None: ...

	@rlocked
	def __iter__(self) -> Iterator[_K]:
		for what in self._get():
			yield what

	@rlocked
	def __len__(self) -> int:
		return len(self._get())

	@rlocked
	def __contains__(self, item):
		return item in self._get()

	@rlocked
	def __eq__(self, other):
		return frozenset(self._get()) == other

	@rlocked
	def __ne__(self, other):
		return frozenset(self._get()) != other

	@rlocked
	def __gt__(self, other):
		return frozenset(self._get()) > other

	@rlocked
	def __ge__(self, other):
		return frozenset(self._get()) >= other

	@rlocked
	def __lt__(self, other):
		return frozenset(self._get()) < other

	@rlocked
	def __le__(self, other):
		return frozenset(self._get()) <= other

	@rlocked
	def copy(self) -> OrderlyFrozenSet[_K]:
		return OrderlyFrozenSet(self._get())

	def __copy__(self) -> OrderlyFrozenSet[_K]:
		return self.copy()

	@rlocked
	def __and__(self, other) -> OrderlyFrozenSet[_K]:
		return OrderlyFrozenSet(filter(other.__contains__, self._get()))

	@rlocked
	def __or__(self, other) -> OrderlyFrozenSet[_K]:
		return OrderlyFrozenSet(chain(self, other))

	@rlocked
	def __sub__(self, other) -> OrderlyFrozenSet[_K]:
		return OrderlyFrozenSet(filterfalse(other.__contains__, self._get()))

	@rlocked
	def __xor__(self, other) -> OrderlyFrozenSet[_K]:
		this = self._get()
		that = frozenset(this)
		excluded = that ^ other
		return OrderlyFrozenSet(
			filter(excluded.__contains__, chain(this, other))
		)

	@rlocked
	def difference(self, *others) -> OrderlyFrozenSet[_K]:
		this = self._get()
		that = set(this)
		for it in others:
			for what in it:
				that.discard(what)
		return OrderlyFrozenSet(filter(that.__contains__, this))

	@rlocked
	def intersection(self, *others) -> OrderlyFrozenSet[_K]:
		this = self._get()
		that = set(this)
		for it in others:
			that.intersection_update(it)
		return OrderlyFrozenSet(filter(that.__contains__, this))

	@rlocked
	def issubset(self, other) -> bool:
		for k in self._get():
			if k not in other:
				return False
		return True

	@rlocked
	def issuperset(self, other) -> bool:
		this = self._get()
		for k in other:
			if k not in this:
				return False
		return True

	@rlocked
	def symmetric_difference(self, other, /) -> OrderlyFrozenSet[_K]:
		this = self._get()
		if not isinstance(other, Container):
			other = set(other)
		return OrderlyFrozenSet(filter(other.__contains__, this))

	@rlocked
	def union(self, *others) -> OrderlyFrozenSet[_K]:
		return OrderlyFrozenSet(chain(self._get(), *others))

	@rlocked
	def isdisjoint(self, other) -> bool:
		return set(self._get()).isdisjoint(other)


class AbstractOrderlyMutableSet[_K](AbstractOrderlySet[_K], MutableSet[_K]):
	"""A set with deterministic order of iteration

	Order is not regarded as significant for the purposes of equality.

	"""

	__slots__ = ()

	@abstractmethod
	def _get(self) -> dict[_K, bool]: ...

	@abstractmethod
	def _set(self, data: dict[_K, bool]) -> None: ...

	@rlocked
	def add(self, item: Hashable) -> None:
		this = self._get()
		this[item] = True
		self._set(this)

	@rlocked
	def discard(self, value):
		this = self._get()
		if value in this:
			del this[value]
		self._set(this)

	@rlocked
	def remove(self, value):
		this = self._get()
		del this[value]
		self._set(this)

	@rlocked
	def pop(self) -> _K:
		this = self._get()
		k, _ = this.popitem()
		return k

	@rlocked
	def clear(self) -> None:
		self._set({})

	@rlocked
	def copy(self):
		return OrderlySet(self._get())

	def __copy__(self):
		return self.copy()

	@rlocked
	def __eq__(self, other) -> bool:
		return self._this_keys() == other

	@rlocked
	def __ne__(self, other) -> bool:
		return self._this_keys() != other

	@rlocked
	def __le__(self, other) -> bool:
		return self._this_keys() <= other

	@rlocked
	def __lt__(self, other) -> bool:
		return self._this_keys() < other

	@rlocked
	def __gt__(self, other) -> bool:
		return self._this_keys() > other

	@rlocked
	def __ge__(self, other) -> bool:
		return self._this_keys() >= other

	@rlocked
	def difference(self, *others) -> OrderlySet[_K]:
		this = self._this_keys()
		that = set(this)
		for it in others:
			for what in it:
				that.discard(what)
		return OrderlySet(filter(that.__contains__, this))

	@rlocked
	def difference_update(self, *others) -> None:
		this = self._this_keys()
		that = set(this)
		for it in others:
			that.difference_update(it)
		self._set(dict.fromkeys(filter(that.__contains__, this), True))

	@rlocked
	def intersection(self, *others) -> OrderlySet[_K]:
		this = self._this_keys()
		that = set(this)
		for it in others:
			that.intersection_update(it)
		return OrderlySet(filter(that.__contains__, this))

	@rlocked
	def intersection_update(self, *others) -> None:
		this = self._this_keys()
		that = set(this)
		for it in others:
			that.intersection_update(it)
		self._set(dict.fromkeys(filter(that.__contains__, this), True))

	@rlocked
	def issubset(self, __s) -> bool:
		this = self._this_keys()
		if isinstance(__s, Set):
			return this <= __s
		for k in this:
			if k not in __s:
				return False
		return True

	@rlocked
	def issuperset(self, __s) -> bool:
		this = self._this_keys()
		if isinstance(__s, Set):
			return this >= __s
		for k in __s:
			if k not in this:
				return False
		return True

	@rlocked
	def symmetric_difference(self, s, /) -> OrderlySet[_K]:
		this = self._this_keys()
		if not isinstance(s, Container):
			s = set(s)
		elif not isinstance(s, Set):
			return OrderlySet(filter(s.__contains__, this))
		that = this.keys() ^ s
		return OrderlySet(filter(that.__contains__, this))

	@rlocked
	def symmetric_difference_update(self, s, /):
		this = self._this_keys()
		if not isinstance(s, Container):
			s = set(s)
		elif not isinstance(s, Set):
			return self._set(dict.fromkeys(filter(s.__contains__, this), True))
		that = this ^ s
		self._set(dict.fromkeys(filter(that.__contains, this), True))

	@rlocked
	def union(self, *others) -> OrderlySet[_K]:
		return OrderlySet(chain(self._get(), *others))

	@rlocked
	def update(self, *others) -> None:
		this = self._get()
		this.update(dict.fromkeys(chain(others), True))
		self._set(this)

	@rlocked
	def isdisjoint(self, other) -> bool:
		this = self._this_keys()
		if isinstance(other, AbstractOrderlyMutableSet):
			return this.isdisjoint(other._get().keys())
		return this.isdisjoint(other)

	@rlocked
	def __iter__(self) -> Iterator[_K]:
		for what in self._get():
			yield what

	def _this_keys(self) -> Set[_K]:
		this = self._get()
		if hasattr(this, "keys"):
			return this.keys()
		if not isinstance(this, set):
			raise TypeError("Not a set", this)
		return this

	@rlocked
	def __isub__(self, other) -> Self:
		this = self._this_keys()
		that = this.keys() - other
		self._set(dict.fromkeys(filter(that.__contains__, this), True))
		return self

	@rlocked
	def __ixor__(self, other) -> Self:
		this = self._this_keys()
		that = this ^ other
		self._set(dict.fromkeys(filter(that.__contains__, this), True))
		return self

	@rlocked
	def __iand__(self, other) -> Self:
		this = self._this_keys()
		that = this & other
		self._set(dict.fromkeys(filter(that.__contains__, this), True))
		return self

	@rlocked
	def __ior__(self, other) -> Self:
		self.update(other)
		return self

	@rlocked
	def __and__(self, other) -> OrderlySet[_K]:
		return OrderlySet(filter(other.__contains__, self._this_keys()))

	@rlocked
	def __or__(self, other) -> OrderlySet[_K]:
		return OrderlySet(chain(self, other))

	@rlocked
	def __sub__(self, other) -> OrderlySet[_K]:
		return OrderlySet(filterfalse(other.__contains__, self._get()))

	@rlocked
	def __xor__(self, other) -> OrderlySet[_K]:
		if isinstance(other, OrderlySet):
			this = self._this_keys()
			that = other._this_keys()
			excluded = this ^ that
			return OrderlySet(filter(excluded.__contains__, chain(this, that)))
		elif isinstance(other, Set):
			this = self._this_keys()
			excluded = this ^ other
			return OrderlySet(
				filter(excluded.__contains__, chain(this, other))
			)
		this = self._this_keys()
		that = set(other)
		excluded = this ^ that
		return OrderlySet(filter(excluded.__contains__, chain(this, that)))


class OrderlySet[_K](AbstractOrderlyMutableSet[_K], set[_K]):
	"""A set with deterministic order of iteration

	Iterates in insertion order.

	Order is not considered significant for the purpose of determining
	equality.

	"""

	__slots__ = ("_data", "_lock")

	_data: dict[_K, bool]

	def __new__(cls, data: Iterable[_K] = ()):
		data = dict.fromkeys(data)
		me = set.__new__(cls, data)
		me._data = data
		me._lock = RLock()
		return me

	def __init__(self, data: Iterable[_K] = ()):
		super().__init__()

	def __repr__(self):
		return f"OrderlySet({list(self._data)})"

	def _get(self) -> dict[_K, bool]:
		return self._data

	def _set(self, data: dict[_K, bool]):
		set.clear(self)
		set.update(self, data)
		self._data = data


class OrderlyFrozenSet[_K](AbstractOrderlySet[_K], frozenset):
	"""A frozenset with deterministic order of iteration

	Iterates in the same order as the data it was constructed with.
	(Repeated elements are discarded.)

	Order is not considered significant for the purpose of determining
	equality.

	"""

	__slots__ = ("_data", "_lock")

	_data: tuple[_K, ...]

	def __new__(cls, data: Iterable[_K] = ()):
		data = tuple(unique_everseen(data))
		me = frozenset.__new__(cls, data)
		me._data = data
		me._lock = RLock()
		return me

	def __init__(self, data: Iterable[_K] = ()):
		super().__init__()

	def __repr__(self):
		return f"OrderlyFrozenSet({self._data})"

	def _get(self) -> tuple[_K, ...]:
		return self._data

	def _set(self, data: tuple[_K, ...]) -> None:
		raise TypeError("Tried to set data of a frozen set", self, data)


class MutableWrapper[_K: Hashable, _V](Collection, ABC):
	__slots__ = ()

	def __iter__(self):
		return iter(self._get())

	def __len__(self):
		return len(self._get())

	def __contains__(self, item):
		return item in self._get()

	def __str__(self):
		return str(self._get())

	@abstractmethod
	def __copy__(self):
		raise NotImplementedError

	def copy(self):
		return self.__copy__()

	@abstractmethod
	def _get(
		self,
	) -> MutableMapping[_K, _V] | MutableSequence[_V] | MutableSet[_K]: ...

	@abstractmethod
	def _set(
		self, v: MutableMapping[_K, _V] | MutableSequence[_V] | MutableSet[_K]
	): ...

	@abstractmethod
	def unwrap(
		self,
	) -> MutableMapping[_K, _V] | MutableSequence[_V] | MutableSet[_K]: ...


class MutableWrapperDictList[_K, _V](MutableWrapper[_K, _V], ABC):
	__slots__ = ()

	def _subset(self, k: _K, v: _V) -> None:
		new = self.__copy__()
		new[k] = v
		self._set(new)

	def __getitem__(
		self, k: _K
	) -> _V | SubDictWrapper[_K, _V] | SubListWrapper[_V] | SubSetWrapper[_V]:
		ret = self._get()[k]
		if isinstance(ret, dict):
			return SubDictWrapper(
				lambda: self._get()[k], partial(self._subset, k)
			)
		if isinstance(ret, list):
			return SubListWrapper(
				lambda: self._get()[k], partial(self._subset, k)
			)
		if isinstance(ret, set):
			return SubSetWrapper(
				lambda: self._get()[k], partial(self._subset, k)
			)
		return ret

	def __setitem__(self, key: _K, value: _V) -> None:
		me = self.__copy__()
		me[key] = value
		self._set(me)

	def __delitem__(self, key: _K) -> None:
		me = self.__copy__()
		del me[key]
		self._set(me)


class MappingUnwrapper[_K, _V](Mapping[_K, _V], ABC):
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

	def unwrap(self):
		return unwrap_items(self.items())


class MutableMappingWrapper[_K, _V](
	MappingUnwrapper[_K, _V],
	MutableWrapperDictList[_K, _V],
	MutableMapping[_K, _V],
	ABC,
):
	__slots__ = ()


@define(eq=False)
class SubDictWrapper[_K, _V](MutableMappingWrapper[_K, _V], dict[_K, _V]):
	__slots__ = ("_rlock",)
	_getter: Callable[[], dict[_K, _V]]
	_setter: Callable[[dict[_K, _V]], None]

	def _get(self) -> dict[_K, _V]:
		return self._getter()

	def _set(self, v: dict[_K, _V]):
		self._setter(v)

	def __copy__(self):
		return dict(self._get())

	def _subset(self, k: _K, v: _V) -> None:
		new = dict(self._get())
		new[k] = v
		self._set(new)


class MutableSequenceWrapper[_T](
	MutableWrapperDictList[int, _T], MutableSequence[_T], ABC
):
	__slots__ = ()

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


@define(eq=False)
class SubListWrapper[_T](MutableSequenceWrapper[_T], list[_T]):
	_getter: Callable[[], list[_T]]
	_setter: Callable[[list[_T]], None]

	def _get(self) -> list[_T]:
		return self._getter()

	def _set(self, v: list[_T]) -> None:
		self._setter(v)

	def __copy__(self) -> list[_T]:
		return list(self._get())

	def insert(self, index: int, object) -> None:
		me = self.__copy__()
		me.insert(index, object)
		self._set(me)

	def append(self, object) -> None:
		me = self.__copy__()
		me.append(object)
		self._set(me)

	def unwrap(self) -> list[_T]:
		return [v.unwrap() if hasattr(v, "unwrap") else v for v in self]


class MutableWrapperSet[_T](
	MutableWrapper, AbstractOrderlyMutableSet[_T], ABC
):
	__slots__ = ()

	def unwrap(self) -> OrderlySet[_T]:
		"""Deep copy myself as a set, all contents unwrapped"""
		unwrapped = OrderlySet()
		for v in self:
			if hasattr(v, "unwrap") and not hasattr(v, "no_unwrap"):
				unwrapped.add(v.unwrap())
			else:
				unwrapped.add(v)
		return unwrapped

	def clear(self) -> None:
		self._set(OrderlySet())

	def __repr__(self):
		strs = map(repr, self)
		return (
			"<"
			+ self.__class__.__name__
			+ " containing {"
			+ ", ".join(strs)
			+ "}>"
		)

@define(repr=False, eq=False, order=False)
class SubSetWrapper[_T](MutableWrapperSet[_T]):
	_getter: Callable[[], MutableSet[_T]]
	_setter: Callable[[MutableSet[_T]], None]
	_lock: RLock = field(init=False, factory=RLock)

	def _get(self) -> dict[_T, bool]:
		return dict.fromkeys(self._getter())

	def _set(self, data: dict[_T, bool]):
		self._setter(OrderlySet(data.keys()))

	def __copy__(self):
		return OrderlySet(self._getter())


_U = TypeVar("_U")
_V = TypeVar("_V")


def unwrap_items(it: Iterable[tuple[_U, _V]]) -> dict[_U, _V]:
	ret = {}
	for k, v in it:
		if hasattr(v, "unwrap") and not hasattr(v, "no_unwrap"):
			ret[k] = v.unwrap()
		else:
			ret[k] = v
	return ret


@define(eq=False, order=False)
class DictWrapper[_K, _V](MutableMappingWrapper[_K, _V], dict[_K, _V]):
	"""A dictionary synchronized with a serialized field.

	This is meant to be used in Lisien entities (graph, node, or
	edge), for when the user stores a dictionary in them.

	"""

	__slots__ = ()
	_getter: Callable[[], dict[_K, _V]]
	_outer: MutableMapping[_K, _V]
	_key: _K

	def __copy__(self):
		return dict(self._getter())

	def _get(
		self,
	) -> dict[_K, _V]:
		return self.unwrap()

	def _set(self, v):
		self._outer[self._key] = v

	def unwrap(self):
		return {
			k: v.unwrap() if hasattr(v, "unwrap") else v
			for (k, v) in self._getter().items()
		}


@define(eq=False)
class ListWrapper[_T](MutableWrapperDictList[int, _T], MutableSequence[_T]):
	"""A list synchronized with a serialized field.

	This is meant to be used in Lisien entities (graph, node, or
	edge), for when the user stores a list in them.

	"""

	__slots__ = ()

	_getter: Callable[[], list[_T]]
	_outer: MutableMapping
	_key: Hashable

	def __eq__(self, other: Sequence) -> bool:
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

	def __copy__(self) -> list[_T]:
		return list(self._getter())

	def _get(
		self,
	) -> list[_T]:
		return self._getter()

	def _set(self, v: list[_T]) -> None:
		self._outer[self._key] = v

	def insert(self, i: int, v: _T) -> None:
		new = self.__copy__()
		new.insert(i, v)
		self._set(new)

	def append(self, v: _T) -> None:
		new = self.__copy__()
		new.append(v)
		self._set(new)

	def unwrap(self) -> list[_T]:
		"""Deep copy myself as a list, with all contents unwrapped"""
		return [
			v.unwrap()
			if hasattr(v, "unwrap") and not hasattr(v, "no_unwrap")
			else v
			for v in self
		]


@define(eq=False, order=False, repr=False)
class SetWrapper[_T](MutableWrapperSet[_T]):
	"""A set synchronized with a serialized field.

	This is meant to be used in Lisien entities (graph, node, or
	edge), for when the user stores a set in them.

	"""

	_getter: Callable[[], MutableSet[_T]]
	_outer: MutableMapping
	_key: Hashable
	_lock: RLock = field(init=False, factory=RLock)

	def _get(self) -> dict[_T, bool]:
		return dict.fromkeys(self._getter())

	def _set(self, v: dict[_T, bool]) -> None:
		self._outer[self._key] = OrderlySet(v.keys())

	def __copy__(self):
		ret = OrderlySet()
		ret._data = self._get()
		return ret


class UnwrappingDict[_K, _V](dict[_K, _V]):
	"""Dict that stores the data from the wrapper classes

	Won't store those objects themselves.

	"""

	__slots__ = ()

	def __setitem__(self, key: _K, value: _V) -> None:
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
