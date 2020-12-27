# This file is part of allegedb, an object relational mapper for versioned graphs.
# Copyright (C) Zachary Spector. public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Wrapper classes to let you store mutable data types in the allegedb ORM"""
from functools import partial
from itertools import zip_longest
from abc import ABC, abstractmethod
from collections.abc import MutableSet, MutableMapping, MutableSequence, Mapping, Sequence, Iterable, Sized, Container


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
    def _set(self):
        raise NotImplementedError

    @abstractmethod
    def unwrap(self):
        raise NotImplementedError


Iterable.register(MutableWrapper)
Sized.register(MutableWrapper)
Container.register(MutableWrapper)


class MutableWrapperDictList(MutableWrapper):
    __slots__ = ()

    def _subset(self, k, v):
        new = self._copy()
        new[k] = v
        self._set(new)

    def __getitem__(self, k):
        ret = self._getter()[k]
        if isinstance(ret, dict):
            return SubDictWrapper(lambda: self._getter()[k], partial(self._subset, k))
        if isinstance(ret, list):
            return SubListWrapper(lambda: self._getter()[k], partial(self._subset, k))
        if isinstance(ret, set):
            return SubSetWrapper(lambda: self._getter()[k], partial(self._subset, k))
        return ret

    def __setitem__(self, key, value):
        me = self._copy()
        me[key] = value
        self._set(me)

    def __delitem__(self, key):
        me = self._copy()
        del me[key]
        self._set(me)


class MutableMappingUnwrapper(MutableMapping):
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
            if hasattr(me, 'unwrap'):
                me = me.unwrap()
            if hasattr(you, 'unwrap'):
                you = you.unwrap()
            if me != you:
                return False
        else:
            return True

    def unwrap(self):
        """Return a deep copy of myself as a dict, and unwrap any wrapper objects in me."""
        return {
            k: v.unwrap() if hasattr(v, 'unwrap') and not hasattr(v, 'no_unwrap') else v
            for (k, v) in self.items()
        }


class MutableMappingWrapper(MutableWrapperDictList, MutableMappingUnwrapper):
    def __eq__(self, other):
        return MutableMappingUnwrapper.__eq__(self, other)

    def unwrap(self):
        return MutableMappingUnwrapper.unwrap(self)


class SubDictWrapper(MutableMappingWrapper, dict):
    __slots__ = ('_getter', '_set')

    def __init__(self, getter, setter):
        self._getter = getter
        self._set = setter

    def _copy(self):
        return dict(self._getter())

    def _subset(self, k, v):
        new = dict(self._getter())
        new[k] = v
        self._set(new)


class MutableSequenceWrapper(MutableWrapperDictList, MutableSequence):
    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Sequence):
            return NotImplemented
        for me, you in zip_longest(self, other):
            if hasattr(me, 'unwrap'):
                me = me.unwrap()
            if hasattr(you, 'unwrap'):
                you = you.unwrap()
            if me != you:
                return False
        else:
            return True

    def unwrap(self):
        """Return a deep copy of myself as a list, and unwrap any wrapper objects in me."""
        return [
            v.unwrap() if hasattr(v, 'unwrap') else v
            for v in self
        ]


class SubListWrapper(MutableSequenceWrapper, list):
    __slots__ = ('_getter', '_set')

    def __init__(self, getter, setter):
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


class MutableWrapperSet(MutableWrapper, MutableSet):
    __slots__ = ()

    def _copy(self):
        return set(self._getter())

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
        """Return a deep copy of myself as a set, and unwrap any wrapper objects in me."""
        return {v.unwrap() if hasattr(v, 'unwrap') and not hasattr(v, 'no_unwrap') else v for v in self}


class SubSetWrapper(MutableWrapperSet, set):
    __slots__ = ('_getter', '_set')

    def __init__(self, getter, setter):
        self._getter = getter
        self._set = setter

    def _copy(self):
        return set(self._getter())


class DictWrapper(MutableMappingWrapper, dict):
    """A dictionary synchronized with a serialized field.

    This is meant to be used in allegedb entities (graph, node, or
    edge), for when the user stores a dictionary in them.

    """
    __slots__ = ('_getter', '_setter', '_outer', '_key')

    def __init__(self, getter, setter, outer, key):
        self._getter = getter
        self._setter = setter
        self._outer = outer
        self._key = key

    def _copy(self):
        return dict(self._getter())

    def _set(self, v):
        self._setter(v)
        self._outer[self._key] = v


class ListWrapper(MutableWrapperDictList, MutableSequence, list):
    """A list synchronized with a serialized field.

    This is meant to be used in allegedb entities (graph, node, or
    edge), for when the user stores a list in them.

    """

    __slots__ = ('_getter', '_setter', '_outer', '_key')

    def __init__(self, getter, setter, outer, key):
        self._outer = outer
        self._key = key
        self._getter = getter
        self._setter = setter

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Sequence):
            return NotImplemented
        for me, you in zip_longest(self, other):
            if hasattr(me, 'unwrap'):
                me = me.unwrap()
            if hasattr(you, 'unwrap'):
                you = you.unwrap()
            if me != you:
                return False
        else:
            return True

    def _copy(self):
        return list(self._getter())

    def _set(self, v):
        self._setter(v)
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
        """Return a deep copy of myself as a list, and unwrap any wrapper objects in me."""
        return [v.unwrap() if hasattr(v, 'unwrap') and not hasattr(v, 'no_unwrap') else v for v in self]


class SetWrapper(MutableWrapperSet, set):
    """A set synchronized with a serialized field.

    This is meant to be used in allegedb entities (graph, node, or
    edge), for when the user stores a set in them.

    """
    __slots__ = ('_getter', '_setter', '_outer', '_key')

    def __init__(self, getter, setter, outer, key):
        self._getter = getter
        self._setter = setter
        self._outer = outer
        self._key = key

    def _set(self, v):
        self._setter(v)
        self._outer[self._key] = v


class UnwrappingDict(dict):
    """Dict that stores the data from the wrapper classes but won't store those objects themselves."""
    def __setitem__(self, key, value):
        if isinstance(value, MutableWrapper):
            value = value.unwrap()
        super(UnwrappingDict, self).__setitem__(key, value)