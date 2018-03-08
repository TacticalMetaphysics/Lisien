# This file is part of allegedb, an object relational mapper for versioned graphs.
# Copyright (C) Zachary Spector. zacharyspector@gmail.com
from functools import partial
from collections.abc import MutableSet, MutableMapping, MutableSequence, Iterable, Sized, Container


class MutableWrapper:
    __slots__ = ()

    def __iter__(self):
        return iter(self._getter())

    def __len__(self):
        return len(self._getter())

    def __contains__(self, item):
        return item in self._getter()

    def __eq__(self, other):
        return self._getter() == other


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


class SubDictWrapper(MutableWrapperDictList, MutableMapping, dict):
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


class SubListWrapper(MutableWrapperDictList, MutableSequence, list):
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


class SubSetWrapper(MutableWrapperSet, set):
    __slots__ = ('_getter', '_set')

    def __init__(self, getter, setter):
        self._getter = getter
        self._set = setter

    def _copy(self):
        return set(self._getter())


class DictWrapper(MutableWrapperDictList, MutableMapping, dict):
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


class SetWrapper(MutableWrapperSet, set):
    __slots__ = ('_getter', '_setter', '_outer', '_key')

    def __init__(self, getter, setter, outer, key):
        self._getter = getter
        self._setter = setter
        self._outer = outer
        self._key = key

    def _set(self, v):
        self._setter(v)
        self._outer[self._key] = v