# This file is part of allegedb, an object relational mapper for versioned graphs.
# Copyright (C) Zachary Spector. zacharyspector@gmail.com
from collections import MutableMapping, MutableSequence
from copy import deepcopy as deepcopy_base
from functools import partial


class CoreDictWrapper(MutableMapping):
    """A dictionary-like object stored serialized.

    This isn't meant to be used on its own; see ``DictWrapper``.

    """
    __slots__ = ['outer', 'outkey']

    def __init__(self, outer, outkey):
        """Store the mapping I'm in and the key used to access me."""
        self.outer = outer
        self.outkey = outkey

    def __contains__(self, wot):
        return self._get().__contains__(wot)

    def _get(self, k=None):
        if k is None:
            return self.outer._get(self.outkey)
        return self._get()[k]

    def _set(self, v):
        self.outer[self.outkey] = v

    def __iter__(self):
        return iter(self._get())

    def __len__(self):
        return len(self._get())

    def __getitem__(self, k):
        r = self._get()[k]
        if isinstance(r, list):
            return CoreListWrapper(self, k)
        elif isinstance(r, dict):
            return CoreDictWrapper(self, k)
        else:
            return r

    def __setitem__(self, k, v):
        me = dict(self._get())
        me[k] = v
        self._set(me)

    def __delitem__(self, k):
        me = dict(self._get())
        del me[k]
        self._set(me)

    def __str__(self):
        return self._get().__str__()

    def __repr__(self):
        return self._get().__repr__()

    def __eq__(self, other):
        return self._get() == other

    def __list__(self):
        return list(self._get())

    def __dict__(self):
        return dict(self._get())

    def copy(self):
        return self._get().copy()


class CoreListWrapper(CoreDictWrapper):
    """A list synchronized with a serialized field.

    This isn't meant to be used on its own; see ``ListWrapper``.

    """
    def append(self, v):
        me = self._get()
        me.append(v)
        self._set(me)

    def insert(self, i, v):
        me = self._get()
        me.insert(i, v)
        self._set(me)


class DictWrapper(MutableMapping, dict):
    """A dictionary synchronized with a serialized field.

    This is meant to be used in allegedb entities (graph, node, or
    edge), for when the user stores a dictionary in them.

    """
    __slots__ = ['_getter', '_setter', '_outer', '_key', '_inner', '_v']

    def __init__(self, getter, setter, outer, key, initval=None):
        if initval:
            if not isinstance(initval, dict):
                raise TypeError("DictWrapper only wraps dicts")
            setter(initval)
        else:
            setter({})
        self._getter = getter
        self._setter = setter
        self._outer = outer
        self._key = key
        self._inner = CoreDictWrapper(outer, key)

    def _get(self, k=None):
        if k is None:
            return self._getter()
        return self._getter()[k]

    def __iter__(self):
        return iter(self._getter())

    def __len__(self):
        return len(self._getter())

    def __eq__(self, other):
        return self._getter() == other

    def _subget(self, k, subk):
        return self._getter()[k][subk]

    def _subset(self, k, subk, v):
        new = dict(self._getter())
        new[k][subk] = v
        self._setter(new)

    def __getitem__(self, k):
        r = self._getter()[k]
        if isinstance(r, dict):
            return DictWrapper(partial(self._subget, k), partial(self._subset, k), self, k, r)
        if isinstance(r, list):
            return ListWrapper(partial(self._subget, k), partial(self._subset, k), self, k, r)
        return r

    def __setitem__(self, k, v):
        new = dict(self._getter())
        new[k] = v
        self._setter(new)
        self._inner[k] = v

    def __delitem__(self, k):
        del self._inner[k]
        new = self._getter()
        del new[k]
        self._setter(new)

    def __repr__(self):
        return repr(self._getter())


class ListWrapper(MutableSequence, list):
    """A list synchronized with a serialized field.

    This is meant to be used in allegedb entities (graph, node, or
    edge), for when the user stores a list in them.

    """

    __slots__ = ['_getter', '_setter', '_inner']

    def __init__(self, getter, setter, outer, key, initval=None):
        if initval:
            if not isinstance(initval, list):
                raise TypeError("ListWrapper only wraps lists")
            setter(initval)
        else:
            setter([])
        self._inner = CoreListWrapper(outer, key)
        self._getter = getter
        self._setter = setter

    def __iter__(self):
        return iter(self._getter())

    def __len__(self):
        return len(self._getter())

    def __eq__(self, other):
        return self._getter() == other

    def _subget(self, i, j):
        return self._getter()[i][j]

    def _subset(self, i, j, v):
        new = list(self._getter())
        new[i][j] = v
        self._setter(new)

    def __getitem__(self, i):
        r = self._getter()[i]
        if isinstance(r, dict):
            return DictWrapper(partial(self._subget, i), partial(self._subset, i), self, i, r)
        if isinstance(r, list):
            return ListWrapper(partial(self._subget, i), partial(self._subset, i), self, i, r)
        return r

    def __setitem__(self, i, v):
        self._inner[i] = v
        new = list(self._getter())
        new[i] = v
        self._setter(new)

    def __delitem__(self, i):
        del self._inner[i]
        new = list(self._getter())
        del new[i]
        self._setter(new)

    def insert(self, i, v):
        self._inner.insert(i, v)
        new = list(self._getter())
        new.insert(i, v)
        self._setter(new)

    def __repr__(self):
        return repr(self._getter())


def deepcopy(obj):
    """Copy all the data from a ``DictWrapper`` into a dictionary."""
    if not isinstance(obj, DictWrapper):
        return deepcopy_base(obj)
    r = {}
    for (k, v) in obj.items():
        if (
            isinstance(v, DictWrapper) or
            isinstance(v, ListWrapper)
        ):
            r[k] = deepcopy_base(v._getter())
        else:
            r[k] = deepcopy_base(v)
    return r
