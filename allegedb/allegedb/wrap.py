# This file is part of allegedb, an object relational mapper for versioned graphs.
# Copyright (C) Zachary Spector. zacharyspector@gmail.com
from collections import MutableMapping, MutableSequence
from copy import deepcopy as deepcopy_base


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
    __slots__ = ['_outer', '_key', '_inner', '_v']

    def __init__(self, outer, key, initval):
        self._outer = outer
        self._key = key
        self._inner = CoreDictWrapper(outer, key)
        self._v = initval
        if not isinstance(self._v, dict):
            raise TypeError(
                "DictWrapper only wraps dicts"
            )

    def _get(self, k=None):
        if k is None:
            return self._v
        return self._v[k]

    def __iter__(self):
        return iter(self._v)

    def __len__(self):
        return len(self._v)

    def __eq__(self, other):
        return self._v == other

    def __getitem__(self, k):
        r = self._v[k]
        if isinstance(r, dict):
            return DictWrapper(self, k, r)
        if isinstance(r, list):
            return ListWrapper(self, k, r)
        return r

    def __setitem__(self, k, v):
        self._v[k] = v
        self._outer[self._key] = self._v

    def __delitem__(self, k):
        del self._inner[k]
        del self._v[k]

    def __repr__(self):
        return repr(self._v)


class ListWrapper(MutableSequence, list):
    """A list synchronized with a serialized field.

    This is meant to be used in allegedb entities (graph, node, or
    edge), for when the user stores a list in them.

    """

    __slots__ = ['_inner', '_v']

    def __init__(self, outer, key, initval=None):
        self._inner = CoreListWrapper(outer, key)
        self._v = initval
        if not isinstance(self._v, list):
            raise TypeError(
                "ListWrapper only wraps lists"
            )

    def __iter__(self):
        return iter(self._v)

    def __len__(self):
        return len(self._v)

    def __eq__(self, other):
        return self._v == other

    def __getitem__(self, i):
        r = self._v[i]
        if isinstance(r, dict):
            return DictWrapper(self, i, r)
        if isinstance(r, list):
            return ListWrapper(self, i, r)
        return r

    def __setitem__(self, i, v):
        self._inner[i] = v
        self._v[i] = v

    def __delitem__(self, i, v):
        del self._inner[i]
        del self._v[i]

    def insert(self, i, v):
        self._inner.insert(i, v)
        self._v.insert(i, v)

    def __repr__(self):
        return repr(self._v)


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
            r[k] = deepcopy_base(v._v)
        else:
            r[k] = deepcopy_base(v)
    return r
