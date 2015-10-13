# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from collections import (
    MutableSequence,
    Callable
)

from .util import listen


class FunList(MutableSequence):
    """Persistent list of functions associated with a rule.

    Stored as JSON lists of strings in the world database. Said
    strings identify functions in the code database.

    """
    def __init__(
            self,
            engine,
            db
    ):
        """Store the rule and the name of the field to put the data in, then
        put the data in it.

        """
        self.engine = engine
        self.db = db
        self._listeners = []

    def __repr__(self):
        return "{}([{}])".format(self.__class__.__name__, self._getlist())

    @property
    def funcstore(self):
        raise NotImplementedError

    def _dispatch(self):
        for f in self._listeners:
            f(self)

    def listener(self, f):
        listen(self._listeners, f)

    def _funn(self, v):
        funn = v.__name__ if isinstance(v, Callable) else v
        if funn not in self.funcstore:
            if not isinstance(v, Callable):
                raise KeyError("No such function: {}".format(v))
            self.funcstore[funn] = v
        return funn

    def _setlist(self, l):
        """Update the rule's record with this new list of strings."""
        if not isinstance(l, list):
            l = list(l)
        self._savelist(l)
        if self.engine.caching:
            self._cache = l

    def _savelist(self, l):
        raise NotImplementedError

    def _getlist(self):
        """Return the list, decoded from JSON, but not yet translated to
        actual functions, just their names.

        """
        if not self.engine.caching:
            return self._loadlist()
        if not hasattr(self, '_cache'):
            self._cache = self._loadlist()
        return self._cache

    def _loadlist(self):
        raise NotImplementedError

    def __iter__(self):
        """Yield a function from the code database for each item in the
        list

        """
        for funn in self._getlist():
            yield self.funcstore[funn]

    def __eq__(self, other):
        """Also return ``True`` if ``other`` contains the names of my
        functions in the correct order.

        """
        return (
            super().__eq__(other) or
            self._getlist() == other
        )

    def __len__(self):
        """Return the length of the list (don't translate it to functions)"""
        return len(self._getlist())

    def __getitem__(self, i):
        """Get the function named by the ith item in the list"""
        return self.funcstore[self._getlist()[i]]

    def __setitem__(self, i, v):
        """If ``v`` is a function, store it and get its name, otherwise it's
        the name of a function. Either way, set the ith item to
        the name.

        """
        l = self._getlist()
        l[i] = self._funn(v)
        self._setlist(l)
        self._dispatch()

    def __delitem__(self, i):
        """Delete item i and save the list"""
        l = self._getlist()
        del l[i]
        self._setlist(l)
        self._dispatch()

    def insert(self, i, v):
        """Insert the name of function ``v`` (or just ``v`` if it's a string)
        into position i

        """
        l = self._getlist()
        l.insert(i, self._funn(v))
        self._setlist(l)
        self._dispatch()
