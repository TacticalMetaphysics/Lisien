# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
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
"""Common utility functions and data structures.

"""
from collections.abc import Set
from operator import attrgetter, add, sub, mul, pow, truediv, floordiv, mod
from functools import partial
from contextlib import contextmanager
from textwrap import dedent
from time import monotonic


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
        'engine', 'entity', 'branch', 'turn', 'tick', 'stat', 'current', 'mungers'
    ]

    def __init__(
            self, entity, stat,
            engine=None, branch=None, turn=None, tick=None, current=False, mungers=[]
    ):
        if engine is None:
            engine = entity.engine
        if branch is None:
            branch = engine.branch
        if turn is None:
            turn = engine.turn
        if tick is None:
            tick = engine.tick
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
            self.engine.turn = turn or self.turn
            self.engine.tick = tick or self.tick
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
            self.entity,
            self.stat,
            "" if self.current else
            ", branch={}, turn={}, tick={}".format(self.branch, self.turn, self.tick),
            len(self.mungers)
        )

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
        return EntityStatAccessor(
            self.entity,
            self.stat,
            self.engine,
            self.branch,
            self.tick,
            self.current,
            self.mungers + [munger]
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
        """Iterate over all the values this stat has had in the given window, inclusive.

        """
        # It might be useful to do this in a way that doesn't change the engine's time, perhaps for thread safety
        engine = self.engine
        entity = self.entity
        oldturn = engine.turn
        oldtick = engine.tick
        stat = self.stat
        for turn in range(beginning, end+1):
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
        source = source[nlidx+1:]
        nlidx = source.index('\n')
    return dedent(source)


def _sort_set_key(v):
    if isinstance(v, tuple):
        return (2,) + tuple(map(repr, v))
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
