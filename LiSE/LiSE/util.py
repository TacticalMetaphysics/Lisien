# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Common utility functions and data structures.

"""
from operator import attrgetter, add, sub, mul, pow, truediv, floordiv, mod
from functools import partial


def getatt(attribute_name):
    """An easy way to make an alias"""
    return property(attrgetter(attribute_name))


def path_len(graph, path, weight=None):
    """Return the number of ticks it will take to follow ``path``,
    assuming the portals' ``weight`` attribute is how long it will
    take to go through that portal--if unspecified, 1 tick.

    """
    n = 0
    path = list(path)  # local copy
    prevnode = path.pop(0)
    while path:
        nextnode = path.pop(0)
        edge = graph.edge[prevnode][nextnode]
        n += edge[weight] if weight and hasattr(edge, weight) else 1
        prevnode = nextnode
    return n


def dict_diff(old, new):
    """Return a dictionary containing the items of ``new`` that are either
    absent from ``old`` or whose values are different; as well as the
    value ``None`` for those keys that are present in ``old``, but
    absent from ``new``.

    Useful for describing changes between two versions of a dict.

    """
    r = {}
    for k in set(old.keys()).union(new.keys()):
        if k in old:
            if k not in new:
                r[k] = None
            elif old[k] != new[k]:
                r[k] = new[k]
        else:  # k in new
            r[k] = new[k]
    return r


def list_diff(old, new):
    r = {}
    for item in old:
        if item not in new:
            r[item] = False
    for item in new:
        if item not in old:
            r[item] = True
    return r


def keycache_iter(keycache, branch, tick, get_iterator):
    if branch not in keycache:
        keycache[branch] = {}
    if tick not in keycache[branch]:
        keycache[branch][tick] = set(get_iterator())
    yield from keycache[branch][tick]


class EntityStatAccessor(object):
    __slots__ = ['engine', 'entity', 'branch', 'tick', 'stat', 'current', 'mungers']

    def __init__(self, entity, stat, engine=None, branch=None, tick=None, current=False, mungers=[]):
        if engine is None:
            engine = entity.engine
        if branch is None and engine is not None:
            branch = engine.branch
        if tick is None and engine is not None:
            tick = engine.tick
        self.current = current
        self.engine = engine
        self.entity = entity
        self.stat = stat
        self.branch = branch
        self.tick = tick
        self.mungers = mungers

    def __call__(self, branch=None, tick=None):
        if self.current:
            res = self.entity[self.stat]
        else:
            time = self.engine.time
            self.engine.time = (branch or self.branch, tick or self.tick)
            res = self.entity[self.stat]
            self.engine.time = time
        for munger in self.mungers:
            res = munger(res)
        return res

    def __hash__(self):
        return hash(self())

    def __ne__(self, other):
        return self() != other

    def __str__(self):
        return str(self())

    def __repr__(self):
        return "EntityStatAccessor({}[{}]{}), {} mungers".format(
            self.entity,
            self.stat,
            "" if self.current else ", branch={}, tick={}".format(self.branch, self.tick),
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
