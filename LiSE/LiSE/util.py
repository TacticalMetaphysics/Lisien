# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Common utility functions and data structures.

"""
from operator import attrgetter, add, sub, mul, pow, truediv, floordiv, mod
from functools import partial
from .reify import reify


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
    try:
        oldset = frozenset(old.items())
        newset = frozenset(new.items())
        if (oldset, newset) in dict_diff.memo:
            return dict_diff.memo[(oldset, newset)]
        r = dict_diff.memo[(oldset, newset)] = {}
    except TypeError:
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
dict_diff.memo = {}


def set_diff(old, new):
    try:
        old = frozenset(old)
        new = frozenset(new)
        if (old, new) in set_diff.memo:
            return set_diff.memo[(old, new)]
        r = set_diff.memo[(old, new)] = {}
    except TypeError:
        r = {}
    for item in old:
        if item not in new:
            r[item] = False
    for item in new:
        if item not in old:
            r[item] = True
    return r
set_diff.memo = {}


def keycache_iter(keycache, branch, tick, get_iterator):
    if branch not in keycache:
        keycache[branch] = {}
    if tick not in keycache[branch]:
        keycache[branch][tick] = set(get_iterator())
    yield from keycache[branch][tick]


class EntityStatAccessor(object):
    __slots__ = [
        'engine', 'entity', 'branch', 'tick', 'stat', 'current', 'mungers'
    ]

    def __init__(
            self, entity, stat,
            engine=None, branch=None, tick=None, current=False, mungers=[]
    ):
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
            "" if self.current else
            ", branch={}, tick={}".format(self.branch, self.tick),
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


def is_chardiff(d):
    if not isinstance(d, dict):
        return False
    diffkeys = (
        'character_stat',
        'node_stat',
        'things',
        'places',
        'portal_stat',
        'portals',
        'avatars',
        'rulebooks',
        'node_rulebooks',
        'portal_rulebooks'
    )
    return any(key in d for key in diffkeys) and not any(
        key not in diffkeys for key in d.keys()
    )


def dedent_sourcelines(sourcelines):
    if sourcelines[0].strip().startswith('@'):
        del sourcelines[0]
    indent = 999
    for line in sourcelines:
        lineindent = 0
        for char in line:
            if char not in ' \t':
                break
            lineindent += 1
        else:
            indent = 0
            break
        indent = min((indent, lineindent))
    return '\n'.join(line[indent:].strip('\n') for line in sourcelines) + '\n'