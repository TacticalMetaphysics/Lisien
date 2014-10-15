# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Common utility functions and data structures.

"""
from collections import Mapping


class CacheError(ValueError):
    pass


class TravelException(Exception):
    """Exception for problems with pathfinding"""
    def __init__(
            self,
            message,
            path=None,
            followed=None,
            traveller=None,
            branch=None,
            tick=None,
            lastplace=None
    ):
        """Store the message as usual, and also the optional arguments:

        ``path``: a list of Place names to show such a path as you found

        ``followed``: the portion of the path actually followed

        ``traveller``: the Thing doing the travelling

        ``branch``: branch during travel

        ``tick``: tick at time of error (might not be the tick at the
        time this exception is raised)

        ``lastplace``: where the traveller was, when the error happened

        """
        self.path = path
        self.followed = followed
        self.traveller = traveller
        self.branch = branch
        self.tick = tick
        self.lastplace = lastplace
        super().__init__(message)


class CompositeDict(Mapping):
    """Read-only mapping that looks up values in a first dict if
    available, then a second dict if possible.

    Assumes the dicts have no overlap.

    """
    def __init__(self, d1, d2):
        """Store dictionaries"""
        self.d1 = d1
        self.d2 = d2

    def __iter__(self):
        """Iterate over both dictionaries' keys"""
        for k in self.d1:
            yield k
        for k in self.d2:
            yield k

    def __len__(self):
        """Sum the lengths of both dictionaries"""
        return len(self.d1) + len(self.d2)

    def __getitem__(self, k):
        """Get an item from ``d1`` if possible, then ``d2``"""
        try:
            return self.d1[k]
        except KeyError:
            return self.d2[k]


# ==Caching==
def keycache_iter(keycache, branch, tick, get_iterator):
    if branch not in keycache:
        keycache[branch] = {}
    if tick not in keycache[branch]:
        keycache[branch][tick] = set(get_iterator())
    yield from keycache[branch][tick]


def cache_get(cache, keycache, branch, tick, key, getter):
    """Utility function to retrieve something from a branch/tick-indexed
    cache if possible, and if not, retrieve it using ``getter`` (and
    add it to the cache).

    """
    if key not in cache:
        cache[key] = {}
    if branch not in cache[key]:
        cache[key][branch] = {}
    if tick not in cache[key][branch]:
        cache[key][branch][tick] = getter(key)
    return cache[key][branch][tick]


def cache_set(cache, keycache, branch, tick, key, value, setter):
    """Utility function to both set ``key = value`` using ``setter`` and
    add ``value`` to the cache, indexed with ``branch`` and then
    ``tick``.

    """
    if key not in cache:
        cache[key] = {}
    if branch not in cache[key]:
        cache[key][branch] = {}
    cache[key][branch][tick] = value
    if branch in keycache:
        try:
            if tick not in keycache[branch]:
                keycache[branch][tick] = set(
                    keycache[branch][
                        max(t for t in keycache[branch]
                            if t < tick)
                    ]
                )
            keycache[branch][tick].add(key)
        except ValueError:
            pass
    setter(key, value)


def cache_del(cache, keycache, branch, tick, key, deleter):
    """Utility function to both delete the key from the
    branch/tick-indexed cache, and delete it using ``deleter``.

    """
    if (
            key in cache and
            branch in cache[key] and
            tick in cache[key][branch]
    ):
        del cache[key][branch][tick]
    if branch in keycache:
        try:
            if tick not in keycache[branch]:
                keycache[branch][tick] = set(
                    keycache[branch][
                        max(t for t in keycache[branch]
                            if t < tick)
                        ]
                )
            keycache[branch][tick].remove(key)
        except ValueError:
            pass
    deleter(key)


def passthru(_):
    """A function that returns its input. Defined here for convenient
    import into lise.kv"""
    return _


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
