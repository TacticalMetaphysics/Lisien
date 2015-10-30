# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Common utility functions and data structures.

"""
import numpy
from json import dumps, loads, JSONEncoder
from collections import Mapping
from gorm.xjson import (
    JSONWrapper,
    JSONReWrapper,
    JSONListWrapper,
    JSONListReWrapper,
    json_deepcopy
)
from gorm.reify import reify
from gorm.window import window_left, window_right


def dispatch(d, key, *args):
    """Call functions in a dictionary to inform them that a value in
    another mapping has changed.

    Keys in the dictionary should be the same as those in the observed
    mapping, with the exception of ``None``, which is for functions to
    call whenever any value of any key has changed.

    """
    assert(isinstance(d, Mapping))
    if key in d:
        for f in d[key]:
            f(*args)
    if None in d:
        for f in d[None]:
            f(*args)


def listen(l, f, k=None):
    """Insert the function into the list or dict supplied, under the key
    given.

    """
    if not callable(f):
        raise TypeError('listeners must be callable')
    if isinstance(l, Mapping):
        l = l[k]
    if f not in l:
        l.append(f)


def unlisten(l, f, k=None):
    """Remove the function from the list or dict.

    If dict, the function will be removed from the list in the dict
    under the key ``k``.

    """
    if not callable(f):
        raise TypeError('listeners must be callable')
    if isinstance(l, Mapping):
        l = l[k]
    if f in l:
        l.remove(f)


def listener(l, f=None, k=None):
    """Put a function into a list or dict, for later use by
    :class:`LiSE.util.dispatch`.

    """
    if f:
        listen(l, f, k)
        return f
    return lambda fun: listen(l, fun, k)


def unlistener(l, f=None, k=None):
    """Remove a function from the list or dict used by
    :class:`LiSE.util.dispatch`.

    """
    if f:
        try:
            unlisten(l, f, k)
            return f
        except KeyError:
            return f
    return lambda fun: unlisten(l, fun, k)


try:
    from sqlalchemy.exc import OperationalError as alchemyOpError
    from sqlite3 import OperationalError as liteOpError
    OperationalError = (alchemyOpError, liteOpError)
except ImportError:
    from sqlite3 import OperationalError


try:
    from sqlalchemy.exc import IntegrityError as alchemyIntegError
    from sqlite3 import IntegrityError as liteIntegError
    IntegrityError = (alchemyIntegError, liteIntegError)
except ImportError:
    from sqlite3 import IntegrityError


class RedundantRuleError(ValueError):
    """Error condition for when you try to run a rule on a (branch,
    tick) it's already been executed.

    """
    pass


class UserFunctionError(SyntaxError):
    """Error condition for when I try to load a user-defined function and
    something goes wrong.

    """
    pass


class CacheError(ValueError):
    """Error condition for something going wrong with a cache"""
    pass


class TravelException(Exception):
    """Exception for problems with pathfinding. Not necessarily an error
    because sometimes somebody SHOULD get confused finding a path.

    """
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


# ==Caching==


class LiSEncoder(JSONEncoder):
    def default(self, o):
        t = type(o)
        if t in numpy.sctypes['int']:
            return int(o)
        elif t in numpy.sctypes['float']:
            return float(o)
        else:
            return super().default(o)


def _keycache(self, keycache, k, meth):
    (branch, tick) = self.engine.time
    if branch not in keycache:
        return
    if tick not in keycache[branch]:
        if tick - 1 in keycache[branch]:
            keycache[branch][tick] = set(keycache[branch][tick-1])
        else:
            return
    getattr(keycache[branch][tick], meth)(k)
    for t in list(keycache[branch].keys()):
        if t > tick:
            del keycache[branch][t]
    for child in self.engine._branch_descendants(branch):
        if child in keycache:
            del keycache[child]


def enkeycache(self, keycache, k):
    _keycache(self, keycache, k, 'add')


def dekeycache(self, keycache, k):
    _keycache(self, keycache, k, 'discard')


def encache(self, cache, k, v):
    """Put ``k=v`` into ``cache`` and delete anything later than it"""
    (branch, tick) = self.engine.time
    if k not in cache:
        cache[k] = {}
    if branch not in cache[k]:
        cache[k][branch] = {}
    if isinstance(v, JSONListWrapper):
        v = JSONListReWrapper(v.outer, v.outkey, list(v))
    elif isinstance(v, JSONWrapper):
        v = JSONReWrapper(v.outer, v.outkey, dict(v))
    elif isinstance(v, dict):
        v = JSONReWrapper(self, k, v)
    elif isinstance(v, list):
        v = JSONListReWrapper(self, k, v)
    cache[k][branch][tick] = v
    for t in list(cache[k][branch].keys()):
        if t > tick:
            del cache[k][branch][t]
    for child in self.engine._branch_descendants(branch):
        if child in cache:
            del cache[child]


def cache_forward(cache, k, branch, tick):
    if (
            k in cache and
            branch in cache[k] and
            tick not in cache[k][branch] and
            tick - 1 in cache[k][branch]
    ):
        cache[k][branch][tick] = cache[k][branch][tick-1]


def needcache(cache, k, branch, tick):
    """Return whether ``k`` lacks a value in ``cache`` for the given
    ``branch`` and ``tick``.

    """
    return (
        k not in cache or
        branch not in cache[k] or
        tick not in cache[k][branch]
    )


def fillcache(engine, real, cache):
    """Copy all the present values in ``real`` into ``cache``, indexed
    appropriately.

    """
    (branch, tick) = engine.time
    for k in real:
        if k not in cache:
            cache[k] = {}
        if branch not in cache[k]:
            cache[k][branch] = {}
        if tick not in cache[k][branch]:
            cache[k][branch][tick] = real[k]


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
    r = {item: True for item in new}
    for item in old:
        if item not in new:
            r[item] = False
    return r


def fire_time_travel_triggers(
        real,
        cache,
        dispatcher,
        branch_then,
        tick_then,
        branch_now,
        tick_now
):
    """For each key in ``cache`` whose value has changed between the two
    times given, call ``dispatcher`` with the key and its current value.

    ``real`` is the mapping whose values ``cache`` caches. It will be
    used when the key is in the cache, but not at either of the times
    given.

    If a key has never been cached, it won't trigger any of its
    listeners, but this is unlikely, because caching happens whenever
    the key is used for any purpose.

    """
    # only fire anything if there is actual time travel going on
    if branch_then == branch_now and tick_then == tick_now:
        return
    for k in cache:
        if (
                branch_then in cache[k] and
                tick_then in cache[k][branch_then] and
                cache[k][branch_then][tick_then] is not None
        ):
            # key was set then
            if (
                    branch_now in cache[k] and
                    tick_now in cache[k][branch_now] and
                    cache[k][branch_now][tick_now] is not None
            ):
                # key is set now
                val_then = cache[k][branch_then][tick_then]
                val_now = cache[k][branch_now][tick_now]
                if val_then != val_now:
                    # key's value changed between then and now
                    dispatcher(k, val_now)
            else:
                # No cached info on the value right now; account for
                # the common case that a single tick has passed and
                # nothing has changed
                if (
                    branch_then == branch_now and
                    tick_then + 1 == tick_now
                ):
                    cache[k][branch_now][tick_now] \
                        = cache[k][branch_now][tick_then]
                    continue
                # otherwise, fetch from db
                try:
                    if branch_now not in cache[k]:
                        cache[k][branch_now] = {}
                    cache[k][branch_now][tick_now] = real[k]
                    if (
                        cache[k][branch_then][tick_then] !=
                        cache[k][branch_now][tick_now]
                    ):
                        dispatcher(k, cache[k][branch_now][tick_now])
                except KeyError:
                    dispatcher(k, None)
        else:
            # key might not have been set then -- if it was, our
            # listeners never heard of it, or it'd be cached
            if (
                    branch_now in cache[k] and
                    tick_now in cache[k][branch_now]
            ):
                if cache[k][branch_now][tick_now] is None:
                    # and they still never heard of it
                    continue
                else:
                    # key is set now
                    dispatcher(k, cache[k][branch_now][tick_now])


def keycache_iter(keycache, branch, tick, get_iterator):
    if branch not in keycache:
        keycache[branch] = {}
    if tick not in keycache[branch]:
        keycache[branch][tick] = set(get_iterator())
    yield from keycache[branch][tick]


# ABCs


class AbstractEngine(object):
    @reify
    def json_dump_hints(self):
        return {}

    @reify
    def json_load_hints(self):
        return {}

    @classmethod
    def _enc_tuple(cls, obj):
        if isinstance(obj, tuple):
            return ['tuple'] + [cls._enc_tuple(v) for v in obj]
        elif isinstance(obj, list):
            return ['list'] + [cls._enc_tuple(v) for v in obj]
        elif isinstance(obj, dict):
            return {
                cls._enc_tuple(k): cls._enc_tuple(v)
                for (k, v) in obj.items()
                }
        elif isinstance(obj, cls.char_cls):
            return ['character', obj.name]
        elif isinstance(obj, cls.node_cls):
            return ['node', obj.character.name, obj.name]
        elif isinstance(obj, cls.portal_cls):
            return ['portal', obj.character.name, obj.nodeA.name, obj.nodeB.name]
        else:
            return obj

    def _dec_tuple(self, obj):
        if isinstance(obj, dict):
            r = {}
            for (k, v) in obj.items():
                r[self._dec_tuple(k)] = self._dec_tuple(v)
            return r
        elif isinstance(obj, list):
            if obj == [] or obj == ["list"]:
                return []
            elif obj == ["tuple"]:
                return tuple()
            elif obj[0] == 'list':
                return [self._dec_tuple(p) for p in obj[1:]]
            elif obj[0] == 'tuple':
                return tuple(self._dec_tuple(p) for p in obj[1:])
            elif obj[0] == 'character':
                return self.character[self._dec_tuple(obj[1])]
            elif obj[0] == 'node':
                return self.character[self._dec_tuple(obj[1])].node[self._dec_tuple(obj[2])]
            elif obj[0] == 'portal':
                return self.character[self._dec_tuple(obj[1])].portal[self._dec_tuple(obj[2])][self._dec_tuple(obj[3])]
            else:
                raise ValueError("Unknown sequence type: {}".format(obj[0]))
        else:
            return obj

    def json_dump(self, obj):
        """JSON dumper that distinguishes lists from tuples, and handles
        pointers to Node, Portal, and Character.

        """
        if isinstance(obj, self.node_cls):
            return dumps(["node", obj.character.name, obj.name])
        if isinstance(obj, self.portal_cls):
            return dumps(["portal", obj.character.name, obj.orign, obj.destn])
        if isinstance(obj, self.char_cls):
            return dumps(["character", obj.name])
        k = str(obj)
        if k not in self.json_dump_hints:
            self.json_dump_hints[k] = dumps(self._enc_tuple(obj), cls=LiSEncoder)
        return self.json_dump_hints[k]

    def json_load(self, s):
        if s is None:
            return None
        if s not in self.json_load_hints:
            self.json_load_hints[s] = self._dec_tuple(loads(s))
        return self.json_load_hints[s]


def stat_validity(k, cache, branch, tick):
    lo = window_left(cache[k][branch].keys(), tick)
    try:
        hi = window_right(cache[k][branch].keys(), tick)
    except ValueError:
        hi = None
    return (lo, hi)


def fire_stat_listeners(
        dispatcher,
        keys,
        cache,
        stats_validity,
        branch_then,
        tick_then,
        branch_now,
        tick_now
):
    def getstat(k, branch, tick):
        return cache[k][branch][window_left(cache[k][branch].keys(), tick)]

    for k in keys:
        try:
            (since, until) = stats_validity[k]
            if (
                branch_then == branch_now and
                tick_now >= since and
                until is None or tick_now < until
            ):
                continue
        except KeyError:
            pass
        try:
            stats_validity[k] = stat_validity(k, cache, branch_now, tick_now)
        except ValueError:
            if k in stats_validity:
                del stats_validity[k]
        try:
            newv = getstat(k, branch_now, tick_now)
            oldv = getstat(k, branch_then, tick_then)
            if newv == oldv:
                continue
        except (KeyError, ValueError):
            continue
        dispatcher(k, newv)
