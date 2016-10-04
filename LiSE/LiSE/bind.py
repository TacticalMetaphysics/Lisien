# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Utilities for binding functions to values."""
from gorm.reify import reify
from collections import defaultdict, Mapping, MutableSequence


def dispatch(d, key, v, *args):
    """Call functions in a dictionary to inform them that a value in
    another mapping has changed.

    Keys in the dictionary should be the same as those in the observed
    mapping, with the exception of ``None``, which is for functions to
    call whenever any value of any key has changed.

    """
    if not isinstance(d, Mapping):
        raise TypeError(
            "Need a mapping of functions, not {}".format(type(d))
        )
    if key in d:
        if d[key] == v:
            return
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


def stat_validity(k, cache, branch, tick):
    """Return the tick when a stat took its current value, and when it'll
    change."""
    lo = cache[k][branch]._past[-1][0]
    try:
        hi = cache[k][branch]._future[0][0]
    except IndexError:
        hi = None
    return (lo, hi)


class SessionList(MutableSequence):
    """List that calls listeners on first item set and last delete."""
    __slots__ = ['_real', '_listeners']

    def __init__(self, listeners=[]):
        self._real = []
        self._listeners = listeners

    def listener(self, f):
        if f not in self._listeners:
            self._listeners.append(f)

    def unlisten(self, f):
        if f in self._listeners:
            self._listeners.remove(f)

    def insert(self, i, v):
        self._real.insert(i, v)

    def __iter__(self):
        return iter(self._real)

    def __len__(self):
        return len(self._real)

    def __getitem__(self, i):
        return self._real[i]

    def __setitem__(self, i, v):
        begun = self._real == []
        self._real[i] = v
        if begun:
            for f in self._listeners:
                f(self)

    def __delitem__(self, i):
        del self._real[i]
        if self._real == []:
            for f in self._listeners:
                f(self)

    def __str__(self):
        return str(self._real)

    def __repr__(self):
        return repr(self._real)

time_dispatcher_listeners = {}
time_dispatcher_validity = defaultdict(dict)
time_dispatch_cache = {}

class TimeDispatcher(object):
    """Mixin class for sim-time-sensitive objects with callback functions.

    The callback functions get called with args ``(key, value, self,
    branch, tick)`` whenever the watched key *appears* to change its
    value. This can happen either because a new value was actually
    set, or because one had already been set in the future, and time
    passed so that the future is now the present.

    """
    @property
    def _listeners(self):
        me = id(self)
        if me not in time_dispatcher_listeners:
            time_dispatcher_listeners[me] = defaultdict(lambda: SessionList([
                self._listen_to_time_if
            ]))
        return time_dispatcher_listeners[me]

    @property
    def _dispatch_validity(self):
        return time_dispatcher_validity[id(self)]

    @property
    def _dispatch_cache(self):
        me = id(self)
        if me not in time_dispatch_cache:
            time_dispatch_cache[me] = defaultdict(lambda: SessionList([
                self._listen_to_time_if
            ]))
        return time_dispatch_cache[me]

    def listener(self, fun=None, key=None):
        return listener(self._listeners, fun, key)

    def unlisten(self, fun=None, key=None):
        return unlistener(self._listeners, fun, key)

    def dispatch(self, k, v):
        if k in self and self[k] is v:
            return
        (branch, tick) = self.engine.time
        d = self._listeners
        if k in d:
            for f in d[k]:
                f(branch, tick, self, k, v)
        if None in d:
            for f in d[None]:
                f(branch, tick, self, k, v)

    def dispatch_time(
            self,
            branch_then,
            tick_then,
            branch_now,
            tick_now
    ):
        for k in set(self.keys()).union(self._dispatch_validity.keys()):
            try:
                (since, until) = self._dispatch_validity[k]
                if (
                    branch_then == branch_now and
                    tick_now >= since and
                    (until is None or tick_now < until)
                ):
                    continue
            except KeyError:
                pass
            try:
                self._dispatch_validity[k] = stat_validity(
                    k, self._dispatch_cache, branch_now, tick_now
                )
            except ValueError:
                if k in self._dispatch_validity:
                    del self._dispatch_validity[k]
            try:
                newv = self._dispatch_cache[k][branch_now][tick_now]
                oldv = self._dispatch_cache[k][branch_then][tick_then]
                if newv == oldv:
                    continue
            except (KeyError, ValueError):
                continue
            self.dispatch(k, newv)

    def _listen_to_time_if(self, b):
        if b:
            self.engine.time_listener(self.dispatch_time)
        else:
            self.engine.time_unlisten(self.dispatch_time)
