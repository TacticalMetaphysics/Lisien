# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Common utility functions and data structures.

"""
from collections import Mapping
from math import sqrt, hypot, atan, pi, sin, cos


phi = (1.0 + sqrt(5))/2.0
"""The golden ratio."""


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


def slope_theta_rise_run(rise, run):
    """Return a radian value expressing the angle at the lower-left corner
    of a triangle ``rise`` high, ``run`` wide.

    If ``run`` is zero, but ``rise`` is positive, return pi / 2. If
    ``run`` is zero, but ``rise`` is negative, return -pi / 2.

    """
    try:
        return atan(rise/run)
    except ZeroDivisionError:
        if rise >= 0:
            return ninety
        else:
            return -1 * ninety


def slope_theta(ox, oy, dx, dy):
    """Get a radian value representing the angle formed at the corner (ox,
    oy) of a triangle with a hypotenuse going from there to (dx,
    dy).

    """
    rise = dy - oy
    run = dx - ox
    return slope_theta_rise_run(rise, run)


def opp_theta_rise_run(rise, run):
    """Inverse of ``slope_theta_rise_run``"""
    try:
        return atan(run/rise)
    except ZeroDivisionError:
        if run >= 0:
            return ninety
        else:
            return -1 * ninety


def opp_theta(ox, oy, dx, dy):
    """Inverse of ``slope_theta``"""
    rise = dy - oy
    run = dx - ox
    return opp_theta_rise_run(rise, run)


def truncated_line(leftx, boty, rightx, topy, r, from_start=False):
    """Return coordinates for two points, very much like the two points
    supplied, but with the end of the line foreshortened by amount r.

    """
    # presumes pointed up and right
    if r == 0:
        return (leftx, boty, rightx, topy)
    rise = topy - boty
    run = rightx - leftx
    length = hypot(rise, run) - r
    theta = slope_theta_rise_run(rise, run)
    if from_start:
        leftx = rightx - cos(theta) * length
        boty = topy - sin(theta) * length
    else:
        rightx = leftx + cos(theta) * length
        topy = boty + sin(theta) * length
    return (leftx, boty, rightx, topy)


def wedge_offsets_core(theta, opp_theta, taillen):
    """Internal use"""
    top_theta = theta - fortyfive
    bot_theta = pi - fortyfive - opp_theta
    xoff1 = cos(top_theta) * taillen
    yoff1 = sin(top_theta) * taillen
    xoff2 = cos(bot_theta) * taillen
    yoff2 = sin(bot_theta) * taillen
    return (
        xoff1, yoff1, xoff2, yoff2)


def wedge_offsets_rise_run(rise, run, taillen):
    """Given a line segment's rise, run, and length, return two new
    points--with respect to the *end* of the line segment--that are good
    for making an arrowhead with.

    The arrowhead is a triangle formed from these points and the point at
    the end of the line segment.

    """
    # theta is the slope of a line bisecting the ninety degree wedge.
    theta = slope_theta_rise_run(rise, run)
    opp_theta = opp_theta_rise_run(rise, run)
    return wedge_offsets_core(theta, opp_theta, taillen)


ninety = pi / 2
"""pi / 2"""

fortyfive = pi / 4
"""pi / 4"""
