# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from math import sqrt, hypot, atan, pi, sin, cos
from re import match, compile, findall


"""Common utility functions and data structures.

The most important are Skeleton, a mapping used to store and maintain
all game data; and SaveableMetaclass, which generates
SQL from metadata declared as class atttributes.

"""

### Constants

THING_TAB = 0
PLACE_TAB = 2
PORTAL_TAB = 3
CHAR_TAB = 4
THING_CAL = 5
PLACE_CAL = 6
PORTAL_CAL = 7
CHAR_CAL = 8


SHEET_ITEM_TYPES = [
    THING_TAB,
    PLACE_TAB,
    PORTAL_TAB,
    CHAR_TAB,
    THING_CAL,
    PLACE_CAL,
    PORTAL_CAL,
    CHAR_CAL]


TABLE_TYPES = [
    THING_TAB,
    PLACE_TAB,
    PORTAL_TAB,
    CHAR_TAB]


CALENDAR_TYPES = [
    THING_CAL,
    PLACE_CAL,
    PORTAL_CAL,
    CHAR_CAL]

int2pytype = {
    0: bool,
    1: int,
    2: float,
    3: unicode}
"""Arbitrary numerical codes for inbuilt Python types"""

pytype2int = dict([
    (value, key) for (key, value) in int2pytype.iteritems()
])

phi = (1.0 + sqrt(5))/2.0
"""The golden ratio."""

portex = compile("Portal\((.+?)->(.+?)\)")
"""Regular expression to recognize portals by name"""

###
# These regexes serve to parse certain database records that represent
# function calls.
#
# Mainly, that means menu items and Effects.
###
ONE_ARG_RE = compile("(.+)")
TWO_ARG_RE = compile("(.+), ?(.+)")
ITEM_ARG_RE = compile("(.+)\.(.+)")
MAKE_SPOT_ARG_RE = compile(
    "(.+)\."
    "(.+),([0-9]+),([0-9]+),?(.*)")
MAKE_PORTAL_ARG_RE = compile(
    "(.+)\.(.+)->"
    "(.+)\.(.+)")
MAKE_THING_ARG_RE = compile(
    "(.+)\.(.+)@(.+)")
PORTAL_NAME_RE = compile(
    "Portal\((.+)->(.+)\)")
NEW_THING_RE = compile(
    "new_thing\((.+)+\)")
NEW_PLACE_RE = compile(
    "new_place\((.+)\)")
CHARACTER_RE = compile(
    "character\((.+)\)")

### End constants
### Begin functions


def passthru(_):
    return _


def upbranch(closet, bones, branch, tick):
    started = False
    first = None
    for bone in bones:
        if bone.tick >= tick:
            started = True
            yield bone._replace(branch=branch)
        if not started:
            assert(bone.tick < tick)
            first = bone
    if first is not None:
        yield first._replace(
            branch=branch, tick=tick)


def selectif(skel, key):
    if key is None:
        for sk in skel.itervalues():
            yield sk
    else:
        try:
            yield skel[key]
        except (KeyError, IndexError):
            return

### End functions


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


class TimestreamException(Exception):
    """Used for time travel related errors that are nothing to do with
continuity."""
    pass


class TimeParadox(Exception):
    """I tried to record some fact at some time, and in so doing,
    contradicted the historical record."""
    pass


class JourneyException(Exception):
    """There was a problem with pathfinding."""
    pass


class KnowledgeException(Exception):
    """I tried to access some information that I was not permitted access to.

    Should be treated like KeyError most of the time. For the purposes
    of the simulation, not having information is the same as
    information not existing. But there may be circumstances where
    they differ for programming purposes.

    """
    pass


class ListItemIterator:
    """Iterate over a list in a way that resembles dict.iteritems()"""
    def __init__(self, l):
        """Initialize for list l"""
        self.l = l
        self.l_iter = iter(l)
        self.i = 0

    def __iter__(self):
        """I'm an iterator"""
        return self

    def __len__(self):
        """Provide the length of the underlying list."""
        return len(self.l)

    def __next__(self):
        """Return a tuple of the current index and its item in the list"""
        it = next(self.l_iter)
        i = self.i
        self.i += 1
        return (i, it)

    def next(self):
        """Return a tuple of the current index and its item in the list"""
        return self.__next__()


class Fabulator(object):
    """Construct objects (or call functions, as you please) as described
    by strings loaded in from the database.

    This doesn't use exec(). You need to supply the functions when you
    construct the Fabulator.

    """
    def __init__(self, fabs):
        """Supply a dictionary full of callables, keyed by the names you want
        to use for them.

        """
        self.fabbers = fabs

    def __call__(self, s):
        """Parse the string into something I can make a callable from. Then
        make it, using the classes in self.fabbers.

        """
        def _call_recursively(inner, outer):
            fun = self.fabbers[outer]
            # pretty sure parentheses are meaningless inside []
            m = findall("(.+)\((.+)\)[,)] *", inner)
            if len(m) == 0:
                return fun(*inner.split(",").strip(" "))
            elif len(m) == 1:
                (infun, inarg) = m[0]
                infun = self.fabbers[infun]
                inargs = inarg.split(",").strip(" ")
                return fun(infun(*inargs))
            else:
                # This doesn't allow any mixing of function-call arguments
                # with text arguments at the same level. Not optimal.
                return fun(*[self._call_recursively(infun, inarg)
                             for (infun, inarg) in m])
        (outer, inner) = match("(.+)\((.+)\)", s).groups()
        return _call_recursively(outer, inner)


class HandleHandler(object):
    def mk_handles(self, *names):
        for name in names:
            self.mk_handle(name)

    def mk_handle(self, name):
        def register_listener(llist, listener):
            if listener not in llist:
                llist.append(listener)

        def registrar(llist):
            return lambda listener: register_listener(llist, listener)

        def unregister_listener(llist, listener):
            while listener in llist:
                llist.remove(listener)

        def unregistrar(llist):
            return lambda listener: unregister_listener(llist, listener)

        llist = []
        setattr(self, '{}_listeners'.format(name), llist)
        setattr(self, 'register_{}_listener'.format(name),
                registrar(llist))
        setattr(self, 'unregister_{}_listener'.format(name),
                unregistrar(llist))
