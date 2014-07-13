# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Common utility functions and data structures.

"""
from math import sqrt, hypot, atan, pi, sin, cos

### Constants
phi = (1.0 + sqrt(5))/2.0
"""The golden ratio."""

### End constants
### Begin functions


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
