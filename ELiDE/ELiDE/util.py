# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) Zachary Spector, ZacharySpector@gmail.com
from kivy.clock import Clock
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from functools import partial
from math import sin, cos, atan, pi, floor, ceil
from array import array

ninety = pi / 2
"""pi / 2"""

fortyfive = pi / 4
"""pi / 4"""


class SelectableRecycleBoxLayout(FocusBehavior, LayoutSelectionBehavior, RecycleBoxLayout):
    pass


class trigger(object):
    """Make a trigger from a method.

    Decorate a method with this and it will become a trigger. Supply a
    numeric parameter to set a timeout.

    Not suitable for methods that expect any arguments other than
    ``dt``. However you should make your method accept ``*args`` for
    compatibility.

    """
    def __init__(self, func_or_timeout):
        if callable(func_or_timeout):
            self.func = func_or_timeout
            self.timeout = 0
        else:
            self.func = None
            self.timeout = func_or_timeout

    def __call__(self, func):
        self.func = func
        return self

    def __get__(self, instance, owner=None):
        if instance is None:
            # EventDispatcher iterates over its attributes before it
            # instantiates.  Don't try making any trigger in that
            # case.
            return
        retval = Clock.create_trigger(
            partial(self.func, instance), self.timeout
        )
        setattr(instance, self.func.__name__, retval)
        return retval


def set_remote_value(loader, remote, k, v):
    if v is None:
        del remote[k]
    else:
        remote[k] = loader(v)


def remote_setter(loader, remote):
    """Return a function taking two arguments, ``k`` and ``v``, which sets
    ``remote[k] = v``, interpreting ``v`` as JSON if possible, or
    deleting ``remote[k]`` if ``v is None``.

    """
    return lambda k, v: set_remote_value(loader, remote, k, v)


def try_load(loader, obj):
    """Return the JSON interpretation the object if possible, or just the
    object otherwise.

    """
    try:
        return loader(obj)
    except (TypeError, ValueError):
        return obj


def dummynum(character, name):
    """Count how many nodes there already are in the character whose name
    starts the same.

    """
    num = 0
    for nodename in character.node:
        nodename = str(nodename)
        if not nodename.startswith(name):
            continue
        try:
            nodenum = int(nodename.lstrip(name))
        except ValueError:
            continue
        num = max((nodenum, num))
    return num


def get_thin_rect_vertices(ox, oy, dx, dy, r):
    """Given the starting point, ending point, and width, return a list of
    vertex coordinates at the corners of the line segment
    (really a thin rectangle).

    """
    if ox < dx:
        leftx = ox
        rightx = dx
        xco = 1
    elif ox > dx:
        leftx = ox * -1
        rightx = dx * -1
        xco = -1
    else:
        return [
            ox - r, oy,
            ox + r, oy,
            ox + r, dy,
            ox - r, dy
        ]
    if oy < dy:
        boty = oy
        topy = dy
        yco = 1
    elif oy > dy:
        boty = oy * -1
        topy = dy * -1
        yco = -1
    else:
        return [
            ox, oy - r,
            dx, oy - r,
            dx, oy + r,
            ox, oy + r
        ]

    rise = topy - boty
    run = rightx - leftx
    theta = atan(rise/run)
    theta_prime = ninety - theta
    xoff = cos(theta_prime) * r
    yoff = sin(theta_prime) * r
    x1 = leftx + xoff
    y1 = boty - yoff
    x2 = rightx + xoff
    y2 = topy - yoff
    x3 = rightx - xoff
    y3 = topy + yoff
    x4 = leftx - xoff
    y4 = boty + yoff
    return [
        x1 * xco, y1 * yco,
        x2 * xco, y2 * yco,
        x3 * xco, y3 * yco,
        x4 * xco, y4 * yco
    ]


class Collide2DPoly(object):
    ''' Collide2DPoly checks whether a point is within a polygon defined by a
    list of corner points.

    Based on http://alienryderflex.com/polygon/

    For example, a simple triangle::

        >>> collider = Collide2DPoly([10., 10., 20., 30., 30., 10.],
        ... cache=True)
        >>> (0.0, 0.0) in collider
        False
        >>> (20.0, 20.0) in collider
        True

    The constructor takes a list of x,y points in the form of [x1,y1,x2,y2...]
    as the points argument. These points define the corners of the
    polygon. The boundary is linearly interpolated between each set of points.
    The x, and y values must be floating points.
    The cache argument, if True, will calculate membership for all the points
    so when collide_point is called it'll just be a table lookup.

    This pure Python version was ported from the kivy.garden.collider
    package.
    '''
    def __init__(self, points, cache=False, **kwargs):
        length = len(points)
        if length % 2:
            raise IndexError()
        if length < 4:
            self.points = None
            return
        count = length // 2
        self.count = count
        self.points = points = array('d', points)
        self.constant = constant = array('d', [0.0] * count)
        self.multiple = multiple = array('d', [0.0] * count)

        self.min_x = min(points[0::2])
        self.max_x = max(points[0::2])
        self.min_y = min(points[1::2])
        self.max_y = max(points[1::2])
        min_x = floor(self.min_x)
        min_y = floor(self.min_y)
        j = count - 1
        if cache:
            for i in range(count):
                points[2 * i] -= min_x
                points[2 * i + 1] -= min_y

        for i in range(count):
            i_x = i * 2
            i_y = i_x + 1
            j_x = j * 2
            j_y = j_x + 1
            if points[j_y] == points[i_y]:
                constant[i] = points[i_x]
                multiple[i] = 0.
            else:
                constant[i] = (points[i_x] - points[i_y] * points[j_x] /
                                (points[j_y] - points[i_y]) +
                                points[i_y] * points[i_x] /
                                (points[j_y] - points[i_y]))
                multiple[i] = ((points[j_x] - points[i_x]) /
                                (points[j_y] - points[i_y]))
            j = i
        if cache:
            width = int(ceil(self.max_x) - min_x + 1.)
            self.width = width
            height = int(ceil(self.max_y) - min_y + 1.)
            self.space = []
            for y in range(height):
                for x in range(width):
                    j = count - 1
                    odd = 0
                    for i in range(count):
                        i_y = i * 2 + 1
                        j_y = j * 2 + 1
                        if (points[i_y] < y and points[j_y] >= y or
                                        points[j_y] < y and points[i_y] >= y):
                            odd ^= y * multiple[i] + constant[i] < x
                        j = i
                    self.space[y * width + x] = odd

    def collide_point(self, x, y):
        points = self.points
        if not points or not (self.min_x <= x <= self.max_x and
                                              self.min_y <= y <= self.max_y):
            return False
        if hasattr(self, 'space'):
            y -= floor(self.min_y)
            x -= floor(self.min_x)
            return self.space[int(y) * self.width + int(x)]

        j = self.count - 1
        odd = 0
        for i in range(self.count):
            i_y = i * 2 + 1
            j_y = j * 2 + 1
            if (points[i_y] < y and points[j_y] >= y or
                            points[j_y] < y and points[i_y] >= y):
                odd ^= y * self.multiple[i] + self.constant[i] < x
            j = i
        return odd

    def __contains__(self, point):
        return self.collide_point(*point)
