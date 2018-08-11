# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
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
from kivy.clock import Clock
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from functools import partial
from math import sin, cos, atan, pi

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