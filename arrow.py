# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from math import hypot, cos, sin
from util import (
    set_line_width,
    wedge_offsets_rise_run,
    truncated_line,
    fortyfive)
import pyglet


class DummySpot:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getattr__(self, attrn):
        if attrn == 'window_x':
            return self.x
        elif attrn == 'window_y':
            return self.y
        elif attrn == 'window_coords':
            return (self.x, self.y)
        else:
            raise AttributeError


class BoldLineOrderedGroup(pyglet.graphics.OrderedGroup):
    def __init__(self, order, parent=None, width=1.0):
        self.width = float(width)
        pyglet.graphics.OrderedGroup.__init__(self, order, parent)

    def set_state(self):
        pyglet.gl.glDisable(pyglet.gl.GL_LINE_SMOOTH)
        set_line_width(self.width)


class SmoothBoldLineOrderedGroup(pyglet.graphics.OrderedGroup):
    def __init__(self, order, parent=None, width=1.0):
        self.width = float(width)
        pyglet.graphics.OrderedGroup.__init__(self, order, parent)

    def set_state(self):
        set_line_width(self.width)
        pyglet.gl.glEnable(pyglet.gl.GL_LINE_SMOOTH)


class Arrow:
    margin = 20
    w = 10

    def __init__(self, board, orig_or_port, dest=None):
        self.board = board
        self.window = board.window
        self.rumor = self.board.rumor
        self.center_shrink = 0
        if dest is None:
            if hasattr(orig_or_port, 'orig'):
                port = orig_or_port
                self.orig = self.board.get_spot(port.orig)
                self.dest = self.board.get_spot(port.dest)
                self.portal = port
            else:
                e = orig_or_port
                self.orig = self.board.get_spot(e.source)
                self.dest = self.board.get_spot(e.target)
            self.center_shrink = self.dest.r
        else:
            orig = orig_or_port
            if hasattr(orig, 'v'):
                self.orig = self.board.get_spot(orig)
            elif isinstance(orig, tuple):
                self.orig = DummySpot(*orig)
            else:
                self.orig = orig
            if hasattr(dest, 'v'):
                self.dest = self.board.get_spot(dest)
                self.center_shrink = self.dest.r
            elif isinstance(dest, tuple):
                self.dest = DummySpot(*dest)
            else:
                self.dest = dest
        self.vertices = ((None, None), (None, None), (None, None))
        self.oldstate = None
        self.order = self.window.edge_order
        self.window.edge_order += 1
        self.selectable = True
        self.overlap_hints = {}
        self.y_at_hints = {}
        self.tweaks = 0

    def __getattr__(self, attrn):
        if attrn == 'ox':
            return self.orig.x
        elif attrn == 'oy':
            return self.orig.y
        elif attrn == 'dx':
            return self.dest.x
        elif attrn == 'dy':
            return self.dest.y
        elif attrn == 'rise':
            return self.dest.y - self.orig.y
        elif attrn == 'run':
            return self.dest.x - self.orig.x
        elif attrn == 'width':
            return self.window.arrow_width
        elif attrn == 'length':
            if "branch" in self.portal.e.attribute_names():
                return self.board.get_edge_len(self.portal.e)
            else:
                return hypot(self.rise, self.run)
        elif attrn in ('m', 'slope'):
            ox = self.orig.x
            oy = self.orig.y
            dx = self.dest.x
            dy = self.dest.y
            if oy == dy:
                return 0
            elif ox == dx:
                return None
            else:
                return self.rise / self.run
        elif attrn == 'window_left':
            if self.orig.window_x < self.dest.window_x:
                return self.orig.window_x - self.margin
            else:
                return self.dest.window_x - self.margin
        elif attrn == 'window_right':
            if self.orig.window_x < self.dest.window_x:
                return self.dest.window_x + self.margin
            else:
                return self.orig.window_x + self.margin
        elif attrn == 'window_bot':
            if self.orig.window_y < self.dest.window_y:
                return self.orig.window_y - self.margin
            else:
                return self.dest.window_y - self.margin
        elif attrn == 'window_top':
            if self.orig.window_y < self.dest.window_y:
                return self.dest.window_y + self.margin
            else:
                return self.orig.window_y + self.margin
        elif attrn == 'b':
            # Returns a pair representing a fraction
            # y = mx + b
            # y - b = mx
            # -b = mx - y
            # b = -mx + y
            # b = y - mx
            if self.m is None:
                return None
            denominator = self.run
            x_numerator = self.rise * self.ox
            y_numerator = denominator * self.oy
            return ((y_numerator - x_numerator), denominator)
        elif attrn == 'highlit':
            return self in self.window.selected
        elif attrn == "tick_along":
            # This is for the timestream.  After clicking an arrow you
            # can get the tick represented by the point at which the
            # arrow was clicked.
            return int(self.e["length"] * self.clicked_along)
        else:
            raise AttributeError(
                "Edge instance has no attribute {0}".format(attrn))

    def y_at(self, x):
        if self.m is None:
            return None
        else:
            b = self.b
            mx = (self.rise * x, self.run)
            y = (mx[0] + b[0], self.run)
            return float(y[0]) / float(self.run)

    def x_at(self, y):
        # y = mx + b
        # y - b = mx
        # (y - b)/m = x
        if self.m is None:
            return self.ox
        else:
            b = self.b
            numerator = y - b[1]
            denominator = b[0]
            return float(numerator) / float(denominator)

    def overlaps(self, x, y):
        """Do I overlap the point (x, y)?

Take my width into account

        """
        # trivial rejections
        if not (
                x > self.window_left and
                x < self.window_right and
                y > self.window_bot and
                y < self.window_top):
            return False
        try:
            perfect_x = self.x_at(y)
        except:
            perfect_x = None
        try:
            perfect_y = self.y_at(x)
        except:
            perfect_y = None
        if perfect_x is None:
            return abs(perfect_y - y) < self.width
        elif perfect_y is None:
            return abs(perfect_x - x) < self.width
        else:
            a = perfect_y - y
            b = perfect_x - x
            dist = hypot(a, b)
            return dist < self.width

    def onclick(self, x, y, buttons, modifiers):
        """In most circumstances, this does nothing; the window will handle my
being selected.

In the case where I'm in a timeline, clicking me initiates time travel.

        """
        if "branch" in self.portal.e.attribute_names():
            branch = self.portal.e["branch"]
            length = self.portal.e["length"]
            start = self.orig.v["tick"]
            self.rumor.time_travel(
                branch, start + (length * self.clicked_along))
            self.window.onscreen = set()

    def get_state_tup(self):
        return ((self.tweaks, self.highlit) +
                self.orig.get_state_tup() +
                self.dest.get_state_tup())

    def reciprocate(self):
        # Return the edge of the portal that connects the same two
        # places in the opposite direction, supposing it exists
        try:
            port = self.portal.reciprocal
            return port.arrows[int(self.board)]
        except KeyError:
            return None

    def delete(self):
        for pair in self.vertices:
            for vertle in pair:
                try:
                    vertle.delete()
                except:
                    pass

    def draw(self, batch, group):
        supergroup = pyglet.graphics.OrderedGroup(
            self.order, group)
        bggroup = SmoothBoldLineOrderedGroup(
            0, supergroup, self.window.arrow_girth)
        fggroup = BoldLineOrderedGroup(
            1, supergroup, self.window.arrow_width)
        owc = self.orig.window_coords
        dwc = self.dest.window_coords
        if None in (owc, dwc):
            return
        (ox, oy) = owc
        (dx, dy) = dwc
        if dy < oy:
            yco = -1
        else:
            yco = 1
        if dx < ox:
            xco = -1
        else:
            xco = 1
        (leftx, boty, rightx, topy) = truncated_line(
            float(ox * xco), float(oy * yco),
            float(dx * xco), float(dy * yco),
            self.center_shrink+1)
        taillen = float(self.window.arrowhead_size)
        rise = topy - boty
        run = rightx - leftx
        if rise == 0:
            xoff1 = cos(fortyfive) * taillen
            yoff1 = xoff1
            xoff2 = xoff1
            yoff2 = -1 * yoff1
        elif run == 0:
            xoff1 = sin(fortyfive) * taillen
            yoff1 = xoff1
            xoff2 = -1 * xoff1
            yoff2 = yoff1
        else:
            (xoff1, yoff1, xoff2, yoff2) = wedge_offsets_rise_run(
                rise, run, taillen)
        x1 = int(rightx - xoff1) * xco
        x2 = int(rightx - xoff2) * xco
        y1 = int(topy - yoff1) * yco
        y2 = int(topy - yoff2) * yco
        endx = int(rightx) * xco
        endy = int(topy) * yco
        if self.highlit:
            bgcolor = (255, 255, 0, 0)
            fgcolor = (0, 0, 0, 0)
        else:
            bgcolor = (64, 64, 64, 64)
            fgcolor = (255, 255, 255, 0)
        lpoints = (x1, y1, endx, endy)
        cpoints = (ox, oy, endx, endy)
        rpoints = (x2, y2, endx, endy)
        lbg = self.window.draw_line(
            lpoints, bgcolor, bggroup, self.vertices[0][0])
        cbg = self.window.draw_line(
            cpoints, bgcolor, bggroup, self.vertices[1][0])
        rbg = self.window.draw_line(
            rpoints, bgcolor, bggroup, self.vertices[2][0])
        lfg = self.window.draw_line(
            lpoints, fgcolor, fggroup, self.vertices[0][1])
        cfg = self.window.draw_line(
            cpoints, fgcolor, fggroup, self.vertices[1][1])
        rfg = self.window.draw_line(
            rpoints, fgcolor, fggroup, self.vertices[2][1])
        self.vertices = ((lbg, lfg), (cbg, cfg), (rbg, rfg))
