# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from math import hypot, cos, sin
from util import (
    set_line_width,
    wedge_offsets_rise_run,
    truncated_line,
    fortyfive)
import pyglet


GL_LINES = pyglet.gl.GL_LINES


class DummySpot:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getattr__(self, attrn):
        if attrn in ('window_x', 'viewport_x', 'board_x'):
            return self.x
        elif attrn in ('window_y', 'viewport_y', 'board_y'):
            return self.y
        elif attrn in ('coords', 'window_coords',
                       'viewport_coords', 'board_coords'):
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
        elif attrn == 'left':
            if self.orig.x < self.dest.x:
                return self.orig.x - self.margin
            else:
                return self.dest.x - self.margin
        elif attrn == 'right':
            if self.orig.x < self.dest.x:
                return self.dest.x + self.margin
            else:
                return self.orig.x + self.margin
        elif attrn == 'bot':
            if self.orig.y < self.dest.y:
                return self.orig.y - self.margin
            else:
                return self.dest.y - self.margin
        elif attrn == 'top':
            if self.orig.y < self.dest.y:
                return self.dest.y + self.margin
            else:
                return self.orig.y + self.margin
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
                x > self.left and
                x < self.right and
                y > self.bot and
                y < self.top):
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

    def reciprocate(self):
        # Return the edge of the portal that connects the same two
        # places in the opposite direction, supposing it exists
        try:
            port = self.portal.reciprocal
            return port.arrows[int(self.board)]
        except KeyError:
            return None


class ArrowWidget:
    selectable = True

    def __init__(self, viewport, arrow):
        self.viewport = viewport
        self.window = self.viewport.window
        self.arrow = arrow
        self.vertices = ((None, None), (None, None), (None, None))
        self.order = self.window.edge_order
        self.window.edge_order += 1

    def __getattr__(self, attrn):
        if attrn == "board_left":
            return self.arrow.left
        elif attrn == "board_right":
            return self.arrow.right
        elif attrn == "board_top":
            return self.arrow.top
        elif attrn == "board_bot":
            return self.arrow.bot
        elif attrn == "viewport_left":
            return self.board_left + self.viewport.offset_x
        elif attrn == "viewport_right":
            return self.board_right + self.viewport.offset_x
        elif attrn == "viewport_bot":
            return self.board_bot + self.viewport.offset_y
        elif attrn == "viewport_top":
            return self.board_top + self.viewport.offset_y
        elif attrn == "window_left":
            return self.viewport_left + self.viewport.window_left
        elif attrn == "window_right":
            return self.viewport_right + self.viewport.window_left
        elif attrn == "window_bot":
            return self.viewport_bot + self.viewport.window_bot
        elif attrn == "window_top":
            return self.viewport_top + self.viewport.window_bot
        elif attrn in ("ox", "board_ox"):
            return self.arrow.ox
        elif attrn in ("oy", "board_oy"):
            return self.arrow.oy
        elif attrn in ("dx", "board_dx"):
            return self.arrow.dx
        elif attrn in ("dy", "board_dy"):
            return self.arrow.dy
        elif attrn == "viewport_ox":
            return self.board_ox + self.viewport.offset_x
        elif attrn == "viewport_dx":
            return self.board_dx + self.viewport.offset_x
        elif attrn == "viewport_dy":
            return self.board_dy + self.viewport.offset_y
        elif attrn == "viewport_oy":
            return self.board_oy + self.viewport.offset_y
        elif attrn == "window_ox":
            return self.viewport_ox + self.viewport.window_left
        elif attrn == "window_dx":
            return self.viewport_dx + self.viewport.window_left
        elif attrn == "window_oy":
            return self.viewport_oy + self.viewport.window_bot
        elif attrn == "window_dy":
            return self.viewport_dy + self.viewport.window_bot
        elif attrn in ("highlit", "selected"):
            return self in self.window.selected
        elif attrn in (
                "rise", "run", "width", "length", "m", "slope"):
            return getattr(attrn, self.arrow)
        else:
            raise AttributeError(
                "ArrowWidget instance has no attribute " + attrn)

    def delete(self):
        for vertle in (self.bg_vertlist, self.fg_vertlist):
            try:
                vertle.delete()
            except:
                pass

    def draw(self, batch, group):
        # group had better be viewported
        if not hasattr(self, 'supergroup'):
            self.supergroup = pyglet.graphics.OrderedGroup(
                self.order, group)
        if not hasattr(self, 'bggroup'):
            self.bggroup = SmoothBoldLineOrderedGroup(
                0, self.supergroup, self.window.arrow_girth)
        if not hasattr(self, 'fggroup'):
            self.fggroup = BoldLineOrderedGroup(
                1, self.supergroup, self.window.arrow_width)
        ox = self.viewport_ox
        dx = self.viewport_dx
        oy = self.viewport_oy
        dy = self.viewport_dy
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
            self.center_shrink + 1)
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
        points = (ox, oy, endx, endy, x1, y1, x2, y2)
        bgcolors = bgcolor * 4
        fgcolors = fgcolor * 4
        try:
            self.bg_vertlist.vertices = list(points)
        except:
            self.bg_vertlist = batch.add_indexed(
                4,
                GL_LINES,
                self.bggroup,
                (0, 1, 2, 1, 3),
                ('v2i', points),
                ('c4b', bgcolors))
        try:
            self.fg_vertlist.vertices = list(points)
        except:
            self.fg_vertlist = batch.add_indexed(
                4,
                GL_LINES,
                self.fggroup,
                (0, 1, 2, 1, 3),
                ('v2i', points),
                ('c4b', fgcolors))
