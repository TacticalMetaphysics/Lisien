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
        self.atrdic = {
            'ox': lambda: self.orig.x,
            'oy': lambda: self.orig.y,
            'dx': lambda: self.dest.x,
            'dy': lambda: self.dest.y,
            'rise': lambda: self.dest.y - self.orig.y,
            'run': lambda: self.dest.x - self.orig.x,
            'length': self.get_length,
            'slope': self.get_slope,
            'm': self.get_slope,
            'left': self.get_left,
            'right': self.get_right,
            'bot': self.get_bot,
            'bottom': self.get_bot,
            'top': self.get_top,
            'b': self.get_b
        }

    def __getattr__(self, attrn):
        try:
            return self.atrdic[attrn]()
        except IndexError:
            raise AttributeError(
                "Edge instance has no attribute {0}".format(attrn))

    def reciprocate(self):
        # Return the edge of the portal that connects the same two
        # places in the opposite direction, supposing it exists
        try:
            port = self.portal.reciprocal
            return port.arrows[int(self.board)]
        except KeyError:
            return None

    def get_length(self):
        if "branch" in self.portal.e.attribute_names():
            return self.board.get_edge_len(self.portal.e)
        else:
            return hypot(self.rise, self.run)

    def get_slope(self):
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

    def get_left(self):
        if self.orig.x < self.dest.x:
            return self.orig.x - self.margin
        else:
            return self.dest.x - self.margin

    def get_right(self):
        if self.orig.x < self.dest.x:
            return self.dest.x + self.margin
        else:
            return self.orig.x + self.margin

    def get_bot(self):
        if self.orig.y < self.dest.y:
            return self.orig.y - self.margin
        else:
            return self.dest.y - self.margin

    def get_top(self):
        if self.orig.y < self.dest.y:
            return self.dest.y + self.margin
        else:
            return self.orig.y + self.margin

    def get_b(self):
        if self.m is None:
            return None
        denominator = self.run
        x_numerator = self.rise * self.ox
        y_numerator = denominator * self.oy
        return ((y_numerator - x_numerator), denominator)


class ArrowWidget:
    selectable = True

    def __init__(self, viewport, arrow):
        self.viewport = viewport
        self.window = self.viewport.window
        self.batch = self.window.batch
        self.arrow = arrow
        self.shaft_bg_vertlist = None
        self.wedge_bg_vertlist = None
        self.shaft_fg_vertlist = None
        self.wedge_fg_vertlist = None
        self.bgvldict = {}
        self.fgvldict = {}
        self.order = self.window.edge_order
        self.window.edge_order += 1
        self.bggroup = SmoothBoldLineOrderedGroup(
            0, self.viewport.arrowgroup, self.viewport.arrow_width)
        self.fggroup = BoldLineOrderedGroup(
            1, self.viewport.arrowgroup, self.viewport.arrow_width)
        self.old_state = None
        def b():
            ab = self.arrow.b
            return (ab[0] + self.viewport.offset_x * ab[1], ab[1])
        def mkstate():
            return (
                self.viewport.window_left,
                self.viewport.window_bot,
                self.viewport.view_left,
                self.viewport.view_bot,
                self.orig.spot.coords,
                self.dest.spot.coords)
        self.atrdic = {
            "board_left": lambda: self.arrow.left,
            "board_right": lambda: self.arrow.right,
            "board_top": lambda: self.arrow.top,
            "board_bot": lambda: self.arrow.bot,
            "viewport_left": lambda: self.board_left + self.viewport.offset_x,
            "viewport_right": lambda: self.board_right + self.viewport.offset_x,
            "viewport_top": lambda: self.board_top + self.viewport.offset_y,
            "viewport_bot": lambda: self.board_bot + self.viewport.offset_y,
            "window_left": lambda: self.viewport_left + self.viewport.window_left,
            "window_right": lambda: self.viewport_right + self.viewport.window_left,
            "window_bot": lambda: self.viewport_bot + self.viewport.window_bot,
            "window_top": lambda: self.viewport_top + self.viewport.window_bot,
            "ox": lambda: self.arrow.ox,
            "oy": lambda: self.arrow.oy,
            "dx": lambda: self.arrow.dx,
            "dy": lambda: self.arrow.dy,
            "viewport_ox": lambda: self.board_ox + self.viewport.offset_x,
            "viewport_dx": lambda: self.board_dx + self.viewport.offset_x,
            "viewport_dy": lambda: self.board_dy + self.viewport.offset_y,
            "viewport_oy": lambda: self.board_oy + self.viewport.offset_y,
            "window_ox": lambda: self.viewport_ox + self.viewport.window_left,
            "window_dx": lambda: self.viewport_dx + self.viewport.window_left,
            "window_oy": lambda: self.viewport_oy + self.viewport.window_bot,
            "window_dy": lambda: self.viewport_dy + self.viewport.window_bot,
            "selected": lambda: self in self.window.selected,
            "orig": lambda: self.viewport.spotdict[str(self.arrow.orig)],
            "dest": lambda: self.viewport.spotdict[str(self.arrow.dest)],
            "in_view": lambda: self.orig.in_view or self.dest.in_view,
            "b": b,
            "width": lambda: self.viewport.arrow_width,
            "state": mkstate}

    def __getattr__(self, attrn):
        if attrn in (
                "rise", "run", "length", "m", "slope",
                "center_shrink", "portal", "e"):
            return getattr(self.arrow, attrn)
        else:
            try:
                return self.atrdic[attrn]()
            except KeyError:
                raise AttributeError(
                    "ArrowWidget instance has no attribute " + attrn)

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
                x > self.viewport_left and
                x < self.viewport_right and
                y > self.viewport_bot and
                y < self.viewport_top):
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

    def pass_focus(self):
        return self.viewport

    def draw(self):
        # group had better be viewported
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
        taillen = float(self.viewport.arrowhead_size)
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
        bgcolors = bgcolor * 2
        fgcolors = fgcolor * 2
        if (ox > 0 and oy > 0) or (endx > 0 and endy > 0):
            try:
                self.bgvldict['shaft'].vertices = [ox, oy, endx, endy]
            except:
                self.bgvldict['shaft'] = self.batch.add(
                    2,
                    pyglet.gl.GL_LINES,
                    self.bggroup,
                    ('v2i', (ox, oy, endx, endy)),
                    ('c4B', bgcolors))
            try:
                self.fgvldict['shaft'].vertices = [ox, oy, endx, endy]
            except:
                self.fgvldict['shaft'] = self.batch.add(
                    2,
                    pyglet.gl.GL_LINES,
                    self.fggroup,
                    ('v2i', (ox, oy, endx, endy)),
                    ('c4B', fgcolors))
        else:
            if 'shaft' in self.bgvldict:
                try:
                    self.bgvldict['shaft'].delete()
                except:
                    pass
                del self.bgvldict['shaft']
            if 'shaft' in self.fgvldict:
                try:
                    self.fgvldict['shaft'].delete()
                except:
                    pass
                del self.fgvldict['shaft']
        if (x1 > 0 and y1 > 0) or (endx > 0 and endy > 0):
            try:
                self.bgvldict['left'].vertices = [x1, y1, endx, endy]
            except:
                self.bgvldict['left'] = self.batch.add(
                    2,
                    pyglet.gl.GL_LINES,
                    self.bggroup,
                    ('v2i', (x1, y1, endx, endy)),
                    ('c4B', bgcolors))
            try:
                self.fgvldict['left'].vertices = [x1, y1, endx, endy]
            except:
                self.fgvldict['left'] = self.batch.add(
                    2,
                    pyglet.gl.GL_LINES,
                    self.fggroup,
                    ('v2i', (x1, y1, endx, endy)),
                    ('c4B', fgcolors))
        else:
            if 'left' in self.bgvldict:
                try:
                    self.bgvldict['left'].delete()
                except:
                    pass
                del self.bgvldict['left']
            if 'left' in self.fgvldict:
                try:
                    self.fgvldict['left'].delete()
                except:
                    pass
                del self.fgvldict['left']
        if (x2 > 0 and y2 > 0) or (endx > 0 and endy > 0):
            try:
                self.bgvldict['right'].vertices = [x2, y2, endx, endy]
            except:
                self.bgvldict['right'] = self.batch.add(
                    2,
                    pyglet.gl.GL_LINES,
                    self.bggroup,
                    ('v2i', (x2, y2, endx, endy)),
                    ('c4B', bgcolors))
            try:
                self.fgvldict['right'].vertices = [x2, y2, endx, endy]
            except:
                self.fgvldict['right'] = self.batch.add(
                    2,
                    pyglet.gl.GL_LINES,
                    self.fggroup,
                    ('v2i', (x2, y2, endx, endy)),
                    ('c4B', fgcolors))
        else:
            if 'right' in self.bgvldict:
                try:
                    self.bgvldict['right'].delete()
                except:
                    pass
                del self.bgvldict['right']
            if 'right' in self.fgvldict:
                try:
                    self.fgvldict['right'].delete()
                except:
                    pass
                del self.fgvldict['right']
