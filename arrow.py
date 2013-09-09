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
    atrdic = {
            'ox': lambda self: self.orig.x,
            'oy': lambda self: self.orig.y,
            'dx': lambda self: self.dest.x,
            'dy': lambda self: self.dest.y,
            'rise': lambda self: self.dest.y - self.orig.y,
            'run': lambda self: self.dest.x - self.orig.x,
            'length': lambda self: {
                True: self.board.get_edge_len(self.portal.e),
                False: hypot(self.rise, self.run)
            }["branch" in self.portal.e.attribute_names()],
            'slope': lambda self: self.get_slope(),
            'm': lambda self: self.get_slope(),
            'left': lambda self: self.get_left(),
            'right': lambda self: self.get_right(),
            'bot': lambda self: self.get_bot(),
            'bottom': lambda self: self.get_bot(),
            'top': lambda self: self.get_top(),
            'b': lambda self: self.get_b()
        }

    def __init__(self, board, orig_or_port, dest=None):
        self.board = board
        self.closet = self.board.closet
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
        try:
            return self.atrdic[attrn](self)
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
    def yint(self):
        ab = self.arrow.b
        return (ab[0] + self.viewport.offset_x * ab[1], ab[1])
    atrdic = {
        "board_left": lambda self: self.arrow.left,
        "board_right": lambda self: self.arrow.right,
        "board_top": lambda self: self.arrow.top,
        "board_bot": lambda self: self.arrow.bot,
        "viewport_left": lambda self: self.board_left + self.viewport.offset_x,
        "viewport_right": lambda self: self.board_right + self.viewport.offset_x,
        "viewport_top": lambda self: self.board_top + self.viewport.offset_y,
        "viewport_bot": lambda self: self.board_bot + self.viewport.offset_y,
        "window_left": lambda self: self.viewport_left + self.viewport.window_left,
        "window_right": lambda self: self.viewport_right + self.viewport.window_left,
        "window_bot": lambda self: self.viewport_bot + self.viewport.window_bot,
        "window_top": lambda self: self.viewport_top + self.viewport.window_bot,
        "ox": lambda self: self.orig.board_x,
        "oy": lambda self: self.orig.board_y,
        "dx": lambda self: self.dest.board_x,
        "dy": lambda self: self.dest.board_y,
        "window_ox": lambda self: self.orig.window_x,
        "window_oy": lambda self: self.orig.window_y,
        "window_dx": lambda self: self.dest.window_x,
        "window_dy": lambda self: self.dest.window_y,
        "selected": lambda self: self in self.window.selected,
        "orig": lambda self: self.viewport.spotdict[str(self.arrow.orig)],
        "dest": lambda self: self.viewport.spotdict[str(self.arrow.dest)],
        "in_view": lambda self: self.orig.in_view or self.dest.in_view,
        "b": lambda self: self.yint(),
        "width": lambda self: self.viewport.arrow_width}

    def __init__(self, viewport, arrow):
        self.viewport = viewport
        self.window = self.viewport.window
        self.batch = self.window.batch
        self.arrow = arrow
        self.bgvl = None
        self.fgvl = None
        self.order = self.window.edge_order
        self.window.edge_order += 1
        self.bggroup = SmoothBoldLineOrderedGroup(
            0, self.viewport.arrowgroup, self.viewport.arrow_width)
        self.fggroup = BoldLineOrderedGroup(
            1, self.viewport.arrowgroup, self.viewport.arrow_width)
        self.old_state = None

    def __getattr__(self, attrn):
        if attrn in (
                "rise", "run", "length", "m", "slope",
                "center_shrink", "portal", "e"):
            return getattr(self.arrow, attrn)
        elif attrn in ArrowWidget.atrdic:
            return ArrowWidget.atrdic[attrn](self)
        else:
            raise AttributeError(
                "ArrowWidget instance has no attribute named {0}".format(attrn))

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
        ox = self.window_ox
        dx = self.window_dx
        oy = self.window_oy
        dy = self.window_dy
        try:
            self.really_draw(ox, oy, dx, dy)
        except:
            if self.bggroup is not None:
                try:
                    self.bggroup.delete()
                except:
                    pass
                self.bggroup = None
            if self.fggroup is not None:
                try:
                    self.fggroup.delete()
                except:
                    pass
                self.fggroup = None

    def really_draw(self, ox, oy, dx, dy):
        # group had better be viewported
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
        if self.selected:
            bgcolor = (255, 255, 0, 0)
            fgcolor = (0, 0, 0, 0)
        else:
            bgcolor = (64, 64, 64, 64)
            fgcolor = (255, 255, 255, 0)
        vees = (ox, oy, endx, endy, x1, y1, endx, endy, x2, y2, endx, endy)
        if self.bgvl is None:
            self.bgvl = self.batch.add(
                6,
                GL_LINES,
                self.bggroup,
                ('v2i', vees),
                ('c4B', bgcolor * 6))
        else:
            self.bgvl.vertices = vees
        if self.fgvl is None:
            self.fgvl = self.batch.add(
                6,
                GL_LINES,
                self.fggroup,
                ('v2i', vees),
                ('c4B', fgcolor * 6))
        else:
            self.fgvl.vertices = vees
