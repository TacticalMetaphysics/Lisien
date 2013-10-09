# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import unicode_literals
ascii = str
str = unicode
from math import hypot, cos, sin
from util import (
    wedge_offsets_rise_run,
    truncated_line,
    fortyfive)
from kivy.graphics import Line, Color
from kivy.graphics.instructions import InstructionGroup
from kivy.uix.widget import Widget


class Arrow(Widget):
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

    def __init__(self, board, portal):
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

    def get_lines_instructions(self, offx=0, offy=0, group=None):
        if self.dest.y < self.orig.y:
            yco = -1
        else:
            yco = 1
        if self.dest.x < self.orig.x:
            xco = -1
        else:
            xco = 1
        (leftx, boty, rightx, topy) = truncated_line(
            float(self.orig.x * xco), float(self.orig.y * yco),
            float(self.dest.x * xco), float(self.dest.y * yco),
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
        points = [self.orig.x + offx, self.orig.y + offy,
                  endx + offx, endy + offy, x1 + offx, y1 + offy,
                  endx + offx, endy + offy, x2 + offx, y2 + offy,
                  endx + offx, endy + offy]
        if self.selected:
            bgcolor = (255, 255, 0)
            fgcolor = (0, 0, 0)
        else:
            bgcolor = (64, 64, 64)
            fgcolor = (255, 255, 255)
        if group is None:
            group = InstructionGroup()
        group.add(Color(*bgcolor))
        group.add(Line(points=points, width=self.w))
        group.add(Color(*fgcolor))
        group.add(Line(points=points, width=self.w/3))
        return group

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
