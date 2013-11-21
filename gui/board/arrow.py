# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import print_function
from math import cos, sin, hypot, atan
from util import (
    wedge_offsets_rise_run,
    truncated_line,
    fortyfive)
from kivy.graphics import Line, Color
from kivy.uix.widget import Widget
from kivy.properties import (
    ObjectProperty,
    BooleanProperty)
from kivy.clock import Clock


class Arrow(Widget):
    margin = 10
    w = 1
    board = ObjectProperty()
    portal = ObjectProperty()
    dragging = BooleanProperty(False)

    def __init__(self, **kwargs):
        Widget.__init__(self, **kwargs)
        self.upd_pos_size()
        orign = unicode(self.portal.origin)
        destn = unicode(self.portal.destination)
        self.board.spotdict[orign].bind(
            pos=self.realign,
            size=self.realign,
            transform=self.realign)
        self.board.spotdict[destn].bind(
            pos=self.realign,
            size=self.realign,
            transform=self.realign)
        self.bg_color = Color(0.25, 0.25, 0.25)
        self.fg_color = Color(1.0, 1.0, 1.0)
        self.bg_line = Line(width=self.w * 1.4)
        self.fg_line = Line(width=self.w)
        self.canvas.add(self.bg_color)
        self.canvas.add(self.bg_line)
        self.canvas.add(self.fg_color)
        self.canvas.add(self.fg_line)

        def startup(*args):
            self.parent.bind(pos=self.realign, size=self.realign)
            self.realign()

        Clock.schedule_once(startup, 0)

    def __unicode__(self):
        return unicode(self.portal)

    def __str__(self):
        return str(self.portal)

    @property
    def reciprocal(self):
        # Return the edge of the portal that connects the same two
        # places in the opposite direction, supposing it exists
        try:
            return self.portal.reciprocal.arrow
        except KeyError:
            return None

    def get_points(self):
        orig = self.board.spotdict[unicode(self.portal.origin)]
        dest = self.board.spotdict[unicode(self.portal.destination)]
        (ox, oy) = orig.pos
        # orig.size is SUPPOSED to be the same as orig.tex.size but
        # sometimes it isn't, because threading
        (ow, oh) = orig.tex.size
        orx = ow / 2
        ory = ow / 2
        ox += orx
        oy += ory
        (dx, dy) = dest.pos
        (dw, dh) = dest.tex.size
        drx = dw / 2
        dry = dh / 2
        dx += drx
        dy += dry

        if drx > dry:
            dr = drx
        else:
            dr = dry
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
            dr + 1)
        taillen = float(self.board.arrowhead_size)
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
        x1 = (rightx - xoff1) * xco
        x2 = (rightx - xoff2) * xco
        y1 = (topy - yoff1) * yco
        y2 = (topy - yoff2) * yco
        endx = rightx * xco
        endy = topy * yco
        r = [ox, oy,
             endx, endy, x1, y1,
             endx, endy, x2, y2,
             endx, endy]
        for coord in r:
            assert(coord > 0.0)
            assert(coord < 1000.0)
        return r

    def get_slope(self):
        orig = self.board.spotdict[unicode(self.portal.origin)]
        dest = self.board.spotdict[unicode(self.portal.destination)]
        ox = orig.x
        oy = orig.y
        dx = dest.x
        dy = dest.y
        if oy == dy:
            return 0
        elif ox == dx:
            return None
        else:
            rise = dy - oy
            run = dx - ox
            return rise / run

    def get_b(self):
        orig = self.board.spotdict[unicode(self.portal.origin)]
        dest = self.board.spotdict[unicode(self.portal.destination)]
        (ox, oy) = orig.pos
        (dx, dy) = dest.pos
        denominator = dx - ox
        x_numerator = (dy - oy) * ox
        y_numerator = denominator * oy
        return ((y_numerator - x_numerator), denominator)

    def realign(self, *args):
        self.upd_pos_size()
        points = self.get_points()
        self.bg_line.points = points
        self.fg_line.points = points

    def upd_pos_size(self, *args):
        orig = self.board.spotdict[unicode(self.portal.origin)]
        dest = self.board.spotdict[unicode(self.portal.destination)]
        (ox, oy) = orig.pos
        (dx, dy) = dest.pos
        w = dx - ox
        h = dy - oy
        self.pos = (ox, oy)
        self.size = (w, h)

    def collide_point(self, x, y):
        if not super(Arrow, self).collide_point(x, y):
            return False
        orig = self.board.spotdict[unicode(self.portal.origin)]
        dest = self.board.spotdict[unicode(self.portal.destination)]
        (ox, oy) = orig.pos
        (dx, dy) = dest.pos
        if ox == dx:
            return abs(y - dy) <= self.w
        elif oy == dy:
            return abs(x - dx) <= self.w
        else:
            correct_angle_a = atan(dy / dx)
            observed_angle_a = atan(y / x)
            error_angle_a = abs(observed_angle_a - correct_angle_a)
            error_seg_len = hypot(x, y)
            return sin(error_angle_a) * error_seg_len <= self.margin

    def on_drop(self):
        pass
