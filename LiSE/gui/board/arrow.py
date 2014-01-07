# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import print_function
from math import cos, sin, hypot, atan
from LiSE.util import (
    wedge_offsets_rise_run,
    truncated_line,
    fortyfive)
from kivy.graphics import Line, Color
from kivy.uix.widget import Widget
from kivy.properties import (
    ObjectProperty,
    ListProperty)
from kivy.clock import Clock


def get_points(ox, orx, oy, ory, dx, drx, dy, dry, taillen):
    ox += orx
    oy += ory
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
    return r


class Arrow(Widget):
    """A widget that points from one :class:`~LiSE.gui.board.Spot` to
    another.

    :class:`Arrow`s are the graphical representations of
    :class:`~LiSE.model.Portal`s. They point from the :class:`Spot`
    representing the :class:`Portal`'s origin, to the one representing
    its destination.

    """
    margin = 10
    """When deciding whether a touch collides with me, how far away can
    the touch get before I should consider it a miss?"""
    w = 1
    """The width of the inner, brighter portion of the :class:`Arrow`. The
    whole :class:`Arrow` will end up thicker."""
    board = ObjectProperty()
    """The board on which I am displayed."""
    portal = ObjectProperty()
    """The portal that I represent."""
    """Pawns that are part-way through me. Each needs to present a
    'progress' property to let me know how far through me they ought to be
    repositioned."""
    pawns_here = ListProperty([])

    def __init__(self, **kwargs):
        """Bind some properties, and put the relevant instructions into the
        canvas--but don't put any point data into the instructions
        just yet. For that, wait until ``on_parent``, when we are
        guaranteed to know the positions of our endpoints.

        """
        Widget.__init__(self, **kwargs)
        self.trigger_repoint = Clock.create_trigger(
            self.repoint, timeout=-1)
        self.board.arrowdict[unicode(self.portal)] = self
        orign = unicode(self.portal.origin)
        destn = unicode(self.portal.destination)
        self.board.spotdict[orign].bind(
            pos=self.trigger_repoint,
            size=self.trigger_repoint)
        self.board.spotdict[destn].bind(
            pos=self.trigger_repoint,
            size=self.trigger_repoint)
        self.board.host.closet.register_time_listener(self.repawn)
        self.bind(pos=self.trigger_repoint)
        self.bind(size=self.trigger_repoint)
        self.bg_color = Color(0.25, 0.25, 0.25)
        self.fg_color = Color(1.0, 1.0, 1.0)
        self.bg_line = Line(width=self.w * 1.4)
        self.fg_line = Line(width=self.w)
        self.canvas.add(self.bg_color)
        self.canvas.add(self.bg_line)
        self.canvas.add(self.fg_color)
        self.canvas.add(self.fg_line)

    def __unicode__(self):
        """Return Unicode name of my :class:`Portal`"""
        return unicode(self.portal)

    def __str__(self):
        """Return string name of my :class:`Portal`"""
        return str(self.portal)

    @property
    def reciprocal(self):
        """If it exists, return the edge of the :class:`Portal` that connects
        the same two places that I do, but in the opposite
        direction. Otherwise, return ``None``.

        """
        # Return the edge of the portal that connects the same two
        # places in the opposite direction, supposing it exists
        try:
            return self.portal.reciprocal.arrow
        except KeyError:
            return None

    def handle_time(self, b, t):
        pass

    def get_points(self):
        """Return the coordinates of the points that describe my shape."""
        orig = self.board.spotdict[unicode(self.portal.origin)]
        dest = self.board.spotdict[unicode(self.portal.destination)]
        (ox, oy) = orig.pos
        try:
            (ow, oh) = orig.size
        except AttributeError:
            (ow, oh) = (0, 0)
        taillen = float(self.board.arrowhead_size)
        orx = ow / 2
        ory = ow / 2
        (dx, dy) = dest.pos
        try:
            (dw, dh) = dest.size
        except AttributeError:
            (dw, dh) = (0, 0)
        drx = dw / 2
        dry = dh / 2
        return get_points(ox, orx, oy, ory, dx, drx, dy, dry, taillen)

    def get_slope(self):
        """Return a float of the increase in y divided by the increase in x,
        both from left to right."""
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
        """Return my Y-intercept.

        I probably don't really hit the left edge of the window, but
        this is where I would, if I were long enough.

        """
        orig = self.board.spotdict[unicode(self.portal.origin)]
        dest = self.board.spotdict[unicode(self.portal.destination)]
        (ox, oy) = orig.pos
        (dx, dy) = dest.pos
        denominator = dx - ox
        x_numerator = (dy - oy) * ox
        y_numerator = denominator * oy
        return ((y_numerator - x_numerator), denominator)

    def repoint(self, *args):
        """Recalculate all my points and redraw. Reposition any pawns on
        me."""
        points = self.get_points()
        self.bg_line.points = points
        self.fg_line.points = points
        origspot = self.board.spotdict[self.portal.origin.name]
        destspot = self.board.spotdict[self.portal.destination.name]
        (ox, oy) = origspot.pos
        (dx, dy) = destspot.pos
        (branch, tick) = self.board.host.sanetime(None, None)

    def collide_point(self, x, y):
        """Return True iff the point falls sufficiently close to my core line
        segment to count as a hit.

        """
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

    def repawn(self, branch, tick):
        for pawn in self.pawns_here:
            locations = pawn.thing.get_locations(
                self.board.facade.observer, branch)
            bone = locations.value_during(tick)
            t1 = bone.tick
            try:
                t2 = locations.key_after(tick)
            except ValueError:
                continue
            if t2 is None:
                continue
            else:
                duration = float(t2 - t1)
                passed = float(tick - t1)
                progress = passed / duration
            os = self.board.spotdict[unicode(self.portal.origin)]
            ds = self.board.spotdict[unicode(self.portal.destination)]
            (ox, oy) = os.pos
            (dx, dy) = ds.pos
            w = dx - ox
            h = dy - oy
            x = ox + w * progress
            y = oy + h * progress
            pawn.pos = (x, y)

    def on_pawns_here(self, i, v):
        (branch, tick) = self.board.host.sanetime(None, None)
        self.repawn(branch, tick)
