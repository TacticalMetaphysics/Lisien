# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
"""That which displays a one-way connection between two places.

An arrow connects two spots, the origin and the destination, and it
points from the origin to the destination, regardless of where on the
screen they are at the moment.

"""
from math import cos, sin, hypot, atan, pi
from kivy.graphics import Line, Color
from kivy.uix.widget import Widget
from kivy.properties import (
    ObjectProperty,
    NumericProperty,
    ListProperty,
    BooleanProperty
)
from kivy.clock import Clock

ninety = pi / 2
"""pi / 2"""

fortyfive = pi / 4
"""pi / 4"""


def get_points(ox, oy, ro, dx, dy, rd, taillen):
    """Return points to use for an arrow from ``ox,oy`` to ``dx,dy`` where
    the origin has dimensions ``2*orx,2*ory``, the destination has
    dimensions ``2*drx,2*dry``, and the bits of the arrow not actually
    connecting the ends of it--the edges of the arrowhead--have length
    ``taillen``.

    """
    # handle special cases;
    # flip the arrow, if needed, to make it point up and right;
    # store coefficients to flip it back again
    if ox < dx:
        leftx = ox
        rightx = dx
        xco = 1
    elif ox > dx:
        leftx = ox * -1
        rightx = dx * -1
        xco = -1
    else:
        off1 = cos(fortyfive) * taillen
        off2 = sin(fortyfive) * taillen
        x0 = ox
        y0 = oy + ro if oy < dy else oy - ro
        endx = dx
        endy = dy - rd if oy < dy else dy + rd
        x1 = endx - off1
        x2 = endx + off1
        y1 = y2 = endy - off2 if oy < dy else endy + off2
        return [x0, y0, endx, endy, x1, y1, endx, endy, x2, y2, endx, endy]
    if oy < dy:
        boty = oy
        topy = dy
        yco = 1
    elif oy > dy:
        boty = oy * -1
        topy = dy * -1
        yco = -1
    else:
        off1 = cos(fortyfive) * taillen
        off2 = sin(fortyfive) * taillen
        x0 = ox + ro if ox < dx else ox - ro
        y0 = oy
        endx = dx - rd if ox < dx else dx + rd
        endy = dy
        y1 = endy - off1
        y2 = endy + off1
        x1 = x2 = endx - off2 if ox < dx else endx + off2
        return [x0, y0, endx, endy, x1, y1, endx, endy, x2, y2, endx, endy]

    rise = topy - boty
    run = rightx - leftx

    # truncate the end so it just touches the destination circle.
    start_theta = atan(rise/run)
    end_theta = atan(run/rise)
    length = hypot(run, rise) - rd
    rightx = leftx + cos(start_theta) * length
    topy = boty + sin(start_theta) * length
    # truncate the start so it's at the very edge of the origin circle.
    length -= ro
    leftx = rightx - sin(end_theta) * length
    boty = topy - cos(end_theta) * length

    # make the little wedge at the end so you can tell which way the
    # arrow's pointing, and flip it all back around to the way it was
    top_theta = start_theta - fortyfive
    bot_theta = pi - fortyfive - end_theta
    xoff1 = cos(top_theta) * taillen
    yoff1 = sin(top_theta) * taillen
    xoff2 = cos(bot_theta) * taillen
    yoff2 = sin(bot_theta) * taillen
    x1 = (rightx - xoff1) * xco
    x2 = (rightx - xoff2) * xco
    y1 = (topy - yoff1) * yco
    y2 = (topy - yoff2) * yco
    startx = leftx * xco
    starty = boty * yco
    endx = rightx * xco
    endy = topy * yco
    return [startx, starty, endx, endy, x1, y1, endx, endy, x2, y2, endx, endy]


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
    pawns_here = ListProperty([])
    points = ListProperty([])
    slope = NumericProperty(0.0, allownone=True)
    y_intercept = NumericProperty(0)
    origin = ObjectProperty()
    destination = ObjectProperty()
    reciprocal = ObjectProperty(None, allownone=True)
    engine = ObjectProperty()
    selected = BooleanProperty()
    repointed = BooleanProperty(True)

    @property
    def reciprocal(self):
        orign = self.portal['origin']
        destn = self.portal['destination']
        if (
                destn in self.board.arrow and
                orign in self.board.arrow[destn]
        ):
            return self.board.arrow[destn][orign]
        else:
            return None

    def __init__(self, **kwargs):
        """Bind some properties, and put the relevant instructions into the
        canvas--but don't put any point data into the instructions
        just yet. For that, wait until ``on_parent``, when we are
        guaranteed to know the positions of our endpoints.

        """
        super().__init__(**kwargs)
        self._trigger_repoint = Clock.create_trigger(
            self._repoint,
            timeout=-1
        )
        self.finalize()

    def finalize(self, *args):
        if None in (
                self.board,
                self.engine,
                self.portal
        ):
            Clock.schedule_once(self.finalize, 0)
            if self.board is None:
                print("no board")
            if self.engine is None:
                print("no engine")
            if self.portal is None:
                print("no portal")
            return
        orign = self.portal["origin"]
        destn = self.portal["destination"]
        self.origin = self.board.spot[orign]
        self.origin.bind(
            pos=self._trigger_repoint,
            size=self._trigger_repoint
        )
        self.destination = self.board.spot[destn]
        self.destination.bind(
            pos=self._trigger_repoint,
            size=self._trigger_repoint
        )
        self.bg_color = Color(*self.board.arrow_bg)
        self.fg_color = Color(*self.board.arrow_fg)
        self.bg_line = Line(width=self.w * 1.4)
        self.fg_line = Line(width=self.w)
        self.canvas.add(self.bg_color)
        self.canvas.add(self.bg_line)
        self.canvas.add(self.fg_color)
        self.canvas.add(self.fg_line)
        self._trigger_repoint()

    def add_widget(self, wid, index=0, canvas=None):
        super().add_widget(wid, index, canvas)
        if not hasattr(wid, 'group'):
            return
        wid._no_use_canvas = True
        mycanvas = (
            self.canvas.before if canvas == 'before' else
            self.canvas.after if canvas == 'after' else
            self.canvas
        )
        mycanvas.remove(wid.canvas)
        pawncanvas = (
            self.board.pawnlayout.canvas.before if canvas == 'before' else
            self.board.pawnlayout.canvas.after if canvas == 'after' else
            self.board.pawnlayout.canvas
        )
        for child in self.children:
            if hasattr(child, 'group') and child.group in pawncanvas.children:
                pawncanvas.remove(child.group)
            pawncanvas.add(child.group)
        self.pospawn(wid)

    def remove_widget(self, wid):
        super().remove_widget(wid)
        for canvas in (
                self.board.pawnlayout.canvas.before,
                self.board.pawnlayout.canvas.after,
                self.board.pawnlayout.canvas
        ):
            if wid.group in canvas.children:
                canvas.remove(wid.group)
        wid._no_use_canvas = False

    def on_points(self, *args):
        """Propagate my points to both my lines"""
        self.bg_line.points = self.points
        self.fg_line.points = self.points
        for pawn in self.children:
            self.pospawn(pawn)

    def pos_along(self, pct):
        """Return coordinates for where a Pawn should be if it has travelled
        along ``pct`` of my length (between 0 and 1).

        Might get complex when I switch over to using beziers for
        arrows, but for now this is quite simple, using distance along
        a line segment.

        """
        if pct < 0 or pct > 1:
            raise ValueError("Invalid portion")
        (ox, oy) = self.origin.center
        (dx, dy) = self.destination.center
        xdist = (dx - ox) * pct
        ydist = (dy - oy) * pct
        return (ox + xdist, oy + ydist)

    def pospawn(self, pawn):
        """Position a :class:`Pawn` that is my child so as to reflect how far
        its :class:`Thing` has gone along my :class:`Portal`.

        """
        pawn.pos = self.pos_along(
            (
                self.engine.tick -
                pawn.thing['arrival_time']
            ) / (
                pawn.thing['next_arrival_time'] -
                pawn.thing['arrival_time']
            )
        )

    def _get_points(self):
        """Return the coordinates of the points that describe my shape."""
        orig = self.origin
        dest = self.destination
        (ox, oy) = orig.center
        ow = orig.width if hasattr(orig, 'width') else 0
        taillen = float(self.board.arrowhead_size)
        ory = ow / 2
        (dx, dy) = dest.center
        (dw, dh) = dest.size if hasattr(dest, 'size') else (0, 0)
        dry = dh / 2
        return get_points(ox, oy, ory, dx, dy, dry, taillen)

    def _get_slope(self):
        """Return a float of the increase in y divided by the increase in x,
        both from left to right."""
        orig = self.origin
        dest = self.destination
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

    def _get_b(self):
        """Return my Y-intercept.

        I probably don't really hit the left edge of the window, but
        this is where I would, if I were long enough.

        """
        orig = self.origin
        dest = self.destination
        (ox, oy) = orig.pos
        (dx, dy) = dest.pos
        denominator = dx - ox
        x_numerator = (dy - oy) * ox
        y_numerator = denominator * oy
        return ((y_numerator - x_numerator), denominator)

    def _repoint(self, *args):
        """Recalculate points, y-intercept, and slope"""
        if None in (self.origin, self.destination):
            Clock.schedule_once(self._repoint, 0)
            return
        self.points = self._get_points()
        self.slope = self._get_slope()
        self.y_intercept = self._get_b()
        self.repointed = True

    def collide_point(self, x, y):
        """Return True iff the point falls sufficiently close to my core line
        segment to count as a hit.

        """
        if not super(Arrow, self).collide_point(x, y):
            return False
        if None in (self.board, self.portal):
            return False
        orig = self.origin
        dest = self.destination
        (ox, oy) = orig.pos
        (dx, dy) = dest.pos
        if 0 in (ox, dx, oy, dy):
            ox += 1
            oy += 1
            dx += 1
            dy += 1
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

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self._touch = touch
            return True
        return False
