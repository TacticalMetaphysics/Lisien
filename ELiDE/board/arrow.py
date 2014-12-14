# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
"""That which displays a one-way connection between two places.

An arrow connects two spots, the origin and the destination, and it
points from the origin to the destination, regardless of where on the
screen they are at the moment.

"""
from math import cos, sin, hypot, atan, pi
from kivy.uix.widget import Widget
from kivy.properties import (
    ReferenceListProperty,
    AliasProperty,
    ObjectProperty,
    NumericProperty,
    ListProperty,
    BooleanProperty,
    StringProperty
)
from kivy.lang import Builder
from kivy.clock import Clock

from ELiDE.kivygarden.collider import Collide2DPoly
from ELiDE.remote import MirrorMapping

ninety = pi / 2
"""pi / 2"""

fortyfive = pi / 4
"""pi / 4"""


def get_collider(ox, oy, dx, dy, r):
    if ox < dx:
        leftx = ox
        rightx = dx
        xco = 1
    elif ox > dx:
        leftx = ox * -1
        rightx = dx * -1
        xco = -1
    else:
        return Collide2DPoly(
            [
                ox - r, oy,
                ox + r, oy,
                ox + r, dy,
                ox - r, dy
            ],
            cache=False
        )
    if oy < dy:
        boty = oy
        topy = dy
        yco = 1
    elif oy > dy:
        boty = oy * -1
        topy = dy * -1
        yco = -1
    else:
        return Collide2DPoly(
            [
                ox, oy - r,
                dx, oy - r,
                dx, oy + r,
                ox, oy + r
            ],
            cache=False
        )

    rise = topy - boty
    run = rightx - leftx
    theta = atan(rise/run)
    theta_prime = ninety - theta
    xoff = sin(theta_prime) * r
    yoff = cos(theta_prime) * r
    x1 = leftx + xoff
    y1 = boty - yoff
    x2 = rightx + xoff
    y2 = topy - yoff
    x3 = rightx - xoff
    y3 = topy + yoff
    x4 = leftx - xoff
    y4 = boty + yoff
    return Collide2DPoly(
        [
            x1 * xco, y1 * yco,
            x2 * xco, y2 * yco,
            x3 * xco, y3 * yco,
            x4 * xco, y4 * yco
        ],
        cache=False
    )


def get_points(ox, oy, ro, dx, dy, rd, taillen, r):
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
        return (
            get_collider(x0, y0, endx, endy, r),
            [x0, y0, endx, endy],
            [x1, y1, endx, endy, x2, y2]
        )
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
        return (
            get_collider(x0, y0, endx, endy, r),
            [x0, y0, endx, endy],
            [x1, y1, endx, endy, x2, y2]
        )

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
    return (
        get_collider(startx, starty, endx, endy, r),
        [startx, starty, endx, endy],
        [x1, y1, endx, endy, x2, y2]
    )


class ArrowWidget(Widget):
    """A widget that points from one :class:`~LiSE.gui.board.Spot` to
    another.

    :class:`Arrow`s are the graphical representations of
    :class:`~LiSE.model.Portal`s. They point from the :class:`Spot`
    representing the :class:`Portal`'s origin, to the one representing
    its destination.

    """
    name = StringProperty()
    margin = NumericProperty(10)
    """When deciding whether a touch collides with me, how far away can
    the touch get before I should consider it a miss?"""
    w = NumericProperty(1)
    """The width of the inner, brighter portion of the :class:`Arrow`. The
    whole :class:`Arrow` will end up thicker."""
    board = ObjectProperty()
    """The board on which I am displayed."""
    pawns_here = ListProperty([])
    trunk_points = ListProperty([])
    head_points = ListProperty([])
    points = ReferenceListProperty(trunk_points, head_points)
    slope = NumericProperty(0.0, allownone=True)
    y_intercept = NumericProperty(0)
    origin = ObjectProperty()
    destination = ObjectProperty()
    engine = ObjectProperty()
    selected = BooleanProperty()
    repointed = BooleanProperty(True)
    bg_scale = NumericProperty(1.4)
    selected = BooleanProperty(False)
    bg_color_unselected = ListProperty()
    bg_color_selected = ListProperty()
    bg_color = ListProperty()
    fg_color_unselected = ListProperty()
    fg_color_selected = ListProperty()
    fg_color = ListProperty()
    arrowhead_size = NumericProperty(10)
    collide_radius = NumericProperty(3)
    collider = ObjectProperty()
    portal = ObjectProperty()
    mirrormap = ObjectProperty()

    def on_portal(self, *args):
        if not (
                self.board and
                self.origin and
                self.destination and
                self.origin.name in self.board.character.portal and
                self.destination.name in self.board.character.portal
        ):
            Clock.schedule_once(self.on_portal, 0)
            return
        self.mirrormap = MirrorMapping(
            remote=self.board.character.portal[self.origin.name][
                self.destination.name
            ]
        )
        self.name = '{}->{}'.format(self.origin.name, self.destination.name)

    def collide_point(self, x, y):
        if not self.collider:
            return False
        return (x, y) in self.collider

    def __init__(self, **kwargs):
        """Create trigger for my _repoint method, otherwise delegate to parent
        class

        """
        self._trigger_repoint = Clock.create_trigger(
            self._repoint,
            timeout=-1
        )
        super().__init__(**kwargs)

    def on_origin(self, *args):
        if self.origin is None:
            Clock.schedule_once(self.on_origin, 0)
            return
        self.origin.bind(
            pos=self._trigger_repoint,
            size=self._trigger_repoint
        )

    def on_destination(self, *args):
        if self.destination is None:
            Clock.schedule_once(self.on_destination, 0)
            return
        self.destination.bind(
            pos=self._trigger_repoint,
            size=self._trigger_repoint
        )

    def on_board(self, *args):
        if None in (
                self.board,
                self.engine,
                self.origin,
                self.destination
        ):
            Clock.schedule_once(self.on_board, 0)
            return
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
        if self.engine is None:
            Clock.schedule_once(self.on_points, 0)
            return
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
        taillen = float(self.arrowhead_size)
        ory = ow / 2
        (dx, dy) = dest.center
        (dw, dh) = dest.size if hasattr(dest, 'size') else (0, 0)
        dry = dh / 2
        return get_points(
            ox, oy, ory, dx, dy, dry, taillen, self.collide_radius
        )

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
        (
            self.collider,
            self.trunk_points,
            self.head_points
        ) = self._get_points()
        self.slope = self._get_slope()
        self.y_intercept = self._get_b()
        (ox, oy) = self.origin.center
        (dx, dy) = self.destination.center
        self.repointed = True


class Arrow(ArrowWidget):
    """ArrowWidget that represents Portal"""
    portal = ObjectProperty()
    """The portal that I represent."""
    reciprocal = AliasProperty(
        lambda self: self._get_reciprocal()
        if None not in (self.board, self.portal) else None,
        lambda self, v: None,
        bind=('portal',)
    )
    grabbed = BooleanProperty(False)

    def _get_reciprocal(self):
        orign = self.portal['origin']
        destn = self.portal['destination']
        if (
                destn in self.board.arrow and
                orign in self.board.arrow[destn]
        ):
            return self.board.arrow[destn][orign]
        else:
            return None


kv = """
#: import Dummy ELiDE.dummy.Dummy
<ArrowWidget>:
    engine: self.board.layout.app.engine if self.board and self.board.layout else None
    bg_color_unselected: [0.5, 0.5, 0.5, 0.5]
    bg_color_selected: [0.0, 1.0, 1.0, 1.0]
    fg_color_unselected: [1.0, 1.0, 1.0, 1.0]
    fg_color_selected: [1.0, 1.0, 1.0, 1.0]
    bg_color: self.bg_color_selected if self.selected else self.bg_color_unselected
    fg_color: self.fg_color_selected if self.selected else self.fg_color_unselected
    bg_scale: 2.0 if self.selected else 1.4
    canvas:
        Color:
            rgba: root.bg_color
        Line:
            width: root.w * root.bg_scale
            points: root.trunk_points
        Line:
            width: root.w * root.bg_scale
            points: root.head_points
        Color:
            rgba: root.fg_color
        Line:
            width: root.w
            points: root.trunk_points
        Line:
            width: root.w
            points: root.head_points
<Arrow>:
    origin: self.board.spot[self.portal['origin']] if self.portal['origin'] in self.board.spot else Dummy()
    destination: self.board.spot[self.portal['destination']] if self.portal['destination'] in self.board.spot else Dummy()
"""
Builder.load_string(kv)
