# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""That which displays a one-way connection between two places.

An arrow connects two spots, the origin and the destination, and it
points from the origin to the destination, regardless of where on the
screen they are at the moment.

"""
from math import hypot, atan, pi, sin, cos
from LiSE.util import (
    fortyfive,
    ninety
)
from LiSE.gui.kivybits import SaveableWidgetMetaclass
from kivy.uix.widget import Widget
from kivy.properties import (
    ObjectProperty,
    ListProperty,
    ReferenceListProperty,
    BoundedNumericProperty,
)
from kivy.clock import Clock


def slope_theta_rise_run(rise, run):
    """Return a radian value expressing the angle at the lower-left corner
    of a triangle ``rise`` high, ``run`` wide.

    If ``run`` is zero, but ``rise`` is positive, return pi / 2. If
    ``run`` is zero, but ``rise`` is negative, return -pi / 2.

    """
    try:
        return atan(rise/run)
    except ZeroDivisionError:
        if rise >= 0:
            return ninety
        else:
            return -1 * ninety


def truncated_line(leftx, boty, rightx, topy, r, from_start=False):
    """Return coordinates for two points, very much like the two points
    supplied, but with the end of the line foreshortened by amount r.

    """
    # presumes pointed up and right
    if r == 0:
        return (leftx, boty, rightx, topy)
    rise = topy - boty
    run = rightx - leftx
    length = hypot(rise, run) - r
    theta = slope_theta_rise_run(rise, run)
    if from_start:
        leftx = rightx - cos(theta) * length
        boty = topy - sin(theta) * length
    else:
        rightx = leftx + cos(theta) * length
        topy = boty + sin(theta) * length
    return (leftx, boty, rightx, topy)


def slope_theta(ox, oy, dx, dy):
    """Get a radian value representing the angle formed at the corner (ox,
    oy) of a triangle with a hypotenuse going from there to (dx,
    dy).

    """
    rise = dy - oy
    run = dx - ox
    return slope_theta_rise_run(rise, run)


def opp_theta_rise_run(rise, run):
    """Inverse of ``slope_theta_rise_run``"""
    try:
        return atan(run/rise)
    except ZeroDivisionError:
        if run >= 0:
            return ninety
        else:
            return -1 * ninety


def opp_theta(ox, oy, dx, dy):
    """Inverse of ``slope_theta``"""
    rise = dy - oy
    run = dx - ox
    return opp_theta_rise_run(rise, run)


def wedge_offsets_core(theta, opp_theta, taillen):
    """Internal use"""
    top_theta = theta - fortyfive
    bot_theta = pi - fortyfive - opp_theta
    xoff1 = cos(top_theta) * taillen
    yoff1 = sin(top_theta) * taillen
    xoff2 = cos(bot_theta) * taillen
    yoff2 = sin(bot_theta) * taillen
    return (
        xoff1, yoff1, xoff2, yoff2)


def wedge_offsets_rise_run(rise, run, taillen):
    """Given a line segment's rise, run, and length, return two new
    points--with respect to the *end* of the line segment--that are good
    for making an arrowhead with.

    The arrowhead is a triangle formed from these points and the point at
    the end of the line segment.

    """
    # theta is the slope of a line bisecting the ninety degree wedge.
    theta = slope_theta_rise_run(rise, run)
    opp_theta = opp_theta_rise_run(rise, run)
    return wedge_offsets_core(theta, opp_theta, taillen)


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
    __metaclass__ = SaveableWidgetMetaclass
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
    bg_r = BoundedNumericProperty(0.25, min=0., max=1.)
    bg_g = BoundedNumericProperty(0.25, min=0., max=1.)
    bg_b = BoundedNumericProperty(0.25, min=0., max=1.)
    bg_a = BoundedNumericProperty(0.8, min=0., max=1.)
    bg_color = ReferenceListProperty(bg_r, bg_g, bg_b, bg_a)
    fg_r = BoundedNumericProperty(1, min=0., max=1.)
    fg_g = BoundedNumericProperty(1, min=0., max=1.)
    fg_b = BoundedNumericProperty(1, min=0., max=1.)
    fg_a = BoundedNumericProperty(1, min=0., max=1.)
    fg_color = ReferenceListProperty(fg_r, fg_g, fg_b, fg_a)
    points = ListProperty()

    def __init__(self, **kwargs):
        """Bind some properties, and put the relevant instructions into the
        canvas--but don't put any point data into the instructions
        just yet. For that, wait until ``on_parent``, when we are
        guaranteed to know the positions of our endpoints.

        """
        self.trigger_repoint = Clock.create_trigger(
            self.repoint, timeout=-1)
        self.trigger_repawn = Clock.create_trigger(
            self.repawn, timeout=-1)
        Widget.__init__(self, **kwargs)
        self.board.arrowdict[unicode(self.portal)] = self
        orign = unicode(self.portal.origin)
        destn = unicode(self.portal.destination)
        self.board.spotdict[orign].bind(
            pos=self.trigger_repoint,
            size=self.trigger_repoint)
        self.board.spotdict[destn].bind(
            pos=self.trigger_repoint,
            size=self.trigger_repoint)
        self.board.host.closet.register_time_listener(
            self.trigger_repawn)
        self.trigger_repoint()
        self.trigger_repawn()

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

    def repoint(self, *args):
        self.points = self.get_points()

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

    def repawn(self, *args):
        (branch, tick) = self.board.host.closet.time
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
