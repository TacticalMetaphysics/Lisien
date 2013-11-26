# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from kivy.properties import (
    BooleanProperty,
    BoundedNumericProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.layout import Layout
from kivy.uix.widget import Widget
from kivy.uix.relativelayout import RelativeLayout
from kivy.clock import Clock
from itemlayout import ItemLayout

SCROLL_FACTOR = 4
CAL_TYPE = {
    "THING": 0,
    "PLACE": 1,
    "PORTAL": 2,
    "STAT": 3,
    "SKILL": 4}


def get_timeline_x(calendar, branch):
    return ((branch - calendar.branch) * calendar.col_width
            + calendar.xmov + calendar.x)


def get_timeline_y(calendar, tick):
    return calendar.ymov + calendar.top + calendar.y - (
        tick - calendar.tick) * calendar.tick_height


class ColorBox(BoxLayout):
    """A BoxLayout with a background of a particular color.

In lise.kv this is filled with a label."""
    color = ListProperty()


class Cell(RelativeLayout):
    """A box to represent an event on the calendar.

It needs a branch, tick_from, tick_to, text, and a calendar to belong
to. It may be styled differently than the calendar, but will default
to the calendar's style.

    """
    bg_color_inactive = ListProperty(None)
    text_color_inactive = ListProperty(None)
    bg_color_active = ListProperty(None)
    text_color_active = ListProperty(None)
    font_name = StringProperty(None, allownone=True)
    font_size = NumericProperty(None, allownone=True)
    branch = NumericProperty()
    tick_from = NumericProperty()
    tick_to = NumericProperty(None, allownone=True)
    text = StringProperty()
    calendar = ObjectProperty()
    rowid = NumericProperty()
    active = BooleanProperty(False)

    def __init__(self, **kwargs):
        calendar = kwargs["calendar"]
        for kwarg in ["bg_color_inactive",
                      "bg_color_active",
                      "text_color_inactive",
                      "text_color_active",
                      "font_name", "font_size"]:
            if kwarg not in kwargs:
                kwargs[kwarg] = getattr(calendar, kwarg)
        super(Cell, self).__init__(
            **kwargs)


class Timeline(Widget):
    """A line drawn atop one of the columns of the calendar, representing
the present moment.

It may be dragged about to achieve time travel. It needs a
KivyConnector so it can tell when to update.

    """
    col_width = NumericProperty()

    def upd_branch(self, calendar, branch):
        self.x = get_timeline_x(calendar, branch)

    def upd_tick(self, calendar, tick):
        self.y = get_timeline_y(calendar, tick)

    def upd_time(self, calendar, branch, tick):
        self.upd_branch(calendar, branch)
        self.upd_tick(calendar, tick)


class Calendar(Layout):
    """A gridlike layout of cells representing events throughout every
branch of the timestream.

It will fill itself in based on what it finds in the Skeleton under
the given keys. Only the events that can be seen at the moment, and a
few just out of view, will be instantiated.

It may be scrolled by dragging. It will snap to some particular branch
and tick when dropped.

A timeline will be drawn on top of it, but that is not instantiated
here. Look in CalendarView below.

    """
    cal_type = NumericProperty()
    character = ObjectProperty()
    keys = ListProperty()
    font_name = StringProperty()
    font_size = NumericProperty()
    branch = NumericProperty(0)
    tick = BoundedNumericProperty(0, min=0)
    ticks_tall = NumericProperty(100)
    ticks_offscreen = NumericProperty(0)
    branches_offscreen = NumericProperty(2)
    spacing_x = NumericProperty()
    spacing_y = NumericProperty()
    branches_wide = NumericProperty()
    col_width = NumericProperty()
    tick_height = NumericProperty()
    xmov = NumericProperty(0)
    xcess = NumericProperty(0)
    ymov = NumericProperty(0)
    ycess = NumericProperty(0)
    referent = ObjectProperty(None)
    skel = ObjectProperty(None)
    force_refresh = BooleanProperty(False)
    bg_color_inactive = ListProperty()
    bg_color_active = ListProperty()
    text_color_inactive = ListProperty()
    text_color_active = ListProperty()
    completedness = NumericProperty()
    timeline = ObjectProperty()

    def on_character(self, i, v):
        """Count toward completion"""
        self.completedness += 1

    def on_keys(self, i, v):
        """Count toward completion"""
        self.completedness += 1

    def on_timeline(self, i, v):
        """Count toward completion"""
        self.completedness += 1

    def on_completedness(self, i, v):
        """When I have everything I need to fetch everything I'm missing, call
self.completed().

        """
        if v == 3:
            self.completed()

    def completed(self):
        """Collect my referent--the object I am about--and my skel--the
portion of the great Skeleton that pertains to my referent. Arrange to
be notified whenever I need to lay myself out again.

        """
        character = self.character
        closet = character.closet
        closet.branch_listeners.append(
            lambda closet, branch: self.timeline.upd_branch(self, branch))
        closet.tick_listeners.append(
            lambda closet, tick: self.timeline.upd_tick(self, tick))
        self.bind(
            size=lambda i, v: self.timeline.upd_time(
                self, closet.branch, closet.tick),
            pos=lambda i, v: self.timeline.upd_time(
                self, closet.branch, closet.tick),
            xmov=lambda i, v: self.timeline.upd_branch(self, closet.branch),
            ymov=lambda i, v: self.timeline.upd_tick(self, closet.tick))
        self.timeline.upd_time(
            self, closet.branch, closet.tick)
        skeleton = closet.skeleton
        ks = []
        for key in self.keys:
            if key is None:
                break
            ks.append(key)
        if self.cal_type == 5:
            print ks
            (dimension, thing) = ks
            self.referent = closet.get_thing(dimension, thing)
            self.skel = skeleton["thing_location"][dimension][thing]
        elif self.cal_type == 6:
            (dimension, place) = ks
            self.referent = closet.get_place(dimension, place)
            self.skel = character.placedict[dimension][place]
        elif self.cal_type == 7:
            (dimension, orig, dest) = ks
            self.referent = closet.get_portal(dimension, orig, dest)
            self.skel = character.portaldict[dimension][orig][dest]
        elif self.cal_type == 8:
            stat = ks[0]
            self.skel = character.statdict[stat]
        elif self.cal_type == 9:
            skill = ks[0]
            self.skel = character.skilldict[skill]
        self.skel.listeners.append(self.refresh_and_layout)
        self.bind(size=lambda i, v: self._trigger_layout(),
                  pos=lambda i, v: self._trigger_layout())
        Clock.schedule_once(self.refresh_and_layout, 0)

    def refresh_and_layout(self, *args):
        """Get rid of my current widgets and make new ones."""
        self.clear_widgets()
        self.force_refresh = True
        self._trigger_layout()

    def branch_x(self, b):
        """Where does the column representing that branch have its left
edge?"""
        b -= self.branch
        return self.x + self.xmov + b * self.col_width

    def tick_y(self, t):
        """Where upon me does the given tick appear?

That's where you'd draw the timeline for it."""
        if t is None:
            return self.y
        else:
            t -= self.tick
            return self.y + self.ymov + self.height - self.tick_height * t

    def refresh(self):
        """Generate cells that are missing. Remove cells that cannot be
seen."""
        minbranch = int(self.branch - self.branches_offscreen)
        maxbranch = int(
            self.branch + self.branches_wide + self.branches_offscreen)
        mintick = int(self.tick - self.ticks_offscreen)
        maxtick = int(self.tick + self.ticks_tall + self.ticks_offscreen)
        # I contain Cells.
        #
        # I should contain those that are visible, or nearly so.
        #
        # Remove those that are neither.
        for child in self.children:
            if (
                    child.branch < minbranch or
                    maxbranch < child.branch or
                    maxtick < child.tick_from or
                    (child.tick_to is not None and
                     child.tick_to < mintick)):
                self.remove_widget(child)
        # Find cells to show
        to_cover = {}
        content = {}
        for branch in xrange(minbranch, maxbranch):
            if branch not in self.skel:
                continue
            to_cover[branch] = set()
            content[branch] = {}
            boneiter = self.skel[branch].iterbones()
            prev = next(boneiter)
            for bone in boneiter:
                if (
                        prev.tick_from < maxtick and
                        rd.tick_from > mintick):
                    # I'll be showing this cell. Choose text for it
                    # based on my type.
                    if self.cal_type == 5:
                        text = prev.location
                    elif self.cal_type == 6:
                        text = prev.place
                    elif self.cal_type == 7:
                        text = "{}->{}".format(
                            prev.origin, prev.destination)
                    elif self.cal_type == 8:
                        text = prev.value
                    else:
                        text = ""
                    to_cover[branch].add(id(prev))
                    content[branch][id(prev)] = (
                        text, prev.tick_from, bone.tick_from)
                if bone.tick_from > maxtick:
                    break
                prev = rd
            # The last cell is infinitely long
            if prev.tick_from < maxtick:
                if self.cal_type == 5:
                    text = prev.location
                elif self.cal_type == 6:
                    text = prev.place
                elif self.cal_type == 7:
                    text = "{}->{}".format(
                        prev.origin, prev.destination)
                elif self.cal_type == 8:
                    text = prev.value
                else:
                    text = ""
                to_cover[branch].add(id(prev))
                content[branch][id(prev)] = (
                    text, prev.tick_from, None)
        # I might already be showing some of these, though.
        #
        # Which ones don't I show?
        uncovered = {}
        covered = {}
        for child in self.children:
            if child.branch not in covered:
                covered[child.branch] = set()
            covered[child.branch].add(child.rowid)
        for (branch, coverage) in to_cover.iteritems():
            if branch not in covered:
                uncovered[branch] = coverage
            else:
                uncovered[branch] = coverage - covered[branch]
        # Construct cells for just the bones that I'm not showing already
        for (branch, rowids) in uncovered.iteritems():
            n = 0
            for rowid in rowids:
                (text, tick_from, tick_to) = content[branch][rowid]
                cell = Cell(
                    calendar=self,
                    branch=branch,
                    text=text,
                    tick_from=tick_from,
                    tick_to=tick_to,
                    rowid=rowid)
                self.add_widget(cell)
                n += 1

    def do_layout(self, *largs):
        """Arrange all the cells into columns sorted by branch, and stack them
as appropriate to their start and end times. Adjust for scrolling as
necessary."""
        if self.parent is None:
            return
        branchwidth = self.col_width
        d_branch = int(self.xmov / branchwidth)
        tickheight = self.tick_height
        d_tick = int(self.ymov / tickheight)
        if abs(d_branch) >= 1 or abs(d_tick) >= 1:
            try:
                self.branch -= d_branch
            except ValueError:
                self.branch = 0
            self.xmov -= d_branch * (branchwidth + self.spacing_y)
            try:
                self.tick += d_tick
            except ValueError:
                self.tick = 0
            self.ymov -= d_tick * tickheight
            self.refresh()
        elif self.force_refresh:
            self.refresh()
            self.force_refresh = False
        for child in self.children:
            x = self.branch_x(child.branch)
            y = self.tick_y(child.tick_to)
            height = self.tick_y(child.tick_from) - y
            hs = self.spacing_y
            ws = self.spacing_x
            child.pos = (x + ws, y + hs)
            child.size = (branchwidth - ws, height - hs)

    def on_touch_down(self, touch):
        """Catch the touch if it hits me."""
        if self.collide_point(touch.x, touch.y):
            touch.grab(self)
            return True

    def on_touch_up(self, touch):
        """Snap to the nearest branch and tick."""
        touch.grab_current = None
        self.xmov = 0
        self.xcess = 0
        self.ymov = 0
        self.ycess = 0
        self._trigger_layout()

    def on_touch_move(self, touch):
        """If I'm being dragged, trigger a layout, but first check to see if
I've been dragged far enough that I'm no longer at the same branch and
tick. If so, adjust my branch and tick to fit."""
        if touch.grab_current is self:
            if self.xcess == 0:
                nuxmov = self.xmov + touch.dx
                if not (self.branch == 0 and nuxmov < 0):
                    self.xmov = nuxmov
                else:
                    self.xcess += touch.dx
            else:
                self.xcess += touch.dx
                if self.xcess < 0:
                    self.xcess = 0
            if self.ycess == 0:
                nuymov = self.ymov + touch.dy
                if not (self.tick == 0 and nuymov < 0):
                    self.ymov = nuymov
                else:
                    self.ycess += touch.dy
            else:
                self.ycess += touch.dy
                if self.ycess > 0:
                    self.ycess = 0
            self._trigger_layout()


class CalendarLayout(RelativeLayout, ItemLayout):
    """Really just a RelativeLayout with some Kivy properties to handle
the parameters of a Calendar."""
    pass
