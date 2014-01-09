# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from kivy.clock import Clock
from kivy.properties import (
    BooleanProperty,
    BoundedNumericProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.layout import Layout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget
from kivy.uix.label import Label

from LiSE.data import (
    THING_CAL,
    PLACE_CAL,
    PORTAL_CAL,
    CHAR_CAL)


SCROLL_FACTOR = 4


def get_timeline_x(calendar, branch):
    return ((branch - calendar.branch) * calendar.col_width
            + calendar.xmov + calendar.x)


def get_timeline_y(calendar, tick):
    return calendar.ymov + calendar.top + calendar.y - (
        tick - calendar.tick) * calendar.tick_height


class ColorBox(BoxLayout):
    color = ListProperty()


class Cell(Widget):
    """A box to represent an event on the calendar.

    It needs a branch, tick_from, tick_to, text, and a calendar to belong
    to.

    """
    text = StringProperty()
    active = BooleanProperty(False)
    bone = ObjectProperty()
    branch = NumericProperty()
    calendar = ObjectProperty()
    tick_from = NumericProperty()
    tick_to = NumericProperty(None, allownone=True)

    def on_text(self, *args):
        print("text {}".format(self.text))

    def on_pos(self, *args):
        print("pos {}".format(self.pos))

    def on_size(self, *args):
        print("size {}".format(self.size))


class Timeline(Widget):
    """A line drawn atop one of the columns of the calendar, representing
    the present moment.

    """

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
    boneatt = StringProperty()
    branch = NumericProperty(0)
    branches_offscreen = NumericProperty(2)
    branches_wide = NumericProperty()
    cal_type = NumericProperty()
    character = ObjectProperty()
    col_width = NumericProperty()
    completedness = NumericProperty()
    font_name = StringProperty()
    font_size = NumericProperty()
    force_refresh = BooleanProperty(False)
    i = NumericProperty()
    key = StringProperty()
    referent = ObjectProperty(None)
    skel = ObjectProperty(None)
    spacing_x = NumericProperty()
    spacing_y = NumericProperty()
    stat = StringProperty()
    tick = BoundedNumericProperty(0, min=0)
    tick_height = NumericProperty()
    ticks_tall = NumericProperty(100)
    ticks_offscreen = NumericProperty(0)
    timeline = ObjectProperty()
    xcess = NumericProperty(0)
    xmov = NumericProperty(0)
    ycess = NumericProperty(0)
    ymov = NumericProperty(0)
    _touch = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super(Calendar, self).__init__(**kwargs)
        self._trigger_remake = Clock.create_trigger(self.remake)
        self._trigger_retime = Clock.create_trigger(self.retime)
        self._trigger_timely_layout = Clock.create_trigger(
            self.timely_layout)
        Clock.schedule_once(self.finalize, 0)

    def finalize(self, *args):
        """Collect my referent--the object I am about--and my skel--the
        portion of the great Skeleton that pertains to my
        referent. Arrange to be notified whenever I need to lay myself
        out again.

        """
        if not (self.character and self.timeline and self.key and self.stat):
            Clock.schedule_once(self.finalize, 0)
            return

        def upd_time(branch, tick):
            self.timeline.upd_branch(self, branch)
            self.timeline.upd_tick(self, tick)

        character = self.character
        closet = character.closet
        closet.register_time_listener(upd_time)
        self.bind(
            size=lambda i, v: self.timeline.upd_time(
                self, closet.branch, closet.tick),
            pos=lambda i, v: self.timeline.upd_time(
                self, closet.branch, closet.tick))
        self.timeline.upd_time(
            self, closet.branch, closet.tick)
        skeleton = closet.skeleton
        if self.cal_type == THING_CAL:
            self.referent = self.character.get_thing(self.key)
            if self.stat == "location":
                self.skel = skeleton["thing_loc"][
                    unicode(self.character)][self.key]
                self.boneatt = "location"
            else:
                self.skel = skeleton["thing_stat"][
                    unicode(self.character)][self.key][self.stat]
                self.boneatt = "value"
        elif self.cal_type == PLACE_CAL:
            self.referent = self.character.get_place(self.key)
            self.skel = skeleton["place_stat"][
                unicode(self.character)][self.key][self.stat]
            self.boneatt = "value"
        elif self.cal_type == PORTAL_CAL:
            if self.stat in ("origin", "destination"):
                self.skel = skeleton["portal_loc"][
                    unicode(self.character)][self.key]
                self.boneatt = self.stat
            else:
                self.skel = skeleton["portal_stat"][
                    unicode(self.character)][self.key][self.stat]
                self.boneatt = "value"
        elif self.cal_type == CHAR_CAL:
            self.skel = skeleton["character_stat"][
                unicode(self.character)][self.key]
            self.boneatt = "value"
        else:
            raise ValueError("Unknown cal_type")
        self.skel.register_set_listener(self.remake)
        self.skel.register_del_listener(self.remake)
        self.bind(size=lambda i, v: self._trigger_layout(),
                  pos=lambda i, v: self._trigger_layout())
        self._trigger_remake()

    def remake(self, *args):
        """Get rid of my current widgets and make new ones."""
        self.clear_widgets()
        self.refresh()

    def branch_x(self, b):
        """Where does the column representing that branch have its left
        edge?"""
        b -= self.branch
        return self.xmov + self.x + b * self.col_width

    def tick_y(self, t):
        """Where upon me does the given tick appear?

        That's where you'd draw the timeline for it."""
        if t is None:
            return 0
        else:
            return self.ymov + self.top - (
                self.tick_height * (self.tick-t))

    def refresh(self):
        """Generate cells that are missing. Remove cells that cannot be
        seen."""
        minbranch = int(self.branch - self.branches_offscreen)
        maxbranch = int(
            self.branch + self.branches_wide + self.branches_offscreen)
        mintick = int(self.tick - self.ticks_offscreen)
        maxtick = int(self.tick + self.ticks_tall + self.ticks_offscreen)
        old_widgets = {}
        for child in self.children:
            old_widgets[child.bone] = child
        self.clear_widgets()
        for branch in xrange(minbranch, maxbranch):
            if branch not in self.skel:
                continue
            boneiter = self.skel[branch].iterbones()
            prev = next(boneiter)
            for bone in boneiter:
                if bone in old_widgets:
                    self.add_widget(old_widgets[bone])
                    print("refreshing w. old bone: {}".format(bone))
                elif (
                        prev.tick < maxtick and
                        bone.tick > mintick):
                    print("refreshing w. new bone: {}".format(bone))
                    cell = Cell(
                        calendar=self,
                        branch=branch,
                        text=getattr(prev, self.boneatt),
                        tick_from=prev.tick,
                        tick_to=bone.tick,
                        bone=bone)
                    self.add_widget(cell)
                if bone.tick > maxtick:
                    break
                prev = bone
            # The last cell is infinitely long
            if prev.tick < maxtick:
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
                assert(text is not None)
                self.add_widget(Cell(
                    calendar=self,
                    branch=branch,
                    text=text,
                    tick_from=prev.tick,
                    tick_to=None))

    def do_layout(self, *args):
        """Reposition all my cells so that they are in the column for their
        branch, start and end at the right place for their tick, and are
        offset by whatever amounts I'm scrolled."""
        for cell in self.children:
            if cell.tick_to is None:
                h = (self.tick + self.ticks_tall -
                     cell.tick_from) * self.tick_height
                if h <= 0:
                    self.remove_widget(cell)
                else:
                    cell.pos = (self.branch_x(cell.branch), 0)
                    cell.size = (h, self.branch_width)
            else:
                y = self.tick_y(cell.tick_to)
                cell.size = (self.branch_width,
                             self.tick_y(cell.tick_from) - y)
                cell.pos = (self.branch_x(cell.branch), y)

    def retime(self, *args):
        """Position my pointers at the timestream so they correspond to the
        apparent position of my top-left corner.

        """
        # you can only move through time by integral amounts; drop the
        # remainder
        xmov = self.xmov - self.xmov % self.branch_width
        ymov = self.ymov - self.ymov % self.tick_height
        if xmov > 0:
            self.branch += xmov / self.branch_width
            self.xmov %= self.branch_width
        if ymov > 0:
            self.tick += ymov / self.tick_height
            self.ymov %= self.tick_height

    def timely_layout(self, *args):
        self.retime(*args)
        self.do_layout(*args)

    def on_touch_down(self, touch):
        if self.collide_point(touch.x, touch.y):
            self._touch = touch
            touch.grab(self)
            touch.ud['calendar'] = self

    def on_touch_move(self, touch):
        if self._touch is touch:
            self.xmov -= touch.dx
            self.ymov -= touch.dy
            self._trigger_timely_layout()
        else:
            touch.ungrab(self)

    def on_touch_up(self, touch):
        _touch = self._touch
        self._touch = None
        if _touch is not touch:
            return
        self._trigger_timely_layout()


class CalendarLayout(RelativeLayout):
    """Really just a RelativeLayout with some Kivy properties to handle
the parameters of a Calendar."""
    character = ObjectProperty()
    item_type = NumericProperty()
    key = StringProperty()
    stat = StringProperty()
    edbut = ObjectProperty()
