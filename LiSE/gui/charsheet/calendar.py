# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from kivy.clock import Clock
from kivy.properties import (
    DictProperty,
    BooleanProperty,
    BoundedNumericProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty,
    ReferenceListProperty)
from kivy.uix.stacklayout import StackLayout
from kivy.uix.stencilview import StencilView
from kivy.uix.layout import Layout
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.logger import Logger
from kivy.graphics import Callback, Color, Line, Triangle

from LiSE.data import (
    THING_CAL,
    PLACE_CAL,
    PORTAL_CAL,
    CHAR_CAL)


SCROLL_FACTOR = 4


class Cell(Label):
    """A box to represent an event on the calendar.

    It needs a branch, tick_from, tick_to, text, and a calendar to belong
    to.

    """
    bg_color = ListProperty()
    text = StringProperty()
    active = BooleanProperty(False)
    bone = ObjectProperty()
    branch = NumericProperty()
    calendar = ObjectProperty()
    tick_from = NumericProperty()
    tick_to = NumericProperty(None, allownone=True)
    label = ObjectProperty()
    color_inst = ObjectProperty()
    rect_inst = ObjectProperty()

    def __init__(self, **kwargs):
        kwargs['size_hint_y'] = None
        super(Cell, self).__init__(**kwargs)


class Timeline(Widget):
    """A line drawn atop one of the columns of the calendar, representing
    the present moment.

    """
    calendar = ObjectProperty()

    def finalize(self, *args):
        if not self.canvas and self.calendar:
            Clock.schedule_once(self.finalize, 0)
            return
        with self.canvas:
            self.cb = Callback(self.upd_time)
            Color(1, 0, 0, 1)
            Line(points=[
                self.x, self.y, self.x+self.calendar.col_width, self.y])
            Triangle(points=[
                self.x, self.y+8, self.x+16, self.y, self.x, self.y-8])

    def upd_time(self, *args):
        branch, tick = self.calendar.character.closet.time
        self.x = ((branch - self.calendar.branch) * self.calendar.col_width
                  + self.calendar.xmov + self.calendar.x)
        self.y = (
            self.calendar.ymov + self.calendar.top + self.calendar.y -
            (tick - self.calendar.tick) * self.calendar.tick_height)


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
    branches_offscreen = NumericProperty()
    branches_wide = NumericProperty()
    cal_type = NumericProperty()
    branches_cells = DictProperty({})
    branches_cols = DictProperty({})
    charsheet = ObjectProperty()
    character = ObjectProperty()
    col_width = NumericProperty()
    completedness = NumericProperty()
    font_name = StringProperty()
    font_size = NumericProperty()
    force_refresh = BooleanProperty(False)
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
    xmov = NumericProperty(0)
    ymov = NumericProperty(0)

    @property
    def minbranch(self):
        return int(max([0, self.branch - self.branches_offscreen]))

    @property
    def maxbranch(self):
        return int(self.branch + self.branches_wide
                   + self.branches_offscreen)

    @property
    def mintick(self):
        return int(max([0, self.tick - self.ticks_offscreen]))

    @property
    def maxtick(self):
        return int(self.tick + self.ticks_tall
                   + self.ticks_offscreen)

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
        if self.charsheet and not self.character:
            self.character = self.charsheet.character
        if not (self.character and self.key and self.stat):
            Clock.schedule_once(self.finalize, 0)
            return

        character = self.character
        closet = character.closet
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
            Logger.debug('Unknown cal_type, finalize later')
            Clock.schedule_once(self.finalize, 0)
            return

        self.timeline = Timeline(calendar=self)

        closet.register_time_listener(self.timeline.upd_time)
        self.bind(
            size=lambda i, v: self.timeline.upd_time(
                closet.branch, closet.tick),
            pos=lambda i, v: self.timeline.upd_time(
                closet.branch, closet.tick))
        self.timeline.upd_time(
            closet.branch, closet.tick)

        self.skel.register_set_listener(self._trigger_remake)
        self.skel.register_del_listener(self._trigger_remake)
        self.bind(size=lambda i, v: self._trigger_timely_layout(),
                  pos=lambda i, v: self._trigger_timely_layout())
        self._trigger_remake()

    def remake(self, *args):
        """Get rid of my current widgets and make new ones."""
        self.refresh()
        self.do_layout()

    def branch_x(self, b):
        """Where does the column representing that branch have its left
        edge?"""
        b -= self.branch
        return self.x + b * self.col_width - self.xmov

    def tick_y(self, t):
        """Where upon me does the given tick appear?

        That's where you'd draw the timeline for it."""
        if t is None:
            return
        else:
            # ticks from the top
            tft = t - self.tick
            if tft < 1:
                return
            # pixels from the top
            pft = tft * self.tick_height
            return self.height - pft + self.ymov

    def refresh(self):
        """Generate cells that are missing. Remove cells that cannot be
        seen."""
        old_widgets = {}
        for child in self.children:
            old_widgets[child.bone] = child
        self.clear_widgets()
        for branch in xrange(self.minbranch, self.maxbranch):
            if branch not in self.skel:
                continue
            if branch not in self.branches_cells:
                self.branches_cells[branch] = {}
            boneiter = self.skel[branch].iterbones()
            prev = next(boneiter)
            i = 0
            for bone in boneiter:
                if bone in old_widgets:
                    self.branches_cells[branch][prev.tick] = old_widgets[bone]
                elif (
                        prev.tick < self.maxtick and
                        bone.tick > self.mintick):
                    cell = Cell(
                        calendar=self,
                        branch=branch,
                        text=getattr(prev, self.boneatt),
                        tick_from=prev.tick,
                        tick_to=bone.tick,
                        bg_color=[1, 1, 1, 1],
                        color=[0, 0, 0, 1],
                        bone=bone)
                    self.branches_cells[branch][prev.tick] = cell
                if bone.tick > self.maxtick:
                    break
                prev = bone
                i += 1
            # The last cell is infinitely long
            if prev.tick < self.maxtick:
                cell = Cell(
                    calendar=self,
                    branch=branch,
                    text=getattr(prev, self.boneatt),
                    tick_from=prev.tick,
                    tick_to=None,
                    bg_color=[1, 0, 0, 1])
                self.branches_cells[branch][prev.tick] = cell

    def do_layout(self, *args):
        """Reposition all my cells so that they are in the column for their
        branch, start and end at the right place for their tick, and are
        offset by whatever amounts I'm scrolled."""
        hi_tick = self.character.closet.timestream.hi_tick
        branches_height = max([hi_tick * self.tick_height, 100])
        for branch in xrange(self.minbranch, self.maxbranch):
            if branch not in self.branches_cells:
                return
            if branch not in self.branches_cols:
                # height of the column is however much needed to hold
                # all the ticks.
                self.branches_cols[branch] = StackLayout()
            branch_col = self.branches_cols[branch]
            branch_col.height = branches_height
            branch_col.y = self.y + self.tick_height * self.tick
            branch_col.x = self.x + (self.col_width + self.spacing_x) * branch
            branch_col.width = self.col_width
            branch_col.clear_widgets()
            final = None
            for tick in sorted(self.branches_cells[branch].keys()):
                cell = self.branches_cells[branch][tick]
                cell.width = self.col_width
                if cell.tick_to is None:
                    final = cell
                    break
                cell.height = (
                    cell.tick_to - cell.tick_from - 1) * self.tick_height
                branch_col.add_widget(cell)
                branch_col.add_widget(
                    Widget(size_hint_y=None, height=self.tick_height))
            if final is not None:
                # 100% arbitrary
                final.height = 100
                final.width = self.col_width
                branch_col.add_widget(final)
            if branch_col not in self.children:
                self.add_widget(branch_col)

    def retime(self, *args):
        """Position my pointers at the timestream so they correspond to the
        apparent position of my top-left corner.

        """
        # you can only move through time by integral amounts; drop the
        # remainder
        xmov = self.xmov - self.xmov % self.col_width
        ymov = self.ymov - self.ymov % self.tick_height
        if xmov != 0:
            self.branch += xmov / self.col_width
            self.xmov %= self.col_width
        if ymov != 0:
            try:
                self.tick += ymov / self.tick_height
                self.ymov %= self.tick_height
            except ValueError:
                self.tick = 0
                self.ymov = 0

    def timely_layout(self, *args):
        self.retime(*args)
        self.do_layout(*args)


class CalendarView(StencilView):
    boneatt = StringProperty()
    cal_type = NumericProperty()
    col_width = NumericProperty()
    key = StringProperty()
    stat = StringProperty()
    i = NumericProperty()
    edbut = ObjectProperty()
    branches_wide = NumericProperty()
    font_name = StringProperty()
    font_size = NumericProperty()
    spacing_x = NumericProperty()
    spacing_y = NumericProperty()
    spacing = ReferenceListProperty(
        spacing_x, spacing_y)
    tick_height = NumericProperty()
    ticks_offscreen = NumericProperty()
    branches_offscreen = NumericProperty()
    offscreen = ReferenceListProperty(
        branches_offscreen, ticks_offscreen)
    calendar = ObjectProperty()
    charsheet = ObjectProperty()
    timeline = ObjectProperty()
    _touch = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        super(CalendarView, self).__init__(**kwargs)
        Clock.schedule_once(self.finalize, 0)

    def finalize(self, *args):
        if not self.charsheet:
            Clock.schedule_once(self.finalize, 0)
            return
        self.calendar = Calendar(
            boneatt=self.boneatt,
            cal_type=self.cal_type,
            key=self.key,
            stat=self.stat,
            branches_wide=self.branches_wide,
            col_width=self.col_width,
            font_name=self.font_name,
            font_size=self.font_size,
            spacing_x=self.spacing_x,
            spacing_y=self.spacing_y,
            tick_height=self.tick_height,
            ticks_offscreen=self.ticks_offscreen,
            branches_offscreen=self.branches_offscreen,
            charsheet=self.charsheet)
        self.calendar.bind(timeline=self.setter('timeline'))
        self.add_widget(self.calendar)
        self.bind(pos=self.calendar.setter('pos'))

    def on_touch_down(self, touch):
        if self.collide_point(touch.x, touch.y):
            self._touch = touch
            touch.grab(self)
            touch.ud['calendar'] = self.calendar
            touch.ud['charsheet'] = self.charsheet
            return True

    def on_touch_move(self, touch):
        if self._touch is touch:
            self.calendar.xmov -= touch.dx
            self.calendar.ymov += touch.dy
            self.calendar._trigger_timely_layout()
            return True
        else:
            touch.ungrab(self)

    def on_touch_up(self, touch):
        _touch = self._touch
        self._touch = None
        if _touch is not touch:
            return
        self.calendar._trigger_timely_layout()
        return True
