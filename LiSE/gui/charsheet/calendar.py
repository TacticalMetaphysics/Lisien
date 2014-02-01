# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from kivy.clock import Clock
from kivy.properties import (
    AliasProperty,
    DictProperty,
    BoundedNumericProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty,
    ReferenceListProperty)
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.stencilview import StencilView
from kivy.uix.layout import Layout
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.logger import Logger
from kivy.graphics import Color, Line, Triangle

from LiSE.util import (
    CALENDAR_TYPES,
    THING_CAL,
    PLACE_CAL,
    PORTAL_CAL,
    CHAR_CAL)
from LiSE.gui.kivybits import EnumProperty


class Cell(Label):
    """A box to represent an event on the calendar.

    It needs a branch, tick_from, tick_to, text, and a calendar to belong
    to.

    """
    bg_color = ListProperty()
    bone = ObjectProperty()
    branch = NumericProperty()
    calendar = ObjectProperty()
    tick_from = NumericProperty()
    tick_to = NumericProperty(None, allownone=True)

    def __init__(self, **kwargs):
        """Not sure why, but setting size_hint_y: None in lise.kv had no
        effect."""
        kwargs['size_hint_y'] = None
        super(Cell, self).__init__(**kwargs)


class Timeline(Widget):
    """A wedge and a line to show where the current moment is on the
    calendar."""
    r = BoundedNumericProperty(1., min=0., max=1.)
    g = BoundedNumericProperty(0., min=0., max=1.)
    b = BoundedNumericProperty(0., min=0., max=1.)
    a = BoundedNumericProperty(1., min=0., max=1.)
    color = ReferenceListProperty(r, g, b, a)
    calendar = ObjectProperty()

    def __init__(self, **kwargs):
        super(Timeline, self).__init__(**kwargs)
        self.colorinst = Color(*self.color)
        self.canvas.add(self.colorinst)
        self.lineinst = Line(
            points=[self.x, self.y, self.x+self.calendar.col_width, self.y])
        self.canvas.add(self.lineinst)
        self.triinst = Triangle(points=[
            self.x, self.y+8, self.x+16, self.y, self.x, self.y-8])
        self.canvas.add(self.triinst)

    def on_color(self, *args):
        if not hasattr(self, 'colorinst'):
            return
        self.colorinst.rgba = self.color

    def on_pos(self, *args):
        if not hasattr(self, 'lineinst'):
            return
        self.lineinst.points = [
            self.x, self.y, self.x+self.calendar.col_width, self.y]
        if not hasattr(self, 'triinst'):
            return
        self.triinst.points = [
            self.x, self.y+8, self.x+16, self.y, self.x, self.y-8]


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
    branches_wide = NumericProperty()
    """The number of columns in the calendar that are visible at
    once. Each represents a branch of the timestream."""
    boneatt = StringProperty()
    """What attribute of its bone each cell should display for its text"""
    cal_type = EnumProperty(permitted=CALENDAR_TYPES)
    """Integer to indicate where in the skeleton to look up the bones for
    the cells"""
    col_width = BoundedNumericProperty(100, min=50)
    """How wide a column of the calendar should be"""
    key = StringProperty()
    """The *most specific* element of the partial key identifying the
    records of the calendar's referent, not including the branch and
    tick.

    """
    stat = StringProperty()
    """Name of the stat the calendar displays. Might be one of the special
    stats like location or origin."""
    font_name = StringProperty()
    """Font to be used for labels in cells"""
    font_size = NumericProperty()
    """Size of font to be used for labels in cells"""
    spacing_x = NumericProperty()
    """Space between columns"""
    spacing_y = NumericProperty()
    """Space between cells"""
    spacing = ReferenceListProperty(
        spacing_x, spacing_y)
    """[spacing_x, spacing_y]"""
    tick_height = NumericProperty()
    """How much screen a single tick should take up"""
    ticks_offscreen = NumericProperty()
    """How far off the screen a cell should be allowed before it's
    deleted--measured in ticks"""
    branches_offscreen = NumericProperty()
    """How many columns should be kept in memory, despite being offscreen"""
    offscreen = ReferenceListProperty(
        branches_offscreen, ticks_offscreen)
    """[branches_offscreen, ticks_offscreen]"""
    branch = BoundedNumericProperty(0, min=0)
    """The leftmost branch I show"""
    tick = BoundedNumericProperty(0, min=0)
    """The topmost tick I show"""
    time = ReferenceListProperty(branch, tick)
    """[branch, tick]"""
    branches_cells = DictProperty()
    """Where I keep my cells when it's inconvenient for them to be my
    children."""
    branches_cols = DictProperty()
    """Where I keep my columns when it's inconvenient for them to be my
    children."""
    referent = ObjectProperty()
    """The sim-object I am about"""
    skel = ObjectProperty()
    """That portion of the grand skeleton I am concerned with"""
    """Position of the timeline"""
    charsheet = ObjectProperty()
    """Character sheet I'm in"""
    character = AliasProperty(
        lambda self: self.charsheet.character
        if self.charsheet else None,
        lambda self, v: None,
        bind=('charsheet',))
    """Conveniently reach the character"""
    closet = AliasProperty(
        lambda self: self.charsheet.character.closet
        if self.charsheet else None,
        lambda self, v: None,
        bind=('charsheet',))
    """Conveniently reach the closet"""
    closetbranch = BoundedNumericProperty(0, min=0)
    """Current branch as a Kivy property"""
    closettick = BoundedNumericProperty(0, min=0)
    """Current tick as a Kivy property"""
    closettime = ReferenceListProperty(closetbranch, closettick)
    """Current (branch, tick) as a Kivy property"""
    ticks_tall = AliasProperty(
        lambda self: self.charsheet.character.closet.timestream.hi_tick
        if self.charsheet else None,
        lambda self, v: None,
        bind=('charsheet',))
    """How many ticks fit in me at once"""
    minbranch = NumericProperty()
    """The lowest branch I have, which may be lower than the lowest branch
    I show.

    """
    maxbranch = NumericProperty()
    """The highest branch I have, which may be higher than the highest
    branch I show.

    """
    mintick = NumericProperty()
    """The lowest tick I have, which may be lower than the lowest tick I
    show.

    """
    maxtick = NumericProperty()
    """The highest tick I have, which may be higher than the highest tick
    I show.

    """

    def __init__(self, **kwargs):
        """Make triggers for remake, retime, and timely_layout methods, then
        call self.finalize()

        """
        super(Calendar, self).__init__(**kwargs)
        self._trigger_remake = Clock.create_trigger(self.remake)
        self.bind(branch=self._trigger_layout,
                  tick=self._trigger_layout)
        self.closet.register_time_listener(self.upd_closettime)

    def upd_closettime(self, b, t):
        self.closettime = [b, t]

    def finalize(self, *args):
        """Collect my referent--the object I am about--and my skel--the
        portion of the great Skeleton that pertains to my
        referent. Arrange to be notified whenever I need to lay myself
        out again.

        """
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

        self.skel.register_set_listener(self._trigger_remake)
        self.skel.register_del_listener(self._trigger_remake)
        self._trigger_remake()

    def remake(self, *args):
        """Get rid of my current widgets and make new ones."""
        self.refresh()
        self._trigger_layout()

    def refresh(self):
        """Generate cells that are missing. Remove cells that cannot be
        seen."""
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
                if bone.tick > self.maxtick:
                    break
                elif (
                        branch in self.branches_cells and
                        prev.tick in self.branches_cells[branch] and
                        bone == self.branches_cells[branch][prev.tick]):
                    continue
                elif (
                        prev.tick < self.maxtick and
                        bone.tick > self.mintick):
                    cell = Cell(
                        calendar=self,
                        branch=branch,
                        text=getattr(prev, self.boneatt),
                        tick_from=prev.tick,
                        tick_to=bone.tick,
                        bone=bone)
                    self.branches_cells[branch][prev.tick] = cell
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
        colheight = max(self.tick_height * self.ticks_tall, 100)
        for branch in xrange(self.minbranch, self.maxbranch):
            if branch not in self.branches_cells:
                return
            if branch not in self.branches_cols:
                self.branches_cols[branch] = StackLayout()
            branch_col = self.branches_cols[branch]
            branch_col.height = colheight
            branch_col.top = self.top
            branch_col.x = self.x + (
                self.col_width + self.spacing_x) * branch
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
                    cell.tick_to - cell.tick_from
                ) * self.tick_height - self.spacing_y
                branch_col.add_widget(cell)
                branch_col.add_widget(
                    Widget(size_hint_y=None, height=self.spacing_y))
            if final is not None:
                final.width = self.col_width
                final.height = max(100, (
                    self.ticks_tall - final.tick_from) * self.tick_height)
                branch_col.add_widget(final)
            if branch_col not in self.children:
                self.add_widget(branch_col)


class CalendarView(StencilView):
    """A StencilView displaying a Calendar and a Timeline."""
    i = NumericProperty()
    """Index in the character sheet"""
    csbone = ObjectProperty()
    """Bone indicating me to the character sheet"""
    calendar = ObjectProperty()
    """I exist to hold this"""
    timeline = ObjectProperty()
    """Must put this in the view not the calendar, because it's supposed
    to stay on top, always

    """
    charsheet = ObjectProperty()
    """Character sheet I'm in"""
    _touch = ObjectProperty(None, allownone=True)
    """For when I've been grabbed and should handle scrolling"""

    @property
    def closet(self):
        return self.charsheet.character.closet

    def __init__(self, **kwargs):
        """Construct the calendar and the timeline atop it."""
        super(CalendarView, self).__init__(**kwargs)
        for questionable in (
                'pos_hint', 'size_hint', 'pos_hint_x',
                'pos_hint_y', 'pos_hint_top', 'pos_hint_right',
                'size_hint_x', 'size_hint_y', 'size',
                'x', 'y', 'width', 'height', 'top', 'right'):
            if questionable in kwargs:
                del kwargs[questionable]
        kwargs['pos'] = (0, 0)
        kwargs['height'] = self.csbone.height
        self.calendar = Calendar(**kwargs)
        self.clayout = RelativeLayout(top=self.top, x=self.x)
        self.closet.timestream.hi_time_listeners.append(
            self.upd_clayout_size)
        self.upd_clayout_size(self.closet.timestream,
                              *self.closet.timestream.hi_time)
        self.add_widget(self.clayout)
        self.clayout.add_widget(self.calendar)
        self.timeline = Timeline(
            color=kwargs['tl_color'] if 'tl_color' in kwargs
            else [1, 0, 0, 1],
            calendar=self.calendar)
        self.closet.register_time_listener(self.upd_tl_pos)
        self.upd_tl_pos(*self.closet.time)
        self.clayout.add_widget(self.timeline)
        self.calendar.finalize()
        self.bind(pos=self.upd_clayout_pos)

    def upd_tl_pos(self, branch, tick):
        x = (self.calendar.col_width + self.calendar.spacing_x) * branch
        y = self.clayout.height - self.calendar.tick_height * tick
        self.timeline.pos = (x, y)

    def upd_clayout_size(self, ts, hi_branch, hi_tick):
        w = (self.calendar.col_width + self.calendar.spacing_x) * hi_branch
        h = self.calendar.tick_height * hi_tick
        self.clayout.size = (w, max(100, h))

    def upd_clayout_pos(self, *args):
        if self._touch:
            return
        self.clayout.x = self.x
        self.clayout.top = self.top

    def on_touch_down(self, touch):
        """Detect grab. If grabbed, put 'calendar' and 'charsheet' into
        touch.ud

        """
        if self.collide_point(touch.x, touch.y):
            self._touch = touch
            touch.grab(self)
            touch.ud['calendar'] = self.calendar
            touch.ud['charsheet'] = self.charsheet
            # clayout's current position
            touch.ud['clox'] = self.clayout.x
            touch.ud['cloy'] = self.clayout.y
            return True

    def on_touch_move(self, touch):
        """If grabbed, move the calendar based on how far the touch has gone
        from where it started

        """
        if self._touch is touch:
            x = touch.ud['clox'] + touch.x - touch.ox
            y = touch.ud['cloy'] + touch.y - touch.oy
            self.clayout.pos = (x, y)
            self.calendar.tick = max(0, int((
                self.clayout.top - self.top) / self.calendar.tick_height))
            self.calendar.branch = max(0, int((
                self.x - self.clayout.x) / (
                self.calendar.col_width + self.calendar.spacing_x)))
            return True
        else:
            self._touch = None
            touch.ungrab(self)

    def on_touch_up(self, touch):
        """If the calendar's been dragged, it should adjust itself so it's at
        whatever time it should be

        """
        _touch = self._touch
        self._touch = None
        if _touch is not touch:
            return
        x = self.clayout.x
        top = self.clayout.top
        x -= x % (self.calendar.col_width + self.calendar.spacing_x)
        x = max(self.x, x)
        top -= top % self.calendar.tick_height
        top = max(self.top, top)
        self.clayout.x = x
        self.clayout.top = top
        return True
