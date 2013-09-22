
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass, TimestreamException
from logging import getLogger
from pyglet.text import Label
from pyglet.graphics import GL_LINES, GL_TRIANGLES, Group, OrderedGroup
from pyglet.gl import glScissor, glEnable, glDisable, GL_SCISSOR_TEST

"""User's view on a given item's schedule.

Usually there should be only one calendar per board, but it can switch
between showing various schedules, or even show many in parallel.

"""


__metaclass__ = SaveableMetaclass


logger = getLogger(__name__)


class BranchConnector:
    """Widget to show where a branch branches off another.

    It's an arrow that leads from the tick on one CalendarCol where
    another branches from it, to the start of the child.

    """
    atrdic = {
        "window_bot": lambda self:
        self.calendar.tick_to_y(self.column.start_tick) +
        self.calendar.offy,
        "window_top": lambda self: self.window_bot + self.space,
        "window_left": lambda self:
        self.column.parent.window_right -
        self.column.parent.style.spacing,
        "window_right": lambda self:
        self.column.window_left +
        self.column.style.spacing +
        self.column.rx,
        "window_center": lambda self:
        (self.column.window_left + self.column.parent.window_right) / 2,
        "in_view": lambda self:
        self.column.in_view or self.column.parent.in_view,
        "wedge_visible": lambda self: (
            self.window_top > 0 and
            self.window_bot < self.window.height and
            self.column.in_view),
        "branch": lambda self: int(self.column)}

    def __init__(self, column, color=(255, 0, 0, 255),
                 wedge_height=8, wedge_width=16):
        self.column = column
        self.calendar = self.column.calendar
        self.batch = self.calendar.batch
        self.window = self.calendar.window
        self.color = color
        self.wedge_height = wedge_height
        self.wedge_rx = wedge_width / 2
        self.space = self.calendar.style.spacing * 2
        self.line_vertlist = None
        self.wedge_vertlist = None

    def __getattr__(self, attrn):
        """Look up computed attributes in the atrdic of the class."""
        try:
            return BranchConnector.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                "BranchConnector has no attribute named {0}".format(attrn))

    def get_wedge(self, batch, group):
        b = self.window_bot
        assert(b is not None)
        c = self.window_right
        t = b + self.wedge_height
        l = c - self.wedge_rx
        r = c + self.wedge_rx
        points = (
            c, b,
            r, t,
            l, t,
            c, b)
        return batch.add(
            4,
            GL_TRIANGLES,
            group,
            ('v2i', points),
            ('c4B', self.color * 4))

    def get_line(self, batch, group):
        y0 = self.window_bot
        if y0 > self.calendar.window_top:
            return
        y2 = y0
        y1 = self.window_top
        if y1 < self.calendar.window_bot:
            return
        x0 = self.window_left
        x1 = self.window_center
        x2 = self.window_right
        verts = (x0, y0, x1, y0,
                 x1, y0, x1, y1,
                 x1, y1, x2, y1,
                 x2, y1, x2, y2)
        return batch.add(
            len(verts) / 2,
            GL_LINES,
            group,
            ('v2i', verts),
            ('c4B', self.color * (len(verts) / 2)))

    def delete(self):
        """Immediately remove from video memory"""
        try:
            self.line_vertlist.delete()
        except (AttributeError, AssertionError):
            pass
        self.line_vertlist = None
        try:
            self.wedge_vertlist.delete()
        except (AttributeError, AssertionError):
            pass
        self.wedge_vertlist = None


class Timeline:
    """A line that goes on top of a CalendarCol to indicate what time it
is.

Also has a little handle that you can drag to do the time warp. Select
which side it should be on by supplying "left" (default) or "right"
for the handle_side keyword argument.

    """
    crosshair = True

    atrdic = {
        "window_y": lambda self:
        self.cal.tick_to_y(self.closet.tick) + self.cal.offy,
        "y": lambda self: self.window_y,
        "window_bot": lambda self: self.window_y,
        "window_top": lambda self: self.window_y,
        "window_left": lambda self:
        self.col.window_left + self.col.style.spacing,
        "window_right": lambda self:
        self.col.window_right - self.col.style.spacing,
        "in_view": lambda self:
        (self.col.branch == self.col.closet.branch and
         self.y > 0 and self.y < self.window.height
         and self.window_right > 0
         and self.window_left < self.window.width),
        "col": lambda self: self.column,
        "cal": lambda self: self.calendar}

    def __init__(self, col, handle_side="left",
                 handle_width=None, handle_height=None,
                 color=(255, 0, 0, 255)):
        """Make a timeline for the given column, optionally specifying
        what side its handle is on, and what color it is.

color is a 4-tuple of Red, Green, Blue, Alpha."""
        self.column = col
        self.calendar = self.col.calendar
        self.batch = self.col.batch
        self.window = self.cal.window
        self.closet = self.col.closet
        self.handle_side = handle_side
        if handle_width is None:
            self.handle_width = self.col.style.spacing * 2
        else:
            self.handle_width = handle_width
        if handle_height is None:
            self.handle_height = self.col.style.spacing * 2
        else:
            self.handle_height = handle_height
        self.ry = self.handle_height / 2
        self.color = color
        self.offx = 0
        self.offy = 0

    def __getattr__(self, attrn):
        """Try to compute the attribute using the lambdas in my
        atrdic."""
        try:
            return self.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                "Timeline instance does not have and cannot "
                "compute attribute {0}".format(attrn))

    def overlaps(self, x, y):
        y = self.y
        t = y + self.ry
        b = y - self.ry
        if self.handle_side == "left":
            r = self.window_left
            l = r - self.handle_width
        else:
            l = self.window_right
            r = l + self.handle_width
        return (
            x > l and
            x < r and
            y > b and
            y < t)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.offx += dx
        if y > self.column.window_top:
            if self.closet.tick != self.calendar.scrolled_to:
                self.closet.time_travel(
                    self.column.branch,
                    max((self.calendar.scrolled_to, 0)))
        elif y < self.column.window_bot:
            if self.closet.tick != self.calendar.bot_tick:
                self.closet.time_travel(
                    self.column.branch, self.calendar.bot_tick)
        else:
            self.offy += dy
        while self.offy > self.calendar.row_height:
            try:
                self.closet.time_travel_inc_tick(-1)
                self.offy -= self.calendar.row_height
            except TimestreamException:
                self.offy = 0
                break
        while self.offy * -1 > self.calendar.row_height:
            try:
                self.closet.time_travel_inc_tick(1)
                self.offy += self.calendar.row_height
            except TimestreamException:
                self.offy = 0
                break
        while self.offx > self.column.width:
            if self.closet.branch == self.closet.timestream.hi_branch:
                self.offx = 0
                break
            self.closet.time_travel_inc_branch(1)
            self.offx -= self.column.width
            self.column = self.calendar.make_col(self.closet.branch)
        while self.offx * -1 > self.column.width:
            if self.closet.branch == 0:
                self.offx = 0
                break
            self.closet.time_travel_inc_branch(-1)
            self.offx += self.column.width
            self.column = self.calendar.make_col(self.closet.branch)

    def get_line(self, batch, group):
        points = (
            self.window_left, self.y,
            self.window_right, self.y)
        return batch.add(
            2,
            GL_LINES,
            group,
            ('v2i', points),
            ('c4B', self.color * 2))

    def get_handle(self, batch, group):
        y = self.y
        b = y - self.ry
        t = y + self.ry
        if self.handle_side == "left":
            r = self.window_left
            l = r - self.handle_width
            points = (r, y,
                      l, t,
                      l, b,
                      r, y)
        else:
            l = self.window_right
            r = l + self.handle_width
            points = (l, y,
                      r, t,
                      r, b,
                      l, y)
        return batch.add(
            4,
            GL_TRIANGLES,
            group,
            ('v2i', points),
            ('c4B', self.color * 4))


class CalendarCell:
    """A block of time in a calendar.

Uses information from the CalendarCol it's in and the Event it
represents to calculate its dimensions and coordinates.

    """
    visible = True

    atrdic = {
        "interactive": lambda self: self.column.calendar.interactive,
        "window": lambda self: self.column.calendar.window,
        "width": lambda self: self.window_right - self.window_left,
        "height": lambda self: {
            True: lambda:
            self.calendar.height - (
                self.tick_from * self.calendar.row_height),
            False: lambda: self.calendar.row_height * len(self)
            }[self.tick_to is None](),
        "window_left": lambda self:
        self.column.window_left + self.column.style.spacing,
        "window_right": lambda self:
        self.column.window_right - self.column.style.spacing,
        "window_top": lambda self:
        self.calendar.height + self.calendar.offy -
        self.column.calendar.row_height * (
            self.tick_from - self.column.calendar.scrolled_to),
        "window_bot": lambda self:
        self.window_top - self.height,
        "in_view": lambda self:
        (self.window_right > 0 or
         self.window_left < self.window.width or
         self.window_top > 0 or
         self.window_bot < self.window.height),
        "same_size": lambda self: self.is_same_size(),
        "label_height": lambda self:
        self.style.fontsize + self.style.spacing,
        "hovered": lambda self: self is self.window.hovered,
        "coverage_dict": lambda self: {
            CAL_TYPE['THING']: lambda self: self.closet.skeleton[
                "character_things"][self._rowdict["character"]][
                self._rowdict["dimension"]][self._rowdict["thing"]],
            CAL_TYPE['PLACE']: lambda self: self.closet.skeleton[
                "character_places"][self._rowdict["character"]][
                self._rowdict["dimension"]][self._rowdict["place"]],
            CAL_TYPE['PORTAL']: lambda self: self.closet.skeleton[
                "character_portals"][self._rowdict["character"]][
                self._rowdict["dimension"]][self._rowdict["origin"]][
                self._rowdict["destination"]],
            CAL_TYPE['SKILL']: lambda self: self.closet.skeleton[
                "character_skills"][self._rowdict["character"]][
                self._rowdict["skill"]],
            CAL_TYPE['STAT']: lambda self: self.closet.skeleton[
                "character_stats"][self._rowdict["character"]][
                self._rowdict["stat"]]}[self.typ]()}

    def __init__(self, col, tick_from, tick_to, text):
        """Get a CalendarCol in the given column, between the start
        and and ticks given, and with the given text."""
        self.column = col
        self.calendar = self.column.calendar
        self.batch = self.column.batch
        self.style = self.column.style
        self.window = self.calendar.window
        self.tick_from = tick_from
        self.tick_to = tick_to
        self.text = text

    def __len__(self):
        """The number of ticks I show. May be lower than the number of
        ticks I *can* show, if I'm part-way off the screen."""
        if self.tick_to is None:
            r = self.calendar.bot_tick - self.tick_from
        else:
            r = self.tick_to - self.tick_from
        if r < 0:
            return 0
        else:
            return r

    def __getattr__(self, attrn):
        """Try using the lambdas in my atrdic to compute the value"""
        return CalendarCell.atrdic[attrn](self)

    def __eq__(self, other):
        return self.tick_from == other.tick_from

    def __hash__(self):
        return hash(self.tick_from)

    def __str__(self):
        """(text) from (tick) to (tick)"""
        return "{0} from {1} to {2}".format(
            self.text, self.tick_from, self.tick_to)

    def get_calendar_bot(self):
        """How far above the bottom of the calendar is my bottom edge?"""
        try:
            return self.calendar.height - self.calendar.row_height * (
                self.tick_to - self.calendar.scrolled_to)
        except TypeError:
            return 0

    def is_same_size(self):
        """Check if I have the same size as the last time I ran this
        method."""
        r = (
            self.old_width == self.width and
            self.old_height == self.height)
        self.old_width = self.width
        self.old_height = self.height
        return r

    def delete(self):
        """Get out of video memory right away."""
        try:
            self.label.delete()
        except AttributeError:
            pass
        self.label = None
        try:
            self.vertl.delete()
        except AttributeError:
            pass
        self.vertl = None

    def get_label(self, batch, group):
        l = self.window_left
        t = self.window_top
        w = self.width
        h = self.height
        if self.style.fontsize > h:
            return
        else:
            l = Label(
                self.text,
                self.style.fontface,
                self.style.fontsize,
                color=self.style.textcolor.tup,
                width=w,
                height=h,
                x=l,
                y=t,
                anchor_x="left",
                anchor_y="top",
                halign="center",
                multiline=True,
                batch=batch,
                group=group)
            while l.content_width > w:
                l.text = l.text[:-1]
            return l

    def get_box(self, batch, group):
        l = self.window_left
        b = self.window_bot
        r = self.window_right
        t = self.window_top
        colors = self.style.fg_inactive.tup * 6
        vees = (l, t, r, t, r, b,
                r, b, l, b, l, t)
        return batch.add(
            6,
            GL_TRIANGLES,
            group,
            ('v2i', vees),
            ('c4B', colors))

CAL_TYPE = {
    "THING": 0,
    "PLACE": 1,
    "PORTAL": 2,
    "STAT": 3,
    "SKILL": 4}


class CalendarGroup(Group):
    def __init__(self, cal):
        super(CalendarGroup, self).__init__(cal.window.calendar_group)
        self.calendar = cal

    def set_state(self):
        glEnable(GL_SCISSOR_TEST)
        glScissor(self.calendar.window_left,
                  self.calendar.window_bot,
                  self.calendar.width,
                  self.calendar.height)

    def unset_state(self):
        glDisable(GL_SCISSOR_TEST)


class TimelineWedgeGroup(Group):
    def __init__(self, cal, parent):
        super(TimelineWedgeGroup, self).__init__(parent)
        self.calendar = cal

    def set_state(self):
        glScissor(self.calendar.window_left - self.calendar.style.spacing,
                  self.calendar.window_bot,
                  self.calendar.width,
                  self.calendar.height)

    def unset_state(self):
        pass


class Calendar:
    """A collection of columns representing values over time for
a given attribute of a character.

Calendars come in several types, each corresponding to one of the
dictionaries in a Character:

THING Calendars usually display where a Thing is located, but may
display the Thing's display name over time.

PLACE and PORTAL Calendars show the display name of a Place, and the
fact that two Places are connected, respectively.

STAT Calendars show the changing value of a particular stat of the
Character over time.

SKILL Calendars show the display name of the EffectDeck used for one
of the Character's skills.

If the Calendar shows a display name, and the display name changes,
the Calendar will show its old value before the tick of the change,
and the new value afterward. You might want to set your display names
programmatically, to make them show some data of interest.

Each column in a Calendar displays one branch of time. Each branch may
appear no more than once in a given Calendar. Every branch of the
Timestream that has yet been created should have a column associated
with it, but that column might not be shown. It will just be kept in
reserve, in case the user tells the Calendar to display that branch.

The columns are arranged such that each branch is as close to its
parent as possible. There is a squiggly arrow pointing to the start of
each branch save for the zeroth, originating at the branch and tick
that the branch split off of.

A line, called the Timeline, will be drawn on the column of the active
branch, indicating the current tick. The timeline has an arrow-shaped
handle, which may be dragged within and between branches to effect
time travel.

    """
    visible = True
    tables = [
        (
            "calendar",
            {"window": "TEXT NOT NULL DEFAULT 'Main'",
             "idx": "INTEGER NOT NULL DEFAULT 0",
             "left": "FLOAT NOT NULL DEFAULT 0.8",
             "right": "FLOAT NOT NULL DEFAULT 1.0",
             "top": "FLOAT NOT NULL DEFAULT 1.0",
             "bot": "FLOAT NOT NULL DEFAULT 0.0",
             "max_cols": "INTEGER NOT NULL DEFAULT 3",
             "style": "TEXT NOT NULL DEFAULT 'default_style'",
             "interactive": "BOOLEAN NOT NULL DEFAULT 1",
             "rows_shown": "INTEGER NOT NULL DEFAULT 240",
             "scrolled_to": "INTEGER DEFAULT 0",
             "scroll_factor": "INTEGER NOT NULL DEFAULT 4",
             "left_branch": "INTEGER NOT NULL DEFAULT 0",
             "type": "INTEGER NOT NULL DEFAULT {0}".format(CAL_TYPE['THING']),
             "character": "TEXT NOT NULL",
             "dimension": "TEXT DEFAULT NULL",
             "thing": "TEXT DEFAULT NULL",
             "thing_show_location": "BOOLEAN DEFAULT 1",
             "place": "TEXT DEFAULT NULL",
             "origin": "TEXT DEFAULT NULL",
             "destination": "TEXT DEFAULT NULL",
             "skill": "TEXT DEFAULT NULL",
             "stat": "TEXT DEFAULT NULL"},
            ("window", "idx"),
            {"window": ("window", "name"),
             "style": ("style", "name"),
             "character, dimension, thing":
             ("character_things", "character, dimension, thing"),
             "character, dimension, place":
             ("character_places", "character, dimension, place"),
             "character, dimension, origin, destination":
             ("character_portals",
              "character, dimension, origin, destination"),
             "character, skill":
             ("character_skills", "character, skill"),
             "character, stat":
             ("character_stats", "character, stat")},
            ["rows_shown>0", "left>=0.0", "left<=1.0", "right<=1.0",
             "left<right", "top>=0.0", "top<=1.0", "bot>=0.0",
             "bot<=1.0", "top>bot", "idx>=0",
             "CASE type "
             "WHEN {0} THEN (dimension NOTNULL AND thing NOTNULL) "
             "WHEN {1} THEN (dimension NOTNULL AND place NOTNULL) "
             "WHEN {2} THEN "
             "(dimension NOTNULL AND "
             "origin NOTNULL AND "
             "destination NOTNULL) "
             "WHEN {3} THEN skill NOTNULL "
             "WHEN {4} THEN stat NOTNULL "
             "ELSE 0 "
             "END".format(
                 CAL_TYPE['THING'],
                 CAL_TYPE['PLACE'],
                 CAL_TYPE['PORTAL'],
                 CAL_TYPE['SKILL'],
                 CAL_TYPE['STAT'])]
        )]

    atrdic = {
        "typ": lambda self: self._rowdict["type"],
        "character": lambda self:
        self.closet.get_character(self._rowdict["character"]),
        "dimension": lambda self:
        self.closet.get_dimension(self._rowdict["dimension"]),
        "thing": lambda self: self.closet.get_thing(
            self._rowdict["dimension"], self._rowdict["thing"]),
        "place": lambda self: self.closet.get_place(
            self._rowdict["dimension"], self._rowdict["place"]),
        "portal": lambda self: self.closet.get_portal(
            self._rowdict["dimension"],
            self._rowdict["origin"],
            self._rowdict["destination"]),
        "interactive": lambda self: self._rowdict["interactive"],
        "rows_shown": lambda self: self._rowdict["rows_shown"],
        "left_branch": lambda self: self._rowdict["left_branch"],
        "left_prop": lambda self: self._rowdict["left"],
        "right_prop": lambda self: self._rowdict["right"],
        "top_prop": lambda self: self._rowdict["top"],
        "bot_prop": lambda self: self._rowdict["bot"],
        "bot_tick": lambda self: self.scrolled_to + self.rows_shown,
        "style": lambda self: self.closet.get_style(self._rowdict["style"]),
        "window_top": lambda self: int(self.top_prop * self.window.height),
        "window_bot": lambda self: int(self.bot_prop * self.window.height),
        "window_left": lambda self: int(self.left_prop * self.window.width),
        "window_right": lambda self: int(self.right_prop * self.window.width),
        "width": lambda self: self.window_right - self.window_left,
        "col_width": lambda self: self.get_col_width(),
        "height": lambda self: self.window_top - self.window_bot,
        "row_height": lambda self: self.height / self.rows_shown,
        "scrolled_to": lambda self: self.sttt(),
        "scroll_factor": lambda self: self._rowdict["scroll_factor"],
        "max_cols": lambda self: self._rowdict["max_cols"],
        "thing_show_location": lambda self: (
            self._rowdict["thing_show_location"]
            not in (0, None, False)),
        "columns": lambda self: CalendarColIter(self),
        "_rowdict": lambda self:
        self.closet.skeleton["calendar"][
            str(self.window)][
            int(self)],
        "dimension": lambda self:
        self.closet.get_dimension(self._rowdict["dimension"]),
        "thing": lambda self:
        self.closet.get_thing(
            self._rowdict["dimension"],
            self._rowdict["thing"]),
        "_location_dict": lambda self:
        self.closet.skeleton["thing_location"][
            self._rowdict["dimension"]][
            self._rowdict["thing"]],
    }

    def __init__(self, window, idx):
        """Get the idx-th calendar in the given window"""
        super(Calendar, self).__setattr__("window", window)
        super(Calendar, self).__setattr__("closet", self.window.closet)
        super(Calendar, self).__setattr__("idx", idx)
        self.batch = self.window.batch
        self.biggroup = CalendarGroup(self)
        self.group = OrderedGroup(0, self.biggroup)
        self.tlgroup = OrderedGroup(1, self.biggroup)
        self.wedgegroup = TimelineWedgeGroup(self, self.tlgroup)
        self.offx = 0
        self.offy = 0
        self.last_draw = None

    def __int__(self):
        """What-th calendar in my window am I?

So, return my index."""
        return self.idx

    def __iter__(self):
        return self.columns

    def __getattr__(self, attrn):
        """Compute the attribute using the correct lambda from my atrdic"""
        try:
            return self.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                "Calendar instance has no attribute {0}".format(attrn))

    def __setattr__(self, attrn, val):
        """Handle setting my field values in my rowdict"""
        if attrn in self._rowdict:
            assert(type(val) is type(self._rowdict[attrn]))
            self._rowdict[attrn] = val
        else:
            super(Calendar, self).__setattr__(attrn, val)

    def get_col_width(self):
        branches = len(self.closet.timestream.branchdict)
        if branches == 1:
            return self.width
        elif self.max_cols < branches:
            return self.width / self.max_cols
        else:
            return self.width / branches

    def sttt(self):
        """Return the tick I'm scrolled to, if any; otherwise pick a
        sensible tick to be at."""
        r = self._rowdict["scrolled_to"]
        if r is None:
            return self.closet.tick
        else:
            return r

    def tick_to_y(self, tick):
        """Given a tick, return the y-coordinate on me that represents it."""
        ticks_from_top = tick - self.scrolled_to
        px_from_cal_top = self.row_height * ticks_from_top
        return self.window_top - px_from_cal_top

    def y_to_tick(self, y):
        """Given a y-coordinate, what tick does it represent?"""
        px_from_cal_top = self.window_top - y
        ticks_from_top = px_from_cal_top / self.row_height
        return ticks_from_top + self.scrolled_to

    def overlaps(self, x, y):
        # My hitbox is a bit bigger than I appear, because sometimes
        # the timeline flows out of my bounds.
        return (
            self.window_left - self.style.spacing < x and
            self.window_right + self.style.spacing > x and
            self.window_bot < y and
            self.window_top > y)

    def chk_overlap(self, x, y):
        col = self.make_col(self.closet.branch)
        tl = Timeline(col)
        if tl.overlaps(x, y):
            return tl
        elif self.overlaps(x, y):
            return self

    def make_col(self, branch):
        return {
            CAL_TYPE['THING']: {
                True: LocationCalendarCol,
                False: ThingCalendarCol}[
                self.thing_show_location],
            CAL_TYPE['PLACE']: PlaceCalendarCol,
            CAL_TYPE['PORTAL']: PortalCalendarCol,
            CAL_TYPE['STAT']: StatCalendarCol,
            CAL_TYPE['SKILL']: SkillCalendarCol
        }[self.typ](self, branch)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.offx += dx
        self.offy += dy
        return self

    def dropped(self, x, y, button, modifiers):
        while self.offx > self.col_width:
            self.left_branch -= 1
            self.offx -= self.col_width
        rightmostbranch = self.closet.timestream.hi_branch
        if rightmostbranch <= self.left_branch:
            self.left_branch = rightmostbranch - 1
        while (
                self.offx * -1 > self.col_width):
            self.left_branch += 1
            self.offx += self.col_width
        self.left_branch = max((self.left_branch, 0))
        while (self.offy > self.row_height):
            self.scrolled_to += 1
            self.offy -= self.row_height
        self.scrolled_to = min((
            self.closet.timestream.latest_tick + self.rows_shown,
            self.scrolled_to))
        while self.offy * -1 > self.row_height:
            self.scrolled_to -= 1
            self.offy += self.row_height
        if self.scrolled_to < -10:
            self.scrolled_to = -10
        self.offx = 0
        self.offy = 0

    def draw(self):
        """Put all the visible CalendarCols and everything in them into my
group in my batch.

While the CalendarCols are lazy, this method is not. Making it lazier
would be good.

        """
        batch = self.window.batch
        drawn = []
        slew = self.offx % self.col_width
        o = self.left_branch - slew
        d = self.left_branch + self.max_cols + slew
        rightmostbranch = self.closet.timestream.hi_branch
        if d > rightmostbranch:
            d = rightmostbranch + 1
        if o < 0:
            o = 0
        for branch in xrange(o, d):
            column = self.make_col(branch)
            colgrp = Group(self.group)
            assert(column.start_tick is not None)
            for cell in iter(column):
                group = colgrp
                gonna_draw = (
                    cell.get_box(batch, group),
                    cell.get_label(batch, group))
                logger.debug("GONNA DRAW {0}".format(cell))
                drawn.extend(gonna_draw)
            if int(column) != 0:
                bc = BranchConnector(column)
                bcgrp = colgrp
                drawn.append(bc.get_line(batch, bcgrp))
                drawn.append(bc.get_wedge(batch, bcgrp))
            if int(column) == self.closet.branch:
                tl = Timeline(column)
                drawn.append(tl.get_line(batch, self.wedgegroup))
                drawn.append(tl.get_handle(batch, self.wedgegroup))
        self.delete()
        self.last_draw = drawn

    def delete(self):
        if self.last_draw is None:
            return
        for drawn in self.last_draw:
            try:
                drawn.delete()
            except AttributeError:
                pass
        self.last_draw = None


class CalendarColCellIter:
    """Lazy iterator over the cells in a column.

First argument is the column itself. Second argument is the skeleton
of its schedule, with the first level key being the branch ID. Third
argument is the name of the field to be displayed in the cell.

    """
    def __init__(self, col, skel, field=None):
        self.column = col
        self.skeleton = skel
        self.realiter = iter(skel)
        self.prevrd = self.skeleton[self.realiter.next()]
        self.last = False
        self.field = field

    def __iter__(self):
        return self

    def next(self):
        if self.last:
            raise StopIteration
        try:
            rd = self.skeleton[self.realiter.next()]
        except StopIteration:
            rd = {"tick_from": None}
            self.last = True
        if self.field is None:
            cc = CalendarCell(
                self.column, self.prevrd["tick_from"] + 1,
                rd["tick_from"], "")
        else:
            cc = CalendarCell(
                self.column, self.prevrd["tick_from"] + 1,
                rd["tick_from"], self.prevrd[self.field])
        self.prevrd = rd
        return cc


class CalendarColIter:
    """Iterator over the columns in a given calendar. Lazily evaluated."""
    def __init__(self, cal):
        self.calendar = cal
        self.branchiter = self.calendar.closet.timestream.branchdict.iterkeys()

    def make_col(self, branch):
        """Return a new column for the given branch."""
        return self.calendar.make_col(branch)

    def __iter__(self):
        return self

    def next(self):
        return self.make_col(self.branchiter.next())


class CalendarCol:
    """A column in a calendar. Represents one branch.

Shows whatever the calendar is about, in that branch."""

    atrdic = {
        "width": lambda self: max((self.calendar.col_width, 1)),
        "rx": lambda self: self.width / 2,
        "height": lambda self: self.window_top - self.window_bot,
        "ry": lambda self: self.height / 2,
        "window_left": lambda self:
        self.calendar.window_left + self.calendar.offx + (
            int(self) -
            self.calendar.left_branch
        ) * self.calendar.col_width,
        "window_right": lambda self:
        self.window_left + self.calendar.col_width,
        "window_top": lambda self: self.calendar.tick_to_y(self.start_tick),
        "window_bot": lambda self: self.calendar.tick_to_y(self.end_tick),
        "parent": lambda self: self.calendar.make_col(
            self.closet.skeleton[
                "timestream"][self.branch]["parent"]),
        "start_tick": lambda self: {
            True: lambda:
            self.closet.skeleton["timestream"][int(self)]["tick_from"],
            False: lambda: None
        }[int(self) in self.closet.skeleton["timestream"]](),
        "end_tick": lambda self: {
            True: lambda:
            self.closet.skeleton["timestream"][int(self)]["tick_to"],
            False: lambda: None
        }[int(self) in self.closet.skeleton["timestream"]]()}

    def __init__(self, calendar, branch, bgcolor=(255, 255, 255, 255)):
        """Get CalendarCol for the given branch in the given
        calendar."""
        self.calendar = calendar
        self.branch = branch
        self.closet = self.calendar.closet
        self.batch = self.calendar.batch
        self.style = self.calendar.style
        self.window = self.calendar.window

    def __getattr__(self, attrn):
        """Use a lambda from my atrdic to compute the attribute."""
        return CalendarCol.atrdic[attrn](self)

    def __int__(self):
        """Return my branch."""
        return self.branch

    def __eq__(self, other):
        return int(self) == int(other)

    def __gt__(self, other):
        return int(self) > int(other)

    def __lt__(self, other):
        return int(self) < int(other)

    def __hash__(self):
        return hash(int(self))

    def pretty_caster(self, *args):
        """Make my content into a single, flat list"""
        unargs = []
        for arg in args:
            if isinstance(arg, tuple) or isinstance(arg, list):
                unargs += self.pretty_caster(*arg)
            else:
                unargs.append(arg)
        return unargs

    def pretty_printer(self, *args):
        """Make my content look good."""
        strings = []
        unargs = self.pretty_caster(*args)
        for unarg in unargs:
            strings.append(str(unarg))
        return ";\n".join(strings)


class LocationCalendarCol(CalendarCol):
    """A column of a calendar displaying a Thing's location over time.

The column only shows its Thing's location during those times when its
Thing is a part of its Tharacter. Other times, the column is
transparent. If all its visible area is transparent, it will still
take up space in its calendar, in case the user scrolls it to
somewhere visible.

The cells in the column are sized to encompass the duration of the
Thing's stay in that location. If the location is a Place, its name is
displayed in the cell. If it is a Portal, a format-string is used
instead, giving something like "in transit from A to B".

    """
    typ = CAL_TYPE['THING']
    cal_attrs = set([
        "character",
        "dimension",
        "thing",
        "location"])

    def __init__(self, calendar, branch):
        """Initialize a LocationCalendarCol representing the given
        branch in the given calendar."""
        CalendarCol.__init__(self, calendar, branch)
        self.dimension = self.calendar.dimension
        self.thing = self.calendar.thing
        self.locations = self.thing.locations[branch]
        self.coverage = self.character.thingdict[
            str(self.dimension)][str(self.thing)][branch]

    def __iter__(self):
        return CalendarColCellIter(self, self.locations, "location")

    def __getattr__(self, attrn):
        """Try looking up the attribute in the calendar first;
        otherwise use lambdas from CalendarCol.atrdic to compute it"""
        if attrn == "sprite":
            return self.calendar.col_sprite_dict[self]
        elif attrn in LocationCalendarCol.cal_attrs:
            return getattr(self.calendar, attrn)
        else:
            try:
                return CalendarCol.atrdic[attrn](self)
            except KeyError:
                raise AttributeError(
                    """LocationCalendarCol does not have and
cannot compute attribute {0}""".format(attrn))


class ThingCalendarCol(CalendarCol):
    def __iter__(self):
        return CalendarColCellIter(self, self.coverage)


class PlaceCalendarCol(CalendarCol):
    def __iter__(self):
        return CalendarColCellIter(self, self.coverage)


class PortalCalendarCol(CalendarCol):
    def __iter__(self):
        return CalendarColCellIter(self, self.coverage)


class StatCalendarCol(CalendarCol):
    def __iter__(self):
        return CalendarColCellIter(self, self.values, "value")


class SkillCalendarCol(CalendarCol):
    def __iter__(self):
        return CalendarColCellIter(self, self.decks, "deck")
