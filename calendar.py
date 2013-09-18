# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass, phi
from logging import getLogger
from pyglet.text import Label
from pyglet.graphics import GL_LINES, GL_TRIANGLES, OrderedGroup, Group
from pyglet.gl import glScissor, glEnable, glDisable, GL_SCISSOR_TEST
from pyglet.image import SolidColorImagePattern
from pyglet.sprite import Sprite

"""User's view on a given item's schedule.

Usually there should be only one calendar per board, but it can switch
between showing various schedules, or even show many in parallel.

"""


__metaclass__ = SaveableMetaclass


logger = getLogger(__name__)


class Wedge:
    """The triangle at the end of a BranchConnector.

It points at the CalendarCol that is the child of the one at the start
of the BranchConnector.

    """
    atrdic = {
        "window_bot": lambda self: self.bc.end[1],
        "window_top": lambda self: self.bc.end[1] + self.height,
        "window_left": lambda self: self.bc.end[0] - self.rx,
        "window_right": lambda self: self.bc.end[0] + self.rx,
        "in_view": lambda self:
        self.window_right > 0 and
        self.window_top > 0 and
        self.window_bot < self.window.height and
        self.window_left < self.window.width}

    def __init__(self, bc, color_tup=(255, 0, 0, 255)):
        """Get a Wedge for the given BranchConnector, and optionally,
        give it a color.

        The color is a 4-tuple of Red, Green, Blue, Alpha."""
        self.bc = bc
        self.color = color_tup
        self.batch = bc.batch
        self.window = bc.window
        width = self.bc.calendar.style.spacing * 2
        self.width = width
        height = int(width / phi)
        self.height = height
        self.rx = width / 2
        self.ry = height / 2
        self.vertlist = None

    def __getattr__(self, attrn):
        """Look up computed attributes in the atrdic of the class."""
        try:
            return Wedge.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                "Wedge instance does not have and cannot compute "
                "attribute {0}".format(attrn))

    def draw(self):
        """If I'm in view, add my vertices to menu_fg_group in my
        window's Batch. Otherwise just delete them."""
        if not self.in_view:
            self.delete()
            return
        (x, y) = self.bc.end
        l = x - self.rx
        c = x
        r = x + self.rx
        t = y + self.height
        b = y
        points = (
            c, b,
            l, t,
            r, t)
        colors = self.bc.color * 3
        self.delete()
        self.vertlist = self.batch.add_indexed(
            3,
            GL_TRIANGLES,
            self.window.menu_fg_group,
            (0, 1, 2, 0),
            ('v2i', points),
            ('c4B', colors))

    def delete(self):
        if self.vertlist is not None:
            try:
                self.vertlist.delete()
            except AttributeError:
                pass
            self.vertlist = None


class BranchConnector:
    """Widget to show where a branch branches off another.

    It's an arrow that leads from the tick on one CalendarCol where
    another branches from it, to the start of the child.

    """
    atrdic = {
        "startx": lambda self: {
            True:
                lambda: self.col1.window_right - self.calendar.style.spacing,
            False:
                lambda: self.col1.window_left - self.calendar.style.spacing
            }[self.col1.window_left < self.col2.window_left](),
        "endx": lambda self: self.col2.window_left + self.col2.rx,
        "starty": lambda self: (
            self.col1.window_top - self.calendar.row_height * (
                self.tick - self.calendar.scrolled_to)),
        "endy": lambda self: (
            self.col2.window_top - self.calendar.row_height * (
                self.tick - self.calendar.scrolled_to)),
        "centerx": lambda self: {
            True:
                lambda:
                self.col1.window_right + self.calendar.style.spacing / 2,
            False:
                lambda:
                self.col2.window_right + self.calendar.style.spacing / 2
            }[self.col1.window_left < self.col2.window_left](),
        "in_view": lambda self: self.col1.in_view or self.col2.in_view,
        "points": lambda self: self.get_points(),
        "start": lambda self: (self.startx, self.starty),
        "end": lambda self: (self.endx, self.endy),
        "wedge_visible": lambda self: self.wedge.in_view()}

    def __init__(self, calendar, col1, col2, tick, color=(255, 0, 0, 255)):
        """Get a BranchConnector for the given calendar, connecting
        the two given columns, and branching off of the first one at
        the given tick.

The optional color argument is a 4-tuple of Red, Green, Blue,
Alpha."""
        self.calendar = calendar
        self.batch = self.calendar.batch
        self.window = self.calendar.window
        self.col1 = col1
        self.col2 = col2
        self.tick = tick
        self.color = color
        self.wedge = Wedge(self)
        self.space = self.calendar.style.spacing * 2
        self.oldpoints = None
        self.oldindices = None

    def __getattr__(self, attrn):
        """Look up computed attributes in the atrdic of the class."""
        try:
            return BranchConnector.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                "BranchConnector has no attribute named {0}".format(attrn))

    def draw(self):
        """If I'm in view, add my vertices to menu_fg_group in my
        window's Batch. Otherwise just delete them."""
        if not self.in_view:
            self.delete()
            return
        points = self.points
        if self.wedge_visible:
            indices = (0, 1, 1, 2, 2, 3, 3, 4)
        elif (
                points[2] > self.calendar.window_left and
                points[2] < self.calendar.window_right):
            indices = (0, 1, 1, 2, 2, 3)
        else:
            indices = (0, 1)
        self.delete()
        colors = self.color * 5
        self.vertlist = self.batch.add_indexed(
            5,
            GL_LINES,
            self.window.menu_fg_group,
            indices,
            ('v2i', points),
            ('c4B', colors))
        if self.wedge_visible:
            self.wedge.draw()

    def get_points(self):
        """Return the points making up the jaggedy line connecting one
        CalendarCol to another--not the points in the Wedge at the
        end."""
        x0 = self.get_startx()
        y = self.col1.window_top - self.calendar.row_height * (
            self.tick - self.calendar.scrolled_to)
        x2 = self.get_centerx()
        x5 = self.col2.window_left + self.col2.rx
        return (
            x0, y,
            x2, y,
            x2, y + self.space,
            x5, y + self.space,
            x5, y)

    def delete(self):
        """Immediately remove from video memory"""
        if self.vertlist is not None:
            try:
                self.vertlist.delete()
            except AttributeError:
                pass
            self.vertlist = None
        self.wedge.delete()


class Handle:
    """The thing on the timeline that you grab to move it."""
    atrdic = {
        "y": lambda self: self.timeline.y,
        "window_y": lambda self: self.timeline.window_y,
        "window_left": lambda self: {
            True: self.timeline.window_left - self.width,
            False: self.timeline.window_right}[self.on_the_left],
        "window_right": lambda self: {
            True: self.timeline.window_left + 1,
            False: self.timeline.window_right + self.width - 1
            }[self.on_the_left],
        "window_top": lambda self: self.y + self.ry,
        "window_bot": lambda self: self.y - self.ry,
        "in_view": lambda self: (
            self.window_right > 0 and
            self.window_top > 0 and
            self.window_bot < self.window.height and
            self.window_left < self.window.width)}

    def __init__(self, timeline, handle_side):
        """Make a handle for the given timeline on its given side,
        "left" by default, possibly "right"."""
        self.timeline = timeline
        self.window = self.timeline.window
        self.on_the_left = handle_side == "left"
        self.vertlist = None
        width = timeline.cal.style.spacing * 2
        self.width = width
        height = int(width * phi)
        self.height = height
        self.rx = width / 2
        self.ry = height / 2
        self.closet = self.timeline.cal.closet
        self.calendar = self.timeline.cal

    def __getattr__(self, attrn):
        """Look up computed attributes in the atrdic of the class."""
        if attrn in self.atrdic:
            return self.atrdic[attrn](self)
        else:
            raise AttributeError(
                "Handle instance does not have and "
                "cannot compute attribute {0}".format(attrn))

    def delete(self):
        """Remove from video memory immediately."""
        try:
            self.vertlist.delete()
        except AttributeError:
            pass
        self.vertlist = None

    def draw(self):
        """If I'm in view, add myself to menu_fg_group in my window's
        Batch."""
        if not self.in_view:
            return
        batch = self.timeline.batch
        group = self.window.menu_fg_group
        colors = self.timeline.color * 3
        points = (
            self.window_right, self.y,
            self.window_left, self.window_bot,
            self.window_left, self.window_top)
        try:
            self.vertlist.vertices = list(points)
            self.vertlist.colors = list(colors)
        except AttributeError:
            self.vertlist = batch.add_indexed(
                3,
                GL_TRIANGLES,
                group,
                (0, 1, 2, 0),
                ('v2i', points),
                ('c4B', colors))

    def overlaps(self, x, y):
        """Check if the given window coordinates are in me."""
        return (
            self.window_left < x and
            x < self.window_right and
            self.window_bot < y and
            y < self.window_top)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        """Move the branch and tick in accordance with how the user drags me.

The tick shall always be the one represented by the calendar at the
height of the mouse. The branch is the one represented by the column
whose center line is nearest me, measured from the side of me that I
point to."""
        pointing_right = self.on_the_left
        nearcol = self.timeline.column
        if pointing_right:
            for column in self.calendar.cols:
                if (
                        column.window_center > self.window_right and
                        column.window_center < nearcol.window_center):
                    nearcol = column
        else:
            for column in self.calendar.cols:
                if (
                        column.window_center < self.window_left and
                        column.window_center > nearcol.window_center):
                    nearcol = column
        branch = nearcol.branch
        tick = self.closet.tick
        if y >= self.calendar.window_top:
            tick = self.calendar.top_tick
        elif y <= self.calendar.window_bot:
            tick = self.calendar.bot_tick
        else:
            tick = self.calendar.y_to_tick(y)
        if branch != self.closet.branch or tick != self.closet.tick:
            self.closet.time_travel(None, branch, tick)


class Timeline:
    """A line that goes on top of a CalendarCol to indicate what time it
is.

Also has a little handle that you can drag to do the time warp. Select
which side it should be on by supplying "left" (default) or "right"
for the handle_side keyword argument.

    """
    atrdic = {
        "calendar_left": lambda self:
        self.col.calendar_left + self.col.style.spacing,
        "calendar_right": lambda self:
        self.calendar_left + self.col.width,
        "calendar_y": lambda self:
        self.cal.height - self.cal.row_height * (
            self.closet.tick - self.cal.scrolled_to),
        "calendar_bot": lambda self: self.calendar_y,
        "calendar_top": lambda self: self.calendar_y,
        "window_y": lambda self:
        self.calendar_y + self.cal.window_bot,
        "y": lambda self: self.window_y,
        "window_bot": lambda self: self.window_y,
        "window_top": lambda self: self.window_y,
        "window_left": lambda self:
        self.calendar_left + self.cal.window_left,
        "window_right": lambda self:
        self.window_left + self.col.width,
        "in_window": lambda self:
        (self.y > 0 and self.y < self.window.height
         and self.window_right > 0
         and self.window_left < self.window.width)}

    def __init__(self, col, handle_side="left", color=(255, 0, 0, 255)):
        """Make a timeline for the given column, optionally specifying
        what side its handle is on, and what color it is.

color is a 4-tuple of Red, Green, Blue, Alpha."""
        self.col = col
        self.cal = self.col.calendar
        self.batch = self.col.batch
        self.window = self.cal.window
        self.closet = self.col.closet
        self.handle = Handle(self, handle_side)
        self.color = color
        self.vertlist = None

    def __getattr__(self, attrn):
        """Try to compute the attribute using the lambdas in my
        atrdic."""
        try:
            return self.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                "Timeline instance does not have and cannot "
                "compute attribute {0}".format(attrn))

    def delete(self):
        """Immediately remove myself and my handle from video
        memory."""
        if self.vertlist is not None:
            try:
                self.vertlist.delete()
            except AttributeError:
                pass
            self.vertlist = None
        self.handle.delete()

    def really_draw(self):
        """Now that I know I'm supposed to draw, add myself and my
        handle to menu_fg_group in my window's Batch."""
        colors = self.color * 2
        points = (
            self.window_left, self.y,
            self.window_right, self.y)
        self.delete()
        assert(self.vertlist is None)
        self.vertlist = self.batch.add(
            2,
            GL_LINES,
            self.window.timeline_group,
            ('v2i', points),
            ('c4B', colors))
        self.handle.draw()

    def draw(self):
        """Really draw if the current branch is mine, otherwise delete."""
        if self.col.branch == self.closet.branch:
            self.really_draw()
        else:
            self.delete()


class CalendarCellGroup(Group):
    """A group to set up a GL_SCISSOR_TEST that will keep the text in
    a calendar cell within that cell."""
    def __init__(self, cell, parent):
        """Get a CalendarCellGroup for the given cell and parent."""
        super(CalendarCellGroup, self).__init__(parent)
        self.cell = cell

    def gettup(self):
        """Get the stuff I need to pass to glScissor()."""
        return (
            self.cell.window_left - 1,
            self.cell.window_bot - 1,
            self.cell.width + 1,
            self.cell.height + 1)

    def set_state(self):
        """Set the scissor test to the bounds of the cell, and enable
        it."""
        tup = self.gettup()
        glScissor(*tup)
        glEnable(GL_SCISSOR_TEST)

    def unset_state(self):
        """Disable the scissor test."""
        glDisable(GL_SCISSOR_TEST)


class CalendarCell:
    """A block of time in a calendar.

Uses information from the CalendarCol it's in and the Event it
represents to calculate its dimensions and coordinates.

    """
    visible = True

    atrdic = {
        "interactive": lambda self: self.column.calendar.interactive,
        "window": lambda self: self.column.calendar.window,
        "calendar_left": lambda self:
        self.column.calendar_left + self.style.spacing,
        "calendar_right": lambda self:
        self.column.calendar_right - self.style.spacing,
        "calendar_top": lambda self:
        (self.calendar.height - self.calendar.row_height *
         (self.tick_from - self.calendar.scrolled_to) -
         self.style.spacing),
        "calendar_bot": lambda self: self.get_calendar_bot(),
        "width": lambda self: self.calendar_right - self.calendar_left,
        "height": lambda self: {
            True: lambda:
            self.calendar.height - (
                self.tick_from * self.calendar.row_height),
            False: lambda: self.calendar.row_height * len(self)
            }[self.tick_to is None](),
        "window_left": lambda self:
        self.calendar_left + self.calendar.window_left,
        "window_right": lambda self:
        self.calendar_right + self.calendar.window_left,
        "window_top": lambda self:
        self.calendar_top + self.calendar.window_bot,
        "window_bot": lambda self:
        self.calendar_bot + self.calendar.window_bot,
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
            return Label(
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

    def get_box(self, batch, group):
        l = self.window_left
        b = self.window_bot
        r = self.window_right
        t = self.window_top
        colors = self.style.fg_inactive.tup * 8
        vees = (l, t, r, t,
                r, t, r, b,
                r, b, l, b,
                l, b, l, t)
        return batch.add(
            8,
            GL_LINES,
            group,
            ('v2i', vees),
            ('c4B', colors))

    def draw_label(self, l, t, w, h):
        """Draw label at the given coordinates with the given width
        and height"""
        if self.style.fontsize > self.height:
            return
        self.label = Label(
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
            batch=self.batch,
            group=self.column.group)

    def draw_box(self, l, b, r, t, color):
        """Draw box with given edges and color"""
        colors = color * 8
        vees = (l, t, r, t, r, t, r, b, r, b, l, b, l, b, l, t)
        self.vertl = self.batch.add(
            8,
            GL_LINES,
            self.window.menu_fg_group,
            ('v2i', vees),
            ('c4B', colors))

    def draw(self):
        """Put myself in the batch"""
        l = self.window_left
        r = self.window_right
        t = self.window_top
        b = self.window_bot
        black = (0, 0, 0, 255)
        if self.label is None:
            self.draw_label(l, t, self.width, self.height)
        elif self.old_label_left != l or self.old_label_top != t:
            try:
                self.label.delete()
            except AttributeError:
                pass
            self.draw_label(l, t, self.width, self.height)
        self.old_label_left = l
        self.old_label_top = t
        if self.vertl is None:
            self.draw_box(l, b, r, t, black)
        else:
            self.vertl.vertices = (
                l, t, r, t, r, t, r, b, r, b, l, b, l, b, l, t)

CAL_TYPE = {
    "THING": 0,
    "PLACE": 1,
    "PORTAL": 2,
    "STAT": 3,
    "SKILL": 4}


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
             "branch_left": "INTEGER NOT NULL DEFAULT 0",
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
        "branch_left": lambda self: self._rowdict["branch_left"],
        "left_prop": lambda self: self._rowdict["left"],
        "right_prop": lambda self: self._rowdict["right"],
        "top_prop": lambda self: self._rowdict["top"],
        "bot_prop": lambda self: self._rowdict["bot"],
        "bot_tick": lambda self: self.top_tick + self.rows_shown,
        "style": lambda self: self.closet.get_style(self._rowdict["style"]),
        "window_top": lambda self: int(self.top_prop * self.window.height),
        "window_bot": lambda self: int(self.bot_prop * self.window.height),
        "window_left": lambda self: int(self.left_prop * self.window.width),
        "window_right": lambda self: int(self.right_prop * self.window.width),
        "width": lambda self: self.window_right - self.window_left,
        "col_width": lambda self: {
            True: lambda: self.width,
            False: lambda: self.width / len(self.cols_shown)
            }[len(self.cols_shown) == 0](),
        "height": lambda self: self.window_top - self.window_bot,
        "row_height": lambda self: self.height / self.rows_shown,
        "scrolled_to": lambda self: self.sttt(),
        "top_tick": lambda self: self.sttt(),
        "scroll_factor": lambda self: self._rowdict["scroll_factor"],
        "max_cols": lambda self: self._rowdict["max_cols"],
        "thing_show_location": lambda self: (
            self._rowdict["thing_show_location"]
            not in (0, None, False)),
        "cols_shown": lambda self: [
            self.coldict[k] for k in xrange(
                self.branch_left, self.branch_left + self.max_cols)
            if k in self.coldict]
    }

    def __init__(self, window, idx):
        """Get the idx-th calendar in the given window"""
        self.window = window
        self.closet = self.window.closet
        self.idx = idx
        self.batch = self.window.batch
        self.old_state = None
        self.tainted = False
        self._rowdict = self.closet.skeleton[
            "calendar"][
            str(self.window)][
            int(self)]
        if self._rowdict["type"] == CAL_TYPE['THING']:
            self.dimension = self.closet.get_dimension(
                self._rowdict["dimension"])
            self.thing = self.closet.get_thing(
                self._rowdict["dimension"], self._rowdict["thing"])
            self.thing.locations.listeners.add(self)
            if self._rowdict["thing_show_location"]:
                self._location_dict = self.closet.skeleton[
                    "thing_location"][
                    self._rowdict["dimension"]][
                    self._rowdict["thing"]]
        self.coldict = {}
        self.branch_to = 0
        for branch in self.closet.timestream.branchdict:
            try:
                self.coldict[branch] = self.make_col(branch)
                self.branch_to = branch
            except KeyError:
                pass

    def __int__(self):
        """What-th calendar in my window am I?

So, return my index."""
        return self.idx

    def __getattr__(self, attrn):
        """Compute the attribute using the correct lambda from my atrdic"""
        try:
            return self.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                "Calendar instance has no attribute {0}".format(attrn))

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
        ticks_from_top = tick - self.top_tick
        px_from_cal_top = self.row_height * ticks_from_top
        return self.window_top - px_from_cal_top

    def y_to_tick(self, y):
        """Given a y-coordinate, what tick does it represent?"""
        px_from_cal_top = self.window_top - y
        ticks_from_top = px_from_cal_top / self.row_height
        return ticks_from_top + self.top_tick

    def overlaps(self, x, y):
        """Is that point in me?"""
        return (
            self.window_left < x and
            self.window_right > x and
            self.window_bot < y and
            self.window_top > y)

    def draw(self):
        """Draw all my columns."""
        if self.visible and len(self.cols_shown) > 0:
            for calcol in self.cols_shown:
                calcol.draw()
        else:
            for calcol in self.cols_shown:
                calcol.delete()

    def make_col(self, branch):
        """Return a new column for the given branch."""
        logger.debug("Calendar making col {0}!".format(branch))
        return {
            CAL_TYPE['THING']: {
                True: LocationCalendarCol,
                False: ThingCalendarCol}[self.thing_show_location],
            CAL_TYPE['PLACE']: PlaceCalendarCol,
            CAL_TYPE['PORTAL']: PortalCalendarCol,
            CAL_TYPE['STAT']: StatCalendarCol,
            CAL_TYPE['SKILL']: SkillCalendarCol
        }[self.typ](self, branch)

    def rearrow(self):
        """Rearrange the BranchConnectors."""
        for coli in xrange(0, len(self.cols_shown) - 1):
            col1 = self.coldict[coli]
            rd = self.closet.timestream.branchdict[
                col1.branch]
            parent = rd["parent"]
            tick_from = rd["tick_from"]
            tick_to = rd["tick_to"]
            if hasattr(col1, 'bc'):
                col1.bc.delete()
            col2 = None
            for calcol in self.cols_shown:
                if calcol.branch == parent:
                    col2 = calcol
                    break
            if (
                    col2 is not None and
                    tick_from > self.scrolled_to and
                    tick_to < self.scrolled_to + self.rows_shown):
                col2.bc = BranchConnector(
                    self, col2, col1, tick_from)


class CalendarColGroup(OrderedGroup):
    """A group to scissor-test columns so the text doesn't run out of
    their cells.

This is technically an OrderedGroup because its parent is to be an
OrderedGroup and Pyglet seems to get confused when an unordered group
is child of an ordered one."""
    order = 0

    def __init__(self, col):
        super(CalendarColGroup, self).__init__(
            CalendarColGroup.order, col.window.menu_fg_group)
        CalendarColGroup.order += 1
        self.col = col

    def gettup(self):
        return (
            self.col.window_left,
            self.col.window_bot,
            self.col.width - self.col.style.spacing,
            self.col.height)

    def set_state(self):
        glEnable(GL_SCISSOR_TEST)
        glScissor(*self.gettup())

    def unset_state(self):
        glDisable(GL_SCISSOR_TEST)


class CalendarColCellIter:
    """Lazy iterator over the cells in a column.

First argument is the column itself. Second argument is the skeleton
of its schedule, with the first level key being the branch ID. Third
argument is the name of the field to be displayed in the cell.

    """
    def __init__(self, col, skel, field=""):
        self.column = col
        self.skeleton = skel
        self.realiter = self.skeleton.iterrows()
        self.field = field

    def __iter__(self):
        return self

    def next(self):
        rd = self.realiter.next()
        return CalendarCell(
            self.column, rd["tick_from"],
            rd["tick_to"], rd[self.field])


class CalendarCol:
    """A column in a calendar. Represents one branch.

Shows whatever the calendar is about, in that branch."""
    atrdic = {
        "width": lambda self: self.calendar.col_width,
        "rx": lambda self: self.width / 2,
        "height": lambda self: self.calendar.height,
        "ry": lambda self: self.height / 2,
        "calendar_left": lambda self: {
            True: lambda: (
                self.calendar.cols_shown.index(self) * self.width),
            False: lambda: self.calendar.width
            }[self in self.calendar.cols_shown](),
        "calendar_right": lambda self: self.calendar_left + self.width,
        "calendar_top": lambda self: self.calendar.height,
        "calendar_bot": lambda self: 0,
        "window_left": lambda self:
        self.calendar.window_left + self.calendar_left,
        "window_right": lambda self:
        self.calendar.window_left + self.calendar_right,
        "window_top": lambda self: self.calendar.window_top,
        "window_bot": lambda self: self.calendar.window_bot,
        "window_center": lambda self: self.window_left + self.rx,
        "in_view": lambda self: (
            int(self) >= self.calendar.branch_left and
            int(self) < self.calendar.branch_left + self.calendar.max_cols),
        "idx": lambda self: self.calendar.cols.index(self)}

    def __init__(self, calendar, branch, bgcolor=(255, 255, 255, 255)):
        """Get CalendarCol for the given branch in the given
        calendar."""
        self.calendar = calendar
        self.branch = branch
        self.closet = self.calendar.closet
        self.batch = self.calendar.batch
        self.style = self.calendar.style
        self.timeline = Timeline(self)
        self.window = self.calendar.window
        self.group = CalendarColGroup(self)
        self.bgpat = SolidColorImagePattern(bgcolor)
        self.sprite = None
        self.oldwidth = None
        self.oldheight = None
        self.oldleft = None
        self.oldbot = None

    def __getattr__(self, attrn):
        """Use a lambda from my atrdic to compute the attribute."""
        return CalendarCol.atrdic[attrn](self)

    def __int__(self):
        """Return my branch."""
        return self.branch

    def delete(self):
        """Remove from video memory"""
        self.timeline.delete()
        for cell in self.celldict.itervalues():
            cell.delete()
        try:
            self.vertl.delete()
        except AttributeError:
            pass
        self.vertl = None

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

    def draw_sprite(self):
        """Draw a flat color background image"""
        self.image = self.bgpat.create_image(self.width, self.height)
        self.sprite = Sprite(
            self.image, self.window_left, self.window_bot,
            batch=self.batch, group=self.window.menu_bg_group)

    def draw(self):
        """Put myself and all my cells into the batch"""
        if self.sprite is None:
            self.draw_sprite()
        elif self.width != self.oldwidth or self.height != self.oldheight:
            oldsprite = self.sprite
            self.draw_sprite()
            try:
                oldsprite.delete()
            except AttributeError:
                pass
            self.oldwidth = self.width
            self.oldheight = self.height
        elif (
                self.oldleft != self.window_left or
                self.oldbot != self.window_bot):
            self.sprite.set_position(self.window_left, self.window_bot)
            self.oldleft = self.window_left
            self.oldbot = self.window_bot
        if hasattr(self, 'bc'):
            self.bc.draw()
        self.timeline.draw()
        if not hasattr(self, 'tlid'):
            self.tlid = id(self.timeline)
        else:
            assert(self.tlid == id(self.timeline))


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
        self.labeldict = {}
        self.boxdict = {}

    def __getattr__(self, attrn):
        """Try looking up the attribute in the calendar first;
        otherwise use lambdas from CalendarCol.atrdic to compute it"""
        if attrn in LocationCalendarCol.cal_attrs:
            return getattr(self.calendar, attrn)
        elif attrn == "cells":
            return CalendarColCellIter(self, self.locations, "location")
        else:
            try:
                return CalendarCol.atrdic[attrn](self)
            except KeyError:
                raise AttributeError(
                    """LocationCalendarCol does not have and
cannot compute attribute {0}""".format(attrn))

    def shows_any_ever(self, tick_from, tick_to):
        for (cover_tick_from, cover_tick_to) in self.coverage.iteritems():
            if tick_to > cover_tick_from or tick_from < cover_tick_to:
                return True
        return False

    def shows_when(self, tick_from, tick_to):
        for (cover_tick_from, cover_tick_to) in self.coverage.iteritems():
            if tick_to > cover_tick_from or tick_from < cover_tick_to:
                # I show part of this cell, but which part?
                if tick_from > cover_tick_from:
                    a = tick_from
                else:
                    a = cover_tick_from
                if tick_to < cover_tick_to:
                    b = tick_to
                else:
                    b = cover_tick_to
                return (a, b)
        return None

    def draw(self):
        super(LocationCalendarCol, self).draw()
        for cell in self.cells:
            if cell in self.boxdict:
                vertl = self.boxdict[cell]
                l = cell.window_left
                r = cell.window_right
                t = cell.window_top
                b = cell.window_bot
                try:
                    vertl.vertices = (
                        l, t, r, t,
                        r, t, r, b,
                        r, b, l, b,
                        l, b, l, t)
                except AttributeError:
                    vertl = cell.get_box(
                        self.batch, self.group)
                self.boxdict[cell] = vertl
            else:
                self.boxdict[cell] = cell.get_box(
                    self.batch, self.group)
            if cell in self.labeldict:
                (label, l, t) = self.labeldict[cell]
                if (
                        label is None or
                        l != cell.window_left or
                        t != cell.window_top):
                    try:
                        label.delete()
                    except AttributeError:
                        pass
                    self.labeldict[cell] = (
                        cell.get_label(self.batch, self.group),
                        cell.window_left,
                        cell.window_top)
            else:
                self.labeldict[cell] = (
                    cell.get_label(self.batch, self.group),
                    cell.window_left,
                    cell.window_top)

class ThingCalendarCol(CalendarCol):
    pass


class PlaceCalendarCol(CalendarCol):
    pass


class PortalCalendarCol(CalendarCol):
    pass


class StatCalendarCol(CalendarCol):
    pass


class SkillCalendarCol(CalendarCol):
    pass
