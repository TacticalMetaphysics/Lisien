# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import (
    SaveableMetaclass,
    TabdictIterator,
    phi)
from pyglet.text import Label
from pyglet.graphics import GL_LINES, GL_TRIANGLES, OrderedGroup

"""User's view on a given item's schedule.

Usually there should be only one calendar per board, but it can switch
between showing various schedules, or even show many in parallel.

"""


__metaclass__ = SaveableMetaclass


class Wedge:
    """Downward pointing wedge, looks much like the Timeline's Handle

    """
    def __init__(self, bc, color_tup=(255, 0, 0, 255)):
        self.bc = bc
        self.batch = self.bc.batch
        self.group = self.bc.wedgegroup
        self.width = self.bc.calendar.style.spacing * 2
        self.height = int(self.width / phi)
        self.rx = self.width / 2
        self.ry = self.height / 2
        self.color = color_tup
        self.vertlist = None

    def __getattr__(self, attrn):
        if attrn == "window_bot":
            return self.bc.end[1]
        elif attrn == "window_top":
            return self.bc.end[1] + self.height
        elif attrn == "window_left":
            return self.bc.end[0] - self.rx
        elif attrn == "window_right":
            return self.bc.end[0] + self.rx
        else:
            raise AttributeError(
                "Wedge instance has no attribute named {0}".format(
                    attrn))

    def draw(self):
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
        try:
            self.vertlist.vertices = list(points)
        except AttributeError:
            self.vertlist = self.batch.add_indexed(
                3,
                GL_TRIANGLES,
                self.group,
                (0, 1, 2, 0),
                ('v2i', points),
                ('c4B', colors))

    def delete(self):
        try:
            self.vertlist.delete()
        except:
            pass
        self.vertlist = None


class BranchConnector:
    """Widget to show where a branch branches off another.

    It's an arrow that leads from the tick on one branch where
    another branches from it, to the start of the latter.

    It operates on the assumption that child branches will always
    be displayed next to their parents, when they are displayed at
    all.

    """
    color = (255, 0, 0, 255)

    def __init__(self, calendar, col1, col2, tick):
        self.calendar = calendar
        self.batch = self.calendar.batch
        self.group = col2.bcgroup
        self.linegroup = self.group
        self.wedgegroup = self.group
        self.col1 = col1
        self.col2 = col2
        self.tick = tick
        self.wedge = Wedge(self)
        self.space = self.calendar.style.spacing * 2

    def __getattr__(self, attrn):
        if attrn == "startx":
            if self.col1.window_left < self.col2.window_left:
                return self.col1.window_right - self.calendar.style.spacing
            else:
                return self.col1.window_left - self.calendar.style.spacing
        elif attrn == "endx":
            return self.col2.window_left + self.col2.rx
        elif attrn == "starty":
            return (
                self.col1.window_top -
                self.calendar.row_height * (
                    self.tick - self.calendar.scrolled_to))
        elif attrn == "endy":
            return (
                self.col2.window_top -
                self.calendar.row_height * (
                    self.tick - self.calendar.scrolled_to))
        elif attrn == "centerx":
            if self.col1.window_left < self.col2.window_left:
                return (self.col1.window_right +
                        self.calendar.style.spacing / 2)
            else:
                return (self.col2.window_right +
                        self.calendar.style.spacing / 2)
        elif attrn == "points":
            x0 = self.startx
            y = self.starty
            x2 = self.centerx
            x5 = self.endx
            return (
                x0, y,
                x2, y,
                x2, y + self.space,
                x5, y + self.space,
                x5, y)
        elif attrn == "start":
            return (
                self.startx,
                self.starty)
        elif attrn == "end":
            return (
                self.endx,
                self.endy)
        else:
            raise AttributeError(
                "BranchConnector has no attribute named {0}".format(attrn))

    def draw(self):
        points = self.points
        try:
            self.vertlist.vertices = list(points)
        except AttributeError:
            colors = self.color * 5
            self.vertlist = self.batch.add_indexed(
                5,
                GL_LINES,
                self.linegroup,
                (0, 1, 1, 2, 2, 3, 3, 4),
                ('v2i', points),
                ('c4B', colors))
        self.wedge.draw()

    def delete(self):
        try:
            self.vertlist.delete()
        except:
            pass
        self.vertlist = None
        self.wedge.delete()


class Handle:
    """The thing on the timeline that you grab to move"""
    def __init__(self, timeline, handle_side):
        self.timeline = timeline
        self.on_the_left = handle_side == "left"
        self.vertlist = None
        self.width = self.timeline.cal.style.spacing * 2
        self.height = int(self.width * phi)
        self.rx = self.width / 2
        self.ry = self.height / 2

    def __getattr__(self, attrn):
        if attrn in ("y", "window_y"):
            return getattr(self.timeline, attrn)
        elif attrn == "window_left":
            if self.on_the_left:
                return self.timeline.window_left - self.width
            else:
                return self.timeline.window_right
        elif attrn == "window_right":
            if self.on_the_left:
                return self.timeline.window_left + 1
            else:
                return self.timeline.window_right + self.width - 1
        elif attrn == "window_top":
            return self.y + self.ry
        elif attrn == "window_bot":
            return self.y - self.ry
        else:
            raise AttributeError(
                "Handle instance has no attribute " + attrn)

    def delete(self):
        try:
            self.vertlist.delete()
        except:
            pass
        self.vertlist = None

    def draw(self):
        batch = self.timeline.batch
        group = self.timeline.col.tlgroup
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


class Timeline:
    """A line that goes on top of a CalendarCol to indicate what time it
is.

Also has a little handle that you can drag to do the time warp. Select
which side it should be on by supplying "left" (default) or "right"
for the handle_side keyword argument.

    """
    color = (255, 0, 0, 255)

    def __init__(self, col, handle_side="left"):
        self.col = col
        self.cal = self.col.calendar
        self.batch = self.col.batch
        self.window = self.cal.window
        self.rumor = self.col.rumor
        self.handle = Handle(self, handle_side)
        self.vertlist = None

    def __getattr__(self, attrn):
        if attrn in ("calendar_y", "calendar_bot", "calendar_top"):
            return self.cal.height - self.cal.row_height * (
                self.rumor.tick - self.cal.scrolled_to)
        elif attrn in ("y", "window_y", "window_bot", "window_top"):
            return self.calendar_y + self.cal.window_bot
        elif attrn == "calendar_left":
            return self.col.calendar_left + self.col.style.spacing
        elif attrn == "window_left":
            return self.calendar_left + self.cal.window_left
        elif attrn == "calendar_right":
            return self.calendar_left + self.col.width
        elif attrn == "window_right":
            return self.window_left + self.col.width
        elif attrn == "in_window":
            return (
                self.y > 0 and
                self.y < self.window.height and
                self.window_right > 0 and
                self.window_left < self.window.width)
        else:
            raise AttributeError(
                "Timeline instance has no attribute " + attrn)

    def delete(self):
        try:
            self.vertlist.delete()
        except:
            pass
        self.vertlist = None
        self.handle.delete()

    def draw(self):
        colors = self.color * 2
        points = (
            self.window_left, self.y,
            self.window_right, self.y)
        try:
            self.vertlist.vertices = list(points)
            self.vertlist.colors = list(colors)
        except AttributeError:
            assert(self.vertlist is None)
            self.vertlist = self.batch.add(
                2,
                GL_LINES,
                self.col.tlgroup,
                ('v2i', points),
                ('c4B', colors))
        self.handle.draw()


class CalendarCell:
    """A block of time in a calendar.

Uses information from the CalendarCol it's in and the Event it
represents to calculate its dimensions and coordinates.

    """
    def __init__(self, col, tick_from, tick_to, text):
        self.col = col
        self.cal = self.col.calendar
        self.batch = self.col.batch
        self.bggroup = OrderedGroup(0, self.col.cellgroup)
        self.textgroup = OrderedGroup(1, self.col.cellgroup)
        self.tick_from = tick_from
        self.tick_to = tick_to
        self.text = text
        self.style = self.col.style
        self.visible = True

    def __len__(self):
        if self.tick_to is None:
            r = self.cal.height - self.tick_from
        else:
            r = self.tick_to - self.tick_from
        if r < 0:
            return 0
        else:
            return r

    def __getattr__(self, attrn):
        if attrn == 'interactive':
            return self.col.calendar.interactive
        elif attrn == 'window':
            return self.col.calendar.window
        elif attrn == 'calendar_left':
            return self.col.calendar_left + self.style.spacing
        elif attrn == 'calendar_right':
            return self.col.calendar_right - self.style.spacing
        elif attrn == 'calendar_top':
            return self.cal.height - self.cal.row_height * (
                self.tick_from - self.cal.scrolled_to) - self.style.spacing
        elif attrn == 'calendar_bot':
            try:
                return self.cal.height - self.cal.row_height * (
                    self.tick_to - self.cal.scrolled_to)
            except TypeError:
                return 0
        elif attrn == 'width':
            return self.calendar_right - self.calendar_left
        elif attrn == 'height':
            return len(self) * self.cal.row_height
        elif attrn == 'window_left':
            return self.calendar_left + self.cal.window_left
        elif attrn == 'window_right':
            return self.calendar_right + self.cal.window_left
        elif attrn == 'window_top':
            return self.calendar_top + self.cal.window_bot
        elif attrn == 'window_bot':
            return self.calendar_bot + self.cal.window_bot
        elif attrn == 'label_height':
            return self.style.fontsize + self.style.spacing
        elif attrn == 'hovered':
            return self is self.window.hovered
        elif attrn == 'vertlist':
            return self.col.vertldict[str(self)]
        elif attrn == 'label':
            return self.col.labeldict[str(self)]
        else:
            raise AttributeError(
                "CalendarCell instance has no such attribute: " +
                attrn)

    def __setattr__(self, attrn, value):
        if attrn == 'vertlist':
            self.col.vertldict[str(self)] = value
        elif attrn == 'label':
            self.col.labeldict[str(self)] = value
        else:
            super(CalendarCell, self).__setattr__(attrn, value)

    def __hash__(self):
        return hash((self.tick_from, self.tick_to, self.text))

    def __str__(self):
        return "{0} from {1} to {2}".format(self.text, self.tick_from, self.tick_to)

    def delete(self):
        try:
            self.label.delete()
        except:
            pass
        self.label = None
        try:
            self.vertlist.delete()
        except:
            pass
        self.vertlist = None

    def draw(self):
        colors = (0, 0, 0, 255) * 4
        l = self.window_left
        r = self.window_right
        t = self.window_top
        b = self.window_bot
        points = (
            l, b,
            l, t,
            r, t,
            r, b)
        try:
            self.vertlist.vertices = list(points)
        except KeyError:
            self.vertlist = self.batch.add_indexed(
                4,
                GL_LINES,
                self.bggroup,
                (0, 1, 1, 2, 2, 3, 3, 0),
                ('v2i', points),
                ('c4B', colors))
        y = self.calendar_top - self.label_height
        try:
            self.label.x = self.window_left
            self.label.y = y
        except:
            self.label = Label(
                self.text,
                self.style.fontface,
                self.style.fontsize,
                color=self.style.textcolor.tup,
                width=self.width,
                height=self.height,
                x=self.window_left,
                y=y,
                multiline=True,
                batch=self.batch,
                group=self.textgroup)


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
        
    visible = True

    def __init__(self, window, idx):
        self.window = window
        self.rumor = self.window.rumor
        self.idx = idx
        self.batch = self.window.batch
        self.group = self.window.calgroup
        self.cols = []
        self.add_thing_cols()
        self.old_state = None

    def __iter__(self):
        return iter(self.cols)

    def __len__(self):
        return len(self.cols)

    def __int__(self):
        return self.idx

    def __getattr__(self, attrn):
        if attrn == "_rowdict":
            return self.rumor.tabdict["calendar"][str(self.window)][int(self)]
        elif attrn in (
                "interactive",
                "rows_shown",
                "scroll_factor",
                "max_cols"):
            return self._rowdict[attrn]
        elif attrn == "typ":
            return self._rowdict["type"]
        elif attrn == "character":
            return self.rumor.characterdict[self._rowdict["character"]]
        elif attrn == "left_prop":
            return self._rowdict["left"]
        elif attrn == "right_prop":
            return self._rowdict["right"]
        elif attrn == "top_prop":
            return self._rowdict["top"]
        elif attrn == "bot_prop":
            return self._rowdict["bot"]
        elif attrn in ("scrolled_to", "top_tick"):
            r = self._rowdict["scrolled_to"]
            if r is None:
                return self.rumor.tick
            else:
                return r
        elif attrn == "bot_tick":
            return self.top_tick + self.rows_shown
        elif attrn == "style":
            return self.rumor.get_style(self._rowdict["style"])
        elif attrn == 'window_top':
            return int(self.top_prop * self.window.height)
        elif attrn == 'window_bot':
            return int(self.bot_prop * self.window.height)
        elif attrn == 'window_left':
            return int(self.left_prop * self.window.width)
        elif attrn == 'window_right':
            return int(self.right_prop * self.window.width)
        elif attrn == 'width':
            return self.window_right - self.window_left
        elif attrn == 'col_width':
            return self.width / len(self.cols)
        elif attrn == 'height':
            return self.window_top - self.window_bot
        elif attrn == 'row_height':
            return self.height / self.rows_shown
        elif attrn == 'visible':
            return self._visible and len(self.cols) > 0
        elif attrn == 'interactive':
            return self._interactive
        elif self.typ == CAL_TYPE['THING']:
            if attrn == 'thing_show_location':
                return self._rowdict['thing_show_location']
            elif attrn == 'location_dict' and self.thing_show_location:
                return self.rumor.tabdict[
                    "thing_location"][
                        self._rowdict["dimension"]][
                            self._rowdict["thing"]]
            elif attrn == 'coverage_dict':
                return self.rumor.tabdict[
                    "character_things"][
                        self._rowdict["character"]][
                            self._rowdict["dimension"]][
                                self._rowdict["thing"]]
        elif self.typ == CAL_TYPE['PLACE']:
            if attrn == 'coverage_dict':
                return self.rumor.tabdict[
                    "character_places"][
                        self._rowdict["character"]][
                            self._rowdict["dimension"]][
                                self._rowdict["place"]]
        elif self.typ == CAL_TYPE['PORTAL']:
            if attrn == 'coverage_dict':
                charn = self._rowdict["character"]
                dimn = self._rowdict["dimension"]
                orign = self._rowdict["origin"]
                destn = self._rowdict["destination"]
                return self.rumor.tabdict[
                    "character_portals"][charn][dimn][orign][destn]
        elif self.typ == CAL_TYPE['SKILL']:
            if attrn == 'coverage_dict':
                return self.rumor.tabdict[
                    "character_skills"][
                        self._rowdict["character"]][
                            self._rowdict["skill"]]
        elif self.typ == CAL_TYPE['STAT']:
            if attrn == 'coverage_dict':
                return self.rumor.tabdict[
                    "character_stats"][
                        self._rowdict["character"]][
                            self._rowdict["stat"]]
        else:
            raise AttributeError(
                "Calendar instance has no such attribute: " +
                attrn)

    def __getitem__(self, i):
        return self.cols[i]

    def __contains__(self, col):
        return col in self.cols

    def __int__(self):
        return self.idx

    def overlaps(self, x, y):
        return (
            self.visible and
            self.interactive and
            self.window_left < x and
            self.window_right > x and
            self.window_bot < y and
            self.window_top > y)

    def draw(self):
        if self.visible and len(self.cols) > 0:
            for calcol in self.cols:
                try:
                    (parent, start, end) = self.rumor.timestream.branchdict[
                        calcol.branch]
                except KeyError:
                    continue
                col1 = calcol
                col1.draw()
        else:
            for calcol in self.cols:
                calcol.delete()

    def remove(self, it):
        self.cols.remove(it)

    def refresh(self):
        for col1 in self.cols:
            (parent, tick_from, tick_to) = self.rumor.timestream.branchdict[
                col1.branch]
            if hasattr(col1, 'bc'):
                col1.bc.delete()
            col2 = None
            for calcol in self.cols:
                if calcol.branch == parent:
                    col2 = calcol
                    break
            if (
                    col2 is not None and
                    tick_from > self.scrolled_to and
                    tick_to < self.scrolled_to + self.rows_shown):
                col2.bc = BranchConnector(
                    self, col2, col1, tick_from)
                col2.refresh()
            col1.refresh()


class CalendarCol:
    def __init__(self, calendar, idx, branch):
        self.calendar = calendar
        self.rumor = self.calendar.rumor
        self.batch = self.calendar.batch
        self.style = self.calendar.style
        self.bggroup = OrderedGroup(0, self.calendar.group)
        self.cellgroup = OrderedGroup(1, self.calendar.group)
        self.tlgroup = OrderedGroup(2, self.calendar.group)
        self.bcgroup = OrderedGroup(3, self.calendar.group)
        self.timeline = Timeline(self)
        self.window = self.calendar.window
        self.idx = idx
        self.branch = branch
        self.old_schedule_hash = 0
        self.vertldict = {}
        self.labeldict = {}
        self.refresh()

    def __iter__(self):
        return self.cells

    def __getattr__(self, attrn):
        if attrn == 'width':
            return self.calendar.col_width
        elif attrn == 'rx':
            return self.width / 2
        elif attrn == 'height':
            return self.calendar.height
        elif attrn == 'ry':
            return self.height / 2
        elif attrn == 'calendar_left':
            return self.idx * self.width
        elif attrn == 'calendar_right':
            return self.calendar_left + self.width
        elif attrn == 'calendar_top':
            return self.calendar.height
        elif attrn == 'calendar_bot':
            return 0
        elif attrn == 'window_left':
            return self.calendar.window_left + self.calendar_left
        elif attrn == 'window_right':
            return self.calendar.window_left + self.calendar_right
        elif attrn == 'window_top':
            return self.calendar.window_top
        elif attrn == 'window_bot':
            return self.calendar.window_bot
        else:
            raise AttributeError(
                "CalendarCol instance has no such attribute: " +
                attrn)

    def __int__(self):
        return self.idx

    def delete(self):
        for cell in self._cells:
            cell.delete()
        try:
            self.timeline.delete()
        except:
            pass
        self.timeline = None

    def pretty_caster(self, *args):
        unargs = []
        for arg in args:
            if isinstance(arg, tuple) or isinstance(arg, list):
                unargs += self.pretty_caster(*arg)
            else:
                unargs.append(arg)
        return unargs

    def pretty_printer(self, *args):
        strings = []
        unargs = self.pretty_caster(*args)
        for unarg in unargs:
            strings.append(str(unarg))
        return ";\n".join(strings)

    def draw(self):
        colors = self.style.bg_inactive.tup * 4
        l = self.window_left
        r = self.window_right
        t = self.window_top
        b = self.window_bot
        points = (
            l, b,
            l, t,
            r, t,
            r, b)
        try:
            self.vertl['bg'] = list(points)
        except KeyError:
            self.vertl['bg'] = self.batch.add_indexed(
                4,
                GL_TRIANGLES,
                self.bggroup,
                [0, 2, 3, 0, 1, 2],
                ('v2i', points),
                ('c4B', colors))
        for cel in self._cells:
            cel.draw()
        if hasattr(self, 'bc'):
            self.bc.draw()
        if (
                self.rumor.branch == self.branch and
                self.timeline.in_window):
            self.timeline.draw()
        else:
            self.timeline.delete()

class LocationCalendarCellIterator:
    def __init__(self, loccol):
        self.col = loccol
        self.lociter = TabdictIterator(self.col.locations)

    def __iter__(self):
        return self

    def next(self):
        rd = self.lociter.next()
        when = self.col.shows_when(rd["tick_from"], rd["tick_to"])
        while (
                when is None or
                when[1] < self.col.calendar.top_tick or
                when[0] > self.col.calendar.bot_tick):
            rd = self.lociter.next()
            when = self.col.shows_when(rd["tick_from"], rd["tick_to"])
        (a, b) = when
        if a < self.col.calendar.top_tick:
            a = self.col.calendar.top_tick
        if b > self.col.calendar.bot_tick:
            b = self.col.calendar.bot_tick
        return CalendarCell(
            self.col, a, b,
            # TODO: if the display name of the place changes while the
            # thing is there, split the calendar cell for each of
            # those names
            self.col.rumor.get_place(
                rd["dimension"], rd["location"]).display_name(
                    rd["branch"], rd["tick_from"]))

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
    def __getattr__(self, attrn):
        winn = str(self.calendar.window)
        cali = int(self.calendar)
        charn = str(self.calendar.character)
        dimn = str(self.calendar.dimension)
        thingn = str(self.calendar.thing)
        if attrn == "_rowdict":
            return self.rumor.tabdict["thing_cal"][winn][cali]
        elif attrn in (
                "character",
                "dimension",
                "thing"):
            return getattr(self.calendar, attrn)
        elif attrn == "typ":
            return COL_TYPE['THING']
        elif attrn == "location":
            return True
        elif attrn == "locations":
            return self.thing.locations[self.branch]
        elif attrn == "coverage":
            return self.character.thingdict[self.branch][dimn][thingn]
        elif attrn == "_cells":
            return LocationCalendarCellIterator(self)
        else:
            raise AttributeError(
                "LocationCalendarCol instance has no attribute named {0}".format(
                    attrn))

        def shows_any_ever(self, tick_from, tick_to):
            for (cover_tick_from, cover_tick_to) in self.coverage.iteritems():
                if tick_to > cover_tick_from or tick_from < cover_tick_to:
                    return True
            return False

        def shows_when(self, tick_from, tick_to):
            for (cover_tick_from, cover_tick_to) in self.coverage.iteritems():
                if tick_to > cover_tick_from and tick_from < cover_tick_to:
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
                        
