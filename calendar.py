# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import (
    SaveableMetaclass,
    ViewportOrderedGroup,
    TabdictIterator,
    phi)
from pyglet.image import SolidColorImagePattern as color_pattern
from pyglet.sprite import Sprite
from pyglet.text import Label
from pyglet.graphics import GL_LINES, GL_TRIANGLES, OrderedGroup

"""User's view on a given item's schedule.

Usually there should be only one calendar per board, but it can switch
between showing various schedules, or even show many in parallel.

"""


__metaclass__ = SaveableMetaclass


class Timeline:
    """A line that goes on top of a CalendarCol to indicate what time it
is.

Also has a little handle that you can drag to do the time warp. Select
which side it should be on by supplying "left" (default) or "right"
for the handle_side keyword argument.

    """
    color = (255, 0, 0, 255)

    class Handle:
        """The handle widget."""
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

    def __init__(self, col, handle_side="left"):
        self.col = col
        self.cal = self.col.calendar
        self.batch = self.col.batch
        self.window = self.cal.window
        self.rumor = self.col.rumor
        self.handle = Timeline.Handle(self, handle_side)
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
            print "new vertlist for timeline for CalendarCol {0}".format(int(self.col))
            print "points: " + repr(points)
            print "colors: " + repr(colors)
            print "group ID: " + str(id(self.col.tlgroup))
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
        self.oldstate = None
        self.was_hovered = False
        self.old_width = None
        self.old_height = None
        self.old_active_image = None
        self.old_inactive_image = None
        self.vertlist = None
        self.label = None
        self.visible = True
        self.tweaks = 0
        self.inactive_pattern = color_pattern(self.style.fg_inactive.tup)
        self.active_pattern = color_pattern(self.style.fg_active.tup)

    def __len__(self):
        if self.tick_to is None:
            return self.cal.height - self.tick_from
        else:
            return self.tick_to - self.tick_from

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
        else:
            raise AttributeError(
                "CalendarCell instance has no such attribute: " +
                attrn)

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
         except AttributeError:
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


COL_TYPE = {
    "THING": 0,
    "STAT": 1,
    "SKILL": 2}


class Calendar:
    """A collection of calendar columns representing at least one
schedule, possibly several.

"""
    # There should be only one character-attribute for one
    # calendar_col, although its record may be in any of several
    # tables.
    tables = [
        (
            "calendar",
            {"window": "text not null default 'Main'",
             "idx": "integer not null default 0",
             "left": "float not null default 0.8",
             "right": "float not null default 1.0",
             "top": "float not null default 1.0",
             "bot": "float not null default 0.0",
             "max_cols": "integer not null default 3",
             "style": "text not null default 'default_style'",
             "interactive": "boolean not null default 1",
             "rows_shown": "integer not null default 240",
             "scrolled_to": "integer default null",
             "scroll_factor": "integer not null default 4"},
            ("window", "idx"),
            {"window": ("window", "name"),
             "style": ("style", "name")},
            ["rows_shown>0", "left>=0.0", "left<=1.0", "right<=1.0",
             "left<right", "top>=0.0", "top<=1.0", "bot>=0.0",
             "bot<=1.0", "top>bot", "idx>=0"]),
        (
            "calendar_col",
            {"window": "text not null default 'Main'",
             "calendar": "integer not null default 0",
             "idx": "integer not null",
             "type": "integer not null"},
            ("window", "calendar", "idx"),
            {"window, calendar": ("calendar", "window, idx")},
            ["idx>=0", "type IN ({0})".format(
                ", ".join([str(it) for it in COL_TYPE.values()])
            )])]
    visible = True

    class BranchConnector:
        """Widget to show where a branch branches off another.

        It's an arrow that leads from the tick on one branch where
        another branches from it, to the start of the latter.

        It operates on the assumption that child branches will always
        be displayed next to their parents, when they are displayed at
        all.

        """
        color = (255, 0, 0, 255)
        class Wedge:
            """Downward pointing wedge, looks much like the Timeline's Handle"""
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
                        "Wedge instance has no attribute named {0}".format(attrn))

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
                        (0,1,2,0),
                        ('v2i', points),
                        ('c4B', colors))

            def delete(self):
                try:
                    self.vertlist.delete()
                except:
                    pass
                self.vertlist = None

        def __init__(self, calendar, col1, col2, tick):
            self.calendar = calendar
            self.batch = self.calendar.batch
            self.group = OrderedGroup(3, self.calendar.group)
            self.linegroup = OrderedGroup(0, self.group)
            self.wedgegroup = OrderedGroup(1, self.group)
            self.col1 = col1
            self.col2 = col2
            self.tick = tick
            self.wedge = self.__class__.Wedge(self)
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
                    return self.col1.window_right + self.calendar.style.spacing / 2
                else:
                    return self.col2.window_right + self.calendar.style.spacing / 2
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

    def __init__(self, rumor, td):
        self.rumor = rumor
        self._tabdict = td
        rd = self._tabdict["calendar"]
        self.window = self.rumor.get_window(rd["window"])
        self.batch = self.window.batch
        self.group = self.window.calgroup
        self.idx = rd["idx"]
        self.left_prop = rd["left"]
        self.right_prop = rd["right"]
        self.top_prop = rd["top"]
        self.bot_prop = rd["bot"]
        self.interactive = rd["interactive"]
        self.rows_shown = rd["rows_shown"]
        self.style = self.rumor.get_style(rd["style"])
        scrolled_to = rd["scrolled_to"]
        if scrolled_to is None:
            scrolled_to = self.rumor.tick
        self.scrolled_to = scrolled_to
        self.scroll_factor = rd["scroll_factor"]
        self.max_cols = rd["max_cols"]
        self.sprite = None
        self.cols = []
        self.add_cols_from_tabdict(td)

    def __iter__(self):
        return iter(self.cols)

    def __len__(self):
        return len(self.cols)

    def __getattr__(self, attrn):
        if attrn == 'window_top':
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

    def get_tabdict(self):
        return {
            "calendar": [
                {
                    "window": str(self.window),
                    "i": self.i,
                    "left": self.left_prop,
                    "right": self.right_prop,
                    "top": self.top_prop,
                    "bot": self.bot_prop,
                    "style": str(self.style),
                    "interactive": self.interactive,
                    "rows_shown": self.rows_shown,
                    "scrolled_to": self.scrolled_to,
                    "scroll_factor": self.scroll_factor}]}

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
                    (parent, start, end) = self.rumor.timestream.branchdict[calcol.branch]
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
            (parent, tick_from, tick_to) = self.rumor.timestream.branchdict[col1.branch]
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
                col1.bc = Calendar.BranchConnector(
                    self, col2, col1, tick_from)
            col1.regen_cells()

    def add_cols_from_tabdict(self, td):
        for rd in TabdictIterator(td):
            if "thing" in rd:
                self.cols.append(CalendarCol(
                    self,
                    rd["character"],
                    COL_TYPE["THING"],
                    rd["idx"],
                    td))


class CalendarCol:
    prelude = [
        ("CREATE VIEW ccstat AS SELECT {0} FROM calendar_col WHERE type={1}".format(
            ", ".join(Calendar.colnames["calendar_col"]),
            COL_TYPE["STAT"])),
        ("CREATE VIEW ccthing AS SELECT {0} FROM calendar_col WHERE type={1}".format(
            ", ".join(Calendar.colnames["calendar_col"]),
            COL_TYPE["THING"])),
        ("CREATE VIEW ccskill AS SELECT {0} FROM calendar_col WHERE type={1}".format(
            ", ".join(Calendar.colnames["calendar_col"]),
            COL_TYPE["SKILL"]))]
    postlude = [
        ("CREATE TRIGGER unical_thing BEFORE INSERT ON calendar_col_thing BEGIN "
         "INSERT INTO calendar_col (window, calendar, idx, type) VALUES "
         "(NEW.window, NEW.calendar, NEW.idx, {0}); "
         "END".format(COL_TYPE["THING"])),
        ("CREATE TRIGGER unical_stat BEFORE INSERT ON calendar_col_stat BEGIN "
         "INSERT INTO calendar_col (window, calendar, idx, type) VALUES "
         "(NEW.window, NEW.calendar, NEW.idx, {0}); "
         "END".format(COL_TYPE["STAT"])),
        ("CREATE TRIGGER unical_skill BEFORE INSERT ON calendar_col_skill BEGIN "
         "INSERT INTO calendar_col (window, calendar, idx, type) VALUES "
         "(NEW.window, NEW.calendar, NEW.idx, {0}); "
         "END".format(COL_TYPE["SKILL"]))]
    tables = [
        (
            "calendar_col_thing",
            {"window": "text not null default 'Main'",
             "calendar": "integer not null default 0",
             "idx": "integer not null",
             "branch": "integer not null default 0",
             "character": "text not null",
             "dimension": "text not null",
             "thing": "text not null",
             "location": "boolean default 1"},
            ("window", "calendar", "idx"),
            {"window, calendar, idx":
             ("ccthing", "window, calendar, idx", "ON DELETE CASCADE"),
             "character, dimension, branch, thing":
             ("character_things", "character, dimension, branch, thing")},
            ["idx>=0"]),
        (
            "calendar_col_stat",
            {"window": "text not null default 'Main'",
             "calendar": "integer not null default 0",
             "idx": "integer not null",
             "branch": "integer not null default 0",
             "character": "text not null",
             "stat": "text not null"},
            ("window", "calendar", "idx"),
            {"window, calendar, idx":
             ("ccstat", "window, calendar, idx", "ON DELETE CASCADE"),
             "character, branch, stat": ("character_skills", "character, branch, stat")},
            ["idx>=0"]),
        (
            "calendar_col_skill",
            {"window": "text not null default 'Main'",
             "calendar": "integer not null default 0",
             "idx": "integer not null",
             "branch": "integer not null default 0",
             "character": "text not null",
             "skill": "text not null"},
            ("window", "calendar", "idx"),
            {"window, calendar, idx":
             ("ccskill", "window, calendar, idx", "ON DELETE CASCADE"),
             "character, branch, skill": ("character_skills", "character, branch, skill")},
            ["idx>=0"])]
    def __init__(self, calendar, character, typ, idx, td):
        self.calendar = calendar
        self.rumor = self.calendar.rumor
        self.batch = self.calendar.batch
        self.style = self.calendar.style
        self.bggroup = OrderedGroup(0, self.calendar.group)
        self.cellgroup = OrderedGroup(1, self.calendar.group)
        self.tlgroup = OrderedGroup(2, self.calendar.group)
        self.timeline = Timeline(self)
        self.window = self.calendar.window
        self.character = self.rumor.get_character(character)
        self.idx = idx
        self.typ = typ
        if self.typ == COL_TYPE["THING"]:
            self._rowdict = td["calendar_col_thing"][str(self.window)][int(self.calendar)][int(self)]
            if self._rowdict["location"]:
                self.get_locations = lambda: self.rumor.get_thing(
                    self._rowdict["dimension"], self._rowdict["thing"]).locations[self.branch]
        elif self.typ == COL_TYPE["STAT"]:
            self._rowdict = td["calendar_col_stat"][str(self.window)][int(self.calendar)][int(self)]
        else:
            self._rowdict = td["calendar_col_skill"][str(self.window)][int(self.calendar)][int(self)]
        self.vertl = None
        self.cells = []
        self.refresh()

    def __iter__(self):
        return iter(self.cells)

    def __getattr__(self, attrn):
        if attrn == 'dimension' and self.typ == COL_TYPE["THING"]:
            return self.rumor.get_dimension(self._dimension)
        elif attrn in self._rowdict:
            return self._rowdict[attrn]
        elif hasattr(self.character, attrn):
            return getattr(self.character, attrn)
        elif attrn == 'width':
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

    def refresh(self):
        self.regen_cells()

    def regen_cells(self):
        for cell in self.cells:
            cell.delete()
        self.cells = []
        if self.typ == COL_TYPE["THING"]:
            if hasattr(self, 'get_locations'):
                scheduledict = self.get_locations()
                # Eliminate those entries when I am not the thing;
                # truncate those when I spend only part of it being
                # the thing
            else:
                scheduledict = self.thingdict
        elif self.typ == COL_TYPE["STAT"]:
            scheduledict = self.statdict
        for (tick_from, val) in scheduledict.iteritems():
            if isinstance(val, tuple):
                tick_to = val[-1]
                text = self.pretty_printer(val[:-1])
            else:
                tick_to = val
                text = "..."
            self.cells.append(CalendarCell(self, tick_from, tick_to, text))

    def delete(self):
        for cell in self.cells:
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
            self.vertl.vertices = list(points)
        except AttributeError:
            self.vertl = self.batch.add_indexed(
                4,
                GL_TRIANGLES,
                self.bggroup,
                [0, 2, 3, 0, 1, 2],
                ('v2i', points),
                ('c4B', colors))
        for cel in self.cells:
            cel.draw()
        if hasattr(self, 'bc'):
            self.bc.draw()
        if (
                self.rumor.branch == self.branch and
                self.timeline.in_window):
            self.timeline.draw()
        else:
            self.timeline.delete()


