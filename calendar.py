# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass, ViewportOrderedGroup
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
        def __init__(self, timeline, handle_side, width=10, height=16):
            self.timeline = timeline
            self.on_the_left = handle_side == "left"
            self.vertlist = None
            self.width = width
            self.rx = self.width / 2
            self.height = height
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

        def draw(self):
            batch = self.timeline.batch
            group = self.timeline.handlegroup
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
        self.tlgroup = OrderedGroup(0, self.col.tlgroup)
        self.handlegroup = OrderedGroup(1, self.col.tlgroup)
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
        else:
            raise AttributeError(
                "Timeline instance has no attribute " + attrn)

    def delete(self):
        try:
            self.vertlist.delete()
        except:
            pass
        try:
            self.handle.delete()
        except:
            pass

    def draw(self):
        colors = self.color * 2
        points = (
            self.window_left, self.y,
            self.window_right, self.y)
        try:
            self.vertlist.vertices = list(points)
            self.vertlist.colors = list(colors)
        except AttributeError:
            self.vertlist = self.batch.add(
                2,
                GL_LINES,
                self.tlgroup,
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
        try:
            self.vertlist.delete()
        except:
            pass

    def draw(self):
        if self.visible and self.height > 0:
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
                    batch=batch,
                    group=self.textgroup)
        else:
            self.delete()


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
        class Wedge:
            """Downward pointing wedge, looks much like the Timeline's Handle"""
            def __init__(self, bc, width=16, height=10):
                self.bc = bc
                self.batch = self.bc.batch
                self.group = self.bg.wedgegroup
                self.width = width
                self.rx = self.width / 2
                self.height = height
                self.ry = self.height / 2
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

        def __init__(self, calendar, col1, tick1, col2, tick2, space=None):
            self.calendar = calendar
            self.batch = self.calendar.batch
            self.group = OrderedGroup(3, self.calendar.group)
            self.linegroup = OrderedGroup(0, self.group)
            self.wedgegroup = Orderedgroup(1, self.group)
            self.col1 = col1
            self.tick1 = tick1
            self.col2 = col2
            self.tick2 = tick2
            if space is None:
                self.space = self.calendar.style.spacing
            else:
                self.space = space

        def __getattr__(self, attrn):
            if attrn == "startx":
                return self.col1.window_left + self.col1.rx
            elif attrn == "endx":
                return self.col2.window_left + self.col2.rx
            elif attrn == "starty":
                return (
                    self.col1.window_top -
                    self.calendar.row_height * (
                        self.tick1 - self.calendar.scrolled_to))
            elif attrn == "endy":
                return (
                    self.col2.window_top -
                    self.calendar.row_height * (
                        self.tick2 - self.calendar.scrolled_to))
            elif attrn == "centerx":
                if self.col1.window_x < self.col2.window_x:
                    return self.col1.window_x + self.calendar.style.spacing / 2
                else:
                    return self.col2.window_x + self.calendar.style.spacing / 2
            elif attrn == "points":
                x0 = self.startx
                y0 = self.starty
                x2 = self.centerx
                x5 = self.endx
                y5 = self.endy
                return (
                    x0, y0,
                    x0, y0 + self.space,
                    x2, y0 + self.space,
                    x2, y5 + self.space,
                    x5, y5 + self.space,
                    x5, y5)
            elif attrn == "start":
                return (
                    self.startx,
                    self.starty)
            elif attrn == "end":
                return (
                    self.endx,
                    self.endy)

    def __init__(self, rumor, td):
        self.rumor = rumor
        rd = td["calendar"]
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
        elif attrn =

        = 'window_bot':
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
        return self.i

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
                calcol.draw()
        else:
            for calcol in self.cols:
                calcol.delete()

    def remove(self, it):
        self.cols.remove(it)

    def refresh(self):
        for col in self.cols:
            col.regen_cells()

    def add_col_from_tabdict(self, td):
        self.cols.append(CalendarCol(self, td))

    def add_cols_from_tabdict(self, td):
        if "calendar_col_thing" in td:
            for rd in td["calendar_col_thing"]:
                self.add_col_from_tabdict(
                    {"calendar_col_thing": rd})
        if "calendar_col_stat" in td:
            for rd in td["calendar_col_stat"]:
                self.add_col_from_tabdict(
                    {"calendar_col_stat": rd})
        if "calendar_col_skill" in td:
            for rd in td["calendar_col_skill"]:
                self.add_col_from_tabdict(
                    {"calendar_col_skill": rd})


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
         "(NEW.window, NEW.calendar, NEW.idx, ?); "
         "END",
         (COL_TYPE["THING"],)),
        ("CREATE TRIGGER unical_stat BEFORE INSERT ON calendar_col_stat BEGIN "
         "INSERT INTO calendar_col (window, calendar, idx, type) VALUES "
         "(NEW.window, NEW.calendar, NEW.idx, ?); "
         "END",
         (COL_TYPE["STAT"],)),
        ("CREATE TRIGGER unical_skill BEFORE INSERT ON calendar_col_skill BEGIN "
         "INSERT INTO calendar_col (window, calendar, idx, type) VALUES "
         "(NEW.window, NEW.calendar, NEW.idx, ?); "
         "END",
         (COL_TYPE["SKILL"],))]
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
    def __init__(self, calendar, td):
        self.calendar = calendar
        self.rumor = self.calendar.rumor
        self.batch = self.calendar.batch
        self.bggroup = OrderedGroup(0, self.calendar.group)
        self.cellgroup = OrderedGroup(1, self.calendar.group)
        self.tlgroup = OrderedGroup(2, self.calendar.group)
        self.timeline = Timeline(self)
        self.window = self.calendar.window
        if "calendar_col_thing" in td:
            rd = td["calendar_col_thing"]
            def is_thing(self, tick=None):
                return self.character.is_thing_with_strs(
                    rd["dimension"], rd["thing"], self.branch, tick)
            self.is_thing = is_thing
            def get_thing(self, tick=None):
                if self.is_thing(branch, tick):
                    return self.rumor.get_thing(
                        rd["dimension"], rd["thing"], self.branch, tick)
                else:
                    return None
            self.get_thing = get_thing
            self.thingdict = self.character.thingdict[self.branch]
            if rd["location"]:
                def glocs(self, tick=None):
                    return self.get_thing(tick).locations[self.branch]
                self.get_locations = gloc
        elif "calendar_col_stat" in td:
            rd = td["calendar_col_stat"]
            self.statdict = self.character.statdict[rd["stat"]]
        else:
            rd = td["calendar_col_skill"]
            self.skilldict = self.character.skilldict[rd["skill"]]
        self.character = self.rumor.get_character(rd["character"])
        self.branch = rd["branch"]
        self.vertl = None
        self.cells = []
        self.regen_cells()

    def __iter__(self):
        return iter(self.cells)

    def __getattr__(self, attrn):
        if attrn == 'dimension':
            return self.rumor.get_dimension(self._dimension)
        elif attrn == 'idx':
            return self.calendar.cols.index(self)
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

    def regen_cells(self, branch=None):
        for cell in self.cells:
            cell.delete()
        self.cells = []
        if hasattr(self, 'is_thing'):
            if hasattr(self, 'get_location'):
                scheduledict = self.get_locations()
                # Eliminate those entries when I am not the thing;
                # truncate those when I spend only part of it being
                # the thing
            else:
                scheduledict = self.thingdict
        elif hasattr(self, 'statdict'):
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
            self.sprite.delete()
        except:
            pass
        try:
            self.timeline.delete()
        except:
            pass
        self.sprite = None
        self.timeline = None
        self.calendar.remove(self)

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
        if self.rumor.branch == self.branch:
            self.timeline.draw()
        else:
            self.timeline.delete()


