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
        batch = self.batch
        if self.visible and self.height > 0:
            colors = (0, 0, 0, 255) * 4
            points = (
                self.window_left, self.window_bot,
                self.window_left, self.window_top,
                self.window_right, self.window_top,
                self.window_right, self.window_bot)
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


class CalendarCol:
    def __init__(self, calendar, scheduledict, branch, interactive, style):
        self.calendar = calendar
        self.rumor = self.calendar.rumor
        self.batch = self.calendar.batch
        self.bggroup = OrderedGroup(0, self.calendar.group)
        self.cellgroup = OrderedGroup(1, self.calendar.group)
        self.tlgroup = OrderedGroup(2, self.calendar.group)
        self.timeline = Timeline(self)
        self.window = calendar.window
        self.scheduledict = scheduledict
        self.branch = branch
        self.visible = False
        self.interactive = interactive
        self.style = style
        self.oldstate = None
        self.old_width = None
        self.old_image = None
        self.sprite = None
        self.oldwidth = None
        self.inactive_pattern = color_pattern(style.bg_inactive.tup)
        self.active_pattern = color_pattern(style.bg_active.tup)
        self.tweaks = 0
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
        elif attrn == 'height':
            return self.calendar.height
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

    def celhash(self):
        hashes = [
            hash(cel.get_state_tup())
            for cel in self.cells]
        return hash(tuple(hashes))

    def get_state_tup(self):
        """Return a tuple containing information enough to tell the difference
between any two states that should appear different on-screen."""
        return (
            self.celhash(),
            self.window_top,
            self.window_bot,
            self.idx,
            self.width,
            self.visible,
            self.interactive,
            self.tweaks)

    def regen_cells(self, branch=None):
        for cell in self.cells:
            cell.delete()
        self.cells = []
        if branch is None:
            branch = self.rumor.branch
        print "generating calendar cells from scheduledict:"
        print self.scheduledict[branch]
        for (tick_from, val) in self.scheduledict[branch].iteritems():
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
        if (
                not hasattr(self, 'image') or
                self.image.width != self.width or
                self.image.height != self.height):
            self.image = self.inactive_pattern.create_image(
                self.width, self.height)
        try:
            assert(self.sprite.width == self.width)
            assert(self.sprite.height == self.height)
            self.sprite.x = self.window_left
            self.sprite.y = self.window_bot
        except:
            self.sprite = Sprite(
                self.image,
                self.window_left,
                self.window_bot,
                batch=self.batch,
                group=self.bggroup)
        for cel in self.cells:
            cel.draw()
        if self.rumor.branch == self.branch:
            self.timeline.draw()
        else:
            self.timeline.delete()


COL_TYPE = {
    "THING": 0,
    "STAT": 1,
    "SKILL": 2}

class Calendar:
    """A collection of calendar columns representing at least one
schedule, possibly several.

"""
    __metaclass__ = SaveableMetaclass
    # There should be only one character-attribute for one
    # calendar_col, although its record may be in any of several
    # tables.
    postlude = [
        "CREATE TRIGGER unical_thing BEFORE INSERT ON calendar_col_thing BEGIN "
        "INSERT INTO calendar_col (window, calendar, idx, type) VALUES "
        "(NEW.window, NEW.calendar, NEW.idx, {0});"
        "END".format(COL_TYPE["THING"]),
        "CREATE TRIGGER unical_stat BEFORE INSERT ON calendar_col_stat BEGIN "
        "INSERT INTO calendar_col (window, calendar, idx, type) VALUES "
        "(NEW.window, NEW.calendar, NEW.idx, {0});"
        "END".format(COL_TYPE["STAT"]),
        "CREATE TRIGGER unical_skill BEFORE INSERT ON calendar_col_skill BEGIN "
        "INSERT INTO calendar_col (window, calendar, idx, type) VALUES "
        "(NEW.window, NEW.calendar, NEW.idx, {0});"
        "END".format(COL_TYPE["SKILL"])]
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
            )])
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
            {"window, calendar, idx, {0}".format(COL_TYPE["THING"]):
             ("calendar_col", "window, calendar, idx, type", "ON DELETE CASCADE"),
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
            {"window, calendar, idx, {0}".format(COL_TYPE["STAT"]):
             ("calendar_col", "window, calendar, idx, type", "ON DELETE CASCADE"),
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
            {"window, calendar, idx, {0}".format(COL_TYPE["SKILL"]):
             ("calendar_col", "window, calendar, idx, type", "ON DELETE CASCADE"),
             "character, branch, skill": ("character_skills", "character, branch, skill")},
            ["idx>=0"])]
    visible = True

    def __init__(
            self, window, idx, left, right, top, bot, style, interactive,
            rows_shown, scrolled_to, scroll_factor, max_cols, branches=None):
        self.window = window
        self.batch = self.window.batch
        self.group = self.window.calgroup
        self.rumor = self.window.rumor
        self.idx = idx
        self.left_prop = left
        self.right_prop = right
        self.top_prop = top
        self.bot_prop = bot
        self.interactive = interactive
        self.rows_shown = rows_shown
        self.style = style
        if scrolled_to is None:
            scrolled_to = self.rumor.tick
        self.scrolled_to = scrolled_to
        self.scroll_factor = scroll_factor
        self.max_cols = max_cols
        if branches is not None:
            self.branches = branches
        self.oldstate = None
        self.sprite = None
        self.tweaks = 0
        self.cols = []
        self.timeline = None

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
        return self.i

    def colhash(self):
        hashes = [
            hash(col.get_state_tup())
            for col in self.cols]
        return hash(tuple(hashes))

    def get_state_tup(self):
        """Return a tuple containing information enough to tell the difference
between any two states that should appear different on-screen."""
        return (
            self.i,
            self.colhash(),
            self.window_left,
            self.window_right,
            self.window_top,
            self.window_bot,
            self.visible,
            self.interactive,
            self.rows_shown,
            self.scrolled_to,
            self.rumor.tick,
            self.tweaks)

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

    def mkcol(self, scheduledict, branch, interactive=True, style=None):
        if style is None:
            style = self.style
        cc = CalendarCol(
            self, scheduledict, branch, interactive, style)
        self.cols.append(cc)
        return cc

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
