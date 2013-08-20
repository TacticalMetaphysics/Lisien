# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass, ViewportOrderedGroup
from pyglet.image import SolidColorImagePattern as color_pattern
from pyglet.sprite import Sprite
from pyglet.text import Label
from pyglet.graphics import GL_LINES, GL_TRIANGLES, OrderedGroup
from logging import getLogger

logger = getLogger(__name__)

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
            self.batch = self.timeline.batch
            self.group = self.timeline.group
            self.on_the_left = handle_side == "left"
            self.vertlist = None
            self.width = width
            self.rx = self.width / 2
            self.height = height
            self.ry = self.height / 2

        def __getattr__(self, attrn):
            if attrn == "calendar_left":
                if self.on_the_left:
                    return self.timeline.calendar_left - self.width
                else:
                    return self.timeline.calendar_right
            elif attrn == "calendar_right":
                if self.on_the_left:
                    return self.timeline.calendar_left
                else:
                    return self.timeline.calendar_right + self.width
            elif attrn == "calendar_top":
                return self.timeline.calendar_y + self.ry
            elif attrn == "calendar_bot":
                return self.timeline.calendar_y - self.ry
            elif attrn == "window_left":
                return self.calendar_left + self.timeline.cal.window_left
            elif attrn == "window_right":
                return self.calendar_right + self.timeline.cal.window_left
            elif attrn == "window_top":
                return self.calendar_top + self.timeline.cal.window_bot
            elif attrn == "window_bot":
                return self.calendar_bot + self.timeline.cal.window_bot
            else:
                raise AttributeError(
                    "Handle instance has no attribute " + attrn)

        def draw(self):
            colors = self.timeline.color * 3
            if self.on_the_left:
                points = (
                    self.calendar_right, self.timeline.calendar_y,
                    self.calendar_left, self.calendar_bot,
                    self.calendar_left, self.calendar_top)
            else:
                points = (
                    self.calendar_left, self.timeline.calendar_y,
                    self.calendar_right, self.calendar_top,
                    self.calendar_right, self.calendar_bot)
            logger.debug("drawing handle at %s", repr(points))
            try:
                self.vertlist.vertices = list(points)
                self.vertlist.colors = list(colors)
            except AttributeError:
                self.vertlist = self.batch.add_indexed(
                    3,
                    GL_TRIANGLES,
                    self.group,
                    (0, 1, 2, 0),
                    ('v2i', points),
                    ('c4b', colors))

    def __init__(self, col, handle_side="left"):
        self.col = col
        self.cal = self.col.calendar
        self.window = self.cal.window
        self.batch = self.window.batch
        self.group = self.col.tlgroup
        self.rumor = self.col.rumor
        self.handle = Timeline.Handle(self, handle_side)
        self.vertlist = None

    def __getattr__(self, attrn):
        if attrn in ("calendar_y", "calendar_bot", "calendar_top"):
            return self.cal.height - self.cal.row_height * (
                self.rumor.tick - self.cal.scrolled_to)
        elif attrn in ("window_y", "window_bot", "window_top"):
            return self.calendar_y + self.cal.window_bot
        elif attrn in ("calendar_left", "calendar_right",
                       "window_left", "window_right"):
            return getattr(self.col, attrn)
        else:
            raise AttributeError(
                "Timeline instance has no attribute " + attrn)

    def draw(self):
        colors = self.color * 2
        ws = self.cal.width_scalar
        hs = self.cal.height_scalar
        points = (
            self.calendar_left * ws, self.calendar_y * hs,
            self.calendar_right * ws, self.calendar_y * hs)
        logger.debug("drawing timeline at %s", repr(points))
        try:
            self.vertlist.vertices = list(points)
            self.vertlist.colors = list(colors)
        except AttributeError:
            self.vertlist = self.batch.add(
                2,
                GL_LINES,
                self.group,
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
        self.batch = self.cal.batch
        self.bggroup = OrderedGroup(0, self.col.cellgroup)
        self.fggroup = OrderedGroup(1, self.col.cellgroup)
        self.tick_from = tick_from
        self.tick_to = tick_to
        self.text = text
        self.style = self.col.style
        self.was_hovered = False
        self.verts = None
        self.label = None
        self.visible = True

    def __len__(self):
        if self.tick_to is None:
            return None
        else:
            return self.tick_to - self.tick_from

    def __getattr__(self, attrn):
        if attrn == 'interactive':
            return self.col.calendar.interactive
        elif attrn == 'window':
            return self.col.calendar.window
        elif attrn == 'calendar_left':
            return self.col.calendar_left + (
                self.style.spacing * self.cal.width_scalar)
        elif attrn == 'calendar_right':
            return self.col.calendar_right - (
                self.style.spacing * self.cal.width_scalar)
        elif attrn == 'calendar_top':
            return self.cal.height - self.cal.row_height * (
                self.tick_from - self.cal.scrolled_to)
        elif attrn == 'calendar_bot':
            try:
                return self.cal.height - self.cal.row_height * (
                    self.tick_to - self.cal.scrolled_to)
            except TypeError:
                return 0
        elif attrn == 'window_left':
            return self.col.window_left + self.style.spacing
        elif attrn == 'window_right':
            return self.window_left + self.col.width
        elif attrn in ('width', 'label_width'):
            return self.col.width
        elif attrn == 'height':
            try:
                return len(self) * self.cal.window_row_height
            except TypeError:
                return self.cal.height
        elif attrn == 'window_top':
            return self.col.window_top - self.cal.row_height * (
                self.tick_to - self.cal.scrolled_to)
        elif attrn == 'window_bot':
            return self.window_top - self.window_height
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
            self.sprite.delete()
        except:
            pass
        try:
            self.label.delete()
        except:
            pass

    def draw(self):
        ws = self.cal.width_scalar
        hs = self.cal.height_scalar
        if self.visible and self.height > 0:
            colors = self.style.fg_inactive.tup * 4
            pointt = (
                self.calendar_left * ws, self.calendar_bot * hs,
                self.calendar_left * ws, self.calendar_top * hs,
                self.calendar_right * ws, self.calendar_top * hs,
                self.calendar_right * ws, self.calendar_bot * hs)
            logger.debug("drawing calcell at %s", repr(pointt))
            pointl = list(pointt)
            try:
                self.verts.vertices = pointl
            except:
                self.verts = self.batch.add_indexed(
                    4,
                    GL_TRIANGLES,
                    self.bggroup,
                    (0, 1, 2, 0, 3, 2, 0),
                    ('v2i', pointt),
                    ('c4B', colors))
            y = self.calendar_top - self.label_height
            try:
                self.label.x = self.calendar_left * ws
                self.label.y = y * hs
                self.label.width = self.label_width * ws
                self.label.height = self.label_height * hs
            except:
                self.label = Label(
                    self.text,
                    self.style.fontface,
                    self.style.fontsize,
                    color=self.style.textcolor.tup,
                    multiline=True,
                    width=self.label_width,
                    height=self.label_height,
                    x=self.calendar_left * ws,
                    y=y * hs,
                    batch=self.batch,
                    group=self.fggroup)
        else:
            self.delete()


class CalendarCol:
    def __init__(self, calendar, branch, scheduledict, interactive, style):
        self.calendar = calendar
        self.branch = branch
        self.window = calendar.window
        self.batch = self.window.batch
        self.bggroup = OrderedGroup(0, self.calendar.supergroup)
        self.cellgroup = OrderedGroup(1, self.calendar.supergroup)
        self.tlgroup = OrderedGroup(2, self.calendar.supergroup)
        self.scheduledict = scheduledict
        self.rumor = self.calendar.rumor
        self.visible = False
        self.interactive = interactive
        self.style = style
        self.verts = None
        self.cells = []
        self.timeline = Timeline(self)
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
        elif attrn == 'bgcolor':
            return self.style.bg_inactive.tup
        elif attrn == 'calendar_left':
            return self.idx * self.width
        elif attrn == 'calendar_right':
            return self.calendar_left + self.width
        elif attrn == 'calendar_top':
            return self.calendar.supergroup.height
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
        elif attrn == 'width':
            return self.calendar.col_width
        else:
            raise AttributeError(
                "CalendarCol instance has no such attribute: " +
                attrn)

    def celhash(self):
        hashes = [
            hash(cel.get_state_tup())
            for cel in self.cells]
        return hash(tuple(hashes))
 
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
        self.sprite = None
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
        ws = self.calendar.width_scalar
        hs = self.calendar.height_scalar
        pointt = (
            self.calendar_left * ws, self.calendar_bot * hs,
            self.calendar_left * ws, self.calendar_top * hs,
            self.calendar_right * ws, self.calendar_top * hs,
            self.calendar_right * ws, self.calendar_bot * hs)
        logger.debug("drawing calcol at %s", repr(pointt))
        pointl = list(pointt)
        try:
            self.verts.vertices = pointl
        except:
            self.verts = self.batch.add_indexed(
                4,
                GL_TRIANGLES,
                self.bggroup,
                (0, 1, 2, 0, 3, 2, 0),
                ('v2i', pointt),
                ('c4B', self.bgcolor * 4))
        for cell in self.cells:
            cell.draw()


class Calendar:
    """A collection of calendar columns representing at least one
schedule, possibly several.

"""
    __metaclass__ = SaveableMetaclass
    tables = [
        (
            "calendar",
            {"window": "text not null default 'Main'",
             "idx": "integer not null default 0",
             "left": "float not null default 0.8",
             "right": "float not null default 1.0",
             "top": "float not null default 1.0",
             "bot": "float not null default 0.0",
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
             "bot<=1.0", "top>bot"])]
    visible = True

    def __init__(
            self, window, idx, left, right, top, bot, style, interactive,
            rows_shown, scrolled_to, scroll_factor):
        self.window = window
        self.batch = self.window.batch
        self.supergroup = ViewportOrderedGroup(
            0, self.window.calgroup, self)
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
        if attrn == 'window_bot':
            return int(self.bot_prop * self.window.height)
        elif attrn == 'window_left':
            return int(self.left_prop * self.window.width)
        elif attrn == 'window_right':
            r = int(self.right_prop * self.window.width)
            return r
        elif attrn == 'window_top':
            return int(self.top_prop * self.window.height)
        elif attrn == "width":
            r = self.window_right - self.window_left
            return r
        elif attrn == "height":
            return self.window_top - self.window_bot
        elif attrn == 'col_width':
            try:
                return self.width / len(self.cols)
            except ZeroDivisionError:
                return self.width
        elif attrn == 'row_height':
            return self.height / self.rows_shown
        elif attrn == 'visible':
            return self._visible and len(self.cols) > 0
        elif attrn == 'interactive':
            return self._interactive
        elif attrn == 'width_scalar':
            return self.supergroup.width / self.width
        elif attrn == 'height_scalar':
            return self.supergroup.height / self.height
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
            self, branch, scheduledict, interactive, style)
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
