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

        def draw(self, batch, group):
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
                    ('c4b', colors))

    def __init__(self, col, handle_side="left"):
        self.col = col
        self.cal = self.col.calendar
        self.window = self.cal.window
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

    def draw(self, batch, group):
        colors = self.color * 2
        points = (
            self.window_left, self.y,
            self.window_right, self.y)
        try:
            self.vertlist.vertices = list(points)
            self.vertlist.colors = list(colors)
        except AttributeError:
            self.vertlist = batch.add(
                2,
                GL_LINES,
                group,
                ('v2i', points),
                ('c4B', colors))
        self.handle.draw(batch, group)


class CalendarCell:
    """A block of time in a calendar.

Uses information from the CalendarCol it's in and the Event it
represents to calculate its dimensions and coordinates.

    """
    def __init__(self, col, tick_from, tick_to, text):
        self.col = col
        self.cal = self.col.calendar
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
        self.sprite = None
        self.label = None
        self.visible = True
        self.tweaks = 0
        self.inactive_pattern = color_pattern(self.style.fg_inactive.tup)
        self.active_pattern = color_pattern(self.style.fg_active.tup)

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
            return self.col.calendar_left + self.style.spacing
        elif attrn == 'calendar_right':
            return self.col.calendar_right - self.style.spacing
        elif attrn == 'calendar_top':
            return self.cal.height - self.cal.row_height * (
                self.tick_from - self.cal.scrolled_to)
        elif attrn == 'calendar_bot':
            return self.cal.height - self.cal.row_height * (
                self.tick_to - self.cal.scrolled_to)
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
            self.sprite.delete()
        except:
            pass
        try:
            self.label.delete()
        except:
            pass

    def draw(self, batch, group):
        if self.visible and self.height > 0:
            if self.hovered:
                image = self.active_pattern.create_image(
                    self.width, self.height)
            else:
                image = self.inactive_pattern.create_image(
                    self.width, self.height)
            if not hasattr(self, 'bggroup'):
                self.bggroup = OrderedGroup(0, group)
            if not hasattr(self, 'fggroup'):
                self.fggroup = OrderedGroup(1, group)
            if self.hovered != self.was_hovered:
                self.sprite = Sprite(
                    image,
                    self.calendar_left,
                    self.calendar_bot,
                    batch=batch,
                    group=self.bggroup)
                self.was_hovered = self.hovered
            else:
                try:
                    self.sprite.x = self.calendar_left
                    self.sprite.y = self.calendar_bot
                except:
                    self.sprite = Sprite(
                        image,
                        self.calendar_left,
                        self.calendar_bot,
                        batch=batch,
                        group=self.bggroup)
            y = self.calendar_top - self.label_height
            try:
                self.label.x = self.calendar_left
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
                    group=self.fggroup)
        else:
            self.delete()


class CalendarCol:
    def __init__(self, calendar, scheduledict, interactive, style):
        self.calendar = calendar
        self.window = calendar.window
        self.scheduledict = scheduledict
        self.board = self.calendar.board
        self.rumor = self.calendar.rumor
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

    def draw(self, batch, group):
        if (
                not hasattr(self, 'image') or
                self.image.width != self.width or
                self.image.height != self.height):
            self.image = self.inactive_pattern.create_image(
                self.width, self.height)
        if not hasattr(self, 'bggroup'):
            self.bggroup = OrderedGroup(0, group)
        if not hasattr(self, 'fggroup'):
            self.fggroup = OrderedGroup(1, group)
        try:
            assert(self.sprite.width == self.width)
            assert(self.sprite.height == self.height)
            self.sprite.x = self.calendar_left
            self.sprite.y = self.calendar_bot
        except:
            self.sprite = Sprite(
                self.image,
                self.calendar_left,
                self.calendar_bot,
                batch=batch,
                group=self.bggroup)
        for cel in self.cells:
            cel.draw(batch, self.fggroup)


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
        if attrn == 'board':
            return self.rumor.boarddict[self._dimension]
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

    def mkcol(self, scheduledict, interactive=True, style=None):
        if style is None:
            style = self.style
        cc = CalendarCol(
            self, scheduledict, interactive, style)
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

    def draw(self, batch, group):
        if not hasattr(self, 'supergroup'):
            self.supergroup = ViewportOrderedGroup(0, group, self)
        if self.visible and len(self.cols) > 0:
            for calcol in self.cols:
                calcol.draw(batch, self.supergroup)
        else:
            for calcol in self.cols:
                calcol.delete()

    def remove(self, it):
        self.cols.remove(it)

    def refresh(self):
        for col in self.cols:
            col.regen_cells()
