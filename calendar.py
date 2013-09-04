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
    atrdic = {
        "window_bot": lambda self: self.bc.end[1],
        "window_top": lambda self: self.bc.end[1] + self.height,
        "window_left": lambda self: self.bc.end[0] - self.rx,
        "window_right": lambda self: self.bc.end[0] + self.rx}

    def __init__(self, bc, color_tup=(255, 0, 0, 255)):
        self.bc = bc
        self.color = color_tup
        self.batch = bc.batch
        self.group = bc.wedgegroup
        width = self.bc.calendar.style.spacing * 2
        self.width = width
        height = int(width / phi)
        self.height = height
        self.rx = width / 2
        self.ry = height / 2
        self.vertlist = None

    def __getattr__(self, attrn):
        assert(hasattr(self, 'atrdic'))
        return Wedge.atrdic[attrn](self)

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
    def get_startx(self):
        if self.col1.window_left < self.col2.window_left:
            return self.col1.window_right - self.calendar.style.spacing
        else:
            return self.col1.window_left - self.calendar.style.spacing
    def get_centerx(self):
        if self.col1.window_left < self.col2.window_left:
            return (self.col1.window_right +
                    self.calendar.style.spacing / 2)
        else:
            return (self.col2.window_right +
                    self.calendar.style.spacing / 2)
    def get_points(self):
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
    atrdic = {
        "startx": lambda self: self.get_startx(),
        "endx": lambda self: self.col2.window_left + self.col2.rx,
        "starty": lambda self: (
            self.col1.window_top - self.calendar.row_height * (
                self.tick - self.calendar.scrolled_to)),
        "endy": lambda self: (
            self.col2.window_top - self.calendar.row_height * (
                self.tick - self.calendar.scrolled_to)),
        "centerx": lambda self: self.get_centerx(),
        "points": lambda self: self.get_points(),
        "start": lambda self: (self.startx, self.starty),
        "end": lambda self: (self.endx, self.endy)}

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
        try:
            return BranchConnector.atrdic[attrn](self)
        except KeyError:
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
        width = timeline.cal.style.spacing * 2
        self.width = width
        height = int(width * phi)
        self.height = height
        self.rx = width / 2
        self.ry = height / 2
        self.atrdic = {
            "y": lambda: self.timeline.y,
            "window_y": lambda: self.timeline.window_y,
            "window_left": lambda: {
                True: self.timeline.window_left - self.width,
                False: self.timeline.window_right}[self.on_the_left],
            "window_right": lambda: {
                True: self.timeline.window_left + 1,
                False: self.timeline.window_right + self.width - 1
                }[self.on_the_left],
            "window_top": lambda: self.y + self.ry,
            "window_bot": lambda: self.y - self.ry}
            

    def __getattr__(self, attrn):
        assert(hasattr(self, 'atrdic'))
        return self.atrdic[attrn]()

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
        self.atrdic = {
            "calendar_left": lambda: self.col.calendar_left + self.col.style.spacing,
            "calendar_right": lambda: self.calendar_left + self.col.width,
            "window_left": lambda: self.calendar_left + self.cal.window_left,
            "window_right": lambda: self.window_left + self.col.width,
            "in_window": lambda: (self.y > 0 and self.y < self.window.height
                                  and self.window_right > 0
                                  and self.window_left < self.window.width)}

    def __getattr__(self, attrn):
        if attrn in ("calendar_y", "calendar_bot", "calendar_top"):
            return self.cal.height - self.cal.row_height * (
                self.rumor.tick - self.cal.scrolled_to)
        elif attrn in ("y", "window_y", "window_bot", "window_top"):
            return self.calendar_y + self.cal.window_bot
        else:
            assert(hasattr(self, 'atrdic'))
            return self.atrdic[attrn]()

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
    visible = True
    def get_calendar_bot(self):
        try:
            return self.cal.height - self.cal.row_height * (
                self.tick_to - self.cal.scrolled_to)
        except TypeError:
            return 0
    def is_same_size(self):
        r = (
            self.old_width == self.width and
            self.old_height == self.height)
        self.old_width = self.width
        self.old_height = self.height
        return r

    atrdic = {
        "interactive": lambda self: self.col.calendar.interactive,
        "window": lambda self: self.col.calendar.window,
        "calendar_left": lambda self: self.col.calendar_left + self.style.spacing,
        'calendar_right': lambda self: self.col.calendar_right - self.style.spacing,
        "calendar_top": lambda self: (self.cal.height - self.cal.row_height * 
                                      (self.tick_from - self.cal.scrolled_to) -
                                      self.style.spacing),
        "calendar_bot": lambda self: self.get_calendar_bot(),
        "width": lambda self: self.calendar_right - self.calendar_left,
        "height": lambda self: len(self) * self.cal.row_height,
        "window_left": lambda self: self.calendar_left + self.cal.window_left,
        "window_right": lambda self: self.calendar_right + self.cal.window_left,
        "window_top": lambda self: self.calendar_top + self.cal.window_bot,
        "window_bot": lambda self: self.calendar_bot + self.cal.window_bot,
        "in_view": lambda self: (self.window_right > 0 and
                                 self.window_left < self.window.width and
                                 self.window_top > 0 and
                                 self.window_bot < self.window.height),
        "same_size": lambda self: self.is_same_size(),
        "label_height": lambda self: self.style.fontsize + self.style.spacing,
        "hovered": lambda self: self is self.window.hovered}

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
        self.old_height = self.height
        self.old_width = self.width
        self.left_side = None
        self.right_side = None
        self.top_side = None
        self.bot_side = None
        self.label = None

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
        return CalendarCell.atrdic[attrn](self)

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
            self.top_side.delete()
        except:
            pass
        self.top_side = None
        try:
            self.left_side.delete()
        except:
            pass
        self.left_side = None
        try:
            self.bot_side.delete()
        except:
            pass
        self.bot_side = None
        try:
            self.right_side.delete()
        except:
            pass
        self.right_side = None

    def draw(self):
        colors = (0, 0, 0, 255) * 2
        l = self.window_left
        r = self.window_right
        t = self.window_top
        b = self.window_bot
        if (t < 0 or b > self.window.height):
            self.delete()
            return
        try:
            self.top_side.vertices = (l, t, r, t)
        except:
            self.top_side = self.batch.add(
                2,
                GL_LINES,
                self.bggroup,
                ('v2i', (l, t, r, t)),
                ('c4B', colors))
        try:
            self.left_side.vertices = (l, b, l, t)
        except:
            self.left_side = self.batch.add(
                2,
                GL_LINES,
                self.bggroup,
                ('v2i', (l, b, l, t)),
                ('c4B', colors))
        try:
            self.bot_side.vertices = (l, b, r, b)
        except:
            self.bot_side = self.batch.add(
                2,
                GL_LINES,
                self.bggroup,
                ('v2i', (l, b, r, b)),
                ('c4B', colors))
        try:
            self.right_side.vertices = (r, b, r, t)
        except:
            self.right_side = self.batch.add(
                2,
                GL_LINES,
                self.bggroup,
                ('v2i', (r, b, r, t)),
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

    def sttt(self):
        r = self._rowdict["scrolled_to"]
        if r is None:
            return self.rumor.tick
        else:
            return r
    def get_col_width(self):
        try:
            return self.width / len(self.cols_shown)
        except ZeroDivisionError:
            return self.width
    atrdic = {
        "typ": lambda self: self._rowdict["type"],
        "character": lambda self: self.rumor.get_character(self._rowdict["character"]),
        "dimension": lambda self: self.rumor.get_dimension(self._rowdict["dimension"]),
        "thing": lambda self: self.rumor.get_thing(
            self._rowdict["dimension"], self._rowdict["thing"]),
        "place": lambda self: self.rumor.get_place(
            self._rowdict["dimension"], self._rowdict["place"]),
        "portal": lambda self: self.rumor.get_portal(
            self._rowdict["dimension"],
            self._rowdict["origin"],
            self._rowdict["destination"]),
        "interactive": lambda self: self._rowdict["interactive"],
        "rows_shown": lambda self: self._rowdict["rows_shown"],
        "left_prop": lambda self: self._rowdict["left"],
        "right_prop": lambda self: self._rowdict["right"],
        "top_prop": lambda self: self._rowdict["top"],
        "bot_prop": lambda self: self._rowdict["bot"],
        "bot_tick": lambda self: self.top_tick + self.rows_shown,
        "style": lambda self: self.rumor.get_style(self._rowdict["style"]),
        "window_top": lambda self: int(self.top_prop * self.window.height),
        "window_bot": lambda self: int(self.bot_prop * self.window.height),
        "window_left": lambda self: int(self.left_prop * self.window.width),
        "window_right": lambda self: int(self.right_prop * self.window.width),
        "width": lambda self: self.window_right - self.window_left,
        "col_width": lambda self: self.get_col_width(),
        "height": lambda self: self.window_top - self.window_bot,
        "row_height": lambda self: self.height / self.rows_shown,
        "scrolled_to": lambda self: self.sttt(),
        "top_tick": lambda self: self.sttt(),
        "scroll_factor": lambda self: self._rowdict["scroll_factor"],
        "max_cols": lambda self: self._rowdict["max_cols"]
    }
        
    visible = True

    def __init__(self, window, idx):
        self.window = window
        self.rumor = self.window.rumor
        self.idx = idx
        self.batch = self.window.batch
        self.group = self.window.calgroup
        self.old_state = None
        self.tainted = False
        self._rowdict = self.rumor.tabdict[
            "calendar"][
                str(self.window)][
                    int(self)]
        self.coldict = {}
        self.cols_shown = set()
        for i in xrange(0, self.rumor.hi_branch):
            self.coldict[i] = self.make_col(i)
        for i in xrange(0, self.max_cols - 1):
            try:
                self.cols_shown.add(self.coldict[i])
            except KeyError:
                break
        self.branch_to = self.rumor.hi_branch
        self.refresh()

    def __int__(self):
        return self.idx

    def __getattr__(self, attrn):
        try:
            return self.atrdic[attrn](self)
        except KeyError:
            pass
        if self.typ == CAL_TYPE['THING']:
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
        raise AttributeError(
            "Calendar instance has no attribute {0}".format(attrn))

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
        if self.visible and len(self.cols_shown) > 0:
            for calcol in self.cols_shown:
                calcol.draw()
        else:
            for calcol in self.cols_shown:
                calcol.delete()

    def make_col(self, branch):
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
        for col1 in self.cols_shown:
            (parent, tick_from, tick_to) = self.rumor.timestream.branchdict[
                col1.branch]
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

    def review(self):
        for col in self.cols_shown:
            col.review()

    def regen(self):
        for col in self.cols_shown:
            col.regen_cells()

    def refresh(self):
        self.regen()
        self.rearrow()
        self.review()

class CalendarCol:
    atrdic = {
        "width": lambda self: self.calendar.col_width,
        "rx": lambda self: self.width / 2,
        "height": lambda self: self.calendar.height,
        "ry": lambda self: self.height / 2,
        "calendar_left": lambda self: int(self) * self.width,
        "calendar_right": lambda self: self.calendar_left + self.width,
        "calendar_top": lambda self: self.calendar.height,
        "calendar_bot": lambda self: 0,
        "window_left": lambda self: self.calendar.window_left + self.calendar_left,
        "window_right": lambda self: self.calendar.window_left + self.calendar_right,
        "window_top": lambda self: self.calendar.window_top,
        "window_bot": lambda self: self.calendar.window_bot,
        "idx": lambda self: self.calendar.cols.index(self)}

    def __init__(self, calendar, branch):
        self.calendar = calendar
        self.branch = branch
        self.rumor = self.calendar.rumor
        self.batch = self.calendar.batch
        self.style = self.calendar.style
        self.bggroup = OrderedGroup(0, self.calendar.group)
        self.cellgroup = OrderedGroup(1, self.calendar.group)
        self.textgroup = OrderedGroup(2, self.calendar.group)
        self.tlgroup = OrderedGroup(3, self.calendar.group)
        self.bcgroup = OrderedGroup(4, self.calendar.group)
        self.timeline = Timeline(self)
        self.window = self.calendar.window
        self.old_schedule_hash = 0
        self.celldict = {}
        self.cells_on_screen = set()
        self.vertl = None

    def __getattr__(self, attrn):
        return CalendarCol.atrdic[attrn](self)

    def __int__(self):
        return self.branch

    def delete(self):
        for cell in self.celldict.itervalues():
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

    def review(self):
        self.cells_on_screen = set()
        for cell in self.celldict.itervalues():
            if cell.in_view:
                self.cells_on_screen.add(cell)

    def refresh(self):
        self.regen_cells()
        self.review()

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
        except:
            self.vertl = self.batch.add_indexed(
                4,
                GL_TRIANGLES,
                self.bggroup,
                [0, 2, 3, 0, 1, 2],
                ('v2i', points),
                ('c4B', colors))
        if hasattr(self, 'bc'):
            self.bc.draw()
        if (
                self.rumor.branch == self.branch and
                self.timeline.in_window):
            self.timeline.draw()
        else:
            self.timeline.delete()
        for cell in self.celldict.itervalues():
            if cell in self.cells_on_screen:
                cell.draw()
            else:
                # Deleting the cell from video RAM, but keeping the
                # object around anyhow in case it's on screen later.
                cell.delete()

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
    atrdic = {
        "locations": lambda self: self.thing.locations[self.branch],
        "coverage": lambda self: self.character.thingdict[
            self.branch][dimn][thingn],
        "thing": lambda self: self.rumor.get_thing(dimn, thingn)
    }

    def __init__(self, calendar, branch):
        CalendarCol.__init__(self, calendar, branch)
        self.refresh()

    def __getattr__(self, attrn):
        if attrn in (
                "character",
                "dimension",
                "thing",
                "location"):
            return getattr(self.calendar, attrn)
        elif attrn in (
                "calendar_left",
                "calendar_top",
                "calendar_right",
                "calendar_bot",
                "window_left",
                "window_top",
                "window_right",
                "window_bot",
                "width",
                "height",
                "rx",
                "ry"):
            return CalendarCol.__getattr__(self, attrn)
        else:
            charn = str(self.calendar.character)
            dimn = str(self.calendar.dimension)
            thingn = str(self.calendar.thing)
            return LocationCalendarCol.atrdic[attrn](self)

    def regen_cells(self):
        for rd in TabdictIterator(self.locations):
            if rd["tick_from"] not in self.celldict:
                cell = CalendarCell(
                    self, rd["tick_from"], rd["tick_to"], rd["location"])
                self.celldict[rd["tick_from"]] = cell
            else:
                cell = self.celldict[rd["tick_from"]]
            cell.tick_to = rd["tick_to"]
            cell.text = rd["location"]
        todel = set()
        for cell in self.celldict.itervalues():
            if cell.tick_from not in self.locations:
                todel.add(cell)
        for cell in todel:
            cell.delete()
            del self.celldict[cell.tick_from]

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
