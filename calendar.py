# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import print_function
from logging import getLogger
from kivy.graphics import (
    Color,
    Triangle,
    Rectangle,
    Line)
from kivy.properties import AliasProperty
from kivy.uix.widget import Widget
from kivy.uix.label import Label


"""User's view on a given item's schedule.

Usually there should be only one calendar per board, but it can switch
between showing various schedules, or even show many in parallel.

"""


logger = getLogger(__name__)


CAL_TYPE = {
    "THING": 0,
    "PLACE": 1,
    "PORTAL": 2,
    "STAT": 3,
    "SKILL": 4}


class BCLine(Line):
    points = AliasProperty(
        lambda self: self.get_points(), lambda self, v: None)

    def __init__(self, bc):
        self.bc = bc
        Line.__init__(self)

    def get_points(self):
        return self.bc.get_line_points()

    def set_points(self, v):
        self.bc.set_line_points(v)


class BCWedge(Triangle):
    points = AliasProperty(
        lambda self: self.get_points(), lambda self, v: None)

    def __init__(self, bc):
        self.bc = bc
        Triangle.__init__(self)

    def get_points(self):
        return self.bc.get_wedge_points()

    def set_points(self, v):
        return self.bc.set_wedge_points(v)


class BranchConnector(Widget):
    wedge_height = 8

    @property
    def x(self):
        """Return the x-coord of the *start* of the line, which is at the tick
upon the parent column when the branch occurred."""
        return (
            self.column.parent.window_right -
            self.column.parent.style.spacing)

    @property
    def y(self):
        return (
            self.column.calendar.tick_to_y(self.column.start_tick) +
            self.column.calendar.offy)

    def __init__(self, column, color=(255, 0, 0)):
        self.column = column
        self.color = color
        Widget.__init__(self)

    def build(self):
        with self.canvas:
            Color(*self.color)
            BCLine(self)
            BCWedge(self)

    def get_line_points(self):
        y0 = self.y
        y2 = y0
        y1 = y0 + self.wedge_height
        x0 = self.x
        # x1 gotta be half way between x0 and the center of my column
        x2 = self.column.left + self.column.width / 2
        dx = x2 - x0
        x1 = x0 + dx
        return [x0, y0, x1, y0, x1, y1, x2, y1, x2, y2]

    def set_line_points(self, v):
        print("Tried to set the line of BranchConnector {} to {}".format(self, v))

    def get_wedge_points(self):
        b = self.y
        c = self.column.left + self.column.width / 2
        rx = self.wedge_width / 2
        l = c - rx
        r = c + rx
        return [c, b, r, t, l, t]

    def set_wedge_points(self, v):
        print("Tried to set the wedge of BranchConnector {} to {}".format(self, v))


class TLLine(Line):
    points = AliasProperty(
        lambda self: self.get_points(),
        lambda self, v: self.set_points(v))

    def __init__(self, tl):
        self.tl = tl
        Line.__init__(self)

    def get_points(self):
        column = self.tl.column
        l = column.left
        r = l + self.tl.handle_width
        c = column.calendar.tick_to_y(column.calendar.charsheet.closet.tick)
        t = c + self.tl.ry
        b = c - self.tl.ry
        return [l, t, r, c, l, b]

    def set_points(self, v):
        print("Tried to set the points of TLLine {} to {}".format(self, v))


class TLWedge(Triangle):
    points = AliasProperty(
        lambda self: self.get_points(),
        lambda self, v: self.set_points(v))

    def __init__(self, tl):
        self.tl = tl
        Triangle.__init__(self)

    def get_points(self):
        column = self.tl.column
        l = column.left
        r = column.right
        y = column.calendar.tick_to_y(column.calendar.charsheet.closet.tick)
        return [l, y, r, y]

    def set_points(self, v):
        print("Tried to set the points of TLWedge {} to {}".format(self, v))


class Timeline(Widget):
    handle_width = 8
    handle_height = 16
    color = (255, 0, 0)

    def __init__(self, column):
        self.column = column
        self.ry = self.handle_height / 2
        self.offx = 0
        self.offy = 0
        Widget.__init__(self)

    def build(self):
        with self.canvas:
            Color(*self.color)
            TLLine(self)
            TLWedge(self)


class CalendarCellBG(Rectangle):
    pos_hint = AliasProperty(lambda self: self.get_pos, lambda self, v: None)
    size_hint = AliasProperty(lambda self: self.get_size, lambda self, v: None)

    def __init__(self, cell):
        self.cell = cell
        Rectangle.__init__(self)

    def get_pos(self):
        branch = self.cell.column.calendar.left_branch
        if branch is None:
            branch = self.cell.column.calendar.charsheet.closet.branch
        x = self.cell.column.calendar.col_width * (
            self.cell.column.branch - branch) + (
                self.cell.column.calendar.offx +
                self.cell.column.calendar.x)
        if self.cell.tick_to is None:
            y = self.cell.column.calendar.window_bot
        else:
            y = self.cell.column.calendar.tick_to_y(
                self.cell.tick_to)
        return (x, y)

    def get_size(self):
        calendar = self.cell.column.calendar
        (cal_w, cal_h) = calendar.size_hint
        w = cal_w / calendar.max_cols
        tick_height = cal_h / calendar.rows_shown
        if self.cell.tick_to is None:
            h = (calendar.bot_tick - self.cell.tick_from) * tick_height
        else:
            h = (self.cell.tick_to - self.cell.tick_from) * tick_height
        return (w, h)


class CalendarCellLabel(Label):
    text = AliasProperty(lambda self: self.cell.text, lambda self, v: None)
    color = AliasProperty(
        lambda self: self.cell.column.calendar.charsheet.style.textcolor.tup,
        lambda self, v: None)
    font_name = AliasProperty(
        lambda self: self.column.calendar.charsheet.style.fontface,
        lambda self, v: None)
    font_size = AliasProperty(
        lambda self: self.column.calendar.charsheet.style.fontsize,
        lambda self, v: None)
    pos_hint = AliasProperty(
        lambda self: self.get_pos(),
        lambda self, v: None)
    size_hint = AliasProperty(
        lambda self: self.cell.column.calendar.charsheet.style.fontsize,
        lambda self, v: None)
    valign = 'top'

    def __init__(self, cell):
        self.cell = cell

    def get_pos(self):
        branch = self.cell.column.calendar.left_branch
        if branch is None:
            branch = self.cell.column.calendar.charsheet.closet.branch
        l = self.cell.column.calendar.col_width * (
            self.cell.column.branch - branch) + (
                self.cell.column.calendar.offx +
                self.cell.column.calendar.x)
        t = self.cell.column.calendar.tick_to_y(self.cell.tick_from)
        return (l, t)


class CalendarCell(Widget):
    def __init__(self, column, tick_from, tick_to, text):
        self.column = column
        self.tick_from = tick_from
        self.tick_to = tick_to
        self.text = text
        Widget.__init__(self)

    def build(self):
        style = self.column.calendar.charsheet.style
        bg = style.bg_active.tup
        textcolor = style.textcolor.tup
        with self.canvas:
            Color(*bg)
            CalendarCellBG(self)
            Color(*textcolor)
            CalendarCellLabel(self)


class CalendarColumn(Widget):
    def __init__(self, calendar, branch):
        self.calendar = calendar
        self.branch = branch
        Widget.__init__(self)

    def __int__(self):
        return self.branch

    def __eq__(self, other):
        return int(self) == int(other)

    def __gt__(self, other):
        return int(self) > int(other)

    def __lt__(self, other):
        return int(self) < int(other)

    @property
    def parent(self):
        return self.calendar.make_col(
            self.calendar.charsheet.closet.skeleton[
                "timestream"][self.branch]["parent"])

    def get_x(self):
        branch = self.calendar.left_branch
        if branch is None:
            branch = self.calendar.charsheet.closet.branch
        return self.calendar.offx + self.calendar.col_width * (
            self.branch - branch) + self.calendar.x


class LocationCalendarColumn(CalendarColumn):
    typ = CAL_TYPE['THING']

    def __iter__(self):
        rowiter = self.calendar.thing.locations.iterrows()
        prev = next(rowiter)
        style = self.calendar.charsheet.style
        for rd in rowiter:
            cc = CalendarCell(
                self, prev["tick_from"], rd["tick_from"], prev["location"])
            (w, h) = cc.size
            if h > style.fontsize + style.spacing:
                yield cc
        yield CalendarCell(
            self, prev["tick_from"], None, prev["location"])

    def build(self):
        for cell in self:
            self.add_widget(cell)


class PlaceCalendarColumn(CalendarColumn):
    pass


class PortalCalendarColumn(CalendarColumn):
    pass


class StatCalendarColumn(CalendarColumn):
    pass


class SkillCalendarColumn(CalendarColumn):
    pass


class Calendar(Widget):
    top_tick = AliasProperty(
        lambda self: self.get_top_tick(),
        lambda self, v: self.charsheet.set_cal_top_tick(self, v))
    offx = AliasProperty(
        lambda self: self.charsheet.get_cal_offx(self),
        lambda self, v: self.charsheet.set_cal_offx(self, v))
    offy = AliasProperty(
        lambda self: self.charsheet.get_cal_offy(self),
        lambda self, v: self.charsheet.set_cal_offy(self, v))

    @property
    def col_width(self):
        branches = self.charsheet.closet.timestream.max_branch() + 1
        (w, h) = self.size
        if branches == 1:
            return w
        elif self.max_cols < branches:
            return w / self.max_cols
        else:
            return w / branches

    def get_top_tick(self):
        r = self.charsheet.get_cal_top_tick(self)
        if r is None:
            return self.charsheet.closet.tick
        else:
            return r

    def __init__(
            self, charsheet, rows_shown, max_cols,
            left_branch, scroll_factor, typ, *keys):
        self.charsheet = charsheet
        self.rows_shown = rows_shown
        self.left_branch = left_branch
        self.scroll_factor = scroll_factor
        self.change_type(typ, *keys)

    def __iter__(self):
        if not hasattr(self, 'data'):
            return iter([])
        else:
            return self.itercolumns

    def __eq__(self, other):
        return (
            self.charsheet is other.charsheet and
            hasattr(other, 'cal_type') and
            self.cal_type == other.cal_type and
            self.skel == other.skel)

    def __ne__(self, othre):
        return (
            self.charsheet is not other.charsheet or
            not hasattr(other, 'cal_type') or
            other.cal_type != self.cal_type or
            self.skel != other.skel)

    def change_type(self, cal_type, *keys):
        self.cal_type = cal_type
        dk = {
            CAL_TYPE["THING"]: "thing",
            CAL_TYPE["PLACE"]: "place",
            CAL_TYPE["PORTAL"]: "portal",
            CAL_TYPE["STAT"]: "stat",
            CAL_TYPE["SKILL"]: "skill"}[cal_type]
        self.skel = self.charsheet.character.get_item_history(dk, *keys)
        self.keys = keys

    def tick_to_y(self, tick):
        ticks_from_top = tick - self.top_tick
        (w, h) = self.size
        px_from_cal_top = (self.rows_shown / h) * ticks_from_top
        (x, y) = self.pos
        t = y + h
        return int(t - px_from_cal_top + self.offy)

    def y_to_tick(self, y):
        (x, y) = self.pos
        (w, h) = self.size
        t = y + h
        px_from_cal_top = t - y
        ticks_from_top = px_from_cal_top / (self.rows_shown / h)
        return ticks_from_top + self.top_tick

    def make_col(self, branch):
        return {
            CAL_TYPE['THING']: LocationCalendarColumn,
            CAL_TYPE['PLACE']: PlaceCalendarColumn,
            CAL_TYPE['PORTAL']: PortalCalendarColumn,
            CAL_TYPE['STAT']: StatCalendarColumn,
            CAL_TYPE['SKILL']: SkillCalendarColumn
        }[self.cal_type](self, branch)
