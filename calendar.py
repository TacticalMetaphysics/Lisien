# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import print_function
from logging import getLogger
from kivy.graphics import (
    Callback,
    Color,
    Triangle,
    Rectangle,
    Line)
from kivy.properties import (
    AliasProperty,
    NumericProperty,
    StringProperty,
    ObjectProperty)
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout


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
    def __init__(self, bc):
        self.bc = bc
        Line.__init__(self, points=self.bc.get_line_points())

    def upd_points(self, *args):
        self.points = self.bc.get_line_points()


class BCWedge(Triangle):
    points = AliasProperty(
        lambda self: self.get_points(), lambda self, v: None)

    def __init__(self, bc):
        self.bc = bc
        Triangle.__init__(self, points=self.bc.get_wedge_points())


class BranchConnector(Widget):
    wedge_height = 8

    def __init__(self, column, color=(255, 0, 0)):
        self.column = column
        self.color = color
        Widget.__init__(
            self,
            x=self.column.parent_branch_col.window_right -
            self.column.parent_branch_col.style.spacing,
            y=self.column.calendar.tick_to_y(self.column.start_tick) +
            self.column.calendar.offy)
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

    def get_wedge_points(self):
        b = self.y
        t = b + self.wedge_height
        c = self.column.left + self.column.width / 2
        rx = self.wedge_width / 2
        l = c - rx
        r = c + rx
        return [c, b, r, t, l, t]


class TLLine(Line):
    def __init__(self, tl):
        self.tl = tl
        Line.__init__(self, points=self.get_points())
        self.tl.column.calendar.closet.bind(
            branch=self.upd_points,
            tick=self.upd_points)

    def get_points(self):
        column = self.tl.column
        l = column.left
        r = l + self.tl.handle_width
        c = column.calendar.tick_to_y(column.calendar.charsheet.closet.tick)
        t = c + self.tl.ry
        b = c - self.tl.ry
        return [l, t, r, c, l, b]

    def upd_points(self, *args):
        self.points = self.get_points()


class TLWedge(Triangle):
    def __init__(self, tl):
        self.tl = tl
        Triangle.__init__(self, points=self.get_points())
        self.tl.column.calendar.closet.bind(
            branch=self.upd_points,
            tick=self.upd_points)

    def get_points(self):
        column = self.tl.column
        l = column.left
        r = column.right
        y = column.calendar.tick_to_y(column.calendar.charsheet.closet.tick)
        return [l, y, r, y]

    def upd_points(self, *args):
        self.points = self.get_points()


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
        with self.canvas:
            Color(*self.color)
            TLLine(self)
            TLWedge(self)


class CalendarCell(Widget):
    column = ObjectProperty()
    tick_from = NumericProperty()
    tick_to = NumericProperty(allownone=True)
    text = StringProperty('')

    def __init__(self, **kwargs):
        Widget.__init__(self, **kwargs)
        style = self.column.calendar.charsheet.style
        with self.canvas:
            Color(*style.bg_active.rgba)
            Rectangle(pos=self.pos, size=self.size)
        layout = BoxLayout(
            size=self.size, pos=self.pos, orientation='vertical',
            scale_hint={'x': None, 'y': self.height})
        label = Label(
            text=self.text,
            pos_hint={'top': 1, 'x': 0},
            valign='top',
            color=style.textcolor.rgba,
            font_name=style.fontface + '.ttf',
            font_size=style.fontsize)
        label.text_size = (None, label.height)
        layout.add_widget(label)
        self.add_widget(layout)

    def __len__(self):
        if self.tick_to is None:
            return 99999999
        return self.tick_to - self.tick_from


class CalendarColumn(RelativeLayout):
    calendar = ObjectProperty()
    branch = NumericProperty()

    @property
    def parent_branch_col(self):
        return self.calendar.make_col(
            self.calendar.charsheet.closet.skeleton[
                "timestream"][self.branch]["parent"])

    def __init__(self, **kwargs):
        RelativeLayout.__init__(self, **kwargs)
        self.calendar.bind(
            top_tick=self._trigger_layout, left_branch=self._trigger_layout,
            col_width=self._trigger_layout, rows_shown=self._trigger_layout)
        for cell in self:
            self.add_widget(cell)

    def __int__(self):
        return self.branch

    def __eq__(self, other):
        return hasattr(other, 'branch') and int(self) == int(other)

    def __gt__(self, other):
        return int(self) > int(other)

    def __lt__(self, other):
        return int(self) < int(other)

    def do_layout(self, *largs):
        for cell in self.children:
            cell.y = cell.tick_from * self.calendar.tick_height
            cell.height = len(cell) * self.calendar.tick_height
        super(CalendarColumn, self).do_layout(*largs)

    def to_parent(self, x, y, relative=False):
        if relative:
            return (x + self.x, self.top - y)
        return (x, y)


class LocationCalendarColumn(CalendarColumn):
    thing = ObjectProperty()

    def __init__(self, **kwargs):
        kwargs["thing"] = kwargs["calendar"].charsheet.character.closet.get_thing(
            kwargs["calendar"].keys[0], kwargs["calendar"].keys[1])
        CalendarColumn.__init__(self, **kwargs)
        

    def __iter__(self):
        rowiter = self.thing.locations.iterrows()
        prev = next(rowiter)
        style = self.calendar.charsheet.style
        for rd in rowiter:
            cc = CalendarCell(
                column=self, tick_from=prev["tick_from"],
                tick_to=rd["tick_from"], text=prev["location"])
            (w, h) = cc.size
            if h > style.fontsize + style.spacing:
                yield cc
        yield CalendarCell(
            column=self, tick_from=prev["tick_from"],
            tick_to=None, text=prev["location"])


class PlaceCalendarColumn(CalendarColumn):
    pass


class PortalCalendarColumn(CalendarColumn):
    pass


class StatCalendarColumn(CalendarColumn):
    pass


class SkillCalendarColumn(CalendarColumn):
    pass


class Calendar(RelativeLayout):
    top_tick = NumericProperty(0)
    left_branch = NumericProperty(0)
    cal_type = NumericProperty(0)
    col_width = NumericProperty(0)
    rows_shown = NumericProperty(0)
    tick_height = AliasProperty(
        lambda self: self.get_tick_height(),
        lambda self, v: None,
        bind=('rows_shown', 'height'))
    bot_tick = AliasProperty(
        lambda self: self.top_tick + self.rows_shown,
        lambda self, v: None,
        bind=('top_tick', 'rows_shown'))

    def __init__(
            self, charsheet, rows_shown, max_cols,
            left_branch, top_tick, scroll_factor, typ, *keys):
        self.charsheet = charsheet
        self.rows_shown = rows_shown
        self.max_cols = max_cols
        self.top_tick = top_tick
        self.left_branch = left_branch
        self.scroll_factor = scroll_factor
        self.change_type(typ, *keys)
        RelativeLayout.__init__(self)
        self.charsheet.character.closet.bind(branch=self._trigger_layout)
        for widget in self:
            self.add_widget(widget)

    def __iter__(self):
        if not hasattr(self, 'skel'):
            return
        else:
            i = self.left_branch
            while i < self.max_cols:
                yield self.make_col(i)
                i += 1

    def __eq__(self, other):
        return (
            other is not None and
            self.charsheet is other.charsheet and
            hasattr(other, 'cal_type') and
            self.cal_type == other.cal_type and
            self.skel == other.skel)

    def __ne__(self, other):
        return (
            other is None or
            self.charsheet is not other.charsheet or
            not hasattr(other, 'cal_type') or
            other.cal_type != self.cal_type or
            self.skel != other.skel)

    def _trigger_layout(self, *largs):
        pass

    def upd_col_width(self):
        branches = self.charsheet.character.closet.timestream.max_branch() + 1
        (w, h) = self.size
        if branches == 1:
            self.col_width = w
        elif self.max_cols < branches:
            self.col_width = w / self.max_cols
        else:
            self.col_width = w / branches

    def get_tick_height(self):
        return self.height / self.rows_shown

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
        return int(t - px_from_cal_top)

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
        }[self.cal_type](calendar=self, branch=branch)
