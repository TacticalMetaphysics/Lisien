# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import print_function
from kivy.graphics import (
    Color,
    Triangle,
    Rectangle,
    Line)
from kivy.properties import (
    NumericProperty,
    StringProperty,
    ObjectProperty,
    ReferenceListProperty)
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView


"""User's view on a given item's schedule.

Usually there should be only one calendar per board, but it can switch
between showing various schedules, or even show many in parallel.

"""


CAL_TYPE = {
    "THING": 0,
    "PLACE": 1,
    "PORTAL": 2,
    "STAT": 3,
    "SKILL": 4}


class BranchConnector(Widget):
    wedge_height = 8

    def on_parent(self, *args):
        self.x = (self.parent.parent_branch_col.window_right -
                  self.parent.parent_branch_col.style.spacing)
        self.y = (self.parent.calendar.tick_to_y(self.column.start_tick) +
                  self.parent.calendar.offy)
        with self.canvas:
            Color(*self.color)
            Line(points=self.get_line_points())
            Triangle(points=self.get_wedge_points())

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


class CalendarCell(BoxLayout):
    bg_color = ObjectProperty(allownone=True)
    bg_rect = ObjectProperty(allownone=True)
    tick_from = NumericProperty()
    tick_to = NumericProperty()
    text = StringProperty()

    def on_parent(self, *args):
        column = self.parent
        calendar = column.parent
        charsheet = calendar.parent
        self.bg_color = charsheet.style.bg_active.kivy_color
        self.canvas.before.add(self.bg_color)

    def calendared(self, *args):
        self.bg_rect = Rectangle(
            pos=self.pos,
            size=self.size)
        label = Label(
            text=self.text,
            valign='top',
            text_size=self.size)
        self.add_widget(label)

        def repos(*args):
            self.bg_rect.pos = self.pos

        def resize(*args):
            self.bg_rect.size = self.size
            label.text_size = self.size
        self.bind(pos=repos)
        self.bind(size=resize)


class CalendarColumn(RelativeLayout):
    branch = NumericProperty()
    tl_line = ObjectProperty(allownone=True)
    tl_wedge = ObjectProperty(allownone=True)
    tl_color = ObjectProperty(allownone=True)
    tl_width = 16
    tl_height = 8

    def __init__(self, **kwargs):
        RelativeLayout.__init__(self, **kwargs)
        if "cells" in kwargs:
            for cell in kwargs["cells"]:
                self.add_widget(cell)

    @property
    def parent_branch_col(self):
        return self.parent.make_col(
            self.parent.parent.closet.skeleton[
                "timestream"][self.branch]["parent"])

    def on_parent(self, *args):
        calendar = self.parent
        charsheet = calendar.parent
        closet = charsheet.character.closet
        (line_points, wedge_points) = self.get_tl_points(closet.tick)
        if self.branch == closet.branch:
            self.tl_color = Color(1.0, 0.0, 0.0, 1.0)
        else:
            self.tl_color = Color(1.0, 0.0, 0.0, 0.0)
        self.canvas.after.add(self.tl_color)
        self.tl_line = Line(points=line_points)
        self.canvas.after.add(self.tl_line)
        self.tl_wedge = Triangle(points=wedge_points)
        self.canvas.after.add(self.tl_wedge)
        for cell in self.children:
            cell.calendared()
        closet.bind(branch=self.upd_tl, tick=self.upd_tl)

    def __int__(self):
        return self.branch

    def __eq__(self, other):
        return hasattr(other, 'branch') and int(self) == int(other)

    def __gt__(self, other):
        return int(self) > int(other)

    def __lt__(self, other):
        return int(self) < int(other)

    def upd_tl(self, *args):
        calendar = self.parent
        (line_points, wedge_points) = self.get_tl_points(calendar.tick)
        self.tl_line.points = line_points
        self.tl_wedge.points = wedge_points

    def get_tl_points(self, tick):
        (l, b) = self.to_parent(0, 0)
        (r, t) = self.to_parent(*self.size)
        try:
            c = self.parent.tick_y(tick)
        except ZeroDivisionError:
            c = self.height
        line_points = self.to_parent(l, c) + self.to_parent(r, c)
        r = self.tl_width
        ry = self.tl_height / 2
        t = c + ry
        b = c - ry
        wedge_points = self.to_parent(l, t) + self.to_parent(
            r, c) + self.to_parent(l, b)
        return (line_points, wedge_points)

    def do_layout(self, *args):
        calendar = self.parent
        charsheet = calendar.parent
        closet = charsheet.character.closet
        tick_height = calendar.height / closet.timestream.hi_tick
        for cell in self.children:
            cell.y = calendar.tick_y(cell.tick_to)
            cell.size = (
                self.width,
                (cell.tick_to - cell.tick_from) * tick_height)


class LocationCalendarColumn(CalendarColumn):
    thing = ObjectProperty()

    def on_parent(self, *args):
        self.thing = self.parent.parent.character.closet.get_thing(
            self.parent.keys[0], self.parent.keys[1])
        rowiter = self.thing.locations.iterrows()
        prev = next(rowiter)
        calendar = self.parent
        charsheet = calendar.parent
        style = charsheet.style
        for rd in rowiter:
            tick_from = prev["tick_from"]
            if rd["tick_from"] is None:
                tick_to = charsheet.character.closet.timestream.hi_tick
            else:
                tick_to = rd["tick_from"]
            cc = CalendarCell(
                tick_from=tick_from, tick_to=tick_to, text=prev["location"])
            self.add_widget(cc)
        super(LocationCalendarColumn, self).on_parent(*args)


class PlaceCalendarColumn(CalendarColumn):
    pass


class PortalCalendarColumn(CalendarColumn):
    pass


class StatCalendarColumn(CalendarColumn):
    pass


class SkillCalendarColumn(CalendarColumn):
    pass


class Calendar(BoxLayout):
    cal_type = NumericProperty(0)
    scroll_factor = NumericProperty(4)
    key0 = StringProperty()
    key1 = StringProperty(allownone=True)
    key2 = StringProperty(allownone=True)
    keys = ReferenceListProperty(key0, key1, key2)

    def __init__(self, **kwargs):
        BoxLayout.__init__(
            self,
            size_hint=(None, None),
            **kwargs)

    def __iter__(self):
        charsheet = self.parent
        closet = charsheet.character.closet
        hi_branch = closet.timestream.hi_branch
        if not hasattr(self, 'skel'):
            return
        else:
            i = 0
            while i < hi_branch:
                yield self.make_col(i)
                i += 1

    def __eq__(self, other):
        return (
            other is not None and
            self.parent is other.parent and
            hasattr(other, 'cal_type') and
            self.cal_type == other.cal_type and
            self.skel == other.skel)

    def __ne__(self, other):
        return (
            other is None or
            self.parent is not other.parent or
            not hasattr(other, 'cal_type') or
            other.cal_type != self.cal_type or
            self.skel != other.skel)

    def upd_size(self, *args):
        self.size = self.children[0].size

    def col_x(self, col):
        (x, y) = col.to_parent(0, 0)
        return x - self.children[0].spacing

    def col_scroll_x(self, col):
        return self.col_x(col) / float(self.width)

    def tick_y(self, tick):
        # assuming that scroll_y == 0.0 indicates the bottom of the
        # longest column, and scroll_y == 1.0 indicates tick zero
        charsheet = self.parent
        hi_tick = charsheet.character.closet.timestream.hi_tick
        tick_height = self.height / hi_tick
        ticks_from_bot = hi_tick - tick
        return ticks_from_bot * tick_height

    def tick_scroll_y(self, tick):
        return self.tick_y(tick) / float(self.height)

    def on_parent(self, *args):
        self.change_type(self.typ, self.keys)
        for widget in self:
            layout.add_widget(widget)

    def change_type(self, cal_type, keys):
        self.cal_type = cal_type
        dk = {
            CAL_TYPE["THING"]: "thing",
            CAL_TYPE["PLACE"]: "place",
            CAL_TYPE["PORTAL"]: "portal",
            CAL_TYPE["STAT"]: "stat",
            CAL_TYPE["SKILL"]: "skill"}[cal_type]
        self.skel = self.parent.character.get_item_history(dk, *keys)
        self.keys = keys

    def make_col(self, branch):
        return {
            CAL_TYPE['THING']: LocationCalendarColumn,
            CAL_TYPE['PLACE']: PlaceCalendarColumn,
            CAL_TYPE['PORTAL']: PortalCalendarColumn,
            CAL_TYPE['STAT']: StatCalendarColumn,
            CAL_TYPE['SKILL']: SkillCalendarColumn
        }[self.cal_type](calendar=self, branch=branch)
