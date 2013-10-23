from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import (
    Color,
    Line,
    Triangle,
    Rectangle)
from kivy.properties import (
    NumericProperty,
    StringProperty,
    ObjectProperty)


fontsize = 10


class CalendarCell(BoxLayout):
    bg_color = ObjectProperty(Color(1.0, 0.0, 0.0, 1.0))
    bg_rect = ObjectProperty(allownone=True)
    tick_from = NumericProperty()
    tick_to = NumericProperty(allownone=True)
    text = StringProperty()

    def __init__(self, **kwargs):
        BoxLayout.__init__(self, size_hint=(None, None), **kwargs)

    def on_parent(self, *args):
        column = self.parent
        calendar = column.parent
        calendar.upd_max_tick(self.tick_from)
        if self.tick_to is not None:
            calendar.upd_max_tick(self.tick_to)

    def calendared(self, *args):
        if self.tick_from == self.tick_to:
            return
        self.canvas.before.add(self.bg_color)
        self.bg_rect = Rectangle(
            pos=self.pos,
            size=self.size)
        self.canvas.before.add(self.bg_rect)

        def repos_rect(*args):
            self.bg_rect.pos = self.pos
        self.bind(pos=repos_rect)

        def resize_rect(*args):
            self.bg_rect.size = self.size
        self.bind(size=resize_rect)

        label = Label(
            text=self.text,
            valign='top',
            text_size=self.size)
        self.add_widget(label)

        def resize_label(*args):
            label.text_size = self.size
        self.bind(size=resize_label)


class CalendarColumn(RelativeLayout):
    tl_line = ObjectProperty(allownone=True)
    tl_wedge = ObjectProperty(allownone=True)
    tl_color = (0.0, 1.0, 0.0, 1.0)
    tl_width = 16
    tl_height = 8

    def __init__(self, **kwargs):
        RelativeLayout.__init__(self, **kwargs)
        if "cells" in kwargs:
            for cell in kwargs["cells"]:
                self.add_widget(cell)

    def on_parent(self, *args):
        (line_points, wedge_points) = self.get_tl_points(0)
        self.canvas.after.add(Color(*self.tl_color))
        self.tl_line = Line(points=line_points)
        self.canvas.after.add(self.tl_line)
        self.tl_wedge = Triangle(points=wedge_points)
        self.canvas.after.add(self.tl_wedge)
        calendar = self.parent
        for cell in self.children:
            cell.calendared()
        calendar.bind(tick=self.upd_tl)

    def upd_tl(self, *args):
        calendar = self.parent
        (line_points, wedge_points) = self.get_tl_points(calendar.tick)
        self.tl_line.points = line_points
        self.tl_wedge.points = wedge_points

    def get_tl_points(self, tick):
        (l, b) = 0, 0
        (r, t) = self.size
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
        myheight = 0
        tick_height = self.parent.tick_height
        for cell in self.children:
            if cell.tick_to is None:
                tick_to = self.parent.max_tick

                def upd_max_tick(*args):
                    cell.tick_to = self.parent.max_tick
                self.parent.bind(max_tick=upd_max_tick)
            else:
                tick_to = cell.tick_to
            cell.y = self.parent.tick_y(tick_to)
            cell.size = (
                self.width,
                (tick_to - cell.tick_from) * tick_height)
            myheight += cell.height
        if self.parent.height < myheight:
            self.parent.height = myheight
        self.height = myheight


class Calendar(BoxLayout):
    tick = NumericProperty(0)
    max_tick = NumericProperty(0)
    tick_height = NumericProperty(10)

    def __init__(self, **kwargs):
        BoxLayout.__init__(
            self, orientation='horizontal', spacing=10, **kwargs)
        if "columns" in kwargs:
            for column in kwargs["columns"]:
                self.add_widget(column)

    def tick_y(self, tick):
        ticks_from_bot = self.max_tick - tick
        return ticks_from_bot * self.tick_height

    def upd_max_tick(self, value):
        if value > self.max_tick:
            self.max_tick = value
        print("calendar's max tick is now {}".format(self.max_tick))

    def do_layout(self, *args):
        super(Calendar, self).do_layout(*args)
        for child in self.children:
            child.do_layout(*args)
