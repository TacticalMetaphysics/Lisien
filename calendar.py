from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import (
    Color,
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
    def __init__(self, **kwargs):
        RelativeLayout.__init__(self, size_hint=(1, None), **kwargs)
        if "cells" in kwargs:
            for cell in kwargs["cells"]:
                self.add_widget(cell)

    def on_parent(self, *args):
        for cell in self.children:
            cell.calendared()

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
