from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stencilview import StencilView
from kivy.graphics import (
    Color,
    Rectangle,
    Callback)
from kivy.properties import (
    BooleanProperty,
    NumericProperty,
    StringProperty,
    ObjectProperty)
from kivy.clock import Clock


fontsize = 10


def sane_colors(**kwargs):
    for colorstr in ("bg_color",):
        if colorstr in kwargs:
            if not isinstance(kwargs[colorstr], Color):
                kwargs[colorstr] = Color(*kwargs[colorstr])


class CalendarCell(StencilView):
    bg_color = ObjectProperty(None, allownone=True)
    text_color = ObjectProperty(None, allownone=True)
    font_name = StringProperty(None, allownone=True)
    font_size = NumericProperty(None, allownone=True)
    bg_rect = ObjectProperty(allownone=True)
    tick_from = NumericProperty()
    tick_to = NumericProperty(allownone=True)
    text = StringProperty()
    calendaredness = BooleanProperty(False)

    def __init__(self, **kwargs):
        sane_colors(**kwargs)
        StencilView.__init__(
            self,
            size_hint_y=None,
            **kwargs)

    def calendared(self, *args):
        if self.calendaredness:
            return
        self.calendaredness = True
        column = self.parent
        calendar = column.parent
        if (
                self.tick_from > calendar.max_tick):
            calendar.max_tick = self.tick_from
        if (
                self.tick_to is not None and
                self.tick_to > calendar.max_tick):
            calendar.max_tick = self.tick_to
        color = calendar.bg_color
        if column.bg_color is not None:
            color = column.bg_color
        if self.bg_color is not None:
            color = self.bg_color
        box = BoxLayout(
            pos=self.pos,
            size=self.size)

        box.canvas.before.add(Color(*color))
        self.bg_rect = Rectangle(
            pos=self.pos,
            size=self.size)
        box.canvas.before.add(self.bg_rect)

        text_color = calendar.text_color
        if column.text_color is not None:
            text_color = column.text_color
        if self.text_color is not None:
            text_color = self.text_color
        font_name = calendar.font_name
        if column.font_name is not None:
            font_name = column.font_name
        if self.font_name is not None:
            font_name = self.font_name
        font_size = calendar.font_size
        if column.font_size is not None:
            font_size = column.font_size
        if self.font_size is not None:
            font_size = self.font_size
        label = Label(
            text=self.text,
            valign='top',
            text_size=self.size,
            color=text_color,
            font_name=font_name,
            font_size=font_size)
        box.add_widget(label)

        def rearrange(*args):
            box.pos = self.pos
            box.size = self.size
            self.bg_rect.pos = box.pos
            self.bg_rect.size = box.size
            label.text_size = self.size
        with self.canvas:
            self.cb = Callback(rearrange)

        self.add_widget(box)


class CalendarColumn(RelativeLayout):
    bg_color = ObjectProperty(None, allownone=True)
    text_color = ObjectProperty(None, allownone=True)
    font_size = NumericProperty(None, allownone=True)
    font_name = StringProperty(None, allownone=True)

    def __init__(self, **kwargs):
        RelativeLayout.__init__(self, **kwargs)
        if "cells" in kwargs:
            for cell in kwargs["cells"]:
                self.add_widget(cell)

    def add_cell(self, text, tick_from, tick_to=None):
        cc = CalendarCell(
            tick_from=tick_from,
            tick_to=tick_to,
            text=text)
        self.add_widget(cc)
        return cc

    def on_parent(self, instance, value):
        if value is not None:
            for cell in self.children:
                cell.calendared()

    def do_layout(self, *args):
        calendar = self.parent
        if calendar is None:
            return
        tick_height = calendar.tick_height
        half_spacing = calendar.spacing / 2
        for cell in self.children:
            if cell.tick_to is not None:
                tick_to = cell.tick_to
            else:
                tick_to = calendar.get_max_col_tick()
            cell.y = calendar.tick_y(tick_to) + half_spacing
            cell.size = (
                self.width,
                ((tick_to - cell.tick_from) * tick_height) - half_spacing)
            cell.cb.ask_update()
        super(CalendarColumn, self).do_layout(*args)


class Calendar(BoxLayout):
    bg_color = ObjectProperty(Color(1.0, 0.0, 0.0, 1.0))
    text_color = ObjectProperty(None, allownone=True)
    font_size = NumericProperty(None, allownone=True)
    font_name = StringProperty(None, allownone=True)
    tick = NumericProperty(0)
    min_ticks = NumericProperty(100)
    max_tick = NumericProperty(0)
    tick_height = NumericProperty(10)

    def __init__(self, **kwargs):
        sane_colors(**kwargs)
        BoxLayout.__init__(
            self, orientation='horizontal', spacing=10, **kwargs)
        if "columns" in kwargs:
            for column in kwargs["columns"]:
                self.add_widget(column)

    def tick_y(self, tick):
        ticks_from_bot = self.max_tick - tick
        return ticks_from_bot * self.tick_height

    def do_layout(self, *args):
        super(Calendar, self).do_layout(*args)
        for child in self.children:
            child.do_layout(*args)

    def get_max_col_tick(self):
        return max((self.max_tick, self.min_ticks))
