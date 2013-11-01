from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import (
    Color,
    Rectangle)
from kivy.properties import (
    NumericProperty,
    StringProperty,
    ObjectProperty,
    ListProperty)


def get_column(what):
    while not isinstance(what, Column):
        what = what.parent
    return what


def get_calendar(what):
    while not isinstance(what, Calendar):
        what = what.parent
    return what


class ColorBox(BoxLayout):
    color = ListProperty()
    rect = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(BoxLayout, self).__init__(**kwargs)
        self.canvas.add(Color(*self.color))
        self.rect = Rectangle(pos=self.pos, size=self.size)
        self.canvas.add(self.rect)

        def reposr(*args):
            self.rect.pos = self.pos

        def resizer(*args):
            self.rect.size = self.size

        self.bind(pos=reposr, size=resizer)


class Cell(RelativeLayout):
    bg_color = ListProperty(None)
    text_color = ListProperty(None)
    font_name = StringProperty(None, allownone=True)
    font_size = NumericProperty(None, allownone=True)
    bg_rect = ObjectProperty(None)
    tick_from = NumericProperty()
    tick_to = NumericProperty(None, allownone=True)
    text = StringProperty()
    calendar = ObjectProperty()
    column = ObjectProperty()

    def __init__(self, **kwargs):
        for kwarg in ["bg_color", "text_color", "font_name", "font_size"]:
            if kwarg not in kwargs:
                if getattr(kwargs["column"], kwarg) in ([], None):
                    kwargs[kwarg] = getattr(kwargs["calendar"], kwarg)
                else:
                    kwargs[kwarg] = getattr(kwargs["column"], kwarg)
        if kwargs["tick_from"] > kwargs["calendar"].max_tick:
            kwargs["calendar"].max_tick = kwargs["tick_from"]
        if kwargs["tick_to"] is not None:
            if kwargs["tick_to"] > kwargs["calendar"].max_tick:
                kwargs["calendar"].max_tick = kwargs["tick_to"]
        super(Cell, self).__init__(
            size_hint=(1, None),
            **kwargs)
        if [] not in (self.bg_color, self.text_color):
            assert(self.bg_color != self.text_color)
        if (
                self.tick_from > self.calendar.max_tick):
            self.calendar.max_tick = self.tick_from
        if (
                self.tick_to is not None and
                self.tick_to > self.calendar.max_tick):
            self.calendar.max_tick = self.tick_to
        with self.canvas.before:
            Color(*self.bg_color)
            Rectangle(pos=self.pos, size=self.size)

    def on_pos(self, *args):
        print("cell pos: {}".format(self.pos))

    def on_size(self, *args):
        print("cell size: {}".format(self.size))

    def get_y(self):
        if self.tick_to is None:
            return self.calendar.y + self.calendar.celspace
        else:
            return self.calendar.tick_y(self.tick_to) + self.calendar.celspace

    def get_height(self):
        if self.tick_to is None:
            return self.calendar.height - self.calendar.celspace
        else:
            return self.get_y() - self.calendar.tick_y(
                self.tick_from) - self.calendar.celspace


class Column(RelativeLayout):
    bg_color = ListProperty(None)
    text_color = ListProperty(None)
    font_size = NumericProperty(None, allownone=True)
    font_name = StringProperty(None, allownone=True)

    def add_cell(self, text, tick_from, tick_to=None):
        assert(self.parent is not None)
        cc = Cell(
            tick_from=tick_from,
            tick_to=tick_to,
            text=text,
            calendar=self.parent,
            column=self)
        self.add_widget(cc)
        return cc

    def on_pos(self, i, v):
        print("column pos: {}".format(v))

    def on_size(self, i, v):
        print("column size: {}".format(v))


class Calendar(GridLayout):
    bg_color = ListProperty()
    text_color = ListProperty()
    font_size = NumericProperty(None, allownone=True)
    font_name = StringProperty(None, allownone=True)
    tick = NumericProperty(0)
    min_ticks = NumericProperty(100)
    max_tick = NumericProperty(0)
    tick_height = NumericProperty(10)
    celspace = NumericProperty(5)

    def tick_y(self, tick):
        return self.top - self.tick_height * tick

    def get_max_col_tick(self):
        return max((self.max_tick, self.min_ticks))

    def on_pos(self, i, v):
        print("calendar pos: {}".format(v))

    def on_size(self, i, v):
        print("calendar size: {}".format(v))

    def on_max_tick(self, i, v):
        print("max tick: {}".format(v))
