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
        else:
            kwargs["tick_to"] = kwargs["calendar"].get_max_col_tick()
        super(Cell, self).__init__(
            size_hint=(1, None),
            **kwargs)


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

    def tick_y(self, tick):
        if tick is None:
            return self.y
        return self.top - self.tick_height * tick

    def ticks_height(self, tick_from, tick_to):
        if tick_to is None:
            tick_to = self.get_max_col_tick()
        span = abs(tick_to - tick_from)
        return self.tick_height * span

    def get_max_col_tick(self):
        return max((self.max_tick, self.min_ticks))

    def on_pos(self, i, v):
        print("calendar pos: {}".format(v))

    def on_size(self, i, v):
        print("calendar size: {}".format(v))

    def on_max_tick(self, i, v):
        print("max tick: {}".format(v))
