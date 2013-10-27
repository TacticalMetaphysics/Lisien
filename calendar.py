from kivy.uix.label import Label
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stencilview import StencilView
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import (
    Color,
    Rectangle,
    Callback)
from kivy.properties import (
    ListProperty,
    ReferenceListProperty,
    BooleanProperty,
    BoundedNumericProperty,
    NumericProperty,
    StringProperty,
    ObjectProperty)


def get_column(what):
    while not isinstance(what, Column):
        what = what.parent
    return what


def get_calendar(what):
    while not isinstance(what, Calendar):
        what = what.parent
    return what


class Cell(StencilView):
    bg_r = BoundedNumericProperty(None, min=0.0, max=1.0, allownone=True)
    bg_g = BoundedNumericProperty(None, min=0.0, max=1.0, allownone=True)
    bg_b = BoundedNumericProperty(None, min=0.0, max=1.0, allownone=True)
    bg_a = BoundedNumericProperty(None, min=0.0, max=1.0, allownone=True)
    bg_color = ReferenceListProperty(bg_r, bg_g, bg_b, bg_a)
    text_r = BoundedNumericProperty(None, min=0.0, max=1.0, allownone=True)
    text_g = BoundedNumericProperty(None, min=0.0, max=1.0, allownone=True)
    text_b = BoundedNumericProperty(None, min=0.0, max=1.0, allownone=True)
    text_a = BoundedNumericProperty(None, min=0.0, max=1.0, allownone=True)
    text_color = ReferenceListProperty(bg_r, bg_g, bg_b, bg_a)
    font_name = StringProperty(None, allownone=True)
    font_size = NumericProperty(None, allownone=True)
    bg_rect = ObjectProperty(allownone=True)
    tick_from = NumericProperty()
    tick_to = NumericProperty(allownone=True)
    text = StringProperty()
    calendaredness = BooleanProperty(False)

    def __init__(self, **kwargs):
        StencilView.__init__(
            self,
            size_hint=(1, None),
            **kwargs)

    def calendared(self, *args):
        if self.calendaredness:
            return
        self.calendaredness = True
        column = get_column(self)
        calendar = get_calendar(column)
        if (
                self.tick_from > calendar.max_tick):
            calendar.max_tick = self.tick_from
        if (
                self.tick_to is not None and
                self.tick_to > calendar.max_tick):
            calendar.max_tick = self.tick_to
        color = calendar.bg_color
        if column.bg_color != [None, None, None, None]:
            color = column.bg_color
        if self.bg_color != [None, None, None, None]:
            color = self.bg_color
        box = BoxLayout(
            pos=self.pos,
            size=self.size)

        box.canvas.before.add(Color(*color))
        box.bg_rect = Rectangle(
            pos=self.pos,
            size=self.size)
        box.canvas.before.add(box.bg_rect)

        text_color = calendar.text_color
        if column.text_color != (None, None, None, None):
            text_color = column.text_color
        if self.text_color != (None, None, None, None):
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
        label_kwargs = {
            'text': self.text,
            'valign': 'top',
            'text_size': self.size}
        if text_color != (None, None, None, None):
            label_kwargs['text_color'] = text_color
        if font_name is not None:
            label_kwargs['font_name'] = font_name
        if font_size is not None:
            label_kwargs['font_size'] = font_size
        label = Label(**label_kwargs)
        box.add_widget(label)

        def rearrange(*args):
            box.pos = self.pos
            box.size = self.size
            box.bg_rect.pos = box.pos
            box.bg_rect.size = box.size
            label.text_size = self.size
        with box.canvas:
            self.cb = Callback(rearrange)

        self.add_widget(box)


class Column(RelativeLayout):
    bg_r = BoundedNumericProperty(None, min=0.0, max=1.0, allownone=True)
    bg_g = BoundedNumericProperty(None, min=0.0, max=1.0, allownone=True)
    bg_b = BoundedNumericProperty(None, min=0.0, max=1.0, allownone=True)
    bg_a = BoundedNumericProperty(None, min=0.0, max=1.0, allownone=True)
    bg_color = ReferenceListProperty(bg_r, bg_g, bg_b, bg_a)
    text_r = BoundedNumericProperty(None, min=0.0, max=1.0, allownone=True)
    text_g = BoundedNumericProperty(None, min=0.0, max=1.0, allownone=True)
    text_b = BoundedNumericProperty(None, min=0.0, max=1.0, allownone=True)
    text_a = BoundedNumericProperty(None, min=0.0, max=1.0, allownone=True)
    text_color = ReferenceListProperty(bg_r, bg_g, bg_b, bg_a)
    font_size = NumericProperty(None, allownone=True)
    font_name = StringProperty(None, allownone=True)
    calendar = ObjectProperty(allownone=True)
    tick_from = NumericProperty(None)
    tick_height = NumericProperty(10)
    spacing = NumericProperty(10)

    def __init__(self, **kwargs):
        super(Column, self).__init__(
            size_hint=(1, None), **kwargs)

    def add_cell(self, text, tick_from, tick_to=None):
        if (
                self.tick_from is None or
                len(self.children) == 0 or
                tick_from < self.tick_from):
            self.tick_from = tick_from
        cc = Cell(
            tick_from=tick_from,
            tick_to=tick_to,
            text=text)
        self.add_widget(cc)
        return cc

    def on_parent(self, instance, value):
        if value is not None:
            assert(isinstance(value, Calendar))
            self.calendar = value
            for cell in self.children:
                cell.calendared()

    def do_layout(self, *args):
        half_spacing = self.tick_height / 2
        for cell in self.children:
            if cell.tick_to is not None:
                tick_to = cell.tick_to
            else:
                tick_to = self.calendar.get_max_col_tick()
            cell.y = self.tick_y(tick_to) + half_spacing
            cell.height = ((tick_to - cell.tick_from) * self.tick_height
                           - half_spacing)
        super(Column, self).do_layout(*args)

    def tick_y(self, tick):
        ticks_from_top = tick - self.tick_from
        pix_from_top = ticks_from_top * self.tick_height
        return self.height - pix_from_top


class Calendar(GridLayout):
    bg_r = BoundedNumericProperty(1.0, min=0.0, max=1.0, allownone=True)
    bg_g = BoundedNumericProperty(0.0, min=0.0, max=1.0, allownone=True)
    bg_b = BoundedNumericProperty(0.0, min=0.0, max=1.0, allownone=True)
    bg_a = BoundedNumericProperty(1.0, min=0.0, max=1.0, allownone=True)
    bg_color = ReferenceListProperty(bg_r, bg_g, bg_b, bg_a)
    text_r = BoundedNumericProperty(1.0, min=0.0, max=1.0, allownone=True)
    text_g = BoundedNumericProperty(1.0, min=0.0, max=1.0, allownone=True)
    text_b = BoundedNumericProperty(1.0, min=0.0, max=1.0, allownone=True)
    text_a = BoundedNumericProperty(1.0, min=0.0, max=1.0, allownone=True)
    text_color = ReferenceListProperty(bg_r, bg_g, bg_b, bg_a)
    font_size = NumericProperty(None, allownone=True)
    font_name = StringProperty(None, allownone=True)
    tick = NumericProperty(0)
    min_ticks = NumericProperty(100)
    max_tick = NumericProperty(0)
    tick_height = NumericProperty(10)

    def __init__(self, **kwargs):
        if "col_default_width" not in kwargs:
            kwargs["col_default_width"] = 100
        if "spacing" not in kwargs:
            kwargs["spacing"] = [10, 0]
        super(Calendar, self).__init__(
            rows=1,
            col_force_default=True,
            size_hint=(None, None),
            **kwargs)

    def tick_y(self, tick):
        ticks_from_bot = self.max_tick - tick
        return ticks_from_bot * self.tick_height

    def get_max_col_tick(self):
        return max((self.max_tick, self.min_ticks))

    def do_layout(self, *args):
        super(Calendar, self).do_layout(*args)
        for child in self.children:
            child.do_layout()
