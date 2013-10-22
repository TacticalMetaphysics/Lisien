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
from kivy.clock import Clock
from kivy.app import App


fontsize = 10


class CalendarCell(BoxLayout):
    bg_color = ObjectProperty(Color(1.0, 0.0, 0.0, 1.0))
    bg_rect = ObjectProperty(allownone=True)
    tick_from = NumericProperty()
    tick_to = NumericProperty()
    text = StringProperty()

    def __init__(self, **kwargs):
        BoxLayout.__init__(self, size_hint=(None, None), **kwargs)

    def on_parent(self, *args):
        column = self.parent
        if column.max_tick < self.tick_to:
            column.max_tick = self.tick_to
        self.bind(tick_to=column.upd_max_tick)

    def calendared(self, *args):
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
            halign='left',
            valign='top',
            text_size=self.size)
        self.add_widget(label)

        def resize_label(*args):
            label.text_size = self.size
        self.bind(size=resize_label)

        column = self.parent
        column._trigger_layout()


class CalendarColumn(RelativeLayout):
    tl_line = ObjectProperty(allownone=True)
    tl_wedge = ObjectProperty(allownone=True)
    tl_color = (0.0, 1.0, 0.0, 1.0)
    tl_width = 16
    tl_height = 8
    max_tick = NumericProperty(0)

    def __init__(self, **kwargs):
        RelativeLayout.__init__(self, **kwargs)

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
        if self.max_tick > calendar.max_tick:
            calendar.max_tick = self.max_tick
        self.bind(max_tick=calendar.upd_max_tick)
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

    def upd_max_tick(self, instance, value):
        if value > self.max_tick:
            self.max_tick = value

    def do_layout(self, *args):
        tick_height = self.parent.height / self.parent.max_tick
        for cell in self.children:
            cell.y = self.parent.tick_y(cell.tick_to)
            cell.size = (
                self.width,
                (cell.tick_to - cell.tick_from) * tick_height)


class Calendar(BoxLayout):
    tick = NumericProperty(0)
    max_tick = NumericProperty(0)

    def __init__(self, **kwargs):
        BoxLayout.__init__(
            self, orientation='horizontal', spacing=10, **kwargs)
        if "columns" in kwargs:
            for column in kwargs["columns"]:
                self.add_widget(column)

    def tick_y(self, tick):
        tick_height = self.height / self.max_tick
        ticks_from_bot = self.max_tick - tick
        return ticks_from_bot * tick_height

    def upd_max_tick(self, instance, value):
        if value > self.max_tick:
            self.max_tick = value


class CalDemoApp(App):
    def build(self):
        cells0 = [
            CalendarCell(tick_from=0, tick_to=10, text="wake up"),
            CalendarCell(tick_from=11, tick_to=20, text="go to school"),
            CalendarCell(tick_from=30, tick_to=50, text="save the world")]
        cells1 = [
            CalendarCell(tick_from=0, tick_to=20, text="fall asleep"),
            CalendarCell(tick_from=21, tick_to=50, text="dream randomly"),
            CalendarCell(tick_from=51, tick_to=55, text="feel horrible")]
        col0 = CalendarColumn(cells=cells0)
        col1 = CalendarColumn(cells=cells1)
        cal = Calendar(columns=[col0, col1])

        def inc_tick(*args):
            cal.tick += 1
        Clock.schedule_interval(inc_tick, 1)
        return cal


CalDemoApp().run()
