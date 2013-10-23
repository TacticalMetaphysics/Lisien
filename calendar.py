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
from closet import load_closet


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
    branch = NumericProperty()
    tl_line = ObjectProperty(allownone=True)
    tl_wedge = ObjectProperty(allownone=True)
    tl_color = (0.0, 1.0, 0.0, 1.0)
    tl_width = 16
    tl_height = 8

    def on_parent(self, *args):
        (line_points, wedge_points) = self.get_tl_points(0)
        self.canvas.after.add(Color(*self.tl_color))
        self.tl_line = Line(points=line_points)
        self.canvas.after.add(self.tl_line)
        self.tl_wedge = Triangle(points=wedge_points)
        self.canvas.after.add(self.tl_wedge)
        calendar = self.parent
        it = calendar.closet.skeleton["thing_location"][
            "Physical"]["mom"][self.branch].iterrows()
        prev = next(it)
        for rd in it:
            cc = CalendarCell(
                tick_from=prev["tick_from"],
                tick_to=rd["tick_from"],
                text=prev["location"])
            self.add_widget(cc)
        cc = CalendarCell(
            tick_from=prev["tick_from"],
            tick_to=None,
            text=prev["location"])
        self.add_widget(cc)
        calendar.closet.bind(tick=self.upd_tl)

    def upd_tl(self, *args):
        calendar = self.parent
        (line_points, wedge_points) = self.get_tl_points(calendar.closet.tick)
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
        for cell in self.children:
            cell.y = self.parent.tick_y(cell.tick_to)
            if cell.tick_to is None:
                cell.size = (
                    self.width, cell.tick_from * self.parent.tick_height)
            else:
                cell.size = (
                    self.width,
                    (cell.tick_to - cell.tick_from) * self.parent.tick_height)


class Calendar(BoxLayout):
    closet = ObjectProperty()
    tick_height = 10

    def __init__(self, **kwargs):
        BoxLayout.__init__(
            self, orientation='horizontal', **kwargs)
        for branch in kwargs["branches"]:
            self.add_widget(CalendarColumn(branch=branch))

    def tick_y(self, tick):
        # it seems like timestream still has hi_tick = 0 when there
        # are other events after
        if tick is None:
            return 0
        max_tick = self.closet.timestream.hi_tick
        ticks_from_bot = max_tick - tick
        return ticks_from_bot * self.tick_height


class CalDemoApp(App):
    def build(self):
        closet = load_closet('default.sqlite')
        closet.load_board('Physical')
        closet.load_charsheet('household')
        Clock.schedule_interval(lambda dt: closet.time_travel_inc_tick(), 1)
        return Calendar(size=(800, 600), closet=closet, branches=[0])


CalDemoApp().run()
