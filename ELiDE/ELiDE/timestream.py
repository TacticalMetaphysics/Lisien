from ELiDE.util import trigger

from kivy.lang import Builder
from kivy.properties import BooleanProperty, NumericProperty
from kivy.graphics import Color, Line
from kivy.uix.recycleview import RecycleView
from kivy.uix.label import Label
from kivy.uix.widget import Widget


class ThornyRectangle(Label):
    left_margin = NumericProperty(10)
    right_margin = NumericProperty(10)
    top_margin = NumericProperty(10)
    bottom_margin = NumericProperty(10)

    left_thorn = BooleanProperty(True)
    right_thorn = BooleanProperty(True)
    top_thorn = BooleanProperty(True)
    bottom_thorn = BooleanProperty(True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(
            **{prop: self._trigger_redraw for prop in
               ['pos', 'size', 'left_margin', 'right_margin',
                'top_margin', 'bottom_margin', 'left_thorn',
                'right_thorn', 'top_thorn', 'bottom_thorn']}
        )
        self._trigger_redraw()

    def collide_point(self, x, y):
        return (
            self.x + self.left_margin < x < self.right - self.right_margin and
            self.y + self.bottom_margin < y < self.top - self.top_margin
        )

    def _redraw_line(self, enabled, name, point_lambda):
        if enabled:
            points = point_lambda()
            if hasattr(self, name):
                the_line = getattr(self, name)
                the_line.points = points
            else:
                the_line = Line(points=points)
            if the_line not in self.canvas.children:
                self.canvas.add(the_line)
        elif hasattr(self, name) and \
                getattr(self, name) in self.canvas.children:
            self.canvas.remove(getattr(self, name))
            delattr(self, name)

    def _get_left_line_points(self):
        return [
            self.x, self.center_y,
            self.x + self.left_margin, self.center_y
        ]

    def _get_right_line_points(self):
        return [
            self.right - self.right_margin, self.center_y,
            self.right, self.center_y
        ]

    def _get_top_line_points(self):
        return [
            self.center_x, self.top,
            self.center_x, self.top - self.top_margin
        ]

    def _get_bottom_line_points(self):
        return [
            self.center_x, self.y,
            self.center_x, self.y + self.bottom_margin
        ]

    def _redraw(self, *args):
        self._color = Color(rgba=[1, 1, 1, 1])
        if self._color not in self.canvas.children:
            self.canvas.add(self._color)
        rectpoints = [
            self.x + self.left_margin, self.y + self.bottom_margin,
            self.right - self.right_margin, self.y + self.bottom_margin,
            self.right - self.right_margin, self.top - self.top_margin,
            self.x + self.left_margin, self.top - self.top_margin,
            self.x + self.left_margin, self.y + self.bottom_margin
        ]
        if hasattr(self, '_rect'):
            self._rect.points = rectpoints
        else:
            self._rect = Line(points=rectpoints)
            self.canvas.add(self._rect)
        self._redraw_line(self.left_thorn, '_left_line',
                          self._get_left_line_points)
        self._redraw_line(self.right_thorn, '_right_line',
                          self._get_right_line_points)
        self._redraw_line(self.top_thorn, '_top_line',
                          self._get_top_line_points)
        self._redraw_line(self.bottom_thorn, '_bot_line',
                          self._get_bottom_line_points)

    _trigger_redraw = trigger(_redraw)


class Cross(Widget):
    draw_left = BooleanProperty(True)
    draw_right = BooleanProperty(True)
    draw_up = BooleanProperty(True)
    draw_down = BooleanProperty(True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(
            draw_left=self._trigger_redraw,
            draw_right=self._trigger_redraw,
            draw_up=self._trigger_redraw,
            draw_down=self._trigger_redraw,
            size=self._trigger_redraw,
            pos=self._trigger_redraw
        )

    def _draw_line(self, enabled, name, get_points):
        if enabled:
            points = get_points()
            if hasattr(self, name):
                getattr(self, name).points = points
            else:
                the_line = Line(points=points)
                setattr(self, name, the_line)
                self.canvas.add(the_line)
        elif hasattr(self, name):
            the_line = getattr(self, name)
            if the_line in self.canvas:
                self.canvas.remove(the_line)
            delattr(self, name)

    def _get_left_points(self):
        return [
            self.x, self.center_y,
            self.center_x, self.center_y
        ]

    def _get_right_points(self):
        return [
            self.center_x, self.center_y,
            self.right, self.center_y
        ]

    def _get_up_points(self):
        return [
            self.center_x, self.center_y,
            self.center_x, self.top
        ]

    def _get_down_points(self):
        return [
            self.center_x, self.center_y,
            self.center_x, self.y
        ]

    def _redraw(self, *args):
        self._draw_line(self.draw_left, '_left_line', self._get_left_points)
        self._draw_line(self.draw_right, '_right_line', self._get_right_points)
        self._draw_line(self.draw_up, '_up_line', self._get_up_points)
        self._draw_line(self.draw_down, '_down_line', self._get_down_points)

    _trigger_redraw = trigger(_redraw)


class Timestream(RecycleView):
    pass


Builder.load_string("""
<Timestream>:
    key_viewclass: 'widget'
    effect_cls: 'ScrollEffect'
    RecycleGridLayout:
        cols: 30
        default_width: 100
        default_height: 100
        default_size_hint: None, None
        height: self.minimum_height
        width: self.minimum_width
        size_hint: None, None
""")

if __name__ == '__main__':
    from kivy.base import runTouchApp
    from kivy.core.window import Window
    from kivy.modules import inspector
    from kivy.uix.boxlayout import BoxLayout

    ts = Timestream()
    layout = BoxLayout()
    layout.add_widget(ts)
    ts.data = [{'text': str(i), 'widget': 'ThornyRectangle' if i % 5 != 0 else 'Cross'} for i in range(500)]
    inspector.create_inspector(Window, layout)
    runTouchApp(layout)
