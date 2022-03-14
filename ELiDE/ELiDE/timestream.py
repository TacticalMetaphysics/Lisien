from ELiDE.util import trigger

from kivy.lang import Builder
from kivy.properties import BooleanProperty, NumericProperty
from kivy.graphics import Color, Line
from kivy.uix.recycleview import RecycleView
from kivy.uix.label import Label


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


class Timestream(RecycleView):
    pass


Builder.load_string("""
<Timestream>:
    viewclass: 'ThornyRectangle'
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
    ts.data = [{'text': str(i)} for i in range(500)]
    inspector.create_inspector(Window, layout)
    runTouchApp(layout)
