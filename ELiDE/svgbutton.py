from kivy.resources import resource_find
from kivy.properties import StringProperty
from kivy.graphics.svg import Svg
from kivy.uix import Button


class SvgButton(Button):
    filename = StringProperty()

    def on_filename(self, *args):
        if not self.filename:
            return
        self.svg = Svg(resource_find(self.filename))
        self.svg.size = self.size
        self.canvas.add(self.svg)

    def on_size(self, *args):
        if hasattr(self, 'svg'):
            self.svg.size = self.size
