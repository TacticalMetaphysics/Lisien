from kivy.uix.layout import Layout
from kivy.properties import ObjectProperty

from ..imagestackproxy import ImageStackProxy


class GridSpot(ImageStackProxy, Layout):
    default_image_paths = ['atlas://rltiles/floor.atlas/floor-stone']
    board = ObjectProperty()

    def do_layout(self, *args):
        for child in self.children:
            child.pos = self.pos
            child.size = self.size
            for rect in child._texture_rectangles.values():
                rect.size = self.size

    def add_widget(self, widget, index=0, canvas='after'):
        self._trigger_layout()
        return super().add_widget(widget, index, canvas)

