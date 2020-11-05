from kivy.properties import NumericProperty, ObjectProperty, OptionProperty
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from .util import trigger
from LiSE.character import grid_2d_8graph
from networkx import grid_2d_graph


class GeneratorDialog(BoxLayout):
    graphboard = ObjectProperty()
    gridboard = ObjectProperty()
    xval = NumericProperty()
    yval = NumericProperty()
    directions = OptionProperty(4, options=[4, 8])
    dismiss = ObjectProperty()

    def generate(self, *args):
        x = int(self.xval)
        y = int(self.yval)
        if x < 1 or y < 1:
            return
        if self.directions == 4:
            self.graphboard.character.copy_from(grid_2d_graph(x, y))
        else:
            assert self.directions == 8
            self.graphboard.character.copy_from(grid_2d_8graph(x, y))
        self.graphboard.update()
        self.gridboard.update()
    _trigger_generate = trigger(generate)


Builder.load_string("""
<GeneratorDialog>:
    orientation: 'vertical'
    BoxLayout:
        orientation: 'horizontal'
        MenuIntInput:
            id: input_x
            hint_text: str(root.xval) if root.xval else 'x'
            set_value: root.setter('xval')
        Label:
            text: 'x'
            size_hint_x: 0.1
        MenuIntInput:
            id: input_y
            hint_text: str(root.yval) if root.yval else 'y'
            set_value: root.setter('yval')
    BoxLayout:
        Button:
            text: 'Cancel'
            on_release:
                root.dismiss()
        Button:
            text: 'OK'
            on_release:
                root._trigger_generate()
                root.dismiss()""")