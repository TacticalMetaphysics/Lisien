from kivy.properties import NumericProperty, ObjectProperty, OptionProperty
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from .util import trigger


class GeneratorDialog(BoxLayout):
    board = ObjectProperty()
    x = NumericProperty()
    y = NumericProperty()
    directions = OptionProperty(4, options=[4, 8])

    def generate(self):
        x = int(self.x)
        y = int(self.y)
        if x < 1 or y < 1:
            raise ValueError("Dimensions must be positive")
        if self.directions == 4:
            self.board.character.grid_2d_graph(x, y)
        else:
            assert self.directions == 8
            self.board.character.grid_2d_8graph(x, y)
        self.board.update()
    _trigger_generate = trigger(generate)


Builder.load_string("""
<GeneratorDialog>:
    BoxLayout:
        orientation: 'vertical'
        BoxLayout:
            orientation: 'horizontal'
            MenuIntInput:
                id: input_x
                setter: root.setter('x')
            Label:
                text: 'x'
                size_hint_x: None
                pos_hint: {'center_y': 0.5}
                texture_size: self.size
            MenuIntInput:
                id: input_y
                setter: root.setter('y')""")