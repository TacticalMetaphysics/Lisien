from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput


class GeneratorDialog(BoxLayout):
    board = ObjectProperty()


class NumberInput(TextInput):
    pass


Builder.load_string("""
<GeneratorDialog>:
    BoxLayout:
        orientation: 'vertical'
        BoxLayout:
            orientation: 'horizontal'
            Text""")