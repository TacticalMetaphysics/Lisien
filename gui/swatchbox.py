"""A graphical selector for "swatches," which may be colors, textures,
or standalone sprites."""
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.properties import ObjectProperty


class Swatch(Button):
    texture = ObjectProperty()
    label = ObjectProperty()


class SwatchBox(GridLayout):
    texdict = ObjectProperty()
    selected = ListProperty([])

    def on_texdict(self, i, v):
        for (key, val) in v.iteritems():
            self.add_widget(Swatch(label=key, texture=val))
