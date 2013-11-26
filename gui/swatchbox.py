"""A graphical selector for "swatches," which may be colors, textures,
or standalone sprites."""
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.image import Image
from kivy.uix.scrollview import ScrollView
from kivy.uix.label import Label
from kivy.properties import (
    NumericProperty,
    StringProperty,
    ObjectProperty)


class Swatch(ToggleButton):
    display_text = StringProperty()
    display_texture = ObjectProperty()


class SwatchBox(GridLayout):
    texdict = ObjectProperty()
    style = ObjectProperty()
    finality = NumericProperty(0)

    def on_texdict(self, i, v):
        self.finality += 1

    def on_style(self, i, v):
        self.finality += 1

    def on_canvas(self, i, v):
        self.finality += 1

    def on_finality(self, i, v):
        if v == 3:
            self.finalize()

    def finalize(self):
        for (key, val) in self.texdict.iteritems():
            self.add_widget(Swatch(
                display_text=key, display_texture=val,
                background_color=self.style.bg_active.rgba,
                color=self.style.text_active.rgba,
                disabled_color=self.style.text_inactive.rgba,
                font_name=self.style.fontface,
                font_size=self.style.fontsize))
