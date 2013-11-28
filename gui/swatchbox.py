# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""A graphical selector for "swatches," which may be colors, textures,
or standalone sprites."""
from kivy.uix.gridlayout import GridLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.properties import (
    NumericProperty,
    ObjectProperty)


class Swatch(ToggleButton):
    display_texture = ObjectProperty()


class SwatchBox(GridLayout):
    texdict = ObjectProperty()
    style = ObjectProperty()
    finality = NumericProperty(0)

    def gen_selection(self):
        for child in self.children:
            if child.state == 'down':
                yield child

    def on_texdict(self, i, v):
        self.finality += 1

    def on_style(self, i, v):
        self.finality += 1

    def on_parent(self, i, v):
        self.finality += 1

    def on_finality(self, i, v):
        if v == 3:
            self.finalize()

    def finalize(self):
        for (key, val) in self.texdict.iteritems():
            self.add_widget(Swatch(
                text=key, display_texture=val,
                font_name=self.style.fontface,
                font_size=self.style.fontsize))
