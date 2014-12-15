# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
"""Widget to display the contents of a :class:`kivy.atlas.Atlas` in
one :class:`kivy.uix.togglebutton.ToggleButton` apiece, arranged in a
:class:`kivy.uix.stacklayout.StackLayout`. The user selects graphics
from the :class:`Pallet`, and the :class:`Pallet` updates its
``selection`` list to show what the user selected."""
from kivy.properties import (
    DictProperty,
    NumericProperty,
    ObjectProperty,
    ReferenceListProperty,
    StringProperty
)
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.stacklayout import StackLayout


class SwatchButton(ToggleButton):
    name = StringProperty()
    texture = ObjectProperty()


kv = """
<SwatchButton>:
    Image:
        center_x: root.center_x
        texture: root.texture
        size: root.texture.size
    Label:
        center_x: root.center_x
        text: root.name
"""
Builder.load_string(kv)


class Pallet(StackLayout):
    atlas = ObjectProperty()
    swatches = DictProperty({})
    swatch_width = NumericProperty(100)
    swatch_height = NumericProperty(100)
    swatch_size = ReferenceListProperty(swatch_width, swatch_height)

    def on_atlas(self, *args):
        if self.atlas is None:
            return
        self.upd_textures()
        self.atlas.bind(textures=self.upd_textures)

    def upd_textures(self, *args):
        for name in list(self.swatches.keys()):
            if name not in self.atlas.textures:
                self.remove_widget(self.swatches[name])
                del self.swatches[name]
        for (name, tex) in self.atlas.textures.items():
            if name in self.swatches and self.swatches[name] != tex:
                self.remove_widget(self.swatches[name])
            if name not in self.swatches or self.swatches[name] != tex:
                self.swatches[name] = SwatchButton(
                    name=name,
                    texture=tex,
                    size_hint=(None, None),
                    size=self.swatch_size
                )
                self.add_widget(self.swatches[name])
