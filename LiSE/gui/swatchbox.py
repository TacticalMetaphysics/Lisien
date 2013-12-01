# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""A graphical selector for "swatches"."""
from kivy.uix.gridlayout import GridLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.properties import (
    NumericProperty,
    ObjectProperty)


class Swatch(ToggleButton):
    """A :class:`ToggleButton` that contains both an :class:`Image` and
    some text."""
    display_texture = ObjectProperty()
    """The ``texture`` of the :class:`Image` to show."""


class SwatchBox(GridLayout):
    """A collection of :class:`Swatch` used to select several
    graphics at once."""
    texdict = ObjectProperty()
    finality = NumericProperty(0)

    def gen_selection(self):
        """Generator for all those :class:`Swatch` which are pressed
        at the moment."""
        for child in self.children:
            if child.state == 'down':
                yield child

    def on_texdict(self, i, v):
        """Increment finality counter."""
        self.finality += 1

    def on_parent(self, i, v):
        """Increment finality counter."""
        self.finality += 1

    def on_finality(self, i, v):
        """If I have a ``texdict`` and a ``parent``, finalize."""
        if v == 2:
            self.finalize()

    def finalize(self):
        """Add one :class:`Swatch` per texture in my ``texdict``."""
        for (key, val) in self.texdict.iteritems():
            self.add_widget(Swatch(
                text=key, display_texture=val))
