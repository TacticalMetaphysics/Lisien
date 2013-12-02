# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""A graphical selector for "swatches"."""
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.properties import (
    ListProperty,
    NumericProperty,
    ObjectProperty)


class Swatch(ToggleButton):
    """A :class:`ToggleButton` that contains both an :class:`Image` and
    some text."""
    display_texture = ObjectProperty()
    """The ``texture`` of the :class:`Image` to show."""


class SwatchBox(ScrollView):
    """A collection of :class:`Swatch` used to select several
    graphics at once."""
    texdict = ObjectProperty()
    cattexlst = ListProperty()
    finality = NumericProperty(0)

    def gen_selection(self):
        """Generator for all those :class:`Swatch` which are pressed
        at the moment."""
        for layout in self.cat_layouts:
            for swatch in layout.children:
                if swatch.state == 'down':
                    yield swatch

    def on_texdict(self, i, v):
        """Increment finality counter."""
        self.finality += 1

    def on_cattexlst(self, i, v):
        """Increment finality counter."""
        self.finality += 1

    def on_parent(self, i, v):
        """Increment finality counter."""
        self.finality += 1

    def on_finality(self, i, v):
        """If final enough, finalize."""
        if v == 3:
            self.finalize()

    def finalize(self):
        """For each category in ``cattexdict``, construct a grid of grouped
        Swatches displaying the images therein."""
        root = GridLayout(cols=1, size_hint_y=None)
        self.add_widget(root)
        self.cat_layouts = []
        i = 0
        for (catname, imgnames) in self.cattexlst:
            l = Label(text=catname)
            root.add_widget(l)
            root.rows_minimum[i] = l.font_size
            i += 1
            layout = GridLayout(cols=5, size_hint_y=None,
                                row_default_height=100,
                                row_force_default=True)
            self.cat_layouts.append(layout)
            root.add_widget(layout)
            for imgname in imgnames:
                swatch = Swatch(
                    display_texture=self.texdict[imgname],
                    text=imgname,
                    group=catname)
                layout.add_widget(swatch)
            root.rows_minimum[i] = 100 * (len(layout.children) / layout.cols)
            i += 1
        rootheight = 0
        for v in root.rows_minimum.itervalues():
            rootheight += v
        root.height = rootheight
