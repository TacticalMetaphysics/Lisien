# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""A graphical selector for "swatches"."""
from kivy.uix.widget import Widget
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.properties import (
    ListProperty,
    NumericProperty,
    ObjectProperty)

from LiSE.gui.kivybits import TexPile


class FrobSwatch(Button):
    """A :class:`Button` that contains both an :class:`Image` and
    some text."""
    box = ObjectProperty()
    """The :class:`SwatchBox` that I belong to."""
    display_texture = ObjectProperty()
    """The ``texture`` of the :class:`Image` to show."""
    img = ObjectProperty(None, allownone=True)
    """The bone of the img."""
    img_tags = ListProperty([])
    """Tags of the img."""
    xoff = NumericProperty(0)
    yoff = NumericProperty(0)
    stackh = NumericProperty(0)
    """When showing a preview of stacked images, mine will be regarded as
this tall."""

    def on_release(self):
        self.box.selection.append(self)


class TogSwatch(ToggleButton, FrobSwatch):
    def on_state(self, i, v):
        try:
            self.box.selection.remove(self)
        except ValueError:
            if v == 'down':
                self.box.selection.append(self)

    def on_release(self):
        pass


class SwatchBox(ScrollView):
    """A collection of :class:`Swatch` used to select several
    graphics at once."""
    closet = ObjectProperty()
    cattexlst = ListProperty()
    finality = NumericProperty(0)
    selection = ListProperty([])
    sellen = NumericProperty(0)

    def on_closet(self, i, v):
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

    def on_selection(self, i, v):
        lv = len(v)
        if lv > i.sellen:
            self.pile.append(
                v[-1].display_texture, v[-1].xoff,
                v[-1].yoff, v[-1].stackh)
        elif lv < i.sellen:
            try:
                self.pile.pop()
            except IndexError:
                pass
        i.sellen = lv

    def undo(self, *args):
        try:
            swatch = self.selection.pop()
            swatch.state = 'normal'
        except IndexError:
            pass

    def finalize(self):
        """For each category in ``cattexlst``, construct a grid of grouped
        Swatches displaying the images therein."""
        root = GridLayout(cols=1, size_hint_y=None)
        self.add_widget(root)
        head = GridLayout(cols=2, size_hint_y=None)
        self.undo_button = Button(text="Undo", on_release=self.undo)
        self.pile = TexPile()
        head.add_widget(self.pile)
        head.add_widget(self.undo_button)
        root.add_widget(head)
        root.rows_minimum[0] = 100
        self.cat_layouts = []
        i = 1
        for (catname, imgnames) in self.cattexlst:
            l = Label(text=catname.strip('!?'))
            root.add_widget(l)
            root.rows_minimum[i] = l.font_size
            i += 1
            layout = GridLayout(cols=5, size_hint_y=None,
                                row_default_height=100,
                                row_force_default=True)
            self.cat_layouts.append(layout)
            root.add_widget(layout)
            for imgname in imgnames:
                tex = self.closet.get_texture(imgname)
                imgbone = self.closet.skeleton[u"img"][imgname]
                kwargs = {
                    'box': self,
                    'display_texture': tex,
                    'xoff': imgbone.off_x,
                    'yoff': imgbone.off_y,
                    'stackh': imgbone.stacking_height,
                    'text': imgname,
                    'img': imgbone}
                if catname[0] == '!':
                    swatch = TogSwatch(**kwargs)
                elif catname[0] == '?':
                    swatch = FrobSwatch(**kwargs)
                else:
                    kwargs['group'] = catname
                    swatch = TogSwatch(**kwargs)
                layout.add_widget(swatch)
            root.rows_minimum[i] = 100 * (len(layout.children) / layout.cols)
            i += 1
        rootheight = 0
        for v in root.rows_minimum.itervalues():
            rootheight += v
        root.height = rootheight
