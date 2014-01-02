# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""A graphical selector for "swatches"."""
from kivy.uix.stacklayout import StackLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.label import Label
from kivy.properties import (
    AliasProperty,
    StringProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty)

from LiSE.gui.kivybits import ClosetTextureStack


class FrobSwatch(Button):
    """A :class:`Button` that contains both an :class:`Image` and
    some text."""
    box = ObjectProperty()
    """The :class:`SwatchBox` that I belong to."""
    img_name = StringProperty()
    img_tags = ListProperty([])
    """Tags of the img."""
    xoff = NumericProperty(0)
    yoff = NumericProperty(0)
    stackh = NumericProperty(0)
    """When showing a preview of stacked images, mine will be regarded as
this tall."""
    display_texture = AliasProperty(
        lambda self: self.box.closet.get_texture(self.img_name)
        if self.box is not None and self.img_name != '' else None,
        lambda self, v: None,
        bind=('box', 'img_name'))
    img = AliasProperty(
        lambda self: self.box.closet.skeleton[u"img"][self.img_name]
        if self.box is not None and self.img_name != '' else None,
        lambda self, v: None,
        bind=('box', 'img_name'))


class TogSwatch(ToggleButton, FrobSwatch):
    def on_box(self, i, v):
        self.bind(state=v.upd_selection)


class SwatchBox(BoxLayout):
    """A collection of :class:`Swatch` used to select several
    graphics at once."""
    closet = ObjectProperty()
    cattexlst = ListProperty()
    finality = NumericProperty(0)
    sellen = NumericProperty(0)
    selection = ListProperty([])

    def upd_selection(self, togswatch, state):
        if state == 'normal':
            while togswatch in self.selection:
                self.selection.remove(togswatch)
        else:
            if togswatch not in self.selection:
                self.selection.append(togswatch)

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
        head = GridLayout(cols=2, size_hint_y=None)
        self.undo_button = Button(text="Undo", on_release=self.undo)
        self.pile = ClosetTextureStack(closet=self.closet)
        head.add_widget(self.pile)
        head.add_widget(self.undo_button)
        self.add_widget(head)
        catview = ScrollView(do_scroll_x=False)
        cats = GridLayout(cols=1, size_hint_y=None)
        catview.add_widget(cats)
        self.add_widget(catview)
        i = 0
        h = 0
        for (catname, imgnames) in self.cattexlst:
            l = Label(text=catname.strip('!?'), size_hint_y=None)
            cats.add_widget(l)
            cats.rows_minimum[i] = l.font_size * 2
            h += cats.rows_minimum[i]
            i += 1
            layout = StackLayout(size_hint_y=None)
            for imgname in imgnames:
                imgbone = self.closet.skeleton[u"img"][imgname]
                fakelabel = Label(text=imgname)
                fakelabel.texture_update()
                w = fakelabel.texture.size[0]
                kwargs = {
                    'box': self,
                    'xoff': imgbone.off_x,
                    'yoff': imgbone.off_y,
                    'stackh': imgbone.stacking_height,
                    'text': imgname,
                    'img_name': imgname,
                    'width': w + l.font_size * 2}
                if catname[0] == '!':
                    swatch = TogSwatch(**kwargs)
                elif catname[0] == '?':
                    swatch = FrobSwatch(**kwargs)
                else:
                    kwargs['group'] = catname
                    swatch = TogSwatch(**kwargs)
                layout.add_widget(swatch)

                def upd_from_swatch(swatch, state):
                    if (
                            state == 'down' and
                            swatch.img_name not in self.pile.names):
                        self.pile.names.append(swatch.img_name)
                    elif (
                            state == 'normal' and
                            swatch.img_name in self.pile.names):
                        self.pile.names.remove(swatch.img_name)
                swatch.bind(state=upd_from_swatch)
            layout.minimum_width = 500
            cats.add_widget(layout)
            cats.rows_minimum[i] = (len(imgnames) / 5) * 100
            h += cats.rows_minimum[i]
            i += 1
        cats.height = h
