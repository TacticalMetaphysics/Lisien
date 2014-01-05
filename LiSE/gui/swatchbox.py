# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""A graphical selector for "swatches"."""
from kivy.uix.stacklayout import StackLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.properties import (
    ListProperty,
    NumericProperty,
    ObjectProperty)

from gamepiece import ImgStack


class FrobSwatch(Button):
    """A :class:`Button` that contains both an :class:`Image` and
    some text."""
    box = ObjectProperty()
    """The :class:`SwatchBox` that I belong to."""
    img = ObjectProperty()
    """Image to show"""
    tags = ListProperty([])
    """List for use in SwatchBox"""

    def on_img(self, *args):
        if not self.img:
            return
        if self.img.name == '':
            Clock.schedule_once(self.on_image, 0)
            return
        self.ids.imgbox.add_widget(Image(
            texture=self.img.texture,
            pos_hint={'center': 0.5, 'top': 1}))


class TogSwatch(ToggleButton, FrobSwatch):
    def on_box(self, i, v):
        self.bind(state=v.upd_selection)


class SwatchBox(BoxLayout):
    """A collection of :class:`Swatch` used to select several
    graphics at once."""
    closet = ObjectProperty()
    categorized_images = ObjectProperty()
    sellen = NumericProperty(0)
    selection = ListProperty([])

    def upd_selection(self, togswatch, state):
        if state == 'normal':
            while togswatch in self.selection:
                self.selection.remove(togswatch)
        else:
            if togswatch not in self.selection:
                self.selection.append(togswatch)

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

    def finalize(self, *args):
        """For each category in ``cattexlst``, construct a grid of grouped
        Swatches displaying the images therein."""
        if not self.closet and self.categorized_images:
            Clock.schedule_once(self.finalize, 0)
            return
        head = GridLayout(cols=2, size_hint_y=None)
        self.undo_button = Button(text="Undo", on_release=self.undo)
        self.pile = ImgStack(closet=self.closet)
        head.add_widget(self.pile)
        head.add_widget(self.undo_button)
        self.add_widget(head)
        catview = ScrollView(do_scroll_x=False)
        cats = GridLayout(cols=1, size_hint_y=None)
        catview.add_widget(cats)
        self.add_widget(catview)
        i = 0
        h = 0
        for (catname, images) in self.categorized_images:
            l = Label(text=catname.strip('!?'), size_hint_y=None)
            cats.add_widget(l)
            cats.rows_minimum[i] = l.font_size * 2
            h += cats.rows_minimum[i]
            i += 1
            layout = StackLayout(size_hint_y=None)
            for image in images:
                fakelabel = Label(text=image.name)
                fakelabel.texture_update()
                w = fakelabel.texture.size[0]
                kwargs = {
                    'box': self,
                    'text': image.name,
                    'image': image,
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
                    bone = self.closet.skeleton[u"img"][swatch.img_name]
                    if (
                            state == 'down' and
                            swatch.img_name not in self.pile.names):
                        self.pile.bones.append(bone)
                    elif (
                            state == 'normal' and
                            swatch.img_name in self.pile.names):
                        self.pile.bones.remove(bone)
                swatch.bind(state=upd_from_swatch)
            layout.minimum_width = 500
            cats.add_widget(layout)
            cats.rows_minimum[i] = (len(images) / 5) * 100
            h += cats.rows_minimum[i]
            i += 1
        cats.height = h
