# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Widget to display the contents of a :class:`kivy.atlas.Atlas` in
one :class:`kivy.uix.togglebutton.ToggleButton` apiece, arranged in a
:class:`kivy.uix.stacklayout.StackLayout`. The user selects graphics
from the :class:`Pallet`, and the :class:`Pallet` updates its
``selection`` list to show what the user selected."""
from kivy.clock import Clock
from kivy.properties import (
    DictProperty,
    NumericProperty,
    ObjectProperty,
    OptionProperty,
    ListProperty,
    ReferenceListProperty,
    StringProperty
)
from kivy.resources import resource_find
from kivy.atlas import Atlas
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.stacklayout import StackLayout
from kivy.graphics import Rectangle
from .util import trigger


class SwatchButton(ToggleButton):
    """Toggle button containing a texture and its name, which, when
    toggled, will report the fact to the :class:`Pallet` it's in.

    """
    tex = ObjectProperty()
    """Texture to display here"""

    def on_state(self, *args):
        if self.state == 'down':
            assert(self not in self.parent.selection)
            if self.parent.selection_mode == 'single':
                for wid in self.parent.selection:
                    if wid is not self:
                        wid.state = 'normal'
                self.parent.selection = [self]
            else:
                self.parent.selection.append(self)
        else:
            if self in self.parent.selection:
                self.parent.selection.remove(self)

    def on_parent(self, *args):
        if not self.canvas or not self.tex:
            Clock.schedule_once(self.on_parent, 0)
            return
        self.canvas.after.clear()
        with self.canvas.after:
            self._img_rect = Rectangle(
                pos=self._get_img_rect_pos(),
                size=self.tex.size,
                texture=self.tex
            )
        self.fbind('pos', self._upd_img_rect_pos)
        self.fbind('tex', self._upd_img_rect_tex)

    def _get_img_rect_pos(self):
        x, y = self.pos
        width, height = self.size
        tw, th = self.tex.size
        return (x + (width / 2 - tw / 2),
                y + height - th)

    @trigger
    def _upd_img_rect_pos(self, *args):
        self._img_rect.pos = self._get_img_rect_pos()

    @trigger
    def _upd_img_rect_tex(self, *args):
        self._img_rect.texture = self.tex
        self._img_rect.size = self.tex.size
        self._img_rect.pos = self._get_img_rect_pos()

    def on_size(self, *args):
        self.text_size = self.size


class Pallet(StackLayout):
    """Many :class:`SwatchButton`, gathered from an :class:`kivy.atlas.Atlas`."""
    atlas = ObjectProperty()
    """:class:`kivy.atlas.Atlas` object I'll make :class:`SwatchButton` from."""
    filename = StringProperty()
    """Path to an atlas; will construct :class:`kivy.atlas.Atlas` when set"""
    swatches = DictProperty({})
    """:class:`SwatchButton` widgets here, keyed by name of their graphic"""
    swatch_width = NumericProperty(100)
    """Width of each and every :class:`SwatchButton` here"""
    swatch_height = NumericProperty(75)
    """Height of each and every :class:`SwatchButton` here"""
    swatch_size = ReferenceListProperty(swatch_width, swatch_height)
    """Size of each and every :class:`SwatchButton` here"""
    selection = ListProperty([])
    """List of :class:`SwatchButton`s that are selected"""
    selection_mode = OptionProperty('single', options=['single', 'multiple'])
    """Whether to allow only a 'single' selected :class:`SwatchButton` (default), or 'multiple'"""

    def on_selection(self, *args):
        Logger.debug(
            'Pallet: {} got selection {}'.format(
                self.filename, self.selection
            )
        )

    def on_filename(self, *args):
        if not self.filename:
            return
        resource = resource_find(self.filename)
        if not resource:
            raise ValueError("Couldn't find atlas: {}".format(self.filename))
        self.atlas = Atlas(resource)

    def on_atlas(self, *args):
        if self.atlas is None:
            return
        self.upd_textures()
        self.atlas.bind(textures=self._trigger_upd_textures)

    def upd_textures(self, *args):
        """Create one :class:`SwatchButton` for each texture"""
        if self.canvas is None:
            Clock.schedule_once(self.upd_textures, 0)
            return
        swatches = self.swatches
        atlas_textures = self.atlas.textures
        remove_widget = self.remove_widget
        add_widget = self.add_widget
        swatch_size = self.swatch_size
        for name, swatch in list(swatches.items()):
            if name not in atlas_textures:
                remove_widget(swatch)
                del swatches[name]
        for (name, tex) in atlas_textures.items():
            if name in swatches and swatches[name] != tex:
                remove_widget(swatches[name])
            if name not in swatches or swatches[name] != tex:
                swatches[name] = SwatchButton(
                    text=name,
                    tex=tex,
                    size_hint=(None, None),
                    size=swatch_size
                )
                add_widget(swatches[name])

    def _trigger_upd_textures(self, *args):
        if hasattr(self, '_scheduled_upd_textures'):
            Clock.unschedule(self._scheduled_upd_textures)
        self._scheduled_upd_textures = Clock.schedule_once(self._trigger_upd_textures)

kv = """
<Pallet>:
    orientation: 'lr-tb'
    padding_y: 100
    size_hint: (None, None)
    height: self.minimum_height
"""
Builder.load_string(kv)


class PalletBox(BoxLayout):
    pallets = ListProperty()
