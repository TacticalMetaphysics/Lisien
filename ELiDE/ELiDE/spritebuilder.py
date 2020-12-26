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
from .pallet import Pallet, PalletBox
from .kivygarden.texturestack import ImageStack
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.properties import (
    ListProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty
)
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen
from .util import trigger
# TODO: let the user import their own sprite art


class SpriteSelector(BoxLayout):
    prefix = StringProperty()
    pallets = ListProperty()
    imgpaths = ListProperty([])
    default_imgpaths = ListProperty()
    preview = ObjectProperty()

    def on_prefix(self, *args):
        if 'textbox' not in self.ids:
            Clock.schedule_once(self.on_prefix, 0)
            return
        self.ids.textbox.text = self.prefix

    def on_imgpaths(self, *args):
        if not self.preview:
            Logger.debug(
                "SpriteSelector: no preview"
            )
            Clock.schedule_once(self.on_imgpaths, 0)
            return
        if hasattr(self, '_imgstack'):
            self.preview.remove_widget(self._imgstack)
        self._imgstack = ImageStack(
            paths=self.imgpaths,
            x=self.preview.center_x - 16,
            y=self.preview.center_y - 16
        )
        self.preview.add_widget(self._imgstack)

    def on_pallets(self, *args):
        for pallet in self.pallets:
            pallet.fbind('selection', self._upd_imgpaths)

    def _upd_imgpaths(self, *args):
        imgpaths = []
        for pallet in self.pallets:
            if pallet.selection:
                for selected in pallet.selection:
                    imgpaths.append(
                        'atlas://{}/{}'.format(
                            pallet.filename,
                            selected.text
                        )
                    )
        self.imgpaths = imgpaths if imgpaths else self.default_imgpaths


class SpriteBuilder(ScrollView):
    prefix = StringProperty()
    imgpaths = ListProperty()
    default_imgpaths = ListProperty()
    data = ListProperty()
    labels = ListProperty()
    pallets = ListProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(
            data=self._trigger_update
        )

    def update(self, *args):
        if self.data is None:
            return
        if not self.canvas:
            Clock.schedule_once(self.update, 0)
            return
        if not hasattr(self, '_palbox'):
            self._palbox = PalletBox(
                orientation='vertical',
                size_hint_y=None
            )
            self.add_widget(self._palbox)
        else:
            self._palbox.clear_widgets()
        if hasattr(self._palbox, '_bound_width'):
            for uid in self._palbox._bound_width:
                self._palbox.unbind_uid('width', uid)
            del self._palbox._bound_width
        self.labels = []
        for pallet in self.pallets:
            if hasattr(pallet, '_bound_minimum_height'):
                pallet.unbind_uid('minimum_height', pallet._bound_minimum_height)
                del pallet._bound_minimum_height
            if hasattr(pallet, '_bound_height'):
                pallet.unbind_uid('height', pallet._bound_height)
                del pallet._bound_height
        self.pallets = []
        for (text, filename) in self.data:
            label = Label(
                text=text,
                size_hint=(None, None),
                halign='center'
            )
            label.texture_update()
            label.height = label.texture.height
            label.width = self._palbox.width
            pallet = Pallet(
                filename=filename,
                size_hint=(None, None)
            )
            pallet.width = self._palbox.width
            self._palbox._bound_width = [
                self._palbox.fbind('width', label.setter('width')),
                self._palbox.fbind('width', pallet.setter('width'))
            ]
            pallet.height = pallet.minimum_height
            pallet._bound_minimum_height = pallet.fbind('minimum_height', pallet.setter('height')),
            pallet._bound_height = pallet.fbind('height', self._trigger_reheight)
            self.labels.append(label)
            self.pallets.append(pallet)
        n = len(self.labels)
        assert(n == len(self.pallets))
        for i in range(0, n):
            self._palbox.add_widget(self.labels[i])
            self._palbox.add_widget(self.pallets[i])
    _trigger_update = trigger(update)

    def reheight(self, *args):
        self._palbox.height = sum(
            wid.height for wid in self.labels + self.pallets
        )
    _trigger_reheight = trigger(reheight)


class SpriteDialog(BoxLayout):
    toggle = ObjectProperty()
    prefix = StringProperty()
    imgpaths = ListProperty()
    default_imgpaths = ListProperty()
    data = ListProperty()
    pallet_box_height = NumericProperty()

    def pressed(self):
        self.prefix = self.ids.selector.prefix
        self.imgpaths = self.ids.selector.imgpaths
        self.toggle()


class PawnConfigDialog(SpriteDialog):
    pass


class SpotConfigDialog(SpriteDialog):
    pass


class PawnConfigScreen(Screen):
    toggle = ObjectProperty()
    data = ListProperty()
    imgpaths = ListProperty()


class SpotConfigScreen(Screen):
    toggle = ObjectProperty()
    data = ListProperty()
    imgpaths = ListProperty()


Builder.load_string("""
<SpriteDialog>:
    orientation: 'vertical'
    SpriteBuilder:
        id: builder
        prefix: root.prefix
        default_imgpaths: root.default_imgpaths
        imgpaths: root.imgpaths
        data: root.data
    SpriteSelector:
        id: selector
        textbox: textbox
        size_hint_y: 0.1
        prefix: root.prefix
        default_imgpaths: root.default_imgpaths
        imgpaths: root.imgpaths
        pallets: builder.pallets
        preview: preview
        TextInput:
            id: textbox
            multiline: False
            write_tab: False
            hint_text: 'Enter name prefix'
        Widget:
            id: preview
        Button:
            text: 'OK'
            on_release: root.pressed()
<PawnConfigScreen>:
    name: 'pawncfg'
    imgpaths: dialog.imgpaths
    PawnConfigDialog:
        id: dialog
        toggle: root.toggle
        default_imgpaths: ['atlas://base.atlas/unseen']
        data: root.data
<SpotConfigScreen>:
    name: 'spotcfg'
    imgpaths: dialog.imgpaths
    SpotConfigDialog:
        id: dialog
        toggle: root.toggle
        default_imgpaths: ['atlas://rltiles/floor/floor-stone']
        data: root.data
""")
