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
from kivy.clock import Clock
from kivy.properties import (
    DictProperty,
    NumericProperty,
    StringProperty,
    ObjectProperty
)
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.textinput import TextInput
from .statlist import BaseStatListView


class FloatInput(TextInput):
    def insert_text(self, s, from_undo=False):
        return super().insert_text(
            ''.join(c for c in s if c in '0123456789.'),
            from_undo
        )


class ControlTypePicker(Button):
    app = ObjectProperty()
    key = ObjectProperty()
    mainbutton = ObjectProperty()
    dropdown = ObjectProperty()
    set_control = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build()

    def set_value(self, k, v):
        if v is None:
            del self.app.selected_proxy[k]
        else:
            self.app.selected_proxy[k] = v

    def build(self, *args):
        if None in (
                self.key,
                self.set_control
        ):
            Clock.schedule_once(self.build, 0)
            return
        self.mainbutton = None
        self.dropdown = None
        self.dropdown = DropDown()
        self.dropdown.bind(
            on_select=lambda instance, x: self.set_control(self.key, x)
        )
        readoutbut = Button(
            text='readout',
            size_hint_y=None,
            height=self.height,
            background_color=(0.7, 0.7, 0.7, 1)
        )
        readoutbut.bind(
            on_release=lambda instance: self.dropdown.select('readout')
        )
        self.dropdown.add_widget(readoutbut)
        textinbut = Button(
            text='textinput',
            size_hint_y=None,
            height=self.height,
            background_color=(0.7, 0.7, 0.7, 1)
        )
        textinbut.bind(
            on_release=lambda instance: self.dropdown.select('textinput')
        )
        self.dropdown.add_widget(textinbut)
        togbut = Button(
            text='togglebutton',
            size_hint_y=None,
            height=self.height,
            background_color=(0.7, 0.7, 0.7, 1)
        )
        togbut.bind(
            on_release=lambda instance: self.dropdown.select('togglebutton')
        )
        self.dropdown.add_widget(togbut)
        sliderbut = Button(
            text='slider',
            size_hint_y=None,
            height=self.height,
            background_color=(0.7, 0.7, 0.7, 1)
        )
        sliderbut.bind(
            on_release=lambda instance: self.dropdown.select('slider')
        )
        self.dropdown.add_widget(sliderbut)
        self.bind(on_release=self.dropdown.open)


class ConfigListItemToggleButton(BoxLayout):
    true_text = StringProperty('0')
    false_text = StringProperty('1')

    def set_true_text(self, *args):
        self.parent.set_config(self.parent.key, 'true_text', self.ids.truetext.text)
        self.true_text = self.ids.truetext.text

    def set_false_text(self, *args):
        self.parent.set_config(self.parent.key, 'false_text', self.ids.falsetext.text)


class ConfigListItemSlider(BoxLayout):
    min = NumericProperty(0.)
    max = NumericProperty(1.)

    def set_min(self, *args):
        minn = float(self.ids.minimum.text)
        try:
            self.parent.set_config(self.parent.key, 'min', minn)
            self.min = minn
        except ValueError:
            self.ids.minimum.text = ''

    def set_max(self, *args):
        maxx = float(self.ids.minimum.text)
        try:
            self.parent.set_config(self.parent.key, 'max', maxx)
            self.max = maxx
        except ValueError:
            self.ids.maximum.text = ''


class ConfigListItemCustomizer(BoxLayout):
    key = ObjectProperty()
    control = StringProperty()
    config = DictProperty()
    set_config = ObjectProperty()

    def on_control(self, *args):
        self.clear_widgets()
        if self.control == 'togglebutton':
            if 'true_text' not in self.config or 'false_text' not in self.config:
                Clock.schedule_once(self.on_control, 0)
                return
            wid = ConfigListItemToggleButton(true_text=self.config['true_text'], false_text=self.config['false_text'])
            self.add_widget(wid)
        elif self.control == 'slider':
            if 'min' not in self.config or 'max' not in self.config:
                Clock.schedule_once(self.on_control, 0)
                return
            wid = ConfigListItemSlider(min=self.config['min'], max=self.config['max'])
            self.add_widget(wid)


class ConfigListItem(BoxLayout):
    key = ObjectProperty()
    config = DictProperty()
    set_control = ObjectProperty()
    set_config = ObjectProperty()
    deleter = ObjectProperty()


class StatListViewConfigurator(BaseStatListView):
    statlist = ObjectProperty()
    _key_cfg_setters = DictProperty()
    _val_text_setters = DictProperty()
    _control_wids = DictProperty()

    def set_control(self, key, value):
        config = self.proxy.get('_config', {})
        if value == 'slider':
            if 'min' not in config:
                self.set_config(key, 'min', 0.0)
            if 'max' not in config:
                self.set_config(key, 'max', 1.0)
        elif value == 'togglebutton':
            if 'true_text' not in config:
                self.set_config(key, 'true_text', '1')
            if 'false_text' not in config:
                self.set_config(key, 'false_text', '0')
        self.set_config(key, 'control', value)

    def munge(self, k, v):
        # makes ConfigListItem
        ret = super().munge(k, v)
        ret['deleter'] = self.del_key
        ret['set_control'] = self.set_control
        ret['set_config'] = self.set_config
        return ret


class StatScreen(Screen):
    statlist = ObjectProperty()
    statcfg = ObjectProperty()
    toggle = ObjectProperty()
    engine = ObjectProperty()
    proxy = ObjectProperty()

    def new_stat(self):
        """Look at the key and value that the user has entered into the stat
        configurator, and set them on the currently selected
        entity.

        """
        key = self.ids.newstatkey.text
        value = self.ids.newstatval.text
        if not (key and value):
            # TODO implement some feedback to the effect that
            # you need to enter things
            return
        try:
            self.proxy[key] = self.engine.unpack(value)
        except (TypeError, ValueError):
            self.proxy[key] = value
        self.ids.newstatkey.text = ''
        self.ids.newstatval.text = ''


Builder.load_string("""
<ConfigListItemCustomization>:
    pos_hint: {'x': 0, 'y': 0}
<ConfigListItemToggleButton>:
    Label:
        text: 'True text:'
    TextInput:
        id: truetext
        hint_text: root.true_text
        on_text_validate: root.set_true_text()
    Label:
        text: 'False text:'
    TextInput:
        id: falsetext
        hint_text: root.false_text
        on_text_validate: root.set_false_text()
<ConfigListItemSlider>:
    Label:
        text: 'Minimum:'
    TextInput:
        id: minimum
        hint_text: str(root.min)
        on_text_validate: root.set_min()
    Label:
        text: 'Maximum:'
    TextInput:
        id: maximum
        hint_text: str(root.max)
        on_text_validate: root.set_max()
<ConfigListItem>:
    height: 30
    Button:
        size_hint_x: 0.4 / 3
        text: 'del'
        on_release: root.deleter(root.key)
    Label:
        size_hint_x: 0.4 / 3
        text: str(root.key)
    ControlTypePicker:
        size_hint_x: 0.4 / 3
        key: root.key
        set_control: root.set_control
        text: root.config['control'] if 'control' in root.config else 'readout'
    ConfigListItemCustomizer:
        size_hint_x: 0.6
        control: root.config['control'] if 'control' in root.config else 'readout'
        config: root.config
        key: root.key
        set_config: root.set_config
<StatScreen>:
    name: 'statcfg'
    statcfg: cfg
    BoxLayout:
        orientation: 'vertical'
        StatListViewConfigurator:
            viewclass: 'ConfigListItem'
            id: cfg
            app: app
            engine: root.engine
            proxy: root.proxy
            statlist: root.statlist
            size_hint_y: 0.95
            RecycleBoxLayout:
                default_size: None, dp(56)
                default_size_hint: 1, None
                size_hint_y: None
                height: self.minimum_height
                orientation: 'vertical'
        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: 0.05
            TextInput:
                id: newstatkey
                multiline: False
                write_tab: False
                hint_text: 'New stat'
            TextInput:
                id: newstatval
                multiline: False
                write_tab: False
                hint_text: 'Value'
            Button:
                id: newstatbut
                text: '+'
                on_release: root.new_stat()
            Button:
                id: closer
                text: 'Close'
                on_release: root.toggle()
""")
