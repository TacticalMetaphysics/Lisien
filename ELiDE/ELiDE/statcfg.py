# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
from kivy.clock import Clock
from kivy.properties import (
    DictProperty,
    NumericProperty,
    StringProperty,
    ReferenceListProperty,
    ObjectProperty,
    OptionProperty
)
from kivy.lang import Builder
from kivy.uix.widget import Widget
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.textinput import TextInput
from .statlist import AbstractStatListView


control_txt = {
    'readout': 'Readout',
    'textinput': 'Text input',
    'togglebutton': 'Toggle button',
    'slider': 'Slider'
}


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
    sett = ObjectProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build()

    def set_value(self, k, v):
        if v is None:
            del self.app.selected_remote[k]
        else:
            self.app.selected_remote[k] = v

    def selected(self, v):
        self.sett(self.key, v)
        self.text = str(v)

    def build(self, *args):
        if None in (
                self.key,
                self.setter
        ):
            Clock.schedule_once(self.build, 0)
            return
        self.mainbutton = None
        self.dropdown = None
        self.dropdown = DropDown()
        self.dropdown.bind(on_select=lambda instance, x: self.selected(x))
        readoutbut = Button(
            text='Readout',
            size_hint_y=None,
            height=self.height
        )
        readoutbut.bind(on_press=lambda instance: self.dropdown.select('readout'))
        self.dropdown.add_widget(readoutbut)
        textinbut = Button(
            text='Text input',
            size_hint_y=None,
            height=self.height
        )
        textinbut.bind(on_press=lambda instance: self.dropdown.select('textinput'))
        self.dropdown.add_widget(textinbut)
        togbut = Button(
            text='Toggle button',
            size_hint_y=None,
            height=self.height
        )
        togbut.bind(on_press=lambda instance: self.dropdown.select('togglebutton'))
        self.dropdown.add_widget(togbut)
        sliderbut = Button(
            text='Slider',
            size_hint_y=None,
            height=self.height
        )
        sliderbut.bind(on_press=lambda instance: self.dropdown.select('slider'))
        self.dropdown.add_widget(sliderbut)
        self.bind(on_press=self.dropdown.open)


class ConfigListItemToggleButton(BoxLayout):
    config = ObjectProperty()


class ConfigListItemSlider(BoxLayout):
    config = ObjectProperty()


class ConfigListItemCustomizer(Widget):
    control = ObjectProperty()
    config = ObjectProperty()

    def on_control(self, *args):
        self.clear_widgets()
        if self.control == 'togglebutton':
            wid = ConfigListItemToggleButton(config=self.config)
            self.add_widget(wid)
        elif self.control == 'slider':
            wid = ConfigListItemSlider(config=self.config)
            self.add_widget(wid)


class ConfigListItem(BoxLayout):
    key = ObjectProperty()
    config = DictProperty()
    sett = ObjectProperty()
    deleter = ObjectProperty()
    control = OptionProperty('readout', options=['readout', 'textinput', 'togglebutton', 'slider'])


class StatListViewConfigurator(AbstractStatListView):
    statlist = ObjectProperty()
    _key_cfg_setters = DictProperty()
    _val_text_setters = DictProperty()
    _control_wids = DictProperty()

    def set_config(self, key, option, value):
        super().set_config(key, option, value)
        self.statlist.set_config(key, option, value)

    def set_configs(self, key, d):
        super().set_configs(key, d)
        self.statlist.config[key] = d

    def inst_set_configs(self, inst, val):
        self.set_configs(inst.key, val)

    def set_control(self, key, control):
        super().set_control(key, control)
        self.statlist.set_control(key, control)

    def inst_set_control(self, inst, val):
        self.set_control(inst.key, val)

    def del_key(self, key):
        del self.mirror[key]
        self.statlist.del_key(key)

    def munge(self, k, v):
        ret = super().munge(k, v)
        ret['on_control'] = self.inst_set_control
        ret['on_config'] = self.inst_set_configs
        ret['deleter'] = self.del_key
        return ret


class StatScreen(Screen):
    statlist = ObjectProperty()
    statcfg = ObjectProperty()
    toggle = ObjectProperty()
    engine = ObjectProperty()
    branch = StringProperty('master')
    tick = NumericProperty(0)
    time = ReferenceListProperty(branch, tick)
    remote = ObjectProperty()

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
            self.remote[key] = self.statlist.mirror[key] = self.statcfg.mirror[key] = self.engine.json_load(value)
        except (TypeError, ValueError):
            self.remote[key] = self.statlist.mirror[key] = self.statcfg.mirror[key] = value
        self.ids.newstatkey.text = ''
        self.ids.newstatval.text = ''


Builder.load_string("""
<ConfigListItem>:
    height: 30
    Button:
        text: 'del'
        on_press: root.deleter(root.key)
    Label:
        text: str(root.key)
    ControlTypePicker:
        key: root.key
        sett: root.sett
        control: root.control
    ConfigListItemCustomizer:
        config: root.config
        control: root.control
<StatScreen>:
    name: 'statcfg'
    statcfg: cfg
    BoxLayout:
        orientation: 'vertical'
        StatListViewConfigurator:
            viewclass: 'ConfigListItem'
            id: cfg
            engine: root.engine
            remote: root.remote
            branch: root.branch
            tick: root.tick
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
                on_press: root.new_stat()
            Button:
                id: closer
                text: 'Close'
                on_press: root.toggle()
""")
