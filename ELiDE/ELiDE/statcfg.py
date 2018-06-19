# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  public@zacharyspector.com
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
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.textinput import TextInput
from .statlist import BaseStatListView


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
            del self.app.selected_proxy[k]
        else:
            self.app.selected_proxy[k] = v

    def build(self, *args):
        if None in (
                self.key,
                self.sett
        ):
            Clock.schedule_once(self.build, 0)
            return
        self.mainbutton = None
        self.dropdown = None
        self.dropdown = DropDown()
        self.dropdown.bind(
            on_select=lambda instance, x: self.sett(self.key, x)
        )
        readoutbut = Button(
            text='Readout',
            size_hint_y=None,
            height=self.height
        )
        readoutbut.bind(
            on_press=lambda instance: self.dropdown.select('readout')
        )
        self.dropdown.add_widget(readoutbut)
        textinbut = Button(
            text='Text input',
            size_hint_y=None,
            height=self.height
        )
        textinbut.bind(
            on_press=lambda instance: self.dropdown.select('textinput')
        )
        self.dropdown.add_widget(textinbut)
        togbut = Button(
            text='Toggle button',
            size_hint_y=None,
            height=self.height
        )
        togbut.bind(
            on_press=lambda instance: self.dropdown.select('togglebutton')
        )
        self.dropdown.add_widget(togbut)
        sliderbut = Button(
            text='Slider',
            size_hint_y=None,
            height=self.height
        )
        sliderbut.bind(
            on_press=lambda instance: self.dropdown.select('slider')
        )
        self.dropdown.add_widget(sliderbut)
        self.bind(on_press=self.dropdown.open)


class ConfigListItemCustomization(BoxLayout):
    config = ObjectProperty()


class ConfigListItemToggleButton(ConfigListItemCustomization):
    def set_true_text(self, *args):
        self.config['true_text'] = self.ids.truetext.text

    def set_false_text(self, *args):
        self.config['false_text'] = self.ids.falsetext.text


class ConfigListItemSlider(ConfigListItemCustomization):
    def set_min(self, *args):
        try:
            self.config['min'] = float(self.ids.minimum.text)
        except ValueError:
            self.ids.minimum.text = ''

    def set_max(self, *args):
        try:
            self.config['max'] = float(self.ids.maximum.text)
        except ValueError:
            self.ids.maximum.text = ''


class ConfigListItemCustomizer(FloatLayout):
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


class StatListViewConfigurator(BaseStatListView):
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

    def set_control(self, key, value):
        self.set_config(key, 'controol', value)

    def del_key(self, key):
        del self.mirror[key]
        self.statlist.del_key(key)

    def munge(self, k, v):
        ret = super().munge(k, v)
        ret['on_config'] = self.inst_set_configs
        ret['deleter'] = self.del_key
        ret['sett'] = self.set_control
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
            self.proxy[key] \
                = self.statlist.mirror[key] \
                = self.statcfg.mirror[key] \
                = self.engine.unpack(value)
        except (TypeError, ValueError):
            self.proxy[key] \
                = self.statlist.mirror[key] \
                = self.statcfg.mirror[key] \
                = value
        self.ids.newstatkey.text = ''
        self.ids.newstatval.text = ''


Builder.load_string("""
<ConfigListItemCustomization>:
    config: self.parent.config if self.parent else {}
    pos_hint: {'x': 0, 'y': 0}
<ConfigListItemToggleButton>:
    Label:
        text: 'True text:'
    TextInput:
        id: truetext
        hint_text: root.config.get('true_text', '1')
        on_text_validate: root.set_true_text()
    Label:
        text: 'False text:'
    TextInput:
        id: falsetext
        hint_text: root.config.get('false_text', '0')
        on_text_validate: root.set_false_text()
<ConfigListItemSlider>:
    Label:
        text: 'Minimum:'
    TextInput:
        id: minimum
        hint_text: str(root.config.get('min', 0.0))
        on_text_validate: root.set_min()
    Label:
        text: 'Maximum:'
    TextInput:
        id: maximum
        hint_text: str(root.config.get('max', 1.0))
        on_text_validate: root.set_max()
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
        control: root.config.get('control', 'readout')
        text: root.config.get('control', 'readout')
    ConfigListItemCustomizer:
        config: root.config
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
                on_press: root.new_stat()
            Button:
                id: closer
                text: 'Close'
                on_press: root.toggle()
""")
