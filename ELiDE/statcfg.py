from kivy.clock import Clock
from kivy.properties import DictProperty, ListProperty, ObjectProperty
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.textinput import TextInput
from kivy.adapters.dictadapter import DictAdapter
from kivy.uix.listview import (
    CompositeListItem,
    ListItemButton,
    ListItemLabel,
    SelectableView
)
from .statlist import StatListView
from .util import try_json_load


control_txt = {
    'readout': 'Readout',
    'textinput': 'Text input',
    'togglebutton': 'Toggle button',
    'slider': 'Slider'
}


class SelectableTextInput(TextInput, SelectableView):
    def select_from_composite(self, *args):
        pass

    def deselect_from_composite(self, *args):
        pass


class IntInput(SelectableTextInput):
    def insert_text(self, s, from_undo=False):
        return super().insert_text(
            ''.join(c for c in s if c in '0123456789'),
            from_undo
        )


class ConfigListItem(CompositeListItem):
    pass


class ControlTypePicker(ListItemButton):
    key = ObjectProperty()
    mainbutton = ObjectProperty()
    dropdown = ObjectProperty()
    setter = ObjectProperty()
    button_kwargs = DictProperty({})
    dropdown_kwargs = DictProperty({})
    control_texts = DictProperty({})
    control_callbacks = DictProperty({})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.build()

    def selected(self, v):
        self.setter(self.key, v)
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


class StatListViewConfigurator(StatListView):
    stat_list = ObjectProperty()

    def set_config(self, key, option, value):
        super().set_config(key, option, value)
        self.stat_list.set_config(key, option, value)

    def set_control(self, key, control):
        super().set_control(key, control)
        self.stat_list.set_control(key, control)

    def del_key(self, key):
        if key in self.mirror:
            del self.remote[key]

    def get_adapter(self):
        return DictAdapter(
            data=self.get_data(),
            cls=ConfigListItem,
            args_converter=lambda i, kv: {
                'cls_dicts': self.get_cls_dicts(*kv)
            },
            selection_mode='multiple',
            allow_empty_selection=True
        )

    def get_cls_dicts(self, key, value):
        control_type = self.control.get(key, 'readout')
        cfg = self.config.get(key, {})
        deldict = {
            'cls': ListItemButton,
            'kwargs': {
                'text': 'del',
                'on_press': lambda inst: self.del_key(key)
            }
        }
        picker_dict = {
            'cls': ControlTypePicker,
            'kwargs': {
                'key': key,
                'text': control_txt[control_type],
                'setter': self.set_control,
                'control_texts': control_txt,
                'dropdown_kwargs': {
                    'canvas': self.canvas.after
                }
            }
        }
        keydict = {
            'cls': ListItemLabel,
            'kwargs': {'text': str(key)}
        }
        valdict = {
            'cls': SelectableTextInput,
            'kwargs': {
                'text': str(value),
                'on_text_validate': lambda i: self.set_value(key, i.text),
                'on_enter': lambda i: self.set_value(key, i.text),
                'on_focus': lambda i, v:
                self.set_value(key, i.text) if not v else None,
                'write_tab': False
            }
        }
        cls_dicts = [
            deldict, keydict, valdict, picker_dict
        ]

        if control_type == 'togglebutton':
            if 'true_text' not in cfg:
                cfg['true_text'] = '1'
            if 'false_text' not in cfg:
                cfg['false_text'] = '0'
            true_text_label_dict = {
                'cls': ListItemLabel,
                'kwargs': {'text': 'True:'}
            }
            true_text_dict = {
                'cls': SelectableTextInput,
                'kwargs': {
                    'multiline': False,
                    'text': str(cfg.get('true_text', '1')),
                    'on_enter': lambda i:
                    self.set_config(key, 'true_text', i.text),
                    'on_text_validate': lambda i:
                    self.set_config(key, 'true_text', i.text),
                    'on_focus': lambda i, foc:
                    self.set_config(key, 'true_text', i.text)
                    if not foc else None,
                    'write_tab': False
                }
            }
            false_text_label_dict = {
                'cls': ListItemLabel,
                'kwargs': {'text': 'False:'}
            }
            false_text_dict = {
                'cls': SelectableTextInput,
                'kwargs': {
                    'multiline': False,
                    'text': str(cfg.get('false_text', '0')),
                    'on_enter': lambda i:
                    self.set_config(key, 'false_text', i.text),
                    'on_text_validate': lambda i:
                    self.set_config(key, 'false_text', i.text),
                    'on_focus': lambda i, foc:
                    self.set_config(key, 'false_text', i.text)
                    if not foc else None,
                    'write_tab': False
                }
            }
            cls_dicts.extend(
                (
                    true_text_label_dict,
                    true_text_dict,
                    false_text_label_dict,
                    false_text_dict
                )
            )

        if control_type == 'slider':
            if 'min' not in cfg:
                cfg['min'] = 0.0
            if 'max' not in cfg:
                cfg['max'] = 1.0
            min_label_dict = {
                'cls': ListItemLabel,
                'kwargs': {'text': 'Minimum:'}
            }
            min_dict = {
                'cls': IntInput,
                'kwargs': {
                    'multiline': False,
                    'text': str(cfg.get('min', 0.0)),
                    'on_enter': lambda i:
                    self.set_config(key, 'min', float(i.text)),
                    'on_text_validate': lambda i:
                    self.set_config(key, 'min', float(i.text)),
                    'on_focus': lambda i, foc:
                    self.set_config(key, 'min', float(i.text))
                    if not foc else None,
                    'write_tab': False
                }
            }
            max_label_dict = {
                'cls': ListItemLabel,
                'kwargs': {'text': 'Maximum:'}
            }
            max_dict = {
                'cls': IntInput,
                'kwargs': {
                    'multiline': False,
                    'hint_text': 'Maximum',
                    'text': str(cfg.get('max', 1.0)),
                    'on_enter': lambda i:
                    self.set_config(key, 'max', float(i.text)),
                    'on_text_validate': lambda i:
                    self.set_config(key, 'max', float(i.text)),
                    'on_focus': lambda i, foc:
                    self.set_config(key, 'max', float(i.text))
                    if not foc else None,
                    'write_tab': False
                }
            }
            cls_dicts.extend(
                (min_label_dict, min_dict, max_label_dict, max_dict)
            )
        return cls_dicts


class StatScreen(Screen):
    remote = ObjectProperty()
    stat_list = ObjectProperty()
    toggle = ObjectProperty()
    time = ListProperty()

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
        self.remote[key] = try_json_load(value)
        self.ids.newstatkey.text = ''
        self.ids.newstatval.text = ''


Builder.load_string("""
<ConfigListItem>:
    height: 30
<StatScreen>:
    name: 'stat_cfg'
    BoxLayout:
        orientation: 'vertical'
        StatListViewConfigurator:
            id: cfg
            stat_list: root.stat_list
            time: root.time
            remote: root.remote
            size_hint_y: 0.95
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
