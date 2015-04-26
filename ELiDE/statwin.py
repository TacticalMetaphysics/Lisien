from kivy.clock import Clock
from kivy.properties import DictProperty, ObjectProperty
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
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
from .statgrid import StatListView, default_cfg
from .util import set_remote_value


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
    button_kwargs = DictProperty()
    dropdown_kwargs = DictProperty()
    control_texts = DictProperty()
    control_callbacks = DictProperty()

    def __init__(self, **kwargs):
        if 'button_kwargs' not in kwargs:
            kwargs['button_kwargs'] = {}
        if 'dropdown_kwargs' not in kwargs:
            kwargs['dropdown_kwargs'] = {}
        super().__init__(**kwargs)
        self.build()

    def selected(self, v):
        self.setter(self.key, v)
        self.text = str(v)

    def build(self, *args):
        if None in (
                self.key,
                self.setter,
                self.button_kwargs,
                self.dropdown_kwargs,
                self.control_texts,
                self.control_callbacks
        ):
            Clock.schedule_once(self.build, 0)
            return
        self.mainbutton = None
        self.dropdown = None
        self.dropdown = DropDown(
            text=self.text,
            on_select=lambda instance, x: self.selected(x),
            **self.dropdown_kwargs
        )
        self.dropdown.add_widget(
            Button(
                text='Readout',
                size_hint_y=None,
                height=self.height,
                on_press=lambda instance: self.dropdown.select('readout')
            )
        )
        self.dropdown.add_widget(
            Button(
                text='Text input',
                size_hint_y=None,
                height=self.height,
                on_press=lambda instance: self.dropdown.select('textinput')
            )
        )
        self.dropdown.add_widget(
            Button(
                text='Toggle button',
                size_hint_y=None,
                height=self.height,
                on_press=lambda instance: self.dropdown.select('togglebutton')
            )
        )
        self.dropdown.add_widget(
            Button(
                text='Slider',
                size_hint_y=None,
                height=self.height,
                on_press=lambda instance: self.dropdown.select('slider')
            )
        )

        self.bind(on_press=self.dropdown.open)


class StatListViewConfigurator(StatListView):
    def del_key(self, key):
        if key in self.mirror:
            del self.remote[key]
        if key in self.adapter.sorted_keys:
            self.adapter.sorted_keys.remove(key)

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
        cfg = self.config.get(key, default_cfg)
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


class StatWindow(BoxLayout):
    layout = ObjectProperty()

    def __init__(self, **kwargs):
        kwargs['orientation'] = 'vertical'
        super().__init__(**kwargs)

    def on_layout(self, *args):
        if self.layout is None:
            return
        if self.canvas is None:
            Clock.schedule_once(self.on_layout, 0)
            return
        cfg = StatListViewConfigurator(
            time=self.layout.time,
            size_hint_y=0.95
        )
        newstatkey = TextInput(
            multiline=False,
            write_tab=False,
            hint_text='New stat'
        )
        newstatval = TextInput(
            multiline=False,
            write_tab=False,
            hint_text='Value'
        )
        newstatbut = Button(
            text='+',
            on_press=lambda inst: set_remote_value(
                cfg.remote,
                newstatkey.text,
                newstatval.text
            )
        )
        close_cfg_but = Button(
            text='Close',
            on_press=lambda inst: self.layout.toggle_stat_cfg()
        )
        buttons = BoxLayout(size_hint_y=0.05)
        buttons.add_widget(newstatkey)
        buttons.add_widget(newstatval)
        buttons.add_widget(newstatbut)
        buttons.add_widget(close_cfg_but)
        self.add_widget(buttons)
        self.add_widget(cfg)

        self.layout._stat_cfg = cfg
        self.layout._newstatkey = newstatkey
        self.layout._newstatval = newstatval
        self.layout._newstatbut = newstatbut
        self.layout._close_stat_cfg_but = close_cfg_but


Builder.load_string(
    """
<ConfigListItem>:
    height: 30
"""
)
