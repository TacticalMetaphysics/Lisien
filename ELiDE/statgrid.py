# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
"""Grid of current values for some entity. Can be changed by the
user. Autoupdates when there's a change for any reason.

"""
from functools import partial
from kivy.properties import (
    NumericProperty,
    StringProperty,
    ReferenceListProperty,
    BooleanProperty,
    DictProperty,
    ObjectProperty
)
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.uix.slider import Slider
from kivy.uix.behaviors import ToggleButtonBehavior
from kivy.uix.textinput import TextInput
from kivy.uix.listview import (
    SelectableView,
    ListView,
    ListItemLabel,
    ListItemButton,
    CompositeListItem
)
from kivy.adapters.dictadapter import DictAdapter
from kivy.lang import Builder
from gorm.xjson import json_load
from ELiDE.remote import MirrorMapping


def try_json_load(obj):
    try:
        return json_load(obj)
    except (TypeError, ValueError):
        return obj


class StatRowTextInput(TextInput):
    def __init__(self, **kwargs):
        kwargs['multiline'] = False
        self._trigger_upd_value = Clock.create_trigger(self.upd_value)
        super().__init__(**kwargs)

        def lost_focus(self, *args):
            if not self.focus:
                self._trigger_upd_value()

        self.bind(
            on_enter=self._trigger_upd_value,
            on_text_validate=self._trigger_upd_value,
            on_focus=lost_focus
        )

    def upd_value(self, *args):
        if self.text == '':
            self.parent.value = None
        else:
            self.parent.value = self.text
        self.parent.set_value()
        self.text = ''


class StatRowToggleButton(ToggleButtonBehavior, ListItemButton):
    def __init__(self, **kwargs):
        self._trigger_upd_value = Clock.create_trigger(self.upd_value)
        super().__init__(**kwargs)
        self.bind(on_touch_up=self._trigger_upd_value)

    def upd_value(self, *args):
        if self.parent is None:
            return
        if (
            self.state == 'normal' and self.parent.value == 0
        ) or (
            self.state == 'down' and self.parent.value == 1
        ):
            return
        if self.state == 'normal':
            self.parent.value = 0
        else:
            self.parent.value = 1
        self.parent.set_value()


class StatRowSlider(SelectableView, Slider):
    need_set = BooleanProperty(False)

    def __init__(self, **kwargs):
        self._trigger_maybe_set = Clock.create_trigger(self.maybe_set)
        super().__init__(**kwargs)
        self.bind(on_touch_up=self._trigger_maybe_set)

    def on_value(self, *args):
        self.need_set = True

    def maybe_set(self, *args):
        if self.need_set:
            self.parent.value = self.value
            self.parent.set_value()
            self.need_set = False


class StatRowListItem(CompositeListItem):
    key = ObjectProperty()
    value = ObjectProperty(None, allownone=True)
    reg = ObjectProperty()
    unreg = ObjectProperty()
    setter = ObjectProperty()

    def set_value(self, *args):
        self.setter(
            try_json_load(self.key),
            try_json_load(self.value)
        )

    def on_parent(self, *args):
        if self.parent is None:
            self.unreg(self)
        else:
            self.reg(self)


control_cls = {
    'readout': lambda v: {
        'cls': ListItemLabel,
        'kwargs': {
            'text': str(v)
        }
    },
    'textinput': lambda v: {
        'cls': StatRowTextInput,
        'kwargs': {
            'text': str(v)
        }
    },
    'togglebutton': lambda v: {
        'cls': StatRowToggleButton,
        'kwargs': {
            'state': 'down' if v else 'normal'
        }
    },
    'slider': lambda v: {
        'cls': StatRowSlider,
        'kwargs': {'value': v, 'text': str(v)}
    }
}


class StatListView(ListView, MirrorMapping):
    control = DictProperty({})
    config = DictProperty({})
    branch = StringProperty('master')
    tick = NumericProperty(0)
    time = ReferenceListProperty(branch, tick)

    def __init__(self, **kwargs):
        kwargs['adapter'] = self.get_adapter()
        self._trigger_sortkeys = Clock.create_trigger(self.sortkeys)
        self._trigger_upd_data = Clock.create_trigger(self.upd_data)
        super().__init__(**kwargs)
        self.bind(mirror=self._trigger_sortkeys)
        self.bind(
            mirror=self._trigger_upd_data,
            branch=self._trigger_upd_data,
            tick=self._trigger_upd_data
        )
        self._listeners = {}

    def get_adapter(self):
        return DictAdapter(
            data=self.get_data(),
            cls=StatRowListItem,
            args_converter=lambda i, kv: {
                'key': kv[0],
                'value': kv[1],
                'reg': self._reg_widget,
                'unreg': self._unreg_widget,
                'setter': self._set_value,
                'cls_dicts': self.get_cls_dicts(*kv)
            },
            selection_mode='multiple',
            allow_empty_selection=True
        )

    def get_cls_dicts(self, key, value):
        keydict = {
            'cls': ListItemLabel,
            'kwargs': {'text': str(key)}
        }
        valdict = control_cls[self.control.get(key, 'textinput')](value)
        override = dict(self.config.get(key, {}))
        # hack to let you choose how to display boolean values
        for (k, v) in override.items():
            Logger.debug('StatListView: overriding {}={}'.format(k, v))
        true_text = override['true_text'] if 'true_text' in override else 'T'
        false_text \
            = override['false_text'] if 'false_text' in override else 'F'
        if 'true_text' in override:
            del override['true_text']
        if 'false_text' in override:
            del override['false_text']
        valdict['kwargs'].update(override)
        valdict['kwargs']['text'] = true_text if value else false_text
        return [keydict, valdict]

    def get_data(self):
        return {
            k: (k, v) for (k, v) in self.mirror.items()
            if v is not None and (
                k[0] != '_' and
                k not in (
                    'character',
                    'name',
                    'location',
                    'next_location',
                    'locations',
                    'arrival_time',
                    'next_arrival_time'
                ) or not isinstance(k, str)
            )
        }

    def upd_data(self, *args):
        self.adapter.data = self.get_data()
        if '_control' not in self.mirror:
            self.control = {}
        else:
            self.control = dict(self.mirror['_control'])
        if '_config' not in self.mirror:
            self.config = {}
        else:
            self.config = dict(self.mirror['_config'])

    def sortkeys(self, *args):
        for key in self.mirror.keys():
            if key not in self.adapter.sorted_keys:
                self.adapter.sorted_keys = sorted(self.mirror.keys())
                return
        for k in set(
                k for k in self.adapter.sorted_keys if k not in self.mirror
        ):
            self.adapter.sorted_keys.remove(k)

    def _reg_widget(self, w, *args):
        if not self.mirror:
            Clock.schedule_once(partial(self._reg_widget, w), 0)
            return

        def listen(*args):
            if w.key not in self.mirror:
                return
            if w.value != self.mirror[w.key]:
                w.value = self.mirror[w.key]
        self._listeners[w.key] = listen
        self.bind(mirror=listen)

    def _unreg_widget(self, w):
        if w.key in self._listeners:
            self.unbind(mirror=self._listeners[w.key])

    def _set_value(self, k, v):
        if v is None:
            del self.remote[k]
        else:
            self.remote[k] = v


kv = """
<StatRowListItem>:
    orientation: 'horizontal'
    height: 30
"""
Builder.load_string(kv)
