# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
"""Grid of current values for some entity. Can be changed by the
user. Autoupdates when there's a change for any reason.

"""
from functools import partial
from kivy.properties import (
    ObjectProperty,
    NumericProperty,
    StringProperty
)
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.uix.textinput import TextInput
from kivy.uix.listview import ListView, CompositeListItem
from kivy.adapters.dictadapter import DictAdapter
from kivy.lang import Builder
from gorm.json import json_load
from ELiDE.remote import MirrorMapping


def try_json_load(obj):
    try:
        return json_load(obj)
    except (TypeError, ValueError):
        return obj


class StatRowValue(TextInput):
    def on_text_validate(self, *args):
        if self.text == '':
            self.parent.value = None
        else:
            self.parent.value = self.text
        self.parent.set_value()
        self.text = ''


class StatRowListItem(CompositeListItem):
    key = ObjectProperty()
    value = ObjectProperty(None, allownone=True)
    reg = ObjectProperty()
    unreg = ObjectProperty()
    setter = ObjectProperty()
    font_name = StringProperty('DroidSans')
    font_size = NumericProperty(10)

    def __init__(self, **kwargs):
        super(CompositeListItem, self).__init__(**kwargs)

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


class StatListView(ListView, MirrorMapping):
    def __init__(self, **kwargs):
        kwargs['adapter'] = DictAdapter(
            data={},
            cls=StatRowListItem,
            args_converter=lambda i, kv: {
                'key': kv[0],
                'value': kv[1],
                'reg': self._reg_widget,
                'unreg': self._unreg_widget,
                'setter': self._set_value
            },
            selection_mode='multiple',
            allow_empty_selection=True
        )
        self._trigger_sortkeys = Clock.create_trigger(self.sortkeys)
        self._trigger_upd_data = Clock.create_trigger(self.upd_data)
        super().__init__(**kwargs)
        self.bind(mirror=self._trigger_sortkeys)
        self.bind(mirror=self._trigger_upd_data)
        self._listeners = {}

    def upd_data(self, *args):
        self.adapter.data = dict(
            (k, (k, v)) for (k, v) in self.mirror.items()
            if v is not None and
            not isinstance(k, str) or (
                k[0] != '_' and
                k not in (
                    'character',
                    'name',
                    'location',
                    'next_location',
                    'locations',
                    'arrival_time',
                    'next_arrival_time'
                )
            )
        )

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
    Label:
        id: keylabel
        font_name: root.font_name
        font_size: root.font_size
        text: str(root.key)
    StatRowValue:
        id: valcell
        font_name: root.font_name
        font_size: root.font_size
        hint_text: str(root.value)
        multiline: False
"""
Builder.load_string(kv)
