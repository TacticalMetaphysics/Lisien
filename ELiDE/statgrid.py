# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
"""Grid of current values for some entity. Can be changed by the
user. Autoupdates when there's a change for any reason.

"""
from functools import partial
from kivy.properties import ObjectProperty
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.textinput import TextInput
from kivy.uix.listview import ListView, CompositeListItem
from kivy.adapters.listadapter import ListAdapter
from LiSE.remote import (
    CharacterRemoteMapping,
    EntityRemoteMapping
)
from gorm.json import json_load


def try_json_load(obj):
    try:
        return json_load(obj)
    except (TypeError, ValueError):
        return obj


class StatRowKey(TextInput):
    def on_text_validate(self, *args):
        if hasattr(self, '_former_key'):
            self.parent.parent.del_key(self._former_key)
        self.parent.key = self.text.lstrip('_')
        self.parent.set_value()
        self.text = ''


class StatRowValue(TextInput):
    def on_text_validate(self, *args):
        self.parent.value = self.text
        self.parent.set_value()
        self.text = ''


class StatRowListItem(CompositeListItem):
    key = ObjectProperty()
    value = ObjectProperty()
    reg = ObjectProperty()
    unreg = ObjectProperty()
    setter = ObjectProperty()

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


class StatListView(ListView):
    remote_map = ObjectProperty()

    def __init__(self, **kwargs):
        kwargs['adapter'] = ListAdapter(
            data=[],
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
        super().__init__(**kwargs)
        self._listed = {}
        self._listeners = {}
        self._trigger_poll = Clock.create_trigger(self.poll)

    def on_remote_map(self, *args):
        if self.remote_map is None:
            return
        self.adapter.data = list(
            (k, v) for (k, v) in self.remote_map.items()
            if not (isinstance(k, str) and k[0] == '_')
        )

        @self.remote_map.listener
        def listen(k, v):
            if v is None:
                self.adapter.data.remove((k, self._listed[k]))
                del self._listed[k]
            else:
                self._listed[k] = v
                self.adapter.data.append((k, v))

    def poll(self, *args):
        for k in self._listeners:
            self.remote_map.fetch(k)

    def _reg_widget(self, w, *args):
        if self.remote_map is None:
            Clock.schedule_once(partial(self._reg_widget, w), 0)
            return

        @self.remote_map.listener(key=w.key)
        def listen(k, v):
            assert(k == w.key)
            w.value = v
        self._listeners[w.key] = listen

    def _unreg_widget(self, w):
        assert(self.remote_map is not None)
        if w.key in self._listeners:
            self.remote_map.unlisten(self._listeners[w.key], key=w.key)

    def _set_value(self, k, v):
        self.remote_map[k] = v
