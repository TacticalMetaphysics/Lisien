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
from kivy.clock import Clock
from kivy.uix.textinput import TextInput
from kivy.uix.listview import ListView, CompositeListItem
from kivy.adapters.listadapter import ListAdapter
from kivy.lang import Builder
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


class StatListView(ListView):
    mirrormap = ObjectProperty()

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

    def on_mirrormap(self, *args):
        def listen(mir, *args):
            for (k, v) in self.mirrormap.mirror:
                if v is None:
                    if k not in self._listed:
                        return
                    self.adapter.data.remove((k, self._listed[k]))
                    del self._listed[k]
                elif k not in self._listed:
                    self._listed[k] = v
                    self.adapter.data.append((k, v))
                else:
                    already = (k, self._listed[k])
                    i = self.adapter.data.index(already)
                    self.adapter.data[i] = (k, v)
                    self._listed[k] = v

        self.mirrormap.bind(mirror=listen)

    def _reg_widget(self, w, *args):
        if self.mirrormap is None:
            Clock.schedule_once(partial(self._reg_widget, w), 0)
            return

        def listen(mir, *args):
            w.value = self.mirrormap.mirror[w.key]
        self._listeners[w.key] = listen
        self.mirrormap.bind(mirror=listen)

    def _unreg_widget(self, w):
        if w.key in self.listeners:
            self.mirrormap.unbind(mirror=self._listeners[w.key])

    def _set_value(self, k, v):
        self.mirror.remote[k] = v


kv = """
<StatRowListItem>:
    orientation: 'horizontal'
    height: 30
    StatRowKey:
        id: keycell
        font_name: root.font_name
        font_size: root.font_size
        hint_text: str(root.key)
        multiline: False
    StatRowValue:
        id: valcell
        font_name: root.font_name
        font_size: root.font_size
        hint_text: str(root.value)
        multiline: False
"""
Builder.load_string(kv)
