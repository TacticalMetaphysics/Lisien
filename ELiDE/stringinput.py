# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from functools import partial
from kivy.clock import Clock
from kivy.properties import (
    AliasProperty,
    NumericProperty,
    StringProperty
)
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout


class StringInput(BoxLayout):
    font_name = StringProperty('DroidSans')
    font_size = NumericProperty(12)
    name = StringProperty()
    source = StringProperty()

    def _get_name(self):
        if 'stringname' not in self.ids:
            return ''
        return self.ids.stringname.text

    def _set_name(self, v, *args):
        if 'stringname' not in self.ids:
            Clock.schedule_once(partial(self._set_name, v), 0)
            return
        self.ids.stringname.text = v

    name = AliasProperty(_get_name, _set_name)

    def _get_source(self):
        if 'string' not in self.ids:
            return ''
        return self.ids.string.text

    def _set_source(self, v, *args):
        if 'string' not in self.ids:
            Clock.schedule_once(partial(self._set_source, v), 0)
            return
        self.ids.string.text = v

    source = AliasProperty(_get_source, _set_source)


kv = """
<StringInput>:
    orientation: 'vertical'
    BoxLayout:
        size_hint_y: 0.05
        Label:
            text: 'Title: '
            size_hint_x: None
            width: self.texture_size[0]
        TextInput:
            id: stringname
            font_name: root.font_name
            font_size: root.font_size
    TextInput:
        id: string
        font_name: root.font_name
        font_size: root.font_size
"""
Builder.load_string(kv)
