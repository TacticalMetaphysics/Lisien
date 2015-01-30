# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from kivy.clock import Clock
from kivy.properties import StringProperty, NumericProperty
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout


class StringInput(BoxLayout):
    font_name = StringProperty('DroidSans')
    font_size = NumericProperty(12)
    name = StringProperty()
    source = StringProperty()

    def on_name(self, *args):
        if 'stringname' not in self.ids:
            Clock.schedule_once(self.on_name, 0)
            return
        if self.ids.stringname.text != self.name:
            self.ids.stringname.text = self.name

    def on_source(self, *args):
        if 'string' not in self.ids:
            Clock.schedule_once(self.on_source, 0)
            return
        if self.ids.string.text != self.source:
            self.ids.stringname.text = self.source


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
