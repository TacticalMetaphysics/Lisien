# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from functools import partial
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.properties import (
    AliasProperty,
    NumericProperty,
    ObjectProperty,
    StringProperty
)
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen
from .stores import StringsEditor
Factory.register('StringsEditor', cls=StringsEditor)




class StringsEdScreen(Screen):
    engine = ObjectProperty()
    toggle = ObjectProperty()

    def add_string(self, *args):
        ed = self.ids.editor
        ed.save()
        newname = self.ids.strname.text
        if newname in self.engine.string:
            return
        self.engine.string[newname] = ed.source = ''
        assert(newname in self.engine.string)
        self.ids.strname.text = ''
        ed.name = newname
        ed._trigger_redata_reselect()


Builder.load_string("""
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
<StringsEdScreen>:
    name: 'strings'
    BoxLayout:
        orientation: 'vertical'
        StringsEditor:
            id: editor
            table: 'strings'
            store: root.engine.string if root.engine else None
            size_hint_y: 0.95
        BoxLayout:
            orientation: 'horizontal'
            size_hint_y: 0.05
            TextInput:
                id: strname
                hint_tex: 'New string name'
            Button:
                id: newstr
                text: 'New'
                on_press: root.add_string()
            Button:
                id: close
                text: 'Close'
                on_press: root.toggle()
""")
