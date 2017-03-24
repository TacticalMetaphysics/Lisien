# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
from functools import partial
from kivy.clock import Clock
from kivy.properties import (
    AliasProperty,
    ObjectProperty
)
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from .stores import StoreList


class StringStoreList(StoreList):
    def iter_data(self):
        yield from self.store.lang_items()


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


class StringInput(BoxLayout):
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
    TextInput:
        id: string
<StringsEdScreen>:
    name: 'strings'
    BoxLayout:
        orientation: 'vertical'
        StringInput:
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
