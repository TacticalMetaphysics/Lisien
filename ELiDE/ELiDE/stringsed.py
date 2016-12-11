# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
from kivy.clock import Clock
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from .stores import (
    StoreEditor,
    StoreList,
    StringInput
)
from .util import trigger


class StringStoreList(StoreList):
    def iter_data(self):
        yield from self.store.lang_items()


class StringsEditor(StoreEditor):
    list_cls = StringStoreList

    def add_editor(self, *args):
        if self.selection is None:
            Clock.schedule_once(self.add_editor, 0)
            return
        self._editor = StringInput(
            font_name=self.font_name,
            font_size=self.font_size,
            name=self.name,
            source=self.source
        )
        self.bind(
            font_name=self._editor.setter('font_name'),
            font_size=self._editor.setter('font_size'),
            name=self._editor.setter('name'),
            source=self._editor.setter('source')
        )
        self.add_widget(self._editor)

    @trigger
    def save(self, *args):
        self.source = self._editor.source
        if self.name != self._editor.name:
            del self.store[self.name]
            self.name = self._editor.name
        self.store[self.name] = self.source


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
