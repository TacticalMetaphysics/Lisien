# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
from functools import partial

from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.uix.recycleview import RecycleView
from kivy.properties import ListProperty, ObjectProperty, StringProperty

from .util import SelectableRecycleBoxLayout


# TODO: Visual preview
# TODO: Background image chooser

class CharactersRecycleBoxLayout(SelectableRecycleBoxLayout):
    character_name = StringProperty()

    def apply_selection(self, index, view, is_selected):
        super().apply_selection(index, view, is_selected)
        self.character_name = view.text


class CharactersView(RecycleView):
    character_name = StringProperty()

    def __init__(self, **kwargs):
        self.i2name = {}
        self.name2i = {}
        super().__init__(**kwargs)


class CharactersScreen(Screen):
    toggle = ObjectProperty()
    charsview = ObjectProperty()
    engine = ObjectProperty()
    character_name = StringProperty()
    names = ListProperty()
    new_board = ObjectProperty()

    def on_character_name(self, *args):
        print('CharactersScreen.character_name = ' + self.character_name)

    def new_character(self, name, *args):
        self.engine.add_character(name)
        self.ids.newname.text = ''
        i = len(self.charsview.data)
        self.charsview.i2name[i] = name
        self.charsview.name2i[name] = i
        self.charsview.data.append({'index': i, 'text': name})
        self.names.append(name)
        self.new_board(name)

    def _trigger_new_character(self, name):
        part = partial(self.new_character, name)
        Clock.unschedule(part)
        Clock.schedule_once(part)

    def _munge_names(self, names):
        for i, name in enumerate(names):
            self.charsview.i2name[i] = name
            self.charsview.name2i[name] = i
            yield {'index': i, 'text': name}

    def on_names(self, *args):
        if not self.charsview:
            Clock.schedule_once(self.on_names, 0)
            return
        self.charsview.data = list(self._munge_names(self.names))

    def on_charsview(self, *args):
        self.charsview.bind(character_name=self.setter('character_name'))


Builder.load_string("""
<CharactersView>:
    viewclass: 'RecycleToggleButton'
    character_name: boxl.character_name
    CharactersRecycleBoxLayout:
        id: boxl
        multiselect: False
        default_size: None, dp(56)
        default_size_hint: 1, None
        size_hint_y: None
        height: self.minimum_height
        orientation: 'vertical'
<CharactersScreen>:
    name: 'chars'
    charsview: charsview
    BoxLayout:
        id: chars
        orientation: 'vertical'
        CharactersView:
            id: charsview
            size_hint_y: 0.9
            character_name: root.character_name
        TextInput:
            id: newname
            hint_text: 'New character name'
            write_tab: False
            multiline: False
        Button:
            text: '+'
            on_press: root._trigger_new_character(newname.text)
        Button:
            text: 'Close'
            on_press: root.toggle()
""")
