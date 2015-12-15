# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
from kivy.lang import Builder
from kivy.adapters.listadapter import ListAdapter
from kivy.uix.listview import ListView, ListItemButton
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen

from kivy.properties import AliasProperty, ObjectProperty, StringProperty


# TODO: Visual preview
# TODO: Background image chooser

class CharListView(ListView):
    character_name = StringProperty()
    set_char = ObjectProperty()

    def __init__(self, **kwargs):
        if 'adapter' not in kwargs:
            kwargs['adapter'] = ListAdapter(
                data=[],
                selection_mode='single',
                allow_empty_selection=True,
                cls=ListItemButton,
                args_converter=lambda i, char: {
                    'size_hint_y': None,
                    'height': 30,
                    'text': char
                }
            )
        super().__init__(**kwargs)

    def on_adapter(self, *args):
        def selchange(*args):
            if len(self.adapter.selection) == 0:
                try:
                    i = self.adapter.data.index(self.character_name)
                except ValueError:
                    return
                view = self.adapter.get_view(i)
                self.adapter.select_list([view])
            elif self.character_name != self.adapter.selection[0].text:
                self.set_char(self.adapter.selection[0].text)
        self.adapter.bind(
            on_selection_change=selchange
        )


class CharactersBox(BoxLayout):
    engine = ObjectProperty()
    toggle = ObjectProperty()
    character = ObjectProperty()
    select_character = ObjectProperty()

    def _get_character_name(self):
        if not self.character:
            return ''
        return self.character.name

    character_name = AliasProperty(_get_character_name, lambda self, v: None, bind=('character',))

    def set_char(self, char):
        self.select_character(self.engine.character[char])


class CharactersScreen(Screen):
    toggle = ObjectProperty()
    select_character = ObjectProperty()
    charsview = ObjectProperty()
    engine = ObjectProperty()
    character = ObjectProperty()

    def on_character(self, *args):
        pass

    def new_character(self, name, *args):
        self.select_character(self.engine.new_character(name))


Builder.load_string("""
<CharactersScreen>:
    name: 'chars'
    charsview: charsview
    CharactersBox:
        id: chars
        orientation: 'vertical'
        toggle: root.toggle
        select_character: root.select_character
        engine: root.engine
        CharListView:
            id: charsview
            character_name: chars.character_name
            set_char: chars.set_char
        BoxLayout:
            size_hint_y: 0.05
            TextInput:
                id: newname
                hint_text: 'New character name'
                write_tab: False
            Button:
                text: '+'
                on_press: root.new_character(self.text)
            Button:
                text: 'Close'
                on_press: root.toggle()
""")
