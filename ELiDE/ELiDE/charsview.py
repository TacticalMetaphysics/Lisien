# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
from kivy.lang import Builder
from kivy.uix.recycleview import RecycleView
from kivy.uix.screenmanager import Screen
from kivy.properties import ListProperty, ObjectProperty, StringProperty


# TODO: Visual preview
# TODO: Background image chooser

class CharactersView(RecycleView):
    character_name = StringProperty()
    set_char = ObjectProperty()
    selection = ObjectProperty()


class CharactersScreen(Screen):
    toggle = ObjectProperty()
    select_character = ObjectProperty()
    charsview = ObjectProperty()
    engine = ObjectProperty()
    character = ObjectProperty()
    character_name = StringProperty()
    data = ListProperty()

    def set_char(self, char):
        self.select_character(self.engine.character[char])

    def new_character(self, name, *args):
        self.select_character(self.engine.new_character(name))


Builder.load_string("""
<CharactersView>:
    viewclass: 'RecycleToggleButton'
    SelectableRecycleBoxLayout:
        default_size: None, dp(56)
        default_size_hint: 1, None
        size_hint_y: None
        height: self.minimum_height
        orientation: 'vertical'
<CharactersScreen>:
    name: 'chars'
    character_name: self.character.name if self.character else ''
    charsview: charsview
    BoxLayout:
        id: chars
        orientation: 'vertical'
        CharactersView:
            id: charsview
            size_hint_y: 0.9
            character_name: root.character_name
            set_char: root.set_char
            data: root.data
        TextInput:
            id: newname
            hint_text: 'New character name'
            write_tab: False
            multiline: False
        Button:
            text: '+'
            on_press: root.new_character(self.text)
        Button:
            text: 'Close'
            on_press: root.toggle()
""")
