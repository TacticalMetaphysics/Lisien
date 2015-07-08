from kivy.lang import Builder
from kivy.properties import ObjectProperty, StringProperty
from kivy.adapters.listadapter import ListAdapter
from kivy.uix.listview import ListView, ListItemButton
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen


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
                i = self.adapter.data.index(self.character_name)
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
    character_name = StringProperty()
    new_character = ObjectProperty()
    select_character = ObjectProperty()

    def set_char(self, char):
        self.select_character(self.engine.character[char])
        self.toggle()


class CharactersScreen(Screen):
    toggle = ObjectProperty()
    select_character = ObjectProperty()
    new_character = ObjectProperty()


Builder.load_string("""
<CharactersScreen>:
    name: 'chars'
    CharactersBox:
        id: chars
        orientation: 'vertical'
        toggle: root.toggle
        select_character: root.select_character
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
                text: 'Cancel'
                on_press: root.toggle()
""")
