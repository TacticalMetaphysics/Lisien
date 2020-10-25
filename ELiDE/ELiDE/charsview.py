# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
        if is_selected:
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
    wallpaper_path = StringProperty()
    names = ListProperty()
    new_board = ObjectProperty()
    push_character_name = ObjectProperty()

    def new_character(self, name, *args):
        self.engine.add_character(name)
        self.ids.newname.text = ''
        i = len(self.charsview.data)
        self.charsview.i2name[i] = name
        self.charsview.name2i[name] = i
        self.charsview.data.append({'index': i, 'text': name})
        self.names.append(name)
        self.new_board(name)
        self.push_character_name(name)

    def _trigger_new_character(self, name):
        part = partial(self.new_character, name)
        if hasattr(self, '_scheduled_new_character'):
            Clock.unschedule(self._scheduled_new_character)
        self._scheduled_new_character = Clock.schedule_once(part)

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
        if not self.push_character_name:
            Clock.schedule_once(self.on_charsview, 0)
            return
        self.charsview.bind(character_name=self.setter('character_name'))
        self.bind(character_name=self.push_character_name)


Builder.load_string("""
#: import resource_find kivy.resources.resource_find
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
            size_hint_y: 0.8
            character_name: root.character_name
        TextInput:
            id: newname
            size_hint_y: 0.1
            hint_text: 'New character name'
            write_tab: False
            multiline: False
        Button:
            text: '+'
            on_release: root._trigger_new_character(newname.text)
            size_hint_y: 0.05
        Button:
            text: 'Close'
            on_release: root.toggle()
            size_hint_y: 0.05
""")
