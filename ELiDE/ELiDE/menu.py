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
import os
from kivy.properties import ObjectProperty
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import Screen
from kivy.clock import Clock
from kivy.lang import Builder


class MenuTextInput(TextInput):
    """Special text input for setting the branch"""
    set_value = ObjectProperty()

    def __init__(self, **kwargs):
        """Disable multiline, and bind ``on_text_validate`` to ``on_enter``"""
        kwargs['multiline'] = False
        super().__init__(**kwargs)
        self.bind(on_text_validate=self.on_enter)

    def on_enter(self, *args):
        """Call the setter and blank myself out so that my hint text shows
        up. It will be the same you just entered if everything's
        working.

        """
        if self.text == '':
            return
        self.set_value(Clock.get_time(), int(self.text))
        self.text = ''
        self.focus = False

    def on_focus(self, *args):
        """If I've lost focus, treat it as if the user hit Enter."""
        if not self.focus:
            self.on_enter(*args)

    def on_text_validate(self, *args):
        """Equivalent to hitting Enter."""
        self.on_enter()


class MenuIntInput(MenuTextInput):
    """Special text input for setting the turn or tick"""
    def insert_text(self, s, from_undo=False):
        """Natural numbers only."""
        return super().insert_text(
            ''.join(c for c in s if c in '0123456789'),
            from_undo
        )


class DirPicker(Screen):
    toggle = ObjectProperty()
    start = ObjectProperty()

    def open(self, path):
        os.chdir(path)
        self.start()
        self.toggle()



Builder.load_string("""
#: import os os
<DirPicker>:
    name: 'mainmenu'
    start: app.trigger_start_subprocess
    BoxLayout:
        orientation: 'vertical'
        Label:
            text: 'Pick a directory to create or load a simulation in'
            size_hint_y: None
        FileChooserListView:
            id: filechooser
            path: os.getcwd()
        Button:
            text: 'Work here'
            size_hint_y: 0.1
            on_release: root.open(filechooser.path)
""")