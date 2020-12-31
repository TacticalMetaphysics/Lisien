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
from kivy.properties import OptionProperty, ObjectProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.modalview import ModalView
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import Screen
from kivy.clock import Clock
from kivy.lang import Builder
from .gen import GridGeneratorDialog


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


class GeneratorButton(Button):
    pass


class WorldStartConfigurator(BoxLayout):
    """Give options for how to initialize the world state"""
    grid_config = ObjectProperty()
    generator_type = OptionProperty(None, options=['grid'], allownone=True)
    dismiss = ObjectProperty()
    toggle = ObjectProperty()
    starter = ObjectProperty()
    init_board = ObjectProperty()
    generator_dropdown = ObjectProperty()

    def on_generator_dropdown(self, *args):
        def select_txt(btn):
            self.generator_dropdown.select(btn.text)
        for opt in ['None', 'Grid']:
            self.generator_dropdown.add_widget(GeneratorButton(text=opt, on_release=select_txt))
        self.generator_dropdown.bind(on_select=self.select_generator_type)
    
    def select_generator_type(self, instance, value):
        self.ids.drop.text = value
        if value == 'None':
            self.ids.controls.clear_widgets()
            self.generator_type = None
        elif value == 'Grid':
            self.ids.controls.clear_widgets()
            self.ids.controls.add_widget(self.grid_config)
            self.grid_config.size = self.ids.controls.size
            self.grid_config.pos = self.ids.controls.pos
            self.generator_type = 'grid'

    def start(self, *args):
        if self.generator_type == 'grid':
            if self.grid_config.validate():
                engine = self.starter()
                self.grid_config.generate(engine)
                self.init_board()
                self.toggle()
                self.dismiss()
            else:
                # TODO show error
                return
        else:
            self.starter()
            self.init_board()
            self.toggle()
            self.dismiss()


class DirPicker(Screen):
    toggle = ObjectProperty()
    start = ObjectProperty()
    init_board = ObjectProperty()

    def open(self, path):
        os.chdir(path)
        if 'world.db' not in os.listdir(path):
            # TODO show a configurator, accept cancellation, extract init params
            if not hasattr(self, 'config_popover'):
                self.config_popover = ModalView()
                self.configurator = WorldStartConfigurator(
                    grid_config=GridGeneratorDialog(),
                    dismiss=self.config_popover.dismiss,
                    toggle=self.toggle,
                    generator_dropdown=DropDown()
                )
                self.config_popover.add_widget(self.configurator)
            self.config_popover.open()
            return
        self.start()
        self.init_board()
        self.toggle()



Builder.load_string("""
#: import os os
<GeneratorButton>:
    size_hint_y: None
    height: self.texture_size[1] + 10
<WorldStartConfigurator>:
    orientation: 'vertical'
    init_board: app.init_board
    starter: app.start_subprocess
    Label:
        text: 'Generate an initial map?'
    Button:
        id: drop
        text: 'None'
        on_release: root.generator_dropdown.open(drop)
    Widget:
        id: controls
        size_hint_y: None
        height: 200
    BoxLayout:
        orientation: 'horizontal'
        Button:
            text: 'OK'
            on_release:
                root.start()
        Button:
            text: 'Cancel'
            on_release:
                controls.clear_widgets()
                controls.size_hint_y = 0
                root._trigger_layout()
                root.dismiss()
<DirPicker>:
    name: 'mainmenu'
    start: app.start_subprocess
    init_board: app.init_board
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