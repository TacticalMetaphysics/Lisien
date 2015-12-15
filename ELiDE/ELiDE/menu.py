# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
from kivy.properties import ObjectProperty
from kivy.uix.textinput import TextInput


class MenuTextInput(TextInput):
    """Special text input for setting the branch"""
    setter = ObjectProperty()

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
        self.setter(self.text)
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
    """Special text input for setting the tick"""
    def insert_text(self, s, from_undo=False):
        """Natural numbers only."""
        return super().insert_text(
            ''.join(c for c in s if c in '0123456789'),
            from_undo
        )
