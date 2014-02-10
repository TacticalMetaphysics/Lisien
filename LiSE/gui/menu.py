# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from kivy.properties import (
    BooleanProperty,
    StringProperty,
    ObjectProperty
)
from kivy.uix.widget import Widget
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.clock import Clock

"""Menus that are actually just stacks of buttons.

I'll probably change over to drop menus at some point."""


class MenuWidget(Widget):
    closet = ObjectProperty()
    fun = ObjectProperty()
    symbolic = BooleanProperty(False)
    stringname = StringProperty()

    def __init__(self, **kwargs):
        super(MenuWidget, self).__init__(**kwargs)
        self.finalize()

    def finalize(self, *args):
        if not (self.closet and self.stringname):
            Clock.schedule_once(self.finalize, 0)
            return
        self.closet.register_text_listener(self.stringname, self.retext)
        self.initext()


class MenuButton(Button, MenuWidget):
    def initext(self):
        if self.symbolic:
            self.font_name = 'LiSE/gui/assets/Entypo.ttf'
            self.font_size = 30
        self.text = self.closet.get_text(self.stringname)

    def on_press(self, *args):
        self.fun()

    def retext(self, skel, k, v):
        if k == self.closet.language:
            self.label.text = v


class MenuIntInput(TextInput, MenuWidget):
    def initext(self):
        self.text = ''
        self.hint_text = self.closet.get_text(self.stringname)

    def on_focus(self, *args):
        if not self.focus:
            try:
                self.fun(int(self.text))
            except ValueError:
                self.initext()

    def retext(self, time):
        self.hint_text = str(time)


class Menu(BoxLayout):
    """A stack of buttons that call functions in the closet."""
    font_name = StringProperty
    closet = ObjectProperty()
