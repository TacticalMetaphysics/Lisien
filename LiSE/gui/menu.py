# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from kivy.properties import (
    BooleanProperty,
    StringProperty,
    ObjectProperty,
    NumericProperty)
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.textinput import TextInput
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button

"""Menus that are actually just stacks of buttons.

I'll probably change over to drop menus at some point."""


class MenuWidget(Widget):
    closet = ObjectProperty()
    fun = ObjectProperty()
    arg = ObjectProperty(None, allownone=True)
    symbolic = BooleanProperty(False)
    stringname = StringProperty()
    completion = NumericProperty(0)

    def on_closet(self, *args):
        self.completion += 1

    def on_stringname(self, *args):
        self.completion += 1

    def on_completion(self, i, v):
        if v == 2:
            self.complete()

    def complete(self):
        self.closet.register_text_listener(self.stringname, self.retext)
        self.initext()

    def do_fun(self):
        if self.arg is None:
            self.fun()
        else:
            self.fun(self.arg)


class MenuButton(Button, MenuWidget):
    def initext(self):
        if self.symbolic:
            self.font_name = 'LiSE/gui/assets/Entypo.ttf'
            self.font_size = 30
        self.text = self.closet.get_text(self.stringname)

    def on_press(self, *args):
        self.do_fun()

    def retext(self, skel, k, v):
        if k == self.closet.language:
            self.label.text = v


class MenuIntInput(TextInput, MenuWidget):
    def initext(self):
        self.hint_text = self.closet.get_text(self.stringname)

    def on_text(self, i, v):
        try:
            i.arg = int(v)
            i.do_fun()
        except:
            pass
        self.text = ''

    def retext(self, time):
        self.hint_text = str(time)


class Menu(BoxLayout):
    """A stack of buttons that call functions in the closet."""
    font_name = StringProperty
    closet = ObjectProperty()
