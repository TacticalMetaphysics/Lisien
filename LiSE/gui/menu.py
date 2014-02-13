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


class MenuButton(Button):
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

    def on_press(self, *args):
        self.fun()

    def retext(self, skel, k, v):
        if k == self.closet.language:
            self.label.text = v

    def initext(self):
        if self.symbolic:
            self.font_name = 'LiSE/gui/assets/Entypo.ttf'
            self.font_size = 30
        self.text = self.closet.get_text(self.stringname)


class MenuTextInput(TextInput):
    closet = ObjectProperty()

    def __init__(self, **kwargs):
        super(MenuTextInput, self).__init__(**kwargs)
        self.finalize()

    def finalize(self, *args):
        if not (
                self.closet and
                self.rehint_registrar and
                self.hint_getter and
                self.value_setter):
            Clock.schedule_once(self.finalize, 0)
            return
        self.rehint_registrar(self.rehint)
        self.rehint()

    def rehint(self, *args):
        self.text = ''
        self.hint_text = self.hint_getter()

    def on_focus(self, *args):
        if not self.focus:
            try:
                self.value_setter(self.text)
            except ValueError:
                pass
            self.rehint()
        super(MenuTextInput, self).on_focus(*args)

    def rehint_registrar(self, reh):
        raise NotImplementedError(
            "Abstract method")

    def hint_getter(self):
        raise NotImplementedError(
            "Abstract method")

    def value_setter(self, v):
        raise NotImplementedError(
            "Abstract method")


class MenuBranchInput(MenuTextInput):
    def rehint_registrar(self, reh):
        self.closet.register_branch_listener(reh)

    def hint_getter(self):
        return str(self.closet.branch)

    def value_setter(self, v):
        self.closet.branch = int(v)


class MenuTickInput(MenuTextInput):
    def rehint_registrar(self, reh):
        self.closet.register_tick_listener(reh)

    def hint_getter(self):
        return str(self.closet.tick)

    def value_setter(self, v):
        self.closet.tick = int(v)


class Menu(BoxLayout):
    """A stack of buttons that call functions in the closet."""
    font_name = StringProperty
    closet = ObjectProperty()
