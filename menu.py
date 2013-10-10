# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableWidgetMetaclass
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.modalview import ModalView
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import (
    AliasProperty,
    StringProperty,
    ObjectProperty)
import re
from logging import getLogger


"""Simple menu widgets"""


logger = getLogger(__name__)


ON_CLICK_RE = re.compile("""([a-zA-Z0-9_]+)\((.*)\)""")


class GetterLabel(Label):
    text = AliasProperty(
        lambda self: self.closet.get_text(self.string_name),
        lambda self, v: None)

    def __init__(self, closet, text):
        self.closet = closet
        self.string_name = text


class MenuItem(Button):
    def __init__(self, menu, text=None, img=None, closer=True):
        self.menu = menu
        self._text = text
        Button.__init__(self, dismiss_on_select=closer)
        if img is not None:
            self.add_widget(self.menu.closet.get_img(img))
        if text is not None:
            self.add_widget(GetterLabel(self.menu.closet, text))


class Menu(object):
    __metaclass__ = SaveableWidgetMetaclass
    _name = StringProperty()
    closet = ObjectProperty()
    tables = [
        ('menu',
         {'name': 'text not null',
          'style': "text not null default 'SmallDark'"},
         ('name',),
         {'style': ('style', 'name')},
         []),
        ('menu_item',
         {'menu': 'text not null',
          'idx': 'integer not null',
          'text': 'text',
          'icon': 'text',
          'on_click': 'text not null',
          'closer': 'boolean not null default 1'},
         ('menu', 'idx'),
         {'menu': ('menu', 'name')},
         [])]

    @property
    def style(self):
        return self.closet.get_style(self._rowdict["style"])

    @property
    def _rowdict(self):
        return self.closet.skeleton["menu"][unicode(self)]

    def __str__(self):
        return self._name

    def build(self):
        for rd in self.closet.skeleton["menu_item"][unicode(self)].iterrows():
            it = MenuItem(self, rd["text"], rd["icon"], rd["closer"])
            self.add_widget(it)
            oc = re.match(ON_CLICK_RE, rd["on_click"])
            if oc is not None:
                (cb, argstr) = oc.groups()
                (call, argre) = it.menu.closet.menu_cbs[cb]
                argmatch = re.match(argre, argstr)
                if argmatch is not None:
                    args = argmatch.groups()
                    on_click = lambda: call(it, *args)
                else:
                    on_click = lambda: call(it)
            else:
                on_click = lambda: None
            it.bind(on_release=on_click)


class MainMenu(DropDown, Menu):
    def __init__(self, closet, **kwargs):
        self._name = "Main"
        self.closet = closet
        DropDown.__init__(self, **kwargs)


class SubMenu(Widget, Menu):
    def __init__(self, closet, name, **kwargs):
        self._name = name
        self.closet = closet
        Widget.__init__(self, **kwargs)

    def build(self):
        self.view = ModalView()
        self.layout = BoxLayout(
            orientation='vertical',
            pos_hint=(0.2, 0.0),
            size_hint=(0.8, 1.0))
        self.view.add(self.layout)
        Menu.build(self)

    def add_widget(self, w):
        self.layout.add_widget(w)
