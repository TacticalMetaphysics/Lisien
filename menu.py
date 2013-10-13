# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableWidgetMetaclass
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import AliasProperty
import re
from logging import getLogger


"""Simple menu widgets"""


logger = getLogger(__name__)


ON_CLICK_RE = re.compile("""([a-zA-Z0-9_]+)\((.*)\)""")


class MenuItem(Button):
    text = AliasProperty(
        lambda self: self.menu.closet.get_text(self.string_name),
        lambda self, v: None)
    font_size = AliasProperty(
        lambda self: self.menu.style.fontsize,
        lambda self, v: None)

    def __init__(self, menu, text=None, img=None, closer=True, **kwargs):
        self.menu = menu
        self.string_name = text
        kwargs["size_hint_x"] = None
        Button.__init__(self, **kwargs)
        if img is not None:
            self.add_widget(self.menu.closet.get_image(img))


class Menu(BoxLayout):
    __metaclass__ = SaveableWidgetMetaclass
    tables = [
        ('menu',
         {'name': 'text not null',
          'x': 'float not null',
          'y': 'float not null',
          'w': 'float not null',
          'h': 'float not null',
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
    pos_hint = AliasProperty(
        lambda self: self.get_pos_hint(),
        lambda self, v: None)
    size_hint = AliasProperty(
        lambda self: self.get_size_hint(),
        lambda self, v: None)
    style = AliasProperty(
        lambda self: self.closet.get_style(
            self.closet.skeleton["menu"][unicode(self)]["style"]),
        lambda self, v: None)

    def __str__(self):
        return self._name

    def __init__(self, closet, name, **kwargs):
        kwargs["orientation"] = "vertical"
        kwargs["spacing"] = 10
        self.closet = closet
        self._name = name
        BoxLayout.__init__(self, **kwargs)
        for rd in self.closet.skeleton["menu_item"][unicode(self)].iterrows():
            it = MenuItem(self, rd["text"], rd["icon"], rd["closer"] == 1)
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

    def get_pos_hint(self):
        rd = self.closet.skeleton["menu"][unicode(self)]
        return {'x': rd['x'], 'y': rd['y']}

    def get_size_hint(self):
        rd = self.closet.skeleton["menu"][unicode(self)]
        return (rd['w'], rd['h'])
