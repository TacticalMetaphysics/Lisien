# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableWidgetMetaclass
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.properties import AliasProperty, StringProperty, ObjectProperty
import re
from logging import getLogger


"""Simple menu widgets"""


logger = getLogger(__name__)


ON_CLICK_RE = re.compile("""([a-zA-Z0-9_]+)\((.*)\)""")


class MenuItem(Button):
    text = AliasProperty(
        lambda self: self.menu.closet.get_text(self.rowdict["text"]),
        lambda self, v: None)
    rowdict = AliasProperty(
        lambda self: self.get_rowdict(),
        lambda self, v: None)
    size_hint_x = ObjectProperty(None)

    def __init__(self, menu, idx, **kwargs):
        self.menu = menu
        self.idx = idx
        Button.__init__(self, **kwargs)
        self.bind(rowdict=self.rowdict.touches)
        self.bind(text=self.rowdict)

    def __int__(self):
        return self.idx

    def __str__(self):
        return self.text

    def get_rowdict(self):
        if not isinstance(self.menu, Menu):
            return None
        return self.menu.closet.skeleton["menu_item"][
            unicode(self.menu)][int(self)]


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
    closet = ObjectProperty(None)
    name = StringProperty('')
    rowdict = AliasProperty(
        lambda self: self.get_rowdict(),
        lambda self, v: None)
    style = AliasProperty(
        lambda self: self.closet.get_style(self.rowdict["style"]),
        lambda self, v: None,
        bind=('rowdict',))
    pos_hint = AliasProperty(
        lambda self: {'x': self.rowdict['x'], 'y': self.rowdict['y']},
        lambda self, v: None,
        bind=('rowdict',))
    size_hint = AliasProperty(
        lambda self: (self.rowdict['w'], self.rowdict['h']),
        lambda self, v: None,
        bind=('rowdict',))
    name = StringProperty('')

    def __init__(self, **kwargs):
        BoxLayout.__init__(self, **kwargs)
        Clock.schedule_once(self.build, 0)

    def __str__(self):
        return self.name

    def get_rowdict(self):
        if self.name != '':
            return self.closet.skeleton["menu"][self.name]

    def build(self, *args):
        self.bind(rowdict=self.rowdict.touches)
        for rd in self.closet.skeleton["menu_item"][unicode(self)].iterrows():
            it = MenuItem(self, rd["idx"])
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
