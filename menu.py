# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableWidgetMetaclass
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.properties import AliasProperty, StringProperty
import re
from logging import getLogger


"""Simple menu widgets"""


logger = getLogger(__name__)


ON_CLICK_RE = re.compile("""([a-zA-Z0-9_]+)\((.*)\)""")


class MenuItem(Button):
    image_name = StringProperty()
    def __init__(self, menu, text=None, img=None, closer=True, **kwargs):
        self.menu = menu
        self.string_name = text
        if self.string_name is not None:
            starttext = self.menu.closet.get_text(self.string_name)
            if self.string_name[0] == "@":
                self.menu.closet.skeleton["strings"][self.string_name[1:]].bind(
                    touches=self.upd_text)
        else:
            starttext = ''
        self.closer = closer
        Button.__init__(self, text=starttext, size_hint_x=None, **kwargs)
        if img is not None:
            self.image_name = img
            icon = Image(
                texture=self.menu.closet.get_texture(self.image_name))
            def upd_icon(*args):
                icon.texture = self.menu.closet.get_texture(self.image_name)
            self.bind(image_name=upd_icon)
            self.add_widget(icon)

    def upd_text(self, *args):
        self.text = self.closet.get_text(self.string_name)


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

    def __str__(self):
        return self.name

    def __init__(self, closet, name, **kwargs):
        self.closet = closet
        self.name = name
        kwargs["orientation"] = "vertical"
        kwargs["spacing"] = 10
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
                    def on_click(*args):
                        call(it, *args)
                else:
                    def on_click(*args):
                        call(it)
            else:
                def on_click(*args):
                    pass
            it.bind(on_release=on_click)

    def get_pos_hint(self):
        rd = self.closet.skeleton["menu"][unicode(self)]
        return {'x': rd['x'], 'y': rd['y']}

    def get_size_hint(self):
        rd = self.closet.skeleton["menu"][unicode(self)]
        return (rd['w'], rd['h'])
