# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableWidgetMetaclass
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.image import Image
from kivy.properties import StringProperty, ObjectProperty, AliasProperty
from re import match, compile
from logging import getLogger


"""Simple menu widgets"""


logger = getLogger(__name__)


ON_CLICK_RE = compile("""([a-zA-Z0-9_]+)\((.*)\)""")


class MenuItem(RelativeLayout):
    menu = ObjectProperty()
    img_name = StringProperty(allownone=True)
    string_name = StringProperty(allownone=True)
    on_click = StringProperty('')

    def __init__(self, **kwargs):
        RelativeLayout.__init__(self, **kwargs)

        but = Button()

        def upd_text(*args):
            but.text = self.menu.closet.get_text(self.string_name)
        self.bind(string_name=upd_text)

        if self.string_name is not None:
            starttext = self.menu.closet.get_text(self.string_name)
            if self.string_name[0] == "@":
                self.menu.closet.skeleton["strings"][
                    self.string_name[1:]].bind(
                    touches=upd_text)
        else:
            starttext = ''

        but.text = starttext

        ocmatch = match(ON_CLICK_RE, self.on_click)
        if ocmatch is not None:
            (ocfn, ocargs) = ocmatch.groups()
            (on_click_fun, ARG_RE) = self.menu.closet.menu_cbs[ocfn]
            on_click_args = match(ARG_RE, ocargs).groups()

            def on_click(*args):
                on_click_fun(*on_click_args)
            but.bind(on_press=on_click)

        self.add_widget(but)

        if self.img_name is not None:
            imgtex = self.menu.closet.get_texture(self.img_name)
            imgsrc = self.menu.closet.skeleton["img"][
                self.img_name]["path"]
            icon = Image(pos=(0, 0), source=imgsrc, texture=imgtex)

            def upd_icon(*args):
                imgsrc = self.menu.closet.skeleton["img"][
                    self.img_name][self.menu.closet.language]["path"]
                icon.source = imgsrc
                icon.texture = self.menu.closet.get_texture(self.img_name)
            self.bind(img_name=upd_icon)
            self.add_widget(icon)


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
    closet = ObjectProperty()
    name = StringProperty()
    pos_hint = AliasProperty(
        lambda self: self.get_pos_hint(),
        lambda self, v: None)
    size_hint = AliasProperty(
        lambda self: self.get_size_hint(),
        lambda self, v: None)

    def __str__(self):
        return self.name

    def __init__(self, **kwargs):
        kwargs["orientation"] = "vertical"
        kwargs["spacing"] = 10
        BoxLayout.__init__(self, **kwargs)
        for rd in self.closet.skeleton["menu_item"][unicode(self)].iterrows():
            it = MenuItem(
                menu=self,
                string_name=rd["text"],
                img_name=rd["icon"])
            self.add_widget(it)

    def get_pos_hint(self):
        rd = self.closet.skeleton["menu"][unicode(self)]
        return {'x': rd['x'], 'y': rd['y']}

    def get_size_hint(self):
        rd = self.closet.skeleton["menu"][unicode(self)]
        return (rd['w'], rd['h'])
