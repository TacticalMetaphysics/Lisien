# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from kivybits import SaveableWidgetMetaclass
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import (
    AliasProperty,
    StringProperty,
    ObjectProperty,
    ListProperty,
    NumericProperty)
from re import match, compile
from logging import getLogger


"""Simple menu widgets"""


logger = getLogger(__name__)


ON_CLICK_RE = compile("""([a-zA-Z0-9_]+)\((.*)\)""")


class MenuButton(Button):
    oncl = ObjectProperty()
    fargs = ListProperty()

    def on_press(self):
        self.oncl(self, *self.fargs)


class Menu(BoxLayout):
    __metaclass__ = SaveableWidgetMetaclass
    tables = [
        ('menu',
         {'name': 'text not null',
          'x': 'float not null',
          'y': 'float not null',
          'w': 'float not null',
          'h': 'float not null',
          'text_style': "text not null default 'default_style'",
          'symbol_style': "text not null default 'solid_symbols'"},
         ('name',),
         {'text_style': ('style', 'name'),
          'symbol_style': ('style', 'name')},
         []),
        ('menu_item',
         {'menu': 'text not null',
          'idx': 'integer not null',
          'text': 'text',
          'on_click': 'text not null',
          'closer': 'boolean not null default 1',
          'symbolic': 'boolean not null default 0'},
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
    text_style = ObjectProperty()
    symbol_style = ObjectProperty()
    completedness = NumericProperty(0)

    def __unicode__(self):
        return self.name

    def __str__(self):
        return str(self.name)

    def on_text_style(self, i, v):
        self.completedness += 1

    def on_symbol_style(self, i, v):
        self.completedness += 1

    def on_completedness(self, i, v):
        if v == 2:
            self.finalize()

    def finalize(self):
        for bone in self.closet.skeleton["menu_item"][unicode(self)].iterbones():
            if bone.symbolic:
                style = self.symbol_style
            else:
                style = self.text_style
            ocmatch = match(ON_CLICK_RE, bone.on_click)
            if ocmatch is not None:
                (ocfn, ocargs) = ocmatch.groups()
                (on_click_fun, ARG_RE) = self.closet.menu_cbs[ocfn]
                ocargm = match(ARG_RE, ocargs)
                if ocargm is None:
                    fargs = []
                else:
                    fargs = list(ocargm.groups())

                it = MenuButton(
                    font_name=style.fontface,
                    font_size=style.fontsize,
                    oncl=on_click_fun,
                    fargs=fargs,
                    text=self.closet.get_text(bone.text))

                def retext(*args):
                    it.text = self.closet.get_text(bone.text)
                self.closet.register_text_listener(bone.text, retext)
                self.add_widget(it)

    def get_pos_hint(self):
        rd = self.closet.skeleton["menu"][unicode(self)]
        return {'x': rd['x'], 'y': rd['y']}

    def get_size_hint(self):
        rd = self.closet.skeleton["menu"][unicode(self)]
        return (rd['w'], rd['h'])
