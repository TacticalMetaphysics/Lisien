# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass
from kivy.graphics import Color
from kivy.properties import StringProperty, AliasProperty
from kivy.event import EventDispatcher


"""Simple data structures to hold style information for text and
things that contain text."""


class LiSEColor(object):
    __metaclass__ = SaveableMetaclass
    """Red, green, blue, and alpha values.

    This is just a container class for the (red, green, blue, alpha)
tuples that Pyglet uses to identify colors. The tup attribute will get
you that.

    """
    tables = [
        ("color",
         {'name': 'text not null',
          'red': 'float not null',
          'green': 'float not null',
          'blue': 'float not null',
          'alpha': 'float not null default 1.0'},
         ("name",),
         {},
         ["red between 0.0 and 1.0",
          "green between 0.0 and 1.0",
          "blue between 0.0 and 1.0"])]

    def __init__(self, closet, name):
        """Return a color with the given name, and the given values for red,
green, blue. Register in db.colordict.

        """
        self.closet = closet
        self.name = name

    def __str__(self):
        return self.name

    @property
    def rowdict(self):
        return self.closet.skeleton["color"][self.name]

    @property
    def red(self):
        return self.rowdict["red"]

    @property
    def green(self):
        return self.rowdict["green"]

    @property
    def blue(self):
        return self.rowdict["blue"]

    @property
    def alpha(self):
        return self.rowdict["alpha"]

    @property
    def rgb(self):
        return (self.red, self.green, self.blue)

    @property
    def rgba(self):
        return (self.red, self.green, self.blue, self.alpha)

    @property
    def kivy_color(self):
        return Color(self.red, self.green, self.blue, self.alpha)


class LiSEStyle(EventDispatcher):
    __metaclass__ = SaveableMetaclass
    """A collection of cogent information for rendering text and things
that contain text."""
    tables = [
        ("style",
         {"name": "text not null",
          "fontface": "text not null",
          "fontsize": "integer not null",
          "textcolor": "text not null",
          "spacing": "integer default 6",
          "bg_inactive": "text not null",
          "bg_active": "text not null",
          "fg_inactive": "text not null",
          "fg_active": "text not null"},
         ("name",),
         {"textcolor": ("color", "name"),
          "bg_inactive": ("color", "name"),
          "bg_active": ("color", "name"),
          "fg_inactive": ("color", "name"),
          "fg_active": ("color", "name")},
         [])]
    name = StringProperty('')
    rowdict = AliasProperty(
        lambda self: self.closet.skeleton["style"][self.name],
        lambda self, v: None)
    bg_inactive = AliasProperty(
        lambda self: self.closet.get_color(self.rowdict["bg_inactive"]),
        lambda self, v: None,
        bind=('rowdict',))
    bg_active = AliasProperty(
        lambda self: self.closet.get_color(self.rowdict["bg_active"]),
        lambda self, v: None,
        bind=('rowdict',))
    fg_inactive = AliasProperty(
        lambda self: self.closet.get_color(self.rowdict["fg_inactive"]),
        lambda self, v: None,
        bind=('rowdict',))
    fg_active = AliasProperty(
        lambda self: self.closet.get_color(self.rowdict["fg_active"]),
        lambda self, v: None,
        bind=('rowdict',))
    textcolor = AliasProperty(
        lambda self: self.closet.get_color(self.rowdict["textcolor"]),
        lambda self, v: None,
        bind=('rowdict',))
    fontface = AliasProperty(
        lambda self: self.rowdict["fontface"],
        lambda self, v: None,
        bind=('rowdict',))
    fontsize = AliasProperty(
        lambda self: self.rowdict["fontsize"],
        lambda self, v: None,
        bind=('rowdict',))
    spacing = AliasProperty(
        lambda self: self.rowdict["spacing"],
        lambda self, v: None,
        bind=('rowdict',))

    def __init__(self, closet, name):
        """Return a style by the given name, with the given face, font size,
spacing, and four colors: active and inactive variants for each of the
foreground and the background.

        """
        self.closet = closet
        self.name = name
        EventDispatcher.__init__(self)
        self.bind(rowdict=self.rowdict.touches)
        self.closet.styledict[str(self)] = self

    def __str__(self):
        return self.name

    def __eq__(self, other):
        """Check we're both Style instances and we have the same name"""
        return (
            isinstance(other, LiSEStyle) and
            str(self) == str(other))
