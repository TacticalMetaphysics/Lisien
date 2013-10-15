# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass
from kivy.graphics import Color


"""Simple data structures to hold style information for text and
things that contain text."""


class Color(Color):
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
          'blue': 'float not null'},
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
        self._name = name
        rd = self.closet.skeleton["color"][str(self)]
        super(Color, self).__init__(rd["red"], rd["green"], rd["blue"])

    def __str__(self):
        return self._name


class Style(object):
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
    color_cols = ["textcolor", "bg_inactive", "bg_active",
                  "fg_inactive", "fg_active"]

    def __init__(self, closet, name):
        """Return a style by the given name, with the given face, font size,
spacing, and four colors: active and inactive variants for each of the
foreground and the background.

        """
        self.closet = closet
        self._name = name
        self.closet.styledict[str(self)] = self

    def __getattr__(self, attrn):
        if attrn == "_rowdict":
            return self.closet.skeleton["style"][str(self)]
        elif attrn in self.color_cols:
            return self.closet.get_color(self._rowdict[attrn])
        elif attrn in (
                "fontface",
                "fontsize",
                "spacing"):
            return self._rowdict[attrn]
        else:
            raise AttributeError(
                "Style instance has no such attribute: {0}".format(attrn))

    def __str__(self):
        return self._name

    def __eq__(self, other):
        """Check we're both Style instances and we have the same name"""
        return (
            isinstance(other, Style) and
            str(self) == str(other))
