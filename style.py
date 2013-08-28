# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass, dictify_row
import pyglet


"""Simple data structures to hold style information for text and
things that contain text."""


__metaclass__ = SaveableMetaclass


class Color:
    """Red, green, blue, and alpha values.

    This is just a container class for the (red, green, blue, alpha)
tuples that Pyglet uses to identify colors. The tup attribute will get
you that.

    """
    tables = [
        ("color",
         {'name': 'text not null',
          'red': 'integer not null ',
          'green': 'integer not null ',
          'blue': 'integer not null ',
          'alpha': 'integer not null default 255 '},
         ("name",),
         {},
         ["red between 0 and 255",
          "green between 0 and 255",
          "blue between 0 and 255",
          "alpha between 0 and 255"])]

    def __init__(self, rumor, name, td):
        """Return a color with the given name, and the given values for red,
green, blue, and alpha. Register in db.colordict.

        """
        self.rumor = rumor
        self._name = name
        self._tabdict = td
        self._rowdict = td["color"][name]

    def __getattr__(self, attrn):
        if attrn in ("r", "red"):
            return self._rowdict["red"]
        elif attrn in ("green", "g"):
            return self._rowdict["green"]
        elif attrn in ("blue", "b"):
            return self._rowdict["blue"]
        elif attrn in ("alpha", "a"):
            return self._rowdict["alpha"]
        elif attrn in ("tup", "tuple"):
            return (self.red, self.green, self.blue, self.alpha)
        elif attrn in ("pat", "pattern"):
            return pyglet.image.SolidColorImagePattern(self.tup)
        else:
            raise AttributeError(
                "Color instance has no such attribute: {0}".format(attrn))

    def __str__(self):
        return self._name

    def __repr__(self):
        """Looks just like the tuple."""
        return "(" + ", ".join(self.tup) + ")"


class Style:
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

    def __init__(self, rumor, name, td):
        """Return a style by the given name, with the given face, font size,
spacing, and four colors: active and inactive variants for each of the
foreground and the background.

With db, register in its styledict.

        """
        self.rumor = rumor
        self._name = name
        self._tabdict = td
        self._rowdict = td["style"][name]

    def __getattr__(self, attrn):
        if attrn in self.color_cols:
            return self.rumor.get_color(self._rowdict[attrn])
        elif attrn in self._rowdict:
            return self._rowdict[attrn]
        else:
            raise AttributeError(
                "Style instance has no such attribute: {0}".format(attrn))

    def __str__(self):
        return self.name

    def __eq__(self, other):
        """Check we're both Style instances and we have the same name"""
        return (
            isinstance(other, Style) and
            self.name == other.name)
