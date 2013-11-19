# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass


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
          "blue between 0.0 and 1.0",
          "alpha between 0.0 and 1.0"])]

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


postlude_template = (
    "create trigger fill_{0}_after_{2} after {2} on style "
    "when NEW.{0} is null begin "
    "update style set {0}=NEW.{1} where name=NEW.name; end")


class LiSEStyle(object):
    __metaclass__ = SaveableMetaclass
    """A collection of cogent information for rendering text and things
that contain text."""
    postlude = [
        postlude_template.format(active, inactive, event)
        for (active, inactive) in [
            ("text_active", "text_inactive"),
            ("bg_active", "bg_inactive"),
            ("fg_active", "fg_inactive")]
        for event in ["insert", "update"]]
    tables = [
        ("style",
         {"name": "text not null",
          "fontface": "text not null",
          "fontsize": "integer not null",
          "spacing": "integer default 6",
          "text_inactive": "text not null",
          "text_active": "text",
          "bg_inactive": "text not null",
          "bg_active": "text",
          "fg_inactive": "text not null",
          "fg_active": "text"},
         ("name",),
         {"text_inactive": ("color", "name"),
          "text_active": ("color", "name"),
          "bg_inactive": ("color", "name"),
          "bg_active": ("color", "name"),
          "fg_inactive": ("color", "name"),
          "fg_active": ("color", "name")},
         ["fontsize>0", "spacing>0"])]

    @property
    def rowdict(self):
        return self.closet.skeleton["style"][self.name]

    @property
    def bg_inactive(self):
        return self.closet.get_color(self.rowdict["bg_inactive"])

    @property
    def bg_active(self):
        return self.closet.get_color(self.rowdict["bg_active"])

    @property
    def fg_inactive(self):
        return self.closet.get_color(self.rowdict["fg_inactive"])

    @property
    def fg_active(self):
        return self.closet.get_color(self.rowdict["fg_active"])

    @property
    def text_inactive(self):
        return self.closet.get_color(self.rowdict["text_inactive"])

    @property
    def text_active(self):
        return self.closet.get_color(self.rowdict["text_active"])

    @property
    def fontface(self):
        return self.rowdict["fontface"]

    @property
    def fontsize(self):
        return self.rowdict["fontsize"]

    @property
    def spacing(self):
        return self.rowdict["spacing"]

    def __init__(self, closet, name):
        """Return a style by the given name, with the given face, font size,
spacing, and four colors: active and inactive variants for each of the
foreground and the background.

        """
        self.closet = closet
        self.name = name
        self.closet.styledict[str(self)] = self

    def __str__(self):
        return self.name

    def __eq__(self, other):
        """Check we're both Style instances and we have the same name"""
        return (
            isinstance(other, LiSEStyle) and
            str(self) == str(other))
