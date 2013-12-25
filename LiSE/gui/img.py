# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE.util import SaveableMetaclass


"""Saveable records about image files."""


class Img(object):
    """A class for keeping track of the various image files that go into
    the GUI.

    This class should never be instantiated, nor even subclassed. It
    exists to save and load records. If you want to display an :class:`Img`,
    use :class:`~kivy.uix.image.Image`.

    """
    __metaclass__ = SaveableMetaclass
    tables = [
        ("img",
         {"columns":
          {"name": "text not null",
           "path": "text not null",
           "off_x": "integer not null default 0",
           "off_y": "integer not null default 0",
           "stacking_height": "integer not null default 0"},
          "primary_key":
          ("name",),
          "checks":
          ["stacking_height >= 0.0"]}),
        ("img_tag",
         {"columns":
          {"img": "text not null",
           "tag": "text not null"},
          "primary_key":
          ("img", "tag"),
          "foreign_keys":
          {"img": ("img", "name")}})]
