# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from kivy.event import EventDispatcher
from kivy.properties import (
    ObjectProperty,
    StringProperty,
    AliasProperty,
    ReferenceListProperty,
)

from LiSE.util import SaveableMetaclass


"""Saveable records about image files."""


class Img(EventDispatcher):
    """A class for keeping track of the various image files that go into
    the GUI.

    """
    __metaclass__ = SaveableMetaclass
    tables = [
        ("img",
         {"columns":
          {"name": "text not null",
           "path": "text not null",
           "stacking_height": "integer not null default 0"},
          "primary_key":
          ("name",),
          "checks":
          ["stacking_height >= 0"]}),
        ("img_tag",
         {"columns":
          {"img": "text not null",
           "tag": "text not null"},
          "primary_key":
          ("img", "tag"),
          "foreign_keys":
          {"img": ("img", "name")}})]
    closet = ObjectProperty()
    name = StringProperty()
    texture = ObjectProperty()
    bone = AliasProperty(
        lambda self: self._get_bone(),
        lambda self, v: self._set_bone(v))
    offx = AliasProperty(
        lambda self: self.bone.offset_x,
        lambda self, v: None,
        bind=('bone',))
    offy = AliasProperty(
        lambda self: self.bone.offset_y,
        lambda self, v: None,
        bind=('bone',))
    offset = ReferenceListProperty(offx, offy)
    stackh = AliasProperty(
        lambda self: self.bone.stacking_height,
        lambda self, v: None,
        bind=('bone',))
    width = AliasProperty(
        lambda self: self.texture.width,
        lambda self, v: None,
        bind=('texture',))
    height = AliasProperty(
        lambda self: self.texture.width,
        lambda self, v: None,
        bind=('texture',))
    size = ReferenceListProperty(width, height)

    def _get_bone(self):
        return self.closet.skeleton[u"img"][self.name]

    def _set_bone(self, v):
        self.closet.skeleton[u"img"][self.name] = v

    def itertags(self):
        for (tag, imgs) in self.closet.img_tag_d.iteritems():
            if self.name in imgs:
                yield tag

    def read_pixel(self, x, y):
        data = self.texture.pixels
        (x, y) = (int(x), int(y))
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError("Position ({}, {}) is out of range.".format(
                x, y))
        index = y * self.width * 4 + x * 4
        raw = data[index:index+4]
        return [ord(c) / 255.0 for c in raw]
