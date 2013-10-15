# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass
from kivy.core.image import ImageData
from kivy.uix.image import Image


class Tex(object):
    __metaclass__ = SaveableMetaclass
    tables = [
        ("img",
         {"name": "text not null",
          "path": "text not null",
          "rltile": "boolean not null DEFAULT 0"},
         ("name",),
         {},
         [])]

    def __init__(self, closet, name):
        self.closet = closet
        self._name = name
        self.closet.imgdict[str(self)] = self
        self._rowdict = self.closet.skeleton["img"][str(self)]
        if self.rltile:
            self.texture = load_rltile(self.path)
        else:
            self.texture = Image(self.path, __no_builder=True).texture

    def __str__(self):
        return self._name

    @property
    def path(self):
        return self._rowdict["path"]

    @property
    def rltile(self):
        return self._rowdict["path"]


def load_rltile(path):
    """Load a Windows bitmap, and replace ffGll -> 00Gll and ff. -> 00."""
    badtex = Image(source=path, __no_builder=True).texture
    imgd = ImageData(
        badtex.width, badtex.height,
        badtex.colorfmt, badtex.pixels, source=path)
    dat = imgd.data
    dat.replace(
        b'\xffGll', b'\x00Gll').replace(
        b'\xff.', b'\x00.')
    badtex.blit_buffer(dat)
    return badtex
