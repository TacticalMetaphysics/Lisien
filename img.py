# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass
from kivy.core.image import ImageData
from kivy.uix import Image
from kivy.graphics.texture import Texture


u"""Container for images to be drawn, maybe."""


__metaclass__ = SaveableMetaclass


textures = {}


class Img(Image):
    """A pretty thin wrapper around a Pyglet image.

Has savers and loaders that work with the LiSE database. The image
itself isn't saved, though. It's loaded, but saving an Img just means
saving the path.

    """
    tables = [
        (u"img",
         {u"name": u"text not null",
          u"path": u"text not null",
          u"rltile": u"boolean not null DEFAULT 0"},
         (u"name",),
         {},
         [])]

    def __init__(self, closet, name):
        u"""Return an Img, and register it with the imgdict of the database
provided."""
        self.closet = closet
        self._name = name
        self.closet.imgdict[str(self)] = self
        self._rowdict = self.closet.skeleton[u"img"][str(self)]
        if self.rltile:
            super(Img, self).__init__(load_rltile(self.path))
        else:
            super(Img, self).__init__(self.path)

    def __str__(self):
        return self._name

    @property
    def path(self):
        return self._rowdict[u"path"]

    @property
    def rltile(self):
        return self._rowdict[u"path"]


def load_rltile(path):
    u"""Load a Windows bitmap, and replace ffGll -> 00Gll and ff. -> 00."""
    badtex = Image(source=path).texture
    imgd = ImageData(
        badtex.width, badtex.height,
        badtex.colorfmt, badtex.pixels, source=path)
    imgd.data = imgd.data.replace(
        '\xffGll', '\x00Gll').replace('\xff.', '\x00.')
    return imgd
