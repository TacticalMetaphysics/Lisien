# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from pyglet.resource import image
from util import SaveableMetaclass


u"""Container for images to be drawn, maybe."""


__metaclass__ = SaveableMetaclass


textures = {}


class Img:
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

    atrdic = {
        u"path": lambda self: self._rowdict[u"path"],
        u"rltile": lambda self: self._rowdict[u"rltile"],
        u"center": lambda self: (self.tex.width / 2, self.tex.height / 2),
        u"width": lambda self: self.tex.width,
        u"height": lambda self: self.tex.height,
        u"tex": lambda self: textures[str(self)]}

    def __init__(self, closet, name):
        u"""Return an Img, and register it with the imgdict of the database
provided."""
        global first_img_loaded
        self.closet = closet
        self._name = name
        self.closet.imgdict[str(self)] = self
        self._rowdict = self.closet.skeleton[u"img"][str(self)]
        if self.rltile:
            textures[str(self)] = load_rltile(self.path)
        else:
            textures[str(self)] = load_regular_img(self.path)

    def __str__(self):
        return self._name

    def __getattr__(self, attrn):
        try:
            return self.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                u"Img instance has no attribute {0}.".format(attrn))


def load_rltile(path):
    u"""Load a Windows bitmap, and replace ffGll -> 00Gll and ff. -> 00."""
    badimg = image(path)
    badimgd = badimg.get_image_data()
    bad_rgba = badimgd.get_data('RGBA', badimgd.pitch)
    good_data = bad_rgba.replace('\xffGll', '\x00Gll')
    good_data = good_data.replace('\xff.', '\x00.')
    badimgd.set_data('RGBA', badimgd.pitch, good_data)
    rtex = badimgd.get_texture()
    return rtex


def load_regular_img(path):
    u"""Load an ordinary PNG image."""
    tex = image(path).get_image_data().get_texture()
    return tex
