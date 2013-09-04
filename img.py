# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from pyglet.resource import image
from util import SaveableMetaclass


"""Container for images to be drawn, maybe."""


__metaclass__ = SaveableMetaclass


class Img:
    """A pretty thin wrapper around a Pyglet image.

Has savers and loaders that work with the LiSE database. The image
itself isn't saved, though. It's loaded, but saving an Img just means
saving the path.

    """
    tables = [
        ("img",
         {"name": "text not null",
          "path": "text not null",
          "rltile": "boolean not null DEFAULT 0"},
         ("name",),
         {},
         [])]

    atrdic = {
        "path": lambda self: self._rowdict["path"],
        "rltile": lambda self: self._rowdict["rltile"],
        "center": lambda self: (self.tex.width / 2, self.tex.height / 2),
        "width": lambda self: self.tex.width,
        "height": lambda self: self.tex.height}

    def __init__(self, rumor, name):
        """Return an Img, and register it with the imgdict of the database
provided."""
        self.rumor = rumor
        self._name = name
        self.rumor.imgdict[str(self)] = self
        self._rowdict = self.rumor.tabdict["img"][str(self)]
        if self.rltile:
            self.tex = load_rltile(self.path)
        else:
            self.tex = load_regular_img(self.path)
        self.texture = self.tex

    def __str__(self):
        return self._name

    def __getattr__(self, attrn):
        try:
            return self.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                "Img instance has no attribute {0}.".format(attrn))


def load_rltile(path):
    """Load a Windows bitmap, and replace ffGll -> 00Gll and ff. -> 00."""
    badimg = image(path)
    badimgd = badimg.get_image_data()
    bad_rgba = badimgd.get_data('RGBA', badimgd.pitch)
    good_data = bad_rgba.replace('\xffGll', '\x00Gll')
    good_data = good_data.replace('\xff.', '\x00.')
    badimgd.set_data('RGBA', badimgd.pitch, good_data)
    rtex = badimgd.get_texture()
    return rtex


def load_regular_img(path):
    """Load an ordinary PNG image."""
    tex = image(path).get_image_data().get_texture()
    return tex
