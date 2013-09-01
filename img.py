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

    def __init__(self, rumor, name):
        """Return an Img, and register it with the imgdict of the database
provided."""
        self.rumor = rumor
        self._name = name
        if self.rltile:
            self.tex = load_rltile(self.path)
        else:
            self.tex = load_regular_img(self.path)
        self.rumor.imgdict[str(self)] = self

    def __str__(self):
        return self._name

    def __getattr__(self, attrn):
        if attrn == "_rowdict":
            return self.rumor.tabdict["img"][str(self)]
        elif attrn == "path":
            return self._rowdict["path"]
        elif attrn == "rltile":
            return self._rowdict["rltile"]
        elif attrn == 'center':
            return (self.tex.width / 2, self.tex.height / 2)
        elif attrn == 'texture':
            return self.tex
        elif attrn == 'width':
            return self.tex.width
        elif attrn == 'height':
            return self.tex.height
        else:
            try:
                return getattr(self.tex, attrn)
            except AttributeError:
                raise AttributeError(
                    "Img instance has no attribute {0}.".format(attrn))

    def get_tabdict(self):
        return {
            "img": {
                "name": self.name,
                "path": self.path,
                "rltile": self.rltile}}


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
