# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableWidgetMetaclass
from kivy.core.image import ImageData
from kivy.uix.image import Image
from kivy.properties import AliasProperty


class LiSEImage(Image):
    __metaclass__ = SaveableWidgetMetaclass
    tables = [
        ("img",
         {"name": "text not null",
          "path": "text not null",
          "rltile": "boolean not null DEFAULT 0"},
         ("name",),
         {},
         [])]
    rowdict = AliasProperty(
        lambda self: self.closet.skeleton["img"][unicode(self)],
        lambda self, v: None)
    path = AliasProperty(
        lambda self: self.rowdict["path"],
        lambda self, v: None,
        bind=('rowdict',))
    rltile = AliasProperty(
        lambda self: self.rowdict["rltile"],
        lambda self, v: None,
        bind=('rowdict',))    

    def __init__(self, closet, name, texture):
        self.closet = closet
        self.name = name
        Image.__init__(self, texture=texture)
        self.bind(rowdict=self.closet.skeleton[
            "img"][unicode(self)].touches)
        self.closet.imgdict[unicode(self)] = self

    def __str__(self):
        return self.name


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
