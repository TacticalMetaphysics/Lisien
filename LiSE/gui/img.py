# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from os import sep
from os.path import abspath

from LiSE import __path__
from LiSE.util import SaveableMetaclass


"""Saveable records about image files."""


imgrows = [
    ('default_wallpaper', ['wallpape.jpg'], 0),
    ('default_spot', ['orb.png'], 0),
    ('default_pawn', ['rltiles', 'hominid', 'unseen.bmp'], 1)]
"""Some special names for the default values.

These need to be inserted in Python, rather than SQL, because the
database will use absolute paths for all the image files, and SQLite
is not clever enough to translate relative paths to absolute.

"""


valfmt = "('{0}', '" + abspath(__path__[-1]) + sep + "{1}', '{2}')"
"""A format string in which to insert values from `imgrows` to make
their relative paths absolute.

Actually this just prepends the path that the LiSE module is in to the
path given in the 1th format argument.

"""


class Img(object):
    """A class for keeping track of the various image files that go into
    the GUI.

    This class should never be instantiated, nor even subclassed. It
    exists to save and load records. If you want to display an :class:`Img`,
    use :class:`~kivy.uix.image.Image`.

    """
    __metaclass__ = SaveableMetaclass
    postlude = [
        "INSERT INTO img (name, path, rltile) VALUES "
        + ", ".join([valfmt.format(
            name,
            sep.join(["gui", "assets"] + path), rltile)
            for (name, path, rltile) in imgrows])]
    tables = [
        ("img",
         {"name": "text not null",
          "path": "text not null",
          "rltile": "boolean not null DEFAULT 0"},
         ("name",),
         {},
         []),
        ("img_tag",
         {"img": "text not null",
          "tag": "text not null"},
         ("img", "tag"),
         {"img": ("img", "name")},
         [])]
