# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from os import sep
from os.path import abspath
from LiSE import __path__
from LiSE.util import SaveableMetaclass


imgrows = [
    ('default_wallpaper', ['wallpape.jpg'], 0),
    ('default_spot', ['orb.png'], 0),
    ('default_pawn', ['rltiles', 'hominid', 'unseen.bmp'], 1)]


valfmt = "('{0}', '" + abspath(__path__[-1]) + sep + "{1}', '{2}')"


class Img(object):
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
