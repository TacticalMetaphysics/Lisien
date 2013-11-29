# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass


class Img(object):
    __metaclass__ = SaveableMetaclass
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
