# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from os import sep
from os.path import abspath

from LiSE import __path__
from LiSE.util import SaveableMetaclass


"""Saveable records about image files."""


whole_imgrows = [
    ('default_wallpaper', ['wallpape.jpg'], 0),
    ('default_spot', ['orb.png'], 0),
    ('default_pawn', ['rltiles', 'hominid', 'unseen.bmp'], 1)]


whole_img_val_fmt = (
    "('{0}', '" + abspath(__path__[-1]) + sep + "gui" + sep +
    "assets" + sep + "{1}', {2})")
"""A format string in which to insert values from `imgrows` to make
their relative paths absolute.

Actually this just prepends the path that the LiSE module is in to the
path given in the 1th format argument.

"""

pixel_city_imgrows = [
    ('sidewalk',     4,   5,   33, 21),
    ('crossroad',    45,  5,   34, 21),
    ('street-ne-sw', 4,   29,  34, 21),
    ('street-nw-se', 46,  30,  34, 21),
    ('block',        97,  175, 34, 29),
    ('spacer',       220, 306, 34, 21),
    ('lobby',        261, 307, 34, 23),
    ('brutalist',    302, 344, 34, 28),
    ('enterprise',   340, 344, 34, 26),
    ('brownstone',   266, 374, 34, 23),
    ('blind',        303, 374, 34, 23),
    ('soviet',       340, 374, 34, 23),
    ('monolith',     340, 400, 34, 23),
    ('olivine',      303, 400, 34, 23),
    ('orange',       265, 401, 34, 23)]


pixel_city_val_fmt = (
    "('{0}', '" + abspath(__path__[-1]) + sep + "gui" + sep +
    "assets" + sep + "pixel-city.png', {1}, {2}, {3}, {4})")


pixel_city_postlude = (
    "INSERT INTO img (name, path, cut_x, cut_y, cut_w, "
    "cut_h) VALUES " +
    ", ".join([pixel_city_val_fmt.format(name, x, 429-h-y, w, h)
               for (name, x, y, w, h) in pixel_city_imgrows]))


spot_imgs = (
    'default_spot', 'sidewalk', 'crossroad', 'street-ne-sw', 'street-nw-se',
    'block', 'spacer', 'lobby', 'brutalist', 'enterprise', 'brownstone',
    'blind', 'soviet', 'monolith', 'olivine', 'orange')


class Img(object):
    """A class for keeping track of the various image files that go into
    the GUI.

    This class should never be instantiated, nor even subclassed. It
    exists to save and load records. If you want to display an :class:`Img`,
    use :class:`~kivy.uix.image.Image`.

    """
    __metaclass__ = SaveableMetaclass
    postlude = [
        "INSERT INTO img (name, path, rltile) VALUES"
        + ", ".join(
            [whole_img_val_fmt.format(name, sep.join(path), rltile)
             for (name, path, rltile) in whole_imgrows]),
        pixel_city_postlude,
        "INSERT INTO img_tag (img, tag) VALUES "
        + ", ".join(["('{}', 'spot')".format(spot) for spot in spot_imgs])]
    tables = [
        ("img",
         {"columns":
          {"name": "text not null",
           "path": "text not null",
           "rltile": "boolean not null DEFAULT 0",
           "cut_x": "integer not null default 0",
           "cut_y": "integer not null default 0",
           "cut_w": "integer default null",
           "cut_h": "integer default null",
           "off_x": "integer not null default 0",
           "off_y": "integer not null default 0",
           "stacking_height": "integer not null default 0"},
          "primary_key":
          ("name",),
          "checks":
          ["stacking_height >= 0.0"]}),
        ("img_tag",
         {"columns":
          {"img": "text not null",
           "tag": "text not null"},
          "primary_key":
          ("img", "tag"),
          "foreign_keys":
          {"img": ("img", "name")}})]
