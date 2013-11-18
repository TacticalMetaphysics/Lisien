from kivy.uix.image import Image
from kivy.core.image import ImageData
from kivy.properties import (
    NumericProperty,
    StringProperty)
from kivy.event import EventDispatcher
from img import Tex
from util import SaveableMetaclass
from kivy.uix.widget import WidgetMetaclass


class KivyConnector(EventDispatcher):
    language = StringProperty()
    dimension = StringProperty()
    branch = NumericProperty()
    tick = NumericProperty()
    hi_branch = NumericProperty()
    hi_tick = NumericProperty()


class SaveableWidgetMetaclass(WidgetMetaclass, SaveableMetaclass):
    pass


def load_textures(cursor, skel, texturedict, names):
    kd = {"img": {}}
    for name in names:
        kd["img"][name] = {"name": name}
    skel.update(
        Tex._select_skeleton(
            cursor, kd))
    r = {}
    for name in names:
        if skel["img"][name]["rltile"] != 0:
            rltex = Image(
                source=skel["img"][name]["path"]).texture
            imgd = ImageData(rltex.width, rltex.height,
                             rltex.colorfmt, rltex.pixels,
                             source=skel["img"][name]["path"])
            fixed = ImageData(
                rltex.width, rltex.height,
                rltex.colorfmt, imgd.data.replace(
                    '\xffGll', '\x00Gll').replace(
                    '\xff.', '\x00.'),
                source=skel["img"][name]["path"])
            rltex.blit_data(fixed)
            r[name] = rltex
        else:
            r[name] = Image(
                source=skel["img"][name]["path"]).texture
    texturedict.update(r)
    return r
