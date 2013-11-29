from kivy.uix.image import Image
from kivy.core.image import ImageData
from img import Img
from util import SaveableMetaclass
from kivy.uix.widget import WidgetMetaclass


class SaveableWidgetMetaclass(WidgetMetaclass, SaveableMetaclass):
    pass


def load_rltile(path):
    rltex = Image(
        source=path).texture
    imgd = ImageData(rltex.width, rltex.height,
                     rltex.colorfmt, rltex.pixels,
                     source=path)
    fixed = ImageData(
        rltex.width, rltex.height,
        rltex.colorfmt, imgd.data.replace(
            '\xffGll', '\x00Gll').replace(
            '\xff.', '\x00.'),
        source=path)
    rltex.blit_data(fixed)
    return rltex


def ins_texture(skel, texturedict, path, texn=None, rltile=False):
    if texn is None:
        texn = path
    if rltile:
        tex = load_rltile(path)
    else:
        tex = Image(source=path).texture
    texturedict[texn] = tex
    bone = {
        u"name": texn,
        u"path": path,
        u"rltile": rltile}
    skel[u"img"][texn] = bone
    return (tex, bone)


def load_textures(cursor, skel, texturedict, names):
    skel.update(
        Img._select_skeleton(
            cursor, {
                u"img": [Img.bonetypes.img(name=n) for n in names],
                u"img_tag": [Img.bonetypes.img_tag(img=n) for n in names]}))
    r = {}
    for name in names:
        if skel[u"img"][name].rltile:
            rltex = load_rltile(skel[u"img"][name].path)
            r[name] = rltex
        else:
            r[name] = Image(
                source=skel[u"img"][name].path).texture
    texturedict.update(r)
    return r
