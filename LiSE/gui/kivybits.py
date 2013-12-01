from kivy.core.image import ImageData
from kivy.uix.image import Image
from kivy.uix.widget import WidgetMetaclass

from LiSE.util import SaveableMetaclass
from img import Img


class SaveableWidgetMetaclass(WidgetMetaclass, SaveableMetaclass):
    """A combination of :class:`~kivy.uix.widget.WidgetMetaclass`
    and :class:`~LiSE.util.SaveableMetaclass`.

    There is no additional functionality beyond what those metaclasses do."""
    pass


def load_rltile(path):
    """Load one of the RLTiles, turn its chroma-key into an alpha
    channel, and return its texture."""
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


def load_textures(cursor, skel, texturedict, textagdict, names):
    """Load all the textures in ``names``. Put their :class:`Bone`s in
    ``skel``, and the textures themselves in ``texturedict``."""
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
    for (img, tag) in skel[u"img_tag"].iterbones():
        if tag not in textagdict:
            textagdict[tag] = set()
        textagdict[tag].add(img)
    return r


def load_textures_tagged(cursor, skel, texturedict, textagdict, tags):
    tagskel = Img._select_skeleton(
        cursor, {u"img_tag": [Img.bonetypes.img_tag(tag=t) for t in tags]})
    skel.update(tagskel)
    imgs = set([bone.img for bone in tagskel.iterbones()])
    return load_textures(cursor, skel, texturedict, textagdict, imgs)


def load_all_textures(cursor, skel, texturedict, textagdict):
    skel.update(Img._select_table_all(cursor, u"img_tag") +
                Img._select_table_all(cursor, u"img"))
    for bone in skel[u"img"].iterbones():
        if bone.rltile:
            texturedict[bone.name] = load_rltile(bone.path)
        else:
            texturedict[bone.name] = Image(
                source=bone.path).texture
    for (img, tag) in skel[u"img_tag"].iterbones():
        if img not in textagdict:
            textagdict[tag] = set()
        textagdict[tag].add(img)
