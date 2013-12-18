from kivy.uix.relativelayout import RelativeLayout
from kivy.core.image import ImageData
from kivy.uix.image import Image
from kivy.uix.widget import (
    WidgetMetaclass)
from kivy.properties import (
    NumericProperty,
    ListProperty,
    ObjectProperty)

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
        bone = skel[u"img"][name]
        if bone.rltile:
            tex = load_rltile(skel[u"img"][name].path)
        else:
            tex = Image(
                source=skel[u"img"][name].path).texture
        w = bone.cut_w
        h = bone.cut_h
        if w is None:
            w = tex.width - bone.cut_x
        if h is None:
            h = tex.height - bone.cut_y
        r[name] = tex.get_region(
            bone.cut_x, bone.cut_y, w, h)
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


class TexPile(RelativeLayout):
    """Several images superimposed, and perhaps offset by differing amounts.

    Presents a list-like API. Append textures (not Images) to it,
    possibly specifying offsets on the x and y axes, and perhaps a
    stacking height, which will be added to the y offset of textures
    appended thereafter.

    """
    imgs = ListProperty([])
    stackhs = ListProperty([])

    def __getitem__(self, i):
        return self.imgs[i]

    def __setitem__(self, i, tex, xoff=0, yoff=0, stackh=0):
        self.imgs[i] = Image(
            texture=tex,
            pos=(xoff, yoff+sum(self.stackhs[:i])),
            size=tex.size)
        self.stackhs[i] = stackh

    def __delitem__(self, i):
        del self.imgs[i]
        del self.stackhs[i]

    def append(self, tex, xoff=0, yoff=0, stackh=0):
        pos = (xoff, yoff+sum(self.stackhs))
        size = tex.size
        self.imgs.append(Image(
            texture=tex,
            pos=pos,
            size=size))
        self.add_widget(self.imgs[-1])
        self.stackhs.append(stackh)

    def pop(self, i=-1):
        self.stackhs.pop(i)
        r = self.imgs.pop(i)
        self.remove_widget(r)
        return r


class LayerTexPile(TexPile):
    imagery = ObjectProperty()
    completedness = NumericProperty(0)
    closet = ObjectProperty()

    def collide_point(self, x, y):
        (x, y) = self.to_widget(x, y)
        for i in xrange(0, len(self.imagery)):
            img = self.imgs[i]
            if img.collide_point(x, y):
                return True
        return False

    def on_imagery(self, *args):
        self.completedness += 1

    def on_closet(self, *args):
        self.completedness += 1

    def on_completedness(self, i, v):
        if v == 2:
            self.upd_from_imagery()

    def upd_from_imagery(self, *args):
        branch = self.closet.branch
        tick = self.closet.tick
        self.clear_widgets()
        for layer in self.imagery:
            bone = self.imagery[layer][branch].value_during(tick)
            tex = self.closet.get_texture(bone.img)
            imgbone = self.closet.skeleton[u"img"][bone.img]
            self.append(tex, imgbone.off_x, imgbone.off_y,
                        imgbone.stacking_height)
