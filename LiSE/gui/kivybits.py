from kivy.core.image import ImageData
from kivy.uix.image import Image
from kivy.uix.widget import (
    Widget,
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
        print tag
        if img not in textagdict:
            textagdict[tag] = set()
        textagdict[tag].add(img)


class ImgPileImg(Image):
    pile = ObjectProperty()
    layer = NumericProperty()


class ImgPile(Widget):
    """Several images superimposed, and perhaps offset by differing amounts."""
    textures = ListProperty([])
    imgs = ListProperty([])
    xoffs = ListProperty([])
    yoffs = ListProperty([])
    stackhs = ListProperty([])

    def append(self, tex, xoff=0, yoff=0, stackh=0):
        i = len(self.textures)
        self.textures.append(tex)
        self.xoffs.append(xoff)
        self.yoffs.append(yoff)
        self.stackhs.append(stackh)
        self.imgs.append(ImgPileImg(pile=self, layer=i))
        self.add_widget(self.imgs[i])

    def pop(self, i=-1):
        r = (self.textures.pop(i),
             self.imgs.pop(i),
             self.xoffs.pop(i),
             self.yoffs.pop(i),
             self.stackhs.pop(i))
        self.remove_widget(r[1])
        return r


class LayerImgPile(ImgPile):
    imagery = ObjectProperty()
    completedness = NumericProperty(0)
    closet = ObjectProperty()

    def collide_point(self, x, y):
        (x, y) = self.to_widget(x, y)
        for i in xrange(0, len(self.imagery)):
            xoff = self.xoffs[i]
            yoff = self.yoffs[i]
            tex = self.textures[i]
            (w, h) = tex.size
            if (
                    x > xoff and
                    y > yoff and
                    x < xoff + w and
                    y < yoff + h):
                return True
        return False

    def on_imagery(self, *args):
        self.completedness += 1
        print("imagery {}".format(self.completedness))

    def on_closet(self, *args):
        self.completedness += 1
        print("closet {}".format(self.completedness))

    def on_completedness(self, i, v):
        if v == 2:
            self.upd_from_imagery()

    def upd_from_imagery(self, *args):
        branch = self.closet.branch
        tick = self.closet.tick
        self.clear_widgets()
        for layer in self.imagery:
            bone = self.imagery[layer][branch][tick]
            imgbone = self.closet.skeleton[u"img"][bone.img]
            tex = self.closet.get_texture(bone.img)
            self.append(tex, bone.off_x, bone.off_y, imgbone.stacking_height)
