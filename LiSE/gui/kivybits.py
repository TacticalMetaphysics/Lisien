from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.image import Image
from kivy.uix.widget import (
    Widget,
    WidgetMetaclass)
from kivy.properties import (
    NumericProperty,
    ListProperty,
    ObjectProperty,
    StringProperty,
    BooleanProperty)

from LiSE.util import SaveableMetaclass
from img import Img


class SaveableWidgetMetaclass(WidgetMetaclass, SaveableMetaclass):
    """A combination of :class:`~kivy.uix.widget.WidgetMetaclass`
    and :class:`~LiSE.util.SaveableMetaclass`.

    There is no additional functionality beyond what those metaclasses do."""
    pass


class TouchlessWidget(Widget):
    def on_touch_down(self, touch):
        return

    def on_touch_move(self, touch):
        return

    def on_touch_up(self, touch):
        return

    def collide_point(self, x, y):
        return

    def collide_widget(self, w):
        return


class CueCard(TouchlessWidget):
    """Widget that looks like TextInput but doesn't take input and can't be
clicked.

This is used to display feedback to the user when it's not serious
enough to get a popup of its own.

    """
    closet = ObjectProperty()
    stringname = StringProperty()
    symbolic = BooleanProperty(False)
    completion = NumericProperty(0)

    def on_closet(self, *args):
        self.completion += 1

    def on_stringname(self, *args):
        self.completion += 1

    def on_completion(self, i, v):
        if v == 2:
            self.complete()

    def complete(self):
        self.closet.register_text_listener(self.stringname, self.retext)
        self.revert_text()

    def revert_text(self):
        self.ids.l.text = self.closet.get_text(self.stringname)

    def retext(self, skel, k, v):
        if k == self.closet.language:
            self.text = v


def load_textures(cursor, skel, texturedict, textagdict, names):
    """Load all the textures in ``names``. Put their :class:`Bone`s in
    ``skel``, and the textures themselves in ``texturedict``."""
    nulimg = Img.bonetypes["img"](
        *[None for field in Img.bonetypes["img"]._fields])
    qd = {
        u"img": [nulimg._replace(name=n) for n in names],
        u"img_tag": [Img.bonetypes["img_tag"](img=n) for n in names]}
    res = Img._select_skeleton(cursor, qd)
    skel.update(res)
    r = {}
    for name in names:
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
        cursor, {u"img_tag": [Img.bonetypes["img_tag"](tag=t) for t in tags]})
    skel.update(tagskel)
    imgs = set([bone.img for bone in tagskel.iterbones()])
    return load_textures(cursor, skel, texturedict, textagdict, imgs)


def load_all_textures(cursor, skel, texturedict, textagdict):
    skel.update(Img._select_table_all(cursor, u"img_tag") +
                Img._select_table_all(cursor, u"img"))
    for bone in skel[u"img"].iterbones():
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
