from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButtonBehavior
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.widget import (
    Widget,
    WidgetMetaclass)
from kivy.properties import (
    AliasProperty,
    DictProperty,
    NumericProperty,
    ListProperty,
    ObjectProperty,
    StringProperty,
    BooleanProperty)
from kivy.graphics import (
    Rectangle,
    InstructionGroup
)
from kivy.clock import Clock

from LiSE import __path__
from LiSE.util import SaveableMetaclass
from texturestack import TextureStack
from img import Img


class ClosetWidget(Widget):
    """Mix-in class for various text-having widget classes, to make their
    text match some named string from the closet."""
    stringname = StringProperty()
    closet = ObjectProperty()
    symbolic = BooleanProperty(False)


class ClosetLabel(Label, ClosetWidget):
    pass


class ClosetButton(Button, ClosetWidget):
    fun = ObjectProperty(None)
    arg = ObjectProperty(None)

    def on_release(self, *args):
        if self.fun is None:
            return
        if self.arg is None:
            self.fun()
        else:
            self.fun(self.arg)


class ClosetToggleButton(ClosetButton, ToggleButtonBehavior):
    pass


class ClosetHintTextInput(TextInput, ClosetWidget):
    failure_string = StringProperty()
    """String to use when the input failed to validate"""
    failure_color = ListProperty([1, 0, 0, 1])
    """Color to turn the input field when it fails"""
    failure_color_timeout = NumericProperty(0.5)
    """Time after which to turn the color back"""
    failure_string_timeout = NumericProperty(3)
    """Time after which to turn the hint_text back"""
    validator = ObjectProperty()
    """Boolean function for whether the input is acceptable"""

    def validate(self):
        """If my text is valid, return True. Otherwise, communicate invalidity
        to the user, and return False.

        I'll communicate invalidity by blinking some other color,
        blanking out my text, and displaying an alternative hint_text
        for a little while.

        """
        if self.validator(self.text):
            return True
        else:
            self.text = ''
            oldcolor = self.color
            self.color = self.failure_color
            self.hint_text = self.closet.get_text(self.failure_string)

            def unfail_color():
                self.color = oldcolor
            Clock.schedule_once(unfail_color, self.failure_color_timeout)

            def unfail_text():
                self.hint_text = self.closet.get_text(self.stringname)
            Clock.schedule_once(unfail_text, self.failure_string_timeout)
            return False


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


class OffsetTextureStack(TextureStack):
    offxs = ListProperty([])
    offys = ListProperty([])

    def on_width(self, *args):
        pass

    def clear(self):
        super(OffsetTextureStack, self).clear()
        self.offxs = []
        self.offys = []

    def insert(self, i, tex, offx=0, offy=0):
        self.suppressor = True
        if not self.canvas:
            Clock.schedule_once(
                lambda dt: self.insert(i, tex, offx, offy), 0)
            return
        self.texs.insert(i, tex)
        self.offxs.insert(i, offx)
        self.offys.insert(i, offy)
        group = self.rectify(tex, offx, offy)
        self.canvas.insert(i, group)
        self.width = max([self.width, tex.width + max([offx, 0])])
        self.height = max([self.height, tex.height + max([offy, 0])])
        self.suppressor = False

    def append(self, tex, offx=0, offy=0):
        self.insert(len(self.texs), tex, offx, offy)

    def __setitem__(self, i, v, offx=0, offy=0):
        self.__delitem__(i)
        self.insert(i, v, offx, offy)

    def __delitem__(self, i):
        super(OffsetTextureStack, self).__delitem__(i)
        del self.offxs[i]
        del self.offys[i]

    def pop(self, i=-1):
        tex = super(OffsetTextureStack, self).pop(i)
        self.offxs.pop(i)
        self.offys.pop(i)
        return tex

    def rectify(self, tex, offx=0, offy=0):
        if offx < 0:
            self.offxs = map(lambda x: x-offx, self.offxs)
            offx = 0
        if offy < 0:
            self.offys = map(lambda y: y-offy, self.offys)
            offy = 0
        rect = Rectangle(
            x=self.x+offx,
            y=self.y+offy,
            pos=self.pos,
            size=tex.size,
            texture=tex)
        self.texture_rectangles[tex] = rect
        group = InstructionGroup()
        group.add(rect)
        self.rectangle_groups[rect] = group
        return group

    def recalc_size(self):
        width = height = 1
        for i in xrange(0, len(self.texs)):
            tex = self.texs[i]
            offx = self.offxs[i]
            offy = self.offys[i]
            assert(offx >= 0 and offy >= 0)
            w = tex.width + offx
            h = tex.height + offy
            width = max([width, w])
            height = max([height, h])
        self.size = (width, height)

    def on_pos(self, *args):
        for i in xrange(0, len(self.texs)):
            tex = self.texs[i]
            offx = self.offxs[i]
            offy = self.offys[i]
            rect = self.texture_rectangles[tex]
            rect.pos = (self.x + offx, self.y + offy)


class ClosetTextureStack(OffsetTextureStack):
    closet = ObjectProperty()
    bones = ListProperty([])
    bone_textures = DictProperty({})

    def on_bones(self, *args):
        if not self.closet:
            Clock.schedule_once(self.on_names, 0)
        i = 0
        for bone in self.bones:
            if len(self.texs) == i:
                tex = self.closet.get_texture(bone.name)
                self.append(tex, offx=bone.off_x, offy=bone.off_y)
                self.name_textures[bone] = tex
            elif bone in self.bone_textures:
                continue
            else:
                tex = self.closet.get_texture(bone.name)
                self.offxs[i] = bone.off_x
                self.offys[i] = bone.off_y
                self[i] = self.bone_textures[bone] = tex
            i += 1

    def update_texture_named(self, name):
        i = self.names.index(name)
        bone = self.closet.skeleton[u"img"][name]
        tex = self.closet.get_texture(name)
        self.offxs[i] = bone.off_x
        self.offys[i] = bone.off_y
        self[i] = tex


class LayerTextureStack(ClosetTextureStack):
    imagery = ObjectProperty()

    def on_imagery(self, *args):
        if not (self.imagery and self.closet):
            Clock.schedule_once(self.on_imagery, 0)
            return
        branch = self.closet.branch
        tick = self.closet.tick
        self.clear()
        for layer in self.imagery:
            imgn = self.imagery[layer][branch].value_during(tick).img
            bone = self.closet.skeleton[u"img"][imgn]
            self.bones.append(bone)
