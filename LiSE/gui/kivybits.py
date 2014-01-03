from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButtonBehavior
from kivy.uix.textinput import TextInput
from kivy.core.image import Image as KImage
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


class Image(KImage):
    """Just an Image that stores some LiSE-specific metadata."""
    offx = NumericProperty(0)
    offy = NumericProperty(0)
    stackh = NumericProperty(0)
    tags = ListProperty([])

    @staticmethod
    def load(self, filename):
        img_d = KImage.load(filename, keep_data=True).__dict__
        return Image(**img_d)

    @staticmethod
    def from_bone(bone):
        return Image(KImage.load(bone.path, keep_data=True).texture,
                     offx=bone.off_x, offy=bone.off_y,
                     stackh=bone.stacking_height)


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


def load_images(cursor, setter, names):
    """Load all the textures in ``names``. Put their :class:`Bone`s in
    ``skel``, and the textures themselves in ``texturedict``."""
    r = {}
    for bone in Img._select_skeleton(cursor, {u"img": [
            Img.bonetype._null()._replace(name=n) for n in names]
    }).iterbones():
        setter(bone)
        r[bone.name] = Image.from_bone(bone)
    for bone in Img._select_skeleton(cursor, {u"img_tag": [
            Img.bonetypes["img_tag"](img=n) for n in names]
    }).iterbones():
        setter(bone)
        image = r[bone.img]
        if bone.tag not in image.tags:
            image.tags.append(bone.tag)
    return r


def load_images_tagged(cursor, setter, tags):
    tagskel = Img._select_skeleton(
        cursor, {u"img_tag": [Img.bonetypes["img_tag"](tag=t) for t in tags]})
    imgs = set([bone.img for bone in tagskel.iterbones()])
    return load_images(cursor, setter, imgs)


def load_all_images(cursor, setter):
    imagedict = {}
    for bone in Img._select_table_all(cursor, u"img").iterbones():
        setter(bone)
    for bone in Img._select_table_all(cursor, u"img_tag").iterbones():
        setter(bone)
        (img, tag) = bone
        image = imagedict[img]
        if tag not in image.tags:
            image.tags.append(tag)
    return imagedict


class ClosetTextureStack(TextureStack):
    closet = ObjectProperty()
    bones = ListProperty([])
    bone_images = DictProperty({})

    def on_bones(self, *args):
        if not self.closet:
            Clock.schedule_once(self.on_names, 0)
        i = 0
        for bone in self.bones:
            if len(self.texs) == i:
                image = self.closet.get_image(bone.name)
                self.append(image.texture,
                            offx=bone.off_x,
                            offy=bone.off_y)
                self.bone_images[bone] = image
            elif bone in self.bone_images:
                continue
            else:
                image = self.closet.get_image(bone.name)
                self.offxs[i] = bone.off_x
                self.offys[i] = bone.off_y
                self[i] = self.bone_images[bone] = image
            i += 1


class ImageryStack(ClosetTextureStack):
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
