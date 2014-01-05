"""Several textures superimposed on one another, and possibly offset
by some amount.

In 2D games where characters can wear different clothes or hold
different equipment, their graphics are often composed of several
graphics layered on one another. This widget simplifies the management
of such compositions.

"""

from kivy.uix.widget import Widget
from kivy.core.image import Image
from kivy.graphics import (
    Rectangle,
    InstructionGroup
)
from kivy.properties import (
    ListProperty,
    DictProperty)
from kivy.clock import Clock


def push_list(L, n):
    for i in xrange(0, len(L)):
        L[i] += n


def first_negative(L):
    for n in L:
        if n < 0:
            return n


def enforce_positivity(L):
    neg = first_negative(L)
    while neg is not None:
        push_list(L, neg)


class TextureStack(Widget):
    """Several textures superimposed on one another, and possibly offset
    by some amount.

    In 2D games where characters can wear different clothes or hold
    different equipment, their graphics are often composed of several
    graphics layered on one another. This widget simplifies the
    management of such compositions.

    """
    texs = ListProperty([])
    """Texture objects"""
    stackhs = ListProperty([])
    """Stacking heights. These are how high (in the y dimension) above the
    previous texture a texture at the same index in ``texs`` should be."""
    texture_rectangles = DictProperty({})
    """Private.

    Rectangle instructions for each of the textures, keyed by the
    texture.

    """
    rectangle_groups = DictProperty({})
    """Private.

    InstructionGroups for each Rectangle--including the BindTexture
    instruction that goes with it. Keyed by the Rectangle.

    """

    def __init__(self, **kwargs):
        super(TextureStack, self).__init__(**kwargs)
        self.bind(texs=self.upd_texs, pos=self.upd_pos)
        if len(self.texs) > 0:
            self.upd_texs()
            self.upd_pos()

    def upd_texs(self, *args):
        if not self.canvas:
            Clock.schedule_once(self.upd_texs, 0)
            return
        i = len(self.texs) - 1
        while i >= 0:
            tex = self.texs[i]
            if tex not in self.texture_rectangles:
                x = self.x
                y = self.y + sum(self.stackhs[:i])
                self.canvas.insert(i, self.rectify(tex, x, y))
            i -= 1

    def upd_pos(self, *args):
        for rect in self.texture_rectangles.itervalues():
            rect.pos = self.pos

    def clear(self):
        if self.canvas:
            self.canvas.clear()
        self.unbind(texs=self.upd_texs)
        self.rectangle_groups = {}
        self.texture_rectangles = {}
        self.texs = []
        self.offxs = []
        self.offys = []
        self.bind(texs=self.upd_texs)
        self.stackhs = []
        self.size = [1, 1]

    def rectify(self, tex, x, y):
        rect = Rectangle(
            pos=(x, y),
            size=tex.size,
            texture=tex)
        self.width = max([self.width, tex.width])
        self.height = max([self.height, tex.height])
        self.texture_rectangles[tex] = rect
        group = InstructionGroup()
        group.add(rect)
        self.rectangle_groups[rect] = group
        return group

    def insert(self, i, tex):
        if not self.canvas:
            Clock.schedule_once(
                lambda dt: TextureStack.insert(
                    self, i, tex), 0)
            return
        self.unbind(texs=self.upd_texs)
        self.texs.insert(i, tex)
        if len(self.stackhs) < len(self.texs):
            self.stackhs.extend([0] * (len(self.texs) - len(self.stackhs)))
        # the handlers for offxs and offys mean that they will not
        # necessarily remain at the value they were when we just
        # inserted them.
        group = self.rectify(tex, self.x, self.y + sum(self.stackhs[:i]))
        self.canvas.insert(i, group)
        self.bind(texs=self.upd_texs)

    def append(self, tex):
        TextureStack.insert(self, len(self.texs), tex)

    def __delitem__(self, i):
        tex = self.texs[i]
        try:
            rect = self.texture_rectangles[tex]
            group = self.rectangle_groups[rect]
            self.canvas.remove(group)
            del self.rectangle_groups[rect]
            del self.texture_rectangles[tex]
        except KeyError:
            pass
        self.unbind(
            offxs=self.upd_offxs,
            offys=self.upd_offys,
            texs=self.upd_texs)
        del self.offxs[i]
        del self.offys[i]
        del self.stackhs[i]
        del self.texs[i]
        self.bind(
            offxs=self.upd_offxs,
            offys=self.upd_offys,
            texs=self.upd_texs)

    def __setitem__(self, i, v):
        if len(self.texs) > 0:
            self.__delitem__(i)
        self.insert(i, v)

    def pop(self, i=-1):
        self.unbind(
            offxs=self.upd_offxs,
            offys=self.upd_offys)
        self.offxs.pop(i)
        self.offys.pop(i)
        self.bind(
            offxs=self.upd_offxs,
            offys=self.upd_offys)
        self.stackhs.pop(i)
        return self.texs.pop(i)


class ImageStack(TextureStack):
    """Instead of supplying textures themselves, supply paths to where the
    texture may be loaded from."""
    paths = ListProperty()

    def on_paths(self, *args):
        super(ImageStack, self).clear()
        for path in self.paths:
            self.append(Image.load(path))

    def clear(self):
        self.paths = []
        super(ImageStack, self).clear()

    def insert(self, i, v):
        self.paths.insert(i, v)

    def append(self, v):
        self.paths.append(v)

    def __delitem__(self, i):
        super(ImageStack, self).__delitem__(i)
        del self.paths[i]

    def pop(self, i=-1):
        self.paths.pop(i)
        return super(ImageStack, self).pop(i)
