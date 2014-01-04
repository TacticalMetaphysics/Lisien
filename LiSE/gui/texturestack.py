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
    BooleanProperty,
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
    offxs = ListProperty([])
    """Integer offsets in the horizontal dimension. The texture with a
    given index will be pushed rightward by the number of pixels at
    the same index in offxs. Negative offsets will be adjusted to
    zero, but will still appear to push the texture to the left,
    because all the other textures will be moved to the right.

    """
    offys = ListProperty([])
    """Integer offsets in the vertical dimension. The texture with a given
    index will be pushed upward by the number of pixels at the same
    index in offys. Negative offsets will be adjusted to zero, but
    will still appear to push the texture downward, because all the
    other textures will be moved upward.

    """
    stackhs = ListProperty([])
    """Stacking heights. A texture with a positive stacking height will
    cause all textures on top of it to be offset in the vertical
    dimension by that amount. Stacking heights are cumulative.

    """
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
        if len(self.texs) == 0:
            self.size = [1, 1]
        else:
            self.recalc_size()
        self.bind(texs=self.upd_texs, offxs=self.upd_offxs,
                  offys=self.upd_offys, pos=self.upd_pos)
        if len(self.texs) > 0:
            self.upd_texs()
            self.upd_offxs()
            self.upd_offys()
            self.upd_pos()

    def upd_texs(self, *args):
        i = len(self.texs) - 1
        while i >= 0:
            tex = self.texs[i]
            if tex not in self.texture_rectangles:
                x = self.x + self.offxs[i]
                y = self.y + self.offys[i] + sum(self.stackhs[:i])
                self.canvas.insert(i, self.rectify(tex, x, y))
            i -= 1
        self.recalc_size()

    def upd_offxs(self, *args):
        enforce_positivity(self.offxs)

    def upd_offys(self, *args):
        enforce_positivity(self.offys)

    def upd_pos(self, *args):
        for i in xrange(0, len(self.texs)):
            tex = self.texs[i]
            offx = self.offxs[i]
            offy = self.offys[i]
            rect = self.texture_rectangles[tex]
            rect.pos = (self.x + offx, self.y + offy)

    def clear(self):
        self.canvas.clear()
        self.unbind(
            texs=self.upd_texs, offxs=self.upd_offxs, offys=self.upd_offys)
        self.rectangle_groups = {}
        self.texture_rectangles = {}
        self.texs = []
        self.offxs = []
        self.offys = []
        self.bind(
            texs=self.upd_offxs, offxs=self.upd_offxs, offys=self.upd_offys)
        self.stackhs = []
        self.size = [1, 1]

    def recalc_size(self):
        width = height = 1
        for i in xrange(0, len(self.texs)):
            width = max([self.texs[i].width + self.offxs[i], width])
            height = max([self.texs[i].height + self.offys[i], height])
        self.size = (width, height)

    def rectify(self, tex, x, y):
        rect = Rectangle(
            pos=(x, y),
            size=tex.size,
            texture=tex)
        self.texture_rectangles[tex] = rect
        group = InstructionGroup()
        group.add(rect)
        self.rectangle_groups[rect] = group
        return group

    def insert(self, i, tex, offx=0, offy=0, stackh=0):
        if not self.canvas:
            Clock.schedule_once(
                lambda dt: TextureStack.insert(
                    self, i, tex, offx, offy, stackh), 0)
            return
        self.unbind(texs=self.upd_texs)
        self.offxs.insert(i, offx)
        self.offys.insert(i, offy+sum(self.stackhs[:i]))
        self.stackhs.insert(i, stackh)
        self.texs.insert(i, tex)
        # the handlers for offxs and offys mean that they will not
        # necessarily remain at the value they were when we just
        # inserted them.
        group = self.rectify(tex, self.x+self.offxs[i], self.y+self.offys[i])
        self.canvas.insert(i, group)
        self.bind(texs=self.upd_texs)

    def append(self, tex, offx=0, offy=0, stackh=0):
        TextureStack.insert(self, len(self.texs), tex, offx, offy, stackh)

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
        offxs = self.offxs
        offys = self.offys
        stackhs = self.stackhs
        for i in xrange(0, len(self.paths)):
            image = Image.load(self.paths[i])
            if i < len(self.texs):
                del self[i]
            while i >= len(offxs):
                offxs.append(0)
            while i >= len(offys):
                offys.append(0)
            while i >= len(stackhs):
                stackhs.append(0)
            super(ImageStack, self).insert(
                i, image.texture, offxs=offxs[i],
                offy=offys[i], stackh=stackhs[i])
        self.offxs = offxs
        self.offys = offys
        self.stackhs = stackhs

    def insert(self, i, v, offx=0, offy=0, stackh=0):
        self.offxs.insert(i, offx)
        self.offys.insert(i, offy)
        self.stackhs.insert(i, stackh)
        self.paths.insert(i, v)

    def __delitem__(self, i):
        super(ImageStack, self).__delitem__(i)
        del self.paths[i]

    def pop(self, i=-1):
        self.paths.pop(i)
        return super(ImageStack, self).pop(i)
