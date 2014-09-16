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
    DictProperty
)
from kivy.clock import Clock
from kivy.resources import resource_find


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
        """Make triggers, bind, and if I have something to show, show it."""
        kwargs['size_hint'] = (None, None)
        super().__init__(**kwargs)
        self._trigger_upd_texs = Clock.create_trigger(
            self._upd_texs, timeout=-1
        )
        self._trigger_upd_pos = Clock.create_trigger(
            self._upd_pos, timeout=-1
        )
        self.bind(pos=self._trigger_upd_pos)

    def on_texs(self, *args):
        if len(self.texs) > 0:
            self._trigger_upd_texs()
            self._trigger_upd_pos()

    def _upd_texs(self, *args):
        if not self.canvas:
            Clock.schedule_once(self.upd_texs, 0)
            return
        self._clear_rects()
        w = h = 0
        for tex in self.texs:
            self.canvas.add(self._rectify(tex, self.x, self.y))
            if tex.width > w:
                w = tex.width
            if tex.height > h:
                h = tex.height
        self.size = (w, h)

    def _upd_pos(self, *args):
        for rect in self.texture_rectangles.values():
            rect.pos = self.pos

    def _clear_rects(self):
        for group in self.rectangle_groups.values():
            self.canvas.remove(group)
        self.rectangle_groups = {}
        self.texture_rectangles = {}

    def clear(self):
        self._clear_rects()
        self.texs = []
        self.stackhs = []
        self.size = [1, 1]

    def _rectify(self, tex, x, y):
        rect = Rectangle(
            pos=(x, y),
            size=tex.size,
            texture=tex
        )
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
        if len(self.stackhs) < len(self.texs):
            self.stackhs.extend([0] * (len(self.texs) - len(self.stackhs)))
        self.texs.insert(i, tex)

    def append(self, tex):
        self.insert(len(self.texs), tex)

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
        del self.stackhs[i]
        del self.texs[i]

    def __setitem__(self, i, v):
        if len(self.texs) > 0:
            self.unbind(texs=self._upd_texs)
            self.__delitem__(i)
            self.bind(texs=self._upd_texs)
        self.insert(i, v)

    def pop(self, i=-1):
        self.stackhs.pop(i)
        return self.texs.pop(i)


class ImageStack(TextureStack):
    """Instead of supplying textures themselves, supply paths to where the
    texture may be loaded from."""
    paths = ListProperty()

    def on_paths(self, *args):
        super().clear()
        for path in self.paths:
            super().append(Image.load(resource_find(path)).texture)

    def clear(self):
        self.paths = []
        super().clear()

    def insert(self, i, v):
        if isinstance(v, str):
            self.paths.insert(i, v)
        else:
            super().insert(i, v)

    def append(self, v):
        if isinstance(v, str):
            self.paths.append(v)
        else:
            super().append(v)

    def __delitem__(self, i):
        super().__delitem__(i)
        del self.paths[i]

    def pop(self, i=-1):
        self.paths.pop(i)
        return super().pop(i)
