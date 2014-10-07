# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
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
    DictProperty,
    BooleanProperty,
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
    """Stacking heights. All textures following one with a nonzero
    stacking height are moved up by that number of pixels (cumulative).

    """
    offxs = ListProperty([])
    """x-offsets. The texture at the same index will be moved to the right
    by the number of pixels in this list.

    """
    offys = ListProperty([])
    """y-offsets. The texture at the same index will be moved upward by
    the number of pixels in this list.

    """
    texture_rectangles = DictProperty({})
    """Private.

    Rectangle instructions for each of the textures, keyed by the
    texture.

    """
    _no_tex_upd = BooleanProperty(False)
    _no_use_canvas = BooleanProperty(False)

    def __init__(self, **kwargs):
        """Make triggers and bind."""
        kwargs['size_hint'] = (None, None)
        self.group = InstructionGroup()
        super().__init__(**kwargs)

    def on_texs(self, *args):
        """Make rectangles for each of the textures and add them to the
        canvas, taking their stacking heights into account.

        """
        if self._no_tex_upd:
            return
        if not self.canvas or not self.texs:
            Clock.schedule_once(self.on_texs, 0)
            return
        texlen = len(self.texs)
        props = ('stackhs', 'offxs', 'offys')
        # Ensure each property is the same length as my texs, padding
        # with 0 as needed
        for prop in props:
            proplen = len(getattr(self, prop))
            if proplen > texlen:
                setattr(self, prop, getattr(self, prop)[:proplen-texlen])
            if texlen > proplen:
                propval = list(getattr(self, prop))
                propval += [0] * (texlen - proplen)
                setattr(self, prop, propval)
        self._clear_rects()
        w = h = 0
        stackh = 0
        i = 0
        (x, y) = self.pos
        if self.group in self.canvas.children:
            self.canvas.remove(self.group)
        for tex in self.texs:
            offx = self.offxs[i] if self.offxs[i] > 0 else 0
            offy = self.offys[i] if self.offys[i] > 0 else 0
            rect = Rectangle(
                pos=(x+offx, y+offy+stackh),
                size=tex.size,
                texture=tex
            )
            self.texture_rectangles[tex] = rect
            self.group.add(rect)
            tw = tex.width + offx
            th = tex.height + offy + stackh
            if tw > w:
                w = tw
            if th > h:
                h = th
            stackh += self.stackhs[i] if self.stackhs[i] > 0 else 0
            i += 1
        if not self._no_use_canvas:
            self.canvas.add(self.group)
        self.size = (w, h)

    def on_pos(self, *args):
        """Move all the rectangles within this widget to reflect the widget's
        position. Take stacking height into account.

        """
        stackh = 0
        i = 0
        (x, y) = self.pos
        for rect in self.texture_rectangles.values():
            rect.pos = (x, y+stackh)
            stackh += self.stackhs[i]
            i += 1

    def _clear_rects(self):
        """Get rid of all my rectangles (but not those of my children)."""
        for rect in self.texture_rectangles.values():
            self.canvas.remove(rect)
        self.texture_rectangles = {}

    def clear(self):
        """Clear my rectangles, ``texs``, and ``stackhs``."""
        self._clear_rects()
        self.texs = []
        self.stackhs = []
        self.size = [1, 1]

    def insert(self, i, tex):
        """Insert the texture into my ``texs``, waiting for the creation of
        the canvas if necessary.

        """
        if not self.canvas:
            Clock.schedule_once(
                lambda dt: TextureStack.insert(
                    self, i, tex), 0)
            return
        self.texs.insert(i, tex)

    def append(self, tex):
        """``self.insert(len(self.texs), tex)``"""
        self.insert(len(self.texs), tex)

    def __delitem__(self, i):
        """Remove a texture, its rectangle, and its stacking height"""
        tex = self.texs[i]
        try:
            rect = self.texture_rectangles[tex]
            self.canvas.remove(rect)
            del self.texture_rectangles[tex]
        except KeyError:
            pass
        del self.stackhs[i]
        del self.texs[i]

    def __setitem__(self, i, v):
        """First delete at ``i``, then insert there"""
        if len(self.texs) > 0:
            self._no_upd_texs = True
            self.__delitem__(i)
            self._no_upd_texs = False
        self.insert(i, v)

    def pop(self, i=-1):
        """Delete the stacking height and texture at ``i``, returning the
        texture.

        """
        self.stackhs.pop(i)
        return self.texs.pop(i)


class ImageStack(TextureStack):
    """Instead of supplying textures themselves, supply paths to where the
    textures may be loaded from.

    """
    paths = ListProperty([])
    pathtexs = DictProperty({})

    def on_paths(self, *args):
        """Make textures from the paths and assign them at the same index"""
        i = 0
        for path in self.paths:
            if path in self.pathtexs:
                if self.texs.indexof(self.pathtexs[path]) == i:
                    continue
            else:
                self.pathtexs[path] = Image.load(resource_find(path)).texture
            if i == len(self.texs):
                self.texs.append(self.pathtexs[path])
            else:
                self.texs[i] = self.pathtexs[path]
            i += 1

    def clear(self):
        """Clear paths, textures, rectangles"""
        self.paths = []
        super().clear()

    def insert(self, i, v):
        """Insert a string to my paths"""
        if not isinstance(v, str):
            raise TypeError("Paths only")
        self.paths.insert(i, v)

    def __delitem__(self, i):
        """Delete texture, rectangle, path"""
        super().__delitem__(i)
        del self.paths[i]

    def pop(self, i=-1):
        """Delete and return a path"""
        r = self.paths[i]
        del self[i]
        return r
