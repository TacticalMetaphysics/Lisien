# This file is part of the kivy-garden project.
# Copyright (c) Zachary Spector, public@zacharyspector.com
# Available under the terms of the MIT license.
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
    InstructionGroup,
    PushMatrix,
    PopMatrix,
    Translate
)
from kivy.graphics.fbo import Fbo
from kivy.properties import (
    AliasProperty,
    DictProperty,
    ListProperty,
    ObjectProperty
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
    texs = ListProperty()
    """Texture objects"""
    offxs = ListProperty()
    """x-offsets. The texture at the same index will be moved to the right
    by the number of pixels in this list.

    """
    offys = ListProperty()
    """y-offsets. The texture at the same index will be moved upward by
    the number of pixels in this list.

    """
    group = ObjectProperty()
    """My ``InstructionGroup``, suitable for addition to whatever ``canvas``."""

    def _get_offsets(self):
        return zip(self.offxs, self.offys)

    def _set_offsets(self, offs):
        offxs = []
        offys = []
        for x, y in offs:
            offxs.append(x)
            offys.append(y)
        self.offxs, self.offys = offxs, offys

    offsets = AliasProperty(
        _get_offsets,
        _set_offsets,
        bind=('offxs', 'offys')
    )
    """List of (x, y) tuples by which to offset the corresponding texture."""
    _texture_rectangles = DictProperty({})
    """Private.

    Rectangle instructions for each of the textures, keyed by the
    texture.

    """

    def __init__(self, **kwargs):
        """Make triggers and bind."""
        kwargs['size_hint'] = (None, None)
        self.translate = Translate(0, 0)
        self.group = InstructionGroup()
        super().__init__(**kwargs)
        self.bind(offxs=self.on_pos, offys=self.on_pos)

    def on_texs(self, *args):
        """Make rectangles for each of the textures and add them to the canvas."""
        if not self.canvas or not self.texs:
            Clock.schedule_once(self.on_texs, 0)
            return
        texlen = len(self.texs)
        # Ensure each property is the same length as my texs, padding
        # with 0 as needed
        for prop in ('offxs', 'offys'):
            proplen = len(getattr(self, prop))
            if proplen > texlen:
                setattr(self, prop, getattr(self, prop)[:proplen-texlen])
            if texlen > proplen:
                propval = list(getattr(self, prop))
                propval += [0] * (texlen - proplen)
                setattr(self, prop, propval)
        self.group.clear()
        self._texture_rectangles = {}
        w = h = 0
        (x, y) = self.pos
        self.translate.x = x
        self.translate.y = y
        self.group.add(PushMatrix())
        self.group.add(self.translate)
        for tex, offx, offy in zip(self.texs, self.offxs, self.offys):
            rect = Rectangle(
                pos=(offx, offy),
                size=tex.size,
                texture=tex
            )
            self._texture_rectangles[tex] = rect
            self.group.add(rect)
            tw = tex.width + offx
            th = tex.height + offy
            if tw > w:
                w = tw
            if th > h:
                h = th
        self.size = (w, h)
        self.group.add(PopMatrix())
        self.canvas.add(self.group)

    def on_pos(self, *args):
        """Translate all the rectangles within this widget to reflect the widget's position.

        """
        (x, y) = self.pos
        self.translate.x = x
        self.translate.y = y

    def clear(self):
        """Clear my rectangles and ``texs``."""
        self.group.clear()
        self._texture_rectangles = {}
        self.texs = []
        self.size = [1, 1]

    def insert(self, i, tex):
        """Insert the texture into my ``texs``, waiting for the creation of
        the canvas if necessary.

        """
        if not self.canvas:
            Clock.schedule_once(
                lambda dt: self.insert(i, tex), 0)
            return
        self.texs.insert(i, tex)

    def append(self, tex):
        """``self.insert(len(self.texs), tex)``"""
        self.insert(len(self.texs), tex)

    def __delitem__(self, i):
        """Remove a texture and its rectangle"""
        tex = self.texs[i]
        try:
            rect = self._texture_rectangles[tex]
            self.canvas.remove(rect)
            del self._texture_rectangles[tex]
        except KeyError:
            pass
        del self.texs[i]

    def __setitem__(self, i, v):
        """First delete at ``i``, then insert there"""
        if len(self.texs) > 0:
            self._no_upd_texs = True
            self.__delitem__(i)
            self._no_upd_texs = False
        self.insert(i, v)

    def pop(self, i=-1):
        """Delete the texture at ``i``, and return it."""
        return self.texs.pop(i)


class ImageStack(TextureStack):
    """Instead of supplying textures themselves, supply paths to where the
    textures may be loaded from.

    """
    paths = ListProperty()
    """List of paths to images you want stacked."""
    pathtexs = DictProperty()
    """Private. Dictionary mapping image paths to textures of the images."""
    pathimgs = DictProperty()
    """Dictionary mapping image paths to ``kivy.core.Image`` objects."""

    def on_paths(self, *args):
        """Make textures from the images in ``paths``, and assign them at the
        same index in my ``texs`` as in my ``paths``.

        """
        for i, path in enumerate(self.paths):
            if path in self.pathtexs:
                if (
                        self.pathtexs[path] in self.texs and
                        self.texs.index(self.pathtexs[path])== i
                ):
                    continue
            else:
                self.pathimgs[path] = img = Image.load(
                    resource_find(path), keep_data=True
                )
                self.pathtexs[path] = img.texture
            if i == len(self.texs):
                self.texs.append(self.pathtexs[path])
            else:
                self.texs[i] = self.pathtexs[path]

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


class TextureStackBatchWidget(Widget):
    """Widget for efficiently drawing many TextureStacks

    Only add TextureStack or ImageStack widgets to this. Avoid adding
    any that are to be changed frequently.

    """
    critical_props = ['texs', 'offxs', 'offys', 'pos']
    """Properties that, when changed on my children, force a redraw."""

    def __init__(self, **kwargs):
        self._trigger_redraw = Clock.create_trigger(self.redraw)
        self._trigger_rebind_children = Clock.create_trigger(self.rebind_children)
        super(TextureStackBatchWidget, self).__init__(**kwargs)

    def on_parent(self, *args):
        if not self.canvas:
            Clock.schedule_once(self.on_parent, 0)
            return
        if not hasattr(self, '_fbo'):
            with self.canvas:
                self._fbo = Fbo(size=self.size)
                self._fbo.add_reload_observer(self.redraw)
                self._translate = Translate(x=self.x, y=self.y)
                self._rectangle = Rectangle(texture=self._fbo.texture, size=self.size)
        self.rebind_children()

    def rebind_children(self, *args):
        child_by_uid = {}
        binds = {prop: self._trigger_redraw for prop in self.critical_props}
        for child in self.children:
            child_by_uid[child.uid] = child
            child.bind(**binds)
        if hasattr(self, '_old_children'):
            old_children = self._old_children
            for uid in set(old_children).difference(child_by_uid):
                old_children[uid].unbind(**binds)
        self.redraw()
        self._old_children = child_by_uid

    def redraw(self, *args):
        fbo = self._fbo
        fbo.bind()
        fbo.clear()
        fbo.clear_buffer()
        fbo.release()
        for child in self.children:
            assert child.canvas not in fbo.children
            fbo.add(child.canvas)

    def on_pos(self, *args):
        if not hasattr(self, '_translate'):
            return
        self._translate.x, self._translate.y = self.pos

    def on_size(self, *args):
        if not hasattr(self, '_rectangle'):
            return
        self._rectangle.size = self._fbo.size = self.size
        self.redraw()

    def add_widget(self, widget, index=0, canvas=None):
        if not isinstance(widget, TextureStack):
            raise TypeError("TextureStackBatch is only for TextureStack")
        if index == 0 or len(self.children) == 0:
            self.children.insert(0, widget)
        else:
            children = self.children
            if index >= len(children):
                index = len(children)

            children.insert(index, widget)
        widget.parent = self
        if hasattr(self, '_fbo'):
            self.rebind_children()

    def remove_widget(self, widget):
        if widget not in self.children:
            return
        self.children.remove(widget)
        widget.parent = None
        if hasattr(self, '_fbo'):
            self.rebind_children()


if __name__ == '__main__':
    from kivy.base import runTouchApp
    from itertools import cycle
    import json

    # I should come up with a prettier demo that has a whole lot of widgets in it

    class DraggyStack(ImageStack):
        def on_touch_down(self, touch):
            if self.collide_point(*touch.pos):
                touch.grab(self)
                self._old_parent = parent = self.parent
                parent.remove_widget(self)
                parent.parent.add_widget(self)
                assert self in parent.parent.children
                assert self.parent == parent.parent
                return True

        def on_touch_move(self, touch):
            if touch.grab_current is self:
                self.center = touch.pos
                return True

        def on_touch_up(self, touch):
            self.parent.remove_widget(self)
            self.pos = self._old_parent.to_local(*self.pos, relative=True)
            self._old_parent.add_widget(self)
            return True

    with open('marsh_davies_island_bg.atlas') as bgf, open('marsh_davies_island_fg.atlas') as fgf:
        pathses = zip(
    ('atlas://marsh_davies_island_bg/' + name for name in json.load(bgf)["marsh_davies_island_bg-0.png"].keys()),
    ('atlas://marsh_davies_island_fg/' + name for name in cycle(json.load(fgf)["marsh_davies_island_fg-0.png"].keys()))
    )
    sbatch = TextureStackBatchWidget(size=(800, 600), pos=(0, 0))
    for i, paths in enumerate(pathses):
        sbatch.add_widget(
            DraggyStack(paths=paths, offxs=[0, 16], offys=[0, 16], pos=(0, 32*i))
        )
    parent = Widget(size=(800, 600), pos=(0, 0))
    parent.add_widget(sbatch)
    runTouchApp(parent)
