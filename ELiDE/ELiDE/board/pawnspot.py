# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Code that draws the box around a Pawn or Spot when it's selected"""
from functools import partial

from kivy.properties import (
    ObjectProperty,
    BooleanProperty,
    ListProperty,
)
from kivy.graphics import (
    InstructionGroup,
    Color,
    Line
)
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.lang import Builder
from ELiDE.kivygarden.texturestack import ImageStack
from ..util import trigger


class PawnSpot(ImageStack):
    """The kind of ImageStack that represents a :class:`Thing` or
    :class:`Place`.

    """
    board = ObjectProperty()
    proxy = ObjectProperty()
    engine = ObjectProperty()
    selected = BooleanProperty(False)
    hit = BooleanProperty(False)
    linecolor = ListProperty()
    name = ObjectProperty()
    use_boardspace = True

    def finalize(self, initial=True):
        """Call this after you've created all the PawnSpot you need and are ready to add them to the board."""
        if getattr(self, '_finalized', False):
            return
        if (
                self.proxy is None or
                not hasattr(self.proxy, 'name')
        ):
            Clock.schedule_once(self.finalize, 0)
            return
        if initial:
            self.name = self.proxy.name
            self.paths = self.proxy.setdefault(
                '_image_paths', self.default_image_paths
            )
            zeroes = [0] * len(self.paths)
            self.offxs = self.proxy.setdefault('_offxs', zeroes)
            self.offys = self.proxy.setdefault('_offys', zeroes)
            self.proxy.connect(self._trigger_pull_from_proxy)
        self.bind(
            paths=self._trigger_push_image_paths,
            offxs=self._trigger_push_offxs,
            offys=self._trigger_push_offys
        )
        self._finalized = True

    def unfinalize(self):
        self.unbind(
            paths=self._trigger_push_image_paths,
            offxs=self._trigger_push_offxs,
            offys=self._trigger_push_offys
        )
        self._finalized = False

    def pull_from_proxy(self, *args):
        initial = not hasattr(self, '_finalized')
        self.unfinalize()
        for key, att in [
                ('_image_paths', 'paths'),
                ('_offxs', 'offxs'),
                ('_offys', 'offys')
        ]:
            if key in self.proxy and self.proxy[key] != getattr(self, att):
                setattr(self, att, self.proxy[key])
        self.finalize(initial)

    def _trigger_pull_from_proxy(self, *args, **kwargs):
        Clock.unschedule(self.pull_from_proxy)
        Clock.schedule_once(self.pull_from_proxy, 0)

    @trigger
    def _trigger_push_image_paths(self, *args):
        self.proxy['_image_paths'] = list(self.paths)

    @trigger
    def _trigger_push_offxs(self, *args):
        self.proxy['_offxs'] = list(self.offxs)

    @trigger
    def _trigger_push_offys(self, *args):
        self.proxy['_offys'] = list(self.offys)

    @trigger
    def _trigger_push_stackhs(self, *args):
        self.proxy['_stackhs'] = list(self.stackhs)

    def on_linecolor(self, *args):
        """If I don't yet have the instructions for drawing the selection box
        in my canvas, put them there. In any case, set the
        :class:`Color` instruction to match my current ``linecolor``.

        """
        if hasattr(self, 'color'):
            self.color.rgba = self.linecolor
            return

        def upd_box_points(*args):
            self.box.points = [
                self.x, self.y,
                self.right, self.y,
                self.right, self.top,
                self.x, self.top,
                self.x, self.y
            ]
        boxgrp = InstructionGroup()
        self.color = Color(*self.linecolor)

        boxgrp.add(self.color)
        self.box = Line()
        upd_box_points()
        self.bind(
            pos=upd_box_points,
            size=upd_box_points
        )
        boxgrp.add(self.box)
        boxgrp.add(Color(1., 1., 1.))
        self.group.add(boxgrp)

    @trigger
    def restack(self, *args):
        childs = sorted(list(self.children), key=lambda child: child.priority, reverse=True)
        self.clear_widgets()
        for child in childs:
            self.add_widget(child)

    def add_widget(self, wid, i=None, canvas=None):
        """Put the widget's canvas in my ``board``'s ``pawnlayout`` rather
        than my own canvas.

        The idea is that all my child widgets are to be instances of
        :class:`Pawn`, and should therefore be drawn after every
        non-:class:`Pawn` widget, so that pawns are on top of spots
        and arrows.

        """
        if i is None:
            for i, child in enumerate(self.children, start=1):
                if wid.priority < child.priority:
                    i = len(self.children) - i
                    break
        super().add_widget(wid, i, canvas)
        self.bind_trigger_pospawn(wid)
        if not hasattr(wid, 'group'):
            return
        wid._no_use_canvas = True
        mycanvas = (
            self.canvas.after if canvas == 'after' else
            self.canvas.before if canvas == 'before' else
            self.canvas
        )
        pawncanvas = (
            self.board.pawnlayout.canvas.after if canvas == 'after' else
            self.board.pawnlayout.canvas.before if canvas == 'before' else
            self.board.pawnlayout.canvas
        )
        mycanvas.remove(wid.canvas)
        for child in self.children:
            if hasattr(child, 'group'):
                if child.group in pawncanvas.children:
                    pawncanvas.remove(child.group)
                pawncanvas.add(child.group)
            else:
                pawncanvas.add(child.canvas)
        self.pospawn(wid)

    def remove_widget(self, wid):
        try:
            self.unbind_trigger_pospawn(wid)
        except KeyError:
            pass
        return super().remove_widget(wid)

    def pospawn(self, pawn, *args):
        """Given some :class:`Pawn` instance that's to be on top of me, set
        its ``pos`` so that it looks like it's on top of me but
        doesn't cover me so much that you can't select me.

        """
        i = 0
        for child in self.children:
            if child is pawn:
                break
            i += 1
        off = i * self.offset
        (x, y) = self.center
        pawn.pos = (x+off, y+off)

    def _upd_pawns_here(self, *args):
        """Move any :class:`Pawn` atop me so it still *is* on top of me,
        presumably after I've moved.

        """
        for pawn in self.children:
            self.pospawn(pawn)
    _trigger_upd_pawns_here = trigger(_upd_pawns_here)

    def _get_pospawn_partial(self, pawn):
        if pawn not in self._pospawn_partials:
            self._pospawn_partials[pawn] = partial(
                self.pospawn, pawn
            )
        return self._pospawn_partials[pawn]

    def _get_pospawn_trigger(self, pawn, *args):
        if pawn not in self._pospawn_triggers:
            self._pospawn_triggers[pawn] = Clock.create_trigger(
                self._get_pospawn_partial(pawn)
            )
        return self._pospawn_triggers[pawn]

    def bind_trigger_pospawn(self, pawn):
        trigger = self._get_pospawn_trigger(pawn)
        pawn.bind(
            pos=trigger,
            size=trigger
        )
        self.bind(
            pos=trigger,
            size=trigger
        )

    def unbind_trigger_pospawn(self, pawn):
        trigger = self._get_pospawn_trigger(pawn)
        pawn.unbind(
            pos=trigger,
            size=trigger
        )
        self.unbind(
            pos=trigger,
            size=trigger
        )


kv = """
<PawnSpot>:
    linecolor: [0., 1., 1., 1.] if self.selected else [0., 0., 0., 0.]
"""
Builder.load_string(kv)
