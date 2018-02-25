# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Code that draws the box around a Pawn or Spot when it's selected"""
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
    _finalized = BooleanProperty()
    selected = BooleanProperty(False)
    hit = BooleanProperty(False)
    linecolor = ListProperty()
    name = ObjectProperty()
    use_boardspace = True

    def finalize(self, initial=True):
        """Call this after you've created all the PawnSpot you need and are ready to add them to the board."""
        if not self._finalized:
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
        initial = self._finalized is None
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


kv = """
<PawnSpot>:
    linecolor: [0., 1., 1., 1.] if self.selected else [0., 0., 0., 0.]
"""
Builder.load_string(kv)
