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
from kivy.logger import Logger
from kivy.lang import Builder
from ELiDE.kivygarden.texturestack import ImageStack
from ..util import trigger


class PawnSpot(ImageStack):
    """The kind of ImageStack that represents a :class:`Thing` or
    :class:`Place`.

    """
    board = ObjectProperty()
    remote = ObjectProperty()
    engine = ObjectProperty()
    selected = BooleanProperty(False)
    hit = BooleanProperty(False)
    linecolor = ListProperty()
    name = ObjectProperty()
    use_boardspace = True

    def on_remote(self, *args):
        if (
                self.remote is None or
                not hasattr(self.remote, 'name')
        ):
            Logger.debug('PawnSpot: bad remote {}'.format(self.remote))
            return
        self.name = self.remote.name
        self.paths = self.remote.get(
            '_image_paths', self.default_image_paths
        )
        zeroes = [0] * len(self.paths)
        self.offxs = self.remote.get('_offxs', zeroes)
        self.offys = self.remote.get('_offys', zeroes)
        self.stackhs = self.remote.get('_stackhs', zeroes)
        self.remote.listener(self._trigger_pull_from_remote)

    def finalize(self):
        self.bind(
            paths=self._trigger_push_image_paths,
            offxs=self._trigger_push_offxs,
            offys=self._trigger_push_offys,
            stackhs=self._trigger_push_stackhs
        )

    def unfinalize(self):
        self.unbind(
            paths=self._trigger_push_image_paths,
            offxs=self._trigger_push_offxs,
            offys=self._trigger_push_offys,
            stackhs=self._trigger_push_stackhs
        )

    def pull_from_remote(self):
        if self.remote['_image_paths'] != self.paths:
            self.paths = list(self.remote['_image_paths'])
        if self.remote['_offxs'] != self.offxs:
            self.offxs = list(self.remote['_offxs'])
        if self.remote['_offys'] != self.offys:
            self.offys = list(self.remote['_offys'])
        if self.remote['_stackhs'] != self.stackhs:
            self.stackhs = list(self.remote['_stackhs'])

    @trigger
    def _trigger_pull_from_remote(self, *args):
        self.unfinalize()
        self.pull_from_remote()
        self.finalize()

    @trigger
    def _trigger_push_image_paths(self, *args):
        self.remote['_image_paths'] = list(self.paths)

    @trigger
    def _trigger_push_offxs(self, *args):
        self.remote['_offxs'] = list(self.offxs)

    @trigger
    def _trigger_push_offys(self, *args):
        self.remote['_offys'] = list(self.offys)

    @trigger
    def _trigger_push_stackhs(self, *args):
        self.remote['_stackhs'] = list(self.stackhs)

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
