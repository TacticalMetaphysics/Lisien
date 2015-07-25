# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Code that draws the box around a Pawn or Spot when it's selected"""
from kivy.properties import (
    ObjectProperty,
    BooleanProperty,
    ListProperty,
    StringProperty,
    NumericProperty,
    ReferenceListProperty
)
from kivy.graphics import (
    InstructionGroup,
    Color,
    Line
)
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.lang import Builder
from ELiDE.kivygarden.texturestack import ImageStack
from ..remote import MirrorMapping
from ..util import trigger


class PawnSpot(ImageStack, MirrorMapping):
    """The kind of ImageStack that represents a :class:`Thing` or
    :class:`Place`.

    """
    board = ObjectProperty()
    engine = ObjectProperty()
    selected = BooleanProperty(False)
    linecolor = ListProperty()
    name = ObjectProperty()
    listen_branch = StringProperty('master')
    listen_tick = NumericProperty(0)
    listen_time = ReferenceListProperty(listen_branch, listen_tick)
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
        self.bind(
            paths=self._trigger_push_image_paths,
            offxs=self._trigger_push_offxs,
            offys=self._trigger_push_offys,
            stackhs=self._trigger_push_stackhs
        )
        if '_image_paths' not in self.remote:
            self.remote['_image_paths'] = list(self.paths)
        if '_offxs' not in self.remote:
            self.remote['_offxs'] = zeroes
        if '_offys' not in self.remote:
            self.remote['_offys'] = zeroes
        if '_stackhs' not in self.remote:
            self.remote['_stackhs'] = zeroes
        self.remote.listeners(
            fun=self._listen_func,
            stats=(
                '_image_paths',
                '_offxs',
                '_offys',
                '_stackhs',
                '_x',
                '_y'
            )
        )
        self.sync()

    def push_image_paths(self, *args):
        self.remote['_image_paths'] = list(self.paths)
    _trigger_push_image_paths = trigger(push_image_paths)

    def push_offxs(self, *args):
        self.remote['_offxs'] = list(self.offxs)
    _trigger_push_offxs = trigger(push_offxs)

    def push_offys(self, *args):
        self.remote['_offys'] = list(self.offys)
    _trigger_push_offys = trigger(push_offys)

    def push_stackhs(self, *args):
        self.remote['_stackhs'] = list(self.stackhs)
    _trigger_push_stackhs = trigger(push_stackhs)

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
    time: self.board.time if self.board else ['master', 0]
    linecolor: [0., 1., 1., 1.] if self.selected else [0., 0., 0., 0.]
"""
Builder.load_string(kv)
