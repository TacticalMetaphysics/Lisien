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

    def __init__(self, **kwargs):
        self._trigger_upd_from_mirror_image_paths = Clock.create_trigger(
            self.upd_from_mirror_image_paths
        )
        self._trigger_upd_to_remote_image_paths = Clock.create_trigger(
            self.upd_to_remote_image_paths
        )
        self._trigger_upd_from_mirror_offxs = Clock.create_trigger(
            self.upd_from_mirror_offxs
        )
        self._trigger_upd_to_remote_offxs = Clock.create_trigger(
            self.upd_to_remote_offxs
        )
        self._trigger_upd_from_mirror_offys = Clock.create_trigger(
            self.upd_from_mirror_offys
        )
        self._trigger_upd_to_remote_offys = Clock.create_trigger(
            self.upd_to_remote_offys
        )
        self._trigger_upd_from_mirror_stackhs = Clock.create_trigger(
            self.upd_from_mirror_stackhs
        )
        self._trigger_upd_to_remote_stackhs = Clock.create_trigger(
            self.upd_to_remote_stackhs
        )
        super().__init__(**kwargs)

    def on_remote(self, *args):
        if not super().on_remote(*args):
            return
        if (
                self.remote is None or
                'name' not in self.remote
        ):
            Logger.debug('PawnSpot: bad remote {}'.format(self.remote))
            return
        self._trigger_upd_from_mirror_image_paths()
        self._trigger_upd_from_mirror_offxs()
        self._trigger_upd_from_mirror_offys()
        self._trigger_upd_from_mirror_stackhs()
        self.bind(
            paths=self._trigger_upd_to_remote_image_paths,
            offxs=self._trigger_upd_to_remote_offxs,
            offys=self._trigger_upd_to_remote_offys,
            stackhs=self._trigger_upd_to_remote_stackhs
        )
        self.name = self.remote['name']
        if '_image_paths' not in self.remote:
            self.remote['_image_paths'] = self._default_image_paths()
        if '_offxs' not in self.remote:
            self.remote['_offxs'] = self._default_offxs()
        if '_offys' not in self.remote:
            self.remote['_offys'] = self._default_offys()
        if '_stackhs' not in self.remote:
            self.remote['_stackhs'] = self._default_stackhs()
        return True

    def on_mirror(self, *args):
        if not self.mirror:
            return
        for k in (
                'name',
                '_image_paths',
                '_offxs',
                '_offys',
                '_stackhs'
        ):
            if k not in self.mirror:
                return
        if self.paths != self.mirror['_image_paths']:
            self._trigger_upd_from_mirror_image_paths()
        if self.offxs != self.mirror['_offxs']:
            self._trigger_upd_from_mirror_offxs()
        if self.offys != self.mirror['_offys']:
            self._trigger_upd_from_mirror_offys()
        if self.stackhs != self.mirror['_stackhs']:
            self._trigger_upd_from_mirror_stackhs()
        return True

    def upd_from_mirror_image_paths(self, *args):
        if not self.mirror:
            if self.remote:
                Logger.debug(
                    'PawnSpot: no mirror of {}'.format(self.remote['name'])
                )
            Clock.schedule_once(self.upd_from_mirror_image_paths, 0)
            return
        if '_image_paths' not in self.mirror:
            return
        self.unbind(
            paths=self._trigger_upd_to_remote_image_paths
        )
        self.paths = self.mirror['_image_paths']
        self.bind(
            paths=self._trigger_upd_to_remote_image_paths
        )

    def upd_to_remote_image_paths(self, *args):
        self.remote['_image_paths'] = self.paths

    def upd_from_mirror_offxs(self, *args):
        if not self.mirror:
            Clock.schedule_once(self.upd_from_mirror_offxs, 0)
            return
        if '_offxs' not in self.mirror:
            return
        self.unbind(
            offxs=self._trigger_upd_to_remote_offxs
        )
        self.offxs = self.mirror['_offxs']
        self.bind(
            offxs=self._trigger_upd_to_remote_offxs
        )

    def upd_to_remote_offxs(self, *args):
        self.remote['_offxs'] = self.offxs

    def upd_from_mirror_offys(self, *args):
        if not self.mirror:
            Clock.schedule_once(self.upd_from_mirror_offys, 0)
        if '_offys' not in self.mirror:
            return
        self.unbind(
            offys=self._trigger_upd_to_remote_offys
        )
        self.offys = self.mirror['_offys']
        self.bind(
            offys=self._trigger_upd_to_remote_offys
        )

    def upd_to_remote_offys(self, *args):
        self.remote['_offys'] = self.offys

    def upd_from_mirror_stackhs(self, *args):
        if not self.mirror:
            Clock.schedule_once(self.upd_from_mirror_stackhs, 0)
            return
        if '_stackhs' not in self.mirror:
            return
        self.unbind(
            stackhs=self._trigger_upd_to_remote_stackhs
        )
        self.stackhs = self.mirror['_stackhs']
        self.bind(
            stackhs=self._trigger_upd_to_remote_stackhs
        )

    def upd_to_remote_stackhs(self, *args):
        self.remote['_stackhs'] = self.stackhs

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

    def _default_image_paths(self):
        """Return a list of paths to use for my graphics by default."""
        return ['atlas://rltiles/base.atlas/unseen']

    def _default_offxs(self):
        """Return a list of integers to use for my x-offsets by default."""
        return [0]

    def _default_offys(self):
        """Return a list of integers to use for my y-offsets by default."""
        return [0]

    def _default_stackhs(self):
        """Return a list of integers to use for my stacking heights by
        default.

        """
        return [0]


kv = """
<PawnSpot>:
    time: self.board.time if self.board else ['master', 0]
    linecolor: [0., 1., 1., 1.] if self.selected else [0., 0., 0., 0.]
"""
Builder.load_string(kv)
