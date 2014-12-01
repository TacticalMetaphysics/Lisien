# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Code that draws the box around a Pawn or Spot when it's selected"""
from kivy.properties import (
    ObjectProperty,
    BooleanProperty,
    ListProperty
)
from kivy.graphics import (
    InstructionGroup,
    Color,
    Line
)
from kivy.clock import Clock
from kivy.lang import Builder
from ELiDE.kivygarden.texturestack import ImageStack


class PawnSpot(ImageStack):
    """The kind of ImageStack that represents a :class:`Thing` or
    :class:`Place`.

    """
    board = ObjectProperty()
    engine = ObjectProperty()
    selected = BooleanProperty(False)
    linecolor = ListProperty()
    remote_map = ObjectProperty()
    name = ObjectProperty()
    use_boardspace = True

    def __init__(self, **kwargs):
        self._trigger_upd_remote_name = Clock.create_trigger(
            self.upd_remote_name
        )
        self._trigger_upd_remote_image_paths = Clock.create_trigger(
            self.upd_remote_image_paths
        )
        self._trigger_upd_remote_offxs = Clock.create_trigger(
            self.upd_remote_offxs
        )
        self._trigger_upd_remote_offys = Clock.create_trigger(
            self.upd_remote_offys
        )
        self._trigger_upd_remote_stacking_heights = Clock.create_trigger(
            self.upd_remote_stacking_heights
        )
        super().__init__(**kwargs)

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

    def on_remote_map(self, *args):
        if self.remote_map is None:
            return
        self.name = self.remote_map['name']
        self._oldname = self.name

        if '_image_paths' not in self.remote_map:
            self.remote_map['_image_paths'] = self._default_paths()
        if '_offxs' not in self.remote_map:
            self.remote_map['_offxs'] = self._default_offxs()
        if '_offys' not in self.remote_map:
            self.remote_map['_offys'] = self._default_offys()
        if '_stacking_heights' not in self.remote_map:
            self.remote_map['_stacking_heights'] = self._default_stackhs()
        if self.paths != self.remote_map["_image_paths"]:
            self.paths = self.remote_map["_image_paths"]
        if self.offxs != self.remote_map['_offxs']:
            self.offxs = self.remote_map['_offxs']
        if self.offys != self.remote_map['_offys']:
            self.offys = self.remote_map['_offys']
        if self.stackhs != self.remote_map["_stacking_heights"]:
            self.stackhs = self.remote_map["_stacking_heights"]

        @self.remote_map.listener(key='name')
        def listen_name(k, v):
            self.unbind(name=self._trigger_upd_remote_name)
            self.name = v
            self.bind(name=self._trigger_upd_remote_name)

        @self.remote_map.listener(key='_image_paths')
        def listen_paths(k, v):
            self.unbind(paths=self._trigger_upd_remote_image_paths)
            self.paths = v
            self.bind(paths=self._trigger_upd_remote_image_paths)

        @self.remote_map.listener(key='_offxs')
        def listen_offxs(k, v):
            self.unbind(offxs=self._trigger_upd_remote_offxs)
            self.offxs = v
            self.bind(offxs=self._trigger_upd_remote_offxs)

        @self.remote_map.listener(key='_offys')
        def listen_offys(k, v):
            self.unbind(offys=self._trigger_upd_remote_offys)
            self.offys = v
            self.bind(offys=self._trigger_upd_remote_offys)

        @self.remote_map.listener(key='_stacking_heights')
        def listen_stackhs(k, v):
            self.unbind(stackhs=self._trigger_upd_remote_stacking_heights)
            self.stackhs = v
            self.bind(stackhs=self._trigger_upd_remote_stacking_heights)

        self.bind(
            name=self._trigger_upd_remote_name,
            paths=self._trigger_upd_remote_image_paths,
            offxs=self._trigger_upd_remote_offxs,
            offys=self._trigger_upd_remote_offys,
            stackhs=self._trigger_upd_remote_stacking_heights
        )

        return True

    def upd_remote_name(self, *args):
        self.remote_map['name'] = self.name

    def upd_remote_image_paths(self, *args):
        self.remote_map['_image_paths'] = self.paths

    def upd_remote_offxs(self, *args):
        self.remote_map['_offxs'] = self.offxs

    def upd_remote_offys(self, *args):
        self.remote_map['_offys'] = self.offys

    def upd_remote_stacking_heights(self, *args):
        self.remote_map['_stacking_heights'] = self.stackhs

    def _default_paths(self):
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
    engine: self.board.layout.app.engine if self.board and self.board.layout else None
    linecolor: [0., 1., 1., 1.] if self.selected else [0., 0., 0., 0.]
"""
Builder.load_string(kv)
