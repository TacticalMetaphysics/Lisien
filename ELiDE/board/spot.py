# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Widgets to represent places. Pawns move around on top of these."""
from kivy.properties import (
    ListProperty,
    ObjectProperty,
    BooleanProperty
)
from kivy.clock import Clock
from ..texturestack import ImageStack


class Spot(ImageStack):
    """The icon that represents a Place.

    The Spot is located on the Board that represents the same
    Dimension that the underlying Place is in. Its coordinates are
    relative to its Board, not necessarily the window the Board is in.

    """
    place = ObjectProperty()
    board = ObjectProperty()
    engine = ObjectProperty()
    _ignore_place = BooleanProperty(False)
    _touch = ObjectProperty(None, allownone=True)
    pawns_here = ListProperty([])

    def __init__(self, **kwargs):
        self._trigger_move_to_touch = Clock.create_trigger(self._move_to_touch)
        self._trigger_upd_pawns_here = Clock.create_trigger(self._upd_pawns_here)
        super().__init__(**kwargs)
        self.board.spot[self.place.name] = self
        self.bind(
            pawns_here=self._trigger_upd_pawns_here,
            center=self._trigger_upd_pawns_here
        )
        self.engine.on_time(
            self.engine.function(
                self.handle_time,
                self.place.name + "_handle_time"
            )
        )
        self.handle_time(*self.engine.time)

    def on_x(self, *args):
        if not self._ignore_place:
            self.place["_x"] = self.x

    def on_y(self, *args):
        if not self._ignore_place:
            self.place["_y"] = self.y

    def on_paths(self, *args):
        if not self._ignore_place:
            self.place["_image_paths"] = self.paths
        super().on_paths(*args)

    def handle_time(self, *args):
        self._ignore_place = True
        try:
            self.pos = (self.place["_x"], self.place["_y"])
        except KeyError:
            self.pos = (100, 100)
        try:
            self.paths = self.place["_image_paths"]
        except KeyError:
            import ELiDE
            self.paths = [ELiDE.__path__[0] + "/assets/orb.png"]
        self._ignore_place = False

    def _upd_pawns_here(self, *args):
        for pawn in self.pawns_here:
            pawn.pos = self.center

    def on_touch_down(self, touch):
        if (
                not self.collide_point(*touch.pos)
                or 'spot' in touch.ud
        ):
            return
        touch.grab(self)
        touch.ud['spot'] = self
        self._touch = touch
        return self

    def on_touch_move(self, touch):
        if 'portaling' in touch.ud or 'pawn' in touch.ud:
            touch.ungrab(self)
            return
        elif 'spot' in touch.ud:
            if touch.ud['spot'] is not self:
                return
            self._touch = touch
            self._trigger_move_to_touch()
            return self

    def _move_to_touch(self, *args):
        if self._touch:
            self.center = self.board.parent.to_local(*self._touch.pos)

    def on_touch_up(self, touch):
        if self._touch:
            self.coords = self.pos
        self._touch = None
        if self.collide_point(*touch.pos):
            return self

    def __repr__(self):
        return "{}@({},{})".format(self.place.name, self.x, self.y)
