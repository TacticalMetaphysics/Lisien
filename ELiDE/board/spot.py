# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Widgets to represent places. Pawns move around on top of these."""
from kivy.properties import (
    ListProperty,
    ObjectProperty,
    BooleanProperty,
    AliasProperty
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
    engine = AliasProperty(
        lambda self: self.board.engine if self.board else None,
        lambda self, v: None,
        bind=('board',)
    )
    _ignore_place = BooleanProperty(False)
    _touchpos = ListProperty([])

    def __init__(self, **kwargs):
        """Deal with triggers and bindings, and arrange to take care of
        changes in game-time.

        """
        self._trigger_move_to_touch = Clock.create_trigger(self._move_to_touch)
        self._trigger_upd_pawns_here = Clock.create_trigger(
            self._upd_pawns_here
        )
        self._trigger_update = Clock.create_trigger(self._update)
        kwargs['size_hint'] = (None, None)
        super().__init__(**kwargs)
        self.board.spot[self.place.name] = self
        self.bind(
            center=self._trigger_upd_pawns_here
        )
        self._ignore_place = True
        self.pos = (
            self.place['_x'] * self.board.width,
            self.place['_y'] * self.board.height
        )
        self._ignore_place = False
        self.paths = self.place['_image_paths']

    def _update(self, *args):
        if self.place['_image_paths'] != self.paths:
            self.paths = self.place['_image_paths']
        if (
                self.place['_x'] != self.x / self.board.width or
                self.place['_y'] != self.y / self.board.height
        ):
            self._ignore_place = True
            self.pos = (
                self.place['_x'] * self.board.width,
                self.place['_y'] * self.board.height
            )
            self._ignore_place = False

    def add_widget(self, pawn, index=0, canvas='after'):
        super().add_widget(pawn, index, canvas)
        pawn.pos = self.center

    def on_paths(self, *args):
        """When I get different imagery, save it in my :class:`Place`"""
        if not self._ignore_place:
            self.place["_image_paths"] = self.paths
        super().on_paths(*args)

    def on_x(self, *args):
        if self._ignore_place:
            return
        self.place['_x'] = self.x / self.board.width

    def on_y(self, *args):
        if self._ignore_place:
            return
        self.place['_y'] = self.y / self.board.height

    def _upd_pawns_here(self, *args):
        """Move any :class:`Pawn` atop me so it still *is* on top of me,
        presumably after I've moved.

        """
        for pawn in self.children:
            pawn.pos = self.center

    def on_touch_down(self, touch):
        """If the touch hits me, grab it and put myself in its userdict"""
        if self.collide_point(*touch.pos):
            self._touch = touch
            return True
        return False

    def on_touch_move(self, touch):
        """If I'm being dragged, move to follow the touch."""
        if self.board.layout.grabbed is not self:
            return
        self._touchpos = touch.pos
        self._trigger_move_to_touch()
        return self

    def _move_to_touch(self, *args):
        if self._touchpos != [] and self.center != self._touchpos:
            self.center = self._touchpos

    def on_touch_up(self, touch):
        if self._touchpos:
            self.coords = self.pos
        self._touchpos = []
        if self.collide_point(*touch.pos):
            return self

    def __repr__(self):
        return "{}@({},{})".format(self.place.name, self.x, self.y)
