# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Widgets to represent places. Pawns move around on top of these."""
from kivy.properties import (
    ListProperty,
    ObjectProperty,
    BooleanProperty,
    NumericProperty
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
    selected = BooleanProperty()
    offset = NumericProperty(4)
    use_boardspace = True
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
        try:
            self.pos = (
                self.place['_x'] * self.board.width,
                self.place['_y'] * self.board.height
            )
        except KeyError:
            (x, y) = self._default_pos()
            self.place['_x'] = x
            self.place['_y'] = y
            self.pos = (
                x * self.board.width,
                y * self.board.height
            )
        self._ignore_place = False
        try:
            self.paths = self.place['_image_paths']
        except KeyError:
            self.place['_image_paths'] = self.paths = self._default_paths()

    def _default_pos(self):
        # If one spot is without a position, maybe the rest of them
        # are too, and so maybe the board should do a full layout.
        self.board.spots_unposd += 1
        return (0.5, 0.5)

    def _default_paths(self):
        return ['orb.png']

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

    def add_widget(self, wid, i=0, canvas=None):
        super().add_widget(wid, i, canvas)
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
            if hasattr(child, 'group') and child.group in pawncanvas.children:
                pawncanvas.remove(child.group)
            pawncanvas.add(child.group)
        self.pospawn(wid)
        wid._trigger_update()

    def pospawn(self, pawn):
        i = 0
        for child in self.children:
            if child is pawn:
                break
            i += 1
        off = i * self.offset
        (x, y) = self.center
        pawn.pos = (x+off, y+off)

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
            self.pospawn(pawn)

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
