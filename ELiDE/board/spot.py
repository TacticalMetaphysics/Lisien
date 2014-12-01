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
from kivy.lang import Builder
from kivy.logger import Logger
from ELiDE.kivygarden.collider import CollideEllipse
from .pawnspot import PawnSpot


class Spot(PawnSpot):
    """The icon that represents a :class:`Place`.

    Each :class:`Spot` is located on the Board that represents the
    :class:`Character` that the underlying :class:`Place` is in. Its
    coordinates are relative to its :class:`Board`, not necessarily
    the window the :class:`Board` is in.

    """
    place = ObjectProperty()
    offset = NumericProperty(4)
    collider = ObjectProperty()
    _ignore_place = BooleanProperty(False)
    _touchpos = ListProperty([])

    def __init__(self, **kwargs):
        """Deal with triggers and bindings, and arrange to take care of
        changes in game-time.

        """
        self._trigger_upd_collider = Clock.create_trigger(self._upd_collider)
        self._trigger_move_to_touch = Clock.create_trigger(self._move_to_touch)
        self._trigger_upd_pawns_here = Clock.create_trigger(
            self._upd_pawns_here
        )
        self._trigger_upd_remote_x = Clock.create_trigger(self.upd_remote_x)
        self._trigger_upd_remote_y = Clock.create_trigger(self.upd_remote_y)
        kwargs['size_hint'] = (None, None)
        super().__init__(**kwargs)
        self.bind(
            center=self._trigger_upd_pawns_here
        )

    def _default_pos(self):
        """Return the position on the board to use when I don't have
        one. Given as a pair of floats between 0 and 1.

        """
        # If one spot is without a position, maybe the rest of them
        # are too, and so maybe the board should do a full layout.
        self.board.spots_unposd += 1
        return (0.5, 0.5)

    def _default_paths(self):
        """Return a list of paths to use for my graphics by default."""
        return ['orb.png']

    def _update(self, *args):
        """Private use. Update my ``paths`` and ``pos`` with what's in my
        ``place``.

        """
        if (
                self.remote_map['_x'] != self.x / self.board.width or
                self.remote_map['_y'] != self.y / self.board.height
        ):
            self._ignore_place = True
            self.pos = (
                self.remote_map['_x'] * self.board.width,
                self.remote_map['_y'] * self.board.height
            )
            self._ignore_place = False


    def add_widget(self, wid, i=0, canvas=None):
        """Put the widget's canvas in my ``board``'s ``pawnlayout`` rather
        than my own canvas.

        The idea is that all my child widgets are to be instances of
        :class:`Pawn`, and should therefore be drawn after every
        non-:class:`Pawn` widget, so that pawns are on top of spots
        and arrows.

        """
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
            if hasattr(child, 'group'):
                if child.group in pawncanvas.children:
                    pawncanvas.remove(child.group)
                pawncanvas.add(child.group)
            else:
                pawncanvas.add(child.canvas)
        self.pospawn(wid)

    def pospawn(self, pawn):
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


    def _upd_collider(self, *args):
        """Update my collider to match my present position. and size.

        Assumes that I am an ellipse.

        """
        (x, y) = self.center
        (w, h) = self.size
        rx = w / 2
        ry = h / 2
        self.collider = CollideEllipse(
            x=x, y=y, rx=rx, ry=ry
        )

    def _upd_pawns_here(self, *args):
        """Move any :class:`Pawn` atop me so it still *is* on top of me,
        presumably after I've moved.

        """
        self._trigger_upd_collider()
        for pawn in self.children:
            self.pospawn(pawn)

    def collide_point(self, x, y):
        """Check my collider."""
        if not self.collider:
            return False
        return (x, y) in self.collider

    def on_touch_move(self, touch):
        """If I'm being dragged, move to follow the touch."""
        if not self.selected:
            return False
        self._touchpos = touch.pos
        self._trigger_move_to_touch()
        return True

    def _move_to_touch(self, *args):
        """Move so I'm centered at my ``touchpos``, and trigger an update of
        my collider.

        """
        if self._touchpos != [] and self.center != self._touchpos:
            self.center = self._touchpos
            self._trigger_upd_collider()

    def on_touch_up(self, touch):
        """Unset ``touchpos``"""
        self._touchpos = []

    def __repr__(self):
        """Give my place's name and my position."""
        return "{}@({},{})".format(self.place.name, self.x, self.y)

    def on_remote_map(self, *args):
        if not PawnSpot.on_remote_map(self, *args):
            return
        try:
            self.pos = (
                self.remote_map['_x'] * self.board.width,
                self.remote_map['_y'] * self.board.height
            )
        except KeyError:
            (x, y) = self._default_pos()
            self.remote_map['_x'] = x
            self.remote_map['_y'] = y
            self.pos = (
                x * self.board.width,
                y * self.board.height
            )

        @self.remote_map.listener(key='_x')
        def listen_x(k, v):
            self.unbind(x=self._trigger_upd_remote_x)
            self.x = v * self.board.width
            self.bind(x=self._trigger_upd_remote_x)

        @self.remote_map.listener(key='_y')
        def listen_y(k, v):
            self.unbind(y=self._trigger_upd_remote_y)
            self.y = v * self.board.height
            self.bind(y=self._trigger_upd_remote_y)

        self.bind(
            x=self._trigger_upd_remote_x,
            y=self._trigger_upd_remote_y,
            pos=self._trigger_upd_collider
        )

    def upd_remote_x(self, *args):
        self.remote_map['_x'] = self.x / self.board.width

    def upd_remote_y(self, *args):
        self.remote_map['_y'] = self.y / self.board.height

kv = """
<Spot>:
    remote_map: EntityRemoteMapping(self.place) if self.place else None
"""
Builder.load_string(kv)
