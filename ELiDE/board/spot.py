# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Widget to represent :class:`Place`s. :class:`Pawn` moves around on
top of these.

"""
from functools import partial
from kivy.properties import (
    AliasProperty,
    ListProperty,
    ObjectProperty,
    NumericProperty,
    BooleanProperty
)
from kivy.clock import Clock
from kivy.logger import Logger
from ELiDE.kivygarden.collider import CollideEllipse
from .pawnspot import PawnSpot
from ..util import trigger


class Spot(PawnSpot):
    """The icon that represents a :class:`Place`.

    Each :class:`Spot` is located on the Board that represents the
    :class:`Character` that the underlying :class:`Place` is in. Its
    coordinates are relative to its :class:`Board`, not necessarily
    the window the :class:`Board` is in.

    """
    offset = NumericProperty(3)
    collider = ObjectProperty()
    collided = BooleanProperty(False)
    place = AliasProperty(
        lambda self: self.remote,
        lambda self, v: self.remote.setter()(v),
        bind=('remote',)
    )
    default_image_paths = ['orb.png']
    default_pos = (0.5, 0.5)
    _touchpos = ListProperty([])

    def __init__(self, **kwargs):
        """Deal with triggers and bindings, and arrange to take care of
        changes in game-time.

        """
        self._pospawn_partials = {}
        self._pospawn_triggers = {}
        kwargs['size_hint'] = (None, None)
        if 'place' in kwargs:
            kwargs['remote'] = kwargs['place']
            del kwargs['place']
        super().__init__(**kwargs)
        self.bind(pos=self._trigger_upd_pawns_here)
        self.bind(
            size=self._trigger_upd_collider,
            pos=self._trigger_upd_collider
        )

    def _upd_collider(self, *args):
        rx = self.width / 2
        ry = self.height / 2
        if (
            not hasattr(self.collider, 'pos') or
            self.collider.pos != self.center or
            self.collider.rx != rx or
            self.collider.ry != ry
        ):
            self.collider = CollideEllipse(
                x=self.center_x, y=self.center_y, rx=rx, ry=ry
            )
    _trigger_upd_collider = trigger(_upd_collider)
        
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

    def _bind_trigger_pospawn(self, pawn):
        trigger = self._get_pospawn_trigger(pawn)
        pawn.bind(
            pos=trigger,
            size=trigger
        )
        self.bind(
            pos=trigger,
            size=trigger
        )

    def _unbind_trigger_pospawn(self, pawn):
        trigger = self._get_pospawn_trigger(pawn)
        pawn.unbind(
            pos=trigger,
            size=trigger
        )
        self.unbind(
            pos=trigger,
            size=trigger
        )

    def _upd_pos(self, *args):
        if self.board is None:
            Clock.schedule_once(self._upd_pos, 0)
            return
        self.pos = (
            int(self.remote.get('_x', self.default_pos[0]) * self.board.width),
            int(self.remote.get('_y', self.default_pos[1]) * self.board.height)
        )

    def listen_pos(self, *args):
        self.remote.listener(
            fun=self._upd_pos,
            stat='pos'
        )

    def unlisten_pos(self, *args):
        self.remote.unlisten(
            fun=self._upd_pos,
            stat='pos'
        )

    def on_remote(self, *args):
        super().on_remote(*args)
        self.listen_pos()
        self._upd_pos()

    def push_pos(self, *args):
        """Set my current position, expressed as proportions of the board's
        width and height, into the ``_x`` and ``_y`` keys of the
        entity in my ``remote`` property, such that it will be
        recorded in the database.

        """
        self.remote['_x'] = self.x / self.board.width
        self.remote['_y'] = self.y / self.board.height
    _trigger_push_pos = trigger(push_pos)

    def add_widget(self, wid, i=0, canvas=None):
        """Put the widget's canvas in my ``board``'s ``pawnlayout`` rather
        than my own canvas.

        The idea is that all my child widgets are to be instances of
        :class:`Pawn`, and should therefore be drawn after every
        non-:class:`Pawn` widget, so that pawns are on top of spots
        and arrows.

        """
        super().add_widget(wid, i, canvas)
        self._bind_trigger_pospawn(wid)
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
            self._unbind_trigger_pospawn(wid)
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

    def collide_point(self, x, y):
        """Check my collider."""
        self._upd_collider()
        return (x, y) in self.collider

    def on_touch_move(self, touch):
        """If I'm being dragged, move to follow the touch."""
        if not self.hit:
            return False
        self._touchpos = touch.pos
        self.center = self._touchpos
        return True

    def on_touch_up(self, touch):
        """Unset ``touchpos``"""
        if not self.hit:
            return False
        if self._touchpos:
            self.center = self._touchpos
            self._touchpos = []
            self._trigger_push_pos()
        self.collided = False
        self.hit = False

    def __repr__(self):
        """Give my name and position."""
        return "{}@({},{})".format(self.name, self.x, self.y)
