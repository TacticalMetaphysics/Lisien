# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Widget to represent :class:`Place`s. :class:`Pawn` moves around on
top of these.

"""
from kivy.clock import Clock

from kivy.properties import (
    AliasProperty,
    ListProperty,
    ObjectProperty,
    NumericProperty,
    BooleanProperty
)
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
        lambda self: self.proxy,
        lambda self, v: self.setter('proxy')(v),
        bind=('proxy',)
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
            kwargs['proxy'] = kwargs['place']
            del kwargs['place']
        super().__init__(**kwargs)
        self.bind(pos=self._trigger_upd_pawns_here)

    def on_board(self, *args):
        self.board.bind(size=self._upd_pos)

    def _upd_pos(self, *args):
        if self.board is None:
            Clock.schedule_once(self._upd_pos, 0)
            return
        self.pos = (
            int(self.proxy.get('_x', self.default_pos[0]) * self.board.width),
            int(self.proxy.get('_y', self.default_pos[1]) * self.board.height)
        )

    def finalize(self, initial=True):
        if initial:
            self._upd_pos()
        super().finalize(initial)

    def push_pos(self, *args):
        """Set my current position, expressed as proportions of the board's
        width and height, into the ``_x`` and ``_y`` keys of the
        entity in my ``proxy`` property, such that it will be
        recorded in the database.

        """
        self.proxy['_x'] = self.x / self.board.width
        self.proxy['_y'] = self.y / self.board.height
    _trigger_push_pos = trigger(push_pos)

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
