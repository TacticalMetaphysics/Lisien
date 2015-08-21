# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Widget representing things that move about from place to place."""
from kivy.properties import (
    AliasProperty,
    BooleanProperty,
    ObjectProperty,
    NumericProperty,
    ReferenceListProperty
)
from kivy.logger import Logger
from .pawnspot import PawnSpot
from ..util import trigger


class Pawn(PawnSpot):
    """A token to represent a :class:`Thing`.

    :class:`Thing` is the LiSE class to represent items that are
    located in some :class:`Place` or other. Accordingly,
    :class:`Pawn`'s coordinates are never set directly; they are
    instead derived from the location of the :class:`Thing`
    represented. That means a :class:`Pawn` will appear next to the
    :class:`Spot` representing the :class:`Place` that its
    :class:`Thing` is in. The exception is if the :class:`Thing` is
    currently moving from its current :class:`Place` to another one,
    in which case the :class:`Pawn` will appear some distance along
    the :class:`Arrow` that represents the :class:`Portal` it's moving
    through.

    """
    _touch_ox_diff = NumericProperty()
    _touch_oy_diff = NumericProperty()
    _touch_opos_diff = ReferenceListProperty(_touch_ox_diff, _touch_oy_diff)
    _touch = ObjectProperty(None, allownone=True)
    travel_on_drop = BooleanProperty(False)
    loc_name = ObjectProperty()
    next_loc_name = ObjectProperty(None, allownone=True)
    thing = AliasProperty(
        lambda self: self.remote,
        lambda self, v: self.remote.setter()(v),
        bind=('remote',)
    )
    default_image_paths = ['atlas://rltiles/base.atlas/unseen']

    def __init__(self, **kwargs):
        if 'thing' in kwargs:
            kwargs['remote'] = kwargs['thing']
            del kwargs['thing']
        super().__init__(**kwargs)

    def on_parent(self, *args):
        if self.parent:
            self._board = self.parent.board
        else:
            if not hasattr(self, '_board'):
                return
            for canvas in (
                    self._board.pawnlayout.canvas.before,
                    self._board.pawnlayout.canvas.after,
                    self._board.pawnlayout.canvas
            ):
                if self.group in canvas.children:
                    canvas.remove(self.group)
            del self._board

    def relocate(self, *args):
        if not self.remote.exists:
            return
        try:
            location = self.board.arrow[self.loc_name][self.next_loc_name]
        except KeyError:
            location = self.board.spot[self.loc_name]
        if location != self.parent:
            self.parent.remove_widget(self)
            location.add_widget(self)
    _trigger_relocate = trigger(relocate)

    def upd_loc_name(self, *args):
        try:
            self.loc_name = self.remote['location']
            self._trigger_relocate()
        except ValueError:
            if self.name in self.board.pawn:
                self.board.rm_pawn(self.name)
            elif self.parent:
                self.parent.remove_widget(self)
    _trigger_upd_loc_name = trigger(upd_loc_name)

    def upd_next_loc_name(self, *args):
        self.next_loc_name = self.remote.get('next_location', None)
        self._trigger_relocate()
    _trigger_upd_next_loc_name = trigger(upd_next_loc_name)

    def listen_loc(self, *args):
        self.remote.listener(
            fun=self._trigger_upd_loc_name,
            stat='location'
        )
        self.remote.listener(
            fun=self._trigger_upd_next_loc_name,
            stat='next_location'
        )
    _trigger_listen_loc = trigger(listen_loc)

    def unlisten_loc(self, *args):
        self.remote.unlisten(
            fun=self._trigger_upd_loc_name,
            stat='location'
        )
        self.remote.unlisten(
            fun=self._trigger_upd_next_loc_name,
            stat='next_location'
        )
    _trigger_unlisten_loc = trigger(unlisten_loc)

    def on_remote(self, *args):
        """In addition to the usual behavior from
        :class:`remote.MirrorMapping`, copy ``loc_name`` from remote's
        'location', ``next_loc_name`` from remote's 'next_location',
        and arrange to keep them both up to date.

        """
        super().on_remote(*args)
        self.loc_name = self.remote['location']
        self.next_loc_name = self.remote.get('next_location', None)
        self._trigger_listen_loc()

    def push_location(self, *args):
        self.remote['location'] = self.loc_name
    _trigger_push_location = trigger(push_location)

    def push_next_location(self, *args):
        self.remote['next_location'] = self.next_loc_name
    _trigger_push_next_location = trigger(push_next_location)

    def add_widget(self, pawn, index=0, canvas='after'):
        """Apart from the normal behavior, bind my ``center`` so that the
        child's lower left corner will always be there, so long as
        it's my child.

        """
        super().add_widget(pawn, index, canvas)
        pawn.pos = self.center
        self.bind(center=pawn.setter('pos'))

    def remove_widget(self, pawn):
        """Unbind my ``center`` from the child before removing it."""
        if pawn not in self.children:
            raise ValueError("Not my child")
        self.unbind(center=pawn.setter('pos'))
        super().remove_widget(pawn)

    def on_touch_move(self, touch):
        """Move with the touch if I'm grabbed."""
        if not (self.selected and self.hit):
            return False
        if not hasattr(self, '_unlistened'):
            self.parent._unbind_trigger_pospawn(self)
            self._unlistened = True
        self.center = touch.pos
        return True

    def on_touch_up(self, touch):
        """See if I've been dropped on a :class:`Spot`. If so, command the
        underlying :class:`Thing` to either travel there or teleport
        there. Otherwise, snap back to my present location.

        """
        if not self.selected:
            return False
        if hasattr(self, '_unlistened'):
            self.parent._bind_trigger_pospawn(self)
            del self._unlistened
        for spot in self.board.spot.values():
            if self.collide_widget(spot) and spot.name != self.loc_name:
                Logger.debug(
                    "pawn: {} will go from {} to {}".format(
                        self.name,
                        self.loc_name,
                        spot.name
                    )
                )
                new_spot = spot
                break
        else:
            parent = self.parent
            parent.remove_widget(self)
            parent.add_widget(self)
            return True

        myplace = self.loc_name
        theirplace = new_spot.name
        if myplace != theirplace:
            if hasattr(self, '_start'):
                del self._start
            if self.travel_on_drop:
                self.thing.travel_to(new_spot.name)
            else:
                self.loc_name = new_spot.name
        self.parent.remove_widget(self)
        new_spot.add_widget(self)
        return True

    def __repr__(self):
        """Give my ``thing``'s name and its location's name."""
        return '{}-in-{}'.format(
            self.name,
            self.loc_name
        )
