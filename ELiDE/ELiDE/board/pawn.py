# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Widget representing things that move about from place to place."""
from kivy.properties import (
    AliasProperty,
    BooleanProperty,
    ObjectProperty,
    NumericProperty,
    ReferenceListProperty
)
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
            self.bind(
                loc_name=self._trigger_relocate,
                next_loc_name=self._trigger_relocate
            )
            if self.remote:
                self._trigger_relocate()
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
            try:
                location = self.board.spot[self.loc_name]
            except KeyError:
                return
        if location != self.parent:
            self.parent.remove_widget(self)
            location.add_widget(self)
    _trigger_relocate = trigger(relocate)

    def on_remote(self, *args):
        super().on_remote(*args)
        if not self.remote or not self.remote.exists:
            return
        self.loc_name = self.remote['location']
        self.next_loc_name = self.remote.get('next_location', None)

    def finalize(self):
        super().finalize()
        self.bind(
            loc_name=self._trigger_push_location,
            next_loc_name=self._trigger_push_next_location
        )

    def unfinalize(self):
        super().unfinalize()
        self.unbind(
            loc_name=self._trigger_push_location,
            next_loc_name=self._trigger_push_next_location
        )

    def pull_from_remote(self):
        relocate = False
        if self.loc_name != self.remote['location']:
            self.loc_name = self.remote['location']  # aliasing? could be trouble
            relocate = True
        if self.next_loc_name != self.remote['next_location']:
            self.next_loc_name = self.remote['next_location']
            relocate = True
        if relocate:
            self.relocate()

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
        if touch.grab_current != self:
            return False
        if hasattr(self.parent, 'place') and \
           not hasattr(self, '_pospawn_unbound'):
            self.parent.unbind_trigger_pospawn(self)
            self._pospawn_bound = True
        self.center = touch.pos
        return True

    def on_touch_up(self, touch):
        """See if I've been dropped on a :class:`Spot`. If so, command the
        underlying :class:`Thing` to either travel there or teleport
        there. Otherwise, snap back to my present location.

        """
        if touch.grab_current != self:
            return False
        if hasattr(self.parent, 'place') and hasattr(self, '_pospawn_unbound'):
            self.parent.bind_trigger_pospawn(self)
            del self._pospawn_unbound
        for spot in self.board.spot.values():
            if self.collide_widget(spot) and spot.name != self.loc_name:
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
