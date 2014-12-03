# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Widget representing things that move about from place to place."""
from kivy.properties import (
    BooleanProperty,
    ObjectProperty,
    NumericProperty,
    ReferenceListProperty
)
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.logger import Logger
from .pawnspot import PawnSpot


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
    thing = ObjectProperty()
    _touch_ox_diff = NumericProperty()
    _touch_oy_diff = NumericProperty()
    _touch_opos_diff = ReferenceListProperty(_touch_ox_diff, _touch_oy_diff)
    travel_on_drop = BooleanProperty(False)
    loc_name = ObjectProperty()
    next_loc_name = ObjectProperty(None, allownone=True)

    def __init__(self, **kwargs):
        self._trigger_upd_remote_location = Clock.create_trigger(
            self.upd_remote_location
        )
        self._trigger_upd_remote_next_location = Clock.create_trigger(
            self.upd_remote_next_location
        )
        super().__init__(**kwargs)

    def __repr__(self):
        """Give my ``thing``'s name and its location's name."""
        return '{}-in-{}'.format(
            self.name,
            self.loc_name
        )

    def on_name(self, *args):
        """Reindex myself in my board's pawn dict, for when my thing gets
        renamed.

        """
        if hasattr(self, '_oldname'):
            del self.board.pawn[self._oldname]
        self.board.pawn[self.name] = self
        self._oldname = self.name

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
        if not self.selected:
            return False
        self.center = touch.pos
        return True

    def on_touch_up(self, touch):
        """See if I've been dropped on a :class:`Spot`. If so, command the
        underlying :class:`Thing` to either travel there or teleport
        there.

        """
        if not self.selected:
            return False
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
            return True

        myplace = self.loc_name
        theirplace = new_spot.name
        if myplace != theirplace:
            if self.travel_on_drop:
                self.thing.travel_to(new_spot.name)
            else:
                self.loc_name = new_spot.name
        return True

    def on_loc_name(self, *args):
        """Move myself to the widget representing my new location."""
        if (
                (
                    hasattr(self.parent, 'place') and
                    self.parent.name != self.loc_name
                ) or (
                    hasattr(self.parent, 'origin') and
                    (
                        self.parent.origin.name !=
                        self.loc_name or
                        self.parent.destination.name !=
                        self.loc_name
                    )
                )
        ):
            try:
                whereat = self.board.arrow[
                    self.loc_name
                    ][
                        self.next_loc_name
                    ]
            except KeyError:
                whereat = self.board.spot[self.loc_name]
            self.parent.remove_widget(self)
            whereat.add_widget(self)

    def on_remote_map(self, *args):
        if not PawnSpot.on_remote_map(self, *args):
            return
        self.loc_name = self.remote_map['location']
        self.next_loc_name = self.remote_map['next_location']

        @self.remote_map.listener(key='location')
        def listen_loc(k, v):
            self.unbind(loc_name=self._trigger_upd_remote_location)
            self.loc_name = v
            self.bind(loc_name=self._trigger_upd_remote_location)

        @self.remote_map.listener(key='next_location')
        def listen_next_loc(k, v):
            self.unbind(next_loc_name=self._trigger_upd_remote_next_location)
            self.next_loc_name = v
            self.bind(next_loc_name=self._trigger_upd_remote_next_location)

        self.bind(
            loc_name=self._trigger_upd_remote_location,
            next_loc_name=self._trigger_upd_remote_next_location,
        )

        return True

    def upd_remote_location(self, *args):
        self.remote_map['location'] = self.loc_name

    def upd_remote_next_location(self, *args):
        self.remote_map['next_location'] = self.next_loc_name


kv = """
<Pawn>:
    remote_map: EntityRemoteMapping(self.thing) if self.thing else None
"""
Builder.load_string(kv)
