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
from kivy.logger import Logger
from .pawnspot import PawnSpot


class Pawn(PawnSpot):
    """A token to represent something that moves about between places.

Pawns are graphical widgets made of one or more textures layered atop
one another. The textures are assumed to be 32x32 pixels.

Pawns represent Things in those cases where a Thing is located
directly in a Place or a Portal. The corresponding Pawn will appear
atop the Place's Spot or the Portal's Arrow.

If a Pawn is currently interactive, it may be dragged to a new Spot,
which has the effect of ordering its Thing to travel there. This takes
some amount of game time. Whenever the game-time changes, the Pawn
will update its position appropriately.

    """
    thing = ObjectProperty()
    _touch_ox_diff = NumericProperty()
    _touch_oy_diff = NumericProperty()
    _touch_opos_diff = ReferenceListProperty(_touch_ox_diff, _touch_oy_diff)
    travel_on_drop = BooleanProperty(False)

    def __init__(self, **kwargs):
        """Arrange to update my textures and my position whenever the relevant
        data change.

        """
        self._trigger_update = Clock.create_trigger(self._update)
        super().__init__(**kwargs)

    def __repr__(self):
        return '{}-in-{}'.format(self.thing.name, self.thing.location.name)

    def _update(self, *args):
        """Private use. Update my ``paths`` and ``stackhs`` with what's in my
        :class:`Thing`.

        """
        if '_image_paths' not in self.thing:
            self.thing['_image_paths'] = self._default_paths()
        if '_offxs' not in self.thing:
            self.thing['_offxs'] = self._default_offxs()
        if '_offys' not in self.thing:
            self.thing['_offys'] = self._default_offys()
        if '_stacking_heights' not in self.thing:
            self.thing['_stacking_heights'] = self._default_stackhs()
        if self.paths != self.thing["_image_paths"]:
            self.paths = self.thing["_image_paths"]
        if self.stackhs != self.thing["_stacking_heights"]:
            self.stackhs = self.thing["_stacking_heights"]
        if (
                (
                    hasattr(self.parent, 'place') and
                    self.parent.place.name != self.thing["location"]
                ) or (
                    hasattr(self.parent, 'origin') and
                    (
                        self.parent.origin.place.name !=
                        self.thing['location'] or
                        self.parent.destination.place.name !=
                        self.thing['next_location']
                    )
                )
        ):
            try:
                whereat = self.board.arrow[
                    self.thing["location"]
                    ][
                        self.thing["next_location"]
                    ]
            except KeyError:
                whereat = self.board.spot[self.thing["location"]]
            self.parent.remove_widget(self)
            whereat.add_widget(self)

    def _default_paths(self):
        return ['atlas://rltiles/base.atlas/unseen']

    def _default_offxs(self):
        return [0]

    def _default_offys(self):
        return [0]

    def _default_stackhs(self):
        return [0]

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
            if self.collide_widget(spot):
                Logger.debug(
                    "pawn: {} will go from {} to {}".format(
                        self.thing.name,
                        self.thing.location.name,
                        spot.place.name
                    )
                )
                new_spot = spot
                break
        else:
            return True

        myplace = self.thing["location"]
        theirplace = new_spot.place.name
        if myplace != theirplace:
            if self.travel_on_drop:
                self.thing.travel_to(new_spot.place.name)
            else:
                self.thing["location"] = new_spot.place.name
                self._update()
        return True
