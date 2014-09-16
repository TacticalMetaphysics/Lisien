# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Widget representing things that move about from place to place."""
from kivy.properties import (
    BooleanProperty,
    ObjectProperty,
    AliasProperty
)
from kivy.clock import Clock
from ELiDE.texturestack import ImageStack


class Pawn(ImageStack):
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
    board = ObjectProperty()
    thing = ObjectProperty()
    _touch = ObjectProperty(None, allownone=True)
    travel_on_drop = BooleanProperty(False)
    engine = AliasProperty(
        lambda self: self.board.engine if self.board else None,
        lambda self, v: None,
        bind=('board',)
    )

    def __init__(self, **kwargs):
        """Arrange to update my textures and my position whenever the relevant
        data change.

        """
        self._trigger_update = Clock.create_trigger(self._update)
        super().__init__(**kwargs)

    def _update(self, *args):
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

    def add_widget(self, pawn, index=0, canvas='after'):
        super().add_widget(pawn, index, canvas)
        pawn.pos = self.center
        self.bind(center=pawn.setter('pos'))

    def remove_widget(self, pawn):
        self.unbind(center=pawn.setter('pos'))
        super().remove_widget(pawn)

    def on_touch_down(self, touch):
        """If the touch hits me, grab it, and put myself in its userdict.

        """
        if (
                not self.collide_point(*touch.pos)
        ):
            return
        self.board.layout.grabbed = self
        touch.grab(self)
        return self

    def on_touch_move(self, touch):
        """Move with the touch if I'm grabbed."""
        # TODO
        pass

    def on_touch_up(self, touch):
        """See if I've been dropped on a :class:`Spot`. If so, command the
        underlying :class:`Thing` to either travel there or teleport
        there.

        """
        if self.board.layout.grabbed is self:
            new_spot = None
            for spot in self.board.spot.values():
                if self.collide_widget(spot):
                    new_spot = spot

            if new_spot:
                myplace = self.thing["location"]
                theirplace = new_spot.place.name
                if myplace != theirplace:
                    self.thing.travel_to(new_spot.place.name)
            return self
        return super().on_touch_up(touch)
