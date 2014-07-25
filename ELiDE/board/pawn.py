# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from kivy.properties import (
    BooleanProperty,
    ObjectProperty
)
from kivy.clock import Clock
from ELiDE.texturestack import ImageStack


"""Widget representing things that move about from place to place."""


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
    where_upon = ObjectProperty()
    engine = ObjectProperty()
    travel_on_drop = BooleanProperty(False)

    def __init__(self, **kwargs):
        """Arrange to update my textures and my position whenever the relevant
        data change.

        """
        self._trigger_updata = Clock.create_trigger(self._updata)
        self._trigger_recenter = Clock.create_trigger(self._recenter)
        super().__init__(**kwargs)
        def on_time(*args):
            self._trigger_updata()
        on_time.__name__ = self.thing.name + "_trigger_updata"
        self.engine.on_time(on_time)
        self.bind(where_upon=self._trigger_recenter)
        self._trigger_updata()

    def handle_time(self, *args):
        self._trigger_updata()

    def _updata(self, *args):
        self.paths = self.thing["_image_paths"] if "_image_paths" in self.thing else []
        self.stackhs = self.thing["_stacking_heights"] if "_stacking_heights" in self.thing else []

    def _recenter(self):
        self.pos = self.where_upon.center

    def dropped(self, x, y, button, modifiers):
        spotto = None
        for spot in self.board.spots():
            if (
                    self.window_left < spot.x and
                    spot.x < self.window_right and
                    self.window_bot < spot.y and
                    spot.y < self.window_top
            ):
                spotto = spot
                break
        if spotto is not None:
            if self.travel_on_drop:
                self.thing.travel_to(spotto.place)
            else:
                self.thing["location"] = spotto.place.name

    def check_spot_collision(self):
        for spot in self.board.spots():
            if self.collide_widget(spot):
                return spot

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            touch.ud["pawn"] = self
            touch.grab(self)
            return self

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            self.center = touch.pos
            return self

    def on_touch_up(self, touch):
        if 'pawn' in touch.ud:
            del touch.ud['pawn']
        if touch.grab_current is self:
            touch.ungrab(self)
            new_spot = None
            for spot in self.board.spots():
                if self.collide_widget(spot):
                    new_spot = spot

            if new_spot:
                myplace = self.thing["location"]
                theirplace = new_spot.place.name
                if myplace != theirplace:
                    self.thing.travel_to(new_spot.place.name)
            self._handle_time(*self.engine.time)
            return self
        return super().on_touch_up(touch)
