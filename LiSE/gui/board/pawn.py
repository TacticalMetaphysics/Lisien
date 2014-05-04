# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from gamepiece import GamePiece
from kivy.properties import (
    AliasProperty,
    ObjectProperty,
    NumericProperty,
    StringProperty,
    ReferenceListProperty)
from kivy.logger import Logger
from kivy.clock import Clock


"""Widget representing things that move about from place to place."""


class Pawn(GamePiece):
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
    bone = ObjectProperty()
    branch = NumericProperty()
    tick = NumericProperty()
    time = ReferenceListProperty(branch, tick)
    _touch = ObjectProperty(None, allownone=True)
    where_upon = ObjectProperty()
    name = StringProperty()
    closet = ObjectProperty()
    graphic_name = StringProperty()
    locskel = AliasProperty(
        lambda self: self.thing.character.closet.skeleton[u'thing_loc'][
            unicode(self.thing.character)][unicode(self.thing)],
        lambda self, v: None,
        bind=('thing',))

    def __init__(self, **kwargs):
        """Arrange to update my textures and my position whenever the relevant
        data change.

        """
        super(Pawn, self).__init__(**kwargs)
        self.closet.register_time_listener(self.handle_time)
        self.board.pawndict[unicode(self.thing)] = self
        self.handle_time(*self.closet.time)

    def __str__(self):
        return str(self.thing)

    def __unicode__(self):
        return unicode(self.thing)

    def reposskel(self, *args):
        self.repos()

    def handle_time(self, b, t):
        try:
            self.time = (b, t)
        except KeyError:
            Clock.schedule_once(lambda dt: self.handle_time(b, t), 0)
            return
        self.repos()

    def on_board(self, *args):
        self.repos()

    def get_loc_bone(self, branch=None, tick=None):
        return self.thing.get_bone(branch, tick)

    def dropped(self, x, y, button, modifiers):
        """When dropped on a spot, if my :class:`Thing` doesn't have anything
        else to do, make it journey there.

        If it DOES have anything else to do, make the journey in
        another branch.

        """
        spotto = None
        for spot in self.board.spots:
            if (
                    self.window_left < spot.x and
                    spot.x < self.window_right and
                    self.window_bot < spot.y and
                    spot.y < self.window_top):
                spotto = spot
                break
        if spotto is not None:
            self.thing.journey_to(spotto.place)
        self.drag_offset_x = 0
        self.drag_offset_y = 0

    def repos(self, *args):
        """Recalculate and reassign my position, based on the apparent
        position of whatever widget I am located on--a :class:`Spot`
        or an :class:`Arrow`.

        """
        (b, t) = self.time
        thingloc = self.thing.get_location(
            self.board.facade.observer, b, t)
        if thingloc is None:
            return
        try:
            new_where_upon = self.board.arrowdict[unicode(thingloc)]
        except KeyError:
            new_where_upon = self.board.spotdict[unicode(thingloc)]
        if self.where_upon is not None:
            self.where_upon.pawns_here.remove(self)
        self.where_upon = new_where_upon
        self.where_upon.pawns_here.append(self)

    def check_spot_collision(self):
        for spot in self.board.spotlayout.children:
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
            new_spot = self.check_spot_collision()

            if new_spot:
                myplace = self.thing.location
                theirplace = new_spot.place
                if myplace != theirplace:
                    self.locskel.unregister_del_listener(self.reposskel)
                    self.thing.journey_to(new_spot.place)
                    self.locskel.register_del_listener(self.reposskel)
            self.repos()
            return self
        return super(Pawn, self).on_touch_up(touch)
