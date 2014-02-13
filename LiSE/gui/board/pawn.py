# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from gamepiece import GamePiece
from kivy.properties import (
    AliasProperty,
    ObjectProperty,
    NumericProperty,
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
    demands = ["board"]
    tables = [
        ("pawn", {
            "columns": {
                "observer": "text not null default 'Omniscient'",
                "observed": "text not null default 'Physical'",
                "host": "text not null default 'Physical'",
                "thing": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "graphic": "text not null",
                "interactive": "boolean default 1"},
            "primary_key": (
                "observer", "observed", "host", "thing",
                "branch", "tick"),
            "foreign_keys": {
                "observer, observed, host": (
                    "board", "observer, observed, host"),
                "observed, host, thing": (
                    "thing", "character, host, name"),
                "graphic": ("graphic", "name")}})]
    board = ObjectProperty()
    thing = ObjectProperty()
    bone = ObjectProperty()
    branch = NumericProperty()
    tick = NumericProperty()
    time = ReferenceListProperty(branch, tick)
    _touch = ObjectProperty(None, allownone=True)
    where_upon = ObjectProperty()
    name = AliasProperty(
        lambda self: self.bone.graphic if self.bone else '',
        lambda self, v: None,
        bind=('bone',))
    character = AliasProperty(
        lambda self: self.thing.character,
        lambda self, v: None,
        bind=('thing',))
    closet = AliasProperty(
        lambda self: self.thing.character.closet,
        lambda self, v: None,
        bind=('thing',))
    locskel = AliasProperty(
        lambda self: self.thing.character.closet.skeleton[u'thing_loc'][
            unicode(self.thing.character)][unicode(self.thing)],
        lambda self, v: None,
        bind=('thing',))
    graphic_name = AliasProperty(
        lambda self: self.bone.graphic if self.bone else '',
        lambda self, v: None,
        bind=('bone',))

    def __init__(self, **kwargs):
        """Arrange to update my textures and my position whenever the relevant
        data change.

        """
        super(Pawn, self).__init__(**kwargs)
        self.bone = self.get_pawn_bone()
        self.closet.register_time_listener(self.handle_time)
        self.board.pawndict[unicode(self.thing)] = self
        self.handle_time(*self.closet.time)
        self.locskel.register_listener(self.reposskel)

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

    def get_pawn_bone(self, branch=None, tick=None):
        (branch, tick) = self.board.host.sanetime(branch, tick)
        return self.board.host.closet.skeleton[u"pawn"][
            unicode(self.board.facade.observer)][
            unicode(self.board.facade.observed)][
            unicode(self.thing.host)][
            unicode(self.thing)][branch].value_during(tick)

    def new_branch(self, parent, branch, tick):
        """Update my part of the :class:`Skeleton` to have this new branch in
        it. Where there is room, fill it with data from the parent branch."""
        prev = None
        started = False
        imagery = self.board.host.closet.skeleton[u"pawn"][
            unicode(self.board.facade.observer)][
            unicode(self.thing.character)][
            unicode(self.board.host)][
            unicode(self.thing)]
        for tick_from in imagery[parent]:
            if tick_from >= tick:
                bone2 = imagery[parent][tick_from]._replace(
                    branch=branch)
                yield bone2
                if (
                        not started and prev is not None and
                        tick_from > tick and prev < tick):
                    bone3 = imagery[parent][prev]._replace(
                        branch=branch, tick_from=tick_from)
                    yield bone3
                    started = True
                prev = tick_from

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
        if self.collide_point(touch.x, touch.y):
            touch.ud["pawn"] = self
            touch.grab(self)
            return True

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            self.center = touch.pos
            return True

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
            return True
        return super(Pawn, self).on_touch_up(touch)
