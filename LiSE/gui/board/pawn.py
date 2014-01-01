# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE.gui.kivybits import (
    SaveableWidgetMetaclass)
from kivy.uix.widget import Widget
from kivy.properties import (
    ObjectProperty)
from kivy.graphics import Rectangle


"""Widget representing things that move about from place to place."""


class Pawn(Widget):
    __metaclass__ = SaveableWidgetMetaclass
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
                "layer": "integer not null default 0",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "img": "text not null default 'default_pawn'",
                "interactive": "boolean default 1"},
            "primary_key": (
                "observer", "observed", "host", "thing",
                "layer", "branch", "tick"),
            "foreign_keys": {
                "observer, observed, host": (
                    "board", "observer, observed, host"),
                "observed, host, thing": (
                    "thing", "character, host, name"),
                "img": ("img", "name")}})]
    board = ObjectProperty()
    thing = ObjectProperty()
    where_upon = ObjectProperty(None)

    def __init__(self, **kwargs):
        """Arrange to update my textures and my position whenever the relevant
data change.

The relevant data are

* The branch and tick, being the two measures of game-time.
* The location data for the Thing I represent, in the table thing_location"""
        def reposskel(*args):
            self.repos()
        kwargs['size_hint'] = (None, None)
        super(Pawn, self).__init__(**kwargs)
        self.board.pawndict[unicode(self.thing)] = self

        skel = self.board.facade.closet.skeleton[u"thing_loc"][
            unicode(self.thing.character)][unicode(self.thing)]
        skel.register_set_listener(reposskel)
        skel.register_del_listener(reposskel)
        self.retex()

    def __str__(self):
        return str(self.thing)

    def __unicode__(self):
        return unicode(self.thing)

    def retex(self, branch=None, tick=None):
        """Clear my canvas and rebuild it with textures from the closet."""
        self.canvas.clear()
        (branch, tick) = self.board.host.sanetime(branch, tick)
        gettex = self.board.host.closet.get_texture
        try:
            skel = self.board.host.closet.skeleton[u"pawn_img"][
                unicode(self.board.observer)][unicode(self.board.observed)][
                unicode(self.board.host)][unicode(self.thing)]
        except KeyError:
            raise NotImplementedError(
                "TODO: Pick the most similar graphic and use it.")
        for layer in skel:
            texbone = skel[layer][branch].value_during(tick)
            texture = gettex(texbone.img)
            self.canvas.add(Rectangle(
                texture=texture,
                size=texture.size,
                pos=self.pos))

    def on_board(self, i, v):
        v.facade.closet.register_time_listener(self.repos)
        try:
            whereami = v.arrowdict[unicode(self.thing.location)]
        except KeyError:
            whereami = v.spotdict[unicode(self.thing.location)]
        whereami.pawns_here.append(self)

    def get_loc_bone(self, branch=None, tick=None):
        return self.thing.get_bone(branch, tick)

    def get_pawn_bone(self, layer=0, branch=None, tick=None):
        (branch, tick) = self.board.host.sanetime(branch, tick)
        return self.board.host.closet.skeleton[u"pawn"][
            unicode(self.board.facade.observer)][
            unicode(self.thing.character)][
            unicode(self.thing.host)][
            unicode(self.thing)][
            layer][branch].value_during(tick)

    def get_img_bone(self, layer=0, branch=None, tick=None):
        (branch, tick) = self.board.host.sanetime(branch, tick)
        pawnbone = self.get_pawn_bone(layer, branch, tick)
        return self.board.host.closet.skeleton[u"img"][pawnbone.img]

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
        for layer in imagery:
            for tick_from in imagery[layer][parent]:
                if tick_from >= tick:
                    bone2 = imagery[layer][parent][tick_from]._replace(
                        branch=branch)
                    yield bone2
                    if (
                            not started and prev is not None and
                            tick_from > tick and prev < tick):
                        bone3 = imagery[layer][parent][prev]._replace(
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

    def repos(self, b=None, t=None):
        """Recalculate and reassign my position, based on the apparent
        position of whatever widget I am located on--a :class:`Spot`
        or an :class:`Arrow`.

        """
        thingloc = self.thing.get_location(self.board.facade.observer, b, t)
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
        for spot in self.board.spot_layout.children:
            if self.collide_widget(spot):
                return spot

    def on_touch_down(self, touch):
        r = self.ids.pile.on_touch_down(touch)
        if r:
            touch.ud["pawn"] = self
            touch.grab(self)
        return r

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            self.center = touch.pos
            return True

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            new_spot = self.check_spot_collision()

            if new_spot:
                myplace = self.thing.location
                theirplace = new_spot.place
                if myplace != theirplace:
                    self.thing.journey_to(new_spot.place)
            self.repos()
            return True
        return super(Pawn, self).on_touch_up(touch)
