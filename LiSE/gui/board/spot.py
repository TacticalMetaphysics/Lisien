# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from gamepiece import GamePiece
from kivy.properties import (
    ListProperty,
    ObjectProperty)
from kivy.clock import Clock
from kivy.logger import Logger


"""Widgets to represent places. Pawns move around on top of these."""


class Spot(GamePiece):
    """The icon that represents a Place.

    The Spot is located on the Board that represents the same
    Dimension that the underlying Place is in. Its coordinates are
    relative to its Board, not necessarily the window the Board is in.

    """
    demands = ["board"]
    provides = ["place"]
    postlude = [
        "CREATE VIEW place AS "
        "SELECT thing.host AS host, thing_loc.location AS place, "
        "thing_loc.branch AS branch, thing_loc.tick AS tick "
        "FROM thing "
        "JOIN thing_loc ON "
        "thing.character=thing_loc.character AND "
        "thing.name=thing_loc.name "
        "WHERE thing_loc.location NOT IN "
        "(SELECT name FROM portal UNION SELECT name FROM thing) "
        "UNION "
        "SELECT portal.host AS host, portal_loc.origin AS place, "
        "portal_loc.branch AS branch, portal_loc.tick AS tick "
        "FROM portal JOIN portal_loc ON "
        "portal.character=portal_loc.character AND "
        "portal.name=portal_loc.name "
        "UNION "
        "SELECT portal.host AS host, portal_loc.destination AS place, "
        "portal_loc.branch AS branch, portal_loc.tick AS tick "
        "FROM portal JOIN portal_loc ON "
        "portal.character=portal_loc.character AND "
        "portal.name=portal_loc.name "
        "UNION "
        "SELECT spot.host AS host, spot.place AS place, "
        "spot.branch AS branch, spot.tick AS tick FROM spot "
        "WHERE observer='Omniscient';",
        "CREATE VIEW place_facade AS "
        "SELECT thing.host AS host, thing_loc_facade.location AS place, "
        "thing_loc_facade.branch AS branch, thing_loc_facade.tick AS tick "
        "FROM thing JOIN thing_loc_facade ON "
        "thing.character=thing_loc_facade.observed AND "
        "thing.name=thing_loc_facade.name "
        "UNION "
        "SELECT portal.host AS host, portal_loc_facade.origin AS place, "
        "portal_loc_facade.branch AS branch, portal_loc_facade.tick AS tick "
        "FROM portal JOIN portal_loc_facade ON "
        "portal.character=portal_loc_facade.observed AND "
        "portal.name=portal_loc_facade.name "
        "UNION "
        "SELECT portal.host AS host, portal_loc_facade.destination AS place, "
        "portal_loc_facade.branch AS branch, portal_loc_facade.tick AS tick "
        "FROM portal JOIN portal_loc_facade ON "
        "portal.character=portal_loc_facade.observed AND "
        "portal.name=portal_loc_facade.name "
        "UNION "
        "SELECT spot.host AS host, spot.place AS place, "
        "spot.branch AS branch, spot.tick AS tick "
        "FROM spot WHERE observer<>'Omniscient';"]
    tables = [
        ("spot", {
            "columns": {
                "observer": "text not null default 'Omniscient'",
                "host": "text not null default 'Physical'",
                "place": "text not null",
                "layer": "integer not null default 0",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "graphic": "text not null default 'default_spot'",
                "interactive": "boolean default 1"},
            "primary_key": (
                "observer", "host", "place",
                "branch", "tick"),
            "foreign_keys": {
                "observer, host": ("board", "observer, observed"),
                "graphic": ("graphic", "name")}}),
        ("spot_coords", {
            "columns": {
                "observer": "text not null default 'Omniscient'",
                "host": "text not null default 'Physical'",
                "place": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "x": "float not null",
                "y": "float not null"},
            "primary_key": (
                "observer", "host", "place", "branch", "tick"),
            "foreign_keys": {
                "observer, host, place": (
                    "spot", "observer, host, place")}})]
    place = ObjectProperty()
    board = ObjectProperty()
    bone = ObjectProperty()
    _touch = ObjectProperty(None, allownone=True)
    pawns_here = ListProperty([])

    def __init__(self, **kwargs):
        if 'board' in kwargs and 'closet' not in kwargs:
            kwargs['closet'] = kwargs['board'].host.closet
        kwargs['bone'] = kwargs['closet'].skeleton[u'spot'][
            unicode(kwargs['board'].facade.observer)][
            unicode(kwargs['place'].character)][
            unicode(kwargs['place'])][
            kwargs['closet'].branch].value_during(
            kwargs['closet'].tick)
        kwargs['graphic_name'] = kwargs['bone'].graphic
        kwargs['imgs'] = kwargs['closet'].get_game_piece(
            kwargs['bone'].graphic).imgs
        self._trigger_move_to_touch = Clock.create_trigger(
            self._move_to_touch)
        super(Spot, self).__init__(**kwargs)
        self.closet.register_time_listener(self.handle_time)
        self.board.spotdict[unicode(self.place)] = self
        self.bind(
            pawns_here=self.upd_pawns_here,
            center=self.upd_pawns_here)

    def __str__(self):
        """Return the name of my :class:`Place`."""
        return str(self.place)

    def __unicode__(self):
        """Return the name of my :class:`Place`."""
        return unicode(self.place)

    def upd_texs(self, *args):
        super(Spot, self).upd_texs(*args)

    def handle_time(self, b, t):
        self.bone = self.get_bone(b, t)
        self.graphic_name = self.bone.graphic
        self.repos(b, t)

    def repos(self, b, t):
        if not self.graphic_bone:
            Clock.schedule_once(lambda dt: self.repos(b, t), 0)
            return
        bone = self.get_coord_bone(b, t)
        x = bone.x + self.graphic_bone.offset_x
        y = bone.y + self.graphic_bone.offset_y
        self.pos = (x, y)

    def upd_pawns_here(self, *args):
        for pawn in self.pawns_here:
            pawn.pos = self.center

    def sanetime(self, branch, tick):
        return self.board.facade.sanetime(branch, tick)

    def get_bone(self, branch=None, tick=None):
        (branch, tick) = self.sanetime(branch, tick)
        return self.closet.skeleton[u"spot"][
            unicode(self.board.facade.observer)][
            unicode(self.board.host)][
            unicode(self.place)][branch].value_during(tick)

    def get_coord_bone(self, branch=None, tick=None):
        (branch, tick) = self.sanetime(branch, tick)
        return self.board.facade.closet.skeleton[u"spot_coords"][
            unicode(self.board.facade.observer)][
            unicode(self.board.host)][unicode(self.place)][
            branch].value_during(tick)

    def get_coords(self, branch=None, tick=None, default=None):
        """Return a pair of coordinates for where I should be on my board,
        either now, or at the given point in time.

        """
        try:
            bone = self.get_coord_bone(branch, tick)
            if bone is None:
                raise KeyError
            else:
                return (bone.x, bone.y)
        except KeyError:
            if default is not None:
                self.set_coords(*default)
                return default

    def set_coords(self, x, y, branch=None, tick=None):
        """Set my coordinates on the :class:`Board`.

        Optional arguments may be used to set my coordinates as of
        some time other than "right now".

        """
        (branch, tick) = self.sanetime(branch, tick)
        bone = self.get_coord_bone(branch, tick)
        self.closet.set_bone(bone._replace(
            x=x, y=y,
            branch=branch, tick=tick))

    def set_center_coords(self, x, y, branch=None, tick=None):
        self.set_coords(x - self.width / 2, y - self.height / 2,
                        branch, tick)

    def new_branch(self, parent, branch=None, tick=None):
        """Copy all the stuff from the parent to the child branch as of the
        given tick.

        """
        (branch, tick) = self.sanetime(branch, tick)
        prev = None
        started = False
        for skel in (
                self.board.host.closet.skeleton[u"spot"][
                    unicode(self.board.facade.observer)][
                    unicode(self.place.character)][
                    unicode(self.place)],
                self.board.host.closet.skeleton[u"spot_coords"][
                    unicode(self.board.facade.observer)][
                    unicode(self.place.character)][
                    unicode(self.place)]):
            for bone in skel.iterbones():
                if bone.tick >= tick:
                    yield bone._replace(branch=branch)
                    if (
                            not started and prev is not None and
                            bone.tick > tick and prev.tick < tick):
                        yield prev._replace(
                            branch=branch,
                            tick=tick)
                    started = True
                    prev = bone

    def on_touch_down(self, touch):
        if touch.grab_current:
            return
        if not self.collide_point(*self.to_local(*touch.pos)):
            return
        if 'spot' in touch.ud:
            return
        touch.grab(self)
        touch.ud['spot'] = self
        self._touch = touch
        return True

    def on_touch_move(self, touch):
        if 'spot' in touch.ud:
            if touch.ud['spot'] is self and 'portaling' not in touch.ud:
                self._touch = touch
                self._trigger_move_to_touch()
            elif (not touch.ud['spot'].collide_point(
                    *touch.ud['spot'].to_local(*touch.pos)) and
                    self.collide_point(*self.to_local(*touch.pos))):
                touch.ud['spot'] = self
        elif self.collide_point(*self.to_local(*touch.pos)):
            touch.ud['spot'] = self

    def _move_to_touch(self, *args):
        self.center = self.to_local(*self._touch.pos)

    def on_touch_up(self, touch):
        if self._touch:
            self.set_coords(*self.pos)
        self._touch = None
        return super(Spot, self).on_touch_up(touch)

    def __repr__(self):
        return "{}@({},{})".format(self.place.name, self.x, self.y)
