# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE.gui.kivybits import (
    SaveableWidgetMetaclass)
from kivy.properties import (
    NumericProperty,
    ObjectProperty)
from kivy.uix.scatter import Scatter


"""Widgets to represent places. Pawns move around on top of these."""


class Spot(Scatter):
    __metaclass__ = SaveableWidgetMetaclass
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
# TODO: query tool for places in facades that do not *currently*
# correspond to any place in any character
    tables = [
        ("spot", {
            "columns": {
                "observer": "text not null default 'Omniscient'",
                "host": "text not null default 'Physical'",
                "place": "text not null",
                "layer": "integer not null default 0",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "img": "text default 'default_spot'",
                "interactive": "boolean default 1"},
            "primary_key": (
                "observer", "host", "place",
                "layer", "branch", "tick"),
            "foreign_keys": {
                "observer, host": ("board", "observer, observed"),
                "img": ("img", "name")}}),
        ("spot_coords", {
            "columns": {
                "observer": "text not null default 'Omniscient'",
                "host": "text not null default 'Physical'",
                "place": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "x": "integer not null",
                "y": "integer not null"},
            "primary_key": (
                "observer", "host", "place", "branch", "tick"),
            "foreign_keys": {
                "observer, host, place": (
                    "spot", "observer, host, place")}})]
    place = ObjectProperty()
    board = ObjectProperty()
    completedness = NumericProperty(0)
    cheatx = NumericProperty(0)
    cheaty = NumericProperty(0)

    def __init__(self, **kwargs):
        super(Spot, self).__init__(**kwargs)
        self.board.spotdict[unicode(self.place)] = self

    def __str__(self):
        """Return the name of my :class:`Place`."""
        return str(self.place)

    def __unicode__(self):
        """Return the name of my :class:`Place`."""
        return unicode(self.place)

    def on_board(self, i, v):
        if v is None:
            return
        v.facade.closet.branch_listeners.append(self.repos)
        v.facade.closet.tick_listeners.append(self.repos)
        self.repos()

    def repos(self, *args):
        """Update my pos to match the database. Keep respecting my transform
        as I can."""
        coords = self.get_coords()
        self.pos = coords

    def set_img(self, img, layer, branch=None, tick_from=None):
        if branch is None:
            branch = self.board.facade.closet.branch
        if tick_from is None:
            tick_from = self.board.facade.closet.tick
        imagery = self.board.facade.closet.skeleton["spot_img"][
            unicode(self.board.dimension)][unicode(self.place)]
        if layer not in imagery:
            imagery[layer] = []
        il = imagery[layer]
        if branch not in il:
            il[branch] = []
        il[branch][tick_from] = self.bonetypes["spot_img"](
            dimension=unicode(self.board),
            place=unicode(self.place),
            branch=branch,
            tick_from=tick_from,
            img=unicode(img))

    def sanetime(self, branch, tick):
        return self.board.facade.sanetime(branch, tick)

    def get_bone(self, layer=0, branch=None, tick=None):
        (branch, tick) = self.sanetime(branch, tick)
        return self.board.facade.closet.skeleton[u"spot"][
            unicode(self.board.facade.observer)][
            unicode(self.board.host)][
            unicode(self.place)][layer][branch].value_during(tick)

    def set_bone(self, bone):
        self.board.skelset(
            self.board.facade.closet.skeleton[u"spot"],
            "place",
            bone)

    def get_coord_bone(self, branch=None, tick=None):
        (branch, tick) = self.sanetime(branch, tick)
        return self.board.facade.closet.skeleton[u"spot_coords"][
            unicode(self.board.facade.observer)][
            unicode(self.board.host)][unicode(self.place)][
            branch].value_during(tick)

    def set_coord_bone(self, bone):
        self.board.skelset(
            self.board.facade.closet.skeleton[u"spot_coords"],
            "place",
            bone)

    def get_coords(self, branch=None, tick=None, default=None):
        """Return a pair of coordinates for where I should be on my board,
        either now, or at the given point in time.

        """
        try:
            bone = self.get_coord_bone(branch, tick)
            if bone is None:
                return None
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
        self.set_coord_bone(bone._replace(
            x=x, y=y,
            branch=branch, tick=tick))

    def new_branch(self, parent, branch=None, tick=None):
        """Copy all the stuff from the parent to the child branch as of the
        given tick.

        """
        (branch, tick) = self.sanetime(branch, tick)
        prev = None
        started = False
        for skel in (self.skel[parent], self.coords[parent]):
            for bone in skel.iterbones():
                if bone.tick >= tick:
                    b2 = bone._replace(branch=branch)
                    self.set_bone(b2)
                    if (
                            not started and prev is not None and
                            bone.tick > tick and prev.tick < tick):
                        b3 = prev._replace(
                            branch=branch,
                            tick=tick)
                        self.set_bone(b3)
                    started = True
                    prev = bone

    def collide_point(self, x, y):
        return self.ids.pile.collide_point(x, y)

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            self.set_coords(*self.pos)
        return super(Spot, self).on_touch_up(touch)
