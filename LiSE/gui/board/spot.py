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
    tables = [
        ("spot_img",
         {"dimension": "text not null default 'Physical'",
          "place": "text not null",
          "layer": "integer not null default 0",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "img": "text not null default 'default_spot'",
          "off_x": "integer not null default 0",
          "off_y": "integer not null default 0",
          "stacking_height": "integer not null default 0"},
         ("dimension", "place", "layer", "branch", "tick_from"),
         {"dimension": ("board", "dimension"),
          "img": ("img", "name")},
         []),
        ("spot_interactive",
         {"dimension": "text not null default 'Physical'",
          "place": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "tick_to": "integer default null"},
         ("dimension", "place", "branch", "tick_from"),
         {"dimension": ("board", "dimension")},
         []),
        ("spot_coords",
         {"dimension": "text not null default 'Physical'",
          "place": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "x": "integer not null default 50",
          "y": "integer not null default 50"},
         ("dimension", "place", "branch", "tick_from"),
         {"dimension": ("board", "dimension")},
         [])]
    place = ObjectProperty()
    board = ObjectProperty()
    coords = ObjectProperty()
    interactivity = ObjectProperty()
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
        self.completedness += 1

    def on_interactivity(self, i, v):
        """Count toward completion"""
        self.completedness += 1

    def on_coords(self, i, v):
        """Count toward completion"""
        self.completedness += 1

    def on_completedness(self, i, v):
        """If completed, trigger ``self.finalize``"""
        if v == 3:
            self.board.closet.branch_listeners.append(self.repos)
            self.board.closet.tick_listeners.append(self.repos)
            self.repos()

    def repos(self, *args):
        """Update my pos to match the database. Keep respecting my transform
        as I can."""
        oldtf = self.transform
        self.transform.identity()
        self.pos = self.get_coords()
        self.apply_transform(oldtf)

    def set_interactive(self, branch=None, tick_from=None, tick_to=None):
        """Declare that I am interactive from the one time to the other."""
        if branch is None:
            branch = self.board.closet.branch
        if tick_from is None:
            tick_from = self.board.closet.tick
        assert branch in self.interactivity, "Make a new branch first"
        self.board.closet.skeleton["spot_interactive"][
            unicode(self.board)][unicode(self.place)][branch][
            tick_from] = self.bonetype(
            dimension=unicode(self.board),
            place=unicode(self.place),
            branch=branch,
            tick_from=tick_from,
            tick_to=tick_to)
        self.upd_interactivity()

    def is_interactive(self, branch=None, tick=None):
        """Am I interactive? Either now, or at the given point in sim-time."""
        if branch is None:
            branch = self.board.closet.branch
        if tick is None:
            tick = self.board.closet.tick
        interactivity = self.closet.skeleton["spot_interactive"][
            unicode(self.board.dimension)][unicode(self.place)]
        if branch not in interactivity:
            return False
        r = interactivity.value_during(tick)
        return (r.tick_to is None or tick <= r.tick_to)

    def new_branch_interactivity(self, parent, branch, tick):
        """Copy interactivity from the parent branch to the child, starting
        from the tick.

        """
        prev = None
        started = False
        interactivity = self.board.closet.skeleton["spot_interactive"][
            unicode(self.board.dimension)][unicode(self.place)]
        for tick_from in interactivity[parent]:
            if tick_from >= tick:
                b2 = interactivity[parent][
                    tick_from]._replace(branch=branch)
                if branch not in interactivity:
                    interactivity[branch] = {}
                interactivity[branch][b2.tick_from] = b2
                if (
                        not started and prev is not None and
                        tick_from > tick and prev < tick):
                    b3 = interactivity[parent][prev].replace(
                        branch=branch, tick_from=tick)
                    interactivity[branch][b3.tick_from] = b3
                started = True
            prev = tick_from
        self.upd_interactivity()

    def set_img(self, img, layer, branch=None, tick_from=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick_from is None:
            tick_from = self.board.closet.tick
        imagery = self.board.closet.skeleton["spot_img"][
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

    def get_coord_bone(self, branch=None, tick=None):
        """Get a bone for coordinates, either now, or at the given point in
        time

        """
        if branch is None:
            branch = self.board.closet.branch
        if tick is None:
            tick = self.board.closet.tick
        return self.coords[branch].value_during(tick)

    def get_coords(self, branch=None, tick=None, default=None):
        """Return a pair of coordinates for where I should be on my board,
        either now, or at the given point in time.

        """
        try:
            bone = self.get_coord_bone(branch, tick)
        except KeyError:
            if default is not None:
                self.set_coords(*default)
                return default
        if bone is None:
            return None
        else:
            return (bone.x, bone.y)

    def set_coords(self, x, y, branch=None, tick_from=None):
        """Set my coordinates on the :class:`Board`.

        Optional arguments may be used to set my coordinates as of
        some time other than "right now".

        """
        if branch is None:
            branch = self.board.closet.branch
        if tick_from is None:
            tick_from = self.board.closet.tick
        self.board.closet.skeleton["spot_coords"][
            unicode(self.board)][unicode(self.place)][
            branch][tick_from] = self.bonetypes.spot_coords(
            dimension=unicode(self.board),
            place=unicode(self.place),
            branch=branch,
            tick_from=tick_from,
            x=x,
            y=y)

    def new_branch_coords(self, parent, branch, tick):
        """Copy coordinate data from the parent branch as of the given
        tick.

        """
        prev = None
        started = False
        coords = self.board.closet.skeleton["spot_coords"][
            unicode(self.board)][unicode(self.place)]
        for tick_from in coords[parent]:
            if tick_from >= tick:
                b2 = coords[parent][tick_from]._replace(branch=branch)
                if branch not in coords:
                    coords[branch] = {}
                coords[branch][b2.tick_from] = b2
                if (
                        not started and prev is not None and
                        tick_from > tick and prev < tick):
                    b3 = coords[branch][prev].replace(
                        branch=branch,
                        tick_from=tick)
                    coords[branch][b3.tick_from] = b3
                started = True
            prev = tick_from

    def new_branch(self, parent, branch, tick):
        """Copy all the stuff from the parent to the child branch as of the
        given tick.

        """
        self.new_branch_imagery(parent, branch, tick)
        self.new_branch_interactivity(parent, branch, tick)
        self.new_branch_coords(parent, branch, tick)

    def new_branch_imagery(self, parent, branch, tick):
        """Copy imagery data from the parent branch to the child, as of the
        given tick.

        """
        prev = None
        started = False
        imagery = self.board.closet.skeleton[u"spot_img"][
            unicode(self.board.dimension)][unicode(self.place)]
        for layer in imagery:
            for tick_from in imagery[layer][parent]:
                if tick_from >= tick:
                    b2 = imagery[layer][parent][
                        tick_from]._replace(branch=branch)
                    if branch not in imagery[layer]:
                        imagery[layer][branch] = {}
                    imagery[layer][branch][b2.tick_from] = b2
                    if (
                            not started and prev is not None and
                            tick_from > tick and prev < tick):
                        b3 = imagery[layer][parent][prev].replace(
                            branch=branch,
                            tick_from=tick)
                        imagery[layer][branch][b3.tick_from] = b3
                    started = True
                prev = tick_from

    def on_touch_up(self, touch):
        """If this is the end of a drag, set my coordinates to wherever I've
        been dragged to."""
        if touch.grab_current is self:
            self.set_coords(*self.pos)
        super(Spot, self).on_touch_up(touch)

    def collide_point(self, x, y):
        return self.ids.pile.collide_point(x, y)
