# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from gui.kivybits import SaveableWidgetMetaclass
from kivy.uix.scatter import Scatter
from kivy.uix.image import Image
from kivy.properties import (
    ListProperty,
    NumericProperty,
    ObjectProperty)
from util import get_bone_during


"""Widget representing things that move about from place to place."""


class PawnImage(Image):
    pawn = ObjectProperty()
    layer = NumericProperty()


class Pawn(Scatter):
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
    tables = [
        ("pawn_img",
         {"dimension": "text not null default 'Physical'",
          "thing": "text not null",
          "layer": "integer not null default 0",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "img": "text not null default 'default_pawn'"},
         ("dimension", "thing", "layer", "branch", "tick_from"),
         {"dimension, thing": ("thing_location", "dimension, name"),
          "img": ("img", "name")},
         ["layer>=0", "branch>=0", "tick_from>=0"]),
        ("pawn_interactive",
         {"dimension": "text not null default 'Physical'",
          "thing": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0"},
         ("dimension", "thing", "branch", "tick_from"),
         {"dimension, thing": ("thing_location", "dimension, name")},
         [])]
    imagery = ObjectProperty()
    interactivity = ObjectProperty()
    board = ObjectProperty()
    thing = ObjectProperty()
    textures = ListProperty()
    where_upon = ObjectProperty(None)

    @property
    def radii(self):
        """Return x and y offsets that will put this Pawn at a slightly
different point on a Spot, so that it's easy to grab the Spot
underneath a Pawn."""
        loc = self.thing.location
        if hasattr(loc, 'origin'):
            ref = self.board.get_spot(loc.origin)
        else:
            ref = self.board.get_spot(loc)
        try:
            (x, y) = self.sizecheat = ref.size
        except AttributeError:
            (x, y) = self.sizecheat
        return (x / 4, y / 2)

    def __init__(self, **kwargs):
        """Arrange to update my textures and my position whenever the relevant
data change.

The relevant data are

* The branch and tick, being the two measures of game-time.
* The imagery in the table pawn_img
* The location data for the Thing I represent, in the table thing_location"""
        super(Pawn, self).__init__(**kwargs)

        self.board.closet.branch_listeners.append(self.repos)
        self.board.closet.tick_listeners.append(self.repos)

        dimn = unicode(self.board.dimension)
        thingn = unicode(self.thing)
        skel = self.board.closet.skeleton

        skel["pawn_img"][dimn][thingn].listeners.append(self.upd_imagery)
        self.upd_imagery()

        skel["thing_location"][dimn][thingn].listeners.append(self.repos)
        self.repos()

    def __str__(self):
        return str(self.thing)

    def __unicode__(self):
        return unicode(self.thing)

    def on_tex(self, i, v):
        if v is None:
            return
        self.size = v.size

    def upd_imagery(self, *args):
        closet = self.board.closet
        branch = closet.branch
        tick = closet.tick
        for layer in self.imagery:
            bone = get_bone_during(self.imagery[layer], branch, tick)
            while len(self.textures) <= layer:
                self.textures.append(None)
            self.textures[layer] = closet.get_texture(bone["img"])
            if len(self.children) <= layer:
                self.add_widget(PawnImage(pawn=self, layer=layer))

    def set_interactive(self, branch=None, tick_from=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick_from is None:
            tick_from = self.board.closet.tick
        self.interactivity[branch][tick_from] = {
            "dimension": str(self.thing.dimension),
            "board": int(self.board),
            "thing": str(self.thing),
            "branch": branch,
            "tick_from": tick_from}

    def get_coords(self, branch=None, tick=None):
        loc = self.thing.get_location(branch, tick)
        if loc is None:
            return None
        if hasattr(loc, 'destination'):
            origspot = self.board.spotdict[unicode(loc.origin)]
            destspot = self.board.spotdict[unicode(loc.destination)]
            oc = origspot.get_coords(branch, tick)
            dc = destspot.get_coords(branch, tick)
            if None in (oc, dc):
                return self.cheat_coords
            (ox, oy) = oc
            (dx, dy) = dc
            prog = self.thing.get_progress(branch, tick)
            odx = dx - ox
            ody = dy - oy
            self.cheat_coords = (int(ox + odx * prog),
                                 int(oy + ody * prog))
            return self.cheat_coords
        elif unicode(loc) in self.board.spotdict:
            spot = self.board.spotdict[unicode(loc)]
            return spot.get_coords()

    def new_branch(self, parent, branch, tick):
        self.new_branch_imagery(parent, branch, tick)
        self.new_branch_interactivity(parent, branch, tick)

    def dropped(self, x, y, button, modifiers):
        """When dropped on a spot, if my thing doesn't have anything else to
do, make it journey there.

If it DOES have anything else to do, make the journey in another branch.

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

    def get_img_rd(self, branch=None, tick=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick is None:
            tick = self.board.closet.tick
        if branch not in self.imagery:
            return None
        prev = None
        for rd in self.imagery[branch].iterbones():
            if rd["tick_from"] > tick:
                break
            else:
                prev = rd
        if prev is None or prev["img"] in ("", None):
            return None
        else:
            return prev

    def get_img_source(self, branch=None, tick=None):
        name = self.get_img_rd(branch, tick)["img"]
        return self.board.closet.skeleton["img"][name]["path"]

    def get_texture(self, branch=None, tick=None):
        rd = self.get_img_rd(branch, tick)
        return self.board.closet.get_texture(rd["img"])

    def get_size(self, branch=None, tick=None):
        return self.get_texture(branch, tick).size

    def new_branch_imagery(self, parent, branch, tick):
        prev = None
        started = False
        imagery = self.board.closet.skeleton["pawn_img"][
            unicode(self.board.dimension)][unicode(self.thing)]
        for layer in imagery:
            for tick_from in imagery[layer][parent]:
                if tick_from >= tick:
                    rd2 = dict(imagery[layer][parent][tick_from])
                    rd2["branch"] = branch
                    if branch not in imagery[layer]:
                        imagery[layer][branch] = {}
                    imagery[layer][branch][rd2["tick_from"]] = rd2
                    if (
                            not started and prev is not None and
                            tick_from > tick and prev < tick):
                        rd3 = dict(imagery[layer][parent][prev])
                        rd3["branch"] = branch
                        rd3["tick_from"] = tick
                        imagery[layer][branch][rd3["tick_from"]] = rd3
                        started = True
                    prev = tick_from
        self.upd_imagery()

    def is_interactive(self, branch=None, tick=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick is None:
            tick = self.board.closet.tick
        interactivity = self.board.closet.skeleton["pawn_interactive"][
            unicode(self.board.dimension)][unicode(self.thing)]
        if branch not in interactivity:
            return False
        for rd in interactivity.iterbones():
            if rd["tick_from"] <= tick and (
                    rd["tick_to"] is None or tick <= rd["tick_to"]):
                return True
        return False

    def new_branch_interactivity(self, parent, branch, tick):
        prev = None
        started = False
        interactivity = self.board.closet.skeleton["pawn_interactive"][
            unicode(self.board.dimension)][unicode(self.thing)]
        for tick_from in interactivity[parent]:
            if tick_from >= tick:
                rd2 = dict(interactivity[parent][tick_from])
                rd2["branch"] = branch
                if branch not in interactivity:
                    interactivity[branch] = {}
                interactivity[branch][rd2["tick_from"]] = rd2
                if (
                        not started and prev is not None and
                        tick_from > tick and prev < tick):
                    rd3 = dict(interactivity[parent][prev])
                    rd3["branch"] = branch
                    rd3["tick_from"] = tick
                    interactivity[branch][rd3["tick_from"]] = rd3
                started = True
            prev = tick_from

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return
        for spot in self.board.spotdict.itervalues():
            if self.collide_widget(spot):
                myplace = self.thing.location
                theirplace = spot.place
                if myplace != theirplace:
                    print("{} journeys to {}".format(self, spot))
                    self.thing.journey_to(spot.place)
                    break
        self.repos()
        return True

    def repos(self, *args):
        if '->' in unicode(self.thing.location):
            from model.portal import Portal
            assert(isinstance(self.thing.location, Portal))
        if self.thing.location is None:
            return
        if self.where_upon is not None:
            if hasattr(self.where_upon, 'portal'):
                for place in (self.where_upon.portal.origin,
                              self.where_upon.portal.destination):
                    self.board.get_spot(place).unbind(
                        transform=self.transform_on_arrow)
            else:
                self.where_upon.unbind(
                    transform=self.transform_on_spot)
        print("relocating to {} of class {}".format(self.thing.location, self.thing.location.__class__))
        if hasattr(self.thing.location, 'origin'):
            self.where_upon = self.board.get_arrow(self.thing.location)
            for place in (self.where_upon.portal.origin,
                          self.where_upon.portal.destination):
                self.board.get_spot(place).bind(
                    transform=self.transform_on_arrow)
            ospot = self.board.get_spot(self.thing.location.origin)
            self.pos = ospot.pos
            self.transform_on_arrow(ospot, ospot.transform)
        else:
            self.where_upon = self.board.get_spot(self.thing.location)
            self.where_upon.bind(transform=self.transform_on_spot)
            self.pos = self.where_upon.pos
            self.transform_on_spot(self.where_upon, self.where_upon.transform)

    def transform_on_spot(self, i, v):
        """Presently, I am located atop the spot. I want to be located a bit
up and to the side, so you can reach the spot below me."""
        self.transform.identity()
        self.apply_transform(v)
        self.transform.translate(self.radii[0], self.radii[1], 0)

    def transform_on_arrow(self, i, v):
        """I am located some ways along the arrow. Work out how far on each
axis and transform so I appear there."""
        origspot = self.board.get_spot(self.where_upon.portal.origin)
        destspot = self.board.get_spot(self.where_upon.portal.destination)
        progress = self.thing.get_progress()
        (orig_x, orig_y) = self.where_upon.pos
        xtrans = (destspot.x - origspot.x) * progress + self.radii[0]
        ytrans = (destspot.y - origspot.y) * progress + self.radii[1]
        self.transform.identity()
        self.apply_transform(v)
        self.transform.translate(xtrans, ytrans, 0)

    def collide_point(self, x, y):
        (x0, y0) = self.to_parent(0, 0)
        (x1, y1) = self.to_parent(*self.size)
        return (
            x > x0 and x1 > x and
            y > y0 and y1 > y)
