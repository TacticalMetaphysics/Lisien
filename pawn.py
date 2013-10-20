# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import (
    SaveableWidgetMetaclass,
    Touchy)
from kivy.uix.image import Image
from kivy.uix.scatter import Scatter
from kivy.properties import (
    DictProperty,
    ObjectProperty,
    BooleanProperty)
from logging import getLogger


logger = getLogger(__name__)


"""Widget representing things that move about from place to place."""


class PawnImage(Image):
    pawn = ObjectProperty()

    def __init__(self, **kwargs):
        pawn = kwargs["pawn"]
        kwargs["texture"] = pawn.get_texture()
        kwargs["size"] = kwargs["texture"].size
        Image.__init__(self, **kwargs)
        self.pawn.bind(imagery=self.upd_tex)

    def upd_tex(self, *args):
        self.texture = self.pawn.get_texture()


class Pawn(Scatter, Touchy):
    __metaclass__ = SaveableWidgetMetaclass
    """A token to represent something that moves about between places."""
    tables = [
        ("pawn_img",
         {"dimension": "text not null default 'Physical'",
          "thing": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "img": "text not null default 'default_pawn'"},
         ("dimension", "thing", "branch", "tick_from"),
         {"dimension, thing": ("thing_location", "dimension, name"),
          "img": ("img", "name")},
         []),
        ("pawn_interactive",
         {"dimension": "text not null default 'Physical'",
          "thing": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0"},
         ("dimension", "thing", "branch", "tick_from"),
         {"dimension, thing": ("thing_location", "dimension, name")},
         [])]
    imagery = DictProperty()
    interactivity = DictProperty()
    board = ObjectProperty()
    thing = ObjectProperty()
    old_tf = ObjectProperty()
    old_tf_i = ObjectProperty()
    dragging = BooleanProperty(False)
    auto_bring_to_front = BooleanProperty(False)

    def __init__(self, **kwargs):
        """Return a pawn on the board for the given dimension, representing
the given thing with the given image. It may be visible or not,
interactive or not.

        """
        kwargs["size_hint"] = (None, None)
        kwargs["imagery"] = dict(kwargs["board"].closet.skeleton["pawn_img"][
            unicode(kwargs["board"])][unicode(kwargs["thing"])])
        kwargs["interactivity"] = dict(
            kwargs["board"].closet.skeleton["pawn_interactive"][
                unicode(kwargs["board"])][unicode(kwargs["thing"])])
        Scatter.__init__(
            self, do_rotation=False, do_scale=False, **kwargs)
        self.pos = self.get_pos()
        self.size = self.get_size()

        (rx, ry) = self.radii
        self.transform.translate(rx, ry, 0)
        self.old_tf = self.transform
        self.old_tf_i = self.transform_inv

        self.board.closet.bind(branch=self.repos, tick=self.repos)
        dimn = unicode(self.board.dimension)
        thingn = unicode(self.thing)
        skel = self.board.closet.skeleton
        skel["pawn_img"][dimn][thingn].bind(touches=self.upd_imagery)
        skel["pawn_interactive"][dimn][thingn].bind(
            touches=self.upd_interactivity)

        self.add_widget(PawnImage(pawn=self, pos=(0, 0)))

    def __str__(self):
        return str(self.thing)

    def __unicode__(self):
        return unicode(self.thing)

    def upd_imagery(self, *args):
        self.imagery = dict(self.board.closet.skeleton["pawn_img"][
            unicode(self.board.dimension)][unicode(self.thing)])

    def upd_interactivity(self, *args):
        self.interactivity = dict(self.board.closet.skeleton[
            "pawn_interactive"][unicode(self.board.dimension)][
            unicode(self.thing)])

    def set_img(self, img, branch=None, tick_from=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick_from is None:
            tick_from = self.board.closet.tick
        self.imagery[branch][tick_from] = {
            "dimension": str(self.thing.dimension),
            "thing": str(self.thing),
            "board": str(self.board),
            "branch": branch,
            "tick_from": tick_from,
            "img": str(img)}

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

    def get_pos(self):
        loc = self.thing.location
        if loc is None:
            return None
        if hasattr(loc, 'destination'):
            origspot = self.board.spotdict[unicode(loc.origin)]
            destspot = self.board.spotdict[unicode(loc.destination)]
            (ox, oy) = origspot.get_coords()
            (dx, dy) = destspot.get_coords()
            prog = self.thing.get_progress()
            odx = dx - ox
            ody = dy - oy
            (x, y) = (float(ox + odx * prog),
                      float(oy + ody * prog))
            return (x, y)
        elif unicode(loc) in self.board.spotdict:
            locspot = self.board.spotdict[unicode(loc)]
            (x, y) = locspot.get_coords()
            return (x, y)

    def get_img_rd(self, branch=None, tick=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick is None:
            tick = self.board.closet.tick
        if branch not in self.imagery:
            return None
        prev = None
        for rd in self.imagery[branch].iterrows():
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
        for tick_from in imagery[parent]:
            if tick_from >= tick:
                rd2 = dict(imagery[parent][tick_from])
                rd2["branch"] = branch
                if branch not in imagery:
                    imagery[branch] = {}
                imagery[branch][rd2["tick_from"]] = rd2
                if (
                        not started and prev is not None and
                        tick_from > tick and prev < tick):
                    rd3 = dict(imagery[parent][prev])
                    rd3["branch"] = branch
                    rd3["tick_from"] = tick
                    imagery[branch][rd3["tick_from"]] = rd3
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
        for rd in interactivity.iterrows():
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
        self.upd_interactivity()

    def on_drop(self):
        for spot in self.board.spotdict.itervalues():
            if self.collide_widget(spot):
                myplace = self.thing.location
                theirplace = spot.place
                if myplace != theirplace:
                    print("{} journeys to {}".format(self, spot))
                    self.thing.journey_to(spot.place)
                    break
        self.repos()

    def repos(self, *args):
        self.transform.identity()
        self.pos = self.get_pos()
        (rx, ry) = self.radii
        self.transform.translate(rx, ry, 0)

    def extra_translate(self, a, t):
        self.transform.identity()
        self.apply_transform(t)
        (rx, ry) = self.radii
        self.transform.translate(rx, ry, 0)

    def collide_point(self, x, y):
        (x0, y0) = self.to_parent(0, 0)
        (x1, y1) = self.to_parent(*self.size)
        return (
            x > x0 and x1 > x and
            y > y0 and y1 > y)

    @property
    def radii(self):
        loc = self.thing.location
        if hasattr(loc, 'destination'):
            locspot = self.board.spotdict[unicode(loc.origin)]
        else:
            locspot = self.board.spotdict[unicode(loc)]
        (lw, lh) = locspot.size
        return (lw / 2, lh / 2)
