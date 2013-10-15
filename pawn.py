# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableWidgetMetaclass
from kivy.uix.image import Image
from kivy.uix.scatter import ScatterPlane
from kivy.properties import DictProperty
from kivy.clock import Clock
from logging import getLogger


logger = getLogger(__name__)


"""Widget representing things that move about from place to place."""


class Pawn(ScatterPlane):
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

    def __init__(self, board, thing):
        """Return a pawn on the board for the given dimension, representing
the given thing with the given image. It may be visible or not,
interactive or not.

        """
        self.board = board
        self.thing = thing
        self.upd_imagery()
        self.upd_interactivity()
        dimn = unicode(self.board.dimension)
        thingn = unicode(self.thing)
        skel = self.board.closet.skeleton
        skel["pawn_img"][dimn][thingn].bind(touches=self.upd_imagery)
        skel["pawn_interactive"][dimn][thingn].bind(
            touches=self.upd_interactivity)
        ScatterPlane.__init__(self)

        img = Image()
        self.collide_point = lambda x, y: img.collide_point(x, y)

        def retex(*args):
            img.texture = self.get_texture()
            img.pos = self.get_coords()
            img.size = self.get_size()

        def startup(*args):
            self.add_widget(img)
            self.board.closet.bind(branch=retex, tick=retex)
            retex()

        Clock.schedule_once(startup, 0)

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
            origspot = self.board.spotdict[str(loc.orig)]
            destspot = self.board.spotdict[str(loc.dest)]
            (ox, oy) = origspot.get_coords()
            (dx, dy) = destspot.get_coords()
            prog = self.pawn.thing.get_progress()
            odx = dx - ox
            ody = dy - oy
            return (float(ox + odx * prog) + self.drag_offset_x - origspot.rx,
                    float(oy + ody * prog) + self.drag_offset_y - origspot.ry)
        elif unicode(loc) in self.board.spotdict:
            locspot = self.board.spotdict[unicode(loc)]
            (x, y) = locspot.get_coords()
            return (x + locspot.rx, y + locspot.ry)

    def get_texture(self, branch=None, tick=None):
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
            return self.board.closet.get_texture(prev["img"])

    def get_size(self, branch=None, tick=None):
        return self.get_texture(branch, tick).size

    def new_branch_imagery(self, parent, branch, tick):
        prev = None
        started = False
        imagery = self.board.closet.skeleton["pawn_img"][
            unicode(self.board.dimension)][unicode(self.place)]
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
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        interactivity = self.closet.skeleton["pawn_interactive"][
            unicode(self.board.dimension)][unicode(self.place)]
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
        interactivity = self.closet.skeleton["pawn_interactive"][
            unicode(self.board.dimension)][unicode(self.place)]
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
