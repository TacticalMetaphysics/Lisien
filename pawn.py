# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import (
    SaveableWidgetMetaclass,
    TerminableInteractivity)
from kivy.uix.image import Image
from kivy.properties import AliasProperty, ObjectProperty
from logging import getLogger


logger = getLogger(__name__)


"""Widget representing things that move about from place to place."""


class Pawn(Image, TerminableInteractivity):
    __metaclass__ = SaveableWidgetMetaclass
    """A token to represent something that moves about between places."""
    width = AliasProperty(
        lambda self: self.get_width(), lambda self, v: None)
    height = AliasProperty(
        lambda self: self.get_height(), lambda self, v: None)
    size = AliasProperty(
        lambda self: self.get_size(), lambda self, v: None,
        bind=('texture',))
    texture = AliasProperty(
        lambda self: self.get_texture(), lambda self, v: None)
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

    def __init__(self, board, thing):
        """Return a pawn on the board for the given dimension, representing
the given thing with the given image. It may be visible or not,
interactive or not.

        """
        self.board = board
        self.thing = thing
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        Image.__init__(self, texture=self.get_texture(), pos=self.get_coords())

    def __str__(self):
        return str(self.thing)

    @property
    def imagery(self):
        return self.board.closet.skeleton[
            "pawn_img"][unicode(self.thing.dimension)][unicode(self.thing)]

    @property
    def interactivity(self):
        return self.board.closet.skeleton[
            "pawn_interactive"][unicode(self.thing.dimension)][
            unicode(self.thing)]

    def retex(self):
        self.texture = self.get_texture()

    def move(self):
        self.pos = self.get_coords()

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

    def overlaps(self, x, y):
        return (
            self.window_left < x and x < self.window_right and
            self.window_bot < y and y < self.window_top)

    def get_pos_hint(self):
        loc = self.thing.location
        if loc is None:
            return None
        if hasattr(loc, 'dest'):
            # actually SpotWidgets
            origspot = self.board.spotdict[str(loc.orig)]
            destspot = self.board.spotdict[str(loc.dest)]
            (ox, oy) = origspot.coords
            (dx, dy) = destspot.coords
            prog = self.pawn.thing.get_progress()
            odx = dx - ox
            ody = dy - oy
            return {'x': int(ox + odx * prog) + self.drag_offset_x,
                    'y': int(oy + ody * prog) + self.drag_offset_y}
        elif str(loc) in self.board.spotdict:
            (x, y) = self.board.spotdict[str(loc)].pos
            return {'x': x + self.drag_offset_x,
                    'y': y + self.drag_offset_y}

    def get_pos(self):
        pos_hint = self.get_pos_hint()
        return (pos_hint['x'], pos_hint['y'])

    def get_size(self):
        img = self.get_texture()
        if img is None:
            return (0, 0)
        else:
            return (img.width, img.height)

    def get_width(self):
        return self.get_size()[0]

    def get_height(self):
        return self.get_size()[0]

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

    def new_branch_imagery(self, parent, branch, tick):
        prev = None
        started = False
        for tick_from in self.imagery[parent]:
            if tick_from >= tick:
                rd2 = dict(self.imagery[parent][tick_from])
                rd2["branch"] = branch
                if branch not in self.imagery:
                    self.imagery[branch] = {}
                self.imagery[branch][rd2["tick_from"]] = rd2
                if (
                        not started and prev is not None and
                        tick_from > tick and prev < tick):
                    rd3 = dict(self.imagery[parent][prev])
                    rd3["branch"] = branch
                    rd3["tick_from"] = tick
                    self.imagery[branch][rd3["tick_from"]] = rd3
                started = True
            prev = tick_from
