# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import (
    SaveableWidgetMetaclass,
    TerminableImg,
    TerminableInteractivity)
from kivy.uix.image import Image
from kivy.properties import AliasProperty
from logging import getLogger


logger = getLogger(__name__)


"""Widget representing things that move about from place to place."""


class Pawn(Image, TerminableImg, TerminableInteractivity):
    __metaclass__ = SaveableWidgetMetaclass
    """A token to represent something that moves about between places."""
    pos_hint = AliasProperty(
        lambda self: self.get_board_coords(), lambda self, v: None)
    size = AliasProperty(
        lambda self: self.get_size(), lambda self, v: None)
    width = AliasProperty(
        lambda self: self.get_width(), lambda self, v: None)
    height = AliasProperty(
        lambda self: self.get_height(), lambda self, v: None)
    texture = AliasProperty(
        lambda self: self.get_texture(), lambda self, v: None)
    closet = AliasProperty(
        lambda self: self.board.closet, lambda self, v: None)
    dimension = AliasProperty(
        lambda self: self.board.dimension, lambda self, v: None)
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
    atrdic = {
        "imagery": lambda self: self.closet.skeleton[
            "pawn_img"][str(self.dimension)][str(self.thing)],
        "interactivity": lambda self: self.closet.skeleton["pawn_interactive"][
            str(self.dimension)][str(self.thing)],
        "img": lambda self: self.get_img(),
        "visible": lambda self: self.img is not None,
        "x": lambda self: self.coords[0],
        "y": lambda self: self.coords[1],
        "width": lambda self: self.img.width,
        "height": lambda self: self.img.height,
        "rx": lambda self: self.width / 2,
        "ry": lambda self: self.height / 2,
        "r": lambda self: {True: self.rx, False: self.ry}[self.rx > self.ry],
        "dimension": lambda self: self.board.dimension,
        "closet": lambda self: self.board.dimension.closet}

    def __init__(self, board, thing):
        """Return a pawn on the board for the given dimension, representing
the given thing with the given image. It may be visible or not,
interactive or not.

        """
        self.board = board
        self.thing = thing
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        Image.__init__(self)

    def __str__(self):
        return str(self.thing)

    def __getattr__(self, attrn):
        try:
            return self.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                "Pawn instance has no such attribute: " +
                attrn)

    def __setattr__(self, attrn, val):
        if attrn == "img":
            self.set_img(val)
        elif attrn == "interactive":
            self.set_interactive(val)
        else:
            object.__setattr__(self, attrn, val)

    def set_img(self, img, branch=None, tick_from=None):
        if branch is None:
            branch = self.closet.branch
        if tick_from is None:
            tick_from = self.closet.tick
        self.imagery[branch][tick_from] = {
            "dimension": str(self.dimension),
            "thing": str(self.thing),
            "board": str(self.board),
            "branch": branch,
            "tick_from": tick_from,
            "img": str(img)}

    def set_interactive(self, branch=None, tick_from=None):
        if branch is None:
            branch = self.closet.branch
        if tick_from is None:
            tick_from = self.closet.tick
        self.interactivity[branch][tick_from] = {
            "dimension": str(self.dimension),
            "board": int(self.board),
            "thing": str(self.thing),
            "branch": branch,
            "tick_from": tick_from}

    def get_coords(self, branch=None, tick=None):
        loc = self.thing.get_location(branch, tick)
        if loc is None:
            return None
        if hasattr(loc, 'dest'):
            origspot = self.board.get_spot(loc.orig)
            destspot = self.board.get_spot(loc.dest)
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
        elif str(loc) in self.board.spotdict:
            spot = self.board.get_spot(loc)
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
        for spot in self.viewport.board.spots:
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

    def pass_focus(self):
        return self.viewport

    def get_board_coords(self):
        loc = self.pawn.thing.location
        if loc is None:
            return None
        if hasattr(loc, 'dest'):
            # actually SpotWidgets
            origspot = self.viewport.spotdict[str(loc.orig)]
            destspot = self.viewport.spotdict[str(loc.dest)]
            (ox, oy) = origspot.coords
            (dx, dy) = destspot.coords
            prog = self.pawn.thing.get_progress()
            odx = dx - ox
            ody = dy - oy
            return (int(ox + odx * prog) + self.drag_offset_x,
                    int(oy + ody * prog) + self.drag_offset_y)
        elif str(loc) in self.viewport.spotdict:
            (x, y) = self.viewport.spotdict[str(loc)].coords
            return (x + self.drag_offset_x,
                    y + self.drag_offset_y)

    def get_size(self):
        img = self.get_img()
        if img is None:
            return [0, 0]
        else:
            return [img.texture.width, img.texture.height]

    def get_width(self):
        return self.get_size()[0]

    def get_height(self):
        return self.get_size()[0]

    def get_texture(self):
        return self.get_img().texture
