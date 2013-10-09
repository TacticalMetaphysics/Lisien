# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import unicode_literals
ascii = str
str = unicode
from util import SaveableMetaclass
from pawn import Pawn
from spot import Spot
from arrow import Arrow
from kivy.uix.scatter import Scatter


"""Class for user's view on gameworld, and support functions."""


__metaclass__ = SaveableMetaclass


class Board(object):
    """A widget notionally representing the game board on which the rest
of the game pieces lie.

Each board represents exactly one dimension in the world model, but
you can have more than one board to a dimension. It has a width and
height in pixels, which do not necessarily match the width or height
of the window it's displayed in--a board may be scrolled horizontally
or vertically. Every board has a static background image, and may have
menus. The menus' positions are relative to the window rather than the
board, but they are linked to the board anyhow, on the assumption that
each board will be open in at most one window at a time.

    """
    tables = [
        ("board",
         {"dimension": "text not null default 'Physical'",
          "wallpaper": "text not null default 'default_wallpaper'"},
         ("dimension", "idx"),
         {"wallpaper": ("img", "name")},
         [])]
    atrdic = {
        "wallpaper": lambda self: self.closet.get_img(
            self._rowdict["wallpaper"]),
        "width": lambda self: self.wallpaper.tex.width,
        "height": lambda self: self.wallpaper.tex.height,
        "places": lambda self: iter(self.dimension.places),
        "portals": lambda self: iter(self.dimension.portals),
        "things": lambda self: iter(self.dimension.things),
        "pawns": lambda self: self.pawndict.itervalues(),
        "spots": lambda self: self.spotdict.itervalues(),
        "arrows": lambda self: self.arrowdict.itervalues(),

    def __init__(self, closet, dimension):
        """Return a board representing the given dimension.

        """
        self.closet = closet
        self.dimension = dimension
        self.pawndict = {}
        self.spotdict = {}
        self.arrowdict = {}
        # self.closet = closet
        # self.dimension = dimension
        # self.idx = idx
        # self.pawndict = {}
        # self.spotdict = {}
        # self.arrowdict = {}
        # self._rowdict = self.closet.skeleton[
        #     "board"][str(self.dimension)][int(self)]
        # while len(self.dimension.boards) <= self.idx:
        #     self.dimension.boards.append(None)
        # self.dimension.boards[self.idx] = self
        # if (
        #         "spot_coords" in self.closet.skeleton and
        #         str(self.dimension) in self.closet.skeleton["spot_coords"]):
        #     for rd in self.closet.skeleton[
        #             "spot_coords"][str(self.dimension)][
        #             int(self)].iterrows():
        #         self.add_spot(rd)
        # if (
        #         "pawn_img" in self.closet.skeleton and
        #         str(self.dimension) in self.closet.skeleton["pawn_img"]):
        #     for rd in self.closet.skeleton[
        #             "pawn_img"][str(self.dimension)][
        #             int(self)].iterrows():
        #         self.add_pawn(rd)
        # for portal in self.dimension.portals:
        #     self.make_arrow(portal)

    def __getattr__(self, attrn):
        return self.atrdic[attrn](self)

    def __int__(self):
        return self.idx

    def add_spot(self, rd):
        assert(rd["dimension"] == str(self.dimension))
        self.spotdict[rd["place"]] = Spot(
            self.closet, self.dimension, self,
            self.dimension.get_place(rd["place"]))

    def add_pawn(self, rd):
        assert(rd["dimension"] == str(self.dimension))
        self.pawndict[rd["thing"]] = Pawn(
            self.closet, self.dimension, self,
            self.dimension.get_thing(rd["thing"]))

    def get_spot_at(self, x, y):
        for spot in self.spots:
            if (
                    spot.board_left < x and x < spot.board_right and
                    spot.board_bot < y and y < spot.board_top):
                return spot
        return None

    def get_arrow_at(self, x, y):
        for arrow in self.arrows:
            if arrow.overlaps(x, y):
                return arrow
        return None

    def get_pawn_at(self, x, y):
        for pawn in self.pawns:
            if (
                    pawn.board_left < x and x < pawn.board_right and
                    pawn.board_bot < y and y < pawn.board_top):
                return pawn
        return None

    def get_pawn(self, thing):
        return self.pawndict[str(thing)]

    def get_spot(self, place):
        if str(place) not in self.spotdict:
            self.make_spot(place)
        return self.spotdict[str(place)]

    def make_spot(self, place):
        place = self.closet.get_place(str(self.dimension), str(place))
        self.spotdict[str(place)] = Spot(
            self.closet, self.dimension, self, place)

    def make_arrow(self, orig_or_port, dest=None):
        if dest is None:
            self.arrowdict[str(orig_or_port)] = Arrow(self, orig_or_port)
        else:
            name = "Portal({0}->{1})".format(orig_or_port, dest)
            self.arrowdict[name] = Arrow(self, orig_or_port, dest)

    def get_arrow(self, orig_or_port, dest=None):
        if dest is None:
            name = str(orig_or_port)
        else:
            name = "Portal({0}->{1})".format(orig_or_port, dest)
        if name not in self.arrowdict:
            self.make_arrow(orig_or_port, dest)
        return self.arrowdict[name]

    def new_branch(self, parent, branch, tick):
        for spot in self.spots:
            spot.new_branch(parent, branch, tick)
        for pawn in self.pawns:
            pawn.new_branch(parent, branch, tick)
        # Arrows don't have branchdicts. Just make them smart enough
        # to handle their portal changing its.

    def overlaps(self, x, y):
        return (
            x > self.window_left and
            x < self.window_right and
            y > self.window_bot and
            y < self.window_top)

    def relativize(self, x, y):
        return (
            x + self.window_left + self.offset_x,
            y + self.window_bot + self.offset_y)

    def get_pawn_at(self, x, y):
        for pawn in PawnIterX(self, x):
            if pawn.window_bot < y and y < pawn.window_top:
                return pawn
        return None

    def get_spot_at(self, x, y):
        for spot in SpotIterX(self, x):
            if spot.window_bot < y and y < spot.window_top:
                return spot
        return None

    def get_arrow_at(self, x, y):
        for arrow in ArrowIterX(self, x):
            if arrow.window_bot < y and y < arrow.window_top:
                return arrow
        return None

    def hover(self, x, y):
        x -= self.window_left
        y -= self.window_bot
        pawn_at = self.get_pawn_at(x, y)
        if pawn_at is not None:
            return pawn_at
        spot_at = self.get_spot_at(x, y)
        if spot_at is not None:
            return spot_at
        arrow_at = self.get_arrow_at(x, y)
        if arrow_at is not None:
            return arrow_at
        else:
            return self

    def on_mouse_drag(self, x, y, dx, dy, button, modifiers):
        self.moved = True
        self.view_left -= dx
        self.view_bot -= dy
        if self.view_left < 0:
            self.view_left = 0
        if self.view_bot < 0:
            self.view_bot = 0
        if self.view_left + self.width > self.board.width:
            self.view_left = self.board.width - self.width
        if self.view_bot + self.height > self.board.height:
            self.view_bot = self.board.height - self.height
        return self

    def draw(self):
        if self.moved:
            self.bgregion = self.wallpaper.tex.get_region(
                self.view_left, self.view_bot,
                self.window.width, self.window.height)
            self.bgsprite = Sprite(
                self.bgregion,
                0, 0,
                batch=self.batch,
                group=self.window.board_bg_group)
            self.moved = False
        for spot in self.spots:
            spot.draw()
        for pawn in self.pawns:
            pawn.draw()
        for arrow in self.arrows:
            arrow.draw()


class SpotIterX(object):
    def __init__(self, bv, x):
        self.spotiter = bv.spots
        self.x = x

    def __iter__(self):
        return self

    def next(self):
        spot = self.spotiter.next()
        while spot.window_left > self.x or spot.window_right < self.x:
            spot = self.spotiter.next()
        return spot


class PawnIterX(object):
    def __init__(self, bv, x):
        self.pawniter = bv.pawns
        self.x = x

    def __iter__(self):
        return self

    def next(self):
        pawn = self.pawniter.next()
        while self.x < pawn.window_left or pawn.window_right < self.x:
            pawn = self.pawniter.next()
        return pawn


class ArrowIterX(object):
    def __init__(self, bv, x):
        self.arrowiter = bv.arrows
        self.x = x

    def __iter__(self):
        return self

    def next(self):
        arrow = self.arrowiter.next()
        while self.x < arrow.window_left or arrow.window_right < self.x:
            arrow = self.arrowiter.next()
        return arrow
