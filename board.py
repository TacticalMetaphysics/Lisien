# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass
from collections import OrderedDict
from pawn import Pawn
from spot import Spot


"""Class for user's view on gameworld, and support functions."""


class BoardPawnIter:
    def __init__(self, board):
        self.thingit = board.things
        self.i = int(board)

    def __iter__(self):
        return self

    def next(self):
        r = self.thingit.next()
        while not hasattr(r, 'pawns') or len(r.pawns) <= self.i:
            r = self.thingit.next()
        return r.pawns[self.i]


class BoardSpotIter:
    def __init__(self, board):
        self.placeit = board.places
        self.i = int(board)

    def __iter__(self):
        return self

    def next(self):
        r = self.placeit.next()
        while not hasattr(r, 'spots'):
            r = self.placeit.next()
        return r.spots[self.i]


class BoardArrowIter:
    def __init__(self, board):
        self.portit = board.portals
        self.i = int(board)

    def __iter__(self):
        return self

    def next(self):
        r = self.portit.next()
        while not hasattr(r, 'arrows'):
            r = self.portit.next()
        return r.arrows[self.i]


class Board:
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
    __metaclass__ = SaveableMetaclass
    tables = [
        ("board",
         {"dimension": "text not null default 'Physical'",
          "i": "integer not null default 0",
          "wallpaper": "text not null default 'default_wallpaper'",
          "width": "integer not null default 4000",
          "height": "integer not null default 3000"},
         ("dimension", "i"),
         {"wallpaper": ("image", "name")},
         []),
    ]

    def __init__(self, window, i, width, height, wallpaper):
        """Return a board representing the given dimension.

        """
        self.window = window
        self.dimension = window.dimension
        self.rumor = self.dimension.rumor
        self.i = i
        self.width = width
        self.height = height
        self.wallpaper = wallpaper
        self.menu_by_name = OrderedDict()

    def __getattr__(self, attrn):
        if attrn == "places":
            return iter(self.dimension.places)
        elif attrn == "things":
            return iter(self.dimension.things)
        elif attrn == "portals":
            return iter(self.dimension.portals)
        elif attrn == "pawns":
            return BoardPawnIter(self)
        elif attrn == "spots":
            return BoardSpotIter(self)
        elif attrn == "arrows":
            return BoardArrowIter(self)
        elif attrn == "menus":
            return self.menu_by_name.itervalues()
        else:
            raise AttributeError("Board has no attribute named " + attrn)

    def __int__(self):
        return self.i

    def get_spot_at(self, x, y):
        for spot in self.spots:
            if (
                    spot.window_left < x < spot.window_right and
                    spot.window_bot < y < spot.window_top):
                return spot
        return None

    def get_tabdict(self):
        return {
            "board": [
                {"dimension": str(self.dimension),
                 "i": int(self),
                 "wallpaper": str(self.wallpaper),
                 "width": self.width,
                 "height": self.height}]}

    def make_pawn(self, thing):
        thing.pawns[int(self)] = Pawn(self, thing)

    def get_pawn(self, thing):
        if int(self) not in thing.pawns:
            self.make_pawn(thing)
        return thing.pawns[int(self)]

    def make_spot(self, place):
        place.spots[int(self)] = Spot(self, place)

    def get_spot(self, place):
        if int(self) not in place.spots:
            self.make_spot(place)
        return place.spots[int(self)]

    def save(self):
        for pawn in self.pawns:
            pawn.save()
        for spot in self.spots:
            spot.save()
        self.coresave()
