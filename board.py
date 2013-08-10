# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass
from collections import OrderedDict
from pawn import Pawn
from spot import Spot
from arrow import Arrow
from pyglet.graphics import OrderedGroup
from pyglet.sprite import Sprite


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
        while (
                not hasattr(r, 'spots') or
                len(r.spots) <= self.i or
                r.spots[self.i] is None):
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
        while (
                not hasattr(r, 'arrows') or
                len(r.arrows) <= self.i or
                r.arrows[self.i] is None):
            r = self.portit.next()
        while not r.extant():
            r.arrows[self.i].delete()
            r = self.portit.next()
        return r.arrows[self.i]


class AbstractBoard:
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

    def __init__(self, window, width, height, wallpaper):
        """Return a board representing the given dimension.

        """
        self.window = window
        self.dimension = window.dimension
        self.rumor = self.dimension.rumor
        self.width = width
        self.height = height
        self.wallpaper = wallpaper
        self.menu_by_name = OrderedDict()
        self.pawndict = {}
        self.spotdict = {}
        self.arrowdict = {}

    def __getattr__(self, attrn):
        if attrn == "places":
            return iter(self.dimension.places)
        elif attrn == "things":
            return iter(self.dimension.things)
        elif attrn == "portals":
            return iter(self.dimension.portals)
        elif attrn == "pawns":
            return self.pawndict.itervalues()
        elif attrn == "spots":
            return self.spotdict.itervalues()
        elif attrn == "arrows":
            return self.arrowdict.itervalues()
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

    def make_pawn(self, thing):
        self.pawndict[str(thing)] = Pawn(self, thing)

    def get_pawn(self, thing):
        if str(thing) not in self.pawndict:
            self.make_pawn(thing)
        return self.pawndict[str(thing)]

    def make_spot(self, place):
        self.spotdict[str(place)] = Spot(self, place)

    def get_spot(self, place):
        if str(place) not in self.spotdict:
            self.make_spot(place)
        return self.spotdict[str(place)]

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

    def draw(self, batch, group):
        if not hasattr(self, 'bggroup'):
            self.bggroup = OrderedGroup(0, group)
        if not hasattr(self, 'arrowgroup'):
            self.arrowgroup = OrderedGroup(1, group)
        if not hasattr(self, 'spotgroup'):
            self.spotgroup = OrderedGroup(2, group)
        if not hasattr(self, 'pawngroup'):
            self.pawngroup = OrderedGroup(3, group)
        if not hasattr(self, 'bgsprite'):
            self.bgsprite = Sprite(
                self.wallpaper.tex,
                self.window.offset_x,
                self.window.offset_y,
                batch=batch,
                group=self.bggroup)
        else:
            self.bgsprite.x = self.window.offset_x
            self.bgsprite.y = self.window.offset_y
        for arrow in self.arrows:
            arrow.draw(batch, self.arrowgroup)
        for spot in self.spots:
            spot.draw(batch, self.spotgroup)
        for pawn in self.pawns:
            pawn.draw(batch, self.pawngroup)


class Board(AbstractBoard):
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
        self.i = i
        super(Board, self).__init__(window, width, height, wallpaper)

    def get_tabdict(self):
        return {
            "board": [
                {"dimension": str(self.dimension),
                 "i": int(self),
                 "wallpaper": str(self.wallpaper),
                 "width": self.width,
                 "height": self.height}]}
