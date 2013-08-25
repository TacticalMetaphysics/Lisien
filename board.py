# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass, ViewportOrderedGroup
from collections import OrderedDict
from pawn import Pawn, PawnWidget
from spot import Spot, SpotWidget
from arrow import Arrow, ArrowWidget
from pyglet.graphics import OrderedGroup
from pyglet.sprite import Sprite


"""Class for user's view on gameworld, and support functions."""


__metaclass__ = SaveableMetaclass


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
    tables = [
        ("board",
         {"dimension": "text not null default 'Physical'",
          "idx": "integer not null default 0",
          "wallpaper": "text not null default 'default_wallpaper'",
          "width": "integer not null default 4000",
          "height": "integer not null default 3000"},
         ("dimension", "idx"),
         {"wallpaper": ("img", "name")},
         [])]

    def __init__(self, dimension, idx, width, height, wallpaper):
        """Return a board representing the given dimension.

        """
        self.dimension = dimension
        self.idx = idx
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
        return self.idx

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

    def new_branch(self, parent, branch, tick):
        for spot in self.spots:
            spot.new_branch(parent, branch, tick)
        for pawn in self.pawns:
            pawn.new_branch(parent, branch, tick)
        # Arrows don't have branchdicts. Just make them smart enough
        # to handle their portal changing its.

    def get_tabdict(self):
        return {
            "board":
                {"dimension": str(self.dimension),
                 "idx": int(self),
                 "wallpaper": str(self.wallpaper),
                 "width": self.width,
                 "height": self.height}}

    def update_from_tabdict(self, tabdict):
        rd = tabdict["board"]
        self.dimension = self.rumor.get_dimension(rd["dimension"])
        self.idx = rd["idx"]
        self.wallpaper = self.rumor.get_img(rd["wallpaper"])
        self.width = rd["width"]
        self.height = rd["height"]

class BoardViewport:
    tables = [
        ("board_viewport",
         {"window": "text not null",
          "dimension": "text not null",
          "board": "integer not null default 0",
          "idx": "integer not null default 0",
          "left": "float not null default 0.0",
          "bot": "float not null default 0.0",
          "top": "float not null default 1.0",
          "right": "float not null default 1.0",
          "view_left": "integer not null default 0",
          "view_bot": "integer not null default 0",
          "arrow_width": "float not null default 1.4",
          "arrowhead_size": "integer not null default 10"},
         ("window", "dimension", "board", "idx"),
         {"window": ("window", "name"),
          "dimension, board": ("board", "dimension, i")},
         ["view_left>=0", "view_bot>=0", "left>=0.0", "bot>=0.0",
          "right>=0.0", "top>=0.0", "left<=1.0", "bot<=1.0",
          "right<=1.0", "top<=1.0", "right>left", "top>bot"])]

    def __init__(self, window, board, idx,
                 left, bot, top, right, view_left, view_bot,
                 arrow_width, arrowhead_size):
        self.board = board
        self.window = window
        self.idx = idx
        self.left_prop = left
        self.bot_prop = bot
        self.top_prop = top
        self.right_prop = right
        self.view_left = view_left
        self.view_bot = view_bot
        self.arrow_width = arrow_width
        self.arrowhead_size = arrowhead_size
        self.batch = self.window.batch
        self.supergroup = OrderedGroup(0, self.window.boardgroup)
        self.bggroup = OrderedGroup(0, self.supergroup)
        self.arrowgroup = OrderedGroup(1, self.supergroup)
        self.spotgroup = OrderedGroup(2, self.supergroup)
        self.pawngroup = OrderedGroup(3, self.supergroup)
        self.pawndict = {}
        self.spotdict = {}
        self.arrowdict = {}
        for (k, v) in self.board.pawndict.iteritems():
            self.pawndict[k] = PawnWidget(self, v)
        for (k, v) in self.board.spotdict.iteritems():
            self.spotdict[k] = SpotWidget(self, v)
        for (k, v) in self.board.arrowdict.iteritems():
            self.arrowdict[k] = ArrowWidget(self, v)

    def __getattr__(self, attrn):
        if attrn == "window_left":
            return int(self.left_prop * self.window.width)
        elif attrn == "window_right":
            return int(self.right_prop * self.window.width)
        elif attrn == "window_bot":
            return int(self.bot_prop * self.window.height)
        elif attrn == "window_top":
            return int(self.top_prop * self.window.height)
        elif attrn == "width":
            return self.window_right - self.window_left
        elif attrn == "height":
            return self.window_top - self.window_bot
        elif attrn == "offset_x":
            return -1 * self.view_left
        elif attrn == "offset_y":
            return -1 * self.view_bot
        elif attrn == "arrows":
            return self.arrowdict.itervalues()
        elif attrn == "spots":
            return self.spotdict.itervalues()
        elif attrn == "pawns":
            return self.pawndict.itervalues()
        elif attrn in (
                "dimension", "idx", "wallpaper"):
            return getattr(self.board, attrn)
        else:
            raise AttributeError(
                "BoardView instance has no attribute " + attrn)

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
        for pawn in self.pawns:
            if pawn.in_view and pawn.overlaps(x, y):
                return pawn
        return None

    def get_spot_at(self, x, y):
        for spot in self.spots:
            if spot.in_view and spot.overlaps(x, y):
                return spot
        return None

    def get_arrow_at(self, x, y):
        for arrow in self.arrows:
            if arrow.in_view and arrow.overlaps(x, y):
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

    def move_with_mouse(self, x, y, dx, dy, button, modifiers):
        self.view_left -= dx
        self.view_bot -= dy
        if self.view_left < 0:
            self.view_left = 0
        if self.view_bot < 0:
            self.view_bot = 0

    def draw(self):
        try:
            self.bgsprite.x = self.offset_x
            self.bgsprite.y = self.offset_y
        except:
            self.bgsprite = Sprite(
                self.wallpaper.tex,
                self.offset_x,
                self.offset_y,
                batch=self.batch,
                group=self.bggroup)
        for arrow in self.arrows:
            arrow.draw()
        for spot in self.spots:
            spot.draw()
        for pawn in self.pawns:
            pawn.draw()

    def get_tabdict(self):
        return {
            "board_viewport": {
                "window": str(self.window),
                "dimension": str(self.board.dimension),
                "board": int(self.board),
                "idx": self.idx,
                "left": self.left_prop,
                "bot": self.bot_prop,
                "top": self.top_prop,
                "right": self.right_prop,
                "view_left": self.view_left,
                "view_bot": self.view_bot}}

    def save(self):
        self.coresave()
        self.board.save()
