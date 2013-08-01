# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import (
    SaveableMetaclass,
    TerminableImg,
    TerminableInteractivity,
    TerminableCoords,
    BranchTicksIter,
    dictify_row)
from logging import getLogger


logger = getLogger(__name__)


"""Widgets to represent places. Pawns move around on top of these."""


class Spot(object, TerminableImg, TerminableInteractivity, TerminableCoords):
    """The icon that represents a Place.

    The Spot is located on the Board that represents the same
    Dimension that the underlying Place is in. Its coordinates are
    relative to its Board, not necessarily the window the Board is in.

    """
    __metaclass__ = SaveableMetaclass
    tables = [
        ("spot_img",
         {"dimension": "text not null default 'Physical'",
          "place": "text not null",
          "board": "integer not null default 0",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "tick_to": "integer default null",
          "img": "text not null default 'default_spot'"},
         ("dimension", "place", "board", "branch", "tick_from"),
         {"dimension, board": ("board", "dimension, i"),
          "img": ("img", "name")},
         []),
        ("spot_interactive",
         {"dimension": "text not null default 'Physical'",
          "place": "text not null",
          "board": "integer not null default 0",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "tick_to": "integer default null"},
         ("dimension", "place", "board", "branch", "tick_from"),
         {"dimension, board": ("board", "dimension, i")},
         []),
        ("spot_coords",
         {"dimension": "text not null default 'Physical'",
          "place": "text not null",
          "board": "integer not null default 0",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "tick_to": "integer default null",
          "x": "integer not null default 50",
          "y": "integer not null default 50"},
         ("dimension", "place", "board", "branch", "tick_from"),
         {"dimension, board": ("board", "dimension, i")},
         [])]
    selectable = True

    def __init__(self, board, place):
        """Return a new spot on the board for the given dimension,
representing the given place with the given image. It will be at the
given coordinates, and visible or interactive as indicated.
        """
        self.board = board
        self.db = self.board.db
        self.window = self.board.window
        self.place = place
        self.interactivity = {}
        self.imagery = {}
        self.coord_dict = {}
        self.indefinite_imagery = {}
        self.indefinite_coords = {}
        self.indefinite_interactivity = {}
        self.grabpoint = None
        self.sprite = None
        self.box_edges = (None, None, None, None)
        self.oldstate = None
        self.newstate = None
        self.tweaks = 0
        self.drag_offset_x = 0
        self.drag_offset_y = 0

    def __getattr__(self, attrn):
        if attrn == 'dimension':
            return self.board.dimension
        elif attrn == 'interactive':
            return self.is_interactive()
        elif attrn == 'img':
            return self.get_img()
        elif attrn == 'hovered':
            return self.window.hovered is self
        elif attrn == 'pressed':
            return self.window.pressed is self
        elif attrn == 'grabbed':
            return self.window.grabbed is self
        elif attrn == 'selected':
            return self in self.window.selected
        elif attrn == 'coords':
            return self.get_coords()
        elif attrn == 'x':
            return self.coords[0]
        elif attrn == 'y':
            return self.coords[1]
        elif attrn == 'window_coords':
            (x, y) = self.coords
            return (
                x + self.drag_offset_x + self.window.offset_x,
                y + self.drag_offset_y + self.window.offset_y)
        elif attrn == 'width':
            myimg = self.img
            if myimg is None:
                return 0
            else:
                return myimg.width
        elif attrn == 'height':
            myimg = self.img
            assert(hasattr(myimg, 'tex'))
            if myimg is None:
                return 0
            else:
                return myimg.height
        elif attrn == 'rx':
            return self.width / 2
        elif attrn == 'ry':
            return self.height / 2
        elif attrn == 'r':
            if self.rx > self.ry:
                return self.rx
            else:
                return self.ry
        elif attrn == 'window_x':
            return self.window_coords[0]
        elif attrn == 'window_y':
            return self.window_coords[1]
        elif attrn == 'window_left':
            return self.window_x - self.rx
        elif attrn == 'window_bot':
            return self.window_y - self.ry
        elif attrn == 'window_top':
            return self.window_y + self.ry
        elif attrn == 'window_right':
            return self.window_x + self.rx
        elif attrn == 'in_window':
            return (self.window_top > 0 and
                    self.window_right > 0) or (
                        self.window_bot < self.window.height and
                        self.window_right < self.window.width)
        elif attrn == 'visible':
            return self.img is not None
        else:
            raise AttributeError(
                "Spot instance has no such attribute: " +
                attrn)

    def __setattr__(self, attrn, val):
        if attrn == "img":
            self.set_img(val)
        elif attrn == "interactive":
            self.set_interactive(val)
        elif attrn == "x":
            raise Exception("Don't set x that way")
        elif attrn == "y":
            raise Exception("Don't set y that way")
        elif attrn == "hovered":
            if val is True:
                self.hovered()
            else:
                self.unhovered()
        elif attrn == "pressed":
            if val is True:
                self.set_pressed()
            else:
                self.unset_pressed()
        else:
            super(Spot, self).__setattr__(attrn, val)

    def __str__(self):
        return str(self.place)

    def __eq__(self, other):
        """Compare the dimension and the name"""
        return (
            isinstance(other, Spot) and
            self.dimension == other.dimension and
            self.name == other.name)

    def onclick(self):
        """Does nothing yet"""
        pass

    def dropped(self, x, y, button, modifiers):
        c = self.get_coords()
        newx = c[0] + self.drag_offset_x
        newy = c[1] + self.drag_offset_y
        self.set_coords(c[0] + self.drag_offset_x, c[1] + self.drag_offset_y)
        self.drag_offset_x = 0
        self.drag_offset_y = 0

    def hovered(self):
        """Become hovered"""
        if not self.hovered:
            self.hovered = True
            self.tweaks += 1

    def unhovered(self):
        """Stop being hovered"""
        if self.hovered:
            self.hovered = False
            self.tweaks += 1

    def set_pressed(self):
        """Become pressed"""
        pass

    def unset_pressed(self):
        """Stop being pressed"""
        pass

    def move_with_mouse(self, x, y, dx, dy, buttons, modifiers):
        """Remember where exactly I was grabbed, then move around with the
mouse, always keeping the same relative position with respect to the
mouse."""
        self.drag_offset_x += dx
        self.drag_offset_y += dy

    def overlaps(self, x, y):
        (myx, myy) = self.window_coords
        return (
            self.visible and
            self.interactive and
            abs(myx - x) < self.rx and
            abs(myy - y) < self.ry)
        
    def get_tabdict(self):
        return {
            "spot_img": [
                {
                    "dimension": str(self.dimension),
                    "place": str(self.place),
                    "board": int(self.board),
                    "branch": branch,
                    "tick_from": tick_from,
                    "tick_to": tick_to,
                    "img": str(img)}
                for (branch, tick_from, tick_to, img) in
                BranchTicksIter(self.imagery)],
            "spot_interactive": [
                {
                    "dimension": str(self.dimension),
                    "place": str(self.place),
                    "board": int(self.board),
                    "branch": branch,
                    "tick_from": tick_from,
                    "tick_to": tick_to}
                for (branch, tick_from, tick_to) in
                BranchTicksIter(self.interactivity)],
            "spot_coords": [
                {
                    "dimension": str(self.dimension),
                    "place": str(self.place),
                    "board": int(self.board),
                    "branch": branch,
                    "tick_from": tick_from,
                    "tick_to": tick_to,
                    "x": x,
                    "y": y}
                for (branch, tick_from, tick_to, x, y) in
                BranchTicksIter(self.coord_dict)]}

    def get_state_tup(self):
        return (
            str(self.board.dimension),
            int(self.board),
            str(self.place),
            str(self.img),
            self.hovered,
            self.pressed,
            self.grabbed,
            self.selected,
            self.window_x,
            self.window_y)
