## This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import unicode_literals
ascii = str
str = unicode
from util import (
    SaveableMetaclass,
    TerminableImg,
    TerminableInteractivity)
from kivy.uix import Image
from logging import getLogger


logger = getLogger(__name__)


__metaclass__ = SaveableMetaclass


"""Widgets to represent places. Pawns move around on top of these."""


class Spot(Image, TerminableImg, TerminableInteractivity):
    """The icon that represents a Place.

    The Spot is located on the Board that represents the same
    Dimension that the underlying Place is in. Its coordinates are
    relative to its Board, not necessarily the window the Board is in.

    """
    tables = [
        ("spot_img",
         {"dimension": "text not null default 'Physical'",
          "place": "text not null",
          "board": "integer not null default 0",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "img": "text not null default 'default_spot'"},
         ("dimension", "board", "place", "branch", "tick_from"),
         {"dimension, board": ("board", "dimension, i"),
          "img": ("img", "name")},
         []),
        ("spot_interactive",
         {"dimension": "text not null default 'Physical'",
          "place": "text not null",
          "board": "integer not null default 0",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0"},
         ("dimension", "board", "place", "branch", "tick_from"),
         {"dimension, board": ("board", "dimension, i")},
         []),
        ("spot_coords",
         {"dimension": "text not null default 'Physical'",
          "place": "text not null",
          "board": "integer not null default 0",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "x": "integer not null default 50",
          "y": "integer not null default 50"},
         ("dimension", "board", "place", "branch", "tick_from"),
         {"dimension, board": ("board", "dimension, i")},
         [])]

    def __init__(self, board, place):
        self.closet = closet
        self.board = board
        self.place = place
        self.vert = self.place.v
        self.coord_lst = self.closet.skeleton["spot_coords"][
            str(self.dimension)][str(self.place)]
        self.interactivity = self.closet.skeleton["spot_interactive"][
            str(self.dimension)][str(self.place)]
        self.imagery = self.closet.skeleton["spot_img"][
            str(self.dimension)][str(self.place)]

    def __str__(self):
        return str(self.place)

    @property
    def vertex(self):
        return self.place.v

    @property
    def interactive(self):
        return self.is_interactive()

    @property
    def coords(self):
        return self.get_coords()

    @property
    def x(self):
        return self.coords[0]

    @property
    def y(self):
        return self.coords[1]

    @property
    def width(self):
        img = self.get_img()
        if img is None:
            return 0
        else:
            return img.width

    @property
    def height(self):
        img = self.get_img()
        if img is None:
            return 0
        else:
            return img.width

    @property
    def rx(self):
        return self.width / 2

    @property
    def ry(self):
        return self.height / 2

    @property
    def r(self):
        rx = self.rx
        ry = self.ry
        if rx > ry:
            return rx
        else:
            return ry

    @property
    def visible(self):
        return self.img is not None

    @property
    def board_left(self):
        return self.x - self.rx

    @property
    def board_right(self):
        return self.x + self.rx

    @property
    def board_bot(self):
        return self.y - self.ry

    @property
    def board_top(self):
        return self.y + self.ry

    def set_interactive(self, branch=None, tick_from=None):
        if branch is None:
            branch = self.closet.branch
        if tick_from is None:
            tick_from = self.closet.tick
        assert branch in self.interactivity, "Make a new branch first"
        self.interactivity[branch][tick_from] = {
            "dimension": str(self.dimension),
            "board": int(self.board),
            "place": str(self.place),
            "branch": branch,
            "tick_from": tick_from}

    def set_img(self, img, branch=None, tick_from=None):
        if branch is None:
            branch = self.closet.branch
        if tick_from is None:
            tick_from = self.closet.tick
        self.imagery[branch][tick_from] = {
            "dimension": str(self.dimension),
            "place": str(self.place),
            "board": int(self.board),
            "branch": branch,
            "tick_from": tick_from,
            "img": str(img)}

    def get_coords(self, branch=None, tick=None):
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        prev = None
        if branch not in self.coord_lst:
            return None
        for tick_from in self.coord_lst[branch]:
            if tick_from == tick:
                rd = self.coord_lst[branch][tick_from]
                return (rd["x"], rd["y"])
            elif tick_from > tick:
                break
            prev = tick_from
        if prev is None:
            return None
        else:
            rd = self.coord_lst[branch][prev]
            return (rd["x"], rd["y"])

    def set_coords(self, x, y, branch=None, tick_from=None):
        if branch is None:
            branch = self.closet.branch
        if tick_from is None:
            tick_from = self.closet.tick
        assert branch in self.coord_lst, "Make a new branch first"
        self.coord_lst[branch][tick_from] = {
            "dimension": str(self.dimension),
            "board": int(self.board),
            "place": str(self.place),
            "branch": branch,
            "tick_from": tick_from,
            "x": x,
            "y": y}

    def new_branch_coords(self, parent, branch, tick):
        prev = None
        started = False
        for tick_from in self.coord_lst[parent]:
            if tick_from >= tick:
                rd2 = dict(self.coord_lst[parent][tick_from])
                rd2["branch"] = branch
                if branch not in self.coord_lst:
                    self.coord_lst[branch] = {}
                self.coord_lst[branch][rd2["tick_from"]] = rd2
                if (
                        not started and prev is not None and
                        tick_from > tick and prev < tick):
                    rd3 = dict(self.coord_lst[branch][prev])
                    rd3["branch"] = branch
                    rd3["tick_from"] = tick
                    self.coord_lst[branch][rd3["tick_from"]] = rd3
                started = True
            prev = tick_from

    def new_branch(self, parent, branch, tick):
        self.new_branch_imagery(parent, branch, tick)
        self.new_branch_interactivity(parent, branch, tick)
        self.new_branch_coords(parent, branch, tick)

    @property
    def pos(self):
        cords = self.spot.get_coords()
        if cords is None:
            return (self.cheatx, self.cheaty)
        (x, y) = cords
        r = (
            x + self.drag_offset_x,
            y + self.drag_offset_y)
        (self.cheatx, self.cheaty) = r
        return r

    @property
    def in_view(self):
        return (
            self.window_right > 0 and
            self.window_left < self.window.width and
            self.window_top > 0 and
            self.window_bot < self.window.height)

    def dropped(self, x, y, button, modifiers):
        (oldx, oldy) = self.spot.coords
        self.spot.set_coords(
            oldx + self.drag_offset_x,
            oldy + self.drag_offset_y)
        self.drag_offset_x = 0
        self.drag_offset_y = 0

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.drag_offset_x += dx
        self.drag_offset_y += dy
        return self

    def overlaps(self, x, y):
        return (
            self.window_left < x and x < self.window_right and
            self.window_bot < y and y < self.window_top)

    def pass_focus(self):
        return self.viewport
