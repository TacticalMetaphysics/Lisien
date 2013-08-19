# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import (
    SaveableMetaclass,
    TerminableImg,
    TerminableInteractivity,
    TerminableCoords,
    BranchTicksIter)
from collections import defaultdict
from pyglet.sprite import Sprite
from logging import getLogger
from igraph import ALL


logger = getLogger(__name__)


__metaclass__ = SaveableMetaclass


"""Widgets to represent places. Pawns move around on top of these."""


class Spot(
        TerminableImg, TerminableInteractivity,
        TerminableCoords):
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

    def get_tabdict(self):
        return {
            "spot_img": [
                {
                    "dimension": str(self.spot.dimension),
                    "place": str(self.spot.place),
                    "board": int(self.spot.board),
                    "branch": branch,
                    "tick_from": tick_from,
                    "tick_to": tick_to,
                    "img": str(img)}
                for (branch, tick_from, tick_to, img) in
                BranchTicksIter(self.spot.imagery)],
            "spot_interactive": [
                {
                    "dimension": str(self.spot.dimension),
                    "place": str(self.spot.place),
                    "board": int(self.spot.board),
                    "branch": branch,
                    "tick_from": tick_from,
                    "tick_to": tick_to}
                for (branch, tick_from, tick_to) in
                BranchTicksIter(self.spot.interactivity)],
            "spot_coords": [
                {
                    "dimension": str(self.spot.dimension),
                    "place": str(self.spot.place),
                    "board": int(self.spot.board),
                    "branch": branch,
                    "tick_from": tick_from,
                    "tick_to": tick_to,
                    "x": x,
                    "y": y}
                for (branch, tick_from, tick_to, x, y) in
                BranchTicksIter(self.spot.coord_dict)]}

    def __init__(self, board, vert):
        self.board = board
        self.rumor = self.board.rumor
        self.vert = vert
        self.interactivity = defaultdict(dict)
        self.imagery = defaultdict(dict)
        self.coord_dict = defaultdict(dict)
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
        elif attrn == 'coords':
            return self.get_coords()
        elif attrn == 'x':
            return self.coords[0]
        elif attrn == 'y':
            return self.coords[1]
        elif attrn == 'width':
            if self.img is None:
                return 0
            else:
                return self.img.width
        elif attrn == 'height':
            if self.img is None:
                return 0
            else:
                return self.img.height
        elif attrn == 'rx':
            return self.width / 2
        elif attrn == 'ry':
            return self.height / 2
        elif attrn == 'r':
            if self.rx > self.ry:
                return self.rx
            else:
                return self.ry
        elif attrn == 'visible':
            return self.img is not None
        elif hasattr(self, 'saver') and hasattr(self.saver, attrn):
            return getattr(self.saver, attrn)
        else:
            raise AttributeError(
                "Spot instance has no such attribute: " +
                attrn)

    def new_branch(self, parent, branch, tick):
        self.new_branch_imagery(parent, branch, tick)
        self.new_branch_interactivity(parent, branch, tick)
        self.new_branch_coords(parent, branch, tick)


class SpotWidget:
    def __init__(self, viewport, spot):
        self.viewport = viewport
        self.spot = spot

    def __getattr__(self, attrn):
        if attrn == "board_left":
            return self.spot.x
        elif attrn == "board_bot":
            return self.spot.y
        elif attrn == "board_top":
            return self.board_bot + self.spot.height
        elif attrn == "board_right":
            return self.board_left + self.spot.width
        elif attrn == "viewport_left":
            return self.board_left + self.viewport.offset_x
        elif attrn == "viewport_bot":
            return self.board_bot + self.viewport.offset_y
        elif attrn == "viewport_top":
            return self.viewport_bot + self.spot.height
        elif attrn == "viewport_right":
            return self.viewport_left + self.spot.width
        elif attrn == "window_left":
            return self.viewport_left + self.viewport.window_left
        elif attrn == "window_bot":
            return self.viewport_bot + self.viewport.window_bot
        elif attrn == "window_top":
            return self.viewport_top + self.viewport.window_bot
        elif attrn == "window_right":
            return self.viewport_right + self.viewport.window_left
        elif attrn in ("place", "img", "board", "get_img", "set_img",
                       "get_coords", "set_coords", "is_interactive",
                       "set_interactive", "new_branch"):
            return getattr(self.spot, attrn)
        else:
            raise AttributeError(
                "SpotWidget instance has no attribute " + attrn)

    def dropped(self, x, y, button, modifiers):
        c = self.get_coords()
        newx = c[0] + self.drag_offset_x
        newy = c[1] + self.drag_offset_y
        self.set_coords(newx, newy)
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
        if self.coords is None:
            return False
        (myx, myy) = self.window_coords
        return (
            self.visible and
            self.interactive and
            abs(myx - x) < self.rx and
            abs(myy - y) < self.ry)

    def draw(self, batch, group):
        if self.visible and self.in_window:
            try:
                self.sprite.x = self.window_left
                self.sprite.y = self.window_bot
            except AttributeError:
                self.sprite = Sprite(
                    self.img.tex,
                    self.window_left,
                    self.window_bot,
                    batch=batch,
                    group=group)
        else:
            try:
                self.sprite.delete()
            except:
                pass
        if self.selected:
            yelo = (255, 255, 0, 0)
            self.box_edges = self.window.draw_box(
                self.window_left,
                self.window_top,
                self.window_right,
                self.window_bot,
                yelo,
                self.window.topgroup,
                self.box_edges)
        else:
            for vertls in self.box_edges:
                try:
                    vertls.delete()
                except:
                    pass
            self.box_edges = (None, None, None, None)

    def delete(self):
        for e in self.place.incident(mode=ALL):
            for arrow in e.arrows:
                arrow.delete()
        try:
            self.sprite.delete()
        except:
            pass
