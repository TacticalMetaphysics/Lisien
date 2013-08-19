# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import (
    SaveableMetaclass,
    TerminableImg,
    TerminableInteractivity,
    BranchTicksIter)
from collections import defaultdict
from pyglet.sprite import Sprite
from logging import getLogger


logger = getLogger(__name__)


__metaclass__ = SaveableMetaclass


"""Widget representing things that move about from place to place."""


class Pawn(TerminableImg, TerminableInteractivity):
    """A token to represent something that moves about between places."""
    tables = [
        ("pawn_img",
         {"dimension": "text not null default 'Physical'",
          "board": "integer not null default 0",
          "thing": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "tick_to": "integer default null",
          "img": "text not null default 'default_pawn'"},
         ("dimension", "board", "thing", "tick_from"),
         {"dimension, board": ("board", "dimension, i"),
          "dimension, thing": ("thing_location", "dimension, name"),
          "img": ("img", "name")},
         []),
        ("pawn_interactive",
         {"dimension": "text not null default 'Physical'",
          "board": "integer not null default 0",
          "thing": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "tick_to": "integer default null"},
         ("dimension", "board", "thing", "tick_from"),
         {"dimension, board": ("board", "dimension, i"),
          "dimension, thing": ("thing_location", "dimension, name")},
         [])]

    def __init__(self, board, thing):
        """Return a pawn on the board for the given dimension, representing
the given thing with the given image. It may be visible or not,
interactive or not.

        """
        self.board = board
        self.rumor = self.board.rumor
        self.thing = thing
        self.imagery = defaultdict(dict)
        self.indefinite_imagery = {}
        self.interactivity = defaultdict(dict)
        self.indefinite_interactivity = {}
        self.grabpoint = None
        self.sprite = None
        self.oldstate = None
        self.tweaks = 0
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.selectable = True
        self.box_edges = (None, None, None, None)
        self.calcol = None

    def __str__(self):
        return str(self.thing)

    def __getattr__(self, attrn):
        if attrn == 'img':
            return self.get_img()
        elif attrn == 'visible':
            return self.img is not None
        elif attrn == 'coords':
            coords = self.get_coords()
            return coords
        elif attrn == 'x':
            coords = self.coords
            return coords[0]
        elif attrn == 'y':
            coords = self.coords
            return coords[1]
        elif attrn == 'width':
            return self.img.width
        elif attrn == 'height':
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
        else:
            raise AttributeError(
                "Pawn instance has no such attribute: " +
                attrn)

    def __setattr__(self, attrn, val):
        if attrn == "img":
            self.set_img(val)
        elif attrn == "interactive":
            self.set_interactive(val)
        else:
            super(Pawn, self).__setattr__(attrn, val)

    def get_coords(self, branch=None, tick=None):
        loc = self.thing.get_location(branch, tick)
        if loc is None:
            return None
        if hasattr(loc, 'dest'):
            origspot = self.board.get_spot(loc.orig)
            destspot = self.board.get_spot(loc.dest)
            (ox, oy) = origspot.get_coords(branch, tick)
            (dx, dy) = destspot.get_coords(branch, tick)
            ox += origspot.drag_offset_x
            dx += destspot.drag_offset_x
            oy += origspot.drag_offset_y
            dy += destspot.drag_offset_y
            prog = self.thing.get_progress(branch, tick)
            odx = dx - ox
            ody = dy - oy
            return (int(ox + odx * prog) + self.window.offset_x,
                    int(oy + ody * prog) + self.window.offset_y)
        elif str(loc) in self.board.spotdict:
            spot = self.board.get_spot(loc)
            coords = spot.get_coords(branch, tick)
            if coords is None:
                return None
            (x, y) = coords
            return (
                x + spot.drag_offset_x + self.window.offset_x,
                y + spot.drag_offset_y + self.window.offset_y)
        else:
            return None

    def get_tabdict(self):
        return {
            "pawn_img": [
                {
                    "dimension": str(self.board.dimension),
                    "board": int(self.board),
                    "thing": str(self.thing),
                    "branch": branch,
                    "tick_from": tick_from,
                    "tick_to": tick_to,
                    "img": str(img)}
                for (branch, tick_from, tick_to, img) in
                BranchTicksIter(self.imagery)],
            "pawn_interactive": [
                {
                    "dimension": str(self.board.dimension),
                    "board": int(self.board),
                    "thing": str(self.thing),
                    "branch": branch,
                    "tick_from": tick_from,
                    "tick_to": tick_to}
                for (branch, tick_from, tick_to) in
                BranchTicksIter(self.interactivity)]}

    def new_branch(self, parent, branch, tick):
        self.new_branch_imagery(parent, branch, tick)
        self.new_branch_interactivity(parent, branch, tick)

    def dropped(self, x, y):
        spot = self.board.get_spot_at(x, y)
        if spot is not None:
            # if the thing is in a *portal*, it is traveling
            self.thing.journey_to(spot.place)
        try:
            self.calcol.regen_cells()
            self.calcol.tweaks += 1
        except:
            pass


class PawnWidget:
    selectable = True

    def __init__(self, viewport, pawn):
        self.pawn = pawn
        self.viewport = viewport
        self.window = self.viewport.window
        self.drag_offset_x = 0
        self.drag_offset_y = 0

    def __getattr__(self, attrn):
        if attrn == "board_left":
            return self.pawn.x
        elif attrn == "board_bot":
            return self.pawn.y
        elif attrn == "board_top":
            return self.pawn.y + self.pawn.height
        elif attrn == "board_right":
            return self.pawn.x + self.pawn.width
        elif attrn == "window_left":
            return self.board_left + self.viewport.window_left
        elif attrn == "window_right":
            return self.board_right + self.viewport.window_left
        elif attrn == "window_bot":
            return self.board_bot + self.viewport.window_bot
        elif attrn == "window_top":
            return self.board_top + self.viewport.window_bot
        elif attrn in ("selected", "highlit"):
            return self in self.window.selected
        elif attrn == "hovered":
            return self is self.window.hovered
        elif attrn == "pressed":
            return self is self.window.pressed
        elif attrn == "grabbed":
            return self is self.window.grabbed
        elif attrn in (
                "img", "visible", "interactive",
                "width", "height"):
            return getattr(self.pawn, attrn)
        else:
            raise AttributeError(
                "PawnWidget instance has no attribute " + attrn)

    def hover(self, x, y):
        return self

    def move_with_mouse(self, x, y, dx, dy, buttons, modifiers):
        self.drag_offset_x += dx
        self.drag_offset_y += dy

    def dropped(self, x, y, button, modifiers):
        """When dropped on a spot, if my thing doesn't have anything else to
do, make it journey there.

If it DOES have anything else to do, make the journey in another branch.

        """
        logger.debug("Dropped the pawn %s at (%d,%d)",
                     str(self), x, y)
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.pawn.dropped(x, y)

    def delete(self):
        try:
            self.sprite.delete()
        except:
            pass

    def draw(self, batch, group):
        if self.coords is not None:
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
        if self.selected:
            yelo = (255, 255, 0, 0)
            self.box_edges = self.window.draw_box(
                self.window_left,
                self.window_top,
                self.window_right,
                self.window_bot,
                yelo,
                group,
                self.box_edges)
        else:
            for edge in self.box_edges:
                try:
                    edge.delete()
                except (AttributeError, AssertionError):
                    pass
            self.box_edges = (None, None, None, None)

    def overlaps(self, x, y):
        if self.visible and self.interactive and self.in_window:
            (myx, myy) = self.get_coords()
            return (
                x > myx and
                y > myy and
                x - myx < self.width and
                y - myy < self.height)
        else:
            return False

    def select(self):
        if self.calcol is None:
            sensical = self.window.sensible_calendar_for(self.thing)
            self.calcol = sensical.mkcol(self.thing.locations)
            self.calcol.visible = True

    def unselect(self):
        if self.calcol is not None:
            self.calcol.delete()
            self.calcol = None
