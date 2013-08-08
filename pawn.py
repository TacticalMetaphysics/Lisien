# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import (
    SaveableMetaclass,
    TerminableImg,
    TerminableInteractivity,
    BranchTicksIter)
from pyglet.sprite import Sprite
from logging import getLogger
from igraph import IN


logger = getLogger(__name__)


"""Widget representing things that move about from place to place."""


class Pawn(object, TerminableImg, TerminableInteractivity):
    """A token to represent something that moves about between places."""
    __metaclass__ = SaveableMetaclass
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
        self.window = self.board.window
        self.rumor = self.window.rumor
        self.thing = thing
        self.imagery = {}
        self.indefinite_imagery = {}
        self.interactivity = {}
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
        elif attrn == 'highlit':
            return self in self.gw.selected
        elif attrn == 'interactive':
            return self.is_interactive()
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
            coords = self.coords
            if coords is None:
                return None
            return coords[0]
        elif attrn == 'y':
            coords = self.coords
            if coords is None:
                return None
            return coords[1]
        elif attrn == 'window_left':
            coords = self.coords
            if coords is None:
                return None
            return coords[0] + self.drag_offset_x
        elif attrn == 'window_bot':
            coords = self.coords
            if coords is None:
                return None
            return coords[1] + self.drag_offset_y
        elif attrn == 'width':
            return self.img.width
        elif attrn == 'height':
            return self.img.height
        elif attrn == 'window_right':
            return self.window_left + self.width
        elif attrn == 'window_top':
            return self.window_bot + self.height
        elif attrn == 'in_window':
            return (
                self.coords is not None and
                self.window_right > 0 and
                self.window_left < self.window.width and
                self.window_top > 0 and
                self.window_bot < self.window.height)
        elif attrn == 'rx':
            return self.width / 2
        elif attrn == 'ry':
            return self.height / 2
        elif attrn == 'r':
            if self.rx > self.ry:
                return self.rx
            else:
                return self.ry
        elif attrn == 'state':
                return self.get_state_tup()
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

    def __eq__(self, other):
        """Essentially, compare the state tuples of the two pawns."""
        return self.state == other.state

    def get_state_tup(self, branch=None, tick=None):
        """Return a tuple containing everything you might need to draw me."""
        return (
            self.get_img(branch, tick),
            self.interactive,
            self.grabpoint,
            self.hovered,
            self.get_coords(branch, tick),
            self.window.view_left,
            self.window.view_bot,
            self.tweaks)

    def move_with_mouse(self, x, y, dx, dy, buttons, modifiers):
        self.drag_offset_x += dx
        self.drag_offset_y += dy
        self.tweaks += 1

    def dropped(self, x, y, button, modifiers):
        """When dropped on a spot, if my thing doesn't have anything else to
do, make it journey there.

If it DOES have anything else to do, make the journey in another branch.

        """
        logger.debug("Dropped the pawn %s at (%d,%d)",
                     str(self), x, y)
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        spot = self.board.get_spot_at(x, y)
        if spot is not None:
            # if the thing is in a *portal*, it is traveling
            if hasattr(self.thing.loc, 'e'):
                self.rumor.split_branch()
            self.thing.journey_to(spot.place)
        try:
            self.calcol.regen_cells()
            self.calcol.tweaks += 1
        except:
            pass

    def draw(self):
        newstate = self.get_state_tup()
        if newstate in self.window.onscreen:
            return
        self.window.onscreen.discard(self.oldstate)
        self.window.onscreen.add(newstate)
        self.oldstate = newstate
        if self.visible and self.in_window:
            try:
                self.sprite.x = self.window_left
                self.sprite.y = self.window_bot
            except AttributeError:
                self.sprite = Sprite(
                    self.img.tex,
                    self.window_left,
                    self.window_bot,
                    batch=self.window.batch,
                    group=self.window.pawngroup)
        if self.selected:
            yelo = (255, 255, 0, 0)
            self.box_edges = self.window.draw_box(
                self.window_left,
                self.window_top,
                self.window_right,
                self.window_bot,
                yelo,
                self.window.higroup,
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
            

    def get_coords(self, branch=None, tick=None):
        loc = self.thing.get_location(branch, tick)
        if loc is None:
            return None
        if hasattr(loc, 'dest'):
            (ox, oy) = loc.orig.spots[int(self.board)].get_coords(branch, tick)
            (dx, dy) = loc.dest.spots[int(self.board)].get_coords(branch, tick)
            ox += loc.orig.spots[int(self.board)].drag_offset_x
            dx += loc.dest.spots[int(self.board)].drag_offset_x
            oy += loc.orig.spots[int(self.board)].drag_offset_y
            dy += loc.dest.spots[int(self.board)].drag_offset_y
            prog = self.thing.get_progress(branch, tick)
            odx = dx - ox
            ody = dy - oy
            return (int(ox + odx * prog) + self.window.offset_x,
                    int(oy + ody * prog) + self.window.offset_y)
        elif hasattr(loc, 'spots'):
            spot = loc.spots[int(self.board)]
            swico = spot.get_coords(branch, tick)
            if swico is None:
                return None
            else:
                (x, y) = swico
                return (
                    x + spot.drag_offset_x + self.window.offset_x,
                    y + spot.drag_offset_y + self.window.offset_y)
        else:
            raise Exception("When trying to get the coordinates of the pawn for {0}, I found that its location {1} had no spots.".format(str(self), str(loc)))

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
