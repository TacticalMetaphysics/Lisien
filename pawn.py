# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import (
    SaveableMetaclass,
    TerminableImg,
    TerminableInteractivity,
    BranchTicksIter,
    LoadError,
    TabdictIterator)
from collections import defaultdict
from pyglet.sprite import Sprite
from pyglet.graphics import OrderedGroup
from pyglet.gl import GL_LINES
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
         ("dimension", "board", "thing", "branch", "tick_from"),
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
         ("dimension", "board", "thing", "branch", "tick_from"),
         {"dimension, board": ("board", "dimension, i"),
          "dimension, thing": ("thing_location", "dimension, name")},
         [])]
    loaded_keys = set()

    def __init__(self, rumor, dimension, board, thing, td):
        """Return a pawn on the board for the given dimension, representing
the given thing with the given image. It may be visible or not,
interactive or not.

        """
        dimn = str(dimension)
        boardi = int(board)
        thingn = str(thing)
        if (dimn, boardi, thingn) in Pawn.loaded_keys:
            raise LoadError("Pawn already loaded: {0}[{1}].{2}".format(dimn, boardi, thingn))
        else:
            Pawn.loaded_keys.add((dimn, boardi, thingn))
        self.rumor = rumor
        self._tabdict = td
        self.dimension = dimension
        self.board = board
        self.thing = thing
        self.indefinite_imagery = {}
        self.indefinite_interactivity = {}
        imgns = set()
        for rd in TabdictIterator(self._tabdict["pawn_img"]):
            imgns.add(rd["img"])
            if rd["tick_to"] is None:
                self.indefinite_imagery[rd["branch"]] = rd["tick_from"]
        imgdict = self.rumor.get_imgs(imgns)
        for rd in TabdictIterator(self._tabdict["pawn_interactive"]):
            if rd["tick_to"] is None:
                self.indefinite_interactivity[rd["branch"]] = rd["tick_from"]
        self.grabpoint = None
        self.sprite = None
        self.oldstate = None
        self.tweaks = 0
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        self.selectable = True
        self.vertlist = None

    def __str__(self):
        return str(self.thing)

    def __getattr__(self, attrn):
        if attrn == "imagery":
            return self._tabdict["pawn_img"][str(self.dimension)][int(self.board)][str(self.thing)]
        elif attrn == "interactivity":
            return self._tabdict["pawn_interactive"][str(self.dimension)][int(self.board)][str(self.thing)]
        elif attrn == 'img':
            return self.get_img()
        elif attrn == 'visible':
            return self.img is not None
        elif attrn == 'coords':
            coords = self.get_coords()
            return coords
        elif attrn == 'x':
            return self.coords[0]
        elif attrn == 'y':
            return self.coords[1]
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

    def set_img(self, img, branch=None, tick_from=None, tick_to=None):
        if branch is None:
            branch = self.rumor.branch
        if tick_from is None:
            tick_from = self.rumor.tick
        if branch in self.indefinite_imagery:
            indef_start = self.indefinite_imagery[branch]
            indef_rd = self.imagery[branch][indef_start]
            if tick_from > indef_start:
                del self.indefinite_imagery[branch]
                indef_rd["tick_to"] = tick_from - 1
                self.imagery[branch][indef_start] = indef_rd
            elif tick_to is None or tick_to > indef_start:
                del self.indefinite_imagery[branch]
                del self.imagery[branch][indef_start]
            elif tick_to == indef_start and str(img) == indef_rd["img"]:
                indef_rd["tick_from"] = tick_from
                return
        self.imagery[branch][tick_from] = {
            "dimension": str(self.dimension),
            "thing": str(self.thing),
            "board": str(self.board),
            "branch": branch,
            "tick_from": tick_from,
            "tick_to": tick_to,
            "img": str(img)}

    def set_interactive(self, branch=None, tick_from=None, tick_to=None):
        if branch is None:
            branch = self.rumor.branch
        if tick_from is None:
            tick_from = self.rumor.tick
        if branch in self.indefinite_interactivity:
            indef_start = self.indefinite_interactivity[branch]
            indef_rd = self.interactivity[branch][indef_start]
            if tick_from > indef_start:
                indef_rd["tick_to"] = tick_from - 1
                del self.indefinite_interactivity[branch]
            elif tick_to is None or tick_to > indef_start:
                del self.interactivity[branch][indef_start]
                del self.indefinite_interactivity[branch]
            elif tick_to == indef_start:
                indef_rd["tick_from"] = tick_from
                return
        self.interactivity[branch][tick_from] = {
            "dimension": str(self.dimension),
            "board": int(self.board),
            "thing": str(self.thing),
            "branch": branch,
            "tick_from": tick_from,
            "tick_to": tick_to}
        if tick_to is None:
            self.indefinite_interactivity[branch] = tick_from

    def get_coords(self, branch=None, tick=None):
        loc = self.thing.get_location(branch, tick)
        if loc is None:
            import pdb
            pdb.set_trace()
            return None
        if hasattr(loc, 'dest'):
            origspot = self.board.get_spot(loc.orig)
            destspot = self.board.get_spot(loc.dest)
            (ox, oy) = origspot.get_coords(branch, tick)
            (dx, dy) = destspot.get_coords(branch, tick)
            prog = self.thing.get_progress(branch, tick)
            odx = dx - ox
            ody = dy - oy
            return (int(ox + odx * prog) + self.drag_offset_x,
                    int(oy + ody * prog) + self.drag_offset_y)
        elif str(loc) in self.board.spotdict:
            spot = self.board.get_spot(loc)
            coords = spot.get_coords(branch, tick)
            return (
                coords[0] + self.drag_offset_x,
                coords[1] + self.drag_offset_y)
        else:
            import pdb
            pdb.set_trace()
            return None

    def new_branch(self, parent, branch, tick):
        self.new_branch_imagery(parent, branch, tick)
        self.new_branch_interactivity(parent, branch, tick)


class PawnWidget:
    selectable = True

    def __init__(self, viewport, pawn):
        self.pawn = pawn
        self.rumor = self.pawn.rumor
        self.viewport = viewport
        self.batch = self.viewport.batch
        self.spritegroup = OrderedGroup(0, self.viewport.pawngroup)
        self.boxgroup = OrderedGroup(1, self.viewport.pawngroup)
        self.window = self.viewport.window
        self.calcol = None

    def __getattr__(self, attrn):
        if attrn == "board_left":
            return self.pawn.x
        elif attrn == "board_bot":
            return self.pawn.y
        elif attrn == "board_top":
            return self.pawn.y + self.pawn.height
        elif attrn == "board_right":
            return self.pawn.x + self.pawn.width
        elif attrn == "viewport_left":
            return self.board_left + self.viewport.offset_x
        elif attrn == "viewport_right":
            return self.board_right + self.viewport.offset_x
        elif attrn == "viewport_bot":
            return self.board_bot + self.viewport.offset_y
        elif attrn == "viewport_top":
            return self.board_top + self.viewport.offset_y
        elif attrn == "window_left":
            return self.viewport_left + self.viewport.window_left
        elif attrn == "window_right":
            return self.viewport_right + self.viewport.window_left
        elif attrn == "window_bot":
            return self.viewport_bot + self.viewport.window_bot
        elif attrn == "window_top":
            return self.viewport_top + self.viewport.window_bot
        elif attrn in ("selected", "highlit"):
            return self in self.window.selected
        elif attrn == "hovered":
            return self is self.window.hovered
        elif attrn == "pressed":
            return self is self.window.pressed
        elif attrn == "grabbed":
            return self is self.window.grabbed
        elif attrn == "in_view":
            return (
                self.viewport_right > 0 and
                self.viewport_left < self.viewport.width and
                self.viewport_top > 0 and
                self.viewport_bot < self.viewport.height)
        elif attrn in (
                "img", "visible", "interactive",
                "width", "height", "thing"):
            return getattr(self.pawn, attrn)
        else:
            raise AttributeError(
                "PawnWidget instance has no attribute " + attrn)

    def hover(self, x, y):
        return self

    def move_with_mouse(self, x, y, dx, dy, buttons, modifiers):
        self.pawn.drag_offset_x += dx
        self.pawn.drag_offset_y += dy

    def dropped(self, x, y, button, modifiers):
        """When dropped on a spot, if my thing doesn't have anything else to
do, make it journey there.

If it DOES have anything else to do, make the journey in another branch.

        """
        spotto = None
        for spot in self.viewport.board.spots:
            if (
                    self.viewport_left < spot.x and
                    spot.x < self.viewport_right and
                    self.viewport_bot < spot.y and
                    spot.y < self.viewport_top):
                spotto = spot
                break
        if spotto is not None:
            self.thing.journey_to(spotto.place)
            try:
                self.calcol.regen_cells()
            except:
                pass
        self.pawn.drag_offset_x = 0
        self.pawn.drag_offset_y = 0

    def delete(self):
        try:
            self.sprite.delete()
        except:
            pass

    def draw(self):
        if (
                None in (self.viewport_left, self.viewport_bot) or
                self.img is None):
            return
        try:
            self.sprite.x = self.viewport_left
            self.sprite.y = self.viewport_bot
        except AttributeError:
            self.sprite = Sprite(
                self.img.tex,
                self.window_left,
                self.window_bot,
                batch=self.batch,
                group=self.spritegroup)
        if self.selected:
            yelo = (255, 255, 0, 255)
            colors = yelo * 4
            points = (
                self.viewport_left, self.viewport_top,
                self.viewport_right, self.viewport_top,
                self.viewport_right, self.viewport_bot,
                self.viewport_left, self.viewport_bot)
            try:
                self.vertlist.vertices = points
            except:
                self.vertlist = self.batch.add_indexed(
                    4,
                    GL_LINES,
                    self.boxgroup,
                    (0, 1, 1, 2, 2, 3, 3, 0),
                    ('v2i', points),
                    ('c4B', colors))
        else:
            try:
                self.vertlist.delete()
            except:
                pass
            self.vertlist = None

    def overlaps(self, x, y):
        return (
            self.viewport_left < x and x < self.viewport_right and
            self.viewport_bot < y and y < self.viewport_top)

    def pass_focus(self):
        return self.viewport

    def select(self):
        if self.calcol is not None:
            self.calcol.visible = True

    def unselect(self):
        if self.calcol is not None:
            self.calcol.visible = False
