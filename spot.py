# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import (
    SaveableMetaclass,
    TerminableImg,
    TerminableInteractivity,
    BranchTicksIter,
    TabdictIterator)
from place import Place
from collections import defaultdict
from pyglet.sprite import Sprite
from pyglet.graphics import OrderedGroup
from pyglet.gl import GL_LINES
from logging import getLogger
from igraph import ALL


logger = getLogger(__name__)


__metaclass__ = SaveableMetaclass


"""Widgets to represent places. Pawns move around on top of these."""


class Spot(TerminableImg, TerminableInteractivity):
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
         ("dimension", "board", "place", "branch", "tick_from"),
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
         ("dimension", "board", "place", "branch", "tick_from"),
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
         ("dimension", "board", "place", "branch", "tick_from"),
         {"dimension, board": ("board", "dimension, i")},
         [])]

    def __init__(self, rumor, dimension, board, place):
        self.rumor = rumor
        self.dimension = dimension
        self.board = board
        self.place = place
        self.indefinite_imagery = {}
        for rd in TabdictIterator(
                self.rumor.tabdict["spot_img"][
                    str(self.dimension)][
                        int(self.board)][str(self.place)]):
            if rd["tick_to"] is None:
                self.indefinite_imagery[rd["branch"]] = rd["tick_from"]
                break
        self.indefinite_coords = {}
        for rd in TabdictIterator(
                self.rumor.tabdict["spot_coords"][
                    str(self.dimension)][
                        int(self.board)][str(self.place)]):
            if rd["tick_to"] is None:
                self.indefinite_coords[rd["branch"]] = rd["tick_from"]
                break
        self.indefinite_interactivity = {}
        for rd in TabdictIterator(self.rumor.tabdict["spot_interactive"][
                str(self.dimension)][
                    int(self.board)][str(self.place)]):
            if rd["tick_to"] is None:
                self.indefinite_interactivity[rd["branch"]] = rd["tick_from"]
                break
        self.drag_offset_x = 0
        self.drag_offset_y = 0

    def __str__(self):
        return str(self.place)

    def __repr__(self):
        return "Spot({0}[{1}].{2})".format(
            str(self.dimension), int(self.board), str(self.place))

    def __getattr__(self, attrn):
        if attrn == "vertex":
            return self.place.v
        elif attrn == "interactivity":
            return self.rumor.tabdict["spot_interactive"][
                str(self.dimension)][int(self.board)][str(self.place)]
        elif attrn == "imagery":
            return self.rumor.tabdict["spot_img"][
                str(self.dimension)][int(self.board)][str(self.place)]
        elif attrn == "coord_dict":
            return self.rumor.tabdict["spot_coords"][
                str(self.dimension)][int(self.board)][str(self.place)]
        elif attrn == 'interactive':
            return self.is_interactive()
        elif attrn == 'img':
            return self.get_img()
        elif attrn == 'coords':
            (x, y) = self.get_coords()
            return (
                x + self.drag_offset_x,
                y + self.drag_offset_y)
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
        elif attrn == "board_left":
            return self.x - self.rx
        elif attrn == "board_bot":
            return self.y - self.ry
        elif attrn == "board_top":
            return self.board_bot + self.height
        elif attrn == "board_right":
            return self.board_left + self.width
        else:
            raise AttributeError(
                "Spot instance has no such attribute: " +
                attrn)

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
            "place": str(self.place),
            "branch": branch,
            "tick_from": tick_from,
            "tick_to": tick_to}
        if tick_to is None:
            self.indefinite_interactivity[branch] = tick_from

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
            "place": str(self.place),
            "board": int(self.board),
            "branch": branch,
            "tick_from": tick_from,
            "tick_to": tick_to,
            "img": str(img)}
        if tick_to is None:
            self.indefinite_imagery[branch] = tick_from

    def get_coords(self, branch=None, tick=None):
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        if branch not in self.coord_dict:
            import pdb
            pdb.set_trace()
            return None
        if (
                branch in self.indefinite_coords and
                tick >= self.indefinite_coords[branch]):
            rd = self.coord_dict[branch][self.indefinite_coords[branch]]
            return (rd["x"], rd["y"])
        for rd in TabdictIterator(self.coord_dict):
            if rd["tick_from"] <= tick and tick <= rd["tick_to"]:
                return (rd["x"], rd["y"])
        import pdb
        pdb.set_trace()
        return None

    def set_coords(self, x, y, branch=None, tick_from=None, tick_to=None):
        print "spot coords set to {0}, {1}".format(x, y)
        if branch is None:
            branch = self.rumor.branch
        if tick_from is None:
            tick_from = self.rumor.tick
        if branch not in self.coord_dict:
            self.coord_dict[branch] = {}
        if branch in self.indefinite_coords:
            itf = self.indefinite_coords[branch]
            rd = self.coord_dict[branch][itf]
            if itf < tick_from:
                # You have cut off an indefinite coord
                rd["tick_to"] = tick_from - 1
                self.coord_dict[branch][itf] = rd
                del self.indefinite_coords[branch]
            elif itf < tick_to:
                # You have overwritten an indefinite coord
                del self.coord_dict[branch][itf]
                del self.indefinite_coords[branch]
            elif itf == tick_to:
                # You have extended an indefinite coord, backward in time
                del self.coord_dict[branch][itf]
                tick_to = None
        self.coord_dict[branch][tick_from] = {
            "dimension": str(self.dimension),
            "board": int(self.board),
            "place": str(self.place),
            "branch": branch,
            "tick_from": tick_from,
            "tick_to": tick_to,
            "x": x,
            "y": y}
        if tick_to is None:
            self.indefinite_coords[branch] = tick_from

    def new_branch_coords(self, parent, branch, tick):
        for rd in TabdictIterator(self.coord_dict[parent]):
            if rd["tick_to"] >= tick or rd["tick_to"] is None:
                if rd["tick_from"] < tick:
                    self.set_coords(
                        rd["x"], rd["y"], branch, tick, rd["tick_to"])
                else:
                    self.set_coords(
                        rd["x"], rd["y"], branch,
                        rd["tick_from"], rd["tick_to"])

    def new_branch(self, parent, branch, tick):
        self.new_branch_imagery(parent, branch, tick)
        self.new_branch_interactivity(parent, branch, tick)
        self.new_branch_coords(parent, branch, tick)


class SpotWidget:
    def __init__(self, viewport, spot):
        self.viewport = viewport
        self.supergroup = OrderedGroup(0, self.viewport.spotgroup)
        self.spritegroup = OrderedGroup(0, self.supergroup)
        self.boxgroup = OrderedGroup(1, self.supergroup)
        self.batch = self.viewport.batch
        self.spot = spot
        self.sprite = None
        self.vertlist = None

    def __getattr__(self, attrn):
        if attrn == "viewport_left":
            return (
                self.board_left +
                self.viewport.view_left)
        elif attrn == "viewport_bot":
            return self.board_bot + self.viewport.view_bot
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
        elif attrn == "in_view":
            return (
                self.viewport_right > 0 and
                self.viewport_left < self.viewport.width and
                self.viewport_top > 0 and
                self.viewport_bot < self.viewport.height)
        elif attrn == "selected":
            return self in self.viewport.window.selected
        elif attrn == "hovered":
            return self is self.viewport.window.hovered
        elif attrn == "pressed":
            return self is self.viewport.window.pressed
        elif attrn == "grabbed":
            return self is self.window.grabbed
        elif attrn in ("place", "img", "board", "vert",
                       "visible", "interactive", "board_left",
                       "board_right", "board_top", "board_bot"):
            return getattr(self.spot, attrn)
        else:
            raise AttributeError(
                "SpotWidget instance has no attribute " + attrn)

    def dropped(self, x, y, button, modifiers):
        self.spot.set_coords(*self.spot.coords)
        self.spot.drag_offset_x = 0
        self.spot.drag_offset_y = 0

    def move_with_mouse(self, x, y, dx, dy, buttons, modifiers):
        """Remember where exactly I was grabbed, then move around with the
mouse, always keeping the same relative position with respect to the
mouse."""
        self.spot.drag_offset_x += dx
        self.spot.drag_offset_y += dy

    def overlaps(self, x, y):
        return (
            self.window_left < x and x < self.window_right and
            self.window_bot < y and y < self.window_top)

    def pass_focus(self):
        return self.viewport

    def draw(self):
        if self.visible and self.in_view:
            try:
                self.sprite.x = self.viewport_left
                self.sprite.y = self.viewport_bot
            except AttributeError:
                self.sprite = Sprite(
                    self.img.tex,
                    self.viewport_left,
                    self.viewport_bot,
                    batch=self.batch,
                    group=self.spritegroup)
        else:
            try:
                self.sprite.delete()
            except:
                pass
        if self.selected:
            yelo = (255, 255, 0, 0)
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
                    (0, 1, 2, 3, 0),
                    ('v2i', points),
                    ('c4b', colors))
        else:
            try:
                self.vertlist.delete()
            except:
                pass
            self.vertlist = None

    def delete(self):
        for e in self.place.incident(mode=ALL):
            for arrow in e.arrows:
                arrow.delete()
        try:
            self.sprite.delete()
        except:
            pass
