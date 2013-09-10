## This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import (
    SaveableMetaclass,
    TerminableImg,
    TerminableInteractivity,
    BranchTicksIter,
    SkeletonIterator)
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

    def get_r(self):
        rx = self.rx
        ry = self.ry
        if rx > ry:
            return rx
        else:
            return ry

    atrdic = {
        "vertex": lambda self: self.place.v,
        "interactive": lambda self: self.is_interactive(),
        "img": lambda self: self.get_img(),
        "coords": lambda self: self.get_coords(),
        "x": lambda self: self.coords[0],
        "y": lambda self: self.coords[1],
        "width": lambda self: {
            True: lambda: 0, False: lambda: self.get_img().width
        }[self.get_img() is None](),
        "height": lambda self: {
            True: lambda: 0, False: lambda: self.get_img().height
        }[self.get_img() is None](),
        "rx": lambda self: self.width / 2,
        "ry": lambda self: self.height / 2,
        "r": lambda self: self.get_r(),
        "visible": lambda self: self.get_img() is not None,
        "board_left": lambda self: self.x - self.rx,
        "board_bot": lambda self: self.y - self.ry,
        "board_top": lambda self: self.y + self.ry,
        "board_right": lambda self: self.x + self.rx}

    def __init__(self, closet, dimension, board, place):
        self.closet = closet
        self.dimension = dimension
        self.board = board
        self.place = place
        self.vert = self.place.v
        self.coord_lst = self.closet.skeleton["spot_coords"][
            str(self.dimension)][
                int(self.board)][str(self.place)]
        self.interactivity = self.closet.skeleton["spot_interactive"][
            str(self.dimension)][
                int(self.board)][str(self.place)]
        self.imagery = self.closet.skeleton["spot_img"][
            str(self.dimension)][
                int(self.board)][str(self.place)]
        self.indefinite_imagery = {}
        for rd in SkeletonIterator(
                self.closet.skeleton["spot_img"][
                    str(self.dimension)][
                        int(self.board)][str(self.place)]):
            if rd["tick_to"] is None:
                self.indefinite_imagery[rd["branch"]] = rd["tick_from"]
        self.indefinite_coords = {}
        for rd in SkeletonIterator(
            self.closet.skeleton["spot_coords"][
                str(self.dimension)][
                int(self.board)][str(self.place)]):
            if rd["tick_to"] is None:
                self.indefinite_coords[rd["branch"]] = rd["tick_from"]
        self.indefinite_interactivity = {}
        for rd in SkeletonIterator(self.closet.skeleton["spot_interactive"][
                str(self.dimension)][
                    int(self.board)][str(self.place)]):
            if rd["tick_to"] is None:
                self.indefinite_interactivity[rd["branch"]] = rd["tick_from"]
        self.drag_offset_x = 0
        self.drag_offset_y = 0

    def __str__(self):
        return str(self.place)

    def __repr__(self):
        return "Spot({0}[{1}].{2})".format(
            str(self.dimension), int(self.board), str(self.place))

    def __getattr__(self, attrn):
        try:
            return Spot.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                "Spot instance has no such attribute: " +
                attrn)

    def set_interactive(self, branch=None, tick_from=None, tick_to=None):
        if branch is None:
            branch = self.closet.branch
        if tick_from is None:
            tick_from = self.closet.tick
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
        assert branch in self.interactivity, "Make a new branch first"
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
            branch = self.closet.branch
        if tick_from is None:
            tick_from = self.closet.tick
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
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        if (
                branch in self.indefinite_coords and
                tick >= self.indefinite_coords[branch]):
            rd = self.coord_lst[branch][self.indefinite_coords[branch]]
            return (rd["x"], rd["y"])
        for rd in SkeletonIterator(self.coord_lst):
            if rd["tick_from"] <= tick and tick <= rd["tick_to"]:
                return (rd["x"], rd["y"])
        import pdb
        pdb.set_trace()
        return None

    def set_coords(self, x, y, branch=None, tick_from=None, tick_to=None):
        if branch is None:
            branch = self.closet.branch
        if tick_from is None:
            tick_from = self.closet.tick
        assert branch in self.coord_lst, "Make a new branch first"
        if branch in self.indefinite_coords:
            itf = self.indefinite_coords[branch]
            rd = self.coord_lst[branch][itf]
            if itf < tick_from:
                # You have cut off an indefinite coord
                rd["tick_to"] = tick_from - 1
                self.coord_lst[branch][itf] = rd
                del self.indefinite_coords[branch]
            elif itf < tick_to:
                # You have overwritten an indefinite coord
                del self.coord_lst[branch][itf]
                del self.indefinite_coords[branch]
            elif itf == tick_to:
                # You have extended an indefinite coord, backward in time
                del self.coord_lst[branch][itf]
                tick_to = None
        self.coord_lst[branch][tick_from] = {
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
        for rd in SkeletonIterator(self.coord_lst[parent]):
            if rd["tick_to"] >= tick or rd["tick_to"] is None:
                if rd["tick_from"] < tick:
                    self.coord_lst[branch][tick] = {
                        "dimension": rd["dimension"],
                        "place": rd["place"],
                        "board": rd["board"],
                        "branch": branch,
                        "tick_from": tick,
                        "tick_to": rd["tick_to"],
                        "x": rd["x"],
                        "y": rd["y"]}
                    if rd["tick_to"] is None:
                        self.indefinite_coords[branch] = tick
                else:
                    self.coord_lst[branch][rd["tick_from"]] = {
                        "dimension": rd["dimension"],
                        "place": rd["place"],
                        "board": rd["board"],
                        "branch": branch,
                        "tick_from": rd["tick_from"],
                        "tick_to": rd["tick_to"],
                        "x": rd["x"],
                        "y": rd["y"]}
                    if rd["tick_to"] is None:
                        self.indefinite_coords[branch] = rd["tick_from"]

    def new_branch(self, parent, branch, tick):
        self.new_branch_imagery(parent, branch, tick)
        self.new_branch_interactivity(parent, branch, tick)
        self.new_branch_coords(parent, branch, tick)


class SpotWidget:
    def get_board_coords(self):
        (x, y) = self.spot.get_coords()
        return (
            x + self.spot.drag_offset_x,
            y + self.spot.drag_offset_y)

    atrdic = {
        "coords": lambda self: self.get_board_coords(),
        "board_x": lambda self: self.coords[0],
        "board_y": lambda self: self.coords[1],
        "window_x": lambda self: self.board_x + self.viewport.offset_x,
        "window_y": lambda self: self.board_y + self.viewport.offset_y,
        "window_left": lambda self: self.window_x - self.spot.rx,
        "window_right": lambda self: self.window_x + self.spot.rx,
        "window_top": lambda self: self.window_y + self.spot.ry,
        "window_bot": lambda self: self.window_y - self.spot.ry,
        "in_view": lambda self: (
            self.window_right > 0 and
            self.window_left < self.window.width and
            self.window_top > 0 and
            self.window_bot < self.window.height),
        "selected": lambda self: self in self.viewport.window.selected,
        "hovered": lambda self: self is self.viewport.window.hovered,
        "pressed": lambda self: self is self.viewport.window.hovered,
        "grabbed": lambda self: self is self.window.grabbed}

    def __init__(self, viewport, spot):
        self.viewport = viewport
        self.window = self.viewport.window
        self.spritegroup = self.viewport.spotgroup
        self.boxgroup = self.viewport.spotgroup
        self.batch = self.viewport.batch
        self.spot = spot
        self.place = self.spot.place
        self.board = self.spot.board
        self.vert = self.spot.vert
        self.sprite = None
        self.vertlist = None
        self.old_window_left = None
        self.old_window_bot = None
        self.old_points = None

    def __str__(self):
        return str(self.spot)

    spotattrs = set(["img","visible", "interactive", "board_left",
                     "board_right", "board_top", "board_bot"])

    def __getattr__(self, attrn):
        if attrn in SpotWidget.atrdic:
            return SpotWidget.atrdic[attrn](self)
        elif attrn in self.spotattrs:
            return getattr(self.spot, attrn)
        else:
            raise AttributeError(
                "SpotWidget instance has no attribute " + attrn)

    def dropped(self, x, y, button, modifiers):
        (oldx, oldy) = self.spot.coords
        self.spot.set_coords(
            oldx + self.spot.drag_offset_x,
            oldy + self.spot.drag_offset_y)
        self.spot.drag_offset_x = 0
        self.spot.drag_offset_y = 0

    def move_with_mouse(self, x, y, dx, dy, buttons, modifiers):
        self.spot.drag_offset_x += dx
        self.spot.drag_offset_y += dy

    def overlaps(self, x, y):
        return (
            self.window_left < x and x < self.window_right and
            self.window_bot < y and y < self.window_top)

    def pass_focus(self):
        return self.viewport

    def actually_draw(self):
        try:
            wl = self.window_left
            wb = self.window_bot
            if self.old_window_left != wl:
                self.sprite.x = wl
                self.old_window_left = wl
            if self.old_window_bot != wb:
                self.sprite.y = wb
                self.old_window_bot = wb
        except AttributeError:
            self.sprite = Sprite(
                self.img.tex,
                self.window_left,
                self.window_bot,
                batch=self.batch,
                group=self.spritegroup)
        if self.selected:
            yelo = (255, 255, 0, 0)
            colors = yelo * 4
            points = (
                self.window_left, self.window_top,
                self.window_right, self.window_top,
                self.window_right, self.window_bot,
                self.window_left, self.window_bot)
            if self.old_points != points:
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
                self.old_points = points
        else:
            if self.vertlist is not None:
                try:
                    self.vertlist.delete()
                except:
                    pass
                self.vertlist = None
            self.old_points = None

    def draw(self):
        if (
                self.img is not None and
                (self.window_top > 0 or
                 self.window_right > 0 or
                 self.window_bot < self.window.height or
                 self.window_left < self.window.width)):
            self.actually_draw()
        else:
            if self.sprite is not None:
                try:
                    self.sprite.delete()
                except:
                    pass
                self.sprite = None
            if self.vertlist is not None:
                try:
                    self.vertlist.delete()
                except:
                    pass
                self.vertlist = None
            self.old_window_left = None
            self.old_window_bot = None
            return



    def delete(self):
        try:
            self.sprite.delete()
        except:
            pass
        self.sprite = None
        try:
            self.vertlist.delete()
        except:
            pass
        self.vertlist = None
