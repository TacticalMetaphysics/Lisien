## This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableWidgetMetaclass, TerminableInteractivity
from kivy.uix.image import Image
from kivy.properties import AliasProperty
from logging import getLogger


logger = getLogger(__name__)


"""Widgets to represent places. Pawns move around on top of these."""


class Spot(Image, TerminableInteractivity):
    __metaclass__ = SaveableWidgetMetaclass
    """The icon that represents a Place.

    The Spot is located on the Board that represents the same
    Dimension that the underlying Place is in. Its coordinates are
    relative to its Board, not necessarily the window the Board is in.

    """
    width = AliasProperty(
        lambda self: self.get_width(), lambda self, v: None)
    height = AliasProperty(
        lambda self: self.get_height(), lambda self, v: None)
    size = AliasProperty(
        lambda self: self.get_size(), lambda self, v: None)
    texture = AliasProperty(
        lambda self: self.get_texture(), lambda self, v: None)
    tables = [
        ("spot_img",
         {"dimension": "text not null default 'Physical'",
          "place": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "img": "text not null default 'default_spot'"},
         ("dimension", "place", "branch", "tick_from"),
         {"dimension": ("board", "dimension"),
          "img": ("img", "name")},
         []),
        ("spot_interactive",
         {"dimension": "text not null default 'Physical'",
          "place": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0"},
         ("dimension", "place", "branch", "tick_from"),
         {"dimension": ("board", "dimension")},
         []),
        ("spot_coords",
         {"dimension": "text not null default 'Physical'",
          "place": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "x": "integer not null default 50",
          "y": "integer not null default 50"},
         ("dimension", "place", "branch", "tick_from"),
         {"dimension": ("board", "dimension")},
         [])]

    def __init__(self, board, place, **kwargs):
        self.board = board
        self.place = place
        self.drag_offset_x = 0
        self.drag_offset_y = 0
        Image.__init__(self)

    def __str__(self):
        return str(self.place)

    @property
    def coord_lst(self):
        return self.board.closet.skeleton["spot_coords"][
            str(self.board)][str(self.place)]

    @property
    def interactivity(self):
        return self.board.closet.skeleton["spot_interactive"][
            str(self.board)][str(self.place)]

    @property
    def imagery(self):
        return self.board.closet.skeleton["spot_img"][
            str(self.board)][str(self.place)]

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

    def move(self):
        self.pos = self.get_pos()

    def get_width(self):
        img = self.get_texture()
        if img is None:
            return 0.
        else:
            return float(img.width)

    def get_height(self):
        img = self.get_texture()
        if img is None:
            return 0.
        else:
            return float(img.height)

    def get_size(self):
        img = self.get_texture()
        if img is None:
            return [0., 0.]
        else:
            return [float(img.width), float(img.height)]

    def get_pos(self):
        cords = self.get_coords()
        if cords is None:
            return (self.cheatx, self.cheaty)
        (x, y) = cords
        r = (
            float(x - self.rx + self.drag_offset_x),
            float(y - self.ry + self.drag_offset_y))
        print "spot {} at ({}, {})".format(self, *r)
        (self.cheatx, self.cheaty) = r
        return r

    def set_interactive(self, branch=None, tick_from=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick_from is None:
            tick_from = self.board.closet.tick
        assert branch in self.interactivity, "Make a new branch first"
        self.interactivity[branch][tick_from] = {
            "dimension": unicode(self.board),
            "place": unicode(self.place),
            "branch": branch,
            "tick_from": tick_from}

    def set_img(self, img, branch=None, tick_from=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick_from is None:
            tick_from = self.board.closet.tick
        self.imagery[branch][tick_from] = {
            "dimension": unicode(self.board),
            "place": unicode(self.place),
            "branch": branch,
            "tick_from": tick_from,
            "img": unicode(img)}

    def get_coords(self, branch=None, tick=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick is None:
            tick = self.board.closet.tick
        prev = None
        if branch not in self.coord_lst:
            return None
        for tick_from in self.coord_lst[branch]:
            if tick_from == tick:
                rd = self.coord_lst[branch][tick_from]
                return (rd["x"] + self.drag_offset_x,
                        rd["y"] + self.drag_offset_y)
            elif tick_from > tick:
                break
            prev = tick_from
        if prev is None:
            return None
        else:
            rd = self.coord_lst[branch][prev]
            return (rd["x"] + self.drag_offset_x, rd["y"] + self.drag_offset_y)

    def set_coords(self, x, y, branch=None, tick_from=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick_from is None:
            tick_from = self.board.closet.tick
        assert branch in self.coord_lst, "Make a new branch first"
        self.coord_lst[branch][tick_from] = {
            "dimension": unicode(self.board),
            "place": unicode(self.place),
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

    def get_texture(self, branch=None, tick=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick is None:
            tick = self.board.closet.tick
        if branch not in self.imagery:
            return None
        prev = None
        for rd in self.imagery[branch].iterrows():
            if rd["tick_from"] > tick:
                break
            else:
                prev = rd
        if prev is None or prev["img"] in ("", None):
            return None
        else:
            return self.board.closet.get_texture(prev["img"])

    def new_branch_imagery(self, parent, branch, tick):
        prev = None
        started = False
        for tick_from in self.imagery[parent]:
            if tick_from >= tick:
                rd2 = dict(self.imagery[parent][tick_from])
                rd2["branch"] = branch
                if branch not in self.imagery:
                    self.imagery[branch] = {}
                self.imagery[branch][rd2["tick_from"]] = rd2
                if (
                        not started and prev is not None and
                        tick_from > tick and prev < tick):
                    rd3 = dict(self.imagery[parent][prev])
                    rd3["branch"] = branch
                    rd3["tick_from"] = tick
                    self.imagery[branch][rd3["tick_from"]] = rd3
                started = True
            prev = tick_from
