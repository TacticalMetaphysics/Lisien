## This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import (
    get_bone_during)
from gui.kivybits import SaveableWidgetMetaclass
from kivy.uix.image import Image
from kivy.properties import (
    NumericProperty,
    DictProperty,
    ObjectProperty,
    BooleanProperty)
from kivy.uix.scatter import Scatter
from logging import getLogger


logger = getLogger(__name__)


"""Widgets to represent places. Pawns move around on top of these."""


class Spot(Scatter):
    __metaclass__ = SaveableWidgetMetaclass
    """The icon that represents a Place.

    The Spot is located on the Board that represents the same
    Dimension that the underlying Place is in. Its coordinates are
    relative to its Board, not necessarily the window the Board is in.

    """
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
    place = ObjectProperty()
    board = ObjectProperty()
    coords = ObjectProperty()
    interactivity = ObjectProperty()
    imagery = ObjectProperty()
    completedness = NumericProperty(0)
    tex = ObjectProperty(None)

    def __str__(self):
        return str(self.place)

    def __unicode__(self):
        return unicode(self.place)

    def on_interactivity(self, i, v):
        self.completedness += 1

    def on_imagery(self, i, v):
        self.completedness += 1

    def on_coords(self, i, v):
        self.completedness += 1
        v.listeners.append(self.repos)
        self.repos()

    def on_completedness(self, i, v):
        if v == 3:
            self.imagery.listeners.append(self.retex)

    def retex(self, *args):
        self.tex = self.get_texture()

    def on_tex(self, i, v):
        if v is None:
            return
        self.size = v.size

    def get_width(self):
        img = self.get_texture()
        if img is None:
            return 0
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
        if self.board is None:
            return (0, 0)
        cords = self.get_coords()
        if cords is None:
            return (self.cheatx, self.cheaty)
        (x, y) = cords
        r = (self.cheatx, self.cheaty) = (x, y)
        return r

    def repos(self, *args):
        oldtf = self.transform
        self.transform.identity()
        self.pos = self.get_pos()
        self.apply_transform(oldtf)

    def set_pos(self, v):
        if self.board is not None:
            self.set_coords(v[0], v[1])

    def set_interactive(self, branch=None, tick_from=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick_from is None:
            tick_from = self.board.closet.tick
        assert branch in self.interactivity, "Make a new branch first"
        self.board.closet.skeleton["spot_interactive"][
            unicode(self.board)][unicode(self.place)][branch][tick_from] = {
            "dimension": unicode(self.board),
            "place": unicode(self.place),
            "branch": branch,
            "tick_from": tick_from}
        self.upd_interactivity()

    def upd_interactivity(self, *args):
        self.interactivity = dict(
            self.board.closet.skeleton["spot_interactive"][
                unicode(self.board)][unicode(self.place)])

    def is_interactive(self, branch=None, tick=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick is None:
            tick = self.board.closet.tick
        interactivity = self.closet.skeleton["spot_interactive"][
            unicode(self.board.dimension)][unicode(self.place)]
        if branch not in interactivity:
            return False
        for rd in interactivity.iterbones():
            if rd["tick_from"] <= tick and (
                    rd["tick_to"] is None or tick <= rd["tick_to"]):
                return True
        return False

    def new_branch_interactivity(self, parent, branch, tick):
        prev = None
        started = False
        interactivity = self.board.closet.skeleton["spot_interactive"][
            unicode(self.board.dimension)][unicode(self.place)]
        for tick_from in interactivity[parent]:
            if tick_from >= tick:
                rd2 = dict(interactivity[parent][tick_from])
                rd2["branch"] = branch
                if branch not in interactivity:
                    interactivity[branch] = {}
                interactivity[branch][rd2["tick_from"]] = rd2
                if (
                        not started and prev is not None and
                        tick_from > tick and prev < tick):
                    rd3 = dict(interactivity[parent][prev])
                    rd3["branch"] = branch
                    rd3["tick_from"] = tick
                    interactivity[branch][rd3["tick_from"]] = rd3
                started = True
            prev = tick_from
        self.upd_interactivity()

    def set_img(self, img, branch=None, tick_from=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick_from is None:
            tick_from = self.board.closet.tick
        imagery = self.board.closet.skeleton["spot_img"][
            unicode(self.board.dimension)][unicode(self.place)]
        assert branch in imagery, "Make a new branch first"
        imagery[branch][tick_from] = {
            "dimension": unicode(self.board),
            "place": unicode(self.place),
            "branch": branch,
            "tick_from": tick_from,
            "img": unicode(img)}
        self.upd_imagery()

    def upd_imagery(self, *args):
        self.imagery = dict(self.board.closet.skeleton["spot_img"][
            unicode(self.board)][unicode(self.place)])

    def get_coord_rd(self, branch=None, tick=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick is None:
            tick = self.board.closet.tick
        return get_bone_during(self.coords, branch, tick)

    def get_coords(self, branch=None, tick=None):
        rd = self.get_coord_rd(branch, tick)
        if rd is None:
            return None
        else:
            return (rd["x"], rd["y"])

    def set_coords(self, x, y, branch=None, tick_from=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick_from is None:
            tick_from = self.board.closet.tick
        self.board.closet.skeleton["spot_coords"][
            unicode(self.board)][unicode(self.place)][
            branch][tick_from] = {
            "dimension": unicode(self.board),
            "place": unicode(self.place),
            "branch": branch,
            "tick_from": tick_from,
            "x": x,
            "y": y}

    def new_branch_coords(self, parent, branch, tick):
        prev = None
        started = False
        coords = self.board.closet.skeleton["spot_coords"][
            unicode(self.board)][unicode(self.place)]
        for tick_from in coords[parent]:
            if tick_from >= tick:
                rd2 = dict(coords[parent][tick_from])
                rd2["branch"] = branch
                if branch not in coords:
                    coords[branch] = {}
                coords[branch][rd2["tick_from"]] = rd2
                if (
                        not started and prev is not None and
                        tick_from > tick and prev < tick):
                    rd3 = dict(coords[branch][prev])
                    rd3["branch"] = branch
                    rd3["tick_from"] = tick
                    coords[branch][rd3["tick_from"]] = rd3
                started = True
            prev = tick_from

    def new_branch(self, parent, branch, tick):
        self.new_branch_imagery(parent, branch, tick)
        self.new_branch_interactivity(parent, branch, tick)
        self.new_branch_coords(parent, branch, tick)

    def get_image_rd(self, branch=None, tick=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick is None:
            tick = self.board.closet.tick
        if branch not in self.imagery:
            return None
        prev = None
        for rd in self.imagery[branch].iterbones():
            if rd["tick_from"] > tick:
                break
            else:
                prev = rd
        if prev is None or prev["img"] in ("", None):
            return None
        else:
            return prev

    def get_texture(self, branch=None, tick=None):
        tn = self.get_image_rd(branch, tick)
        if tn is not None:
            return self.board.closet.get_texture(tn["img"])

    def new_branch_imagery(self, parent, branch, tick):
        prev = None
        started = False
        imagery = self.board.closet.skeleton["spot_img"][
            unicode(self.board.dimension)][unicode(self.place)]
        for tick_from in imagery[parent]:
            if tick_from >= tick:
                rd2 = dict(imagery[parent][tick_from])
                rd2["branch"] = branch
                if branch not in imagery:
                    imagery[branch] = {}
                imagery[branch][rd2["tick_from"]] = rd2
                if (
                        not started and prev is not None and
                        tick_from > tick and prev < tick):
                    rd3 = dict(imagery[parent][prev])
                    rd3["branch"] = branch
                    rd3["tick_from"] = tick
                    imagery[branch][rd3["tick_from"]] = rd3
                started = True
            prev = tick_from
        self.upd_imagery()

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            self.set_coords(*self.pos)
        super(Spot, self).on_touch_up(touch)
