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
          "tick_from": "integer not null default 0",
          "tick_to": "integer default null"},
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
    cheatx = NumericProperty(0)
    cheaty = NumericProperty(0)

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

    def on_completedness(self, i, v):
        if v == 3:
            self.imagery.listeners.append(self.retex)
            self.repos()

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
        r = interactivity.value_during(tick)
        return (r.tick_to is None or tick <= r.tick_to)

    def new_branch_interactivity(self, parent, branch, tick):
        prev = None
        started = False
        interactivity = self.board.closet.skeleton["spot_interactive"][
            unicode(self.board.dimension)][unicode(self.place)]
        for tick_from in interactivity[parent]:
            if tick_from >= tick:
                b2 = interactivity[parent][tick_from].replace(branch=branch)
                if branch not in interactivity:
                    interactivity[branch] = {}
                interactivity[branch][b2.tick_from] = b2
                if (
                        not started and prev is not None and
                        tick_from > tick and prev < tick):
                    b3 = interactivity[parent][prev].replace(
                        branch=branch, tick_from=tick)
                    interactivity[branch][b3.tick_from] = b3
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
        imagery[branch][tick_from] = self.bonetypes["spot_img"](
            dimension=unicode(self.board),
            place=unicode(self.place),
            branch=branch,
            tick_from=tick_from,
            img=unicode(img))
        self.upd_imagery()

    def get_coord_bone(self, branch=None, tick=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick is None:
            tick = self.board.closet.tick
        return self.coords[branch].value_during(tick)

    def get_coords(self, branch=None, tick=None):
        bone = self.get_coord_bone(branch, tick)
        if bone is None:
            return None
        else:
            return (bone.x, bone.y)

    def set_coords(self, x, y, branch=None, tick_from=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick_from is None:
            tick_from = self.board.closet.tick
        self.board.closet.skeleton["spot_coords"][
            unicode(self.board)][unicode(self.place)][
            branch][tick_from] = self.bonetypes["spot_coords"](
                dimension=unicode(self.board),
                place=unicode(self.place),
                branch=branch,
                tick_from=tick_from,
                x=x,
                y=y)

    def new_branch_coords(self, parent, branch, tick):
        prev = None
        started = False
        coords = self.board.closet.skeleton["spot_coords"][
            unicode(self.board)][unicode(self.place)]
        for tick_from in coords[parent]:
            if tick_from >= tick:
                b2 = coords[parent][tick_from]._replace(branch=branch)
                if branch not in coords:
                    coords[branch] = {}
                coords[branch][b2["tick_from"]] = b2
                if (
                        not started and prev is not None and
                        tick_from > tick and prev < tick):
                    b3 = coords[branch][prev].replace(
                        branch=branch,
                        tick_from=tick)
                    coords[branch][b3.tick_from] = b3
                started = True
            prev = tick_from

    def new_branch(self, parent, branch, tick):
        self.new_branch_imagery(parent, branch, tick)
        self.new_branch_interactivity(parent, branch, tick)
        self.new_branch_coords(parent, branch, tick)

    def get_image_bone(self, branch=None, tick=None):
        if branch is None:
            branch = self.board.closet.branch
        if tick is None:
            tick = self.board.closet.tick
        if branch not in self.imagery:
            return None
        r = self.imagery[branch].value_during(tick)
        if r.img in ("", None):
            return None
        return r

    def get_texture(self, branch=None, tick=None):
        tn = self.get_image_bone(branch, tick)
        if tn is not None:
            return self.board.closet.get_texture(tn.img)

    def new_branch_imagery(self, parent, branch, tick):
        prev = None
        started = False
        imagery = self.board.closet.skeleton["spot_img"][
            unicode(self.board.dimension)][unicode(self.place)]
        for tick_from in imagery[parent]:
            if tick_from >= tick:
                b2 = imagery[parent][tick_from].replace(branch=branch)
                if branch not in imagery:
                    imagery[branch] = {}
                imagery[branch][b2.tick_from] = b2
                if (
                        not started and prev is not None and
                        tick_from > tick and prev < tick):
                    b3 = imagery[parent][prev].replace(
                        branch=branch,
                        tick_from=tick)
                    imagery[branch][b3.tick_from] = b3
                started = True
            prev = tick_from
        self.upd_imagery()

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            self.set_coords(*self.pos)
        super(Spot, self).on_touch_up(touch)
