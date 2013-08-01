# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass, BranchTicksIter, dictify_row
from logging import getLogger


logger = getLogger(__name__)


class Portal:
    __metaclass__ = SaveableMetaclass
    tables = [
        ("portal_existence",
         {"dimension": "text not null DEFAULT 'Physical'",
          "origin": "text not null",
          "destination": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "tick_to": "integer default null"},
         ("dimension", "origin", "destination", "branch", "tick_from"),
         {},
         [])]

    def __init__(self, dimension, orig, dest):
        self.dimension = dimension
        self.orig = orig
        self.dest = dest
        self.db = self.dimension.db
        self.existence = {}

    def __repr__(self):
        return "Portal({0}->{1})".format(str(self.orig), str(self.dest))

    def __int__(self):
        return self.dimension.portals_by_orign_destn.values().index(self)

    def __len__(self):
        # eventually this will represent something like actual physical length
        return 1

    def extant(self, branch=None, tick=None):
        if branch is None:
            branch = self.db.branch
        if tick is None:
            tick = self.db.tick
        if branch not in self.existence:
            return False
        for (tick_from, tick_to) in self.existence[branch].iteritems():
            if tick_from <= tick and (tick_to is None or tick <= tick_to):
                return True
        return False

    def exist(self, branch=None, tick_from=None, tick_to=None):
        if branch is None:
            branch = self.db.branch
        if tick_from is None:
            tick_from = self.db.tick
        if branch not in self.existence:
            self.existence[branch] = {}
        self.existence[branch][tick_from] = tick_to

    def admits(self, traveler):
        """Return True if I want to let the given thing enter me, False
otherwise."""
        return True

    def touches(self, place):
        return self.orig == place or self.dest == place

    def find_neighboring_portals(self):
        return self.orig.portals + self.dest.portals

    def notify_moving(self, thing, amount):
        """Handler for when a thing moves through me by some amount of my
length. Does nothing by default."""
        pass

    def get_tabdict(self):
        return {
            "portal_existence": [
                {
                    "dimension": str(self.dimension),
                    "origin": str(self.orig),
                    "destination": str(self.dest),
                    "branch": branch,
                    "tick_from": tick_from,
                    "tick_to": tick_to}
                for (branch, tick_from, tick_to) in
                BranchTicksIter(self.existence)]}
