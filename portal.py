# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass
from logging import getLogger


logger = getLogger(__name__)


class Portal:
    __metaclass__ = SaveableMetaclass
    tables = [
        ("portal",
         {"dimension": "text not null DEFAULT 'Physical'",
          "origin": "text not null",
          "destination": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "extant": "boolean not null default 0"},
         ("dimension", "origin", "destination", "branch", "tick_from"),
         {},
         [])]

    def __init__(self, closet, dimension, origin, destination):
        self.closet = closet
        self.dimension = dimension
        self.graph = self.dimension.graph
        self.orig = origin
        self.origin = origin
        self.dest = destination
        self.destination = destination
        self.dimension.graph.add_edge(self.origi, self.desti, portal=self)

    def __getattr__(self, attrn):
        if attrn == "origi":
            return self.orig.index
        elif attrn == "desti":
            return self.dest.index
        elif attrn in ("e", "edge"):
            return self.graph.es[self.graph.get_eid(self.origi, self.desti)]
        elif attrn == "existence":
            return self.closet.skeleton["portal"][
                str(self.dimension)][str(self.orig)][str(self.dest)]
        else:
            raise AttributeError(
                "Portal instance has no attribute named " + attrn)

    def __repr__(self):
        return "Portal({0}->{1})".format(
            str(self.orig), str(self.dest))

    def __int__(self):
        return self.e.index

    def __len__(self):
        # eventually this will represent something like actual physical length
        return 1

    def admits(self, traveler):
        """Return True if I want to let the given thing enter me, False
otherwise."""
        return True

    def set_existence(self, exist=True, branch=None, tick_from=None):
        if branch is None:
            branch = self.closet.branch
        if tick_from is None:
            tick_from = self.closet.tick
        self.existence[branch][tick_from] = {
            "dimension": str(self.dimension),
            "origin": str(self.origin),
            "destination": str(self.destination),
            "branch": branch,
            "tick_from": tick_from,
            "extant": exist}

    def new_branch(self, parent, branch, tick):
        prev = None
        started = False
        for rd in self.existence[parent].iterrows():
            if rd["tick_from"] >= tick:
                rd2 = dict(rd)
                rd2["branch"] = branch
                self.existence[branch][rd2["tick_from"]] = rd2
                if (
                        not started and prev is not None and
                        rd["tick_from"] > tick and prev["tick_from"] < tick):
                    rd3 = dict(prev)
                    rd3["branch"] = branch
                    rd3["tick_from"] = tick
                    self.existence[branch][rd3["tick_from"]] = rd3
                started = True
            prev = rd
