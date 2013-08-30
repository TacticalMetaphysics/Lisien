# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass, TabdictIterator
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
          "tick_to": "integer default null"},
         ("dimension", "origin", "destination", "branch", "tick_from"),
         {},
         [])]

    def __init__(self, rumor, dimension, origin, destination):
        self.rumor = rumor
        self.dimension = dimension
        self.orig = origin
        self.dest = destination
        self.indefinite_existence = {}
        for rd in TabdictIterator(self.existence):
            if rd["tick_to"] is None:
                self.indefinite_existence[rd["branch"]] = rd["tick_from"]
        # now make the edge
        self.dimension.graph.add_edge(self.origi, self.desti, portal=self)

    def __getattr__(self, attrn):
        if attrn == "origin":
            return self.orig
        elif attrn == "origi":
            return self.orig.index
        elif attrn == "destination":
            return self.dest
        elif attrn == "desti":
            return self.dest.index
        elif attrn in ("e", "edge"):
            return self.graph.es[self.graph.get_eid(self.origi, self.desti)]
        elif attrn == "existence":
            return self.rumor.tabdict["portal"][
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

    def new_branch(self, parent, branch, tick):
        if branch not in self.existence:
            self.existence[branch] = {}
        for rd in TabdictIterator(self.existence):
            if rd["tick_to"] is None or rd["tick_to"] >= tick:
                rd2 = dict(rd)
                if rd2["tick_from"] < tick:
                    rd2["tick_from"] = tick
                    self.existence[branch][tick] = rd2
                    if rd2["tick_to"] is None:
                        self.indefinite_existence[branch] = tick
                else:
                    self.existence[branch][rd2["tick_from"]] = rd2["tick_to"]
                    if rd2["tick_to"] is None:
                        self.indefinite_existence[branch] = rd2["tick_from"]
