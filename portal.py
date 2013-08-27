# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass, BranchTicksIter
from place import Place
from logging import getLogger
from igraph import Edge


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

    def __init__(self, dimension, e):
        assert(isinstance(e, Edge))
        self.dimension = dimension
        self.rumor = self.dimension.rumor
        self.e = e

    def __getattr__(self, attrn):
        if attrn in ("orig", "origin"):
            return Place(
                self.dimension,
                self.dimension.graph.vs[self.e.source])
        elif attrn in ("dest", "destination"):
            return Place(
                self.dimension,
                self.dimension.graph.vs[self.e.target])
        elif attrn in self.e.attribute_names():
            return self.e[attrn]
        else:
            raise AttributeError(
                "Portal instance has no attribute named " + attrn)

    def __repr__(self):
        return "Portal({0}->{1})".format(str(self.orig), str(self.dest))

    def __int__(self):
        return self.e.index

    def __len__(self):
        # eventually this will represent something like actual physical length
        return 1

    def admits(self, traveler):
        """Return True if I want to let the given thing enter me, False
otherwise."""
        return True

    def extant(self, branch=None, tick=None):
        return self.dimension.portal_extant(self.e, branch, tick)

    def extant_between(self, branch=None, tick_from=None, tick_to=None):
        return self.dimension.portal_extant_between(
            self.e, branch, tick_from, tick_to)

    def persist(self, branch=None, tick_from=None, tick_to=None):
        self.dimension.persist_portal(self.e, branch, tick_from, tick_to)

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
