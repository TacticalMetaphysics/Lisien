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

    def __init__(self, dimension, e):
        self.dimension = dimension
        self.rumor = self.dimension.rumor
        self.e = e

    def __getattr__(self, attrn):
        if attrn == "orig":
            return Place(self.dimension, self.source)
        elif attrn == "dest":
            return Place(self.dimension, self.target)
        else:
            try:
                return self.e[attrn]
            except KeyError:
                raise AttributeError(
                    "Portal instance has no attribute named " + attrn)

    def __setattr__(self, attrn, val):
        if attrn in self.e.get_attributes():
            self.e[attrn] = val
        else:
            super(Portal, self).__setattr__(attrn, val)

    def __repr__(self):
        return "Portal({0}->{1})".format(str(self.orig), str(self.dest))

    def __int__(self):
        return self.e.index

    def __len__(self):
        # eventually this will represent something like actual physical length
        return 1

    def __getattr__(self, attrn):
        try:
            return self.e[attrn]
        except KeyError:
            raise AttributeError(
                "Portal instance has no attribute named " + attrn)

    def __contains__(self, that):
        try:
            return that.location.e is self.e
        except:
            return False

    def admits(self, traveler):
        """Return True if I want to let the given thing enter me, False
otherwise."""
        return True

    def extant(self, branch=None, tick=None):
        return self.dimension.portal_extant(self.e, branch, tick)

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
