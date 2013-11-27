# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass
from logging import getLogger


logger = getLogger(__name__)


class Portal(object):
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
         ["origin not like '%->%'", "destination not like '%->%'"])]

    def __init__(self, closet, dimension, origin, destination):
        self.closet = closet
        self.dimension = dimension
        self.graph = self.dimension.graph
        self.origin = origin
        self.destination = destination
        self.dimension.graph.add_edge(
            self.origin.index, self.destination.index, portal=self)

    def __str__(self):
        return "{}->{}".format(str(self.origin), str(self.destination))

    def __unicode__(self):
        return u"{}->{}".format(unicode(self.origin), unicode(self.destination))

    def __repr__(self):
        return "Portal({0}->{1})".format(
            self.origin, self.destination)

    def __int__(self):
        return self.e.index

    def __len__(self):
        # eventually this will represent something like actual physical length
        return 1

    @property
    def edge(self):
        return self.dimension.graph.es[
            self.dimension.graph.get_eid(
                self.origin.index, self.destination.index)]

    @property
    def existence(self):
        return self.closet.skeleton["portal"][
            unicode(self.dimension)][unicode(self.origin)][
            unicode(self.destination)]

    def admits(self, traveler):
        """Return True if I want to let the given thing enter me, False
otherwise."""
        return True

    def set_existence(self, branch=None, tick_from=None, tick_to=None):
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
            "tick_to": tick_to}
        self.closet.timestream.upbranch(branch)
        self.closet.timestream.uptick(tick_from)
        if tick_to is not None:
            self.closet.timestream.uptick(tick_to)

    def new_branch(self, parent, branch, tick):
        prev = None
        started = False
        for bone in self.existence[parent].iterbones():
            if bone.tick_from >= tick:
                bone2 = bone._replace(branch=branch)
                self.existence[branch][bone2.tick_from] = bone2
                if (
                        not started and prev is not None and
                        bone2.tick_from > tick and prev.tick_from < tick):
                    bone3 = prev._replace(branch=branch, tick_from=tick)
                    self.existence[branch][bone3.tick_from] = bone3
                started = True
            prev = bone
