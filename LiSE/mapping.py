# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from collections import MutableMapping
from gorm.graph import (
    GraphSuccessorsMapping,
    DiGraphPredecessorsMapping
)
from LiSE.graph import (
    Thing,
    Place,
    Portal
)

class CharacterThingMapping(MutableMapping):
    def __init__(self, character):
        self.character = character
        self.worldview = character.worldview
        self.name = character.name

    def __iter__(self):
        """Iterate over nodes that have locations, and are therefore
        Things. Yield their names.

        """
        seen = set()
        for (branch, tick) in self.worldview._active_branches():
            self.worldview.cursor.execute(
                "SELECT things.node, things.location FROM things JOIN ("
                "SELECT graph, node, branch, MAX(tick) AS tick FROM things "
                "WHERE graph=? "
                "AND branch=? "
                "AND tick<=? "
                "GROUP BY graph, node, branch) AS hitick "
                "ON things.graph=hitick.graph "
                "AND things.node=hitick.node "
                "AND things.branch=hitick.branch "
                "AND things.tick=hitick.tick;",
                (
                    self.name,
                    branch,
                    tick
                )
            )
            for (node, loc) in self.worldview.cursor.fetchall():
                if loc and node not in seen:
                    yield node
                seen.add(node)

    def __len__(self):
        n = 0
        for th in self:
            n += 1
        return n

    def __getitem__(self, thing):
        for (branch, rev) in self.worldview._active_branches():
            self.worldview.cursor.execute(
                "SELECT things.node, things.location FROM things JOIN ("
                "SELECT graph, node, branch, MAX(tick) AS tick FROM things "
                "WHERE graph=? "
                "AND node=? "
                "AND branch=? "
                "AND tick<=? "
                "GROUP BY graph, node, branch) AS hitick "
                "ON things.graph=hitick.graph "
                "AND things.node=hitick.node "
                "AND things.branch=hitick.branch "
                "AND things.tick=hitick.tick;",
                (
                    self.name,
                    thing,
                    branch,
                    rev
                )
            )
            for (thing, loc) in self.worldview.cursor.fetchall():
                if not loc:
                    raise KeyError("Thing doesn't exist right now")
                return Thing(self.character, thing)
        raise KeyError("Thing has never existed")

    def __setitem__(self, thing, val):
        th = Thing(self.character, thing)
        th.clear()
        th.exists = True
        th.update(val)

    def __delitem__(self, thing):
        Thing(self.character, thing).clear()

    def __repr__(self):
        return repr(dict(self))


class CharacterPlaceMapping(MutableMapping):
    def __init__(self, character):
        self.character = character
        self.worldview = character.worldview
        self.name = character.name

    def __iter__(self):
        things = set()
        things_seen = set()
        for (branch, rev) in self.worldview._active_branches():
            self.worldview.cursor.execute(
                "SELECT things.node, things.location FROM things JOIN ("
                "SELECT graph, node, branch, MAX(rev) AS rev FROM things "
                "WHERE graph=? "
                "AND branch=? "
                "AND rev<=? "
                "GROUP BY graph, node, branch) AS hirev ON "
                "things.graph=hirev.graph "
                "AND things.node=hirev.node "
                "AND things.branch=hirev.branch "
                "AND things.rev=hirev.rev;",
                (
                    self.character.name,
                    branch,
                    rev
                )
            )
            for (thing, loc) in self.worldview.cursor.fetchall():
                if thing not in things_seen and loc:
                    things.add(thing)
                things_seen.add(thing)
        for node in self.worldview._iternodes(self.character.name):
            if node not in things:
                yield node

    def __len__(self):
        n = 0
        for place in self:
            n += 1
        return n

    def __getitem__(self, place):
        if place in iter(self):
            return Place(self.character, place)
        raise KeyError("No such place")

    def __setitem__(self, place, v):
        pl = Place(self.character, place)
        pl.clear()
        pl.exists = True
        pl.update(v)

    def __delitem__(self, place):
        Place(self.character, place).clear()

    def __repr__(self):
        return repr(dict(self))


class CharacterPortalSuccessorsMapping(GraphSuccessorsMapping):
    class Successors(GraphSuccessorsMapping.Successors):
        def _getsub(self, nodeB):
            return Portal(self.graph, self.nodeA, nodeB)


class CharacterPortalPredecessorsMapping(DiGraphPredecessorsMapping):
    class Predecessors(DiGraphPredecessorsMapping.Predecessors):
        def _getsub(self, nodeA):
            return Portal(self.graph, nodeA, self.nodeB)
