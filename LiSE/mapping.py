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
                "SELECT things.thing, things.location FROM things JOIN ("
                "SELECT character, thing, branch, MAX(tick) AS tick FROM things "
                "WHERE character=? "
                "AND branch=? "
                "AND tick<=? "
                "GROUP BY character, thing, branch) AS hitick "
                "ON things.character=hitick.character "
                "AND things.thing=hitick.thing "
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
                "SELECT things.thing, things.location FROM things JOIN ("
                "SELECT character, thing, branch, MAX(tick) AS tick FROM things "
                "WHERE character=? "
                "AND thing=? "
                "AND branch=? "
                "AND tick<=? "
                "GROUP BY character, thing, branch) AS hitick "
                "ON things.character=hitick.character "
                "AND things.thing=hitick.thing "
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
        th.dontcall = True
        th.clear()
        th.exists = True
        th.update(val)
        del th.dontcall
        (branch, tick) = self.character.worldview.time
        for fun in self.character.on_set_thing:
            fun(thing, val, branch, tick)

    def __delitem__(self, thing):
        Thing(self.character, thing).clear()
        (branch, tick) = self.character.worldview.time
        for fun in self.character.on_del_thing:
            fun(thing, branch, tick)

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
                "SELECT things.thing, things.location FROM things JOIN ("
                "SELECT character, thing, branch, MAX(tick) AS tick FROM things "
                "WHERE character=? "
                "AND branch=? "
                "AND tick<=? "
                "GROUP BY character, thing, branch) AS hitick ON "
                "things.character=hitick.character "
                "AND things.thing=hitick.thing "
                "AND things.branch=hitick.branch "
                "AND things.tick=hitick.tick;",
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
        pl.dontcall = True
        pl.clear()
        pl.exists = True
        pl.update(v)
        del pl.dontcall
        (branch, tick) = self.character.worldview.time
        for fun in self.character.on_set_place:
            fun(place, v, branch, tick)

    def __delitem__(self, place):
        Place(self.character, place).clear()
        (branch, tick) = self.character.worldview.time
        for fun in self.character.on_del_place:
            fun(place, branch, tick)

    def __repr__(self):
        return repr(dict(self))


class CharacterPortalSuccessorsMapping(GraphSuccessorsMapping):
    class Successors(GraphSuccessorsMapping.Successors):
        def _getsub(self, nodeB):
            return Portal(self.graph, self.nodeA, nodeB)

        def __setitem__(self, nodeB, value):
            p = Portal(self.graph, self.nodeA, nodeB)
            p.dontcall = True
            p.clear()
            p.exists = True
            p.update(value)
            del p.dontcall
            (branch, tick) = self.graph.worldview.time
            for fun in self.graph.on_set_portal:
                fun(self.nodeA, nodeB, value, branch, tick)

        def __delitem__(self, nodeB):
            super().__delitem__(nodeB)
            (branch, tick) = self.graph.worldview.time
            for fun in self.graph.on_del_portal:
                fun(self.nodeA, nodeB, branch, tick)


class CharacterPortalPredecessorsMapping(DiGraphPredecessorsMapping):
    class Predecessors(DiGraphPredecessorsMapping.Predecessors):
        def _getsub(self, nodeA):
            return Portal(self.graph, nodeA, self.nodeB)

        def __setitem__(self, nodeA, value):
            p = Portal(self.graph, nodeA, self.nodeB)
            p.dontcall = True
            p.clear()
            p.exists = True
            p.update(value)
            del p.dontcall
            (branch, tick) = self.graph.worldview.time
            for fun in self.graph.on_set_portal:
                fun(nodeA, self.nodeB, value, branch, tick)

        def __delitem__(self, nodeA):
            super().__delitem__(nodeA)
            (branch, tick) = self.graph.worldview.time
            for fun in self.graph.on_del_portal:
                fun(nodeA, self.nodeB, branch, tick)
