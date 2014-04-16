# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from collections import Mapping
from igraph import Graph

from LiSE.orm import (
    SaveableMetaclass,
    Skeleton
)
from thing import Thing
from place import Place
from portal import Portal


class CharacterNameCreatorMapping(Mapping):
    def __init__(self, character, cls):
        self.character = character
        self.cls = cls
        self.d = {}

    def __getitem__(self, name):
        if name not in self.d:
            self.d[name] = self.cls(self.character, name)


class Character(object):
    __metaclass__ = SaveableMetaclass
    demands = ["thing"]
    provides = ["character"]
    tables = [
        ("character_stat", {
            "columns": {
                "character": "text not null",
                "key": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "value": "text"},
            "primary_key": (
                "character", "key", "branch", "tick")})]

    def __init__(self, closet, name):
        self.closet = closet
        self.name = name
        self.thing_d = CharacterNameCreatorMapping(self, Thing)
        self.place_d = CharacterNameCreatorMapping(self, Place)
        self.portal_d = CharacterNameCreatorMapping(self, Portal)
        self.regraph()

    def __str__(self):
        return str(self.name)

    def __unicode__(self):
        return unicode(self.name)

    def iter_triggers(self):
        pass

    def regraph(self, branch=None, tick=None):
        """Recreate my graph and fill it with places, portals, and things.

        """
        (branch, tick) = self.closet.sanetime(branch, tick)
        self.graph = Graph(directed=True)
        for portalbone in self.iter_portal_loc_bones_hosted(branch, tick):
            for placename in (portalbone.origin, portalbone.destination):
                try:
                    self.graph.vs.find(name=placename)
                except ValueError:  # check what errors it really throws
                    self.graph.add_vertex(
                        name=placename,
                        contents=set())
            self.graph.add_edge(
                portalbone.origin,
                portalbone.destination,
                name=portalbone.name)
        for thingbone in self.iter_thing_loc_bones_hosted(branch, tick):
            try:
                v = self.graph.vs.find(name=thingbone.location)
                v["contents"].add(thingbone.name)
            except ValueError:
                self.graph.add_vertex(
                    name=thingbone.location,
                    contents=set([thingbone.name]))

    def _iter_bones_hosted(self, tabn, branch, tick, xkeys=[]):
        """Iterate over the bones that are hosted in a tab, at a branch and
        tick.

        """
        skel = self.closet.skeleton[tabn][unicode(self)]
        (branch, tick) = self.closet.sanetime(branch, tick)
        for name in skel:
            subskel = skel[name]
            for key in xkeys:
                subskel = subskel[key]
            if branch in subskel:
                yield subskel[branch].value_during(tick)

    def iter_thing_loc_bones_hosted(self, branch, tick):
        """Iterate over all the thing_loc bones that are active at the
        moment.

        """
        for b in self._iter_bones_hosted(
                u"thing_loc", branch, tick):
            yield b

    def iter_portal_loc_bones_hosted(self, branch, tick):
        """Iterate over all the portal_loc bones that are active at the
        moment.

        """
        for b in self._iter_bones_hosted(
                u"portal_loc", branch, tick):
            yield b
