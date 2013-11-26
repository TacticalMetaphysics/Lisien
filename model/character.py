# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass, TimeParadox
from collections import deque


"""Things that should have character sheets."""


class Character(object):
    __metaclass__ = SaveableMetaclass
    """An incorporeal object connecting corporeal ones together across
dimensions, indicating that they represent one thing and have that
thing's attributes.

Every item in LiSE's world model must be part of a Character, though
it may be the only member of that Character. Where items can only have
generic attributes, Characters have all the attributes of the items
that make them up, and possibly many more. There are no particular
restrictions on what manner of attribute a Character can have, so long
as it is not used by the physics of any dimension.

Otherwise, Characters can be treated much like three-dimensional
dictionaries, wherein you may look up the Character's attributes. The
key is composed of the dimension an item of this character is in, the
item's name, and the name of the attribute.

    """

    prelude = [
        "CREATE VIEW place AS "
        "SELECT dimension, location AS name FROM thing_location UNION "
        "SELECT dimension, origin AS name FROM portal UNION "
        "SELECT dimension, destination AS name FROM portal UNION "
        "SELECT dimension, place AS name FROM spot_coords"]
    postlude = [
        "CREATE VIEW character AS "
        "SELECT character FROM character_things UNION "
        "SELECT character FROM character_places UNION "
        "SELECT character FROM character_portals UNION "
        "SELECT inner_character FROM character_subcharacters UNION "
        "SELECT outer_character FROM character_subcharacters UNION "
        "SELECT character FROM character_skills UNION "
        "SELECT character FROM character_stats"]
    demands = ["thing_location", "portal", "spot_coords"]
    provides = ["character"]
    tables = [
        ("character_things",
         {"character": "text not null",
          "dimension": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "tick_to": "integer default null",
          "thing": "text not null",
          "pawn": "text",
          "interactive": "boolean not null default 1"},
         ("character", "dimension", "thing", "branch", "tick_from"),
         {"dimension, thing": ("thing", "dimension, name")},
         []),
        ("character_places",
         {"character": "text not null",
          "dimension": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "tick_to": "integer default null",
          "place": "text not null"},
         ("character", "dimension", "place", "branch", "tick_from"),
         {"dimension, place": ("place", "dimension, name")},
         []),
        ("character_portals",
         {"character": "text not null",
          "dimension": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "tick_to": "integer default null",
          "origin": "text not null",
          "destination": "text not null"},
         ("character", "dimension", "origin", "destination",
          "branch", "tick_from"),
         {"dimension, origin, destination":
          ("portal", "dimension, origin, destination")},
         []),
        ("character_subcharacters",
         {"outer_character": "text not null",
          "inner_character": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "tick_to": "integer default 0"},
         ("outer_character", "inner_character", "branch", "tick_from"),
         {},
         []),
        ("character_skills",
         {"character": "text not null",
          "skill": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "tick_to": "integer default null"},
         ("character", "skill", "branch", "tick_from"),
         {},
         []),
        ("character_stats",
         {"character": "text not null",
          "stat": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "value": "text"},
         ("character", "stat", "branch", "tick_from"),
         {},
         [])]

    def __init__(self, closet, name):
        self._name = name
        self.closet = closet
        self.update_handlers = set()
        td = self.closet.skeleton
        if "character_things" not in td:
            td["character_things"] = {}
        if name not in td["character_things"]:
            td["character_things"][name] = {}
        self.thingdict = td["character_things"][name]
        if "character_stats" not in td:
            td["character_stats"] = {}
        if name not in td["character_stats"]:
            td["character_stats"][name] = {}
        self.statdict = td["character_stats"][name]
        if "character_skills" not in td:
            td["character_skills"] = {}
        if name not in td["character_skills"]:
            td["character_skills"][name] = {}
        self.skilldict = td["character_skills"][name]
        if "character_portals" not in td:
            td["character_portals"] = {}
        if name not in td["character_portals"]:
            td["character_portals"][name] = {}
        self.portaldict = td["character_portals"][name]
        if "character_places" not in td:
            td["character_places"] = {}
        if name not in td["character_places"]:
            td["character_places"][name] = {}
        self.placedict = td["character_places"][name]
        if name not in td["character_subcharacters"]:
            td["character_subcharacters"][name] = {}
        self.subchardict = td["character_subcharacters"][name]

    def __str__(self):
        return str(self._name)

    def __unicode__(self):
        return unicode(self._name)

    def get_item_history(self, mydict, *keys):
        if mydict == "thing":
            #(dimension, thing) = keys
            if (
                    keys[0] in self.thingdict and
                    keys[1] in self.thingdict[keys[0]]):
                return self.thingdict[keys[0]][keys[1]]
        elif mydict == "place":
            #(dimension, place) = keys
            if (
                    keys[0] in self.thingdict and
                    keys[1] in self.thingdict[keys[0]]):
                return self.placedict[keys[0]][keys[1]]
        elif mydict == "portal":
            #(dimension, origin, destination) = keys
            if (
                    keys[0] in self.thingdict and
                    keys[1] in self.thingdict[keys[0]] and
                    keys[2] in self.thingdict[keys[0]][keys[1]]):
                return self.portaldict[keys[0]][keys[1]][keys[2]]
        elif mydict == "stat":
            if keys[0] in self.statdict:
                return self.statdict[keys[0]]
        elif mydict == "skill":
            if keys[0] in self.skilldict:
                return self.skilldict[keys[0]]
        else:
            raise AttributeError(
                "I do not have that dictionary.")

    def has_thing_by_key(self, dimension, thing, branch=None, tick=None):
        if tick is None:
            tick = self.closet.tick
        if (
                dimension not in self.thingdict or
                thing not in self.thingdict[dimension] or
                (branch is not None and
                 branch not in self.thingdict[dimension][thing])):
            return False
        if branch is None:
            for branch in self.thingdict[dimension][thing]:
                try:
                    return self.thingdict[dimension][thing][
                        branch].value_during(tick)
                except KeyError:
                    continue
            return None
        else:
            try:
                return self.thingdict[dimension][thing][
                    branch].value_during(tick)
            except KeyError:
                return None

    def add_thing_by_bone(self, bone):
        if self.has_thing_by_key(
                bone.dimension, bone.thing,
                bone.branch, bone.tick_from):
            raise TimeParadox("I already have that then")
        if bone.dimension not in self.thingdict:
            self.thingdict[bone.dimension] = {}
        if bone.thing not in self.thingdict[bone.dimension]:
            self.thingdict[bone.dimension][bone.thing] = {}
        if bone.branch not in self.thingdict[bone.dimension][bone.thing]:
            self.thingdict[bone.dimension][bone.thing][bone.branch] = {}
        self.thingdict[
            bone.dimension][bone.thing][bone.branch][bone.tick_from] = rd

    def add_thing_by_key(self, dimension, thing, branch=None, tick=None):
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        self.add_thing_by_bone(self.bonetypes["character_thing"](
            character=unicode(self),
            dimension=unicode(dimension),
            thing=unicode(thing),
            branch=branch,
            tick_from=tick))

    def add_thing(self, thing, branch=None, tick_from=None, tick_to=None):
        self.add_thing_by_key(
            unicode(thing.dimension), unicode(thing), branch, tick_from, tick_to)

    def has_place_by_key(self, dimension, place, branch=None, tick=None):
        if tick is None:
            tick = self.closet.tick
        if (
                dimension not in self.placedict or
                place not in self.placedict[dimension] or
                (branch is not None and
                 branch not in self.placedict[dimension][place])):
            return False
        if branch is None:
            for branch in self.placedict[dimension][place]:
                try:
                    return self.placedict[dimension][place][branch].value_during(tick) is not None
                except KeyError:
                    continue
            return False
        else:
            try:
                return self.placedict[dimension][place][branch].value_during(tick) is not None
            except KeyError:
                return False

    def has_place(self, place, branch=None, tick=None):
        return self.has_place_by_key(
            unicode(place.dimension), unicode(place), branch, tick)

    def add_place_by_bone(self, bone):
        if self.has_place_by_key(
                bone.dimension, bone.place,
                bone.branch, bone.tick_from):
            raise TimeParadox(
                "Tried to assign Place to Character when "
                "it was already assigned there")
        if bone.dimension not in self.placedict:
            self.placedict[bone.dimension] = {}
        if bone.place not in self.placedict[bone.dimension]:
            self.placedict[bone.dimension][bone.place] = {}
        if bone.branch not in self.placedict[bone.dimension][bone.place]:
            self.placedict[bone.dimension][bone.place][bone.branch] = {}
        self.placedict[
            bone.dimension][bone.place][bone.branch][
                bone.tick_from] = self.bonetypes["character_place"](
                    character=unicode(self),
                    dimension=bone.dimension,
                    place=bone.place,
                    branch=bone.branch,
                    tick_from=bone.tick_from)

    def add_place_by_key(self, dimension, place,
                         branch=None, tick=None):
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        self.placedict[dimension][place][branch][tick] = {
            "character": str(self),
            "dimension": dimension,
            "place": place,
            "branch": branch,
            "tick_from": tick}

    def add_place(self, place, branch=None, tick=None):
        self.add_place_by_key(str(place.dimension), str(place),
                              branch, tick)

    def has_portal_by_key(self, dimension, origin, destination,
                          branch=None, tick=None):
        if tick is None:
            tick = self.closet.tick
        if (
                dimension not in self.portaldict or
                origin not in self.portaldict[dimension] or
                destination not in self.portaldict[dimension][origin] or
                (branch is not None and branch not in
                 self.portaldict[dimension][origin][destination])):
            return False
        if branch is None:
            rditer = self.portaldict[dimension][origin][destination].iterbones()
        else:
            rditer = self.portaldict[dimension][origin][destination][
                branch].iterbones()
        for rd in rditer:
            if rd["tick_from"] >= tick:
                if rd["tick_to"] is None or rd["tick_to"] <= tick:
                    return True
        return False

    def add_portal_by_bone(self, bone):
        if self.has_portal_by_key(
                bone.dimension, bone.origin, bone.destination,
                bone.branch, bone.tick_from):
            raise TimeParadox("I already have that then")
        if bone.dimension not in self.portaldict:
            self.portaldict[bone.dimension] = {}
        if bone.origin not in self.portaldict[bone.dimension]:
            self.portaldict[bone.dimension][bone.origin] = {}
        if bone.destination not in self.portaldict[
                bone.dimension][bone.origin]:
            self.portaldict[bone.dimension][
                bone.origin][bone.destination] = []
        if bone.branch not in self.portaldict[
                bone.dimension][bone.origin][bone.destination]:
            self.portaldict[bone.dimension][
                bone.origin][bone.destination][bone.branch] = []
        self.portaldict[bone.dimension][
            bone.origin][bone.destination][
            bone.branch][bone.tick_from] = rd

    def _portal_by_key(self, dimension, origin, destination, extant,
                       branch=None, tick=None):
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        self.add_portal_by_bone(self.bonetypes["character_portal"](
            character=str(self),
            dimension=dimension,
            origin=origin,
            destination=destination,
            branch=branch,
            tick_from=tick,
            extant=extant))

    def add_portal_by_key(self, dimension, origin, destination,
                          branch=None, tick=None):
        self._portal_by_key(dimension, origin, destination, True, branch, tick)

    def add_portal(self, portal, branch=None, tick=None):
        self.add_portal_by_key(
            str(portal.dimension), str(portal.origin), str(portal.destination),
            branch, tick)

    def remove_portal_by_key(self, dimension, origin, destination,
                             branch=None, tick=None):
        self._portal_by_key(
            dimension, origin, destination, False, branch, tick)

    def remove_portal(self, portal, branch=None, tick=None):
        self.remove_portal_by_key(
            str(portal.dimension), str(portal.origin),
            str(portal.destination), branch, tick)

    def thing_skel_branch_iter(self, branch, dimension=None, thing=None):
        if dimension is None:
            for dimension in self.thingdict:
                for rd in self.thing_skel_branch_iter(
                        branch, dimension, thing):
                    yield rd
        elif thing is None:
            for thing in self.thingdict[dimension]:
                for rd in self.thing_skel_branch_iter(
                        branch, dimension, thing):
                    yield rd
        else:
            for rd in self.thingdict[dimension][thing][branch].iterbones():
                yield rd

    def place_skel_branch_iter(self, branch, dimension=None, place=None):
        if dimension is None:
            for dimension in self.placedict:
                for rd in self.place_skel_branch_iter(
                        branch, dimension, place):
                    yield rd
        elif place is None:
            for place in self.placedict[dimension]:
                for rd in self.place_skel_branch_iter(
                        branch, dimension, place):
                    yield rd
        else:
            for rd in self.placedict[dimension][place][branch].iterbones():
                yield rd

    def portal_skel_branch_iter(
            self, branch, dimension=None,
            origin=None, destination=None):
        if dimension is None:
            for dimension in self.portaldict:
                for rd in self.portal_skel_branch_iter(
                        branch, dimension, origin, destination):
                    yield rd
        elif origin is None:
            for origin in self.portaldict[dimension]:
                for rd in self.portal_skel_branch_iter(
                        branch, dimension, origin, destination):
                    yield rd
        elif destination is None:
            for destination in self.portaldict[dimension][origin]:
                for rd in self.portal_skel_branch_iter(
                        branch, dimension, origin, destination):
                    yield rd
        else:
            for rd in self.portaldict[
                    dimension][origin][destination][branch].iterbones():
                yield rd

    def stat_skel_branch_iter(
            self, branch, stat=None):
        if stat is None:
            for stat in self.statdict:
                for rd in self.statdict[stat][branch].iterbones():
                    yield rd
        else:
            for rd in self.statdict[stat][branch].iterbones():
                yield rd

    def skill_skel_branch_iter(
            self, branch, skill=None):
        if skill is None:
            for skill in self.skilldict:
                for rd in self.skilldict[skill][branch].iterbones():
                    yield rd
        else:
            for rd in self.skilldict[skill][branch].iterbones():
                yield rd

    def new_branch(self, parent, branch, tick):
        l = [
            (self.add_thing_by_rd,
             self.thing_skel_branch_iter(parent))]
        for (assigner, iterator) in l:
            prev = None
            started = False
            for rd in iterator:
                if rd["tick_from"] >= tick:
                    rd2 = dict(rd)
                    rd2["branch"] = branch
                    assigner(rd2)
                    # I have the feeling that this test is too much.
                    if (
                            not started and prev is not None and
                            rd["tick_from"] > tick and
                            prev["tick_from"] < tick):
                        rd3 = dict(prev)
                        rd3["branch"] = branch
                        rd3["tick_from"] = tick
                        assigner(rd3)
                    started = True
                prev = rd
