# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import unicode_literals
ascii = str
str = unicode
from util import SaveableMetaclass, TimeParadox
from collections import deque


"""Things that should have character sheets."""


__metaclass__ = SaveableMetaclass


class ThingSkelBranchIter(object):
    def __init__(self, thingdict, branch):
        self.thingdict = thingdict
        self.branch = branch
        self.level1 = self.thingdict.iterkeys()
        self.k1 = self.level1.next()
        self.level2 = self.thingdict[self.k1].iterkeys()
        self.k2 = self.level2.next()
        self.level3 = self.thingdict[self.k1][self.k2][self.branch].iterrows()

    def __iter__(self):
        return self

    def next(self):
        try:
            return self.level3.next()
        except StopIteration:
            try:
                self.k2 = self.level2.next()
                self.level3 = self.thingdict[
                    self.k1][self.k2][self.branch].iterrows()
                return self.level3.next()
            except StopIteration:
                self.k1 = self.level1.next()
                self.level2 = self.thingdict[self.k1].iterkeys()
                self.k2 = self.level2.next()
                self.level3 = self.thingdict[
                    self.k1][self.k2][self.branch].iterrows()
                return self.level3.next()


class Character:
    """An incorporeal object connecting corporeal ones together across
dimensions, indicating that they represent one thing and have that
thing's attributes.

Every item in LiSE's world model must be part of a Character, though
it may be the only member of that Character. Where items can only have
generic attributes, Characters have all the attributes of the items
that make them up, and possibly many more. There are no particular
restrictions on what manner of attribute a Character can have, so long
as it is not used by the physics of any dimension.

Characters may contain EventDecks. These may represent skills the
character has, in which case every EventCard in the EventDeck
represents something can happen upon using the skill, regardless of
what it's used on or where. "Success" and "failure" are appropriate
EventCards in this case, though there may be finer distinctions to be
made between various kinds of success and failure with a given skill.

However, the EventCards that go in a Character's EventDeck to
represent a skill should never represent anything particular to any
use-case of the skill. Those EventCards should instead be in the
EventDeck of those other Characters--perhaps people, perhaps places,
perhaps tools--that the skill may be used on, with, or for. All of
those Characters' relevant EventDecks will be used in constructing a
new one, called the OutcomeDeck, and the outcome of the event will be
drawn from that.

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
        "SELECT character FROM character_skills UNION "
        "SELECT character FROM character_stats"]
    demands = ["thing_location", "portal"]
    provides = ["character"]
    tables = [
        ("character_things",
         {"character": "text not null",
          "dimension": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "tick_to": "integer default null",
          "thing": "text not null"},
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
        ("character_skills",
         {"character": "text not null",
          "skill": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "deck": "text"},
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
        assert(len(closet.skeleton['img']) > 1)
        self._name = name
        self.closet = closet
        self.update_handlers = set()
        td = self.closet.skeleton
        if "character_things" not in td:
            td["character_things"] = {}
        if str(self) not in td["character_things"]:
            td["character_things"][str(self)] = {}
        self.thingdict = td["character_things"][str(self)]
        if "character_stats" not in td:
            td["character_stats"] = {}
        if str(self) not in td["character_stats"]:
            td["character_stats"][str(self)] = {}
        self.statdict = td["character_stats"][str(self)]
        if "character_skills" not in td:
            td["character_skills"] = {}
        if str(self) not in td["character_skills"]:
            td["character_skills"][str(self)] = {}
        self.skilldict = td["character_skills"][str(self)]
        if "character_portals" not in td:
            td["character_portals"] = {}
        if str(self) not in td["character_portals"]:
            td["character_portals"][str(self)] = {}
        self.portaldict = td["character_portals"][str(self)]
        if "character_places" not in td:
            td["character_places"] = {}
        if str(self) not in td["character_places"]:
            td["character_places"][str(self)] = {}
        self.placedict = td["character_places"][str(self)]

    def __str__(self):
        return self._name

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
            rditer = self.thingdict[dimension][thing].iterrows()
        else:
            rditer = self.thingdict[dimension][thing][branch].iterrows()
        for rd in rditer:
            if rd["tick_from"] <= tick:
                if rd["tick_to"] is None or rd["tick_to"] <= tick:
                    return True
        return False

    def add_thing_by_rd(self, rd):
        if self.has_thing_by_key(
                rd["dimension"], rd["thing"],
                rd["branch"], rd["tick_from"]):
            raise TimeParadox("I already have that then")
        if rd["dimension"] not in self.thingdict:
            self.thingdict[rd["dimension"]] = {}
        if rd["thing"] not in self.thingdict[rd["dimension"]]:
            self.thingdict[rd["dimension"]][rd["thing"]] = {}
        if rd["branch"] not in self.thingdict[rd["dimension"]][rd["thing"]]:
            self.thingdict[rd["dimension"]][rd["thing"]][rd["branch"]] = {}
        self.thingdict[
            rd["dimension"]][rd["thing"]][rd["branch"]][rd["tick_from"]] = rd

    def add_thing_by_key(self, dimension, thing, branch=None, tick=None):
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        self.add_thing_by_rd({
            "character": str(self),
            "dimension": str(dimension),
            "thing": str(thing),
            "branch": branch,
            "tick_from": tick})

    def add_thing(self, thing, branch=None, tick_from=None, tick_to=None):
        self.add_thing_by_key(
            str(thing.dimension), str(thing), branch, tick_from, tick_to)

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
            rditer = self.placedict[dimension][place].iterrows()
        else:
            rditer = self.placedict[dimension][place][branch].iterrows()
        for rd in rditer:
            if rd["tick_from"] <= tick:
                if rd["tick_to"] is None or rd["tick_to"] <= tick:
                    return True
        return False

    def has_place(self, place, branch=None, tick=None):
        return self.has_place_by_key(
            str(place.dimension), str(place), branch, tick)

    def add_place_by_rd(self, rd):
        if self.has_place_by_key(
                rd["dimension"], rd["place"],
                rd["branch"], rd["tick_from"]):
            raise TimeParadox(
                "Tried to assign Place to Character when "
                "it was already assigned there")
        if rd["dimension"] not in self.placedict:
            self.placedict[rd["dimension"]] = {}
        if rd["place"] not in self.placedict[rd["dimension"]]:
            self.placedict[rd["dimension"]][rd["place"]] = {}
        if rd["branch"] not in self.placedict[rd["dimension"]][rd["place"]]:
            self.placedict[rd["dimension"]][rd["place"]][rd["branch"]] = {}
        self.placedict[
            rd["dimension"]][rd["place"]][rd["branch"]][rd["tick_from"]] = {
            "character": str(self),
            "dimension": rd["dimension"],
            "place": rd["place"],
            "branch": rd["branch"],
            "tick_from": rd["tick_from"]}

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
            rditer = self.portaldict[dimension][origin][destination].iterrows()
        else:
            rditer = self.portaldict[dimension][origin][destination][
                branch].iterrows()
        for rd in rditer:
            if rd["tick_from"] >= tick:
                if rd["tick_to"] is None or rd["tick_to"] <= tick:
                    return True
        return False

    def add_portal_by_rd(self, rd):
        if self.has_portal_by_key(
                rd["dimension"], rd["origin"], rd["destination"],
                rd["branch"], rd["tick_from"]):
            raise TimeParadox("I already have that then")
        if rd["dimension"] not in self.portaldict:
            self.portaldict[rd["dimension"]] = {}
        if rd["origin"] not in self.portaldict[rd["dimension"]]:
            self.portaldict[rd["dimension"]][rd["origin"]] = {}
        if rd["destination"] not in self.portaldict[
                rd["dimension"]][rd["origin"]]:
            self.portaldict[rd["dimension"]][
                rd["origin"]][rd["destination"]] = []
        if rd["branch"] not in self.portaldict[
                rd["dimension"]][rd["origin"]][rd["destination"]]:
            self.portaldict[rd["dimension"]][
                rd["origin"]][rd["destination"]][rd["branch"]] = []
        self.portaldict[rd["dimension"]][
            rd["origin"]][rd["destination"]][
            rd["branch"]][rd["tick_from"]] = rd

    def _portal_by_key(self, dimension, origin, destination, extant,
                       branch=None, tick=None):
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        self.add_portal_by_rd({
            "character": str(self),
            "dimension": dimension,
            "origin": origin,
            "destination": destination,
            "branch": branch,
            "tick_from": tick,
            "extant": extant})

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

    def get_stat_rd_triad(self, name, value, branch=None, tick=None):
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        r = deque([], 3)
        for rd in self.statdict[branch][tick].iterrows():
            if rd["stat"] == name and value in (
                    "ANY_VALUE", rd["value"]):
                r.append(rd)
            if tick is not None and rd["tick_from"] > tick:
                break
        return tuple(r)

    def has_stat_with_value(self, name, value,
                            branch=None, tick=None):
        rds = self.get_stat_rd_triad(name, value, branch, tick)
        return rds[1] is not None and rds[1]["value"] == value

    def get_stat_value(self, name, branch=None, tick=None):
        rds = self.get_stat_rd_triad(name, None, branch, tick)
        return rds[1]["value"]

    def has_stat(self, name, branch=None, tick=None):
        rds = self.get_stat_rd_triad(name, "ANY_VALUE", branch, tick)
        return rds[1] is not None

    def add_stat_by_rd(self, rd):
        if self.has_stat(rd["stat"], rd["branch"], rd["tick_from"]):
            raise TimeParadox("I already have that then")
        if rd["stat"] not in self.statdict:
            self.statdict[rd["stat"]] = []
        if rd["branch"] not in self.statdict[rd["stat"]]:
            self.statdict[rd["stat"]] = []
        self.statdict[rd["stat"]][rd["branch"]][rd["tick_from"]] = rd

    def add_stat(self, name, value, branch=None, tick=None):
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        self.add_stat_by_rd({
            "character": str(self),
            "stat": name,
            "value": value,
            "branch": branch,
            "tick_from": tick})

    def remove_stat(self, name, branch=None, tick=None):
        self.add_stat(name, None, branch, tick)

    def get_skill_rd_triad(self, name, branch=None, tick=None):
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        r = deque([], 3)
        for rd in self.skilldict[name].iterrows():
            r.append(rd)
            if rd["tick_from"] > tick:
                break
        return tuple(r)

    def has_skill(self, name, branch=None, tick=None):
        rds = self.get_skill_rd_triad(name, branch, tick)
        return None not in (rds[1], rds[1]["deck"])

    def add_skill_by_rd(self, rd):
        if self.has_skill(rd["skill"], rd["branch"], rd["tick_from"]):
            raise TimeParadox("I already have that then")
        if rd["skill"] not in self.skilldict:
            self.skilldict[rd["skill"]] = []
        if rd["branch"] not in self.skilldict[rd["skill"]]:
            self.skilldict[rd["skill"]][rd["branch"]] = []
        self.skilldict[rd["skill"]][rd["branch"]][rd["tick_from"]] = rd

    def add_skill(self, name, deck, branch=None, tick=None):
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        self.add_skill_by_rd({
            "character": str(self),
            "skill": name,
            "deck": deck,
            "branch": branch,
            "tick_from": tick})

    def remove_skill(self, name, branch=None, tick=None):
        self.add_skill(name, None, branch, tick)

    def new_branch(self, parent, branch, tick):
        l = [
            (self.add_thing_by_rd,
             ThingSkelBranchIter(self.thingdict, parent))]
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
