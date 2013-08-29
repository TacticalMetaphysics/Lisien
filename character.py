# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass, TabdictIterator
from thing import Thing
from collections import defaultdict


"""Things that should have character sheets."""


__metaclass__ = SaveableMetaclass


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
          "tick_to": "integer default null",
          "effect_deck": "text not null"},
         ("character", "skill", "branch", "tick_from"),
         {"effect_deck": ("effect_deck", "name")},
         []),
        ("character_stats",
         {"character": "text not null",
          "stat": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "tick_to": "integer default null",
          "value": "text not null"},
         ("character", "stat", "branch", "tick_from"),
         {},
         [])]

    def __init__(self, rumor, name, td):
        self._name = name
        self.rumor = rumor
        self.update_handlers = set()
        self.indefinite_skill = {}
        self.indefinite_stat = {}
        self.indefinite_port = {}
        self.indefinite_place = {}
        self.skilldict = {}
        self.statdict = {}
        self.portdict = {}
        self.placedict = {}
        if "character_things" in td and str(self) in td["character_things"]:
            self.thingdict = {}
            self.indefinite_thing = {}
            for rd in TabdictIterator(td["character_things"][str(self)]):
                self.add_thing_with_strs(**rd)
                self.rumor.get_thing(
                    rd["dimension"],
                    rd["thing"]).register_update_handler(self.update)
        if "character_stats" in td and str(self) in td["character_stats"]:
            for rd in TabdictIterator(td["character_stats"][str(self)]):
                self.set_stat(**rd)
        if "character_skills" in td and str(self) in td["character_skills"]:
            for rd in TabdictIterator(td["character_skills"][str(self)]):
                self.set_skill(**rd)
        if "character_portals" in td and str(self) in td["character_portals"]:
            for rd in TabdictIterator(td["character_portals"][str(self)]):
                self.add_portal_with_strs(**rd)
        if "character_places" in td and str(self) in td["character_places"]:
            for rd in TabdictIterator(td["character_places"][str(self)]):
                self.add_place_with_strs(**rd)
        self.rumor.characterdict[str(self)] = self

    def __str__(self):
        return self._name

    def set_stat(
            self, stat, val,
            branch=None, tick_from=None, tick_to=None, **kwargs):
        if branch is None:
            branch = self.rumor.branch
        if tick_from is None:
            tick_from = self.rumor.tick
        if branch not in self.attribdict:
            self.attribdict[branch] = {}
        if stat not in self.statdict[branch]:
            self.statdict[branch][stat] = {}
        if (
                branch in self.indefinite_stat and
                stat in self.indefinite_stats[branch]):
            ifrom = self.indefinite_stat[branch][stat]
            (ival, ito) = self.statdict[branch][ifrom][stat]
            if tick_from > ifrom:
                self.attribdict[branch][stat][ifrom] = (ival, tick_from - 1)
                del self.indefinite_attrib[branch][stat]
            elif tick_from == ifrom or tick_to > ifrom:
                del self.attribdict[branch][stat][ifrom]
                del self.indefinite_stat[branch][stat]
        self.attribdict[branch][stat][tick_from] = (val, tick_to)
        if tick_to is None:
            if branch not in self.indefinite_attrib:
                self.indefinite_attrib[branch] = {}
            self.indefinite_attrib[branch][stat] = tick_from

    def get_stat(self, stat, branch=None, tick=None):
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        if branch not in self.statdict:
            return None
        for (tick_from, (val, tick_to)) in self.statdict[branch].iteritems():
            if tick_from <= tick and (tick_to is None or tick <= tick_to):
                return val
        return None

    def set_skill(
            self, skill, val,
            branch=None, tick_from=None, tick_to=None, **kwargs):
        if branch is None:
            branch = self.rumor.branch
        if tick_from is None:
            tick_from = self.rumor.tick
        if branch not in self.skilldict:
            self.skilldict[branch] = {}
        if skill not in self.skilldict[branch]:
            self.skilldict[branch][skill] = {}
        if (
                branch in self.indefinite_skill and
                skill in self.indefinite_skill[branch]):
            ifrom = self.indefinite_skill[branch][skill]
            (ival, ito) = self.skilldict[branch][skill][ifrom]
            if tick_from > ifrom:
                self.skilldict[branch][skill][ifrom] = (ival, tick_from - 1)
                del self.indefinite_skill[branch][skill]
            elif tick_from == ifrom or tick_to > ifrom:
                del self.skilldict[branch][skill][ifrom]
                del self.indefinite_skill[branch][skill]
        self.skilldict[branch][skill][tick_from] = (val, tick_to)
        if tick_to is None:
            if branch not in self.indefinite_skill:
                self.indefinite_skill[branch] = {}
            self.indefinite_skill[branch][skill] = tick_from

    def get_skill(self, skill, branch=None, tick=None):
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        if branch not in self.skilldict:
            return None
        for (tick_from, (val, tick_to)) in self.skilldict[branch].iteritems():
            if tick_from <= tick and (tick_to is None or tick <= tick_to):
                return val
        return None

    def add_thing_with_strs(
            self, dimension, thing,
            branch=None, tick_from=None, tick_to=None, **kwargs):
        if branch is None:
            branch = self.rumor.branch
        if tick_from is None:
            tick_from = self.rumor.tick
        if branch not in self.thingdict:
            self.thingdict[branch] = {}
        if dimension not in self.thingdict[branch]:
            self.thingdict[branch][dimension] = {}
        if thing not in self.thingdict[branch][dimension]:
            self.thingdict[branch][dimension][thing] = {}
        if branch not in self.indefinite_thing:
            self.indefinite_thing[branch] = {}
        if dimension not in self.indefinite_thing[branch]:
            self.indefinite_thing[branch][dimension] = {}
        if thing not in self.indefinite_thing[branch][dimension]:
            self.indefinite_thing[branch][dimension][thing] = {}
        try:
            ifrom = self.indefinite_thing[branch][dimension][thing]
            if tick_from > ifrom:
                self.thingdict[branch][dimension][thing][ifrom] = tick_from - 1
                del self.indefinite_thing[branch][dimension][thing]
            elif tick_from == ifrom or tick_to > ifrom:
                del self.thingdict[branch][dimension][thing][ifrom]
                del self.indefinite_thing[branch][dimension][thing]
        except KeyError:
            pass
        self.thingdict[branch][dimension][thing][tick_from] = tick_to
        if tick_to is None:
            self.indefinite_thing[branch][dimension][thing] = tick_from

    def add_thing(self, thing, branch=None, tick=None):
        dimn = str(thing.dimension)
        thingn = str(thing)
        self.add_thing_with_strs(dimn, thingn, branch, tick)

    def rm_thing_with_strs(self, dimension, thing, branch, tick):
        try:
            del self.indefinite_thing[branch][dimension][thing]
        except KeyError:
            pass
        del self.thingdict[branch][dimension][thing][tick]

    def rm_thing(self, thing, branch, tick):
        dimn = str(thing.dimension)
        thingn = str(thing)
        self.rm_thing_with_strs(dimn, thingn, branch, tick)

    def is_thing_with_strs(self, dimension, thing, branch=None, tick=None):
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        if not (
                branch in self.thingdict and
                dimension in self.thingdict[branch] and
                thing in self.thingdict[branch][dimension]):
            return False
        if (
                branch in self.indefinite_thing and
                dimension in self.indefinite_thing[branch] and
                thing in self.indefinite_thing[branch][dimension]):
            tick_from = self.indefinite_thing[branch][dimension][thing]
            if tick >= tick_from:
                return True
        for (tick_from, tick_to) in self.thingdict[branch][dimension][thing].iteritems():
            if tick_from <= tick and tick <= tick_to:
                return True
        return False

    def is_thing(self, thing, branch=None, tick=None):
        dimn = str(thing.dimension)
        thingn = str(thing)
        return self.is_thing_with_strs(dimn, thingn, branch, tick)

    def were_thing_with_strs(self, dimn, thingn):
        """Was I ever, will I ever, be this thing?"""
        for branch in self.thingdict:
            if dimn not in self.thingdict[branch]:
                continue
            if thingn in self.thingdict[branch][dimn]:
                return True
        return False

    def get_things(self, branch=None, tick=None):
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        r = set()
        if branch not in self.thingdict:
            return r
        for rd in TabdictIterator[self.thingdict[branch]]:
            r.add(self.rumor.get_thing(rd["dimension"], rd["thing"]))
        return r

    def add_place_with_strs(
            self, dimension, place,
            branch=None, tick_from=None, tick_to=None, **kwargs):
        if branch is None:
            branch = self.rumor.branch
        if tick_from is None:
            tick_from = self.rumor.tick
        if (
                branch in self.indefinite_place and
                dimension in self.indefinite_place[branch] and
                place in self.indefinite_place[branch][dimension]):
            ifrom = self.indefinite_place[branch][dimension][place]
            if tick_from > ifrom:
                self.placedict[branch][dimension][place][ifrom] = tick_from - 1
                del self.indefinite_place[branch][dimension][place]
            elif tick_to > ifrom:
                del self.placedict[branch][dimension][place][ifrom]
                del self.indefinite_place[branch][dimension][place]
            if tick_to == ifrom:
                self.placedict[branch][dimension][place][tick_from] = None
                self.indefinite_place[branch][dimension][place] = tick_from
                return
        self.placedict[branch][dimension][place][tick_from] = tick_to
        if tick_to is None:
            self.indefinite_thing[branch][dimension][place] = tick_from

    def add_place(
            self, place, branch=None, tick_from=None, tick_to=None):
        dimn = str(place.dimension)
        placen = str(place)
        self.add_place_with_strs(dimn, placen, branch, tick_from, tick_to)

    def rm_place_with_strs(self, dimension, place, branch, tick):
        try:
            del self.indefinite_place[branch][dimension][place]
        except KeyError:
            pass
        del self.placedict[branch][dimension][place][tick]

    def rm_place(self, place, branch, tick):
        dimn = str(place.dimension)
        placen = str(place)
        self.rm_place_with_strs(dimn, placen, branch, tick)

    def is_place_with_strs(self, dimension, place, branch=None, tick=None):
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        if not (
                branch in self.placedict and
                dimension in self.placedict[branch] and
                place in self.placedict[branch][dimension]):
            return False
        for rd in TabdictIterator(self.placedict[branch][dimension][place]):
            if rd["tick_from"] <= tick and (
                    rd["tick_to"] is None or tick <= rd["tick_to"]):
                return True
        return False

    def get_places(self, branch=None, tick=None):
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        r = set()
        if branch not in self.placedict:
            return r
        for rd in TabdictIterator(self.placedict[branch]):
            if rd["tick_from"] <= r and (
                    rd["tick_to"] is None or tick <= rd["tick_to"]):
                r.add(self.rumor.get_place(rd["dimension"], rd["place"]))
        return r

    def is_place(self, place, branch=None, tick=None):
        dimn = str(place.dimension)
        placen = str(place)
        return self.is_place_with_strs(dimn, placen, branch, tick)

    def were_place_with_strs(self, dimension, place):
        for branch in self.placedict:
            if dimension not in self.placedict[branch]:
                continue
            if place in self.placedict[branch][dimension]:
                return True
        return False

    def were_place(self, place):
        return self.were_place_with_strs(str(place.dimension), str(place))

    def add_portal_with_strs(
            self, dimension, origin, destination,
            branch=None, tick_from=None, tick_to=None, **kwargs):
        if branch is None:
            branch = self.rumor.branch
        if tick_from is None:
            tick_from = self.rumor.tick
        try:
            ifrom = self.indefinite_portal[
                branch][dimension][origin][destination]
            if tick_from > ifrom:
                self.portaldict[
                    branch][dimension][origin][destination][ifrom] = {
                        "dimension": dimension,
                        "origin": origin,
                        "destination": destination,
                        "tick_from": ifrom,
                        "tick_to": tick_from - 1}
                del self.indefinite_portal[
                    branch][dimension][origin][destination]
            elif tick_to > ifrom:
                del self.portaldict[
                    branch][dimension][origin][destination][ifrom]
                del self.indefinite_portal[
                    branch][dimension][origin][destination]
            if tick_to == ifrom:
                self.portaldict[
                    branch][dimension][origin][destination][tick_from] = {
                        "dimension": dimension,
                        "origin": origin,
                        "destination": destination,
                        "tick_from": tick_from,
                        "tick_to": None}
                self.indefinite_portal[
                    branch][dimension][origin][destination] = tick_from
                return
        except KeyError:
            pass
        self.portaldict[
            branch][dimension][origin][destination][tick_from] = tick_to
        if tick_to is None:
            self.indefinite_portal[
                branch][dimension][origin][destination] = tick_from

    def add_portal(self, portal, branch=None, tick_from=None, tick_to=None):
        dimn = str(portal.dimension)
        orign = str(portal.origin)
        destn = str(portal.destination)
        self.add_portal_with_strs(
            dimn, orign, destn, branch, tick_from, tick_to)

    def rm_portal_with_strs(
            self, dimension, origin, destination, branch, tick):
        try:
            del self.indefinite_portal[branch][dimension][origin][destination]
        except KeyError:
            pass
        del self.portdict[branch][dimension][origin][destination]

    def rm_portal(self, portal, branch, tick):
        dimn = str(portal.dimension)
        orign = str(portal.origin)
        destn = str(portal.destination)
        self.rm_portal_with_strs(dimn, orign, destn, branch, tick)

    def is_portal_with_strs(
            self, dimension, origin, destination, branch=None, tick=None):
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        if not (
                branch in self.portdict and
                dimension in self.portdict[branch] and
                origin in self.portdict[branch][dimension] and
                destination in self.portdict[branch][dimension][origin]):
            return False
        for rd in TabdictIterator(
                self.portdict[branch][dimension][origin][destination]):
            if rd["tick_from"] <= tick and (
                    rd["tick_to"] is None or tick <= rd["tick_to"]):
                return True
        return False

    def is_portal(self, portal, branch=None, tick=None):
        dimn = str(portal.dimension)
        orign = str(portal.origin)
        destn = str(portal.destination)
        return self.is_portal_with_strs(dimn, orign, destn, branch, tick)

    def get_portals(self, branch=None, tick=None):
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        r = set()
        if branch not in self.portdict:
            return r
        for rd in TabdictIterator(self.portdict[branch]):
            if rd["tick_from"] <= tick and (
                    rd["tick_to"] is None or tick <= rd["tick_to"]):
                port = self.rumor.get_portal(
                    rd["dimension"], rd["origin"], rd["destination"])
                r.add(port)
        return r

    def were_portal_with_strs(self, dimension, origin, destination):
        for branch in self.portdict:
            if dimension not in self.portdict[branch]:
                continue
            if (
                    origin in self.portdict[branch][dimension] and
                    destination in self.portdict[branch][dimension][origin]):
                return True
        return False

    def were_portal(self, portal):
        dimn = str(portal.dimension)
        orign = str(portal.origin)
        destn = str(portal.destination)
        return self.were_portal_with_strs(dimn, orign, destn)

    def register_update_handler(self, that):
        self.update_handlers.add(that)

    def update(self, wot):
        if (isinstance(wot, Thing) and self.is_thing(wot)):
            for handler in self.update_handlers:
                handler(self)
