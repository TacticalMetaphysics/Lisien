# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass, SkeletonIterator, TimeParadox
from thing import Thing


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
          "deck": "text not null"},
         ("character", "skill", "branch", "tick_from"),
         {},
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

    def __init__(self, closet, name):
        assert(len(closet.skeleton['img']) > 1)
        self._name = name
        self.closet = closet
        self.update_handlers = set()
        self.indefinite_skill = {}
        self.indefinite_stat = {}
        self.indefinite_port = {}
        self.indefinite_place = {}
        self.skilldict = {}
        self.statdict = {}
        self.portdict = {}
        self.placedict = {}
        td = self.closet.skeleton
        try:
            self.thingdict = td["character_things"][str(self)]
            self.indefinite_thing = {}
            for rd in self.thingdict.iterrows():
                if rd["dimension"] not in self.indefinite_thing:
                    self.indefinite_thing[rd["dimension"]] = {}
                if rd["thing"] not in self.indefinite_thing[rd["dimension"]]:
                    self.indefinite_thing[rd["dimension"]][rd["thing"]] = {}
                if rd["tick_to"] is None:
                    self.indefinite_thing[rd["dimension"]][
                        rd["thing"]][rd["branch"]] = rd["tick_from"]
        except KeyError:
            if "character_things" not in td:
                td["character_things"] = {}
            if str(self) not in td["character_things"]:
                td["character_things"][str(self)] = {}
            self.thingdict = td["character_things"][str(self)]
            self.indefinite_thing = {}
        try:
            self.statdict = td["character_stats"][str(self)]
            self.indefinite_stat = {}
            for rd in self.statdict.iterrows():
                if rd["stat"] not in self.indefinite_stat:
                    self.indefinite_stat[rd["stat"]] = {}
                if rd["tick_to"] is None:
                    self.indefinite_stat[
                        rd["stat"]][rd["branch"]] = rd["tick_from"]
        except KeyError:
            if "character_stats" not in td:
                td["character_stats"] = {}
            if str(self) not in td["character_stats"]:
                td["character_stats"][str(self)] = {}
            self.statdict = td["character_stats"][str(self)]
            self.indefinite_stat = {}
        try:
            self.skilldict = td["character_skills"][str(self)]
            self.indefinite_skill = {}
            for rd in self.skilldict.iterrows():
                if rd["skill"] not in self.indefinite_skill:
                    self.indefinite_skill[rd["skill"]] = {}
                if rd["tick_to"] is None:
                    self.indefinite_skill[
                        rd["skill"]][rd["branch"]] = rd["tick_from"]
        except KeyError:
            if "character_skills" not in td:
                td["character_skills"] = {}
            if str(self) not in td["character_skills"]:
                td["character_skills"][str(self)] = {}
            self.skilldict = td["character_skills"][str(self)]
            self.indefinite_skill = {}
        try:
            self.portaldict = td["character_portals"][str(self)]
            self.indefinite_portal = {}
            for rd in self.portaldict.iterrows():
                if rd["dimension"] not in self.indefinite_portal:
                    self.indefinite_portal[rd["dimension"]] = {}
                if rd["origin"] not in self.indefinite_portal[rd["dimension"]]:
                    self.indefinite_portal[rd["dimension"]][rd["origin"]] = {}
                if rd["destination"] not in self.indefinite_portal[rd["dimension"]][rd["origin"]]:
                    self.indefinite_portal[rd["dimension"]][rd["origin"]][rd["destination"]]
                if rd["tick_to"] is None:
                    self.indefinite_portal[rd["dimension"]][rd["origin"]][rd["destination"]][rd["branch"]] = rd["tick_from"]
        except KeyError:
            if "character_portals" not in td:
                td["character_portals"] = {}
            if str(self) not in td["character_portals"]:
                td["character_portals"][str(self)] = {}
            self.portaldict = td["character_portals"][str(self)]
            self.indefinite_portal = {}
        try:
            self.placedict = td["character_places"][str(self)]
            self.indefinite_place = {}
            for rd in self.placedict.iterrows():
                if rd["dimension"] not in self.indefinite_place:
                    self.indefinite_place[rd["dimension"]] = {}
                if rd["place"] not in self.indefinite_place[rd["dimension"]]:
                    self.indefinite_place[rd["dimension"]][rd["place"]] = {}
                if rd["tick_to"] is None:
                    self.indefinite_place[rd["dimension"]][rd["place"]][rd["branch"]] = rd["tick_from"]
            self.closet.characterdict[str(self)] = self
        except KeyError:
            if "character_places" not in td:
                td["character_places"] = {}
            if str(self) not in td["character_places"]:
                td["character_places"][str(self)] = {}
            self.placedict = td["character_places"][str(self)]
            self.indefinite_place = {}

    def __str__(self):
        return self._name

    def add_thing_by_key(self, dimension, name,
                         branch=None, tick_from=None, tick_to=None):
        if branch is None:
            branch = self.closet.branch
        if tick_from is None:
            tick_from = self.closet.tick
        if tick_to is None:
            if branch in self.indefinite_thing[dimension][name]:
                ifrom = self.indefinite_thing[dimension][name][branch]
                irow = self.thingdict[dimension][name][branch][ifrom]
                if irow["tick_from"] < tick_from:
                    irow["tick_to"] = tick_from - 1
                    self.indefinite_thing[dimension][name][branch] = tick_from
                else:
                    raise TimeParadox(
                        "I'll have this thing as of {0}, so I need to not have it already at that time".format(ifrom))
            else:
                self.indefinite_thing[dimension][name][branch] = tick_from
        self.thingdict[dimension][name][branch][tick_from] = {
            "dimension": dimension,
            "thing": name,
            "branch": branch,
            "tick_from": tick_from,
            "tick_to": tick_to}

    def add_thing(self, thing, branch=None, tick_from=None, tick_to=None):
        self.add_thing_by_key(str(thing.dimension), str(thing), branch, tick_from, tick_to)

    def register_update_handler(self, that):
        self.update_handlers.add(that)

    def update(self, wot):
        for handler in self.update_handlers:
            handler(self)
