# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass


"""Things that should have character sheets."""


__metaclass__ = SaveableMetaclass


class Character:
    """An incorporeal object connecting corporeal ones together across
dimensions, indicating that they represent one thing and have that
thing's attributes.

Every item in LiSE's world model must be part of a Character, though
it may be the only member of that Character. Where items can only have
generic attributes appropriate to the dimension they occupy,
Characters have all the attributes of the items that make them up, and
possibly many more. There are no particular restrictions on what
manner of attribute a Character can have, so long as it is not used by
the physics of any dimension.

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

    def __init__(self, rumor, name):
        self.name = name
        self.rumor = rumor
        self.thingdict = {}
        self.indefinite_thing = {}
        self.skilldict = {}
        self.indefinite_skill = {}
        self.statdict = {}
        self.indefinite_stat = {}

    def set_stat(self, stat, val, branch=None, tick_from=None, tick_to=None):
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

    def set_skill(self, skill, val, branch=None, tick_from=None, tick_to=None):
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
            self, dimn, thingn,
            branch=None, tick_from=None, tick_to=None):
        if branch is None:
            branch = self.rumor.branch
        if tick_from is None:
            tick_from = self.rumor.tick
        if branch not in self.thingdict:
            self.thingdict[branch] = {}
        if dimn not in self.thingdict[branch]:
            self.thingdict[branch][dimn] = {}
        if thingn not in self.thingdict[branch][dimn]:
            self.thingdict[branch][dimn][thingn] = {}
        if (
                branch in self.indefinite_thing and
                dimn in self.indefinite_thing[branch] and
                thingn in self.indefinite_thing[branch][dimn]):
            ifrom = self.indefinite_thing[branch][dimn][thingn]
            if tick_from > ifrom:
                self.thingdict[branch][dimn][thingn][ifrom] = tick_from - 1
                del self.indefinite_thing[branch][dimn][thingn]
            elif tick_from == ifrom or tick_to > ifrom:
                del self.thingdict[branch][dimn][thingn][ifrom]
                del self.indefinite_thing[branch][dimn][thingn]
        self.thingdict[branch][dimn][thingn][tick_from] = tick_to
        if tick_to is None:
            if branch not in self.indefinite_thing:
                self.indefinite_thing[branch] = {}
            if dimn not in self.indefinite_thing[branch]:
                self.indefinite_thing[branch][dimn] = {}
            self.indefinite_thing[branch][dimn][thingn] = tick_from

    def add_thing(self, thing, branch=None, tick=None):
        dimn = str(thing.dimension)
        thingn = str(thing)
        self.add_thing_with_strs(dimn, thingn, branch, tick)

    def is_thing_with_strs(self, dimn, thingn, branch=None, tick=None):
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        if not (
                branch in self.thingdict and
                dimn in self.thingdict[branch] and
                thingn in self.thingdict[branch][dimn]):
            return False
        for (tick_from, tick_to) in self.thingdict[
                branch][dimn][thingn].iteritems():
            if tick_from <= tick and (tick_to is None or tick <= tick_to):
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
        for (dimn, thingsd) in self.thingdict[branch].iteritems():
            for (thingn, ticksd) in thingsd.iteritems():
                for (tick_from, tick_to) in ticksd.iteritems():
                    if tick_from <= tick and (
                            tick_to is None or tick <= tick_to):
                        dim = self.rumor.dimensiondict[dimn]
                        thing = dim.things_by_name[thingn]
                        r.add(thing)
        return r
