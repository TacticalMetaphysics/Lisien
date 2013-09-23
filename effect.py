# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import unicode_literals
ascii = str
str = unicode
from util import SaveableMetaclass
from random import randrange


"""Ways to change the game world."""


__metaclass__ = SaveableMetaclass


class Effect(object):
    """One change to one attribute of one character. It may occur at any
given time.

When fired, an effect will call a callback function named in its
constructor. The function should be registered by that name in the
effect_cbs dictionary in the RumorMill of the character (accessible
through the character's closet attribute). This is how it will be called:

callback(effect, branch, tick)

callback's return value may be:

* a Thing object, which will be added to the character if it isn't part already

* an EffectDeck object, which will be the new value of the skill named
  in the Effect's constructor

* anything else, which will be the new value of the stat named in the
  Effect's constructor

The callback should always return the same type.

Note that the character object is an attribute of every effect acting
on it. You can make use of this to have your callbacks depend on the
character's history.

Effects are only ever fired by EffectDecks, though the EffectDeck may
contain only a single Effect.

    """
    atrdic = {
        "character": lambda self:
        self.closet.get_character(self._rowdict["character"]),
        "key": lambda self: self._rowdict["key"],
        "callback": lambda self:
        self.closet.effect_cbs[self._rowdict["callback"]]}

    tables = [
        ("effect",
         {"name": "text not null",
          "character": "text not null",
          "key": "text",  # May be null, but only when adding a thing
                          # to the character
          "callback": "text not null"},
         ("name",),
         {},
         [])]

    def __init__(self, closet, name):
        self.closet = closet
        self._name = name
        self.closet.effectdict[str(self)] = self
        self._rowdict = self.closet.skeleton["effect"][str(self)]

    def __getattr__(self, attrn):
        try:
            return self.atrdic[attrn]()
        except KeyError:
            raise AttributeError(
                "Effect instance has no attribute named {0}".format(
                    attrn))

    def __str__(self):
        return self._name

    def do(self, branch=None, tick=None):
        """Call the callback, and add its results to this character."""
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        nuval = self.cb(self, branch, tick)
        if hasattr(nuval, 'locations'):
            if self.character.is_thing(nuval, branch, tick):
                # redundant
                return
            self.character.add_thing(nuval, branch, tick)
        elif hasattr(nuval, 'effects'):
            self.character.set_skill(self.key, nuval, branch, tick)
        else:
            self.character.set_stat(self.key, nuval, branch, tick)


DRAW_FILO = 0
DRAW_FIFO = 1
DRAW_RANDOM = 2
ROLL_RANDOM = 3


class EffectDeck(object):
    """Several effects in a specified order, which may happen all at once
or one at a time.

The order may change during play, and effects may be added or
removed. An EffectDeck may contain many of a single event, in which
case it's to be fired that many times.

The effects may be 'drawn' from the deck in one of four orders:
DRAW_FILO ("from the top"), DRAW_FIFO ("from the bottom"),
DRAW_RANDOM, and ROLL_RANDOM. The difference between the two random
draw-orders is in what happens after the draw. If the order is
ROLL_RANDOM, the effect that was drawn will remain in the deck,
causing it to behave more like a die. DRAW_RANDOM will discard the
effect as usual. Though the contents of the deck may change over time,
the draw-order is eternal.

You may supply an index to the draw() method to draw an effect from a
particular point in the deck. In any case, effects drawn from
EffectDecks still need to be fired, if indeed that's what you want to
do with it.

Call the do() method to draw and fire all the effects. Give the
keyword argument reset=True to reset the effects back to where they
were right after firing them.

    """
    tables = [
        ("effect_deck",
         {"name": "text not null",
          "draw_order": "integer not null default 0"},
         ("name",),
         {},
         ["draw_order>=0", "draw_order<=3"]),
        ("effect_deck_link",
         {"deck": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null default 0",
          "idx": "integer not null default 0",
          "effect": "text not null"},
         ("deck", "branch", "tick_from", "idx"),
         {"deck": ("effect_deck", "name"),
          "effect": ("effect", "name")},
         ["idx>=0"])]

    atrdic = {
        "effects": lambda self: self.get_effects(),
        "draw_order": lambda self: self._rowdict["draw_order"]}

    def __init__(self, closet, name):
        assert(len(closet.skeleton['img']) > 1)
        self.closet = closet
        self._name = name
        self.reset_to = {}
        self.indefinite_effects = {}
        self.closet.effectdeckdict[name] = self
        self._rowdict = self.closet.skeleton["effect_deck"][str(self)]
        self._card_links = self.closet.skeleton["effect_deck_link"][str(self)]

    def __len__(self):
        return len(self.effects)

    def __str__(self):
        return self._name

    def __getattr__(self, attrn):
        try:
            return self.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                "EffectDeck instance has no attribute named " + attrn)

    def __setattr__(self, attrn, val):
        if attrn == "effects":
            self.set_effects(val)
        else:
            super(EffectDeck, self).__setattr__(attrn, val)

    def set_effects(self, effects, branch=None, tick=None):
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        if branch not in self.effects:
            self.effects[branch] = []
        i = 0
        for effect in effects:
            self._card_links[branch][tick][i] = {
                "deck": str(self),
                "branch": branch,
                "tick_from": tick,
                "idx": i,
                "effect": effect}
            i += 1

    def get_effects(self, branch=None, tick=None):
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        tick_from = None
        r = []
        for rd in self._card_links[branch].iterrows():
            if tick_from is None:
                if rd["tick_from"] >= tick:
                    tick_from = rd["tick_from"]
                    r.append(rd["name"])
            elif rd["tick_from"] == tick_from:
                r.append(rd["name"])
            else:
                effd = self.closet.get_effects(r)
                return [effd[effn] for effn in r]
        return []

    def draw(self, i=None, branch=None, tick=None):
        effs = self.get_effects(branch, tick)
        if i is None:
            if self.draw_order == DRAW_FILO:
                r = effs.pop()
            elif self.draw_order == DRAW_FIFO:
                r = effs.pop(0)
            elif self.draw_order == DRAW_RANDOM:
                i = randrange(0, len(effs) - 1)
                r = effs.pop(i)
            elif self.draw_order == ROLL_RANDOM:
                i = randrange(0, len(effs) - 1)
                r = effs[i]
            else:
                raise Exception("What kind of draw order is that?")
        else:
            r = effs.pop(i)
        if self.draw_order != ROLL_RANDOM:
            self.set_effects(effs, branch, tick)
        return r

    def do(self, reset=False, branch=None, tick=None):
        if branch is None:
            branch = self.closet.branch
        if tick is None:
            tick = self.closet.tick
        effs = self.get_effects(branch, tick)
        if branch not in self.reset_to:
            self.reset_to[branch] = {}
        self.reset_to[branch][tick] = effs
        for eff in effs:
            eff.do()
        if not reset:
            self.set_effects([], branch, tick)
