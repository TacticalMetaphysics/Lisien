# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass, dictify_row
from random import randrange


"""Ways to change the game world."""


__metaclass__ = SaveableMetaclass


class Effect:
    """One change to one attribute of one character. It may occur at any given time.

When fired, an effect will call a callback function named in its
constructor. The function should be registered by that name in the
effect_cbs dictionary in the RumorMill of the character (accessible
through the character's rumor attribute). This is how it will be called:

callback(effect, branch, tick)

callback's return value may be:

* a Thing object, which will be added to the character if it isn't part already

* an EffectDeck object, which will be the new value of the skill named in the Effect's constructor

* anything else, which will be the new value of the stat named in the Effect's constructor

The callback should always return the same type.

Note that the character object is an attribute of every effect acting
on it. You can make use of this to have your callbacks depend on the
character's history.

Effects are only ever fired by EffectDecks, though the EffectDeck may
contain only a single Effect.

    """
    tables = [
        ("effect",
         {"name": "text not null",
          "character": "text not null",
          "key": "text",  # May be null, but only when adding a thing to the character
          "callback": "text not null"},
         ("name",),
         {},
         [])]
    def __init__(self, name, chara, key, cbname):
        self.name = name
        self.character = chara
        self.rumor = self.character.rumor
        self.cb = self.db.effect_cbs[cbname]
        self.key = key
        self.occurrences = {}

    def __str__(self):
        return self.name

    def schedule_occurrence(self, branch, tick):
        if branch not in self.occurrences:
            self.occurrences[branch] = set()
        self.occurrences[branch].add(tick)

    def do(self, branch=None, tick=None):
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        nuval = self.cb(self, branch, tick)
        if hasattr(nuval, 'locations'):
            self.character.add_thing(nuval, branch, tick)
        elif hasattr(nuval, 'effects'):
            self.character.set_skill(self.key, nuval, branch, tick)
        else:
            self.character.set_stat(self.key, nuval, branch, tick)

    def occurs(self, branch=None, tick=None):
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        return tick in self.occurrences[branch]


DRAW_FILO = 0
DRAW_FIFO = 1
DRAW_RANDOM = 2
ROLL_RANDOM = 3


class EffectDeck:
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
        ("effect_deck":
         {"name": "text not null",
          "draw": "integer not null default 0"},
         ("name",),
         {},
         ["draw>=0", "draw<=3"]),
        ("effect_deck_link",
         {"deck": "text not null",
          "tick_from": "integer not null default 0",
          "idx": "integer not null default 0",
          "effect": "text not null",
          "tick_to": "integer"},
         ("deck", "tick_from", "idx"),
         {"deck": ("effect_deck", "name"),
          "effect": ("effect", "name")},
         ["idx>=0"])]

    def __init__(self, rumor, name, draw_order=DRAW_FILO):
        self.rumor = rumor
        self.name = name
        self.draw_order = draw_order
        self.reset_to = {}
        self.effects = {}
        self.indefinite_effects = {}

    def __len__(self):
        return len(self.effects)

    def __str__(self):
        return self.name

    def __getattr__(self, attrn):
        if attrn == "effects":
            return self.get_effects()
        else:
            raise AttributeError("EffectDeck instance has no attribute named " + attrn)

    def __setattr__(self, attrn, val):
        if attrn == "effects":
            self.set_effects(val)
        else:
            super(EffectDeck, self).__setattr__(attrn, val)

    def set_effects(self, effects, branch=None, tick_from=None, tick_to=None):
        if branch is None:
            branch = self.rumor.branch
        if tick_from is None:
            tick_from = self.rumor.tick
        if branch in self.indefinite_effects:
            ifrom = self.indefinite_effects[branch]
            (ieffects, ito) = self.effects[branch][ifrom]  # ito should be None
            if tick_from > ito:
                self.effects[branch][ifrom] = (ieffects, tick_from - 1)
                del self.indefinite_effects[branch]
            elif tick_to > ito:
                del self.effects[branch][ifrom]
                del self.indefinite_effects[branch]
        if branch not in self.effects:
            self.effects[branch] = {}
        self.effects[branch][tick_from] = (effects, tick_to)
        if tick_to is None:
            self.indefinite_effects[branch] = tick_from

    def get_effects(self, branch=None, tick=None):
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        if branch in self.indefinite_effects and self.indefinite_effects[branch] <= tick:
            return self.effects[self.indefinite_effects[branch]][0]
        for (tick_from, (val, tick_to)) in self.effects[branch].iteritems():
            if tick_from <= tick and tick <= tick_to:
                return val

    def get_tabdict(self):
        r = {"effect_deck": []}
        for branch in self.effects:
            for (tick_from, (effects, tick_to)) in self.effects[branch].iteritems():
                i = 0
                for effect in effects:
                    r["effect_deck"].append(
                        {"deck": str(self),
                         "tick_from": tick_from,
                         "idx": i,
                         "effect": str(effect),
                         "tick_to": tick_to})
                    i += 1
        return r

    def get_keydict(self):
        r = {"effect_deck": []}
        for branch in self.effects:
            for (tick_from, (effects, tick_to)) in self.effects[branch].iteritems():
                for i in xrange(tick_from, tick_to):
                    r["effect_deck"].append(
                        {"deck": str(self),
                         "tick_from": tick_from,
                         "idx": i})
        return r

    def draw(self, i=None, branch=None, tick=None):

        effs = self.get_effects(branch, tick)
        if i is None:
            if self.draw_order == DRAW_FILO:
                r = effs.pop()
            elif self.draw_order == DRAW_FIFO:
                r = effs.pop(0)
            elif self.draw_order == DRAW_RANDOM:
                i = self.rumor.randrange(0, len(effs)-1)
                r = effs.pop(i)
            elif self.draw_order == ROLL_RANDOM:
                i = self.rumor.randrange(0, len(effs)-1)
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
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        effs = self.get_effects(branch, tick)
        if branch not in self.reset_to:
            self.reset_to[branch] = {}
        self.reset_to[branch][tick] = effs
        for eff in effs:
            eff.do()
        if not reset:
            self.set_effects([], branch, tick)
