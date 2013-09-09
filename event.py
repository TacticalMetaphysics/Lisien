# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Containers for EffectDecks that have beginnings, middles, and ends.

Events, in LiSE, resemble events in programming generally insofar as
you register listeners in the form of EffectDecks, and the listeners
get called when the event is fired. Events here actually have *three*
EffectDecks registered, one each for the moment the event starts, the
moment it ends, and the various ticks in between.

Events get passed to the effect decks, which may or may not use them
for anything in particular.

"""
from util import SaveableMetaclass, dictify_row
import logging


logger = logging.getLogger(__name__)


class SenselessEvent(Exception):
    """Exception raised when trying to fire events that can't happen
anywhere ever."""
    pass


class ImpossibleEvent(Exception):
    """Exception raised when trying to fire events that can't happen given
the present circumstances."""
    pass


class IrrelevantEvent(Exception):
    """Exception raised when trying to fire events that could happen if
they hadn't already, or if some other event hadn't already done the
same thing."""
    pass


class ImpracticalEvent(Exception):
    """Exception raised when trying to fire events that could happen, but
are bad ideas. They may be allowed to pass anyway if the characters
involved *insist*."""
    pass


__metaclass__ = SaveableMetaclass


class Event:
    """A class for things that happen over time, having a beginning, a
middle, and an end.

Events are composed of three EffectDecks, called 'commence,'
'proceed,' and 'conclude.' Events are scheduled with a tick-from
and a tick-to. commence will be fired on tick-from; conclude on
tick-to; and proceed on every tick between them, non-inclusive.

    """
    tables = [
        ("event",
         {"name": "text not null",
          "text": "text not null",
          "commence": "text",
          "proceed": "text",
          "conclude": "text"},
         ("name",),
         {"commence": ("effect_deck", "name"),
          "proceed": ("effect_deck", "name"),
          "conclude": ("effect_deck", "name")},
         []),
        ("scheduled_event",
         {"event": "text not null",
          "branch": "integer not null default 0",
          "tick_from": "integer not null",
          "tick_to": "integer not null"},
         ("event", "branch", "tick_from"),
         {"event": ("event", "name")},
         [])]

    def __init__(self, db, name,
                 text, commence_effects,
                 proceed_effects, conclude_effects):
        """Return an Event with the given name, text, and ongoing-status, and
the three given effect decks. Register with db.eventdict.

        """
        self.name = name
        self._text = text
        self.commence_effects = commence_effects
        self.proceed_effects = proceed_effects
        self.conclude_effects = conclude_effects
        self.occurrences = {}
        self.db = db

    def __getattr__(self, attrn):
        if attrn == "text":
            if self._text[0] == "@":
                return self.db.get_text[self._text[1:]]
            else:
                return self._text
        else:
            raise AttributeError("Event has no such attribute")

    def __repr__(self):
        if hasattr(self, 'start') and hasattr(self, 'length'):
            return "{0}[{1}->{2}]".format(
                str(self),
                self.start,
                self.end)

    def __len__(self):
        return self.end - self.start

    def schedule(self, branch, tick_from, tick_to):
        if branch not in self.occurrences:
            self.occurrences[branch] = {}
        self.occurrences[tick_from] = tick_to

    def is_happening(self, branch=None, tick=None):
        if branch is None:
            branch = self.db.branch
        if tick is None:
            tick = self.db.tick
        if branch not in self.occurrences:
            return False
        for (tick_from, tick_to) in self.occurrences.iteritems():
            if tick_from <= tick and tick <= tick_to:
                return True
        return False

    def is_commencing(self, branch=None, tick=None):
        if branch is None:
            branch = self.db.branch
        if tick is None:
            tick = self.db.tick
        if branch not in self.occurrences:
            return False
        return tick in self.occurrences[branch]

    def is_proceeding(self, branch=None, tick=None):
        if branch is None:
            branch = self.db.branch
        if tick is None:
            tick = self.db.tick
        if branch not in self.occurrences:
            return False
        for (tick_from, tick_to) in self.occurrences[branch].iteritems():
            if tick_from < tick and tick < tick_to:
                return True
        return False

    def is_concluding(self, branch=None, tick=None):
        if branch is None:
            branch = self.db.branch
        if tick is None:
            tick = self.db.tick
        if branch not in self.occurrences:
            return False
        return tick in self.occurrences[branch].values()

    def commence(self, reset=True, branch=None, tick=None):
        """Perform all commence effects."""
        if self.commence_effects is not None:
            self.commence_effects.do(
                self, reset, branch, tick)

    def proceed(self, reset=True, branch=None, tick=None):
        """Perform all proceed effects."""
        if self.proceed_effects is not None:
            self.proceed_effects.do(
                self, reset, branch, tick)

    def conclude(self, reset=True, branch=None, tick=None):
        """Perform all conclude effects."""
        if self.conclude_effects is not None:
            self.conclude_effects.do(
                self, reset, branch, tick)
