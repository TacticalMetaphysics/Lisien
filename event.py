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
from util import SaveableMetaclass, stringlike
from effect import (
    PortalEntryEffectDeck,
    PortalProgressEffectDeck,
    PortalExitEffectDeck)
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
    """A class for things that can happen. Normally represented as
cards.

Events are kept in EventDecks, which are in turn contained by
Characters. When something happens involving one or more characters,
the relevant EventDecks from the participating Characters will be put
together into an AttemptDeck. One card will be drawn from this, called
the attempt card. It will identify what kind of EventDeck should be
taken from the participants and compiled in the same manner into an
OutcomeDeck. From this, the outcome card will be drawn.

The effects of an event are determined by both the attempt card and
the outcome card. An attempt card might specify that only favorable
outcomes should be put into the OutcomeDeck; the attempt card might
therefore count itself for a success card. But further, success cards
may have their own effects irrespective of what particular successful
outcome occurs. This may be used, for instance, to model that kind of
success that strains a person terribly and causes them injury.

    """
    tables = [
        ("event",
         {"name": "text not null",
          "text": "text not null",
          "commence_effect_deck": "text",
          "proceed_effect_deck": "text",
          "conclude_effect_deck": "text"},
         ("name",),
         {"commence_effect_deck": ("effect_deck", "name"),
          "proceed_effect_deck": ("effect_deck", "name"),
          "conclude_effect_deck": ("effect_deck", "name")},
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

    def get_tabdict(self):
        occur_rows = set()
        occorder = (
            "event",
            "branch",
            "tick_from",
            "tick_to")
        for branch in self.occurrences:
            for (tick_from, tick_to) in self.occurrences[branch].iteritems():
                occur_rows.add((
                    str(self),
                    branch,
                    tick_from,
                    tick_to))
        return {
            "event": [{"name": str(self),
                       "text": self._text,
                       "commence_effect_deck": str(self.commence_effect_deck),
                       "proceed_effect_deck": str(self.proceed_effect_deck),
                       "conclude_effect_deck": str(self.conclude_effect_deck)}],
            "scheduled_event": [dictify_row(row, occorder)
                                for row in iter(occur_rows)]}

    def unravel(self):
        """Dereference the effect decks.

If the event text begins with @, it's a pointer; look up the real
value in the db.

        """
        for deck in (self.commence_effects, self.proceed_effects,
                     self.conclude_effects):
            if deck is not None:
                deck.unravel()

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

    def commence(self):
        """Perform all commence effects, and set self.ongoing to True."""
        if self.commence_effects is not None:
            self.commence_effects.do(self)

    def proceed(self):
        """Perform all proceed effects."""
        if self.proceed_effects is not None:
            self.proceed_effects.do(self)

    def conclude(self):
        """Perform all conclude effects, and set self.ongoing to False."""
        if self.conclude_effects is not None:
            self.conclude_effects.do(self)


class PortalTravelEvent(Event):
    """Event representing a thing's travel through a single portal, from
one place to another."""
    name_format = "PortalTravelEvent {0}: {1}: {2}-{3}->{4}"
    text_format = "Travel from {0} to {1}"

    def __init__(self, db, thing, portal, ongoing):
        dimname = thing.dimension.name
        origname = str(portal.orig)
        destname = str(portal.dest)
        name = self.name_format.format(
            dimname, thing.name, origname, str(portal), destname)
        text = self.text_format.format(origname, destname)
        commence_effects = PortalEntryEffectDeck(db, thing, portal)
        proceed_effects = PortalProgressEffectDeck(db, thing)
        conclude_effects = PortalExitEffectDeck(db, thing)
        Event.__init__(
            self, db, name, text, ongoing,
            commence_effects, proceed_effects, conclude_effects)


class EventDeck:
    """A deck representing events that might get scheduled at some point."""
    tables = [
        ("event_deck_link",
         {"deck": "text not null",
          "idx": "integer not null",
          "event": "text not null"},
         ("deck", "idx"),
         {"event": ("event", "name")},
         [])]

    def __init__(self, db, name, event_list):
        """Return an EventDeck with the given name, containing the given
events. Register with db.eventdeckdict.

        """
        self.name = name
        self.events = event_list
        db.eventdeckdict[self.name] = self
        self.db = db

    def get_tabdict(self):
        rowdicts = []
        for i in xrange(0, len(self.events)):
            rowdicts.append({
                "deck": self.name,
                "idx": i,
                "event": self.events[i].name})
        return {"event_deck_link": rowdicts}

    def unravel(self):
        for i in xrange(0, len(self.events)):
            if stringlike(self.events[i]):
                self.events[i] = self.db.eventdict[self.events[i]]
