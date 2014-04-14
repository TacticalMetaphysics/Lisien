# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE.orm import SaveableMetaclass
from LiSE.util import Fabulator
from cause import Cause
from effect import Effect


class Implicator(object):
    """An event handler for only those events that happen in the
game-world.

Implicator is used to decide what Effects ought to be applied to the
world, given that a particular combination of Causes is active.

    """
    __metaclass__ = SaveableMetaclass
    tables = [
        ("cause_event",
         {"columns":
          {"cause": "text not null",
           "event": "text not null"},
          "primary_key":
          ("cause", "event")}),
        ("event_effect",
         {"columns":
          {"event": "text not null",
           "effect": "text not null"},
          "primary_key":
          ("event", "effect")}),
        ("event",
         {"columns":
          {"name": "text not null",
           "priority": "integer not null default 0"},
          "primary_key":
          ("name",)})]

    def __init__(self, closet):
        """Remember the closet, and construct Fabulators for Cause and
Effect.

Fabulators are loaders for classes. Look them up in util.py."""
        import cause_cbs
        import effect_cbs

        def causes():
            """Generator for the callback functions that Cause is meant to
encapsulate."""
            for fun in dir(cause_cbs):
                yield getattr(cause_cbs, fun)

        def effects():
            """Generator for the callback functions that Effect is meant to
encapsulate."""
            for fun in dir(effect_cbs):
                yield getattr(effect_cbs, fun)

        self.closet = closet
        # make_cause should contain a plain Cause for every callback
        make_cause_fabdict = dict([
            (fun.__name__, self.cause_maker(fun))
            for fun in causes])
        # It should also contain those Causes that contain other
        # Causes.  The idea is you should be able to construct Causes
        # by writing out the constructor in the appropriate database
        # field, then letting the Fabulator do its thing.
        make_cause_fabdict.update(dict([
            (clsn, self.mkmaker(getattr(cause, clsn)))
            for clsn in dir(cause) if clsn != "Cause"]))
        # same deal for effects
        make_effect_fabdict = dict([
            (fun.__name__, self.effect_maker(fun))
            for fun in effects])
        make_effect_fabdict.update(dict([
            (clsn, self.mkmaker(getattr(effect, clsn)))
            for clsn in dir(effect) if clsn != "Effect"]))
        self.make_cause = Fabulator(make_cause_fabdict)
        self.make_effect = Fabulator(make_effect_fabdict)

    def mkmaker(self, constructor):
        def maker(*args):
            constructor(self, *args)
        return maker

    def cause_maker(self, fun):
        def maker(*args):
            return Cause(self, test=fun(*args))
        return maker

    def effect_maker(self, fun):
        def maker(*args):
            return Effect(self, doer=fun(*args))
        return maker

    def event_maker(self, name, priority=0):
        return type(name, (Event,), {'priority': priority})

    def poll_char(self, character, branch, tick):
        """Collect and return all the changes to be made to the character at
the given game-time.

        """
        evs = []
        for cause in self.cause_event:
            if cause.validate_time(branch, tick):
                evs.append(self.cause_event[cause](
                    self, cause, character, branch, tick))
        evs.sort()
        changes = []
        for event in evs:
            if event.cause.test(character, branch, tick):
                eff = self.event_effect[event]
                changes.append(eff(event))
        return changes
