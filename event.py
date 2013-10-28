# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass


class Implicator(object):
    """A relation between Causes and Effects by way of Event classes.

Implicator is used to decide what Effects ought to be applied to the
world, given that a particular combination of Causes is active.

    """
    __metaclass__ = SaveableMetaclass
    tables = [
        ("cause_event",
         {"cause": "text not null",
          "event": "text not null"},
         ("cause", "event"),
         {},
         []),
        ("event_effect",
         {"event": "text not null",
          "effect": "text not null"},
         ("event", "effect"),
         {},
         [])]

    def __init__(self, closet):
        self.closet = closet
        self.cause_event = {}
        self.event_effect = {}


class AbstractEvent(object):
    """Information about something happening in the world.

This kind of event only happens in the simulated world. It has no
concept of, eg., user input.

    """
    def __init__(self, cause, branch, tick):
        if type(self) is AbstractEvent:
            raise TypeError(
                "AbstractEvent should not be instantiated. Subclass it.")
        self.cause = cause
        self.branch = branch
        self.tick = tick
        self.changes = []

    def get_change(self, effect):
        """Ask an Effect what change it has on the world, given myself. Store
the response in self.changes for Implicator to use."""
        self.changes.append(effect(self.cause, self.branch, self.tick))


class Cause(object):
    """Listen to the world, and fire an event on a specified condition."""
    def __init__(self, tester=None):
        if tester is not None:
            self.test = tester
        if not callable(self):
            raise TypeError(
                "Cause is not callable. Does it have a test method?")

    def __call__(self, character, branch, tick):
        r = self.test(character, branch, tick)
        if not isinstance(r, bool):
            raise TypeError(
                "test returned non-Boolean")
        return r


class CompoundCause(Cause):
    """Several Causes that trigger a particular Effect only if they are
all active at once.

Construct me with an iterable of all the Causes you want
considered. I'll only trigger when they are all true for the same
character on the same branch and tick.

    """
    def __init__(self, causes):
        self.causes = causes

    def test(self, character, branch, tick):
        for cause in self.causes:
            if not cause(character, branch, tick):
                return False
        return True


class Effect(object):
    """Respond to an event fired by a Cause."""
    def __call__(self, event):
        r = self.do(event)
        if not isinstance(r, tuple):
            raise TypeError(
                "do returned non-tuple")
        return r
