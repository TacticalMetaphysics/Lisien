# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import SaveableMetaclass, Fabulator, tabclas


class Cause(object):
    """Listen to the world, and fire an event on a specified condition.

In fact it is Implicator that fires the event. I only test whether a
particular precondition for that event holds true.

A Cause must have a method called test, which returns True when the
Cause is triggered. It may also have a method called validate, which
returns True when the Cause *may* be triggered--but the validate
method does not receive a Character to check.

    """
    def __init__(self):
        if not callable(self):
            raise TypeError(
                "Cause is not callable. Does it have a test() method?")

    def __call__(self, character, branch, tick, validate=True):
        if validate and not self.validate(branch, tick):
            return False
        r = self.test(character, branch, tick)
        if not isinstance(r, bool):
            raise TypeError(
                "test returned non-Boolean")
        return r

    def validate(self, branch, tick):
        return True


class CompoundCause(Cause):
    """Combine many causes into one. Trigger only when they all pass."""
    def __init__(self, causes):
        self.causes = causes

    def test(self, character, branch, tick):
        for cause in self.causes:
            if not cause(character, branch, tick):
                return False
        return True


class Event(object):
    """Information about something happening in the world, when the
effects are yet unknown.

This kind of event only happens in the simulated world. It has no
concept of, eg., user input.

    """
    def __init__(self, cause, branch, tick):
        if type(self) is Event:
            raise TypeError(
                "Event should not be instantiated. Subclass it.")
        self.cause = cause
        self.branch = branch
        self.tick = tick


class Effect(object):
    """Respond to an Event fired by Implicator in response to a Cause.

An Effect does its thing in its do() method, which will receive a
Character, along with the current branch and tick. It must be
stateless, with no side effects. Instead, it describes a change to the
world using a tuple. Each Effect may only have one change.

    """

    def __call__(self, event):
        r = self.do(event.character, event.branch, event.tick)
        if not isinstance(r, tuple):
            raise TypeError(
                "do returned non-tuple")
        return r


class ChangeException(Exception):
    """Generic exception for something gone wrong with a Change."""
    pass


class Change(object):
    """A change to a single value in the world state."""
    def __init__(self, val, *keys):
        """The given value will be put into the place in the Skeleton
specified in the remaining arguments. The keys are in the order
they'll be looked up in the skeleton--table name first, field name
last.

        """
        global tabclas
        self.clas = tabclas[keys[0]]
        for fn in keys:
            if fn not in self.clas.colnames[keys[0]]:
                raise ChangeException(
                    "That table doesn't have that field.")
        self.keys = keys
        self.value = val


class Implicator(object):
    """An event handler for only those events that happen in the
game-world.

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
    make_cause = Fabulator(
        {"CompoundCause": ("CompoundCause\((.+?)\)", CompoundCause)})

    def __init__(self, closet):
        self.closet = closet

    def iter_cause_event(self):
        """Generate the causes and effects, like dict.iteritems()."""
        for (causen, effectn) in self.closet.skeleton["cause_event"]:
            yield (
                self.closet.get_cause(causen),
                self.closet.get_effect(effectn))

    def poll(self, branch, tick):
        """Collect and return all the changes to be made to the world at the
given game-time."""
        r = []
        for (cause, event) in self.iter_cause_event():
            if cause.validate(branch, tick):
                for character in self.closet.characters:
                    if cause(character, branch, tick, validate=False):
                        r.append(self.event_effect[event](event))
        return r
