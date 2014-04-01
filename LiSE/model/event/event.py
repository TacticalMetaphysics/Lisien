# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE.orm import SaveableMetaclass


class AbstractEvent(object):
    """Something that can happen."""
    def __init__(self, cause, character, branch, tick):
        if self.__class__ is AbstractEvent:
            raise NotImplementedError(
                "Subclass AbstractEvent, don't instantiate it.")


class ImplicatorException(Exception):
    pass


class Implicator(object):
    """An event handler for those events that take place in the simulated
    world.

    Whenever new ticks are added to a timeline, the Implicator will
    take a look at the world-state during each of them in
    turn. Boolean functions called 'causes' will be called to evaluate
    whether their associated Event should happen. When an Event
    happens, it is instantiated, and the instance is used to create
    bones describing the changes to the world state. The bones are set
    into the skeleton, and thereby change the world.

    """
    __metaclass__ = SaveableMetaclass
    demands = ["character"]
    tables = (
        "ticks_implicated", {
            "columns": {
                "branch": "integer not null",
                "tick": "integer not null"},
            "primary_key": ("branch", "tick"),
            "checks": ["branch>=0", "tick>=0"]})

    def __init__(self, closet, cause_event_d):
        """Set local variables, most of which are taken from ``closet``.

        """
        self.set_bone = closet.set_bone
        self.get_character = closet.get_character
        self.character_d = closet.character_d
        self.cause_event_d = cause_event_d
        # load all ticks that have been implicated already, to ensure
        # that they are not implicated twice
        closet.select_class_all(Implicator)
        self.implicated = closet.skeleton["ticks_implicated"]

    def tick_implicated(self, branch, tick):
        """Check if I've handled the tick in the branch"""
        return (branch in self.implicated and
                tick in self.implicated[branch])

    def handle_tick(self, branch, tick):
        """A new tick has been generated in this here branch.
        Deal with it."""
        if self.tick_implicated(branch, tick):
            raise ImplicatorException(
                "This tick has already been implicated.")
        for (cause, event) in self.cause_event_d.iteritems():
            for character in self.character_d.itervalues():
                if cause(character):
                    happening = event(cause, character, branch, tick)
                    for bone in happening.iterbones():
                        self.set_bone(bone)
        self.set_bone(self.bonetype(branch=branch, tick=tick))
