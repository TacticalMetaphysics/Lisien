# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com


class Event(object):
    """Information about something happening in the world, when the
effects are yet unknown.

This kind of event only happens in the simulated world. It has no
concept of, eg., user input.

    """
    def __init__(self, imp, cause, character, branch, tick, priority=0):
        self.implicator = imp
        if type(self) is Event:
            raise TypeError(
                "Event should not be instantiated. Subclass it.")
        self.cause = cause
        self.character = character
        self.branch = branch
        self.tick = tick
        self.priority = priority
