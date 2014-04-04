# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE.orm import SaveableMetaclass


class AbstractEvent(object):
    """Something that can happen."""
    def __init__(self, cause, branch, tick, **kwargs):
        self.cause = cause
        self.branch = branch
        self.tick = tick
        # kwargs should really be handled by subclasses, but I'll
        # store them in an instance var anyway to make debugging
        # easier
        self.kwargs = kwargs

    def iter_bones_to_set(self):
        """Implement an iterator over bones here. They will be set, and thus
        change the world.

        """
        raise NotImplementedError(
            "Abstract class")


class DiegeticEventHandler(object):
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
        "ticks_evented", {
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
        # load all ticks that have been handed already, to ensure
        # that they are not handled twice
        closet.select_class_all(DiegeticEventHandler)
        self.handled = closet.skeleton["ticks_implicated"]

    def iter_events(self, branch, tick):
        """Iterate over all events for all characters and all of their
        facades.

        """
        for character in self.character_d.itervalues():
            for introspection in character.iter_triggers():
                kwargs = introspection(character, branch, tick)
                if kwargs:
                    event_cls = self.cause_event_d[introspection]
                    kwargs['character'] = character
                    yield event_cls(
                        introspection,
                        branch,
                        tick,
                        **kwargs)
            for facade in character.facade_d.itervalues():
                for inquiry in facade.iter_triggers():
                    kwargs = inquiry(facade, branch, tick)
                    if kwargs:
                        event_cls = self.cause_event_d[inquiry]
                        kwargs['facade'] = facade
                        yield event_cls(
                            inquiry,
                            branch,
                            tick,
                            **kwargs)

    def tick_handled(self, branch, tick):
        """Check if I've handled the tick in the branch"""
        return (branch in self.handled and
                tick in self.handled[branch])

    def handle_events(self, branch, tick):
        """Handle events in the given branch and tick.

        Raise ValueError if the branch/tick has been handled
        already. In that case, the correct thing is to make a new
        branch and handle the tick there.

        """
        if self.tick_handled(branch, tick):
            raise ValueError(
                "I already handled tick {} of branch {}".format(
                    tick, branch))
        if branch not in self.handled:
            self.handled[branch] = {}
        r = list(self.iter_events(branch, tick))
        self.handled[branch][tick] = r
        return r
