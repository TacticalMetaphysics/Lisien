# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE.orm import SaveableMetaclass


class AbstractEvent(object):
    """Something that can happen."""
    def __init__(self, character, cause, branch, tick, **kwargs):
        # character may actually be of Facade class, but they have the
        # same API so I'm using the name of the parent class by
        # default
        self.character = character
        self.cause = cause
        self.branch = branch
        self.tick = tick

    def iter_bones_to_set(self):
        """Implement an iterator over bones here. They will be set, and thus
        change the world.

        """
        raise NotImplementedError(
            "Abstract class")


class ThingEvent(AbstractEvent):
    """Produces some change or other in a Thing.

    Accepts optional kwargs:

    name: the name of the Thing
    host: the character to locate the Thing in
    
    Other (key, value) pairs in kwargs get interpreted as assignments
    to the thing's stats.

    In the case that it moves a Thing, the Thing in question will
    teleport from one location to the next; no pathfinding happens
    here. To destroy a Thing, set its location to None.

    It is possible to change a thing's host with this event. This
    probably isn't what you want; the thing's host defines whether,
    eg., it's a physical item or a concept, so you should generally
    only set that when you're creating a thing the first
    time. ThingEvent does not enforce this, however.

    """
    def __init__(self, character, cause, branch, tick, **kwargs):
        super(ThingEvent, self).__init__(
            character, cause, branch, tick, **kwargs)
        if 'name' in kwargs:
            self.name = kwargs['name']
            del kwargs['name']
        else:
            closet = self.character.closet
            numeral = closet.get_global('top_generic_thing') + 1
            closet.set_global('top_generic_thing', numeral)
            self.name = 'generic_thing_{}'.format(numeral)
        if 'location' in kwargs:
            self.location = kwargs['location']
            del kwargs['location']
        if 'host' in kwargs:
            self.host = kwargs['host']
            del kwargs['host']
        self.stats = kwargs

    def iter_bones_to_set(self):
        from LiSE.model import Thing
        closet = self.character.closet
        skel = closet.skeleton['thing']
        charn = unicode(self.character)
        name = unicode(self.name)
        if (
                charn not in skel or
                self.name not in skel[charn] or
                hasattr(self, 'host')):
            yield Thing.bonetypes['thing'](
                character=charn,
                name=name,
                host=(
                    unicode(self.host)
                    if hasattr(self, 'host')
                    else u'Physical'
                )
            )
        if hasattr(self, 'location'):
            yield Thing.bonetypes['thing_loc'](
                character=charn,
                name=name,
                branch=int(self.branch),
                tick=int(self.tick),
                location=unicode(self.location)
            )
        for (key, value) in self.stats.iteritems():
            yield Thing.bonetypes['thing_stat'](
                character=charn,
                name=name,
                key=unicode(key),
                branch=int(self.branch),
                tick=int(self.tick),
                value=unicode(value)
            )


class DiegeticEventHandler(object):
    """An event handler for those events that take place in the simulated
    world.

    Whenever new ticks are added to a timeline, the Implicator will
    take a look at the world-state during each of them in
    turn. Functions called 'causes' will be called to evaluate
    whether their associated Event should happen. When an Event
    happens, it is instantiated, and the instance is used to create
    bones describing the changes to the world state. The bones are set
    into the skeleton, and thereby change the world.

    """
    __metaclass__ = SaveableMetaclass
    demands = ["character"]
    tables = [
        (
            "ticks_evented",
            {
                "columns":
                [
                    {
                        'name': 'branch',
                        'type': 'integer'
                    }, {
                        'name': 'tick',
                        'type': 'integer'
                    }
                ],
                "primary_key": ("branch", "tick"),
                "checks": ["branch>=0", "tick>=0"]
            }
        )
    ]

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
        self.handled = closet.skeleton["ticks_evented"]

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
                        character,
                        introspection,
                        branch,
                        tick,
                        **kwargs
                    )
            for facade in character.facade_d.itervalues():
                for inquiry in facade.iter_triggers():
                    kwargs = inquiry(facade, branch, tick)
                    if kwargs:
                        event_cls = self.cause_event_d[inquiry]
                        kwargs['facade'] = facade
                        yield event_cls(
                            facade,
                            inquiry,
                            branch,
                            tick,
                            **kwargs
                        )

    def tick_handled(self, branch, tick):
        """Check if I've handled the tick in the branch"""
        return (
            branch in self.handled and
            tick in self.handled[branch]
        )

    def handle_events(self, branch, tick):
        """Handle events in the given branch and tick.

        Raise ValueError if the branch/tick has been handled
        already. In that case, the correct thing is to make a new
        branch and handle the tick there.

        """
        if self.tick_handled(branch, tick):
            raise ValueError(
                "I already handled tick {} of branch {}".format(
                    tick,
                    branch
                )
            )
        if branch not in self.handled:
            self.handled[branch] = {}
        r = list(self.iter_events(branch, tick))
        self.handled[branch][tick] = r
        return r
