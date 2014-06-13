# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE.util import TimeParadox
from LiSE.orm import SaveableMetaclass
from container import Contents
from stats import Stats


class Thing(object):
    """The sort of item that has a particular location at any given time.

    If a Thing is in a Place, it is standing still. If it is in a
    Portal, it is moving through that Portal however fast it must in
    order to arrive at the other end when it is scheduled to. If it is
    in another Thing, then it is wherever that is, and moving the
    same.

    """
    __metaclass__ = SaveableMetaclass
    tables = [
        (
            "thing_stat",
            {
                "columns":
                [
                    {
                        'name': 'character',
                        'type': 'text'
                    }, {
                        'name': 'name',
                        'type': 'text'
                    }, {
                        'name': 'key',
                        'type': 'text',
                        'default': 'exists'
                    }, {
                        'name': 'branch',
                        'type': 'integer',
                        'default': 0
                    }, {
                        'name': 'tick',
                        'type': 'integer',
                        'default': 0
                    }, {
                        'name': 'value',
                        'type': 'text',
                        'nullable': True
                    }, {
                        'name': 'type',
                        'type': 'text',
                        'default': 'text'
                    }
                ],
                "primary_key":
                ("character", "name", "key", "branch", "tick"),
                "checks":
                ["type in ('text', 'real', 'boolean', 'integer')"]
            }
        )
    ]

    def __init__(self, character, name):
        def make_stat_bone(branch, tick, key, value):
            return Thing.bonetypes['thing_stat'](
                character=character.name,
                name=name,
                key=key,
                branch=branch,
                tick=tick,
                value=value,
                type={
                    str: 'text',
                    unicode: 'text',
                    bool: 'boolean',
                    int: 'integer',
                    float: 'real'
                }[type(value)]
            )

        self.stats = Stats(
            character.closet,
            ['thing_stat', character.name, name],
            make_stat_bone
        )
        self.contents = Contents(character, name)
        self.character = character
        self.name = name

    def __str__(self):
        return str(self.name)

    def __unicode__(self):
        return unicode(self.name)

    def __repr__(self):
        return "Thing({})".format(self.name)

    def __getitem__(self, key):
        if key == 'name':
            return self.name
        elif key == 'character':
            return unicode(self.character)
        else:
            return self.stats[key]

    def __setitem__(self, key, value):
        if key == 'contents':
            for thing in value:
                if not isinstance(thing, Thing):
                    try:
                        thing = self.character.thing_d[thing]
                    except KeyError:
                        raise KeyError(
                            "Character {} has no Thing named {}".format(
                                self.character.name,
                                thing
                            )
                        )
                if thing['location'] != self.name:
                    thing['location'] = self.name
            self.character.thing_contents_d[self.name] = value
        else:
            self.stats[key] = value
