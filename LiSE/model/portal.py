# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE.orm import SaveableMetaclass
from container import Contents
from stats import Stats


class Portal(object):
    __metaclass__ = SaveableMetaclass
    tables = [
        (
            "portal_stat",
            {
                "columns":
                [
                    {
                        'name': 'character',
                        'type': 'text'
                    }, {
                        'name': 'origin',
                        'type': 'text'
                    }, {
                        'name': 'destination',
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
                (
                    "character",
                    "origin",
                    "destination",
                    "key",
                    "branch",
                    "tick"
                ),
                "checks":
                ["type in ('text', 'real', 'boolean', 'integer')"]
            }
        )
    ]

    def __init__(self, character, origin, destination):
        origin = unicode(origin)
        destination = unicode(destination)

        def make_stat_bone(branch, tick, key, value):
            return self.bonetypes["portal_stat"](
                character=character.name,
                origin=origin,
                destination=destination,
                branch=branch,
                tick=tick,
                key=key,
                value=value,
                type={
                    str: 'text',
                    unicode: 'text',
                    int: 'integer',
                    float: 'real',
                    bool: 'boolean'
                }[type(value)]
            )

        self.stats = Stats(
            character.closet,
            ["portal_stat", character.name, origin, destination],
            make_stat_bone
        )
        self.contents = Contents(
            character,
            '{}->{}'.format(origin, destination)
        )
        self.character = character
        self.origin = self.character.place_d[origin]
        self.destination = self.character.place_d[destination]

    def __eq__(self, other):
        return (
            hasattr(other, 'character') and
            hasattr(other, 'origin') and
            hasattr(other, 'destination') and
            self.character == other.character and
            self.origin == other.origin and
            self.destination == other.destination
        )

    def __hash__(self):
        return hash((self.character, self.origin, self.destination))

    def __str__(self):
        return "{}->{}".format(self.origin, self.destination)

    def __unicode__(self):
        return u"{}->{}".format(self.origin, self.destination)

    def __repr__(self):
        return "Portal({0}->{1})".format(
            self.origin,
            self.destination
        )

    def __getitem__(self, key):
        if key == 'origin':
            return unicode(self.origin)
        elif key == 'destination':
            return unicode(self.destination)
        elif key == 'character':
            return unicode(self.character)
        else:
            return self.stats[key]

    def __setitem__(self, key, value):
        self.stats[key] = value
