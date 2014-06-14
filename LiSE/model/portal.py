# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE.orm import SaveableMetaclass
from container import Contents
from collections import MutableMapping


class Portal(MutableMapping):
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
            return self.character.get_portal_stat(self.name, key)

    def __setitem__(self, key, value):
        self.character.set_portal_stat(self['origin'], self['destination'], key, value)

    def __delitem__(self, key):
        self.character.del_portal_stat(self['origin'], self['destination'], key)

    def __iter__(self):
        for stat in self.character.iter_portal_stats(self['origin'], self['destination']):
            yield stat

    def __len__(self):
        return self.character.len_portal_stats(self['origin'], self['destination'])
