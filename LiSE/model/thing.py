# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE.orm import SaveableMetaclass
from container import Contents
from collections import MutableMapping


class Thing(MutableMapping):
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
                        'name': 'thing',
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
            return self.character.get_thing_stat(self.name, key)

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
            self.character.set_thing_stat(self.name, key, value)

    def __delitem__(self, key):
        self.character.del_thing_stat(self.name, key)

    def __iter__(self):
        for stat in self.character.iter_thing_stats(self.name):
            yield stat

    def __len__(self):
        return self.character.len_thing_stats(self.name)
