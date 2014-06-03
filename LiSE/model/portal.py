# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from place import Place, Container
from LiSE.util import upbranch


class Portal(Container):
    tables = [
        (
            "portal_stat",
            {
                "columns":
                {
                    "character": "text not null",
                    "origin": "text not null",
                    "destination": "text not null",
                    "key": "text not null default 'exists'",
                    "branch": "integer not null default 0",
                    "tick": "integer not null default 0",
                    "value": "text",
                    "type": "text not null default 'text'"},
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
        self.character = character
        self.origin = origin
        self.destination = destination

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

    @property
    def origplace(self):
        return self.character.get_place(self.origin)

    @property
    def destplace(self):
        return self.character.get_place(self.destination)

    @property
    def contents(self):
        return self.character.graph[self.origin][self.destination]['contents']

    def _get_end(self, bprop):
        bone = self.loc_bone
        endn = getattr(bone, bprop)
        if endn not in self.character.place_d:
            self.character.place_d[endn] = Place(self.character, endn)
        return self.character.place_d[endn]

    @property
    def origin(self):
        return self._get_end('origin')

    @property
    def destination(self):
        return self._get_end('destination')

    def new_branch(self, parent, branch, tick):
        skel = (
            self.character.closet.skeleton
            [u"portal_stat"]
            [unicode(self.character)]
            [unicode(self.origin)]
            [unicode(self.destination)]
            [parent]
        )
        for bone in upbranch(
                self.character.closet,
                skel.iterbones(),
                branch,
                tick
        ):
            yield bone
