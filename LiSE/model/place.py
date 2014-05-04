# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from container import Container


class Place(Container):
    """Places where things may be.

    Places are vertices in a character's graph where things can be,
    and where portals can lead. A place's name must be unique within
    its character.

    You don't need to create a bone for each and every place you
    use--link to it with a portal, or put a thing there, and it will
    exist. Place bones are only for when a place needs stats.

    """
    tables = [
        ("place_stat", {
            "columns": {
                "character": "text not null",
                "name": "text not null",
                "key": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "value": "text"},
            "primary_key": (
                "character", "name", "key", "branch", "tick")})]

    def __init__(self, character, name):
        """Initialize a place in a character by a name"""
        self.character = character
        self.name = name

    def __contains__(self, that):
        """Is that here?"""
        return that in self.character.graph[self.name].contents

    def __str__(self):
        return str(self.name)

    def __unicode__(self):
        return unicode(self.name)

    def __eq__(self, other):
        return (
            self.character == other.character and
            self.name == other.name)

    def __hash__(self):
        return hash((self.character, self.name))

    def __getitem__(self, key):
        return self.character.graph[self.name][key]
