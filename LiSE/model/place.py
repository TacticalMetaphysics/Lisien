# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from container import Container


class Place(Container):
    """Places where things may be.

    Places are vertices in a character's graph where things can be,
    and where portals can lead. A place's name must be unique within
    its character.

    Unlike things and portals, places can't be hosted by characters
    other than the one they are part of. When something is "hosted" in
    a character, that always means it's in a place that's in the
    character.

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
                "character", "name", "key", "branch", "tick")}),
        ("place_stat_facade", {
            "columns": {
                "observer": "text not null",
                "observed": "text not null",
                "name": "text not null",
                "key": "text not null",
                "branch": "integer not null default 0",
                "tick": "integer not null default 0",
                "value": "text"},
            "primary_key": (
                "observer", "observed", "name", "key", "branch", "tick")})]

    @property
    def v(self):
        """My iGraph vertex object"""
        return self.character.graph.vs.find(name=self.name)

    def __init__(self, character, name):
        """Initialize a place in a character by a name"""
        self.character = character
        self.name = name
        self.character.graph.add_vertex(name=self.name, place=self)

    def __contains__(self, that):
        """Is that here?"""
        return self.contains(that)

    def __str__(self):
        return str(self.name)

    def __unicode__(self):
        return unicode(self.name)

    def iter_portals(self, mode="both", observer=None, branch=None, tick=None):
        """Iterate over portals incident on this place.

        By default this includes portals leading from and to
        here. Change this by setting mode to 'in' or 'out'.

        """
        if observer:
            charfac = self.character.get_facade(observer)
        else:
            charfac = self.character
        for portal in charfac.iter_portals_incident_on_place(
                self.name, mode, observer, branch, tick):
            yield portal

    def get_stat(self, stat, observer=None, branch=None, tick=None):
        return self.get_subjectively(
            'get_place_stat', observer, [stat, branch, tick])

    def iter_stat_keys(self, observer=None, branch=None, tick=None):
        for key in self.get_subjectively(
                'iter_place_stat_keys', observer, [branch, tick]):
            yield key
