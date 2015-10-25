# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from .node import Node
from .util import dispatch


class Place(Node):
    """The kind of node where a Thing might ultimately be located."""
    extrakeys = {
        'name',
        'character'
    }

    def __getitem__(self, key):
        """Return my name if ``key=='name'``, my character's name if
        ``key=='character'``, the names of everything located in me if
        ``key=='contents'``, or the value of the stat named ``key``
        otherwise.

        """
        if key == 'name':
            return self.name
        elif key == 'character':
            return self.character.name
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.engine.caching:
            (branch, tick) = self.engine.time
            dispatch(self._stat_listeners, key, branch, tick, self, key, value)

    def __repr__(self):
        """Return my character and name"""
        return "{}.place[{}]".format(
            self['character'],
            self['name']
        )

    def _get_json_dict(self):
        (branch, tick) = self.engine.time
        return {
            "type": "Place",
            "version": 0,
            "character": self["character"],
            "name": self["name"],
            "branch": branch,
            "tick": tick,
            "stat": dict(self)
        }

    def dump(self):
        """Return a JSON representation of my present state"""
        return self.engine.json_dump(self._get_json_dict())

    def delete(self, nochar=False):
        super().delete()
        if not nochar:
            del self.character.place[self.name]
