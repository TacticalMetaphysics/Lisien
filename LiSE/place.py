# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from gorm.xjson import json_dump
from .node import Node
from .util import dispatch, encache


class Place(Node):
    """The kind of node where a Thing might ultimately be located."""

    def __iter__(self):
        yield 'name'
        yield 'character'
        yield from super().__iter__()

    def __contains__(self, key):
        if key in ('name', 'character'):
            return True
        return super().__contains__(key)

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
        if not self.engine.caching:
            return
        (branch, tick) = self.engine.time
        if (
                branch in self._keycache and
                tick in self._keycache[branch]
        ):
            self._keycache[branch][tick].add(key)
        encache(self._cache, key, value, branch, tick)
        dispatch(self._stat_listeners, key, branch, tick, self, key, value)

    def __delitem__(self, key):
        super().__delitem__(key)
        if not self.engine.caching:
            return
        (branch, tick) = self.engine.time
        if (
                branch in self._keycache and
                tick in self._keycache[branch]
        ):
            self._keycache[branch][tick].remove(key)
        encache(self._cache, key, None, branch, tick)

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
        return json_dump(self._get_json_dict())
