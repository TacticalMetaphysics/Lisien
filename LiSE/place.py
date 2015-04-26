# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from gorm.xjson import json_dump
from .node import Node
from .util import (
    dispatch,
    cache_forward,
    needcache,
    encache,
    enkeycache,
    dekeycache
)


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
            if not self.engine.caching:
                return super().__getitem__(key)
            (branch, tick) = self.engine.time
            cache_forward(self._cache, key, branch, tick)
            if needcache(self._cache, key, branch, tick):
                value = super().__getitem__(key)
                encache(
                    self, self._cache, key, value
                )
            r = self._cache[key][branch][tick]
            if r is None:
                raise KeyError("Key {} not set now".format(key))
            return r

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if self.engine.caching:
            encache(self, self._cache, key, value)
            enkeycache(self, self._keycache, key)
            (branch, tick) = self.engine.time
            dispatch(self._stat_listeners, key, branch, tick, self, key, value)

    def __delitem__(self, key):
        super().__delitem__(key)
        if not self.engine.caching:
            return
        dekeycache(self, self._cache, key)
        encache(self, self._cache, key, None)

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
        return json_dump(self._get_json_dict())

    def delete(self):
        del self.character.place[self.name]
