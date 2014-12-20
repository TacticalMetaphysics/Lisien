# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from gorm.xjson import json_dump
from .node import Node
from .util import (
    dispatch, needcache, encache, JSONReWrapper, JSONListReWrapper
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
            if needcache(self._cache, key, branch, tick):
                value = super().__getitem__(key)
                if isinstance(value, dict):
                    value = JSONReWrapper(self, key, value)
                elif isinstance(value, list):
                    value = JSONListReWrapper(self, key, value)
                encache(
                    self._cache, key, value, branch, tick
                )
            return self._cache[key][branch][tick]

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
        if isinstance(value, list):
            value = JSONListReWrapper(self, key, value)
        elif isinstance(value, dict):
            value = JSONReWrapper(self, key, value)
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
