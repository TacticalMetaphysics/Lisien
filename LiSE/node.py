# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from collections import defaultdict
import gorm.graph
from .util import (
    dispatch,
    listen,
    listener,
    fire_time_travel_triggers,
    encache
)
from .rule import RuleBook, RuleMapping


class Node(gorm.graph.Node):
    """Superclass for both Thing and Place"""
    @property
    def rulebook(self):
        if not hasattr(self, '_rulebook'):
            self._rulebook = RuleBook(
                self.engine,
                self._rulebook_name
            )
        return self._rulebook

    @rulebook.setter
    def rulebook(self, v):
        if not isinstance(v, str) or isinstance(v, RuleBook):
            raise TypeError("Use a :class:`RuleBook` or the name of one")
        self._rulebook_name = v.name if isinstance(v, RuleBook) else v
        dispatch(self._rulebook_listeners, self._rulebook_name, self, v)

    @property
    def _rulebook_name(self):
        try:
            return self.engine.db.node_rulebook(
                self.character.name,
                self.name
            )
        except KeyError:
            self.engine.db.set_node_rulebook(
                self.character.name,
                self.name,
                (self.character.name, self.name)
            )
            return (self.character.name, self.name)

    @_rulebook_name.setter
    def _rulebook_name(self, v):
        self.engine.db.set_node_rulebook(
            self.character.name,
            self.name,
            v
        )

    @property
    def rule(self):
        return RuleMapping(self.engine, self.rulebook)

    def __init__(self, character, name):
        """Store character and name, and maybe initialize a couple caches"""
        self.character = character
        self.engine = character.engine
        self.name = name
        self._rulebook_listeners = []
        self._stat_listeners = defaultdict(list)
        if self.engine.caching:
            self._keycache = {}
            self._cache = {}

        @self.engine.on_time
        def time_travel_triggers(
                engine,
                branch_then,
                tick_then,
                branch_now,
                tick_now
        ):
            fire_time_travel_triggers(
                engine,
                self,
                self._cache,
                self._dispatch_stat,
                branch_then,
                tick_then,
                branch_now,
                tick_now
            )
        super().__init__(character, name)

    def listener(self, f=None, stat=None):
        return listener(self._stat_listeners, f, stat)

    def _dispatch_stat(self, k, v):
        (branch, tick) = self.engine.time
        dispatch(self._stat_listeners, k, branch, tick, self, k, v)

    def rulebook_listener(self, f):
        listen(self._rulebook_listeners, f)

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        if self.engine.caching:
            (branch, tick) = self.engine.time
            encache(self._cache, k, v, branch, tick)
            if branch in self._keycache and tick in self._keycache[branch]:
                self._keycache[branch][tick].add(k)
        self._dispatch_stat(k, v)

    def __delitem__(self, k):
        super().__delitem__(k)
        if self.engine.caching:
            (branch, tick) = self.engine.time
            encache(self._cache, k, None, branch, tick)
            if branch in self._keycache and tick in self._keycache[branch]:
                self._keycache[branch][tick].remove(k)
        self._dispatch_stat(k, None)

    def _portal_dests(self):
        """Iterate over names of nodes you can get to from here"""
        return self.engine.db.nodeBs(
            self.character.name,
            self.name,
            *self.engine.time
        )

    def _portal_origs(self):
        """Iterate over names of nodes you can get here from"""
        return self.engine.db.nodeAs(
            self.character.name,
            self.name,
            *self.engine.time
        )

    def _user_names(self):
        """Iterate over names of characters that have me as an avatar"""
        return self.engine.db.avatar_users(
            self.character.name,
            self.name,
            *self.engine.time
        )

    def users(self):
        """Iterate over characters this is an avatar of."""
        for charn in self._user_names():
            yield self.engine.character[charn]

    def portals(self):
        """Iterate over :class:`Portal` objects that lead away from me"""
        for destn in self._portal_dests():
            yield self.character.portal[self.name][destn]

    def preportals(self):
        """Iterate over :class:`Portal` objects that lead to me"""
        for orign in self._portal_origs():
            yield self.character.preportal[self.name][orign]

    def contents(self):
        """Iterate over :class:`Thing` objects located in me"""
        for thing in self.character.thing.values():
            if thing['location'] == self.name:
                yield thing

    def delete(self):
        """Get rid of this, starting now.

        Apart from deleting the node, this also informs all its users
        that it doesn't exist and therefore can't be their avatar
        anymore.

        """
        del self.character.place[self.name]
        if self.name in self.character.portal:
            del self.character.portal[self.name]
        if self.name in self.character.preportal:
            del self.character.preportal[self.name]
        for user in self.users():
            user.del_avatar(self.character.name, self.name)

    def __bool__(self):
        return self.name in self.character.node
