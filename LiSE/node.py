# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from collections import defaultdict
import gorm.graph
from .util import (
    dispatch,
    listener,
    fire_time_travel_triggers,
    encache,
    enkeycache,
    dekeycache
)
from .rule import RuleBook, RuleFollower
from .rule import RuleMapping as BaseRuleMapping


class RuleMapping(BaseRuleMapping):
    def __init__(self, node):
        super().__init__(node.engine, node.rulebook)
        self.character = node.character
        self.engine = self.character.engine

    def __iter__(self):
        return self.engine.db.node_rules(
            self.character.name,
            self.name,
            *self.engine.time
        )


class Node(gorm.graph.Node, RuleFollower):
    """Superclass for both Thing and Place"""
    def _rule_names_activeness(self):
        return self.engine.db.current_rules_node(
            self.character.name,
            self.name,
            *self.engine.time
        )

    def _get_rule_mapping(self):
        return RuleMapping(self)

    def _get_rulebook_name(self):
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

    def _get_rulebook(self):
        return RuleBook(
            self.engine,
            self._get_rulebook_name()
        )

    def _set_rulebook_name(self, v):
        self.engine.db.set_node_rulebook(
            self.character.name,
            self.name,
            v
        )

    def _dispatch_stat(self, k, v):
        if k in self and self[k] == v:
            return
        (branch, tick) = self.engine.time
        dispatch(self._stat_listeners, k, branch, tick, self, k, v)

    def __init__(self, character, name):
        """Store character and name, and initialize caches"""
        self.character = character
        self.engine = character.engine
        self.name = name
        self._rulebook_listeners = []
        self._stat_listeners = defaultdict(list)
        self._keycache = {}
        self._cache = {}

        @self.engine.time_listener
        def time_travel_triggers(
                branch_then,
                tick_then,
                branch_now,
                tick_now
        ):
            fire_time_travel_triggers(
                self,
                self._cache,
                self._dispatch_stat,
                branch_then,
                tick_then,
                branch_now,
                tick_now
            )

        if self.engine.caching:
            def cache_branch(branch):
                for (key, tick, value) in self.engine.db.node_stat_branch_data(
                        self.character.name, self.name, branch
                ):
                    if key not in self._cache:
                        self._cache[key] = {}
                    if branch not in self._cache:
                        self._cache[key][branch] = {}
                    self._cache[key][branch][tick] = value

            branch = self.engine.branch
            cache_branch(branch)
            self._branches_cached = {branch, }

            @self.engine.time_listener
            def cache_new_branch(
                    branch_then,
                    tick_then,
                    branch_now,
                    tick_now
            ):
                if branch_now not in self._branches_cached:
                    cache_branch(branch_now)
                    self._branches_cached.add(branch_now)

        super().__init__(character, name)

    def listener(self, f=None, stat=None):
        return listener(self._stat_listeners, f, stat)

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        if self.engine.caching:
            encache(self, self._cache, k, v)
            enkeycache(self, self._keycache, k)
        self._dispatch_stat(k, v)

    def __delitem__(self, k):
        super().__delitem__(k)
        if self.engine.caching:
            (branch, tick) = self.engine.time
            encache(self, self._cache, k, None)
            dekeycache(self, self._keycache, k)
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
