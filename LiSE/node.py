# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from collections import defaultdict, Mapping

import gorm.graph

from .util import (
    dispatch,
    listener,
    unlisten,
    fire_time_travel_triggers,
    encache,
    enkeycache,
    dekeycache,
    reify
)
from . import rule


class RuleMapping(rule.RuleMapping):
    """Version of :class:`LiSE.rule.RuleMapping` that works more easily with a node. """
    def __init__(self, node):
        """Initialize with node's engine and rulebook. Store character and engine. """
        super().__init__(node.engine, node.rulebook)
        self.character = node.character
        self.engine = self.character.engine

    def __iter__(self):
        """Iterate over the names of rules for this node."""
        return self.engine.db.node_rules(
            self.character.name,
            self.name,
            *self.engine.time
        )


class UserMapping(Mapping):
    def __init__(self, node):
        self.node = node
        self.engine = node.engine

    def __iter__(self):
        yield from self.node._user_names()

    def __len__(self):
        return len(self.node._user_names())

    def __getitem__(self, k):
        if k not in self.node._user_names():
            raise KeyError("{} not used by {}".format(
                self.node.name, k
            ))
        return self.engine.character[k]


class Node(gorm.graph.Node, rule.RuleFollower):
    """Superclass for both Thing and Place"""
    def _rule_names_activeness(self):
        return self.engine.db.current_rules_node(
            self.character.name,
            self.name,
            *self.engine.time
        )

    def _cache_keys(self):
        (branch, tick) = self.engine.time
        if branch not in self._keycache:
            self._keycache[branch] = {}
        if tick not in self._keycache[branch]:
            self._keycache[branch][tick] = self.extrakeys.union(set(
                k for k in super().__iter__()
            ))

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
        return rule.RuleBook(
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

    def _get_user_names(self):
        yield from self.engine.db.avatar_users(
            self.character.name,
            self.name,
            *self.engine.time
        )

    @reify
    def _user_mapping(self):
        return UserMapping(self)

    @property
    def user(self):
        usernames = list(self._get_user_names())
        if len(usernames) == 1:
            return self.engine.character[usernames[0]]
        else:
            return self._user_mapping

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

    def __iter__(self):
        """Iterate over a cached set of keys if possible and caching's
        enabled.

        Iterate over some special keys too.

        """
        if not self.engine.caching:
            yield from super().__iter__()
            yield from self.extrakeys
            return
        self._cache_keys()
        (branch, tick) = self.engine.time
        yield from list(self._keycache[branch][tick])
        yield from self.extrakeys

    def __contains__(self, k):
        """Handle extra keys, then delegate."""
        if k in self.extrakeys:
            return True
        return super().__contains__(k)

    def listener(self, f=None, stat=None):
        """Arrange to call a function whenever a stat changes.

        If no stat is provided, changes to any stat will result in a call.

        """
        return listener(self._stat_listeners, f, stat)

    def unlisten(self, f=None, stat=None):
        """Stop calling a function when a stat changes.

        If the function wasn't passed to ``self.listener`` in the same way, this won't do anything.

        """
        return unlisten(self._stat_listeners, f, stat)

    def __setitem__(self, k, v):
        """Set a stat.

        Stats are time-sensitive. Values set to stats will appear to change to their predecessors, or disappear
        entirely, if the sim-time is set to a tick before the value was set.

        """
        super().__setitem__(k, v)
        if self.engine.caching:
            encache(self, self._cache, k, v)
            enkeycache(self, self._keycache, k)
        self._dispatch_stat(k, v)

    def __delitem__(self, k):
        """Delete a stat.

        Stats are time-sensitive, so the stat may appear to pop back into existence if you set the tick to one before
        when the stat was deleted.

        """
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
        if not self.engine.caching:
            return list(self.engine.db.avatar_users(
                self.character.name,
                self.name,
                *self.engine.time
            ))
        if not hasattr(self, '_user_cache'):
            self._user_cache = list(self.engine.db.avatar_users(
                self.character.name,
                self.name,
                *self.engine.time
            ))
        return self._user_cache

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
        if self.name in self.character.portal:
            del self.character.portal[self.name]
        if self.name in self.character.preportal:
            del self.character.preportal[self.name]
        for contained in list(self.contents()):
            contained.delete()
        for user in list(self.users()):
            user.del_avatar(self.character.name, self.name)
        (branch, tick) = self.engine.time
        self.engine.db.exist_node(
            self.character.name,
            self.name,
            branch,
            tick,
            False
        )

    def one_way_portal(self, other, **stats):
        """Connect a portal from here to another node, and return it."""
        return self.character.new_portal(
            self, other, symmetrical=False, **stats
        )

    def one_way(self, other, **stats):
        return self.one_way_portal(other, **stats)

    def two_way_portal(self, other, **stats):
        """Connect these nodes with a two-way portal and return it."""
        return self.character.new_portal(
            self, other, symmetrical=True, **stats
        )

    def two_way(self, other, **stats):
        return self.two_way_portal(other, **stats)

    def new_thing(self, name, **stats):
        """Create a new thing, located here, and return it."""
        return self.character.new_thing(
            name, self.name, None, **stats
        )

    @property
    def portal(self):
        """Return a mapping of portals to other nodes."""
        return self.character.portal[self.name]

    def __bool__(self):
        """Return whether I really exist in the world model, ie. in my character."""
        return self.name in self.character.node
