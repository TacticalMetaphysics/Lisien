# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""A base class for nodes that can be in a character.

Every actual node that you're meant to use will be a place or
thing. This module is for what they have in common.

"""
from collections import Mapping

import gorm.graph
from gorm.reify import reify

from .util import getatt
from .bind import TimeDispatcher
from . import rule


class RuleMapping(rule.RuleMapping):
    """Version of :class:`LiSE.rule.RuleMapping` that works more easily
    with a node.

    """
    def __init__(self, node):
        """Initialize with node's engine, character, and rulebook."""
        super().__init__(node.engine, node.rulebook)
        self.node = node

    character = getatt('node.character')

    def __iter__(self):
        if self.engine.caching:
            for (rule, active) in self.node._rule_names_activeness():
                if active:
                    yield rule
            return
        return self.engine.db.node_rules(
            self.character.name,
            self.name,
            *self.engine.time
        )


class UserMapping(Mapping):
    """A mapping of the characters that have a particular node as an avatar.

    Getting characters from here isn't any better than getting them from
    the engine direct, but with this you can do things like use the
    .get() method to get a character if it's a user and otherwise
    get something else; or test whether the character's name is in
    the keys; and so on.

    """
    def __init__(self, node):
        """Store the node"""
        self.node = node

    engine = getatt('node.engine')

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


class Node(gorm.graph.Node, rule.RuleFollower, TimeDispatcher):
    """Superclass for both Thing and Place"""
    def _rule_names_activeness(self):
        if self.engine.caching:
            cache = self.engine._active_rules_cache[self._get_rulebook_name()]
            for rule in cache:
                for (branch, tick) in self.engine._active_branches():
                    if branch not in cache[rule]:
                        continue
                    try:
                        yield (
                            rule,
                            cache[rule][branch][tick]
                        )
                        break
                    except ValueError:
                        continue
            return
        return self.engine.db.current_rules_node(
            self.character.name,
            self.name,
            *self.engine.time
        )

    def _get_rule_mapping(self):
        return RuleMapping(self)

    def _get_rulebook_name(self):
        if self.engine.caching:
            cache = self.engine._nodes_rulebooks_cache
            if (
                    self.character.name not in cache or
                    self.name not in cache[self.character.name]
            ):
                return (self.character.name, self.name)
            return cache[self.character.name][self.name]
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

    def _user_names(self):
        if self.engine.caching:
            cache = self.engine._avatarness_cache.user_order[self.character.name][self.name]
            for user in cache:
                for (branch, tick) in self.engine._active_branches():
                    try:
                        if cache[user][branch][tick]:
                            yield user
                        break
                    except KeyError:
                        continue
            return
        yield from self.engine.db.avatar_users(
            self.character.name,
            self.name,
            *self.engine.time
        )

    @reify
    def user(self):
        return UserMapping(self)

    @property
    def portal(self):
        """Return a mapping of portals to other nodes."""
        return self.character.portal[self.name]

    def __init__(self, character, name):
        """Store character and name, and initialize caches"""
        self.character = character
        self.engine = character.engine
        self.gorm = self.engine
        self.graph = self.character
        self.name = self.node = name
        if self.engine.caching:
            (branch, tick) = self.engine.time
            self._dispatch_cache = self.engine._node_val_cache[self.character.name][self.name]

    def __iter__(self):
        yield from super().__iter__()
        yield from self.extrakeys
        return

    def __contains__(self, k):
        """Handle extra keys, then delegate."""
        if k in self.extrakeys:
            return True
        return super().__contains__(k)

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        self.dispatch(k, v)

    def __delitem__(self, k):
        super().__delitem__(k)
        self.dispatch(k, None)

    def _portal_dests(self):
        """Iterate over names of nodes you can get to from here"""
        if self.engine.caching:
            cache = self.engine._edges_cache[self.character.name][self.name]
            (branch, tick) = self.engine.time
            for nodeB in cache:
                try:
                    if cache[nodeB][0][branch][tick]:
                        yield nodeB
                except (KeyError, ValueError):
                    continue
            return
        yield from self.engine.db.nodeBs(
            self.character.name,
            self.name,
            *self.engine.time
        )

    def _portal_origs(self):
        """Iterate over names of nodes you can get here from"""
        if self.engine.caching:
            cache = self.engine._edges_cache[self.character.name]
            (branch, tick) = self.engine.time
            for nodeA in cache:
                if self.name not in cache[nodeA]:
                    continue
                try:
                    if cache[nodeA][self.name][0][branch][tick]:
                        yield nodeA
                except (KeyError, ValueError):
                    continue
            return
        yield from self.engine.db.nodeAs(
            self.character.name,
            self.name,
            *self.engine.time
        )

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
        for user in list(self.user.values()):
            user.del_avatar(self.character.name, self.name)
        (branch, tick) = self.engine.time
        self.engine.db.exist_node(
            self.character.name,
            self.name,
            branch,
            tick,
            False
        )
        if self.engine.caching:
            self.engine._nodes_cache[self.character.name][self.name][branch][tick] = False

    def one_way_portal(self, other, **stats):
        """Connect a portal from here to another node, and return it."""
        return self.character.new_portal(
            self, other, symmetrical=False, **stats
        )

    def one_way(self, other, **stats):
        """Connect a portal from here to another node, and return it."""
        return self.one_way_portal(other, **stats)

    def two_way_portal(self, other, **stats):
        """Connect these nodes with a two-way portal and return it."""
        return self.character.new_portal(
            self, other, symmetrical=True, **stats
        )

    def two_way(self, other, **stats):
        """Connect these nodes with a two-way portal and return it."""
        return self.two_way_portal(other, **stats)

    def new_thing(self, name, statdict={}, **stats):
        """Create a new thing, located here, and return it."""
        return self.character.new_thing(
            name, self.name, None, statdict, **stats
        )

    def __bool__(self):
        return self.name in self.character.node

    def __eq__(self, other):
        return (
            isinstance(other, Node) and
            self.character == other.character and
            self.name == other.name
        )

    def __hash__(self):
        return hash((self.character.name, self.name))
