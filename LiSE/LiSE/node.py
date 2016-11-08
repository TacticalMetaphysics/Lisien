# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""A base class for nodes that can be in a character.

Every actual node that you're meant to use will be a place or
thing. This module is for what they have in common.

"""
from collections import Mapping

from networkx import shortest_path, shortest_path_length

import gorm.graph

from .util import getatt
from .query import StatusAlias
from .bind import TimeDispatcher
from . import rule


class RuleMapping(rule.RuleMapping):
    """Version of :class:`LiSE.rule.RuleMapping` that works more easily
    with a node.

    """
    __slots__ = ['node']

    def __init__(self, node):
        """Initialize with node's engine, character, and rulebook."""
        super().__init__(node.engine, node.rulebook)
        self.node = node

    character = getatt('node.character')

    def __iter__(self):
        for (rul, active) in self.node._rule_names_activeness():
            if active:
                yield rul


class UserMapping(Mapping):
    """A mapping of the characters that have a particular node as an avatar.

    Getting characters from here isn't any better than getting them from
    the engine direct, but with this you can do things like use the
    .get() method to get a character if it's a user and otherwise
    get something else; or test whether the character's name is in
    the keys; and so on.

    """
    __slots__ = ['node']

    def __init__(self, node):
        """Store the node"""
        self.node = node

    engine = getatt('node.engine')

    def __iter__(self):
        yield from self.node._user_names()

    def __len__(self):
        n = 0
        for user in self.node._user_names():
            n += 1
        return n

    def __getitem__(self, k):
        if len(self) == 1:
            me = self.engine.character[next(self.node._user_names())]
            if k in me:
                return me[k]
        if k not in self.node._user_names():
            raise KeyError("{} not used by {}".format(
                self.node.name, k
            ))
        return self.engine.character[k]

    def __setitem__(self, k, v):
        if len(self) != 1:
            raise KeyError(
                "More than one user. "
                "Look up the one you want to set a stat on."
            )
        me = self.engine.character[next(self.node._user_names())]
        me[k] = v

    def __getattr__(self, attr):
        if len(self) == 1:
            me = self.engine.character[next(self.node._user_names())]
            if hasattr(me, attr):
                return getattr(me, attr)


class Node(gorm.graph.Node, rule.RuleFollower, TimeDispatcher):
    """The fundamental graph component, which edges (in LiSE, "portals")
    go between.

    Every LiSE node is either a thing or a place. They share in common
    the abilities to follow rules; to be connected by portals; and to
    contain things.

    """
    __slots__ = ['user', 'graph', 'gorm', 'node', '_getitem_dispatch', '_setitem_dispatch']

    def _get_rule_mapping(self):
        return RuleMapping(self)

    def _get_rulebook_name(self):
        cache = self.engine._nodes_rulebooks_cache._data
        key = (self.character.name, self.name)
        if key not in cache:
            return key
        return cache[key]

    def _get_rulebook(self):
        return rule.RuleBook(
            self.engine,
            self._get_rulebook_name()
        )

    def _set_rulebook_name(self, v):
        self.engine._set_node_rulebook(
            self.character.name,
            self.name,
            v
        )

    def _user_names(self):
        cache = self.engine._avatarness_cache.user_order
        if self.character.name not in cache or \
           self.name not in cache[self.character.name]:
            return
        cache = cache[self.character.name][self.name]
        seen = set()
        for user in cache:
            if user in seen:
                continue
            for (branch, tick) in self.engine._active_branches():
                if branch in cache[user]:
                    if cache[user][branch][tick]:
                        yield user
                    seen.add(user)
                    break

    @property
    def portal(self):
        """Return a mapping of portals connecting this node to its neighbors."""
        return self.character.portal[self.name]

    @property
    def engine(self):
        return self.gorm

    @property
    def character(self):
        return self.graph

    @property
    def name(self):
        return self.node

    def __init__(self, character, name):
        """Store character and name, and initialize caches"""
        self.user = UserMapping(self)
        self.graph = character
        self.gorm = character.engine
        self.node = name

    def __iter__(self):
        yield from super().__iter__()
        yield from self.extrakeys
        return

    def clear(self):
        for key in super().__iter__():
            del self[key]

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
        yield from self.engine._edges_cache.iter_entities(self.character.name, self.name, *self.engine.time)

    def _portal_origs(self):
        """Iterate over names of nodes you can get here from"""
        cache = self.engine._edges_cache.predecessors[self.character.name][self.name]
        for nodeB in cache:
            for (b, t) in self.engine._active_branches():
                if b in cache[nodeB][0]:
                    if b != self.engine.branch:
                        self.engine._edges_cache.store(self.character.name, self.name, nodeB, 0, *self.engine.time)
                    if cache[nodeB][0][b][t]:
                        yield nodeB
                        break

    def portals(self):
        """Iterate over :class:`Portal` objects that lead away from me"""
        for destn in self._portal_dests():
            yield self.character.portal[self.name][destn]

    def successors(self):
        """Iterate over nodes with edges leading from here to there."""
        for destn in self._portal_dests():
            yield self.character.node[destn]

    def preportals(self):
        """Iterate over :class:`Portal` objects that lead to me"""
        for orign in self._portal_origs():
            yield self.character.preportal[self.name][orign]

    def predecessors(self):
        """Iterate over nodes with edges leading here from there."""
        for orign in self._portal_origs():
            yield self.character.node[orign]

    def _sane_dest_name(self, dest):
        if isinstance(dest, Node):
            if dest.character != self.character:
                raise ValueError(
                    "{} not in {}".format(dest.name, self.character.name)
                )
            return dest.name
        else:
            if dest in self.character.node:
                return dest
            raise ValueError("{} not in {}".format(dest, self.character.name))

    def shortest_path_length(self, dest, weight=None):
        """Return the length of the path from me to ``dest``.

        Raise ``ValueError`` if ``dest`` is not a node in my character
        or the name of one.

        """

        return shortest_path_length(
            self.character, self.name, self._sane_dest_name(dest), weight
        )

    def shortest_path(self, dest, weight=None):
        """Return a list of node names leading from me to ``dest``.

        Raise ``ValueError`` if ``dest`` is not a node in my character
        or the name of one.

        """
        return shortest_path(
            self.character, self.name, self._sane_dest_name(dest), weight
        )

    def path_exists(self, dest, weight=None):
        """Return whether there is a path leading from me to ``dest``.

        With ``weight``, only consider edges that have a stat by the
        given name.

        Raise ``ValueError`` if ``dest`` is not a node in my character
        or the name of one.

        """
        try:
            return bool(self.shortest_path_length(dest, weight))
        except KeyError:
            return False

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
        self.engine._exist_node(self.character.name, self.name, False)

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

    def historical(self, stat):
        """Return a reference to the values that a stat has had in the past.

        You can use the reference in comparisons to make a history
        query, and execute the query by calling it, or passing it to
        ``self.engine.ticks_when``.

        """
        return StatusAlias(
            entity=self,
            stat=stat
        )

    def __bool__(self):
        return self.name in self.character.node
