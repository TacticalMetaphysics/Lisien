# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""A base class for nodes that can be in a character.

Every actual node that you're meant to use will be a place or
thing. This module is for what they have in common.

"""
from collections import Mapping, ValuesView

from networkx import shortest_path, shortest_path_length

import allegedb.graph
from allegedb.cache import HistoryError

from .util import getatt
from .query import StatusAlias
from . import rule
from .exc import AmbiguousUserError


class RuleMapping(rule.RuleMapping):
    """Version of :class:`LiSE.rule.RuleMapping` that works more easily
    with a node.

    """
    def __init__(self, node):
        """Initialize with node's engine, character, and rulebook."""
        super().__init__(node.engine, node.rulebook)
        self.node = node


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

    def _user_names(self):
        cache = self.engine._avatarness_cache.user_order
        if self.node.character.name not in cache or \
           self.node.name not in cache[self.node.character.name]:
            return
        cache = cache[self.node.character.name][self.node.name]
        seen = set()
        for user in cache:
            if user in seen:
                continue
            for (branch, turn, tick) in self.node.engine._iter_parent_btt():
                if branch in cache[user]:
                    branchd = cache[user][branch]
                    try:
                        if turn in branchd:
                            if branchd[turn].get(tick, False):
                                yield user
                        elif branchd.rev_gettable(turn):
                            turnd = branchd[turn]
                            if turnd[turnd.end]:
                                yield user
                        seen.add(user)
                        break
                    except HistoryError as ex:
                        if ex.deleted:
                            break

    def __iter__(self):
        yield from self._user_names()

    def __len__(self):
        n = 0
        for user in self._user_names():
            n += 1
        return n

    def __contains__(self, item):
        if item in self.engine.character:
            item = self.engine.character[item]
        if hasattr(item, 'avatar'):
            charn = self.node.character.name
            nn = self.node.name
            return charn in item.avatar and nn in item.avatar[charn]
        return False

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("{} not used by {}".format(
                self.node.name, k
            ))
        return self.engine.character[k]


class NodeContentValues(ValuesView):
    def __iter__(self):
        node = self._mapping.node
        nodem = node.character.node
        try:
            for name in node.engine._node_contents_cache.retrieve(
                    node.character.name, node.name, *node.engine.btt()
            ):
                yield nodem[name]
        except KeyError:
            return

    def __contains__(self, item):
        return item.location == self._mapping.node


class NodeContent(Mapping):
    __slots__ = ('node',)

    def __init__(self, node):
        self.node = node

    def __iter__(self):
        try:
            yield from self.node.engine._node_contents_cache.retrieve(
                self.node.character.name, self.node.name, *self.node.engine.btt()
            )
        except KeyError:
            return

    def __len__(self):
        try:
            return len(self.node.engine._node_contents_cache.retrieve(
                self.node.character.name, self.node.name, *self.node.engine.btt()
            ))
        except KeyError:
            return 0

    def __contains__(self, item):
        try:
            return self.node.character.thing[item].location == self.node
        except KeyError:
            return False

    def __getitem__(self, item):
        if item not in self:
            raise KeyError
        return self.node.character.thing[item]

    def values(self):
        return NodeContentValues(self)


class DestsValues(ValuesView):
    def __contains__(self, item):
        return item.origin == self._mapping.node


class Dests(Mapping):
    __slots__ = ('node',)

    def __init__(self, node):
        self.node = node

    def __iter__(self):
        yield from self.node.engine._edges_cache.iter_successors(
            self.node.character.name, self.node.name, *self.node.engine.btt()
        )

    def __len__(self):
        return self.node.engine._edges_cache.count_successors(
            self.node.character.name, self.node.name, *self.node.engine.btt()
        )

    def __contains__(self, item):
        return self.node.engine._edges_cache.has_successor(
            self.node.character.name, self.node.name, item, *self.node.engine.btt()
        )

    def __getitem__(self, item):
        if item not in self:
            raise KeyError
        return self.node.character.portal[self.node.name][item]

    def values(self):
        return DestsValues(self)


class OrigsValues(ValuesView):
    def __contains__(self, item):
        return item.destination == self._mapping.node


class Origs(Mapping):
    __slots__ = ('node',)

    def __init__(self, node):
        self.node = node

    def __iter__(self):
        return self.node.engine._edges_cache.iter_predecessors(
            self.node.character.name, self.node.name, *self.node.engine.btt()
        )

    def __contains__(self, item):
        return self.node.engine._edges_cache.has_predecessor(
            self.node.character.name, self.node.name, item, *self.node.engine.btt()
        )

    def __len__(self):
        return self.node.engine._edges_cache.count_predecessors(
            self.node.character.name, self.node.name, *self.node.engine.btt()
        )

    def __getitem__(self, item):
        if item not in self:
            raise KeyError
        return self.node.character.portal[item][self.node.name]

    def values(self):
        return OrigsValues(self)


class UserDescriptor:
    """Give a node's user if there's only one

    If there are many users, but one of them has the same name as this node, give that one.

    Otherwise, raise AmbiguousUserError.

    """
    usermapping = UserMapping

    def __get__(self, instance, owner):
        mapping = self.usermapping(instance)
        it = iter(mapping)
        try:
            k = next(it)
        except StopIteration:
            raise AmbiguousUserError("No users")
        try:
            next(it)
            raise AmbiguousUserError("{} users. Use the ``users`` property".format(len(mapping)))
        except StopIteration:
            return mapping[k]
        except AmbiguousUserError:
            if instance.name in mapping:
                return mapping[instance.name]
            raise


class Node(allegedb.graph.Node, rule.RuleFollower):
    """The fundamental graph component, which edges (in LiSE, "portals")
    go between.

    Every LiSE node is either a thing or a place. They share in common
    the abilities to follow rules; to be connected by portals; and to
    contain things.

    """
    __slots__ = ()
    engine = getatt('db')
    character = getatt('graph')
    name = getatt('node')
    no_unwrap = True

    def _get_rule_mapping(self):
        return RuleMapping(self)

    def _get_rulebook_name(self):
        try:
            return self.engine._nodes_rulebooks_cache.retrieve(
                self.character.name, self.name, *self.engine.btt()
            )
        except KeyError:
            return self.character.name, self.name

    def _get_rulebook(self):
        return rule.RuleBook(
            self.engine,
            self._get_rulebook_name()
        )

    def _set_rulebook_name(self, rulebook):
        character = self.character.name
        node = self.name
        cache = self.engine._nodes_rulebooks_cache
        try:
            if rulebook == cache.retrieve(character, node, *self.engine.btt()):
                return
        except KeyError:
            pass
        branch, turn, tick = self.engine.nbtt()
        cache.store(character, node, branch, turn, tick, rulebook)
        self.engine.query.set_node_rulebook(character, node, branch, turn, tick, rulebook)

    @property
    def portal(self):
        """Return a mapping of portals connecting this node to its neighbors."""
        return Dests(self)
    successor = adj = edge = portal

    @property
    def preportal(self):
        return Origs(self)
    predecessor = pred = preportal

    user = UserDescriptor()

    @property
    def users(self):
        return UserMapping(self)

    def __init__(self, character, name):
        """Store character and name, and initialize caches"""
        super().__init__(character, name)
        self.db = character.engine

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
        self.send(self, key=k, val=v)

    def __delitem__(self, k):
        super().__delitem__(k)
        self.send(self, key=k, val=None)

    def portals(self):
        """Iterate over :class:`Portal` objects that lead away from me"""
        yield from self.portal.values()

    def successors(self):
        """Iterate over nodes with edges leading from here to there."""
        for port in self.portal.values():
            yield port.destination

    def preportals(self):
        """Iterate over :class:`Portal` objects that lead to me"""
        yield from self.preportal.values()

    def predecessors(self):
        """Iterate over nodes with edges leading here from there."""
        for port in self.preportal.values():
            yield port.origin

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

    @property
    def content(self):
        return NodeContent(self)

    def contents(self):
        return self.content.values()

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
        for user in list(self.users.values()):
            user.del_avatar(self.character.name, self.name)
        branch, turn, tick = self.engine.nbtt()
        self.engine._nodes_cache.store(
            self.character.name, self.name,
            branch, turn, tick, False
        )
        self.engine.query.exist_node(
            self.character.name, self.name,
            branch, turn, tick, False
        )
        self.character.node.send(self.character.node, key=self.name, val=None)

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

    def new_thing(self, name, **stats):
        """Create a new thing, located here, and return it."""
        return self.character.new_thing(
            name, self.name, **stats
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