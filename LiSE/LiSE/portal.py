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
"""Directed edges, as used by LiSE."""

from .allegedb.graph import Edge
from .allegedb import HistoryError

from .util import getatt
from .query import StatusAlias
from .rule import RuleFollower
from .rule import RuleMapping as BaseRuleMapping


class RuleMapping(BaseRuleMapping):
    """Mapping to get rules followed by a portal."""

    def __init__(self, portal):
        """Store portal, engine, and rulebook."""
        super().__init__(portal.engine, portal.rulebook)
        self.portal = portal


class Portal(Edge, RuleFollower):
    """Connection between two Places that Things may travel along.

    Portals are one-way, but you can make one appear two-way by
    setting the ``symmetrical`` key to ``True``,
    eg. ``character.add_portal(orig, dest, symmetrical=True)``.
    The portal going the other way will appear to have all the
    stats of this one, and attempting to set a stat on it will
    set it here instead.

    """
    __slots__ = ('graph', 'orig', 'dest', 'idx', 'origin', 'destination', '_rulebook')
    character = getatt('graph')
    engine = getatt('db')
    no_unwrap = True

    def __init__(self, graph, orig, dest, idx=0):
        super().__init__(graph, orig, dest, idx)
        self.origin = graph.node[orig]
        self.destination = graph.node[dest]

    @property
    def _cache(self):
        return self.db._edge_val_cache[
            self.character.name][self.orig][self.dest][0]

    def _rule_name_activeness(self):
        rulebook_name = self._get_rulebook_name()
        cache = self.engine._active_rules_cache
        if rulebook_name not in cache:
            return
        cache = cache[rulebook_name]
        for rule in cache:
            for (branch, turn, tick) in self.engine._iter_parent_btt():
                if branch not in cache[rule]:
                    continue
                try:
                    yield (rule, cache[rule][branch][turn][tick])
                    break
                except ValueError:
                    continue
                except HistoryError as ex:
                    if ex.deleted:
                        break
        raise KeyError("{}->{} has no rulebook?".format(
            self.orig, self.dest
        ))

    def _get_rulebook_name(self):
        try:
            return self.engine._portals_rulebooks_cache.retrieve(
                self.character.name, self.orig, self.dest, *self.engine._btt()
            )
        except KeyError:
            return (self.character.name, self.orig, self.dest)

    def _set_rulebook_name(self, rulebook):
        character = self.character
        orig = self.orig
        dest = self.dest
        cache = self.engine._portals_rulebooks_cache
        try:
            if rulebook == cache.retrieve(character, orig, dest, *self.engine._btt()):
                return
        except KeyError:
            pass
        branch, turn, tick = self.engine._nbtt()
        cache.store(character, orig, dest, branch, turn, tick, rulebook)
        self.engine.query.set_portal_rulebook(character, orig, dest, branch, turn, tick, rulebook)

    def _get_rule_mapping(self):
        return RuleMapping(self)

    def __getitem__(self, key):
        """Get the present value of the key.

        If I am a mirror of another Portal, return the value from that
        Portal instead.

        """
        if key == 'origin':
            return self.orig
        elif key == 'destination':
            return self.dest
        elif key == 'character':
            return self.character.name
        elif key == 'is_mirror':
            try:
                return super().__getitem__(key)
            except KeyError:
                return False
        elif 'is_mirror' in self and self['is_mirror']:
            return self.character.preportal[
                self.orig
            ][
                self.dest
            ][
                key
            ]
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        """Set ``key``=``value`` at the present game-time.

        If I am a mirror of another Portal, set ``key``==``value`` on
        that Portal instead.

        """
        if key in ('origin', 'destination', 'character'):
            raise KeyError("Can't change " + key)
        elif 'is_mirror' in self and self['is_mirror']:
            self.reciprocal[key] = value
            return
        elif key == 'symmetrical' and value:
            if (
                    self.dest not in self.character.portal or
                    self.orig not in
                    self.character.portal[self.dest]
            ):
                self.character.add_portal(self.dest, self.orig)
            self.character.portal[
                self.dest
            ][
                self.orig
            ][
                "is_mirror"
            ] = True
            self.send(self, key='symmetrical', val=False)
            return
        elif key == 'symmetrical' and not value:
            try:
                self.character.portal[
                    self.dest
                ][
                    self.orig
                ][
                    "is_mirror"
                ] = False
            except KeyError:
                pass
            self.send(self, key='symmetrical', val=False)
            return
        super().__setitem__(key, value)

    def __repr__(self):
        """Describe character, origin, and destination"""
        return "{}.character[{}].portal[{}][{}]".format(
            repr(self.engine),
            repr(self['character']),
            repr(self['origin']),
            repr(self['destination'])
        )

    def __bool__(self):
        """It means something that I exist, even if I have no data."""
        return self.orig in self.character.portal and \
               self.dest in self.character.portal[self.orig]

    @property
    def reciprocal(self):
        """If there's another Portal connecting the same origin and
        destination that I do, but going the opposite way, return
        it. Else raise KeyError.

        """
        try:
            return self.character.portal[self.dest][self.orig]
        except KeyError:
            raise KeyError("This portal has no reciprocal")

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

    def update(self, d):
        """Works like regular update, but only actually updates when the new
        value and the old value differ. This is necessary to prevent
        certain infinite loops.

        :arg d: a dictionary

        """
        for (k, v) in d.items():
            if k not in self or self[k] != v:
                self[k] = v

    def delete(self):
        """Remove myself from my :class:`Character`.

        For symmetry with :class:`Thing` and :class`Place`.

        """
        self.clear()
        branch, turn, tick = self.engine._nbtt()
        self.engine._edges_cache.store(
            self.character.name,
            self.origin.name,
            self.destination.name,
            0,
            branch,
            turn,
            tick,
            None
        )
        self.engine.query.exist_edge(
            self.character.name,
            self.origin.name,
            self.destination.name,
            branch, turn, tick, False
        )
        try:
            del self.engine._edge_objs[
                (self.graph.name, self.orig, self.dest)
            ]
        except KeyError:
            pass
        self.character.portal[self.origin.name].send(
            self.character.portal[self.origin.name],
            key='dest', val=None
        )

    def unwrap(self):
        return {
            k: v.unwrap() if hasattr(v, 'unwrap') and not hasattr(v, 'no_unwrap')
            else v for (k, v) in self.items()
        }
