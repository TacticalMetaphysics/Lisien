# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Directed edges, as used by LiSE."""

from allegedb.graph import Edge
from allegedb.cache import HistoryError

from .exc import CacheError
from .util import getatt
from .query import StatusAlias
from .rule import RuleFollower
from .rule import RuleMapping as BaseRuleMapping


class RuleMapping(BaseRuleMapping):
    """Mapping to get rules followed by a portal."""
    __slots__ = 'portal',

    def __init__(self, portal):
        """Store portal, engine, and rulebook."""
        super().__init__(portal.engine, portal.rulebook)
        self.portal = portal

    character = getatt('portal.character')
    orign = getatt('portal._orign')
    destn = getatt('portal._destn')

    def __iter__(self):
        for (rule, active) in self.portal._rule_names_activeness():
            if active:
                yield rule


class Portal(Edge, RuleFollower):
    """Connection between two Places that Things may travel along.

    Portals are one-way, but you can make one appear two-way by
    setting the ``symmetrical`` key to ``True``,
    eg. ``character.add_portal(orig, dest, symmetrical=True)``.
    The portal going the other way will appear to have all the
    stats of this one, and attempting to set a stat on it will
    set it here instead.

    """
    character = getatt('graph')
    engine = getatt('db')

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
        cache = self.engine._portals_rulebooks_cache
        if self.character.name not in cache or \
            self.orig not in cache[self.character.name] or \
            self.dest not in cache[
                self.character.name][self.orig]:
            return
        cache = cache[self.character.name][self.orig][self.dest]
        for (branch, turn, tick) in self.engine._iter_parent_btt():
            if branch in cache:
                try:
                    return cache[branch][turn][tick]
                except HistoryError as ex:
                    if ex.deleted:
                        break
        raise CacheError(
            "Rulebook for portal {}->{} in character {} is not cached.".format(
                self.orig, self.dest, self.character.name
            )
        )

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
        return "{}.portal[{}][{}]".format(
            self['character'],
            self['origin'],
            self['destination']
        )

    def __bool__(self):
        """It means something that I exist, even if I have no data."""
        return self.orig in self.character.portal and \
               self.dest in self.character.portal[self.orig]

    @property
    def origin(self):
        """Return the Place object that is where I begin"""
        return self.character.place[self.orig]

    @property
    def destination(self):
        """Return the Place object at which I end"""
        return self.character.place[self.dest]

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
        return StatusAlias(
            entity=self,
            stat=stat
        )

    def contents(self):
        """Iterate over Thing instances that are presently travelling through
        me.

        """
        for thing in self.character.thing.values():
            if thing['locations'] == (self.orig, self.dest):
                yield thing

    def new_thing(self, name, statdict={}, **kwargs):
        """Create and return a thing located in my origin and travelling to my
        destination."""
        return self.character.new_thing(
            name, self.orig, self.dest, statdict, **kwargs
        )

    def update(self, d):
        """Works like regular update, but only actually updates when the new
        value and the old value differ. This is necessary to prevent
        certain infinite loops.

        """
        for (k, v) in d.items():
            if k not in self or self[k] != v:
                self[k] = v

    def delete(self):
        """Remove myself from my :class:`Character`.

        For symmetry with :class:`Thing` and :class`Place`.

        """
        branch, turn, tick = self.engine.btt()
        self.engine._edges_cache.store(
            self.character.name,
            self.origin.name,
            self.destination.name,
            0,
            branch,
            turn,
            tick,
            None,
            planning=self.engine.planning,
            forward=self.engine.forward
        )
        self.engine.query.exist_edge(
            self.character.name,
            self.origin.name,
            self.destination.name,
            branch, turn, tick, False
        )
        try:
            del self.engine._portal_objs[
                (self.graph.name, self.orig, self.dest)
            ]
        except KeyError:
            pass
        self.character.portal[self.origin.name].send(
            self.character.portal[self.origin.name],
            key='dest', val=None
        )
