# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Directed edges, as used by LiSE."""

from gorm.graph import Edge

from .exc import CacheError
from .util import getatt
from .query import StatusAlias
from .bind import TimeDispatcher
from .rule import RuleFollower
from .rule import RuleMapping as BaseRuleMapping


class RuleMapping(BaseRuleMapping):
    """Mapping to get rules followed by a portal."""
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


class Portal(Edge, RuleFollower, TimeDispatcher):
    """Connection between two Places that Things may travel along.

    Portals are one-way, but you can make one appear two-way by
    setting the ``symmetrical`` key to ``True``,
    eg. ``character.add_portal(orig, dest, symmetrical=True)``.
    The portal going the other way will appear to have all the
    stats of this one, and attempting to set a stat on it will
    set it here instead.

    """
    @property
    def _cache(self):
        return self._dispatch_cache

    def _rule_name_activeness(self):
        rulebook_name = self._get_rulebook_name()
        cache = self.engine._active_rules_cache
        if rulebook_name not in cache:
            return
        cache = cache[rulebook_name]
        for rule in cache:
            for (branch, tick) in self.engine._active_branches():
                if branch not in cache[rule]:
                    continue
                try:
                    yield (rule, cache[rule][branch][tick])
                    break
                except ValueError:
                    continue
        raise KeyError("{}->{} has no rulebook?".format(self._origin, self._destination))

    def _get_rulebook_name(self):
        cache = self.engine._portals_rulebooks_cache
        if self.character.name not in cache or \
            self._origin not in cache[self.character.name] or \
            self._destination not in cache[self.character.name][self._origin]:
            return
        cache = cache[self.character.name][self._origin][self._destination]
        for (branch, tick) in self.engine._active_branches():
            if branch in cache:
                return cache[branch][tick]
        raise CacheError("Rulebook for portal {}->{} in character {} is not cached.".format(
            self._origin, self._destination, self.character.name
        ))

    def _get_rule_mapping(self):
        return RuleMapping(self)

    def __init__(self, character, origin, destination):
        """Remember what portal I am, and initialize caches."""
        self._origin = origin
        self._destination = destination
        self.character = character
        self.engine = character.engine
        self._keycache = {}
        self._existence = {}

        self._dispatch_cache = self.engine._edge_val_cache[
            self.character.name][self._origin][self._destination][0]

        super().__init__(character, self._origin, self._destination)

    def __getitem__(self, key):
        """Get the present value of the key.

        If I am a mirror of another Portal, return the value from that
        Portal instead.

        """
        if key == 'origin':
            return self._origin
        elif key == 'destination':
            return self._destination
        elif key == 'character':
            return self.character.name
        elif key == 'is_mirror':
            try:
                return super().__getitem__(key)
            except KeyError:
                return False
        elif 'is_mirror' in self and self['is_mirror']:
            return self.character.preportal[
                self._origin
            ][
                self._destination
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
                    self._destination not in self.character.portal or
                    self._origin not in
                    self.character.portal[self._destination]
            ):
                self.character.add_portal(self._destination, self._origin)
            self.character.portal[
                self._destination
            ][
                self._origin
            ][
                "is_mirror"
            ] = True
            return
        elif key == 'symmetrical' and not value:
            try:
                self.character.portal[
                    self._destination
                ][
                    self._origin
                ][
                    "is_mirror"
                ] = False
            except KeyError:
                pass
            return
        super().__setitem__(key, value)
        self.dispatch(key, value)

    def __delitem__(self, key):
        """Invalidate my :class:`Character`'s cache of portal traits"""
        super().__delitem__(key)
        self.dispatch(key, None)

    def __repr__(self):
        """Describe character, origin, and destination"""
        return "{}.portal[{}][{}]".format(
            self['character'],
            self['origin'],
            self['destination']
        )

    def __bool__(self):
        """It means something that I exist, even if I have no data but my name."""
        return self._origin in self.character.portal and \
            self._destination in self.character.portal[self._origin]

    def __eq__(self, other):
        return (
            isinstance(other, Portal) and
            self.character == other.character and
            self._origin == other._origin and
            self._destination == other._destination
        )

    def __hash__(self):
        return hash((self.character.name, self._origin, self._destination))

    @property
    def origin(self):
        """Return the Place object that is where I begin"""
        return self.character.place[self._origin]

    @property
    def destination(self):
        """Return the Place object at which I end"""
        return self.character.place[self._destination]

    @property
    def reciprocal(self):
        """If there's another Portal connecting the same origin and
        destination that I do, but going the opposite way, return
        it. Else raise KeyError.

        """
        try:
            return self.character.portal[self._destination][self._origin]
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
            if thing['locations'] == (self._origin, self._destination):
                yield thing

    def new_thing(self, name, statdict={}, **kwargs):
        """Create and return a thing located in my origin and travelling to my
        destination."""
        return self.character.new_thing(
            name, self._origin, self._destination, statdict, **kwargs
        )

    def update(self, d):
        """Works like regular update, but only actually updates when the new
        value and the old value differ. This is necessary to prevent
        certain infinite loops.

        """
        for (k, v) in d.items():
            if k not in self or self[k] != v:
                self[k] = v

    def _get_json_dict(self):
        (branch, tick) = self.engine.time
        return {
            "type": "Portal",
            "version": 0,
            "branch": branch,
            "tick": tick,
            "character": self.character.name,
            "origin": self._origin,
            "destination": self._destination,
            "stat": dict(self)
        }

    def dump(self):
        """Return a JSON representation of my present state"""
        return self.engine.json_dump(self._get_json_dict())

    def delete(self):
        """Remove myself from my :class:`Character`.

        For symmetry with :class:`Thing` and :class`Place`.

        """
        del self.character.portal[self.origin.name][self.destination.name]
        (branch, tick) = self.engine.time
        self.engine._edges_cache[self.character.name][self.origin.name][self.destination.name][branch][tick] = False
