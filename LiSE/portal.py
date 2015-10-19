from collections import defaultdict

from gorm.graph import Edge

from .util import (
    dispatch,
    listener,
    unlisten,
    stat_validity,
    fire_stat_listeners
)
from .rule import RuleFollower
from .rule import RuleMapping as BaseRuleMapping


class RuleMapping(BaseRuleMapping):
    def __init__(self, portal):
        super().__init__(portal.engine, portal.rulebook)
        self.character = portal.character
        self.engine = portal.engine
        self.orign = portal._origin
        self.destn = portal._destn

    def __iter__(self):
        return self.engine.db.portal_rules(
            self.character.name,
            self.orign,
            self.destn,
            *self.engine.time
        )


class Portal(Edge, RuleFollower):
    """Connection between two Places that Things may travel along.

    Portals are one-way, but you can make one appear two-way by
    setting the ``symmetrical`` key to ``True``,
    eg. ``character.add_portal(orig, dest, symmetrical=True)``

    """
    def _rule_name_activeness(self):
        return self.engine.db.current_rules_portal(
            self.character.name,
            self._origin,
            self._destination,
            *self.engine.time
        )

    def _get_rulebook_name(self):
        return self.engine.db.portal_rulebook(
            self.character.name,
            self._origin,
            self._destination
        )

    def _get_rule_mapping(self):
        return RuleMapping(self)

    def __init__(self, character, origin, destination):
        """Initialize a Portal in a character from an origin to a
        destination

        """
        self._origin = origin
        self._destination = destination
        self._stat_listeners = defaultdict(list)
        self.character = character
        self.engine = character.engine
        self._keycache = {}
        self._cache = {}
        self._existence = {}

        if self.engine.caching:
            (branch, tick) = self.engine.time
            self._stats_validity = {}
            cache = self.engine.gorm._edges_cache[
                self.character.name][self._origin][self._destination][0]
            for k in cache:
                try:
                    self._stats_validity[k] = stat_validity(
                        k,
                        cache,
                        branch,
                        tick
                    )
                except ValueError:
                    pass

            @self.engine.time_listener
            def fire_my_stat_listeners(
                    branch_then,
                    tick_then,
                    branch_now,
                    tick_now
            ):
                fire_stat_listeners(
                    lambda k, v: dispatch(self._stat_listeners, k, branch_now, tick_now, self, k, v),
                    self.keys(),
                    cache,
                    self._stats_validity,
                    branch_then,
                    tick_then,
                    branch_now,
                    tick_now
                )

        super().__init__(character, self._origin, self._destination)

    def _dispatch_stat(self, k, v):
        if k in self and self[k] == v:
            return
        (branch, tick) = self.engine.time
        dispatch(self._stat_listeners, k, branch, tick, self, k, v)

    def listener(self, f=None, stat=None):
        return listener(self._stat_listeners, f, stat)

    def unlisten(self, f=None, stat=None):
        return unlisten(self._stat_listeners, f, stat)

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
        if not self.engine.caching:
            super().__setitem__(key, value)
            return
        if key in self.character._portal_traits:
            self.character._portal_traits = set()
        super().__setitem__(key, value)
        (branch, tick) = self.engine.time
        self._dispatch_stat(key, value)

    def __delitem__(self, key):
        """Invalidate my :class:`Character`'s cache of portal traits"""
        if not self.engine.caching:
            super().__delitem__(key)
            return
        if key in self.character._portal_traits:
            self.character._portal_traits = set()
        self._dispatch_stat(key, None)

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

    def contents(self):
        """Iterate over Thing instances that are presently travelling through
        me.

        """
        for thing in self.character.thing.values():
            if thing['locations'] == (self._origin, self._destination):
                yield thing

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
