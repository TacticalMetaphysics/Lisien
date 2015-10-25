# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
import networkx as nx
from .node import Node
from .util import (
    path_len,
    CacheError,
    TravelException
)
from gorm.window import window_left


class Thing(Node):
    """The sort of item that has a particular location at any given time.

    If a Thing is in a Place, it is standing still. If it is in a
    Portal, it is moving through that Portal however fast it must in
    order to arrive at the other end when it is scheduled to. If it is
    in another Thing, then it is wherever that is, and moving the
    same.

    """
    extrakeys = {
        'name',
        'character',
        'location',
        'next_location',
        'arrival_time',
        'next_arrival_time'
    }

    def __contains__(self, key):
        if key in self.extrakeys:
            return True
        return super().__contains__(key)

    def __getitem__(self, key):
        """Return one of my stats stored in the database, or a few
        special cases:

        ``name``: return the name that uniquely identifies me within
        my Character

        ``character``: return the name of my character

        ``location``: return the name of my location

        ``arrival_time``: return the tick when I arrived in the
        present location

        ``next_location``: if I'm in transit, return where to, else return None

        ``next_arrival_time``: return the tick when I'm going to
        arrive at ``next_location``

        ``locations``: return a pair of ``(location, next_location)``

        """
        if key == 'name':
            return self.name
        elif key == 'character':
            return self.character.name
        elif key == 'location':
            return self['locations'][0]
        elif key == 'arrival_time':
            if not self.engine.caching:
                return self._get_arrival_time()
            cache = self.engine._things_cache[self.character.name][self.name]
            for (branch, tick) in self.engine._active_branches():
                if branch in cache:
                    return window_left(cache[branch].keys(), tick)
            raise CacheError("Locations not cached correctly")
        elif key == 'next_location':
            return self['locations'][1]
        elif key == 'next_arrival_time':
            if not self.engine.caching:
                return self._get_next_arrival_time()
            cache = self.engine._things_cache[self.character.name][self.name]
            for (branch, tick) in self.engine._active_branches():
                if branch in cache:
                    return min(t for t in cache[branch].keys() if t > tick)
            return None
        elif key == 'locations':
            if not self.engine.caching:
                return self._loc_and_next()
            cache = self.engine._things_cache[self.character.name][self.name]
            for (branch, tick) in self.engine._active_branches():
                if branch in cache:
                    return cache[branch][
                        window_left(cache[branch].keys(), tick)
                    ]
            raise CacheError("Locations not cached correctly")
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        """Set ``key``=``value`` for the present game-time."""
        if key == 'name':
            raise ValueError("Can't change names")
        elif key == 'character':
            raise ValueError("Can't change characters")
        elif key == 'location':
            self['locations'] = (value, None)
        elif key == 'arrival_time':
            raise ValueError("Read-only")
        elif key == 'next_location':
            self['locations'] = (self['location'], value)
        elif key == 'next_arrival_time':
            raise ValueError("Read-only")
        elif key == 'locations':
            self._set_loc_and_next(*value)
            if not self.engine.caching:
                return
            (branch, tick) = self.engine.time
            self.engine._things_cache\
                [self.character.name][self.name][branch][tick] = value
            self._dispatch_stat('locations', value)
        else:
            super().__setitem__(key, value)
            self._dispatch_stat(key, value)

    def __delitem__(self, key):
        """As of now, this key isn't mine."""
        if key in self.extrakeys:
            raise ValueError("Can't delete {}".format(key))
        super().__delitem__(key)
        self._dispatch_stat(key, None)

    def __repr__(self):
        """Return my character, name, and location"""
        if self['next_location'] is not None:
            return "{}.thing[{}]@{}->{}".format(
                self['character'],
                self['name'],
                self['location'],
                self['next_location']
            )
        return "{}.thing[{}]@{}".format(
            self['character'],
            self['name'],
            self['location']
        )

    def _get_arrival_time(self):
        """Query the database for when I arrive at my present location."""
        if self.engine.caching:
            cache = self.engine._things_cache[self.character.name][self.name]
            for (branch, tick) in self.engine._active_branches():
                try:
                    return window_left(cache[branch].keys(), tick)
                except (KeyError, ValueError):
                    continue
            raise CacheError("Thing seems never to have arrived where it is")
        loc = self['location']
        return self.engine.db.arrival_time_get(
            self.character.name,
            self.name,
            loc,
            *self.engine.time
        )

    def _get_next_arrival_time(self):
        """Query the database for when I will arrive at my next location, or
        ``None`` if I'm not traveling.

        """
        if self.engine.caching:
            cache = self.engine._things_cache[self.character.name][self.name]
            for (branch, tick) in self.engine._active_branches():
                try:
                    return min(t for t in cache[branch] if t > tick)
                except (KeyError, ValueError):
                    continue
            return None
        nextloc = self['next_location']
        if nextloc is None:
            return None
        return self.engine.db.next_arrival_time_get(
            self.character.name,
            self.name,
            nextloc,
            *self.engine.time
        )

    def delete(self, nochar=False):
        super().delete()
        if not nochar:
            del self.character.thing[self.name]
        (branch, tick) = self.engine.time
        self.engine.db.thing_loc_and_next_set(
            self.character.name,
            self.name,
            branch,
            tick,
            None,
            None
        )
        if self.engine.caching:
            self.engine._things_cache[self.character.name][self.name][branch][tick] = (None, None)

    def clear(self):
        """Unset everything."""
        for k in list(self.keys()):
            if k not in (
                    'name',
                    'character',
                    'location',
                    'next_location',
                    'arrival_time',
                    'next_arrival_time',
                    'locations'
            ):
                del self[k]

    @property
    def container(self):
        """If I am in transit, this is the Portal I'm moving through. Otherwise
        it's the Thing or Place I'm located in.

        """
        (a, b) = self['locations']
        try:
            return self.character.portal[a][b]
        except KeyError:
            try:
                return self.character.thing[a]
            except KeyError:
                return self.character.place[a]

    @property
    def location(self):
        """The Thing or Place I'm in. If I'm in transit, it's where I
        started.

        """
        if not self['location']:
            return None
        return self.character.node[self['location']]

    @property
    def next_location(self):
        """If I'm not in transit, this is None. If I am, it's where I'm
        headed.

        """
        locn = self['next_location']
        if not locn:
            return None
        try:
            return self.character.thing[locn]
        except KeyError:
            return self.character.place[locn]

    def _loc_and_next(self):
        """Private method that returns a pair in which the first item is my
        present ``location`` and the second is my ``next_location``,
        to which I am presently travelling.

        """
        if self.engine.caching:
            cache = self.engine._things_cache[self.character.name][self.name]
            for (branch, tick) in self.engine._active_branches():
                try:
                    return cache[branch][
                        window_left(cache[branch].keys(), tick)
                    ]
                except (KeyError, ValueError):
                    continue
            raise CacheError("Thing loc and next weren't cached right")
        return self.engine.db.thing_loc_and_next_get(
            self.character.name,
            self.name,
            *self.engine.time
        )

    def _set_loc_and_next(self, loc, nextloc):
        """Private method to simultaneously set ``location`` and
        ``next_location``

        """
        (branch, tick) = self.engine.time
        self.engine.db.thing_loc_and_next_set(
            self.character.name,
            self.name,
            branch,
            tick,
            loc,
            nextloc
        )
        if self.engine.caching:
            self.engine._things_cache[self.character.name][self.name][branch][tick] = (loc, nextloc)

    def go_to_place(self, place, weight=''):
        """Assuming I'm in a :class:`Place` that has a :class:`Portal` direct
        to the given :class:`Place`, schedule myself to travel to the
        given :class:`Place`, taking an amount of time indicated by
        the ``weight`` stat on the :class:`Portal`, if given; else 1
        tick.

        Return the number of ticks to travel.

        """
        if hasattr(place, 'name'):
            placen = place.name
        else:
            placen = place
        curloc = self["location"]
        orm = self.character.engine
        curtick = orm.tick
        ticks = self.character.portal[curloc][placen].get(weight, 1)
        self['next_location'] = placen
        orm.tick += ticks
        self['locations'] = (placen, None)
        orm.tick = curtick
        return ticks

    def follow_path(self, path, weight=None):
        """Go to several :class:`Place`s in succession, deciding how long to
        spend in each by consulting the ``weight`` stat of the
        :class:`Portal` connecting the one :class:`Place` to the next.

        Return the total number of ticks the travel will take. Raise
        :class:`TravelException` if I can't follow the whole path,
        either because some of its nodes don't exist, or because I'm
        scheduled to be somewhere else.

        """
        curtick = self.character.engine.tick
        prevplace = path.pop(0)
        if prevplace != self['location']:
            raise ValueError("Path does not start at my present location")
        subpath = [prevplace]
        for place in path:
            if (
                    prevplace not in self.character.portal or
                    place not in self.character.portal[prevplace]
            ):
                raise TravelException(
                    "Couldn't follow portal from {} to {}".format(
                        prevplace,
                        place
                    ),
                    path=subpath,
                    traveller=self
                )
            subpath.append(place)
            prevplace = place
        ticks_total = 0
        prevsubplace = subpath.pop(0)
        subsubpath = [prevsubplace]
        for subplace in subpath:
            if prevsubplace != self["location"]:
                l = self["location"]
                fintick = self.character.engine.tick
                self.character.engine.tick = curtick
                raise TravelException(
                    "When I tried traveling to {}, at tick {}, "
                    "I ended up at {}".format(
                        prevsubplace,
                        fintick,
                        l
                    ),
                    path=subpath,
                    followed=subsubpath,
                    branch=self.character.engine.branch,
                    tick=fintick,
                    lastplace=l,
                    traveller=self
                )
            portal = self.character.portal[prevsubplace][subplace]
            tick_inc = portal.get(weight, 1)
            self.go_to_place(subplace, weight)
            self.character.engine.tick += tick_inc
            ticks_total += tick_inc
            subsubpath.append(subplace)
            prevsubplace = subplace
        self.character.engine.tick = curtick
        return ticks_total

    def travel_to(self, dest, weight=None, graph=None):
        """Find the shortest path to the given :class:`Place` from where I am
        now, and follow it.

        If supplied, the ``weight`` stat of the :class:`Portal`s along
        the path will be used in pathfinding, and for deciding how
        long to stay in each Place along the way.

        The ``graph`` argument may be any NetworkX-style graph. It
        will be used for pathfinding if supplied, otherwise I'll use
        my :class:`Character`. In either case, however, I will attempt
        to actually follow the path using my :class:`Character`, which
        might not be possible if the supplied ``graph`` and my
        :class:`Character` are too different. If it's not possible,
        I'll raise a :class:`TravelException`, whose ``subpath``
        attribute holds the part of the path that I *can* follow. To
        make me follow it, pass it to my ``follow_path`` method.

        Return value is the number of ticks the travel will take.

        """
        destn = dest.name if hasattr(dest, 'name') else dest
        graph = self.character if graph is None else graph
        path = nx.shortest_path(graph, self["location"], destn, weight)
        return self.follow_path(path, weight)

    def travel_to_by(self, dest, arrival_tick, weight=None, graph=None):
        """Arrange to travel to ``dest`` such that I arrive there at
        ``arrival_tick``.

        Optional argument ``weight`` indicates what attribute of
        portals will indicate how long they take to go through.

        Optional argument ``graph`` is the graph to perform
        pathfinding with. If it contains a viable path that my
        character does not, you'll get a TravelException.

        """
        curtick = self.character.engine.tick
        if arrival_tick <= curtick:
            raise ValueError("travel always takes positive amount of time")
        destn = dest.name if hasattr(dest, 'name') else dest
        graph = self.character if graph is None else graph
        curloc = self["location"]
        path = nx.shortest_path(graph, curloc, destn, weight)
        travel_time = path_len(graph, path, weight)
        start_tick = arrival_tick - travel_time
        if start_tick <= curtick:
            raise self.TravelException(
                "path too heavy to follow by the specified tick",
                path=path,
                traveller=self
            )
        self.character.engine.tick = start_tick
        self.follow_path(path, weight)
        self.character.engine.tick = curtick

    def _get_json_dict(self):
        (branch, tick) = self.character.engine.time
        return {
            "type": "Thing",
            "version": 0,
            "character": self.character.name,
            "name": self.name,
            "branch": branch,
            "tick": tick,
            "stat": dict(self)
        }

    def dump(self):
        """Return a JSON representation of my present state only, not any of
        my history.

        """
        return self.engine.json_dump(self._get_json_dict())
