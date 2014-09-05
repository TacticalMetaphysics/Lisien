# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com


"""The basic data model of LiSE, based on NetworkX DiGraph objects
with various additions and conveniences.

"""

from collections import (
    Mapping,
    MutableMapping,
    Callable
)
import networkx as nx
from sqlite3 import IntegrityError
from gorm.graph import (
    DiGraph,
    GraphNodeMapping,
    GraphEdgeMapping,
    GraphSuccessorsMapping,
    DiGraphPredecessorsMapping,
    json_dump,
    json_load
)
from LiSE.util import path_len
from LiSE.rule import RuleBook, RuleMapping
from LiSE.funlist import FunList


def cache_get(branch, tick, cache, key, getter):
    if key not in cache:
        cache[key] = {}
    if branch not in cache[key]:
        cache[key][branch] = {}
    if tick not in cache[key][branch]:
        cache[key][branch][tick] = getter(key)
    return cache[key][branch][tick]


def cache_set(branch, tick, cache, key, value, setter):
    if key not in cache:
        cache[key] = {}
    if branch not in cache[key]:
        cache[key][branch] = {}
    cache[key][branch][tick] = value
    setter(key, value)


def cache_del(branch, tick, cache, key, deleter):
    if (
            key in cache and
            branch in cache[key] and
            tick in cache[key][branch]
    ):
        del cache[key][branch][tick]
    deleter(key)


class RuleFollower(object):
    """Object that has a rulebook associated, which you can get a
    RuleMapping into

    """
    @property
    def rulebook(self):
        n = self.engine.cursor.execute(
            "SELECT {}_rulebook FROM characters WHERE character=?;".format(
                self._book
            ),
            (self.character._name,)
        ).fetchone()[0]
        return RuleBook(self.engine, n)

    @rulebook.setter
    def rulebook(self, v):
        if not isinstance(v, str) or isinstance(v, RuleBook):
            raise TypeError("Use a :class:`RuleBook` or the name of one")
        n = v.name if isinstance(v, RuleBook) else v
        self.engine.cursor.execute(
            "UPDATE characters SET {}_rulebook=? WHERE character=?;".format(
                self._book
            ),
            (n, self.character._name)
        )

    @property
    def rule(self):
        return RuleMapping(self.character, self.rulebook, self._book)


class TravelException(Exception):
    """Exception for problems with pathfinding"""
    def __init__(
            self,
            message,
            path=None,
            followed=None,
            traveller=None,
            branch=None,
            tick=None,
            lastplace=None
    ):
        """Store the message as usual, and also the optional arguments:

        ``path``: a list of Place names to show such a path as you found

        ``followed``: the portion of the path actually followed

        ``traveller``: the Thing doing the travelling

        ``branch``: branch during travel

        ``tick``: tick at time of error (might not be the tick at the
        time this exception is raised)

        ``lastplace``: where the traveller was, when the error happened

        """
        self.path = path
        self.followed = followed
        self.traveller = traveller
        self.branch = branch
        self.tick = tick
        self.lastplace = lastplace
        super().__init__(message)


class CompositeDict(Mapping):
    """Read-only mapping that looks up values in a first dict if
    available, then a second dict if possible.

    Assumes the dicts have no overlap.

    """
    def __init__(self, d1, d2):
        """Store dictionaries"""
        self.d1 = d1
        self.d2 = d2

    def __iter__(self):
        """Iterate over both dictionaries' keys"""
        for k in self.d1:
            yield k
        for k in self.d2:
            yield k

    def __len__(self):
        """Sum the lengths of both dictionaries"""
        return len(self.d1) + len(self.d2)

    def __getitem__(self, k):
        """Get an item from ``d1`` if possible, then ``d2``"""
        try:
            return self.d1[k]
        except KeyError:
            return self.d2[k]


class ThingPlace(GraphNodeMapping.Node):
    """Superclass for both Thing and Place"""
    def __init__(self, character, name):
        self.character = character
        self.engine = character.engine
        self._name = json_dump(name)
        if self.engine.caching:
            self._keycache = {}
            self._statcache = {}
        super().__init__(character, name)

    @property
    def name(self):
        return json_load(self._name)

    @property
    def exists(self):
        return self.engine._node_exists(self.graph.name, self.node)

    @exists.setter
    def exists(self, v):
        GraphNodeMapping.Node.exists.fset(self, v)
        if not self.engine.caching:
            return
        (branch, tick) = self.engine.time
        if self.character.name not in self.engine._existence:
            self.engine._existence[self.character.name] = {}
        d = self.engine._existence[self.character.name]
        if self.name not in d:
            d[self.name] = {}
        if branch not in d[self.name]:
            d[self.name][branch] = {}
        d[self.name][branch][tick] = v

    def _portal_dests(self):
        """Iterate over names of nodes you can get to from here"""
        seen = set()
        for (branch, tick) in self.gorm._active_branches():
            self.gorm.cursor.execute(
                "SELECT edges.nodeB, edges.extant FROM edges JOIN "
                "(SELECT graph, nodeA, nodeB, idx, branch, MAX(rev) AS rev "
                "FROM edges "
                "WHERE graph=? "
                "AND nodeA=? "
                "AND branch=? "
                "AND rev<=? "
                "GROUP BY graph, nodeA, nodeB, idx, branch) AS hirev "
                "ON edges.graph=hirev.graph "
                "AND edges.nodeA=hirev.nodeA "
                "AND edges.nodeB=hirev.nodeB "
                "AND edges.idx=hirev.idx "
                "AND edges.branch=hirev.branch "
                "AND edges.rev=hirev.rev;",
                (
                    self.character._name,
                    self._name,
                    branch,
                    tick
                )
            )
            for (d, exists) in self.gorm.cursor.fetchall():
                dest = json_load(d)
                if exists and dest not in seen:
                    yield dest
                seen.add(dest)

    def _portal_origs(self):
        """Iterate over names of nodes you can get here from"""
        seen = set()
        for (branch, tick) in self.gorm._active_branches():
            self.gorm.cursor.execute(
                "SELECT edges.nodeA, edges.extant FROM edges JOIN "
                "(SELECT graph, nodeA, nodeB, idx, branch, MAX(rev) AS rev "
                "FROM edges "
                "WHERE graph=? "
                "AND nodeB=? "
                "AND branch=? "
                "AND rev<=? "
                "GROUP BY graph, nodeA, nodeB, idx, branch) AS hirev "
                "ON edges.graph=hirev.graph "
                "AND edges.nodeA=hirev.nodeA "
                "AND edges.nodeB=hirev.nodeB "
                "AND edges.idx=hirev.idx "
                "AND edges.rev=hirev.rev;",
                (
                    self.character._name,
                    self._name,
                    branch,
                    tick
                )
            )
            for (o, exists) in self.gorm.cursor.fetchall():
                orig = json_load(o)
                if exists and orig not in seen:
                    yield orig
                seen.add(orig)

    def _user_names(self):
        """Iterate over names of characters that have me as an avatar"""
        seen = set()
        for (branch, tick) in self.engine._active_branches():
            self.engine.cursor.execute(
                "SELECT avatars.avatar_graph FROM avatars JOIN ("
                "SELECT character_graph, avatar_graph, avatar_node, "
                "branch, MAX(tick) AS tick "
                "FROM avatars WHERE "
                "avatar_graph=? AND "
                "avatar_node=? AND "
                "branch=? AND "
                "tick<=? GROUP BY "
                "character_graph, avatar_graph, avatar_node, "
                "branch) AS hitick "
                "ON avatars.character_graph=hitick.character_graph "
                "AND avatars.avatar_graph=hitick.avatar_graph "
                "AND avatars.avatar_node=hitick.avatar_node "
                "AND avatars.branch=hitick.branch "
                "AND avatars.tick=hitick.tick;",
                (
                    self.character._name,
                    self._name,
                    branch,
                    tick
                )
            )
            for row in self.engine.cursor.fetchall():
                charn = json_load(row[0])
                if charn not in seen:
                    yield charn
                    seen.add(charn)

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


class Thing(ThingPlace):
    """The sort of item that has a particular location at any given time.

    If a Thing is in a Place, it is standing still. If it is in a
    Portal, it is moving through that Portal however fast it must in
    order to arrive at the other end when it is scheduled to. If it is
    in another Thing, then it is wherever that is, and moving the
    same.

    """
    def __iter__(self):
        extrakeys = [
            'name',
            'character',
            'location',
            'next_location',
            'arrival_time',
            'next_arrival_time'
        ]
        if not self.engine.caching:
            yield from extrakeys
            yield from super().__iter__()
            return
        (branch, tick) = self.engine.time
        if branch not in self._keycache:
            self._keycache[branch] = {}
        if tick not in self._keycache[branch]:
            self._keycache[branch][tick] = extrakeys + list(super().__iter__())
        yield from self._keycache[branch][tick]

    def __getitem__(self, key):
        """Return one of my attributes stored in the database, with a few
        special exceptions:

        ``name``: return the name that uniquely identifies me within
        my Character

        ``character``: return the name of my character

        ``location``: return the name of my location

        ``arrival_time``: return the tick when I arrived in the
        present location

        ``next_location``: if I'm in transit, return where to, else return None

        ``next_arrival_time``: return the tick when I'm going to
        arrive at ``next_location``

        ``locations``: return a pair of (``location``, ``next_location``)

        """
        def getcache(getter):
            if not self.engine.caching:
                return getter(key)
            (branch, tick) = self.engine.time
            return cache_get(
                branch,
                tick,
                self._statcache,
                key,
                getter
            )
        if key == 'name':
            return self.name
        elif key == 'character':
            return self.character.name
        elif key == 'location':
            return self['locations'][0]
        elif key == 'arrival_time':
            if not self.engine.caching:
                return self._get_arrival_time()
            (branch, tick) = self.engine.time
            arrt = lambda: max(
                t for t in
                self._statcache['locations'][branch]
                if t <= tick
            )
            try:
                return arrt()
            except ValueError:
                getcache(lambda k: self._loc_and_next())
                return arrt()
        elif key == 'next_location':
            return self['locations'][1]
        elif key == 'next_arrival_time':
            if not self.engine.caching:
                return self._get_next_arrival_time()
            (branch, tick) = self.engine.time
            narrt = lambda: min(
                t for t in
                self._statcache['locations'][branch]
                if t > tick
            )
            try:
                return narrt()
            except ValueError:
                try:
                    getcache(lambda k: self._loc_and_next())
                    return narrt()
                except ValueError:
                    return None
        elif key == 'locations':
            return getcache(lambda k: self._loc_and_next())
        else:
            return getcache(super().__getitem__)

    def __setitem__(self, key, value):
        """Set ``key``==``value`` for the present game-time."""
        def setcache(setter):
            if not self.engine.caching:
                setter(key, value)
                return
            (branch, tick) = self.engine.time
            cache_set(
                branch,
                tick,
                self._statcache,
                key,
                value,
                setter
            )
            self.character.changect += 1

        if key == 'name':
            raise ValueError("Can't change names")
        elif key == 'character':
            raise ValueError("Can't change characters")
        elif key == 'location':
            if not self.engine.caching:
                self._set_loc_and_next(value, None)
                return
            (branch, tick) = self.engine.time
            cache_set(
                branch,
                tick,
                self._statcache,
                'locations',
                (value, None),
                lambda k, v: self._set_loc_and_next(*v)
            )
        elif key == 'arrival_time':
            raise ValueError("Read-only")
        elif key == 'next_location':
            if not self.engine.caching:
                self._set_loc_and_next(self['location'], value)
                return
            (branch, tick) = self.engine.time
            cache_set(
                branch,
                tick,
                self._statcache,
                'locations',
                (self['location'], value),
                lambda k, v: self._set_loc_and_next(*v)
            )
        elif key == 'next_arrival_time':
            raise ValueError("Read-only")
        elif key == 'locations':
            setcache(lambda k, v: self._set_loc_and_next(*v))
        else:
            setcache(super().__setitem__)

    def __delitem__(self, key):
        if key in (
                'name',
                'character',
                'location',
                'arrival_time',
                'next_location',
                'next_arrival_time',
                'locations'
        ):
            raise ValueError("Read-only")
        if not self.engine.caching:
            super().__delitem__(key)
            return
        (branch, tick) = self.engine.time
        cache_del(
            branch,
            tick,
            self._statcache,
            key,
            super().__delitem__
        )
        self.character.changect += 1

    def _get_arrival_time(self):
        curloc = json_dump(self['location'])
        for (branch, tick) in self.gorm._active_branches():
            data = self.gorm.cursor.execute(
                "SELECT MAX(tick) FROM things "
                "WHERE character=? "
                "AND thing=? "
                "AND location=? "
                "AND branch=? "
                "AND tick<=?;",
                (
                    self.character._name,
                    self._name,
                    curloc,
                    branch,
                    tick
                )
            ).fetchall()
            if len(data) == 0:
                continue
            elif len(data) > 1:
                raise ValueError(
                    "How do you get more than one record from that?"
                )
            else:
                return data[0][0]
        raise ValueError("I don't seem to have arrived where I am?")

    def _get_next_arrival_time(self):
        nextloc = json_dump(self['next_location'])
        if nextloc is None:
            return None
        for (branch, tick) in self.engine._active_branches():
            data = self.engine.cursor.execute(
                "SELECT MIN(tick) FROM things "
                "WHERE character=? "
                "AND thing=? "
                "AND location=? "
                "AND branch=? "
                "AND tick>?;",
                (
                    json_dump(self["character"]),
                    json_dump(self.name),
                    nextloc,
                    branch,
                    tick
                )
            ).fetchall()
            if len(data) == 0:
                continue
            elif len(data) > 1:
                raise ValueError(
                    "How do you get more than one record from that?"
                )
            else:
                return data[0][0]

    def delete(self):
        del self.character.thing[self.name]

    def clear(self):
        for k in self:
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
        self['locations'] = (None, None)
        self.exists = False

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
        for (branch, tick) in self.gorm._active_branches():
            self.gorm.cursor.execute(
                "SELECT location, next_location FROM things JOIN ("
                "SELECT character, thing, branch, MAX(tick) AS tick "
                "FROM things "
                "WHERE character=? "
                "AND thing=? "
                "AND branch=? "
                "AND tick<=? "
                "GROUP BY character, thing, branch) AS hitick "
                "ON things.character=hitick.character "
                "AND things.thing=hitick.thing "
                "AND things.branch=hitick.branch "
                "AND things.tick=hitick.tick;",
                (
                    self.character._name,
                    self._name,
                    branch,
                    tick
                )
            )
            data = self.gorm.cursor.fetchall()
            if len(data) == 0:
                continue
            elif len(data) > 1:
                raise ValueError("Silly data in things table")
            else:
                (l, nl) = data[0]
                return (json_load(l), json_load(nl) if nl else None)
        raise ValueError("No location set")

    def _set_loc_and_next(self, loc, nextloc):
        """Private method to simultaneously set ``location`` and
        ``next_location``

        """
        (branch, tick) = self.character.engine.time
        charn = self.character._name
        myn = self._name
        locn = json_dump(loc)
        nextlocn = json_dump(nextloc)
        try:
            self.character.engine.cursor.execute(
                "INSERT INTO things ("
                "character, "
                "thing, "
                "branch, "
                "tick, "
                "location, "
                "next_location) VALUES ("
                "?, ?, ?, ?, ?, ?);",
                (
                    charn,
                    myn,
                    branch,
                    tick,
                    locn,
                    nextlocn
                )
            )
        except IntegrityError:
            self.character.engine.cursor.execute(
                "UPDATE things SET location=?, next_location=? "
                "WHERE character=? "
                "AND thing=? "
                "AND branch=? "
                "AND tick=?;",
                (
                    locn,
                    nextlocn,
                    charn,
                    myn,
                    branch,
                    tick
                )
            )
        self.character.changect += 1

    def go_to_place(self, place, weight=''):
        """Assuming I'm in a Place that has a Portal direct to the given
        Place, schedule myself to travel to the given Place, taking an
        amount of time indicated by the ``weight`` stat on the Portal,
        if given; else 1 tick.

        Return the number of ticks to travel.

        """
        if hasattr(place, 'name'):
            placen = place.name
        else:
            placen = place
        for fun in self.character.travel_reqs:
            if not fun(self.name, placen):
                (branch, tick) = self.engine.time
                raise TravelException(
                    "{} cannot travel through {}".format(self.name, placen),
                    traveller=self,
                    branch=branch,
                    tick=tick,
                    lastplace=self.location
                )
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
        """Go to several Places in succession, deciding how long to spend in
        each by consulting the ``weight`` attribute of the Portal
        connecting the one Place to the next.

        Return the total number of ticks the travel will take. Raise
        TravelException if I can't follow the whole path, either
        because some of its nodes don't exist, or because I'm
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
        """Find the shortest path to the given Place from where I am now, and
        follow it.

        If supplied, the ``weight`` stat of the Portals along the path
        will be used in pathfinding, and for deciding how long to stay
        in each Place along the way.

        The ``graph`` argument may be any NetworkX-style graph. It
        will be used for pathfinding if supplied, otherwise I'll use
        my Character. In either case, however, I will attempt to
        actually follow the path using my Character, which might not
        be possible if the supplied ``graph`` and my Character are too
        different. If it's not possible, I'll raise a TravelException,
        whose ``subpath`` attribute holds the part of the path that I
        *can* follow. To make me follow it, pass it to my
        ``follow_path`` method.

        Return value is the number of ticks the travel will take.

        """
        destn = dest.name if hasattr(dest, 'name') else dest
        graph = self.character if graph is None else graph
        if graph is None and '_paths' in self.character.graph:
            # use a cached path
            paths = self.character._paths
            path = paths[weight][self['location']][destn]
        elif hasattr(graph, 'graph') and '_paths' in graph.graph:
            # use a cached path from the given graph
            paths = graph._paths
            path = paths[weight][self['location']][destn]
        else:
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
        return json_dump(self._get_json_dict())


class Place(ThingPlace):
    """The kind of node where a Thing might ultimately be located."""
    def __getitem__(self, key):
        """Return my name if ``key``=='name', my character's name if
        ``key``=='character', the names of everything located in me if
        ``key``=='contents', or the value of the stat named ``key``
        otherwise.

        """
        if key == 'name':
            return self.name
        elif key == 'character':
            return self.character.name
        else:
            if not self.engine.caching:
                return super().__getitem__(key)
            (branch, tick) = self.engine.time
            return cache_get(
                branch,
                tick,
                self._statcache,
                key,
                super().__getitem__
            )

    def __setitem__(self, key, value):
        if not self.engine.caching:
            super().__setitem__(key, value)
            return
        (branch, tick) = self.engine.time
        cache_set(
            branch,
            tick,
            self._statcache,
            key,
            value,
            super().__setitem__
        )
        self.character.changect += 1

    def __delitem__(self, key):
        if not self.engine.caching:
            super().__delitem__(key)
            return
        (branch, tick) = self.engine.time
        cache_del(
            branch,
            tick,
            self._statcache,
            key,
            super().__delitem__
        )
        self.character.changect += 1

    def _get_json_dict(self):
        (branch, tick) = self.engine.time
        return {
            "type": "Place",
            "version": 0,
            "character": self["character"],
            "name": self["name"],
            "branch": branch,
            "tick": tick,
            "stat": dict(self)
        }

    def dump(self):
        """Return a JSON representation of my present state"""
        return json_dump(self._get_json_dict())


class Portal(GraphEdgeMapping.Edge):
    """Connection between two Places that Things may travel along.

    Portals are one-way, but you can make one appear two-way by
    setting the ``symmetrical`` key to ``True``,
    eg. ``character.add_portal(orig, dest, symmetrical=True)``

    """
    @property
    def exists(self):
        if not self.engine.caching:
            return GraphEdgeMapping.Edge.exists.fget(self)
        (branch, rev) = self.engine.time
        if branch not in self._existence:
            self._existence[branch] = {}
        if rev not in self._existence[branch]:
            self._existence[branch][
                rev
            ] = GraphEdgeMapping.Edge.exists.fget(self)
        return self._existence[branch][rev]

    @exists.setter
    def exists(self, v):
        GraphEdgeMapping.Edge.exists.fset(self, v)
        if not self.engine.caching:
            return
        (branch, rev) = self.engine.time
        if branch not in self._existence:
            self._existence[branch] = {}
        self._existence[branch][rev] = v

    def __init__(self, character, origin, destination):
        """Initialize a Portal in a character from an origin to a
        destination

        """
        self._origin = origin
        self._destination = destination
        self.character = character
        self.engine = character.engine
        if self.engine.caching:
            self._keycache = {}
            self._statcache = {}
            self._existence = {}
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
            if not self.engine.caching:
                return super().__getitem__(key)
            (branch, tick) = self.engine.time
            cache_get(
                branch,
                tick,
                self._statcache,
                key,
                super().__getitem__
            )

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
        if not self.engine.caching:
            super().__setitem__(key, value)
            return
        if key in self.character._portal_traits:
            del self.character.graph['_paths']
            self.character._portal_traits = set()
        (branch, tick) = self.engine.time
        cache_set(
            branch,
            tick,
            self._statcache,
            key,
            value,
            super().__setitem__
        )
        self.character.changect += 1

    def __delitem__(self, key):
        """Invalidate my :class:`Character`'s cache of portal traits"""
        if not self.engine.caching:
            super().__delitem__(key)
            return
        if key in self.character._portal_traits:
            del self.character.graph['_paths']
            self.character._portal_traits = set()
        (branch, tick) = self.engine.time
        cache_del(
            branch,
            tick,
            self._statcache,
            key,
            super().__delitem__
        )
        self.character.changect += 1

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
        return json_dump(self._get_json_dict())


class CharacterThingMapping(MutableMapping, RuleFollower):
    """:class:`Thing` objects that are in a :class:`Character`"""
    _book = "thing"

    def __init__(self, character):
        """Store the character and initialize cache (if caching)"""
        self.character = character
        self.engine = character.engine
        self.name = character.name
        if self.engine.caching:
            self._cache = {}

    def __contains__(self, k):
        """Check the cache first, if it exists"""
        if hasattr(self, '_cache') and k in self._cache:
            return True
        return super().__contains__(k)

    def __iter__(self):
        """Iterate over nodes that have locations, and are therefore
        Things. Yield their names.

        """
        seen = set()
        myn = json_dump(self.name)
        for (branch, tick) in self.engine._active_branches():
            self.engine.cursor.execute(
                "SELECT things.thing, things.location FROM things JOIN ("
                "SELECT character, thing, branch, MAX(tick) AS tick "
                "FROM things "
                "WHERE character=? "
                "AND branch=? "
                "AND tick<=? "
                "GROUP BY character, thing, branch) AS hitick "
                "ON things.character=hitick.character "
                "AND things.thing=hitick.thing "
                "AND things.branch=hitick.branch "
                "AND things.tick=hitick.tick "
                "LEFT OUTER JOIN "
                "(SELECT nodes.graph, nodes.node, nodes.branch, nodes.rev, "
                "nodes.extant FROM nodes JOIN "
                "(SELECT graph, node, branch, MAX(rev) AS rev FROM nodes "
                "WHERE graph=? "
                "AND branch=? "
                "AND rev<=? GROUP BY graph, node, branch) AS hirev ON "
                "nodes.graph=hirev.graph AND "
                "nodes.node=hirev.node AND "
                "nodes.branch=hirev.branch AND "
                "nodes.rev=hirev.rev) AS existence ON "
                "things.character=existence.graph AND "
                "things.thing=existence.node "
                "WHERE existence.extant;",
                (
                    myn,
                    branch,
                    tick,
                    myn,
                    branch,
                    tick
                )
            )
            for (n, l) in self.engine.cursor.fetchall():
                node = json_load(n)
                loc = json_load(l)
                if loc and node not in seen:
                    yield node
                seen.add(node)

    def __len__(self):
        """Just iterate and count stuff"""
        n = 0
        for th in self:
            n += 1
        return n

    def __getitem__(self, thing):
        """Check the cache first. If the key isn't there, try retrieving it
        from the database.

        """
        try:
            if thing in self._cache:
                return self._cache[thing]
        except AttributeError:
            pass
        myn = json_dump(self.name)
        thingn = json_dump(thing)
        for (branch, rev) in self.engine._active_branches():
            self.engine.cursor.execute(
                "SELECT things.thing, things.location FROM things JOIN ("
                "SELECT character, thing, branch, MAX(tick) AS tick "
                "FROM things "
                "WHERE character=? "
                "AND thing=? "
                "AND branch=? "
                "AND tick<=? "
                "GROUP BY character, thing, branch) AS hitick "
                "ON things.character=hitick.character "
                "AND things.thing=hitick.thing "
                "AND things.branch=hitick.branch "
                "AND things.tick=hitick.tick "
                "LEFT OUTER JOIN "
                "(SELECT nodes.graph, nodes.node, nodes.branch, "
                "nodes.rev, nodes.extant "
                "FROM nodes JOIN "
                "(SELECT graph, node, branch, MAX(rev) AS rev "
                "FROM nodes "
                "WHERE graph=? "
                "AND node=? "
                "AND branch=? "
                "AND rev<=? GROUP BY graph, node, branch) AS hirev ON "
                "nodes.graph=hirev.graph AND "
                "nodes.node=hirev.node AND "
                "nodes.branch=hirev.branch AND "
                "nodes.rev=hirev.rev) AS existence ON "
                "things.character=existence.graph AND "
                "things.thing=existence.node "
                "WHERE existence.extant;",
                (
                    myn,
                    thingn,
                    branch,
                    rev,
                    myn,
                    thingn,
                    branch,
                    rev
                )
            )
            for (th, l) in self.engine.cursor.fetchall():
                thing = json_load(th)
                loc = json_load(l)
                if not loc:
                    raise KeyError("Thing does not exist")
                r = Thing(self.character, thing)
                try:
                    self._cache[thing] = r
                except AttributeError:
                    pass
                return r
        raise KeyError("Thing does not exist")

    def __setitem__(self, thing, val):
        """Clear out any existing :class:`Thing` by this name and make a new
        one out of ``val`` (assumed to be a mapping of some kind)

        """
        th = Thing(self.character, thing)
        th.clear()
        th.exists = True
        th.update(val)
        self.character.changect += 1
        try:
            self._cache[thing] = th
        except AttributeError:
            pass

    def __delitem__(self, thing):
        """Delete the thing from the cache and the database"""
        if hasattr(self, '_cache') and thing in self._cache:
            th = self._cache[thing]
            del self._cache[thing]
        else:
            th = Thing(self.character, thing)
        th.exists = False
        th.clear()
        self.character.changect += 1
        assert(thing not in self)

    def __repr__(self):
        """Represent myself as a dict"""
        return repr(dict(self))


class CharacterPlaceMapping(MutableMapping, RuleFollower):
    """:class:`Place` objects that are in a :class:`Character`"""
    _book = "place"

    def __init__(self, character):
        """Store the character and initialize the cache (if caching)"""
        self.character = character
        self.engine = character.engine
        self.name = character.name
        if self.engine.caching:
            self._cache = {}

    def _things(self):
        """Private method. Return a set of names of things in the character."""
        things = set()
        things_seen = set()
        charn = json_dump(self.character.name)
        for (branch, rev) in self.engine._active_branches():
            self.engine.cursor.execute(
                "SELECT things.thing, things.location FROM things JOIN ("
                "SELECT character, thing, branch, MAX(tick) AS tick "
                "FROM things "
                "WHERE character=? "
                "AND branch=? "
                "AND tick<=? "
                "GROUP BY character, thing, branch) AS hitick ON "
                "things.character=hitick.character "
                "AND things.thing=hitick.thing "
                "AND things.branch=hitick.branch "
                "AND things.tick=hitick.tick;",
                (
                    charn,
                    branch,
                    rev
                )
            )
            for (th, l) in self.engine.cursor.fetchall():
                thing = json_load(th)
                loc = json_load(l)
                if thing not in things_seen and loc:
                    things.add(thing)
                things_seen.add(thing)
            return things

    def __iter__(self):
        """Iterate over names of places."""
        things = self._things()
        for node in self.engine._iternodes(self.character.name):
            if node not in things:
                yield node

    def __contains__(self, k):
        """Check the cache first, then query the database to see if ``k`` is
        the name of a place.

        """
        if hasattr(self, '_cache') and k in self._cache:
            return True
        for node in self.engine._iternodes(self.character.name):
            if node == k:
                return k not in self._things()
        return False

    def __len__(self):
        """Iterate and count"""
        n = 0
        for place in self:
            n += 1
        return n

    def __getitem__(self, place):
        """Get the place from the cache if I can, otherwise check that it
        exists, and if it does, cache and return it

        """
        if hasattr(self, '_cache') and place in self._cache:
            return self._cache[place]
        if place in self:
            r = Place(self.character, place)
            if hasattr(self, '_cache'):
                self._cache[place] = r
            return r
        raise KeyError("No such place")

    def __setitem__(self, place, v):
        """Wipe out any existing place by that name, and replace it with one
        described by ``v``

        """
        pl = Place(self.character, place)
        pl.clear()
        pl.exists = True
        pl.update(v)
        self.character.changect += 1
        if hasattr(self, '_cache'):
            self._cache[place] = pl

    def __delitem__(self, place):
        """Delete place from both cache and database"""
        if hasattr(self, '_cache') and place in self._cache:
            self._cache[place].clear()
            del self._cache[place]
            self.character.changect += 1
        else:
            Place(self.character, place).clear()

    def __repr__(self):
        """Represent myself as a dictionary"""
        return repr(dict(self))


class CharacterThingPlaceMapping(MutableMapping):
    """Replacement for gorm's GraphNodeMapping that does Place and Thing"""
    def __init__(self, character):
        """Store the character"""
        self.character = character
        self.engine = character.engine
        self.name = character.name

    def __iter__(self):
        """Iterate over all nodes presently existing"""
        seen = set()
        myn = json_dump(self.name)
        for (branch, rev) in self.engine._active_branches():
            self.engine.cursor.execute(
                "SELECT nodes.node, nodes.extant FROM nodes JOIN "
                "(SELECT graph, node, branch, MAX(rev) AS rev "
                "FROM nodes WHERE "
                "graph=? AND "
                "branch=? AND "
                "rev<=? GROUP BY graph, node, branch) AS hirev "
                "ON nodes.graph=hirev.graph "
                "AND nodes.node=hirev.node "
                "AND nodes.branch=hirev.branch "
                "AND nodes.rev=hirev.rev;",
                (
                    myn,
                    branch,
                    rev
                )
            )
            for (n, extant) in self.engine.cursor.fetchall():
                node = json_load(n)
                if extant and node not in seen:
                    yield node
                seen.add(node)

    def __len__(self):
        """Count nodes that exist"""
        n = 0
        for node in iter(self):
            n += 1
        return n

    def __getitem__(self, k):
        """Return a :class:`Thing` or :class:`Place` as appropriate"""
        if k in self.character.thing:
            return self.character.thing[k]
        else:
            try:
                return self.character.place[k]
            except KeyError:
                raise KeyError("No such Thing or Place in this Character")

    def __setitem__(self, k, v):
        """Assume you're trying to create a :class:`Place`"""
        self.character.place[k] = v

    def __delitem__(self, k):
        """Delete place or thing"""
        if k in self.character.place:
            del self.character.place[k]
        else:
            try:
                del self.character.thing[k]
            except KeyError:
                raise KeyError("No such Thing or Place in this Character")


class CharacterPortalSuccessorsMapping(GraphSuccessorsMapping, RuleFollower):
    _book = "portal"

    class Successors(GraphSuccessorsMapping.Successors):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.graph.engine.caching:
                self._cache = {}

        def __contains__(self, k):
            if hasattr(self, '_cache') and k in self._cache:
                return True
            return super().__contains__(k)

        def _getsub(self, nodeB):
            if hasattr(self, '_cache'):
                if nodeB not in self._cache:
                    self._cache[nodeB] = Portal(self.graph, self.nodeA, nodeB)
                return self._cache[nodeB]
            return Portal(self.graph, self.nodeA, nodeB)

        def __setitem__(self, nodeB, value):
            p = Portal(self.graph, self.nodeA, nodeB)
            p.clear()
            p.exists = True
            p.update(value)
            self.graph.changect += 1
            if '_paths' in self.graph.graph:
                del self.graph.graph['_paths']
                self.graph._paths = {}
            if hasattr(self, '_cache'):
                self._cache[nodeB] = p

        def __delitem__(self, nodeB):
            if hasattr(self, '_cache') and nodeB in self._cache:
                n = self._cache[nodeB]
                if not n.exists:
                    raise KeyError("No such node")
                n.clear()
                del self._cache[nodeB]
                self.graph.changect += 1
            else:
                super().__delitem__(nodeB)


class CharacterPortalPredecessorsMapping(
        DiGraphPredecessorsMapping,
        RuleFollower
):
    _book = "portal"

    class Predecessors(DiGraphPredecessorsMapping.Predecessors):
        def _getsub(self, nodeA):
            if nodeA in self.graph.portal:
                if (
                        self.graph.engine.caching and
                        self.nodeB not in self.graph.portal[nodeA]._cache
                ):
                    self.graph.portal[nodeA]._cache[self.nodeB] = Portal(
                        self.graph,
                        nodeA,
                        self.nodeB
                    )
                return self.graph.portal[nodeA][self.nodeB]
            return Portal(self.graph, nodeA, self.nodeB)

        def __setitem__(self, nodeA, value):
            if nodeA in self.graph.portal:
                if (
                        self.graph.engine.caching and
                        self.nodeB not in self.graph.portal[nodeA]._cache
                ):
                    self.graph.portal[nodeA]._cache[self.nodeB] = Portal(
                        self.graph,
                        nodeA,
                        self.nodeB
                    )
            p = self.graph.portal[nodeA][self.nodeB]
            p.clear()
            p.exists = True
            p.update(value)
            self.graph.changect += 1
            if '_paths' in self.graph.graph:
                del self.graph.graph['_paths']
                if hasattr(self.graph, '_paths'):
                    self.graph._paths = {}

        def __delitem__(self, nodeA):
            if (
                    self.graph.engine.caching and
                    nodeA in self.graph.portal and
                    self.nodeB in self.graph.portal[nodeA]
            ):
                n = self.graph.portal[nodeA]._cache[self.nodeB]
                n.clear()
                del self.graph.portal[nodeA].cache[self.nodeB]
            else:
                super().__delitem__(nodeA)
            self.graph.changect += 1


class CharacterAvatarGraphMapping(Mapping, RuleFollower):
    _book = "avatar"

    def __init__(self, char):
        """Remember my character"""
        self.character = char
        self.engine = char.engine
        self.name = char.name
        self._name = char._name

    def __call__(self, av):
        """Add the avatar. It must be an instance of Place or Thing."""
        if av.__class__ not in (Place, Thing):
            raise TypeError("Only Things and Places may be avatars")
        self.character.add_avatar(av.name, av.character.name)

    def _datadict(self):
        """Get avatar-ness data and return it"""
        d = {}
        for (branch, rev) in self.engine._active_branches():
            self.engine.cursor.execute(
                "SELECT "
                "avatars.avatar_graph, "
                "avatars.avatar_node, "
                "avatars.is_avatar FROM avatars "
                "JOIN ("
                "SELECT character_graph, avatar_graph, avatar_node, "
                "branch, MAX(tick) AS tick FROM avatars WHERE "
                "character_graph=? AND "
                "branch=? AND "
                "tick<=? GROUP BY character_graph, avatar_graph, "
                "avatar_node, branch) AS hitick ON "
                "avatars.character_graph=hitick.character_graph AND "
                "avatars.avatar_graph=hitick.avatar_graph AND "
                "avatars.avatar_node=hitick.avatar_node AND "
                "avatars.branch=hitick.branch AND "
                "avatars.tick=hitick.tick;",
                (
                    json_dump(self.name),
                    branch,
                    rev
                )
            )
            for (graph, node, avatar) in self.engine.cursor.fetchall():
                g = json_load(graph)
                n = json_load(node)
                is_avatar = bool(avatar)
                if g not in d:
                    d[g] = {}
                if n not in d[g]:
                    d[g][n] = is_avatar
        return d

    def __iter__(self):
        """Iterate over every avatar graph that has at least one avatar node
        in it presently

        """
        d = self._datadict()
        for graph in d:
            for node in d[graph]:
                if d[graph][node]:
                    yield graph
                    break

    def __len__(self):
        """Number of graphs in which I have an avatar"""
        n = 0
        for g in self:
            n += 1
        return n

    def __getitem__(self, g):
        """Get the CharacterAvatarMapping for the given graph, if I have any
        avatars in it.

        If I have avatars in only one graph, behave as a proxy to that
        graph's CharacterAvatarMapping.

        Unless I have only one avatar anywhere, in which case be a
        proxy to that.

        """
        d = self._datadict()
        if g in d:
            return self.CharacterAvatarMapping(self, g)
        elif len(d.keys()) == 1:
            avm = self.CharacterAvatarMapping(self, list(d.keys())[0])
            if len(avm.keys()) == 1:
                return avm[list(avm.keys())[0]][g]
            else:
                return avm[g]
        raise KeyError("No avatar in {}".format(g))

    def __getattr__(self, attr):
        """If I've got only one avatar, return its attribute"""
        d = self._datadict()
        if len(d.keys()) == 1:
            avs = self.CharacterAvatarMapping(self, list(d.keys())[0])
            if len(avs) == 1:
                av = list(avs.keys())[0]
                if attr == av:
                    return avs[attr]
                else:
                    return getattr(avs[list(avs.keys())[0]], attr)
        raise AttributeError

    def __repr__(self):
        """Represent myself like a dictionary"""
        d = {}
        for k in self:
            d[k] = dict(self[k])
        return repr(d)

    class CharacterAvatarMapping(Mapping):
        """Mapping of avatars of one Character in another Character."""
        def __init__(self, outer, graphn):
            """Store the character and the name of the "graph", ie. the other
            character.

            """
            self.character = outer.character
            self.engine = outer.engine
            self.name = outer.name
            self._name = outer._name
            self.graph = graphn
            self._graph = json_dump(graphn)

        def _branchdata(self, branch, rev):
            return self.engine.cursor.execute(
                "SELECT "
                "avatars.avatar_node, "
                "avatars.is_avatar FROM avatars JOIN ("
                "SELECT character_graph, avatar_graph, avatar_node, "
                "branch, MAX(tick) AS tick FROM avatars "
                "WHERE character_graph=? "
                "AND avatar_graph=? "
                "AND branch=? "
                "AND tick<=? GROUP BY "
                "character_graph, avatar_graph, avatar_node, branch"
                ") AS hitick ON "
                "avatars.character_graph=hitick.character_graph "
                "AND avatars.avatar_graph=hitick.avatar_graph "
                "AND avatars.avatar_node=hitick.avatar_node "
                "AND avatars.branch=hitick.branch "
                "AND avatars.tick=hitick.tick;",
                (
                    json_dump(self.name),
                    json_dump(self.graph),
                    branch,
                    rev
                )
            ).fetchall()

        def __getattr__(self, attrn):
            """If I don't have such an attribute, but I contain exactly one
            avatar, and *it* has the attribute, return the
            avatar's attribute.

            """
            seen = set()
            counted = 0
            for (branch, rev) in self.engine._active_branches():
                if counted > 1:
                    break
                for (node, extant) in self._branchdata(branch, rev):
                    if counted > 1:
                        break
                    n = json_load(node)
                    x = bool(extant)
                    if x and n not in seen:
                        counted += 1
                    seen.add(n)
            if counted == 1:
                node = self.engine.character[self.graph].node[seen.pop()]
                if hasattr(node, attrn):
                    return getattr(node, attrn)
            raise AttributeError("No such attribute: " + attrn)

        def __iter__(self):
            """Iterate over the names of all the presently existing nodes in the
            graph that are avatars of the character

            """
            seen = set()
            for (branch, rev) in self.engine._active_branches():
                for (node, extant) in self._branchdata(branch, rev):
                    n = json_load(node)
                    x = bool(extant)
                    if (
                            x and
                            n not in seen and
                            self.engine._node_exists(self.graph, n)
                    ):
                        yield n
                    seen.add(n)

        def __contains__(self, av):
            for (branch, rev) in self.engine._active_branches():
                self.engine.cursor.execute(
                    "SELECT avatars.is_avatar FROM avatars JOIN ("
                    "SELECT character_graph, avatar_graph, avatar_node, "
                    "branch, MAX(tick) AS tick FROM avatars "
                    "WHERE character_graph=? "
                    "AND avatar_graph=? "
                    "AND avatar_node=? "
                    "AND branch=? "
                    "AND tick<=? GROUP BY "
                    "character_graph, avatar_graph, avatar_node, "
                    "branch) AS hitick ON "
                    "avatars.character_graph=hitick.character_graph "
                    "AND avatars.avatar_graph=hitick.avatar_graph "
                    "AND avatars.avatar_node=hitick.avatar_node "
                    "AND avatars.branch=hitick.branch "
                    "AND avatars.tick=hitick.tick;",
                    (
                        json_dump(self.name),
                        json_dump(self.graph),
                        json_dump(av),
                        branch,
                        rev
                    )
                )
                try:
                    return (
                        self.engine.cursor.fetchone()[0] and
                        self.engine._node_exists(av)
                    )
                except (TypeError, IndexError):
                    continue
            return False

        def __len__(self):
            """Number of presently existing nodes in the graph that are avatars of
            the character"""
            n = 0
            for a in self:
                n += 1
            return n

        def __getitem__(self, av):
            """Return the Place or Thing by the given name in the graph, if it's
            my avatar and it exists.

            If I contain exactly *one* Place or Thing, and you're
            not trying to get it by its name, delegate to its
            __getitem__. It's common for one Character to have
            exactly one avatar in another Character, and when that
            happens, it's nice not to have to specify the avatar's
            name.

            """
            if av in self:
                return self.engine.character[self.graph].node[av]
            if len(self.keys()) == 1:
                k = list(self.keys())[0]
                return self.engine.character[self.graph].node[k]
            raise KeyError("No such avatar")

        def __repr__(self):
            """Represent myself like a dictionary"""
            d = {}
            for k in self:
                d[k] = dict(self[k])
            return repr(d)


class SenseFuncWrap(object):
    """Wrapper for a sense function that looks it up in the code store if
    provided with its name, and prefills the first two arguments.

    """
    def __init__(self, character, fun):
        """Store the character and the function, looking up the function if
        needed

        """
        self.character = character
        self.engine = character.engine
        if isinstance(fun, str):
            self.fun = self.engine.sense[fun]
        else:
            self.fun = fun
        if not isinstance(self.fun, Callable):
            raise TypeError("Function is not callable")

    def __call__(self, observed):
        """Call the function, prefilling the engine and observer arguments"""
        if isinstance(observed, str):
            observed = self.engine.character[observed]
        return self.fun(self.engine, self.character, Facade(observed))


class CharacterSense(object):
    """Mapping for when you've selected a sense for a character to use
    but haven't yet specified what character to look at

    """
    def __init__(self, container, sensename):
        """Store the container and the name of the sense"""
        self.container = container
        self.engine = self.container.engine
        self._sensename = json_dump(sensename)
        self.observer = self.container.character

    @property
    def sensename(self):
        return json_load(self._sensename)

    @property
    def func(self):
        """Return the function most recently associated with this sense"""
        for (branch, tick) in self.engine._active_branches():
            data = self.engine.cursor.execute(
                "SELECT function FROM senses JOIN "
                "(SELECT character, sense, branch, MAX(tick) AS tick "
                "FROM senses WHERE "
                "character=? AND "
                "sense=? AND "
                "branch=? AND "
                "tick<=? GROUP BY character, sense, branch) AS hitick "
                "ON senses.character=hitick.character "
                "AND senses.sense=hitick.sense "
                "AND senses.branch=hitick.branch "
                "AND senses.tick=hitick.tick;",
                (
                    self.observer._name,
                    self._sensename,
                    branch,
                    tick
                )
            ).fetchone()
            if data is None:
                continue
            return SenseFuncWrap(self.observer, data[0])

    def __call__(self, observed):
        """Call my sense function and make sure it returns the right type,
        then return that.

        """
        r = self.func(observed)
        if not (
                isinstance(r, Character) or
                isinstance(r, Facade)
        ):
            raise TypeError(
                "Sense function did not return a character-like object"
            )
        return r


class CharacterSenseMapping(MutableMapping, RuleFollower):
    """Used to view other Characters as seen by one, via a particular sense"""
    _book = "character"

    def __init__(self, character):
        """Store the character"""
        self.character = character
        self.engine = character.engine

    def __iter__(self):
        """Iterate over active sense names"""
        seen = set()
        for (branch, tick) in self.engine._active_branches():
            self.engine.cursor.execute(
                "SELECT sense, active FROM senses JOIN ("
                "SELECT character, sense, branch, MAX(tick) AS tick "
                "FROM senses WHERE "
                "(character='' OR character=?) "
                "AND branch=? "
                "AND tick<=? "
                "GROUP BY character, sense, branch) AS hitick "
                "ON senses.character=hitick.character "
                "AND senses.sense=hitick.sense "
                "AND senses.branch=hitick.branch "
                "AND senses.tick=hitick.tick;",
                (
                    json_dump(self.character.name),
                    branch,
                    tick
                )
            )
            for (sense, active) in self.engine.cursor.fetchall():
                if active and sense not in seen:
                    yield sense
                seen.add(sense)

    def __len__(self):
        """Count active senses"""
        n = 0
        for sense in iter(self):
            n += 1
        return n

    def __getitem__(self, k):
        """Get a :class:`CharacterSense` named ``k`` if it exists"""
        for (branch, tick) in self.engine._active_branches():
            data = self.engine.cursor.execute(
                "SELECT active FROM senses JOIN ("
                "SELECT character, sense, branch, MAX(tick) AS tick "
                "FROM senses WHERE "
                "character IN ('', ?) "
                "AND sense=? "
                "AND branch=? "
                "AND tick<=? "
                "GROUP BY character, sense, branch) AS hitick "
                "ON senses.character=hitick.character "
                "AND senses.sense=hitick.sense "
                "AND senses.branch=hitick.branch "
                "AND senses.tick=hitick.tick;",
                (
                    json_dump(self.character.name),
                    json_dump(k),
                    branch,
                    tick
                )
            ).fetchall()
            if len(data) == 0:
                continue
            elif len(data) > 1:
                raise ValueError("Silly data in senses table")
            else:
                return CharacterSense(self.character, k)
        raise KeyError("Sense isn't active or doesn't exist")

    def __setitem__(self, k, v):
        """Use the function for the sense from here on out"""
        if isinstance(v, str):
            funn = v
        else:
            funn = v.__name__
        if funn not in self.engine.sense:
            if not isinstance(v, Callable):
                raise TypeError("Not a function")
            self.engine.sense[funn] = v
        sense = json_dump(k)
        charn = self.character._name
        (branch, tick) = self.engine.time
        try:
            self.engine.cursor.execute(
                "INSERT INTO senses "
                "(character, sense, branch, tick, function, active) "
                "VALUES "
                "(?, ?, ?, ?, ?, ?);",
                (
                    charn,
                    sense,
                    branch,
                    tick,
                    funn,
                    True
                )
            )
        except IntegrityError:
            self.engine.cursor.execute(
                "UPDATE senses SET function=?, active=? "
                "WHERE character=? "
                "AND sense=? "
                "AND branch=? "
                "AND tick=?;",
                (
                    funn,
                    True,
                    charn,
                    sense,
                    branch,
                    tick
                )
            )

    def __delitem__(self, k):
        """Stop having the given sense"""
        (branch, tick) = self.engine.time
        charn = json_dump(self.character.name)
        sense = json_dump(k)
        try:
            self.engine.cursor.execute(
                "INSERT INTO senses "
                "(character, sense, branch, tick, active) "
                "VALUES "
                "(?, ?, ?, ?, ?);",
                (
                    charn,
                    sense,
                    branch,
                    tick,
                    False
                )
            )
        except IntegrityError:
            self.engine.cursor.execute(
                "UPDATE senses SET active=? WHERE "
                "character=? AND "
                "sense=? AND "
                "branch=? AND "
                "tick=?;",
                (
                    False,
                    charn,
                    sense,
                    branch,
                    tick
                )
            )

    def __call__(self, fun, name=None):
        """Decorate the function so it's mine now"""
        if not isinstance(fun, Callable):
            raise TypeError(
                "I need a function here"
            )
        if name is None:
            name = fun.__name__
        self[name] = fun


class FacadePlace(MutableMapping):
    @property
    def name(self):
        return self['name']

    def contents(self):
        for thing in self.facade.thing.values():
            if thing.container is self:
                yield thing

    def __init__(self, facade, real):
        self.facade = facade
        self._real = real
        self._patch = {}
        self._masked = set()

    def __iter__(self):
        seen = set()
        for k in self._real:
            if k not in self._masked:
                yield k
            seen.add(k)
        for k in self._patch:
            if (
                    k not in self._masked and
                    k not in seen
            ):
                yield k

    def __len__(self):
        n = 0
        for k in self:
            n += 1
        return n

    def __getitem__(self, k):
        if k in self._masked:
            raise KeyError("masked")
        if k in self._patch:
            return self._patch[k]
        return self._real[k]

    def __setitem__(self, k, v):
        self._masked.discard(k)
        self._patch[k] = v

    def __delitem__(self, k):
        self._masked.add(k)


class FacadeThing(FacadePlace):
    @property
    def location(self):
        try:
            return self.facade.node[self['location']]
        except KeyError:
            return None

    @property
    def next_location(self):
        try:
            return self.facade.node[self['next_location']]
        except KeyError:
            return None

    @property
    def container(self):
        try:
            return self.facade.portal[self['location']][
                self['next_location']]
        except KeyError:
            return self.location


class FacadePortal(FacadePlace):
    @property
    def origin(self):
        return self.facade.node[self['origin']]

    @property
    def destination(self):
        return self.facade.node[self['destination']]


class FacadeEntityMapping(MutableMapping):
    def __init__(self, facade):
        self.facade = facade
        self._patch = {}
        self._masked = set()

    def __contains__(self, k):
        return (
            k not in self._masked and (
                k in self._patch or
                k in self._get_inner_map()
            )
        )

    def __iter__(self):
        seen = set()
        for k in self._get_inner_map():
            if k not in self._masked:
                yield k
            seen.add(k)
        for k in self._patch:
            if k not in seen:
                yield k

    def __len__(self):
        n = 0
        for k in self:
            n += 1
        return n

    def __getitem__(self, k):
        if k in self._masked:
            raise KeyError("masked")
        if k in self._patch:
            return self._patch[k]
        return self.facadecls(self.facade, self._get_inner_map()[k])

    def __setitem__(self, k, v):
        if not isinstance(v, self.facadecls):
            if not isinstance(v, self.innercls):
                raise TypeError(
                    "Need :class:``Thing`` or :class:``FacadeThing``"
                )
            v = self.facadecls(self.facade, v)
        self._masked.discard(k)
        self._patch[k] = v

    def __delitem__(self, k):
        self._masked.add(k)


class FacadeThingMapping(FacadeEntityMapping):
    facadecls = FacadeThing
    innercls = Thing

    def _get_inner_map(self):
        return self.facade.character.thing


class FacadePlaceMapping(FacadeEntityMapping):
    facadecls = FacadePlace
    innercls = Place

    def _get_inner_map(self):
        return self.facade.character.place


class FacadePortalSuccessors(FacadeEntityMapping):
    facadecls = FacadePortal
    innercls = Portal

    def __init__(self, facade, origname):
        super().__init__(facade)
        self._origname = origname

    def _get_inner_map(self):
        return self.facade.character.portal[self._origname]


class FacadePortalPredecessors(FacadeEntityMapping):
    facadecls = FacadePortal
    innercls = Portal

    def __init__(self, facade, destname):
        super().__init__(facade)
        self._destname = destname

    def _get_inner_map(self):
        return self.facade.character.preportal[self._destname]


class FacadePortalMapping(FacadeEntityMapping):
    def __getitem__(self, node):
        if node in self._masked:
            raise KeyError("masked")
        if node in self._patch:
            return self._patch[node]
        return self.cls(self.facade, node)

    def __setitem__(self, node, value):
        self._masked.discard(node)
        v = self.cls(self.facade, node)
        v.update(value)
        self._patch[node] = v

    def __delitem__(self, node):
        self._masked.add(node)


class FacadePortalSuccessorsMapping(FacadePortalMapping):
    cls = FacadePortalSuccessors

    def _get_inner_map(self):
        return self.facade.character.portal


class FacadePortalPredecessorsMapping(FacadePortalMapping):
    cls = FacadePortalPredecessors

    def _get_inner_map(self):
        return self.facade.character.preportal


class FacadeStatsMapping(MutableMapping):
    def __init__(self, facade):
        self.facade = facade
        self._patch = {}
        self._masked = set()

    def __iter__(self):
        seen = set()
        for k in self.facade.graph:
            if k not in self._masked:
                yield k
            seen.add(k)
        for k in self._patch:
            if k not in seen:
                yield k

    def __len__(self):
        n = 0
        for k in self:
            n += 1
        return n

    def __contains__(self, k):
        if k in self._masked:
            return False
        return (
            k in self._patch or
            k in self.facade.graph
        )

    def __getitem__(self, k):
        if k in self._masked:
            raise KeyError("masked")
        if k in self._patch:
            return self._patch[k]
        return self.facade.graph[k]

    def __setitem__(self, k, v):
        self._masked.discard(k)
        self._patch[k] = v

    def __delitem__(self, k):
        self._masked.add(k)


class Facade(nx.DiGraph):
    def __init__(self, character):
        self.character = character
        self.thing = FacadeThingMapping(self)
        self.place = FacadePlaceMapping(self)
        self.node = CompositeDict(self.thing, self.place)
        self.portal = FacadePortalSuccessorsMapping(self)
        self.succ = self.edge = self.adj = self.portal
        self.preportal = FacadePortalPredecessorsMapping(self)
        self.pred = self.preportal
        self.graph = FacadeStatsMapping(self)


class CharStatCache(MutableMapping):
    """Caching dict-alike for character stats"""
    def __init__(self, char):
        """Store character, initialize cache"""
        self.character = char
        self.engine = char.engine
        self._real = char.graph
        self._cache = {}

    def __iter__(self):
        """Iterate over underlying keys"""
        return iter(self._real)

    def __len__(self):
        """Length of underlying graph"""
        return len(self._real)

    def __getitem__(self, k):
        """Use the cache if I can"""
        (branch, tick) = self.engine.time
        if k not in self._cache:
            self._cache[k] = {}
        if branch not in self._cache[k]:
            self._cache[k][branch] = {}
        d = self._cache[k][branch]
        if tick in d:
            return d[tick]
        else:
            try:
                return d[max(t for t in d if t < tick)]
            except ValueError:
                # This might cause issues when rewriting history,
                # but I think that adequate discipline with the
                # tracking of ticks-handled should take care of it
                d[tick] = self._real[k]
                return d[tick]

    def __setitem__(self, k, v):
        """Cache new value and set it the normal way"""
        (branch, tick) = self.engine.time
        if k not in self._cache:
            self._cache[k] = {}
        if branch not in self._cache[k]:
            self._cache[k][branch] = {}
        self._cache[k][branch][tick] = v
        self._real[k] = v
        self.__getitem__(k)
        self.character.changect += 1

    def __delitem__(self, k):
        """Clear the cached value and delete the normal way"""
        (branch, tick) = self.engine.time
        if branch in self._cache[k]:
            for staletick in list(
                    t for t in self._cache[k][branch]
                    if t < tick
            ):
                del self._cache[k][branch][staletick]
        del self._real[k]
        self.character.changect += 1


class Character(DiGraph, RuleFollower):
    """A graph that follows game rules and has a containment hierarchy.

    Nodes in a Character are subcategorized into Things and
    Places. Things have locations, and those locations may be Places
    or other Things. A Thing might also travel, in which case, though
    it will spend its travel time located in its origin node, it may
    spend some time contained by a Portal (i.e. an edge specialized
    for Character). If a Thing is not contained by a Portal, it's
    contained by whatever it's located in.

    """
    _book = "character"

    def __init__(self, engine, name, data=None, **attr):
        """Store engine and name, and set up mappings for Thing, Place, and
        Portal

        """
        super().__init__(engine.gorm, name, data, **attr)
        self.character = self
        (ct,) = engine.cursor.execute(
            "SELECT COUNT(*) FROM characters WHERE character=?;",
            (self._name,)
        ).fetchone()
        if ct == 0:
            d = {}
            for mapp in ('character', 'avatar', 'thing', 'place', 'portal'):
                if mapp + '_rulebook' in attr:
                    rulebook = attr[mapp + 'rulebook']
                    bookname = rulebook.name if isinstance(
                        rulebook,
                        RuleBook
                    ) else str(rulebook)
                    d[mapp] = bookname
                else:
                    d[mapp] = mapp + ":" + self._name
            engine.cursor.execute(
                "INSERT INTO characters "
                "(character, "
                "character_rulebook, "
                "avatar_rulebook, "
                "thing_rulebook, "
                "place_rulebook, "
                "portal_rulebook) "
                "VALUES (?, ?, ?, ?, ?, ?);",
                (
                    self._name,
                    d['character'],
                    d['avatar'],
                    d['thing'],
                    d['place'],
                    d['portal']
                )
            )
        self.engine = engine
        self.thing = CharacterThingMapping(self)
        self.place = CharacterPlaceMapping(self)
        self.node = CharacterThingPlaceMapping(self)
        self.portal = CharacterPortalSuccessorsMapping(self)
        self.adj = self.portal
        self.succ = self.adj
        self.preportal = CharacterPortalPredecessorsMapping(self)
        self.pred = self.preportal
        self.avatar = CharacterAvatarGraphMapping(self)
        self.sense = CharacterSenseMapping(self)
        self.travel_reqs = FunList(
            self.engine,
            self.engine.prereq,
            'travel_reqs',
            ['character'],
            [name],
            'reqs'
        )
        self.changect = 0
        self._copychanges = {}
        self._copies = {}
        if engine.caching:
            self.stat = CharStatCache(self)
        else:
            self.stat = self.graph
        self._portal_traits = set()
        if self.engine.caching:
            self._paths = (
                self.graph['_paths']
                if '_paths' in self.graph
                else {}
            )

    def travel_req(self, fun):
        """Decorator for tests that :class:`Thing`s have to pass before they
        can go thru :class:`Portal's

        """
        self.travel_reqs.append(fun)

    def cache_paths(self):
        """Calculate all shortest paths in me, and cache them, to avoid having
        to do real pathfinding later.

        The cache will be deleted when a Portal is added or removed,
        or when any trait that all Portals have is changed or deleted
        on any of them.

        """
        path_d = {}
        # one set of shortest paths for every trait that all Portals have
        self._portal_traits = set()
        for (o, d, port) in self.in_edges_iter(data=True):
            for trait in port:
                self._portal_traits.add(trait)
        for (o, d, port) in self.in_edges_iter(data=True):
            for trait in self._portal_traits:
                if trait not in port:
                    self._portal_traits.remove(trait)
        traits = self._portal_traits + set([None])
        for trait in traits:
            path_d[trait] = nx.shortest_path(self, weight=trait)
        if self.engine.caching:
            self._paths = path_d
        self.graph['_paths'] = path_d

    def add_place(self, name, **kwargs):
        """Create a new Place by the given name, and set its initial
        attributes based on the keyword arguments (if any).

        """
        super(Character, self).add_node(name, **kwargs)

    def add_places_from(self, seq):
        """Take a series of place names and add the lot."""
        super().add_nodes_from(seq)

    def add_thing(self, name, location, next_location=None, **kwargs):
        """Create a Thing, set its location and next_location (if provided),
        and set its initial attributes from the keyword arguments (if
        any).

        """
        super(Character, self).add_node(name, **kwargs)
        self.place2thing(name, location)
        assert(name in self.thing and name not in self.place)

    def add_things_from(self, seq):
        for tup in seq:
            name = tup[0]
            location = tup[1]
            next_loc = tup[2] if len(tup) > 2 else None
            kwargs = tup[3] if len(tup) > 3 else {}
            self.add_thing(name, location, next_loc, **kwargs)

    def place2thing(self, name, location, next_location=None):
        """Turn a Place into a Thing with the given location and (if provided)
        next_location. It will keep all its attached Portals.

        """
        (branch, tick) = self.engine.time
        if hasattr(self.place, '_cache') and name in self.place._cache:
            del self.place._cache[name]
        myname = json_dump(self.name)
        thingname = json_dump(name)
        locn = json_dump(location)
        nlocn = json_dump(next_location) if next_location else None
        try:
            self.engine.cursor.execute(
                "INSERT INTO things "
                "(character, thing, branch, tick, location, next_location) "
                "VALUES "
                "(?, ?, ?, ?, ?, ?);",
                (
                    myname,
                    thingname,
                    branch,
                    tick,
                    locn,
                    nlocn
                )
            )
        except IntegrityError:
            self.engine.cursor.execute(
                "UPDATE things SET location=?, next_location=? "
                "WHERE character=? "
                "AND thing=? "
                "AND branch=? "
                "AND tick=?;",
                (
                    locn,
                    nlocn,
                    myname,
                    thingname,
                    branch,
                    tick
                )
            )

    def thing2place(self, name):
        """Unset a Thing's location, and thus turn it into a Place."""
        self.place2thing(name, None)

    def add_portal(self, origin, destination, symmetrical=False, **kwargs):
        """Connect the origin to the destination with a :class:`Portal`.

        Keyword arguments are the :class:`Portal`'s
        attributes. Exception: if keyword ``symmetrical`` == ``True``,
        a mirror-:class:`Portal` will be placed in the opposite
        direction between the same nodes. It will always appear to
        have the placed :class:`Portal`'s stats, and any change to the
        mirror :class:`Portal`'s stats will affect the placed
        :class:`Portal`.

        """
        if origin.__class__ in (Place, Thing):
            origin = origin.name
        if destination.__class__ in (Place, Thing):
            destination = destination.name
        super(Character, self).add_edge(origin, destination, **kwargs)
        if symmetrical:
            self.add_portal(destination, origin, is_mirror=True)

    def add_portals_from(self, seq, symmetrical=False):
        """Take a sequence of (origin, destination) pairs and make a
        :class:`Portal` for each.

        Actually, triples are acceptable too, in which case the third
        item is a dictionary of stats for the new :class:`Portal`.

        If optional argument ``symmetrical`` is set to ``True``, all
        the :class:`Portal` instances will have a mirror portal going
        in the opposite direction, which will always have the same
        stats.

        """
        for tup in seq:
            orig = tup[0]
            dest = tup[1]
            kwargs = tup[2] if len(tup) > 2 else {}
            if symmetrical:
                kwargs['symmetrical'] = True
            self.add_portal(orig, dest, **kwargs)

    def add_avatar(self, host, name):
        """Start keeping track of a :class:`Thing` or :class:`Place` in a
        different :class:`Character`.

        """
        (branch, tick) = self.engine.time
        if isinstance(host, Character):
            host = host.name
        h = json_dump(host)
        n = json_dump(name)
        # This will create the node if it doesn't exist. Otherwise
        # it's redundant but harmless.
        try:
            self.engine.cursor.execute(
                "INSERT INTO nodes (graph, node, branch, rev, extant) "
                "VALUES (?, ?, ?, ?, ?);",
                (
                    h,
                    n,
                    branch,
                    tick,
                    True
                )
            )
        except IntegrityError:
            pass
        # Declare that the node is my avatar
        try:
            self.engine.cursor.execute(
                "INSERT INTO avatars ("
                "character_graph, avatar_graph, avatar_node, "
                "branch, tick, is_avatar"
                ") VALUES (?, ?, ?, ?, ?, ?);",
                (
                    self._name,
                    h,
                    n,
                    branch,
                    tick,
                    True
                )
            )
        except IntegrityError:
            self.engine.cursor.execute(
                "UPDATE avatars SET is_avatar=? WHERE "
                "character_graph=? AND "
                "avatar_graph=? AND "
                "avatar_node=? AND "
                "branch=? AND "
                "tick=?;",
                (
                    True,
                    self._name,
                    h,
                    n,
                    branch,
                    tick
                )
            )
        self.changect += 1

    def del_avatar(self, host, name):
        """Way to delete avatars for if you don't want to do it in the avatar
        mapping for some reason

        """
        h = json_dump(host)
        n = json_dump(name)
        (branch, tick) = self.engine.time
        try:
            self.engine.cursor.execute(
                "INSERT INTO avatars "
                "(character_graph, avatar_graph, avatar_node, "
                "branch, tick, is_avatar) "
                "VALUES (?, ?, ?, ?, ?, ?);",
                (
                    self._name,
                    h,
                    n,
                    branch,
                    tick,
                    False
                )
            )
        except IntegrityError:
            self.engine.cursor.execute(
                "UPDATE avatars SET is_avatar=? WHERE "
                "character_graph=? AND "
                "avatar_graph=? AND "
                "avatar_node=? AND "
                "branch=? AND "
                "tick=?;",
                (
                    False,
                    self._name,
                    h,
                    n,
                    branch,
                    tick
                )
            )
        self.changect += 1

    def iter_portals(self):
        """All portals"""
        for o in self.portal:
            for port in self.portal[o].values():
                yield port

    def iter_avatars(self):
        """Iterate over all my avatars, regardless of what character they are
        in.

        """
        seen = set()
        for (branch, tick) in self.engine._active_branches():
            data = self.engine.cursor.execute(
                "SELECT avatars.avatar_graph, avatars.avatar_node, is_avatar "
                "FROM avatars JOIN "
                "(SELECT character_graph, avatar_graph, avatar_node, branch, "
                "MAX(tick) AS tick FROM avatars "
                "WHERE character_graph=? "
                "AND branch=? "
                "AND tick<=? GROUP BY "
                "character_graph, avatar_graph, avatar_node, branch) "
                "AS hitick "
                "ON avatars.character_graph=hitick.character_graph "
                "AND avatars.avatar_graph=hitick.avatar_graph "
                "AND avatars.branch=hitick.branch "
                "AND avatars.tick=hitick.tick "
                "LEFT OUTER JOIN "
                "(SELECT nodes.graph, nodes.node, "
                "nodes.branch, nodes.rev, nodes.extant FROM nodes JOIN "
                "(SELECT graph, node, branch, MAX(rev) AS rev "
                "FROM nodes WHERE "
                "branch=? AND "
                "rev<=? GROUP BY graph, node, branch) AS hirev ON "
                "nodes.graph=hirev.graph AND "
                "nodes.node=hirev.node AND "
                "nodes.branch=hirev.branch AND "
                "nodes.rev=hirev.rev) AS existence ON "
                "avatars.avatar_graph=existence.graph AND "
                "avatars.avatar_node=existence.node WHERE "
                "existence.extant;",
                (
                    self._name,
                    branch,
                    tick,
                    branch,
                    tick
                )
            ).fetchall()
            for (g, n, a) in data:
                graphn = json_load(g)
                noden = json_load(n)
                is_avatar = bool(a)
                if (graphn, noden) not in seen and is_avatar:
                    yield self.engine.character[graphn].node[noden]
                seen.add((graphn, noden))
