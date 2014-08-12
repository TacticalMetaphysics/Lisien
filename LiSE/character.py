# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
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
    DiGraphPredecessorsMapping
)
from LiSE.util import path_len
from LiSE.rule import Rule
from LiSE.funlist import FunList


class TravelException(Exception):
    """Exception for problems with pathfinding"""
    def __init__(self, message, path=None, followed=None, traveller=None, branch=None, tick=None, lastplace=None):
        """Store the message as usual, and also the optional arguments:

        ``path``: a list of Place names to show such a path as you found

        ``followed``: the portion of the path actually followed

        ``traveller``: the Thing doing the travelling

        ``branch``: branch during travel

        ``tick``: tick at time of error (might not be the tick at the time this exception is raised)

        ``lastplace``: where the traveller was, when the error happened

        """
        self.path = path
        self.followed = followed
        self.traveller = traveller
        self.branch = branch
        self.tick = tick
        self.lastplace = lastplace
        super().__init__(message)


class EntityImage(dict):
    def __init__(self, entity, branch, tick):
        self.name = entity.name
        self.branch = branch
        self.tick = tick
        self.time = (branch, tick)
        self.update(entity)

    def _copy(self, character, obj):
        if isinstance(obj, Thing):
            return ThingImage(obj, self.branch, self.tick)
        elif isinstance(obj, Place):
            return PlaceImage(character, obj, self.branch, self.tick)
        elif isinstance(obj, Portal):
            return PortalImage(character, obj, self.branch, self.tick)
        elif obj is None:
            return None
        else:
            return obj.copy()


class PlaceImage(EntityImage):
    def __init__(self, character, thingplace, branch, tick):
        super().__init__(thingplace, branch, tick)
        if isinstance(character, Character):
            self.portals = [self._copy(character, port) for port in thingplace.portals()]
            self.preportals = [self._copy(character, port) for port in thingplace.preportals()]


class ThingImage(PlaceImage):
    """How a Thing appeared at a given game-time"""
    def __init__(self, character, thing, branch, tick):
        super().__init__(thing, branch, tick)
        if isinstance(character, Character):
            self.container = self._copy(character, thing.container)
            self.location = self._copy(character, thing.location)
            self.next_location = self._copy(character, thing.next_location)


class PortalImage(EntityImage):
    def __init__(self, character, portal, branch, tick):
        super().__init__(portal, branch, tick)
        if isinstance(character, Character):
            self.origin = self._copy(character, portal.origin)
            self.destination = self._copy(character, portal.destination)
            self.reciprocal = self._copy(character, portal.reciprocal)


class CharacterImage(nx.DiGraph):
    def __init__(self, character, branch, tick):
        super().__init__(self, data=character)
        self.branch = branch
        self.tick = tick
        self.character = character
        self.place = {}
        for place in character.place.values():
            pli = PlaceImage(self, place, branch, tick)
            pli.contents = []
        self.portal = {}
        self.preportal = {}
        for o in character.portal:
            if o not in self.portal:
                self.portal[o] = {}
            for (d, portal) in character.portal[o].items():
                if d not in self.preportal:
                    self.preportal[d] = {}
                cp = PortalImage(self, portal, branch, tick)
                cp.origin = self.place[portal['origin']]
                cp.destination = self.place[portal['destination']]
                cp.contents = []
                self.portal[o][d] = cp
                self.preportal[d][o] = cp
        self.thing = {}
        for thing in character.thing.values():
            thi = ThingImage(self, thing, branch, tick)
            (locn1, locn2) = thing['locations']
            thi.contents = []
            self.thing[thing.name] = thi
        for thi in self.thing.values():
            if thi['location'] in self.place:
                thi.location = self.place[thi['location']]
            elif thi['location'] in self.thing:
                thi.location = self.thing[thi['location']]
            else:
                raise ValueError("Invalid location for thing")
            if thi['next_location'] is None:
                thi.next_location = None
                thi.container = thi.location
            elif thi['next_location'] in self.place:
                thi.next_location = self.place['next_location']
                thi.container = self.portal[thi['location']][thi['next_location']]
            elif thi['location'] in self.thing:
                thi.next_location = self.thing['next_location']
                thi.container = self.portal[thi['location']][thi['next_location']]
            else:
                raise ValueError("Invalid next_location for thing")
            thi.container.contents.append(thi)


class ThingPlace(GraphNodeMapping.Node):
    def __getitem__(self, name):
        """For when I'm the only avatar in a Character, and so you don't need
        my name to select me, but a name was provided anyway.

        """
        if name == self.name:
            return self
        return super().__getitem__(name)

    def __setitem__(self, k, v):
        if k == "name":
            raise KeyError("Can't set name")
        elif k == "character":
            raise KeyError("Can't set character")
        super().__setitem__(k, v)

    def _contents_names(self):
        things_seen = set()
        for (branch, tick) in self.gorm._active_branches():
            self.gorm.cursor.execute(
                "SELECT things.node FROM things JOIN ("
                "SELECT graph, node, branch, MAX(tick) AS rev FROM things "
                "WHERE graph=? "
                "AND branch=? "
                "AND tick<=? "
                "GROUP BY graph, node, branch) AS hitick "
                "ON things.graph=hitick.graph "
                "AND things.node=hitick.node "
                "AND things.branch=hitick.branch "
                "AND things.tick=hitick.tick "
                "WHERE things.location=?;",
                (
                    self.character.name,
                    branch,
                    tick,
                    self.name
                )
            )
            for (thing,) in self.gorm.cursor.fetchall():
                if thing not in things_seen:
                    yield thing
                things_seen.add(thing)

    def _portal_dests(self):
        seen = set()
        for (branch, tick) in self.gorm._active_branches():
            self.gorm.cursor.execute(
                "SELECT edges.nodeB, edges.extant FROM edges JOIN "
                "(SELECT graph, nodeA, nodeB, idx, branch, MAX(rev) AS rev FROM edges "
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
                    self.character.name,
                    self.name,
                    branch,
                    tick
                )
            )
            for (dest, exists) in self.gorm.cursor.fetchall():
                if exists and dest not in seen:
                    yield dest
                seen.add(dest)

    def _portal_origs(self):
        seen = set()
        for (branch, tick) in self.gorm._active_branches():
            self.gorm.cursor.execute(
                "SELECT edges.nodeA, edges.extant FROM edges JOIN "
                "(SELECT graph, nodeA, nodeB, idx, branch, MAX(rev) AS rev FROM edges "
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
                    self.character.name,
                    self.name,
                    branch,
                    tick
                )
            )
            for (orig, exists) in self.gorm.cursor.fetchall():
                if exists and orig not in seen:
                    yield orig
                seen.add(orig)

    def _user_names(self):
        seen = set()
        for (branch, tick) in self.engine._active_branches():
            self.engine.cursor.execute(
                "SELECT avatars.avatar_graph FROM avatars JOIN ("
                "SELECT character_graph, avatar_graph, avatar_node, branch, MAX(tick) AS tick "
                "FROM avatars WHERE "
                "avatar_graph=? AND "
                "avatar_node=? AND "
                "branch=? AND "
                "tick<=? GROUP BY "
                "character_graph, avatar_graph, avatar_node, branch) AS hitick "
                "ON avatars.character_graph=hitick.character_graph "
                "AND avatars.avatar_graph=hitick.avatar_graph "
                "AND avatars.avatar_node=hitick.avatar_node "
                "AND avatars.branch=hitick.branch "
                "AND avatars.tick=hitick.tick;",
                (
                    self.character.name,
                    self.name,
                    branch,
                    tick
                )
            )
            for row in self.engine.cursor.fetchall():
                charn = row[0]
                if charn not in seen:
                    yield charn
                    seen.add(charn)

    def users(self):
        """Iterate over characters this is an avatar of. Usually there will only be one.

        """
        for charn in self._user_names():
            yield self.engine.character[charn]

    def contents(self):
        """Iterate over the Things that are located here."""
        for thingn in self._contents_names():
            yield self.character.thing[thingn]

    def portals(self):
        for destn in self._portal_dests():
            yield self.character.portal[self.name][destn]

    def preportals(self):
        for orign in self._portal_origs():
            yield self.character.preportal[self.name][orign]


class Thing(ThingPlace):
    """The sort of item that has a particular location at any given time.

    If a Thing is in a Place, it is standing still. If it is in a
    Portal, it is moving through that Portal however fast it must in
    order to arrive at the other end when it is scheduled to. If it is
    in another Thing, then it is wherever that is, and moving the
    same.

    """
    def __init__(self, character, name):
        """Initialize a Thing in a Character with a name"""
        self.character = character
        self.engine = character.engine
        self.name = name
        super().__init__(character, name)

    def __iter__(self):
        # I'm only going to iterate over *some* of the special keys
        # implemented in __getitem__, the ones that are also writable
        # in __setitem__. This is to make it easy to copy a Thing as
        # though it's an ordinary Node.
        for extrakey in (
                'name',
                'character',
                'location',
                'next_location',
                'arrival_time',
                'next_arrival_time'
        ):
            yield extrakey
        for key in iter(super()):
            yield key

    def __getitem__(self, key):
        """Return one of my attributes stored in the database, with a few special exceptions:

        ``name``: return the name that uniquely identifies me within my Character

        ``character``: return the name of my character

        ``location``: return the name of my location

        ``arrival_time``: return the tick when I arrived in the present location

        ``next_location``: if I'm in transit, return where to, else return None

        ``next_arrival_time``: return the tick when I'm going to arrive at ``next_location``

        ``locations``: return a pair of (``location``, ``next_location``)

        """
        if key == 'name':
            return self.name
        elif key == 'character':
            return self.character.name
        elif key == 'location':
            return self._loc_and_next()[0]
        elif key == 'arrival_time':
            curloc = self['location']
            for (branch, tick) in self.gorm._active_branches():
                self.gorm.cursor.execute(
                    "SELECT MAX(tick) FROM things "
                    "WHERE character=? "
                    "AND thing=? "
                    "AND location=? "
                    "AND branch=? "
                    "AND tick<=?;",
                    (
                        self["character"],
                        self.name,
                        curloc,
                        branch,
                        tick
                    )
                )
                data = self.engine.orm.cursor.fetchall()
                if len(data) == 0:
                    continue
                elif len(data) > 1:
                    raise ValueError("How do you get more than one record from that?")
                else:
                    return data[0]
            raise ValueError("I don't seem to have arrived where I am?")
        elif key == 'next_location':
            return self._loc_and_next()[1]
        elif key == 'next_arrival_time':
            nextloc = self['next_location']
            if nextloc is None:
                return None
            for (branch, tick) in self.engine.orm._active_branches():
                self.engine.orm.cursor.execute(
                    "SELECT MIN(tick) FROM things "
                    "WHERE character=? "
                    "AND thing=? "
                    "AND location=? "
                    "AND branch=? "
                    "AND tick>?;",
                    (
                        self["character"],
                        self.name,
                        nextloc,
                        branch,
                        tick
                    )
                )
                data = self.engine.orm.cursor.fetchall()
                if len(data) == 0:
                    continue
                elif len(data) > 1:
                    raise ValueError("How do you get more than one record from that?")
                else:
                    return data[0]
        elif key == 'locations':
            return self._loc_and_next()
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        """Set ``key``==``value`` for the present game-time."""
        if key == 'name':
            raise ValueError("Can't change names")
        elif key == 'character':
            raise ValueError("Can't change characters")
        elif key == 'location':
            self._set_loc_and_next(value, self['next_location'])
        elif key == 'arrival_time':
            raise ValueError("Read-only")
        elif key == 'next_location':
            self._set_loc_and_next(self['location'], value)
        elif key == 'next_arrival_time':
            raise ValueError("Read-only")
        elif key == 'locations':
            self._set_loc_and_next(*value)
        else:
            super().__setitem__(key, value)

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
        """The Thing or Place I'm in. If I'm in transit, it's where I started."""
        locn = self['location']
        try:
            return self.character.thing[locn]
        except KeyError:
            return self.character.place[locn]

    @property
    def next_location(self):
        """If I'm not in transit, this is None. If I am, it's where I'm headed."""
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
                "SELECT character, thing, branch, MAX(tick) AS tick FROM things "
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
                    self.character.name,
                    self.name,
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
                return data[0]
        raise ValueError("No location set")

    def _set_loc_and_next(self, loc, nextloc):
        """Private method to simultaneously set ``location`` and ``next_location``"""
        (branch, tick) = self.character.engine.time
        self.character.engine.cursor.execute(
            "DELETE FROM things WHERE "
            "character=? AND "
            "thing=? AND "
            "branch=? AND "
            "tick=?;",
            (
                self.character.name,
                self.name,
                branch,
                tick
            )
        )
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
                self.character.name,
                self.name,
                branch,
                tick,
                loc,
                nextloc
            )
        )

    def copy(self):
        (branch, tick) = self.engine.time
        return ThingImage(self.character, self, branch, tick)

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

    def follow_path(self, path, weight=''):
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
            place = place.name if hasattr(place, 'name') else place
            if (
                    prevplace not in self.character.portal or
                    place not in self.character.portal[prevplace]
            ):
                raise TravelException(
                    "Couldn't follow portal from {} to {}".format(
                        prevplace.name,
                        place.name
                    ),
                    path=subpath,
                    traveller=self
                )
            subpath.append(place)
        ticks_total = 0
        prevsubplace = subpath.pop(0)
        subsubpath = [prevsubplace]
        for subplace in subpath:
            if prevsubplace != self["location"]:
                l = self["location"]
                fintick = self.character.engine.tick
                self.character.engine.tick = curtick
                raise TravelException(
                    "When I tried traveling to {}, at tick {}, I ended up at {}".format(
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
            self.character.engine.tick += tick_inc
            if self["location"] not in (subplace, prevsubplace):
                l = self["location"]
                fintick = self.character.engine.tick
                self.character.engine.tick = curtick
                raise TravelException(
                    "I couldn't go to {} at tick {} because I was in {}".format(
                        subplace,
                        fintick,
                        l
                    ),
                    path=subpath,
                    followed=subsubpath,
                    traveller=self,
                    branch=self.character.engine.branch,
                    tick=fintick,
                    lastplace=l
                )
            self.character.engine.tick -= tick_inc
            self.go_to_place(subplace, weight)
            self.character.engine.tick += tick_inc
            ticks_total += tick_inc
            subsubpath.append(subplace)
            prevsubplace = subplace
        self.character.engine.tick = curtick
        return ticks_total

    def travel_to(self, dest, weight='', graph=None):
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
        path = nx.shortest_path(graph, self["location"], destn, weight)
        return self.follow_path(path, weight)

    def travel_to_by(self, dest, arrival_tick, weight='', graph=None):
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


class Place(ThingPlace):
    """The kind of node where a Thing might ultimately be located."""
    def __init__(self, character, name):
        """Initialize a place in a character by a name"""
        self.character = character
        self.engine = character.engine
        self.name = name
        super(Place, self).__init__(character, name)

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
        elif key == 'contents':
            return self.contents_names()
        else:
            return super(Place, self).__getitem__(key)

    def copy(self):
        (branch, tick) = self.engine.time
        return PlaceImage(self.character, self, branch, tick)


class Portal(GraphEdgeMapping.Edge):
    """Connection between two Places that Things may travel along.

    Portals are one-way, but you can make one appear two-way by
    setting the ``symmetrical`` key to ``True``,
    eg. ``character.add_portal(orig, dest, symmetrical=True)``

    """
    def __init__(self, character, origin, destination):
        """Initialize a Portal in a character from an origin to a destination"""
        self._origin = origin
        self._destination = destination
        self.character = character
        self.engine = character.engine
        super().__init__(character, self._origin, self._destination)

    def __getitem__(self, key):
        """Get the present value of the key.

        If I am a mirror of another Portal, return the value from that Portal instead.
        
        """
        if key == 'origin':
            return self._origin
        elif key == 'destination':
            return self._destination
        elif key == 'character':
            return self.character.name
        elif key == 'is_mirror':
            return super().__getitem__(key)
        elif 'is_mirror' in self and super().__getitem__('is_mirror'):
            return self.character.preportal[self._origin][self._destination][key]
        else:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        """Set ``key``=``value`` at the present game-time.

        If I am a mirror of another Portal, set ``key``==``value`` on
        that Portal instead.

        """
        if key in ('origin', 'destination', 'character'):
            raise KeyError("Can't change " + key)
        if 'is_mirror' in self and super().__getitem__('is_mirror'):
            self.reciprocal[key] = value
            return
        elif key == 'symmetrical' and value:
            if (
                    self._destination not in self.character.portal or
                    self._origin not in self.character.portal[self._destination]
            ):
                self.character.add_portal(self._destination, self._origin)
                self.character.portal[self._destination][self._origin]["is_mirror"] = True
        elif key == 'symmetrical' and not value:
            try:
                self.character.portal[self._destination][self._origin]["is_mirror"] = False
            except KeyError:
                pass
        super().__setitem__(key, value)

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

    def _contents_names(self):
        """Private method to iterate over the names of the Things that are
        travelling along me at the present."""
        r = set()
        for (branch, tick) in self.gorm._active_branches():
            self.gorm.cursor.execute(
                "SELECT things.node FROM things JOIN ("
                "SELECT graph, node, branch, MAX(tick) AS tick "
                "FROM things WHERE "
                "graph=? AND "
                "branch=? AND "
                "tick<=? "
                "GROUP BY graph, node, branch) AS hitick "
                "ON things.graph=hitick.graph "
                "AND things.node=hitick.node "
                "AND things.branch=hitick.branch "
                "AND things.tick=hitick.tick "
                "WHERE location=? "
                "AND next_location=?;",
                (
                    self.character.name,
                    branch,
                    tick,
                    self['origin'],
                    self['destination']
                )
            )
            for (thing,) in self.gorm.cursor.fetchall():
                r.add(thing)
        return r

    def contents(self):
        """Iterate over Thing instances that are presently travelling through
        me.

        """
        for thingn in self._contents_names():
            yield self.character.thing[thingn]

    def update(self, d):
        """Works like regular update, but only actually updates when the new
        value and the old value differ. This is necessary to prevent
        certain infinite loops.

        """
        for (k, v) in d.items():
            if self[k] != v:
                self[k] = v

    def copy(self):
        (branch, tick) = self.engine.time
        return PortalImage(self.character, self, branch, tick)


class CharacterThingMapping(MutableMapping):
    def __init__(self, character):
        self.character = character
        self.engine = character.engine
        self.name = character.name

    def __iter__(self):
        """Iterate over nodes that have locations, and are therefore
        Things. Yield their names.

        """
        seen = set()
        for (branch, tick) in self.engine._active_branches():
            self.engine.cursor.execute(
                "SELECT things.thing, things.location FROM things JOIN ("
                "SELECT character, thing, branch, MAX(tick) AS tick FROM things "
                "WHERE character=? "
                "AND branch=? "
                "AND tick<=? "
                "GROUP BY character, thing, branch) AS hitick "
                "ON things.character=hitick.character "
                "AND things.thing=hitick.thing "
                "AND things.branch=hitick.branch "
                "AND things.tick=hitick.tick;",
                (
                    self.name,
                    branch,
                    tick
                )
            )
            for (node, loc) in self.engine.cursor.fetchall():
                if loc and node not in seen:
                    yield node
                seen.add(node)

    def __len__(self):
        n = 0
        for th in self:
            n += 1
        return n

    def __getitem__(self, thing):
        for (branch, rev) in self.engine._active_branches():
            self.engine.cursor.execute(
                "SELECT things.thing, things.location FROM things JOIN ("
                "SELECT character, thing, branch, MAX(tick) AS tick FROM things "
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
                    self.name,
                    thing,
                    branch,
                    rev
                )
            )
            for (thing, loc) in self.engine.cursor.fetchall():
                if not loc:
                    raise KeyError("Thing doesn't exist right now")
                return Thing(self.character, thing)
        raise KeyError("Thing has never existed")

    def __setitem__(self, thing, val):
        th = Thing(self.character, thing)
        th.clear()
        th.exists = True
        th.update(val)

    def __delitem__(self, thing):
        Thing(self.character, thing).clear()

    def __repr__(self):
        return repr(dict(self))


class CharacterPlaceMapping(MutableMapping):
    def __init__(self, character):
        self.character = character
        self.engine = character.engine
        self.name = character.name

    def __iter__(self):
        things = set()
        things_seen = set()
        for (branch, rev) in self.engine._active_branches():
            self.engine.cursor.execute(
                "SELECT things.thing, things.location FROM things JOIN ("
                "SELECT character, thing, branch, MAX(tick) AS tick FROM things "
                "WHERE character=? "
                "AND branch=? "
                "AND tick<=? "
                "GROUP BY character, thing, branch) AS hitick ON "
                "things.character=hitick.character "
                "AND things.thing=hitick.thing "
                "AND things.branch=hitick.branch "
                "AND things.tick=hitick.tick;",
                (
                    self.character.name,
                    branch,
                    rev
                )
            )
            for (thing, loc) in self.engine.cursor.fetchall():
                if thing not in things_seen and loc:
                    things.add(thing)
                things_seen.add(thing)
        for node in self.engine._iternodes(self.character.name):
            if node not in things:
                yield node

    def __len__(self):
        n = 0
        for place in self:
            n += 1
        return n

    def __getitem__(self, place):
        if place in iter(self):
            return Place(self.character, place)
        raise KeyError("No such place")

    def __setitem__(self, place, v):
        pl = Place(self.character, place)
        pl.clear()
        pl.exists = True
        pl.update(v)

    def __delitem__(self, place):
        Place(self.character, place).clear()

    def __repr__(self):
        return repr(dict(self))


class CharacterPortalSuccessorsMapping(GraphSuccessorsMapping):
    class Successors(GraphSuccessorsMapping.Successors):
        def _getsub(self, nodeB):
            return Portal(self.graph, self.nodeA, nodeB)

        def __setitem__(self, nodeB, value):
            p = Portal(self.graph, self.nodeA, nodeB)
            p.clear()
            p.exists = True
            p.update(value)


class CharacterPortalPredecessorsMapping(DiGraphPredecessorsMapping):
    class Predecessors(DiGraphPredecessorsMapping.Predecessors):
        def _getsub(self, nodeA):
            return Portal(self.graph, nodeA, self.nodeB)

        def __setitem__(self, nodeA, value):
            p = Portal(self.graph, nodeA, self.nodeB)
            p.clear()
            p.exists = True
            p.update(value)


class CharRules(MutableMapping):
    """Maps rule names to rules the Character is following, and is also a
    decorator to create said rules from action functions.

    Decorating a function with this turns the function into the
    first action of a new rule of the same name, and applies the
    rule to the character. Add more actions with the @rule.action
    decorator, and add prerequisites with @rule.prereq

    """
    def __init__(self, char):
        """Store the character"""
        self.character = char
        self.engine = char.engine
        self.name = char.name

    def __call__(self, v):
        """If passed a Rule, activate it. If passed a string, get the rule by
        that name and activate it. If passed a function (probably
        because I've been used as a decorator), make a rule with the
        same name as the function, with the function itself being the
        first action of the rule, and activate that rule.

        """
        if isinstance(v, Rule):
            self._activate_rule(v)
        elif isinstance(v, Callable):
            # create a new rule performing the action v
            vname = self.engine.function(v)
            rule = Rule(
                    self.engine,
                    vname
            )
            rule.actions.append(vname)
            self._activate_rule(rule)
        else:
            # v is the name of a rule. Maybe it's been created
            # previously or maybe it'll get initialized in Rule's
            # __init__.
            self._activate_rule(Rule(self.engine, v))

    def __iter__(self):
        """Iterate over all rules presently in effect"""
        seen = set()
        for (branch, tick) in self.engine._active_branches():
            self.engine.cursor.execute(
                "SELECT char_rules.rule, char_rules.active "
                "FROM char_rules JOIN ("
                "SELECT character, rule, branch, MAX(tick) AS tick "
                "FROM char_rules WHERE "
                "character=? AND "
                "branch=? AND "
                "tick<=? GROUP BY character, rule, branch) AS hitick "
                "ON char_rules.character=hitick.character "
                "AND char_rules.rule=hitick.rule "
                "AND char_rules.character=hitick.character "
                "AND char_rules.branch=hitick.branch "
                "AND char_rules.tick=hitick.tick;",
                (
                    self.character.name,
                    branch,
                    tick
                )
            )
            for (rule, active) in self.engine.cursor.fetchall():
                if active and rule not in seen:
                    yield rule
                seen.add(rule)

    def __len__(self):
        """Count the rules presently in effect"""
        n = 0
        for rule in self:
            n += 1
        return n

    def __getitem__(self, rulen):
        """Get the rule by the given name, if it is in effect"""
        # make sure the rule is active at the moment
        for (branch, tick) in self.engine._active_branches():
            self.engine.cursor.execute(
                "SELECT char_rules.active "
                "FROM char_rules JOIN ("
                "SELECT character, rule, branch, MAX(tick) AS tick "
                "FROM char_rules WHERE "
                "character=? AND "
                "rule=? AND "
                "branch=? AND "
                "tick<=? GROUP BY character, rule, branch) AS hitick "
                "ON char_rules.character=hitick.character "
                "AND char_rules.rule=hitick.rule "
                "AND char_rules.branch=hitick.branch "
                "AND char_rules.tick=hitick.tick;",
                (
                    self.character.name,
                    rulen,
                    branch,
                    tick
                )
            )
            data = self.engine.cursor.fetchall()
            if len(data) == 0:
                continue
            elif len(data) > 1:
                raise ValueError("Silly data in char_rules table")
            else:
                (active,) = data[0]
                if not active:
                    raise KeyError("No such rule at the moment")
                return Rule(self.engine, rulen)
        raise KeyError("No such rule, ever")

    def __setitem__(self, k, v):
        oldn = v.__name__
        v.__name__ = k
        self(v)
        v.__name__ = oldn

    def __getattr__(self, attrn):
        """For easy use with decorators, allow accessing my contents like
        attributes

        """
        try:
            return self[attrn]
        except KeyError:
            raise AttributeError

    def _activate_rule(self, rule):
        """Indicate that the rule is active and should be followed.

        """
        (branch, tick) = self.engine.time
        try:
            self.engine.cursor.execute(
                "INSERT INTO char_rules "
                "(character, rule, branch, tick, active) "
                "VALUES (?, ?, ?, ?, ?);",
                (
                    self.character.name,
                    rule.name,
                    branch,
                    tick,
                    True
                )
            )
        except IntegrityError:
            self.engine.cursor.execute(
                "UPDATE char_rules SET active=1 WHERE "
                "character=? AND "
                "rule=? AND "
                "branch=? AND "
                "tick=?;",
                (
                    self.character.name,
                    rule.name,
                    branch,
                    tick
                )
            )

    def __delitem__(self, rulen):
        """Deactivate the rule"""
        (branch, tick) = self.engine.time
        try:
            self.engine.cursor.execute(
                "INSERT INTO char_rules "
                "(character, rule, branch, tick, active) "
                "VALUES (?, ?, ?, ?, ?);",
                (
                    self.name,
                    rulen,
                    branch,
                    tick,
                    False
                )
            )
        except IntegrityError:
            self.engine.cursor.execute(
                "UPDATE char_rules SET active=? "
                "WHERE character=? "
                "AND rule=? "
                "AND branch=? "
                "AND tick=?;",
                (
                    False,
                    self.name,
                    rulen,
                    branch,
                    tick
                )
            )


class CharacterAvatarGraphMapping(Mapping):
    def __init__(self, char):
        """Remember my character"""
        self.char = char
        self.engine = char.engine
        self.name = char.name

    def __call__(self, av):
        """Add the avatar. It must be an instance of Place or Thing."""
        if av.__class__ not in (Place, Thing):
            raise TypeError("Only Things and Places may be avatars")
        self.char.add_avatar(av.name, av.character.name)

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
                    self.name,
                    branch,
                    rev
                )
            )
            for (graph, node, avatar) in self.engine.cursor.fetchall():
                is_avatar = bool(avatar)
                if graph not in d:
                    d[graph] = {}
                if node not in d[graph]:
                    d[graph][node] = is_avatar
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
        avatars in it. Otherwise raise KeyError.

        """
        d = self._datadict()[g]
        for node in d:
            if d[node]:
                return self.CharacterAvatarMapping(self, g)
        raise KeyError("No avatars in {}".fengineat(g))

    def __repr__(self):
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
            self.char = outer.char
            self.engine = outer.engine
            self.name = outer.name
            self.graph = graphn

        def __getattr__(self, attrn):
            """If I don't have such an attribute, but I contain exactly one
            avatar, and *it* has the attribute, return the
            avatar's attribute.

            """
            if len(self) == 1:
                return getattr(self[next(iter(self))], attrn)
            return super(Character.CharacterAvatarMapping, self).__getattr__(attrn)

        def __iter__(self):
            """Iterate over the names of all the presently existing nodes in the
            graph that are avatars of the character

            """
            seen = set()
            for (branch, rev) in self.engine._active_branches():
                self.engine.cursor.execute(
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
                        self.name,
                        self.graph,
                        branch,
                        rev
                    )
                )
                for (node, extant) in self.engine.cursor.fetchall():
                    if extant and node not in seen:
                        yield node
                    seen.add(node)

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
                        self.name,
                        self.graph,
                        av,
                        branch,
                        rev
                    )
                )
                try:
                    return bool(self.engine.cursor.fetchone()[0])
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
                if self.engine._is_thing(self.graph, av):
                    return Thing(
                        self.char.engine.character[self.graph],
                        av
                    )
                else:
                    return Place(
                        self.char.engine.character[self.graph],
                        av
                    )
            if len(self) == 1:
                return self[next(iter(self))][av]
            raise KeyError("No such avatar")

        def __repr__(self):
            d = {}
            for k in self:
                d[k] = dict(self[k])
            return repr(d)


class SenseCharacterMapping(Mapping):
    def __init__(self, container, sensename):
        self.container = container
        self.engine = self.container.engine
        self.sensename = sensename
        self.observer = self.container.character

    @property
    def fun(self):
        return self.engine.function[self.sensename]

    def __iter__(self):
        for char in self.engine.character.values():
            test = self.fun(self.engine, self.observer, char)
            if test is None:  # The sense does not apply to the char
                continue
            elif not isinstance(test, CharacterImage):
                raise TypeError("Sense function did not return CharacterImage")
            else:
                yield char.name

    def __len__(self):
        return len(self.engine.character)

    def __getitem__(self, name):
        observed = self.engine.character[name]
        r = self.fun(self.engine, self.observer, observed)
        if not isinstance(r, CharacterImage):
            raise TypeError("Sense function did not return TransientCharacter")
        return r


class CharacterSenseMapping(MutableMapping, Callable):
    """Used to view other Characters as seen by one, via a particular sense"""
    def __init__(self, character):
        self.character = character
        self.engine = character.engine

    def __iter__(self):
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
                    self.character.name,
                    branch,
                    tick
                )
            )
            for (sense, active) in self.engine.cursor.fetchall():
                if active and sense not in seen:
                    yield sense
                seen.add(sense)

    def __len__(self):
        n = 0
        for sense in iter(self):
            n += 1
        return n

    def __getitem__(self, k):
        for (branch, tick) in self.engine._active_branches():
            self.engine.cursor.execute(
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
                    self.character.name,
                    k,
                    branch,
                    tick
                )
            )
            data = self.engine.cursor.fetchall()
            if len(data) == 0:
                continue
            elif len(data) > 1:
                raise ValueError("Silly data in senses table")
            else:
                return SenseCharacterMapping(self, k)
        raise KeyError("Sense isn't active or doesn't exist")

    def __setitem__(self, k, v):
        v.__name__ = k
        self(v)

    def __delitem__(self, k):
        (branch, tick) = self.engine.time
        try:
            self.engine.cursor.execute(
                "INSERT INTO senses "
                "(character, sense, branch, tick, active) "
                "VALUES "
                "(?, ?, ?, ?, 0);",
                (
                    self.character.name,
                    k,
                    branch,
                    tick
                )
            )
        except IntegrityError:
            self.engine.cursor.execute(
                "UPDATE senses SET active=0 WHERE "
                "character=? AND "
                "sense=? AND "
                "branch=? AND "
                "tick=?;",
                (
                    self.character.name,
                    k,
                    branch,
                    tick
                )
            )

    def __call__(self, fun):
        funn = self.engine.function(fun)
        (branch, tick) = self.engine.time
        try:
            self.engine.cursor.execute(
                "INSERT INTO senses "
                "(character, sense, branch, tick, active) "
                "VALUES "
                "(?, ?, ?, ?, 1);",
                (
                    self.character.name,
                    funn,
                    branch,
                    tick
                )
            )
        except IntegrityError:
            raise ValueError("Looks like this character already has that sense?")


class Character(DiGraph):
    """A graph that follows game rules and has a containment hierarchy.

    Nodes in a Character are subcategorized into Things and
    Places. Things have locations, and those locations may be Places
    or other Things. A Thing might also travel, in which case, though
    it will spend its travel time located in its origin node, it may
    spend some time contained by a Portal (i.e. an edge specialized
    for Character). If a Thing is not contained by a Portal, it's
    contained by whatever it's located in.

    """
    def __init__(self, engine, name, data=None, **attr):
        """Store engine and name, and set up mappings for Thing, Place, and
        Portal

        """
        super().__init__(engine.gorm, name, data=None, **attr)
        self.engine = engine
        self.thing = CharacterThingMapping(self)
        self.place = CharacterPlaceMapping(self)
        self.portal = CharacterPortalSuccessorsMapping(self)
        self.preportal = CharacterPortalPredecessorsMapping(self)
        self.avatar = CharacterAvatarGraphMapping(self)
        self.rule = CharRules(self)
        self.sense = CharacterSenseMapping(self)
        self.travel_reqs = FunList(self.engine, 'travel_reqs', ['character'], [name], 'reqs')

    def travel_req(self, fun):
        self.travel_reqs.append(fun)

    def add_place(self, name, **kwargs):
        """Create a new Place by the given name, and set its initial
        attributes based on the keyword arguments (if any).

        """
        super(Character, self).add_node(name, **kwargs)

    def add_places_from(self, seq):
        super().add_nodes_from(seq)

    def add_thing(self, name, location, next_location=None, **kwargs):
        """Create a Thing, set its location and next_location (if provided),
        and set its initial attributes from the keyword arguments (if
        any).

        """
        super(Character, self).add_node(name, **kwargs)
        self.place2thing(name, location)

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
        self.engine.cursor.execute(
            "INSERT INTO things ("
            "graph, node, branch, tick, location, next_location"
            ") VALUES ("
            "?, ?, ?, ?, ?"
            ");",
            (
                self.name,
                name,
                branch,
                tick,
                location,
                next_location
            )
        )

    def thing2place(self, name):
        """Unset a Thing's location, and thus turn it into a Place."""
        self.place2thing(name, None)

    def add_portal(self, origin, destination, **kwargs):
        """Connect the origin to the destination with a Portal. Keyword
        arguments are the Portal's attributes.

        """
        if origin.__class__ in (Place, Thing):
            origin = origin.name
        if destination.__class__ in (Place, Thing):
            destination = destination.name
        super(Character, self).add_edge(origin, destination, **kwargs)
        if 'symmetrical' in kwargs and kwargs['symmetrical']:
            self.add_portal(destination, origin, is_mirror=True)

    def add_portals_from(self, seq, symmetrical=False):
        for tup in seq:
            orig = tup[0]
            dest = tup[1]
            kwargs = tup[2] if len(tup) > 2 else {}
            if symmetrical:
                kwargs['symmetrical'] = True
            self.add_portal(orig, dest, **kwargs)

    def add_avatar(self, name, host, location=None, next_location=None):
        (branch, tick) = self.engine.time
        if isinstance(host, Character):
            host = host.name
        # This will create the node if it doesn't exist. Otherwise
        # it's redundant but harmless.
        try:
            self.engine.cursor.execute(
                "INSERT INTO nodes (graph, node, branch, rev, extant) "
                "VALUES (?, ?, ?, ?, ?);",
                (
                    host,
                    name,
                    branch,
                    tick,
                    True
                )
            )
        except IntegrityError:
            pass
        if location:
            # This will convert the node into a Thing if it isn't already
            self.engine.cursor.execute(
                "INSERT INTO things ("
                "character, thing, branch, tick, location, next_location"
                ") VALUES (?, ?, ?, ?, ?, ?);",
                (
                    host,
                    name,
                    branch,
                    tick,
                    location,
                    next_location
                )
            )
        # Declare that the node is my avatar
        self.engine.cursor.execute(
            "INSERT INTO avatars ("
            "character_graph, avatar_graph, avatar_node, "
            "branch, tick, is_avatar"
            ") VALUES (?, ?, ?, ?, ?, ?);",
            (
                self.name,
                host,
                name,
                branch,
                tick,
                True
            )
        )
