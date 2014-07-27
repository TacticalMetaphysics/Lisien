# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from collections import (
    Mapping,
    MutableMapping,
    Callable
)
from networkx import shortest_path
from gorm.graph import (
    DiGraph,
    GraphNodeMapping,
    GraphEdgeMapping,
    GraphSuccessorsMapping,
    DiGraphPredecessorsMapping
)
from LiSE.util import path_len
from LiSE.rule import Rule


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


class ThingPlace(GraphNodeMapping.Node):
    def __getitem__(self, name):
        """For when I'm the only avatar in a Character, and so you don't need
        my name to select me, but a name was provided anyway.

        """
        if name == self.name:
            return self
        elif name == "exists":
            return self.exists
        return super().__getitem__(name)

    def __setitem__(self, k, v):
        if k == "name":
            raise KeyError("Can't set name")
        elif k == "character":
            raise KeyError("Can't set character")
        elif k == "exists":
            self.exists = v
        super().__setitem__(k, v)

    def _contents_names(self):
        r = set()
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
                    r.add(thing)
                things_seen.add(thing)
        return r

    def contents(self):
        """Iterate over the Things that are located here."""
        for thingn in self._contents_names():
            yield self.character.thing[thingn]


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
        self.worldview = character.worldview
        self.name = name
        super(Thing, self).__init__(character, name)

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
            return super(Thing, self).__getitem__(key)

    def __setitem__(self, key, value):
        """Set ``key``==``value`` for the present game-time."""
        if key == 'name':
            raise ValueError("Can't change names")
        elif key == 'character':
            raise ValueError("Can't change characters")
        elif key == 'location':
            self._set_loc_and_next(value, self['next_location'])
        elif key == 'next_location':
            self._set_loc_and_next(self['location'], value)
        elif key == 'locations':
            self._set_loc_and_next(*value)
        else:
            super(Thing, self).__setitem__(key, value)
        if not hasattr(self, 'dontcall'):
            (branch, tick) = self.character.worldview.time
            for fun in self.character.on_set_thing_item:
                fun(self, key, value, branch, tick)

    def __delitem__(self, k):
        super().__delitem__(k)
        if not hasattr(self, 'dontcall'):
            (branch, tick) = self.character.worldview.time
            for fun in self.character.on_del_thing_item:
                fun(self, k, branch, tick)

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
        locn = self['location']
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
        (branch, tick) = self.character.worldview.time
        self.character.worldview.cursor.execute(
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
        self.character.worldview.cursor.execute(
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
        curloc = self["location"]
        orm = self.character.worldview
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
        curtick = self.character.worldview.tick
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
                fintick = self.character.worldview.tick
                self.character.worldview.tick = curtick
                raise TravelException(
                    "When I tried traveling to {}, at tick {}, I ended up at {}".format(
                        prevsubplace,
                        fintick,
                        l
                    ),
                    path=subpath,
                    followed=subsubpath,
                    branch=self.character.worldview.branch,
                    tick=fintick,
                    lastplace=l,
                    traveller=self
                )
            portal = self.character.portal[prevsubplace][subplace]
            tick_inc = portal.get(weight, 1)
            self.character.worldview.tick += tick_inc
            if self["location"] not in (subplace, prevsubplace):
                l = self["location"]
                fintick = self.character.worldview.tick
                self.character.worldview.tick = curtick
                raise TravelException(
                    "I couldn't go to {} at tick {} because I was in {}".format(
                        subplace,
                        fintick,
                        l
                    ),
                    path=subpath,
                    followed=subsubpath,
                    traveller=self,
                    branch=self.character.worldview.branch,
                    tick=fintick,
                    lastplace=l
                )
            self.character.worldview.tick -= tick_inc
            self.go_to_place(subplace, weight)
            self.character.worldview.tick += tick_inc
            ticks_total += tick_inc
            subsubpath.append(subplace)
            prevsubplace = subplace
        self.character.worldview.tick = curtick
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
        path = shortest_path(graph, self["location"], destn, weight)
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
        curtick = self.character.worldview.tick
        if arrival_tick <= curtick:
            raise ValueError("travel always takes positive amount of time")
        destn = dest.name if hasattr(dest, 'name') else dest
        graph = self.character if graph is None else graph
        curloc = self["location"]
        path = shortest_path(graph, curloc, destn, weight)
        travel_time = path_len(graph, path, weight)
        start_tick = arrival_tick - travel_time
        if start_tick <= curtick:
            raise self.TravelException(
                "path too heavy to follow by the specified tick",
                path=path,
                traveller=self
            )
        self.character.worldview.tick = start_tick
        self.follow_path(path, weight)
        self.character.worldview.tick = curtick


class Place(ThingPlace):
    """The kind of node where a Thing might ultimately be located."""
    def __init__(self, character, name):
        """Initialize a place in a character by a name"""
        self.character = character
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

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        if not hasattr(self, 'dontcall'):
            (branch, tick) = self.character.worldview.time
            for fun in self.character.on_set_place_item:
                fun(self, k, v, branch, tick)

    def __delitem__(self, k):
        super().__delitem__(k)
        if not hasattr(self, 'dontcall'):
            (branch, tick) = self.character.worldview.time
            for fun in self.character.on_del_place_item:
                fun(self, k, branch, tick)


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
        if not hasattr(self, 'dontcall'):
            (branch, tick) = self.character.worldview.time
            for fun in self.character.on_set_portal_item:
                fun(self, key, value, branch, tick)

    def __delitem__(self, k):
        super().__delitem__(k)
        if not hasattr(self, 'dontcall'):
            (branch, tick) = self.character.worldview.time
            for fun in self.on_del_portal_item:
                fun(self, k, branch, tick)

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


class CharacterThingMapping(MutableMapping):
    def __init__(self, character):
        self.character = character
        self.worldview = character.worldview
        self.name = character.name

    def __iter__(self):
        """Iterate over nodes that have locations, and are therefore
        Things. Yield their names.

        """
        seen = set()
        for (branch, tick) in self.worldview._active_branches():
            self.worldview.cursor.execute(
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
            for (node, loc) in self.worldview.cursor.fetchall():
                if loc and node not in seen:
                    yield node
                seen.add(node)

    def __len__(self):
        n = 0
        for th in self:
            n += 1
        return n

    def __getitem__(self, thing):
        for (branch, rev) in self.worldview._active_branches():
            self.worldview.cursor.execute(
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
            for (thing, loc) in self.worldview.cursor.fetchall():
                if not loc:
                    raise KeyError("Thing doesn't exist right now")
                return Thing(self.character, thing)
        raise KeyError("Thing has never existed")

    def __setitem__(self, thing, val):
        th = Thing(self.character, thing)
        th.dontcall = True
        th.clear()
        th.exists = True
        th.update(val)
        del th.dontcall
        (branch, tick) = self.character.worldview.time
        for fun in self.character.on_set_thing:
            fun(thing, val, branch, tick)

    def __delitem__(self, thing):
        Thing(self.character, thing).clear()
        (branch, tick) = self.character.worldview.time
        for fun in self.character.on_del_thing:
            fun(thing, branch, tick)

    def __repr__(self):
        return repr(dict(self))


class CharacterPlaceMapping(MutableMapping):
    def __init__(self, character):
        self.character = character
        self.worldview = character.worldview
        self.name = character.name

    def __iter__(self):
        things = set()
        things_seen = set()
        for (branch, rev) in self.worldview._active_branches():
            self.worldview.cursor.execute(
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
            for (thing, loc) in self.worldview.cursor.fetchall():
                if thing not in things_seen and loc:
                    things.add(thing)
                things_seen.add(thing)
        for node in self.worldview._iternodes(self.character.name):
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
        pl.dontcall = True
        pl.clear()
        pl.exists = True
        pl.update(v)
        del pl.dontcall
        (branch, tick) = self.character.worldview.time
        for fun in self.character.on_set_place:
            fun(place, v, branch, tick)

    def __delitem__(self, place):
        Place(self.character, place).clear()
        (branch, tick) = self.character.worldview.time
        for fun in self.character.on_del_place:
            fun(place, branch, tick)

    def __repr__(self):
        return repr(dict(self))


class CharacterPortalSuccessorsMapping(GraphSuccessorsMapping):
    class Successors(GraphSuccessorsMapping.Successors):
        def _getsub(self, nodeB):
            return Portal(self.graph, self.nodeA, nodeB)

        def __setitem__(self, nodeB, value):
            p = Portal(self.graph, self.nodeA, nodeB)
            p.dontcall = True
            p.clear()
            p.exists = True
            p.update(value)
            del p.dontcall
            (branch, tick) = self.graph.worldview.time
            for fun in self.graph.on_set_portal:
                fun(self.nodeA, nodeB, value, branch, tick)

        def __delitem__(self, nodeB):
            super().__delitem__(nodeB)
            (branch, tick) = self.graph.worldview.time
            for fun in self.graph.on_del_portal:
                fun(self.nodeA, nodeB, branch, tick)


class CharacterPortalPredecessorsMapping(DiGraphPredecessorsMapping):
    class Predecessors(DiGraphPredecessorsMapping.Predecessors):
        def _getsub(self, nodeA):
            return Portal(self.graph, nodeA, self.nodeB)

        def __setitem__(self, nodeA, value):
            p = Portal(self.graph, nodeA, self.nodeB)
            p.dontcall = True
            p.clear()
            p.exists = True
            p.update(value)
            del p.dontcall
            (branch, tick) = self.graph.worldview.time
            for fun in self.graph.on_set_portal:
                fun(nodeA, self.nodeB, value, branch, tick)

        def __delitem__(self, nodeA):
            super().__delitem__(nodeA)
            (branch, tick) = self.graph.worldview.time
            for fun in self.graph.on_del_portal:
                fun(nodeA, self.nodeB, branch, tick)


class CharRules(Mapping):
    """Maps rule names to rules the Character is following, and is also a
    decorator to create said rules from action functions.

    Decorating a function with this turns the function into the
    first action of a new rule of the same name, and applies the
    rule to the character. Add more actions with the @rule.action
    decorator, and add prerequisites with @rule.prereq

    """
    def __init__(self, orm, char):
        """Store the character"""
        self.orm = orm
        self.character = char
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
            vname = self.orm.function(v)
            self._activate_rule(
                Rule(
                    self.orm,
                    vname,
                    actions=[vname]
                )
            )
        else:
            # v is the name of a rule. Maybe it's been created
            # previously or maybe it'll get initialized in Rule's
            # __init__.
            self._activate_rule(Rule(self.orm, v))

    def __iter__(self):
        """Iterate over all rules presently in effect"""
        seen = set()
        for (branch, tick) in self.orm._active_branches():
            self.orm.cursor.execute(
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
                "AND char_rules.tick=hitick.tick;"
                (
                    self.character.name,
                    branch,
                    tick
                )
            )
            for (rule, active) in self.orm.cursor.fetchall():
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
        for (branch, tick) in self.orm._active_branches():
            self.orm.cursor.execute(
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
            data = self.orm.cursor.fetchall()
            if len(data) == 0:
                continue
            elif len(data) > 1:
                raise ValueError("Silly data in char_rules table")
            else:
                (active,) = data[0]
                if not active:
                    raise KeyError("No such rule at the moment")
                return Rule(self.orm, rulen)
        raise KeyError("No such rule, ever")

    def __getattr__(self, attrn):
        """For easy use with decorators, allow accessing my contents like
        attributes

        """
        try:
            return self[attrn]
        except KeyError as err:
            raise AttributeError(err.message)

    def _activate_rule(self, rule):
        """Indicate that the rule is active and should be followed. Add the
        given arguments to whatever's there.

        """
        (branch, tick) = self.orm.time
        self.orm.cursor.execute(
            "DELETE FROM char_rules WHERE "
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
        self.orm.cursor.execute(
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

    def __delitem__(self, rule):
        """Deactivate the rule"""
        if isinstance(rule, Rule):
            rulen = rule.name
        else:
            rulen = self.orm.function(rule)
        (branch, tick) = self.orm.time
        self.orm.cursor.execute(
            "DELETE FROM char_rules WHERE "
            "character=? AND "
            "rule=? AND "
            "branch=? AND "
            "tick=?;",
            (
                self.name,
                rulen,
                branch,
                tick
            )
        )
        self.orm.cursor.execute(
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


class CharacterAvatarGraphMapping(Mapping):
    def __init__(self, char):
        """Remember my character"""
        self.char = char
        self.worldview = char.worldview
        self.name = char.name

    def __call__(self, av):
        """Add the avatar. It must be an instance of Place or Thing."""
        if av.__class__ not in (Place, Thing):
            raise TypeError("Only Things and Places may be avatars")
        self.char.add_avatar(av.name, av.character.name)

    def _datadict(self):
        """Get avatar-ness data and return it"""
        d = {}
        for (branch, rev) in self.worldview._active_branches():
            self.worldview.cursor.execute(
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
            for (graph, node, avatar) in self.worldview.cursor.fetchall():
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
        raise KeyError("No avatars in {}".format(g))

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
            self.worldview = outer.worldview
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
            for (branch, rev) in self.worldview._active_branches():
                self.worldview.cursor.execute(
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
                for (node, extant) in self.worldview.cursor.fetchall():
                    if extant and node not in seen:
                        yield node
                    seen.add(node)

        def __contains__(self, av):
            for (branch, rev) in self.worldview._active_branches():
                self.worldview.cursor.execute(
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
                    return bool(self.worldview.cursor.fetchone()[0])
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
                if self.worldview._is_thing(self.graph, av):
                    return Thing(
                        self.char.worldview.character[self.graph],
                        av
                    )
                else:
                    return Place(
                        self.char.worldview.character[self.graph],
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
    def __init__(self, worldview, name, on_set_thing=[], on_del_thing=[], on_set_thing_item=[], on_del_thing_item=[], on_set_place=[], on_del_place=[], on_set_place_item=[], on_del_place_item=[], on_set_portal=[], on_del_portal=[], on_set_portal_item=[], on_del_portal_item=[]):
        """Store worldview and name, and set up mappings for Thing, Place, and
        Portal

        """
        super(Character, self).__init__(worldview.gorm, name)
        self.worldview = worldview
        self.on_set_place = on_set_place
        self.on_del_place = on_del_place
        self.on_set_place_item = on_set_place_item
        self.on_del_place_item = on_del_place_item
        self.on_set_thing = on_set_thing
        self.on_del_thing = on_del_thing
        self.on_set_thing_item = on_set_thing_item
        self.on_del_thing_item = on_del_thing_item
        self.on_set_portal = on_set_portal
        self.on_del_portal = on_del_portal
        self.on_set_portal_item = on_set_portal_item
        self.on_del_portal_item = on_del_portal_item
        self.thing = CharacterThingMapping(self)
        self.place = CharacterPlaceMapping(self)
        self.portal = CharacterPortalSuccessorsMapping(self)
        self.preportal = CharacterPortalPredecessorsMapping(self)
        self.avatar = CharacterAvatarGraphMapping(self)

    def add_place(self, name, **kwargs):
        """Create a new Place by the given name, and set its initial
        attributes based on the keyword arguments (if any).

        """
        super(Character, self).add_node(name, **kwargs)

    def add_thing(self, name, location, next_location=None, **kwargs):
        """Create a Thing, set its location and next_location (if provided),
        and set its initial attributes from the keyword arguments (if
        any).

        """
        super(Character, self).add_node(name, **kwargs)
        self.place2thing(name, location)

    def place2thing(self, name, location, next_location=None):
        """Turn a Place into a Thing with the given location and (if provided)
        next_location. It will keep all its attached Portals.

        """
        (branch, tick) = self.worldview.time
        self.worldview.cursor.execute(
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
            super().add_edge(destination, origin, is_mirror=True)

    def add_avatar(self, name, host, location=None, next_location=None):
        (branch, tick) = self.worldview.time
        if isinstance(host, Character):
            host = host.name
        # This will create the node if it doesn't exist. Otherwise
        # it's redundant but harmless.
        self.worldview.cursor.execute(
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
        if location:
            # This will convert the node into a Thing if it isn't already
            self.worldview.cursor.execute(
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
        self.worldview.cursor.execute(
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
