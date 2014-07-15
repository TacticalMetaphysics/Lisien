from networkx import shortest_path
from gorm.graph import (
    GraphNodeMapping,
    GraphEdgeMapping,
)
from LiSE.util import path_len


class ThingPlace(GraphNodeMapping.Node):
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
        self.character = character
        self.orm = character.orm
        self.name = name
        super(Thing, self).__init__(character, name)

    def __getitem__(self, key):
        if key == 'name':
            return self.name
        elif key == 'character':
            return self.character.name
        elif key == 'location':
            return self._loc_and_next()[0]
        elif key == 'next_location':
            return self._loc_and_next()[1]
        elif key == 'locations':
            return self._loc_and_next()
        else:
            return super(Thing, self).__getitem__(key)

    def __setitem__(self, key, value):
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

    def rule(self, v):
        """Make a Rule on my Character that takes me as an argument.

        """
        self.character.rule(v, [self])

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
                return data.pop()
        raise ValueError("No location set")

    def _set_loc_and_next(self, loc, nextloc):
        (branch, tick) = self.character.orm.time
        self.character.orm.cursor.execute(
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
        self.character.orm.cursor.execute(
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

    def go_to_place(self, place, weight=None):
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
        orm = self.character.orm
        curtick = orm.tick
        if weight:
            ticks = self.character.portal[curloc][placen][weight]
        else:
            ticks = 1
        self['next_location'] = placen
        orm.tick += ticks
        self['locations'] = (placen, None)
        orm.tick = curtick
        return ticks

    def follow_path(self, path, weight=None):
        """Go to several Places in succession, deciding how long to spend in
        each by consulting the ``weight`` attribute of the Portal
        connecting the one Place to the next.

        Return the total number of ticks the travel will take.

        """
        curtick = self.character.orm.tick
        prevplace = path.pop(0)
        if prevplace != self['location']:
            raise ValueError("Path does not start at my present location")
        subpath = []
        for place in path:
            place = place.name if hasattr(place, 'name') else place
            if (
                    prevplace not in self.character.portal or
                    place not in self.character.portal[prevplace]
            ):
                raise self.TravelException(subpath)
            subpath.append(place)
        ticks_total = 0
        for subplace in subpath:
            tick_inc = self.go_to_place(subplace, weight)
            ticks_total += tick_inc
            self.character.orm.tick += tick_inc
        self.character.orm.tick = curtick
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
        path = shortest_path(graph, self["location"], destn, weight)
        return self.follow_path(path, weight)

    def travel_to_by(self, dest, arrival_tick, weight=None, graph=None):
        curtick = self.character.orm.tick
        if arrival_tick <= curtick:
            raise ValueError("travel always takes positive amount of time")
        destn = dest.name if hasattr(dest, 'name') else dest
        graph = self.character if graph is None else graph
        curloc = self["location"]
        path = shortest_path(graph, curloc, destn, weight)
        travel_time = path_len(graph, path, weight)
        start_tick = arrival_tick - travel_time
        if start_tick <= curtick:
            raise self.TravelException("path too heavy to follow by the specified tick")
        self.character.orm.tick = start_tick
        self.follow_path(path, weight)
        self.character.orm.tick = curtick

    class TravelException(Exception):
        """Exception for when a Thing travelling someplace doesn't work
        right

        """
        def __init__(self, subpath):
            """Remember the path and put it in my message"""
            self.subpath = subpath
            Exception.__init__(
                self,
                "Couldn't follow the whole path, only {}".format(subpath)
            )


class Place(ThingPlace):
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

    def rule(self, v):
        """Make a Rule on my Character that takes me as an argument."""
        self.character.rule(v, [self])


class Portal(GraphEdgeMapping.Edge):
    """Connection between two Places that Things may travel along.

    Portals are one-way. If the connection between two Places is
    symmetrical, you may find it convenient to make a rule to update
    the properties of the portal in one direction to match those in
    the other direction.

    """
    def __init__(self, character, origin, destination):
        self._origin = origin
        self._destination = destination
        self.character = character
        super(Portal, self).__init__(character, self._origin, self._destination)

    def __getitem__(self, key):
        if key == 'origin':
            return self._origin
        elif key == 'destination':
            return self._destination
        elif key == 'character':
            return self.character.name
        else:
            return super(Portal, self).__getitem__(key)

    @property
    def origin(self):
        return self.character.place[self._origin]

    @property
    def destination(self):
        return self.character.place[self._destination]

    def rule(self, v):
        """Make a new Rule on my Character that takes me as an argument."""
        self.character.rule(v, [self])

    def _contents_names(self):
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
        for thingn in self._contents_names():
            yield self.character.thing[thingn]
