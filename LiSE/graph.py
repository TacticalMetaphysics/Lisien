from networkx import shortest_path
from gorm.graph import (
    GraphNodeMapping,
    GraphEdgeMapping,
)
from LiSE.util import path_len


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
        return super().__getitem__(name)

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
