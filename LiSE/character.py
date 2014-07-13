# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from collections import (
    Mapping,
    MutableMapping
)
from json import dumps as jsonned
from json import loads as unjsonned
from networkx import shortest_path
from gorm.graph import (
    GraphNodeMapping,
    GraphEdgeMapping,
    DiGraph,
    GraphSuccessorsMapping,
    DiGraphPredecessorsMapping
)
from LiSE.util import path_len
from LiSE.rule import Rule


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
    def __init__(self, orm, name):
        """Store orm and name, and set up mappings for Thing, Place, and
        Portal

        """
        super(Character, self).__init__(orm.gorm, name)
        self.orm = orm
        self.thing = self.CharacterThingMapping(self)
        self.place = self.CharacterPlaceMapping(self)
        self.portal = self.CharacterPortalSuccessorsMapping(self)
        self.preportal = self.CharacterPortalPredecessorsMapping(self)
        self.avatar = self.CharacterAvatarGraphMapping(self)
        self.rule = self.CharRules(self)

    def _arg_parse(self, arg):
        if arg == 'character':
            return self
        elif arg[:6] == 'thing:':
            return self.thing[arg[6:]]
        elif arg[:6] == 'place:':
            return self.place[arg[6:]]
        elif arg[:7] == 'portal:':
            (a, b) = arg[7:].split("|")
            return self.portal[a][b]
        else:
            raise ValueError("Can't parse arg {}".format(arg))

    def _arg_enc(self, arg):
        if arg is self:
            return 'character'
        elif isinstance(arg, Thing):
            return 'thing:' + arg.name
        elif isinstance(arg, Place):
            return 'place:' + arg.name
        elif isinstance(arg, Portal):
            return 'portal:{}|{}'.format(
                arg['origin'],
                arg['destination']
            )
        else:
            raise ValueError("Cas't encode arg {}".format(arg))

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
        (branch, tick) = self.orm.time
        self.orm.cursor.execute(
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

    def add_avatar(self, name, host, location=None, next_location=None):
        (branch, tick) = self.orm.time
        if isinstance(host, Character):
            host = host.name
        # This will create the node if it doesn't exist. Otherwise
        # it's redundant but harmless.
        self.orm.cursor.execute(
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
            self.orm.cursor.execute(
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
        self.orm.cursor.execute(
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

    class CharacterThingMapping(MutableMapping):
        def __init__(self, character):
            self.character = character
            self.orm = character.orm
            self.name = character.name

        def __iter__(self):
            """Iterate over nodes that have locations, and are therefore
            Things. Yield their names.

            """
            seen = set()
            for (branch, tick) in self.orm._active_branches():
                self.orm.cursor.execute(
                    "SELECT things.node, things.location FROM things JOIN ("
                    "SELECT graph, node, branch, MAX(tick) AS tick FROM things "
                    "WHERE graph=? "
                    "AND branch=? "
                    "AND tick<=? "
                    "GROUP BY graph, node, branch) AS hitick "
                    "ON things.graph=hitick.graph "
                    "AND things.node=hitick.node "
                    "AND things.branch=hitick.branch "
                    "AND things.tick=hitick.tick;",
                    (
                        self.character.name,
                        branch,
                        tick
                    )
                )
                for (node, loc) in self.orm.cursor.fetchall():
                    if loc and node not in seen:
                        yield node
                    seen.add(node)

        def __len__(self):
            n = 0
            for th in self:
                n += 1
            return n

        def __getitem__(self, thing):
            for (branch, rev) in self.orm._active_branches():
                self.orm.cursor.execute(
                    "SELECT things.node, things.location FROM things JOIN ("
                    "SELECT graph, node, branch, MAX(tick) AS tick FROM things "
                    "WHERE graph=? "
                    "AND node=? "
                    "AND branch=? "
                    "AND tick<=? "
                    "GROUP BY graph, node, branch) AS hitick "
                    "ON things.graph=hitick.graph "
                    "AND things.node=hitick.node "
                    "AND things.branch=hitick.branch "
                    "AND things.tick=hitick.tick;",
                    (
                        self.character.name,
                        thing,
                        branch,
                        rev
                    )
                )
                for (thing, loc) in self.orm.cursor.fetchall():
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
            self.orm = character.orm
            self.name = character.name

        def __iter__(self):
            things = set()
            things_seen = set()
            for (branch, rev) in self.orm._active_branches():
                self.orm.cursor.execute(
                    "SELECT things.node, things.location FROM things JOIN ("
                    "SELECT graph, node, branch, MAX(rev) AS rev FROM things "
                    "WHERE graph=? "
                    "AND branch=? "
                    "AND rev<=? "
                    "GROUP BY graph, node, branch) AS hirev ON "
                    "things.graph=hirev.graph "
                    "AND things.node=hirev.node "
                    "AND things.branch=hirev.branch "
                    "AND things.rev=hirev.rev;",
                    (
                        self.character.name,
                        branch,
                        rev
                    )
                )
                for (thing, loc) in self.orm.cursor.fetchall():
                    if thing not in things_seen and loc:
                        things.add(thing)
                    things_seen.add(thing)
            for node in self.orm._iternodes(self.character.name):
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


    class CharacterPortalPredecessorsMapping(DiGraphPredecessorsMapping):
        class Predecessors(DiGraphPredecessorsMapping.Predecessors):
            def _getsub(self, nodeA):
                return Portal(self.graph, nodeA, self.nodeB)


    class CharacterAvatarGraphMapping(Mapping):
        def __init__(self, char):
            """Remember my character"""
            self.char = char
            self.orm = char.orm
            self.name = char.name

        def __call__(self, av):
            """Add the avatar. It must be an instance of Place or Thing."""
            if av.__class__ not in (Place, Thing):
                raise TypeError("Only Things and Places may be avatars")
            self.char.add_avatar(av.name, av.character.name)

        def _datadict(self):
            """Get avatar-ness data and return it"""
            d = {}
            for (branch, rev) in self.orm._active_branches():
                self.orm.cursor.execute(
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
                for (graph, node, avatar) in self.orm.cursor.fetchall():
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
                self.orm = outer.orm
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
                for (branch, rev) in self.orm._active_branches():
                    self.orm.cursor.execute(
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
                    for (node, extant) in self.orm.cursor.fetchall():
                        if extant and node not in seen:
                            yield node
                        seen.add(node)

            def __contains__(self, av):
                for (branch, rev) in self.orm._active_branches():
                    self.orm.cursor.execute(
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
                        return bool(self.orm.cursor.fetchone()[0])
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
                    if self.orm._is_thing(self.graph, av):
                        return Thing(
                            self.orm.get_character(self.graph),
                            av
                        )
                    else:
                        return Place(
                            self.orm.character[self.graph],
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


    class CharRules(Mapping):
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
            self.orm = char.orm
            self.name = char.name

        def __call__(self, v, xargs=[]):
            """If passed a Rule, activate it. If passed a string, get the rule by
            that name and activate it. If passed a function (probably
            because I've been used as a decorator), make a rule with
            the same name as the function, with the function itself
            being the first action of the rule, and activate that
            rule.

            """
            if isinstance(v, Rule):
                self._activate_rule(v, xargs)
            elif callable(v):
                # create a new rule performing the action v
                vname = self.orm.function(v)
                self._activate_rule(
                    Rule(
                        self.orm,
                        vname,
                        actions=[vname]
                    ),
                    xargs
                )
            else:
                # v is the name of a rule. Maybe it's been created
                # previously or maybe it'll get initialized in Rule's
                # __init__.
                self._activate_rule(Rule(self.orm, v), xargs)

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
                    "AND char_rules.tick=hitick.tick;"
                    (
                        self.character.name,
                        branch,
                        tick
                    )
                )
                for (rule, active) in self.orm.fetchall():
                    if active and rule not in seen:
                        yield rule
                    seen.add(rule)

        def __len__(self):
            """Count the rules presently in effect"""
            seen = set()
            n = 0
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
                    "AND char_rules.tick=hitick.tick;"
                    (
                        self.character.name,
                        branch,
                        tick
                    )
                )
            for (rule, active) in self.orm.fetchall():
                if active and rule not in seen:
                    n += 1
                seen.add(rule)
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

        def args(self, rule):
            """Return the arguments that the Rule should pass to its actions and
            prereqs

            """
            if hasattr(rule, 'name'):
                rule = rule.name
            for (branch, tick) in self.orm._active_branches():
                self.orm.cursor.execute(
                    "SELECT args FROM char_rules JOIN ("
                    "SELECT character, rule, branch, MAX(tick) AS tick "
                    "FROM char_rules WHERE "
                    "character=? AND "
                    "rule=? AND "
                    "branch=? AND "
                    "tick<=? GROUP BY character, rule, branch) "
                    "AS hitick ON "
                    "char_rules.character=hitick.character "
                    "AND char_rules.rule=hitick.rule "
                    "AND char_rules.branch=hitick.branch "
                    "AND char_rules.tick=hitick.tick;",
                    (
                        self.character.name,
                        rule,
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
                    return [
                        self.character._arg_parse(self.character, arg)
                        for arg in unjsonned(data[0][0])
                    ]
            return [self.character]

        def _activate_rule(self, rule, args=[]):
            """Indicate that the rule is active and should be followed. Add the
            given arguments to whatever's there.

            """
            (branch, tick) = self.orm.time
            args = self.args(rule) + args
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
                "(character, rule, branch, tick, active, args) "
                "VALUES (?, ?, ?, ?, ?, ?);",
                (
                    self.character.name,
                    rule.name,
                    branch,
                    tick,
                    True,
                    jsonned(
                        [self.character._arg_enc(arg) for arg in args]
                    )
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
