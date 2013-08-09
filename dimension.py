# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from place import Place
from thing import Thing
from portal import Portal
from logging import getLogger
from igraph import Graph, InternalError


logger = getLogger(__name__)


class PlaceIter:
    def __init__(self, dim):
        self.dim = dim
        self.realit = iter(dim.graph.vs)

    def __iter__(self):
        return self

    def next(self):
        return Place(self.dim, self.realit.next())


class PortIter:
    def __init__(self, dim):
        self.dim = dim
        self.realit = iter(dim.graph.es)

    def __iter__(self):
        return self

    def next(self):
        return Portal(self.dim, self.realit.next())


"""Class and loaders for dimensions--the top of the world hierarchy."""


class Dimension:
    """Container for a given view on the game world, sharing no things,
places, or portals with any other dimension, but possibly sharing
characters."""
    def __init__(self, rumor, name):
        """Return a dimension with the given name.

Probably useless unless, once you're sure you've put all your places,
portals, and things in the db, you call the Dimension object's unravel(db)
method. Thereafter, it will have dictionaries of all those items,
keyed with their names.

        """
        self.name = name
        self.rumor = rumor
        self.boards = []
        self.thingdict = {}
        self.graph = Graph(directed=True)

    def __hash__(self):
        """Return the hash of this dimension's name, since the database
constrains it to be unique."""
        return hash(self.name)

    def __str__(self):
        return self.name

    def __getattr__(self, attrn):
        if attrn == "places":
            return PlaceIter(self)
        elif attrn == "placenames":
            try:
                return self.graph.vs["name"]
            except KeyError:
                return []
        elif attrn == "portals":
            return PortIter(self)
        elif attrn == "things":
            return self.thingdict.itervalues()
        else:
            raise AttributeError(
                "Dimension instance has no attribute named " +
                attrn)

    def get_igraph_layout(self, layout_type):
        """Return a Graph layout, of the kind that igraph uses, representing
this dimension, and laid out nicely."""
        return self.graph.layout(layout=layout_type)

    def have_place(self, name):
        if len(self.graph.vs) == 0:
            return False
        return name in self.graph.vs["name"]

    def get_place(self, iname):
        vnames = self.graph.vs["name"]
        if not isinstance(iname, int):
            iname = vnames.index(iname)
        v = self.graph.vs[iname]
        return Place(self, v)

    def make_place(self, name, existence=None, indef_exist=None, spots=None):
        if existence is None:
            existence = {}
        if indef_exist is None:
            indef_exist = {}
        if spots is None:
            spots = []
        self.graph.add_vertex(
            name=name,
            existence=existence,
            indef_exist=indef_exist,
            spots=spots)
        return self.get_place(name)

    def have_portal(self, orig, dest):
        vnames = self.graph.vs["name"]
        if isinstance(orig, Place):
            orig = int(orig)
        elif not isinstance(orig, int):
            orig = vnames.index(orig)
        if isinstance(dest, Place):
            dest = int(dest)
        elif not isinstance(dest, int):
            dest = vnames.index(dest)
        lgv = len(self.graph.vs)
        if orig >= lgv or dest >= lgv:
            return False
        return self.graph[orig, dest] > 0

    def make_portal(
            self, orig, dest,
            existence=None, indef_exist=None, arrows=None):
        vertns = self.graph.vs["name"]
        if hasattr(orig, 'v'):
            orig = int(orig)
        elif not isinstance(orig, int):
            orig = vertns.index(orig)
        if hasattr(dest, 'v'):
            dest = int(dest)
        elif not isinstance(dest, int):
            dest = vertns.index(dest)
        if existence is None:
            existence = {}
        if indef_exist is None:
            indef_exist = {}
        if arrows is None:
            arrows = []
        self.graph.add_edge(
            orig, dest,
            existence=existence,
            indef_exist=indef_exist,
            arrows=arrows)
        return self.get_portal(orig, dest)

    def get_portal(self, orig, dest):
        vertns = self.graph.vs["name"]
        if isinstance(orig, Place):
            orig = int(orig)
        elif not isinstance(orig, int):
            orig = vertns.index(orig)
        if isinstance(dest, Place):
            dest = int(dest)
        elif not isinstance(dest, int):
            dest = vertns.index(dest)
        try:
            eid = self.graph.get_eid(orig, dest)
            return Portal(self, self.graph.es[eid])
        except InternalError:
            return None

    def make_thing(self, name, locations=None, indef_locs=None):
        if locations is None:
            locations = {}
        if indef_locs is None:
            indef_locs = {}
        self.thingdict[name] = Thing(self, name, locations, indef_locs)
        return self.thingdict[name]

    def get_thing(self, name):
        return self.thingdict[name]

    def portal_extant(self, e, branch=None, tick=None):
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        if branch in e["indef_exist"]:
            if tick > e["indef_exist"][branch]:
                return True
        if branch not in e["existence"]:
            return False
        for (tick_from, tick_to) in e["existence"][branch].iteritems():
            if tick_from <= tick and tick <= tick_to:
                return True
        return False

    def portal_extant_between(
            self, e, branch=None, tick_from=None, tick_to=None):
        if branch is None:
            branch = self.rumor.branch
        if tick_from is None:
            tick_from = self.rumor.tick
        if tick_to is None:
            if branch not in e["indef_exist"]:
                return False
            ifrom = e["indef_exist"][branch]
            return ifrom < tick_from
        if branch not in e["existence"]:
            return False
        # search for an existence window that either matches or
        # contains the one given
        for (tick_before, tick_after) in e["existence"][branch].iteritems():
            if tick_before <= tick_from and tick_after >= tick_to:
                return True
        return False

    def persist_portal(self, e, branch=None, tick_from=None, tick_to=None):
        if branch is None:
            branch = self.rumor.branch
        if tick_from is None:
            tick_from = self.rumor.tick
        if branch in e["indef_exist"]:
            ifrom = e["indef_exist"][branch]
            if tick_from > ifrom:
                e["existence"][branch][ifrom] = tick_from - 1
                del e["indef_exist"][branch]
            elif tick_from == ifrom:
                del e["indef_exist"][branch]
            elif tick_to >= ifrom:
                del e["existence"][branch][ifrom]
                e["existence"][branch][tick_from] = None
                e["indef_exist"][branch] = tick_from
                return
        if tick_to is None:
            e["indef_exist"][branch] = tick_from
        if branch not in e["existence"]:
            e["existence"][branch] = {}
        e["existence"][branch][tick_from] = tick_to

    def save(self):
        for portal in self.portals:
            portal.save()
        for thing in self.things:
            thing.save()
