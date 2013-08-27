# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from place import Place
from thing import Thing
from portal import Portal
from logging import getLogger
from igraph import Graph, InternalError
from collections import defaultdict
from util import TabdictIterator, stringlike


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
        self.realit = iter(dim.graph.es)

    def __iter__(self):
        return self

    def next(self):
        return self.realit.next()["portal"]


"""Class and loaders for dimensions--the top of the world hierarchy."""


class Dimension:
    """Container for a given view on the game world, sharing no things,
places, or portals with any other dimension, but possibly sharing
characters."""
    def __init__(self, rumor, name, td):
        """Return a dimension with the given name.

Probably useless unless, once you're sure you've put all your places,
portals, and things in the db, you call the Dimension object's unravel(db)
method. Thereafter, it will have dictionaries of all those items,
keyed with their names.

        """
        self._name = name
        self.rumor = rumor
        self._tabdict = td
        self.boards = []
        self.thingdict = {}
        self.graph = Graph(directed=True)
        for rd in TabdictIterator(td["portal"]):
            Portal(self.rumor, self, rd["origin"], rd["destination"], td)
        for rd in TabdictIterator(td["thing_location"]):
            self.thingdict[rd["thing"]] = Thing(self.rumor, self, rd["thing"], td)
        self.rumor.dimensiondict[str(self)] = self

    def __hash__(self):
        """Return the hash of this dimension's name, since the database
constrains it to be unique."""
        return hash(self.name)

    def __str__(self):
        return self._name

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
        try:
            if not isinstance(iname, int):
                vnames = self.graph.vs["name"]
                name = iname
                iname = vnames.index(iname)
            v = self.graph.vs[iname]
            return Place(self, v)
        except (IndexError, KeyError, ValueError):
            return self.make_place(iname)

    def make_place(self, name):
        i = len(self.graph.vs)
        self.graph.add_vertex(name=name)
        return Place(self, self.graph.vs[i])

    def have_portal(self, orig, dest):
        (origi, orig) = self.sanitize_vert(orig)
        (desti, dest) = self.sanitize_vert(dest)
        lgv = len(self.graph.vs)
        if origi >= lgv or desti >= lgv:
            return False
        return self.graph[origi, desti] > 0

    def get_portal(self, orig, dest):
        (origi, orig) = self.sanitize_vert(orig)
        (desti, dest) = self.sanitize_vert(dest)
        return self.graph.es[self.graph.get_eid(origi, desti)]["portal"]

    def get_thing(self, name):
        return self.thingdict[name]

    def new_branch(self, parent, branch, tick):
        for thing in self.things:
            thing.new_branch(parent, branch, tick)
        for e in self.graph.es:
            e["portal"].new_branch(parent, branch, tick)

    def sanitize_vert(self, v):
        if isinstance(v, int):
            i = v
            v = self.graph.vs[i]
        elif isinstance(v, Place):
            v = v.v
            i = v.i
        elif stringlike(v):
            vname = str(v)
            vnames = self.graph.vs["name"]
            i = vnames.index(vname)
            v = self.graph.vs[i]
        else:
            i = v.index
        return (i, v)

    def sanitize_edge(self, e):
        if isinstance(e, int):
            i = e
            e = self.graph.es[i]
        elif isinstance(e, Portal):
            e = e.e
            i = e.index
        elif stringlike(e):
            if e[:6] == "Portal":
                e = e[6:]
            if e[0] == "(":
                e = e[1:]
            if e[-1] == ")":
                e = e[:-1]
            (orign, destn) = e.split("->")
            i = self.graph.get_eid(orign, destn)
            e = self.graph.es[i]
        else:
            i = e.index
        return (i, e)
