# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from place import Place
from thing import Thing
from portal import Portal
from spot import Spot
from board import Board
from pawn import Pawn
from img import Img
from arrow import Arrow
from logging import getLogger
from igraph import Graph
from collections import OrderedDict
from util import DictValues2DIterator


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
        self.rumor.dimensiondict[str(self)] = self
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
        elif attrn == "portals":
            return PortIter(self)
        elif attrn == "things":
            return self.thingdict.itervalues()
        else:
            return super(Dimension, self).__getattr__(attrn)

    def get_igraph_layout(self, layout_type):
        """Return a Graph layout, of the kind that igraph uses, representing
this dimension, and laid out nicely."""
        return self.graph.layout(layout=layout_type)

    def add_place(self, name, existence={}, indef_exist={}, spots=[]):
        self.graph.add_vertex({
            "name": name,
            "contents": set(),
            "existence": existence,
            "indef_exist": indef_exist,
            "spots": spots})

    def get_place(self, iname):
        return Place(self, self.graph.vs[iname])

    def add_portal(self, orig, dest, existence={}, indef_exist={}, arrows=[]):
        self.graph.add_edge(orig, dest, {
            "existence": existence,
            "indef_exist": indef_exist,
            "arrows": arrows})

    def get_portal(self, orig, dest):
        if isinstance(orig, Place):
            orig = int(orig)
        if isinstance(dest, Place):
            dest = int(dest)
        return Portal(self, self.graph.es[orig, dest])

    def add_thing(self, name):
        self.thingdict[name] = Thing(self, name)

    def get_thing(self, name):
        return self.thingdict[name]

    def portal_extant(self, e, branch=None, tick=None):
        if branch is None:
            branch = self.rumor.branch
        if tick is None:
            tick = self.rumor.tick
        if branch in e["indefinite_existence"]:
            if tick > e["indefinite_existence"][branch]:
                return True
        if branch not in e["existence"]:
            return False
        for (tick_from, tick_to) in e["existence"][branch].iteritems():
            if tick_from <= tick and tick <= tick_to:
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
        if tick_to is None:
            e["indef_exist"][branch] = tick_from
        if branch not in e["existence"]:
            e["existence"][branch] = {}
        e["existence"][branch][tick_from] = tick_to
