# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from igraph import Graph
from igraph._igraph import InternalError as IgraphError

from re import match

from LiSE.util import portex


"""Class and loaders for dimensions--the top of the world hierarchy."""


class Dimension:
    """Container for a given view on the game world, sharing no things,
places, or portals with any other dimension, but possibly sharing
facades."""
    def __init__(self, facade, label):
        """Return a dimension with the given name.

Probably useless unless, once you're sure you've put all your places,
portals, and things in the db, you call the Dimension object's unravel(db)
method. Thereafter, it will have dictionaries of all those items,
keyed with their names.

        """
        self.facade = facade
        self.label = label
        self.remake_graph()
        self.closet.dimension_d[unicode(self)] = self

    def __str__(self):
        return str(self.label)

    def __unicode__(self):
        return unicode(self.label)

    def add_place_bone(self, bone):
        """Add a vertex for this bone, if I don't have one already. Otherwise
        update it.

        """
        diff = bone.idx + 1 - len(self.graph.vs)
        if diff > 0:
            self.graph.add_vertices(diff)
        self.graph.vs[bone.idx]["bone"] = bone
        self.graph.vs[bone.idx]["name"] = bone.label

    def add_portal_bone(self, bone):
        """Add an edge for this bone, if I don't have one already. Otherwise
        update it.

        """
        if (
                bone.origin in self.graph.vs["name"] and
                bone.destination in self.graph.vs["name"]):
            extant = self.graph.es.select(_within=[
                bone.origin, bone.destination])
            if len(extant) > 0:
                try:
                    e = extant.find(name=bone.label)
                    e["bone"] = bone
                    return
                except IgraphError:
                    pass
        self.graph.add_edge(
            bone.origin,
            bone.destination,
            bone=bone,
            name=bone.label)

    def get_location(self, label):
        try:
            return self.graph.vs.find(name=label)["place"]
        except IgraphError:
            try:
                m = match(portex, label)
                (origin, destination) = m.groups()
                oi = self.graph.vs["name"].index(origin)
                di = self.graph.vs["name"].index(destination)
                ei = self.graph.get_eid(oi, di)
                return self.graph.es[ei]["portal"]
            except (IgraphError, IndexError):
                return self.get_thing(label)

    def populate_graph(self, branch=None, tick=None):
        """Add a Vertex for each Place, and an Edge for each Portal--or, if
        there is one already, update the existing one with the most
        recent information.

        The Places and Portals that exist at the present diegetic time
        will be used, unless you specify some other time in the
        optional parameters ``branch`` and ``tick``.

        """
        for place_bone in self.facade.iter_place_bones(branch, tick):
            self.add_place_bone(place_bone)
        for portal_bone in self.facade.iter_portal_bones(branch, tick):
            self.add_portal_bone(portal_bone)

    def remake_graph(self, branch=None, tick=None):
        """Throw out the old graph and make a new one."""
        self.graph = Graph(directed=True)
        self.populate_graph(branch, tick)

    def update_graph(self, branch=None, tick=None):
        """Delete vertices and edges for Places and Portals that do not
        presently exist. Then populate.

        """
        dead_vs = []
        for v in self.graph.vs:
            try:
                place_bone = self.facade.get_place_bone(
                    v["name"], branch, tick)
                v["bone"] = place_bone
            except KeyError:
                # This place doesn't exist anymore; neither should its vertex.
                dead_vs.append(v.index)
        self.graph.delete_vertices(dead_vs)
        dead_es = []
        for e in self.graph.es:
            try:
                portal_bone = self.facade.get_portal_bone(
                    e["name"], branch, tick)
                e["bone"] = portal_bone
            except KeyError:
                # This portal doesn't exist anymore; neither should its edge.
                dead_es.append(e.index)
        self.graph.delete_edges(dead_es)
        self.populate_graph(branch, tick)

    def get_igraph_layout(self, layout_type):
        """Return a Graph layout, of the kind that igraph uses, representing
this dimension, and laid out nicely."""
        return self.graph.layout(layout=layout_type)
