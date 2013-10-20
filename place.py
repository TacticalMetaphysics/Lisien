# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from igraph import Vertex, OUT


class Place:
    """Where you go when you have to be someplace.

Places have no database records of their own. They are considered to
exist so long as there's a portal from there, a portal to there, or a
thing located there.

    """
    def __init__(self, dimension, v):
        assert(isinstance(v, Vertex))
        if v["name"] == "Physical.Portal(myroom->livingroom)":
            import pdb
            pdb.set_trace()
        self.dimension = dimension
        self.closet = self.dimension.closet
        self.v = v

    def __getattr__(self, attrn):
        if attrn in self.v.attribute_names():
            return self.v[attrn]
        elif attrn == "index":
            return self.v.index
        else:
            raise AttributeError(
                "Place instance has no attribute named " + attrn)

    def __contains__(self, that):
        try:
            return that.location.v is self.v
        except AttributeError:
            return False

    def __int__(self):
        return self.v.index

    def __str__(self):
        return self.v["name"]

    def __repr__(self):
        return "Place({0}.{1})".format(str(self.dimension), self.v["name"])

    def get_contents(self, branch=None, tick=None):
        if branch is None:
            branch = self.dimension.closet.branch
        if tick is None:
            tick = self.dimension.closet.tick
        r = set()
        for thingn in self.dimension.closet.skeleton[
                "thing_location"][unicode(self.dimension)]:
            prev = None
            for rd in self.dimension.closet.skeleton[
                    "thing_location"][unicode(self.dimension)][
                    thingn][branch].iterrows():
                if rd["tick_from"] == tick:
                    if rd["location"] == unicode(self):
                        thing = self.dimension.get_thing(thingn)
                        r.add(thing)
                    break
                elif rd["tick_from"] > tick:
                    if (
                            prev is not None and
                            prev["location"] == unicode(self)):
                        thing = self.dimension.get_thing(thingn)
                        r.add(thing)
                    break
                else:
                    prev = rd
        return r

    def incident(self, mode=OUT):
        return self.dimension.graph.incident(int(self), mode)

    def display_name(self, branch=None, tick=None):
        # Stub.
        #
        # TODO: Look up a display name in a table or dictionary,
        # perhaps using get_text
        return self.v["name"]
