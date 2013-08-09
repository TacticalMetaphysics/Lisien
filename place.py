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
        self.dimension = dimension
        self.rumor = self.dimension.rumor
        self.v = v

    def __getattr__(self, attrn):
        if attrn in self.v.attribute_names():
            return self.v[attrn]
        else:
            raise AttributeError(
                "Place instance has no attribute named " + attrn)

    def __contains__(self, that):
        try:
            return that.location.v is self.v
        except:
            return False

    def __int__(self):
        return self.v.index

    def __str__(self):
        return self.v["name"]

    def __repr__(self):
        return str(self.dimension) + "." + str(self)

    def incident(self, mode=OUT):
        return self.dimension.graph.incident(int(self), mode)
