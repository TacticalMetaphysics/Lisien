# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
class Place:
    """Where you go when you have to be someplace.

Places have no database records of their own. They are considered to
exist so long as there's a portal from there, a portal to there, or a
thing located there.

    """
    def __init__(self, dimension, name):
        self.name = name
        self.dimension = dimension
        self.db = self.dimension.db
        self.contents = set()

    def __contains__(self, that):
        return that.location == self

    def __int__(self):
        return self.dimension.places_by_name.values().index(self)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __getattr__(self, attrn):
        if attrn == 'portals':
            if str(self) in self.dimension.portals_by_orign_destn:
                return self.dimension.portals_by_orign_destn[str(self)].itervalues()
            else:
                return iter([])
        else:
            raise AttributeError("Place has no attribute named " + attrn)

    def update_contents(self):
        for thing in self.dimension.things:
            if thing.location == self:
                self.contents.add(thing)
            else:
                self.contents.discard(thing)

    def save(self):
        pass
