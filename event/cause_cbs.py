# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Factories for callbacks for Cause."""


def thing_in(dimension, thing, location):
    dimn = unicode(dimension)
    thingn = unicode(thing)
    locn = unicode(location)

    def thing_is_in(character, branch, tick):
        if dimn not in character.thingdict:
            return False
        if thingn not in character.thingdict[dimn]:
            return False
        if branch not in character.thingdict[dimn][thingn]:
            return False
        thing = character.closet.get_thing(dimn, thingn)
        return unicode(thing.get_location(branch, tick)) == locn
    return thing_is_in
