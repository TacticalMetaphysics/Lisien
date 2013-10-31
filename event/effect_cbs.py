# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com


def teleport_to(dimension, thing, destination):
    dimn = unicode(dimension)
    thingn = unicode(thing)
    destn = unicode(destination)

    def teleport(character, branch, tick):
        thing = character.thingdict[dimn][thingn]
        thing.set_location(dimn, destn, branch, tick)
    return teleport


def vaporize_thing(dimension, thing):
    dimn = unicode(dimension)
    thingn = unicode(thing)

    def vaporize(character, branch, tick):
        assert(dimn in character.thingdict and
               thingn in character.thingdict[dimn] and
               branch in character.thingdict[dimn][thingn])
        thing = character.closet.get_thing(dimn, thingn)
        thing.set_location(None, branch, tick)
