# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""A dwarf hunting a kobold that hides in the bushes.

This script will initialize LiSEworld.db and LiSEcode.db to run the
simulation described. To view it, run ELiDE from the same directory
as you ran this script from.

"""


import networkx as nx


def inittest(
        engine,
        mapsize=(10, 10),
        dwarf_pos=(0, 0),
        kobold_pos=(9, 9),
        shrubberies=20,
        dwarf_sight_radius=2,
        kobold_sprint_chance=.1
):
    # initialize world
    phys = engine.new_character('physical', data=nx.grid_2d_graph(*mapsize))
    kobold = phys.new_thing("kobold", kobold_pos)
    kobold['sprint_chance'] = kobold_sprint_chance
    kobold['_image_paths'] = ['atlas://rltiles/base.atlas/kobold_m']
    dwarf = phys.new_thing("dwarf", dwarf_pos)
    dwarf['sight_radius'] = dwarf_sight_radius
    dwarf['seen_kobold'] = False
    dwarf['_image_paths'] = ['atlas://rltiles/base.atlas/dwarf_m']
    # randomly place the shrubberies and add their locations to shrub_places
    n = 0
    # these are sorted as a way to make them shuffle the same whether
    # I'm using the cache or not
    locs = sorted(list(phys.place.keys()))
    engine.shuffle(locs)
    shrub_places = []
    while n < shrubberies:
        loc = locs.pop()
        phys.add_thing(
            "shrub" + str(n),
            loc,
            cover=1,
            _image_paths=['atlas://rltiles/dc-mon.atlas/fungus']
        )
        shrub_places.append(loc)
        n += 1
    print('{} shrubberies: {}'.format(n, shrub_places))
    kobold['shrub_places'] = shrub_places

    # If the kobold is not in a shrubbery, it will try to get to one.
    # If it is, there's a chance it will try to get to another.
    @kobold.rule
    def shrubsprint(engine, character, thing):
        print('shrub_places: {}'.format(thing['shrub_places']))
        shrub_places = sorted(list(thing['shrub_places']))
        if thing['location'] in shrub_places:
            shrub_places.remove(thing['location'])
        print('shrub_places after: {}'.format(thing['shrub_places']))
        thing.travel_to(engine.choice(shrub_places))

    @shrubsprint.trigger
    def uncovered(engine, character, thing):
        for shrub_candidate in thing.location.contents():
            if shrub_candidate.name[:5] == "shrub":
                return False
        engine.info("kobold uncovered")
        return True

    @shrubsprint.trigger
    def breakcover(engine, character, thing):
        if engine.random() < thing['sprint_chance']:
            engine.info("kobold breaking cover")
            return True

    @shrubsprint.prereq
    def not_traveling(engine, character, thing):
        if thing['next_location'] is not None:
            engine.info("kobold already travelling to {}".format(thing['next_location']))
        return thing['next_location'] is None

    @dwarf.rule
    def kill(engine, character, thing):
        character.thing['kobold'].delete()
        print("===KOBOLD DIES===")

    @kill.trigger
    def sametile(engine, character, thing):
        try:
            return (
                thing['location'] == character.thing['kobold']['location']
            )
        except KeyError:
            return False

    @kill.prereq
    def kobold_alive(engine, character, thing):
        return 'kobold' in character.thing

    def aware(engine, character, thing):
        # calculate the distance from dwarf to kobold
        from math import hypot
        try:
            bold = character.thing['kobold']
        except KeyError:
            return False
        (dx, dy) = bold['location']
        (ox, oy) = thing['location']
        xdist = abs(dx - ox)
        ydist = abs(dy - oy)
        dist = hypot(xdist, ydist)
        # if it's <= the dwarf's sight radius, the dwarf is aware of the kobold
        return dist <= thing['sight_radius']

    kill.prereq(aware)

    @dwarf.rule
    def go2kobold(engine, character, thing):
        thing.travel_to(character.thing['kobold']['location'])

    go2kobold.trigger(aware)

    go2kobold.prereqs = ['kobold_alive']

    @dwarf.rule
    def wander(engine, character, thing):
        dests = sorted(list(character.place.keys()))
        dests.remove(thing['location'])
        thing.travel_to(engine.choice(dests))

    @wander.trigger
    def standing_still(engine, character, thing):
        return thing['next_location'] is None


if __name__ == '__main__':
    from utiltest import mkengine, clear_off, seed, caching
    clear_off()
    with mkengine(random_seed=seed, caching=caching) as engine:
        inittest(engine)
        engine.commit()
        print('shrub_places beginning: {}'.format(
            engine.character['physical'].thing['kobold']['shrub_places']
        ))
