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
    locs = list(phys.place.keys())
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

    # Basic day night cycle
    @engine.action
    def time_slipping(character, *, daylen: int, nightlen: int, twilightlen: float=0.0):
        if 'hour' not in character.stat:
            character.stat['hour'] = 0
            character.stat['day_period'] = 'dawn' if twilightlen else 'day'
            return
        twi_margin = twilightlen / 2
        hour = character.stat['hour'] = (character.stat['hour'] + 1) % (daylen + nightlen)
        if twilightlen:
            if hour < twi_margin or hour > daylen + nightlen - twi_margin:
                character.stat['day_period'] = 'dawn'
            elif twi_margin < hour < daylen - twi_margin:
                character.stat['day_period'] = 'day'
            elif daylen - twi_margin < hour < daylen + twi_margin:
                character.stat['day_period'] = 'dusk'
            else:
                character.stat['day_period'] = 'night'
        else:
            character.stat['day_period'] = 'day' if hour < daylen else 'night'


    # If the kobold is not in a shrubbery, it will try to get to one.
    # If it is, there's a chance it will try to get to another.
    @kobold.rule
    def shrubsprint(thing):
        print('shrub_places: {}'.format(thing['shrub_places']))
        shrub_places = sorted(list(thing['shrub_places']))
        if thing['location'] in shrub_places:
            shrub_places.remove(thing['location'])
        print('shrub_places after: {}'.format(thing['shrub_places']))
        thing.travel_to(thing.engine.choice(shrub_places))

    @shrubsprint.trigger
    def uncovered(thing):
        for shrub_candidate in thing.location.contents():
            if shrub_candidate.name[:5] == "shrub":
                return False
        thing.engine.info("kobold uncovered")
        return True

    @shrubsprint.trigger
    def breakcover(thing):
        if thing.engine.random() < thing['sprint_chance']:
            thing.engine.info("kobold breaking cover")
            return True

    @shrubsprint.prereq
    def not_traveling(thing):
        if thing['next_location'] is not None:
            thing.engine.info("kobold already travelling to {}".format(thing['next_location']))
        return thing['next_location'] is None

    @dwarf.rule
    def kill(thing):
        thing.character.thing['kobold'].delete()
        print("===KOBOLD DIES===")

    @kill.trigger
    def sametile(thing):
        try:
            return (
                thing['location'] == thing.character.thing['kobold']['location']
            )
        except KeyError:
            return False

    @kill.prereq
    def kobold_alive(thing):
        return 'kobold' in thing.character.thing

    def aware(thing):
        # calculate the distance from dwarf to kobold
        from math import hypot
        try:
            bold = thing.character.thing['kobold']
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
    def go2kobold(thing):
        thing.travel_to(thing.character.thing['kobold']['location'])

    go2kobold.trigger(aware)

    go2kobold.prereqs = ['kobold_alive']

    @dwarf.rule
    def wander(thing):
        dests = sorted(list(thing.character.place.keys()))
        dests.remove(thing['location'])
        thing.travel_to(thing.engine.choice(dests))

    @wander.trigger
    def standing_still(thing):
        return thing['next_location'] is None


if __name__ == '__main__':
    from LiSE.engine import Engine
    from os import remove
    try:
        remove('LiSEworld.db')
    except FileNotFoundError:
        pass
    with Engine('LiSEworld.db', random_seed=69105) as engine:
        inittest(engine, shrubberies=20)
        engine.commit()
        print('shrub_places beginning: {}'.format(
            engine.character['physical'].thing['kobold']['shrub_places']
        ))
