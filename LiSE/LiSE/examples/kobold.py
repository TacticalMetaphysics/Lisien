# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""A dwarf hunting a kobold that hides in the bushes.

This script will initialize world.db and the code libraries to run the
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
    assert len(phys.place) == mapsize[0] * mapsize[1]
    kobold = phys.new_thing("kobold", kobold_pos)
    assert len(phys.thing) == 1
    kobold['sprint_chance'] = kobold_sprint_chance
    kobold['_image_paths'] = ['atlas://rltiles/base.atlas/kobold_m']
    dwarf = phys.new_thing("dwarf", dwarf_pos)
    dwarf['sight_radius'] = dwarf_sight_radius
    dwarf['seen_kobold'] = False
    dwarf['_image_paths'] = ['atlas://rltiles/base.atlas/dwarf_m']
    # randomly place the shrubberies and add their locations to shrub_places
    locs = list(phys.place.keys())
    engine.shuffle(locs)
    shrub_places = []
    while len(shrub_places) < shrubberies:
        loc = locs.pop()
        phys.add_thing(
            "shrub" + str(len(shrub_places)),
            loc,
            cover=1,
            _image_paths=['atlas://rltiles/dc-mon.atlas/fungus'],
            _group='shrub'
        )
        shrub_places.append(loc)
    print('{} shrubberies: {}'.format(len(shrub_places), shrub_places))
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
        shrub_places = sorted(list(thing['shrub_places']))
        if thing['location'] in shrub_places:
            shrub_places.remove(thing['location'])
            assert thing['location'] not in shrub_places
        whereto = thing.engine.choice(shrub_places)
        thing.travel_to(whereto)

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
        if thing.next_location is not None:
            thing.engine.info("kobold already travelling to {}".format(thing.next_location))
            return False
        else:
            return True

    @engine.method
    def set_kill_flag(eng):
        eng.character['physical'].thing['dwarf']['kill'] = True

    @dwarf.rule
    def fight(thing):
        method = thing.engine.method
        return "Kill kobold?", [("Kill", method.set_kill_flag), ("Spare", None)]

    @fight.trigger
    def sametile(thing):
        try:
            return (
                thing['location'] == thing.character.thing['kobold']['location']
            )
        except KeyError:
            return False

    @fight.prereq
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

    fight.prereq(aware)

    @dwarf.rule
    def kill_kobold(thing):
        del thing.character.thing['kobold']
        del thing['kill']

    kill_kobold.trigger(kobold_alive)

    @kill_kobold.prereq
    def unmerciful(thing):
        return thing.get('kill', False)

    @dwarf.rule
    def go2kobold(thing):
        thing.travel_to(thing.character.thing['kobold']['location'])

    go2kobold.trigger(aware)

    go2kobold.prereqs = ['kobold_alive']

    @go2kobold.prereq
    def kobold_not_here(thing):
        return 'kobold' not in thing.location.content

    @dwarf.rule
    def wander(thing):
        dests = sorted(list(thing.character.place.keys()))
        dests.remove(thing['location'])
        thing.travel_to(thing.engine.choice(dests))

    @wander.trigger
    def standing_still(thing):
        return thing.next_location is None


if __name__ == '__main__':
    from LiSE.engine import Engine
    from os import remove
    with Engine(random_seed=69105, clear=True) as engine:
        inittest(engine, shrubberies=20, kobold_sprint_chance=.9)
        engine.commit()
        print('shrub_places beginning: {}'.format(
            engine.character['physical'].thing['kobold']['shrub_places']
        ))
