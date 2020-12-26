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
"""Sickle cell anemia vs. malaria, a classic example of population genetics.

This script will initialize world.db and the game code libraries to run the
simulation described. To view it, run ELiDE from the same directory
as you ran this script from.

"""


import networkx as nx
from LiSE import Engine
from os import remove


def install(
        engine,
        n_creatures=5,
        n_sickles=3,
        malaria_chance=.05,
        mate_chance=.05,
        mapsize=(1, 1),
        startpos=(0, 0)
):
    """Natural Selection on Sickle Cell Anemia

    If anyone carries a pair of sickle betaglobin genes, they die of
    sickle cell anemia.

    Individuals with 1x betaglobin, 1x sickle betaglobin are immune to
    malaria.

    """
    initmap = nx.grid_2d_graph(*mapsize)
    phys = engine.new_character("physical", data=initmap)
    species = engine.new_character(
        "species",
        mate_chance=mate_chance,
        malaria_chance=malaria_chance,
        n_creatures=n_creatures,
    )
    for n in range(0, n_creatures):
        name = "critter" + str(n)
        phys.add_thing(
            name=name,
            location=startpos,
            sickle_a=(n < n_sickles),
            sickle_b=False,
            male=engine.coinflip(),
            last_mate_turn=-1
        )
        assert name in phys.thing
        assert name not in phys.place
        assert name in phys.node, "couldn't add node {} to phys.node".format(name)
        assert hasattr(phys.node[name], 'location')
        species.add_avatar("physical", name)
        assert hasattr(species.avatar['physical'][name], 'location')

    # putting dieoff earlier in the code than mate means that dieoff will
    # be followed before mate is
    @species.avatar.rule
    def dieoff(critter):
        critter.delete()
        assert (critter.name not in critter.character.node)
        if critter['from_malaria']:
            return 'malaria'
        else:
            return 'anemia'

    @species.avatar.rule
    def mate(critter):
        """If I share my location with another critter, attempt to mate"""
        suitors = list(
            oc for oc in critter.location.contents()
            if oc['male'] != critter['male']
        )
        assert (len(suitors) > 0)
        other_critter = critter.engine.choice(suitors)
        sickles = [
            critter['sickle_a'],
            critter['sickle_b'],
            other_critter['sickle_a'],
            other_critter['sickle_b']
        ]
        engine.shuffle(sickles)
        name = "critter" + str(species.stat["n_creatures"])
        species.stat["n_creatures"] += 1
        engine.character["physical"].add_thing(
            name,
            critter["location"],
            sickle_a=sickles.pop(),
            sickle_b=sickles.pop(),
            male=engine.coinflip(),
            last_mate_turn=engine.turn
        )
        species.add_avatar("physical", name)
        critter['last_mate_turn'] = other_critter['last_mate_turn'] = \
            engine.turn
        return 'mated'

    @mate.prereq
    def once_per_turn(critter):
        return critter['last_mate_turn'] < critter.engine.turn

    @mate.prereq
    def mate_present(critter):
        for oc in critter.location.contents():
            if oc['male'] != critter['male']:
                return True
        return False

    @mate.trigger
    def in_the_mood(critter):
        return critter.engine.random() < critter.user.stat['mate_chance']

    @dieoff.trigger
    def sickle2(critter):
        r = critter['sickle_a'] and critter['sickle_b']
        if r:
            critter['from_malaria'] = False
        return r

    @dieoff.trigger
    def malaria(critter):
        r = (
                critter.engine.random() < critter.user.stat['malaria_chance'] and not
        (critter['sickle_a'] or critter['sickle_b'])
        )
        if r:
            critter['from_malaria'] = True
        return r

    # it would make more sense to keep using species.avatar.rule, this
    # is just a test
    @phys.thing.rule
    def wander(critter):
        dests = list(critter.character.place.keys())
        dests.remove(critter['location'])
        dest = critter.engine.choice(dests)
        critter.travel_to(dest)

    @wander.trigger
    def not_travelling(critter):
        return critter.next_location is None

    @wander.prereq
    def big_map(critter):
        return len(critter.character.place) > 1


def sickle_cell_test(
        engine,
        n_creatures=5,
        n_sickles=3,
        malaria_chance=.05,
        mate_chance=.05,
        mapsize=(1, 1),
        startpos=(0, 0),
        turns=100
):
    install(engine, n_creatures, n_sickles, malaria_chance, mate_chance, mapsize, startpos)
    species = engine.character['species']
    print(
        "Starting with {} creatures, of which {} have "
        "at least one sickle betaglobin.".format(
            len(species.avatar['physical']),
            sum(
                1 for critter in species.avatar['physical'].values()
                if critter['sickle_a'] or critter['sickle_b']
            )
        )
    )

    for i in range(0, turns):
        malaria_dead = 0
        anemia_dead = 0
        born = 0
        while engine.turn < i:
            r = engine.next_turn()
            if not r:
                continue
            r = r[0]
            if isinstance(r, Exception):
                raise r
            if 'malaria' in r:
                malaria_dead += 1
            if 'anemia' in r:
                anemia_dead += 1
            if 'mated' in r:
                born += 1
        print("On turn {}, {} critters were born; "
              "{} died of malaria, and {} of sickle cell anemia, "
              "leaving {} alive.".format(
                  i, born, malaria_dead, anemia_dead,
                  len(engine.character['species'].avatar['physical'])
              ))
    print(
        "Of the remaining {} creatures, {} have a sickle betaglobin.".format(
            len(species.avatar['physical']),
            sum(
                1 for critter in species.avatar['physical'].values()
                if critter['sickle_a'] or critter['sickle_b']
            )
        )
    )


if __name__ == '__main__':
    with Engine(random_seed=69105, clear=True) as engine:
        sickle_cell_test(engine)
