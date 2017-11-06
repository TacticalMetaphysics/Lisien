# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Sickle cell anemia vs. malaria, a classic example of population genetics.

This script will initialize LiSEworld.db and LiSEcode.db to run the
simulation described. To view it, run ELiDE from the same directory
as you ran this script from.

"""


import networkx as nx
from LiSE import Engine
from os import remove


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
        species.add_avatar("physical", name)


# putting dieoff earlier in the code than mate means that dieoff will
# be followed before mate is
    @species.avatar.rule
    def dieoff(critter):
        critter.delete()
        assert(not critter.exists)
        assert(critter.name not in critter.character.thing)
        return critter['from_malaria']

    @species.avatar.rule
    def mate(critter):
        """If I share my location with another critter, attempt to mate"""
        n = 0
        suitors = list(
            oc for oc in critter.location.contents()
            if oc['male'] != critter['male']
        )
        assert(len(suitors) > 0)
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
        critter['last_mate_turn'] = other_critter['last_mate_turn'] =\
            engine.turn
        n += 1
        return n

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
        return critter.engine.random() < critter.character.stat['mate_chance']

    @dieoff.trigger
    def sickle2(critter):
        r = critter['sickle_a'] and critter['sickle_b']
        if r:
            critter['from_malaria'] = False
        return r

    @dieoff.trigger
    def malaria(critter):
        r = (
            critter.engine.random() < critter.character.stat['malaria_chance'] and not
            (critter['sickle_a'] or critter['sickle_b'])
        )
        if r:
            critter['from_malaria'] = True
        return r

    # it would make more sense to keep using species.avatar.rule, this
    # is just a test
    @phys.thing.rule
    def wander(critter):
        dest = critter.engine.choice(list(critter.character.place.keys()))
        critter.travel_to(dest)

    @wander.trigger
    def not_travelling(critter):
        return critter['next_location'] is None

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
        r = engine.next_turn()
        print("On tick {}, {} critters were born; "
              "{} died of malaria, and {} of sickle cell anemia, "
              "leaving {} alive.".format(
                  i,
                  sum(tup[0][0] for tup in r if tup[0] and tup[1] == 'mate'),
                  sum(
                      1 for tup in r if tup[0] and
                      tup[0][0] and tup[1] == 'dieoff'
                  ),
                  sum(
                      1 for tup in r if tup[0] and not
                      tup[0][0] and tup[1] == 'dieoff'
                  ),
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

try:
    remove('LiSEworld.db')
except OSError:
    pass
with Engine('LiSEworld.db', random_seed=69105) as engine:
    sickle_cell_test(engine)
