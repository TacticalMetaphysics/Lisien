import networkx as nx
from LiSE import Engine
from os import remove
from cProfile import run

def clear_off():
    for fn in ('LiSEworld.db', 'LiSEcode.db'):
        try:
            remove(fn)
        except OSError:
            pass

def mkengine(w='LiSEworld.db'):
    return Engine(
        worlddb=w,
        codedb='LiSEcode.db'
    )

def sickle_cell_test(
        engine,
        n_creatures=5,
        malaria_chance=.05,
        migrate_chance=.5,
        mate_chance=.1,
        mapsize=(6, 6),
        startpos=(3, 3),
        ticks=100
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
        migrate_chance=migrate_chance,
        n_creatures=n_creatures
    )
    for n in range(0, n_creatures):
        name = "critter" + str(n)
        stats = {
            'sickle_a': True,
            'sickle_b': False,
            'male': engine.coinflip(),
            'last_mate_tick': -1
        }
        phys.add_thing(
            name,
            startpos,
            **stats
        )
        thing = phys.thing[name]
        for stat in stats:
            assert(stat in thing.keys())
            assert(stat in thing)
            assert(thing[stat] == stats[stat])
        species.add_avatar("physical", name)

    @species.avatar.rule
    def mate(engine, character, critter):
        """If I share my location with another critter, attempt to mate"""
        n = 0
        suitors = list(
            oc for oc in critter.location.contents()
            if oc['male'] != critter['male']
        )
        other_critter = engine.choice(suitors)
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
            last_mate_tick=engine.tick
        )
        species.add_avatar("physical", name)
        critter['last_mate_tick'] = other_critter['last_mate_tick'] = engine.tick
        n += 1
        return n

    @mate.prereq
    def once_per_tick(engine, character, critter):
        return critter['last_mate_tick'] < engine.tick

    @mate.prereq
    def mate_present(engine, character, critter):
        for oc in critter.location.contents():
            if oc['male'] != critter['male']:
                return True
        return False

    @mate.prereq
    def in_the_mood(engine, character, critter):
        return engine.random() < character.stat['mate_chance']

    for i in range(0, ticks):
        print(engine.next_tick())

clear_off()
with mkengine(':memory:') as engine:
    run('sickle_cell_test(engine)')
