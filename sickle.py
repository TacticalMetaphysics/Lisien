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


def mkengine(w='LiSEworld.db', caching=True, seed=None):
    return Engine(
        worlddb=w,
        codedb='LiSEcode.db',
        caching=caching,
        random_seed=seed
    )


def sickle_cell_test(
        engine,
        n_creatures=5,
        n_sickles=3,
        malaria_chance=.05,
        mate_chance=.05,
        mapsize=(1, 1),
        startpos=(0, 0),
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
        malaria_chance=malaria_chance,
        n_creatures=n_creatures,
    )
    for n in range(0, n_creatures):
        name = "critter" + str(n)
        stats = {
            'sickle_a': n < n_sickles,
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


# putting dieoff earlier in the code than mate means that dieoff will
# be followed before mate is
    @species.avatar.rule
    def dieoff(engine, character, critter):
        critter.delete()
        assert(not critter.exists)
        assert(critter.name not in critter.character.thing)
        return critter['from_malaria']

    @species.avatar.rule
    def mate(engine, character, critter):
        """If I share my location with another critter, attempt to mate"""
        n = 0
        suitors = list(
            oc for oc in critter.location.contents()
            if oc['male'] != critter['male']
        )
        assert(len(suitors) > 0)
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
        critter['last_mate_tick'] = other_critter['last_mate_tick'] =\
            engine.tick
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

    @mate.trigger
    def in_the_mood(engine, character, critter):
        return engine.random() < character.stat['mate_chance']

    @dieoff.trigger
    def sickle2(engine, character, critter):
        r = critter['sickle_a'] and critter['sickle_b']
        if r:
            critter['from_malaria'] = False
        return r

    @dieoff.trigger
    def malaria(engine, character, critter):
        r = (
            engine.random() < character.stat['malaria_chance'] and not
            (critter['sickle_a'] or critter['sickle_b'])
        )
        if r:
            critter['from_malaria'] = True
        return r

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

    for i in range(0, ticks):
        r = engine.next_tick()
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

clear_off()
with mkengine('LiSEworld.db', seed=69105) as engine:
    run('sickle_cell_test(engine)')
