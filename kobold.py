from cProfile import run
from initbold import inittest
from utiltest import mkengine, clear_off, seed


def runtest(engine):
    # run sim for 100 tick
    for n in range(0, 100):
        engine.next_tick()
        kobold_alive = 'kobold' in engine.character['physical'].thing
        print(
            "On tick {}, the dwarf is at {}, "
            "and the kobold is {}{}{}".format(
                n,
                engine.character['physical'].thing['dwarf']['location'],
                "alive" if kobold_alive else "dead",
                " at " + str(engine.character['physical'].thing['kobold']['location'])
                if kobold_alive else "",
                " (in a shrubbery)" if kobold_alive and len([
                    th for th in
                    engine.character['physical'].thing['kobold'].location.contents()
                    if th.name[:5] == "shrub"
                ]) > 0 else ""
            )
        )


if __name__ == '__main__':
    clear_off()
    with mkengine(':memory:', random_seed=seed, caching=False) as engine:
        inittest(engine)
        engine.commit()
        runtest(engine)
