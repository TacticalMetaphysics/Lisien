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


def mkengine(w='sqlite:///LiSEworld.db', *args, **kwargs):
    return Engine(
        worlddb=w,
        codedb='LiSEcode.db',
        *args,
        **kwargs
    )


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
    phys.add_thing(
        "kobold",
        kobold_pos
    )
    phys.add_thing(
        "dwarf",
        dwarf_pos
    )
    # since the kobold and the dwarf follow different rulesets they
    # need different Characters.
    kobold = engine.new_character('kobold')
    kobold.add_avatar('physical', 'kobold')
    dwarf = engine.new_character('dwarf')
    dwarf.add_avatar('physical', 'dwarf')
    dwarf.stat['sight_radius'] = dwarf_sight_radius
    dwarf.stat['seen_kobold'] = False
    # the kobold dashes between the shrubberies, so it needs to know
    # where they are
    kobold.stat['shrub_places'] = []
    kobold.stat['sprint_chance'] = kobold_sprint_chance
    # randomly place the shrubberies and add their locations to the list
    n = 0
    locs = list(phys.place.keys())
    engine.shuffle(locs)
    shrub_places = []
    while n < shrubberies:
        loc = locs.pop()
        phys.add_thing(
            "shrub" + str(n),
            loc,
            cover=1
        )
        shrub_places.append(loc)
        n += 1
    kobold.stat['shrub_places'] = shrub_places

    # If the kobold is not in a shrubbery, it will try to get to one.
    # If it is, there's a chance it will try to get to another one, anyway.
    @kobold.avatar.rule
    def shrubsprint(engine, character, avatar):
        """Sprint to a location, other than the one I'm in already, which has
        a shrub in it.

        """
        shrub_places = list(character.stat['shrub_places'])
        if avatar['location'] in shrub_places:
            shrub_places.remove(avatar['location'])
        avatar.travel_to(engine.choice(shrub_places))

    @shrubsprint.trigger
    def uncovered(engine, character, avatar):
        """Return True when I'm *not* in a place with a shrub in it."""
        for shrub_candidate in avatar.location.contents():
            if shrub_candidate.name[:5] == "shrub":
                return False
        return True

    @shrubsprint.trigger
    def breakcover(engine, character, avatar):
        """Return True when I *am* in a place with a shrub in it, but elect to
        sprint anyway.

        """
        # This is checked after uncovered so I don't need to re-check
        # for shrubbery.
        return engine.random() < character.stat['sprint_chance']

    @shrubsprint.prereq
    def notsprinting(engine, character, avatar):
        """Only start a new sprint when not already sprinting."""
        return avatar['next_arrival_time'] is None

    # The dwarf's eyesight is not very good.
    @dwarf.sense
    def sight(engine, observer, seen):
        """A sense to simulate short-range vision that can't see anything
        hiding in a shrubbery.

        This gives poor performance and is overkill for this purpose,
        but demonstrates the concept of a sense adequately.

        """
        assert('sight_radius' in observer.stat)
        from math import hypot
        (dwarfx, dwarfy) = observer.avatar['location']
        for place in list(seen.place.keys()):
            (x, y) = place
            dx = dwarfx - x
            dy = dwarfy - y
            if hypot(dx, dy) > observer.stat['sight_radius']:
                del seen.place[place]
            else:
                del_kobold = False
                for thing in seen.place[place].contents():
                    if thing['name'][:5] == "shrub":
                        for thing in seen.place[place].contents():
                            if thing['name'] == "kobold":
                                del_kobold = True
                                break
                        break
                if del_kobold:
                    del seen.thing['kobold']
                    break
        return seen

    # If the dwarf is on the same spot as the kobold, and is aware of
    # the kobold, kill the kobold.
    @dwarf.avatar.rule
    def kill(engine, character, avatar):
        # the avatar's character is 'physical', and not 'dwarf';
        # character 'dwarf' merely tracks the avatar
        avatar.character.thing['kobold'].delete()
        print("===KOBOLD DIES===")

    @kill.trigger
    def alive(engine, character, avatar):
        return 'kobold' in avatar.character.thing

    @kill.prereq
    def aware(engine, character, avatar):
        return 'kobold' in character.sense['sight']('physical').thing

    @kill.prereq
    def sametile(engine, character, avatar):
        return (
            avatar['location'] == avatar.character.thing['kobold']['location']
        )

    # If the dwarf is not on the same spot as the kobold, but is aware
    # of the kobold, the dwarf tries to get to the kobold.
    @dwarf.avatar.rule
    def go2kobold(engine, character, avatar):
        avatar.travel_to(avatar.character.thing['kobold']['location'])

    # only run while the kobold is alive
    @go2kobold.trigger
    def kobold_lives(engine, character, avatar):
        return 'kobold' in avatar.character.thing.keys()

    # reusing the prereq from the previous rule
    go2kobold.prereqs.append('aware')

    # If I'm not otherwise committed, wander about
    @dwarf.avatar.rule
    def wander(engine, character, avatar):
        dests = list(avatar.character.place.keys())
        dests.remove(avatar['location'])
        avatar.travel_to(engine.choice(dests))

    # Run whenever I'm not moving
    @wander.trigger
    def notmoving(engine, character, avatar):
        return avatar['next_location'] is None


def runtest(engine):
    # run sim for 100 tick
    for n in range(0, 100):
        engine.next_tick()
        kobold_alive = 'kobold' in engine.character['physical'].thing
        print(
            "On tick {}, the dwarf is at {}, "
            "and the kobold is {}{}{}".format(
                n,
                engine.character['dwarf'].avatar['location'],
                "alive" if kobold_alive else "dead",
                " at " + str(engine.character['kobold'].avatar['location'])
                if kobold_alive else "",
                " (in a shrubbery)" if kobold_alive and len([
                    th for th in
                    engine.character['kobold'].avatar.location.contents()
                    if th.name[:5] == "shrub"
                ]) > 0 else ""
            )
        )


if __name__ == '__main__':
    clear_off()
    with mkengine(random_seed=69107, caching=False) as engine:
        inittest(engine)
        engine.commit()
        run('runtest(engine)')
