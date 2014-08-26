import networkx as nx
from LiSE import Engine
from os import remove


def clear_off():
    for fn in ('LiSEWorld.db', 'LiSEcode.db'):
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


def kobold_hunt_test(
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
    while n < shrubberies:
        loc = locs.pop()
        phys.add_thing(
            "shrub" + str(n),
            loc,
            cover=1
        )
        # I suspect this might not save the list correctly
        kobold.stat['shrub_places'].append(loc)
        n += 1

    # If the kobold is not in a shrubbery, it will try to get to one.
    # If it is, there's a chance it will try to get to another one, anyway.
    @kobold.avatar.rule
    def shrubsprint(engine, character, avatar):
        """Sprint to a location, other than the one I'm in already, which has
        a shrub in it.

        """
        shrub_places = character.stat['shrub_places']
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
    def sight(engine, observer, observed):
        from math import hypot
        r = observer.stat['sight_radius']
        seen = observed.copy()
        (dwarfx, dwarfy) = observer.avatar['location']
        for place in list(seen.place.keys()):
            (x, y) = place
            dx = dwarfx - x
            dy = dwarfy - y
            if hypot(dx, dy) > r:
                # Ought to remove any things here, and any connected portals.
                # Ought to cancel any movement destined here.
                del seen.place[place]
            else:
                cont = list(observed.place[place].contents())
                # is there a shrub?
                shrub = False
                for thing in cont:
                    if thing.name[:5] == "shrub":
                        shrub = True
                        break
                # is there a kobold?
                kobold = False
                for thing in cont:
                    if thing.name == "kobold":
                        kobold = True
                        break
                if shrub:
                    if kobold:
                        if 'kobold' in seen.thing:
                            del seen.thing['kobold']
                    # when the kobold disappears into shrubbery the
                    # dwarf forgets it was ever there
                    observer.stat['seen_kobold'] = False
                else:
                    if kobold:
                        observer.stat['seen_kobold'] = True
        return seen

    # If the dwarf is on the same spot as the kobold, and is aware of
    # the kobold, kill the kobold.
    @dwarf.avatar.rule
    def kill(engine, character, avatar):
        # the avatar's character is 'physical', and not 'dwarf';
        # character 'dwarf' merely tracks the avatar
        del avatar.character.thing['kobold']

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

    # run sim for 100 tick
    for n in range(0, 100):
        engine.next_tick()
        kobold_alive = 'kobold' in engine.character['physical'].thing
        print(
            "On tick {}, the kobold is {}".format(
                n,
                "alive" if kobold_alive else "dead"
            )
        )


if __name__ == '__main__':
    clear_off()
    kobold_hunt_test(mkengine(seed=69105))
