# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE import __path__
from sys import argv
from os.path import sep

import gettext
import argparse
import shelve


parser = argparse.ArgumentParser(
    description='Pick a database and UI')
parser.add_argument('-w', '--world')
parser.add_argument('-s', '--shelf')
parser.add_argument('--gui', action='store_true')
parser.add_argument('maindotpy')


def lise():
    print(argv)

    _ = gettext.translation('LiSE', sep.join([__path__[0], 'localedir']),
                            ['en']).gettext
    parsed = parser.parse_args(argv)

    print(_("Starting LiSE with world {}, shelf {}, path {}".format(
        parsed.world, parsed.shelf, __path__[-1])))

    if parsed.gui:
        # start up the gui
        from gui.app import LiSEApp
        LiSEApp(dbfn=parsed.file, gettext=_,
                observer_name='Omniscient',
                observed_name='Player',
                host_name='Physical').run()
    else:
        print("I'll implement a proper command line interface eventually. "
              "For now, running unit tests.")
        import os
        from LiSE.orm import Closet
        from LiSE.data import mkdb
        dbfn = parsed.world if parsed.world else "lise.world"
        shfn = parsed.shelf if parsed.shelf else "lise.shelf"
        try:
            os.remove(dbfn)
        except OSError:
            pass
        try:
            os.remove(shfn)
        except OSError:
            pass

        print("Initializing database.")
        conn = mkdb(dbfn, __path__[-1], kivy=False)
        print("Loading closet.")
        closet = Closet(connector=conn, shelf=shelve.open(shfn))
        closet.load_characters([
            'Omniscient',
            'Physical',
            'Player'])
        print("Loaded successfully.")
        print("Creating places and portals.")
        phys = closet.get_character("Physical")
        phys.make_place("Home")
        phys.make_place("Work")
        phys.make_portal("Home", "Work")
        phys.make_portal("Work", "Home")
        print("Created places and portals.")
        print("Creating NPC.")
        npc = closet.make_character("NonPlayer")
        avatar = phys.make_thing("npc")
        avatar["location"] = "Home"
        npc.add_avatar(avatar)
        print("Created avatar with location.")
        closet.timestream.tick = 1
        npc.avatars["Physical"].travel_to("Work")
        closet.timestream.tick = 100
        npc.avatars["Physical"].travel_to("Home")
        print("Scheduled 1 commute")
        for i in xrange(1, 110):
            closet.timestream.tick = i
            print("Location at tick {}: {}".format(
                closet.timestream.tick,
                npc.avatars["Physical"]["location"]))
        # negative ticks are legal; they are intended for eg. world
        # generation prior to game-start, but aren't treated specially
        closet.timestream.tick = -1
        closet.timestream.branch = 1
        print("Switched to branch 1")
        print("Creating rules for commute")

        @npc.rule
        def home2work(npc):
            """Rule to schedule a new trip to work every day."""
            # arrange to get to work by 9 o'clock
            npc.avatars["Physical"].travel_to_by(
                "Work",
                npc.closet.timestream.tick+9)

        h2w = npc.rules["home2work"]

        @h2w.prereq
        def daystart(npc, rule):
            """Run at midnight only."""
            return npc.closet.timestream.tick % 24 == 0

        @h2w.prereq
        def home_all_day(npc, rule):
            """Run if I'm scheduled to be at Home for this tick and the
            following twenty-four.

            """
            present = npc.closet.timestream.tick
            for t in xrange(present, present+24):
                # The branch and tick pointers will be reset by the
                # event handler once this function returns, don't
                # worry.
                npc.closet.timestream.tick = t
                if npc.avatars["Physical"]["location"] != "Home":
                    return False
            return True

        @npc.rule
        def work2home(npc):
            """Rule to go home when work's over, at 5 o'clock."""
            # Leave, go home, arrive whenever
            npc.avatars["Physical"].travel_to("Home")

        w2h = npc.rules["work2home"]

        @w2h.prereq
        def closing_time(npc, rule):
            """Run at 5pm only."""
            return npc.closet.timestream.tick % 24 == 17

        @w2h.prereq
        def at_work(npc, rule):
            """Run only when I'm at Work."""
            return npc.avatars["Physical"]["location"] == "Work"
        print("Testing rules.\n---")
        print("Ideal case:")
        # Run the clock for a few days
        for t in xrange(0, 72):
            closet.timestream.tick = t
            print("At tick {}, NPC's location is {}".format(
                t, npc.avatars["Physical"]["location"]))
        closet.timestream.tick = -1
        closet.timestream.branch = 2
        

if __name__ == '__main__':
    lise()
