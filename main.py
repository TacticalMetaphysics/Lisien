# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE import LiSE, __path__
from sys import argv
import os
from os.path import sep

import gettext
import argparse
import sqlite3
import anydbm


parser = argparse.ArgumentParser(
    description='Pick a database and UI')
parser.add_argument('-w', '--world')
parser.add_argument('-c', '--code')
parser.add_argument('--gui', action='store_true')
parser.add_argument('maindotpy')


def lise():
    print(argv)

    _ = gettext.translation('LiSE', sep.join([__path__[0], 'localedir']),
                            ['en']).gettext
    parsed = parser.parse_args(argv)

    print(_("Starting LiSE with world {}, code {}, path {}".format(
        parsed.world, parsed.code, __path__[-1])))

    if parsed.gui:
        # start up the gui
        print("GUI not implemented yet")
    else:
        print("I'll implement a proper command line interface eventually. "
              "For now, running unit tests.")
        dbfn = parsed.world if parsed.world else "lise.world"
        codefn = parsed.code if parsed.code else "lise.code"
        try:
            os.remove(dbfn)
        except OSError:
            pass
        try:
            os.remove(codefn)
        except OSError:
            pass
        print("Initializing database.")
        lise = LiSE(
            world_filename=dbfn,
            code_filename=codefn,
            gettext=_
        )
        print("Creating places and portals.")
        lise.tick = -1
        phys = lise.new_character("Physical")
        phys.add_place("Home")
        phys.add_place("Work")
        phys.add_portal("Home", "Work")
        phys.add_portal("Work", "Home")
        print("Created places and portals.")
        print("Creating NPC.")
        npc = lise.new_character("NonPlayer")
        npc.add_avatar("npc", "Physical", "Home")
        print("Created avatar with location.")
        lise.tick = 1
        npc.avatar["Physical"]["npc"].travel_to("Work")
        lise.tick = 100
        npc.avatar["Physical"]["npc"].travel_to("Home")
        print("Scheduled 1 commute")
        for i in xrange(1, 110):
            lise.tick = i
            print("Location at tick {}: {}".format(
                lise.tick,
                npc.avatar["Physical"]["npc"]["location"]))
        # negative ticks are legal; they are intended for eg. world
        # generation prior to game-start, but aren't treated specially
        lise.tick = -1
        lise.branch = 'test'
        print("Switched to branch 'test'")
        print("Creating rules for commute")

        @npc.rule
        def home2work(lise, rule, npc):
            """Rule to schedule a new trip to work by 9 o'clock."""
            # arrange to get to work by 9 o'clock
            print("Will go to work in 9 hours")
            npc.avatar["Physical"].travel_to_by(
                "Work",
                lise.tick+9)

        h2w = npc.rule["home2work"]

        @h2w.prereq
        def daystart(lise, rule, npc):
            """Run at midnight only."""
            return lise.tick % 24 == 0

        @h2w.prereq
        def home_all_day(lise, rule, npc):
            """Run if I'm scheduled to be at Home for this tick and the
            following twenty-four.

            """
            present = lise.tick
            for t in xrange(present, present+24):
                # The branch and tick pointers will be reset by the
                # event handler once this function returns, don't
                # worry.
                lise.tick = t
                # I really only have one avatar, but doing it this way
                # I don't need to care what its name is, and it
                # handles the case where I have more than one avatar.
                for avatar in npc.avatar["Physical"].values():
                    if avatar["location"] != "Home":
                        return False
            print("Home all day.")
            return True

        @npc.rule
        def work2home(lise, rule, npc):
            """Rule to go home when work's over, at 5 o'clock."""
            # Leave, go home, arrive whenever
            print("Leaving work now.")
            npc.avatar["Physical"].travel_to("Home")

        w2h = npc.rule["work2home"]

        @w2h.prereq
        def closing_time(lise, rule, npc):
            """Run at 5pm only."""
            return lise.tick % 24 == 17

        @w2h.prereq
        def at_work(lise, rule, npc):
            """Run only when I'm at Work."""
            return npc.avatar["Physical"]["location"] == "Work"
        print("Testing rules.\n---")
        print("Ideal case:")
        # Run the clock for a few days
        lise.tick = -1
        while lise.tick < 76:
            now = lise.tick
            lise.advance()
            if lise.tick != now:
                print("Location at tick {}: {}".format(lise.tick, npc.avatar["Physical"]["location"]))

if __name__ == '__main__':
    lise()
