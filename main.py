# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE import __path__
from sys import argv
from os.path import sep

import gettext
import argparse


parser = argparse.ArgumentParser(
    description='Pick a database and UI')
parser.add_argument('-f', '--file')
parser.add_argument('--no-gui', action='store_false')
parser.add_argument('maindotpy')


def lise():
    print(argv)

    _ = gettext.translation('LiSE', sep.join([__path__[0], 'localedir']),
                            ['en']).gettext
    parsed = parser.parse_args(argv)

    print(_("Starting LiSE with database {}, path {}".format(
        parsed.file, __path__[-1])))

    if parsed.no_gui:
        # start up the gui
        from LiSE.gui.app import LiSEApp
        LiSEApp(dbfn=parsed.file, gettext=_,
                observer_name='Omniscient',
                observed_name='Player',
                host_name='Physical').run()
    else:
        print("I'll implement a proper command line interface eventually. "
              "For now, running unit tests.")
        from LiSE.orm import mkdb, load_closet
        dbfn = parsed.file if parsed.file else "lisetest.sqlite"
        from sqlite3 import connect, OperationalError

        with connect(dbfn) as conn:
            try:
                print("Testing database connectivity.")
                for tab in "thing", "place", "portal":
                    conn.execute("SELECT * FROM {};".format(tab))
            except (IOError, OperationalError):
                print("Database not ready; initializing it.")
                mkdb(dbfn, __path__[-1], kivy=False)
        print("Loading closet.")
        closet = load_closet(
            dbfn, _,
            load_characters=['Omniscient', 'Player', 'Physical'])
        print("Loaded successfully.")


if __name__ == '__main__':
    lise()
