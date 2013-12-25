# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE import __path__
from LiSE.gui.app import LiSEApp
from sys import argv
from os.path import abspath
from os import environ


def lise():
    print(argv)
    lang = "eng"
    dbfn = None

    if "LISELANG" in environ:
        lang = environ["LISELANG"]
    else:
        lang = "eng"

    if argv[-1][-4:] == "lise":
        dbfn = argv[-1]

    print("Starting LiSE with database {}, language {}, path {}".format(
        dbfn, lang, __path__[-1]))

    LiSEApp(dbfn=dbfn, lang=lang,
            lise_path=abspath(__path__[-1]),
            observer_name='Omniscient',
            observed_name='Player',
            host_name='Physical').run()


if __name__ == '__main__':
    lise()
