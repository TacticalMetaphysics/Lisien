# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE import __path__
from LiSE import closet
from LiSE.gui.app import LiSEApp
from sys import argv
from os.path import abspath


def lise():
    i = 0
    lang = "eng"
    dbfn = None
    debugfn = ""
    DEBUG = False
    for arg in argv:
        if arg == "-l":
            try:
                lang = argv[i + 1]
            except:
                raise ValueError("Couldn't parse language")
        elif arg == "-d":
            DEBUG = True
        elif DEBUG:
            debugfn = arg
            try:
                df = open(debugfn, 'w')
                df.write('--begin debug file for LiSE--\n')
                df.close()
            except IOError:
                exit("Couldn't write to the debug log file "
                     "\"{}\".".format(debugfn))
        elif arg[-4:] == "lise":
            dbfn = arg
        i += 1

    print("Starting LiSE with database {}, language {}, path {}".format(
        dbfn, lang, __path__[-1]))

    LiSEApp(dbfn=dbfn, lang=lang,
            lise_path=abspath(__path__[-1]),
            menu_name='Main',
            dimension_name='Physical',
            character_name='household').run()


if __name__ == '__main__':
    lise()
