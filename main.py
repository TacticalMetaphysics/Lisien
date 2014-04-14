# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE import __path__
from LiSE.gui.app import LiSEApp
from sys import argv
from os.path import sep

import gettext


def lise():
    print(argv)
    dbfn = None

    _ = gettext.translation('LiSE', sep.join([__path__[0], 'localedir']),
                            ['en']).gettext

    if argv[-1][-4:] == "lise":
        dbfn = argv[-1]

    print(_("Starting LiSE with database {}, path {}".format(
        dbfn, __path__[-1])))

    LiSEApp(dbfn=dbfn, gettext=_,
            observer_name='Omniscient',
            observed_name='Player',
            host_name='Physical').run()


if __name__ == '__main__':
    lise()
