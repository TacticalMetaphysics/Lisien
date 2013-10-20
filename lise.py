# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
import logging
import closet
from app import LiSEApp
from sys import argv
from os import remove
from sqlite3 import connect, DatabaseError

i = 0
lang = "eng"
dbfn = "default.sqlite"
debugfn = ""
DEBUG = False
for arg in argv:
    if arg == "-l":
        try:
            lang = argv[i + 1]
        except:
            raise Exception("Couldn't parse language")
    elif arg == "-d":
        DEBUG = True
    elif DEBUG:
        debugfn = arg
    elif arg[-2:] != "py":
        try:
            connect(arg).cursor().execute("SELECT * FROM game;")
            dbfn = arg
        except DatabaseError:
            print("Couldn't connect to the database named {0}.".format(arg))
    i += 1
if DEBUG:
    if debugfn == "":
        logging.basicConfig(level=logging.DEBUG)
    else:
        try:
            remove(debugfn)
        except OSError:
            pass
        logging.basicConfig(level=logging.DEBUG, filename=debugfn)
    logger = logging.getLogger()


closet = closet.load_closet(dbfn, lang)

LiSEApp(closet=closet, menu_name='Main',
        dimension_name='Physical',
        character_name='household').run()
closet.save_game()
closet.end_game()
