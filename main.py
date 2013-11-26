# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
import closet
from gui.app import LiSEApp
from sys import argv
from sqlite3 import connect, DatabaseError

i = 0
lang = "eng"
defdbfn = "default.sqlite"
dbfn = ""
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
            print("Couldn't connect to the database named \"{0}\". "
                  "Defaulting to \"{1}\".".format(arg, defdbfn))
            dbfn = defdbfn
    i += 1
if dbfn == "":
    dbfn = defdbfn


try:
    open(dbfn, 'r')
    conn = connect(dbfn)
except IOError:
    conn = closet.mkdb(dbfn)


closet = closet.load_closet(dbfn, lang, kivy=True)

LiSEApp(closet=closet, menu_name='Main',
        dimension_name='Physical',
        character_name='household').run()
closet.save_game()
closet.end_game()
