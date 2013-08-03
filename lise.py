# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
import pyglet
import logging
from rumor import load_game
from sys import argv
from sqlite3 import connect, DatabaseError

logging.basicConfig(level=logging.DEBUG, filename="debug.log")
logger = logging.getLogger()


i = 0
lang = "English"
dbfn = "default.sqlite"
for arg in argv:
    if arg == "-l":
        try:
            lang = argv[i+1]
        except:
            raise Exception("Couldn't parse language")
    elif arg == "-d":
        DEBUG = True
    elif arg[-2:] != "py":
        try:
            connect(arg).cursor().execute("SELECT * FROM game;")
            dbfn = arg
        except DatabaseError:
            print "Couldn't connect to the database named {0}.".format(arg)
    i += 1
tocky = 0.0
rumor = load_game(dbfn, lang)
gw = rumor.load_window('Main')
gamespeed = 0.1

def incdb(ticky):
    global tocky
    tocky += ticky
    while tocky >= gamespeed:
        tocky -= gamespeed
        db.tick += 1
    db.tick += 1
pyglet.clock.schedule_interval(incdb, gamespeed)
pyglet.app.run()
