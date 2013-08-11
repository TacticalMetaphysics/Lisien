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
clock = pyglet.clock.Clock()
pyglet.clock.set_default(clock)
rumor = load_game(dbfn, lang)
gw = rumor.load_window('Main')

def update(ts):
    gw.update(ts)

pyglet.clock.schedule(update)
pyglet.clock.schedule_interval(rumor.update, rumor.game_speed)
pyglet.app.run()
