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
rumor = load_game(dbfn, lang)
#tw = rumor.get_timestream(300, 900, 10, 1.4, 0, 0)
gw = rumor.load_window('Main')


pyglet.clock.schedule(gw.update)
#pyglet.clock.schedule(tw.update)
pyglet.clock.schedule_interval(gw.rumor.increment_time, gw.rumor.game_speed)
pyglet.app.run()
