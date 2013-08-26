# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
import os
from sqlite3 import connect, OperationalError
from rltileins import ins_rltiles
from util import schemata, saveables
import board, card, calendar, character, dimension, effect, gui
import img, menu, pawn, portal, spot, style, thing, rumor

"""Make an empty database of LiSE's schema. By default it will be
called default.sqlite and include the RLTiles (in folder
./rltiles). Put sql files in the folder ./init and they'll be executed
in their sort order, after the schema is defined.

"""


def mkdb(DB_NAME='default.sqlite'):
    try:
        os.remove(DB_NAME)
    except OSError:
        pass
    conn = connect(DB_NAME)
    c = conn.cursor()
    
    def read_sql(filen):
        sqlfile = open(filen, "r")
        sql = sqlfile.read()
        sqlfile.close()
        c.executescript(sql)
    
    c.execute(
        "CREATE TABLE game"
        " (front_board TEXT DEFAULT 'Physical', front_branch INTEGER DEFAULT 0, "
        "tick INTEGER DEFAULT 0,"
        " seed INTEGER DEFAULT 0, hi_place INTEGER DEFAULT 0, hi_portal INTEGER"
        " DEFAULT 0, hi_thing INTEGER DEFAULT 0, hi_branch INTEGER DEFAULT 0);")
    c.execute(
        "CREATE TABLE strings (stringname TEXT NOT NULL, language TEXT NOT"
        " NULL DEFAULT 'English', string TEXT NOT NULL, PRIMARY KEY(stringname,"
        " language));")
    
    
    done = set()
    while saveables != []:
        (demands, provides, prelude, tablenames, postlude) = saveables.pop(0)
        print tablenames
        if 'character_things' in tablenames:
            pass
        breakout = False
        for demand in iter(demands):
            if demand not in done:
                saveables.append((demands, provides, prelude, tablenames, postlude))
                breakout = True
                break
        if breakout:
            continue
        if tablenames == []:
            while prelude != []:
                pre = prelude.pop()
                if isinstance(pre, tuple):
                    c.execute(*pre)
                else:
                    c.execute(pre)
            while postlude != []:
                post = postlude.pop()
                if isinstance(post, tuple):
                    c.execute(*post)
                else:
                    c.execute(post)
            continue
        try:
            while prelude != []:
                pre = prelude.pop()
                if isinstance(pre, tuple):
                    c.execute(*pre)
                else:
                    c.execute(pre)
        except OperationalError as e:
            print "OperationalError during prelude to {0}:".format(tn)
            print e
            saveables.append((demands, provides, prelude, tablenames, postlude))
            continue
        breakout = False
        while tablenames != []:
            tn = tablenames.pop(0)
            if tn == "calendar":
                pass
            try:
                c.execute(schemata[tn])
                done.add(tn)
            except OperationalError as e:
                print "OperationalError while creating table {0}:".format(tn)
                print e
                breakout = True
                break
        if breakout:
            saveables.append((demands, provides, prelude, tablenames, postlude))
            continue
        try:
            while postlude != []:
                post = postlude.pop()
                if isinstance(post, tuple):
                    c.execute(*post)
                else:
                    c.execute(post)
        except OperationalError as e:
            print "OperationalError during postlude from {0}:".format(tn)
            print e
            saveables.append((demands, provides, prelude, tablenames, postlude))
            continue
        done.update(provides)
                
    oldhome = os.getcwd()
    os.chdir('sql')
    initfiles = sorted(os.listdir('.'))
    for initfile in initfiles:
        if initfile[-3:] == "sql":  # weed out automatic backups and so forth
            print "reading SQL from file " + initfile
            read_sql(initfile)
    
    os.chdir(oldhome)
    
    print "indexing the RLTiles"
    ins_rltiles(c, 'rltiles')
    
    # print "indexing the dumb effects"
    # efns = db.c.execute("SELECT on_click FROM menu_item").fetchall()
    # for row in efns:
    #     print row[0]
    #     dumb_effect(db, row[0])
    
    c.close()
    conn.commit()

if __name__ == "__main__":
    mkdb()
