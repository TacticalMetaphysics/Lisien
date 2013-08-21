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
        (prelude, tablenames, finish) = saveables.pop(0)
        if tablenames == []:
            for pre in prelude:
                c.execute(pre)
            for fin in finish:
                c.execute(fin)
            continue
        breakout = False
        try:
            for pre in prelude:
                c.execute(pre)
        except OperationalError:
            saveables.append((prelude, tablenames, finish))
            continue
        while tablenames != []:
            tn = tablenames.pop(0)
            try:
                c.execute(schemata[tn])
                done.add(tn)
            except OperationalError:
                breakout = True
                break
        if breakout:
            saveables.append((prelude, tablenames, finish))
            continue
        try:
            for fin in finish:
                c.execute(fin)
        except OperationalError:
            saveables.append((prelude, tablenames, finish))
            continue
                
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
