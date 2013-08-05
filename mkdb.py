# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
import os
import re
from sqlite3 import OperationalError
from rltileins import ins_rltiles
from util import schemata
import board, card, calendar, character, dimension, effect, gui, img, menu, pawn, portal, spot, style, thing, rumor

"""Make an empty database of LiSE's schema. By default it will be
called default.sqlite and include the RLTiles (in folder
./rltiles). Put sql files in the folder ./init and they'll be executed
in their sort order, after the schema is defined.

"""


DB_NAME = 'default.sqlite'


def read_sql(db, filen):
    sqlfile = open(filen, "r")
    sql = sqlfile.read()
    sqlfile.close()
    db.c.executescript(sql)

def dumb_effect(db, effn):
    mat = re.match("(.+)\((.*)\)", effn)
    (name, arg) = mat.groups()
    qrystr = "INSERT INTO effect (name, func, arg) VALUES (?, ?, ?)"
    qrytup = (effn, name, arg)
    db.c.execute(qrystr, qrytup)
    

try:
    os.remove(DB_NAME)
except OSError:
    pass

db = rumor.RumorMill(DB_NAME)
db.c.execute(
    "CREATE TABLE game"
    " (front_board TEXT DEFAULT 'Physical', front_branch INTEGER DEFAULT 0, "
    "tick INTEGER DEFAULT 0,"
    " seed INTEGER DEFAULT 0, hi_place INTEGER DEFAULT 0, hi_portal INTEGER"
    " DEFAULT 0, hi_thing INTEGER DEFAULT 0, hi_branch INTEGER DEFAULT 0);")
db.c.execute(
    "CREATE TABLE strings (stringname TEXT NOT NULL, language TEXT NOT"
    " NULL DEFAULT 'English', string TEXT NOT NULL, PRIMARY KEY(stringname,"
    " language));")


done = set()

while schemata != []:
    s = schemata.pop()
    print s
    (tabn, reqs, schema, prelude) = s
    if tabn in done:
        continue
    for req in reqs:
        if req not in done:
            schemata.insert(0, (tabn, reqs, schema, prelude))
            continue
    print "creating " + tabn
    try:
        if prelude is not None:
            db.c.execute(prelude)
        db.c.execute(schema)
        done.add(tabn)
    except OperationalError as oe:
        raise OperationalError(
            str(oe) + " while trying to execute: \n" + schema)

oldhome = os.getcwd()
os.chdir('sql')
initfiles = sorted(os.listdir('.'))
for initfile in initfiles:
    if initfile[-3:] == "sql":  # weed out automatic backups and so forth
        print "reading SQL from file " + initfile
        read_sql(db, initfile)

os.chdir(oldhome)

print "indexing the RLTiles"
ins_rltiles(db.c, 'rltiles')

# print "indexing the dumb effects"
# efns = db.c.execute("SELECT on_click FROM menu_item").fetchall()
# for row in efns:
#     print row[0]
#     dumb_effect(db, row[0])

db.c.close()
db.conn.commit()
