# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from __future__ import unicode_literals
import os
import sqlite3
from sys import argv


def isdir(p):
    try:
        os.chdir(p)
        return True
    except:
        return False


def allsubdirs_core(doing, done):
    if len(doing) == 0:
        return done
    here = doing.pop()
    if isdir(here):
        done.add(here + '/')
        inside = (
            [here + '/' + there for there in
             os.listdir(here) if there[0] != '.'])
        doing.update(set(inside))


def allsubdirs(path):
    inpath = os.path.realpath(path)
    indoing = set()
    indoing.add(inpath)
    indone = set()
    result = None
    while result is None:
        result = allsubdirs_core(indoing, indone)
    return iter(result)


def ins_rltiles(curs, dirname):
    here = os.getcwd()
    directories = os.path.abspath(dirname).split("/")
    home = "/".join(directories[:-1]) + "/"
    dirs = allsubdirs(dirname)
    for dir in dirs:
        for bmp in os.listdir(dir):
            if bmp[-4:] != ".bmp":
                continue
            qrystr = """insert or replace into img
(name, path, rltile) values (?, ?, ?)"""
            bmpr = bmp.replace('.bmp', '')
            dirr = dir.replace(home, '') + bmp
            curs.execute(qrystr, (bmpr, dirr, True))
    os.chdir(here)

if __name__ == '__main__':
    db = sqlite3.connect(argv[-2])
    curs = db.cursor()
    ins_rltiles(curs, argv[-1])
    curs.close()
    db.commit()
