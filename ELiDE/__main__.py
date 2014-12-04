# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
import LiSE
import sys

def lise():
    print("args: {}".format(sys.argv))
    print("path: {}".format(sys.path))

    from ELiDE.app import ELiDEApp
    app = ELiDEApp()
    sys.setrecursionlimit(10000)
    app.run()

if __name__ == '__main__':
    lise()
