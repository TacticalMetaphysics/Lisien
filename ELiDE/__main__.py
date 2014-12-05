# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
import cProfile


def elide():
    from ELiDE.app import ELiDEApp
    app = ELiDEApp()
    app.run()

if __name__ == '__main__':
    cProfile.run('elide()')
