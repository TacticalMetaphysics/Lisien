# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  public@zacharyspector.com


def elide():
    from ELiDE.app import ELiDEApp
    app = ELiDEApp()
    app.run()

if __name__ == '__main__':
    elide()
