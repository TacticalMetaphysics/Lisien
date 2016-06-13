# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Some utility functions I found useful in testing."""


from LiSE.engine import Engine
from os import remove


def clear_off():
    for fn in ('LiSEworld.db', 'LiSEcode.db'):
        try:
            remove(fn)
        except OSError:
            pass


def mkengine(w='sqlite:///LiSEworld.db', *args, **kwargs):
    return Engine(
        worlddb=w,
        codedb='LiSEcode.db',
        *args,
        **kwargs
    )


seed = 69105
caching = True
