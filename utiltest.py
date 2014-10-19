from LiSE import Engine
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
