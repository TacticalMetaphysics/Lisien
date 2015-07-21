# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
from gorm.xjson import json_load
from kivy.clock import Clock
from functools import partial


class trigger(object):
    """Make a trigger from a method.

    Decorate a method with this and it will become a trigger. Supply a numeric parameter to set a timeout.

    Not suitable for methods that expect any arguments other than ``dt``. However you should make your method
    accept ``*args`` for compatibility.

    """
    def __init__(self, func_or_timeout):
        if callable(func_or_timeout):
            self.func = func_or_timeout
            self.timeout = 0
        else:
            self.func = None
            self.timeout = func_or_timeout

    def __call__(self, func):
        self.func = func
        return self

    def __get__(self, instance, owner=None):
        if instance is None:
            # EventDispatcher iterates over its attributes before it instantiates.
            # Don't try making any trigger in that case.
            return
        retval = Clock.create_trigger(
            partial(self.func, instance), self.timeout
        )
        setattr(instance, self.func.__name__, retval)
        return retval


def set_remote_value(remote, k, v):
    if v is None:
        del remote[k]
    else:
        remote[k] = try_json_load(v)


def remote_setter(remote):
    """Return a function taking two arguments, ``k`` and ``v``, which sets
    ``remote[k] = v``, interpreting ``v`` as JSON if possible, or
    deleting ``remote[k]`` if ``v is None``.

    """
    return lambda k, v: set_remote_value(remote, k, v)


def try_json_load(obj):
    """Return the JSON interpretation the object if possible, or just the
    object otherwise.

    """
    try:
        return json_load(obj)
    except (TypeError, ValueError):
        return obj


def dummynum(character, name):
    """Count how many nodes there already are in the character whose name
    starts the same.

    """
    num = 0
    for nodename in character.node:
        nodename = str(nodename)
        if not nodename.startswith(name):
            continue
        try:
            nodenum = int(nodename.lstrip(name))
        except ValueError:
            continue
        num = max((nodenum, num))
    return num