# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from util import tabclas


class ChangeException(Exception):
    """Generic exception for something gone wrong with a Change."""
    pass


class Change(object):
    """A change to a single value in the world state."""
    def __init__(self, val, *keys):
        """The given value will be put into the place in the Skeleton
specified in the remaining arguments. The keys are in the order
they'll be looked up in the skeleton--table name first, field name
last.

        """
        global tabclas
        self.clas = tabclas[keys[0]]
        for fn in keys:
            if fn not in self.clas.colnames[keys[0]]:
                raise ChangeException(
                    "That table doesn't have that field.")
        self.keys = keys
        self.value = val
