from collections import MutableMapping
from util import ListItemIterator


LIST_COEFFICIENT = 100


class Skeleton(MutableMapping):
    atrdic = {
        "subtype": lambda self: self.getsubtype(),
        "rowdict": lambda self: self.isrowdict()
        }

    def isrowdict(self):
        for that in self.it.itervalues():
            if that.__class__ in (dict, list):
                return False
        return True

    def getsubtype(self):
        if len(self.it) == 0:
            raise ValueError(
                "Can't decide what the subtype is if there's nothing here.")
        if self.typ is dict:
            return typ(self.it.iteritems().next()))
        else: # self.typ is list
            return typ(self.it[0])

    def __init__(self, it):
        if isinstance(it, dict):
            self.typ = dict
        elif isinstance(it, list):
            self.typ = list
        else:
            raise ValueError(
                "Skeleton may only contain dict or list.")
        self.it = it

    def __getattr__(self, attrn):
        try:
            return self.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
        "Skeleton instance does not have and cannot compute "
        "attribute {0}".format(attrn))

    def __getitem__(self, k):
        return Skeleton(self.it[k])

    def __setitem__(self, k, v):
        if self.typ is dict:
            if self.subtype in (None, type(v)):
                self[k] = v
            else:
                raise TypeError(
            "This part of the skeleton takes {0}, not {1}".format(
                self.typ, type(v)))
        elif self.typ is list:
            if not isinstance(k, int):
                raise TypeError(
                    "Indices at this level must be integers")
            if self.subtype not in (None, type(v)):
                raise TypeError(
            "This part of the skeleton takes {0}, not {1}".format(
                self.typ, type(v)))
            while len(self.it) <= k:
                self.it.extend([self.typ()] * LIST_COEFFICIENT)
            self.it[k] = v
        else: # self.typ is None
            if self.rowdict:
                self.it[k] = v
            elif self.subtype is None:
                if isinstance(k, int):
                    self.it = []
                else:
                    self.it = {}
                self.__setitem__(k, v)
            else:
                raise TypeError(
            "This part of the skeleton takes {0}, not {1}".format(
                self.typ, type(v)))

    def __delitem__(self, k):
        if self.typ is list:
            self.it[k] = None
        else:
            del self.it[k]

    def __contains__(self, what):
        if self.typ is list:
            return what > -1 and what < len(self.it)
        else:
            return what in self.it

    def __iter__(self):
        if self.typ is list:
            return xrange(0, len(self.it) - 1)
        else:
            return iter(self.it)

    def iteritems(self):
        if self.typ is list:
            return ListItemIterator(self.it)
        else:
            return self.it.iteritems()

    def keys(self):
        if self.typ is list:
            return range(0, len(self.it) - 1)
        else:
            return self.it.keys()
