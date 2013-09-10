from collections import MutableMapping
from copy import copy
from util import ListItemIterator



LIST_COEFFICIENT = 100


class Skeleton(MutableMapping):
    atrdic = {
        "subtype": lambda self: self.getsubtype(),
        "rowdict": lambda self: self.isrowdict()
        }

    def __init__(self, it, listeners=None):
        if isinstance(it, dict):
            self.typ = dict
        elif isinstance(it, list):
            self.typ = list
        elif isinstance(it, Skeleton):
            return it
        else:
            raise ValueError(
                "Skeleton may only contain dict or list.")
        self.it = it
        if listeners is None:
            self.listeners = set()
        else:
            self.listeners = listeners

    def __getattr__(self, attrn):
        try:
            return self.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
        "Skeleton instance does not have and cannot compute "
        "attribute {0}".format(attrn))

    def __getitem__(self, k):
        return self.it[k]

    def __setitem__(self, k, v):
        if self.typ is dict:
            if self.subtype in (None, type(v)):
                self.it[k] = Skeleton(v)
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
            self.it[k] = Skeleton(v)
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
            for listener in self.listeners:
                listener.on_skel_assign(k, v)

    def __delitem__(self, k):
        if self.typ is list:
            self.it[k] = None
        else:
            for listener in self.listeners:
                listener.on_skel_delete(k)
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
            return self.it.iterkeys()

    def __len__(self):
        return len(self.it)

    def __add__(self, other):
        newness = self.copy()
        newness += other
        return newness

    def __iadd__(self, other):
        self.update(other)

    def __sub__(self, other):
        newness = self.copy()
        newness -= other
        return newness

    def __isub__(self, other):
        if other.__class__ in (dict, Skeleton)
            kitr = other.iteritems()
        else:
            kitr = ListItemIterator(other)
        for (k, v) in kitr:
            if k not in self:
                continue
            elif self[k] == v:
                del self[k]
            else:
                self[k] -= v

    def copy(self):
        # Shallow copy
        return Skeleton(self.it)

    def deepcopy(self):
        newness = Skeleton(self.typ())
        for (k, v) in self.iteritems():
            if isinstance(v, Skeleton):
                newness[k] = v.deepcopy()
            else:
                assert self.rowdict, "I contain something I shouldn't"
                newness[k] = copy(v)
        return newness

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

    def iteritems(self):
        return self.it.iteritems()

    def isrowdict(self):
        for that in self.it.itervalues():
            if that.__class__ in (dict, list):
                return False
        return True

    def getsubtype(self):
        if len(self.it) == 0:
            return None
        if self.typ is dict:
            return typ(self.it.iteritems().next())
        else: # self.typ is list
            return typ(self.it[0])

    def update(self, skellike):
        if skellike.__class__ in (dict, Skeleton):
            kitr = skellike.iteritems()
        else:
            kitr = ListItemIterator(skellike)
        for (k, v) in kitr:
            if k not in self:
                self[k] = v
            elif isinstance(self[k], Skeleton):
                self[k].update(v)
            elif self[k] != v:
                self[k] = v
