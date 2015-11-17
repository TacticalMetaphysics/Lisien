# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Common utility functions and data structures.

"""
from operator import attrgetter
import numpy
from json import dumps, loads, JSONEncoder
from collections import Mapping
from gorm.reify import reify


def getatt(attribute_name):
    """An easy way to make an alias"""
    return property(attrgetter(attribute_name))


try:
    from sqlalchemy.exc import OperationalError as alchemyOpError
    from sqlite3 import OperationalError as liteOpError
    OperationalError = (alchemyOpError, liteOpError)
except ImportError:
    from sqlite3 import OperationalError


try:
    from sqlalchemy.exc import IntegrityError as alchemyIntegError
    from sqlite3 import IntegrityError as liteIntegError
    IntegrityError = (alchemyIntegError, liteIntegError)
except ImportError:
    from sqlite3 import IntegrityError


class RedundantRuleError(ValueError):
    """Error condition for when you try to run a rule on a (branch,
    tick) it's already been executed.

    """
    pass


class UserFunctionError(SyntaxError):
    """Error condition for when I try to load a user-defined function and
    something goes wrong.

    """
    pass


class CacheError(ValueError):
    """Error condition for something going wrong with a cache"""
    pass


class TravelException(Exception):
    """Exception for problems with pathfinding. Not necessarily an error
    because sometimes somebody SHOULD get confused finding a path.

    """
    def __init__(
            self,
            message,
            path=None,
            followed=None,
            traveller=None,
            branch=None,
            tick=None,
            lastplace=None
    ):
        """Store the message as usual, and also the optional arguments:

        ``path``: a list of Place names to show such a path as you found

        ``followed``: the portion of the path actually followed

        ``traveller``: the Thing doing the travelling

        ``branch``: branch during travel

        ``tick``: tick at time of error (might not be the tick at the
        time this exception is raised)

        ``lastplace``: where the traveller was, when the error happened

        """
        self.path = path
        self.followed = followed
        self.traveller = traveller
        self.branch = branch
        self.tick = tick
        self.lastplace = lastplace
        super().__init__(message)


def path_len(graph, path, weight=None):
    """Return the number of ticks it will take to follow ``path``,
    assuming the portals' ``weight`` attribute is how long it will
    take to go through that portal--if unspecified, 1 tick.

    """
    n = 0
    path = list(path)  # local copy
    prevnode = path.pop(0)
    while path:
        nextnode = path.pop(0)
        edge = graph.edge[prevnode][nextnode]
        n += edge[weight] if weight and hasattr(edge, weight) else 1
        prevnode = nextnode
    return n


class LiSEncoder(JSONEncoder):
    def default(self, o):
        t = type(o)
        if t in numpy.sctypes['int']:
            return int(o)
        elif t in numpy.sctypes['float']:
            return float(o)
        else:
            return super().default(o)


def dict_diff(old, new):
    """Return a dictionary containing the items of ``new`` that are either
    absent from ``old`` or whose values are different; as well as the
    value ``None`` for those keys that are present in ``old``, but
    absent from ``new``.

    Useful for describing changes between two versions of a dict.

    """
    r = {}
    for k in set(old.keys()).union(new.keys()):
        if k in old:
            if k not in new:
                r[k] = None
            elif old[k] != new[k]:
                r[k] = new[k]
        else:  # k in new
            r[k] = new[k]
    return r


def list_diff(old, new):
    r = {item: True for item in new}
    for item in old:
        if item not in new:
            r[item] = False
    return r


def keycache_iter(keycache, branch, tick, get_iterator):
    if branch not in keycache:
        keycache[branch] = {}
    if tick not in keycache[branch]:
        keycache[branch][tick] = set(get_iterator())
    yield from keycache[branch][tick]


class AbstractEngine(object):
    @reify
    def json_dump_hints(self):
        return {}

    @reify
    def json_load_hints(self):
        return {}

    def _enc_tuple(self, obj):
        if isinstance(obj, list):
            return ["list"] + [self._enc_tuple(v) for v in obj]
        elif isinstance(obj, tuple):
            return ["tuple"] + [self._enc_tuple(v) for v in obj]
        elif isinstance(obj, dict):
            return ["dict"] + [
                [self._enc_tuple(k), self._enc_tuple(v)]
                for (k, v) in obj.items()
            ]
        elif isinstance(obj, self.char_cls):
            return ['character', obj.name]
        elif isinstance(obj, self.node_cls):
            return ['node', obj.character.name, obj.name]
        elif isinstance(obj, self.portal_cls):
            return ['portal', obj.character.name, obj.nodeA.name, obj.nodeB.name]
        else:
            return obj

    def _dec_tuple(self, obj):
        if isinstance(obj, list) or isinstance(obj, tuple):
            if obj == [] or obj == ["list"]:
                return []
            elif obj == ["tuple"]:
                return tuple()
            elif obj == ['dict']:
                return {}
            elif obj[0] == 'list':
                return [self._dec_tuple(p) for p in obj[1:]]
            elif obj[0] == 'tuple':
                return tuple(self._dec_tuple(p) for p in obj[1:])
            elif obj[0] == 'dict':
                return {
                    self._dec_tuple(k): self._dec_tuple(v)
                    for (k, v) in obj[1:]
                }
            elif obj[0] == 'character':
                return self.character[self._dec_tuple(obj[1])]
            elif obj[0] == 'node':
                return self.character[self._dec_tuple(obj[1])].node[self._dec_tuple(obj[2])]
            elif obj[0] == 'portal':
                return self.character[self._dec_tuple(obj[1])].portal[self._dec_tuple(obj[2])][self._dec_tuple(obj[3])]
            else:
                raise ValueError("Unknown sequence type: {}".format(obj[0]))
        else:
            return obj

    def json_dump(self, obj):
        """JSON dumper that distinguishes lists from tuples, and handles
        pointers to Node, Portal, and Character.

        """
        if isinstance(obj, self.node_cls):
            return dumps(["node", obj.character.name, obj.name])
        if isinstance(obj, self.portal_cls):
            return dumps(["portal", obj.character.name, obj.orign, obj.destn])
        if isinstance(obj, self.char_cls):
            return dumps(["character", obj.name])
        k = str(obj)
        if k not in self.json_dump_hints:
            self.json_dump_hints[k] = dumps(self._enc_tuple(obj), cls=LiSEncoder)
        return self.json_dump_hints[k]

    def json_load(self, s):
        if s is None:
            return None
        if s not in self.json_load_hints:
            self.json_load_hints[s] = self._dec_tuple(loads(s))
        return self.json_load_hints[s]
