# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com


"""The basic data model of LiSE, based on NetworkX DiGraph objects
with various additions and conveniences.

"""

from collections import (
    defaultdict,
    Mapping,
    MutableMapping,
    Callable
)
import networkx as nx
from gorm.graph import (
    DiGraph,
    GraphSuccessorsMapping,
    DiGraphPredecessorsMapping
)
from .util import (
    CompositeDict,
    keycache_iter,
    cache_del,
    dispatch,
    listen,
    listener
)
from .rule import RuleBook
from .rule import CharRuleMapping as RuleMapping
from .funlist import FunList
from .thingplace import (
    Thing,
    Place
)
from .portal import Portal


class RuleFollower(object):
    """Mixin class that has a rulebook associated, which you can get a
    RuleMapping into

    """

    @property
    def _rulebook_listeners(self):
        if not hasattr(self, '_rbl'):
            self._rbl = []
        return self._rbl

    @_rulebook_listeners.setter
    def _rulebook_listeners(self, v):
        self._rbl = v

    def _dispatch_rulebook(self, v):
        for f in self._rulebook_listeners:
            f(self, v)

    def rulebook_listener(self, f):
        listen(self._rulebook_listeners, f)

    @property
    def rulebook(self):
        return RuleBook(
            self.engine,
            self.engine.db.get_rulebook_char(
                self._book,
                self.character.name
            )
        )

    @rulebook.setter
    def rulebook(self, v):
        if not isinstance(v, str) or isinstance(v, RuleBook):
            raise TypeError("Use a :class:`RuleBook` or the name of one")
        n = v.name if isinstance(v, RuleBook) else v
        self.engine.db.upd_rulebook_char(self._book, n, self.character.name)
        self._dispatch_rulebook(v)

    @property
    def rule(self):
        if not hasattr(self, '_rule_mapping'):
            self._rule_mapping = RuleMapping(
                self.character, self.rulebook, self._book
            )
        return self._rule_mapping


class CharacterThingMapping(MutableMapping, RuleFollower):
    """:class:`Thing` objects that are in a :class:`Character`"""
    _book = "thing"

    def __init__(self, character):
        """Store the character and initialize cache (if caching)"""
        self.character = character
        self.engine = character.engine
        self.name = character.name
        self._thing_listeners = defaultdict(list)
        if self.engine.caching:
            self._cache = {}
            self._keycache = {}

            @self.engine.on_time
            def recache(eng, branch_then, tick_then, b, t):
                if b not in self._keycache:
                    self._keycache[b] = {}
                self._keycache[b][t] = set(
                    self._iter_thing_names()
                )

    def _dispatch_thing(self, k, v):
        (b, t) = self.engine.time
        dispatch(self._thing_listeners, k, b, t, self, k, v)
        if v is None:
            if b in self._keycache:
                try:
                    self._keycache[b][t].remove(k)
                except KeyError:
                    self._keycache[b][t] = set(
                        self._iter_thing_names()
                    )
            else:
                self._keycache[b] = {
                    t: set(
                        self._iter_thing_names()
                    )
                }
        else:
            if b in self._keycache:
                try:
                    self._keycache[b][t].add(k)
                except KeyError:
                    self._keycache[b][t] = set(
                        self._iter_thing_names()
                    )
            else:
                self._keycache[b] = {
                    t: set(
                        self._iter_thing_names()
                    )
                }

    def listener(self, f=None, thing=None):
        return listener(self._thing_listeners, f, thing)

    def __contains__(self, k):
        """Check the cache first, if it exists"""
        if not self.engine.caching:
            return k in self._iter_thing_names()
        (branch, tick) = self.engine.time
        if branch not in self._keycache:
            self._keycache[branch] = {}
        try:
            return k in self._keycache[branch][tick]
        except KeyError:
            self._keycache[branch][tick] = set(self._iter_thing_names())
        return k in self._keycache[branch][tick]

    def _iter_thing_names(self):
        """Iterate over the names of things *in the database*."""
        for (n, l) in self.engine.db.thing_loc_items(
            self.character.name,
            *self.engine.time
        ):
            yield n

    def __iter__(self):
        """Iterate over nodes that have locations, and are therefore
        Things. Yield their names.

        """
        if not self.engine.caching:
            yield from self._iter_thing_names()
            return
        (branch, tick) = self.engine.time
        if branch not in self._keycache:
            self._keycache[branch] = {}
        if tick not in self._keycache[branch]:
                self._keycache[branch][tick] = set(self._iter_thing_names())
        yield from self._keycache[branch][tick]

    def __len__(self):
        """Just iterate and count stuff"""
        n = 0
        for th in self:
            n += 1
        return n

    def __getitem__(self, thing):
        """Check the cache first. If the key isn't there, try retrieving it
        from the database.

        """
        if self.engine.caching and thing in self and thing in self._cache:
            return self._cache[thing]
        (th, l) = self.engine.db.thing_and_loc(
            self.character.name,
            thing,
            *self.engine.time
        )
        r = Thing(self.character, th)
        if self.engine.caching:
            self._cache[thing] = r
        return r

    def __setitem__(self, thing, val):
        """Clear out any existing :class:`Thing` by this name and make a new
        one out of ``val`` (assumed to be a mapping of some kind)

        """
        if not isinstance(val, Mapping):
            raise TypeError('Things are made from Mappings')
        if 'location' not in val:
            raise ValueError('Thing needs location')
        self.engine.gorm.db.exist_node(
            self.character.name,
            thing,
            self.engine.branch,
            self.engine.tick,
            True
        )
        th = Thing(self.character, thing)
        th.clear()
        th.update(val)
        self._dispatch_thing(thing, th)
        if self.engine.caching:
            self._cache[thing] = th
            (branch, tick) = self.engine.time
            if branch in self._keycache:
                for t in list(self._keycache[branch].keys()):
                    if t > tick:
                        del self._keycache[branch][t]
                if tick in self._keycache[branch]:
                    self._keycache[branch][tick].add(self.name)
                    return
                try:
                    self._keycache[branch][tick] = set(
                        self._keycache[branch][
                            max(t for t in self._keycache[branch]
                                if t < tick)
                        ]
                    )
                    self._keycache[branch][tick].add(self.name)
                except ValueError:
                    pass

    def __delitem__(self, thing):
        """Delete the thing from the cache and the database"""
        (branch, tick) = self.engine.time
        if self.engine.caching:
            if thing in self._cache:
                del self._cache[thing]
            if branch in self._keycache:
                for t in list(self._keycache[branch].keys()):
                    if t > tick:
                        del self._keycache[branch][t]
                if tick in self._keycache[branch]:
                    self._keycache[branch][tick].remove(thing)
                    return
                try:
                    self._keycache[branch][tick] = set(
                        self._keycache[branch][
                            max(t for t in self._keycache[branch]
                                if t < tick)
                        ]
                    )
                    self._keycache[branch][tick].remove(self.name)
                except ValueError:
                    pass
        self.engine.db.thing_loc_and_next_del(
            self.character.name,
            self.name,
            branch,
            tick
        )
        self._dispatch_thing(thing, None)

    def __repr__(self):
        """Represent myself as a dict"""
        return repr(dict(self))


class CharacterPlaceMapping(MutableMapping, RuleFollower):
    """:class:`Place` objects that are in a :class:`Character`"""
    _book = "place"

    def __init__(self, character):
        """Store the character and initialize the cache (if caching)"""
        self.character = character
        self.engine = character.engine
        self.name = character.name
        self._place_listeners = defaultdict(list)
        if self.engine.caching:
            self._cache = {}
            self._keycache = {}

    def _dispatch_place(self, k, v):
        (branch, tick) = self.engine.time
        dispatch(self._place_listeners, k, branch, tick, self, k, v)

    def listener(self, f=None, place=None):
        return listener(self._place_listeners, f, place)

    def _things(self):
        """Private method. Return a set of names of things in the character."""
        return set(
            thing for (thing, loc) in
            self.engine.db.character_things_items(
                self.character.name,
                *self.engine.time
            )
        )

    def _iter_place_names(self):
        things = self._things()
        for node in self.engine.db.nodes_extant(
                self.character.name,
                *self.engine.time
        ):
            if node not in things:
                yield node

    def __iter__(self):
        """Iterate over names of places."""
        if not self.engine.caching:
            yield from self._iter_place_names()
            return
        (branch, tick) = self.engine.time
        if branch not in self._keycache:
            self._keycache[branch] = {
                tick: set(self._iter_place_names())
            }
        elif tick not in self._keycache[branch]:
            self._keycache[branch][tick] = set(
                self._iter_place_names()
            )
        yield from self._keycache[branch][tick]

    def __contains__(self, k):
        """Check the cache first, if it exists"""
        if self.engine.caching:
            (branch, tick) = self.engine.time
            if (
                    branch in self._keycache and
                    tick in self._keycache[branch]
            ):
                return k in self._keycache[branch][tick]
        return k in self._iter_place_names()

    def __len__(self):
        """Iterate and count"""
        n = 0
        for place in self:
            n += 1
        return n

    def _getplace(self, place):
        nodenames = set(self.character.node.keys())
        thingnames = self._things()
        if place in nodenames.difference(thingnames):
            return Place(self.character, place)
        raise KeyError("No such place")

    def __getitem__(self, place):
        """Get the place from the cache if I can, otherwise check that it
        exists, and if it does, cache and return it

        """
        if not self.engine.caching:
            return self._getplace(place)
        if place not in self:
            raise KeyError("No such place")
        # not using cache_get because creating Place objects is expensive
        if place not in self._cache:
            self._cache[place] = Place(self.character, place)
        return self._cache[place]

    def __setitem__(self, place, v):
        """Wipe out any existing place by that name, and replace it with one
        described by ``v``

        """
        if not self.engine.caching:
            pl = Place(self.character, place)
        else:
            if place not in self._cache:
                self._cache[place] = Place(self.character, place)
            pl = self._cache[place]
        self._dispatch_place(place, v)
        (branch, tick) = self.engine.time
        self.engine.db.exist_node(
            self.character.name,
            place,
            branch,
            tick,
            True
        )
        pl.clear()
        pl.update(v)
        if not self.engine.caching:
            return
        if branch not in self._keycache:
            self._keycache[branch] = {}
        if tick in self._keycache[branch]:
            self._keycache[branch][tick].add(place)
        else:
            self._keycache[branch][tick] = set(self._iter_place_names())

    def __delitem__(self, place):
        """Delete place from both cache and database"""
        (branch, tick) = self.engine.time
        if self.engine.caching:
            if place in self._cache:
                del self._cache[place]
            if (
                    branch in self._keycache and
                    tick in self._keycache[branch] and
                    place in self._keycache[branch][tick]
            ):
                self._keycache[branch][tick].remove(place)
        self.engine.gorm.db.exist_node(
            self.character.name,
            place,
            branch,
            tick,
            False
        )
        self._dispatch_place(place, None)

    def __repr__(self):
        """Represent myself as a dictionary"""
        return repr(dict(self))


class CharacterThingPlaceMapping(MutableMapping):
    """Replacement for gorm's GraphNodeMapping that does Place and Thing"""
    def __init__(self, character):
        """Store the character"""
        self.character = character
        self.engine = character.engine
        self.name = character.name
        if self.engine.caching:
            self._keycache = {}

    def __iter__(self):
        if not self.engine.caching:
            yield from self.engine.db.nodes_extant(
                self.character.name,
                *self.engine.time
            )
            return
        (branch, tick) = self.engine.time
        yield from keycache_iter(
            self._keycache,
            branch,
            tick,
            lambda: self.engine.db.nodes_extant(
                self.character.name, *self.engine.time
            )
        )

    def __len__(self):
        """Count nodes that exist"""
        n = 0
        for node in iter(self):
            n += 1
        return n

    def __getitem__(self, k):
        """Return a :class:`Thing` or :class:`Place` as appropriate"""
        if k in self.character.thing:
            return self.character.thing[k]
        elif k in self.character.place:
            return self.character.place[k]
        else:
            raise KeyError("No such Thing or Place in this Character")

    def __setitem__(self, k, v):
        """Assume you're trying to create a :class:`Place`"""
        self.character.place[k] = v

    def __delitem__(self, k):
        """Delete place or thing"""
        if (
                k not in self.character.thing and
                k not in self.character.place
        ):
            raise KeyError("No such thing or place")
        if k in self.character.thing:
            del self.character.thing[k]
        if k in self.character.place:
            del self.character.place[k]


class CharacterPortalSuccessorsMapping(GraphSuccessorsMapping, RuleFollower):
    _book = "portal"

    @property
    def _cache(self):
        if not hasattr(self, '_c'):
            self._c = {}
        return self._c

    @property
    def _portal_listeners(self):
        if not hasattr(self, '_pl'):
            self._pl = defaultdict(list)
        return self._pl

    def _dispatch_portal(self, o, d, p):
        dispatch(
            self._portal_listeners,
            o,
            self.gorm.branch,
            self.gorm.rev,
            self,
            self.graph.node[o],
            self.graph.node[d],
            p
        )

    def listener(self, f=None, place=None):
        return listener(self._portal_listeners, f, place)

    def __getitem__(self, nodeA):
        if self.gorm.db.node_exists(
                self.graph.name,
                nodeA,
                self.gorm.branch,
                self.gorm.rev
        ):
            if nodeA not in self._cache:
                self._cache[nodeA] = self.Successors(self, nodeA)
            return self._cache[nodeA]
        raise KeyError("No such node")

    def __setitem__(self, nodeA, val):
        if nodeA not in self._cache:
            self._cache[nodeA] = self.Successors(self, nodeA)
        sucs = self._cache[nodeA]
        sucs.clear()
        sucs.update(val)

    def __delitem__(self, nodeA):
        bs = list(self[nodeA].keys())
        super().__delitem__(nodeA)
        for b in bs:
            self._dispatch_portal(nodeA, b, None)

    class Successors(GraphSuccessorsMapping.Successors):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.engine = self.graph.engine
            self._portal_listeners = defaultdict(list)
            if self.engine.caching:
                self._cache = {}
                self._keycache = {}

        def _dispatch_portal(self, nodeB, portal):
            dispatch(
                self._portal_listeners,
                nodeB,
                self,
                self.container.graph.node[self.nodeA],
                self.container.graph.node[nodeB],
                portal
            )
            self.container._dispatch_portal(self.nodeA, nodeB, portal)

        def listener(self, f=None, nodeB=None):
            return listener(self._portal_listeners, f, nodeB)

        def _getsub(self, nodeB):
            if hasattr(self, '_cache'):
                if nodeB not in self._cache:
                    self._cache[nodeB] = Portal(self.graph, self.nodeA, nodeB)
                return self._cache[nodeB]
            return Portal(self.graph, self.nodeA, nodeB)

        def __contains__(self, nodeB):
            if not self.engine.caching:
                return super().__contains__(nodeB)
            (branch, tick) = self.engine.time
            if branch not in self._keycache:
                self._keycache[branch] = {}
            if tick not in self._keycache[branch]:
                self._keycache[branch][tick] = set(iter(self))
            return nodeB in self._keycache[branch][tick]

        def __getitem__(self, nodeB):
            if not self.engine.caching:
                return super().__getitem__(nodeB)
            if nodeB in self:
                if nodeB not in self._cache:
                    self._cache[nodeB] = Portal(self.graph, self.nodeA, nodeB)
                return self._cache[nodeB]
            raise KeyError("No such portal")

        def __setitem__(self, nodeB, value):
            (branch, tick) = self.engine.time
            if self.engine.caching:
                if (
                        branch in self._keycache and
                        tick in self._keycache[branch]
                ):
                    self._keycache[branch][tick].add(nodeB)
                if nodeB not in self._cache:
                    self._cache[nodeB] = Portal(self.graph, self.nodeA, nodeB)
                p = self._cache[nodeB]
            else:
                p = Portal(self.graph, self.nodeA, nodeB)
            self.engine.db.exist_edge(
                self.graph.name,
                self.nodeA,
                nodeB,
                0,
                branch,
                tick,
                True
            )
            p.clear()
            p.update(value)
            self._dispatch_portal(nodeB, p)

        def __delitem__(self, nodeB):
            if not self.engine.caching:
                super().__delitem__(nodeB)
                return
            (branch, tick) = self.engine.time
            cache_del(
                self._cache,
                self._keycache,
                branch,
                tick,
                nodeB,
                super().__delitem__
            )
            self._dispatch_portal(nodeB, None)


class CharacterPortalPredecessorsMapping(
        DiGraphPredecessorsMapping,
        RuleFollower
):
    _book = "portal"

    class Predecessors(DiGraphPredecessorsMapping.Predecessors):
        def _getsub(self, nodeA):
            if not self.graph.engine.caching:
                return Portal(self.graph, nodeA, self.nodeB)
            if nodeA in self.graph.portal:
                if (
                        self.graph.engine.caching and
                        self.nodeB not in self.graph.portal[nodeA]._cache
                ):
                    self.graph.portal[nodeA]._cache[self.nodeB] = Portal(
                        self.graph,
                        nodeA,
                        self.nodeB
                    )
                return self.graph.portal[nodeA][self.nodeB]
            return Portal(self.graph, nodeA, self.nodeB)

        def __setitem__(self, nodeA, value):
            if nodeA in self.graph.portal:
                if (
                        self.graph.engine.caching and
                        self.nodeB not in self.graph.portal[nodeA]._cache
                ):
                    self.graph.portal[nodeA]._cache[self.nodeB] = Portal(
                        self.graph,
                        nodeA,
                        self.nodeB
                    )
            p = self.graph.portal[nodeA][self.nodeB]
            p.clear()
            p.update(value)


class CharacterAvatarGraphMapping(Mapping, RuleFollower):
    _book = "avatar"

    def __init__(self, char):
        """Remember my character"""
        self.character = char
        self.engine = char.engine
        self.name = char.name
        self._name = char._name

    @property
    def listener(self, f=None, graph=None):
        return self.character.avatar_listener(f, graph)

    def __call__(self, av):
        """Add the avatar. It must be an instance of Place or Thing."""
        if av.__class__ not in (Place, Thing):
            raise TypeError("Only Things and Places may be avatars")
        self.character.add_avatar(av.name, av.character.name)

    def _datadict(self):
        if self.engine.caching:
            return self._avatarness_cache()
        else:
            return self._avatarness_db()

    def _avatarness_cache(self):
        ac = self.character._avatar_cache
        d = {}
        for (branch, rev) in self.engine._active_branches():
            for g in ac:
                if g not in d:
                    d[g] = {}
                for n in ac[g]:
                    if n in d[g]:
                        continue
                    if branch in ac[g][n]:
                        try:
                            if g not in d:
                                d[g] = {}
                            d[g][n] = ac[g][n][branch][
                                max(
                                    t for t in ac[g][n][branch]
                                    if t <= rev
                                )
                            ]
                        except KeyError:
                            pass
        return d

    def _avatarness_db(self):
        """Get avatar-ness data and return it"""
        return self.engine.db.avatarness(
            self.character.name, *self.engine.time
        )

    def __iter__(self):
        """Iterate over every avatar graph that has at least one avatar node
        in it presently

        """
        d = self._datadict()
        for graph in d:
            for node in d[graph]:
                if d[graph][node]:
                    yield graph
                    break

    def __len__(self):
        """Number of graphs in which I have an avatar"""
        n = 0
        for g in self:
            n += 1
        return n

    def __getitem__(self, g):
        """Get the CharacterAvatarMapping for the given graph, if I have any
        avatars in it.

        If I have avatars in only one graph, behave as a proxy to that
        graph's CharacterAvatarMapping.

        Unless I have only one avatar anywhere, in which case be a
        proxy to that.

        """
        d = (
            self.character._avatar_cache
            if self.engine.caching
            else self._datadict()
        )
        if g in d:
            return self.CharacterAvatarMapping(self, g)
        elif len(d.keys()) == 1:
            avm = self.CharacterAvatarMapping(self, list(d.keys())[0])
            if len(avm.keys()) == 1:
                return avm[list(avm.keys())[0]][g]
            else:
                return avm[g]
        raise KeyError("No avatar in {}".format(g))

    def __getattr__(self, attr):
        """If I've got only one avatar, return its attribute"""
        d = self._datadict()
        if len(d.keys()) == 1:
            avs = self.CharacterAvatarMapping(self, list(d.keys())[0])
            if len(avs) == 1:
                av = list(avs.keys())[0]
                if attr == av:
                    return avs[attr]
                else:
                    return getattr(avs[list(avs.keys())[0]], attr)
        raise AttributeError

    def __repr__(self):
        """Represent myself like a dictionary"""
        d = {}
        for k in self:
            d[k] = dict(self[k])
        return repr(d)

    class CharacterAvatarMapping(Mapping):
        """Mapping of avatars of one Character in another Character."""
        def __init__(self, outer, graphn):
            """Store the character and the name of the "graph", ie. the other
            character.

            """
            self.character = outer.character
            self.engine = outer.engine
            self.name = outer.name
            self.graph = graphn

        def _branchdata(self, branch, rev):
            if self.engine.caching:
                return self._branchdata_cache(branch, rev)
            else:
                return self.engine.db.avatar_branch_data(
                    self.character.name, self.graph, branch, rev
                )

        def _branchdata_cache(self, branch, rev):
            ac = self.character._avatar_cache
            return [
                (
                    node,
                    ac[self.graph][node][branch][
                        max(
                            t for t in ac[self.graph][node][branch]
                            if t <= rev
                        )
                    ]
                )
                for node in ac[self.graph]
                if branch in ac[self.graph][node]
            ]

        def __getattr__(self, attrn):
            """If I don't have such an attribute, but I contain exactly one
            avatar, and *it* has the attribute, return the
            avatar's attribute.

            """
            seen = set()
            counted = 0
            for (branch, rev) in self.engine._active_branches():
                if counted > 1:
                    break
                for (n, extant) in self._branchdata(branch, rev):
                    if counted > 1:
                        break
                    x = bool(extant)
                    if x and n not in seen:
                        counted += 1
                    seen.add(n)
            if counted == 1:
                node = self.engine.character[self.graph].node[seen.pop()]
                if hasattr(node, attrn):
                    return getattr(node, attrn)
            raise AttributeError("No such attribute: " + attrn)

        def __iter__(self):
            """Iterate over the names of all the presently existing nodes in the
            graph that are avatars of the character

            """
            seen = set()
            for (branch, rev) in self.engine.gorm._active_branches():
                for (n, x) in self._branchdata(branch, rev):
                    if (
                            x and
                            n not in seen and
                            self.engine._node_exists(self.graph, n)
                    ):
                        yield n
                    seen.add(n)

        def __contains__(self, av):
            fun = (
                self._contains_when_cache
                if self.engine.caching
                else self._contains_when_db
            )
            for (branch, tick) in self.engine.gorm._active_branches():
                r = fun(av, branch, tick)
                if r is None:
                    continue
                return r
            return False

        def _contains_when_cache(self, av, branch, rev):
            ac = self.character._avatar_cache
            if av not in ac:
                return False
            for node in ac[av]:
                try:
                    if ac[av][branch][
                            max(
                                t for t in ac[av][branch]
                                if t <= rev
                            )
                    ]:
                        return True
                except KeyError:
                    continue

        def _contains_when_db(self, av, branch, tick):
            return self.engine.db.is_avatar_of(
                self.character.name,
                self.graph,
                av,
                branch,
                tick
            )

        def __len__(self):
            """Number of presently existing nodes in the graph that are avatars of
            the character"""
            n = 0
            for a in self:
                n += 1
            return n

        def __getitem__(self, av):
            """Return the Place or Thing by the given name in the graph, if it's
            my avatar and it exists.

            If I contain exactly *one* Place or Thing, and you're
            not trying to get it by its name, delegate to its
            __getitem__. It's common for one Character to have
            exactly one avatar in another Character, and when that
            happens, it's nice not to have to specify the avatar's
            name.

            """
            if av in self:
                return self.engine.character[self.graph].node[av]
            if len(self.keys()) == 1:
                k = list(self.keys())[0]
                return self.engine.character[self.graph].node[k]
            raise KeyError("No such avatar")

        def __repr__(self):
            """Represent myself like a dictionary"""
            d = {}
            for k in self:
                d[k] = dict(self[k])
            return repr(d)


class SenseFuncWrap(object):
    """Wrapper for a sense function that looks it up in the code store if
    provided with its name, and prefills the first two arguments.

    """
    def __init__(self, character, fun):
        """Store the character and the function, looking up the function if
        needed

        """
        self.character = character
        self.engine = character.engine
        if isinstance(fun, str):
            self.fun = self.engine.sense[fun]
        else:
            self.fun = fun
        if not isinstance(self.fun, Callable):
            raise TypeError("Function is not callable")

    def __call__(self, observed):
        """Call the function, prefilling the engine and observer arguments"""
        if isinstance(observed, str):
            observed = self.engine.character[observed]
        return self.fun(self.engine, self.character, Facade(observed))


class CharacterSense(object):
    """Mapping for when you've selected a sense for a character to use
    but haven't yet specified what character to look at

    """
    def __init__(self, container, sensename):
        """Store the container and the name of the sense"""
        self.container = container
        self.engine = self.container.engine
        self.sensename = sensename
        self.observer = self.container.character

    @property
    def func(self):
        """Return the function most recently associated with this sense"""
        fn = self.engine.db.sense_func_get(
            self.observer.name,
            self.sensename,
            *self.engine.time
        )
        if fn is not None:
            return SenseFuncWrap(self.observer, fn)

    def __call__(self, observed):
        """Call my sense function and make sure it returns the right type,
        then return that.

        """
        r = self.func(observed)
        if not (
                isinstance(r, Character) or
                isinstance(r, Facade)
        ):
            raise TypeError(
                "Sense function did not return a character-like object"
            )
        return r


class CharacterSenseMapping(MutableMapping, RuleFollower):
    """Used to view other Characters as seen by one, via a particular sense"""
    _book = "character"

    def __init__(self, character):
        """Store the character"""
        self.character = character
        self.engine = character.engine
        self._listeners = defaultdict(list)

    def _dispatch(self, k, v):
        (branch, tick) = self.engine.time
        dispatch(self._listeners, k, branch, tick, self, k, v)

    def listener(self, f=None, sense=None):
        return listener(self._listeners, f, sense)

    def __iter__(self):
        """Iterate over active sense names"""
        yield from self.engine.db.sense_active_items(
            self.character.name, *self.engine.time
        )

    def __len__(self):
        """Count active senses"""
        n = 0
        for sense in iter(self):
            n += 1
        return n

    def __getitem__(self, k):
        """Get a :class:`CharacterSense` named ``k`` if it exists"""
        if not self.engine.db.sense_is_active(
                self.character.name,
                k,
                *self.engine.time
        ):
            raise KeyError("Sense isn't active or doesn't exist")
        return CharacterSense(self.character, k)

    def __setitem__(self, k, v):
        """Use the function for the sense from here on out"""
        if isinstance(v, str):
            funn = v
        else:
            funn = v.__name__
        if funn not in self.engine.sense:
            if not isinstance(v, Callable):
                raise TypeError("Not a function")
            self.engine.sense[funn] = v
        (branch, tick) = self.engine.time
        self.engine.db.sense_fun_set(
            self.character.name,
            k,
            branch,
            tick,
            funn,
            True
        )
        self._dispatch(k, v)

    def __delitem__(self, k):
        """Stop having the given sense"""
        (branch, tick) = self.engine.time
        self.engine.db.sense_set(
            self.character.name,
            k,
            branch,
            tick,
            False
        )
        self._dispatch(k, None)

    def __call__(self, fun, name=None):
        """Decorate the function so it's mine now"""
        if not isinstance(fun, Callable):
            raise TypeError(
                "I need a function here"
            )
        if name is None:
            name = fun.__name__
        self[name] = fun


class FacadePlace(MutableMapping):
    @property
    def name(self):
        return self['name']

    def contents(self):
        for thing in self.facade.thing.values():
            if thing.container is self:
                yield thing

    def __init__(self, facade, real):
        self.facade = facade
        self._real = real
        self._patch = {}
        self._masked = set()
        self._listeners = defaultdict(list)

    def _dispatch(self, k, v):
        dispatch(self._listeners, k, self, k, v)

    def listener(self, f=None, stat=None):
        return listener(self._listeners, f, stat)

    def __iter__(self):
        seen = set()
        for k in self._real:
            if k not in self._masked:
                yield k
            seen.add(k)
        for k in self._patch:
            if (
                    k not in self._masked and
                    k not in seen
            ):
                yield k

    def __len__(self):
        n = 0
        for k in self:
            n += 1
        return n

    def __getitem__(self, k):
        if k in self._masked:
            raise KeyError("masked")
        if k in self._patch:
            return self._patch[k]
        return self._real[k]

    def __setitem__(self, k, v):
        self._masked.discard(k)
        self._patch[k] = v
        self._dispatch(k, v)

    def __delitem__(self, k):
        self._masked.add(k)
        self._dispatch(k, None)


class FacadeThing(FacadePlace):
    @property
    def location(self):
        try:
            return self.facade.node[self['location']]
        except KeyError:
            return None

    @property
    def next_location(self):
        try:
            return self.facade.node[self['next_location']]
        except KeyError:
            return None

    @property
    def container(self):
        if self['next_location'] is None:
            return self.location
        try:
            return self.facade.portal[self['location']][
                self['next_location']]
        except KeyError:
            return self.location


class FacadePortal(FacadePlace):
    @property
    def origin(self):
        return self.facade.node[self['origin']]

    @property
    def destination(self):
        return self.facade.node[self['destination']]


class FacadeEntityMapping(MutableMapping):
    def __init__(self, facade):
        self.facade = facade
        self._patch = {}
        self._masked = set()
        self._listeners = defaultdict(list)

    def dispatch(self, k, v):
        dispatch(self._listeners, k, self, k, v)

    def listener(self, f=None, stat=None):
        return listener(self._listeners, f, stat)

    def __contains__(self, k):
        return (
            k not in self._masked and (
                k in self._patch or
                k in self._get_inner_map()
            )
        )

    def __iter__(self):
        seen = set()
        for k in self._get_inner_map():
            if k not in self._masked:
                yield k
            seen.add(k)
        for k in self._patch:
            if k not in seen:
                yield k

    def __len__(self):
        n = 0
        for k in self:
            n += 1
        return n

    def __getitem__(self, k):
        if k in self._masked:
            raise KeyError("masked")
        if k in self._patch:
            return self._patch[k]
        return self.facadecls(self.facade, self._get_inner_map()[k])

    def __setitem__(self, k, v):
        if not isinstance(v, self.facadecls):
            if not isinstance(v, self.innercls):
                raise TypeError(
                    "Need :class:``Thing`` or :class:``FacadeThing``"
                )
            v = self.facadecls(self.facade, v)
        self._masked.discard(k)
        self._patch[k] = v
        self._dispatch(k, v)

    def __delitem__(self, k):
        self._masked.add(k)
        self.dispatch(k, None)


class FacadeThingMapping(FacadeEntityMapping):
    facadecls = FacadeThing
    innercls = Thing

    def _get_inner_map(self):
        return self.facade.character.thing


class FacadePlaceMapping(FacadeEntityMapping):
    facadecls = FacadePlace
    innercls = Place

    def _get_inner_map(self):
        return self.facade.character.place


class FacadePortalSuccessors(FacadeEntityMapping):
    facadecls = FacadePortal
    innercls = Portal

    def __init__(self, facade, origname):
        super().__init__(facade)
        self._origname = origname

    def _get_inner_map(self):
        return self.facade.character.portal[self._origname]


class FacadePortalPredecessors(FacadeEntityMapping):
    facadecls = FacadePortal
    innercls = Portal

    def __init__(self, facade, destname):
        super().__init__(facade)
        self._destname = destname

    def _get_inner_map(self):
        return self.facade.character.preportal[self._destname]


class FacadePortalMapping(FacadeEntityMapping):
    def __getitem__(self, node):
        if node in self._masked:
            raise KeyError("masked")
        if node in self._patch:
            return self._patch[node]
        return self.cls(self.facade, node)

    def __setitem__(self, node, value):
        self._masked.discard(node)
        v = self.cls(self.facade, node)
        v.update(value)
        self._patch[node] = v

    def __delitem__(self, node):
        self._masked.add(node)


class FacadePortalSuccessorsMapping(FacadePortalMapping):
    cls = FacadePortalSuccessors

    def _get_inner_map(self):
        return self.facade.character.portal


class FacadePortalPredecessorsMapping(FacadePortalMapping):
    cls = FacadePortalPredecessors

    def _get_inner_map(self):
        return self.facade.character.preportal


class FacadeStatsMapping(MutableMapping):
    def __init__(self, facade):
        self.facade = facade
        self._patch = {}
        self._masked = set()

    def __iter__(self):
        seen = set()
        for k in self.facade.graph:
            if k not in self._masked:
                yield k
            seen.add(k)
        for k in self._patch:
            if k not in seen:
                yield k

    def __len__(self):
        n = 0
        for k in self:
            n += 1
        return n

    def __contains__(self, k):
        if k in self._masked:
            return False
        return (
            k in self._patch or
            k in self.facade.graph
        )

    def __getitem__(self, k):
        if k in self._masked:
            raise KeyError("masked")
        if k in self._patch:
            return self._patch[k]
        return self.facade.graph[k]

    def __setitem__(self, k, v):
        self._masked.discard(k)
        self._patch[k] = v

    def __delitem__(self, k):
        self._masked.add(k)


class Facade(nx.DiGraph):
    def __init__(self, character):
        self.character = character
        self.thing = FacadeThingMapping(self)
        self.place = FacadePlaceMapping(self)
        self.node = CompositeDict(self.thing, self.place)
        self.portal = FacadePortalSuccessorsMapping(self)
        self.succ = self.edge = self.adj = self.portal
        self.preportal = FacadePortalPredecessorsMapping(self)
        self.pred = self.preportal
        self.graph = FacadeStatsMapping(self)


class CharStatCache(MutableMapping):
    """Caching dict-alike for character stats"""
    def __init__(self, char):
        """Store character, initialize cache"""
        self.character = char
        self.engine = char.engine
        self._real = char.graph
        self._cache = {}

        def fillcache(cache, branch, tick):
            assert(self.engine.time == (branch, tick))
            if branch not in cache:
                cache[branch] = {}
                for k, v in self._real.items():
                    cache[branch][k] = {tick: v}
            else:
                for k in cache[branch]:
                    if tick not in cache[branch][k]:
                        try:
                            cache[branch][k][tick] = cache[branch][k][
                                max(t for t in cache[branch][k] if t < tick)
                            ]
                        except ValueError:
                            # probably shouldn't happen so much
                            cache[branch][k][tick] = self._real[k]

        @self.engine.on_time
        def time_travel_triggers(
                engine,
                branch_then,
                tick_then,
                branch_now,
                tick_now
        ):
            """Fire such triggers as needed for time travel from
            the given time to the present moment.

            """
            then = (branch_then, tick_then)
            now = (branch_now, tick_now)
            # prevent recursion
            self.engine.locktime = True
            # use a local cache if not really caching
            cache = self._cache if self.engine.caching else {}
            self.engine.time = then
            fillcache(cache, *then)
            self.engine.time = now
            fillcache(cache, *now)
            # Anything that differs between these two times
            # is cause for a trigger to fire.
            for k in cache[branch_then]:
                existed = (
                    tick_then in cache[branch_then][k] and
                    cache[branch_then][k][tick_then] is not None
                )
                exists = (
                    tick_now in cache[branch_now][k] and
                    cache[branch_now][k][tick_now] is not None
                )
                if existed and not exists:
                    # deleted
                    self._dispatch(k, None)
                elif exists and not existed:
                    # added
                    self._dispatch(k, cache[branch_now][k][tick_now])
                elif exists and existed:
                    if (
                            cache[branch_then][k][tick_then] !=
                            cache[branch_now][k][tick_now]
                    ):
                        # changed
                        self._dispatch(k, cache[branch_now][k][tick_now])
            del self.engine.locktime

    def listener(self, f=None, stat=None):
        return self.character.stat_listener(f, stat)

    def _dispatch(self, k, v):
        (branch, tick) = self.engine.time
        dispatch(
            self.character._stat_listeners,
            k,
            branch,
            tick,
            self.character,
            k,
            v
        )

    def __iter__(self):
        """Iterate over underlying keys"""
        return iter(self._real)

    def __len__(self):
        """Length of underlying graph"""
        return len(self._real)

    def __getitem__(self, k):
        """Use the cache if I can"""
        if not self.engine.caching:
            return self._real[k]
        (branch, tick) = self.engine.time
        if k not in self._cache:
            self._cache[k] = {}
        if branch not in self._cache[k]:
            self._cache[k][branch] = {}
        d = self._cache[k][branch]
        if tick not in d:
            d[tick] = self._real[k]
        if d[tick] is None:
            raise KeyError('Not set now')
        return d[tick]

    def __setitem__(self, k, v):
        """Cache new value and set it the normal way"""
        assert(v is not None)
        self._real[k] = v
        self._dispatch(k, v)
        if not self.engine.caching:
            return
        (branch, tick) = self.engine.time
        if k not in self._cache:
            self._cache[k] = {}
        if branch not in self._cache[k]:
            self._cache[k][branch] = {}
        self._cache[k][branch][tick] = v

    def __delitem__(self, k):
        """Clear the cached value and delete the normal way"""
        del self._real[k]
        self._dispatch(k, None)
        if not self.engine.caching:
            return
        (branch, tick) = self.engine.time
        if branch in self._cache[k]:
            for staletick in list(
                    t for t in self._cache[k][branch]
                    if t < tick
            ):
                del self._cache[k][branch][staletick]
            self._cache[k][branch][tick] = None


class Character(DiGraph, RuleFollower):
    """A graph that follows game rules and has a containment hierarchy.

    Nodes in a Character are subcategorized into Things and
    Places. Things have locations, and those locations may be Places
    or other Things. A Thing might also travel, in which case, though
    it will spend its travel time located in its origin node, it may
    spend some time contained by a Portal (i.e. an edge specialized
    for Character). If a Thing is not contained by a Portal, it's
    contained by whatever it's located in.

    """
    _book = "character"

    def __init__(self, engine, name, data=None, **attr):
        """Store engine and name, and set up mappings for Thing, Place, and
        Portal

        """
        super().__init__(engine.gorm, name, data, **attr)
        self.character = self
        self.engine = engine
        d = {}
        for mapp in ('character', 'avatar', 'thing', 'place', 'portal'):
            if mapp + '_rulebook' in attr:
                rulebook = attr[mapp + 'rulebook']
                bookname = rulebook.name if isinstance(
                    rulebook,
                    RuleBook
                ) else str(rulebook)
                d[mapp] = bookname
            else:
                d[mapp] = mapp + ":" + self._name
        self.engine.db.init_character(
            self.name,
            d['character'],
            d['avatar'],
            d['thing'],
            d['place'],
            d['portal']
        )
        self.thing = CharacterThingMapping(self)
        self.place = CharacterPlaceMapping(self)
        self.node = CharacterThingPlaceMapping(self)
        self.portal = CharacterPortalSuccessorsMapping(self)
        self.adj = self.portal
        self.succ = self.adj
        self.preportal = CharacterPortalPredecessorsMapping(self)
        self.pred = self.preportal
        self.avatar = CharacterAvatarGraphMapping(self)
        self.sense = CharacterSenseMapping(self)
        self.travel_reqs = FunList(
            self.engine,
            self.engine.prereq,
            'travel_reqs',
            ['character'],
            [name],
            'reqs'
        )
        self.stat = CharStatCache(self)
        if engine.caching:
            self._avatar_cache = ac = {}
            # I'll cache this ONE table in full, because iterating
            # over avatars seems to take a lot of time.
            for (g, n, b, t, a) in self.engine.db.avatars_ever(self.name):
                if g not in ac:
                    ac[g] = {}
                if n not in ac[g]:
                    ac[g][n] = {}
                if b not in ac[g][n]:
                    ac[g][n][b] = {}
                ac[g][n][b][t] = a
        self._stat_listeners = defaultdict(list)
        self._avatar_listeners = defaultdict(list)
        self._portal_traits = set()
        for stat in self.stat:
            assert(stat in self.stat)

    def _dispatch_stat(self, k, v):
        dispatch(self._stat_listeners, k, self, k, v)

    def stat_listener(self, f=None, stat=None):
        return listener(self._stat_listeners, f, stat)

    def _dispatch_avatar(self, k, v, ex):
        dispatch(
            self._avatar_listeners,
            k,
            self,
            self.engine.character[k],
            self.engine.character[k].node[v],
            ex
        )

    def avatar_listener(self, f=None, graph=None):
        return listener(self._avatar_listeners, f, graph)

    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        self._dispatch_stat(k, v)

    def __delitem__(self, k):
        super().__delitem__(k)
        self._dispatch_stat(k, None)

    def facade(self):
        return Facade(self)

    def travel_req(self, fun):
        """Decorator for tests that :class:`Thing`s have to pass before they
        can go thru :class:`Portal's

        """
        self.travel_reqs.append(fun)

    def add_place(self, name, **kwargs):
        """Create a new Place by the given name, and set its initial
        attributes based on the keyword arguments (if any).

        """
        self.place[name] = kwargs

    def add_places_from(self, seq):
        """Take a series of place names and add the lot."""
        super().add_nodes_from(seq)

    def new_place(self, name, **kwargs):
        self.add_place(name, **kwargs)
        return self.place[name]

    def add_thing(self, name, location, next_location=None, **kwargs):
        """Create a Thing, set its location and next_location (if provided),
        and set its initial attributes from the keyword arguments (if
        any).

        """
        super().add_node(name, **kwargs)
        self.place2thing(name, location, next_location)

    def add_things_from(self, seq):
        for tup in seq:
            name = tup[0]
            location = tup[1]
            next_loc = tup[2] if len(tup) > 2 else None
            kwargs = tup[3] if len(tup) > 3 else {}
            self.add_thing(name, location, next_loc, **kwargs)

    def new_thing(self, name, location, next_location=None, **kwargs):
        self.add_thing(name, location, next_location, **kwargs)
        return self.thing[name]

    def place2thing(self, name, location, next_location=None):
        """Turn a Place into a Thing with the given location and (if provided)
        next_location. It will keep all its attached Portals.

        """
        (branch, tick) = self.engine.time
        self.engine.db.thing_loc_and_next_set(
            self.name,
            name,
            branch,
            tick,
            location,
            next_location
        )
        if not self.engine.caching:
            return
        if (
                branch in self.place._keycache and
                tick in self.place._keycache[branch]
        ):
            self.place._keycache[branch][tick].discard(name)
        cache = self.thing._keycache
        if branch in cache and tick in cache[branch]:
            cache[branch][tick].add(name)
        else:
            if branch not in cache:
                cache[branch] = {}
            cache[branch][tick] = set(
                self.thing._iter_thing_names()
            )

    def thing2place(self, name):
        """Unset a Thing's location, and thus turn it into a Place."""
        self.engine.db.thing_loc_and_next_del(
            self.name,
            name,
            *self.engine.time
        )

    def add_portal(self, origin, destination, symmetrical=False, **kwargs):
        """Connect the origin to the destination with a :class:`Portal`.

        Keyword arguments are the :class:`Portal`'s
        attributes. Exception: if keyword ``symmetrical`` == ``True``,
        a mirror-:class:`Portal` will be placed in the opposite
        direction between the same nodes. It will always appear to
        have the placed :class:`Portal`'s stats, and any change to the
        mirror :class:`Portal`'s stats will affect the placed
        :class:`Portal`.

        """
        if origin.__class__ in (Place, Thing):
            origin = origin.name
        if destination.__class__ in (Place, Thing):
            destination = destination.name
        super(Character, self).add_edge(origin, destination, **kwargs)
        if symmetrical:
            self.add_portal(destination, origin, is_mirror=True)

    def new_portal(self, origin, destination, symmetrical=False, **kwargs):
        self.add_portal(origin, destination, symmetrical, **kwargs)
        return self.portal[origin][destination]

    def add_portals_from(self, seq, symmetrical=False):
        """Take a sequence of (origin, destination) pairs and make a
        :class:`Portal` for each.

        Actually, triples are acceptable too, in which case the third
        item is a dictionary of stats for the new :class:`Portal`.

        If optional argument ``symmetrical`` is set to ``True``, all
        the :class:`Portal` instances will have a mirror portal going
        in the opposite direction, which will always have the same
        stats.

        """
        for tup in seq:
            orig = tup[0]
            dest = tup[1]
            kwargs = tup[2] if len(tup) > 2 else {}
            if symmetrical:
                kwargs['symmetrical'] = True
            self.add_portal(orig, dest, **kwargs)

    def add_avatar(self, a, b=None):
        """Start keeping track of a :class:`Thing` or :class:`Place` in a
        different :class:`Character`.

        """
        if b is None:
            if not (
                    isinstance(a, Place) or
                    isinstance(a, Thing)
            ):
                raise TypeError(
                    'when called with one argument, '
                    'it must be a place or thing'
                )
            g = a.character.name
            n = a.name
        else:
            if isinstance(a, Character):
                g = a.name
            elif not isinstance(a, str):
                raise TypeError(
                    'when called with two arguments, '
                    'the first is a character or its name'
                )
            else:
                g = a
            if isinstance(b, Place) or isinstance(b, Thing):
                n = b.name
            elif not isinstance(b, str):
                raise TypeError(
                    'when called with two arguments, '
                    'the second is a thing/place or its name'
                )
            else:
                n = b
        (branch, tick) = self.engine.time
        if self.engine.caching:
            ac = self._avatar_cache
            if g not in ac:
                ac[g] = {}
            if n not in ac[g]:
                ac[g][n] = {}
            if branch not in ac[g][n]:
                ac[g][n][branch] = {}
            ac[g][n][branch][tick] = True
        # This will create the node if it doesn't exist. Otherwise
        # it's redundant but harmless.
        self.engine.db.exist_node(
            g,
            n,
            branch,
            tick,
            True
        )
        # Declare that the node is my avatar
        self.engine.db.avatar_set(
            self.character.name,
            g,
            n,
            branch,
            tick,
            True
        )
        self._dispatch_avatar(g, n, True)

    def del_avatar(self, node):
        """This is no longer my avatar, though it still exists on its own"""
        g = node.character.name
        n = node.name
        (branch, tick) = self.engine.time
        if self.engine.caching:
            ac = self._avatar_cache
            if g not in ac:
                ac[g] = {}
            if n not in ac[g]:
                ac[g][n] = {}
            if branch not in ac[g][n]:
                ac[g][n][branch] = {}
            ac[g][n][branch][tick] = False
        self.engine.db.avatar_set(
            self.character.name,
            g,
            n,
            branch,
            tick,
            False
        )
        self._dispatch_avatar(g, n, False)

    def portals(self):
        """Iterate over all portals"""
        for o in self.portal:
            for port in self.portal[o].values():
                yield port

    def avatars(self):
        """Iterate over all my avatars, regardless of what character they are
        in.

        """
        if not self.engine.caching:
            for (g, n, a) in self._db_iter_avatar_rows():
                if a:
                    yield self.engine.character[g].node[n]
            return
        ac = self._avatar_cache
        seen = set()
        for (branch, tick) in self.engine._active_branches():
            for g in ac:
                for n in ac[g]:
                    if (
                            (g, n) not in seen and
                            branch in ac[g][n]
                    ):
                        seen.add((g, n))
                        if ac[g][n][branch][
                                max(t for t in ac[g][n][branch] if t <= tick)
                        ]:
                            # the character or avatar may have been
                            # deleted from the world. It remains
                            # "mine" in case it comes back, but don't
                            # yield things that don't exist.
                            if (
                                    g in self.engine.character and
                                    n in self.engine.character[g]
                            ):
                                yield self.engine.character[g].node[n]

    def _db_iter_avatar_rows(self):
        yield from self.engine.db.avatars_now(
            self.character.name,
            *self.engine.time
        )
