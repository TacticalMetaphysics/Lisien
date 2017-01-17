# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Proxy objects to access LiSE entities from another process."""
import sys
import syslog
import logging
from os import getpid
from collections import (
    Mapping,
    MutableMapping,
    MutableSequence
)
from threading import Thread, Lock
from multiprocessing import Process, Pipe, Queue, ProcessError
from queue import Empty

from .engine import AbstractEngine
from .character import Facade
from allegedb.xjson import JSONReWrapper, JSONListReWrapper
from .util import reify
from allegedb.cache import PickyDefaultDict, StructuredDefaultDict
from .handle import EngineHandle


class CachingProxy(MutableMapping):
    @property
    def rulebook(self):
        if not hasattr(self, '_rulebook'):
            self._rulebook = self._get_rulebook()
        return self._rulebook

    @rulebook.setter
    def rulebook(self, v):
        rb = v.name if hasattr(v, 'name') else v
        self._rulebook = v if isinstance(v, RuleBookProxy) \
                         else RuleBookProxy(self.engine, rb)
        self._set_rulebook(rb)
        self.dispatch('rulebook', rb)

    def __init__(self, engine_proxy):
        self._listeners = []
        self.engine = engine_proxy
        self.exists = True

    def __bool__(self):
        return bool(self.exists)

    def __iter__(self):
        yield from self._cache

    def __len__(self):
        return len(self._cache)

    def __contains__(self, k):
        return k in self._cache

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("No such key: {}".format(k))
        return self._cache[k]

    def __setitem__(self, k, v):
        self._set_item(k, v)
        self._cache[k] = self._cache_munge(k, v)
        self.dispatch('setitem', k, v)

    def __delitem__(self, k):
        if k not in self:
            raise KeyError("No such key: {}".format(k))
        self._del_item(k)
        del self._cache[k]
        self.dispatch('delitem', k)

    def _apply_diff(self, diff):
        for (k, v) in diff.items():
            if v is None:
                if k in self._cache:
                    del self._cache[k]
                    self.dispatch('delitem', k)
            elif k not in self._cache or self._cache[k] != v:
                self._cache[k] = v
                self.dispatch('setitem', k, v)

    def update_cache(self):
        diff = self._get_diff()
        self.exists = diff is not None
        if not self.exists:
            self._cache = {}
            self.dispatch('deleted')
            return
        self._apply_diff(diff)

    def _get_diff(self):
        raise NotImplementedError("Abstract method")

    def _cache_munge(self, k, v):
        raise NotImplementedError("Abstract method")

    def _set_item(self, k, v):
        raise NotImplementedError("Abstract method")

    def _del_item(self, k):
        raise NotImplementedError("Abstract method")

    def listener(self, fun):
        if fun not in self._listeners:
            self._listeners.append(fun)

    def unlisten(self, fun):
        if fun in self._listeners:
            self._listeners.remove(fun)

    def dispatch(self, *args, **kwargs):
        for fun in self._listeners:
            fun(self, *args, **kwargs)


class CachingEntityProxy(CachingProxy):
    def _cache_munge(self, k, v):
        return self.engine.json_rewrap(v)

    def __repr__(self):
        return "{}({}) {}".format(
            self.__class__.__name__, self._cache, self.name
        )


class NodeProxy(CachingEntityProxy):
    @property
    def character(self):
        return self.engine.character[self._charname]

    @property
    def _cache(self):
        return self.engine._node_stat_cache[self._charname][self.name]

    def _get_rulebook(self):
        return RuleBookProxy(self.engine, self._get_rulebook_name())

    def _get_rulebook_name(self):
        r = self.engine.handle(
            command='get_node_rulebook',
            char=self._charname,
            node=self.name
        )
        if r is None:
            self.engine.handle(
                command='set_node_rulebook',
                char=self._charname,
                node=self.name,
                rulebook=(self._charname, self.name),
                silent=True
            )
            return (self._charname, self.name)
        return r

    def _set_rulebook(self, rb):
        self.engine.handle(
            'set_node_rulebook',
            char=self._charname, node=self.name, rulebook=rb
        )

    def __init__(self, engine_proxy, charname, nodename):
        self._charname = charname
        self.name = nodename
        super().__init__(engine_proxy)

    def __iter__(self):
        yield from super().__iter__()
        yield 'character'
        yield 'name'

    def __eq__(self, other):
        return (
            isinstance(other, NodeProxy) and
            self._charname == other._charname and
            self.name == other.name
        )

    def __hash__(self):
        return hash((self._charname, self.name))

    def __contains__(self, k):
        if k in ('character', 'name'):
            return True
        return super().__contains__(k)

    def __getitem__(self, k):
        if k == 'character':
            return self._charname
        elif k == 'name':
            return self.name
        return super().__getitem__(k)

    def _get_state(self):
        return self.engine.handle(
            command='node_stat_copy',
            char=self._charname,
            node=self.name
        )

    def _get_diff(self):
        return self.engine.handle(
            command='node_stat_diff',
            char=self._charname,
            node=self.name
        )

    def _set_item(self, k, v):
        if k == 'name':
            raise KeyError("Nodes can't be renamed")
        self.engine.handle(
            command='set_node_stat',
            char=self._charname,
            node=self.name,
            k=k, v=v,
            silent=True,
            branching=True
        )

    def _del_item(self, k):
        if k == 'name':
            raise KeyError("Nodes need names")
        self.engine.handle(
            command='del_node_stat',
            char=self._charname,
            node=self.name,
            k=k,
            silent=True,
            branching=True
        )

    def delete(self):
        self.engine.del_node(self._charname, self.name)


class PlaceProxy(NodeProxy):
    def __repr__(self):
        return "proxy to {}.place[{}]".format(
            self._charname,
            self.name
        )


class ThingProxy(NodeProxy):
    @property
    def location(self):
        return self.engine.character[self._charname].node[self._location]

    @location.setter
    def location(self, v):
        if isinstance(v, NodeProxy):
            if v.character != self.character:
                raise ValueError(
                    "Things can only be located in their character. "
                    "Maybe you want an avatar?"
                )
            locn = v.name
        elif v in self.character.node:
            locn = v
        else:
            raise TypeError("Location must be a node or the name of one")
        self._set_location(locn)

    @property
    def next_location(self):
        if self._next_location is None:
            return None
        return self.engine.character[self._charname].node[self._next_location]

    @next_location.setter
    def next_location(self, v):
        if isinstance(v, NodeProxy):
            if v.character != self.character:
                raise ValueError(
                    "Things can only be located in their character. "
                    "Maybe you want an avatar?"
                )
            locn = v.name
        elif v in self.character.node:
            locn = v
        else:
            raise TypeError("Location must be a node or the name of one")
        self._set_next_location(locn)

    def __init__(
            self, engine, character, name, location, next_location,
            arrival_time, next_arrival_time
    ):
        if location is None:
            raise TypeError("Things must have locations")
        super().__init__(engine, character, name)
        self._location = location
        self._next_location = next_location
        self._arrival_time = arrival_time
        self._next_arrival_time = next_arrival_time

    def __iter__(self):
        yield from super().__iter__()
        yield from {
            'location',
            'next_location',
            'arrival_time',
            'next_arrival_time'
        }

    def __getitem__(self, k):
        if k in {
                'location',
                'next_location',
                'arrival_time',
                'next_arrival_time'
        }:
            return getattr(self, '_' + k)
        return super().__getitem__(k)

    def _set_location(self, v):
        self._location = v
        self.engine.handle(
            command='set_thing_location',
            char=self.character.name,
            thing=self.name,
            loc=v
        )
        self.dispatch('location', v)

    def _set_next_location(self, v):
        self._next_location = v
        self.engine.handle(
            command='set_thing_next_location',
            char=self.character.name,
            thing=self.name,
            loc=v
        )
        self.dispatch('next_location', v)

    def __setitem__(self, k, v):
        if k == 'location':
            self._set_location(v)
        elif k == 'next_location':
            self._set_next_location(v)
        elif k in {'arrival_time', 'next_arrival_time'}:
            raise ValueError("Read-only")
        else:
            super().__setitem__(k, v)

    def __repr__(self):
        if self._next_location is not None:
            return "proxy to {}.thing[{}]@{}->{}".format(
                self._charname,
                self.name,
                self._location,
                self._next_location
            )
        return "proxy to {}.thing[{}]@{}".format(
            self._charname,
            self.name,
            self._location
        )

    def update_cache(self):
        (loc, next_loc, arrt, next_arrt) = self.engine.handle(
            command='get_thing_special_stats',
            char=self._charname, thing=self.name
        )
        if loc is None:
            self.exists = False
            self._cache = {}
            return
        if loc != self._location:
            self._location = loc
            self.dispatch('location', loc)
        if next_loc != self._next_location:
            self._next_location = next_loc
            self.dispatch('next_location', next_loc)
        if arrt != self._arrival_time:
            self._arrival_time = arrt
            self.dispatch('arrival_time', arrt)
        if next_arrt != self._next_arrival_time:
            self._next_arrival_time = next_arrt
            self.dispatch('next_arrival_time', next_arrt)
        super().update_cache()

    def follow_path(self, path, weight=None):
        self.engine.handle(
            command='thing_follow_path',
            char=self._charname,
            thing=self.name,
            path=path,
            weight=weight,
            silent=True
        )

    def go_to_place(self, place, weight=None):
        if hasattr(place, 'name'):
            place = place.name
        self.engine.handle(
            command='thing_go_to_place',
            char=self._charname,
            thing=self.name,
            place=place,
            weight=weight,
            silent=True
        )

    def travel_to(self, dest, weight=None, graph=None):
        if hasattr(dest, 'name'):
            dest = dest.name
        if hasattr(graph, 'name'):
            graph = graph.name
        self.engine.handle(
            command='thing_travel_to',
            char=self._charname,
            thing=self.name,
            dest=dest,
            weight=weight,
            graph=graph,
            silent=True
        )

    def travel_to_by(self, dest, arrival_tick, weight=None, graph=None):
        if hasattr(dest, 'name'):
            dest = dest.name
        if hasattr(graph, 'name'):
            graph = graph.name
        self.engine.handle(
            command='thing_travel_to_by',
            char=self._charname,
            thing=self.name,
            dest=dest,
            arrival_tick=arrival_tick,
            weight=weight,
            graph=graph,
            silent=True
        )


class PortalProxy(CachingEntityProxy):
    def _set_rulebook(self, rb):
        self.engine.handle(
            command='set_portal_rulebook',
            char=self._charname,
            orig=self._origin,
            dest=self._destination,
            rulebook=rb,
            silent=True
        )

    def _get_rulebook_name(self):
        return self.engine.handle(
            command='get_portal_rulebook',
            char=self._charname,
            orig=self._origin,
            dest=self._destination
        )

    def _get_rulebook(self):
        return RuleBookProxy(self.engine, self._get_rulebook_name())

    @property
    def _cache(self):
        return self.engine._portal_stat_cache[self._charname][
            self._origin][self._destination]

    @property
    def character(self):
        return self.engine.character[self._charname]

    @property
    def origin(self):
        return self.character.node[self._origin]

    @property
    def destination(self):
        return self.character.node[self._destination]

    def _get_diff(self):
        return self.engine.handle(
            commnad='portal_stat_diff',
            char=self._charname,
            orig=self._origin,
            dest=self._destination
        )

    def _set_item(self, k, v):
        self.engine.handle(
            command='set_portal_stat',
            char=self._charname,
            orig=self._origin,
            dest=self._destination,
            k=k, v=v,
            silent=True,
            branching=True
        )

    def _del_item(self, k):
        self.engine_handle(
            command='del_portal_stat',
            char=self._charname,
            orig=self._origin,
            dest=self._destination,
            k=k,
            silent=True,
            branching=True
        )

    def __init__(self, engine_proxy, charname, nodeAname, nodeBname):
        self._charname = charname
        self._origin = nodeAname
        self._destination = nodeBname
        super().__init__(engine_proxy)

    def __eq__(self, other):
        return (
            hasattr(other, 'character') and
            hasattr(other, 'origin') and
            hasattr(other, 'destination') and
            self.character == other.character and
            self.origin == other.origin and
            self.destination == other.destination
        )

    def __repr__(self):
        return "proxy to {}.portal[{}][{}]".format(
            self._charname,
            self._origin,
            self._destination
        )

    def __getitem__(self, k):
        if k == 'origin':
            return self._origin
        elif k == 'destination':
            return self._destination
        elif k == 'character':
            return self._charname
        return super().__getitem__(k)

    def delete(self):
        self.engine.del_portal(self._charname, self._origin, self._destination)


class NodeMapProxy(MutableMapping):
    @property
    def character(self):
        return self.engine.character[self._charname]

    def __init__(self, engine_proxy, charname):
        self.engine = engine_proxy
        self._charname = charname

    def __iter__(self):
        yield from self.character.thing
        yield from self.character.place

    def __len__(self):
        return len(self.character.thing) + len(self.character.place)

    def __getitem__(self, k):
        if k in self.character.thing:
            return self.character.thing[k]
        else:
            return self.character.place[k]

    def __setitem__(self, k, v):
        self.character.place[k] = v

    def __delitem__(self, k):
        if k in self.character.thing:
            del self.character.thing[k]
        else:
            del self.character.place[k]


class ThingMapProxy(CachingProxy):
    @property
    def rulebook(self):
        return self.engine._character_rulebooks_cache[self.name]['thing']

    @property
    def character(self):
        return self.engine.character[self.name]

    @property
    def _cache(self):
        return self.engine._things_cache[self.name]

    def __init__(self, engine_proxy, charname):
        self.name = charname
        super().__init__(engine_proxy)

    def __eq__(self, other):
        return self is other

    def _apply_diff(self, diff):
        for (
                thing, (
                    location,
                    next_location,
                    arrival_time,
                    next_arrival_time
                )
        ) in diff.items():
            if location:
                if thing in self._cache:
                    thisthing = self._cache[thing]
                    if thisthing._location != location:
                        thisthing._location = location
                        thisthing.dispatch('location', location)
                    if thisthing._next_location != next_location:
                        thisthing._next_location = next_location
                        thisthing.dispatch('next_location', next_location)
                    if thisthing._arrival_time != arrival_time:
                        thisthing._arrival_time = arrival_time
                        thisthing.dispatch('arrival_time', arrival_time)
                    if thisthing._next_arrival_time != next_arrival_time:
                        thisthing._next_arrival_time = next_arrival_time
                        thisthing.dispatch(
                            'next_arrival_time', next_arrival_time
                        )
                else:
                    self._cache[thing] = ThingProxy(
                        self.engine,
                        self.name,
                        thing,
                        location,
                        next_location,
                        arrival_time,
                        next_arrival_time
                    )
            elif thing in self._cache:
                thing.dispatch('deleted')
                del self._cache[thing]

    def _get_diff(self):
        return self.engine.handle(
            command='character_things_diff',
            char=self.name
        )

    def _cache_munge(self, k, v):
        return ThingProxy(
            self.engine, self.name, *self.engine.handle(
                'get_thing_special_stats', char=self.name, thing=k
            )
        )

    def _set_item(self, k, v):
        self.engine.handle(
            command='set_thing',
            char=self.name,
            thing=k,
            statdict=v,
            silent=True,
            branching=True
        )
        self._cache[k] = ThingProxy(
            self.engine, self.name,
            v.pop('location'), v.pop('next_location', None),
            v.pop('arrival_time', None), v.pop('next_arrival_time', None)
        )
        self.engine._node_stat_cache[self.name][k] = v

    def _del_item(self, k):
        self.engine.handle(
            command='del_node',
            char=self.name,
            node=k,
            silent=True,
            branching=True
        )
        del self._cache[k]
        del self.engine._node_stat_cache[self.name][k]


class PlaceMapProxy(CachingProxy):
    @property
    def rulebook(self):
        return self.engine._character_rulebooks_cache[self.name]['place']

    @property
    def character(self):
        return self.engine.character[self.name]

    @property
    def _cache(self):
        return self.engine._character_places_cache[self.name]

    def __init__(self, engine_proxy, character):
        self.name = character
        super().__init__(engine_proxy)

    def __eq__(self, other):
        return self is other

    def _apply_diff(self, diff):
        for (place, ex) in diff.items():
            if ex:
                if place not in self._cache:
                    self._cache[place] = PlaceProxy(
                        self.engine,
                        self.name,
                        place
                    )
            else:
                if place in self._cache:
                    del self._cache[place]

    def _get_diff(self):
        return self.engine.handle(
            command='character_places_diff',
            char=self.name
        )

    def _cache_munge(self, k, v):
        return PlaceProxy(
            self.engine, self.name, k
        )

    def _set_item(self, k, v):
        self.engine.handle(
            command='set_place',
            char=self.name,
            place=k, statdict=v,
            silent=True,
            branching=True
        )
        self.engine._node_stat_cache[self.name][k] = v

    def _del_item(self, k):
        self.engine.handle(
            command='del_node',
            char=self.name,
            node=k,
            silent=True,
            branching=True
        )
        del self.engine._node_stat_cache[self.name][k]


class SuccessorsProxy(CachingProxy):
    @property
    def _cache(self):
        return self.engine._character_portals_cache.successors[
            self._charname][self._nodeA]

    def __init__(self, engine_proxy, charname, nodeAname):
        self._charname = charname
        self._nodeA = nodeAname
        super().__init__(engine_proxy)

    def __eq__(self, other):
        return (
            isinstance(other, SuccessorsProxy) and
            self.engine is other.engine and
            self._charname == other._charname and
            self._nodeA == other._nodeA
        )

    def _get_state(self):
        return {
            node: self._cache[node] if node in self._cache else
            PortalProxy(self.engine, self._charname, self._nodeA, node)
            for node in self.engine.handle(
                command='node_successors',
                char=self._charname,
                node=self._nodeA
            )
        }

    def _apply_diff(self, diff):
        raise NotImplementedError(
            "Apply the diff on CharSuccessorsMappingProxy"
        )

    def _get_diff(self):
        return self.engine.handle(
            command='node_successors_diff',
            char=self._charname,
            node=self._nodeA
        )

    def _cache_munge(self, k, v):
        if isinstance(v, PortalProxy):
            assert v._origin == self._nodeA
            assert v._destination == k
            return v
        return PortalProxy(
            self.engine,
            self._charname,
            self._nodeA,
            k
        )

    def _set_item(self, nodeB, value):
        self.engine.handle(
            command='set_portal',
            char=self._charname,
            orig=self._nodeA,
            dest=nodeB,
            statdict=value,
            silent=True,
            branching=True
        )

    def _del_item(self, nodeB):
        self.engine.del_portal(self._charname, self._nodeA, nodeB)


class CharSuccessorsMappingProxy(CachingProxy):
    @property
    def rulebook(self):
        return self.engine._character_rulebooks_cache[self._charname]['portal']

    @property
    def character(self):
        return self.engine.character[self._charname]

    @property
    def _cache(self):
        return self.engine._character_portals_cache.successors[self.name]

    def __init__(self, engine_proxy, charname):
        self.name = charname
        super().__init__(engine_proxy)

    def __eq__(self, other):
        return (
            isinstance(other, CharSuccessorsMappingProxy) and
            other.engine is self.engine and
            other.name == self.name
        )

    def _cache_munge(self, k, v):
        return {
            vk: PortalProxy(self.engine, self.name, vk, vv)
            for (vk, vv) in v.items()
        }

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("No portals from {}".format(k))
        return SuccessorsProxy(
            self.engine,
            self.name,
            k
        )

    def _apply_diff(self, diff):
        for ((o, d), ex) in diff.items():
            if ex:
                if d not in self._cache[o]:
                    self._cache[o][d] = PortalProxy(
                        self.engine,
                        self.name,
                        o, d
                    )
            else:
                if o in self._cache and d in self._cache[o]._cache:
                    del self._cache[o]._cache[d]
                    if not self._cache[o]._cache:
                        del self._cache[o]

    def _get_diff(self):
        return self.engine.handle(
            command='character_nodes_with_successors_diff',
            character=self.name
        )

    def _set_item(self, nodeA, val):
        self.engine.handle(
            command='character_set_node_successors',
            character=self.name,
            node=nodeA,
            val=val,
            silent=True,
            branching=True
        )

    def _del_item(self, nodeA):
        for nodeB in self[nodeA]:
            self.engine.del_portal(self.name, nodeA, nodeB)


class PredecessorsProxy(MutableMapping):
    @property
    def character(self):
        return self.engine.character[self._charname]

    def __init__(self, engine_proxy, charname, nodeBname):
        self.engine = engine_proxy
        self._charname = charname
        self.name = nodeBname

    def __iter__(self):
        return iter(self.engine._character_portals_cache.predecessors[
            self._charname][self.name])

    def __len__(self):
        return len(self.engine._character_portals_cache.predecessors[
            self._charname][self.name])

    def __contains__(self, k):
        return k in self.engine._character_portals_cache.predecessors[
            self._charname][self.name]

    def __getitem__(self, k):
        return self.engine._character_portals_cache.predecessors[
            self._charname][self.name][k]

    def __setitem__(self, k, v):
        self.engine._place_stat_cache[self._charname][k] = v
        self.engine._character_portals_cache.store(
            self._charname,
            self.name,
            k,
            PortalProxy(self.engine, self._charname, k, self.name)
        )
        self.engine.handle(
            command='set_place',
            char=self._charname,
            place=k,
            statdict=v,
            silent=True
        )
        self.engine.handle(
            'set_portal',
            (self._charname, k, self.name),
            silent=True
        )

    def __delitem__(self, k):
        self.engine.del_portal(self._charname, k, self.name)


class CharPredecessorsMappingProxy(MutableMapping):
    @property
    def rulebook(self):
        return self.character.portal.rulebook

    def __init__(self, engine_proxy, charname):
        self.engine = engine_proxy
        self.name = charname
        self._cache = {}

    def __contains__(self, k):
        return k in self.engine._character_portals_cache.predecessors[self.name]

    def __iter__(self):
        return iter(self.engine._character_portals_cache.predecessors[self.name])

    def __len__(self):
        return len(self.engine._character_portals_cache.predecessors[self.name])

    def __getitem__(self, k):
        if k not in self:
            raise KeyError(
                "No predecessors to {} (if it even exists)".format(k)
            )
        if k not in self._cache:
            self._cache[k] = PredecessorsProxy(self.engine, self.name, k)
        return self._cache[k]

    def __setitem__(self, k, v):
        for pred, proxy in v.items():
            self.engine._character_portals_cache.store(
                self.name,
                pred,
                k,
                proxy
            )
        self.engine.handle(
            command='character_set_node_predecessors',
            char=self.name,
            node=k,
            preds=v,
            silent=True
        )

    def __delitem__(self, k):
        for v in self[k]:
            self.engine.del_portal(self.name, k, v)
        if k in self._cache:
            del self._cache[k]


class CharStatProxy(CachingEntityProxy):
    @property
    def _cache(self):
        return self.engine._char_stat_cache[self.name]

    def __init__(self, engine_proxy, character):
        self.name = character
        super().__init__(engine_proxy)

    def __eq__(self, other):
        return (
            isinstance(other, CharStatProxy) and
            self.engine is other.engine and
            self.name == other.name
        )

    def _get_state(self):
        return self.engine.handle(
            command='character_stat_copy',
            char=self.name
        )

    def _get_diff(self):
        return self.engine.handle(
            command='character_stat_diff',
            char=self.name
        )

    def _set_item(self, k, v):
        self.engine.handle(
            command='set_character_stat',
            char=self.name,
            k=k, v=v,
            silent=True,
            branching=True
        )

    def _del_item(self, k):
        self.engine.handle(
            command='del_character_stat',
            char=self.name,
            k=k,
            silent=True,
            branching=True
        )


class RuleProxy(object):
    @property
    def triggers(self):
        return self.all_rules._cache[self.name]['triggers']

    @triggers.setter
    def triggers(self, v):
        self.all_rules._cache[self.name]['triggers'] = v

    @property
    def prereqs(self):
        return self.all_rules._cache[self.name]['prereqs']

    @prereqs.setter
    def prereqs(self, v):
        self.all_rules._cache[self.name]['prereqs'] = v

    @property
    def actions(self):
        return self.all_rules._cache[self.name]['actions']

    @actions.setter
    def actions(self, v):
        self.all_rules._cache[self.name]['actions'] = v

    def __init__(self, all_rules, rulename):
        self.all_rules = all_rules
        self.engine = all_rules.engine
        self.name = self._name = rulename

    def __eq__(self, other):
        return (
            hasattr(other, 'name') and
            self.name == other.name
        )


class RuleBookProxy(MutableSequence):
    @property
    def _cache(self):
        return self.all_rulebooks._cache[self.name]

    def __init__(self, all_rulebooks,  bookname):
        self.all_rulebooks = all_rulebooks
        self.engine = all_rulebooks.engine
        self.name = bookname
        self._proxy_cache = {}

    def __iter__(self):
        for k in self._cache:
            if k not in self._proxy_cache:
                self._proxy_cache[k] = RuleProxy(self.engine, k)
            yield self._proxy_cache[k]

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, i):
        k = self._cache[i]
        if k not in self._proxy_cache:
            self._proxy_cache[k] = RuleProxy(self.engine, k)
        return self._proxy_cache[k]

    def __setitem__(self, i, v):
        if isinstance(v, RuleProxy):
            v = v._name
        self._cache[i] = v
        self.engine.handle(
            command='set_rulebook_rule',
            rulebook=self.name,
            i=i,
            rule=v,
            silent=True
        )

    def __delitem__(self, i):
        del self._cache[i]
        self.engine.handle(
            command='del_rulebook_rule',
            rulebook=self.name,
            i=i,
            silent=True
        )

    def insert(self, i, v):
        if isinstance(v, RuleProxy):
            v = v._name
        self._cache.insert(i, v)
        self.engine.handle(
            command='ins_rulebook_rule',
            rulebook=self.name,
            i=i,
            rule=v,
            silent=True
        )


class AvatarMapProxy(Mapping):
    @property
    def rulebook(self):
        return self.engine._character_rulebooks_cache[
            self.character.name]['avatar']

    def __init__(self, character):
        self.character = character

    def __iter__(self):
        yield from self.character.engine._character_avatars_cache[
            self.character.name]

    def __len__(self):
        return len(self.character.engine._character_avatars_cache[
            self.character.name])

    def __contains__(self, k):
        return k in self.character.engine._character_avatars_cache[
            self.character.name]

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("{} has no avatar in {}".format(
                self.character.name, k
            ))
        return self.GraphAvatarsProxy(
            self.character, self.character.engine.character[k]
        )

    def __getattr__(self, attr):
        vals = self.values()
        if not vals:
            raise AttributeError(
                "No attribute {}, and no graph to delegate to".format(attr)
            )
        elif len(vals) > 1:
            raise AttributeError(
                "No attribute {}, and more than one graph".format(attr)
            )
        else:
            return getattr(next(iter(vals)), attr)

    class GraphAvatarsProxy(Mapping):
        def __init__(self, character, graph):
            self.character = character
            self.graph = graph

        def __iter__(self):
            yield from self.character.engine._character_avatars_cache[
                self.character.name][self.graph.name]

        def __len__(self):
            return len(self.character.engine._character_avatars_cache[
                self.character.name][self.graph.name])

        def __contains__(self, k):
            cache = self.character.engine._character_avatars_cache[
                self.character.name]
            return self.graph.name in cache and k in cache[self.graph.name]

        def __getitem__(self, k):
            if k not in self:
                raise KeyError("{} has no avatar {} in graph {}".format(
                    self.character.name, k, self.graph.name
                ))
            return self.graph.node[k]

        def __getattr__(self, attr):
            vals = self.values()
            if not vals:
                raise AttributeError(
                    "No attribute {}, "
                    "and no avatar to delegate to".format(attr)
                )
            elif len(vals) > 1:
                raise AttributeError(
                    "No attribute {}, and more than one avatar"
                )
            else:
                return getattr(next(iter(vals)), attr)


class CharacterProxy(MutableMapping):
    @property
    def rulebook(self):
        return self.engine._character_rulebooks_cache[self.name]['character']

    @reify
    def avatar(self):
        return AvatarMapProxy(self)

    def __init__(self, engine_proxy, charname):
        self.engine = engine_proxy
        self.name = charname
        self.adj = self.succ = self.portal = CharSuccessorsMappingProxy(
            self.engine, self.name
        )
        self.pred = self.preportal = CharPredecessorsMappingProxy(
            self.engine, self.name
        )
        self.thing = ThingMapProxy(self.engine, self.name)
        self.place = PlaceMapProxy(self.engine, self.name)
        self.node = NodeMapProxy(self.engine, self.name)
        self.stat = CharStatProxy(self.engine, self.name)

    def __bool__(self):
        return True

    def __eq__(self, other):
        if hasattr(other, 'engine'):
            oe = other.engine
        else:
            return False
        return (
            self.engine is oe and
            hasattr(other, 'name') and
            self.name == other.name
        )

    def __iter__(self):
        yield from self.engine.handle(
            command='character_nodes',
            char=self.name
        )

    def __len__(self):
        return self.engine.handle(
            command='character_nodes_len',
            char=self.name
        )

    def __contains__(self, k):
        if k == 'name':
            return True
        return k in self.node

    def __getitem__(self, k):
        if k == 'name':
            return self.name
        return self.node[k]

    def __setitem__(self, k, v):
        self.node[k] = v

    def __delitem__(self, k):
        del self.node[k]

    def _apply_diff(self, diff):
        self.stat._apply_diff(diff['character_stat'])
        self.thing._apply_diff(diff['things'])
        self.place._apply_diff(diff['places'])
        self.portal._apply_diff(diff['portals'])
        for (node, nodediff) in diff['node_stat'].items():
            if node not in self.engine._node_stat_cache[self.name]:
                self.engine._node_stat_cache[self.name][node] = nodediff
            else:
                self.node[node]._apply_diff(nodediff)
        for (orig, destdiff) in diff['portal_stat'].items():
            for (dest, portdiff) in destdiff.items():
                if orig in self.portal and dest in self.portal[orig]:
                    self.portal[orig][dest]._apply_diff(portdiff)
                else:
                    self.engine._portal_stat_cache[
                        self.name][orig][dest] = portdiff

    def add_place(self, name, **kwargs):
        self[name] = kwargs

    def add_places_from(self, seq):
        self.engine.handle(
            command='add_places_from',
            char=self.name,
            seq=list(seq),
            silent=True
        )
        for pln in seq:
            self.place._cache[pln] = PlaceProxy(
                self.engine, self.name, pln
            )

    def add_nodes_from(self, seq):
        self.add_places_from(seq)

    def add_thing(self, name, location, next_location=None, **kwargs):
        self.engine.handle(
            command='add_thing',
            char=self.name,
            thing=name,
            loc=location,
            next_loc=next_location,
            statdict=kwargs,
            silent=True
        )
        self.thing._cache[name] = ThingProxy(
            self.engine, self.name, name, location, next_location,
            self.engine.tick, None
        )

    def add_things_from(self, seq):
        self.engine.handle(
            command='add_things_from',
            char=self.name,
            seq=list(seq),
            silent=True
        )
        for thn in seq:
            self.thing._cache[thn] = ThingProxy(
                self.engine, self.name, thn
            )

    def new_place(self, name, **kwargs):
        self.add_place(name, **kwargs)
        return self.place[name]

    def new_thing(self, name, location, next_location=None, **kwargs):
        self.add_thing(name, location, next_location, **kwargs)
        return self.thing[name]

    def place2thing(self, node, location, next_location=None):
        self.engine.handle(
            command='place2thing',
            char=self.name,
            node=node,
            loc=location,
            next_loc=next_location,
            silent=True
        )

    def add_portal(self, origin, destination, symmetrical=False, **kwargs):
        self.engine.handle(
            command='add_portal',
            char=self.name,
            orig=origin,
            dest=destination,
            symmetrical=symmetrical,
            statdict=kwargs,
            silent=True
        )
        self.portal._cache[origin][destination] = PortalProxy(
            self.engine,
            self.name,
            origin,
            destination
        )

    def add_portals_from(self, seq, symmetrical=False):
        l = list(seq)
        self.engine.handle(
            command='add_portals_from',
            char=self.name,
            seq=l,
            symmetrical=symmetrical,
            silent=True
        )
        for (origin, destination) in l:
            if origin not in self.portal._cache:
                self.portal._cache[origin] = SuccessorsProxy(
                    self.engine,
                    self.name,
                    origin
                )
            self.portal[origin]._cache[destination] = PortalProxy(
                self.engine,
                self.name,
                origin,
                destination
            )

    def new_portal(self, origin, destination, symmetrical=False, **kwargs):
        self.add_portal(origin, destination, symmetrical, **kwargs)
        return self.portal[origin][destination]

    def portals(self):
        yield from self.engine.handle(
            command='character_portals',
            char=self.name
        )

    def add_avatar(self, graph, node):
        self.engine.handle(
            command='add_avatar',
            char=self.name,
            graph=graph,
            node=node,
            silent=True
        )

    def del_avatar(self, graph, node):
        self.engine.handle(
            command='del_avatar',
            char=self.name,
            graph=graph,
            node=node,
            silent=True
        )

    def avatars(self):
        yield from self.engine.handle(
            command='character_avatars',
            char=self.name
        )

    def facade(self):
        return Facade(self)


class CharacterMapProxy(MutableMapping):
    def __init__(self, engine_proxy):
        self.engine = engine_proxy

    def __iter__(self):
        return iter(self.engine._char_cache.keys())

    def __contains__(self, k):
        return k in self.engine._char_cache

    def __len__(self):
        return len(self.engine._char_cache)

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("No character: {}".format(k))
        return self.engine._char_cache[k]

    def __setitem__(self, k, v):
        if isinstance(v, CharacterProxy):
            return
        self.engine.handle(
            command='set_character',
            char=k,
            data=v,
            silent=True
        )
        self.engine._char_cache[k] = CharacterProxy(self.engine, k)

    def __delitem__(self, k):
        self.engine.handle(
            command='del_character',
            char=k,
            silent=True
        )
        if k in self.engine._char_cache:
            del self.engine._char_cache[k]


class StringStoreProxy(MutableMapping):
    @property
    def language(self):
        if not hasattr(self, '_l'):
            self._l = self.engine.handle(command='get_language')
        return self._l

    @language.setter
    def language(self, v):
        self.engine.handle(command='set_language', lang=v)
        self._l = v

    def __init__(self, engine_proxy):
        self.engine = engine_proxy
        self._cache = self.engine.handle('strings_diff')

    def __iter__(self):
        yield from self._cache

    def __contains__(self, k):
        return k in self._cache

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, k):
        return self._cache[k]

    def __setitem__(self, k, v):
        self._cache[k] = v
        self.engine.handle(command='set_string', k=k, v=v, silent=True)

    def __delitem__(self, k):
        del self._cache[k]
        self.engine.handle(command='del_string', k=k, silent=True)

    def lang_items(self, lang=None):
        if lang is None or lang == self.language:
            yield from self._cache.items()
        else:
            yield from self.engine.handle(
                command='get_string_lang_items', lang=lang
            )


class EternalVarProxy(MutableMapping):
    def __init__(self, engine_proxy):
        self.engine = engine_proxy
        self._cache = self.engine.handle('eternal_diff')

    def __contains__(self, k):
        return k in self._cache

    def __iter__(self):
        yield from self.engine.handle(command='eternal_keys')

    def __len__(self):
        return self.engine.handle(command='eternal_len')

    def __getitem__(self, k):
        return self.engine.handle(command='get_eternal', k=k)

    def __setitem__(self, k, v):
        self._cache[k] = v
        self.engine.handle(
            'set_eternal',
            k=k, v=v,
            silent=True
        )

    def __delitem__(self, k):
        del self._cache[k]
        self.engine.handle(
            command='del_eternal',
            k=k,
            silent=True
        )


class GlobalVarProxy(MutableMapping):
    def __init__(self, engine_proxy):
        self.engine = engine_proxy
        self._cache = self.engine.handle('universal_diff')

    def __iter__(self):
        return iter(self._cache)

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, k):
        return self._cache[k]

    def __setitem__(self, k, v):
        self._cache[k] = v
        self.engine.handle('set_universal', k=k, v=v)

    def __delitem__(self, k):
        del self._cache[k]
        self.engine.handle('del_universal', k=k)


class AllRuleBooksProxy(Mapping):
    def __init__(self, engine_proxy):
        self.engine = engine_proxy
        self._cache = self.engine.handle('all_rulebooks_diff')

    def __iter__(self):
        yield from self._cache

    def __len__(self):
        return len(self._cache)

    def __contains__(self, k):
        return k in self._cache

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("No rulebook: {}".format(k))
        return self._cache[k]


class AllRulesProxy(Mapping):
    def __init__(self, engine_proxy):
        self.engine = engine_proxy
        self._cache = self.engine.handle('all_rules_diff')
        self._proxy_cache = {}

    def __iter__(self):
        return iter(self._cache)

    def __len__(self):
        return len(self._cache)

    def __contains__(self, k):
        return k in self._cache

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("No rule: {}".format(k))
        if k not in self._proxy_cache:
            self._proxy_cache[k] = RuleProxy(self, k)
        return self._proxy_cache[k]

    def new_empty(self, k):
        self.engine.handle(command='new_empty_rule', rule=k, silent=True)
        self._cache[k] = []
        self._proxy_cache[k] = RuleProxy(self, k)
        return self._proxy_cache[k]


class FuncStoreProxy(object):
    def __init__(self, engine_proxy, store):
        self.engine = engine_proxy
        self._store = store
        self._cache = self.engine.handle('source_diff', store=store)

    def __iter__(self):
        return iter(self._cache)

    def __len__(self):
        return len(self._cache)

    def plain(self, k):
        return self._cache[k]

    def iterplain(self):
        yield from self._cache.values()

    def set_source(self, func_name, source):
        self.engine.handle(
            command='set_source', store=self._store, k=func_name, v=source
        )
        self._cache[func_name] = source


class ChangeSignatureError(TypeError):
    pass


class PortalObjCache(object):
    def __init__(self):
        self.successors = StructuredDefaultDict(2, PortalProxy)
        self.predecessors = StructuredDefaultDict(2, PortalProxy)

    def store(self, char, u, v, obj):
        self.successors[char][u][v] = obj
        self.predecessors[char][v][u] = obj

    def delete(self, char, u, v):
        del self.successors[char][u][v]
        del self.predecessors[char][v][u]


class EngineProxy(AbstractEngine):
    char_cls = CharacterProxy
    thing_cls = ThingProxy
    place_cls = PlaceProxy
    portal_cls = PortalProxy

    @property
    def branch(self):
        return self._branch

    @branch.setter
    def branch(self, v):
        self.handle(command='set_branch', branch=v, silent=True)
        self._branch = v
        for f in self._branch_listeners:
            f(self, v)

    def branch_listener(self, f):
        if f not in self._branch_listeners:
            self._branch_listeners.append(f)

    @property
    def tick(self):
        return self._tick

    @tick.setter
    def tick(self, v):
        self.handle('set_tick', tick=v, silent=True)
        self._tick = v
        for f in self._tick_listeners:
            f(self, v)

    def tick_listener(self, f):
        if f not in self._tick_listeners:
            self._tick_listeners.append(f)

    @property
    def time(self):
        return (self._branch, self._tick)

    @time.setter
    def time(self, v):
        (branch, tick) = (self._branch, self._tick) = v
        self.handle(command='set_time', branch=branch, tick=tick, silent=True)

    def __init__(
            self, handle_out, handle_in, logger,
            do_game_start=False,  install_modules=[]
    ):
        self._handle_out = handle_out
        self._handle_out_lock = Lock()
        self._handle_in = handle_in
        self._handle_in_lock = Lock()
        self._handle_lock = Lock()
        self._branch_listeners = []
        self._tick_listeners = []
        self.logger = logger
        self.method = FuncStoreProxy(self, 'method')
        self.eternal = EternalVarProxy(self)
        self.universal = GlobalVarProxy(self)
        self.character = CharacterMapProxy(self)
        self.string = StringStoreProxy(self)
        self.rulebook = AllRuleBooksProxy(self)
        self.rule = AllRulesProxy(self)
        self.action = FuncStoreProxy(self, 'action')
        self.prereq = FuncStoreProxy(self, 'prereq')
        self.trigger = FuncStoreProxy(self, 'trigger')
        self.function = FuncStoreProxy(self, 'function')
        (self._branch, self._tick) = self.handle(command='get_watched_time')

        for module in install_modules:
            self.handle('install_module',  module=module)  # not silenced
        if do_game_start:
            # not silenced; mustn't do anything before the game has started
            self.handle('do_game_start')

        self._node_stat_cache = StructuredDefaultDict(1, dict)
        self._portal_stat_cache = StructuredDefaultDict(2, dict)
        self._char_stat_cache = PickyDefaultDict(dict)
        self._things_cache = StructuredDefaultDict(1, ThingProxy)
        self._character_places_cache = StructuredDefaultDict(1, PlaceProxy)
        self._character_rulebooks_cache = StructuredDefaultDict(
            1, RuleBookProxy
        )
        self._char_node_rulebooks_cache = StructuredDefaultDict(
            1, RuleBookProxy
        )
        self._char_port_rulebooks_cache = StructuredDefaultDict(
            2, RuleBookProxy
        )
        self._character_portals_cache = PortalObjCache()
        self._character_avatars_cache = PickyDefaultDict(dict)
        self._char_cache = {}
        self._rules_cache = self.handle('all_rules_diff')
        self._rulebooks_cache = self.handle('all_rulebooks_diff')
        charsdiffs = self.handle('get_chardiffs', chars='all')
        for char in charsdiffs:
            self._char_cache[char] = CharacterProxy(self, char)
            self._char_stat_cache[char] = charsdiffs[char]['character_stat']
            for origin, destinations in charsdiffs[
                    char]['portal_stat'].items():
                for destination,  stats in destinations.items():
                    self._portal_stat_cache[char][origin][destination] = stats
            for node,  stats in charsdiffs[char]['node_stat'].items():
                self._node_stat_cache[char][node] = stats
            self._character_avatars_cache[char] = charsdiffs[char]['avatars']
            for rbtype, rb in charsdiffs[char]['rulebooks'].items():
                self._character_rulebooks_cache[char][rbtype] \
                    = RuleBookProxy(self.rulebook, rb)
            for node, rb in charsdiffs[char]['node_rulebooks'].items():
                self._char_node_rulebooks_cache[char][node] \
                    = RuleBookProxy(self.rulebook, rb)
            for origin, destinations in charsdiffs[
                    char]['portal_rulebooks'].items():
                for destination, rulebook in destinations.items():
                    self._char_port_rulebooks_cache[
                        char][origin][destination] = RuleBookProxy(
                            self.rulebook, rulebook
                        )
            for (
                    thing, (loc, nxloc, arrt, nxarrt)
            ) in charsdiffs[char]['things'].items():
                if loc:
                    self._things_cache[char][thing] \
                        = ThingProxy(
                            self, char, thing, loc, nxloc, arrt, nxarrt
                        )
            for (place, ex) in charsdiffs[char]['places'].items():
                if ex:
                    self._character_places_cache[char][place] \
                        = PlaceProxy(self, char, place)
            for (orig, dest), ex in charsdiffs[char]['portals'].items():
                if ex:
                    self._character_portals_cache.store(
                        char, orig, dest, PortalProxy(self, char, orig, dest)
                    )

    def delistify(self, obj):
        if not (isinstance(obj, list) or isinstance(obj, tuple)):
            return obj
        if obj[0] == 'character':
            name = self.delistify(obj[1])
            if name not in self._chars_cache:
                self._chars_cache[name] = CharacterProxy(self, name)
            return self._chars_cache[name]
        elif obj[0] == 'node':
            charname = self.delistify(obj[1])
            nodename = self.delistify(obj[2])
            if charname in self._things_cache and \
               nodename in self._things_cache[charname]:
                return self._things_cache[charname][nodename]
            if charname in self._character_places_cache and \
               nodename in self._character_places_cache[charname]:
                return self._character_places_cache[charname][nodename]
            # I hate that I have to ask the subprocess about this.
            # Maybe change the serialization to always reflect
            # the distinction between place and thing
            (loc, nextloc, arrt, nextarrt) \
                = self.handle(
                    'get_thing_special_stats', char=charname, thing=nodename
                )
            if loc is not None:
                self._things_cache[charname][nodename] \
                    = ThingProxy(
                        self, charname, nodename, loc, nextloc, arrt, nextarrt
                    )
                return self._things_cache[charname][nodename]
            else:
                self._character_places_cache[charname][nodename] \
                    = PlaceProxy(self, charname, nodename)
                return self._character_places_cache[charname][nodename]
        elif obj[0] == 'portal':
            charname = self.delistify(obj[1])
            origname = self.delistify(obj[2])
            destname = self.delistify(obj[3])
            cache = self._character_portals_cache
            if not (
                    charname in cache and
                    origname in cache[charname] and
                    destname in cache[charname][origname]
            ):
                cache[charname][origname][destname] \
                    = PortalProxy(self, charname, origname, destname)
            return cache[charname][origname][destname]
        else:
            return super().delistify(obj)

    def send(self, obj, blocking=True, timeout=-1):
        self._handle_out_lock.acquire(blocking, timeout)
        self._handle_out.send(obj)
        self._handle_out_lock.release()

    def recv(self, blocking=True, timeout=-1):
        self._handle_in_lock.acquire(blocking, timeout)
        data = self._handle_in.recv()
        self._handle_in_lock.release()
        return data

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)

    def critical(self, msg):
        self.logger.critical(msg)

    def handle(self, cmd=None, **kwargs):
        if 'command' in kwargs:
            cmd = kwargs['command']
        elif cmd:
            kwargs['command'] = cmd
        else:
            raise TypeError("No command")
        branching = kwargs.pop('branching', False)
        self._handle_lock.acquire()
        if 'silent' not in kwargs:
            kwargs['silent'] = False
        self.send(self.json_dump(kwargs))
        if not kwargs['silent']:
            command,  result = self.recv()
            assert cmd == command, \
                "Sent command {} but received results for {}".format(
                    cmd, command
                )
            self._handle_lock.release()
            r = self.json_load(result)
            if branching:
                self._branch = r
                for f in self._branch_listeners:
                    f(self, r)
            return r
        self._handle_lock.release()

    def json_rewrap(self, r):
        if isinstance(r, tuple):
            if r[0] in ('JSONListReWrapper', 'JSONReWrapper'):
                cls = JSONReWrapper if r[0] == 'JSONReWrapper' \
                      else JSONListReWrapper
                if r[1] == 'character':
                    (charn, k, v) = r[2:]
                    return cls(CharacterProxy(self, charn), k, v)
                elif r[1] == 'node':
                    (char, noden, k, v) = r[2:]
                    try:
                        node = self.character[char].node[noden]
                    except KeyError:
                        (loc, nxtloc, arrt, nxtarrt) = self.handle(
                            'get_thing_special_stats',
                            char=char,
                            thing=noden
                        )
                        if loc is not None:
                            node = self._things_cache[char][noden] \
                                   = ThingProxy(
                                       self,
                                       char,
                                       noden,
                                       loc,
                                       nxtloc,
                                       arrt,
                                       nxtarrt
                                   )
                        else:
                            node = self._character_places_cache[
                                char][noden] = PlaceProxy(
                                    self, char, noden
                                )
                    return cls(node, k, v)
                else:
                    assert (r[1] == 'portal')
                    (char, nodeA, nodeB, k, v) = r[2:]
                    return cls(PortalProxy(self, char, nodeA, nodeB), k, v)
            else:
                return tuple(self.json_rewrap(v) for v in r)
        elif isinstance(r, dict):
            # These can't have been stored in a stat
            return {k: self.json_rewrap(v) for (k, v) in r.items()}
        elif isinstance(r, list):
            return [self.json_rewrap(v) for v in r]
        return r

    def json_load(self, s):
        return self.json_rewrap(super().json_load(s))

    def _call_with_recv(self, *cbs, **kwargs):
        received = self.json_load(self.recv()[1])
        for cb in cbs:
            cb(received, **kwargs)
        return received

    def _upd_chars_caches(self, chardiffs, **kwargs):
        deleted = set(self.character.keys())
        for (char, chardiff) in chardiffs.items():
            if char not in self._char_cache:
                self._char_cache[char] = CharacterProxy(self, char)
            self.character[char]._apply_diff(chardiff)
            deleted.discard(char)
        for char in deleted:
            del self._char_cache[char]

    def _inc_tick(self, *args):
        self._tick += 1

    def _set_time(self, *args, **kwargs):
        self._branch = kwargs['branch']
        self._tick = kwargs['tick']

    def _pull_async(self, chars, cb):
        if not callable(cb):
            raise TypeError("Uncallable callback")
        self.send(self.json_dump({
            'silent': False,
            'command': 'get_chardiffs',
            'chars': chars
        }))
        cbs = [self._upd_chars_caches]
        if cb:
            cbs.append(cb)
        self._call_with_recv(cbs)

    def pull(self, chars='all', cb=None, sync=True):
        """Update the state of all my proxy objects from the real objects."""
        if sync:
            diffs = self.handle('get_chardiffs', chars=chars)
            self._upd_chars_caches(diffs)
            if cb:
                cb(diffs)
        else:
            Thread(
                target=self._pull_async,
                args=(chars, cb)
            ).start()

    def next_tick(self, chars=[], cb=None, silent=False):
        if cb and not chars:
            raise TypeError("Callback requires chars")
        if not callable(cb):
            raise TypeError("Uncallable callback")
        if chars:
            self.send(self.json_dump({
                'silent': False,
                'command': 'next_tick',
                'chars': chars
            }))
            args = [self._inc_tick, self._upd_chars_caches]
            if cb:
                args.append(cb)
            if silent:
                Thread(
                    target=self._call_with_recv,
                    args=args
                ).start()
            else:
                return self._call_with_recv(*args)
        elif silent:
            self.handle(command='next_tick', chars=[], silent=True)
        else:
            return self.handle(command='next_tick', chars='all')

    def time_travel(self, branch, tick, char=None, cb=None):
        # TODO: multiple chars
        if cb and not char:
            raise TypeError("Callbacks require char name")
        if cb is not None and not callable(cb):
            raise TypeError("Uncallable callback")
        if char:
            self.send(self.json_dump({
                'command': 'time_travel',
                'silent': False,
                'branch': branch,
                'tick': tick,
                'chars': [char]
            }))
            Thread(
                target=self._call_with_recv,
                args=(self._set_time, self._upd_chars_caches, cb)
                if cb else (self._set_time, self._upd_chars_caches),
                kwargs={'branch': branch, 'tick': tick}
            ).start()
        else:
            self.handle(
                command='time_travel',
                branch=branch,
                tick=tick,
                chars=[],
                silent=True
            )

    def add_character(self, char, data={}, **attr):
        if char in self._char_cache:
            raise KeyError("Character already exists")
        assert char not in self._char_stat_cache
        self._char_cache[char] = CharacterProxy(self, char)
        self._char_stat_cache[char] = attr
        placedata = data.get('place', data.get('node', {}))
        for place, stats in placedata.items():
            assert place not in self._character_places_cache[char]
            assert place not in self._node_stat_cache[char]
            self._character_places_cache[char][place] \
                = PlaceProxy(self.engine,  char,  place)
            self._node_stat_cache[char][place] = stats
        thingdata = data.get('thing',  {})
        for thing, stats in thingdata.items():
            assert thing not in self._things_cache[char]
            assert thing not in self._node_stat_cache[char]
            if 'location' not in stats:
                raise ValueError('Things must always have locations')
            if 'arrival_time' in stats or 'next_arrival_time' in stats:
                raise ValueError('The arrival_time stats are read-only')
            loc = stats.pop('location')
            nxtloc = stats.pop('next_location') \
                     if 'next_location' in stats else None
            self._things_cache[char][thing] \
                = ThingProxy(loc, nxtloc, self.engine.rev, None)
            self._node_stat_cache[char][thing] = stats
        portdata = data.get('edge', data.get('portal', data.get('adj',  {})))
        for orig, dests in portdata.items():
            assert orig not in self._character_portals_cache[char]
            assert orig not in self._portal_stat_cache[char]
            for dest, stats in dests.items():
                assert dest not in self._character_portals_cache[char][orig]
                assert dest not in self._portal_stat_cache[char][orig]
                self._character_portals_cache[char][orig][dest] \
                    = PortalProxy(self.engine, char, orig, dest)
                self._portal_stat_cache[char][orig][dest] = stats
        self.handle(
            command='add_character', char=char, data=data, attr=attr,
            silent=True, branching=True
        )

    def new_character(self, char, **attr):
        self.add_character(char, **attr)
        return self._chars_cache[char]

    def del_character(self, char):
        if char not in self._chars_cache:
            raise KeyError("No such character")
        del self._chars_cache[char]
        del self._char_stat_cache[char]
        del self._character_places_cache[char]
        del self._things_cache[char]
        del self._character_portals_cache[char]
        self.handle(command='del_character', char=char, silent=True, branching=True)

    def del_node(self, char, node):
        if char not in self._chars_cache:
            raise KeyError("No such character")
        if node not in self._character_places_cache[char] and \
           node not in self._things_cache[char]:
            raise KeyError("No such node")
        if node in self._things_cache[char]:
            del self._things_cache[char][node]
        if node in self._character_places_cache[char]:  # just to be safe
            del self._character_places_cache[char][node]
        self.handle(
            command='del_node',
            char=char,
            node=node,
            silent=True,
            branching=True
        )

    def del_portal(self, char, orig, dest):
        if char not in self._chars_cache:
            raise KeyError("No such character")
        self._character_portals_cache.delete(char, orig, dest)
        self.handle(
            command='del_portal',
            char=char,
            orig=orig,
            dest=dest,
            silent=True,
            branching=True
        )

    def commit(self):
        self.handle('commit', silent=True)

    def close(self):
        self.handle(command='close', silent=True)
        self.send('shutdown')


def subprocess(
    args, kwargs, handle_out_pipe, handle_in_pipe, logq
):
    def log(typ, data):
        if typ == 'command':
            (cmd, kvs) = data
            logs = "LiSE proc {}: calling {}({})".format(
                getpid(),
                cmd,
                ",  ".join("{}={}".format(k,  v) for k,  v in kvs.items())
            )
        else:
            logs = "LiSE proc {}: returning {} (of type {})".format(
                getpid(),
                data,
                repr(type(data))
            )
        logq.put(('debug', logs))
        syslog.syslog(logs)
    engine_handle = EngineHandle(args, kwargs, logq)

    while True:
        inst = handle_out_pipe.recv()
        if inst == 'shutdown':
            handle_out_pipe.close()
            handle_in_pipe.close()
            logq.close()
            return 0
        instruction = engine_handle.json_load(inst)
        silent = instruction.pop('silent',  False)
        cmd = instruction.pop('command')
        log('command', (cmd, instruction))
        r = getattr(engine_handle, cmd)(**instruction)
        if silent:
            continue
        log('result', r)
        handle_in_pipe.send((cmd,  engine_handle.json_dump(r)))


class RedundantProcessError(ProcessError):
    """Raised when EngineProcessManager is asked to start a process that
    has already started.

    """


class EngineProcessManager(object):
    def start(self, *args, **kwargs):
        if hasattr(self, 'engine_proxy'):
            raise RedundantProcessError("Already started")
        (handle_out_pipe_recv, self._handle_out_pipe_send) = Pipe(duplex=False)
        (handle_in_pipe_recv, handle_in_pipe_send) = Pipe(duplex=False)
        self.logq = Queue()
        handlers = []
        logl = {
            'debug': logging.DEBUG,
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }
        loglevel = logging.INFO
        if 'loglevel' in kwargs:
            if kwargs['loglevel'] in logl:
                loglevel = logl[kwargs['loglevel']]
            else:
                loglevel = kwargs['loglevel']
            del kwargs['loglevel']
        if 'logger' in kwargs:
            self.logger = kwargs['logger']
            del kwargs['logger']
        else:
            self.logger = logging.getLogger(__name__)
            stdout = logging.StreamHandler(sys.stdout)
            stdout.set_name('stdout')
            handlers.append(stdout)
            handlers[0].setLevel(loglevel)
        if 'logfile' in kwargs:
            try:
                fh = logging.FileHandler(kwargs['logfile'])
                handlers.append(fh)
                handlers[-1].setLevel(loglevel)
            except OSError:
                pass
            del kwargs['logfile']
        do_game_start = kwargs.pop('do_game_start') \
                        if 'do_game_start' in kwargs else False
        install_modules = kwargs.pop('install_modules') \
                          if 'install_modules' in kwargs else []
        formatter = logging.Formatter(
            fmt='[{levelname}] LiSE.proxy({process})\t{message}',
            style='{'
        )
        for handler in handlers:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self._p = Process(
            name='LiSE Life Simulator Engine (core)',
            target=subprocess,
            args=(
                args,
                kwargs,
                handle_out_pipe_recv,
                handle_in_pipe_send,
                self.logq
            )
        )
        self._p.daemon = True
        self._p.start()
        self._logthread = Thread(
            target=self.sync_log_forever,
            name='log',
            daemon=True
        )
        self._logthread.start()
        self.engine_proxy = EngineProxy(
            self._handle_out_pipe_send,
            handle_in_pipe_recv,
            self.logger,
            do_game_start,
            install_modules
        )
        return self.engine_proxy

    def sync_log(self, limit=None, block=True):
        n = 0
        while limit is None or n < limit:
            try:
                (level, message) = self.logq.get(block=block)
                getattr(self.logger, level)(message)
                print(message)
                n += 1
            except Empty:
                return

    def sync_log_forever(self):
        while True:
            self.sync_log(1)

    def shutdown(self):
        self.engine_proxy.close()
        self._p.join()
        del self.engine_proxy
