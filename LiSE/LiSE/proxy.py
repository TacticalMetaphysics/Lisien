# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Proxy objects to access LiSE entities from another process.

Each proxy class is meant to emulate the equivalent LiSE class,
and any change you make to a proxy will be made in the corresponding
entity in the LiSE core.

"""
import sys
import logging
from abc import abstractmethod
from random import Random
from collections.abc import (
    Mapping,
    MutableMapping,
    MutableSequence
)
from functools import partial
from threading import Thread, Lock
from multiprocessing import Process, Pipe, Queue, ProcessError
from concurrent.futures import ThreadPoolExecutor
from queue import Empty

from blinker import Signal

from .allegedb import HistoryError
from .allegedb.cache import PickyDefaultDict, StructuredDefaultDict
from .allegedb.wrap import DictWrapper, ListWrapper, SetWrapper, UnwrappingDict
from .engine import AbstractEngine
from .character import Facade, AbstractCharacter
from .reify import reify
from .util import getatt
from .handle import EngineHandle
from .xcollections import AbstractLanguageDescriptor
from .node import NodeContent, UserMapping, UserDescriptor
from .place import Place
from .thing import Thing
from .portal import Portal


class CachingProxy(MutableMapping, Signal):
    """Abstract class for proxies to LiSE entities or mappings thereof"""
    def __init__(self):
        super().__init__()
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
        return self._cache_get_munge(k, self._cache[k])

    def __setitem__(self, k, v):
        self._set_item(k, v)
        self._cache[k] = self._cache_set_munge(k, v)
        self.send(self, key=k, value=v)

    def __delitem__(self, k):
        if k not in self:
            raise KeyError("No such key: {}".format(k))
        self._del_item(k)
        del self._cache[k]
        self.send(self, key=k, value=None)

    def _apply_delta(self, delta):
        for (k, v) in delta.items():
            if v is None:
                if k in self._cache:
                    del self._cache[k]
                    self.send(self, key=k, value=None)
            elif k not in self._cache or self._cache[k] != v:
                self._cache[k] = v
                self.send(self, key=k, value=v)

    def _cache_get_munge(self, k, v):
        return v

    def _cache_set_munge(self, k, v):
        return v

    @abstractmethod
    def _set_item(self, k, v):
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def _del_item(self, k):
        raise NotImplementedError("Abstract method")


class CachingEntityProxy(CachingProxy):
    """Abstract class for proxy objects representing LiSE entities"""
    def _cache_get_munge(self, k, v):
        if isinstance(v, dict):
            return DictWrapper(
                lambda: self._cache[k], partial(self._set_item, k), self, k)
        elif isinstance(v, list):
            return ListWrapper(
                lambda: self._cache[k], partial(self._set_item, k), self, k)
        elif isinstance(v, set):
            return SetWrapper(
                lambda: self._cache[k], partial(self._set_item, k), self, k)
        return v

    def __repr__(self):
        return "<{}({}) {} at {}>".format(
            self.__class__.__name__, self._cache, self.name, id(self)
        )


class RulebookProxyDescriptor(object):
    """Descriptor that makes the corresponding RuleBookProxy if needed"""
    def __get__(self, inst, cls):
        if inst is None:
            return self
        try:
            proxy = inst._get_rulebook_proxy()
        except KeyError:
            proxy = RuleBookProxy(
                inst.engine, inst._get_default_rulebook_name())
            inst._set_rulebook_proxy(proxy)
        return proxy

    def __set__(self, inst, val):
        if hasattr(val, 'name'):
            if not isinstance(val, RuleBookProxy):
                raise TypeError
            rb = val
            val = val.name
        elif val in inst.engine._rulebooks_cache:
            rb = inst.engine._rulebooks_cache[val]
        else:
            rb = RuleBookProxy(inst.engine, val)
        inst._set_rulebook(val)
        inst._set_rulebook_proxy(rb)
        inst.send(inst, rulebook=rb)


class ProxyUserMapping(UserMapping):
    """A mapping to the ``CharacterProxy``s that have this node as an avatar"""
    def _user_names(self):
        for user, avatars in self.node.engine._avatar_characters_cache[
                self.node._charname].items():
            if self.node.name in avatars:
                yield user


class ProxyUserDescriptor(UserDescriptor):
    usermapping = ProxyUserMapping


class NodeProxy(CachingEntityProxy):
    rulebook = RulebookProxyDescriptor()

    @property
    def users(self):
        return ProxyUserMapping(self)
    @property
    def character(self):
        return self.engine.character[self._charname]

    @property
    def _cache(self):
        return self.engine._node_stat_cache[self._charname][self.name]

    def _get_default_rulebook_name(self):
        return self._charname, self.name

    def _get_rulebook_proxy(self):
        return self.engine._char_node_rulebooks_cache[
            self._charname][self.name]

    def _set_rulebook_proxy(self, rb):
        self.engine._char_node_rulebooks_cache[self._charname][self.name] = rb

    def _set_rulebook(self, rb):
        self.engine.handle(
            'set_node_rulebook',
            char=self._charname, node=self.name, rulebook=rb, block=False,
            branching=True
        )

    user = ProxyUserDescriptor()

    def __init__(self, character, nodename):
        self.engine = character.engine
        self._charname = character.name
        self.name = nodename
        super().__init__()

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

    def _set_item(self, k, v):
        if k == 'name':
            raise KeyError("Nodes can't be renamed")
        self.engine.handle(
            command='set_node_stat',
            char=self._charname,
            node=self.name,
            k=k, v=v,
            block=False,
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
            block=False,
            branching=True
        )

    def delete(self):
        self.engine.del_node(self._charname, self.name)

    @property
    def content(self):
        return NodeContent(self)

    def contents(self):
        return self.content.values()


class PlaceProxy(NodeProxy):
    def __repr__(self):
        return "<proxy to {}.place[{}] at {}>".format(
            self._charname,
            repr(self.name),
            id(self)
        )


Place.register(PlaceProxy)


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

    def __init__(
            self, character, name, location=None, **kwargs
    ):
        if location is None and getattr(
                character.engine, '_initialized', True):
            raise ValueError("Thing must have location")
        super().__init__(character, name)
        self._location = location
        self._cache.update(kwargs)

    def __iter__(self):
        yield from super().__iter__()
        yield 'location'

    def __getitem__(self, k):
        if k == 'location':
            return self._location
        return super().__getitem__(k)

    def _apply_delta(self, delta):
        for (k, v) in delta.items():
            if v is None:
                if k in self._cache:
                    del self._cache[k]
                    self.send(self, key=k, value=None)
            elif k == 'location':
                self._location = v
                self.send(self, key=k, value=v)
            elif k not in self._cache or self._cache[k] != v:
                self._cache[k] = v
                self.send(self, key=k, value=v)

    def _set_location(self, v):
        self._location = v
        self.engine.handle(
            command='set_thing_location',
            char=self.character.name,
            thing=self.name,
            loc=v,
            block=False,
            branching=True
        )
        self.send(self, key='location', value=v)

    def __setitem__(self, k, v):
        if k == 'location':
            self._set_location(v)
        else:
            super().__setitem__(k, v)

    def __repr__(self):
        return "<proxy to {}.thing[{}]@{} at {}>".format(
            self._charname,
            self.name,
            self._location,
            id(self)
        )

    def follow_path(self, path, weight=None):
        self.engine.handle(
            command='thing_follow_path',
            char=self._charname,
            thing=self.name,
            path=path,
            weight=weight,
            block=False
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
            block=False
        )

    def travel_to(self, dest, weight=None, graph=None):
        if hasattr(dest, 'name'):
            dest = dest.name
        if hasattr(graph, 'name'):
            graph = graph.name
        return self.engine.handle(
            command='thing_travel_to',
            char=self._charname,
            thing=self.name,
            dest=dest,
            weight=weight,
            graph=graph
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
            block=False
        )


Thing.register(ThingProxy)


class PortalProxy(CachingEntityProxy):
    rulebook = RulebookProxyDescriptor()

    def _get_default_rulebook_name(self):
        return self._charname, self._origin, self._destination

    def _get_rulebook_proxy(self):
        return self.engine._char_port_rulebooks_cache[
            self._charname][self._origin][self._destination]

    def _set_rulebook_proxy(self, rb):
        self.engine._char_port_rulebooks_cache[self._charname][
            self._origin][self._destination] = rb

    def _set_rulebook(self, rb):
        self.engine.handle(
            command='set_portal_rulebook',
            char=self._charname,
            orig=self._origin,
            dest=self._destination,
            rulebook=rb,
            block=False
        )

    def _get_rulebook_name(self):
        return self.engine.handle(
            command='get_portal_rulebook',
            char=self._charname,
            orig=self._origin,
            dest=self._destination
        )

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

    @property
    def reciprocal(self):
        if self._origin not in self.character.pred or self._destination not in self.character.pred[self._origin]:
            return None
        return self.character.pred[self._origin][self._destination]

    def _set_item(self, k, v):
        self.engine.handle(
            command='set_portal_stat',
            char=self._charname,
            orig=self._origin,
            dest=self._destination,
            k=k, v=v,
            block=False,
            branching=True
        )

    def _del_item(self, k):
        self.engine_handle(
            command='del_portal_stat',
            char=self._charname,
            orig=self._origin,
            dest=self._destination,
            k=k,
            block=False,
            branching=True
        )

    def __init__(self, character, origname, destname):
        self.engine = character.engine
        self._charname = character.name
        self._origin = origname
        self._destination = destname
        super().__init__()

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
        return "<proxy to {}.portal[{}][{}] at {}>".format(
            self._charname,
            repr(self._origin),
            repr(self._destination),
            id(self)
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


Portal.register(PortalProxy)


class NodeMapProxy(MutableMapping, Signal):
    rulebook = RulebookProxyDescriptor()

    def _get_default_rulebook_name(self):
        return self._charname, 'character_node'

    def _get_rulebook_proxy(self):
        return self.engine._character_rulebooks_cache[self._charname]['node']

    def _set_rulebook_proxy(self, rb):
        self.engine._character_rulebooks_cache[self._charname]['node'] = rb

    def _set_rulebook(self, rb):
        self.engine.handle(
            'set_character_node_rulebook',
            char=self._charname,
            rulebook=rb,
            block=False,
            branching=True
        )

    @property
    def character(self):
        return self.engine.character[self._charname]

    def __init__(self, engine_proxy, charname):
        super().__init__()
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

    def patch(self, patch):
        """Change a bunch of node stats at once.

        This works similarly to ``update``, but only accepts a dict-like
        argument, and it recurses one level.

        The patch is sent to the LiSE core all at once, so this is faster than
        using ``update``, too.

        :param patch: a dictionary. Keys are node names, values are other dicts
        describing updates to the nodes, where a value of None means delete the
        stat. Other values overwrite.

        """
        self.engine.handle(
            'update_nodes',
            char=self.character.name,
            patch=patch,
            block=False
        )
        for node, stats in patch.items():
            nodeproxycache = self[node]._cache
            for k, v in stats.items():
                if v is None:
                    del nodeproxycache[k]
                else:
                    nodeproxycache[k] = v


class ThingMapProxy(CachingProxy):
    rulebook = RulebookProxyDescriptor()

    def _get_default_rulebook_name(self):
        return self.name, 'character_thing'

    def _get_rulebook_proxy(self):
        return self.engine._character_rulebooks_cache[self.name]['thing']

    def _set_rulebook_proxy(self, rb):
        self.engine._character_rulebooks_cache[self.name]['thing'] = rb

    def _set_rulebook(self, rb):
        self.engine.handle(
            'set_character_thing_rulebook',
            char=self.name,
            rulebook=rb,
            block=False,
            branching=True
        )

    @property
    def character(self):
        return self.engine.character[self.name]

    @property
    def _cache(self):
        return self.engine._things_cache[self.name]

    def __init__(self, engine_proxy, charname):
        self.engine = engine_proxy
        self.name = charname
        super().__init__()

    def __eq__(self, other):
        return self is other

    def _cache_set_munge(self, k, v):
        return ThingProxy(
            self, *self.engine.handle(
                'get_thing_special_stats', char=self.name, thing=k
            )
        )

    def _set_item(self, k, v):
        self.engine.handle(
            command='set_thing',
            char=self.name,
            thing=k,
            statdict=v,
            block=False,
            branching=True
        )
        self._cache[k] = ThingProxy(
            self.engine, self.name, v.pop('location')
        )
        self.engine._node_stat_cache[self.name][k] = v

    def _del_item(self, k):
        self.engine.handle(
            command='del_node',
            char=self.name,
            node=k,
            block=False,
            branching=True
        )
        del self._cache[k]
        del self.engine._node_stat_cache[self.name][k]


class PlaceMapProxy(CachingProxy):
    rulebook = RulebookProxyDescriptor()

    def _get_default_rulebook_name(self):
        return self.name, 'character_place'

    def _get_rulebook_proxy(self):
        return self.engine._character_rulebooks_cache[self.name]['place']

    def _set_rulebook_proxy(self, rb):
        self.engine._character_rulebooks_cache[self.name]['place'] = rb

    def _set_rulebook(self, rb):
        self.engine.handle(
            'set_character_place_rulebook',
            char=self.name, rulebook=rb,
            block=False, branching=True
        )

    @property
    def character(self):
        return self.engine.character[self.name]

    @property
    def _cache(self):
        return self.engine._character_places_cache[self.name]

    def __init__(self, engine_proxy, character):
        self.engine = engine_proxy
        self.name = character
        super().__init__()

    def __eq__(self, other):
        return self is other

    def _cache_set_munge(self, k, v):
        return PlaceProxy(self, k)

    def _set_item(self, k, v):
        self.engine.handle(
            command='set_place',
            char=self.name,
            place=k, statdict=v,
            block=False,
            branching=True
        )
        self.engine._node_stat_cache[self.name][k] = v

    def _del_item(self, k):
        self.engine.handle(
            command='del_node',
            char=self.name,
            node=k,
            block=False,
            branching=True
        )
        del self.engine._node_stat_cache[self.name][k]


class SuccessorsProxy(CachingProxy):
    @property
    def _cache(self):
        return self.engine._character_portals_cache.successors[
            self._charname][self._orig]

    def __init__(self, engine_proxy, charname, origname):
        self.engine = engine_proxy
        self._charname = charname
        self._orig = origname
        super().__init__()

    def __eq__(self, other):
        return (
            isinstance(other, SuccessorsProxy) and
            self.engine is other.engine and
            self._charname == other._charname and
            self._orig == other._orig
        )

    def _get_state(self):
        return {
            node: self._cache[node] if node in self._cache else
            PortalProxy(self.engine, self._charname, self._orig, node)
            for node in self.engine.handle(
                command='node_successors',
                char=self._charname,
                node=self._orig
            )
        }

    def _apply_delta(self, delta):
        raise NotImplementedError(
            "Apply the delta on CharSuccessorsMappingProxy"
        )

    def _cache_set_munge(self, k, v):
        if isinstance(v, PortalProxy):
            assert v._origin == self._orig
            assert v._destination == k
            return v
        return PortalProxy(self, self._orig, k)

    def _set_item(self, dest, value):
        self.engine.handle(
            command='set_portal',
            char=self._charname,
            orig=self._orig,
            dest=dest,
            statdict=value,
            block=False,
            branching=True
        )

    def _del_item(self, dest):
        self.engine.del_portal(self._charname, self._orig, dest)


class CharSuccessorsMappingProxy(CachingProxy):
    rulebook = RulebookProxyDescriptor()

    def _get_default_rulebook_anme(self):
        return self.name, 'character_portal'

    def _get_rulebook_proxy(self):
        return self.engine._character_rulebooks_cache[self.name]['portal']

    def _set_rulebook_proxy(self, rb):
        self.engine._character_rulebooks_cache[self.name]['portal'] = rb

    def _set_rulebook(self, rb):
        self.engine.handle(
            'set_character_portal_rulebook',
            char=self.character.name, rulebook=rb, block=False, branching=True
        )

    @property
    def character(self):
        return self.engine.character[self.name]

    @property
    def _cache(self):
        return self.engine._character_portals_cache.successors[self.name]

    def __init__(self, engine_proxy, charname):
        self.engine = engine_proxy
        self.name = charname
        super().__init__()

    def __eq__(self, other):
        return (
            isinstance(other, CharSuccessorsMappingProxy) and
            other.engine is self.engine and
            other.name == self.name
        )

    def _cache_set_munge(self, k, v):
        return {
            vk: PortalProxy(self, vk, vv)
            for (vk, vv) in v.items()
        }

    def __getitem__(self, k):
        return SuccessorsProxy(
            self.engine,
            self.name,
            k
        )

    def _apply_delta(self, delta):
        for o, ds in delta.items():
            for d, ex in ds.items():
                if ex:
                    if d not in self._cache[o]:
                        self._cache[o][d] = PortalProxy(
                            self.character,
                            o, d
                        )
                else:
                    if o in self._cache and d in self._cache[o]:
                        del self._cache[o][d]
                        if len(self._cache[o]) == 0:
                            del self._cache[o]

    def _set_item(self, orig, val):
        self.engine.handle(
            command='character_set_node_successors',
            character=self.name,
            node=orig,
            val=val,
            block=False,
            branching=True
        )

    def _del_item(self, orig):
        for dest in self[orig]:
            self.engine.del_portal(self.name, orig, dest)


class PredecessorsProxy(MutableMapping):
    @property
    def character(self):
        return self.engine.character[self._charname]

    def __init__(self, engine_proxy, charname, destname):
        self.engine = engine_proxy
        self._charname = charname
        self.name = destname

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
            block=False,
            branching=True
        )
        self.engine.handle(
            'set_portal',
            (self._charname, k, self.name),
            block=False, branching=True
        )

    def __delitem__(self, k):
        self.engine.del_portal(self._charname, k, self.name)


class CharPredecessorsMappingProxy(MutableMapping):
    def __init__(self, engine_proxy, charname):
        self.engine = engine_proxy
        self.name = charname
        self._cache = {}

    def __contains__(self, k):
        return k in self.engine._character_portals_cache.predecessors[
            self.name]

    def __iter__(self):
        return iter(
            self.engine._character_portals_cache.predecessors[self.name])

    def __len__(self):
        return len(
            self.engine._character_portals_cache.predecessors[self.name])

    def __getitem__(self, k):
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
            block=False,
            branching=True
        )

    def __delitem__(self, k):
        for v in list(self[k]):
            self.engine.del_portal(self.name, v, k)
        if k in self._cache:
            del self._cache[k]


class CharStatProxy(CachingEntityProxy):
    @property
    def _cache(self):
        return self.engine._char_stat_cache[self.name]

    def __init__(self, engine_proxy, character):
        self.engine = engine_proxy
        self.name = character
        super().__init__()

    def __eq__(self, other):
        return (
            isinstance(other, CharStatProxy) and
            self.engine is other.engine and
            self.name == other.name
        )

    def _get(self, k=None):
        if k is None:
            return self
        return self._cache[k]

    def _get_state(self):
        return self.engine.handle(
            command='character_stat_copy',
            char=self.name
        )

    def _set_item(self, k, v):
        self.engine.handle(
            command='set_character_stat',
            char=self.name,
            k=k, v=v,
            block=False,
            branching=True
        )

    def _del_item(self, k):
        self.engine.handle(
            command='del_character_stat',
            char=self.name,
            k=k,
            block=False,
            branching=True
        )


class RuleProxy(Signal):
    @staticmethod
    def _nominate(v):
        ret = []
        for whatever in v:
            if hasattr(whatever, 'name'):
                ret.append(whatever.name)
            else:
                assert isinstance(whatever, str)
                ret.append(whatever)
        return ret

    @property
    def _cache(self):
        return self.engine._rules_cache.setdefault(self.name, {})

    @property
    def triggers(self):
        return self._cache.setdefault('triggers', [])

    @triggers.setter
    def triggers(self, v):
        self._cache['triggers'] = v
        self.engine.handle(
            'set_rule_triggers', rule=self.name,
            triggers=self._nominate(v), block=False)
        self.send(self, triggers=v)

    @property
    def prereqs(self):
        return self._cache.setdefault('prereqs', [])

    @prereqs.setter
    def prereqs(self, v):
        self._cache['prereqs'] = v
        self.engine.handle(
            'set_rule_prereqs', rule=self.name,
            prereqs=self._nominate(v), block=False)
        self.send(self, prereqs=v)

    @property
    def actions(self):
        return self._cache.setdefault('actions', [])

    @actions.setter
    def actions(self, v):
        self._cache['actions'] = v
        self.engine.handle(
            'set_rule_actions', rule=self.name,
            actions=self._nominate(v), block=False)
        self.send(self, actions=v)

    def __init__(self, engine, rulename):
        super().__init__()
        self.engine = engine
        self.name = self._name = rulename

    def __eq__(self, other):
        return (
            hasattr(other, 'name') and
            self.name == other.name
        )


class RuleBookProxy(MutableSequence, Signal):
    @property
    def _cache(self):
        return self.engine._rulebooks_cache.setdefault(self.name, [])

    def __init__(self, engine, bookname):
        super().__init__()
        self.engine = engine
        self.name = bookname
        self._proxy_cache = engine._rule_obj_cache

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
            block=False,
            branching=True
        )
        self.send(self, i=i, val=v)

    def __delitem__(self, i):
        del self._cache[i]
        self.engine.handle(
            command='del_rulebook_rule',
            rulebook=self.name,
            i=i,
            block=False,
            branching=True
        )
        self.send(self, i=i, val=None)

    def insert(self, i, v):
        if isinstance(v, RuleProxy):
            v = v._name
        self._cache.insert(i, v)
        self.engine.handle(
            command='ins_rulebook_rule',
            rulebook=self.name,
            i=i,
            rule=v,
            block=False,
            branching=True
        )
        for j in range(i, len(self)):
            self.send(self, i=j, val=self[j])


class AvatarMapProxy(Mapping):
    rulebook = RulebookProxyDescriptor()
    engine = getatt('character.engine')

    def _get_default_rulebook_name(self):
        return self.character.name, 'avatar'

    def _get_rulebook_proxy(self):
        return self.engine._character_rulebooks_cache[
            self.character.name]['avatar']

    def _set_rulebook_proxy(self, rb):
        self.engine._character_rulebooks_cache[
            self.character.name]['avatar'] = rb

    def _set_rulebook(self, rb):
        self.engine.handle(
            'set_avatar_rulebook',
            char=self.character.name, rulebook=rb, block=False, branching=True
        )

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


class CharacterProxy(AbstractCharacter):
    rulebook = RulebookProxyDescriptor()
    adj_cls = CharSuccessorsMappingProxy
    pred_cls = CharPredecessorsMappingProxy

    def copy_from(self, g):
        # can't handle multigraphs
        self.engine.handle('character_copy_from', char=self.name, nodes=g._node, adj=g._adj, block=False, branching=True)
        for node, nodeval in g.nodes.items():
            if node not in self.node:
                    if nodeval and 'location' in nodeval:
                        self.thing._cache[node] = prox = ThingProxy(
                            self, node, nodeval['location']
                        )
                        self.thing.send(self.thing, key=node, value=prox)
                    else:
                        self.place._cache[node] = prox = PlaceProxy(
                            self, node
                        )
                        self.place.send(self.place, key=node, value=prox)
                    self.node.send(self.node, key=node, value=prox)
        for orig in g.adj:
            for dest, edge in g.adj[orig].items():
                if orig in self.portal and dest in self.portal[orig]:
                    self.portal[orig][dest]._apply_delta(edge)
                else:
                    self.portal._cache[orig][dest] = PortalProxy(
                        self, orig, dest
                    )
                    self.engine._portal_stat_cache[
                        self.name][orig][dest] = edge

    def thing2place(self, name):
        # TODO
        raise NotImplementedError("TODO")

    def _get_default_rulebook_name(self):
        return self.name, 'character'

    def _get_rulebook_proxy(self):
        return self.engine._character_rulebooks_cache[self.name]['character']

    def _set_rulebook_proxy(self, rb):
        self.engine._character_rulebooks_cache[self.name]['character'] = rb

    def _set_rulebook(self, rb):
        self.engine.handle(
            'set_character_rulebook',
            char=self.name, rulebook=rb, block=False, branching=True
        )

    @reify
    def avatar(self):
        return AvatarMapProxy(self)

    @staticmethod
    def PortalSuccessorsMapping(self):
        return CharSuccessorsMappingProxy(self.engine, self.name)

    @staticmethod
    def PortalPredecessorsMapping(self):
        return CharPredecessorsMappingProxy(self.engine, self.name)

    @staticmethod
    def ThingMapping(self):
        return ThingMapProxy(self.engine, self.name)

    @staticmethod
    def PlaceMapping(self):
        return PlaceMapProxy(self.engine, self.name)

    @staticmethod
    def ThingPlaceMapping(self):
        return NodeMapProxy(self.engine, self.name)

    def __init__(self, engine_proxy, charname):
        self.db = engine_proxy
        self.name = charname
        self.graph = CharStatProxy(self.engine, self.name)

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

    def _apply_delta(self, delta):
        delta = delta.copy()
        for node, ex in delta.pop('nodes', {}).items():
            if ex:
                if node not in self.node:
                    nodeval = delta.get('node_val', {}).get(node, None)
                    if nodeval and 'location' in nodeval:
                        self.thing._cache[node] = prox = ThingProxy(
                            self, node, nodeval['location']
                        )
                        self.thing.send(self.thing, key=node, value=prox)
                    else:
                        self.place._cache[node] = prox = PlaceProxy(
                            self, node
                        )
                        self.place.send(self.place, key=node, value=prox)
                    self.node.send(self.node, key=node, value=prox)
            else:
                if node in self.place._cache:
                    del self.place._cache[node]
                    self.place.send(self.place, key=node, value=None)
                elif node in self.thing._cache:
                    del self.thing._cache[node]
                    self.thing.send(self.thing, key=node, value=None)
                else:
                    self.engine.warning(
                        "Diff deleted {} but it was never created here"
                            .format(node))
                self.node.send(self.node, key=node, value=None)
        self.portal._apply_delta(delta.pop('edges', {}))
        for (node, nodedelta) in delta.pop('node_val', {}).items():
            if node not in self.node or node not in \
                    self.engine._node_stat_cache[self.name]:
                self.engine._node_stat_cache[self.name][node] = nodedelta
            else:
                self.node[node]._apply_delta(nodedelta)
        for (orig, destdelta) in delta.pop('edge_val', {}).items():
            for (dest, portdelta) in destdelta.items():
                if orig in self.portal and dest in self.portal[orig]:
                    self.portal[orig][dest]._apply_delta(portdelta)
                else:
                    self.engine._portal_stat_cache[
                        self.name][orig][dest] = portdelta
        if delta.pop('character_rulebook',
                     self.rulebook.name) != self.rulebook.name:
            self._set_rulebook_proxy(
                self.engine._rulebooks_cache[delta.pop('character_rulebook')])
        if delta.pop('avatar_rulebook',
                     self.avatar.rulebook.name) != self.avatar.rulebook.name:
            self.avatar._set_rulebook_proxy(
                self.engine._rulebooks_cache[delta.pop('avatar_rulebook')])
        if delta.pop('character_thing_rulebook',
                     self.thing.rulebook.name) != self.thing.rulebook.name:
            self.thing._set_rulebook_proxy(
                self.engine._rulebooks_cache[
                    delta.pop('character_thing_rulebook')])
        if delta.pop('character_place_rulebook',
                     self.place.rulebook.name) != self.place.rulebook.name:
            self.place._set_rulebook_proxy(
                self.engine._rulebooks_cache[
                    delta.pop('character_place_rulebook')])
        if delta.pop('character_portal_rulebook',
                     self.portal.rulebook.name) != self.portal.rulebook.name:
            self.portal._set_rulebook_proxy(
                self.engine._rulebooks_cache[
                    delta.pop('character_portal_rulebook')])
        for noden, rb in delta.pop('node_rulebooks', {}).items():
            node = self.node[noden]
            if node.rulebook.name != rb:
                node._set_rulebook_proxy(self.engine._rulebooks_cache[rb])
        portrb = delta.pop('portal_rulebooks', {})
        for orign in portrb:
            for destn, rb in portrb[orign].items():
                port = self.portal[orign][destn]
                if port.rulebook.name != rb:
                    port._set_rulebook_proxy(self.engine._rulebooks_cache[rb])
        self.stat._apply_delta(delta)

    def add_place(self, name, **kwargs):
        self.engine.handle(
            command='set_place',
            char=self.name,
            place=name,
            statdict=kwargs,
            block=False,
            branching=True
        )
        self.place._cache[name] = PlaceProxy(
            self, name
        )
        self.engine._node_stat_cache[self.name][name] = kwargs

    def add_places_from(self, seq):
        self.engine.handle(
            command='add_places_from',
            char=self.name,
            seq=list(seq),
            block=False,
            branching=True
        )
        placecache = self.place._cache
        nodestatcache = self.engine._node_stat_cache[self.name]
        for pln in seq:
            if isinstance(pln, tuple):
                placecache[pln[0]] = PlaceProxy(
                    self, *pln
                )
                if len(pln) > 1:
                    nodestatcache[pln[0]] = pln[1]
            else:
                placecache[pln] = PlaceProxy(
                    self, pln
                )

    def add_nodes_from(self, seq):
        self.add_places_from(seq)

    def add_thing(self, name, location, **kwargs):
        self.engine.handle(
            command='add_thing',
            char=self.name,
            thing=name,
            loc=location,
            statdict=kwargs,
            block=False,
            branching=True
        )
        self.thing._cache[name] = ThingProxy(
            self, name, location, **kwargs
        )

    def add_things_from(self, seq):
        self.engine.handle(
            command='add_things_from',
            char=self.name,
            seq=list(seq),
            block=False,
            branching=True
        )
        for name, location in seq:
            self.thing._cache[name] = ThingProxy(
                self, name, location
            )

    def new_place(self, name, **kwargs):
        self.add_place(name, **kwargs)
        return self.place[name]

    def new_thing(self, name, location, **kwargs):
        self.add_thing(name, location, **kwargs)
        return self.thing[name]

    def remove_node(self, node):
        if node not in self.node:
            raise KeyError("No such node: {}".format(node))
        name = self.name
        self.engine.handle(
            'del_node', char=name, node=node, block=False, branching=True)
        placecache = self.place._cache
        thingcache = self.thing._cache
        if node in placecache:
            del placecache[node]
        else:
            del thingcache[node]
        portscache = self.engine._character_portals_cache
        to_del = {(node, dest) for dest in portscache.successors[name][node]}
        to_del.update((orig, node) for orig in portscache.predecessors[name][node])
        for u, v in to_del:
            portscache.delete(name, u, v)
        if node in portscache.successors[name]:
            del portscache.successors[name][node]
        if node in portscache.predecessors[name]:
            del portscache.predecessors[name][node]

    def remove_place(self, place):
        placemap = self.place
        if place not in placemap:
            raise KeyError("No such place: {}".format(place))
        name = self.name
        self.engine.handle('del_node', char=name, node=place,
                           block=False, branching=True)
        del placemap._cache[place]
        portscache = self.engine._character_portals_cache
        del portscache.successors[name][place]
        del portscache.predecessors[name][place]

    def remove_thing(self, thing):
        thingmap = self.thing
        if thing not in thingmap:
            raise KeyError("No such thing: {}".format(thing))
        name = self.name
        self.engine.handle('del_node', char=name, node=thing,
                           block=False, branching=True)
        del thingmap._cache[thing]
        portscache = self.engine._character_portals_cache
        del portscache.successors[name][thing]
        del portscache.predecessors[name][thing]

    def place2thing(self, node, location):
        # TODO: cache
        self.engine.handle(
            command='place2thing',
            char=self.name,
            node=node,
            loc=location,
            block=False,
            branching=True
        )

    def add_portal(self, origin, destination, symmetrical=False, **kwargs):
        self.engine.handle(
            command='add_portal',
            char=self.name,
            orig=origin,
            dest=destination,
            symmetrical=symmetrical,
            statdict=kwargs,
            block=False,
            branching=True
        )
        self.engine._character_portals_cache.store(
            self.name,
            origin,
            destination,
            PortalProxy(self, origin, destination)
        )
        node = self._node
        placecache = self.place._cache

        if origin not in node:
            placecache[origin] = PlaceProxy(self, origin)
        if destination not in node:
            placecache[destination] = PlaceProxy(self, destination)
        if symmetrical:
            self.engine._character_portals_cache.store(
                self.name,
                destination,
                origin,
                PortalProxy(self, destination, origin)
            )
            self.engine._portal_stat_cache[self.name][
                destination][origin]['is_mirror'] = True

    def remove_portal(self, origin, destination):
        char_port_cache = self.engine._character_portals_cache
        cache = char_port_cache.successors[self.name]
        if origin not in cache or destination not in cache[origin]:
            raise KeyError("No portal from {} to {}".format(
                origin, destination))
        self.engine.handle(
            'del_portal', char=self.name, orig=origin, dest=destination,
            block=False, branching=True
        )
        char_port_cache.delete(self.name, origin, destination)

    remove_edge = remove_portal

    def add_portals_from(self, seq, symmetrical=False):
        l = list(seq)
        self.engine.handle(
            command='add_portals_from',
            char=self.name,
            seq=l,
            symmetrical=symmetrical,
            block=False,
            branching=True
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
        # TODO: cache
        self.engine.handle(
            command='add_avatar',
            char=self.name,
            graph=graph,
            node=node,
            block=False,
            branching=True
        )

    def remove_avatar(self, graph, node):
        # TODO: cache
        self.engine.handle(
            command='remove_avatar',
            char=self.name,
            graph=graph,
            node=node,
            block=False,
            branching=True
        )

    def avatars(self):
        yield from self.engine.handle(
            command='character_avatars',
            char=self.name
        )

    def facade(self):
        return Facade(self)

    def grid_2d_8graph(self, m, n):
        self.engine.handle('grid_2d_8graph', character=self.name, m=m, n=n, cb=self.engine._upd_caches)

    def grid_2d_graph(self, m, n, periodic=False):
        self.engine.handle('grid_2d_graph', character=self.name, m=m, n=n, periodic=periodic, cb=self.engine._upd_caches)


class CharacterMapProxy(MutableMapping, Signal):
    def __init__(self, engine_proxy):
        super().__init__()
        self.engine = engine_proxy

    def __iter__(self):
        return iter(self.engine._char_cache.keys())

    def __contains__(self, k):
        return k in self.engine._char_cache

    def __len__(self):
        return len(self.engine._char_cache)

    def __getitem__(self, k):
        return self.engine._char_cache[k]

    def __setitem__(self, k, v):
        self.engine.handle(
            command='set_character',
            char=k,
            data=v,
            block=False,
            branching=True
        )
        self.engine._char_cache[k] = CharacterProxy(self.engine, k)
        self.send(self, key=k, val=v)

    def __delitem__(self, k):
        self.engine.handle(
            command='del_character',
            char=k,
            block=False,
            branching=True
        )
        if k in self.engine._char_cache:
            del self.engine._char_cache[k]
        self.send(self, key=k, val=None)


class ProxyLanguageDescriptor(AbstractLanguageDescriptor):
    def _get_language(self, inst):
        if not hasattr(inst, '_language'):
            inst._language = inst.engine.handle(command='get_language')
        return inst._language

    def _set_language(self, inst, val):
        inst._language = val
        delta = inst.engine.handle(command='set_language', lang=val)
        cache = inst._cache
        for k, v in delta.items():
            if k in cache:
                if v is None:
                    del cache[k]
                elif cache[k] != v:
                    cache[k] = v
                    inst.send(inst, key=k, string=v)
            elif v is not None:
                cache[k] = v
                inst.send(inst, key=k, string=v)


class StringStoreProxy(Signal):
    language = ProxyLanguageDescriptor()

    def __init__(self, engine_proxy):
        super().__init__()
        self.engine = engine_proxy

    def load(self):
        self._cache = self.engine.handle('strings_delta')

    def __getattr__(self, k):
        try:
            return self._cache[k]
        except KeyError:
            raise AttributeError

    def __setattr__(self, k, v):
        if k in ('_cache', 'engine', 'language', '_language', 'receivers',
                 '_by_receiver', '_by_sender', '_weak_senders'):
            super().__setattr__(k, v)
            return
        self._cache[k] = v
        self.engine.handle(command='set_string', k=k, v=v, block=False)
        self.send(self, key=k, string=v)

    def __delattr__(self, k):
        del self._cache[k]
        self.engine.handle(command='del_string', k=k, block=False)
        self.send(self, key=k, string=None)

    def lang_items(self, lang=None):
        if lang is None or lang == self.language:
            yield from self._cache.items()
        else:
            yield from self.engine.handle(
                command='get_string_lang_items', lang=lang
            )


class EternalVarProxy(MutableMapping):
    @property
    def _cache(self):
        return self.engine._eternal_cache

    def __init__(self, engine_proxy):
        self.engine = engine_proxy

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
            block=False, silent=True
        )

    def __delitem__(self, k):
        del self._cache[k]
        self.engine.handle(
            command='del_eternal',
            k=k,
            block=False, silent=True
        )

    def _update_cache(self, data):
        for k, v in data.items():
            if v is None:
                del self._cache[k]
            else:
                self._cache[k] = v


class GlobalVarProxy(MutableMapping, Signal):
    @property
    def _cache(self):
        return self.engine._universal_cache

    def __init__(self, engine_proxy):
        super().__init__()
        self.engine = engine_proxy

    def __iter__(self):
        return iter(self._cache)

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, k):
        return self._cache[k]

    def __setitem__(self, k, v):
        self._cache[k] = v
        self.engine.handle('set_universal', k=k, v=v,
                           block=False, branching=True)
        self.send(self, key=k, value=v)

    def __delitem__(self, k):
        del self._cache[k]
        self.engine.handle('del_universal', k=k, block=False, branching=True)
        self.send(self, key=k, value=None)

    def _update_cache(self, data):
        for k, v in data.items():
            if v is None:
                if k not in self._cache:
                    continue
                del self._cache[k]
                self.send(self, key=k, value=None)
            else:
                self._cache[k] = v
                self.send(self, key=k, value=v)


class AllRuleBooksProxy(Mapping):
    @property
    def _cache(self):
        return self.engine._rulebooks_cache

    def __init__(self, engine_proxy):
        self.engine = engine_proxy

    def __iter__(self):
        yield from self._cache

    def __len__(self):
        return len(self._cache)

    def __contains__(self, k):
        return k in self._cache

    def __getitem__(self, k):
        if k not in self:
            self.engine.handle('new_empty_rulebook', rulebook=k, block=False)
            self._cache[k] = []
        return self._cache[k]


class AllRulesProxy(Mapping):
    @property
    def _cache(self):
        return self.engine._rules_cache

    def __init__(self, engine_proxy):
        self.engine = engine_proxy
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
            self._proxy_cache[k] = RuleProxy(self.engine, k)
        return self._proxy_cache[k]

    def new_empty(self, k):
        self.engine.handle(command='new_empty_rule', rule=k, block=False)
        self._cache[k] = {'triggers': [], 'prereqs': [], 'actions': []}
        self._proxy_cache[k] = RuleProxy(self.engine, k)
        return self._proxy_cache[k]


class FuncProxy(object):
    __slots__ = 'store', 'func'

    def __init__(self, store, func):
        self.store = store
        self.func = func

    def __call__(self, *args, block=True, cb=None, **kwargs):
        return self.store.engine.handle(
            'call_stored_function',
            store=self.store._store,
            func=self.func,
            args=args,
            kwargs=kwargs,
            block=block,
            cb=cb
        )

    def __str__(self):
        return self.store._cache[self.func]


class FuncStoreProxy(Signal):
    def __init__(self, engine_proxy, store):
        super().__init__()
        self.engine = engine_proxy
        self._store = store

    def load(self):
        self._cache = self.engine.handle('source_delta', store=self._store)

    def __getattr__(self, k):
        if k in self._cache:
            return FuncProxy(self, k)
        else:
            raise AttributeError

    def __setattr__(self, func_name, source):
        if func_name in ('engine', '_store', '_cache', 'receivers',
                         '_by_sender', '_by_receiver', '_weak_senders'):
            super().__setattr__(func_name, source)
            return
        self.engine.handle(
            command='store_source',
            store=self._store,
            v=source,
            name=func_name,
            block=False
        )
        self._cache[func_name] = source

    def __delattr__(self, func_name):
        self.engine.handle(
            command='del_source', store=self._store, k=func_name, block=False
        )
        del self._cache[func_name]

    def get_source(self, func_name):
        return self.engine.handle(
            command='get_source', store=self._store, name=func_name
        )


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
        succs = self.successors
        if char not in succs:
            raise KeyError(char)
        succmap = succs[char]
        if u not in succmap:
            raise KeyError((char, u))
        succu = succmap[u]
        if v not in succu:
            raise KeyError((char, u, v))
        del succu[v]
        if not succu:
            del succmap[u]
        preds = self.predecessors
        if char not in preds:
            raise KeyError(char)
        predmap = preds[char]
        if v not in predmap:
            raise KeyError((char, v))
        predv = predmap[v]
        if u not in predv:
            raise KeyError((char, v, u))
        del predv[u]
        if not predv:
            del predmap[v]

    def delete_char(self, char):
        del self.successors[char]
        del self.predecessors[char]


class TimeSignal(Signal):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def __iter__(self):
        yield self.engine.branch
        yield self.engine.tick

    def __len__(self):
        return 2

    def __getitem__(self, i):
        if i in ('branch', 0):
            return self.engine.branch
        if i in ('tick', 1):
            return self.engine.tick

    def __setitem__(self, i, v):
        if i in ('branch', 0):
            self.engine.time_travel(v, self.engine.tick)
        if i in ('tick', 1):
            self.engine.time_travel(self.engine.branch, v)


class TimeDescriptor(object):
    times = {}

    def __get__(self, inst, cls):
        if id(inst) not in self.times:
            self.times[id(inst)] = TimeSignal(inst)
        return self.times[id(inst)]

    def __set__(self, inst, val):
        inst.time_travel(*val)


class RandoProxy(Random):
    """Proxy to a randomizer"""
    def __init__(self, engine, seed=None):
        self.engine = engine
        self._handle = engine.handle
        self.gauss_next = None
        if seed:
            self.seed(seed)

    def seed(self, a=None, version=2):
        self._handle(
            cmd='call_randomizer',
            method='seed',
            a=a,
            version=version,
            block=False
        )

    def getstate(self):
        return self._handle(
            cmd='call_randomizer',
            method='getstate'
        )

    def setstate(self, state):
        return self._handle(
            cmd='call_randomizer',
            method='setstate',
            state=state
        )

    def _randbelow(self, n, int=int, maxsize=1, type=type,
                   Method=None, BuiltinMethod=None):
        return self._handle(
            cmd='call_randomizer',
            method='_randbelow',
            n=n,
            maxsize=maxsize
        )

    def random(self):
        return self._handle(
            cmd='call_randomizer',
            method='random'
        )


class EngineProxy(AbstractEngine):
    """An engine-like object for controlling a LiSE process

    Don't instantiate this directly. Use :class:`EngineProcessManager` instead.
    The ``start`` method will return an :class:`EngineProxy` instance.

    """
    char_cls = CharacterProxy
    thing_cls = ThingProxy
    place_cls = PlaceProxy
    portal_cls = PortalProxy
    time = TimeDescriptor()

    @property
    def branch(self):
        return self._branch

    @branch.setter
    def branch(self, v):
        self.time_travel(v, self.turn)

    @property
    def turn(self):
        return self._turn

    @turn.setter
    def turn(self, v):
        self.time_travel(self.branch, v)

    def __init__(
            self, handle_out, handle_in, logger,
            do_game_start=False,  install_modules=[],
            submit_func=None, threads=None
    ):
        if submit_func:
            self._submit = submit_func
        else:
            self._threadpool = ThreadPoolExecutor(threads)
            self._submit = self._threadpool.submit
        self._handle_out = handle_out
        self._handle_out_lock = Lock()
        self._handle_in = handle_in
        self._handle_in_lock = Lock()
        self._handle_lock = Lock()
        self._commit_lock = Lock()
        self.logger = logger

        for module in install_modules:
            self.handle('install_module',  module=module)  # not silenced
        if do_game_start:
            # not silenced; mustn't do anything before the game has started
            self.handle('do_game_start')

        self._node_stat_cache = StructuredDefaultDict(1, UnwrappingDict)
        self._portal_stat_cache = StructuredDefaultDict(2, UnwrappingDict)
        self._char_stat_cache = PickyDefaultDict(UnwrappingDict)
        self._things_cache = StructuredDefaultDict(1, ThingProxy)
        self._character_places_cache = StructuredDefaultDict(1, PlaceProxy)
        self._character_rulebooks_cache = StructuredDefaultDict(
            1, RuleBookProxy, kwargs_munger=lambda inst, k: {
                'engine': self,
                'bookname': (inst.key, k)
            }
        )
        self._char_node_rulebooks_cache = StructuredDefaultDict(
            1, RuleBookProxy, kwargs_munger=lambda inst, k: {
                'engine': self,
                'bookname': (inst.key, k)
            }
        )
        self._char_port_rulebooks_cache = StructuredDefaultDict(
            2, RuleBookProxy, kwargs_munger=lambda inst, k: {
                'engine': self,
                'bookname': (inst.parent.key, inst.key, k)
            }
        )
        self._character_portals_cache = PortalObjCache()
        self._character_avatars_cache = PickyDefaultDict(dict)
        self._avatar_characters_cache = PickyDefaultDict(dict)
        self._rule_obj_cache = {}
        self._rulebook_obj_cache = {}
        self._char_cache = {}
        self.character = CharacterMapProxy(self)
        self.eternal = EternalVarProxy(self)
        self.universal = GlobalVarProxy(self)
        self.rulebook = AllRuleBooksProxy(self)
        self.rule = AllRulesProxy(self)
        self.method = FuncStoreProxy(self, 'method')
        self.action = FuncStoreProxy(self, 'action')
        self.prereq = FuncStoreProxy(self, 'prereq')
        self.trigger = FuncStoreProxy(self, 'trigger')
        self.function = FuncStoreProxy(self, 'function')
        self.string = StringStoreProxy(self)
        self.rando = RandoProxy(self)
        self.send(self.pack({'command': 'get_watched_btt'}))
        self._branch, self._turn, self._tick = self.unpack(self.recv()[-1])
        self.method.load()
        self.action.load()
        self.prereq.load()
        self.trigger.load()
        self.function.load()
        self.string.load()
        self._rules_cache = self.handle('all_rules_delta')
        for rule in self._rules_cache:
            self._rule_obj_cache[rule] = RuleProxy(self, rule)
        self._rulebooks_cache = self.handle('all_rulebooks_delta')
        self._eternal_cache = self.handle('eternal_delta')
        self._universal_cache = self.handle('universal_delta')
        deltas = self.handle('get_char_deltas', chars='all')
        for char, delta in deltas.items():
            if char not in self.character:
                self._char_cache[char] = CharacterProxy(self, char)
            self.character[char]._apply_delta(delta)

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
        """Send a command to the LiSE core.

        The only positional argument should be the name of a
        method in :class:``EngineHandle``. All keyword arguments
        will be passed to it, with the exceptions of
        ``cb``, ``branching``, and ``silent``.

        With ``block=False``, don't wait for a result.
        This is best for when you want to make some change to the game
        state and already know what effect it will have.

        With ``branching=True``, handle paradoxes by creating new
        branches of history. I will switch to the new branch if needed.
        If I have an attribute ``branching_cb``, I'll call it if and
        only if the branch changes upon completing a command with
        ``branching=True``.

        With a function ``cb``, I will call ``cb`` when I get
        a result. If ``block=False`` this will happen in a thread.
        ``cb`` will be called with keyword arguments ``command``,
        the same command you asked for; ``result``, the value returned
        by it, possibly ``None``; and the present ``branch``,
        ``turn``, and ``tick``, possibly different than when you called
        ``handle``.

        If any of ``branching``, ``cb``, or ``future`` are ``True``,
        I will return a ``Future``. The ``Future``'s return value
        is a tuple of ``(command, branch, turn, tick, result)``.

        """
        if 'command' in kwargs:
            cmd = kwargs['command']
        elif cmd:
            kwargs['command'] = cmd
        else:
            raise TypeError("No command")
        branching = kwargs.get('branching', False)
        cb = kwargs.pop('cb', None)
        future = kwargs.pop('future', False)
        self._handle_lock.acquire()
        if kwargs.pop('block', True):
            assert not kwargs.get('silent')
            self.debug('EngineProxy: sending {}'.format(kwargs))
            self.send(self.pack(kwargs))
            command, branch, turn, tick, result = self.recv()
            assert cmd == command, \
                "Sent command {} but received results for {}".format(
                    cmd, command
                )
            r = self.unpack(result)
            self.debug('EngineProxy: received {}'.format(
                (command, branch, turn, tick, r)))
            if (branch, turn, tick) != self._btt():
                self._branch = branch
                self._turn = turn
                self._tick = tick
                self.time.send(self, branch=branch, turn=turn, tick=tick)
            if isinstance(r, Exception):
                self._handle_lock.release()
                raise r
            if cb:
                cb(command=command, branch=branch,
                   turn=turn, tick=tick, result=r)
            self._handle_lock.release()
            return r
        else:
            kwargs['silent'] = not (branching or cb or future)
            self.debug('EngineProxy: asynchronously sending {}'.format(kwargs))
            self.send(self.pack(kwargs))
            if branching:
                # what happens if more than one branching call
                # is happening at once?
                return self._submit(self._branching, cb)
            elif cb:
                return self._submit(self._callback, cb)
            if future:
                return self._submit(self._unpack_recv)
        self._handle_lock.release()

    def _unpack_recv(self):
        command, branch, turn, tick, result = self.recv()
        self._handle_lock.release()
        return command, branch, turn, tick, self.unpack(result)

    def _callback(self, cb):
        command, branch, turn, tick, result = self.recv()
        self._handle_lock.release()
        res = self.unpack(result)
        self.debug('EngineProxy: received, with callback {}: {}'.format(
            cb, (command, branch, turn, tick, res))
        )
        ex = None
        if isinstance(res, Exception):
            ex = res
        try:
            if isinstance(res[0], Exception):
                ex = res[0]
        except TypeError:
            pass
        if ex:
            self.warning(
                "{} raised by command {}, trying to run callback {} with it"
                    .format(repr(ex), command, cb))
        cb(command=command, branch=branch, turn=turn, tick=tick, result=res)
        return command, branch, turn, tick, res

    def _branching(self, cb=None):
        command, branch, turn, tick, result = self.recv()
        self._handle_lock.release()
        r = self.unpack(result)
        self.debug('EngineProxy: received, with branching, {}'.format(
            (command, branch, turn, tick, r)))
        if (branch, turn, tick) != (self._branch, self._turn, self._tick):
            self._branch = branch
            self._turn = turn
            self._tick = tick
            self.time.send(self, branch=branch, turn=turn, tick=tick)
            if hasattr(self, 'branching_cb'):
                self.branching_cb(command=command, branch=branch,
                                  turn=turn, tick=tick, result=r)
        if cb:
            cb(command=command, branch=branch, turn=turn, tick=tick, result=r)
        return command, branch, turn, tick, r

    def _call_with_recv(self, *cbs, **kwargs):
        cmd, branch, turn, tick, res = self.recv()
        received = self.unpack(res)
        self.debug('EngineProxy: received {}'.format(
            (cmd, branch, turn, tick, received)))
        if isinstance(received, Exception):
            raise received
        for cb in cbs:
            cb(command=cmd, branch=branch,
               turn=turn, tick=tick, result=received, **kwargs)
        return received

    def _upd_caches(self, command, branch, turn, tick, result, no_del=False):
        deleted = set(self.character.keys())
        result, deltas = result
        self.eternal._update_cache(deltas.pop('eternal', {}))
        self.universal._update_cache(deltas.pop('universal', {}))
        # I think if you travel back to before a rule was created
        # it'll show up empty.
        # That's ok I guess
        for rule, delta in deltas.pop('rules', {}).items():
            if rule in self._rules_cache:
                self._rules_cache[rule].update(delta)
            else:
                delta.setdefault('triggers', [])
                delta.setdefault('prereqs', [])
                delta.setdefault('actions', [])
                self._rules_cache[rule] = delta
            if rule not in self._rule_obj_cache:
                self._rule_obj_cache[rule] = RuleProxy(self, rule)
            ruleproxy = self._rule_obj_cache[rule]
            ruleproxy.send(ruleproxy, **delta)
        rulebookdeltas = deltas.pop('rulebooks', {})
        self._rulebooks_cache.update(rulebookdeltas)
        for rulebook, delta in rulebookdeltas.items():
            if rulebook not in self._rulebook_obj_cache:
                self._rulebook_obj_cache = RuleBookProxy(self, rulebook)
            rulebookproxy = self._rulebook_obj_cache[rulebook]
            # the "delta" is just the rules list, for now
            rulebookproxy.send(rulebookproxy, rules=delta)
        for (char, chardelta) in deltas.items():
            if char not in self._char_cache:
                self._char_cache[char] = CharacterProxy(self, char)
            chara = self.character[char]
            chara._apply_delta(chardelta)
            deleted.discard(char)
        if no_del:
            return
        for char in deleted:
            del self._char_cache[char]

    def _btt(self):
        return self._branch, self._turn, self._tick

    def _set_time(self, command, branch, turn, tick, result, **kwargs):
        self._branch = branch
        self._turn = turn
        self._tick = tick
        self.time.send(self, branch=branch, turn=turn, tick=tick)

    def is_parent_of(self, parent, child):
        return self.handle('is_parent_of', parent=parent, child=child)

    def _pull_async(self, chars, cb):
        if not callable(cb):
            raise TypeError("Uncallable callback")
        self.send(self.pack({
            'silent': False,
            'command': 'get_char_deltas',
            'chars': chars
        }))
        cbs = [self._upd_caches]
        if cb:
            cbs.append(cb)
        self._call_with_recv(cbs)

    def pull(self, chars='all', cb=None, block=True):
        """Update the state of all my proxy objects from the real objects."""
        if block:
            deltas = self.handle('get_char_deltas', chars=chars, cb=self._upd_deltas)
            if cb:
                cb(deltas)
        else:
            return self._submit(self._pull_async, chars, cb)

    def _upd_and_cb(self, cb, *args, **kwargs):
        self._upd_caches(*args, no_del=True, **kwargs)
        self._set_time(*args, no_del=True, **kwargs)
        if cb:
            cb(*args, **kwargs)

    # TODO: make this into a Signal, like it is in the LiSE core
    def next_turn(self, cb=None, block=False):
        if cb and not callable(cb):
            raise TypeError("Uncallable callback")
        return self.handle(
            'next_turn',
            block=block,
            cb=partial(self._upd_and_cb, cb)
        )

    def time_travel(self, branch, turn, tick=None, chars='all',
                    cb=None, block=True):
        """Move to a different point in the timestream

        Needs ``branch`` and ``turn`` arguments. The ``tick`` is
        optional; if unspecified, you'll travel to the last tick
        in the turn.

        May take a callback function ``cb``, which will receive a
        dictionary describing changes to the characters in ``chars``.
        ``chars`` defaults to 'all', indicating that every character
        should be included, but may be a list of character names
        to include.

        With ``block=True`` (the default), wait until finished computing
        differences before returning. Otherwise my ``branch``, ``turn``,
        and ``tick`` will stay where they are until that's done.

        """
        if cb and not chars:
            raise TypeError("Callbacks require chars")
        if cb is not None and not callable(cb):
            raise TypeError("Uncallable callback")
        return self.handle(
            'time_travel',
            block=block,
            branch=branch,
            turn=turn,
            tick=tick,
            chars=chars,
            cb=partial(self._upd_and_cb, cb)
        )

    def add_character(self, char, data={}, block=False, **attr):
        if char in self._char_cache:
            raise KeyError("Character already exists")
        assert char not in self._char_stat_cache
        if not isinstance(data, dict):
            # it's a networkx graph
            data = {
                'place': {k: v for k, v in data._node.items() if 'location' not in v},
                'thing': {k: v for k, v in data._node.items() if 'location' in v},
                'edge': data._adj
            }
        self._char_cache[char] = character = CharacterProxy(self, char)
        self._char_stat_cache[char] = attr
        placedata = data.get('place', data.get('node', {}))
        for place, stats in placedata.items():
            assert place not in self._character_places_cache[char]
            assert place not in self._node_stat_cache[char]
            self._character_places_cache[char][place] \
                = PlaceProxy(character,  place)
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
            self._things_cache[char][thing] \
                = ThingProxy(char, thing, loc)
            self._node_stat_cache[char][thing] = stats
        portdata = data.get('edge', data.get('portal', data.get('adj',  {})))
        for orig, dests in portdata.items():
            assert orig not in self._character_portals_cache.successors[char]
            assert orig not in self._portal_stat_cache[char]
            for dest, stats in dests.items():
                assert dest not in self._character_portals_cache.successors[char][orig]
                assert dest not in self._portal_stat_cache[char][orig]
                self._character_portals_cache.store(char, orig, dest, PortalProxy(self.character[char], orig, dest))
                self._portal_stat_cache[char][orig][dest] = stats
        self.handle(
            command='add_character', char=char, data=data, attr=attr,
            block=block, branching=True
        )

    def new_character(self, char, **attr):
        self.add_character(char, block=True, **attr)
        return self._char_cache[char]
    new_graph = new_character

    def del_character(self, char):
        if char not in self._char_cache:
            raise KeyError("No such character")
        del self._char_cache[char]
        del self._char_stat_cache[char]
        del self._character_places_cache[char]
        del self._things_cache[char]
        self._character_portals_cache.delete_char(char)
        self.handle(command='del_character', char=char,
                    block=False, branching=True)
    del_graph = del_character

    def del_node(self, char, node):
        if char not in self._char_cache:
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
            block=False,
            branching=True
        )

    def del_portal(self, char, orig, dest):
        if char not in self._char_cache:
            raise KeyError("No such character")
        self._character_portals_cache.delete(char, orig, dest)
        self.handle(
            command='del_portal',
            char=char,
            orig=orig,
            dest=dest,
            block=False,
            branching=True
        )

    def commit(self):
        self._commit_lock.acquire()
        self.handle('commit', block=False, cb=self._release_commit_lock)

    def _release_commit_lock(self, *, command, branch, turn, tick, result):
        self._commit_lock.release()

    def close(self):
        self._commit_lock.acquire()
        self._commit_lock.release()
        self.handle('close')
        self.send('shutdown')

    def _node_contents(self, character, node):
        # very slow. do better
        for thing in self.character[character].thing.values():
            if thing['location'] == node:
                yield thing.name


def subprocess(
    args, kwargs, handle_out_pipe, handle_in_pipe, logq, loglevel
):
    def log(typ, data):
        from os import getpid
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
    engine_handle = EngineHandle(args, kwargs, logq, loglevel=loglevel)

    while True:
        inst = handle_out_pipe.recv()
        if inst == 'shutdown':
            handle_out_pipe.close()
            handle_in_pipe.close()
            logq.close()
            return 0
        instruction = engine_handle.unpack(inst)
        silent = instruction.pop('silent',  False)
        cmd = instruction.pop('command')

        branching = instruction.pop('branching', False)
        try:
            if branching:
                try:
                    r = getattr(engine_handle, cmd)(**instruction)
                except HistoryError:
                    engine_handle.increment_branch()
                    r = getattr(engine_handle, cmd)(**instruction)
            else:
                r = getattr(engine_handle, cmd)(**instruction)
        except Exception as e:
            log('exception', repr(e))
            handle_in_pipe.send((
                cmd, engine_handle.branch,
                engine_handle.turn, engine_handle.tick,
                engine_handle.pack(e)
            ))
            continue
        if silent:
            continue
        handle_in_pipe.send((
            cmd, engine_handle.branch, engine_handle.turn, engine_handle.tick,
            engine_handle.pack(r)
        ))
        if hasattr(engine_handle, '_after_ret'):
            engine_handle._after_ret()
            del engine_handle._after_ret


class RedundantProcessError(ProcessError):
    """Asked to start a process that has already started"""


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
                self.logq,
                loglevel
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
                if isinstance(level, int):
                    level = {
                        10: 'debug',
                        20: 'info',
                        30: 'warning',
                        40: 'error',
                        50: 'critical'
                    }[level]
                getattr(self.logger, level)(message)
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
