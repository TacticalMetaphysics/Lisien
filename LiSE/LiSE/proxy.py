# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3.
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

To use these, first instantiate an ``EngineProcessManager``, then
call its ``start`` method with the same arguments you'd give a real
``Engine``. You'll get an ``EngineProxy``, which acts like the underlying
``Engine`` for most purposes.

"""

import os
import sys
import logging
from abc import abstractmethod
from random import Random
from collections.abc import Mapping, MutableMapping, MutableSequence
from functools import partial, cached_property
from threading import Thread, Lock
from multiprocessing import Process, Pipe, Queue, ProcessError
from concurrent.futures import ThreadPoolExecutor
from queue import Empty
from time import monotonic
from types import MethodType
from typing import Hashable, Tuple, Optional, Iterator, List, Union

import networkx as nx
from blinker import Signal
import zlib
import msgpack

from .allegedb import OutOfTimelineError, Key
from .allegedb.cache import PickyDefaultDict, StructuredDefaultDict
from .allegedb.wrap import DictWrapper, ListWrapper, SetWrapper, UnwrappingDict
from .character import Facade
from .util import (
	getatt,
	AbstractEngine,
	MsgpackExtensionType,
	AbstractCharacter,
)
from .xcollections import (
	AbstractLanguageDescriptor,
	FunctionStore,
	StringStore,
)
from .node import NodeContent, UserMapping, Place, Thing
from .portal import Portal


class CachingProxy(MutableMapping, Signal):
	"""Abstract class for proxies to LiSE entities or mappings thereof"""

	_cache: dict
	rulebook: "RuleBookProxy"
	engine: "EngineProxy"

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
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		self._set_item(k, v)
		self._cache[k] = self._cache_set_munge(k, v)
		self.send(self, key=k, value=v)

	def __delitem__(self, k):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		if k not in self:
			raise KeyError("No such key: {}".format(k))
		self._del_item(k)
		del self._cache[k]
		self.send(self, key=k, value=None)

	@abstractmethod
	def _apply_delta(self, delta):
		raise NotImplementedError("_apply_delta")

	def _cache_get_munge(self, k, v):
		return v

	def _cache_set_munge(self, k, v):
		return v

	@abstractmethod
	def _set_rulebook_proxy(self, k):
		raise NotImplementedError("_set_rulebook_proxy")

	@abstractmethod
	def _set_item(self, k, v):
		raise NotImplementedError("Abstract method")

	@abstractmethod
	def _del_item(self, k):
		raise NotImplementedError("Abstract method")


class CachingEntityProxy(CachingProxy):
	"""Abstract class for proxy objects representing LiSE entities"""

	name: Hashable

	def _cache_get_munge(self, k, v):
		if isinstance(v, dict):
			return DictWrapper(
				lambda: self._cache[k], partial(self._set_item, k), self, k
			)
		elif isinstance(v, list):
			return ListWrapper(
				lambda: self._cache[k], partial(self._set_item, k), self, k
			)
		elif isinstance(v, set):
			return SetWrapper(
				lambda: self._cache[k], partial(self._set_item, k), self, k
			)
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
				inst.engine, inst._get_default_rulebook_name()
			)
			inst._set_rulebook_proxy(proxy)
		return proxy

	def __set__(self, inst, val):
		if inst.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		if hasattr(val, "name"):
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
	"""A mapping to the ``CharacterProxy``s that have this node as a unit"""

	def _user_names(self):
		for user, avatars in self.node.engine._unit_characters_cache[
			self.node._charname
		].items():
			if self.node.name in avatars:
				yield user


class ProxyNeighborMapping(Mapping):
	__slots__ = ("_node",)

	def __init__(self, node: "NodeProxy") -> None:
		self._node = node

	def __iter__(self) -> Iterator[Key]:
		seen = set()
		for k in self._node.character.adj[self._node.name]:
			yield k
			seen.add(k)
		for k in self._node.character.pred[self._node.name]:
			if k not in seen:
				yield k

	def __len__(self) -> int:
		return len(
			self._node.character.adj[self._node.name].keys()
			| self._node.character.pred[self._node.name].keys()
		)

	def __getitem__(self, item: Key) -> "NodeProxy":
		if (
			item in self._node.character.adj[self._node.name]
			or item in self._node.character.pred[self._node.name]
		):
			return self._node.character.node[item]
		raise KeyError("Not a neighbor")


class NodeProxy(CachingEntityProxy):
	rulebook = RulebookProxyDescriptor()

	@property
	def user(self):
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
		return self.engine._char_node_rulebooks_cache[self._charname][
			self.name
		]

	def _set_rulebook_proxy(self, rb):
		self.engine._char_node_rulebooks_cache[self._charname][self.name] = (
			RuleBookProxy(self.engine, rb)
		)

	def _set_rulebook(self, rb):
		self.engine.handle(
			"set_node_rulebook",
			char=self._charname,
			node=self.name,
			rulebook=rb,
			branching=True,
		)
		self._set_rulebook_proxy(rb)

	def __init__(self, character: "CharacterProxy", nodename: Key, **stats):
		self.engine = character.engine
		self._charname = character.name
		self.name = nodename
		self._cache.update(stats)
		super().__init__()

	def __iter__(self):
		yield from super().__iter__()
		yield "character"
		yield "name"

	def __eq__(self, other):
		return (
			isinstance(other, NodeProxy)
			and self._charname == other._charname
			and self.name == other.name
		)

	def __contains__(self, k):
		if k in ("character", "name"):
			return True
		return super().__contains__(k)

	def __getitem__(self, k):
		if k == "character":
			return self._charname
		elif k == "name":
			return self.name
		return super().__getitem__(k)

	def _set_item(self, k, v):
		if k == "name":
			raise KeyError("Nodes can't be renamed")
		self.engine.handle(
			command="set_node_stat",
			char=self._charname,
			node=self.name,
			k=k,
			v=v,
			branching=True,
		)

	def _del_item(self, k):
		if k == "name":
			raise KeyError("Nodes need names")
		self.engine.handle(
			command="del_node_stat",
			char=self._charname,
			node=self.name,
			k=k,
			branching=True,
		)

	def delete(self):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		self.engine.del_node(self._charname, self.name)

	@property
	def content(self):
		return NodeContent(self)

	def contents(self):
		return self.content.values()

	@property
	def neighbor(self):
		return ProxyNeighborMapping(self)

	def neighbors(self):
		return self.neighbor.values()

	def add_thing(self, name, **kwargs):
		return self.character.add_thing(name, self.name, **kwargs)

	def new_thing(self, name, **kwargs):
		return self.character.new_thing(name, self.name, **kwargs)

	def shortest_path(
		self, dest: Union[Key, "NodeProxy"], weight: Key = None
	) -> List[Key]:
		"""Return a list of node names leading from me to ``dest``.

		Raise ``ValueError`` if ``dest`` is not a node in my character
		or the name of one.

		"""
		return nx.shortest_path(
			self.character, self.name, self._plain_dest_name(dest), weight
		)

	def _plain_dest_name(self, dest):
		if isinstance(dest, NodeProxy):
			if dest.character != self.character:
				raise ValueError(
					"{} not in {}".format(dest.name, self.character.name)
				)
			return dest.name
		else:
			if dest in self.character.node:
				return dest
			raise ValueError("{} not in {}".format(dest, self.character.name))


class PlaceProxy(NodeProxy):
	def __repr__(self):
		return "<proxy to {}.place[{}] at {}>".format(
			self._charname, repr(self.name), id(self)
		)

	def _apply_delta(self, delta):
		for k, v in delta.items():
			if k == "rulebook":
				if k != self.rulebook.name:
					self._set_rulebook_proxy(k)
					self.send(self, key="rulebook", value=v)
					self.character.place.send(self, key="rulebook", value=v)
					self.character.node.send(self, key="rulebook", value=v)
				continue
			if v is None:
				if k in self._cache:
					del self._cache[k]
					self.send(self, key=k, value=None)
					self.character.place.send(self, key=k, value=None)
					self.character.node.send(self, key=k, value=None)
			elif k not in self._cache or self._cache[k] != v:
				self._cache[k] = v
				self.send(self, key=k, value=v)
				self.character.place.send(self, key=k, value=v)
				self.character.node.send(self, key=k, value=v)


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
					"Maybe you want a unit?"
				)
			locn = v.name
		elif v in self.character.node:
			locn = v
		else:
			raise TypeError("Location must be a node or the name of one")
		self._set_location(locn)

	def __init__(
		self,
		character: "CharacterProxy",
		name: Key,
		location: Key = None,
		**kwargs,
	):
		if location is None and getattr(
			character.engine, "_initialized", True
		):
			raise ValueError("Thing must have location")
		super().__init__(character, name)
		self._location = location
		self._cache.update(kwargs)

	def __iter__(self):
		yield from super().__iter__()
		yield "location"

	def __getitem__(self, k):
		if k == "location":
			return self._location
		return super().__getitem__(k)

	def _apply_delta(self, delta):
		for k, v in delta.items():
			if k == "rulebook":
				if v != self.rulebook.name:
					self._set_rulebook_proxy(k)
					self.send(self, key="rulebook", value=v)
					self.character.thing.send(self, key="rulebook", value=v)
					self.character.node.send(self, key="rulebook", value=v)
			elif v is None:
				if k in self._cache:
					del self._cache[k]
					self.send(self, key=k, value=None)
					self.character.thing.send(self, key=k, value=None)
					self.character.node.send(self, key=k, value=None)
			elif k == "location":
				self._location = v
				self.send(self, key=k, value=v)
				self.character.thing.send(self, key=k, value=v)
				self.character.node.send(self, key=k, value=v)
			elif k not in self._cache or self._cache[k] != v:
				self._cache[k] = v
				self.send(self, key=k, value=v)
				self.character.thing.send(self, key=k, value=v)
				self.character.node.send(self, key=k, value=v)

	def _set_location(self, v):
		self._location = v
		self.engine.handle(
			command="set_thing_location",
			char=self.character.name,
			thing=self.name,
			loc=v,
			branching=True,
		)

	def __setitem__(self, k, v):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		if k == "location":
			self._set_location(v)
		elif k == "rulebook":
			self._set_rulebook(v)
		else:
			super().__setitem__(k, v)
		self.send(self, key=k, value=v)
		self.character.thing.send(self, key=k, value=v)
		self.character.node.send(self, key=k, value=v)

	def __repr__(self):
		return "<proxy to {}.thing[{}]@{} at {}>".format(
			self._charname, self.name, self._location, id(self)
		)

	def follow_path(self, path, weight=None):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		self.engine.handle(
			command="thing_follow_path",
			char=self._charname,
			thing=self.name,
			path=path,
			weight=weight,
		)

	def go_to_place(self, place, weight=None):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		if hasattr(place, "name"):
			place = place.name
		self.engine.handle(
			command="thing_go_to_place",
			char=self._charname,
			thing=self.name,
			place=place,
			weight=weight,
		)

	def travel_to(self, dest, weight=None, graph=None):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		if hasattr(dest, "name"):
			dest = dest.name
		if hasattr(graph, "name"):
			graph = graph.name
		return self.engine.handle(
			command="thing_travel_to",
			char=self._charname,
			thing=self.name,
			dest=dest,
			weight=weight,
			graph=graph,
		)

	def travel_to_by(self, dest, arrival_tick, weight=None, graph=None):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		if hasattr(dest, "name"):
			dest = dest.name
		if hasattr(graph, "name"):
			graph = graph.name
		self.engine.handle(
			command="thing_travel_to_by",
			char=self._charname,
			thing=self.name,
			dest=dest,
			arrival_tick=arrival_tick,
			weight=weight,
			graph=graph,
		)


Thing.register(ThingProxy)


class PortalProxy(CachingEntityProxy):
	rulebook = RulebookProxyDescriptor()

	def _apply_delta(self, delta):
		for k, v in delta.items():
			if k == "rulebook":
				if k != self.rulebook.name:
					self._set_rulebook_proxy(k)
				continue
			if v is None:
				if k in self._cache:
					del self._cache[k]
					self.send(self, key=k, value=None)
					self.character.portal.send(self, key=k, value=None)
			elif k not in self._cache or self._cache[k] != v:
				self._cache[k] = v
				self.send(self, key=k, value=v)
				self.character.portal.send(self, key=k, value=v)

	def _get_default_rulebook_name(self):
		return self._charname, self._origin, self._destination

	def _get_rulebook_proxy(self):
		return self.engine._char_port_rulebooks_cache[self._charname][
			self._origin
		][self._destination]

	def _set_rulebook_proxy(self, rb):
		self.engine._char_port_rulebooks_cache[self._charname][self._origin][
			self._destination
		] = RuleBookProxy(self.engine, rb)

	def _set_rulebook(self, rb):
		self.engine.handle(
			command="set_portal_rulebook",
			char=self._charname,
			orig=self._origin,
			dest=self._destination,
			rulebook=rb,
		)

	def _get_rulebook_name(self):
		return self.engine.handle(
			command="get_portal_rulebook",
			char=self._charname,
			orig=self._origin,
			dest=self._destination,
		)

	@property
	def _cache(self):
		return self.engine._portal_stat_cache[self._charname][self._origin][
			self._destination
		]

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
		if (
			self._origin not in self.character.pred
			or self._destination not in self.character.pred[self._origin]
		):
			return None
		return self.character.pred[self._origin][self._destination]

	def _set_item(self, k, v):
		self.engine.handle(
			command="set_portal_stat",
			char=self._charname,
			orig=self._origin,
			dest=self._destination,
			k=k,
			v=v,
			branching=True,
		)
		self.send(self, k=k, v=v)
		self.character.portal.send(self, k=k, v=v)

	def _del_item(self, k):
		self.engine.handle(
			command="del_portal_stat",
			char=self._charname,
			orig=self._origin,
			dest=self._destination,
			k=k,
			branching=True,
		)
		self.character.portal.send(self, k=k, v=None)
		self.send(self, k=k, v=None)

	def __init__(self, character, origname, destname):
		self.engine = character.engine
		self._charname = character.name
		self._origin = origname
		self._destination = destname
		super().__init__()

	def __eq__(self, other):
		return (
			hasattr(other, "character")
			and hasattr(other, "origin")
			and hasattr(other, "destination")
			and self.character == other.character
			and self.origin == other.origin
			and self.destination == other.destination
		)

	def __repr__(self):
		return "<proxy to {}.portal[{}][{}] at {}>".format(
			self._charname,
			repr(self._origin),
			repr(self._destination),
			id(self),
		)

	def __getitem__(self, k):
		if k == "origin":
			return self._origin
		elif k == "destination":
			return self._destination
		elif k == "character":
			return self._charname
		return super().__getitem__(k)

	def delete(self):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		self.engine.del_portal(self._charname, self._origin, self._destination)


Portal.register(PortalProxy)


class NodeMapProxy(MutableMapping, Signal):
	rulebook = RulebookProxyDescriptor()

	def _get_default_rulebook_name(self):
		return self._charname, "character_node"

	def _get_rulebook_proxy(self):
		return self.engine._character_rulebooks_cache[self._charname]["node"]

	def _set_rulebook_proxy(self, rb):
		self.engine._character_rulebooks_cache[self._charname]["node"] = (
			RuleBookProxy(self.engine, rb)
		)

	def _set_rulebook(self, rb):
		self.engine.handle(
			"set_character_node_rulebook",
			char=self._charname,
			rulebook=rb,
			branching=True,
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
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		self.character.place[k] = v

	def __delitem__(self, k):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
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
			"update_nodes", char=self.character.name, patch=patch
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
		return self.name, "character_thing"

	def _get_rulebook_proxy(self):
		return self.engine._character_rulebooks_cache[self.name]["thing"]

	def _set_rulebook_proxy(self, rb):
		self.engine._character_rulebooks_cache[self.name]["thing"] = (
			RuleBookProxy(self.engine, rb)
		)

	def _set_rulebook(self, rb):
		self.engine.handle(
			"set_character_thing_rulebook",
			char=self.name,
			rulebook=rb,
			branching=True,
		)

	def _apply_delta(self, delta):
		raise NotImplementedError("_apply_delta")

	@property
	def character(self):
		return self.engine.character[self.name]

	@property
	def _cache(self):
		return self.engine._things_cache.setdefault(self.name, {})

	def __init__(self, engine_proxy, charname):
		self.engine = engine_proxy
		self.name = charname
		super().__init__()

	def __eq__(self, other):
		return self is other

	def _cache_set_munge(self, k, v):
		return ThingProxy(
			self,
			*self.engine.handle(
				"get_thing_special_stats", char=self.name, thing=k
			),
		)

	def _set_item(self, k, v):
		self.engine.handle(
			command="set_thing",
			char=self.name,
			thing=k,
			statdict=v,
			branching=True,
		)
		self._cache[k] = ThingProxy(self.engine, self.name, v.pop("location"))
		self.engine._node_stat_cache[self.name][k] = v

	def _del_item(self, k):
		self.engine.handle(
			command="del_node", char=self.name, node=k, branching=True
		)
		del self._cache[k]
		del self.engine._node_stat_cache[self.name][k]

	def patch(self, d: dict):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		places = d.keys() & self.character.place.keys()
		if places:
			raise KeyError(f"Tried to patch places on thing mapping: {places}")
		self.character.node.patch(d)


class PlaceMapProxy(CachingProxy):
	rulebook = RulebookProxyDescriptor()

	def _get_default_rulebook_name(self):
		return self.name, "character_place"

	def _get_rulebook_proxy(self):
		return self.engine._character_rulebooks_cache[self.name]["place"]

	def _set_rulebook_proxy(self, rb):
		self.engine._character_rulebooks_cache[self.name]["place"] = (
			RuleBookProxy(self.engine, rb)
		)

	def _set_rulebook(self, rb):
		self.engine.handle(
			"set_character_place_rulebook",
			char=self.name,
			rulebook=rb,
			branching=True,
		)

	def _apply_delta(self, delta):
		raise NotImplementedError("_apply_delta")

	@property
	def character(self):
		return self.engine.character[self.name]

	@property
	def _cache(self):
		return self.engine._character_places_cache.setdefault(self.name, {})

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
			command="set_place",
			char=self.name,
			place=k,
			statdict=v,
			branching=True,
		)
		self.engine._node_stat_cache[self.name][k] = v

	def _del_item(self, k):
		self.engine.handle(
			command="del_node", char=self.name, node=k, branching=True
		)
		del self.engine._node_stat_cache[self.name][k]

	def patch(self, d: dict):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		things = d.keys() & self.character.thing.keys()
		if things:
			raise KeyError(f"Tried to patch things on place mapping: {things}")
		self.character.node.patch(d)


class SuccessorsProxy(CachingProxy):
	@property
	def _cache(self):
		return self.engine._character_portals_cache.successors[self._charname][
			self._orig
		]

	def _set_rulebook_proxy(self, k):
		raise NotImplementedError(
			"Set the rulebook on the .portal attribute, not this"
		)

	def __init__(self, engine_proxy, charname, origname):
		self.engine = engine_proxy
		self._charname = charname
		self._orig = origname
		super().__init__()

	def __eq__(self, other):
		return (
			isinstance(other, SuccessorsProxy)
			and self.engine is other.engine
			and self._charname == other._charname
			and self._orig == other._orig
		)

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
			command="set_portal",
			char=self._charname,
			orig=self._orig,
			dest=dest,
			statdict=value,
			branching=True,
		)

	def _del_item(self, dest):
		self.engine.del_portal(self._charname, self._orig, dest)


class CharSuccessorsMappingProxy(CachingProxy):
	rulebook = RulebookProxyDescriptor()

	def _get_default_rulebook_anme(self):
		return self.name, "character_portal"

	def _get_rulebook_proxy(self):
		return self.engine._character_rulebooks_cache[self.name]["portal"]

	def _set_rulebook_proxy(self, rb):
		self.engine._character_rulebooks_cache[self.name]["portal"] = (
			RuleBookProxy(self.engine, rb)
		)

	def _set_rulebook(self, rb):
		self.engine.handle(
			"set_character_portal_rulebook",
			char=self.character.name,
			rulebook=rb,
			branching=True,
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
			isinstance(other, CharSuccessorsMappingProxy)
			and other.engine is self.engine
			and other.name == self.name
		)

	def _cache_set_munge(self, k, v):
		return {vk: PortalProxy(self, vk, vv) for (vk, vv) in v.items()}

	def __getitem__(self, k):
		return SuccessorsProxy(self.engine, self.name, k)

	def _apply_delta(self, delta):
		for o, ds in delta.items():
			cache = self._cache[o]
			for d, stats in ds.items():
				if d not in cache:
					cache[d] = PortalProxy(self.character, o, d)
				cache[d]._apply_delta(stats)

	def _set_item(self, orig, val):
		self.engine.handle(
			command="character_set_node_successors",
			character=self.name,
			node=orig,
			val=val,
			branching=True,
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
		return iter(
			self.engine._character_portals_cache.predecessors[self._charname][
				self.name
			]
		)

	def __len__(self):
		return len(
			self.engine._character_portals_cache.predecessors[self._charname][
				self.name
			]
		)

	def __contains__(self, k):
		return (
			k
			in self.engine._character_portals_cache.predecessors[
				self._charname
			][self.name]
		)

	def __getitem__(self, k):
		return self.engine._character_portals_cache.predecessors[
			self._charname
		][self.name][k]

	def __setitem__(self, k, v):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		self.engine._character_portals_cache.store(
			self._charname,
			self.name,
			k,
			PortalProxy(self.engine, self._charname, k, self.name),
		)
		self.engine.handle(
			command="set_place",
			char=self._charname,
			place=k,
			statdict=v,
			branching=True,
		)
		self.engine.handle(
			"set_portal", (self._charname, k, self.name), branching=True
		)

	def __delitem__(self, k):
		self.engine.del_portal(self._charname, k, self.name)


class CharPredecessorsMappingProxy(MutableMapping):
	def __init__(self, engine_proxy, charname):
		self.engine = engine_proxy
		self.name = charname
		self._cache = {}

	def __contains__(self, k):
		return (
			k in self.engine._character_portals_cache.predecessors[self.name]
		)

	def __iter__(self):
		return iter(
			self.engine._character_portals_cache.predecessors[self.name]
		)

	def __len__(self):
		return len(
			self.engine._character_portals_cache.predecessors[self.name]
		)

	def __getitem__(self, k):
		if k not in self._cache:
			self._cache[k] = PredecessorsProxy(self.engine, self.name, k)
		return self._cache[k]

	def __setitem__(self, k, v):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		for pred, proxy in v.items():
			self.engine._character_portals_cache.store(
				self.name, pred, k, proxy
			)
		self.engine.handle(
			command="character_set_node_predecessors",
			char=self.name,
			node=k,
			preds=v,
			branching=True,
		)

	def __delitem__(self, k):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
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
			isinstance(other, CharStatProxy)
			and self.engine is other.engine
			and self.name == other.name
		)

	def _set_rulebook_proxy(self, k):
		raise NotImplementedError(
			"Set rulebooks on the Character proxy, not this"
		)

	def _get(self, k=None):
		if k is None:
			return self
		return self._cache[k]

	def _set_item(self, k, v):
		if k == "name":
			raise KeyError("Can't change names")
		self.engine.handle(
			command="set_character_stat",
			char=self.name,
			k=k,
			v=v,
			branching=True,
		)

	def _del_item(self, k):
		self.engine.handle(
			command="del_character_stat", char=self.name, k=k, branching=True
		)

	def _apply_delta(self, delta):
		for k, v in delta.items():
			if k == "rulebook":
				if k != self.rulebook.name:
					self._set_rulebook_proxy(k)
				continue
			if v is None:
				if k in self._cache:
					del self._cache[k]
					self.send(self, key=k, value=None)
			elif k not in self._cache or self._cache[k] != v:
				self._cache[k] = v
				self.send(self, key=k, value=v)


class RuleProxy(Signal):
	@staticmethod
	def _nominate(v):
		ret = []
		for whatever in v:
			if hasattr(whatever, "name"):
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
		return self._cache.setdefault("triggers", [])

	@triggers.setter
	def triggers(self, v):
		self._cache["triggers"] = v
		self.engine.handle(
			"set_rule_triggers",
			rule=self.name,
			triggers=self._nominate(v),
			branching=True,
		)
		self.send(self, triggers=v)

	@property
	def prereqs(self):
		return self._cache.setdefault("prereqs", [])

	@prereqs.setter
	def prereqs(self, v):
		self._cache["prereqs"] = v
		self.engine.handle(
			"set_rule_prereqs",
			rule=self.name,
			prereqs=self._nominate(v),
			branching=True,
		)
		self.send(self, prereqs=v)

	@property
	def actions(self):
		return self._cache.setdefault("actions", [])

	@actions.setter
	def actions(self, v):
		self._cache["actions"] = v
		self.engine.handle(
			"set_rule_actions",
			rule=self.name,
			actions=self._nominate(v),
			branching=True,
		)
		self.send(self, actions=v)

	def __init__(self, engine, rulename):
		super().__init__()
		self.engine = engine
		self.name = self._name = rulename

	def __eq__(self, other):
		return hasattr(other, "name") and self.name == other.name


class RuleBookProxy(MutableSequence, Signal):
	@property
	def _cache(self):
		return self.engine._rulebooks_cache.setdefault(self.name, ([], 0.0))[0]

	@property
	def priority(self):
		return self.engine._rulebooks_cache.setdefault(self.name, ([], 0.0))[1]

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
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		if isinstance(v, RuleProxy):
			v = v._name
		self._cache[i] = v
		self.engine.handle(
			command="set_rulebook_rule",
			rulebook=self.name,
			i=i,
			rule=v,
			branching=True,
		)
		self.send(self, i=i, val=v)

	def __delitem__(self, i):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		del self._cache[i]
		self.engine.handle(
			command="del_rulebook_rule",
			rulebook=self.name,
			i=i,
			branching=True,
		)
		self.send(self, i=i, val=None)

	def insert(self, i, v):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		if isinstance(v, RuleProxy):
			v = v._name
		self._cache.insert(i, v)
		self.engine.handle(
			command="ins_rulebook_rule",
			rulebook=self.name,
			i=i,
			rule=v,
			branching=True,
		)
		for j in range(i, len(self)):
			self.send(self, i=j, val=self[j])


class UnitMapProxy(Mapping):
	rulebook = RulebookProxyDescriptor()
	engine = getatt("character.engine")

	def _get_default_rulebook_name(self):
		return self.character.name, "unit"

	def _get_rulebook_proxy(self):
		return self.engine._character_rulebooks_cache[self.character.name][
			"unit"
		]

	def _set_rulebook_proxy(self, rb):
		self.engine._character_rulebooks_cache[self.character.name]["unit"] = (
			RuleBookProxy(self.engine, rb)
		)

	def _set_rulebook(self, rb):
		self.engine.handle(
			"set_unit_rulebook",
			char=self.character.name,
			rulebook=rb,
			branching=True,
		)

	def __init__(self, character):
		self.character = character

	def __iter__(self):
		yield from self.character.engine._character_units_cache[
			self.character.name
		]

	def __len__(self):
		return len(
			self.character.engine._character_units_cache[self.character.name]
		)

	def __contains__(self, k):
		return (
			k
			in self.character.engine._character_units_cache[
				self.character.name
			]
		)

	def __getitem__(self, k):
		if k not in self:
			raise KeyError(
				"{} has no unit in {}".format(self.character.name, k)
			)
		return self.GraphUnitsProxy(
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

	class GraphUnitsProxy(Mapping):
		def __init__(self, character, graph):
			self.character = character
			self.graph = graph

		def __iter__(self):
			yield from self.character.engine._character_units_cache[
				self.character.name
			][self.graph.name]

		def __len__(self):
			return len(
				self.character.engine._character_units_cache[
					self.character.name
				][self.graph.name]
			)

		def __contains__(self, k):
			cache = self.character.engine._character_units_cache[
				self.character.name
			]
			return self.graph.name in cache and k in cache[self.graph.name]

		def __getitem__(self, k):
			if k not in self:
				raise KeyError(
					"{} has no unit {} in graph {}".format(
						self.character.name, k, self.graph.name
					)
				)
			return self.graph.node[k]

		@property
		def only(self):
			if len(self) != 1:
				raise AttributeError("No unit, or more than one")
			return next(iter(self.values()))


class CharacterProxy(AbstractCharacter):
	rulebook = RulebookProxyDescriptor()
	adj_cls = CharSuccessorsMappingProxy
	pred_cls = CharPredecessorsMappingProxy

	def copy_from(self, g):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		# can't handle multigraphs
		self.engine.handle(
			"character_copy_from",
			char=self.name,
			nodes=g._node,
			adj=g._adj,
			branching=True,
		)
		for node, nodeval in g.nodes.items():
			if node not in self.node:
				if nodeval and "location" in nodeval:
					self.thing._cache[node] = ThingProxy(
						self, node, nodeval["location"]
					)
				else:
					self.place._cache[node] = PlaceProxy(self, node)
		for orig in g.adj:
			for dest, edge in g.adj[orig].items():
				if orig in self.portal and dest in self.portal[orig]:
					self.portal[orig][dest]._apply_delta(edge)
				else:
					self.portal._cache[orig][dest] = PortalProxy(
						self, orig, dest
					)
					self.engine._portal_stat_cache[self.name][orig][dest] = (
						edge
					)

	def _get_default_rulebook_name(self):
		return self.name, "character"

	def _get_rulebook_proxy(self):
		return self.engine._character_rulebooks_cache[self.name]["character"]

	def _set_rulebook_proxy(self, rb):
		self.engine._character_rulebooks_cache[self.name]["character"] = (
			RuleBookProxy(self.engine, rb)
		)

	def _set_rulebook(self, rb):
		self.engine.handle(
			"set_character_rulebook",
			char=self.name,
			rulebook=rb,
			branching=True,
		)

	@cached_property
	def unit(self):
		return UnitMapProxy(self)

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

	def __init__(self, engine_proxy, charname, *, init_rulebooks=False):
		assert (
			not init_rulebooks
		), "Can't initialize rulebooks in CharacterProxy"
		self.db = engine_proxy
		self.name = charname
		self.graph = CharStatProxy(self.engine, self.name)

	def __bool__(self):
		return True

	def __eq__(self, other):
		if hasattr(other, "engine"):
			return (
				self.engine is other.engine
				and hasattr(other, "name")
				and self.name == other.name
			)
		else:
			return False

	def _apply_delta(self, delta):
		delta = delta.copy()
		for node, ex in delta.pop("nodes", {}).items():
			if ex:
				if node not in self.node:
					nodeval = delta.get("node_val", {}).get(node, None)
					if nodeval and "location" in nodeval:
						self.thing._cache[node] = prox = ThingProxy(
							self, node, nodeval["location"]
						)
						self.thing.send(prox, key=None, value=True)
					else:
						self.place._cache[node] = prox = PlaceProxy(self, node)
						self.place.send(prox, key=None, value=True)
					self.node.send(prox, key=None, value=True)
			else:
				prox = self.node[node]
				if node in self.place._cache:
					del self.place._cache[node]
					self.place.send(prox, key=None, value=False)
				elif node in self.thing._cache:
					del self.thing._cache[node]
					self.thing.send(prox, key=None, value=False)
				else:
					self.engine.warning(
						"Diff deleted {} but it was never created here".format(
							node
						)
					)
				self.node.send(prox, key=None, value=False)
		for (orig, dest), ex in delta.pop("edges", {}).items():
			if ex:
				self.engine._character_portals_cache.store(
					self.name, orig, dest, PortalProxy(self, orig, dest)
				)
				self.portal.send(self.portal[orig][dest], key=None, value=True)
			else:
				prox = self.portal[orig][dest]
				try:
					self.engine._character_portals_cache.delete(
						self.name, orig, dest
					)
				except KeyError:
					pass
				self.portal.send(prox, key=None, value=False)
		self.portal._apply_delta(delta.pop("edge_val", {}))
		nodemap = self.node
		name = self.name
		engine = self.engine
		node_stat_cache = engine._node_stat_cache
		for node, nodedelta in delta.pop("node_val", {}).items():
			if node not in nodemap or node not in node_stat_cache[name]:
				rulebook = nodedelta.pop("rulebook", None)
				node_stat_cache[name][node] = nodedelta
				if rulebook:
					nodemap[node]._set_rulebook_proxy(rulebook)
			else:
				nodemap[node]._apply_delta(nodedelta)
		portmap = self.portal
		portal_stat_cache = self.engine._portal_stat_cache
		for orig, destdelta in delta.pop("edge_val", {}).items():
			if orig in portmap:
				destmap = portmap[orig]
				for dest, portdelta in destdelta.items():
					if dest in destmap:
						destmap[dest]._apply_delta(portdelta)
			else:
				porig = portal_stat_cache[name][orig]
				for dest, portdelta in destdelta.items():
					rulebook = portdelta.pop("rulebook", None)
					porig[dest] = portdelta
					if rulebook:
						porig[dest]._set_rulebook_proxy(rulebook)
		rulebooks = delta.pop("rulebooks", None)
		if rulebooks:
			rulebooks = rulebooks.copy()
			charrb = rulebooks.pop("character", self.rulebook.name)
			if charrb != self.rulebook.name:
				self._set_rulebook_proxy(charrb)
			avrb = rulebooks.pop("unit", self.unit.rulebook.name)
			if avrb != self.unit.rulebook.name:
				self.unit._set_rulebook_proxy(avrb)
			cthrb = rulebooks.pop("thing", self.thing.rulebook.name)
			if cthrb != self.thing.rulebook.name:
				self.thing._set_rulebook_proxy(cthrb)
			cplrb = rulebooks.pop("place", self.place.rulebook.name)
			if cplrb != self.place.rulebook.name:
				self.place._set_rulebook_proxy(cplrb)
			cporb = rulebooks.pop("portal", self.portal.rulebook.name)
			if cporb != self.portal.rulebook.name:
				self.portal._set_rulebook_proxy(cporb)
		self.stat._apply_delta(delta)

	def add_place(self, name, **kwargs):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		self.engine.handle(
			command="set_place",
			char=self.name,
			place=name,
			statdict=kwargs,
			branching=True,
		)
		self.place._cache[name] = PlaceProxy(self, name)
		self.engine._node_stat_cache[self.name][name] = kwargs

	def add_places_from(self, seq):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		self.engine.handle(
			command="add_places_from",
			char=self.name,
			seq=list(seq),
			branching=True,
		)
		placecache = self.place._cache
		nodestatcache = self.engine._node_stat_cache[self.name]
		for pln in seq:
			if isinstance(pln, tuple):
				placecache[pln[0]] = PlaceProxy(self, *pln)
				if len(pln) > 1:
					nodestatcache[pln[0]] = pln[1]
			else:
				placecache[pln] = PlaceProxy(self, pln)

	def add_nodes_from(self, seq, **attrs):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		self.add_places_from(seq)

	def add_thing(self, name, location, **kwargs):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		self.engine.handle(
			command="add_thing",
			char=self.name,
			thing=name,
			loc=location,
			statdict=kwargs,
			branching=True,
		)
		self.thing._cache[name] = thing = ThingProxy(
			self, name, location, **kwargs
		)
		self.thing.send(thing, key=None, value=True)
		self.node.send(thing, key=None, value=True)

	def add_things_from(self, seq, **attrs):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		self.engine.handle(
			command="add_things_from",
			char=self.name,
			seq=list(seq),
			branching=True,
		)
		for name, location in seq:
			self.thing._cache[name] = thing = ThingProxy(self, name, location)
			self.thing.send(thing, key=None, value=True)
			self.node.send(thing, key=None, value=True)

	def remove_node(self, node):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		if node not in self.node:
			raise KeyError("No such node: {}".format(node))
		name = self.name
		self.engine.handle("del_node", char=name, node=node, branching=True)
		placecache = self.place._cache
		thingcache = self.thing._cache
		if node in placecache:
			it = placecache[node]
			it.send(it, key=None, value=False)
			self.place.send(it, key=None, value=False)
			del placecache[node]
		else:
			it = thingcache[node]
			it.send(it, key=None, value=False)
			self.thing.send(it, key=None, value=False)
			del thingcache[node]
		self.node.send(it, key=None, value=False)
		portscache = self.engine._character_portals_cache
		to_del = {(node, dest) for dest in portscache.successors[name][node]}
		to_del.update(
			(orig, node) for orig in portscache.predecessors[name][node]
		)
		for u, v in to_del:
			portscache.delete(name, u, v)
		if node in portscache.successors[name]:
			del portscache.successors[name][node]
		if node in portscache.predecessors[name]:
			del portscache.predecessors[name][node]

	def remove_place(self, place):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		placemap = self.place
		if place not in placemap:
			raise KeyError("No such place: {}".format(place))
		name = self.name
		self.engine.handle("del_node", char=name, node=place, branching=True)
		del placemap._cache[place]
		portscache = self.engine._character_portals_cache
		del portscache.successors[name][place]
		del portscache.predecessors[name][place]

	def remove_thing(self, thing):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		thingmap = self.thing
		if thing not in thingmap:
			raise KeyError("No such thing: {}".format(thing))
		name = self.name
		self.engine.handle("del_node", char=name, node=thing, branching=True)
		del thingmap._cache[thing]
		portscache = self.engine._character_portals_cache
		del portscache.successors[name][thing]
		del portscache.predecessors[name][thing]

	def add_portal(self, origin, destination, **kwargs):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		symmetrical = kwargs.pop("symmetrical", False)
		self.engine.handle(
			command="add_portal",
			char=self.name,
			orig=origin,
			dest=destination,
			symmetrical=symmetrical,
			statdict=kwargs,
			branching=True,
		)
		self.engine._character_portals_cache.store(
			self.name,
			origin,
			destination,
			PortalProxy(self, origin, destination),
		)
		if symmetrical:
			self.engine._character_portals_cache.store(
				self.name,
				destination,
				origin,
				PortalProxy(self, destination, origin),
			)
		node = self._node
		placecache = self.place._cache

		if origin not in node:
			placecache[origin] = PlaceProxy(self, origin)
		if destination not in node:
			placecache[destination] = PlaceProxy(self, destination)

	def remove_portal(self, origin, destination):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		char_port_cache = self.engine._character_portals_cache
		cache = char_port_cache.successors[self.name]
		if origin not in cache or destination not in cache[origin]:
			raise KeyError(
				"No portal from {} to {}".format(origin, destination)
			)
		self.engine.handle(
			"del_portal",
			char=self.name,
			orig=origin,
			dest=destination,
			branching=True,
		)
		char_port_cache.delete(self.name, origin, destination)

	remove_edge = remove_portal

	def add_portals_from(self, seq, symmetrical=False):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		l = list(seq)
		self.engine.handle(
			command="add_portals_from",
			char=self.name,
			seq=l,
			symmetrical=symmetrical,
			branching=True,
		)
		for origin, destination in l:
			if origin not in self.portal._cache:
				self.portal._cache[origin] = SuccessorsProxy(
					self.engine, self.name, origin
				)
			self.portal[origin]._cache[destination] = PortalProxy(
				self, origin, destination
			)

	def portals(self):
		yield from self.engine.handle(
			command="character_portals", char=self.name
		)

	def add_unit(self, graph, node=None):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		# TODO: cache
		if node is None:
			node = graph.name
			graph = graph.character.name
		self.engine.handle(
			command="add_unit",
			char=self.name,
			graph=graph,
			node=node,
			branching=True,
		)

	def remove_unit(self, graph, node=None):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		# TODO: cache
		if node is None:
			node = graph.name
			graph = graph.character.name
		self.engine.handle(
			command="remove_unit",
			char=self.name,
			graph=graph,
			node=node,
			branching=True,
		)

	def units(self):
		yield from self.engine.handle(
			command="character_units", char=self.name
		)

	def facade(self):
		return Facade(self)

	def grid_2d_8graph(self, m, n):
		self.engine.handle(
			"grid_2d_8graph",
			character=self.name,
			m=m,
			n=n,
			cb=self.engine._upd_caches,
		)

	def grid_2d_graph(self, m, n, periodic=False):
		self.engine.handle(
			"grid_2d_graph",
			character=self.name,
			m=m,
			n=n,
			periodic=periodic,
			cb=self.engine._upd_caches,
		)


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
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		self.engine.handle(
			command="set_character", char=k, data=v, branching=True
		)
		self.engine._char_cache[k] = CharacterProxy(self.engine, k)
		self.send(self, key=k, val=v)

	def __delitem__(self, k):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		self.engine.handle(command="del_character", char=k, branching=True)
		if k in self.engine._char_cache:
			del self.engine._char_cache[k]
		self.send(self, key=k, val=None)


class ProxyLanguageDescriptor(AbstractLanguageDescriptor):
	def _get_language(self, inst):
		if not hasattr(inst, "_language"):
			inst._language = inst.engine.handle(command="get_language")
		return inst._language

	def _set_language(self, inst, val):
		inst._language = val
		delta = inst.engine.handle(command="set_language", lang=val)
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
	_cache: dict

	def __init__(self, engine_proxy):
		super().__init__()
		self.engine = engine_proxy

	def load(self):
		self._cache = self.engine.handle("strings_copy")

	def __getattr__(self, k):
		try:
			return self._cache[k]
		except KeyError:
			raise AttributeError

	def __setattr__(self, k, v):
		if k in (
			"_cache",
			"engine",
			"language",
			"_language",
			"receivers",
			"_by_receiver",
			"_by_sender",
			"_weak_senders",
			"is_muted",
		):
			super().__setattr__(k, v)
			return
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		self._cache[k] = v
		self.engine.handle(command="set_string", k=k, v=v)
		self.send(self, key=k, string=v)

	def __delattr__(self, k):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		del self._cache[k]
		self.engine.handle(command="del_string", k=k)
		self.send(self, key=k, string=None)

	def lang_items(self, lang=None):
		if lang is None or lang == self.language:
			yield from self._cache.items()
		else:
			yield from self.engine.handle(
				command="get_string_lang_items", lang=lang
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
		yield from self.engine.handle(command="eternal_keys")

	def __len__(self):
		return self.engine.handle(command="eternal_len")

	def __getitem__(self, k):
		return self.engine.handle(command="get_eternal", k=k)

	def __setitem__(self, k, v):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		self._cache[k] = v
		self.engine.handle("set_eternal", k=k, v=v)

	def __delitem__(self, k):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		del self._cache[k]
		self.engine.handle(command="del_eternal", k=k)

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
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		self._cache[k] = v
		self.engine.handle("set_universal", k=k, v=v, branching=True)
		self.send(self, key=k, value=v)

	def __delitem__(self, k):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		del self._cache[k]
		self.engine.handle("del_universal", k=k, branching=True)
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
			self.engine.handle("new_empty_rulebook", rulebook=k)
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
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		self.engine.handle(command="new_empty_rule", rule=k)
		self._cache[k] = {"triggers": [], "prereqs": [], "actions": []}
		self._proxy_cache[k] = RuleProxy(self.engine, k)
		return self._proxy_cache[k]


class FuncProxy(object):
	__slots__ = "store", "func"

	def __init__(self, store, func):
		self.store = store
		self.func = func

	def __call__(self, *args, cb=None, **kwargs):
		return self.store.engine.handle(
			"call_stored_function",
			store=self.store._store,
			func=self.func,
			args=args[1:] if self.store._store == "method" else args,
			kwargs=kwargs,
			cb=partial(self.store.engine._upd_and_cb, cb=cb),
		)[0]

	def __str__(self):
		return self.store._cache[self.func]


class FuncStoreProxy(Signal):
	_cache: dict

	def __init__(self, engine_proxy, store):
		super().__init__()
		self.engine = engine_proxy
		self._store = store

	def load(self):
		self._cache = self.engine.handle("source_copy", store=self._store)
		self._cache["truth"] = "def truth(*args):\n\treturn True"

	def __getattr__(self, k):
		if k in super().__getattribute__("_cache"):
			return FuncProxy(self, k)
		else:
			raise AttributeError(k)

	def __setattr__(self, func_name, source):
		if func_name in (
			"engine",
			"_store",
			"_cache",
			"receivers",
			"_by_sender",
			"_by_receiver",
			"_weak_senders",
			"is_muted",
		):
			super().__setattr__(func_name, source)
			return
		self.engine.handle(
			command="store_source", store=self._store, v=source, name=func_name
		)
		self._cache[func_name] = source

	def __delattr__(self, func_name):
		self.engine.handle(
			command="del_source", store=self._store, k=func_name
		)
		del self._cache[func_name]

	def get_source(self, func_name):
		if func_name == "truth":
			return "def truth(*args):\n\treturn True"
		return self.engine.handle(
			command="get_source", store=self._store, name=func_name
		)


class ChangeSignatureError(TypeError):
	pass


class PortalObjCache(object):
	def __init__(self):
		self.successors = StructuredDefaultDict(2, PortalProxy)
		self.predecessors = StructuredDefaultDict(2, PortalProxy)

	def store(self, char: Key, u: Key, v: Key, obj: PortalProxy) -> None:
		self.successors[char][u][v] = obj
		self.predecessors[char][v][u] = obj

	def delete(self, char: Key, u: Key, v: Key) -> None:
		for mapp, a, b in [(self.successors, u, v), (self.predecessors, v, u)]:
			if char not in mapp:
				raise KeyError(char)
			submap = mapp[char]
			if a not in submap:
				raise KeyError((char, a))
			submap_a = submap[a]
			if b not in submap_a:
				raise KeyError((char, a, b))
			del submap_a[b]
			if not submap_a:
				del submap[a]

	def delete_char(self, char: Key) -> None:
		del self.successors[char]
		del self.predecessors[char]


class TimeSignal(Signal):
	def __init__(self, engine: "EngineProxy"):
		super().__init__()
		self.engine = engine

	def __iter__(self):
		yield self.engine.branch
		yield self.engine.tick

	def __len__(self):
		return 2

	def __getitem__(self, i):
		if i in ("branch", 0):
			return self.engine.branch
		if i in ("tick", 1):
			return self.engine.tick

	def __setitem__(self, i, v):
		if self.engine._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		if i in ("branch", 0):
			self.engine.time_travel(v, self.engine.tick)
		if i in ("tick", 1):
			self.engine.time_travel(self.engine.branch, v)


class TimeDescriptor(object):
	times = {}

	def __get__(self, inst, cls):
		if id(inst) not in self.times:
			self.times[id(inst)] = TimeSignal(inst)
		return self.times[id(inst)]

	def __set__(self, inst, val):
		if inst._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
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
			cmd="call_randomizer", method="seed", a=a, version=version
		)

	def getstate(self):
		return self._handle(cmd="call_randomizer", method="getstate")

	def setstate(self, state):
		return self._handle(
			cmd="call_randomizer", method="setstate", state=state
		)

	def _randbelow(
		self, n, int=int, maxsize=1, type=type, Method=None, BuiltinMethod=None
	):
		return self._handle(
			cmd="call_randomizer", method="_randbelow", n=n, maxsize=maxsize
		)

	def random(self):
		return self._handle(cmd="call_randomizer", method="random")


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
	def main_branch(self) -> str:
		return self.handle("main_branch")

	def snap_keyframe(self) -> dict:
		if self._worker:
			raise RuntimeError("Can't snap a keyframe in a worker process")
		return self.handle("snap_keyframe")

	def game_start(self) -> None:
		if self._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		self.handle("game_start", cb=self._upd_from_game_start)

	def _node_exists(self, char, node) -> bool:
		return self.handle("node_exists", char=char, node=node)

	def _upd_from_game_start(self, command, branch, turn, tick, result):
		(
			start_kf,
			eternal,
			functions,
			methods,
			triggers,
			prereqs,
			actions,
		) = result
		self._initialized = False
		self._eternal_cache = eternal
		self.function._cache = functions
		self.method._cache = methods
		self.trigger._cache = triggers
		self.prereq._cache = prereqs
		self.action._cache = actions
		self._replace_state_with_kf(start_kf)
		self._initialized = True

	def switch_main_branch(self, branch: str) -> None:
		if self._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		if (
			self.branch != self.main_branch
			or self.turn != 0
			or self._tick != 0
		):
			raise ValueError("Go to the start of time first")
		kf = self.handle(
			"switch_main_branch", branch=branch, cb=self._set_time
		)
		assert self.branch == branch
		self._replace_state_with_kf(kf)

	def _replace_state_with_kf(self, kf):
		self._char_stat_cache = PickyDefaultDict(UnwrappingDict)
		things = self._things_cache = {}
		places = self._character_places_cache = {}
		node_stats = self._node_stat_cache = StructuredDefaultDict(
			1, UnwrappingDict
		)

		portals = self._character_portals_cache = PortalObjCache()
		portal_stats = self._portal_stat_cache = StructuredDefaultDict(
			2, UnwrappingDict
		)
		if kf is None:
			self._char_cache = {}
			self._universal_cache = {}
			return
		self._universal_cache = kf["universal"]
		rc = self._rules_cache = {}
		for rule, triggers in kf["triggers"].items():
			if rule in rc:
				rc[rule]["triggers"] = list(triggers)
			else:
				rc[rule] = {
					"triggers": list(triggers),
					"prereqs": [],
					"actions": [],
				}
		for rule, prereqs in kf["prereqs"].items():
			if rule in rc:
				rc[rule]["prereqs"] = list(prereqs)
			else:
				rc[rule] = {
					"triggers": [],
					"prereqs": list(prereqs),
					"actions": [],
				}
		for rule, actions in kf["actions"].items():
			if rule in rc:
				rc[rule]["actions"] = list(actions)
			else:
				rc[rule] = {
					"triggers": [],
					"prereqs": [],
					"actions": list(actions),
				}
		self._char_cache = chars = {
			graph: CharacterProxy(self, graph) for (graph,) in kf["graph_val"]
		}
		for (graph,), stats in kf["graph_val"].items():
			if "character_rulebook" in stats:
				chars[graph]._set_rulebook_proxy(
					stats.pop("character_rulebook")
				)
			if "unit_rulebook" in stats:
				chars[graph].unit._set_rulebook_proxy(
					stats.pop("unit_rulebook")
				)
			if "character_thing_rulebook" in stats:
				chars[graph].thing._set_rulebook_proxy(
					stats.pop("character_thing_rulebook")
				)
			if "character_place_rulebook" in stats:
				chars[graph].place._set_rulebook_proxy(
					stats.pop("character_place_rulebook")
				)
			if "character_portal_rulebook" in stats:
				chars[graph].portal._set_rulebook_proxy(
					stats.pop("character_portal_rulebook")
				)
			if "units" in stats:
				self._character_units_cache[graph] = stats.pop("units")
			self._char_stat_cache[graph] = stats
		for (char, node), stats in kf["node_val"].items():
			if "location" in stats:
				if char not in things:
					things[char] = {}
				things[char][node] = ThingProxy(
					chars[char], node, stats.pop("location")
				)
			else:
				if char not in places:
					places[char] = {}
				places[char][node] = PlaceProxy(chars[char], node)
			node_stats[char][node] = stats
		for (char,), nodes in kf["nodes"].items():
			if char not in places:
				places[char] = charplaces = {}
				for node in nodes:
					charplaces[node] = PlaceProxy(chars[char], node)
			else:
				for node in nodes:
					if not (
						(char in things and node in things[char])
						or (char in places and node in places[char])
					):
						places[char][node] = PlaceProxy(chars[char], node)
		for (char, orig, dest), exists in kf["edges"].items():
			portals.store(
				char, orig, dest, PortalProxy(chars[char], orig, dest)
			)
		for (char, orig, dest, _), stats in kf["edge_val"].items():
			portal_stats[char][orig][dest] = stats
		self._rulebooks_cache = kf["rulebook"]

	def _pull_kf_now(self, *args, **kwargs):
		self._replace_state_with_kf(self.handle("snap_keyframe"))

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

	@property
	def tick(self):
		return self._tick

	@tick.setter
	def tick(self, v: int):
		self.time_travel(self.branch, self.turn, v)

	def __init__(
		self,
		handle_out,
		handle_in,
		logger,
		do_game_start=False,
		install_modules=(),
		submit_func=None,
		threads=None,
		prefix=None,
	):
		self.closed = False
		if submit_func:
			self._submit = submit_func
		else:
			self._threadpool = ThreadPoolExecutor(threads)
			self._submit = self._threadpool.submit
		self._handle_out = handle_out
		self._handle_out_lock = Lock()
		self._handle_in = handle_in
		self._handle_in_lock = Lock()
		self._round_trip_lock = Lock()
		self._commit_lock = Lock()
		self.logger = logger
		self.character = self.graph = CharacterMapProxy(self)
		self.eternal = EternalVarProxy(self)
		self.universal = GlobalVarProxy(self)
		self.rulebook = AllRuleBooksProxy(self)
		self.rule = AllRulesProxy(self)
		if prefix is None:
			self.method = FuncStoreProxy(self, "method")
			self.action = FuncStoreProxy(self, "action")
			self.prereq = FuncStoreProxy(self, "prereq")
			self.trigger = FuncStoreProxy(self, "trigger")
			self.function = FuncStoreProxy(self, "function")
			self._worker = False
			self.rando = RandoProxy(self)
			self.string = StringStoreProxy(self)
		else:
			self.method = FunctionStore(os.path.join(prefix, "method.py"))
			self.action = FunctionStore(os.path.join(prefix, "action.py"))
			self.prereq = FunctionStore(os.path.join(prefix, "prereq.py"))
			self.trigger = FunctionStore(os.path.join(prefix, "trigger.py"))
			self.function = FunctionStore(os.path.join(prefix, "function.py"))
			self.string = StringStore(self, prefix)
			self._worker = True

		self._node_stat_cache = StructuredDefaultDict(1, UnwrappingDict)
		self._portal_stat_cache = StructuredDefaultDict(2, UnwrappingDict)
		self._char_stat_cache = PickyDefaultDict(UnwrappingDict)
		self._things_cache = StructuredDefaultDict(1, ThingProxy)
		self._character_places_cache = StructuredDefaultDict(1, PlaceProxy)
		self._character_rulebooks_cache = StructuredDefaultDict(
			1,
			RuleBookProxy,
			kwargs_munger=lambda inst, k: {
				"engine": self,
				"bookname": (inst.key, k),
			},
		)
		self._char_node_rulebooks_cache = StructuredDefaultDict(
			1,
			RuleBookProxy,
			kwargs_munger=lambda inst, k: {
				"engine": self,
				"bookname": (inst.key, k),
			},
		)
		self._char_port_rulebooks_cache = StructuredDefaultDict(
			2,
			RuleBookProxy,
			kwargs_munger=lambda inst, k: {
				"engine": self,
				"bookname": (inst.parent.key, inst.key, k),
			},
		)
		self._character_portals_cache = PortalObjCache()
		self._character_units_cache = PickyDefaultDict(dict)
		self._unit_characters_cache = PickyDefaultDict(dict)
		self._rule_obj_cache = {}
		self._rulebook_obj_cache = {}
		self._char_cache = {}
		if prefix is None:
			self.send_bytes(self.pack({"command": "get_btt"}))
			received = self.unpack(self.recv_bytes())
			self._branch, self._turn, self._tick = received[-1]
			self.send_bytes(self.pack({"command": "branches"}))
			self._branches = self.unpack(self.recv_bytes())[-1]
			self.method.load()
			self.action.load()
			self.prereq.load()
			self.trigger.load()
			self.function.load()
			self.string.load()
			self._eternal_cache = self.handle("eternal_copy")
			self._initialized = False
			self._pull_kf_now()
			self._initialized = True
			for module in install_modules:
				self.handle("install_module", module=module)
			if do_game_start:
				self.handle("do_game_start", cb=self._upd_caches)

	def __getattr__(self, item):
		meth = super().__getattribute__("method").__getattr__(item)
		return MethodType(meth, self)

	def _reimport_triggers(self):
		self.trigger.reimport()

	def _eval_trigger(self, name, entity):
		return getattr(self.trigger, name)(entity)

	def _call_function(self, name: str, *args, **kwargs):
		return getattr(self.function, name)(*args, **kwargs)

	def _reimport_functions(self):
		self.function.reimport()

	def _call_method(self, name: str, *args, **kwargs):
		return MethodType(getattr(self.method, name), self)(*args, **kwargs)

	def _reimport_methods(self):
		self.method.reimport()

	def send_bytes(self, obj, blocking=True, timeout=-1):
		compressed = zlib.compress(obj)
		self._handle_out_lock.acquire(blocking, timeout)
		self._handle_out.send_bytes(compressed)
		self._handle_out_lock.release()

	def recv_bytes(self, blocking=True, timeout=-1):
		self._handle_in_lock.acquire(blocking, timeout)
		data = self._handle_in.recv_bytes()
		self._handle_in_lock.release()
		return zlib.decompress(data)

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

	def handle(self, cmd=None, *, cb: Optional[callable] = None, **kwargs):
		"""Send a command to the LiSE core.

		The only positional argument should be the name of a
		method in :class:``EngineHandle``. All keyword arguments
		will be passed to it, with the exceptions of
		``cb``, ``branching``, and ``silent``.

		With ``branching=True``, handle paradoxes by creating new
		branches of history. I will switch to the new branch if needed.
		If I have an attribute ``branching_cb``, I'll call it if and
		only if the branch changes upon completing a command with
		``branching=True``.

		With a function ``cb``, I will call ``cb`` when I get
		a result.
		``cb`` will be called with keyword arguments ``command``,
		the same command you asked for; ``result``, the value returned
		by it, possibly ``None``; and the present ``branch``,
		``turn``, and ``tick``, possibly different than when you called
		``handle``.`.

		"""
		if self._worker:
			return
		if self.closed:
			raise RedundantProcessError(f"Already closed: {id(self)}")
		if "command" in kwargs:
			cmd = kwargs["command"]
		elif cmd:
			kwargs["command"] = cmd
		else:
			raise TypeError("No command")
		assert not kwargs.get("silent")
		self.debug(f"EngineProxy: sending {cmd}")
		start_ts = monotonic()
		with self._round_trip_lock:
			self.send_bytes(self.pack(kwargs))
			received = self.recv_bytes()
		command, branch, turn, tick, r = self.unpack(received)
		self.debug(
			"EngineProxy: received {} in {:,.2f} seconds".format(
				(command, branch, turn, tick), monotonic() - start_ts
			)
		)
		if (branch, turn, tick) != self._btt():
			self._branch = branch
			self._turn = turn
			self._tick = tick
			self.time.send(self, branch=branch, turn=turn, tick=tick)
		if isinstance(r, Exception):
			raise r
		if cmd != command:
			raise RuntimeError(
				f"Sent command {cmd}, but received results for {command}"
			)
		if cb:
			cb(command=command, branch=branch, turn=turn, tick=tick, result=r)
		return r

	def _unpack_recv(self):
		ret = self.unpack(self.recv_bytes())
		return ret

	def _callback(self, cb):
		command, branch, turn, tick, res = self.unpack(self.recv_bytes())
		self.debug(
			"EngineProxy: received, with callback {}: {}".format(
				cb, (command, branch, turn, tick, res)
			)
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
				"{} raised by command {}, trying to run callback {} with it".format(
					repr(ex), command, cb
				)
			)
		cb(command=command, branch=branch, turn=turn, tick=tick, result=res)
		return command, branch, turn, tick, res

	def _branching(self, cb=None):
		command, branch, turn, tick, r = self.unpack(self.recv_bytes())
		self.debug(
			"EngineProxy: received, with branching, {}".format(
				(command, branch, turn, tick, r)
			)
		)
		if (branch, turn, tick) != (self._branch, self._turn, self._tick):
			self._branch = branch
			self._turn = turn
			self._tick = tick
			self.time.send(self, branch=branch, turn=turn, tick=tick)
			if hasattr(self, "branching_cb"):
				self.branching_cb(
					command=command,
					branch=branch,
					turn=turn,
					tick=tick,
					result=r,
				)
		if cb:
			cb(command=command, branch=branch, turn=turn, tick=tick, result=r)
		return command, branch, turn, tick, r

	def _call_with_recv(self, *cbs, **kwargs):
		cmd, branch, turn, tick, received = self.unpack(self.recv_bytes())
		self.debug(
			"EngineProxy: received {}".format(
				(cmd, branch, turn, tick, received)
			)
		)
		if isinstance(received, Exception):
			raise received
		for cb in cbs:
			cb(
				command=cmd,
				branch=branch,
				turn=turn,
				tick=tick,
				result=received,
				**kwargs,
			)
		return received

	def _upd_caches(self, command, branch, turn, tick, result):
		result, deltas = result
		self.eternal._update_cache(deltas.pop("eternal", {}))
		self.universal._update_cache(deltas.pop("universal", {}))
		# I think if you travel back to before a rule was created
		# it'll show up empty.
		# That's ok I guess
		for rule, delta in deltas.pop("rules", {}).items():
			if rule in self._rules_cache:
				self._rules_cache[rule].update(delta)
			else:
				delta.setdefault("triggers", [])
				delta.setdefault("prereqs", [])
				delta.setdefault("actions", [])
				self._rules_cache[rule] = delta
			if rule not in self._rule_obj_cache:
				self._rule_obj_cache[rule] = RuleProxy(self, rule)
			ruleproxy = self._rule_obj_cache[rule]
			ruleproxy.send(ruleproxy, **delta)
		rulebookdeltas = deltas.pop("rulebooks", {})
		self._rulebooks_cache.update(rulebookdeltas)
		for rulebook, delta in rulebookdeltas.items():
			if rulebook not in self._rulebook_obj_cache:
				self._rulebook_obj_cache[rulebook] = RuleBookProxy(
					self, rulebook
				)
			rulebookproxy = self._rulebook_obj_cache[rulebook]
			# the "delta" is just the rules list, for now
			rulebookproxy.send(rulebookproxy, rules=delta)
		to_delete = set()
		for char, chardelta in deltas.items():
			if chardelta is None:
				to_delete.add(char)
				continue
			if char not in self._char_cache:
				self._char_cache[char] = CharacterProxy(self, char)
			chara = self.character[char]
			chara._apply_delta(chardelta)
		for char in to_delete & self._char_cache.keys():
			del self._char_cache[char]

	def _btt(self):
		return self._branch, self._turn, self._tick

	def _set_time(self, command, branch, turn, tick, result, **kwargs):
		self._branch = branch
		self._turn = turn
		self._tick = tick
		parent, turn_from, tick_from, turn_to, tick_to = self._branches.get(
			branch, (None, turn, tick, turn, tick)
		)
		if branch not in self._branches or (turn, tick) > (turn_to, tick_to):
			self._branches[branch] = parent, turn_from, tick_from, turn, tick
		self.time.send(self, branch=branch, turn=turn, tick=tick)

	def branches(self) -> set:
		return self._branches

	def branch_start(self, branch: str) -> Tuple[int, int]:
		return self._branches[branch][1]

	def branch_end(self, branch: str) -> Tuple[int, int]:
		return self._branches[branch][2]

	def branch_parent(self, branch: str) -> Optional[str]:
		return self._branches[branch][0]

	def apply_choices(self, choices, dry_run=False, perfectionist=False):
		if self._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		return self.handle(
			"apply_choices",
			choices=choices,
			dry_run=dry_run,
			perfectionist=perfectionist,
		)

	def _upd(self, *args, **kwargs):
		self._upd_caches(*args, **kwargs)
		self._set_time(*args, no_del=True, **kwargs)

	def _upd_and_cb(self, cb, *args, **kwargs):
		self._upd(*args, **kwargs)
		if cb:
			cb(*args, **kwargs)

	# TODO: make this into a Signal, like it is in the LiSE core
	def next_turn(self, cb=None):
		if self._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		if cb and not callable(cb):
			raise TypeError("Uncallable callback")
		return self.handle("next_turn", cb=partial(self._upd_and_cb, cb))

	def time_travel(self, branch, turn, tick=None, cb=None):
		"""Move to a different point in the timestream

		Needs ``branch`` and ``turn`` arguments. The ``tick`` is
		optional; if unspecified, you'll travel to the last tick
		in the turn.

		May take a callback function ``cb``, which will receive a
		dictionary describing changes to the characters in ``chars``.
		``chars`` defaults to 'all', indicating that every character
		should be included, but may be a list of character names
		to include.

		"""
		if self._worker:
			raise RuntimeError(
				"Tried to change the world state in a worker process"
			)
		if cb is not None and not callable(cb):
			raise TypeError("Uncallable callback")
		return self.handle(
			"time_travel",
			branch=branch,
			turn=turn,
			tick=tick,
			cb=partial(self._upd_and_cb, cb),
		)

	def add_character(self, char, data=None, **attr):
		if char in self._char_cache:
			raise KeyError("Character already exists")
		assert (
			char not in self._char_stat_cache
		), "Tried to create a character when there was already stat data for it"
		if data is None:
			data = {}
		if not isinstance(data, dict):
			# it's a networkx graph
			data = {
				"place": {
					k: v for k, v in data._node.items() if "location" not in v
				},
				"thing": {
					k: v for k, v in data._node.items() if "location" in v
				},
				"edge": data._adj,
			}
		self._char_cache[char] = character = CharacterProxy(self, char)
		self._char_stat_cache[char] = attr
		placedata = data.get("place", data.get("node", {}))
		for place, stats in placedata.items():
			assert (
				char not in self._character_places_cache
				or place not in self._character_places_cache[char]
			), "Tried to create a character when there was already place data for it"
			assert (
				char not in self._node_stat_cache
				or place not in self._node_stat_cache[char]
			), "Tried to create a character when there was already node-value data for it"
			if char in self._character_places_cache:
				self._character_places_cache[char][place] = PlaceProxy(
					character, place
				)
			else:
				self._character_places_cache[char] = {
					place: PlaceProxy(character, place)
				}
			self._node_stat_cache[char][place] = stats
		thingdata = data.get("thing", {})
		for thing, stats in thingdata.items():
			assert (
				char not in self._things_cache
				or thing not in self._things_cache[char]
			), "Tried to create a character when there was already thing data for it"
			assert (
				char not in self._node_stat_cache
				or thing not in self._node_stat_cache[char]
			), "Tried to create a character when there was already node-stat data for it"
			if "location" not in stats:
				raise ValueError("Things must always have locations")
			loc = stats.pop("location")
			if char in self._things_cache:
				self._things_cache[char][thing] = ThingProxy(
					character, thing, loc
				)
			else:
				self._things_cache[char] = {
					thing: ThingProxy(character, thing, loc)
				}
			if char in self._node_stat_cache:
				self._node_stat_cache[char][thing] = stats
			else:
				self._node_stat_cache[char] = {thing: stats}
		portdata = data.get("edge", data.get("portal", data.get("adj", {})))
		for orig, dests in portdata.items():
			assert (
				char not in self._character_portals_cache.successors
				or orig not in self._character_portals_cache.successors[char]
			), "Tried to create a character when there was already successor data for it"
			assert (
				char not in self._portal_stat_cache
				or orig not in self._portal_stat_cache[char]
			), "Tried to create a character when there was already portal-stat data for it"
			for dest, stats in dests.items():
				assert (
					(
						char not in self._character_portals_cache.successors
						or dest
						not in self._character_portals_cache.successors[char][
							orig
						]
					)
					and (
						char not in self._portal_stat_cache
						or orig not in self._portal_stat_cache[char]
						or dest not in self._portal_stat_cache[char][orig]
					)
				), "Tried to create a character when there was already portal data for it"
				self._character_portals_cache.store(
					char,
					orig,
					dest,
					PortalProxy(character, orig, dest),
				)
				self._portal_stat_cache[char][orig][dest] = stats
		self.handle(
			command="add_character",
			char=char,
			data=data,
			attr=attr,
			branching=True,
		)

	def new_character(self, char, **attr):
		self.add_character(char, **attr)
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
		self.handle(command="del_character", char=char, branching=True)

	del_graph = del_character

	def del_node(self, char, node):
		if char not in self._char_cache:
			raise KeyError("No such character")
		if (
			node not in self._character_places_cache[char]
			and node not in self._things_cache[char]
		):
			raise KeyError("No such node")
		if node in self._things_cache[char]:
			del self._things_cache[char][node]
		if node in self._character_places_cache[char]:  # just to be safe
			del self._character_places_cache[char][node]
		self.handle(command="del_node", char=char, node=node, branching=True)

	def del_portal(self, char, orig, dest):
		if char not in self._char_cache:
			raise KeyError("No such character")
		self._character_portals_cache.delete(char, orig, dest)
		self.handle(
			command="del_portal",
			char=char,
			orig=orig,
			dest=dest,
			branching=True,
		)

	def commit(self):
		self._commit_lock.acquire()
		self.handle("commit", cb=self._release_commit_lock)

	def _release_commit_lock(self, *, command, branch, turn, tick, result):
		self._commit_lock.release()

	def close(self):
		self._commit_lock.acquire()
		self._commit_lock.release()
		self.handle("close")
		with self._handle_out_lock:
			self._handle_out.send_bytes(b"shutdown")
		self.closed = True

	def _node_contents(self, character, node):
		# very slow. do better
		for thing in self.character[character].thing.values():
			if thing["location"] == node:
				yield thing.name


def engine_subprocess(args, kwargs, input_pipe, output_pipe, logq, loglevel):
	"""Loop to handle one command at a time and pipe results back"""
	from .handle import EngineHandle

	engine_handle = EngineHandle(*args, logq=logq, loglevel=loglevel, **kwargs)
	compress = zlib.compress
	decompress = zlib.decompress
	pack = engine_handle.pack

	while True:
		inst = input_pipe.recv_bytes()
		if inst == b"shutdown":
			input_pipe.close()
			output_pipe.close()
			if logq:
				logq.close()
			return 0
		instruction = engine_handle.unpack(decompress(inst))
		if isinstance(instruction, dict) and "__use_msgspec__" in instruction:
			import msgspec.msgpack

			instruction = msgspec.msgpack.decode(instruction["__real__"])
		silent = instruction.pop("silent", False)
		cmd = instruction.pop("command")

		branching = instruction.pop("branching", False)
		try:
			if branching:
				try:
					r = getattr(engine_handle, cmd)(**instruction)
				except OutOfTimelineError:
					engine_handle.increment_branch()
					r = getattr(engine_handle, cmd)(**instruction)
			else:
				r = getattr(engine_handle, cmd)(**instruction)
		except AssertionError:
			raise
		except Exception as e:
			output_pipe.send_bytes(
				compress(
					engine_handle.pack(
						(
							cmd,
							engine_handle._real.branch,
							engine_handle._real.turn,
							engine_handle._real.tick,
							e,
						)
					)
				)
			)
			continue
		if silent:
			continue
		resp = msgpack.Packer().pack_array_header(5)
		resp += (
			pack(cmd)
			+ pack(engine_handle._real.branch)
			+ pack(engine_handle._real.turn)
			+ pack(engine_handle._real.tick)
		)
		if hasattr(getattr(engine_handle, cmd), "prepacked"):
			if isinstance(r, dict):
				resp += msgpack.Packer().pack_map_header(len(r))
				for k, v in r.items():
					resp += k + v
			elif isinstance(r, tuple):
				pacr = msgpack.Packer()
				pacr.pack_ext_type(
					MsgpackExtensionType.tuple.value,
					msgpack.Packer().pack_array_header(len(r)) + b"".join(r),
				)
				resp += pacr.bytes()
			elif isinstance(r, list):
				resp += msgpack.Packer().pack_array_header(len(r)) + b"".join(
					r
				)
			else:
				resp += r
		else:
			resp += pack(r)
		output_pipe.send_bytes(compress(resp))
		if hasattr(engine_handle, "_after_ret"):
			engine_handle._after_ret()
			del engine_handle._after_ret


class WorkerLogger:
	def __init__(self, logq):
		self._logq = logq

	def debug(self, msg):
		if not self._logq:
			print(msg)
		self._logq.put((10, msg))

	def info(self, msg):
		if not self._logq:
			print(msg)
		self._logq.put((20, msg))

	def warning(self, msg):
		if not self._logq:
			print(msg)
		self._logq.put((30, msg))

	def error(self, msg):
		if not self._logq:
			print(msg)
		self._logq.put((40, msg))

	def critical(self, msg):
		if not self._logq:
			print(msg)
		self._logq.put((50, msg))


def worker_subprocess(prefix: str, in_pipe: Pipe, out_pipe: Pipe, logq: Queue):
	eng = EngineProxy(None, None, WorkerLogger(logq), prefix=prefix)
	pack = eng.pack
	unpack = eng.unpack
	compress = zlib.compress
	decompress = zlib.decompress
	eng._branches = eng.unpack(zlib.decompress(in_pipe.recv_bytes()))
	eng._initialized = False
	while True:
		inst = in_pipe.recv_bytes()
		if inst == b"shutdown":
			in_pipe.close()
			if logq:
				logq.close()
			out_pipe.send_bytes(b"done")
			out_pipe.close()
			return 0
		(uid, method, args, kwargs) = unpack(decompress(inst))
		try:
			ret = getattr(eng, method)(*args, **kwargs)
		except Exception as ex:
			ret = ex
			if uid < 0:
				import traceback

				traceback.print_exc(file=sys.stderr)
				raise
		if uid >= 0:
			out_pipe.send_bytes(compress(pack((uid, ret))))
		eng._initialized = True


class RedundantProcessError(ProcessError):
	"""Asked to start a process that has already started"""


class EngineProcessManager(object):
	def __init__(self, *args, **kwargs):
		self._args = args
		self._kwargs = kwargs

	def start(self, *args, **kwargs):
		"""Start LiSE in a subprocess, and return a proxy to it"""
		if hasattr(self, "engine_proxy"):
			raise RedundantProcessError("Already started")
		(handle_out_pipe_recv, self._handle_out_pipe_send) = Pipe(duplex=False)
		(handle_in_pipe_recv, handle_in_pipe_send) = Pipe(duplex=False)
		self.logq = Queue()
		handlers = []
		logl = {
			"debug": logging.DEBUG,
			"info": logging.INFO,
			"warning": logging.WARNING,
			"error": logging.ERROR,
			"critical": logging.CRITICAL,
		}
		loglevel = logging.INFO
		if "loglevel" in kwargs:
			if kwargs["loglevel"] in logl:
				loglevel = logl[kwargs["loglevel"]]
			else:
				loglevel = kwargs["loglevel"]
			del kwargs["loglevel"]
		if "logger" in kwargs:
			self.logger = kwargs["logger"]
			del kwargs["logger"]
		else:
			self.logger = logging.getLogger(__name__)
			stdout = logging.StreamHandler(sys.stdout)
			stdout.set_name("stdout")
			handlers.append(stdout)
			handlers[0].setLevel(loglevel)
		if "logfile" in kwargs:
			try:
				fh = logging.FileHandler(kwargs["logfile"])
				handlers.append(fh)
				handlers[-1].setLevel(loglevel)
			except OSError:
				pass
			del kwargs["logfile"]
		do_game_start = (
			kwargs.pop("do_game_start") if "do_game_start" in kwargs else False
		)
		install_modules = (
			kwargs.pop("install_modules")
			if "install_modules" in kwargs
			else []
		)
		formatter = logging.Formatter(
			fmt="[{levelname}] LiSE.proxy({process}) t{message}", style="{"
		)
		for handler in handlers:
			handler.setFormatter(formatter)
			self.logger.addHandler(handler)
		self._p = Process(
			name="LiSE Life Simulator Engine (core)",
			target=engine_subprocess,
			args=(
				args or self._args,
				kwargs or self._kwargs,
				handle_out_pipe_recv,
				handle_in_pipe_send,
				self.logq,
				loglevel,
			),
		)
		self._p.start()
		self._logthread = Thread(
			target=self.sync_log_forever, name="log", daemon=True
		)
		self._logthread.start()
		self.engine_proxy = EngineProxy(
			self._handle_out_pipe_send,
			handle_in_pipe_recv,
			self.logger,
			do_game_start,
			install_modules,
		)
		return self.engine_proxy

	def sync_log(self, limit=None, block=True):
		"""Get log messages from the subprocess, and log them in this one"""
		n = 0
		while limit is None or n < limit:
			try:
				(level, message) = self.logq.get(block=block)
				if isinstance(level, int):
					level = {
						10: "debug",
						20: "info",
						30: "warning",
						40: "error",
						50: "critical",
					}[level]
				getattr(self.logger, level)(message)
				n += 1
			except Empty:
				return

	def sync_log_forever(self):
		"""Continually call ``sync_log``, for use in a subthread"""
		while True:
			self.sync_log(1)

	def shutdown(self):
		"""Close the engine in the subprocess, then join the subprocess"""
		self.engine_proxy.close()
		self._p.join()
		del self.engine_proxy

	def __enter__(self):
		return self.start()

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.shutdown()
