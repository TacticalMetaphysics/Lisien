# This file is part of Lisien, a framework for life simulation games.
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
"""Proxy objects to access lisien entities from another process.

Each proxy class is meant to emulate the equivalent lisien class,
and any change you make to a proxy will be made in the corresponding
entity in the lisien core.

To use these, first instantiate an ``EngineProcessManager``, then
call its ``start`` method with the same arguments you'd give a real
``Engine``. You'll get an ``EngineProxy``, which acts like the underlying
``Engine`` for most purposes.

"""

from __future__ import annotations


import ast
import io
import logging
import os
import sys
import zlib
from abc import ABC, abstractmethod
from collections.abc import Mapping, MutableMapping, MutableSequence
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property, partial
from multiprocessing import Pipe, Process, ProcessError, Queue
from queue import Empty
from random import Random
from threading import Lock, Thread
from time import monotonic
from types import MethodType
from typing import Hashable, Iterator

import astunparse
import msgpack
import networkx as nx
from blinker import Signal

from ..cache import PickyDefaultDict, StructuredDefaultDict
from ..exc import OutOfTimelineError, WorkerProcessReadOnlyError
from ..facade import CharacterFacade
from ..node import NodeContent, Place, Thing, UserMapping
from ..portal import Portal
from ..typing import Key, DeltaDict
from ..util import (
	AbstractCharacter,
	AbstractEngine,
	KeyClass,
	MsgpackExtensionType,
	TimeSignalDescriptor,
	getatt,
	repr_call_sig,
)
from ..wrap import DictWrapper, ListWrapper, SetWrapper, UnwrappingDict
from ..xcollections import (
	AbstractLanguageDescriptor,
	FunctionStore,
	StringStore,
)


class CachingProxy(MutableMapping, Signal):
	"""Abstract class for proxies to lisien entities or mappings thereof"""

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

	def setdefault(self, k, default=None):
		if k not in self:
			if default is None:
				raise KeyError("No such key", k)
			self[k] = default
			return default
		return self[k]

	def __setitem__(self, k, v):
		if self.engine._worker:
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		self._set_item(k, v)
		self._cache[k] = self._cache_set_munge(k, v)
		self.send(self, key=k, value=v)

	def __delitem__(self, k):
		if self.engine._worker:
			raise WorkerProcessReadOnlyError(
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
	def _set_item(self, k, v):
		raise NotImplementedError("Abstract method")

	@abstractmethod
	def _del_item(self, k):
		raise NotImplementedError("Abstract method")


class CachingEntityProxy(CachingProxy):
	"""Abstract class for proxy objects representing lisien entities"""

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


class RuleMapProxy(MutableMapping, Signal):
	@property
	def _cache(self):
		return self.engine._rulebooks_cache.setdefault(self.name, ([], 0.0))[0]

	@property
	def priority(self):
		return self.engine._rulebooks_cache.setdefault(self.name, ([], 0.0))[1]

	def __init__(self, engine, rulebook_name):
		super().__init__()
		self.engine = engine
		self.name = rulebook_name
		self._proxy_cache = engine._rule_obj_cache

	def __iter__(self):
		return iter(self._cache)

	def __len__(self):
		return len(self._cache)

	def __getitem__(self, key):
		if key in self._cache:
			if key not in self._proxy_cache:
				self._proxy_cache[key] = RuleProxy(self.engine, key)
			return self._proxy_cache[key]
		raise KeyError("Rule not assigned to rulebook", key, self.name)

	def __setitem__(self, k, v):
		if self.engine._worker:
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		if isinstance(v, RuleProxy):
			v = v._name
		else:
			RuleProxy(self.engine, k).actions = v
			v = k
		if k in self._cache:
			return
		i = len(self._cache)
		self._cache.append(k)
		self.engine.handle(
			command="set_rulebook_rule",
			rulebook=self.name,
			i=i,
			rule=v,
			branching=True,
		)
		self.send(self, key=k, val=v)

	def __delitem__(self, key):
		if self.engine._worker:
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		i = self._cache.index(key)
		if i is None:
			raise KeyError("Rule not set in rulebook", key, self.name)
		del self._cache[i]
		self.engine.handle(
			command="del_rulebook_rule",
			rulebook=self.name,
			i=i,
			branching=True,
		)
		self.send(self, key=key, val=None)


class RuleFollowerProxyDescriptor:
	def __set__(self, inst, val):
		if inst.engine._worker:
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		if isinstance(val, RuleBookProxy):
			rb = val
			val = val.name
		elif isinstance(val, RuleMapProxy):
			if val.name in inst.engine._rulebooks_cache:
				rb = inst.engine._rulebooks_cache[val.name]
				val = val.name
			else:
				rb = inst.engine._rulebooks_cache[val.name] = RuleBookProxy(
					inst.engine, val.name
				)
				val = val.name
		inst._set_rulebook(val)
		inst.send(inst, rulebook=val)


class RuleMapProxyDescriptor(RuleFollowerProxyDescriptor):
	def __get__(self, instance, owner):
		if instance is None:
			return self
		if hasattr(instance, "_rule_map_proxy"):
			return instance._rule_map_proxy
		elif instance._get_rulebook_name() in instance.engine._rulebooks_cache:
			proxy = RuleMapProxy(
				instance.engine, instance._get_rulebook_name()
			)
			instance._rule_map_proxy = proxy
		else:
			proxy = RuleMapProxy(
				instance.engine, instance._get_default_rulebook_name()
			)
			instance._rule_map_proxy = proxy
		return proxy


class RulebookProxyDescriptor(RuleFollowerProxyDescriptor):
	"""Descriptor that makes the corresponding RuleBookProxy if needed"""

	def __get__(self, inst, cls):
		if inst is None:
			return self
		return inst._get_rulebook_proxy()


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


class RuleFollowerProxy(ABC):
	rule = RuleMapProxyDescriptor()
	rulebook = RulebookProxyDescriptor()
	engine: "EngineProxy"

	@abstractmethod
	def _get_default_rulebook_name(self) -> tuple:
		pass

	@abstractmethod
	def _get_rulebook_name(self) -> Key:
		pass

	def _get_rulebook_proxy(self) -> "RuleBookProxy":
		try:
			name = self._get_rulebook_name()
		except KeyError:
			name = self._get_default_rulebook_name()
		if name not in self.engine._rulebook_obj_cache:
			self.engine._rulebook_obj_cache[name] = RuleBookProxy(
				self.engine, name
			)
		return self.engine._rulebook_obj_cache[name]

	@abstractmethod
	def _set_rulebook_name(self, rb: Key) -> None:
		pass


class NodeProxy(CachingEntityProxy, RuleFollowerProxy):
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

	def _get_rulebook_name(self):
		return self.engine._char_node_rulebooks_cache[self._charname][
			self.name
		]

	def _set_rulebook_name(self, rb):
		self.engine.handle(
			"set_node_rulebook",
			char=self._charname,
			node=self.name,
			rulebook=rb,
			branching=True,
		)
		self.engine._char_node_rulebooks_cache[self._charname][self.name] = rb

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
			raise WorkerProcessReadOnlyError(
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
		self, dest: Key | NodeProxy, weight: Key = None
	) -> list[Key]:
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
					self.engine._char_node_rulebooks_cache[self._charname][
						self.name
					] = v
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
					self.engine._char_node_rulebooks_cache[self._charname][
						self.name
					] = v
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
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
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


class PortalProxy(CachingEntityProxy, RuleFollowerProxy):
	def _apply_delta(self, delta):
		for k, v in delta.items():
			if k == "rulebook":
				if v != self.rulebook.name:
					self.engine._char_port_rulebooks_cache[self._charname][
						self._origin
					][self._destination] = v
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

	def _get_rulebook_name(self):
		return self.engine._char_port_rulebooks_cache[self._charname][
			self._origin
		][self._destination]

	def _set_rulebook_name(self, rb):
		self.engine.handle(
			command="set_portal_rulebook",
			char=self._charname,
			orig=self._origin,
			dest=self._destination,
			rulebook=rb,
		)
		self.engine._char_port_rulebooks_cache[self._charname][self._origin][
			self._destination
		] = rb

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
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		self.engine.del_portal(self._charname, self._origin, self._destination)


Portal.register(PortalProxy)


class NodeMapProxy(MutableMapping, Signal, RuleFollowerProxy):
	def _get_default_rulebook_name(self):
		return self._charname, "character_node"

	def _get_rulebook_name(self):
		return self.engine._character_rulebooks_cache[self._charname]["node"]

	def _set_rulebook_name(self, rb: Key):
		self.engine.handle(
			"set_character_node_rulebook",
			char=self._charname,
			rulebook=rb,
			branching=True,
		)
		self.engine._character_rulebooks_cache[self._charname]["node"] = rb

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
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		self.character.place[k] = v

	def __delitem__(self, k):
		if self.engine._worker:
			raise WorkerProcessReadOnlyError(
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

		The patch is sent to the lisien core all at once, so this is faster than
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


class ThingMapProxy(CachingProxy, RuleFollowerProxy):
	def _get_default_rulebook_name(self):
		return self.name, "character_thing"

	def _get_rulebook_name(self) -> Key:
		return self.engine._character_rulebooks_cache[self.name]["thing"]

	def _set_rulebook_name(self, rb: Key) -> None:
		self.engine.handle(
			"set_character_thing_rulebook",
			char=self.name,
			rulebook=rb,
			branching=True,
		)
		self.engine._character_rulebooks_cache[self.name]["thing"] = rb

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
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		places = d.keys() & self.character.place.keys()
		if places:
			raise KeyError(f"Tried to patch places on thing mapping: {places}")
		self.character.node.patch(d)


class PlaceMapProxy(CachingProxy, RuleFollowerProxy):
	def _get_default_rulebook_name(self):
		return self.name, "character_place"

	def _get_rulebook_name(self) -> Key:
		return self.engine._character_rulebooks_cache[self.name]["place"]

	def _set_rulebook_name(self, rb: Key) -> None:
		self.engine.handle(
			"set_character_place_rulebook",
			char=self.name,
			rulebook=rb,
			branching=True,
		)
		self.engine._character_rulebooks_cache[self.name]["place"] = rb

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
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		things = d.keys() & self.character.thing.keys()
		if things:
			raise KeyError(f"Tried to patch things on place mapping: {things}")
		self.character.node.patch(d)


class SuccessorsProxy(CachingProxy):
	@property
	def _cache(self):
		succ = self.engine._character_portals_cache.successors
		if self._charname not in succ:
			raise KeyError("No portals in this character")
		succc = succ[self._charname]
		if self._orig not in succc:
			raise KeyError(
				"No successors to this portal", self._charname, self._orig
			)
		return succc[self._orig]

	def _set_rulebook_name(self, k):
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


class CharSuccessorsMappingProxy(CachingProxy, RuleFollowerProxy):
	def _get_default_rulebook_name(self):
		return self.name, "character_portal"

	def _get_rulebook_name(self) -> Key:
		return self.engine._character_rulebooks_cache[self.name]["portal"]

	def _set_rulebook_name(self, rb: Key) -> None:
		self.engine.handle(
			"set_character_portal_rulebook",
			char=self.character.name,
			rulebook=rb,
			branching=True,
		)
		self.engine._character_rulebooks_cache[self.name]["portal"] = rb

	@property
	def character(self):
		return self.engine.character[self.name]

	@property
	def _cache(self):
		return self.engine._character_portals_cache.successors.setdefault(
			self.name, {}
		)

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
		if k not in self:
			raise KeyError("No successors to this node", self.name, k)
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
			raise WorkerProcessReadOnlyError(
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


class CharPredecessorsMappingProxy(MutableMapping, Signal):
	def __init__(self, engine_proxy, charname):
		super().__init__()
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
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
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

	def _set_rulebook_name(self, k):
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
			assert k != "rulebook"
			if v is None:
				if k in self._cache:
					del self._cache[k]
					self.send(self, key=k, value=None)
			elif k not in self._cache or self._cache[k] != v:
				self._cache[k] = v
				self.send(self, key=k, value=v)


class FuncListProxy(MutableSequence, Signal):
	def __init__(self, rule_proxy: "RuleProxy", key: str):
		super().__init__()
		self.rule = rule_proxy
		self._key = key

	def __iter__(self):
		return iter(self.rule._cache.get(self._key, ()))

	def __len__(self):
		return len(self.rule._cache.get(self._key, ()))

	def __getitem__(self, item):
		if self._key not in self.rule._cache:
			raise IndexError(item)
		return self.rule._cache[self._key][item]

	def _handle_send(self):
		self.rule.engine.handle(
			f"set_rule_{self._key}",
			**{
				"rule": self.rule.name,
				"branching": True,
				self._key: self.rule._nominate(self.rule._cache[self._key]),
			},
		)

	def __setitem__(self, key, value):
		if isinstance(value, str):
			value = getattr(getattr(self.rule.engine, self._key), value)
		self.rule._cache[self._key] = value
		self._handle_send()

	def __delitem__(self, key):
		if self._key not in self.rule._cache:
			raise IndexError(key)
		del self.rule._cache[self._key][key]
		self._handle_send()

	def insert(self, index, value):
		if isinstance(value, str):
			value = getattr(getattr(self.rule.engine, self._key), value)
		self.rule._cache.insert(index, value)
		self._handle_send()


class FuncListProxyDescriptor:
	def __init__(self, key):
		self._key = key

	def __get__(self, instance, owner):
		attname = f"_{self._key}_proxy"
		if not hasattr(instance, attname):
			setattr(instance, attname, FuncListProxy(instance, self._key))
		return getattr(instance, attname)

	def __set__(self, instance, value):
		to_set = []
		for v in value:
			if isinstance(v, FuncProxy):
				to_set.append(v)
			elif not isinstance(v, str):
				raise TypeError(f"Need FuncListProxy or str, got {type(v)}")
			else:
				to_set.append(
					getattr(
						getattr(instance.engine, self._key.removesuffix("s")),
						v,
					)
				)
		instance._cache[self._key] = to_set
		self.__get__(instance, None)._handle_send()


class RuleProxy(Signal):
	triggers = FuncListProxyDescriptor("triggers")
	prereqs = FuncListProxyDescriptor("prereqs")
	actions = FuncListProxyDescriptor("actions")

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
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
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


class UnitMapProxy(Mapping, RuleFollowerProxy):
	engine = getatt("character.engine")

	def _get_default_rulebook_name(self):
		return self.character.name, "unit"

	def _get_rulebook_name(self) -> Key:
		return self.engine._character_rulebooks_cache[self.character.name][
			"unit"
		]

	def _set_rulebook_name(self, rb: Key) -> None:
		self.engine.handle(
			"set_unit_rulebook",
			char=self.character.name,
			rulebook=rb,
			branching=True,
		)
		self.engine._character_rulebooks_cache[self.character.name]["unit"] = (
			rb
		)

	@property
	def only(self):
		if len(self) == 0:
			raise AttributeError("No units")
		elif len(self) > 1:
			raise AttributeError("Units in more than one graph")
		return next(iter(self.values()))

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


class CharacterProxy(AbstractCharacter, RuleFollowerProxy):
	adj_cls = CharSuccessorsMappingProxy
	pred_cls = CharPredecessorsMappingProxy

	def copy_from(self, g):
		if self.engine._worker:
			raise WorkerProcessReadOnlyError(
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

	def _get_rulebook_name(self) -> Key:
		return self.engine._character_rulebooks_cache[self.name]["character"]

	def _set_rulebook_name(self, rb: Key) -> None:
		self.engine.handle(
			"set_character_rulebook",
			char=self.name,
			rulebook=rb,
			branching=True,
		)
		self.engine._character_rulebooks_cache[self.name]["character"] = rb

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
		assert not init_rulebooks, (
			"Can't initialize rulebooks in CharacterProxy"
		)
		self.db = engine_proxy
		self.name = charname
		self.graph = CharStatProxy(self.engine, self.name)

	def __repr__(self):
		return f"{self.db}.character[{self.name}]"

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
			elif node in self.node:
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
		for orig, dests in delta.pop("edges", {}).items():
			for dest, ex in dests.items():
				if ex:
					self.engine._character_portals_cache.store(
						self.name, orig, dest, PortalProxy(self, orig, dest)
					)
					self.portal.send(
						self.portal[orig][dest], key=None, value=True
					)
				elif orig in self.portal and dest in self.portal[orig]:
					prox = self.portal[orig][dest]
					try:
						self.engine._character_portals_cache.delete(
							self.name, orig, dest
						)
						assert dest not in self.portal[orig]
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
					nodemap[node]._set_rulebook_name(rulebook)
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
						self.engine._char_port_rulebooks_cache[name][orig][
							dest
						] = rulebook
		rulebooks = delta.pop("rulebooks", None)
		if rulebooks:
			ruc = self.engine._character_rulebooks_cache[name]
			if (
				"character" in rulebooks
				and rulebooks["character"] != self.rulebook.name
			):
				ruc["character"] = rulebooks["character"]
			if (
				"unit" in rulebooks
				and rulebooks["unit"] != self.unit.rulebook.name
			):
				ruc["unit"] = rulebooks["unit"]
			if (
				"thing" in rulebooks
				and rulebooks["thing"] != self.thing.rulebook.name
			):
				ruc["thing"] = rulebooks["thing"]
			if (
				"place" in rulebooks
				and rulebooks["place"] != self.place.rulebook.name
			):
				ruc["place"] = rulebooks["place"]
			if (
				"portal" in rulebooks
				and rulebooks["portal"] != self.portal.rulebook.name
			):
				ruc["portal"] = rulebooks["portal"]
		self.stat._apply_delta(delta)

	def add_place(self, name, **kwargs):
		if self.engine._worker:
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		self.add_places_from(seq)

	def add_thing(self, name, location, **kwargs):
		if self.engine._worker:
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
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
		return CharacterFacade(self)

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
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		self.engine.handle(
			command="set_character", char=k, data=v, branching=True
		)
		self.engine._char_cache[k] = CharacterProxy(self.engine, k)
		self.send(self, key=k, val=v)

	def __delitem__(self, k):
		if self.engine._worker:
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		self._cache[k] = v
		self.engine.handle(command="set_string", k=k, v=v)
		self.send(self, key=k, string=v)

	def __delattr__(self, k):
		if self.engine._worker:
			raise WorkerProcessReadOnlyError(
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
		return iter(self._cache)

	def __len__(self):
		return len(self._cache)

	def __getitem__(self, k):
		return self._cache[k]

	def __setitem__(self, k, v):
		if self.engine._worker:
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		self._cache[k] = v
		self.engine.handle("set_eternal", k=k, v=v)

	def __delitem__(self, k):
		if self.engine._worker:
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		self._cache[k] = v
		self.engine.handle("set_universal", k=k, v=v, branching=True)
		self.send(self, key=k, value=v)

	def __delitem__(self, k):
		if self.engine._worker:
			raise WorkerProcessReadOnlyError(
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
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		self.engine.handle(command="new_empty_rule", rule=k)
		self._cache[k] = {"triggers": [], "prereqs": [], "actions": []}
		self._proxy_cache[k] = RuleProxy(self.engine, k)
		return self._proxy_cache[k]


class FuncProxy(object):
	__slots__ = "store", "name"

	def __init__(self, store, func):
		self.store = store
		self.name = func

	def __call__(self, *args, cb=None, **kwargs):
		return self.store.engine.handle(
			"call_stored_function",
			store=self.store._store,
			func=self.name,
			args=args[1:] if self.store._store == "method" else args,
			kwargs=kwargs,
			cb=partial(self.store.engine._upd_and_cb, cb=cb),
		)[0]

	def __str__(self):
		return self.store._cache[self.name]


class FuncStoreProxy(Signal):
	_cache: dict

	def __init__(self, engine_proxy, store):
		super().__init__()
		self.engine = engine_proxy
		self._store = store
		self._proxy_cache = {}

	def load(self):
		self._cache = self.engine.handle("source_copy", store=self._store)

	def __getattr__(self, k):
		if k in super().__getattribute__("_cache"):
			proxcache = super().__getattribute__("_proxy_cache")
			if k not in proxcache:
				proxcache[k] = FuncProxy(self, k)
			return proxcache[k]
		else:
			raise AttributeError(k)

	def __setattr__(self, func_name, source):
		if func_name in (
			"engine",
			"_store",
			"_cache",
			"_proxy_cache",
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


class PortalObjCache:
	def __init__(self):
		self.successors = {}
		self.predecessors = {}

	def store(self, char: Key, u: Key, v: Key, obj: PortalProxy) -> None:
		succ = self.successors
		if char in succ:
			char_us = succ[char]
			if u in char_us:
				char_us[u][v] = obj
			else:
				char_us[u] = {v: obj}
		else:
			succ[char] = {u: {v: obj}}
		pred = self.predecessors
		if char in pred:
			char_vs = pred[char]
			if v in char_vs:
				char_vs[v][u] = obj
			else:
				char_vs[v] = {u: obj}
		else:
			pred[char] = {v: {u: obj}}

	def delete(self, char: Key, u: Key, v: Key) -> None:
		succ = self.successors
		if char in succ:
			succ_us = succ[char]
			if u in succ_us:
				del succ_us[u][v]
			if not succ_us:
				del succ[char]
		pred = self.predecessors
		if char in pred:
			pred_vs = pred[char]
			if v in pred_vs:
				del pred_vs[v][u]
			if not pred_vs:
				del pred[char]

	def delete_char(self, char: Key) -> None:
		if char in self.successors:
			del self.successors[char]
			del self.predecessors[char]


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


class NextTurnProxy(Signal):
	def __init__(self, engine: "EngineProxy"):
		super().__init__()
		self.engine = engine

	def __call__(self) -> tuple[list, DeltaDict]:
		return self.engine.handle(
			"next_turn",
			cb=partial(self.engine._upd_and_cb, partial(self.send, self)),
		)


class EngineProxy(AbstractEngine):
	"""An engine-like object for controlling a lisien process

	Don't instantiate this directly. Use :class:`EngineProcessManager` instead.
	The ``start`` method will return an :class:`EngineProxy` instance.

	"""

	char_cls = CharacterProxy
	thing_cls = ThingProxy
	place_cls = PlaceProxy
	portal_cls = PortalProxy
	time = TimeSignalDescriptor()
	is_proxy = True

	def _set_btt(self, branch: str, turn: int, tick: int, cb=None):
		return self.handle(
			"time_travel",
			branch=branch,
			turn=turn,
			tick=tick,
			cb=partial(self._upd_and_cb, cb=cb),
		)

	def _start_branch(
		self, parent: str, branch: str, turn: int, tick: int
	) -> None:
		self.handle(
			"start_branch", parent=parent, branch=branch, turn=turn, tick=tick
		)

	def _extend_branch(self, branch: str, turn: int, tick: int) -> None:
		self.handle("extend_branch", branch=branch, turn=turn, tick=tick)

	def load_at(self, branch: str, turn: int, tick: int) -> None:
		self.handle("load_at", branch=branch, turn=turn, tick=tick)

	def turn_end(self, branch: str = None, turn: int = None) -> int:
		if self._worker:
			raise NotImplementedError("Need to cache turn ends in workers")
		return self.handle("turn_end", branch=branch, turn=turn)

	def turn_end_plan(self, branch: str = None, turn: int = None) -> int:
		if self._worker:
			raise NotImplementedError("Need to cache plans in workers")
		return self.handle("turn_end_plan", branch=branch, turn=turn)

	@property
	def main_branch(self) -> str:
		return self.handle("main_branch")

	def snap_keyframe(self) -> dict:
		if self._worker:
			raise WorkerProcessReadOnlyError(
				"Can't snap a keyframe in a worker process"
			)
		return self.handle("snap_keyframe")

	def game_init(self) -> None:
		if self._worker:
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		self.handle("game_init", cb=self._upd_from_game_start)

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
		self.function.reimport()
		self.method._cache = methods
		self.method.reimport()
		self.trigger._cache = triggers
		self.trigger.reimport()
		self.prereq._cache = prereqs
		self.prereq.reimport()
		self.action._cache = actions
		self.action.reimport()
		for func, mod in [
			(functions, self.function),
			(methods, self.method),
			(triggers, self.trigger),
			(prereqs, self.prereq),
			(actions, self.action),
		]:
			unimported = set(func).difference(dir(mod._module))
			if unimported:
				self.warning(
					f"some functions not imported from {mod._filename}: {unimported}."
				)
			self.debug(
				f"imported functions from {mod._filename}: {dir(mod._module)}"
			)
		self._replace_state_with_kf(start_kf)
		self._branch = branch
		self._turn = turn
		self._tick = tick
		self._initialized = True

	def switch_main_branch(self, branch: str) -> None:
		if self._worker:
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		if (
			self.branch != self.main_branch
			or self.turn != 0
			or self._tick != 0
		):
			raise ValueError("Go to the start of time first")
		kf = self.handle(
			"switch_main_branch", branch=branch, cb=self._upd_time
		)
		assert self.branch == branch
		self._replace_state_with_kf(kf)

	def _replace_state_with_kf(self, result, **kwargs):
		things = self._things_cache
		places = self._character_places_cache

		portals = self._character_portals_cache
		if result is None:
			self._char_cache = {}
			self._universal_cache = {}
			return
		self._universal_cache = result["universal"]
		rc = self._rules_cache = {}
		for rule, triggers in result["triggers"].items():
			triglist = []
			if isinstance(self.trigger, FuncStoreProxy):
				for func in triggers:
					if not hasattr(self.trigger, func):
						self.trigger._proxy_cache[func] = FuncProxy(
							self.trigger, func
						)
					triglist.append(getattr(self.trigger, func))
			else:
				for func in triggers:
					if not hasattr(self.trigger, func):
						self.warning(
							f"didn't find {func} in trigger file {self.trigger._filename}"
						)
					triglist.append(getattr(self.trigger, func))
			if rule in rc:
				rc[rule]["triggers"] = triglist
			else:
				rc[rule] = {
					"triggers": triglist,
					"prereqs": [],
					"actions": [],
				}
		for rule, prereqs in result["prereqs"].items():
			preqlist = []
			if isinstance(self.prereq, FuncStoreProxy):
				for func in prereqs:
					if not hasattr(self.prereq, func):
						self.prereq._proxy_cache[func] = FuncProxy(
							self.prereq, func
						)
					preqlist.append(getattr(self.prereq, func))
			else:
				for func in prereqs:
					if not hasattr(self.prereq, func):
						self.warning(
							f"didn't find {func} in prereq file {self.trigger._filename}"
						)
					preqlist.append(getattr(self.prereq, func))
			if rule in rc:
				rc[rule]["prereqs"] = preqlist
			else:
				rc[rule] = {
					"triggers": [],
					"prereqs": preqlist,
					"actions": [],
				}
		for rule, actions in result["actions"].items():
			actlist = []
			if isinstance(self.action, FuncStoreProxy):
				for func in actions:
					if not hasattr(self.action, func):
						self.action._proxy_cache[func] = FuncProxy(
							self.action, func
						)
					actlist.append(getattr(self.action, func))
			else:
				for func in actions:
					if not hasattr(self.action, func):
						self.warning(
							f"didn't find {func} in action file {self.action._filename}"
						)
					actlist.append(getattr(self.action, func))
			if rule in rc:
				rc[rule]["actions"] = actlist
			else:
				rc[rule] = {
					"triggers": [],
					"prereqs": [],
					"actions": actlist,
				}
		self._rulebooks_cache = result["rulebook"]
		self._char_cache = chars = {
			graph: CharacterProxy(self, graph)
			if (graph not in self._char_cache)
			else self._char_cache[graph]
			for graph in (
				result["graph_val"].keys()
				| result["nodes"].keys()
				| result["node_val"].keys()
				| result["edges"].keys()
				| result["edge_val"].keys()
				| self._char_cache.keys()
			)
		}
		for graph, stats in result["graph_val"].items():
			if "character_rulebook" in stats:
				self._character_rulebooks_cache[graph]["character"] = (
					stats.pop("character_rulebook")
				)
			if "unit_rulebook" in stats:
				self._character_rulebooks_cache[graph]["unit"] = stats.pop(
					"unit_rulebook"
				)
			if "character_thing_rulebook" in stats:
				self._character_rulebooks_cache[graph]["thing"] = stats.pop(
					"character_thing_rulebook"
				)
			if "character_place_rulebook" in stats:
				self._character_rulebooks_cache[graph]["place"] = stats.pop(
					"character_place_rulebook"
				)
			if "character_portal_rulebook" in stats:
				self._character_rulebooks_cache[graph]["portal"] = stats.pop(
					"character_portal_rulebook"
				)
			if "units" in stats:
				self._character_units_cache[graph] = stats.pop("units")
			else:
				self._character_units_cache[graph] = {}
			self._char_stat_cache[graph] = stats
		nodes_to_delete = {
			(char, node)
			for char in things.keys() | places.keys()
			for node in things.get(char, {}).keys()
			| places.get(char, {}).keys()
		}
		for char, nodes in result["nodes"].items():
			for node, ex in nodes.items():
				if ex:
					if not (
						(char in things and node in things[char])
						or (char in places and node in places[char])
					):
						places[char][node] = PlaceProxy(chars[char], node)
					nodes_to_delete.discard((char, node))
				else:
					if char in things and node in things[char]:
						del things[char][node]
					if char in places and node in places[char]:
						del places[char][node]
		for char, node in nodes_to_delete:
			if node in things[char]:
				del things[char][node]
			else:
				del places[char][node]
		for char, nodestats in result["node_val"].items():
			for node, stats in nodestats.items():
				if "location" in stats:
					if char not in things or node not in things[char]:
						things[char][node] = ThingProxy(
							chars[char], node, stats.pop("location")
						)
					else:
						things[char][node]._location = stats.pop("location")
					if char in places and node in places[char]:
						del places[char][node]
				else:
					if char not in places or node not in places[char]:
						places[char][node] = PlaceProxy(chars[char], node)
					if char in things and node in things[char]:
						del things[char][node]
				self._node_stat_cache[char][node] = stats
		edges_to_delete = {
			(char, orig, dest)
			for char in portals.successors
			for orig in portals.successors[char]
			for dest in portals.successors[char][orig]
		}
		for char, origs in result["edges"].items():
			for orig, dests in origs.items():
				for dest, exists in dests.items():
					if (
						char in portals.successors
						and orig in portals.successors[char]
						and dest in portals.successors[char][orig]
					):
						if exists:
							edges_to_delete.discard((char, orig, dest))
						else:
							del portals.successors[char][orig][dest]
							del portals.predecessors[char][dest][orig]
						continue
					if exists:
						edges_to_delete.discard((char, orig, dest))
					else:
						continue
					that = PortalProxy(chars[char], orig, dest)
					portals.store(char, orig, dest, that)
		for char, orig, dest in edges_to_delete:
			portals.delete(char, orig, dest)
		for char, origs in result["edge_val"].items():
			for orig, dests in origs.items():
				for dest, stats in dests.items():
					self._portal_stat_cache[char][orig][dest] = stats

	def _pull_kf_now(self, *args, **kwargs):
		self._replace_state_with_kf(self.handle("snap_keyframe"))

	@property
	def branch(self):
		return self._branch

	@branch.setter
	def branch(self, v):
		self._set_btt(v, self.turn, self.tick)

	@property
	def turn(self):
		return self._turn

	@turn.setter
	def turn(self, v):
		self._set_btt(self.branch, v, self.tick)

	@property
	def tick(self):
		return self._tick

	@tick.setter
	def tick(self, v: int):
		self._set_btt(self.branch, self.turn, v)

	def _btt(self):
		return self._branch, self._turn, self._tick

	def __init__(
		self,
		handle_out,
		handle_in,
		logger,
		install_modules=(),
		submit_func: callable = None,
		threads: int = None,
		prefix: str = None,
		i: int = None,
		replay_file: str | os.PathLike | io.TextIOBase = None,
		eternal: dict = None,
		branches: dict = None,
	):
		if eternal is None:
			eternal = {"language": "eng"}
		if branches is None:
			branches = {"trunk": (None, 0, 0, 0, 0)}
		self._eternal_cache = eternal
		self._branches_d = branches
		replay_txt = None
		if replay_file is not None:
			if not isinstance(replay_file, io.TextIOBase):
				if os.path.exists(replay_file):
					with open(replay_file, "rt") as rf:
						replay_txt = rf.read().replace(
							"<lisien.proxy.EngineProxy>", "eng"
						)
				else:
					replay_file = open(replay_file, "wt")
			elif "w" in replay_file.mode or "a" in replay_file.mode:
				self._replay_file = replay_file
			elif "r" in replay_file.mode:
				replay_txt = replay_file.read().replace(
					"<lisien.proxy.EngineProxy>", "eng"
				)
		self.i = i
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
			self.next_turn = NextTurnProxy(self)
			self.method = FuncStoreProxy(self, "method")
			self.action = FuncStoreProxy(self, "action")
			self.prereq = FuncStoreProxy(self, "prereq")
			self.trigger = FuncStoreProxy(self, "trigger")
			self.function = FuncStoreProxy(self, "function")
			self._worker = False
			self._rando = RandoProxy(self)
			self.string = StringStoreProxy(self)
		else:

			def next_turn():
				raise WorkerProcessReadOnlyError(
					"Can't advance time in a worker process"
				)

			self.next_turn = next_turn
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
			KeyClass,
			kwargs_munger=lambda inst, k: {
				"engine": self,
				"bookname": (inst.key, k),
			},
		)
		self._char_node_rulebooks_cache = StructuredDefaultDict(
			1,
			KeyClass,
			kwargs_munger=lambda inst, k: {
				"engine": self,
				"bookname": (inst.key, k),
			},
		)
		self._char_port_rulebooks_cache = StructuredDefaultDict(
			2,
			KeyClass,
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
			self._branches_d = self.unpack(self.recv_bytes())[-1]
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
			if replay_txt is not None:
				replay = ast.parse(replay_txt)
				for expr in replay.body:
					if isinstance(expr.value, ast.Call):
						method = expr.value.func.id
						args = []
						kwargs = {}
						for arg in expr.value.args:
							if isinstance(arg.value, ast.Subscript):
								whatmap = arg.value.value.attr
								key = arg.value.slice.value
								args.append(getattr(self, whatmap)[key])
							elif hasattr(arg.value, "value"):
								args.append(arg.value.value)
							else:
								args.append(astunparse.unparse(arg.value))
						for kw in expr.value.keywords:
							if isinstance(kw.value, ast.Subscript):
								whatmap = kw.value.value.attr
								key = kw.value.slice.value
								kwargs[kw.arg] = getattr(self, whatmap)[key]
							else:
								if hasattr(kw.value, "value"):
									kwargs[kw.arg] = kw.value.value
								else:
									kwargs[kw.arg] = astunparse.unparse(
										kw.value
									)
						self.handle(method, *args, **kwargs)

	def __repr__(self):
		return "<lisien.proxy.EngineProxy>"

	def __getattr__(self, item):
		meth = super().__getattribute__("method").__getattr__(item)
		return MethodType(meth, self)

	def _reimport_code(self):
		self.function.reimport()
		self.method.reimport()
		self.trigger.reimport()
		self.prereq.reimport()
		self.action.reimport()

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

	def send_bytes(self, obj, blocking=True, timeout=1):
		compressed = zlib.compress(obj)
		self._handle_out_lock.acquire(blocking, timeout)
		self._handle_out.send_bytes(compressed)
		self._handle_out_lock.release()

	def recv_bytes(self, blocking=True, timeout=1):
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

	def handle(self, cmd=None, *, cb: callable = None, **kwargs):
		"""Send a command to the lisien core.

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
		if hasattr(self, "_replay_file"):
			self._replay_file.write(repr_call_sig(cmd, **kwargs) + "\n")
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

	def _upd_time(self, command, branch, turn, tick, result, **kwargs):
		then = self._btt()
		self._branch = branch
		self._turn = turn
		self._tick = tick
		if branch not in self._branches_d:
			self._branches_d[branch] = (None, turn, tick, turn, tick)
		else:
			parent, turn_from, tick_from, turn_to, tick_to = self._branches_d[
				branch
			]
			if (turn, tick) > (turn_to, tick_to):
				self._branches_d[branch] = (
					parent,
					turn_from,
					tick_from,
					turn,
					tick,
				)
		self.time.send(self, then=then, now=(branch, turn, tick))

	def apply_choices(self, choices, dry_run=False, perfectionist=False):
		if self._worker:
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		return self.handle(
			"apply_choices",
			choices=choices,
			dry_run=dry_run,
			perfectionist=perfectionist,
		)

	def _reimport_all(self):
		for store in (
			self.function,
			self.method,
			self.prereq,
			self.trigger,
			self.action,
		):
			if hasattr(store, "reimport"):
				store.reimport()

	def _upd(self, *args, **kwargs):
		self._upd_caches(*args, **kwargs)
		self._reimport_all()
		self._upd_time(*args, no_del=True, **kwargs)

	def _upd_and_cb(self, cb, *args, **kwargs):
		self._upd(*args, **kwargs)
		if cb:
			cb(*args, **kwargs)

	def _add_character(
		self, char, data: tuple | dict | nx.Graph = None, **attr
	):
		if char in self._char_cache:
			raise KeyError("Character already exists")
		if data is None:
			data = {}
		if isinstance(data, nx.Graph):
			data = {
				"place": {
					k: v for k, v in data.nodes.items() if "location" not in v
				},
				"thing": {
					k: v for k, v in data.nodes.items() if "location" in v
				},
				"edge": {
					orig: {dest: edges for (dest, edges) in dests.items()}
					for (orig, dests) in data.edges.items()
				},
			}
		elif isinstance(data, tuple):
			nodes, edges, graph_val = data
			data = graph_val.copy()
			data["place"] = {
				k: v for k, v in nodes.items() if "location" not in v
			}
			data["thing"] = {k: v for k, v in nodes.items() if "location" in v}
			data["edge"] = edges
		elif not isinstance(data, dict):
			raise TypeError(
				f"Can't make a character out of {type(data)}", data
			)
		self._char_cache[char] = character = CharacterProxy(self, char)
		self._char_stat_cache[char] = attr
		placedata = data.get("place", data.get("node", {}))
		for place, stats in placedata.items():
			if (
				char not in self._character_places_cache
				or place not in self._character_places_cache[char]
			):
				self._character_places_cache[char][place] = PlaceProxy(
					character, place
				)
			self._node_stat_cache[char][place] = stats
		thingdata = data.get("thing", {})
		for thing, stats in thingdata.items():
			if "location" not in stats:
				raise ValueError("Things must always have locations")
			loc = stats.pop("location")
			if (
				char not in self._things_cache
				or thing not in self._things_cache[char]
			):
				self._things_cache[char][thing] = ThingProxy(
					character, thing, loc
				)
			self._node_stat_cache[char][thing] = stats
		portdata = data.get("edge", data.get("portal", data.get("adj", {})))
		for orig, dests in portdata.items():
			for dest, stats in dests.items():
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
			**attr,
			branching=True,
		)

	def add_character(self, char, data=None, **attr):
		if self._worker:
			raise WorkerProcessReadOnlyError(
				"Tried to change world state in a worker process"
			)
		self._add_character(char, data, **attr)

	def new_character(self, char, **attr):
		self.add_character(char, **attr)
		return self._char_cache[char]

	new_graph = new_character

	def _del_character(self, char):
		if char not in self._char_cache:
			raise KeyError("No such character")
		del self._char_cache[char]
		if char in self._char_stat_cache:
			del self._char_stat_cache[char]
		if char in self._character_places_cache:
			del self._character_places_cache[char]
		if char in self._things_cache:
			del self._things_cache[char]
		self._character_portals_cache.delete_char(char)
		self.handle(command="del_character", char=char, branching=True)

	def del_character(self, char):
		if self._worker:
			raise WorkerProcessReadOnlyError(
				"tried to change world state in a worker process"
			)
		self._del_character(char)

	del_graph = del_character

	def del_node(self, char, node):
		if char not in self._char_cache:
			raise KeyError("No such character")
		if (
			node not in self._character_places_cache[char]
			and node not in self._things_cache[char]
		):
			raise KeyError("No such node")
		successors = list(self._character_portals_cache.successors[char][node])
		predecessors = list(
			self._character_portals_cache.predecessors[char][node]
		)
		cont = list(self._node_contents(char, node))
		if node in self._things_cache[char]:
			del self._things_cache[char][node]
		for contained in cont:
			del self._things_cache[char][contained]
		if node in self._character_places_cache[char]:  # just to be safe
			del self._character_places_cache[char][node]
		for succ in successors:
			self._character_portals_cache.delete(char, node, succ)
		for pred in predecessors:
			self._character_portals_cache.delete(char, pred, node)
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
	def __init__(self, logq, i):
		self._logq = logq
		self._i = i

	def debug(self, msg):
		if not self._logq:
			print(msg)
		self._logq.put((10, f"worker {self._i}: {msg}"))

	def info(self, msg):
		if not self._logq:
			print(msg)
		self._logq.put((20, f"worker {self._i}: {msg}"))

	def warning(self, msg):
		if not self._logq:
			print(msg)
		self._logq.put((30, f"worker {self._i}: {msg}"))

	def error(self, msg):
		if not self._logq:
			print(msg)
		self._logq.put((40, f"worker {self._i}: {msg}"))

	def critical(self, msg):
		if not self._logq:
			print(msg)
		self._logq.put((50, f"worker {self._i}: {msg}"))


def worker_subprocess(
	i: int,
	prefix: str,
	branches: dict,
	eternal: dict,
	in_pipe: Pipe,
	out_pipe: Pipe,
	logq: Queue,
):
	from pickle import loads

	from ..util import repr_call_sig

	logger = WorkerLogger(logq, i)
	eng = EngineProxy(
		None,
		None,
		logger,
		prefix=prefix,
		i=i,
		eternal=eternal,
		branches=branches,
	)
	pack = eng.pack
	unpack = eng.unpack
	compress = zlib.compress
	decompress = zlib.decompress
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
		logger.debug(repr_call_sig(method, *args, **kwargs))
		if isinstance(method, str):
			method = getattr(eng, method)
		elif isinstance(method, bytes):
			method = loads(method)
		try:
			ret = method(*args, **kwargs)
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


class EngineProcessManager:
	"""Container for a Lisien proxy and a logger for it

	Make sure the :class:`EngineProcessManager` instance lasts as long as the
	:class:`lisien.proxy.EngineProxy` returned from its :method:`start`
	method. Call the :method:`EngineProcessManager.shutdown` method
	when you're done with the :class:`lisien.proxy.EngineProxy`. That way,
	we can join the thread that listens to the subprocess's logs.

	"""

	loglevel = logging.DEBUG

	def __init__(self, *args, **kwargs):
		self._args = args
		self._kwargs = kwargs

	def start(self, *args, **kwargs):
		"""Start lisien in a subprocess, and return a proxy to it"""
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
		loglevel = self.loglevel
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
		replay_file = kwargs.pop("replay_file", "") or None
		install_modules = (
			kwargs.pop("install_modules")
			if "install_modules" in kwargs
			else []
		)
		formatter = logging.Formatter(
			fmt="[{levelname}] lisien.proxy({process}) t{message}", style="{"
		)
		for handler in handlers:
			handler.setFormatter(formatter)
			self.logger.addHandler(handler)
		self._p = Process(
			name="lisien Life Simulator Engine (core)",
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
			install_modules,
			replay_file=replay_file,
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

	def __enter__(self):
		return self.start()

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.shutdown()
