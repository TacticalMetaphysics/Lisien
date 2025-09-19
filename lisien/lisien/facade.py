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
import os
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from functools import cached_property
from operator import attrgetter
from threading import RLock
from typing import Any, Mapping, MutableMapping, MutableSequence, Type

import networkx as nx
from blinker import Signal

from .cache import Cache, TurnEndDict, TurnEndPlanDict, UnitnessCache
from .collections import CompositeDict, FunctionStore
from .exc import NotInKeyframeError, TotalKeyError
from .types import CharName, DiGraph, Edge, Key, Node, NodeName
from .util import (
	AbstractCharacter,
	AbstractEngine,
	AbstractThing,
	SignalDict,
	TimeSignalDescriptor,
	getatt,
	print_call_sig,
	timer,
)
from .wrap import MutableMappingUnwrapper


class FacadeEntity(MutableMapping, Signal, ABC):
	exists = True
	character = getatt("graph")

	@property
	def rulebook(self):
		if "rulebook" in self._patch:
			return self._patch["rulebook"]
		return self._real.rulebook

	@rulebook.setter
	def rulebook(self, rbname):
		self._patch["rulebook"] = rbname

	@abstractmethod
	def _get_real(self, name):
		raise NotImplementedError()

	def __init__(self, mapping, real_or_name=None, **kwargs):
		self.facade = self.graph = getattr(mapping, "facade", mapping)
		self._mapping = mapping
		is_name = not hasattr(real_or_name, "name") and not hasattr(
			real_or_name, "orig"
		)
		if is_name:
			try:
				self._real = self._get_real(real_or_name)
			except (KeyError, AttributeError):
				pass  # Entity created for Facade. No underlying real entity.
		else:
			self._real = real_or_name
		self._patch = {
			k: v.unwrap() if hasattr(v, "unwrap") else v
			for (k, v) in kwargs.items()
		}

	def __contains__(self, item):
		patch = self._patch
		if item in patch:
			return patch[item] is not ...
		if hasattr(self, "_real"):
			return item in self._real
		return False

	def __iter__(self):
		patch = self._patch
		ks = patch.keys()
		if hasattr(self, "_real"):
			ks |= self._real.keys()
		for k in ks:
			if k not in patch or patch[k] is not ...:
				yield k

	def __len__(self):
		n = 0
		for _ in self:
			n += 1
		return n

	def __getitem__(self, k):
		if k in self._patch:
			if self._patch[k] is ...:
				raise KeyError("{} has been masked.".format(k))
			return self._patch[k]
		if not hasattr(self, "_real"):
			raise KeyError(f"{k} unset, and no underlying Thing")
		ret = self._real[k]
		if hasattr(ret, "unwrap"):  # a wrapped mutable object from the
			# allegedb.wrap module
			ret = ret.unwrap()
			self._patch[k] = ret  # changes will be reflected in the
		# facade but not the original
		return ret

	@abstractmethod
	def _set_plan(self, k, v):
		raise NotImplementedError()

	def __setitem__(self, k, v):
		if k == "name":
			raise KeyError("Can't change names")
		if hasattr(v, "unwrap"):
			v = v.unwrap()
		if self.character.engine._planning:
			return self._set_plan(k, v)
		self._patch[k] = v

	def __delitem__(self, k):
		self._patch[k] = ...

	def apply(self):
		self._real.update(self._patch)
		self._patch = {}

	def unwrap(self):
		return {
			k: v.unwrap() if hasattr(v, "unwrap") else v
			for (k, v) in self.items()
		}


getname = attrgetter("name")


class FacadeEntityMapping(MutableMappingUnwrapper, Signal, ABC):
	"""Mapping that contains entities in a Facade.

	All the entities are of the same type, ``cls``, possibly
	being distorted views of entities of the type ``innercls``.

	"""

	cls: Type[FacadeEntity]

	@abstractmethod
	def _get_inner_map(self):
		raise NotImplementedError("Missing _get_inner_map")

	def _make(self, k, v):
		if isinstance(v, dict):
			for badkey in ("character", "engine", "name"):
				if badkey in v:
					del v[badkey]
			return self.cls(self, k, **v)
		return self.cls(self, v)

	engine = getatt("facade.engine")

	def __init__(self, facade, _=None):
		"""Store the facade."""
		super().__init__()
		self.facade = facade
		self._patch = {}

	def __contains__(self, k):
		if k in self._patch:
			return self._patch[k] is not ...
		return k in self._get_inner_map()

	def __iter__(self):
		seen = set()
		for k in self._patch:
			if k not in seen and self._patch[k] is not ...:
				yield k
			seen.add(k)
		for k in self._get_inner_map():
			if k not in seen:
				yield k

	def __len__(self):
		n = 0
		for k in self:
			n += 1
		return n

	def __getitem__(self, k):
		if k not in self and not self.engine._mockup:
			raise KeyError(k)
		if k not in self._patch:
			inner = self._get_inner_map()
			if k in inner:
				v = inner[k]
			else:
				v = {}
			self._patch[k] = self._make(k, v)
		ret = self._patch[k]
		if ret is ...:
			raise KeyError(k)
		if type(ret) is not self.cls:
			ret = self._patch[k] = self._make(k, ret)
		return ret

	def __setitem__(self, k, v):
		if not isinstance(v, self.cls):
			v = self._make(k, v)
		self._patch[k] = v
		if self is not self.facade.node:
			self.facade.node.send(self, key=k, value=v)

	def __delitem__(self, k):
		if k not in self:
			raise KeyError("{} not present".format(k))
		that = self[k]
		# Units don't work when we're wrapping an EngineProxy or nothing.
		# I'll fix that at some point I guess.
		# 2025-02-06
		if hasattr(self.facade.engine._real, "_unitness_cache") and hasattr(
			that, "users"
		):
			user: CharacterFacade
			for user in list(that.users()):
				user.remove_unit(self.facade.name, k)
		self._patch[k] = ...


class FacadeRulebook(MutableSequence, ABC):
	name: Any
	_fake: list
	engine: "EngineFacade"

	def __iter__(self):
		return iter(self._fake)

	def __getitem__(self, item):
		_ = self._fake[item]
		return FacadeRule(self.engine, item)

	def __setitem__(self, index, value):
		name = getattr(value, "name", value)
		self._fake[index] = name

	def __delitem__(self, index):
		del self._fake[index]

	def __len__(self):
		return len(self._fake)


class FacadeRule:
	class FakeFuncList(MutableSequence):
		def __init__(self, rule, typ):
			self._rule = rule
			self._type = typ

		@property
		def _me(self):
			return getattr(self._rule, f"_fake_{self._type}s")

		def __iter__(self):
			return iter(self._me)

		def __len__(self):
			return len(self._me)

		def __getitem__(self, item):
			realeng = self._rule._engine._real
			return getattr(realeng, self._type)

		def __setitem__(self, key, value):
			self._me[key] = getattr(value, "name", value)

		def __delitem__(self, key):
			del self._me[key]

	def __init__(self, engine, name):
		self._engine = engine
		self.name = name
		realeng = engine._real
		realrule = realeng.rule[name]
		self._fake_triggers = list(map(getname, realrule.triggers))
		self._fake_prereqs = list(map(getname, realrule.prereqs))
		self._fake_actions = list(map(getname, realrule.actions))
		self.triggers = self.FakeFuncList(self, "trigger")
		self.prereqs = self.FakeFuncList(self, "prereq")
		self.actions = self.FakeFuncList(self, "action")

	def apply(self):
		realeng = self._engine._real
		realrule = realeng.rule[self.name]
		realtrigs = list(map(getname, realrule.triggers))
		if self._fake_triggers != realtrigs:
			realrule.triggers = self._fake_triggers
		realpreqs = list(map(getname, realrule.prereqs))
		if self._fake_prereqs != realpreqs:
			realrule.prereqs = self._fake_prereqs
		realacts = list(map(getname, realrule.actions))
		if self._fake_actions != realacts:
			realrule.actions = self._fake_actions


class FacadeNode(FacadeEntity, Node):
	class FacadeNodeUser(Mapping):
		__slots__ = ("_entity",)

		@property
		def only(self):
			if len(self) != 1:
				raise AttributeError("No user, or more than one")
			return self[next(iter(self))]

		def __init__(self, node):
			self._entity = node

		def __iter__(self):
			engine = self._entity.engine
			charn = self._entity.character.name
			return engine._unitness_cache.leader_cache.iter_keys(
				charn, self._entity.name, *engine._btt()
			)

		def __len__(self):
			engine = self._entity.engine
			charn = self._entity.character.name
			return engine._unitness_cache.leader_cache.count_keys(
				charn, self._entity.name, *engine._btt()
			)

		def __contains__(self, item):
			engine = self._entity.engine
			charn = self._entity.character.name
			try:
				return bool(
					engine._unitness_cache.leader_cache.retrieve(
						charn, self._entity.name, item, *engine._btt()
					)
				)
			except KeyError:
				return False

		def __getitem__(self, item):
			if item not in self:
				raise KeyError("Not used by that character", item)
			engine = self._entity.engine
			return engine.character[item]

	class FacadeNodeContent(Mapping):
		__slots__ = ("_entity",)

		def __init__(self, node):
			self._entity = node

		def __iter__(self):
			if hasattr(self._entity, "engine") and hasattr(
				self._entity.engine, "_node_contents_cache"
			):
				# The real contents cache is wrapped by the facade engine.
				try:
					return self._entity.engine._node_contents_cache.retrieve(
						self._entity.character.name,
						self._entity.name,
						*self._entity.engine._btt(),
					)
				except KeyError:
					return
			char = self._entity.character
			myname = self._entity.name
			for name, thing in char.thing.items():
				if thing["location"] == myname:
					yield name

		def __len__(self):
			# slow
			return len(set(self))

		def __contains__(self, item):
			return (
				item in self._entity.character.thing
				and self._entity.character.thing[item]["location"]
				== self._entity.name
			)

		def __getitem__(self, item):
			if item not in self:
				raise KeyError("Not contained here", item, self._entity.name)
			return self._entity.character.thing[item]

	@property
	def portal(self):
		return self.facade.portal[self["name"]]

	def successors(self):
		for dest in self.portal:
			yield self.character.place[dest]

	def contents(self):
		return self.content.values()

	def __init__(self, mapping, real_or_name=None, **kwargs):
		self.name = self.node = getattr(real_or_name, "name", real_or_name)
		super().__init__(mapping, real_or_name, **kwargs)

	def __getitem__(self, item):
		if item == "name":
			return self.name
		return super().__getitem__(item)

	@property
	def content(self):
		return self.FacadeNodeContent(self)

	@property
	def user(self):
		return self.FacadeNodeUser(self)

	def users(self):
		return self.user.values()

	def _set_plan(self, k, v):
		self.character.engine._planned[self.character.engine._curplan][
			self.character.engine.turn
		].append((self.character.name, self.name, k, v))


class FacadeThing(FacadeNode, AbstractThing):
	def __init__(self, mapping, real_or_name, **kwargs):
		from .node import Thing

		location = kwargs.get("location")
		if location is None and not (isinstance(real_or_name, Thing)):
			raise TypeError(
				"FacadeThing needs to wrap a real Thing, or have a location of its own"
			)
		super().__init__(mapping, real_or_name, **kwargs)
		self.character.thing._patch[
			getattr(real_or_name, "name", real_or_name)
		] = self

	def _get_real(self, name):
		return self.character.character.thing[name]

	@property
	def location(self):
		return self.facade.node[self["location"]]

	@location.setter
	def location(self, v):
		if isinstance(v, (FacadePlace, FacadeThing)):
			v = v.name
		if v not in self.facade.node:
			raise KeyError("Location {} not present".format(v))
		self["location"] = v

	def delete(self):
		del self.character.thing[self.name]


class FacadePlace(FacadeNode):
	"""Lightweight analogue of Place for Facade use."""

	def __init__(self, mapping, real_or_name, **kwargs):
		from .node import Place

		super().__init__(mapping, real_or_name, **kwargs)
		if isinstance(mapping, CharacterFacade):
			mapping.place._patch[real_or_name] = self
			return
		if not isinstance(real_or_name, Place):
			if real_or_name in mapping._patch:
				real_or_name = mapping._patch[real_or_name]
			else:
				mapping._patch[real_or_name] = self
				return
		self.character.place._patch[real_or_name.name] = self

	def _get_real(self, name):
		return self.character.character.place[name]

	def add_thing(self, name):
		self.facade.add_thing(name, self.name)

	def new_thing(self, name):
		return self.facade.new_thing(name, self.name)

	def delete(self):
		del self.character.place[self.name]


class FacadePortalMapping(FacadeEntityMapping, ABC):
	cls: Type[FacadeEntityMapping]

	def __getitem__(self, node):
		if node not in self:
			raise KeyError("No such node: {}".format(node))
		if node not in self._patch:
			self._patch[node] = self.cls(self.facade, node)
		ret = self._patch[node]
		if ret is ...:
			raise KeyError("masked")
		if type(ret) is not self.cls:
			nuret = self.cls(self.facade, node)
			if type(ret) is dict:
				nuret._patch = ret
			else:
				nuret.update(ret)
			ret = nuret
		return ret


class FacadePortal(FacadeEntity, Edge):
	"""Lightweight analogue of Portal for Facade use."""

	def __init__(self, mapping, other, **kwargs):
		if hasattr(mapping, "orig"):
			self.orig = mapping.orig
			self.dest = other
		else:
			self.dest = mapping.dest
			self.orig = other
		if hasattr(mapping, "facade"):
			facade = mapping.facade
		else:
			facade = mapping
		try:
			super().__init__(
				facade.character.node[self.orig],
				facade.character.node[self.dest],
				**kwargs,
			)
			self._real = facade.character.portal[self.orig][self.dest]
		except (KeyError, AttributeError):
			if self.orig in facade.node:
				origin = facade.node[self.orig]
			else:
				origin = facade.new_place(self.orig)
			if self.dest in facade.node:
				destination = facade.node[self.dest]
			else:
				destination = facade.new_place(self.dest)
			super().__init__(origin, destination, **kwargs)
			self._real = {}

	def __getitem__(self, item):
		if item == "origin":
			return self.orig
		if item == "destination":
			return self.dest
		return super().__getitem__(item)

	def __setitem__(self, k, v):
		if k in ("origin", "destination"):
			raise TypeError("Portals have fixed origin and destination")
		super().__setitem__(k, v)
		self.character.portal._tampered = True
		self.character.portal[self.orig]._tampered = True

	@property
	def origin(self):
		return self.facade.node[self.orig]

	@property
	def destination(self):
		return self.facade.node[self.dest]

	def _get_real(self, name):
		return self.character.character.portal[self._mapping.orig][name]

	def _set_plan(self, k, v):
		self.character.engine._planned[self.character.engine._curplan][
			self.character.engine.turn
		].append((self.character.name, self.orig, self.dest, k, v))

	def delete(self):
		del self.character.portal[self.orig][self.dest]
		self.character.portal._tampered = True
		self.character.portal[self.orig]._tampered = True


class FacadePortalSuccessors(FacadeEntityMapping):
	cls = FacadePortal
	innercls: type

	def __init__(self, facade, origname):
		from .portal import Portal

		self.innercls = Portal
		super().__init__(facade, origname)
		self.orig = origname

	def _make(self, k, v):
		return self.cls(self, k, **v)

	def _get_inner_map(self):
		try:
			return self.facade.character.portal[self.orig]
		except AttributeError:
			if not hasattr(self, "_inner_map"):
				self._inner_map = SignalDict()
			return self._inner_map


class FacadePortalPredecessors(FacadeEntityMapping):
	cls = FacadePortal
	innercls: type

	def __init__(self, facade, destname):
		from .portal import Portal

		self.innercls = Portal
		super().__init__(facade, destname)
		self.dest = destname

	def _make(self, k, v):
		return self.cls(self.facade.portal[k], v)

	def _get_inner_map(self):
		try:
			return self.facade.character.preportal[self.dest]
		except AttributeError:
			return {}


class CharacterFacade(AbstractCharacter):
	engine = getatt("db")

	def __getstate__(self):
		ports = {}
		for o in self.portal:
			if o not in ports:
				ports[o] = {}
			for d in self.portal[o]:
				ports[o][d] = dict(self.portal[o][d])
		things = {k: dict(v) for (k, v) in self.thing.items()}
		places = {k: dict(v) for (k, v) in self.place.items()}
		stats = {
			k: v.unwrap() if hasattr(v, "unwrap") else v
			for (k, v) in self.graph.items()
		}
		return things, places, ports, stats

	def __setstate__(self, state):
		self.character = None
		self.graph = self.StatMapping(self)
		(
			self.thing._patch,
			self.place._patch,
			self.portal._patch,
			self.graph._patch,
		) = state

	def add_places_from(self, seq, **attrs):
		for place in seq:
			self.add_place(place, **attrs)

	def add_things_from(self, seq, **attrs):
		for thing in seq:
			self.add_thing(thing, **attrs)

	def thing2place(self, name):
		self.place[name] = self.thing.pop(name)

	def place2thing(self, name, location):
		it = self.place.pop(name)
		it["location"] = location
		self.thing[name] = it

	def add_portals_from(self, seq, **attrs):
		for it in seq:
			self.add_portal(*it, **attrs)

	def remove_unit(self, a, b=None):
		if b is None:
			if not isinstance(a, FacadeNode):
				raise TypeError("Need a node or character")
			charn = a.character.name
			noden = a.name
		else:
			charn = a
			if isinstance(b, FacadeNode):
				noden = b.name
			else:
				noden = b
		self.engine._unitness_cache.store(
			self.name, charn, noden, *self.engine._btt(), False
		)

	def add_place(self, name, **kwargs):
		self.place[name] = kwargs

	def add_node(self, name, **kwargs):
		"""Version of add_node that assumes it's a place"""
		self.place[name] = kwargs

	def remove_node(self, node):
		"""Version of remove_node that handles place or thing"""
		if node in self.thing:
			del self.thing[node]
		else:
			del self.place[node]

	def remove_place(self, place):
		del self.place[place]

	def remove_thing(self, thing):
		del self.thing[thing]

	def add_thing(self, name, location, **kwargs):
		kwargs["location"] = location
		self.thing[name] = kwargs

	def add_portal(self, orig, dest, **kwargs):
		self.portal[orig][dest] = kwargs
		self.portal[orig]._tampered = True
		self.portal._tampered = True

	def remove_portal(self, origin, destination):
		del self.portal[origin][destination]
		self.portal._tampered = True
		self.portal[origin]._tampered = True

	def add_edge(self, orig, dest, **kwargs):
		"""Wrapper for add_portal"""
		self.add_portal(orig, dest, **kwargs)

	def add_unit(self, a, b=None):
		if b is None:
			if not isinstance(a, FacadeNode):
				raise TypeError("Need a node or character")
			charn = a.character.name
			noden = a.name
		else:
			charn = a
			if isinstance(b, FacadeNode):
				noden = b.name
			else:
				noden = b
		self.engine._unitness_cache.store(
			self.name, charn, noden, *self.engine._btt(), True
		)

	def __init__(self, engine=None, character=None, init_rulebooks=None):
		if engine is None:
			engine = self.db = EngineFacade(getattr(character, "db", None))
		elif isinstance(engine, EngineFacade):
			self.db = engine
		else:
			raise TypeError(
				"Can't instantiate CharacterFacade with this for an engine",
				engine,
			)
		if isinstance(character, AbstractCharacter):
			self.character = character
			if hasattr(character, "name"):
				engine.character._patch[character.name] = self
				self._name = character.name
			else:
				self._name = character
		else:
			self._name = character
			self.character = None

		self._stat_map = self.StatMapping(self)
		self._rb_patch = {}

	@property
	def graph(self):
		return self._stat_map

	@graph.setter
	def graph(self, v):
		self._stat_map.clear()
		self._stat_map.update(v)

	def portals(self):
		for ds in self.portal.values():
			yield from ds.values()

	class UnitGraphMapping(Mapping):
		class UnitMapping(Mapping):
			def __init__(self, character, graph_name):
				self.character = character
				self.graph_name = graph_name

			def __iter__(self):
				for key in self.character.engine._unitness_cache.iter_keys(
					self.character.name,
					self.graph_name,
					*self.character.engine._btt(),
				):
					if key in self:
						yield key

			def __len__(self):
				return self.character.engine._unitness_cache.count_keys(
					self.character.name,
					self.graph_name,
					*self.character.engine._btt(),
				)

			def __contains__(self, item):
				try:
					return self.character.engine._unitness_cache.retrieve(
						self.character.name,
						self.graph_name,
						item,
						*self.character.engine._btt(),
					)
				except KeyError:
					return False

			def __getitem__(self, item):
				if item not in self:
					if not self.character.engine._mockup:
						raise KeyError(
							"Not a unit of this character in this graph",
							item,
							self.character.name,
							self.graph_name,
						)
					self.character.add_unit(
						self.character.engine.character[self.graph_name].node[
							item
						]
					)
				return self.character.engine.character[self.graph_name].node[
					item
				]

		def __init__(self, character):
			self.character = character

		def __iter__(self):
			engine = self.character.engine
			name = self.character.name
			now = self.character.engine._btt()
			for key in engine._unitness_cache.iter_keys(name, *now):
				if key in self:
					yield key

		def __len__(self):
			return self.character.engine._unitness_cache.count_keys(
				self.character.name, *self.character.engine._btt()
			)

		def __contains__(self, item):
			now = self.character.engine._btt()
			name = self.character.name
			engine = self.character.engine
			try:
				engine._unitness_cache.retrieve(name, item, *now)
				return True
			except KeyError:
				return False

		def __getitem__(self, item):
			if item not in self and not self.character.engine._mockup:
				raise KeyError(
					"Character has no units in graph",
					self.character.name,
					item,
				)
			return self.UnitMapping(self.character, item)

	class ThingMapping(FacadeEntityMapping):
		cls = FacadeThing
		innercls: type

		def __init__(self, facade, _=None):
			from .node import Thing

			self.innercls = Thing
			super().__init__(facade, _)

		def _get_inner_map(self):
			try:
				return self.facade.character.thing
			except AttributeError:
				return {}

		def patch(self, d: dict):
			places = d.keys() & self.facade.place.keys()
			if places:
				raise KeyError(
					f"Tried to patch places on thing mapping: {places}"
				)
			self.facade.node.patch(d)

	class PlaceMapping(FacadeEntityMapping):
		cls = FacadePlace
		innercls: type

		def __init__(self, facade, _=None):
			from .node import Place

			if not isinstance(facade, CharacterFacade):
				raise TypeError("Need CharacterFacade")

			self.innercls = Place
			super().__init__(facade, _)

		def _get_inner_map(self):
			if isinstance(self.facade.character, nx.Graph) and not isinstance(
				self.facade.character, AbstractCharacter
			):
				return self.facade.character._node
			try:
				return self.facade.character.place
			except AttributeError:
				return {}

		def patch(self, d: dict):
			things = d.keys() & self.facade.thing.keys()
			if things:
				raise KeyError(
					f"Tried to patch things on place mapping: {things}"
				)
			self.facade.node.patch(d)

	def ThingPlaceMapping(self, *args):
		return CompositeDict(self.place, self.thing)

	class PortalSuccessorsMapping(FacadePortalMapping):
		cls = FacadePortalSuccessors

		def __contains__(self, item):
			return item in self.facade.node

		def _get_inner_map(self):
			try:
				return self.facade.character._adj
			except AttributeError:
				return {}

	class PortalPredecessorsMapping(FacadePortalMapping):
		cls = FacadePortalPredecessors

		def __contains__(self, item):
			return item in self.facade._node

		def _get_inner_map(self):
			try:
				return self.facade.character.pred
			except AttributeError:
				return {}

	class StatMapping(MutableMappingUnwrapper, Signal):
		def __init__(self, facade):
			super().__init__()
			self.facade = facade
			self._patch = {}

		def copy(self):
			d = {}
			if hasattr(self.facade.character, "graph"):
				for k, v in self.facade.character.graph.items():
					if k not in self._patch:
						d[k] = v
					elif self._patch[k] is not ...:
						d[k] = self._patch[k]
			for k, v in self._patch.items():
				if v is not ...:
					d[k] = v
			return d

		def __iter__(self):
			seen = set()
			if hasattr(self.facade.character, "graph"):
				for k in self.facade.character.graph:
					if k not in self._patch:
						yield k
						seen.add(k)
			for k, v in self._patch.items():
				if k not in seen and v is not ...:
					yield k

		def __len__(self):
			n = 0
			for k in self:
				n += 1
			return n

		def __contains__(self, k):
			if k in self._patch:
				return self._patch[k] is not ...
			if (
				hasattr(self.facade.character, "graph")
				and k in self.facade.character.graph
			):
				return True
			return False

		def __getitem__(self, k):
			if k not in self._patch and hasattr(
				self.facade.character, "graph"
			):
				ret = self.facade.character.graph[k]
				if not hasattr(ret, "unwrap"):
					return ret
				self._patch[k] = ret.unwrap()
			if self._patch[k] is ...:
				return KeyError("masked", k)
			return self._patch[k]

		def __setitem__(self, k, v):
			if hasattr(self.facade, "engine") and self.facade.engine._planning:
				self.facade.engine._planned[
					self.facade.character.engine._curplan
				][self.facade.engine.turn].append((self.facade.name, k, v))
				return
			self._patch[k] = v

		def __delitem__(self, k):
			self._patch[k] = ...

		def __repr__(self):
			toshow = {}
			if hasattr(self.facade.character, "graph"):
				for k in (
					self._patch.keys() | self.facade.character.graph.keys()
				):
					if k in self._patch:
						if self._patch[k] is not ...:
							toshow[k] = self._patch[k]
					elif k in self.facade.character.graph:
						v = self.facade.character.graph[k]
						if hasattr(v, "unwrap") and not hasattr(
							v, "no_unwrap"
						):
							v = v.unwrap()
						toshow[k] = v
			return f"<StatMapping {toshow}>"

	def apply(self):
		"""Do all my changes for real in a batch"""
		realchar = self.character
		realstat = realchar.stat
		realthing = realchar.thing
		realplace = realchar.place
		realport = realchar.portal
		realeng = self.engine._real
		for k, v in self.stat._patch.items():
			if v is ...:
				del realstat[k]
			else:
				realstat[k] = v
		self.stat._patch = {}
		for k, v in self.thing._patch.items():
			if v is ...:
				del realthing[k]
			elif k not in realthing:
				if isinstance(v, FacadeThing):
					v = v._patch
				if "name" in v:
					assert v.pop("name") == k
				realchar.add_thing(k, **v)
			else:
				v.apply()
		self.thing._patch = {}
		for k, v in self.place._patch.items():
			if v is ...:
				del realplace[k]
			elif k not in realplace:
				realchar.add_place(k, **v)
			else:
				v.apply()
		self.place._patch = {}
		if getattr(self.portal, "_tampered", False):
			for orig, dests in self.portal._patch.items():
				if not getattr(dests, "_tampered", False):
					continue
				for dest, v in dests.items():
					if v is ...:
						del realport[orig][dest]
					elif orig not in realport or dest not in realport[orig]:
						realchar.add_portal(orig, dest, **v)
					else:
						v.apply()
				del dests._tampered
			del self.portal._tampered
		self.portal._patch = {}


class EngineFacade(AbstractEngine):
	char_cls = CharacterFacade
	thing_cls = FacadeThing
	place_cls = FacadePlace
	portal_cls = FacadePortal
	time = TimeSignalDescriptor()

	@cached_property
	def function(self):
		return FunctionStore(None)

	@cached_property
	def method(self):
		return FunctionStore(None)

	@cached_property
	def trigger(self):
		return FunctionStore(None)

	@cached_property
	def prereq(self):
		return FunctionStore(None)

	@cached_property
	def action(self):
		return FunctionStore(None)

	class FacadeUniversalMapping(Signal, MutableMapping):
		def __init__(self, engine: AbstractEngine):
			super().__init__()
			assert not isinstance(engine, EngineFacade)
			self.engine = engine
			self._patch = {}
			self._deleted = set()
			self.closed = False

		def _effective_keys(self):
			if self.engine:
				return (
					self.engine.universal.keys() | self._patch.keys()
				) - self._deleted
			return self._patch.keys() - self._deleted

		def __iter__(self):
			yield from self._effective_keys()

		def __len__(self):
			return len(self._effective_keys())

		def __contains__(self, item):
			return item not in self._deleted and (
				item in self._patch
				and (not self.engine or item in self.engine.universal)
			)

		def __getitem__(self, item):
			if item in self._patch:
				ret = self._patch[item]
				if ret is ...:
					raise KeyError("Universal key deleted", item)
				return ret
			elif self.engine and item in self.engine.universal:
				return self.engine.universal[item]
			else:
				raise KeyError("No universal key", item)

		def __setitem__(self, key, value):
			self._patch[key] = value
			if value is not ...:
				self._deleted.discard(key)
			self.send(self, key=key, value=value)

		def __delitem__(self, key):
			if key not in self.engine.universal:
				raise KeyError("No key to delete", key)
			self._patch[key] = ...
			self._deleted.add(key)
			self.send(self, key=key, value=...)

	class FacadeCharacterMapping(Mapping):
		def __init__(self, engine: "EngineFacade"):
			assert isinstance(engine, EngineFacade)
			self.engine = engine
			self._patch = {}

		def __getitem__(self, key, /):
			realeng = self.engine._real
			if realeng and key not in realeng.character:
				raise KeyError("No character", key)
			if key not in self._patch:
				if realeng:
					fac = CharacterFacade(self.engine, realeng.character[key])
				elif self.engine._mockup:
					fac = CharacterFacade(self.engine, key)
				else:
					raise KeyError("No character", key)
				self._patch[key] = fac
			return self._patch[key]

		def __len__(self):
			return len(self.engine.character)

		def __iter__(self):
			return iter(self.engine.character)

		def apply(self):
			for pat in self._patch.values():
				pat.apply()
			self._patch = {}

	class FacadeCache(Cache):
		def __init__(self, cache, name):
			self._created = cache.db._btt()
			super().__init__(cache.db, name)
			self._real = cache

		def retrieve(self, *args, search=False):
			try:
				return super().retrieve(*args, search=search)
			except (NotInKeyframeError, TotalKeyError):
				return self._real.retrieve(*args, search=search)

		def _get_keycache(
			self, parentity, branch, turn, tick, forward: bool = None
		):
			if forward is None:
				forward = self._real.db._forward
			# Find the last effective keycache before the facade was created.
			# Get the additions and deletions since then.
			# Apply those to the keycache and return it.
			kc = set(
				self._real._get_keycache(
					parentity, *self._created, forward=forward
				)
			)
			added, deleted = self._get_adds_dels(
				parentity, branch, turn, tick, stoptime=self._created
			)
			return frozenset((kc | added) - deleted)

	class FacadeUnitnessCache(FacadeCache, UnitnessCache):
		def __init__(self, cache):
			self._created = cache.db._btt()
			UnitnessCache.__init__(self, cache.db, "unitness_cache")
			self.user_cache = EngineFacade.FacadeCache(
				cache.leader_cache, "user_cache"
			)
			self._real = cache

	def __init__(self, real: AbstractEngine | None, mock=False):
		assert not isinstance(real, EngineFacade)
		self._mockup = mock
		if real is not None:
			for alias in (
				"submit",
				"load_at",
				"function",
				"method",
				"trigger",
				"prereq",
				"action",
				"string",
				"log",
				"debug",
				"info",
				"warning",
				"error",
				"critical",
			):
				try:
					setattr(self, alias, getattr(real, alias))
				except AttributeError:
					print(f"{alias} not implemented on {type(real)}")
		elif mock:
			import sys
			from unittest.mock import MagicMock

			from .collections import FunctionStore, StringStore

			for funcs in ("function", "method", "trigger", "prereq", "action"):
				setattr(self, funcs, FunctionStore(None))
			self.string = StringStore({}, None)
			for mockery in ("submit", "load_at"):
				setattr(self, mockery, MagicMock())
			if "kivy" in sys.modules:
				from kivy.logger import Logger

				logger = Logger
			else:
				from logging import getLogger

				logger = getLogger("lisien")
			for loggish in (
				"log",
				"debug",
				"info",
				"warning",
				"error",
				"critical",
			):
				setattr(self, loggish, getattr(logger, loggish))
		self.closed = False
		self._real = real
		self._planning = False
		self._planned = defaultdict(lambda: defaultdict(list))
		self.character = self.FacadeCharacterMapping(self)
		self.universal = self.FacadeUniversalMapping(real)
		self._rando = random.Random()
		self.world_lock = RLock()
		if real is not None:
			self._rando.setstate(real.universal["rando_state"])
			self.branch, self.turn, self.tick = real._btt()
			self._branches_d = real._branches_d.copy()
			self._turn_end = TurnEndDict(self)
			self._turn_end_plan = TurnEndPlanDict(self)
			if not hasattr(real, "is_proxy"):
				self._turn_end.update(real._turn_end)
				self._turn_end_plan.update(real._turn_end_plan)
				self._nodes_cache = self.FacadeCache(
					real._nodes_cache, "nodes_cache"
				)
				self._things_cache = self.FacadeCache(
					real._things_cache, "things_cache"
				)
				self._unitness_cache = self.FacadeUnitnessCache(
					real._unitness_cache
				)
		else:
			self._branches_d = {
				"trunk": (
					None,
					0,
					0,
					0,
					0,
				)
			}
			self._turn_end_plan = {}
			self.branch = "trunk"
			self.turn = 0
			self.tick = 0

	def handle(self, *args, **kwargs):
		print_call_sig("handle", *args, **kwargs)

	def _get_node(
		self, char: AbstractCharacter | CharName, node: NodeName
	) -> Node:
		return self.character[char].node[node]

	def _btt(self):
		return self.branch, self.turn, self.tick

	def _set_btt(self, branch: str, turn: int, tick: int) -> None:
		(self.branch, self.turn, self.tick) = (branch, turn, tick)

	def _extend_branch(self, branch: str, turn: int, tick: int) -> None:
		if branch in self._branches_d:
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
		else:
			self._branches_d[branch] = None, turn, tick, turn, tick

	def _start_branch(
		self, parent: str, branch: str, turn: int, tick: int
	) -> None:
		self._branches_d[branch] = (parent, turn, tick, turn, tick)
		self._extend_branch(branch, turn, tick)

	def export(
		self,
		name: str | None,
		path: str | os.PathLike | None = None,
		indent: bool = True,
	) -> None:
		raise RuntimeError("Can't export facades")

	@classmethod
	def from_archive(
		cls,
		path: str | os.PathLike,
		prefix: str | os.PathLike | None = ".",
		**kwargs,
	) -> AbstractEngine:
		raise RuntimeError(
			"Can't import archived Lisien games into facades. Use a regular Engine."
		)

	def load_at(self, branch: str, turn: int, tick: int) -> None:
		pass

	def turn_end(self, branch: str = None, turn: int = None) -> int:
		if branch is None:
			branch = self.branch
		if turn is None:
			turn = self.turn
		return self._turn_end[branch, turn]

	def turn_end_plan(self, branch: str = None, turn: int = None) -> int:
		if branch is None:
			branch = self.branch
		if turn is None:
			turn = self.turn
		return self._turn_end_plan[branch, turn]

	def _nbtt(self):
		self.tick += 1
		return self._btt()

	@contextmanager
	def batch(self):
		self.info(
			"Facades already batch all changes, so this batch does nothing"
		)
		yield

	@contextmanager
	def plan(self):
		if getattr(self, "_planning", False):
			raise RuntimeError("Already planning")
		self._planning = True
		start_time = self._btt()
		if hasattr(self, "_curplan"):
			self._curplan += 1
		else:
			# Will break if used in a proxy, which I want to do eventually...
			self._curplan = self._real._last_plan + 1
		yield self._curplan
		self._planning = False
		self._set_btt(*start_time)

	def add_character(
		self,
		name: Key,
		data: nx.Graph | DiGraph = None,
		layout: bool = False,
		node: dict = None,
		edge: dict = None,
		**kwargs,
	):
		self.character._patch[name] = char = CharacterFacade(self, name)
		if data:
			char.become(data)
		if node:
			char.node.update(node)
		if edge:
			char.adj.update(edge)

	def apply(self):
		realeng = self._real
		self.character.apply()
		if not getattr(self, "_planned", None):
			return
		# Do I actually need these sorts? Insertion order's preserved...
		for plan_num in sorted(self._planned):
			with (
				timer(
					f"seconds to apply plan {plan_num}",
					logfun=self._real.debug,
				),
				realeng.plan(),
			):  # resets time at end of block
				for turn in sorted(self._planned[plan_num]):
					# Not setting `realeng.turn` the normal way, because that
					# would save the state of the randomizer, which is not
					# relevant here
					realeng._oturn = turn
					for tup in self._planned[plan_num][turn]:
						if len(tup) == 3:
							char, k, v = tup
							realeng.character[char].stat[k] = v
						elif len(tup) == 4:
							char, node, k, v = tup
							realchar = realeng.character[char]
							if node in realchar.node:
								if k is ...:
									realchar.remove_node(node)
								elif k == "location":
									# assume the location really exists, since
									# it did while planning
									now = realeng._nbtt()
									realeng._things_cache.store(
										char, node, *now, v
									)
									realeng.query.set_thing_loc(
										char, node, *now, v
									)
								else:
									realchar.node[node][k] = v
							elif k == "location":
								realchar.add_thing(node, v)
							else:
								realchar.add_place(node, **{k: v})
						elif len(tup) == 5:
							char, orig, dest, k, v = tup
							realchar = realeng.character[char]
							if (
								orig in realchar.portal
								and dest in realchar.portal[orig]
							):
								if k is ...:
									realchar.remove_portal(orig, dest)
								else:
									realchar.portal[orig][dest][k] = v
							else:
								realchar.add_portal(orig, dest, **{k: v})
