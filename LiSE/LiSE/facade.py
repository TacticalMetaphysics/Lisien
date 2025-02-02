from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from functools import cached_property
from operator import attrgetter
from threading import RLock
from typing import Type, MutableMapping, Mapping, MutableSequence, Any

import networkx as nx
from blinker import Signal

from LiSE.allegedb.cache import Cache
from LiSE.allegedb.wrap import MutableMappingUnwrapper
from LiSE.util import (
	AbstractCharacter,
	getatt,
	AbstractEngine,
	AbstractThing,
	SignalDict,
)
from LiSE.xcollections import CompositeDict


class FacadeEntity(MutableMapping, Signal, ABC):
	exists = True

	@property
	def rulebook(self):
		if "rulebook" in self._patch:
			return self._patch["rulebook"]
		return self._real.rulebook

	@rulebook.setter
	def rulebook(self, rbname):
		self._patch["rulebook"] = rbname

	@property
	def engine(self):
		return self.character.engine

	db = engine

	@abstractmethod
	def _get_real(self, name):
		raise NotImplementedError()

	def __init__(self, mapping, real_or_name=None, **kwargs):
		super().__init__()
		self.facade = self.character = getattr(mapping, "facade", mapping)
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
			return patch[item] is not None
		if hasattr(self, "_real"):
			return item in self._real
		return False

	def __iter__(self):
		patch = self._patch
		ks = patch.keys()
		if hasattr(self, "_real"):
			ks |= self._real.keys()
		for k in ks:
			if k not in patch or patch[k] is not None:
				yield k

	def __len__(self):
		n = 0
		for _ in self:
			n += 1
		return n

	def __getitem__(self, k):
		if k in self._patch:
			if self._patch[k] is None:
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
		self._patch[k] = None

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

	All the entities are of the same type, ``facadecls``, possibly
	being distorted views of entities of the type ``innercls``.

	"""

	facadecls: Type[FacadeEntity]

	@abstractmethod
	def _get_inner_map(self):
		raise NotImplementedError("Missing _get_inner_map")

	def _make(self, k, v):
		if isinstance(v, dict):
			for badkey in ("character", "engine", "name"):
				if badkey in v:
					del v[badkey]
			return self.facadecls(self, k, **v)
		return self.facadecls(self, v)

	engine = getatt("facade.engine")

	def __init__(self, facade, _=None):
		"""Store the facade."""
		super().__init__()
		self.facade = facade
		self._patch = {}

	def __contains__(self, k):
		if k in self._patch:
			return self._patch[k] is not None
		return k in self._get_inner_map()

	def __iter__(self):
		seen = set()
		for k in self._patch:
			if k not in seen and self._patch[k] is not None:
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
		if k not in self:
			raise KeyError
		if k not in self._patch:
			self._patch[k] = self._make(k, self._get_inner_map()[k])
		ret = self._patch[k]
		if ret is None:
			raise KeyError
		if type(ret) is not self.facadecls:
			ret = self._patch[k] = self._make(k, ret)
		return ret

	def __setitem__(self, k, v):
		if not isinstance(v, self.facadecls):
			v = self._make(k, v)
		self._patch[k] = v
		if self is not self.facade.node:
			self.facade.node.send(self, key=k, value=v)

	def __delitem__(self, k):
		if k not in self:
			raise KeyError("{} not present".format(k))
		self._patch[k] = None


class FacadeRulebook(MutableSequence, ABC):
	name: Any
	_fake: list
	engine: EngineFacade

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


class FacadeNode(FacadeEntity, ABC):
	class FacadeNodeUser(Mapping):
		__slots__ = ("_entity",)

		def __init__(self, node):
			self._entity = node

		def __iter__(self):
			engine = self._entity.engine
			charn = self._entity.character.name
			return engine._unitness_cache.user_cache.iter_keys(
				charn, self._entity.name, *engine._btt()
			)

		def __len__(self):
			engine = self._entity.engine
			charn = self._entity.character.name
			return engine._unitness_cache.user_cache.count_keys(
				charn, self._entity.namee, *engine._btt()
			)

		def __contains__(self, item):
			engine = self._entity.engine
			charn = self._entity.character.name
			try:
				return bool(
					engine._unitness_cache.user_cache.retrieve(
						charn, self._entity.name, item, *engine._btt()
					)
				)
			except KeyError:
				return False

		def __getitem__(self, item):
			if item not in self:
				raise KeyError("Not used by that character", item)
			engine = self._entity.engine
			charn = self._entity.character.name
			return engine.character[charn]

	class FacadeNodeContent(Mapping):
		__slots__ = ("_entity",)

		def __init__(self, node):
			self._entity = node

		def __iter__(self):
			if hasattr(self._entity.engine, "_node_contents_cache"):
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

	def __init__(self, mapping, real_or_name=None, **kwargs):
		self.name = getattr(real_or_name, "name", real_or_name)
		super().__init__(mapping, real_or_name, **kwargs)

	def __iter__(self):
		yield "name"
		yield from super().__iter__()

	def __getitem__(self, item):
		if item == "name":
			return self.name
		return super().__getitem__(item)

	@property
	def content(self):
		return self.FacadeNodeContent(self)

	def contents(self):
		for thing in self.facade.thing.values():
			# it seems like redundant FacadeNode are being created sometimes
			if thing["location"] == self.name:
				yield thing

	@property
	def user(self):
		return self.FacadeNodeUser(self)

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


class FacadePlace(FacadeNode):
	"""Lightweight analogue of Place for Facade use."""

	def __init__(self, mapping, real_or_name, **kwargs):
		from .node import Place

		super().__init__(mapping, real_or_name, **kwargs)
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


class FacadePortalMapping(FacadeEntityMapping, ABC):
	cls: Type[FacadeEntityMapping]

	def __getitem__(self, node):
		if node not in self:
			raise KeyError("No such node: {}".format(node))
		if node not in self._patch:
			self._patch[node] = self.cls(self.facade, node)
		ret = self._patch[node]
		if ret is None:
			raise KeyError("masked")
		if type(ret) is not self.cls:
			nuret = self.cls(self.facade, node)
			if type(ret) is dict:
				nuret._patch = ret
			else:
				nuret.update(ret)
			ret = nuret
		return ret


class FacadePortal(FacadeEntity):
	"""Lightweight analogue of Portal for Facade use."""

	def __init__(self, mapping, other, **kwargs):
		super().__init__(mapping, other, **kwargs)
		if hasattr(mapping, "orig"):
			self.orig = mapping.orig
			self.dest = other
		else:
			self.dest = mapping.dest
			self.orig = other
		try:
			self._real = self.facade.character.portal[self.orig][self.dest]
		except (KeyError, AttributeError):
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


class FacadePortalSuccessors(FacadeEntityMapping):
	facadecls = FacadePortal
	innercls: type

	def __init__(self, facade, origname):
		from .portal import Portal

		self.innercls = Portal
		super().__init__(facade, origname)
		self.orig = origname

	def _make(self, k, v):
		return self.facadecls(self, k, **v)

	def _get_inner_map(self):
		try:
			return self.facade.character.portal[self.orig]
		except AttributeError:
			if not hasattr(self, "_inner_map"):
				self._inner_map = SignalDict()
			return self._inner_map


class FacadePortalPredecessors(FacadeEntityMapping):
	facadecls = FacadePortal
	innercls: type

	def __init__(self, facade, destname):
		from .portal import Portal

		self.innercls = Portal
		super().__init__(facade, destname)
		self.dest = destname

	def _make(self, k, v):
		return self.facadecls(self.facade.portal[k], v)

	def _get_inner_map(self):
		try:
			return self.facade.character.preportal[self.dest]
		except AttributeError:
			return {}


class CharacterFacade(AbstractCharacter, nx.DiGraph):
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

	def remove_portal(self, origin, destination):
		del self.portal[origin][destination]

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

	def __init__(self, character=None, engine=None):
		self.character = character
		self.db = EngineFacade(engine or getattr(character, "db", None))
		self._stat_map = self.StatMapping(self)
		self._rb_patch = {}
		if character:
			self.db.character._patch[character.name] = self

	@property
	def graph(self):
		return self._stat_map

	@graph.setter
	def graph(self, v):
		self._stat_map.clear()
		self._stat_map.update(v)

	class ThingMapping(FacadeEntityMapping):
		facadecls = FacadeThing
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
		facadecls = FacadePlace
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

		def __iter__(self):
			seen = set()
			if hasattr(self.facade.character, "graph"):
				for k in self.facade.character.graph:
					if k not in self._patch:
						yield k
						seen.add(k)
			for k, v in self._patch.items():
				if k not in seen and v is not None:
					yield k

		def __len__(self):
			n = 0
			for k in self:
				n += 1
			return n

		def __contains__(self, k):
			if k in self._patch:
				return self._patch[k] is not None
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
			if self._patch[k] is None:
				return KeyError
			return self._patch[k]

		def __setitem__(self, k, v):
			if self.facade.engine._planning:
				self.facade.engine._planned[
					self.facade.character.engine._curplan
				][self.facade.engine.turn].append((self.facade.name, k, v))
				return
			self._patch[k] = v

		def __delitem__(self, k):
			self._patch[k] = None

	def apply(self):
		"""Do all my changes for real in a batch"""
		realchar = self.character
		realstat = realchar.stat
		realthing = realchar.thing
		realplace = realchar.place
		realport = realchar.portal
		realeng = self.engine._real
		for k, v in self.stat._patch.items():
			if v is None:
				del realstat[k]
			else:
				realstat[k] = v
		self.stat._patch = {}
		for k, v in self.thing._patch.items():
			if v is None:
				del realthing[k]
			elif k not in realthing:
				realchar.add_thing(k, **v)
			else:
				v.apply()
		self.thing._patch = {}
		for k, v in self.place._patch.items():
			if v is None:
				del realplace[k]
			elif k not in realplace:
				realchar.add_place(k, **v)
			else:
				v.apply()
		self.place._patch = {}
		for orig, dests in self.portal._patch.items():
			for dest, v in dests.items():
				if v is None:
					del realport[orig][dest]
				elif orig not in realport or dest not in realport[orig]:
					realchar.add_portal(orig, dest, **v)
				else:
					v.apply()
		self.portal._patch = {}


class EngineFacade(AbstractEngine):
	class FacadeUniversalMapping(MutableMapping):
		def __init__(self, engine: AbstractEngine):
			assert not isinstance(engine, EngineFacade)
			self.engine = engine
			self._patch = {}

		def __iter__(self):
			return iter(self._patch.keys() | self.engine.universal.keys())

		def __len__(self):
			return len(self._patch.keys() | self.engine.universal.keys())

		def __contains__(self, item):
			return item in self._patch or item in self.engine.universal

		def __getitem__(self, item):
			if item in self._patch:
				ret = self._patch[item]
				if ret is None:
					raise KeyError("Universal key deleted", item)
				return ret
			elif item in self.engine.universal:
				return self.engine.universal[item]
			else:
				raise KeyError("No universal key", item)

		def __setitem__(self, key, value):
			self._patch[key] = value

		def __delitem__(self, key):
			if key not in self.engine.universal:
				raise KeyError("No key to delete", key)
			self._patch[key] = None

	class FacadeCharacterMapping(Mapping):
		def __init__(self, engine: "EngineFacade"):
			assert isinstance(engine, EngineFacade)
			self.engine = engine
			self._patch = {}

		def __getitem__(self, key, /):
			realeng = self.engine._real
			if key not in realeng.character:
				raise KeyError("No character", key)
			if key not in self._patch:
				self._patch[key] = CharacterFacade(
					realeng.character[key], engine=self.engine
				)
			return self._patch[key]

		def __len__(self):
			return len(self.engine.character)

		def __iter__(self):
			return iter(self.engine.character)

		def apply(self):
			for pat in self._patch.values():
				pat.apply()
			rando_state = self.engine._rando.getstate()
			realeng = self.engine._real
			if rando_state != realeng._rando.getstate():
				realeng._rando.setstate(rando_state)
				realeng.universal["rando_state"] = rando_state
			self._patch = {}

	class FacadeCache(Cache):
		def __init__(self, cache, name):
			super().__init__(cache.db, name)
			self._real = cache

		def retrieve(self, *args):
			try:
				return super().retrieve(*args)
			except KeyError:
				return self._real.retrieve(*args)

	def __init__(self, real: AbstractEngine):
		assert not isinstance(real, EngineFacade)
		self.closed = False
		self._real = real
		self._planning = False
		self._planned = defaultdict(lambda: defaultdict(list))
		self.character = self.FacadeCharacterMapping(self)
		self.universal = self.FacadeUniversalMapping(real)
		self._rando = random.Random()
		self.world_lock = RLock()
		if real is not None:
			self._rando.setstate(real._rando.getstate())
			self.branch, self.turn, self.tick = real._btt()
			self._branches = deepcopy(real._branches)
			if hasattr(real, "is_proxy"):
				self._turn_end_plan = {}
			else:
				self._turn_end_plan = deepcopy(real._turn_end_plan)
				self._nodes_cache = self.FacadeCache(
					real._nodes_cache, "nodes_cache"
				)
				self._things_cache = self.FacadeCache(
					real._things_cache, "things_cache"
				)
				self._unitness_cache = self.FacadeCache(
					real._unitness_cache, "unitness_cache"
				)
				self._unitness_cache.user_cache = self.FacadeCache(
					real._unitness_cache.user_cache, "user_cache"
				)
		else:
			self._branches = {}
			self._turn_end_plan = {}

	def _btt(self):
		return self.branch, self.turn, self.tick

	def _nbtt(self):
		self.tick += 1
		return self._btt()

	@contextmanager
	def plan(self):
		if getattr(self, "_planning", False):
			raise RuntimeError("Already planning")
		self._planning = True
		if hasattr(self, "_curplan"):
			self._curplan += 1
		else:
			# Will break if used in a proxy, which I want to do eventually...
			self._curplan = self._real._last_plan + 1
		yield self._curplan
		self._planning = False

	def load_at(self, branch, turn, tick):
		self._real.load_at(branch, turn, tick)

	def apply(self):
		realeng = self._real
		self.character.apply()
		if not getattr(self, "_planned", None):
			return
		# Do I actually need these sorts? Insertion order's preserved...
		for plan_num in sorted(self._planned):
			with realeng.plan():  # resets time at end of block
				for turn in sorted(self._planned[plan_num]):
					realeng.turn = turn
					for tup in self._planned[plan_num][turn]:
						if len(tup) == 3:
							char, k, v = tup
							realeng.character[char].stat[k] = v
						elif len(tup) == 4:
							char, node, k, v = tup
							realchar = realeng.character[char]
							if node in realchar.node:
								if k is None:
									realchar.remove_node(node)
								else:
									realchar.node[node][k] = v
							elif k == "location":
								realchar.add_thing(node, v)
							else:
								realchar.add_place(node, k=v)
						elif len(tup) == 5:
							char, orig, dest, k, v = tup
							realchar = realeng.character[char]
							if (
								orig in realchar.portal
								and dest in realchar.portal[orig]
							):
								if k is None:
									realchar.remove_portal(orig, dest)
								else:
									realchar.portal[orig][dest][k] = v
							else:
								realchar.add_portal(orig, dest, k=v)
