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
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections import UserDict, defaultdict
from contextlib import contextmanager
from functools import cached_property, partial
from logging import Logger
from operator import attrgetter
from random import Random
from threading import RLock
from types import FunctionType, MethodType
from typing import (
	TYPE_CHECKING,
	ClassVar,
	Literal,
	Mapping,
	MutableMapping,
	MutableSequence,
	Optional,
)
from unittest.mock import MagicMock

import networkx as nx
from attr import Factory
from attrs import define, field
from blinker import Signal

from .cache import (
	Cache,
	LeaderSetCache,
	NodesCache,
	ThingsCache,
	TurnEndDict,
	TurnEndPlanDict,
	UnitnessCache,
)
from .collections import (
	CompositeDict,
	FunctionStore,
	StringStore,
	TriggerStore,
)
from .exc import NotInKeyframeError, TotalKeyError
from .types import (
	AbstractBookmarkMapping,
	AbstractCharacter,
	AbstractEngine,
	AbstractThing,
	Branch,
	CharName,
	DiGraph,
	Edge,
	Key,
	KeyHint,
	Node,
	NodeName,
	RulebookName,
	RuleName,
	SignalDict,
	Stat,
	Tick,
	Turn,
	Value,
	ValueHint,
	validate_time,
)
from .util import getatt, print_call_sig, timer
from .wrap import MappingUnwrapper

if TYPE_CHECKING:
	from .engine import Engine


class FacadeEntity(
	MutableMapping[Stat | Literal["rulebook"], Value], Signal, ABC
):
	character: CharacterFacade
	exists: ClassVar[bool] = True

	def __init__(self, character: CharacterFacade):
		super().__init__()
		self.character = character

	@property
	@abstractmethod
	def _real(
		self,
	) -> DiGraph | Node | Edge | dict[Stat | Literal["rulebook"], Value]: ...

	@cached_property
	def _patch(self) -> dict[Stat | Literal["rulebook"], Value | type(...)]:
		return {}

	@cached_property
	def engine(self):
		return self.character.engine

	@property
	def rulebook(self):
		if "rulebook" in self._patch:
			return self._patch["rulebook"]
		if hasattr(self._real, "rulebook"):
			return self._real.rulebook
		raise AttributeError("No rulebook")

	@rulebook.setter
	def rulebook(self, rbname):
		self._patch["rulebook"] = rbname

	def __contains__(self, item: Stat | KeyHint) -> bool:
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

	def __getitem__(self, k: Stat | KeyHint):
		k = Stat(Key(k))
		if k in self._patch:
			if self._patch[k] is ...:
				raise KeyError("{} has been masked.".format(k))
			return self._patch[k]
		if not hasattr(self, "_real"):
			raise KeyError(f"{k} unset, and no underlying Thing")
		ret = self._real[k]
		if hasattr(ret, "unwrap"):  # a wrapped mutable object
			ret = ret.unwrap()
			self._patch[k] = ret
			# changes will be reflected in the
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
		self._patch.clear()

	def unwrap(self):
		return {
			k: v.unwrap() if hasattr(v, "unwrap") else v
			for (k, v) in self.items()
		}


getname = attrgetter("name")


class FacadeEntityMapping[_NAME: Key, _CLS: Node | Edge | DiGraph](
	MappingUnwrapper[_NAME, _CLS], MutableMapping[_NAME, _CLS], Signal, ABC
):
	"""Mapping that contains entities in a Facade.

	All the entities are of the same type, ``cls``, possibly
	being distorted views of entities of the type ``innercls``.

	"""

	character: CharacterFacade
	cls: ClassVar[type[FacadeEntity]]

	def __init__(self, character: CharacterFacade):
		super().__init__()
		self.character = character

	@cached_property
	def _patch(self) -> dict[_NAME, _CLS | type(...)]:
		return {}

	@abstractmethod
	def _get_inner_map(self): ...

	@abstractmethod
	def _make(self, k, v): ...

	engine = getatt("character.engine")

	def __eq__(self, other: Mapping) -> bool:
		if not isinstance(other, Mapping):
			return False
		if self.keys() != other.keys():
			return False
		for k in self:
			if self[k] != other[k]:
				return False
		return True

	def __ne__(self, other: Mapping):
		if not isinstance(other, Mapping):
			return True
		if self.keys() != other.keys():
			return True
		for k in self:
			if self[k] != other[k]:
				return True
		return False

	def __contains__(self, k: _NAME):
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

	def __getitem__(self, k: _NAME) -> _CLS:
		if k not in self and not getattr(self.engine, "_mock", False):
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

	def __setitem__(self, k: _NAME, v: _CLS) -> None:
		if not isinstance(v, self.cls):
			v = self._make(k, v)
		self._patch[k] = v
		if self is not self.character.node:
			self.character.node.send(self, key=k, value=v)

	def __delitem__(self, k: _NAME) -> None:
		if k not in self:
			raise KeyError("{} not present".format(k))
		that = self[k]
		# Units don't work when we're wrapping an EngineProxy or nothing.
		# I'll fix that at some point I guess.
		# 2025-02-06
		if hasattr(self.character.engine._real, "_unitness_cache") and hasattr(
			that, "users"
		):
			user: CharacterFacade
			for user in list(that.users()):
				user.remove_unit(self.character.name, k)
		self._patch[k] = ...


class FacadeRulebook(MutableSequence, ABC):
	engine: EngineFacade
	name: Key
	_fake: ClassVar[list]

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

		def insert(self, index, value):
			self._me.insert(index, value)

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
	name: NodeName

	def __init__(
		self,
		character: CharacterFacade,
		name: NodeName,
		**kwargs,
	):
		kwargs: dict[Stat | Literal["rulebook"], Value]
		super().__init__(character)
		self.name = name
		self.update(kwargs)

	def __getitem__(
		self, item: Literal["name"] | Stat | KeyHint
	) -> NodeName | Value:
		if item == "name":
			return self.name
		return super().__getitem__(item)

	class FacadeNodeLeader(Mapping[CharName, DiGraph]):
		__slots__ = ("_entity",)

		@property
		def only(self):
			if len(self) != 1:
				raise AttributeError("No user, or more than one")
			return self[next(iter(self))]

		def __init__(self, node: Node):
			self._entity = node

		def __iter__(self):
			engine = self._entity.engine
			charn = self._entity.character.name
			return engine._unitness_cache.leader_cache.iter_keys(
				charn, self._entity.name, *engine.time
			)

		def __len__(self):
			engine = self._entity.engine
			charn = self._entity.character.name
			return engine._unitness_cache.leader_cache.count_keys(
				charn, self._entity.name, *engine.time
			)

		def __contains__(self, item):
			engine = self._entity.engine
			charn = self._entity.character.name
			try:
				return bool(
					engine._unitness_cache.leader_cache.retrieve(
						charn, self._entity.name, item, *engine.time
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
						*self._entity.engine.time,
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
		return self.character.portal[self.name]

	@property
	def _real(self) -> Node:
		return self.character.character.node[self.name]

	def successors(self):
		for dest in self.portal:
			yield self.character.place[dest]

	def contents(self):
		return self.content.values()

	@property
	def content(self):
		return self.FacadeNodeContent(self)

	@property
	def leader(self):
		return self.FacadeNodeLeader(self)

	def leaders(self):
		return self.leader.values()

	def _set_plan(self, k, v):
		self.character.engine._planned[self.character.engine._curplan][
			self.character.engine.turn
		].append((self.character.name, self.name, k, v))

	def add_thing(
		self,
		node: NodeName | KeyHint,
		**stats: dict[Stat | KeyHint, Value | ValueHint],
	):
		self.character.add_thing(node, self.name, **stats)

	def __eq__(self, other: Mapping) -> bool:
		if not isinstance(other, Mapping):
			return False
		if self.keys() != other.keys() - {"name"}:
			return False
		if "name" in other and self.name != other["name"]:
			return False
		if hasattr(other, "name"):
			if other.name != self.name:
				return False
		for k in self:
			if self[k] != other[k]:
				return False
		return True

	def __ne__(self, other: Mapping) -> bool:
		if not isinstance(other, Mapping):
			return True
		if self.keys() != other.keys() - {"name"}:
			return True
		if "name" in other and self.name != other["name"]:
			return True
		if hasattr(other, "name"):
			if other.name != self.name:
				return False
		for k in self:
			if self[k] != other[k]:
				return True
		return False


Node.register(FacadeNode)


class FacadeThing(AbstractThing, FacadeNode):
	__slots__ = ()
	character: CharacterFacade

	def __init__(
		self,
		character: CharacterFacade,
		name: NodeName,
		location: Node | NodeName,
		**kwargs,
	):
		if hasattr(location, "name"):
			location = location.name
		super().__init__(character, name, location=location, **kwargs)

	def _get_real(self, name):
		return self.character.character.thing[name]

	def to_place(self) -> FacadePlace:
		ret = self.character.place._patch[self.name] = FacadePlace(
			self.character, self.name
		)
		self.character.thing._patch[self.name] = ...
		return ret

	@property
	def location(self):
		return self.character.node[self["location"]]

	@location.setter
	def location(self, v):
		if isinstance(v, (FacadePlace, FacadeThing)):
			v = v.name
		if v not in self.character.node:
			raise KeyError("Location {} not present".format(v))
		self["location"] = v

	def delete(self):
		del self.character.thing[self.name]


class FacadePlace(FacadeNode):
	"""Lightweight analogue of Place for Facade use."""

	@property
	def _real(self) -> Node:
		try:
			return self.character.character.place[self.name]
		except KeyError as ex:
			raise AttributeError(
				"No real Place", self.character.name, self.name
			) from ex

	def to_thing(self, location: NodeName) -> FacadeThing:
		self.character.place._patch[self.name] = ...
		ret = self.character.thing._patch[self.name] = FacadeThing(
			self.character, self.name, location
		)
		return ret

	def new_thing(
		self,
		name: NodeName | KeyHint,
		**stats: dict[KeyHint, ValueHint] | dict[Stat, Value],
	):
		return self.character.new_thing(name, self.name, **stats)

	def delete(self):
		del self.character.place[self.name]


class FacadePortalSubMapping(FacadeEntityMapping, ABC):
	character: CharacterFacade
	node: NodeName

	def __init__(self, character: CharacterFacade, node: NodeName):
		super().__init__(character)
		self.node = node


class FacadePortalMapping(FacadeEntityMapping, ABC):
	cls: ClassVar[type[FacadePortalSubMapping]]

	def __getitem__(self, node):
		if node not in self:
			raise KeyError("No such node: {}".format(node))
		if node not in self._patch:
			self._patch[node] = self.cls(self.character, node)
		ret = self._patch[node]
		if ret is ...:
			raise KeyError("masked", node)
		if type(ret) is not self.cls:
			nuret = self.cls(self.character, node)
			if type(ret) is dict:
				nuret._patch = ret
			else:
				nuret.update(ret)
			ret = nuret
		return ret

	def _make(self, k, v):
		ret = self.cls(self.character, k)
		ret.update(v)
		return ret


class FacadePortal(FacadeEntity):
	"""Lightweight analogue of Portal for Facade use."""

	orig: NodeName
	dest: NodeName

	def __init__(
		self,
		character: CharacterFacade,
		orig: NodeName,
		dest: NodeName,
		**kwargs: dict[Stat | Literal["rulebook"], Value],
	):
		super().__init__(character)
		self.orig = orig
		self.dest = dest
		self._patch.update(kwargs)

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
	def _real(
		self,
	) -> DiGraph | Node | Edge | dict[Stat | Literal["rulebook"], Value]:
		try:
			return self.character.character.portal[self.orig][self.dest]
		except KeyError as ex:
			raise AttributeError(
				"No real Portal", self.character.name, self.orig, self.dest
			) from ex

	@property
	def origin(self):
		return self.character.node[self.orig]

	@property
	def destination(self):
		return self.character.node[self.dest]

	@property
	def _get_real(self) -> Edge:
		return self.character.character.portal[self.orig][self.dest]

	def _set_plan(self, k, v):
		self.character.engine._planned[self.character.engine._curplan][
			self.character.engine.turn
		].append((self.character.name, self.orig, self.dest, k, v))

	def delete(self):
		del self.character.portal[self.orig][self.dest]
		self.character.portal._tampered = True
		self.character.portal[self.orig]._tampered = True


class FacadePortalSuccessors(FacadePortalSubMapping):
	cls: ClassVar[type] = FacadePortal

	def __init__(self, character: CharacterFacade, orig: NodeName):
		super().__init__(character, orig)

	@cached_property
	def innercls(self):
		from .portal import Portal

		return Portal

	@cached_property
	def orig(self) -> NodeName:
		return self.node

	def _make(self, k, v):
		return self.cls(self.character, self.node, k, **v)

	def _get_inner_map(self):
		try:
			return self.character.character.portal[self.orig]
		except AttributeError:
			if not hasattr(self, "_inner_map"):
				self._inner_map = SignalDict()
			return self._inner_map


class FacadePortalPredecessors(FacadePortalSubMapping):
	cls: ClassVar[type] = FacadePortal

	def __init__(self, character: CharacterFacade, dest: NodeName):
		super().__init__(character, dest)

	@cached_property
	def innercls(self):
		from .portal import Portal

		return Portal

	@cached_property
	def dest(self) -> NodeName:
		return self.node

	def _make(self, k, v):
		return self.cls(self.character, k, self.node, **v)

	def _get_inner_map(self):
		try:
			return self.character.character.preportal[self.dest]
		except AttributeError:
			return {}


@define(eq=False)
class CharacterFacade(AbstractCharacter):
	engine: EngineFacade
	_name: CharName

	_rb_patch: dict = field(init=False, factory=dict)

	@property
	def character(self):
		if engine := getattr(self.engine, "_real", None):
			return engine.character[self._name]
		return None

	def __copy__(self):
		ret = CharacterFacade(engine=self.engine, name=self.name)
		ret.graph = self.StatMapping(ret)
		(
			ret.thing._patch,
			ret.place._patch,
			ret.portal._patch,
			ret.graph._patch,
		) = (
			self.thing.unwrap(),
			self.place.unwrap(),
			self.portal.unwrap(),
			self.graph.unwrap(),
		)
		return ret

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

	def remove_unit(
		self, a: Node | NodeName | KeyHint, b: FacadeNode | None = None
	):
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
				noden = NodeName(Key(b))
		branch, turn, tick = self.engine.time
		self.engine._unitness_cache.store(
			self.name, charn, noden, branch, turn, tick, False
		)

	def add_place(
		self,
		name: NodeName | KeyHint,
		**kwargs: dict[KeyHint, ValueHint]
		| dict[Stat | Literal["rulebook"], Value],
	) -> None:
		self.place[name] = FacadePlace(self, name, **kwargs)

	def add_node(
		self,
		name: NodeName | KeyHint,
		**kwargs: dict[KeyHint, ValueHint]
		| dict[Stat | Literal["rulebook"], Value],
	) -> None:
		"""Version of add_node that assumes it's a place"""
		self.add_place(name, **kwargs)

	def remove_node(self, node: NodeName | KeyHint) -> None:
		"""Version of remove_node that handles place or thing"""
		node = NodeName(Key(node))
		if node in self.thing:
			del self.thing[node]
		else:
			del self.place[node]

	def remove_place(self, place: NodeName | KeyHint) -> None:
		del self.place[place]

	def remove_thing(self, thing: NodeName | KeyHint) -> None:
		del self.thing[thing]

	def add_thing(
		self,
		name: NodeName | KeyHint,
		location: NodeName | KeyHint,
		**kwargs: dict[Stat, Value] | dict[KeyHint, ValueHint],
	) -> None:
		stats: dict[Stat | Literal["location"], Value] = {
			Stat(Key(k)): Value(v) for (k, v) in kwargs.items()
		}
		stats["location"] = Value(location)
		self.thing[name] = FacadeThing(self, name, **stats)

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
			self.name, charn, noden, *self.engine.time, True
		)

	def portals(self):
		for ds in self.portal.values():
			yield from ds.values()

	class UnitGraphMapping(Mapping[CharName, Mapping[NodeName, Node]]):
		class UnitMapping(Mapping[NodeName, Node]):
			def __init__(self, character, graph_name):
				self.character = character
				self.graph_name = graph_name

			def __iter__(self):
				for key in self.character.engine._unitness_cache.iter_keys(
					self.character.name,
					self.graph_name,
					*self.character.engine.time,
				):
					if key in self:
						yield key

			def __len__(self):
				return self.character.engine._unitness_cache.count_keys(
					self.character.name,
					self.graph_name,
					*self.character.engine.time,
				)

			def __contains__(self, item: NodeName | KeyHint):
				try:
					return self.character.engine._unitness_cache.retrieve(
						self.character.name,
						self.graph_name,
						item,
						*self.character.engine.time,
					)
				except KeyError:
					return False

			def __getitem__(self, item: NodeName | KeyHint):
				item = NodeName(item)
				if item not in self:
					if not self.character.engine._mock:
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

		def __init__(self, character: CharacterFacade):
			self.character = character

		def __iter__(self):
			engine = self.character.engine
			name = self.character.name
			now = self.character.engine.time
			for key in engine._unitness_cache.iter_keys(name, *now):
				if key in self:
					yield key

		def __len__(self):
			return self.character.engine._unitness_cache.count_keys(
				self.character.name, *self.character.engine.time
			)

		def __contains__(self, item: NodeName | KeyHint):
			now = self.character.engine.time
			name = self.character.name
			engine = self.character.engine
			try:
				engine._unitness_cache.retrieve(name, NodeName(item), *now)
				return True
			except KeyError:
				return False

		def __getitem__(self, item: NodeName | KeyHint):
			item = NodeName(item)
			if item not in self and not getattr(
				self.character.engine, "_mockup", None
			):
				raise KeyError(
					"Character has no units in graph",
					self.character.name,
					item,
				)
			return self.UnitMapping(self.character, item)

	class ThingMapping(FacadeEntityMapping):
		cls: ClassVar[type] = FacadeThing

		@cached_property
		def innercls(self):
			from .node import Thing

			return Thing

		def _make(self, k, v):
			return self.cls(self.character, k, location=v["location"])

		def _get_inner_map(self):
			try:
				return self.character.character.thing
			except AttributeError:
				return {}

		def patch(self, d: dict):
			places = d.keys() & self.character.place.keys()
			if places:
				raise KeyError(
					f"Tried to patch places on thing mapping: {places}"
				)
			self.character.node.patch(d)

	@cached_property
	def thing(self) -> ThingMapping:
		return self.ThingMapping(self)

	class PlaceMapping(FacadeEntityMapping):
		cls: ClassVar[type] = FacadePlace

		@cached_property
		def innercls(self):
			from .node import Place

			return Place

		def __eq__(self, other):
			return super().__eq__(other)

		def _make(self, k, v):
			return FacadePlace(self.character, k, **v)

		def _get_inner_map(self):
			if isinstance(
				self.character.character, nx.Graph
			) and not isinstance(self.character.character, AbstractCharacter):
				return self.character.character.node
			try:
				return self.character.character.place
			except AttributeError:
				return {}

		def patch(self, d: dict):
			things = d.keys() & self.character.thing.keys()
			if things:
				raise KeyError(
					f"Tried to patch things on place mapping: {things}"
				)
			self.character.node.patch(d)

	@cached_property
	def place(self) -> PlaceMapping:
		return self.PlaceMapping(self)

	def ThingPlaceMapping(
		self, *args
	) -> CompositeDict[NodeName, FacadePlace | FacadeThing]:
		return CompositeDict(self.place, self.thing)

	@cached_property
	def node(self) -> CompositeDict[NodeName, FacadePlace | FacadeThing]:
		return CompositeDict(self.place, self.thing)

	class PortalSuccessorsMapping(FacadePortalMapping):
		cls: ClassVar[type] = FacadePortalSuccessors

		def __contains__(self, item):
			return item in self.character.node

		def __eq__(self, other):
			return super().__eq__(other)

		def _make(self, k, v):
			ret = self.cls(self.character, k)
			ret.update(v)
			return ret

		def _get_inner_map(self):
			try:
				return self.character.character._adj
			except AttributeError:
				return {}

	@cached_property
	def adj(self) -> PortalSuccessorsMapping:
		return self.PortalSuccessorsMapping(self)

	class PortalPredecessorsMapping(FacadePortalMapping):
		cls = FacadePortalPredecessors

		def __contains__(self, item):
			return item in self.character._node

		def _get_inner_map(self):
			try:
				return self.character.character.pred
			except AttributeError:
				return {}

	@cached_property
	def pred(self) -> PortalPredecessorsMapping:
		return self.PortalPredecessorsMapping(self)

	class StatMapping(
		MappingUnwrapper[Stat, Value], MutableMapping[Stat, Value], Signal
	):
		def __init__(self, character):
			super().__init__()
			self.character = character
			self._patch = {}

		def copy(self):
			d = {}
			if hasattr(self.character.character, "graph"):
				for k, v in self.character.character.graph.items():
					if k not in self._patch:
						d[k] = v
					elif self._patch[k] is not ...:
						d[k] = self._patch[k]
			for k, v in self._patch.items():
				if v is not ...:
					d[k] = v
			return d

		def __iter__(self):
			if not self.character.character:
				return iter(self._patch)
			seen = set()
			if hasattr(self.character.character, "graph"):
				for k in self.character.character.graph:
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
				hasattr(self.character.character, "graph")
				and k in self.character.character.graph
			):
				return True
			return False

		def __getitem__(self, k):
			if k not in self._patch and hasattr(
				self.character.character, "graph"
			):
				ret = self.character.character.graph[k]
				if not hasattr(ret, "unwrap"):
					return ret
				self._patch[k] = ret.unwrap()
			if self._patch[k] is ...:
				return KeyError("masked", k)
			return self._patch[k]

		def __setitem__(self, k, v):
			if (
				hasattr(self.character, "engine")
				and self.character.engine._planning
			):
				self.character.engine._planned[
					self.character.character.engine._curplan
				][self.character.engine.turn].append(
					(self.character.name, k, v)
				)
				return
			self._patch[k] = v

		def __delitem__(self, k):
			self._patch[k] = ...

		def __repr__(self):
			toshow = {}
			if hasattr(self.character.character, "graph"):
				for k in (
					self._patch.keys() | self.character.character.graph.keys()
				):
					if k in self._patch:
						if self._patch[k] is not ...:
							toshow[k] = self._patch[k]
					elif k in self.character.character.graph:
						v = self.character.character.graph[k]
						if hasattr(v, "unwrap") and not hasattr(
							v, "no_unwrap"
						):
							v = v.unwrap()
						toshow[k] = v
			return f"<StatMapping {toshow}>"

	stat_map_cls = graph_map_cls = StatMapping

	def apply(self):
		"""Do all my changes for real in a batch"""
		realchar = self.character
		realstat = realchar.stat
		realthing = realchar.thing
		realplace = realchar.place
		realport = realchar.portal
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
		if hasattr(self.portal, "_tampered") and self.portal._tampered:
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


class EternalFacade(MutableMapping):
	def __init__(self, engine: EngineFacade):
		self.engine = engine
		self._patch = {}

	def __setitem__(self, key, value, /):
		self._patch[key] = value

	def __delitem__(self, key, /):
		real = getattr(self.engine, "_real", None)
		if real:
			if key not in real.eternal:
				raise KeyError("Key not set in underlying engine", key)
			self._patch[key] = ...
		else:
			del self._patch[key]

	def __getitem__(self, key, /):
		if key in self._patch:
			value = self._patch[key]
			if value is ...:
				raise KeyError("Value has been masked", key)
			return value
		real = getattr(self.engine, "_real", None)
		if not real:
			raise KeyError(
				"Value never set, and no underlying engine to find it in"
			)
		return real.eternal[key]

	def __len__(self):
		real = getattr(self.engine, "_real", None)
		if real:
			n = len(real.eternal)
			for key, value in self._patch.items():
				if value is ...:
					if key in real.eternal:
						n -= 1
				elif key not in real.eternal:
					n += 1
			return n
		else:
			return len(self._patch)

	def __iter__(self):
		real = getattr(self.engine, "_real", None)
		patch = self._patch
		if real:
			for k in real.eternal:
				if k not in patch or patch[k] is not ...:
					yield k
		else:
			yield from patch.keys()


@define
class EngineFacade(AbstractEngine):
	_real: AbstractEngine | None = field(alias="real")
	_mock: bool = field(alias="mock", default=False)
	_planning = field(default=False, init=False)
	_planned = field(
		factory=partial(defaultdict, partial(defaultdict, list)), init=False
	)
	_curplan: int = field(init=False, default=0)
	keyframe_interval: int | None = field(init=False, default=None)

	def _detect_kf_interval_override(self) -> bool | None:
		return False

	@staticmethod
	def _validate_branches_d(self, attr, branches_d):
		if self._real and hasattr(self._real, "_branches_d"):
			branches_d.update(self._real._branches_d)

	_branches_d: dict[
		Branch, tuple[Optional[Branch], Turn, Tick, Turn, Tick]
	] = field(
		factory=lambda: {
			Branch("trunk"): (None, Turn(0), Tick(0), Turn(0), Tick(0))
		},
		validator=_validate_branches_d,
		init=False,
	)

	def _char_exists(self, character: CharName) -> bool:
		if character in self.character._patch:
			return self.character._patch[character] is not ...
		if self._real:
			return self._real._char_exists(character)
		return False

	def _node_exists(self, character: CharName, node: NodeName) -> bool:
		if character in self.character._patch:
			cha = self.character._patch[character]
			if cha is ...:
				return False
			if not isinstance(cha, CharacterFacade):
				raise TypeError(
					"Invalid facade cache", self, self.character._patch, cha
				)
			if node in cha.place._patch or node in cha.thing._patch:
				return (
					cha.place._patch.get(node, ...) is not ...
					and cha.thing._patch.get(node, ...) is not ...
				)
		if self._real:
			return self._real._node_exists(character, node)
		return False

	def _edge_exists(
		self, character: CharName, orig: NodeName, dest: NodeName
	) -> bool:
		if character in self.character._patch:
			cha = self.character._patch[character]
			if cha is ...:
				return False
			if not isinstance(cha, CharacterFacade):
				raise TypeError(
					"Invalid facade cache", self, self.character._patch, cha
				)
			if orig not in cha.adj._patch:
				if self._real:
					return self._real._edge_exists(character, orig, dest)
				return False
			if dest not in cha.adj._patch[orig]._patch:
				if self._real:
					return self._real._edge_exists(character, orig, dest)
				return False
			return cha.adj._patch[orig]._patch[dest] is not ...
		if self._real:
			return self._real._edge_exists(character, orig, dest)
		return False

	@staticmethod
	def _validate_turn_end(self, attr, turn_end):
		if self._real and not self._real.is_proxy:
			turn_end.update(self._real._turn_end)

	_turn_end: TurnEndDict = field(
		default=Factory(TurnEndDict, takes_self=True),
		validator=_validate_turn_end,
		init=False,
	)

	@staticmethod
	def _validate_turn_end_plan(self, attr, turn_end_plan):
		if self._real and not self._real.is_proxy:
			turn_end_plan.update(self._real._turn_end_plan)

	_turn_end_plan = field(
		default=Factory(TurnEndPlanDict, takes_self=True), init=False
	)

	@staticmethod
	def _get_branch(self) -> Branch:
		try:
			return self._real.branch
		except AttributeError:
			return Branch("trunk")

	_obranch: Branch = field(
		default=Factory(_get_branch, takes_self=True), init=False
	)

	@staticmethod
	def _get_turn(self) -> Turn:
		try:
			return self._real.turn
		except AttributeError:
			return Turn(0)

	_oturn: Turn = field(
		default=Factory(_get_turn, takes_self=True), init=False
	)

	@staticmethod
	def _get_tick(self) -> Tick:
		try:
			return self._real.tick
		except AttributeError:
			return Tick(0)

	_otick: Tick = field(
		default=Factory(_get_tick, takes_self=True), init=False
	)
	closed: bool = field(default=False, init=False)

	char_cls: ClassVar = CharacterFacade
	thing_cls: ClassVar = FacadeThing
	place_cls: ClassVar = FacadePlace
	portal_cls: ClassVar = FacadePortal

	def _function_store_creator(self, attr: str) -> FunctionStore:
		if self._mock:
			return FunctionStore(None, module=attr)
		try:
			return getattr(self._real, attr)
		except AttributeError:
			return FunctionStore(None, module=attr)

	@cached_property
	def submit(self):
		if self._mock:
			return MagicMock()
		return self._real.submit

	@cached_property
	def load_at(self):
		if self._mock:
			return MagicMock()
		return self._real.load_at

	@cached_property
	def function(self) -> FunctionStore[str, FunctionType]:
		return self._function_store_creator("function")

	@cached_property
	def method(self) -> FunctionStore[str, MethodType]:
		return self._function_store_creator("method")

	@cached_property
	def trigger(self) -> TriggerStore:
		if self._mock:
			return TriggerStore(None, module="trigger")
		try:
			return self._real.trigger
		except AttributeError:
			return TriggerStore(None, module="trigger")

	@cached_property
	def prereq(self) -> FunctionStore[str, FunctionType]:
		return self._function_store_creator("prereq")

	@cached_property
	def action(self) -> FunctionStore[str, FunctionType]:
		return self._function_store_creator("action")

	@cached_property
	def string(self) -> StringStore:
		try:
			return self._real.string
		except AttributeError:
			return StringStore({}, None)

	@cached_property
	def _nodes_cache(self) -> FacadeNodesCache | None:
		if self._real and hasattr(self._real, "_nodes_cache"):
			return self.FacadeNodesCache(
				self._real._nodes_cache, "nodes_cache"
			)
		return None

	@cached_property
	def _unitness_cache(self) -> FacadeUnitnessCache | None:
		if self._real and hasattr(self._real, "_unitness_cache"):
			return self.FacadeUnitnessCache(self._real._unitness_cache)
		return None

	@cached_property
	def _things_cache(self) -> FacadeThingsCache | None:
		if self._real and hasattr(self._real, "_things_cache"):
			return self.FacadeThingsCache(
				self._real._things_cache, "things_cache"
			)
		return None

	@cached_property
	def world_lock(self):
		return RLock()

	@cached_property
	def eternal(self):
		return EternalFacade(self)

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

		def __getitem__(self, key: CharName | KeyHint, /):
			realeng = self.engine._real
			if key not in self._patch:
				if realeng:
					if key not in realeng.character:
						raise KeyError("No such character", key)
					fac = CharacterFacade(engine=self.engine, name=key)
				elif getattr(self.engine, "_mock", None):
					fac = CharacterFacade(engine=self.engine, name=key)
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

	class FacadeCache[*_PARENT, _ENTITY, _KEY, _VALUE, _KEYFRAME](
		Cache[*_PARENT, _ENTITY, _KEY, _VALUE, _KEYFRAME]
	):
		def __init__(self, cache, name):
			self._created = cache.engine.time
			super().__init__(cache.engine, name)
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
				forward = self._real.engine._forward
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

	class FacadeNodesCache(
		FacadeCache[CharName, NodeName, bool, dict[NodeName, bool]],
		NodesCache,
	): ...

	class FacadeThingsCache(
		FacadeCache[CharName, NodeName, NodeName, dict[NodeName, NodeName]],
		ThingsCache,
	): ...

	class FacadeUnitnessCache(
		FacadeCache[
			CharName,
			CharName,
			NodeName,
			bool,
			dict[CharName, dict[NodeName, bool]],
		],
		UnitnessCache,
	):
		def __init__(self, cache):
			self._created = cache.engine.time
			UnitnessCache.__init__(self, cache.engine, "unitness_cache")
			self.user_cache = EngineFacade.FacadeLeaderCache(
				cache.leader_cache, "leader_cache"
			)
			self._real = cache

	class FacadeLeaderCache(
		FacadeCache[
			CharName,
			NodeName,
			CharName,
			bool,
			dict[NodeName, frozenset[CharName]],
		],
		LeaderSetCache,
	): ...

	class FacadeBookmarkMapping(AbstractBookmarkMapping, UserDict):
		def __init__(self, engine: EngineFacade):
			self.engine = engine
			super().__init__()

		def __call__(self, key: KeyHint) -> None:
			self.data[key] = tuple(self.engine.time)

		def __setitem__(self, key, value, /):
			if len(value) != 3:
				raise TypeError("Not a full time with a branch", value)
			self.data[key] = validate_time(value)

		def __delitem__(self, key, /):
			del self.data[key]

		def __getitem__(self, key, /):
			return self.data[key]

		def __len__(self):
			return len(self.data)

		def __iter__(self):
			return iter(self.data)

	@property
	def bookmark(self) -> AbstractBookmarkMapping:
		return self.FacadeBookmarkMapping(self)

	@cached_property
	def character(self):
		return self.FacadeCharacterMapping(self)

	@property
	def universal(self):
		return self.FacadeUniversalMapping(self._real)

	@property
	def rulebook(
		self,
	) -> MutableMapping[KeyHint | RulebookName, FacadeRulebook]:
		return {}

	@property
	def rule(self) -> MutableMapping[KeyHint | RuleName, FacadeRule]:
		return {}

	@property
	def trunk(self) -> Branch:
		return self._real.trunk

	@property
	def branch(self) -> Branch:
		return Branch(self._obranch)

	@branch.setter
	def branch(self, branch: str | Branch) -> None:
		self._obranch = Branch(branch)

	@property
	def turn(self) -> Turn:
		return Turn(self._oturn)

	@turn.setter
	def turn(self, turn: int | Turn) -> None:
		self._oturn = Turn(turn)

	@property
	def tick(self) -> Tick:
		return Tick(self._otick)

	@tick.setter
	def tick(self, tick: int | Tick) -> None:
		self._otick = tick

	_rando: Random = field(factory=Random, init=False)

	@cached_property
	def logger(self) -> Logger:
		if self._real:
			return self._real.logger
		elif self._mock:
			import sys

			if "kivy" in sys.modules:
				from kivy.logger import Logger

				return Logger
		from logging import getLogger

		return getLogger("lisien")

	def handle(self, *args, **kwargs):
		print_call_sig("handle", *args, **kwargs)

	def _get_node(
		self, char: AbstractCharacter | CharName, node: NodeName
	) -> Node:
		return self.character[char].node[node]

	def _btt(self):
		return self.branch, self.turn, self.tick

	def _set_btt(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		(self.branch, self.turn, self.tick) = (branch, turn, tick)

	def _time_warp(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		self._set_btt(branch, turn, tick)

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
		"""Do all my batched changes on the underlying real engine."""
		if self._real is None:
			raise TypeError("Can't apply changes to nothing")
		from lisien import Engine

		if not isinstance(self._real, Engine):
			raise TypeError(
				"Currently, we can only apply changes to the core Lisien engine"
			)
		realeng: Engine = self._real
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
					# The ``store`` calls are all ``loading=True`` because that
					# disables keycaching. If you cache keys (stats and node
					# contents and stuff) big batches get really slow.
					for tup in self._planned[plan_num][turn]:
						if len(tup) == 3:
							char, k, v = tup
							now = realeng._nbtt()
							realeng._graph_val_cache.store(
								char, k, *now, v, loading=True
							)
							realeng.database.graph_val_set(char, k, *now, v)
						elif len(tup) == 4:
							char, node, k, v = tup
							now = realeng._nbtt()
							if realeng._nodes_cache.node_exists(
								char, node, *realeng.time
							):
								if k is ...:
									realeng._nodes_cache.store(
										char, node, *now, False, loading=True
									)
									realeng.database.exist_node(
										char, node, *now, False
									)
								elif k == "location":
									# assume the location really exists, since
									# it did while planning
									realeng._things_cache.store(
										char, node, *now, v, loading=True
									)
									realeng.database.set_thing_loc(
										char, node, *now, v
									)
								else:
									realeng._node_val_cache.store(
										char, node, k, *now, v, loading=True
									)
									realeng.database.node_val_set(
										char, node, k, *now, v
									)
							elif k == "location":
								if not realeng._nodes_cache.node_exists(
									char, node, *realeng.time
								):
									realeng._nodes_cache.store(
										char, node, *now, True, loading=True
									)
									realeng.database.exist_node(
										char, node, *now, True
									)
									now = realeng._nbtt()
								realeng._things_cache.store(
									char, node, *now, v, loading=True
								)
								realeng.database.set_thing_loc(
									char, node, *now, v
								)
							else:
								realeng._nodes_cache.store(
									char, node, *now, True, loading=True
								)
								realeng.database.exist_node(
									char, node, *now, True
								)
								now = realeng._nbtt()
								realeng._node_val_cache.store(
									char, node, k, *now, v, loading=True
								)
								realeng.database.node_val_set(
									char, node, k, *now, v
								)
						elif len(tup) == 5:
							char, orig, dest, k, v = tup
							now = realeng._nbtt()
							if not realeng._edges_cache.has_predecessor(
								char, orig, dest, *realeng.time
							):
								realeng._edges_cache.store(
									char, orig, dest, *now, True, loading=True
								)
								realeng.database.exist_edge(
									char, orig, dest, *now, True
								)
								now = realeng._nbtt()
							realeng._edge_val_cache.store(
								char, orig, dest, k, *now, v, loading=True
							)
							realeng.database.edge_val_set(
								char, orig, dest, k, *now, v
							)
						else:
							raise TypeError(
								"Not a valid change for a plan", tup
							)

	def close(self):
		self.apply()
