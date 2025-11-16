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

from functools import cached_property
from types import EllipsisType
from typing import (
	TYPE_CHECKING,
	Iterable,
	Iterator,
	Mapping,
	MutableMapping,
	Optional,
)

import networkx as nx
from blinker import Signal

from lisien.exc import AmbiguousLeaderError
from lisien.facade import CharacterFacade
from lisien.node import Place, Thing
from lisien.portal import Portal
from lisien.proxy.abc import (
	CachingEntityProxy,
	CachingProxy,
	RuleFollowerProxy,
)
from lisien.types import (
	AbstractCharacter,
	CharDelta,
	CharName,
	Edge,
	EdgeValDict,
	Key,
	KeyHint,
	Node,
	NodeName,
	NodeValDict,
	RulebookName,
	Stat,
	StatDict,
	Tick,
	Value,
	ValueHint,
)

from ..util import getatt

if TYPE_CHECKING:
	from .engine import EngineProxy


class ProxyLeaderMapping(Mapping):
	"""A mapping to the ``CharacterProxy``s that have this node as a unit"""

	def __init__(self, node: NodeProxy):
		self.node = node

	def __iter__(self):
		try:
			return iter(self._user_names())
		except KeyError:
			return iter(())

	def __len__(self):
		try:
			return len(self._user_names())
		except KeyError:
			return 0

	def __contains__(self, item: KeyHint) -> bool:
		try:
			return item in self._user_names()
		except KeyError:
			return False

	def __getitem__(self, item: KeyHint):
		if item not in self:
			raise KeyError("Not a leader of this node", item, self.node.name)
		return self.node.engine.character[item]

	@property
	def only(self) -> NodeProxy:
		if len(self) == 1:
			return next(iter(self.values()))
		raise AmbiguousLeaderError("No leaders, or more than one")

	def _user_names(self):
		return self.node.engine._unit_characters_cache[self.node._charname][
			self.node.name
		]


@Node.register
class NodeProxy(CachingEntityProxy, RuleFollowerProxy):
	name: NodeName

	@property
	def graph(self) -> CharacterProxy:
		return self.character

	@property
	def user(self) -> ProxyUserMapping:
		return ProxyUserMapping(self)

	@property
	def character(self) -> CharacterProxy:
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

	def _set_rulebook_name(self, rb: RulebookName):
		self.engine.handle(
			"set_node_rulebook",
			char=self._charname,
			node=self.name,
			rulebook=rb,
			branching=True,
		)
		self.engine._char_node_rulebooks_cache[self._charname][self.name] = rb

	def __init__(self, character: CharacterProxy, nodename: NodeName, **stats):
		self.engine = character.engine
		self._charname = character.name
		self.name = nodename
		self._cache.update(stats)
		super().__init__()

	def __eq__(self, other: NodeProxy):
		return (
			isinstance(other, NodeProxy)
			and self._charname == other._charname
			and self.name == other.name
		)

	def __contains__(self, k: KeyHint):
		if k in ("character", "name"):
			return True
		return super().__contains__(k)

	def __getitem__(self, k: KeyHint):
		if k == "character":
			return self._charname
		elif k == "name":
			return self.name
		return super().__getitem__(k)

	def _set_item(self, k: KeyHint, v: ValueHint):
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

	def _del_item(self, k: KeyHint):
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
		self._worker_check()
		self.engine.del_node(self._charname, self.name)

	@property
	def leader(self) -> ProxyLeaderMapping:
		return ProxyLeaderMapping(self)

	@property
	def content(self):
		return NodeContentProxy(self)

	def contents(self):
		return self.content.values()

	@property
	def portal(self):
		return NodePortalProxy(self)

	def portals(self):
		return self.portal.values()

	@property
	def preportal(self):
		return NodePreportalProxy(self)

	def preportals(self):
		return self.preportal.values()

	@property
	def neighbor(self):
		return ProxyNeighborMapping(self)

	def neighbors(self):
		return self.neighbor.values()

	def add_thing(self, name: KeyHint, **kwargs):
		return self.character.add_thing(NodeName(name), self.name, **kwargs)

	def new_thing(self, name: KeyHint, **kwargs):
		return self.character.new_thing(NodeName(name), self.name, **kwargs)

	def add_portal(self, dest: KeyHint, **kwargs):
		self.character.add_portal(self.name, NodeName(dest), **kwargs)

	def new_portal(self, dest: KeyHint, **kwargs):
		dest = getattr(dest, "name", dest)
		self.add_portal(dest, **kwargs)
		return self.character.portal[self.name][dest]

	def shortest_path(
		self, dest: KeyHint | NodeProxy, weight: Key = None
	) -> list[NodeName]:
		"""Return a list of node names leading from me to ``dest``.

		Raise ``ValueError`` if ``dest`` is not a node in my character
		or the name of one.

		"""
		return nx.shortest_path(
			self.character, self.name, self._plain_dest_name(dest), weight
		)

	def _plain_dest_name(self, dest: KeyHint | NodeProxy) -> NodeName:
		if isinstance(dest, NodeProxy):
			if dest.character != self.character:
				raise ValueError(
					"{} not in {}".format(dest.name, self.character.name)
				)
			return dest.name
		else:
			if dest in self.character.node:
				return NodeName(Key(dest))
			raise ValueError("{} not in {}".format(dest, self.character.name))


@Place.register
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
			elif k == "location":
				if v is ...:
					if (
						self.character.name in self.engine._things_cache
						and self.name
						in self.engine._things_cache[self.character.name]
					):
						del self.engine._things_cache[self.character.name][
							self.name
						]
					self.engine._character_places_cache[self.character.name][
						self.name
					] = self
				else:
					self.engine._things_cache[self.character.name][
						self.name
					] = ThingProxy(self.character, self.name, v, **self._cache)
					del self.engine._character_places_cache[
						self.character.name
					][self.name]
				continue
			if v is ...:
				if k in self._cache:
					del self._cache[k]
					self.send(self, key=k, value=...)
					self.character.place.send(self, key=k, value=...)
					self.character.node.send(self, key=k, value=...)
			elif k not in self._cache or self._cache[k] != v:
				self._cache[k] = v
				self.send(self, key=k, value=v)
				self.character.place.send(self, key=k, value=v)
				self.character.node.send(self, key=k, value=v)


@Thing.register
class ThingProxy(NodeProxy):
	@property
	def location(self) -> NodeProxy:
		return self.engine.character[self._charname].node[self._location]

	@location.setter
	def location(self, v: NodeProxy | KeyHint | None):
		if isinstance(v, NodeProxy):
			if v.character != self.character:
				raise ValueError(
					"Things can only be located in their character. "
					"Maybe you want a unit?"
				)
			locn = v.name
		elif v in self.character.node or v is None:
			locn = v
		else:
			raise TypeError("Location must be a node or the name of one")
		self._set_location(locn)

	def __init__(
		self,
		character: CharacterProxy,
		name: NodeName,
		location: NodeName | None = None,
		**kwargs: StatDict,
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

	def __getitem__(self, k: KeyHint):
		if k == "location":
			return self._location
		return super().__getitem__(k)

	def _apply_delta(self, delta: dict[KeyHint, ValueHint]):
		for k, v in delta.items():
			if k == "rulebook":
				if v is ...:
					cnrc = self.engine._char_node_rulebooks_cache
					if (
						self._charname in cnrc
						and self.name in cnrc[self._charname]
					):
						del cnrc[self._charname][self.name]
					self.send(self, key="rulebook", value=...)
					self.character.thing.send(self, key="rulebook", value=...)
					self.character.node.send(self, key="rulebook", value=...)
				elif v != self.rulebook.name:
					self.engine._char_node_rulebooks_cache[self._charname][
						self.name
					] = v
					self.send(self, key="rulebook", value=v)
					self.character.thing.send(self, key="rulebook", value=v)
					self.character.node.send(self, key="rulebook", value=v)
			elif v is ...:
				if k == "location":
					del self.engine._things_cache[self.character.name][
						self.name
					]
					self.engine._character_places_cache[self.character.name][
						self.name
					] = PlaceProxy(self.character, self.name, **self._cache)
					continue
				if k in self._cache:
					del self._cache[k]
					self.send(self, key=k, value=...)
					self.character.thing.send(self, key=k, value=...)
					self.character.node.send(self, key=k, value=...)
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

	def _set_location(self, v: NodeName | None):
		if v is None:
			del self.engine._things_cache[self.character.name][self.name]
			self.engine._character_places_cache[self.character.name][
				self.name
			] = PlaceProxy(self.character, self.name, **self._cache)
		self._location = v
		self.engine.handle(
			command="set_thing_location",
			char=self.character.name,
			thing=self.name,
			loc=v,
			branching=True,
		)

	def __setitem__(self, k: KeyHint, v: ValueHint):
		self._worker_check()
		if k == "location":
			self._set_location(NodeName(Key(v)))
		elif k == "rulebook":
			self._set_rulebook_name(RulebookName(Key(v)))
		else:
			super().__setitem__(k, v)
		self.send(self, key=k, value=v)
		self.character.thing.send(self, key=k, value=v)
		self.character.node.send(self, key=k, value=v)

	def __repr__(self):
		return "<proxy to {}.thing[{}]@{} at {}>".format(
			self._charname, self.name, self._location, id(self)
		)

	def follow_path(
		self, path: list[NodeName], weight: Stat | EllipsisType = ...
	):
		self._worker_check()
		self.engine.handle(
			command="thing_follow_path",
			char=self._charname,
			thing=self.name,
			path=path,
			weight=weight,
		)

	def go_to_place(
		self, place: NodeProxy | NodeName, weight: Stat | EllipsisType = ...
	):
		self._worker_check()
		if hasattr(place, "name"):
			place = place.name
		self.engine.handle(
			command="thing_go_to_place",
			char=self._charname,
			thing=self.name,
			place=place,
			weight=weight,
		)

	def travel_to(
		self,
		dest: NodeProxy,
		weight: KeyHint | Stat | EllipsisType = ...,
		graph: KeyHint | CharName | EllipsisType = ...,
	):
		self._worker_check()
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

	def travel_to_by(
		self,
		dest: NodeProxy,
		arrival_tick: Tick,
		weight: KeyHint | Stat | EllipsisType = ...,
		graph: KeyHint | CharName | EllipsisType = ...,
	):
		self._worker_check()
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


@Portal.register
class PortalProxy(CachingEntityProxy, RuleFollowerProxy):
	@property
	def orig(self):
		return self._origin

	@property
	def dest(self):
		return self._destination

	def _apply_delta(self, delta):
		for k, v in delta.items():
			if k == "rulebook":
				if v != self.rulebook.name:
					self.engine._char_port_rulebooks_cache[self._charname][
						self._origin
					][self._destination] = v
				continue
			if v is ...:
				if k in self._cache:
					del self._cache[k]
					self.send(self, key=k, value=...)
					self.character.portal.send(self, key=k, value=...)
			elif k not in self._cache or self._cache[k] != v:
				self._cache[k] = v
				self.send(self, key=k, value=v)
				self.character.portal.send(self, key=k, value=v)

	def _get_default_rulebook_name(self) -> RulebookName:
		return RulebookName(
			Key((self._charname, self._origin, self._destination))
		)

	def _get_rulebook_name(self) -> RulebookName:
		return self.engine._char_port_rulebooks_cache[self._charname][
			self._origin
		][self._destination]

	def _set_rulebook_name(self, rb: RulebookName) -> None:
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
	def _cache(self) -> StatDict:
		return self.engine._portal_stat_cache[self._charname][self._origin][
			self._destination
		]

	@property
	def character(self) -> CharacterProxy:
		return self.engine.character[self._charname]

	@property
	def graph(self) -> CharacterProxy:
		return self.engine.character[self._charname]

	@property
	def origin(self) -> NodeProxy:
		return self.character.node[self._origin]

	@property
	def destination(self) -> NodeProxy:
		return self.character.node[self._destination]

	@property
	def reciprocal(self) -> PortalProxy:
		if (
			self._origin not in self.character.pred
			or self._destination not in self.character.pred[self._origin]
		):
			return None
		return self.character.pred[self._origin][self._destination]

	def _set_item(self, k: Stat, v: Value) -> None:
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

	def _del_item(self, k: Stat) -> None:
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

	def __init__(
		self, character: CharacterProxy, origname: NodeName, destname: NodeName
	):
		self.engine = character.engine
		self._charname = character.name
		self._origin = origname
		self._destination = destname
		super().__init__()

	def __eq__(self, other: Edge):
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

	def __getitem__(self, k: Stat) -> NodeName | CharName | Value:
		if k == "origin":
			return self._origin
		elif k == "destination":
			return self._destination
		elif k == "character":
			return self._charname
		return super().__getitem__(k)

	def delete(self) -> None:
		self._worker_check()
		self.engine.del_portal(self._charname, self._origin, self._destination)


class NodeMapProxy(MutableMapping, Signal):
	@property
	def character(self):
		return self.engine.character[self._charname]

	def __init__(self, engine_proxy: "EngineProxy", charname: CharName):
		super().__init__()
		self.engine = engine_proxy
		self._charname = charname

	def __iter__(self):
		yield from self.character.thing
		yield from self.character.place

	def __len__(self):
		return len(self.character.thing) + len(self.character.place)

	def __getitem__(self, k: NodeName) -> ThingProxy | PlaceProxy:
		if k in self.character.thing:
			return self.character.thing[k]
		else:
			return self.character.place[k]

	def __setitem__(self, k: NodeName, v: StatDict) -> None:
		self.engine._worker_check()
		self.character.place[k] = v

	def __delitem__(self, k: NodeName) -> None:
		self.engine._worker_check()
		if k in self.character.thing:
			del self.character.thing[k]
		else:
			del self.character.place[k]

	def patch(self, patch: NodeValDict):
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
	def _get_default_rulebook_name(self) -> RulebookName:
		return RulebookName(Key(("character_thing_rulebook", self.name)))

	def _get_rulebook_name(self) -> RulebookName:
		return self.engine._character_rulebooks_cache[self.name]["thing"]

	def _set_rulebook_name(self, rb: RulebookName) -> None:
		self.engine.handle(
			"set_character_thing_rulebook",
			char=self.name,
			rulebook=rb,
			branching=True,
		)
		self.engine._character_rulebooks_cache[self.name]["thing"] = rb

	def _apply_delta(self, delta: NodeValDict) -> None:
		raise NotImplementedError("_apply_delta")

	@property
	def character(self) -> CharacterProxy:
		return self.engine.character[self.name]

	@property
	def _cache(self):
		return self.engine._things_cache.setdefault(self.name, {})

	def __init__(self, engine_proxy: "EngineProxy", charname: CharName):
		self.engine: "EngineProxy" = engine_proxy
		self.name: CharName = charname
		super().__init__()

	def __eq__(self, other):
		return self is other

	def _cache_set_munge(self, k: NodeName, v: StatDict):
		return ThingProxy(
			self.character,
			*self.engine.handle(
				"get_thing_special_stats", char=self.name, thing=k
			),
		)

	def _set_item(self, k: NodeName, v: StatDict):
		self.engine.handle(
			command="set_thing",
			char=self.name,
			thing=k,
			statdict=v,
			branching=True,
		)
		self._cache[k] = ThingProxy(
			self.character, k, NodeName(Key(v.pop("location")))
		)
		self.engine._node_stat_cache[self.name][k] = v

	def _del_item(self, k: Stat) -> None:
		self.engine.handle(
			command="del_node", char=self.name, node=k, branching=True
		)
		del self.engine._node_stat_cache[self.name][k]

	def patch(self, d: NodeValDict) -> None:
		self._worker_check()
		places = d.keys() & self.character.place.keys()
		if places:
			raise KeyError(f"Tried to patch places on thing mapping: {places}")
		self.character.node.patch(d)


class PlaceMapProxy(CachingProxy, RuleFollowerProxy):
	def _get_default_rulebook_name(self) -> RulebookName:
		return RulebookName(
			Key(
				(
					"character_place_rulebook",
					self.name,
				)
			)
		)

	def _get_rulebook_name(self) -> RulebookName:
		return self.engine._character_rulebooks_cache[self.name]["place"]

	def _set_rulebook_name(self, rb: RulebookName) -> None:
		self.engine.handle(
			"set_character_place_rulebook",
			char=self.name,
			rulebook=rb,
			branching=True,
		)
		self.engine._character_rulebooks_cache[self.name]["place"] = rb

	def _apply_delta(self, delta: NodeValDict) -> None:
		raise NotImplementedError("_apply_delta")

	@property
	def character(self) -> CharacterProxy:
		return self.engine.character[self.name]

	@property
	def _cache(self) -> StatDict:
		return self.engine._character_places_cache.setdefault(self.name, {})

	def __init__(self, engine_proxy: "EngineProxy", character: CharName):
		self.engine: "EngineProxy" = engine_proxy
		self.name: CharName = character
		super().__init__()

	def __eq__(self, other):
		return self is other

	def _cache_set_munge(self, k: NodeName, v: StatDict) -> PlaceProxy:
		return PlaceProxy(self.character, k)

	def _set_item(self, k: NodeName, v: StatDict) -> None:
		self.engine.handle(
			command="set_place",
			char=self.name,
			place=k,
			statdict=v,
			branching=True,
		)
		self.engine._node_stat_cache[self.name][k] = v

	def _del_item(self, k: Stat) -> None:
		self.engine.handle(
			command="del_node", char=self.name, node=k, branching=True
		)
		del self.engine._node_stat_cache[self.name][k]

	def patch(self, d: NodeValDict) -> None:
		self._worker_check()
		things = d.keys() & self.character.thing.keys()
		if things:
			raise KeyError(f"Tried to patch things on place mapping: {things}")
		self.character.node.patch(d)


class SuccessorsProxy(CachingProxy):
	@property
	def _cache(self):
		succ = self.engine._character_portals_cache.successors
		return succ.setdefault(self._charname, {}).setdefault(self._orig, {})

	def _set_rulebook_name(self, k: RulebookName):
		raise NotImplementedError(
			"Set the rulebook on the .portal attribute, not this"
		)

	def __init__(
		self,
		engine_proxy: "EngineProxy",
		charname: CharName,
		origname: NodeName,
	):
		self.engine = engine_proxy
		self._charname = charname
		self._orig = origname
		super().__init__()

	def __eq__(self, other: SuccessorsProxy):
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

	def _cache_set_munge(self, k: NodeName, v: PortalProxy | StatDict):
		if isinstance(v, PortalProxy):
			assert v._origin == self._orig
			assert v._destination == k
			return v
		return PortalProxy(
			self.engine.character[self._charname], self._orig, k
		)

	def _set_item(self, dest: NodeName, value: StatDict) -> None:
		self.engine.handle(
			command="set_portal",
			char=self._charname,
			orig=self._orig,
			dest=dest,
			statdict=value,
			branching=True,
		)

	def _del_item(self, dest: NodeName) -> None:
		self.engine.del_portal(self._charname, self._orig, dest)


class CharSuccessorsMappingProxy(CachingProxy, RuleFollowerProxy):
	def _get_default_rulebook_name(self) -> RulebookName:
		return "character_portal_rulebook", self.name

	def _get_rulebook_name(self) -> RulebookName:
		return self.engine._character_rulebooks_cache[self.name]["portal"]

	def _set_rulebook_name(self, rb: RulebookName) -> None:
		self.engine.handle(
			"set_character_portal_rulebook",
			char=self.character.name,
			rulebook=rb,
			branching=True,
		)
		self.engine._character_rulebooks_cache[self.name]["portal"] = rb

	@property
	def character(self) -> CharacterProxy:
		return self.engine.character[self.name]

	@property
	def _cache(self) -> dict[NodeName, dict[NodeName, PortalProxy]]:
		return self.engine._character_portals_cache.successors.setdefault(
			self.name, {}
		)

	def __init__(self, engine_proxy: "EngineProxy", charname: CharName):
		self.engine: "EngineProxy" = engine_proxy
		self.name: CharName = charname
		super().__init__()

	def __eq__(self, other: CharSuccessorsMappingProxy):
		return (
			isinstance(other, CharSuccessorsMappingProxy)
			and other.engine is self.engine
			and other.name == self.name
		)

	def _cache_set_munge(self, k: None, v: dict[NodeName, NodeName]):
		return {
			vk: PortalProxy(self.character, vk, vv) for (vk, vv) in v.items()
		}

	def __contains__(self, k: NodeName):
		return k in self.character.node

	def __getitem__(self, k: NodeName) -> SuccessorsProxy:
		if k not in self.character.node:
			raise KeyError("No such node in this character", self.name, k)
		return SuccessorsProxy(self.engine, self.name, k)

	def _apply_delta(self, delta: EdgeValDict) -> None:
		for o, ds in delta.items():
			cache = self._cache[o]
			for d, stats in ds.items():
				if d not in cache:
					cache[d] = PortalProxy(self.character, o, d)
				cache[d]._apply_delta(stats)

	def _set_item(self, orig: NodeName, val: dict[NodeName, StatDict]) -> None:
		self.engine.handle(
			command="character_set_node_successors",
			character=self.name,
			node=orig,
			val=val,
			branching=True,
		)

	def _del_item(self, orig: NodeName) -> None:
		for dest in self[orig]:
			self.engine.del_portal(self.name, orig, dest)


class PredecessorsProxy(MutableMapping):
	@property
	def character(self) -> CharacterProxy:
		return self.engine.character[self._charname]

	def _worker_check(self):
		self.engine._worker_check()

	def __init__(
		self,
		engine_proxy: "EngineProxy",
		charname: CharName,
		destname: NodeName,
	):
		self.engine: "EngineProxy" = engine_proxy
		self._charname: CharName = charname
		self.name: NodeName = destname

	def __iter__(self) -> Iterator[NodeName]:
		preds = self.engine._character_portals_cache.predecessors
		if (
			self._charname not in preds
			or self.name not in preds[self._charname]
		):
			return iter(())
		return iter(preds[self._charname][self.name])

	def __len__(self):
		preds = self.engine._character_portals_cache.predecessors
		if (
			self._charname not in preds
			or self.name not in preds[self._charname]
		):
			return 0
		return len(preds[self._charname][self.name])

	def __contains__(self, k: NodeName):
		preds = self.engine._character_portals_cache.predecessors
		return (
			self._charname in preds
			and self.name in preds[self._charname]
			and k in preds[self._charname][self.name]
		)

	def __getitem__(self, k: NodeName):
		return self.engine._character_portals_cache.predecessors[
			self._charname
		][self.name][k]

	def __setitem__(self, k: NodeName, v: dict[NodeName, StatDict]):
		self._worker_check()
		self.engine._character_portals_cache.store(
			self._charname,
			self.name,
			k,
			PortalProxy(self.engine.character[self._charname], k, self.name),
		)
		self.engine.handle(
			command="set_place",
			char=self._charname,
			place=k,
			statdict=v,
			branching=True,
		)
		self.engine.handle(
			"set_portal",
			char=self._charname,
			orig=self.name,
			dest=k,
			value=v,
			branching=True,
		)

	def __delitem__(self, k: NodeName) -> None:
		self.engine.del_portal(self._charname, k, self.name)


class CharPredecessorsMappingProxy(MutableMapping, Signal):
	@property
	def _cache(self) -> dict[NodeName, dict[NodeName, PortalProxy]]:
		return self.engine._character_portals_cache.predecessors.setdefault(
			self.name, {}
		)

	def _worker_check(self) -> None:
		self.engine._worker_check()

	def __init__(self, engine_proxy: "EngineProxy", charname: CharName):
		super().__init__()
		self.engine: "EngineProxy" = engine_proxy
		self.name: CharName = charname
		self._obj_cache = {}

	def __contains__(self, k: NodeName):
		return k in self.engine.character[self.name].node

	def __iter__(self) -> Iterator[NodeName]:
		try:
			return iter(
				self.engine._character_portals_cache.predecessors[self.name]
			)
		except KeyError:
			return iter(())

	def __len__(self):
		try:
			return len(
				self.engine._character_portals_cache.predecessors[self.name]
			)
		except KeyError:
			return 0

	def __getitem__(self, k: NodeName) -> PredecessorsProxy:
		if k not in self:
			raise KeyError("No such node in this character", self.name, k)
		if k not in self._obj_cache:
			self._obj_cache[k] = PredecessorsProxy(self.engine, self.name, k)
		return self._obj_cache[k]

	def __setitem__(self, k: NodeName, v: dict[NodeName, PortalProxy]):
		self._worker_check()
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

	def __delitem__(self, k: NodeName):
		self._worker_check()
		for v in list(self[k]):
			self.engine.del_portal(self.name, v, k)


class CharStatProxy(CachingEntityProxy):
	@property
	def _cache(self):
		return self.engine._char_stat_cache[self.name]

	def __init__(self, character: CharacterProxy):
		self.engine = character.engine
		self.name = character.name
		super().__init__()

	def __eq__(self, other: dict):
		if not hasattr(other, "keys") or not callable(other.keys):
			return False
		if self.keys() != other.keys():
			return False
		for k, v in self.items():
			if hasattr(v, "unwrap"):
				v = v.unwrap()
			if v != other[k]:
				return False
		return True

	def unwrap(self):
		return dict(self)

	def _set_rulebook_name(self, k: RulebookName):
		raise NotImplementedError(
			"Set rulebooks on the Character proxy, not this"
		)

	def _get(self, k: Optional[Key] = None):
		if k is None:
			return self
		return self._cache[k]

	def _set_item(self, k: Key, v: Value):
		if k == "name":
			raise KeyError("Can't change names")
		self.engine.handle(
			command="set_character_stat",
			char=self.name,
			k=k,
			v=v,
			branching=True,
		)

	def _del_item(self, k: Key):
		self.engine.handle(
			command="del_character_stat", char=self.name, k=k, branching=True
		)

	def _apply_delta(self, delta: StatDict):
		for k, v in delta.items():
			assert k != "rulebook"
			if v is ...:
				if k in self._cache:
					del self._cache[k]
					self.send(self, key=k, value=None)
			elif k not in self._cache or self._cache[k] != v:
				self._cache[k] = v
				self.send(self, key=k, value=v)


class UnitMapProxy(Mapping, RuleFollowerProxy, Signal):
	engine = getatt("character.engine")

	def _get_default_rulebook_name(self) -> RulebookName:
		return RulebookName(Key(("unit_rulebook", self.character.name)))

	def _get_rulebook_name(self) -> RulebookName:
		return self.engine._character_rulebooks_cache[self.character.name][
			"unit"
		]

	def _set_rulebook_name(self, rb: RulebookName) -> None:
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
	def only(self) -> NodeProxy:
		if len(self) == 0:
			raise AttributeError("No units")
		elif len(self) > 1:
			raise AttributeError("Units in more than one graph")
		return next(iter(self.values()))

	def __init__(self, character: CharacterProxy):
		super().__init__()
		self.character = character

	def __iter__(self) -> Iterator[NodeProxy]:
		yield from self.character.engine._character_units_cache[
			self.character.name
		]

	def __len__(self):
		return len(
			self.character.engine._character_units_cache[self.character.name]
		)

	def __contains__(self, k: CharName):
		return (
			k
			in self.character.engine._character_units_cache[
				self.character.name
			]
		)

	def __getitem__(self, k: CharName) -> GraphUnitsProxy:
		if k not in self:
			raise KeyError(
				"{} has no unit in {}".format(self.character.name, k)
			)
		return self.GraphUnitsProxy(
			self.character, self.character.engine.character[k]
		)

	class GraphUnitsProxy(Mapping):
		def __init__(self, character: CharacterProxy, graph: CharacterProxy):
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

		def __contains__(self, k: NodeName):
			cache = self.character.engine._character_units_cache[
				self.character.name
			]
			return self.graph.name in cache and k in cache[self.graph.name]

		def __getitem__(self, k: NodeName):
			if k not in self:
				raise KeyError(
					"{} has no unit {} in graph {}".format(
						self.character.name, k, self.graph.name
					)
				)
			return self.graph.node[k]

		@property
		def only(self) -> NodeProxy:
			if len(self) != 1:
				raise AttributeError("No unit, or more than one")
			return next(iter(self.values()))


class CharacterProxy(AbstractCharacter, RuleFollowerProxy):
	adj_cls = CharSuccessorsMappingProxy
	pred_cls = CharPredecessorsMappingProxy
	graph_map_cls = CharStatProxy

	def copy_from(self, g: AbstractCharacter) -> None:
		self._worker_check()
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

	def _get_default_rulebook_name(self) -> RulebookName:
		return RulebookName(Key(("character_rulebook", self.name)))

	def _get_rulebook_name(self) -> RulebookName:
		return self.engine._character_rulebooks_cache[self.name]["character"]

	def _set_rulebook_name(self, rb: RulebookName) -> None:
		self.engine.handle(
			"set_character_rulebook",
			char=self.name,
			rulebook=rb,
			branching=True,
		)
		self.engine._character_rulebooks_cache[self.name]["character"] = rb

	@cached_property
	def unit(self) -> UnitMapProxy:
		return UnitMapProxy(self)

	@staticmethod
	def PortalSuccessorsMapping(self) -> CharSuccessorsMappingProxy:
		return CharSuccessorsMappingProxy(self.engine, self.name)

	@staticmethod
	def PortalPredecessorsMapping(self) -> CharPredecessorsMappingProxy:
		return CharPredecessorsMappingProxy(self.engine, self.name)

	@staticmethod
	def ThingMapping(self) -> ThingMapProxy:
		return ThingMapProxy(self.engine, self.name)

	@staticmethod
	def PlaceMapping(self) -> PlaceMapProxy:
		return PlaceMapProxy(self.engine, self.name)

	@staticmethod
	def ThingPlaceMapping(self) -> NodeMapProxy:
		return NodeMapProxy(self.engine, self.name)

	def __init__(
		self,
		engine: EngineProxy,
		name: CharName,
		*,
		init_rulebooks: bool = False,
	):
		assert not init_rulebooks, (
			"Can't initialize rulebooks in CharacterProxy"
		)
		self.engine = engine
		self._name = name

	def __repr__(self):
		return f"{self.engine}.character[{repr(self.name)}]"

	def __bool__(self):
		return self._name in self.engine.character

	def __eq__(self, other):
		if hasattr(other, "engine"):
			return (
				self.engine is other.engine
				and hasattr(other, "name")
				and self.name == other.name
			)
		else:
			return False

	def _apply_delta(self, delta: CharDelta):
		delta = delta.copy()
		deleted_nodes = set()
		for node, ex in delta.pop("nodes", {}).items():
			if ex:
				if node not in self.node:
					nodeval = delta.get("node_val", {}).get(node, None)
					if nodeval and "location" in nodeval:
						self.thing._cache[node] = prox = ThingProxy(
							self, node, NodeName(Key(nodeval["location"]))
						)
						self.thing.send(prox, key=None, value=True)
					else:
						self.place._cache[node] = prox = PlaceProxy(self, node)
						self.place.send(prox, key=None, value=True)
					self.node.send(prox, key=None, value=True)
			elif node in self.node:
				deleted_nodes.add(node)
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
			else:
				deleted_nodes.add(node)
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
		nodedelts: NodeValDict = delta.pop("node_val", {})
		for node, nodedelta in nodedelts.items():
			if node in deleted_nodes:
				continue
			elif node not in node_stat_cache[name]:
				rulebook = nodedelta.pop("rulebook", None)
				node_stat_cache[name][node] = nodedelta
				if rulebook:
					nodemap[node]._set_rulebook_name(rulebook)
			else:
				nodemap[node]._apply_delta(nodedelta)
		portmap = self.portal
		portal_stat_cache = self.engine._portal_stat_cache
		portdelts: EdgeValDict = delta.pop("edge_val", {})
		for orig, destdelta in portdelts.items():
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
		ruc = self.engine._character_rulebooks_cache[name]
		if "character_rulebook" in delta:
			ruc["character"] = delta.pop("character_rulebook")
		if "unit_rulebook" in delta:
			ruc["unit"] = delta.pop("unit_rulebook")
		if "character_thing_rulebook" in delta:
			ruc["thing"] = delta.pop("character_thing_rulebook")
		if "character_place_rulebook" in delta:
			ruc["place"] = delta.pop("character_place_rulebook")
		if "character_portal_rulebook" in delta:
			ruc["portal"] = delta.pop("character_portal_rulebook")
		units = delta.pop("units", {})
		unit_graphs = self.engine._character_units_cache[self.name]
		for unit_graph in units:
			graph_units = unit_graphs.setdefault(unit_graph, {})
			for unit, is_unit in units[unit_graph].items():
				if is_unit:
					graph_units[unit] = True
				elif unit in graph_units:
					del graph_units[unit]
		self.stat._apply_delta(delta)

	def add_place(self, name: NodeName, **kwargs: StatDict) -> None:
		self._worker_check()
		self.engine.handle(
			command="set_place",
			char=self.name,
			place=name,
			statdict=kwargs,
			branching=True,
		)
		self.place._cache[name] = PlaceProxy(self, name)
		self.engine._node_stat_cache[self.name][name] = kwargs

	def add_places_from(self, seq: Iterable[NodeName]):
		self._worker_check()
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

	def add_nodes_from(self, seq: Iterable[NodeName], **attrs):
		self._worker_check()
		self.add_places_from(seq)

	def add_thing(
		self, name: NodeName, location: NodeName, **kwargs: StatDict
	) -> None:
		self._worker_check()
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

	def _worker_check(self):
		self.engine._worker_check()

	def add_things_from(self, seq: Iterable[NodeName], **attrs) -> None:
		self._worker_check()
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

	def place2thing(self, place: NodeName, location: NodeName) -> None:
		self._worker_check()
		self.engine.handle(
			command="place2thing",
			char=self.name,
			place=place,
			loc=location,
			branching=True,
		)
		if place in self.place._cache:
			del self.place._cache[place]
		self.place.send(place, key=None, value=False)
		if place not in self.thing._cache:
			self.thing._cache[place] = ThingProxy(self, place, location)
		self.thing.send(place, key=None, value=True)

	def thing2place(self, thing: NodeName) -> None:
		self._worker_check()
		self.engine.handle(
			command="thing2place",
			char=self.name,
			thing=thing,
			branching=True,
		)
		if thing in self.thing._cache:
			del self.thing._cache[thing]
		self.thing.send(thing, key=None, value=False)
		if thing not in self.place._cache:
			self.place._cache[thing] = PlaceProxy(self, thing)
		self.place.send(thing, key=None, value=True)

	def remove_node(self, node: NodeName) -> None:
		self._worker_check()
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
		to_del = set()
		if (
			name in portscache.successors
			and node in portscache.successors[name]
		):
			to_del.update(
				(node, dest) for dest in portscache.successors[name][node]
			)
		if (
			name in portscache.predecessors
			and node in portscache.predecessors[name]
		):
			to_del.update(
				(orig, node) for orig in portscache.predecessors[name][node]
			)
		for u, v in to_del:
			portscache.delete(name, u, v)
		if (
			name in portscache.successors
			and node in portscache.successors[name]
		):
			del portscache.successors[name][node]
		if name in portscache.successors and not portscache.successors[name]:
			del portscache.successors[name]
		if (
			name in portscache.predecessors
			and node in portscache.predecessors[name]
		):
			del portscache.predecessors[name][node]
		if (
			name in portscache.predecessors
			and not portscache.predecessors[name]
		):
			del portscache.predecessors[name]

	def remove_place(self, place: NodeName) -> None:
		self._worker_check()
		placemap = self.place
		if place not in placemap:
			raise KeyError("No such place: {}".format(place))
		name = self.name
		self.engine.handle("del_node", char=name, node=place, branching=True)
		del placemap._cache[place]
		portscache = self.engine._character_portals_cache
		del portscache.successors[name][place]
		del portscache.predecessors[name][place]

	def remove_thing(self, thing: NodeName) -> None:
		self._worker_check()
		thingmap = self.thing
		if thing not in thingmap:
			raise KeyError("No such thing: {}".format(thing))
		name = self.name
		self.engine.handle("del_node", char=name, node=thing, branching=True)
		del thingmap._cache[thing]
		portscache = self.engine._character_portals_cache
		del portscache.successors[name][thing]
		del portscache.predecessors[name][thing]

	def add_portal(
		self, origin: NodeName, destination: NodeName, **kwargs: StatDict
	) -> None:
		self._worker_check()
		symmetrical = kwargs.pop("symmetrical", False)
		origin = getattr(origin, "name", origin)
		destination = getattr(destination, "name", destination)
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

	def remove_portal(self, origin: NodeName, destination: NodeName) -> None:
		self._worker_check()
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

	def add_portals_from(
		self, seq: Iterable[tuple[NodeName, NodeName]], symmetrical=False
	):
		self._worker_check()
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

	def portals(self) -> Iterator[PortalProxy]:
		for ds in self.portal.values():
			yield from ds.values()

	def add_unit(
		self, graph: CharName | CharacterProxy, node: Optional[NodeName] = None
	) -> None:
		self._worker_check()
		if node is None:
			node = graph.name
			graph = graph.character.name
		self.engine._character_units_cache[self.name].setdefault(
			graph, set()
		).add(node)
		self.engine._unit_characters_cache[graph].setdefault(node, set()).add(
			self.name
		)
		self.engine.handle(
			command="add_unit",
			char=self.name,
			graph=graph,
			node=node,
			branching=True,
		)

	def remove_unit(
		self, graph: CharName | CharacterProxy, node: Optional[NodeName] = None
	) -> None:
		self._worker_check()
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

	def units(self) -> Iterator[NodeProxy]:
		yield from self.engine.handle(
			command="character_units", char=self.name
		)

	def facade(self) -> CharacterFacade:
		return CharacterFacade(character=self)

	def grid_2d_8graph(self, m: int, n: int) -> None:
		self.engine.handle(
			"grid_2d_8graph",
			character=self.name,
			m=m,
			n=n,
			cb=self.engine._upd_caches,
		)

	def grid_2d_graph(self, m: int, n: int, periodic: bool = False) -> None:
		self.engine.handle(
			"grid_2d_graph",
			character=self.name,
			m=m,
			n=n,
			periodic=periodic,
			cb=self.engine._upd_caches,
		)


class ProxyUserMapping(Mapping):
	"""A mapping to the ``CharacterProxy``s that have this node as a unit"""

	def __init__(self, node: NodeProxy):
		self.node = node

	def __iter__(self):
		try:
			return iter(self._user_names())
		except KeyError:
			return iter(())

	def __len__(self):
		try:
			return len(self._user_names())
		except KeyError:
			return 0

	def __contains__(self, item: CharName) -> bool:
		try:
			return item in self._user_names()
		except KeyError:
			return False

	def __getitem__(self, item: CharName):
		if item not in self:
			raise KeyError("Not a user of this node", item, self.node.name)
		return self.node.engine.character[item]

	@property
	def only(self):
		if len(self) == 1:
			return next(iter(self.values()))
		raise AmbiguousLeaderError("No users, or more than one")

	def _user_names(self):
		return self.node.engine._unit_characters_cache[self.node._charname][
			self.node.name
		]


class ProxyNeighborMapping(Mapping):
	__slots__ = ("_node",)

	def __init__(self, node: NodeProxy) -> None:
		self._node = node

	def __iter__(self) -> Iterator[NodeName]:
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

	def __getitem__(self, item: NodeName) -> NodeProxy:
		if (
			item in self._node.character.adj[self._node.name]
			or item in self._node.character.pred[self._node.name]
		):
			return self._node.character.node[item]
		raise KeyError("Not a neighbor")


class NodeContentProxy(Mapping):
	def __init__(self, node: NodeProxy):
		self.node = node

	def __getitem__(self, key: NodeName, /):
		if key not in self.node.character.thing:
			raise KeyError("No such thing", key, self.node.character.name)
		thing = self.node.character.thing[key]
		if thing.location != self.node:
			raise KeyError(
				"Not located here",
				key,
				self.node.character.name,
				self.node.name,
			)
		return thing

	def __len__(self):
		n = 0
		for _ in self:
			n += 1
		return n

	def __iter__(self) -> Iterator[NodeName]:
		return self.node.engine._node_contents(
			self.node.character.name, self.node.name
		)

	def __contains__(self, item: NodeName):
		return (
			item in self.node.character.thing
			and self.node.character.thing[item].location == self.node
		)


class NodePortalProxy(Mapping):
	def __init__(self, node):
		self.node = node

	def __getitem__(self, key: NodeName, /) -> PortalProxy:
		return self.node.character.portal[self.node.name][key]

	def __len__(self):
		try:
			return len(self.node.character.portal[self.node.name])
		except KeyError:
			return 0

	def __iter__(self):
		return iter(self.node.character.portal[self.node.name])


class NodePreportalProxy(Mapping):
	def __init__(self, node: NodeProxy):
		self.node = node

	def __getitem__(self, key: NodeName, /) -> PortalProxy:
		return self.node.engine._character_portals_cache.predecessors[
			self.node.character.name
		][self.node.name][key]

	def __len__(self):
		try:
			return len(
				self.node.engine._character_portals_cache.predecessors[
					self.node.character.name
				][self.node.name]
			)
		except KeyError:
			return 0

	def __iter__(self) -> Iterator[NodeName]:
		return iter(
			self.node.engine._character_portals_cache.predecessors[
				self.node.character.name
			][self.node.name]
		)
