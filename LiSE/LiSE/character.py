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
"""The top level of the LiSE world model, the Character.

Based on NetworkX DiGraph objects with various additions and
conveniences.

A Character is a graph that follows rules. Its rules may be assigned
to run on only some portion of it. Each Character has a ``stat`` property that
acts very much like a dictionary, in which you can store game-time-sensitive
data for the rules to use.

You can designate some nodes in one Character as units of another,
and then assign a rule to run on all of a Character's units. This is
useful for the common case where someone in your game has a location
in the physical world (here, a Character, called 'physical') but also
has a behavior flowchart, or a skill tree, that isn't part of the
physical world. In that case, the flowchart is the person's Character,
and their node in the physical world is a unit of it.

"""

from abc import abstractmethod, ABC
from collections.abc import Mapping, MutableMapping
from itertools import chain
from types import MethodType
from typing import Type
from blinker import Signal

import networkx as nx
from .allegedb.cache import FuturistWindowDict, PickyDefaultDict
from .allegedb.graph import (
	DiGraph,
	GraphNodeMapping,
	DiGraphSuccessorsMapping,
	DiGraphPredecessorsMapping,
)
from .allegedb.wrap import MutableMappingUnwrapper

from .xcollections import CompositeDict
from .rule import RuleMapping
from .rule import RuleFollower as BaseRuleFollower
from .node import Node, Place, Thing
from .portal import Portal
from .util import (
	getatt,
	singleton_get,
	timer,
	AbstractEngine,
	AbstractCharacter,
)
from .exc import WorldIntegrityError
from .query import StatusAlias


def grid_2d_8graph(m, n):
	"""Make a 2d graph that's connected 8 ways, with diagonals"""
	me = nx.Graph()
	nodes = me.nodes
	add_node = me.add_node
	add_edge = me.add_edge
	for i in range(m):
		for j in range(n):
			add_node((i, j))
			if i > 0:
				add_edge((i, j), (i - 1, j))
				if j > 0:
					add_edge((i, j), (i - 1, j - 1))
			if j > 0:
				add_edge((i, j), (i, j - 1))
			if (i - 1, j + 1) in nodes:
				add_edge((i, j), (i - 1, j + 1))
	return me


class CharRuleMapping(RuleMapping):
	"""Get rules by name, or make new ones by decorator

	You can access the rules in this either dictionary-style or as
	attributes. This is for convenience if you want to get at a rule's
	decorators, eg. to add an Action to the rule.

	Using this as a decorator will create a new rule, named for the
	decorated function, and using the decorated function as the
	initial Action.

	Using this like a dictionary will let you create new rules,
	appending them onto the underlying :class:`RuleBook`; replace one
	rule with another, where the new one will have the same index in
	the :class:`RuleBook` as the old one; and activate or deactivate
	rules. The name of a rule may be used in place of the actual rule,
	so long as the rule already exists.

	You can also set a rule active or inactive by setting it to
	``True`` or ``False``, respectively. Inactive rules are still in
	the rulebook, but won't be followed.

	"""

	def __init__(self, character, rulebook, booktyp):
		"""Initialize as usual for the ``rulebook``, mostly.

		My ``character`` property will be the one passed in, and my
		``_table`` will be the ``booktyp`` with ``"_rules"`` appended.

		"""
		super().__init__(rulebook.engine, rulebook)
		self.character = character
		self._table = booktyp + "_rules"


class RuleFollower(BaseRuleFollower):
	"""Mixin class. Has a rulebook, which you can get a RuleMapping into."""

	character: AbstractCharacter
	engine: AbstractEngine
	_book: str

	def _get_rule_mapping(self):
		return CharRuleMapping(self.character, self.rulebook, self._book)

	@abstractmethod
	def _get_rulebook_cache(self):
		pass

	def _get_rulebook_name(self):
		try:
			return self._get_rulebook_cache().retrieve(
				self.character.name, *self.engine._btt()
			)
		except KeyError:
			return self.character.name, self._book

	def _set_rulebook_name(self, n):
		branch, turn, tick = self.engine._nbtt()
		self.engine.query._set_rulebook_on_character(
			self._book, self.character.name, branch, turn, tick, n
		)
		self._get_rulebook_cache().store(
			self.character.name, branch, turn, tick, n
		)

	def __contains__(self, k):
		return self.engine._active_rules_cache.contains_key(
			self._get_rulebook_name(), *self.engine._btt()
		)


class FacadeEntity(MutableMapping, Signal, ABC):
	exists = True

	def __init__(self, mapping, _=None, **kwargs):
		super().__init__()
		self.facade = self.character = mapping.facade
		self._real = mapping
		self._patch = {
			k: v.unwrap() if hasattr(v, "unwrap") else v
			for (k, v) in kwargs.items()
		}

	def __contains__(self, item):
		patch = self._patch
		return item in self._real or (
			item in patch and patch[item] is not None
		)

	def __iter__(self):
		seen = set()
		for k in self._real:
			if k not in self._patch:
				yield k
				seen.add(k)
		for k in self._patch:
			if self._patch[k] is not None and k not in seen:
				yield k

	def __len__(self):
		n = 0
		for k in self:
			n += 1
		return n

	def __getitem__(self, k):
		if k in self._patch:
			if self._patch[k] is None:
				raise KeyError("{} has been masked.".format(k))
			return self._patch[k]
		ret = self._real[k]
		if hasattr(ret, "unwrap"):  # a wrapped mutable object from the
			# allegedb.wrap module
			ret = ret.unwrap()
			self._patch[k] = ret  # changes will be reflected in the
		# facade but not the original
		return ret

	def __setitem__(self, k, v):
		if k == "name":
			raise KeyError("Can't change names")
		if hasattr(v, "unwrap"):
			v = v.unwrap()
		self._patch[k] = v

	def __delitem__(self, k):
		self._patch[k] = None


class FacadeNode(FacadeEntity, ABC):
	@property
	def name(self):
		return self["name"]

	@property
	def portal(self):
		return self.facade.portal[self["name"]]

	def contents(self):
		for thing in self.facade.thing.values():
			# it seems like redundant FacadeNode are being created sometimes
			if thing["location"] == self.name:
				yield thing


class FacadePlace(FacadeNode):
	"""Lightweight analogue of Place for Facade use."""

	def __init__(self, mapping, real_or_name, **kwargs):
		super().__init__(mapping, real_or_name, **kwargs)
		if isinstance(real_or_name, (Place, FacadePlace)):
			self._real = real_or_name
		else:
			self._real = {"name": real_or_name}

	def add_thing(self, name):
		self.facade.add_thing(name, self.name)

	def new_thing(self, name):
		return self.facade.new_thing(name, self.name)


class FacadeThing(FacadeNode):
	def __init__(self, mapping, real_or_name, **kwargs):
		location = kwargs.pop("location", None)
		super().__init__(mapping, real_or_name, **kwargs)
		if location is None and not (
			isinstance(real_or_name, Thing)
			or isinstance(real_or_name, FacadeThing)
		):
			raise TypeError(
				"FacadeThing needs to wrap a real Thing or another "
				"FacadeThing, or have a location of its own."
			)
		self._real = {
			"name": real_or_name.name
			if hasattr(real_or_name, "name")
			else real_or_name,
			"location": location,
		}

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
		kwargs = dict(v)
		for badkey in ("character", "engine", "name"):
			if badkey in kwargs:
				del kwargs[badkey]
		return self.facadecls(self, k, **kwargs)

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


class FacadePortalSuccessors(FacadeEntityMapping):
	facadecls = FacadePortal
	innercls = Portal

	def __init__(self, facade, origname):
		super().__init__(facade, origname)
		self.orig = origname

	def _make(self, k, v):
		return self.facadecls(self, k, **v)

	def _get_inner_map(self):
		try:
			return self.facade.character.portal[self.orig]
		except AttributeError:
			return {}


class FacadePortalPredecessors(FacadeEntityMapping):
	facadecls = FacadePortal
	innercls = Portal

	def __init__(self, facade, destname):
		super().__init__(facade, destname)
		self.dest = destname

	def _make(self, k, v):
		return self.facadecls(self.facade.portal[k], v)

	def _get_inner_map(self):
		try:
			return self.facade.character.preportal[self.dest]
		except AttributeError:
			return {}


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


class Facade(AbstractCharacter, nx.DiGraph):
	engine = getatt("character.engine")
	db = getatt("character.engine")

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
		raise NotImplementedError("Facades don't have units")

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
		raise NotImplementedError("Facades don't have units")

	def __init__(self, character=None):
		"""Store the character."""
		super().__init__()
		self.character = character
		self.graph = self.StatMapping(self)

	class ThingMapping(FacadeEntityMapping):
		facadecls = FacadeThing
		innercls = Thing

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
		innercls = Place

		def _get_inner_map(self):
			try:
				return self.facade.character._node
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
			self._patch[k] = v

		def __delitem__(self, k):
			self._patch[k] = None


class Character(DiGraph, AbstractCharacter, RuleFollower):
	"""A digraph that follows game rules and has a containment hierarchy

	Nodes in a Character are subcategorized into Things and
	Places. Things have locations, and those locations may be Places
	or other Things. To get at those, use the `thing` and `place`
	mappings -- but in situations where the distinction does not matter,
	you may simply address the Character as a mapping, as in NetworkX.

	Characters may have units in other Characters. These are just
	nodes. You can apply rules to a Character's units, and thus to
	any collection of nodes you want, perhaps in many different
	Characters. The ``unit`` attribute handles this. It is a mapping,
	keyed by the other Character's name, then by the name of the node
	that is this Character's unit. In the common case where a
	Character has exactly one unit, it may be retrieved as
	``unit.only``. When it has more than one unit, but only has
	any units in a single other Character, you can get the mapping
	of units in that Character as ``unit.node``. Add units with the
	``add_unit`` method and remove them with ``del_unit``.

	You can assign rules to Characters with their ``rule`` attribute,
	typically using it as a decorator (see :mod:`LiSE.rule`). You can do the
	same to some of Character's
	attributes:

	* ``thing.rule`` to make a rule run on all Things in this Character
	  every turn
	* ``place.rule`` to make a rule run on all Places in this Character
	  every turn
	* ``node.rule`` to make a rule run on all Things and Places in this
	  Character every turn
	* ``unit.rule`` to make a rule run on all the units this
	  Character has every turn, regardless of what Character the
	  unit is in
	* ``adj.rule`` to make a rule run on all the edges this Character
	  has every turn

	"""

	_book = "character"

	def remove_portal(self, origin, destination):
		__doc__ = self.remove_edge.__doc__
		super().remove_edge(origin, destination)

	@property
	def character(self):
		return self

	def _get_rulebook_cache(self):
		return self.engine._characters_rulebooks_cache

	def __bool__(self):
		try:
			return (
				self.engine._graph_cache.retrieve(
					self.name, *self.engine._btt()
				)
				!= "Deleted"
			)
		except KeyError:
			return False

	def __repr__(self):
		return "{}.character[{}]".format(repr(self.engine), repr(self.name))

	def __init__(self, engine, name, *, init_rulebooks=True):
		super().__init__(engine, name)
		self._avatars_cache = PickyDefaultDict(FuturistWindowDict)
		if not init_rulebooks:
			return
		cachemap = {
			"character": engine._characters_rulebooks_cache,
			"unit": engine._units_rulebooks_cache,
			"character_thing": engine._characters_things_rulebooks_cache,
			"character_place": engine._characters_places_rulebooks_cache,
			"character_portal": engine._characters_portals_rulebooks_cache,
		}
		branch, turn, tick = engine._btt()
		for rulebook, cache in cachemap.items():
			rulebook_name = (name, rulebook)
			engine.query._set_rulebook_on_character(
				rulebook, name, branch, turn, tick, rulebook_name
			)
			cache.store((name, rulebook), branch, turn, tick, rulebook_name)

	class ThingMapping(MutableMappingUnwrapper, RuleFollower, Signal):
		""":class:`Thing` objects that are in a :class:`Character`"""

		_book = "character_thing"

		engine = getatt("character.engine")
		name = getatt("character.name")

		def _get_rulebook_cache(self):
			return self.engine._characters_things_rulebooks_cache

		def __init__(self, character):
			"""Store the character and initialize cache."""
			super().__init__()
			self.character = character

		def __iter__(self):
			cache = self.engine._things_cache
			char = self.name
			branch, turn, tick = self.engine._btt()
			for key in cache.iter_keys(char, branch, turn, tick):
				try:
					if (
						cache.retrieve(char, key, branch, turn, tick)
						is not None
					):
						yield key
				except KeyError:
					continue

		def __contains__(self, thing):
			branch, turn, tick = self.engine._btt()
			args = self.character.name, thing, branch, turn, tick
			cache = self.engine._things_cache
			return cache.contains_key(*args)

		def __len__(self):
			return self.engine._things_cache.count_keys(
				self.character.name, *self.engine._btt()
			)

		def __getitem__(self, thing):
			if thing not in self:
				raise KeyError("No such thing: {}".format(thing))
			return self._make_thing(thing)

		def _make_thing(self, thing, val=None):
			cache = self.engine._node_objs
			if isinstance(val, Thing):
				th = cache[self.name, thing] = val
			elif (self.name, thing) in cache:
				th = cache[(self.name, thing)]
				if type(th) is not Thing:
					th = cache[self.name, thing] = Thing(self.character, thing)
			else:
				th = cache[(self.name, thing)] = Thing(self.character, thing)
			return th

		def __setitem__(self, thing, val):
			if not isinstance(val, Mapping):
				raise TypeError("Things are made from Mappings")
			if "location" not in val:
				raise ValueError("Thing needs location")
			created = thing not in self
			self.engine._exist_node(self.character.name, thing)
			self.engine._set_thing_loc(
				self.character.name, thing, val["location"]
			)
			th = self._make_thing(thing, val)
			th.clear()
			th.update(val)

		def __delitem__(self, thing):
			self[thing].delete()

		def __repr__(self):
			return "{}.character[{}].thing".format(
				repr(self.engine), repr(self.name)
			)

	class PlaceMapping(MutableMappingUnwrapper, RuleFollower, Signal):
		""":class:`Place` objects that are in a :class:`Character`"""

		_book = "character_place"

		def _get_rulebook_cache(self):
			return self.engine._characters_places_rulebooks_cache

		def update(self, __m: dict, **kwargs) -> None:
			self.character.node.update(__m, **kwargs)

		def __init__(self, character):
			"""Store the character."""
			super().__init__()
			self.character = character
			self.engine = engine = character.engine
			charn = character.name
			nodes_cache = engine._nodes_cache
			things_cache = engine._things_cache
			iter_nodes = nodes_cache.iter_entities
			nodes_contains = nodes_cache.contains_entity
			things_contains = things_cache.contains_entity
			btt = engine._btt
			self._iter_stuff = (iter_nodes, things_contains, charn, btt)
			self._len_stuff = (
				nodes_cache.count_entities,
				things_cache.count_entities,
				charn,
				btt,
			)
			self._contains_stuff = (
				nodes_contains,
				things_contains,
				charn,
				btt,
			)
			self._get_stuff = self._contains_stuff + (
				engine._node_objs,
				character,
			)
			self._set_stuff = (
				engine._node_exists,
				engine._exist_node,
				engine._get_node,
				charn,
				character,
			)

		def __iter__(self):
			iter_nodes, things_contains, charn, btt = self._iter_stuff
			branch, turn, tick = btt()
			for node in iter_nodes(charn, branch, turn, tick):
				if not things_contains(charn, node, branch, turn, tick):
					yield node

		def __len__(self):
			count_nodes, count_things, charn, btt = self._len_stuff
			branch, turn, tick = btt()
			return count_nodes(charn, branch, turn, tick) - count_things(
				charn, branch, turn, tick
			)

		def __contains__(self, place):
			nodes_contains, things_contains, charn, btt = self._contains_stuff
			branch, turn, tick = btt()
			return nodes_contains(
				charn, place, branch, turn, tick
			) and not things_contains(charn, place, branch, turn, tick)

		def __getitem__(self, place):
			(nodes_contains, things_contains, charn, btt, cache, character) = (
				self._get_stuff
			)
			branch, turn, tick = btt()
			if not nodes_contains(
				charn, place, branch, turn, tick
			) or things_contains(charn, place, branch, turn, tick):
				raise KeyError("No such place: {}".format(place))
			if (charn, place) not in cache or not isinstance(
				cache[(charn, place)], Place
			):
				ret = cache[(charn, place)] = Place(character, place)
				return ret
			return cache[(charn, place)]

		def __setitem__(self, place, v):
			(node_exists, exist_node, get_node, charn, character) = (
				self._set_stuff
			)
			exist_node(charn, place, True)
			pl = get_node(character, place)
			if not isinstance(pl, Place):
				raise KeyError(
					"{} is a {}, not a place".format(place, type(pl).__name__)
				)
			pl.update(v)

		def __delitem__(self, place):
			self[place].delete()

		def __repr__(self):
			return "{}.character[{}].place".format(
				repr(self.character.engine), repr(self.character.name)
			)

	class ThingPlaceMapping(GraphNodeMapping, Signal):
		"""GraphNodeMapping but for Place and Thing"""

		_book = "character_node"

		character = getatt("graph")
		engine = getatt("db")
		name = getatt("character.name")

		def __init__(self, character):
			"""Store the character."""
			super().__init__(character)
			Signal.__init__(self)
			engine = character.engine
			charn = character.name
			self._contains_stuff = contains_stuff = (
				engine._node_exists,
				charn,
			)
			self._getitem_stuff = contains_stuff + (
				engine._get_node,
				character,
			)
			self._delitem_stuff = contains_stuff + (
				engine._is_thing,
				character.thing,
				character.place,
			)
			self._placemap = character.place

		def __contains__(self, k):
			node_exists, charn = self._contains_stuff
			return node_exists(charn, k)

		def __getitem__(self, k):
			node_exists, charn, get_node, character = self._getitem_stuff
			if not node_exists(charn, k):
				raise KeyError("No such node: " + str(k))
			return get_node(character, k)

		def __setitem__(self, k, v):
			self._placemap[k] = v

		def __delitem__(self, k):
			(node_exists, charn, is_thing, thingmap, placemap) = (
				self._delitem_stuff
			)
			if not node_exists(charn, k):
				raise KeyError("No such node: " + str(k))
			if is_thing(charn, k):
				del thingmap[k]
			else:
				del placemap[k]

	node_map_cls = ThingPlaceMapping

	class PortalSuccessorsMapping(DiGraphSuccessorsMapping, RuleFollower):
		"""Mapping of nodes that have at least one outgoing edge.

		Maps them to another mapping, keyed by the destination nodes,
		which maps to Portal objects.

		"""

		_book = "character_portal"

		character = getatt("graph")
		engine = getatt("graph.engine")

		def __init__(self, graph):
			super().__init__(graph)
			engine = graph.engine
			charn = graph.name
			self._cporh = engine._characters_portals_rulebooks_cache
			self._getitem_stuff = (engine._node_exists, charn, self._cache)
			self._setitem_stuff = (self._cache, self.Successors)

		def _get_rulebook_cache(self):
			return self._cporh

		def __getitem__(self, orig):
			node_exists, charn, cache = self._getitem_stuff
			if node_exists(charn, orig):
				if orig not in cache:
					cache[orig] = self.Successors(self, orig)
				return cache[orig]
			raise KeyError("No such node")

		def __delitem__(self, orig):
			super().__delitem__(orig)

		def update(self, other, **kwargs):
			"""Recursively update the stats of all portals

			Input should be a dictionary of dictionaries of dictionaries
			--just like networkx ``DiGraph._edge``.

			This will create portals as needed, but will only delete
			them if you set their value to ``None``. Likewise, stats
			not specified in the input will be left untouched, if they
			are already present, but you can set them to ``None`` to
			delete them.

			"""
			engine = self.engine
			planning = engine._planning
			forward = engine._forward
			branch, turn, start_tick = engine._btt()
			exist_edge = engine.query.exist_edge
			edge_val_set = engine.query.edge_val_set
			store_edge = engine._edges_cache.store
			store_edge_val = engine._edge_val_cache.store
			iter_edge_keys = engine._edge_val_cache.iter_entity_keys
			charn = self.character.name
			tick = start_tick + 1
			with timer(
				"seconds spent updating PortalSuccessorsMapping", engine.debug
			):
				for orig, dests in chain(other.items(), kwargs.items()):
					for dest, kvs in dests.items():
						if kvs is None:
							for k in iter_edge_keys(
								charn,
								orig,
								dest,
								0,
								branch,
								turn,
								start_tick,
								forward=forward,
							):
								store_edge_val(
									charn,
									orig,
									dest,
									0,
									k,
									branch,
									turn,
									tick,
									None,
									planning=planning,
									forward=forward,
									loading=True,
								)
								edge_val_set(
									charn,
									orig,
									dest,
									0,
									k,
									branch,
									turn,
									tick,
									None,
								)
								tick += 1
							store_edge(
								charn,
								orig,
								dest,
								0,
								branch,
								turn,
								tick,
								False,
								planning=planning,
								forward=forward,
								loading=True,
							)
							exist_edge(
								charn, orig, dest, 0, branch, turn, tick, False
							)
							tick += 1
						else:
							store_edge(
								charn,
								orig,
								dest,
								0,
								branch,
								turn,
								tick,
								True,
								planning=planning,
								forward=forward,
								loading=True,
							)
							exist_edge(
								charn, orig, dest, 0, branch, turn, tick, True
							)
							tick += 1
							for k, v in kvs.items():
								store_edge_val(
									charn,
									orig,
									dest,
									0,
									k,
									branch,
									turn,
									tick,
									v,
									planning=planning,
									forward=forward,
									loading=True,
								)
								edge_val_set(
									charn,
									orig,
									dest,
									0,
									k,
									branch,
									turn,
									tick,
									v,
								)
								tick += 1
			engine.tick = tick

		class Successors(DiGraphSuccessorsMapping.Successors):
			"""Mapping for possible destinations from some node."""

			engine = getatt("graph.engine")

			def __init__(self, container, orig):
				super().__init__(container, orig)
				graph = self.graph
				engine = graph.engine
				self._getitem_stuff = (engine._get_edge, graph, orig)
				self._setitem_stuff = (
					engine._edge_exists,
					engine._exist_edge,
					graph.name,
					orig,
					engine._get_edge,
					graph,
					engine.query.edge_val_set,
					engine._edge_val_cache.store,
					engine._nbtt,
				)

			def __getitem__(self, dest):
				get_edge, graph, orig = self._getitem_stuff
				if dest in self:
					return get_edge(graph, orig, dest, 0)
				raise KeyError("No such portal: {}->{}".format(orig, dest))

			def __setitem__(self, dest, value):
				if value is None:
					del self[dest]
					return
				(
					edge_exists,
					exist_edge,
					charn,
					orig,
					get_edge,
					graph,
					db_edge_val_set,
					edge_val_cache_store,
					nbtt,
				) = self._setitem_stuff
				exist_edge(charn, orig, dest)
				for k, v in value.items():
					branch, turn, tick = nbtt()
					db_edge_val_set(
						charn, orig, dest, 0, k, branch, turn, tick, v
					)
					edge_val_cache_store(
						charn, orig, dest, 0, k, branch, turn, tick, v
					)

			def __delitem__(self, dest):
				if dest not in self:
					raise KeyError("No portal to {}".format(dest))
				self[dest].delete()

			def update(self, other, **kwargs):
				charn = self.graph.name
				orig = self.orig
				engine = self.engine
				store_edge = engine._edges_cache.store
				exist_edge = engine.query.exist_edge
				store_edge_val = engine._edge_val_cache.store
				set_edge_val = engine.query.edge_val_set
				iter_edge_keys = engine._edge_val_cache.iter_entity_keys
				planning = engine._planning
				forward = engine._forward
				branch, turn, start_tick = engine._btt()
				tick = start_tick + 1
				for dest, val in chain(other.items(), kwargs.items()):
					if val is None:
						for k in iter_edge_keys(
							charn, orig, dest, 0, branch, turn, start_tick
						):
							store_edge_val(
								charn,
								orig,
								dest,
								0,
								k,
								branch,
								turn,
								tick,
								None,
								planning=planning,
								forward=forward,
								loading=True,
							)
							set_edge_val(
								charn,
								orig,
								dest,
								0,
								k,
								branch,
								turn,
								tick,
								None,
							)
							tick += 1
						store_edge(
							charn,
							orig,
							dest,
							0,
							branch,
							turn,
							tick,
							None,
							planning=planning,
							forward=forward,
							loading=True,
						)
						exist_edge(
							charn, orig, dest, 0, branch, turn, tick, None
						)
						tick += 1

	adj_cls = PortalSuccessorsMapping

	class PortalPredecessorsMapping(DiGraphPredecessorsMapping, RuleFollower):
		"""Mapping of nodes that have at least one incoming edge.

		Maps to another mapping keyed by the origin nodes, which maps to
		Portal objects.

		"""

		_book = "character_portal"

		def __init__(self, graph):
			super().__init__(graph)
			self._cporc = graph.engine._characters_portals_rulebooks_cache

		def _get_rulebook_cache(self):
			return self._cporc

		class Predecessors(DiGraphPredecessorsMapping.Predecessors):
			"""Mapping of possible origins from some destination."""

			def __init__(self, container, dest):
				super().__init__(container, dest)
				graph = self.graph
				self._setitem_stuff = (
					graph,
					graph.name,
					dest,
					self.db._edge_objs,
				)

			def __setitem__(self, orig, value):
				graph, graph_name, dest, portal_objs = self._setitem_stuff
				key = (graph_name, orig, dest)
				if key not in portal_objs:
					portal_objs[key] = Portal(graph, orig, dest)
				p = portal_objs[key]
				p.clear()
				p.update(value)
				p.engine._exist_edge(graph_name, dest, orig)

	pred_cls = PortalPredecessorsMapping

	class UnitGraphMapping(Mapping, RuleFollower):
		"""A mapping of other characters in which one has a unit."""

		_book = "unit"

		engine = getatt("character.engine")
		name = getatt("character.name")

		def _get_rulebook_cache(self):
			return self._avrc

		def __init__(self, char):
			"""Remember my character."""
			self.character = char
			self._char_av_cache = {}
			engine = char.engine
			self._avrc = engine._units_rulebooks_cache
			self._add_av = char.add_unit
			avcache = engine._unitness_cache
			get_char_graphs = avcache.get_char_graphs
			charn = char.name
			btt = engine._btt
			self._iter_stuff = (get_char_graphs, charn, btt)
			self._node_stuff = (
				self._get_char_av_cache,
				avcache.get_char_only_graph,
				charn,
				btt,
			)
			self._only_stuff = (
				avcache.get_char_only_av,
				charn,
				btt,
				engine._get_node,
				engine.character,
			)

		def __call__(self, av):
			"""Add the unit

			It must be an instance of Place or Thing.

			"""
			if av.__class__ not in (Place, Thing):
				raise TypeError("Only Things and Places may be units")
			self._add_av(av.name, av.character.name)

		def __iter__(self):
			"""Iterate over graphs with unit nodes in them"""
			get_char_graphs, charn, btt = self._iter_stuff
			return iter(get_char_graphs(charn, *btt()))

		def __contains__(self, k):
			get_char_graphs, charn, btt = self._iter_stuff
			return k in get_char_graphs(charn, *btt())

		def __len__(self):
			"""Number of graphs in which I have a unit."""
			get_char_graphs, charn, btt = self._iter_stuff
			return len(get_char_graphs(charn, *btt()))

		def _get_char_av_cache(self, g):
			if g not in self:
				raise KeyError
			if g not in self._char_av_cache:
				self._char_av_cache[g] = self.CharacterUnitMapping(self, g)
			return self._char_av_cache[g]

		def __getitem__(self, g):
			return self._get_char_av_cache(g)

		@property
		def node(self):
			"""If I have units in only one graph, return a map of them

			Otherwise, raise AttributeError.

			"""
			get_char_av_cache: MethodType
			get_char_av_cache, get_char_only_graph, charn, btt = (
				self._node_stuff
			)
			try:
				return get_char_av_cache(get_char_only_graph(charn, *btt()))
			except KeyError:
				raise AttributeError(
					"I have no unit, or I have units in many graphs"
				)

		@property
		def only(self):
			"""If I have only one unit, this is it

			Otherwise, raise AttributeError.

			"""
			get_char_only_av, charn, btt, get_node, charmap = self._only_stuff
			try:
				charn, noden = get_char_only_av(charn, *btt())
				return get_node(charmap[charn], noden)
			except (KeyError, TypeError):
				raise AttributeError("I have no unit, or more than one unit")

		class CharacterUnitMapping(Mapping):
			"""Mapping of units of one Character in another Character."""

			def __init__(self, outer, graphn):
				"""Store this character and the name of the other one"""
				self.character = character = outer.character
				self.engine = engine = outer.engine
				self.name = name = outer.name
				self.graph = graphn
				avcache = engine._unitness_cache
				btt = engine._btt
				self._iter_stuff = iter_stuff = (
					avcache.get_char_graph_avs,
					name,
					graphn,
					btt,
				)
				get_node = engine._get_node
				self._getitem_stuff = iter_stuff + (
					get_node,
					graphn,
					engine.character,
				)
				self._only_stuff = (get_node, engine.character, graphn)

			def __iter__(self):
				"""Iterate over names of unit nodes"""
				get_char_graph_avs, name, graphn, btt = self._iter_stuff
				return iter(get_char_graph_avs(name, graphn, *btt()))

			def __contains__(self, av):
				get_char_graph_avs, name, graphn, btt = self._iter_stuff
				return av in get_char_graph_avs(name, graphn, *btt())

			def __len__(self):
				"""Number of units of this character in that graph"""
				get_char_graph_avs, name, graphn, btt = self._iter_stuff
				return len(get_char_graph_avs(name, graphn, *btt()))

			def __getitem__(self, av):
				(
					get_char_graph_avs,
					name,
					graphn,
					btt,
					get_node,
					graphn,
					charmap,
				) = self._getitem_stuff
				if av in get_char_graph_avs(name, graphn, *btt()):
					return get_node(charmap[graphn], av)
				raise KeyError("No unit: {}".format(av))

			@property
			def only(self):
				"""If I have only one unit, return it; else error"""
				mykey = singleton_get(self.keys())
				if mykey is None:
					raise AttributeError("No unit, or more than one")
				get_node, charmap, graphn = self._only_stuff
				return get_node(charmap[graphn], mykey)

			def __repr__(self):
				return "{}.character[{}].unit".format(
					repr(self.engine), repr(self.name)
				)

	def facade(self):
		"""Return a temporary copy of this Character

		A Facade looks like its :class:`Character`, but doesn't do any of the
		stuff Characters do to save changes to the database, nor enable
		time travel. This makes it much speedier to work with.

		"""
		return Facade(self)

	def add_place(self, node_for_adding, **attr):
		"""Add a new Place"""
		self.add_node(node_for_adding, **attr)

	def add_places_from(self, seq, **attrs):
		"""Take a series of place names and add the lot."""
		super().add_nodes_from(seq, **attrs)

	def remove_place(self, place):
		"""Remove an existing Place"""
		if place in self.place:
			self.remove_node(place)
		else:
			raise KeyError("No such place: {}".format(place))

	def remove_thing(self, thing):
		"""Remove an existing Thing"""
		if thing in self.thing:
			self.remove_node(thing)
		else:
			raise KeyError("No such thing: {}".format(thing))

	def add_thing(self, name, location, **kwargs):
		"""Make a new Thing and set its location"""
		if name in self.thing:
			raise WorldIntegrityError(
				"Already have a Thing named {}".format(name)
			)
		starter = self.node_dict_factory()
		starter.update(kwargs)
		if isinstance(location, Node):
			location = location.name
		starter["location"] = location
		self.thing[name] = starter
		if name not in self._succ:
			self._succ[name] = self.adjlist_inner_dict_factory()
			self._pred[name] = self.adjlist_inner_dict_factory()

	def add_things_from(self, seq, **attrs):
		"""Make many new Things"""
		for tup in seq:
			name = tup[0]
			location = tup[1]
			kwargs = tup[2] if len(tup) > 2 else attrs
			self.add_thing(name, location, **kwargs)

	def place2thing(self, name, location):
		"""Turn a Place into a Thing with the given location.

		It will keep all its attached Portals.

		"""
		self.engine._set_thing_loc(self.name, name, location)
		if (self.name, name) in self.engine._node_objs:
			obj = self.engine._node_objs[self.name, name]
			thing = Thing(self, name)
			for port in obj.portal.values():
				port.origin = thing
			for port in obj.preportal.values():
				port.destination = thing
			self.engine._node_objs[self.name, name] = thing

	def thing2place(self, name):
		"""Unset a Thing's location, and thus turn it into a Place."""
		self.engine._set_thing_loc(self.name, name, None)
		if (self.name, name) in self.engine._node_objs:
			thing = self.engine._node_objs[self.name, name]
			place = Place(self, name)
			for port in thing.portals():
				port.origin = place
			for port in thing.preportals():
				port.destination = place
			self.engine._node_objs[self.name, name] = place

	def add_portal(self, origin, destination, **kwargs):
		"""Connect the origin to the destination with a :class:`Portal`.

		Keyword arguments are attributes of the :class:`Portal`.

		"""
		if isinstance(origin, Node):
			origin = origin.name
		if isinstance(destination, Node):
			destination = destination.name
		super().add_edge(origin, destination, **kwargs)

	def new_portal(self, origin, destination, **kwargs):
		"""Create a portal and return it"""
		self.add_portal(origin, destination, **kwargs)
		return self.engine._get_edge(self, origin, destination, 0)

	def add_portals_from(self, seq, **kwargs):
		"""Make portals for a sequence of (origin, destination) pairs

		Actually, triples are acceptable too, in which case the third
		item is a dictionary of stats for the new :class:`Portal`.
		"""
		for tup in seq:
			orig = tup[0]
			dest = tup[1]
			kwarrgs = tup[2] if len(tup) > 2 else kwargs
			self.add_portal(orig, dest, **kwarrgs)

	def add_unit(self, a, b=None):
		"""Start keeping track of a unit.

		Units are nodes in other characters that are in some sense part of
		this one. A common example in strategy games is when a general leads
		an army: the general is one :class:`Character`, with a graph
		representing the state of its AI; the battle map is another character;
		and the general's units, though not in the general's
		:class:`Character`, are still under their command, and therefore
		follow rules defined on the general's ``unit`` property.

		"""
		if self.engine._planning:
			raise NotImplementedError(
				"Currently can't add units within a plan"
			)
		if b is None:
			if not isinstance(a, (Place, Thing)):
				raise TypeError(
					"when called with one argument, "
					"it must be a place or thing"
				)
			g = a.character.name
			n = a.name
		else:
			if isinstance(a, Character):
				g = a.name
			elif not isinstance(a, str):
				raise TypeError(
					"when called with two arguments, "
					"the first is a character or its name"
				)
			else:
				g = a
			if isinstance(b, (Place, Thing)):
				n = b.name
			elif not isinstance(b, str):
				raise TypeError(
					"when called with two arguments, "
					"the second is a thing/place or its name"
				)
			else:
				n = b
		# This will create the node if it doesn't exist. Otherwise
		# it's redundant but harmless.
		self.engine._exist_node(g, n)
		# Declare that the node is my unit
		branch, turn, tick = self.engine._nbtt()
		self.engine._remember_unitness(
			self.name, g, n, branch=branch, turn=turn, tick=tick
		)

	def remove_unit(self, a, b=None):
		"""This is no longer my unit, though it still exists"""
		if self.engine._planning:
			raise NotImplementedError(
				"Currently can't remove units within a plan"
			)
		if b is None:
			if not isinstance(a, Node):
				raise TypeError(
					"In single argument form, "
					"del_unit requires a Node object "
					"(Thing or Place)."
				)
			g = a.character.name
			n = a.name
		else:
			g = a.name if isinstance(a, Character) else a
			n = b.name if isinstance(b, Node) else b
		self.engine._remember_unitness(self.character.name, g, n, False)

	def portals(self):
		"""Iterate over all portals."""
		for o in self.adj.values():
			yield from o.values()

	def units(self):
		"""Iterate over all my units

		Regardless of what character they are in.

		"""
		charname = self.character.name
		branch, turn, tick = self.engine._btt()
		charmap = self.engine.character
		avit = self.engine._unitness_cache.iter_entities
		makenode = self.engine._get_node
		for graph in charmap:
			for node in avit(charname, graph, branch, turn, tick):
				yield makenode(charmap[graph], node)

	def historical(self, stat):
		"""Get a historical view on the given stat

		This functions like the value of the stat, but changes
		when you time travel. Comparisons performed on the
		historical view can be passed to ``engine.turns_when``
		to find out when the comparison held true.

		"""
		return StatusAlias(entity=self, stat=stat, engine=self.engine)
