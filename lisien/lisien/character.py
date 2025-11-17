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
"""The top level of the lisien world model, the Character.

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

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from functools import cached_property
from itertools import chain
from types import MethodType
from typing import TYPE_CHECKING, Callable, Iterable, Iterator

import networkx as nx
from blinker import Signal

from .exc import WorldIntegrityError
from .facade import CharacterFacade, EngineFacade
from .node import Node, Place, Thing
from .portal import Portal
from .rule import RuleBook
from .rule import RuleFollower as BaseRuleFollower
from .rule import RuleMapping
from .types import (
	AbstractCharacter,
	BaseMutableDiGraphMapping,
	Branch,
	CharacterStatAlias,
	CharName,
	DiGraphMappingMixin,
	DiGraphPredecessorsMapping,
	DiGraphSuccessorsMapping,
	GraphNodeMapping,
	Key,
	KeyHint,
	NodeName,
	RulebookName,
	RulebookTypeStr,
	Stat,
	StatDict,
	ThingDict,
	Tick,
	Time,
	TimeSignal,
	Turn,
	UnitsAlias,
	Value,
	ValueHint,
)
from .util import getatt, singleton_get, timer, unwrap

if TYPE_CHECKING:
	from .engine import Engine


def grid_2d_8graph(m: int, n: int) -> nx.Graph:
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


	"""

	def __init__(
		self,
		character: Character,
		rulebook: RuleBook,
		booktyp: RulebookTypeStr,
	):
		"""Initialize as usual for the ``rulebook``, mostly.

		My ``character`` property will be the one passed in, and my
		``_table`` will be the ``booktyp`` with ``"_rules"`` appended.

		"""
		super().__init__(rulebook.engine, rulebook)
		self.character = character
		self._table = booktyp + "_rules"


class RuleFollower(BaseRuleFollower):
	"""Mixin class. Has a rulebook, which you can get a RuleMapping into."""

	character: Character
	engine: Engine
	_book: RulebookTypeStr

	def _get_rule_mapping(self):
		return CharRuleMapping(self.character, self.rulebook, self._book)

	@abstractmethod
	def _get_rulebook_cache(self):
		pass

	def _get_rulebook_name(self) -> RulebookName:
		try:
			return self._get_rulebook_cache().retrieve(
				self.character.name, *self.engine.time
			)
		except KeyError:
			ret = RulebookName(
				Key((self._book + "_rulebook", self.character.name))
			)
			self._set_rulebook_name(ret)
			return ret

	def _set_rulebook_name(self, n: RulebookName):
		branch, turn, tick = self.engine._nbtt()
		set_rb: Callable[
			[CharName, Branch, Turn, Tick, RulebookName], None
		] = getattr(self.engine.db, f"set_{self._book}_rulebook")
		set_rb(
			self.character.name,
			branch,
			turn,
			tick,
			n,
		)
		self._get_rulebook_cache().store(
			self.character.name, branch, turn, tick, n
		)

	def __contains__(self, k: KeyHint) -> bool:
		return self.engine._active_rules_cache.contains_key(
			self._get_rulebook_name(), *self.engine.time
		)


class MutableCharacterMapping(BaseMutableDiGraphMapping, RuleFollower): ...


class Character(AbstractCharacter, RuleFollower):
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
	typically using it as a decorator (see :mod:`lisien.rule`). You can do the
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

	engine: Engine
	name: CharName
	_book = "character"

	@staticmethod
	def node_dict_factory() -> StatDict:
		return {}

	def remove_portal(self, origin: KeyHint, destination: KeyHint):
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
				self.engine._graph_cache.retrieve(self.name, *self.engine.time)
				!= "Deleted"
			)
		except KeyError:
			return False

	def __repr__(self):
		return "{}.character[{}]".format(repr(self.engine), repr(self.name))

	def __new__(
		cls, engine: Engine, name: CharName, *, init_rulebooks: bool = False
	):
		return super().__new__(cls, engine, name)

	def __init__(
		self, engine: Engine, name: CharName, *, init_rulebooks: bool = False
	):
		name = name
		super().__init__(engine, name)
		if not init_rulebooks:
			return
		cachemap = {
			"character_rulebook": engine._characters_rulebooks_cache,
			"unit_rulebook": engine._units_rulebooks_cache,
			"character_thing_rulebook": engine._characters_things_rulebooks_cache,
			"character_place_rulebook": engine._characters_places_rulebooks_cache,
			"character_portal_rulebook": engine._characters_portals_rulebooks_cache,
		}
		branch, turn, tick = engine.time
		for rulebook, cache in cachemap.items():
			rulebook_name = RulebookName(Key((rulebook, name)))
			set_rb: Callable[
				[CharName, Branch, Turn, Tick, RulebookName], None
			] = getattr(engine.db, f"set_{rulebook}")
			set_rb(name, branch, turn, tick, rulebook_name)
			cache.store(name, branch, turn, tick, rulebook_name)

	class ThingMapping(MutableCharacterMapping):
		""":class:`Thing` objects that are in a :class:`Character`"""

		_book = "character_thing"

		engine: Engine = getatt("character.engine")
		name: CharName = getatt("character.name")

		def _get_rulebook_cache(self):
			return self.engine._characters_things_rulebooks_cache

		def __init__(self, character: Character):
			"""Store the character and initialize cache."""
			super().__init__(character)
			self.character = character

		def __iter__(self):
			cache = self.engine._things_cache
			char = self.name
			branch, turn, tick = self.engine.time
			return cache.iter_things(char, branch, turn, tick)

		def __contains__(self, thing):
			branch, turn, tick = self.engine.time
			args = self.character.name, thing, branch, turn, tick
			cache = self.engine._things_cache
			return cache.thing_exists(*args)

		def __len__(self):
			return self.engine._things_cache.count_keys(
				self.character.name, *self.engine.time
			)

		def __getitem__(self, thing: KeyHint):
			if thing not in self:
				raise KeyError("No such thing: {}".format(thing))
			return self._make_thing(NodeName(Key(thing)))

		def _make_thing(self, thing: NodeName, val: Thing | None = None):
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

		def __setitem__(self, thing: KeyHint, val: Mapping | dict):
			if not isinstance(val, Mapping):
				raise TypeError("Things are made from Mappings")
			if "location" not in val:
				raise ValueError("Thing needs location")
			thing = NodeName(Key(thing))
			val = dict(val)
			branch, turn, tick = self.engine._nbtt()
			self.engine._nodes_cache.store(
				self.character.name, thing, branch, turn, tick, True
			)
			self.engine._things_cache.store(
				self.character.name, thing, branch, turn, tick, val["location"]
			)
			self.engine.db.exist_node(
				self.character.name, thing, branch, turn, tick, True
			)
			self.engine.db.set_thing_loc(
				self.character.name,
				thing,
				branch,
				turn,
				tick,
				val.pop("location"),
			)
			th = self._make_thing(thing)
			th.clear()
			th.update({k: v for (k, v) in val.items() if k != "location"})

		def __delitem__(self, thing: KeyHint):
			self[thing].delete()

		def __repr__(self):
			return "{}.character[{}].thing".format(
				repr(self.engine), repr(self.name)
			)

	class PlaceMapping(MutableCharacterMapping):
		""":class:`Place` objects that are in a :class:`Character`"""

		_book = "character_place"

		def _get_rulebook_cache(self):
			return self.engine._characters_places_rulebooks_cache

		def update(self, __m: dict, **kwargs) -> None:
			self.character.node.update(__m, **kwargs)

		def __init__(self, character: Character):
			super().__init__(character)
			self.character = character
			self.engine = character.engine

		@cached_property
		def _iter_stuff(
			self,
		) -> tuple[
			Callable[[CharName, Branch, Turn, Tick], Iterator[NodeName]],
			Callable[[CharName, NodeName, Branch, Turn, Tick], bool],
			Callable[[CharName, NodeName, Branch, Turn, Tick], bool],
			CharName,
			Time,
		]:
			return (
				self.engine._nodes_cache.iter_nodes,
				self.engine._nodes_cache.node_exists,
				self.engine._things_cache.thing_exists,
				self.character.name,
				self.engine.time,
			)

		def __iter__(self):
			iter_nodes, nodes_contains, things_contains, charn, btt = (
				self._iter_stuff
			)
			branch, turn, tick = btt
			for node in iter_nodes(charn, branch, turn, tick):
				if nodes_contains(
					charn, node, branch, turn, tick
				) and not things_contains(charn, node, branch, turn, tick):
					yield node

		def __len__(self):
			n = 0
			for n, _ in enumerate(self, start=1):
				pass
			return n

		@cached_property
		def _contains_stuff(
			self,
		) -> tuple[
			Callable[[CharName, NodeName, Branch, Turn, Tick], bool],
			Callable[[CharName, NodeName, Branch, Turn, Tick], bool],
			CharName,
			Time,
		]:
			return (
				self.engine._nodes_cache.node_exists,
				self.engine._things_cache.thing_exists,
				self.character.name,
				self.engine.time,
			)

		def __contains__(self, place: KeyHint) -> bool:
			nodes_contains, things_contains, charn, btt = self._contains_stuff
			branch, turn, tick = btt
			place = NodeName(Key(place))
			return nodes_contains(
				charn, place, branch, turn, tick
			) and not things_contains(charn, place, branch, turn, tick)

		@cached_property
		def _get_stuff(
			self,
		) -> tuple[
			Callable[[CharName, NodeName, Branch, Turn, Tick], bool],
			Callable[[CharName, NodeName, Branch, Turn, Tick], bool],
			CharName,
			Time,
			dict,
			Character,
		]:
			return (
				self.engine._nodes_cache.node_exists,
				self.engine._things_cache.thing_exists,
				self.character.name,
				self.engine.time,
				self.engine._node_objs,
				self.character,
			)

		def __getitem__(self, place: KeyHint) -> Place:
			(nodes_contains, things_contains, charn, btt, cache, character) = (
				self._get_stuff
			)
			branch, turn, tick = btt
			place = NodeName(Key(place))
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

		@cached_property
		def _set_stuff(
			self,
		) -> tuple[
			Callable[[CharName, NodeName], bool],
			Callable[[CharName, NodeName, bool], None],
			Callable[[CharName, NodeName], Place | Thing],
			CharName,
			Character,
		]:
			return (
				self.engine._node_exists,
				self.engine._exist_node,
				self.engine._get_node,
				self.character.name,
				self.character,
			)

		def __setitem__(self, place: KeyHint, v: Place | Mapping | dict):
			(node_exists, exist_node, get_node, charn, character) = (
				self._set_stuff
			)
			place = NodeName(Key(place))
			exist_node(charn, place, True)
			pl = get_node(character, place)
			if not isinstance(pl, Place):
				raise KeyError(
					"{} is a {}, not a place".format(place, type(pl).__name__)
				)
			pl.update(v)

		def __delitem__(self, place: KeyHint):
			self[place].delete()

		def __repr__(self):
			return "{}.character[{}].place".format(
				repr(self.character.engine), repr(self.character.name)
			)

	class ThingPlaceMapping(GraphNodeMapping):
		"""GraphNodeMapping but for Place and Thing"""

		name: CharName = getatt("character.name")

		def __init__(self, character: Character):
			"""Store the character."""
			super().__init__(character)
			Signal.__init__(self)
			self.engine = character.engine
			self._placemap = character.place

		@cached_property
		def _contains_stuff(
			self,
		) -> tuple[Callable[[CharName, NodeName], bool], CharName]:
			return self.engine._node_exists, self.character.name

		def __contains__(self, k: KeyHint):
			node_exists, charn = self._contains_stuff
			return node_exists(charn, NodeName(Key(k)))

		@cached_property
		def _getitem_stuff(
			self,
		) -> tuple[
			Callable[[CharName, NodeName], bool],
			CharName,
			Callable[[CharName | Character, NodeName], Place | Thing],
			Character,
		]:
			return (
				self.engine._node_exists,
				self.character.name,
				self.engine._get_node,
				self.character,
			)

		def __getitem__(self, k: KeyHint) -> Thing | Place:
			node_exists, charn, get_node, character = self._getitem_stuff
			k = NodeName(Key(k))
			if not node_exists(charn, k):
				raise KeyError("No such node: " + str(k))
			return get_node(character, k)

		def __setitem__(self, k: KeyHint, v: Place | Mapping | dict):
			self._placemap[k] = v

		@cached_property
		def _delitem_stuff(
			self,
		) -> tuple[
			Callable[[CharName, NodeName], bool],
			CharName,
			Callable[[CharName, NodeName], bool],
			MutableCharacterMapping,
			MutableCharacterMapping,
		]:
			return (
				self.engine._node_exists,
				self.character.name,
				self.engine._is_thing,
				self.character.thing,
				self.character.place,
			)

		def __delitem__(self, k: KeyHint):
			(node_exists, charn, is_thing, thingmap, placemap) = (
				self._delitem_stuff
			)
			k = NodeName(Key(k))
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
		character: Character

		@cached_property
		def _cporh(self):
			return self.character.engine._characters_portals_rulebooks_cache

		@cached_property
		def _getitem_stuff(self):
			return (
				self.character.engine._node_exists,
				self.character.name,
				self._cache,
			)

		@cached_property
		def _setitem_stuff(self):
			return (self._cache, self.Successors)

		def _get_rulebook_cache(self):
			return self._cporh

		def __getitem__(self, orig: KeyHint) -> Successors:
			node_exists, charn, cache = self._getitem_stuff
			orig = NodeName(Key(orig))
			if node_exists(charn, orig):
				if orig not in cache:
					cache[orig] = self.Successors(self, orig)
				return cache[orig]
			raise KeyError("No such node")

		def __delitem__(self, orig: KeyHint):
			super().__delitem__(orig)

		def update(
			self, other: dict[KeyHint, dict[KeyHint, ValueHint]], **kwargs
		):
			"""Recursively update the stats of all portals

			Input should be a dictionary of dictionaries of dictionaries
			--just like networkx ``DiGraph._edge``.

			This will create portals as needed, but will only delete
			them if you set their value to ``None``. Likewise, stats
			not specified in the input will be left untouched, if they
			are already present, but you can set them to ``None`` to
			delete them.

			"""
			kwargs: dict[KeyHint, dict[KeyHint, ValueHint]]
			engine = self.engine
			planning = engine._planning
			forward = engine._forward
			branch, turn, start_tick = engine.time
			exist_edge = engine.db.exist_edge
			edge_val_set = engine.db.edge_val_set
			store_edge = engine._edges_cache.store
			store_edge_val = engine._edge_val_cache.store
			iter_edge_keys = engine._edge_val_cache._iter_entities_or_keys
			charn = self.character.name
			tick = start_tick + 1
			with timer(
				"seconds spent updating PortalSuccessorsMapping", engine.debug
			):
				for orig, dests in chain(other.items(), kwargs.items()):
					orig = NodeName(Key(orig))
					for dest, kvs in dests.items():
						dest = NodeName(Key(dest))
						if kvs is ...:
							for k in iter_edge_keys(
								charn,
								orig,
								dest,
								branch,
								turn,
								start_tick,
								forward=forward,
							):
								k = Stat(k)
								store_edge_val(
									charn,
									orig,
									dest,
									k,
									branch,
									turn,
									tick,
									Value(...),
									planning=planning,
									forward=forward,
									loading=True,
								)
								edge_val_set(
									charn,
									orig,
									dest,
									k,
									branch,
									turn,
									tick,
									Value(...),
								)
								tick += 1
							store_edge(
								charn,
								orig,
								dest,
								branch,
								turn,
								tick,
								False,
								planning=planning,
								forward=forward,
								loading=True,
							)
							exist_edge(
								charn, orig, dest, branch, turn, tick, False
							)
							tick += 1
						else:
							store_edge(
								charn,
								orig,
								dest,
								branch,
								turn,
								tick,
								True,
								planning=planning,
								forward=forward,
								loading=True,
							)
							exist_edge(
								charn, orig, dest, branch, turn, tick, True
							)
							tick += 1
							for k, v in kvs.items():
								k = Stat(Key(k))
								v = Value(v)
								store_edge_val(
									charn,
									orig,
									dest,
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
									k,
									branch,
									turn,
									tick,
									v,
								)
								tick += 1
			self.engine.tick = tick

		class Successors(DiGraphSuccessorsMapping.Successors):
			"""Mapping for possible destinations from some node."""

			character: Character
			engine: Engine

			@cached_property
			def _getitem_stuff(
				self,
			) -> tuple[
				Callable[[Character | CharName, NodeName, NodeName], Portal],
				Character,
				NodeName,
			]:
				return self.engine._get_edge, self.character, self.orig

			@cached_property
			def _setitem_stuff(
				self,
			) -> tuple[
				Callable[[CharName, NodeName, NodeName], bool],
				Callable[[CharName, NodeName, NodeName, bool], None],
				CharName,
				NodeName,
				Callable[[Character | CharName, NodeName, NodeName], Portal],
				Character,
				Callable[
					[
						CharName,
						NodeName,
						NodeName,
						Key,
						Branch,
						Turn,
						Tick,
						Value,
					],
					None,
				],
				Callable[
					[
						CharName,
						NodeName,
						NodeName,
						Stat,
						Branch,
						Turn,
						Tick,
						Value,
					],
					None,
				],
				Callable[[], Time],
			]:
				engine = self.engine
				graph = self.character
				return (
					engine._edge_exists,
					engine._exist_edge,
					graph.name,
					self.orig,
					engine._get_edge,
					graph,
					engine.db.edge_val_set,
					engine._edge_val_cache.store,
					engine._nbtt,
				)

			def __getitem__(self, dest: KeyHint) -> Portal:
				get_edge, graph, orig = self._getitem_stuff
				if dest in self:
					return get_edge(graph, orig, NodeName(Key(dest)))
				raise KeyError("No such portal", graph, orig, dest)

			def __setitem__(
				self,
				dest: KeyHint,
				value: Portal | Mapping | dict[KeyHint, ValueHint] | type(...),
			):
				if value is ...:
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
				dest = NodeName(Key(dest))
				exist_edge(charn, orig, dest, True)
				for k, v in value.items():
					k = Stat(Key(k))
					v = Value(v)
					branch, turn, tick = nbtt()
					db_edge_val_set(
						charn, orig, dest, k, branch, turn, tick, v
					)
					edge_val_cache_store(
						charn, orig, dest, k, branch, turn, tick, v
					)

			def __delitem__(self, dest: KeyHint):
				if dest not in self:
					raise KeyError("No portal to {}".format(dest))
				self[dest].delete()

			def update(
				self,
				other: dict[KeyHint, dict[KeyHint, ValueHint]] | None = None,
				**kwargs,
			):
				other: dict[NodeName, StatDict]
				kwargs: dict[NodeName, StatDict]
				if other is None:
					it = kwargs.items()
				else:
					it = chain(other.items(), kwargs.items())
				for dest, vs in it:
					self.character.add_edge(self.orig, dest, **vs)

	adj_cls = PortalSuccessorsMapping

	class PortalPredecessorsMapping(DiGraphPredecessorsMapping, RuleFollower):
		"""Mapping of nodes that have at least one incoming edge.

		Maps to another mapping keyed by the origin nodes, which maps to
		Portal objects.

		"""

		_book = "character_portal"

		def __init__(self, graph: Character):
			super().__init__(graph)
			self._cporc = graph.engine._characters_portals_rulebooks_cache

		def _get_rulebook_cache(self):
			return self._cporc

		class Predecessors(DiGraphPredecessorsMapping.Predecessors):
			"""Mapping of possible origins from some destination."""

			character: Character
			engine: Engine

			@cached_property
			def _setitem_stuff(
				self,
			) -> tuple[Character, CharName, NodeName, dict]:
				return (
					self.character,
					self.character.name,
					self.dest,
					self.engine._edge_objs,
				)

			def __setitem__(self, orig: NodeName, value: StatDict):
				graph, graph_name, dest, portal_objs = self._setitem_stuff
				key = (graph_name, orig, dest)
				if key not in portal_objs:
					portal_objs[key] = Portal(graph, orig, dest)
				p = portal_objs[key]
				p.engine._exist_edge(graph_name, orig, dest)
				p.clear()
				p.update(value)

	pred_cls = PortalPredecessorsMapping

	class UnitGraphMapping(Mapping, DiGraphMappingMixin, RuleFollower):
		"""A mapping of other characters in which one has a unit."""

		class CharacterUnitMapping(Mapping):
			"""Mapping of units of one Character in another Character."""

			def __init__(
				self, outer: Character.UnitGraphMapping, graphn: CharName
			):
				"""Store this character and the name of the other one"""
				self.character = outer.character
				self.engine = engine = outer.engine
				self.name = name = outer.name
				self.graph = graphn
				unitcache = engine._unitness_cache
				btt = engine.time
				self._iter_stuff = iter_stuff = (
					unitcache.get_char_graph_units,
					unitcache.contains_unit,
					name,
					graphn,
					btt,
				)
				self._contains_stuff = (
					unitcache._base_retrieve,
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
				self._only_stuff = (
					get_node,
					engine.character,
					graphn,
					engine.time,
				)

			def __iter__(self):
				"""Iterate over names of unit nodes"""
				get_char_graph_avs, validate, name, graphn, btt = (
					self._iter_stuff
				)
				for unit in get_char_graph_avs(name, graphn, *btt):
					if validate(name, graphn, unit, *btt):
						yield unit

			def __contains__(self, av: KeyHint):
				base_retrieve, name, graphn, btt = self._contains_stuff
				return (
					base_retrieve(
						(name, graphn, av, *btt()),
						store_hint=False,
						retrieve_hint=False,
					)
					is True
				)

			def __len__(self):
				"""Number of units of this character in that graph"""
				get_char_graph_avs, validate, name, graphn, btt = (
					self._iter_stuff
				)
				n = 0
				for that in get_char_graph_avs(name, graphn, *btt()):
					if validate(name, graphn, that, *btt()):
						n += 1
				return n

			def __getitem__(self, av: KeyHint) -> Place | Thing:
				(
					get_char_graph_avs,
					unitness_cache_has,
					name,
					graphn,
					btt,
					get_node,
					graphn,
					charmap,
				) = self._getitem_stuff
				if unitness_cache_has(name, graphn, av, *btt()):
					return get_node(charmap[graphn], av)
				raise KeyError("No unit: {}".format(av))

			@property
			def only(self) -> Place | Thing:
				"""If I have only one unit, return it; else error"""

				get_node, charmap, graphn, btt = self._only_stuff
				mykey = singleton_get(self.keys())
				if mykey is None:
					raise AttributeError(
						"No unit, or more than one",
						self.character.name,
						*btt,
					)
				try:
					return get_node(charmap[graphn], mykey)
				except KeyError as ex:
					raise AttributeError(
						"The unit doesn't exist",
						graphn,
						mykey,
						*btt,
					) from ex

			def __repr__(self):
				return "{}.character[{}].unit".format(
					repr(self.engine), repr(self.name)
				)

		_book = "unit"

		engine = getatt("character.engine")
		name = getatt("character.name")

		def _get_rulebook_cache(self):
			return self.engine._units_rulebooks_cache

		def __init__(self, char: Character):
			"""Remember my character."""
			self.character = char

		@cached_property
		def _char_unit_cache(self) -> dict[CharName, CharacterUnitMapping]:
			return {}

		@cached_property
		def _iter_stuff(
			self,
		) -> tuple[
			Callable[[CharName, Branch, Turn, Tick], Iterator[CharName]],
			CharName,
			TimeSignal,
		]:
			return (
				self.engine._unitness_cache.iter_char_graphs,
				self.character.name,
				self.engine.time,
			)

		def __call__(self, unit: Place | Thing):
			"""Add the unit

			It must be an instance of Place or Thing.

			"""
			self.character.add_unit(unit)

		def __iter__(self):
			"""Iterate over graphs with unit nodes in them"""
			get_char_graphs, charn, btt = self._iter_stuff
			branch, turn, tick = btt
			for graph in get_char_graphs(charn, branch, turn, tick):
				if graph in self:
					yield graph

		@cached_property
		def _contains_stuff(
			self,
		) -> tuple[
			Callable[[CharName, CharName, Branch, Turn, Tick], bool],
			CharName,
			TimeSignal,
		]:
			return (
				self.engine._unitness_cache.dict_cache.contains_graph,
				self.character.name,
				self.engine.time,
			)

		def __contains__(self, k: KeyHint | CharName):
			retrieve, charn, btt = self._contains_stuff
			got = retrieve(charn, CharName(Key(k)), *btt)
			return got is not ... and not isinstance(got, Exception)

		def __len__(self):
			"""Number of graphs in which I have a unit."""
			n = 0
			for n, _ in enumerate(self, start=1):
				pass
			return n

		def _get_char_unit_cache(self, g: CharName):
			if g not in self:
				raise KeyError
			if g not in self._char_unit_cache:
				self._char_unit_cache[g] = self.CharacterUnitMapping(self, g)
			return self._char_unit_cache[g]

		def __getitem__(self, g):
			return self._get_char_unit_cache(g)

		@cached_property
		def _node_stuff(
			self,
		) -> tuple[
			Callable[[CharName], CharacterUnitMapping],
			Callable[[CharName], CharName],
			CharName,
			TimeSignal,
		]:
			return (
				self._get_char_unit_cache,
				self.engine._unitness_cache.get_char_only_graph,
				self.character.name,
				self.engine.time,
			)

		@property
		def node(self) -> dict[NodeName, Place | Thing]:
			"""If I have units in only one graph, return a map of them

			Otherwise, raise AttributeError.

			"""
			get_char_av_cache: MethodType
			get_char_av_cache, get_char_only_graph, charn, btt = (
				self._node_stuff
			)
			try:
				return get_char_av_cache(get_char_only_graph(charn, *btt))
			except KeyError:
				raise AttributeError(
					"I have no unit, or I have units in many graphs"
				)

		@cached_property
		def _only_stuff(
			self,
		) -> tuple[
			Callable[
				[CharName, Branch, Turn, Tick], tuple[CharName, NodeName]
			],
			CharName,
			TimeSignal,
			Callable[[Character | CharName, NodeName], Thing | Place | None],
			Mapping[CharName, Character],
		]:
			return (
				self.engine._unitness_cache.get_char_only_unit,
				self.character.name,
				self.engine.time,
				self.engine._get_node,
				self.engine.character,
			)

		@property
		def only(self) -> Place | Thing:
			"""If I have only one unit, this is it

			Otherwise, raise AttributeError.

			"""
			get_char_only_av, charn, btt, get_node, charmap = self._only_stuff
			try:
				charn, noden = get_char_only_av(charn, *btt)
				return get_node(charmap[charn], noden)
			except (KeyError, TypeError):
				raise AttributeError("I have no unit, or more than one unit")

	def facade(self) -> CharacterFacade:
		"""Return a temporary copy of this Character

		A Facade looks like its :class:`Character`, but doesn't do any of the
		stuff Characters do to save changes to the database, nor enable
		time travel. This makes it much speedier to work with.

		"""
		return CharacterFacade(character=self)

	def __copy__(self) -> CharacterFacade:
		return self.facade()

	def __deepcopy__(self, memo) -> CharacterFacade:
		eng = EngineFacade(None)
		me = eng.new_character(self.name)
		for stat, val in self.stat.items():
			me.stat[stat] = me.deep_copy(unwrap(val), memo)
		for plname, place in self.place.items():
			fake = me.new_place(plname)
			for k, v in place.items():
				fake[k] = me.deep_copy(unwrap(v), memo)
		for thname, thing in self.thing.items():
			fake = me.new_thing(thname, thing.location.name)
			for k, v in thing.items():
				fake[k] = me.deep_copy(unwrap(v), memo)
		for orig, dests in self.portal.items():
			for dest, portal in dests.items():
				fake = me.new_portal(orig, dest)
				for k, v in portal.items():
					fake[k] = me.deep_copy(unwrap(v), memo)
		return me

	def add_place(self, node_for_adding: KeyHint | NodeName, **attr):
		"""Add a new Place"""
		attr: StatDict
		self.place[node_for_adding] = attr

	def add_places_from(self, seq: Iterable[KeyHint | NodeName], **attrs):
		attrs: StatDict
		for place in seq:
			self.add_place(place, **attrs)

	def remove_place(self, place: KeyHint | NodeName) -> None:
		"""Remove an existing Place"""
		if place in self.place:
			self.remove_node(place)
		else:
			raise KeyError("No such place: {}".format(place))

	def remove_thing(self, thing: KeyHint | NodeName) -> None:
		"""Remove an existing Thing"""
		if thing in self.thing:
			self.remove_node(thing)
		else:
			raise KeyError("No such thing: {}".format(thing))

	def add_thing(
		self,
		name: KeyHint | NodeName,
		location: KeyHint | NodeName,
		**kwargs: dict[KeyHint | Stat, ValueHint | Value],
	):
		"""Make a new Thing and set its location"""
		if name in self.thing:
			raise WorldIntegrityError(
				"Already have a Thing named {}".format(name)
			)
		starter: ThingDict = self.node_dict_factory()
		starter.update((Stat(Key(k)), Value(v)) for (k, v) in kwargs.items())
		if isinstance(location, Node):
			location = location.name
		starter["location"] = Value(location)
		self.thing[name] = starter
		if name not in self._succ:
			self._succ[name] = self.adjlist_inner_dict_factory()
			self._pred[name] = self.adjlist_inner_dict_factory()

	def add_things_from(
		self,
		seq: Iterable[
			tuple[KeyHint | NodeName, KeyHint | NodeName]
			| tuple[
				KeyHint | NodeName,
				KeyHint | NodeName,
				dict[KeyHint | Stat, ValueHint | Value],
			]
		],
		**attrs,
	) -> None:
		"""Make many new Things"""
		attrs: StatDict
		for tup in seq:
			name = tup[0]
			location = tup[1]
			kwargs = tup[2] if len(tup) > 2 else attrs
			self.add_thing(name, location, **kwargs)

	def place2thing(
		self, name: KeyHint | NodeName, location: KeyHint | NodeName
	) -> None:
		"""Turn a Place into a Thing with the given location.

		It will keep all its attached Portals.

		"""
		name = NodeName(Key(name))
		location = NodeName(Key(location))
		self.engine._set_thing_loc(self.name, name, location)
		if (self.name, name) in self.engine._node_objs:
			obj = self.engine._node_objs[self.name, name]
			thing = Thing(self, name)
			for port in obj.portal.values():
				port.origin = thing
			for port in obj.preportal.values():
				port.destination = thing
			self.engine._node_objs[self.name, name] = thing

	def thing2place(self, name: KeyHint | NodeName) -> None:
		"""Unset a Thing's location, and thus turn it into a Place."""
		name = NodeName(Key(name))
		self.engine._set_thing_loc(self.name, name, None)
		if (self.name, name) in self.engine._node_objs:
			thing = self.engine._node_objs[self.name, name]
			place = Place(self, name)
			for port in thing.portals():
				port.origin = place
			for port in thing.preportals():
				port.destination = place
			self.engine._node_objs[self.name, name] = place

	def add_portal(
		self,
		origin: KeyHint | NodeName,
		destination: KeyHint | NodeName,
		**kwargs,
	) -> None:
		"""Connect the origin to the destination with a :class:`Portal`.

		Keyword arguments are attributes of the :class:`Portal`.

		"""
		kwargs: StatDict
		if isinstance(origin, Node):
			origin = origin.name
		if origin not in self.place and origin not in self.thing:
			self.add_place(origin)
		if isinstance(destination, Node):
			destination = destination.name
		if destination not in self.place:
			self.add_place(destination)
		branch, turn, tick = self.engine._nbtt()
		self.engine._edges_cache.store(
			self.name, origin, destination, branch, turn, tick, True
		)
		self.engine.db.exist_edge(
			self.name, origin, destination, branch, turn, tick, True
		)
		for k, v in kwargs.items():
			branch, turn, tick = self.engine._nbtt()
			if k == "rulebook":
				if not isinstance(v, Key):
					raise TypeError("Not a valid rulebook name", v)
				v = RulebookName(v)
				self.engine._portals_rulebooks_cache.store(
					self.name, origin, destination, branch, turn, tick, v
				)
				self.engine.db.set_portal_rulebook(
					self.name, origin, destination, branch, turn, tick, v
				)
				continue
			self.engine._edge_val_cache.store(
				self.name, origin, destination, k, branch, turn, tick, v
			)
			self.engine.db.edge_val_set(
				self.name, origin, destination, k, branch, turn, tick, v
			)

	def new_portal(
		self,
		origin: KeyHint | NodeName,
		destination: KeyHint | NodeName,
		**kwargs,
	) -> Portal:
		"""Create a portal and return it"""
		kwargs: StatDict
		self.add_portal(origin, destination, **kwargs)
		return self.engine._get_edge(
			self, NodeName(Key(origin)), NodeName(Key(destination))
		)

	def add_portals_from(
		self,
		seq: Iterable[
			tuple[KeyHint, KeyHint]
			| tuple[KeyHint, KeyHint, dict[KeyHint, ValueHint]]
		],
		**kwargs,
	):
		"""Make portals for a sequence of (origin, destination) pairs

		Actually, triples are acceptable too, in which case the third
		item is a dictionary of stats for the new :class:`Portal`.
		"""
		for tup in seq:
			orig = tup[0]
			dest = tup[1]
			kwarrgs = tup[2] if len(tup) > 2 else kwargs
			self.add_portal(orig, dest, **kwarrgs)

	def add_unit(
		self,
		a: Place | Thing | Character | KeyHint,
		b: Place | Thing | KeyHint | None = None,
	) -> None:
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
			elif not isinstance(a, Key):
				raise TypeError(
					"when called with two arguments, "
					"the first is a character or its name"
				)
			else:
				g = a
			if isinstance(b, (Place, Thing)):
				n = b.name
			elif not isinstance(b, Key):
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
		self.engine._unitness_cache.store(
			self.name, g, n, branch, turn, tick, True
		)
		self.engine.db.unit_set(self.name, g, n, branch, turn, tick, True)

	def remove_unit(
		self,
		a: Place | Thing | Character | KeyHint,
		b: Place | Thing | KeyHint | None = None,
	):
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
		branch, turn, tick = self.engine._nbtt()
		self.engine._unitness_cache.store(
			self.name, g, n, branch, turn, tick, False
		)
		self.engine.db.unit_set(self.name, g, n, branch, turn, tick, False)

	def historical(self, key: KeyHint) -> UnitsAlias | CharacterStatAlias:
		"""Get a historical view on the given stat

		This functions like the value of the stat, but changes
		when you time travel. Comparisons performed on the
		historical view can be passed to ``engine.turns_when``
		to find out when the comparison held true.

		"""
		if key == "units":
			return UnitsAlias(
				entity=self.stat, stat=Stat(Key("units")), engine=self.engine
			)
		return CharacterStatAlias(
			entity=self.stat, stat=Stat(Key(key)), engine=self.engine
		)
