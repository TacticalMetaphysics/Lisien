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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.f
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence, Set
from itertools import chain
from types import GenericAlias
from typing import (
	TYPE_CHECKING,
	Annotated,
	Any,
	Callable,
	Literal,
	MutableMapping,
	NewType,
	TypeAlias,
	TypeGuard,
)

import networkx
import networkx as nx
from annotated_types import Ge
from networkx import NetworkXError

from .wrap import (
	DictWrapper,
	ListWrapper,
	MutableMappingUnwrapper,
	OrderlySet,
	SetWrapper,
	SpecialMapping,
	unwrap_items,
	wrapval,
)

if TYPE_CHECKING:
	from .engine import Engine

_Key = str | int | float | None | tuple["_Key", ...] | frozenset["_Key"]


def is_valid_key(obj: _Key) -> TypeGuard[Key]:
	"""Is this an object that Lisien can serialize as a key?"""
	return (
		obj is None
		or isinstance(obj, (str, int, float))
		or (
			isinstance(obj, (tuple, frozenset))
			and all(is_valid_key(elem) for elem in obj)
		)
	)


class _KeyMeta(type):
	def __instancecheck__(self, instance) -> TypeGuard[Key]:
		return is_valid_key(instance)

	def __call__(self, obj: _Key) -> Key:
		if is_valid_key(obj):
			return obj
		raise TypeError("Not a valid key", obj)

	def __class_getitem__(cls, item):
		return GenericAlias(cls, item)


class Key(metaclass=_KeyMeta):
	def __new__(cls, obj: _Key) -> Key:
		if not is_valid_key(obj):
			raise TypeError("Invalid key")
		return obj


_Value: TypeAlias = (
	_Key
	| dict[_Key, "_Value"]
	| tuple["_Value", ...]
	| list["_Value"]
	| set["_Value"]
	| frozenset["_Value"]
	| DictWrapper
	| ListWrapper
	| SetWrapper
	| Set["_Value"]
	| OrderlySet["_Value"]
	| Mapping[_Key, "_Value"]
	| type(...)
)


def is_valid_value(obj: _Value) -> TypeGuard[Value]:
	"""Is this an object that Lisien can serialize as a value?"""
	return (
		obj is ...
		or is_valid_key(obj)
		or isinstance(obj, Node)
		or isinstance(obj, Edge)
		or isinstance(obj, DiGraph)
		or (
			isinstance(obj, (dict, DictWrapper))
			and all(map(is_valid_key, obj.keys()))
			and all(map(is_valid_value, obj.values()))
		)
		or (
			isinstance(obj, (Set, Sequence))
			and isinstance(obj, Iterable)
			and all(map(is_valid_value, obj))
		)
		or (
			isinstance(obj, nx.DiGraph)
			and all(map(is_valid_key, obj.graph.keys()))
			and all(map(is_valid_value, obj.graph.values()))
			and all(
				is_valid_key(k) and is_valid_value(v)
				for node in obj.nodes().values()
				for (k, v) in node.items()
			)
			and all(
				is_valid_key(orig)
				and is_valid_key(dest)
				and is_valid_key(k)
				and is_valid_value(v)
				for orig in obj.adj
				for dest in obj.adj[orig]
				for (k, v, *_) in obj.adj[orig][dest]
			)
		)
	)


class _ValueMeta(type):
	def __instancecheck__(self, instance) -> TypeGuard[Value]:
		return is_valid_value(instance)

	def __call__(self, obj: _Value) -> Value:
		if is_valid_value(obj):
			return obj
		raise TypeError("Not a valid value", obj)

	def __class_getitem__(cls, item):
		return GenericAlias(cls, item)


class Value(metaclass=_ValueMeta):
	def __new__(cls, obj: _Value) -> Value:
		if not is_valid_value(obj):
			raise TypeError("Invalid value")
		return obj


Stat = NewType("Stat", Key)
EternalKey = NewType("EternalKey", Key)
UniversalKey = NewType("UniversalKey", Key)
Branch = NewType("Branch", str)
Turn = NewType("Turn", Annotated[int, Ge(0)])
Tick = NewType("Tick", Annotated[int, Ge(0)])
Time: TypeAlias = tuple[Branch, Turn, Tick]
LinearTime: TypeAlias = tuple[Turn, Tick]
TimeWindow: TypeAlias = tuple[Branch, Turn, Tick, Turn, Tick]
Plan = NewType("Plan", Annotated[int, Ge(0)])
CharName = NewType("CharName", Key)
NodeName = NewType("NodeName", Key)
EntityKey: TypeAlias = (
	tuple[CharName]
	| tuple[CharName, NodeName]
	| tuple[CharName, NodeName, NodeName]
)
RulebookName = NewType("RulebookName", Key)
RulebookPriority = NewType("RulebookPriority", float)
RuleName = NewType("RuleName", str)
RuleNeighborhood: TypeAlias = Annotated[int, Ge(0)] | None
RuleBig = NewType("RuleBig", bool)
RuleFunc: TypeAlias = Callable[[Any], bool]
FuncName = NewType("FuncName", str)
FuncStoreName: TypeAlias = Literal[
	"trigger", "prereq", "action", "function", "method"
]
TriggerFuncName = NewType("TriggerFuncName", FuncName)
PrereqFuncName = NewType("PrereqFuncName", FuncName)
ActionFuncName = NewType("ActionFuncName", FuncName)
RuleFuncName: TypeAlias = TriggerFuncName | PrereqFuncName | ActionFuncName
UniversalKeyframe: TypeAlias = dict[UniversalKey, Value]
RuleKeyframe = dict[
	Literal["triggers", "prereqs", "actions", "neighborhood", "big"],
	list[TriggerFuncName]
	| list[PrereqFuncName]
	| list[ActionFuncName]
	| RuleNeighborhood
	| RuleBig,
]
RulebookKeyframe: TypeAlias = dict[
	RulebookName, tuple[list[RuleName], RulebookPriority]
]
UniversalRowType: TypeAlias = tuple[UniversalKey, Branch, Turn, Tick, Value]
RulebookRowType: TypeAlias = tuple[
	RulebookName,
	Branch,
	Turn,
	Tick,
	tuple[list[RuleName], RulebookPriority],
]
RuleRowType: TypeAlias = tuple[
	RuleName,
	Branch,
	Turn,
	Tick,
	list[TriggerFuncName]
	| list[PrereqFuncName]
	| list[ActionFuncName]
	| RuleNeighborhood
	| RuleBig,
]
TriggerRowType: TypeAlias = tuple[
	RuleName, Branch, Turn, Tick, list[TriggerFuncName]
]
PrereqRowType: TypeAlias = tuple[
	RuleName, Branch, Turn, Tick, list[PrereqFuncName]
]
ActionRowType: TypeAlias = tuple[
	RuleName, Branch, Turn, Tick, list[ActionFuncName]
]
RuleNeighborhoodRowType: TypeAlias = tuple[
	RuleName, Branch, Turn, Tick, RuleNeighborhood
]
RuleBigRowType: TypeAlias = tuple[RuleName, Branch, Turn, Tick, RuleBig]
GraphTypeStr: TypeAlias = Literal["DiGraph", "Deleted"]
GraphRowType: TypeAlias = tuple[CharName, Branch, Turn, Tick, GraphTypeStr]
NodeRowType: TypeAlias = tuple[CharName, NodeName, Branch, Turn, Tick, bool]
EdgeRowType: TypeAlias = tuple[
	CharName, NodeName, NodeName, Branch, Turn, Tick, bool
]
GraphValRowType: TypeAlias = tuple[CharName, Stat, Branch, Turn, Tick, Value]
NodeValRowType: TypeAlias = tuple[
	CharName, NodeName, Stat, Branch, Turn, Tick, Value
]
EdgeValRowType: TypeAlias = tuple[
	CharName, NodeName, NodeName, Stat, Branch, Turn, Tick, Value
]
ThingRowType: TypeAlias = tuple[
	CharName, NodeName, Branch, Turn, Tick, NodeName
]
UnitRowType: TypeAlias = tuple[
	CharName, CharName, NodeName, Branch, Turn, Tick, bool
]
CharRulebookRowType: TypeAlias = tuple[
	CharName, Branch, Turn, Tick, RulebookName
]
NodeRulebookRowType: TypeAlias = tuple[
	CharName, NodeName, Branch, Turn, Tick, RulebookName
]
PortalRulebookRowType: TypeAlias = tuple[
	CharName, NodeName, NodeName, Branch, Turn, Tick, RulebookName
]
StatDict: TypeAlias = dict[Stat | Literal["rulebook"], Value]
CharDict: TypeAlias = dict[
	Stat
	| Literal[
		"units",
		"character_rulebook",
		"unit_rulebook",
		"character_thing_rulebook",
		"character_place_rulebook",
		"character_portal_rulebook",
	],
	Value,
]
GraphValKeyframe: TypeAlias = dict[CharName, CharDict]
NodeValDict: TypeAlias = dict[NodeName, StatDict]
NodeKeyframe = NodeValDict
GraphNodeValKeyframe: TypeAlias = dict[CharName, NodeValDict]
EdgeValDict: TypeAlias = dict[NodeName, dict[NodeName, StatDict]]
EdgeKeyframe = EdgeValDict
GraphEdgeValKeyframe: TypeAlias = dict[CharName, EdgeValDict]
NodesDict: TypeAlias = dict[NodeName, bool]
GraphNodesKeyframe: TypeAlias = dict[CharName, NodesDict]
EdgesDict: TypeAlias = dict[NodeName, dict[NodeName, bool]]
GraphEdgesKeyframe: TypeAlias = dict[CharName, EdgesDict]
UnitsDict: TypeAlias = dict[CharName, dict[NodeName, bool]]
CharDelta: TypeAlias = dict[
	Stat
	| Literal[
		"character_rulebook",
		"unit_rulebook",
		"character_thing_rulebook",
		"character_place_rulebook",
		"character_portal_rulebook",
		"nodes",
		"node_val",
		"edges",
		"edge_val",
		"rulebooks",
		"units",
	],
	NodesDict
	| NodeValDict
	| EdgesDict
	| EdgeValDict
	| RulebookName
	| UnitsDict
	| Value,
]
DeltaDict: TypeAlias = dict[
	CharName,
	CharDelta | None,
]
KeyframeTuple: TypeAlias = tuple[
	CharName,
	Branch,
	Turn,
	Tick,
	GraphNodeValKeyframe,
	GraphEdgeValKeyframe,
	GraphValKeyframe,
]
Keyframe: TypeAlias = dict[
	Literal[
		"universal",
		"triggers",
		"prereqs",
		"actions",
		"neighborhood",
		"big",
		"rulebook",
		"nodes",
		"edges",
		"node_val",
		"edge_val",
		"graph_val",
	],
	GraphValKeyframe
	| GraphNodesKeyframe
	| GraphNodeValKeyframe
	| GraphEdgesKeyframe
	| GraphEdgeValKeyframe
	| dict[UniversalKey, Value]
	| dict[RuleName, list[TriggerFuncName]]
	| dict[RuleName, list[PrereqFuncName]]
	| dict[RuleName, list[ActionFuncName]]
	| dict[RuleName, RuleNeighborhood]
	| dict[RuleName, RuleBig]
	| dict[RulebookName, tuple[list[RuleName], RulebookPriority]],
]
SlightlyPackedDeltaType: TypeAlias = dict[
	bytes,
	dict[
		bytes,
		bytes
		| dict[
			bytes,
			bytes | dict[bytes, bytes | dict[bytes, bytes]],
		],
	],
]
RulebookTypeStr: TypeAlias = Literal[
	"character",
	"unit",
	"character_thing",
	"character_place",
	"character_portal",
]
CharacterRulebookTypeStr: TypeAlias = Literal[
	"character_rulebook",
	"unit_rulebook",
	"character_thing_rulebook",
	"character_place_rulebook",
	"character_portal_rulebook",
]


class EntityCollisionError(ValueError):
	"""For when there's a discrepancy between the kind of entity you're creating and the one by the same name"""


def getatt(attribute_name):
	"""An easy way to make an alias"""
	from operator import attrgetter

	ret = property(attrgetter(attribute_name))
	ret.__doc__ = "Alias to `{}`".format(attribute_name)
	return ret


_alleged_receivers = defaultdict(list)


class AllegedMapping(MutableMappingUnwrapper, SpecialMapping, ABC):
	"""Common amenities for mappings"""

	__slots__ = ()

	def clear(self):
		"""Delete everything"""
		for k in list(self.keys()):
			if k in self:
				del self[k]


class AbstractEntityMapping(AllegedMapping, ABC):
	__slots__ = ()
	db: "Engine"

	@abstractmethod
	def _get_cache(
		self, key: Key, branch: Branch, turn: Turn, tick: Tick
	) -> dict:
		raise NotImplementedError

	def _get_cache_now(self, key):
		return self._get_cache(key, *self.db._btt())

	@abstractmethod
	def _cache_contains(self, key, branch, turn, tick):
		raise NotImplementedError

	@abstractmethod
	def _set_db(self, key, branch, turn, tick, value):
		"""Set a value for a key in the database (not the cache)."""
		raise NotImplementedError

	@abstractmethod
	def _set_cache(self, key, branch, turn, tick, value):
		raise NotImplementedError

	def _del_db(self, key, branch, turn, tick):
		"""Delete a key from the database (not the cache)."""
		self._set_db(key, branch, turn, tick, ...)

	def _del_cache(self, key, branch, turn, tick):
		self._set_cache(key, branch, turn, tick, ...)

	def __getitem__(self, key):
		"""If key is 'graph', return myself as a dict, else get the present
		value of the key and return that

		"""

		return wrapval(self, key, self._get_cache_now(key))

	def __contains__(self, item):
		return item == "name" or self._cache_contains(item, *self.db._btt())

	def __setitem__(self, key, value):
		"""Set key=value at the present branch and revision"""
		if value is ...:
			raise ValueError(
				"Lisien uses the ellipsis to indicate that a key's been deleted"
			)
		branch, turn, tick = self.db._nbtt()
		try:
			if self._get_cache(key, branch, turn, tick) != value:
				self._set_cache(key, branch, turn, tick, value)
		except KeyError:
			self._set_cache(key, branch, turn, tick, value)
		self._set_db(key, branch, turn, tick, value)

	def __delitem__(self, key):
		branch, turn, tick = self.db._nbtt()
		self._del_cache(key, branch, turn, tick)
		self._del_db(key, branch, turn, tick)


class GraphMapping(AbstractEntityMapping):
	"""Mapping for graph attributes"""

	__slots__ = (
		"graph",
		"db",
		"_iter_stuff",
		"_cache_contains_stuff",
		"_len_stuff",
		"_get_stuff",
		"_set_db_stuff",
		"_set_cache_stuff",
		"_del_db_stuff",
		"_get_cache_stuff",
	)

	def __init__(self, graph):
		super().__init__(graph)
		self.graph = graph
		self.db = db = graph.db
		btt = db._btt
		graph_val_cache = db._graph_val_cache
		graphn = graph.name
		self._iter_stuff = (graph_val_cache.iter_keys, graphn, btt)
		self._cache_contains_stuff = (graph_val_cache.contains_key, graphn)
		self._len_stuff = (graph_val_cache.count_keys, graphn, btt)
		self._get_stuff = (self._get_cache, btt)
		graph_val_set = db.query.graph_val_set
		self._set_db_stuff = (graph_val_set, graphn)
		self._set_cache_stuff = (graph_val_cache.store, graphn)
		self._del_db_stuff = (graph_val_set, graphn)
		self._get_cache_stuff = (graph_val_cache.retrieve, graphn)

	def __iter__(self):
		iter_entity_keys, graphn, btt = self._iter_stuff
		yield "name"
		yield from iter_entity_keys(graphn, *btt())

	def __repr__(self):
		return f"<{self.__class__.__name__} for {self.graph.name} containing {dict(unwrap_items(self.items()))}>"

	def _cache_contains(self, key, branch, turn, tick):
		contains_key, graphn = self._cache_contains_stuff
		return contains_key(graphn, key, branch, turn, tick)

	def __len__(self):
		count_keys, graphn, btt = self._len_stuff
		return 1 + count_keys(graphn, *btt())

	def __getitem__(self, item):
		if item == "name":
			return self.graph.name
		return super().__getitem__(item)

	def __setitem__(self, key, value):
		if key == "name":
			raise KeyError("name cannot be changed after creation")
		super().__setitem__(key, value)

	def _get_cache(self, key, branch, turn, tick):
		retrieve, graphn = self._get_cache_stuff
		return retrieve(graphn, key, branch, turn, tick)

	def _get(self, key):
		get_cache, btt = self._get_stuff
		return get_cache(key, *btt())

	def _set_db(self, key, branch, turn, tick, value):
		graph_val_set, graphn = self._set_db_stuff
		graph_val_set(graphn, key, branch, turn, tick, value)

	def _set_cache(self, key, branch, turn, tick, value):
		store, graphn = self._set_cache_stuff
		store(graphn, key, branch, turn, tick, value)

	def _del_db(self, key, branch, turn, tick):
		graph_val_set, graphn = self._del_db_stuff
		graph_val_set(graphn, key, branch, turn, tick, ...)

	def clear(self):
		keys = set(self.keys())
		keys.remove("name")
		for k in keys:
			del self[k]

	def unwrap(self):
		return unwrap_items(self.items())

	def __eq__(self, other):
		if hasattr(other, "unwrap"):
			other = other.unwrap()
		other = other.copy()
		me = self.unwrap().copy()
		if "name" not in other:
			del me["name"]
		return me == other


class Node(AbstractEntityMapping):
	"""Mapping for node attributes"""

	__slots__ = (
		"graph",
		"name",
		"db",
		"_iter_stuff",
		"_cache_contains_stuff",
		"_len_stuff",
		"_get_cache_stuff",
		"_set_db_stuff",
		"_set_cache_stuff",
	)

	def _validate_node_type(self):
		return True

	def __init__(self, graph, node):
		"""Store name and graph"""
		super().__init__(graph)
		self.graph = graph
		self.name = node
		self.db = db = graph.db
		node_val_cache = db._node_val_cache
		graphn = graph.name
		btt = db._btt
		self._iter_stuff = (
			node_val_cache.iter_keys,
			graphn,
			node,
			btt,
		)
		self._cache_contains_stuff = (
			node_val_cache.contains_key,
			graphn,
			node,
		)
		self._len_stuff = (
			node_val_cache.count_keys,
			graphn,
			node,
			btt,
		)
		self._get_cache_stuff = (node_val_cache.retrieve, graphn, node)
		self._set_db_stuff = (db.query.node_val_set, graphn, node)
		self._set_cache_stuff = (db._node_val_cache.store, graphn, node)

	def __repr__(self):
		return "<{}(graph={}, name={})>".format(
			self.__class__.__name__, repr(self.graph), repr(self.name)
		)

	def __str__(self):
		return (
			f"Node of class {self.__class__.__name__} "
			f"in graph {self.graph.name} named {self.name}"
		)

	def __iter__(self):
		iter_entity_keys, graphn, node, btt = self._iter_stuff
		return iter_entity_keys(graphn, node, *btt())

	def _cache_contains(self, key, branch, turn, tick):
		contains_key, graphn, node = self._cache_contains_stuff
		return contains_key(graphn, node, key, branch, turn, tick)

	def __len__(self):
		count_entity_keys, graphn, node, btt = self._len_stuff
		return count_entity_keys(graphn, node, *btt())

	def _get_cache(self, key, branch, turn, tick):
		retrieve, graphn, node = self._get_cache_stuff
		return retrieve(graphn, node, key, branch, turn, tick)

	def _set_db(self, key, branch, turn, tick, value):
		node_val_set, graphn, node = self._set_db_stuff
		node_val_set(graphn, node, key, branch, turn, tick, value)

	def _set_cache(self, key, branch, turn, tick, value):
		store, graphn, node = self._set_cache_stuff
		store(graphn, node, key, branch, turn, tick, value)

	def __eq__(self, other):
		if not hasattr(other, "keys") or not callable(other.keys):
			return False
		if not hasattr(other, "name"):
			return False
		if self.name != other.name:
			return False
		if not hasattr(other, "graph"):
			return False
		if self.graph.name != other.graph.name:
			return False
		if self.keys() != other.keys():
			return False
		for key in self:
			if self[key] != other[key]:
				return False
		return True


class Edge(AbstractEntityMapping):
	"""Mapping for edge attributes"""

	__slots__ = (
		"graph",
		"orig",
		"dest",
		"db",
		"_iter_stuff",
		"_cache_contains_stuff",
		"_len_stuff",
		"_get_cache_stuff",
		"_set_db_stuff",
		"_set_cache_stuff",
	)

	set_db_time = set_cache_time = 0

	def __init__(self, graph, orig, dest):
		super().__init__(graph)
		self.graph = graph
		self.db = db = graph.db
		self.orig = orig
		self.dest = dest
		edge_val_cache = db._edge_val_cache
		graphn = graph.name
		btt = db._btt
		self._iter_stuff = (
			edge_val_cache.iter_keys,
			graphn,
			orig,
			dest,
			btt,
		)
		self._cache_contains_stuff = (
			edge_val_cache.contains_key,
			graphn,
			orig,
			dest,
		)
		self._len_stuff = (
			edge_val_cache.count_keys,
			graphn,
			orig,
			dest,
			btt,
		)
		self._get_cache_stuff = (
			edge_val_cache.retrieve,
			graphn,
			orig,
			dest,
		)
		self._set_db_stuff = (db.query.edge_val_set, graphn, orig, dest)
		self._set_cache_stuff = (edge_val_cache.store, graphn, orig, dest)

	def __repr__(self):
		return "<{} in graph {} from {} to {} containing {}>".format(
			self.__class__.__name__,
			self.graph.name,
			self.orig,
			self.dest,
			dict(self),
		)

	def __str__(self):
		return str(dict(self))

	def __iter__(self):
		iter_entity_keys, graphn, orig, dest, btt = self._iter_stuff
		return iter_entity_keys(graphn, orig, dest, *btt())

	def _cache_contains(self, key, branch, turn, tick):
		contains_key, graphn, orig, dest = self._cache_contains_stuff
		return contains_key(graphn, orig, dest, key, branch, turn, tick)

	def __len__(self):
		count_entity_keys, graphn, orig, dest, btt = self._len_stuff
		return count_entity_keys(graphn, orig, dest, *btt())

	def _get_cache(self, key, branch, turn, tick):
		retrieve, graphn, orig, dest = self._get_cache_stuff
		return retrieve(graphn, orig, dest, key, branch, turn, tick)

	def _set_db(self, key, branch, turn, tick, value):
		edge_val_set, graphn, orig, dest = self._set_db_stuff
		edge_val_set(graphn, orig, dest, key, branch, turn, tick, value)

	def _set_cache(self, key, branch, turn, tick, value):
		store, graphn, orig, dest = self._set_cache_stuff
		store(graphn, orig, dest, key, branch, turn, tick, value)


class GraphNodeMapping(AllegedMapping):
	"""Mapping for nodes in a graph"""

	__slots__ = ("graph",)

	db = getatt("graph.db")
	"""Alias to ``self.graph.db``"""

	def __init__(self, graph):
		super().__init__(graph)
		self.graph = graph

	def __iter__(self):
		"""Iterate over the names of the nodes"""
		now = self.db._btt()
		gn = self.graph.name
		nc = self.db._nodes_cache
		for entity in nc.iter_entities(gn, *now):
			if entity in self:
				yield entity

	def __eq__(self, other):
		from collections.abc import Mapping

		if not isinstance(other, Mapping):
			return NotImplemented
		if self.keys() != other.keys():
			return False
		for k in self.keys():
			me = self[k]
			you = other[k]
			if hasattr(me, "unwrap") and not hasattr(me, "no_unwrap"):
				me = me.unwrap()
			if hasattr(you, "unwrap") and not hasattr(you, "no_unwrap"):
				you = you.unwrap()
			if me != you:
				return False
		else:
			return True

	def __contains__(self, node):
		"""Return whether the node exists presently"""
		return self.db._nodes_cache.contains_entity(
			self.graph.name, node, *self.db._btt()
		)

	def __len__(self):
		"""How many nodes exist right now?"""
		return self.db._nodes_cache.count_entities(
			self.graph.name, *self.db._btt()
		)

	def __getitem__(self, node):
		"""If the node exists at present, return it, else throw KeyError"""
		if node not in self:
			raise KeyError
		return self.db._get_node(self.graph, node)

	def __setitem__(self, node, dikt):
		"""Only accept dict-like values for assignment. These are taken to be
		dicts of node attributes, and so, a new GraphNodeMapping.Node
		is made with them, perhaps clearing out the one already there.

		"""
		created = False
		db = self.db
		graph = self.graph
		gname = graph.name
		if not db._node_exists(gname, node):
			created = True
			db._exist_node(gname, node, True)
		n = db._get_node(graph, node)
		n.clear()
		n.update(dikt)

	def __delitem__(self, node):
		"""Indicate that the given node no longer exists"""
		if node not in self:
			raise KeyError("No such node")
		for succ in self.graph.adj[node]:
			del self.graph.adj[node][succ]
		for pred in self.graph.pred[node]:
			del self.graph.pred[node][pred]
		branch, turn, tick = self.db._nbtt()
		self.db.query.exist_node(
			self.graph.name, node, branch, turn, tick, False
		)
		self.db._nodes_cache.store(
			self.graph.name, node, branch, turn, tick, False
		)
		key = (self.graph.name, node)
		if node in self.db._node_objs:
			del self.db._node_objs[key]

	def __repr__(self):
		return f"<{self.__class__.__name__} containing {', '.join(map(repr, self.keys()))}>"

	def update(self, m, /, **kwargs):
		for node, value in chain(m.items(), kwargs.items()):
			if value is ...:
				del self[node]
			elif node not in self:
				self[node] = value
			else:
				self[node].update(value)


class GraphEdgeMapping(AllegedMapping):
	"""Provides an adjacency mapping and possibly a predecessor mapping
	for a graph.

	"""

	__slots__ = ("graph", "_cache")

	db = getatt("graph.db")
	"""Alias to ``self.graph.db``"""

	def __init__(self, graph):
		super().__init__(graph)
		self.graph = graph
		self._cache = {}

	def __eq__(self, other):
		"""Compare dictified versions of the edge mappings within me.

		As I serve custom Predecessor or Successor classes, which
		themselves serve the custom Edge class, I wouldn't normally be
		comparable to a networkx adjacency dictionary. Converting
		myself and the other argument to dicts allows the comparison
		to work anyway.

		"""
		if not hasattr(other, "keys"):
			return False
		if self.keys() != other.keys():
			return False
		for k in self.keys():
			if dict(self[k]) != dict(other[k]):
				return False
		return True

	def __iter__(self):
		return iter(self.graph.node)


class AbstractSuccessors(GraphEdgeMapping):
	__slots__ = ("graph", "container", "orig", "_cache")

	db = getatt("graph.db")
	"""Alias to ``self.graph.db``"""

	def _order_nodes(self, node):
		raise NotImplementedError

	def __init__(self, container, orig):
		"""Store container and node"""
		super().__init__(container.graph)
		self.container = container
		self.orig = orig

	def __iter__(self):
		"""Iterate over node IDs that have an edge with my orig"""
		for that in self.db._edges_cache.iter_successors(
			self.graph.name, self.orig, *self.db._btt()
		):
			if that in self:
				yield that

	def __contains__(self, dest):
		"""Is there an edge leading to ``dest`` at the moment?"""
		orig, dest = self._order_nodes(dest)
		return self.db._edges_cache.has_successor(
			self.graph.name, orig, dest, *self.db._btt()
		)

	def __len__(self):
		"""How many nodes touch an edge shared with my orig?"""
		n = 0
		for n, _ in enumerate(self, start=1):
			pass
		return n

	def _make_edge(self, dest):
		return Edge(self.graph, *self._order_nodes(dest))

	def __getitem__(self, dest):
		"""Get the edge between my orig and the given node"""
		if dest not in self:
			raise KeyError("No edge {}->{}".format(self.orig, dest))
		orig, dest = self._order_nodes(dest)
		return self.db._get_edge(self.graph, orig, dest)

	def __setitem__(self, dest, value):
		"""Set the edge between my orig and the given dest to the given
		value, a mapping.

		"""
		real_dest = dest
		orig, dest = self._order_nodes(dest)
		created = dest not in self
		if orig not in self.graph.node:
			self.graph.add_node(orig)
		if dest not in self.graph.node:
			self.graph.add_node(dest)
		branch, turn, tick = self.db._nbtt()
		self.db.query.exist_edge(
			self.graph.name, orig, dest, 0, branch, turn, tick, True
		)
		self.db._edges_cache.store(
			self.graph.name, orig, dest, branch, turn, tick, True
		)
		e = self[real_dest]
		e.clear()
		e.update(value)

	def __delitem__(self, dest):
		"""Remove the edge between my orig and the given dest"""
		branch, turn, tick = self.db._nbtt()
		orig, dest = self._order_nodes(dest)
		self.db.query.exist_edge(
			self.graph.name, orig, dest, 0, branch, turn, tick, False
		)
		self.db._edges_cache.store(
			self.graph.name, orig, dest, branch, turn, tick, None
		)

	def __repr__(self):
		cls = self.__class__
		return "<{}.{} object containing {}>".format(
			cls.__module__, cls.__name__, dict(self)
		)

	def clear(self):
		"""Delete every edge with origin at my orig"""
		for dest in list(self):
			del self[dest]


class GraphSuccessorsMapping(GraphEdgeMapping):
	"""Mapping for Successors (itself a MutableMapping)"""

	__slots__ = ("graph",)

	class Successors(AbstractSuccessors):
		__slots__ = ("graph", "container", "orig", "_cache")

		def _order_nodes(self, dest):
			if dest < self.orig:
				return (dest, self.orig)
			else:
				return (self.orig, dest)

	def __getitem__(self, orig):
		if orig not in self._cache:
			self._cache[orig] = self.Successors(self, orig)
		return self._cache[orig]

	def __setitem__(self, key, val):
		"""Wipe out any edges presently emanating from orig and replace them
		with those described by val

		"""
		if key in self:
			sucs = self[key]
			sucs.clear()
		else:
			sucs = self._cache[key] = self.Successors(self, key)
		if val:
			sucs.update(val)

	def __delitem__(self, key):
		"""Wipe out edges emanating from orig"""
		self[key].clear()
		del self._cache[key]

	def __iter__(self):
		for node in self.graph.node:
			if node in self:
				yield node

	def __len__(self):
		n = 0
		for node in self.graph.node:
			if node in self:
				n += 1
		return n

	def __contains__(self, key):
		return key in self.graph.node

	def __repr__(self):
		cls = self.__class__
		return "<{}.{} object containing {}>".format(
			cls.__module__,
			cls.__name__,
			{
				k: {k2: dict(v2) for (k2, v2) in v.items()}
				for (k, v) in self.items()
			},
		)


class DiGraphSuccessorsMapping(GraphSuccessorsMapping):
	__slots__ = ("graph",)

	class Successors(AbstractSuccessors):
		__slots__ = ("graph", "container", "orig", "_cache")

		def _order_nodes(self, dest):
			return (self.orig, dest)


class DiGraphPredecessorsMapping(GraphEdgeMapping):
	"""Mapping for Predecessors instances, which map to Edges that end at
	the dest provided to this

	"""

	__slots__ = ("graph",)

	def __contains__(self, dest):
		for orig in self.db._edges_cache.iter_predecessors(
			self.graph.name, dest, *self.db._btt()
		):
			try:
				if self.db._edges_cache.retrieve(
					self.graph.name, orig, dest, *self.db._btt()
				):
					return True
			except KeyError:
				continue
		return False

	def __getitem__(self, dest):
		"""Return a Predecessors instance for edges ending at the given
		node

		"""
		if dest not in self.graph.node:
			raise KeyError("No such node", dest)
		if dest not in self._cache:
			self._cache[dest] = self.Predecessors(self, dest)
		return self._cache[dest]

	def __setitem__(self, key, val):
		"""Interpret ``val`` as a mapping of edges that end at ``dest``"""
		created = key not in self
		if key not in self._cache:
			self._cache[key] = self.Predecessors(self, key)
		preds = self._cache[key]
		preds.clear()
		preds.update(val)

	def __delitem__(self, key):
		"""Delete all edges ending at ``dest``"""
		it = self[key]
		it.clear()
		del self._cache[key]

	def __iter__(self):
		return iter(self.graph.node)

	def __len__(self):
		return len(self.graph.node)

	class Predecessors(GraphEdgeMapping):
		"""Mapping of Edges that end at a particular node"""

		__slots__ = ("graph", "container", "dest")

		def __init__(self, container, dest):
			"""Store container and node ID"""
			super().__init__(container.graph)
			self.container = container
			self.dest = dest

		def __iter__(self):
			"""Iterate over the edges that exist at the present (branch, rev)"""
			for orig in self.db._edges_cache.iter_predecessors(
				self.graph.name, self.dest, *self.db._btt()
			):
				if orig in self:
					yield orig

		def __contains__(self, orig):
			"""Is there an edge from ``orig`` at the moment?"""
			return self.db._edges_cache.has_predecessor(
				self.graph.name, self.dest, orig, *self.db._btt()
			)

		def __len__(self):
			"""How many edges exist at this rev of this branch?"""
			n = 0
			for n, _ in enumerate(self, start=1):
				pass
			return n

		def _make_edge(self, orig):
			return Edge(self.graph, orig, self.dest)

		def __getitem__(self, orig):
			"""Get the edge from the given node to mine"""
			if orig not in self:
				raise KeyError(orig)
			return self.graph.adj[orig][self.dest]

		def __setitem__(self, orig, value):
			"""Use ``value`` as a mapping of edge attributes, set an edge from the
			given node to mine.

			"""
			branch, turn, tick = self.db._nbtt()
			try:
				e = self[orig]
				e.clear()
			except KeyError:
				self.db.query.exist_edge(
					self.graph.name,
					orig,
					self.dest,
					branch,
					turn,
					tick,
					True,
				)
				e = self._make_edge(orig)
			e.update(value)
			self.db._edges_cache.store(
				self.graph.name, orig, self.dest, branch, turn, tick, True
			)

		def __delitem__(self, orig):
			"""Unset the existence of the edge from the given node to mine"""
			branch, turn, tick = self.db._nbtt()
			self.db.query.exist_edge(
				self.graph.name, orig, self.dest, branch, turn, tick, False
			)
			self.db._edges_cache.store(
				self.graph.name, orig, self.dest, branch, turn, tick, None
			)


def unwrapped_dict(d):
	ret = {}
	for k, v in d.items():
		if hasattr(v, "unwrap") and not getattr(v, "no_unwrap", False):
			ret[k] = v.unwrap()
		else:
			ret[k] = v
	return ret


class DiGraph(networkx.DiGraph, ABC):
	"""A version of the networkx.DiGraph class that stores its state in a
	database.

	"""

	adj_cls = DiGraphSuccessorsMapping
	pred_cls = DiGraphPredecessorsMapping
	graph_map_cls = GraphMapping
	node_map_cls = GraphNodeMapping
	_statmap: graph_map_cls
	_nodemap: node_map_cls
	_adjmap: adj_cls
	_predmap: pred_cls

	def __repr__(self):
		return "<{} object named {} containing {} nodes, {} edges>".format(
			self.__class__, self.name, len(self.nodes), len(self.edges)
		)

	def _nodes_state(self):
		return {
			noden: {
				k: v for (k, v) in unwrapped_dict(node).items() if k != "name"
			}
			for noden, node in self._node.items()
		}

	def _edges_state(self):
		ret = {}
		ismul = self.is_multigraph()
		for orig, dests in self.adj.items():
			if orig not in ret:
				ret[orig] = {}
			origd = ret[orig]
			for dest, edge in dests.items():
				if ismul:
					if dest not in origd:
						origd[dest] = edges = {}
					else:
						edges = origd[dest]
					for i, val in edge.items():
						edges[i] = unwrapped_dict(val)
				else:
					origd[dest] = unwrapped_dict(edge)
		return ret

	def _val_state(self):
		return {
			k: v
			for (k, v) in unwrapped_dict(self.graph).items()
			if k != "name"
		}

	def __init__(self, db, name):  # user shouldn't instantiate directly
		self._name = name
		self.db = db

	def __bool__(self):
		return self._name in self.db._graph_objs

	@property
	def graph(self):
		if not hasattr(self, "_statmap"):
			self._statmap = self.graph_map_cls(self)
		return self._statmap

	@graph.setter
	def graph(self, v):
		self.graph.clear()
		self.graph.update(v)

	@property
	def node(self):
		if not hasattr(self, "_nodemap"):
			self._nodemap = self.node_map_cls(self)
		return self._nodemap

	_node = node

	@property
	def adj(self):
		if not hasattr(self, "_adjmap"):
			self._adjmap = self.adj_cls(self)
		return self._adjmap

	edge = succ = _succ = _adj = adj

	@property
	def pred(self):
		if not hasattr(self, "pred_cls"):
			raise TypeError("Undirected graph")
		if not hasattr(self, "_predmap"):
			self._predmap = self.pred_cls(self)
		return self._predmap

	_pred = pred

	@property
	def name(self):
		return self._name

	@name.setter
	def name(self, v):
		raise TypeError("graphs can't be renamed")

	def remove_node(self, n):
		"""Version of remove_node that minimizes writes"""
		if n not in self._node:
			raise NetworkXError("The node %s is not in the digraph." % (n,))
		nbrs = list(self._succ[n])
		for u in nbrs:
			del self._pred[u][n]  # remove all edges n-u in digraph
		pred = list(self._pred[n])
		for u in pred:
			del self._succ[u][n]  # remove all edges n-u in digraph
		del self._node[n]

	def remove_edge(self, u, v):
		"""Version of remove_edge that's much like normal networkx but only
		deletes once, since the database doesn't keep separate adj and
		succ mappings

		"""
		try:
			del self.succ[u][v]
		except KeyError:
			raise NetworkXError(
				"The edge {}-{} is not in the graph.".format(u, v)
			)

	def remove_edges_from(self, ebunch):
		"""Version of remove_edges_from that's much like normal networkx but only
		deletes once, since the database doesn't keep separate adj and
		succ mappings

		"""
		for e in ebunch:
			(u, v) = e[:2]
			if u in self.succ and v in self.succ[u]:
				del self.succ[u][v]

	def add_edge(self, u, v, attr_dict=None, **attr):
		"""Version of add_edge that only writes to the database once"""
		if attr_dict is None:
			attr_dict = attr
		else:
			try:
				attr_dict.update(attr)
			except AttributeError:
				raise NetworkXError(
					"The attr_dict argument must be a dictionary."
				)
		if u not in self.node:
			self.node[u] = {}
		if v not in self.node:
			self.node[v] = {}
		if u in self.adj:
			datadict = self.adj[u].get(v, {})
		else:
			self.adj[u] = {v: {}}
			datadict = self.adj[u][v]
		datadict.update(attr_dict)
		self.succ[u][v] = datadict

	def add_edges_from(self, ebunch, attr_dict=None, **attr):
		"""Version of add_edges_from that only writes to the database once"""
		if attr_dict is None:
			attr_dict = attr
		else:
			try:
				attr_dict.update(attr)
			except AttributeError:
				raise NetworkXError("The attr_dict argument must be a dict.")
		for e in ebunch:
			ne = len(e)
			if ne == 3:
				u, v, dd = e
				assert hasattr(dd, "update")
			elif ne == 2:
				u, v = e
				dd = {}
			else:
				raise NetworkXError(
					"Edge tupse {} must be a 2-tuple or 3-tuple.".format(e)
				)
			if u not in self.node:
				self.node[u] = {}
			if v not in self.node:
				self.node[v] = {}
			datadict = self.adj.get(u, {}).get(v, {})
			datadict.update(attr_dict)
			datadict.update(dd)
			self.succ[u][v] = datadict
			assert u in self.succ
			assert v in self.succ[u]

	def clear(self):
		"""Remove all nodes and edges from the graph.

		Unlike the regular networkx implementation, this does *not*
		remove the graph's name. But all the other graph, node, and
		edge attributes go away.

		"""
		self.adj.clear()
		self.node.clear()
		self.graph.clear()

	def add_node(self, node_for_adding, **attr):
		"""Version of add_node that minimizes writes"""
		if node_for_adding not in self._succ:
			self._succ[node_for_adding] = self.adjlist_inner_dict_factory()
			self._pred[node_for_adding] = self.adjlist_inner_dict_factory()
			self._node[node_for_adding] = self.node_dict_factory()
		self._node[node_for_adding].update(attr)
