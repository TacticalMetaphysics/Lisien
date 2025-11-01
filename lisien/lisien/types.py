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

import builtins
import os
from _operator import (
	attrgetter,
	add,
	sub,
	mul,
	pow,
	truediv,
	floordiv,
	mod,
	ge,
	gt,
	le,
	lt,
	eq,
)
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from collections.abc import Iterable, Mapping, Sequence, Set
from concurrent.futures import Future
from enum import Enum
from functools import wraps, partial, cached_property
from itertools import chain
from random import Random
from types import GenericAlias, ModuleType, FunctionType, MethodType
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
	TypeVar,
	Iterator,
	Mapping,
	Type,
	Optional,
	AbstractSet,
	KeysView,
	Iterable,
	Hashable,
	Sequence,
	get_origin,
	get_args,
	Union,
)

import networkx
import networkx as nx
from annotated_types import Ge, Le
from blinker import Signal
from networkx import NetworkXError
from tblib import Traceback

from . import exc
from .exc import WorkerProcessReadOnlyError, TimeError
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
	from .rule import RuleBook, Rule

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
			isinstance(obj, (list, ListWrapper))
			and all(map(is_valid_value, obj))
		)
		or (
			isinstance(obj, (dict, DictWrapper))
			and all(map(is_valid_key, obj.keys()))
			and all(map(is_valid_value, obj.values()))
		)
		or (
			isinstance(obj, (Set, Sequence, SetWrapper))
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


def stat(k: _Key) -> Stat:
	return Stat(Key(k))


EternalKey = NewType("EternalKey", Key)


def ekey(k: _Key) -> EternalKey:
	return EternalKey(Key(k))


UniversalKey = NewType("UniversalKey", Key)


def ukey(k: _Key) -> UniversalKey:
	return UniversalKey(Key(k))


Branch = NewType("Branch", str)
Turn = NewType("Turn", Annotated[int, Ge(0)])
Tick = NewType("Tick", Annotated[int, Ge(0)])
Time: TypeAlias = tuple[Branch, Turn, Tick]
LinearTime: TypeAlias = tuple[Turn, Tick]
TimeWindow: TypeAlias = tuple[Branch, Turn, Tick, Turn, Tick]
Plan = NewType("Plan", Annotated[int, Ge(0)])
CharName = NewType("CharName", Key)


def charn(k) -> CharName:
	if not isinstance(k, Key):
		raise TypeError("Invalid character name", k)
	return CharName(k)


NodeName = NewType("NodeName", Key)


def nodename(k) -> NodeName:
	if not isinstance(k, Key):
		raise TypeError("Invalid node name", k)
	return NodeName(k)


EntityKey: TypeAlias = (
	tuple[CharName]
	| tuple[CharName, NodeName]
	| tuple[CharName, NodeName, NodeName]
)
RulebookName = NewType("RulebookName", Key)


def rbname(k) -> RulebookName:
	if not isinstance(k, Key):
		raise TypeError("Invalid rulebook name", k)
	return RulebookName(k)


RulebookPriority = NewType("RulebookPriority", float)
RuleName = NewType("RuleName", str)


def rulename(s) -> RuleName:
	if not isinstance(s, str):
		raise TypeError("Invalid rule name", s)
	return RuleName(s)


RuleNeighborhood: TypeAlias = Annotated[int, Ge(0)] | None
RuleBig = NewType("RuleBig", bool)
RuleFunc: TypeAlias = Callable[[Any], bool]
FuncName = NewType("FuncName", str)
FuncStoreName: TypeAlias = Literal[
	"trigger", "prereq", "action", "function", "method"
]
TriggerFuncName = NewType("TriggerFuncName", FuncName)


def trigfuncn(s: str) -> TriggerFuncName:
	return TriggerFuncName(FuncName(s))


PrereqFuncName = NewType("PrereqFuncName", FuncName)


def preqfuncn(s: str) -> PrereqFuncName:
	return PrereqFuncName(FuncName(s))


ActionFuncName = NewType("ActionFuncName", FuncName)


def actfuncn(s: str) -> ActionFuncName:
	return ActionFuncName(FuncName(s))


RuleFuncName: TypeAlias = TriggerFuncName | PrereqFuncName | ActionFuncName
UniversalKeyframe: TypeAlias = dict[UniversalKey, Value]
RuleKeyframe: TypeAlias = dict[
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
UniversalRowType: TypeAlias = tuple[Branch, Turn, Tick, UniversalKey, Value]
RulebookRowType: TypeAlias = tuple[
	Branch,
	Turn,
	Tick,
	RulebookName,
	list[RuleName],
	RulebookPriority,
]
RuleRowType: TypeAlias = tuple[
	Branch,
	Turn,
	Tick,
	RuleName,
	list[TriggerFuncName]
	| list[PrereqFuncName]
	| list[ActionFuncName]
	| RuleNeighborhood
	| RuleBig,
]
TriggerRowType: TypeAlias = tuple[
	Branch, Turn, Tick, RuleName, list[TriggerFuncName]
]
PrereqRowType: TypeAlias = tuple[
	Branch, Turn, Tick, RuleName, list[PrereqFuncName]
]
ActionRowType: TypeAlias = tuple[
	Branch, Turn, Tick, RuleName, list[ActionFuncName]
]
RuleNeighborhoodRowType: TypeAlias = tuple[
	Branch, Turn, Tick, RuleName, RuleNeighborhood
]
RuleBigRowType: TypeAlias = tuple[Branch, Turn, Tick, RuleName, RuleBig]
GraphTypeStr: TypeAlias = Literal["DiGraph", "Deleted"]
GraphRowType: TypeAlias = tuple[Branch, Turn, Tick, CharName, GraphTypeStr]
NodeRowType: TypeAlias = tuple[Branch, Turn, Tick, CharName, NodeName, bool]
EdgeRowType: TypeAlias = tuple[
	Branch, Turn, Tick, CharName, NodeName, NodeName, bool
]
GraphValRowType: TypeAlias = tuple[Branch, Turn, Tick, CharName, Stat, Value]
NodeValRowType: TypeAlias = tuple[
	Branch, Turn, Tick, CharName, NodeName, Stat, Value
]
EdgeValRowType: TypeAlias = tuple[
	Branch, Turn, Tick, CharName, NodeName, NodeName, Stat, Value
]
ThingRowType: TypeAlias = tuple[
	Branch, Turn, Tick, CharName, NodeName, NodeName
]
UnitRowType: TypeAlias = tuple[
	Branch, Turn, Tick, CharName, CharName, NodeName, bool
]
CharRulebookRowType: TypeAlias = tuple[
	Branch, Turn, Tick, CharName, RulebookName
]
NodeRulebookRowType: TypeAlias = tuple[
	Branch, Turn, Tick, CharName, NodeName, RulebookName
]
PortalRulebookRowType: TypeAlias = tuple[
	Branch, Turn, Tick, CharName, NodeName, NodeName, RulebookName
]
CharacterRulesHandledRowType: TypeAlias = tuple[
	Branch,
	Turn,
	CharName,
	RulebookName,
	RuleName,
	Tick,
]
PortalRulesHandledRowType: TypeAlias = tuple[
	Branch,
	Turn,
	CharName,
	NodeName,
	NodeName,
	RulebookName,
	RuleName,
	Tick,
]
NodeRulesHandledRowType: TypeAlias = tuple[
	Branch,
	Turn,
	CharName,
	NodeName,
	RulebookName,
	RuleName,
	Tick,
]
UnitRulesHandledRowType: TypeAlias = tuple[
	Branch,
	Turn,
	CharName,
	CharName,
	NodeName,
	RulebookName,
	RuleName,
	Tick,
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
		"rules",
		"units",
	],
	NodesDict
	| NodeValDict
	| EdgesDict
	| EdgeValDict
	| RulebookName
	| UnitsDict
	| dict[
		RuleName,
		dict[
			Literal["triggers", "prereqs", "actions"],
			list[TriggerFuncName]
			| list[PrereqFuncName]
			| list[ActionFuncName],
		],
	]
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
		try:
			if self._get_cache_now(key) == value:
				return
		except KeyError:
			pass
		branch, turn, tick = self.db._nbtt()
		self._set_cache(key, branch, turn, tick, value)
		self._set_db(key, branch, turn, tick, value)

	def __delitem__(self, key):
		self._get_cache_now(key)  # deliberately raise KeyError if unset
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


PackSignature: TypeAlias = Callable[
	[Key | EternalKey | UniversalKey | Stat | Value], bytes
]
UnpackSignature: TypeAlias = Callable[[bytes], Value]
LoadedCharWindow: TypeAlias = dict[
	Literal[
		"nodes",
		"edges",
		"graph_val",
		"node_val",
		"edge_val",
		"things",
		"units",
		"character_rulebook",
		"unit_rulebook",
		"character_thing_rulebook",
		"character_place_rulebook",
		"character_portal_rulebook",
		"node_rulebook",
		"portal_rulebook",
	],
	list[NodeRowType]
	| list[EdgeRowType]
	| list[GraphValRowType]
	| list[NodeValRowType]
	| list[EdgeValRowType]
	| list[ThingRowType]
	| list[UnitRowType]
	| list[CharRulebookRowType]
	| list[NodeRulebookRowType]
	| list[PortalRulebookRowType],
]
LoadedDict: TypeAlias = dict[
	Literal[
		"universals",
		"rulebooks",
		"rule_triggers",
		"rule_prereqs",
		"rule_actions",
		"rule_neighborhood",
		"rule_big",
		"character_rules_handled",
		"unit_rules_handled",
		"character_thing_rules_handled",
		"character_place_rules_handled",
		"character_portal_rules_handled",
		"node_rules_handled",
		"portal_rules_handled",
		"graphs",
	]
	| CharName,
	list[UniversalRowType]
	| list[RulebookRowType]
	| list[RuleRowType]
	| list[CharacterRulesHandledRowType]
	| list[UnitRulesHandledRowType]
	| list[NodeRulesHandledRowType]
	| list[PortalRulesHandledRowType]
	| list[GraphRowType]
	| LoadedCharWindow,
]


class MsgpackExtensionType(Enum):
	"""Type codes for packing special lisien types into msgpack"""

	tuple = 0x00
	frozenset = 0x01
	set = 0x02
	exception = 0x03
	graph = 0x04
	character = 0x7F
	place = 0x7E
	thing = 0x7D
	portal = 0x7C
	ellipsis = 0x7B
	function = 0x7A
	method = 0x79
	trigger = 0x78
	prereq = 0x77
	action = 0x76


class get_rando:
	"""Attribute getter for randomization functions

	Aliases functions of a randomizer, wrapped so that they won't run in
	planning mode, and will save the randomizer's state after every call.

	"""

	__slots__ = ("_getter", "_wrapfun", "_instance")
	_getter: Callable[[], Callable]

	def __init__(self, attr, *attrs):
		self._getter = attrgetter(attr, *attrs)

	def __get__(self, instance, owner) -> Callable:
		if hasattr(self, "_wrapfun") and self._instance is instance:
			return self._wrapfun
		retfun: Callable = self._getter(instance)

		@wraps(retfun)
		def remembering_rando_state(*args, **kwargs):
			if instance._planning:
				raise exc.PlanError("Don't use randomization in a plan")
			ret = retfun(*args, **kwargs)
			instance.universal["rando_state"] = instance._rando.getstate()
			return ret

		self._wrapfun = remembering_rando_state
		self._instance = instance
		return remembering_rando_state


class SignalDict(Signal, dict):
	def __setitem__(self, __key, __value):
		super().__setitem__(__key, __value)
		self.send(self, key=__key, value=__value)

	def __delitem__(self, __key):
		super().__delitem__(__key)
		self.send(self, key=__key, value=None)


_T = TypeVar("_T")


class EntityAccessor(ABC):
	__slots__ = (
		"engine",
		"entity",
		"branch",
		"turn",
		"tick",
		"stat",
		"current",
		"mungers",
	)

	def __init__(
		self,
		entity: GraphMapping | Node | Edge,
		stat: Stat,
		engine: AbstractEngine | None = None,
		branch: Branch | None = None,
		turn: Turn | None = None,
		tick: Tick | None = None,
		current: bool = False,
		mungers: list[Callable] | None = None,
	):
		if engine is None:
			engine = entity.db
		if branch is None:
			branch = engine.branch
		if turn is None:
			turn = engine.turn
		if mungers is None:
			mungers = []
		self.current = current
		self.engine = engine
		self.entity = entity
		self.stat = stat
		self.branch = branch
		self.turn = turn
		self.tick = tick
		self.mungers = mungers

	def __ne__(self, other):
		return self() != other

	def __str__(self):
		return str(self())

	def __repr__(self):
		return "EntityStatAccessor({}[{}]{}), {} mungers".format(
			self.entity,
			self.stat,
			""
			if self.current
			else ", branch={}, turn={}, tick={}".format(
				self.branch, self.turn, self.tick
			),
			len(self.mungers),
		)

	def __gt__(self, other):
		return self() > other

	def __ge__(self, other):
		return self() >= other

	def __lt__(self, other):
		return self() < other

	def __le__(self, other):
		return self() <= other

	def __eq__(self, other):
		return self() == other

	def munge(self, munger: callable):
		return EntityStatAccessor(
			self.entity,
			self.stat,
			self.engine,
			self.branch,
			self.turn,
			self.tick,
			self.current,
			self.mungers + [munger],
		)

	def __add__(self, other):
		return self.munge(partial(add, other))

	def __sub__(self, other):
		return self.munge(partial(sub, other))

	def __mul__(self, other):
		return self.munge(partial(mul, other))

	def __rpow__(self, other, modulo=None):
		return self.munge(partial(pow, other, modulo=modulo))

	def __rdiv__(self, other):
		return self.munge(partial(truediv, other))

	def __rfloordiv__(self, other):
		return self.munge(partial(floordiv, other))

	def __rmod__(self, other):
		return self.munge(partial(mod, other))

	def __contains__(self, item):
		return item in self()

	def __getitem__(self, k):
		return self.munge(lambda x: x[k])

	@abstractmethod
	def _get_value_now(self) -> Value: ...

	def __call__(
		self,
		branch: Branch | None = None,
		turn: Turn | None = None,
		tick: Tick | None = None,
	):
		if self.current:
			res = self._get_value_now()
		else:
			time_was = self.engine.time
			self.engine.branch = branch or self.branch
			self.engine.turn = turn if turn is not None else self.turn
			if tick is not None:
				self.engine.tick = tick
			elif self.tick is not None:
				self.engine.tick = self.tick
			res = self._get_value_now()
			self.engine.time = time_was
		for munger in self.mungers:
			res = munger(res)
		return res

	def iter_history(self, beginning: Turn, end: Turn) -> Iterator[Value]:
		"""Iterate over all the values this stat has had in the given window, inclusive."""
		# It might be useful to do this in a way that doesn't change the
		# engine's time, perhaps for thread safety
		engine = self.engine
		oldturn = engine.turn
		oldtick = engine.tick
		for turn in range(beginning, end + 1):
			engine.turn = turn
			try:
				y = self._get_value_now()
			except KeyError:
				yield None
				continue
			if hasattr(y, "unwrap"):
				y = y.unwrap()
			yield y
		engine.turn = oldturn
		engine.tick = oldtick


class UnitsAccessor(EntityAccessor):
	entity: AbstractCharacter

	def _get_value_now(self) -> dict[CharName, list[NodeName]]:
		ret = {}
		for graph in self.entity.unit:
			ret[graph] = []
			for node in self.entity.unit[graph]:
				ret[graph].append(node)
		return ret


class CharacterStatAccessor(EntityAccessor):
	entity: AbstractCharacter

	def _get_value_now(self) -> Value:
		return self.entity.stat[self.stat]


class EntityStatAccessor(EntityAccessor):
	def _get_value_now(self) -> Value:
		return self.entity[self.stat]


class SizedDict(OrderedDict):
	"""A dictionary that discards old entries when it gets too big."""

	def __init__(self, max_entries: Annotated[int, Ge(0)] = 1000):
		self._n = max_entries
		super().__init__()

	def __setitem__(self, key, value):
		while len(self) > self._n:
			self.popitem(last=False)
		super().__setitem__(key, value)


class FakeFuture(Future):
	"""A 'Future' that calls its function immediately and sets the result"""

	def __init__(self, func: Callable, *args, **kwargs):
		super().__init__()
		self.set_result(func(*args, **kwargs))


class AbstractBookmarkMapping(MutableMapping, Callable):
	@abstractmethod
	def __call__(self, key: _Key) -> None: ...


class AbstractEngine(ABC):
	"""Parent class to the real Engine as well as EngineProxy.

	Implements serialization and the __getattr__ for stored methods.

	By default, the deserializers will refuse to create lisien entities.
	If you want them to, use my ``loading`` property to open a ``with``
	block, in which deserialized entities will be created as needed.

	"""

	thing_cls: type
	place_cls: type
	portal_cls: type
	char_cls: type
	character: Mapping[_Key, Type[char_cls]]
	eternal: MutableMapping[_Key, _Value]
	universal: MutableMapping[_Key, _Value]
	rulebook: MutableMapping[_Key, "RuleBook"]
	rule: MutableMapping[_Key, "Rule"]
	trunk: Branch
	branch: Branch
	turn: Turn
	tick: Tick
	time: Time
	function: ModuleType | AbstractFunctionStore
	method: ModuleType | AbstractFunctionStore
	trigger: ModuleType | AbstractFunctionStore
	prereq: ModuleType | AbstractFunctionStore
	action: ModuleType | AbstractFunctionStore
	bookmark: AbstractBookmarkMapping
	_rando: Random
	_branches_d: dict[
		Optional[Branch], tuple[Optional[Branch], Turn, Tick, Turn, Tick]
	]

	@cached_property
	def logger(self):
		if hasattr(self, "_logger"):
			return self._logger
		from logging import getLogger

		return getLogger("lisien")

	def log(self, level, msg, *args, **kwargs):
		self.logger.log(level, msg, *args, **kwargs)

	def debug(self, msg, *args, **kwargs):
		self.log(10, msg, *args, **kwargs)

	def info(self, msg, *args, **kwargs):
		self.log(20, msg, *args, **kwargs)

	def warning(self, msg, *args, **kwargs):
		self.log(30, msg, *args, **kwargs)

	def error(self, msg, *args, **kwargs):
		self.log(40, msg, *args, **kwargs)

	def critical(self, msg, *args, **kwargs):
		self.log(50, msg, *args, **kwargs)

	def is_ancestor_of(self, parent: Branch, child: Branch) -> bool:
		"""Return whether ``child`` is a branch descended from ``parent``

		At any remove.

		"""
		branches = self.branches()
		if parent not in branches:
			raise ValueError("Not a branch", parent)
		if child not in branches:
			raise ValueError("Not a branch", child)
		if parent is None or parent == child or parent == self.trunk:
			return True
		if child == self.trunk:
			return False
		if self.branch_parent(child) == parent:
			return True
		return self.is_ancestor_of(parent, self.branch_parent(child))

	@cached_property
	def pack(self) -> Callable[[Value], bytes]:
		try:
			import msgpack._cmsgpack
			import msgpack
		except ImportError:
			import umsgpack

			return partial(
				umsgpack.packb, ext_handlers=self._umsgpack_pack_handlers
			)

		def pack_set(s):
			return msgpack.ExtType(
				MsgpackExtensionType.set.value, packer(list(s))
			)

		handlers = {
			type(...): lambda _: msgpack.ExtType(
				MsgpackExtensionType.ellipsis.value, b""
			),
			nx.Graph: lambda graf: msgpack.ExtType(
				MsgpackExtensionType.graph.value,
				packer(
					[
						"Graph",
						graf._node,
						graf._adj,
						graf.graph,
					]
				),
			),
			nx.DiGraph: lambda graf: msgpack.ExtType(
				MsgpackExtensionType.graph.value,
				packer(["DiGraph", graf._node, graf._adj, graf.graph]),
			),
			nx.MultiGraph: lambda graf: msgpack.ExtType(
				MsgpackExtensionType.graph.value,
				packer(["MultiGraph", graf._node, graf._adj, graf.graph]),
			),
			nx.MultiDiGraph: lambda graf: msgpack.ExtType(
				MsgpackExtensionType.graph.value,
				packer(["MultiDiGraph", graf._node, graf._adj, graf.graph]),
			),
			tuple: lambda tup: msgpack.ExtType(
				MsgpackExtensionType.tuple.value, packer(list(tup))
			),
			frozenset: lambda frozs: msgpack.ExtType(
				MsgpackExtensionType.frozenset.value, packer(list(frozs))
			),
			set: pack_set,
			FunctionType: lambda func: msgpack.ExtType(
				getattr(MsgpackExtensionType, func.__module__).value,
				packer(func.__name__),
			),
			MethodType: lambda meth: msgpack.ExtType(
				MsgpackExtensionType.method.value, packer(meth.__name__)
			),
			Exception: lambda exc: msgpack.ExtType(
				MsgpackExtensionType.exception.value,
				packer(
					[
						exc.__class__.__name__,
						Traceback(exc.__traceback__).to_dict()
						if hasattr(exc, "__traceback__")
						else None,
					]
					+ list(exc.args)
				),
			),
		}

		def pack_handler(obj):
			if isinstance(obj, Exception):
				typ = Exception
			else:
				typ = type(obj)
			if typ in handlers:
				return handlers[typ](obj)
			elif isinstance(obj, DiGraph):
				return msgpack.ExtType(
					MsgpackExtensionType.character.value, packer(obj.name)
				)
			elif isinstance(obj, AbstractThing):
				return msgpack.ExtType(
					MsgpackExtensionType.thing.value,
					packer([obj.character.name, obj.name]),
				)
			elif isinstance(obj, Node):
				return msgpack.ExtType(
					MsgpackExtensionType.place.value,
					packer([obj.graph.name, obj.name]),
				)
			elif isinstance(obj, Edge):
				return msgpack.ExtType(
					MsgpackExtensionType.portal.value,
					packer(
						[
							obj.graph.name,
							obj.orig,
							obj.dest,
						]
					),
				)
			elif isinstance(obj, Set):
				return pack_set(obj)
			elif isinstance(obj, Mapping):
				return dict(obj)
			elif isinstance(obj, list):
				return list(obj)
			raise TypeError("Can't pack {}".format(typ))

		packer = partial(
			msgpack.packb,
			default=pack_handler,
			strict_types=True,
			use_bin_type=True,
		)
		return packer

	@cached_property
	def _unpack_handlers(self):
		char_cls = self.char_cls
		place_cls = self.place_cls
		portal_cls = self.portal_cls
		thing_cls = self.thing_cls
		excs = {
			# builtin exceptions
			"AssertionError": AssertionError,
			"AttributeError": AttributeError,
			"EOFError": EOFError,
			"FloatingPointError": FloatingPointError,
			"GeneratorExit": GeneratorExit,
			"ImportError": ImportError,
			"IndexError": IndexError,
			"KeyError": KeyError,
			"KeyboardInterrupt": KeyboardInterrupt,
			"MemoryError": MemoryError,
			"NameError": NameError,
			"NotImplementedError": NotImplementedError,
			"OSError": OSError,
			"OverflowError": OverflowError,
			"RecursionError": RecursionError,
			"ReferenceError": ReferenceError,
			"RuntimeError": RuntimeError,
			"StopIteration": StopIteration,
			"IndentationError": IndentationError,
			"TabError": TabError,
			"SystemError": SystemError,
			"SystemExit": SystemExit,
			"TypeError": TypeError,
			"UnboundLocalError": UnboundLocalError,
			"UnicodeError": UnicodeError,
			"UnicodeEncodeError": UnicodeEncodeError,
			"UnicodeDecodeError": UnicodeDecodeError,
			"UnicodeTranslateError": UnicodeTranslateError,
			"ValueError": ValueError,
			"ZeroDivisionError": ZeroDivisionError,
			# networkx exceptions
			"HasACycle": nx.exception.HasACycle,
			"NodeNotFound": nx.exception.NodeNotFound,
			"PowerIterationFailedConvergence": nx.exception.PowerIterationFailedConvergence,
			"ExceededMaxIterations": nx.exception.ExceededMaxIterations,
			"AmbiguousSolution": nx.exception.AmbiguousSolution,
			"NetworkXAlgorithmError": nx.exception.NetworkXAlgorithmError,
			"NetworkXException": nx.exception.NetworkXException,
			"NetworkXError": nx.exception.NetworkXError,
			"NetworkXNoCycle": nx.exception.NetworkXNoCycle,
			"NetworkXNoPath": nx.exception.NetworkXNoPath,
			"NetworkXNotImplemented": nx.exception.NetworkXNotImplemented,
			"NetworkXPointlessConcept": nx.exception.NetworkXPointlessConcept,
			"NetworkXUnbounded": nx.exception.NetworkXUnbounded,
			"NetworkXUnfeasible": nx.exception.NetworkXUnfeasible,
			# lisien exceptions
			"NonUniqueError": exc.NonUniqueError,
			"AmbiguousUserError": exc.AmbiguousLeaderError,
			"AmbiguousLeaderError": exc.AmbiguousLeaderError,
			"RulesEngineError": exc.RulesEngineError,
			"RuleError": exc.RuleError,
			"RedundantRuleError": exc.RedundantRuleError,
			"UserFunctionError": exc.UserFunctionError,
			"WorldIntegrityError": exc.WorldIntegrityError,
			"CacheError": exc.CacheError,
			"TravelException": exc.TravelException,
			"OutOfTimelineError": exc.OutOfTimelineError,
			"HistoricKeyError": exc.HistoricKeyError,
			"NotInKeyframeError": exc.NotInKeyframeError,
			"WorkerProcessReadOnlyError": exc.WorkerProcessReadOnlyError,
		}

		def unpack_graph(ext: bytes) -> nx.Graph:
			if hasattr(ext, "data"):  # umsgpack.Ext
				ext = ext.data
			cls, node, adj, graph = self.unpack(ext)
			blank = {
				"Graph": nx.Graph,
				"DiGraph": nx.DiGraph,
				"MultiGraph": nx.MultiGraph,
				"MultiDiGraph": nx.MultiDiGraph,
			}[cls]()
			blank._node = node
			blank._adj = adj
			blank.graph = graph
			return blank

		def unpack_exception(ext: bytes) -> Exception:
			data: tuple[str, dict | None] = self.unpack(
				getattr(ext, "data", ext)
			)
			if data[0] not in excs:
				return Exception(*data)
			ret = excs[data[0]](*data[2:])
			if data[1] is not None:
				ret.__traceback__ = Traceback.from_dict(data[1]).to_traceback()
			return ret

		def unpack_char(ext: bytes) -> char_cls:
			charn = self.unpack(getattr(ext, "data", ext))
			return char_cls(self, charn, init_rulebooks=False)

		def unpack_place(ext: bytes) -> place_cls:
			charn, placen = self.unpack(getattr(ext, "data", ext))
			return place_cls(
				char_cls(self, charn, init_rulebooks=False), placen
			)

		def unpack_thing(ext: bytes) -> thing_cls:
			charn, thingn = self.unpack(getattr(ext, "data", ext))
			# Breaks if the thing hasn't been instantiated yet, not great
			return self.character[charn].thing[thingn]

		def unpack_portal(ext: bytes) -> portal_cls:
			charn, orign, destn = self.unpack(getattr(ext, "data", ext))
			return portal_cls(
				char_cls(self, charn, init_rulebooks=False), orign, destn
			)

		def unpack_seq(t: type[_T], ext: bytes) -> _T:
			unpacked = self.unpack(getattr(ext, "data", ext))
			if not isinstance(unpacked, list):
				raise TypeError("Tried to unpack", type(unpacked), t)
			return t(unpacked)

		def unpack_func(store: AbstractFunctionStore, ext: bytes) -> Callable:
			unpacked = self.unpack(getattr(ext, "data", ext))
			if not isinstance(unpacked, str):
				raise TypeError("Tried to unpack as func", type(unpacked))
			return getattr(store, unpacked)

		return {
			MsgpackExtensionType.ellipsis.value: lambda _: ...,
			MsgpackExtensionType.graph.value: unpack_graph,
			MsgpackExtensionType.character.value: unpack_char,
			MsgpackExtensionType.place.value: unpack_place,
			MsgpackExtensionType.thing.value: unpack_thing,
			MsgpackExtensionType.portal.value: unpack_portal,
			MsgpackExtensionType.tuple.value: partial(unpack_seq, tuple),
			MsgpackExtensionType.frozenset.value: partial(
				unpack_seq, frozenset
			),
			MsgpackExtensionType.set.value: partial(unpack_seq, set),
			MsgpackExtensionType.function.value: partial(
				unpack_func, self.function
			),
			MsgpackExtensionType.method.value: partial(
				unpack_func, self.method
			),
			MsgpackExtensionType.trigger.value: partial(
				unpack_func, self.trigger
			),
			MsgpackExtensionType.prereq.value: partial(
				unpack_func, self.prereq
			),
			MsgpackExtensionType.action.value: partial(
				unpack_func, self.action
			),
			MsgpackExtensionType.exception.value: unpack_exception,
		}

	@cached_property
	def unpack(
		self,
	) -> Callable[
		[bytes],
		Value
		| nx.DiGraph
		| char_cls
		| place_cls
		| thing_cls
		| portal_cls
		| Exception
		| Callable,
	]:
		try:
			import msgpack._cmsgpack
			import msgpack
		except ImportError:
			import umsgpack

			return partial(
				umsgpack.unpackb, ext_handlers=self._unpack_handlers
			)

		def unpack_handler(
			code: MsgpackExtensionType, data: bytes
		) -> Value | Exception | msgpack.ExtType:
			if code in self._unpack_handlers:
				return self._unpack_handlers[code](data)
			return msgpack.ExtType(code, data)

		def unpacker(b: bytes):
			the_unpacker = msgpack.Unpacker(
				ext_hook=unpack_handler, raw=False, strict_map_key=False
			)
			the_unpacker.feed(b)
			# Deliberately only returning the initial item;
			# others are likely to be null bytes as a result of the
			# way browsers work, and anyway if you really want more
			# you can just pack a list
			return the_unpacker.unpack()

		return unpacker

	@cached_property
	def _umsgpack_pack_handlers(self):
		import umsgpack

		return {
			type(...): lambda _: umsgpack.Ext(
				MsgpackExtensionType.ellipsis.value, b""
			),
			nx.Graph: lambda graf: umsgpack.Ext(
				MsgpackExtensionType.graph.value,
				self.pack(
					[
						"Graph",
						graf._node,
						graf._adj,
						graf.graph,
					]
				),
			),
			nx.DiGraph: lambda graf: umsgpack.Ext(
				MsgpackExtensionType.graph.value,
				self.pack(["DiGraph", graf._node, graf._adj, graf.graph]),
			),
			nx.MultiGraph: lambda graf: umsgpack.Ext(
				MsgpackExtensionType.graph.value,
				self.pack(["MultiGraph", graf._node, graf._adj, graf.graph]),
			),
			nx.MultiDiGraph: lambda graf: umsgpack.Ext(
				MsgpackExtensionType.graph.value,
				self.pack(["MultiDiGraph", graf._node, graf._adj, graf.graph]),
			),
			tuple: lambda tup: umsgpack.Ext(
				MsgpackExtensionType.tuple.value, self.pack(list(tup))
			),
			frozenset: lambda frozs: umsgpack.Ext(
				MsgpackExtensionType.frozenset.value, self.pack(list(frozs))
			),
			set: lambda s: umsgpack.Ext(
				MsgpackExtensionType.set.value, self.pack(list(s))
			),
			FunctionType: lambda func: umsgpack.Ext(
				getattr(MsgpackExtensionType, func.__module__).value,
				self.pack(func.__name__),
			),
			MethodType: lambda meth: umsgpack.Ext(
				MsgpackExtensionType.method.value, self.pack(meth.__name__)
			),
			Exception: lambda exc: umsgpack.Ext(
				MsgpackExtensionType.exception.value,
				self.pack(
					[
						exc.__class__.__name__,
						Traceback(exc.__traceback__).to_dict()
						if hasattr(exc, "__traceback__")
						else None,
					]
					+ list(exc.args)
				),
			),
			self.char_cls: lambda obj: umsgpack.Ext(
				MsgpackExtensionType.character.value, self.pack(obj.name)
			),
			self.thing_cls: lambda obj: umsgpack.Ext(
				MsgpackExtensionType.thing.value,
				self.pack([obj.character.name, obj.name]),
			),
			self.place_cls: lambda obj: umsgpack.Ext(
				MsgpackExtensionType.place.value,
				self.pack([obj.graph.name, obj.name]),
			),
			self.portal_cls: lambda obj: umsgpack.Ext(
				MsgpackExtensionType.portal.value,
				self.pack(
					[
						obj.graph.name,
						obj.orig,
						obj.dest,
					]
				),
			),
		}

	@abstractmethod
	def _get_node(
		self, char: AbstractCharacter | CharName, node: NodeName
	) -> Node: ...

	@abstractmethod
	def _btt(self) -> tuple[Branch, Turn, Tick]: ...

	def branches(self) -> KeysView:
		return self._branches_d.keys()

	def branch_parent(self, branch: Branch | None) -> Branch | None:
		if branch is None or branch not in self._branches_d:
			return None
		return self._branches_d[branch][0]

	def _branch_start(self, branch: Branch | None = None) -> LinearTime:
		if branch is None:
			branch = self.branch
		_, turn, tick, _, _ = self._branches_d[branch]
		return turn, tick

	def _branch_end(self, branch: Branch | None = None) -> LinearTime:
		if branch is None:
			branch = self.branch
		_, _, _, turn, tick = self._branches_d[branch]
		return turn, tick

	@abstractmethod
	def _start_branch(
		self, parent: Branch, branch: Branch, turn: Turn, tick: Tick
	) -> None: ...

	@abstractmethod
	def _set_btt(self, branch: Branch, turn: Turn, tick: Tick) -> None: ...

	@abstractmethod
	def _extend_branch(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> None: ...

	@abstractmethod
	def load_at(self, branch: Branch, turn: Turn, tick: Tick) -> None: ...

	def branch_start_turn(self, branch: Branch | None = None) -> Turn:
		return self._branch_start(branch)[0]

	def branch_start_tick(self, branch: Branch | None = None) -> Tick:
		return self._branch_start(branch)[1]

	def branch_end_turn(self, branch: Branch | None = None) -> Turn:
		return self._branch_end(branch)[0]

	def branch_end_tick(self, branch: Branch | None = None) -> Tick:
		return self._branch_end(branch)[1]

	@abstractmethod
	def turn_end(
		self, branch: Branch | None = None, turn: Turn | None = None
	) -> Tick: ...

	@abstractmethod
	def turn_end_plan(
		self, branch: Branch | None = None, turn: Turn | None = None
	) -> Tick: ...

	@abstractmethod
	def add_character(
		self,
		name: _Key,
		data: nx.Graph | DiGraph | None = None,
		layout: bool = False,
		node: NodeValDict | None = None,
		edge: EdgeValDict | None = None,
		**kwargs,
	): ...

	def new_character(
		self,
		name: _Key,
		data: nx.Graph | DiGraph | None = None,
		layout: bool = False,
		node: NodeValDict | None = None,
		edge: EdgeValDict | None = None,
		**kwargs,
	):
		self.add_character(name, data)
		return self.character[name]

	@abstractmethod
	def export(
		self,
		name: str | None,
		path: str | os.PathLike | None = None,
		indent: bool = True,
	) -> str | os.PathLike: ...

	@classmethod
	@abstractmethod
	def from_archive(
		cls,
		path: str | os.PathLike,
		prefix: str | os.PathLike | None = ".",
		**kwargs,
	) -> AbstractEngine: ...

	def coin_flip(self) -> bool:
		"""Return True or False with equal probability."""
		return self.choice((True, False))

	def die_roll(self, d: Annotated[int, Ge(1)]) -> int:
		"""Roll a die with ``d`` faces. Return the result."""
		return self.randint(1, d)

	def dice(
		self, n: Annotated[int, Ge(1)], d: Annotated[int, Ge(1)]
	) -> Iterable[int]:
		"""Roll ``n`` dice with ``d`` faces, and yield the results.

		This is an iterator. You'll get the result of each die in
		succession.

		"""
		for i in range(0, n):
			yield self.die_roll(d)

	def dice_check(
		self,
		n: Annotated[int, Ge(1)],
		d: Annotated[int, Ge(1)],
		target: Annotated[int, Ge(0)],
		comparator: str | Callable[[int, int], bool] = "<=",
	) -> bool:
		"""Roll ``n`` dice with ``d`` sides, sum them, and compare

		If ``comparator`` is provided, use it instead of the default <=.
		You may use a string like '<' or '>='.

		"""
		from operator import eq, ge, gt, le, lt, ne

		comps: dict[str, Callable] = {
			">": gt,
			"<": lt,
			">=": ge,
			"<=": le,
			"=": eq,
			"==": eq,
			"!=": ne,
		}
		if not callable(comparator):
			comparator = comps[comparator]
		return comparator(sum(self.dice(n, d)), target)

	def chance(self, f: Annotated[float, Ge(0.0), Le(1.0)]) -> bool:
		"""Return True or False with a given unit probability

		Supply a float between 0.0 and 1.0 to express the probability--
		or use `percent_chance`

		"""
		if f <= 0.0:
			return False
		if f >= 1.0:
			return True
		return f > self._rando.random()

	def percent_chance(
		self,
		pct: Annotated[int, Ge(0), Le(100)]
		| Annotated[float, Ge(0.0), Le(100.0)],
	) -> bool:
		"""Return True or False with a given percentile probability

		Values not between 0 and 100 are treated as though they
		were 0 or 100, whichever is nearer.

		"""
		return self.chance(pct / 100)

	betavariate = get_rando("_rando.betavariate")
	choice = get_rando("_rando.choice")
	expovariate = get_rando("_rando.expovariate")
	gammavariate = get_rando("_rando.gammavariate")
	gauss = get_rando("_rando.gauss")
	getrandbits = get_rando("_rando.getrandbits")
	lognormvariate = get_rando("_rando.lognormvariate")
	normalvariate = get_rando("_rando.normalvariate")
	paretovariate = get_rando("_rando.paretovariate")
	randint = get_rando("_rando.randint")
	random = get_rando("_rando.random")
	randrange = get_rando("_rando.randrange")
	sample = get_rando("_rando.sample")
	shuffle = get_rando("_rando.shuffle")
	triangular = get_rando("_rando.triangular")
	uniform = get_rando("_rando.uniform")
	vonmisesvariate = get_rando("_rando.vonmisesvariate")
	weibullvariate = get_rando("_rando.weibullvariate")


class AbstractCharacter(DiGraph):
	"""The Character API, with all requisite mappings and graph generators.

	Mappings resemble those of a NetworkX digraph:

	* ``thing`` and ``place`` are subsets of ``node``
	* ``edge``, ``adj``, and ``succ`` are aliases of ``portal``
	* ``pred`` is an alias to ``preportal``
	* ``stat`` is a dict-like mapping of data that changes over game-time,
	to be used in place of graph attributes

	"""

	engine = getatt("db")
	no_unwrap = True
	name: Hashable
	db: AbstractEngine

	@staticmethod
	def is_directed():
		return True

	@staticmethod
	def is_multigraph():
		return False

	@abstractmethod
	def add_place(self, name: NodeName, **kwargs):
		pass

	def add_node(self, name: NodeName, **kwargs):
		self.add_place(name, **kwargs)

	@abstractmethod
	def add_places_from(self, seq: Iterable, **attrs):
		pass

	def add_nodes_from(self, seq: Iterable, **attrs):
		self.add_places_from(seq, **attrs)

	def new_place(self, name: NodeName, **kwargs):
		"""Add a Place and return it.

		If there's already a Place by that name, put a number on the end.

		"""
		if name not in self.node:
			self.add_place(name, **kwargs)
			return self.place[name]
		if isinstance(name, str):
			n = 0
			while name + str(n) in self.node:
				n += 1
			self.add_place(name + str(n), **kwargs)
			return self.place[name]
		raise KeyError("Already have a node named {}".format(name))

	def new_node(self, name: NodeName, **kwargs):
		return self.new_place(name, **kwargs)

	@abstractmethod
	def add_thing(self, name: NodeName, location: NodeName, **kwargs):
		pass

	@abstractmethod
	def add_things_from(self, seq: Iterable, **attrs):
		pass

	def new_thing(self, name: NodeName, location: NodeName, **kwargs):
		"""Add a Thing and return it.

		If there's already a Thing by that name, put a number on the end.

		"""
		if name not in self.node:
			self.add_thing(name, location, **kwargs)
			return self.thing[name]
		if isinstance(name, str):
			if name in self.node:
				n = 0
				while name + str(n) in self.node:
					n += 1
				name = name + str(n)
			self.add_thing(name, location, **kwargs)
			return self.thing[name]
		raise KeyError("Already have a thing named {}".format(name))

	@abstractmethod
	def place2thing(self, place: NodeName, location: NodeName) -> None: ...

	@abstractmethod
	def thing2place(self, thing: NodeName) -> None: ...

	def remove_node(self, node):
		if node in self.node:
			self.node[node].delete()

	def remove_nodes_from(self, nodes: Iterable[NodeName]):
		for node in nodes:
			if node in self.node:
				self.node[node].delete()

	@abstractmethod
	def add_portal(self, orig: NodeName, dest: NodeName, **kwargs):
		pass

	def add_edge(self, orig: NodeName, dest: NodeName, **kwargs):
		self.add_portal(orig, dest, **kwargs)

	def new_portal(self, orig: NodeName, dest: NodeName, **kwargs):
		self.add_portal(orig, dest, **kwargs)
		return self.portal[orig][dest]

	@abstractmethod
	def add_portals_from(self, seq: Iterable, **attrs):
		pass

	def add_edges_from(self, seq: Iterable, **attrs):
		self.add_portals_from(seq, **attrs)

	@abstractmethod
	def remove_portal(self, origin: NodeName, destination: NodeName):
		pass

	def remove_portals_from(self, seq: Iterable[tuple[NodeName, NodeName]]):
		for orig, dest in seq:
			del self.portal[orig][dest]

	def remove_edges_from(self, seq: Iterable[tuple[NodeName, NodeName]]):
		self.remove_portals_from(seq)

	@abstractmethod
	def remove_place(self, place: NodeName):
		pass

	def remove_places_from(self, seq: Iterable[NodeName]):
		for place in seq:
			self.remove_place(place)

	@abstractmethod
	def remove_thing(self, thing: NodeName) -> None:
		pass

	def remove_things_from(self, seq: Iterable[NodeName]) -> None:
		for thing in seq:
			self.remove_thing(thing)

	@abstractmethod
	def add_unit(
		self, a: CharName | Node, b: Optional[NodeName] = None
	) -> None:
		pass

	@abstractmethod
	def remove_unit(
		self, a: CharName | Node, b: Optional[NodeName] = None
	) -> None:
		pass

	def __eq__(self, other: AbstractCharacter):
		return isinstance(other, AbstractCharacter) and self.name == other.name

	def __iter__(self):
		return iter(self.node)

	def __len__(self):
		return len(self.node)

	def __bool__(self):
		try:
			return self.name in self.db.character
		except AttributeError:
			return False  # we can't "really exist" when we've no engine

	def __contains__(self, k: NodeName):
		return k in self.node

	def __getitem__(self, k: NodeName):
		return self.adj[k]

	ThingMapping: type[SpecialMapping]

	@cached_property
	def thing(self) -> ThingMapping:
		return self.ThingMapping(self)

	PlaceMapping: type[SpecialMapping]

	@cached_property
	def place(self) -> PlaceMapping:
		return self.PlaceMapping(self)

	ThingPlaceMapping: type[SpecialMapping]

	@cached_property
	def _node(self) -> ThingPlaceMapping:
		return self.ThingPlaceMapping(self)

	node: ThingPlaceMapping = getatt("_node")
	nodes: ThingPlaceMapping = getatt("_node")

	PortalSuccessorsMapping: type[SpecialMapping]

	@cached_property
	def _succ(self) -> PortalSuccessorsMapping:
		return self.PortalSuccessorsMapping(self)

	portal: PortalSuccessorsMapping = getatt("_succ")
	adj: PortalSuccessorsMapping = getatt("_succ")
	succ: PortalSuccessorsMapping = getatt("_succ")
	edge: PortalSuccessorsMapping = getatt("_succ")
	_adj: PortalSuccessorsMapping = getatt("_succ")

	PortalPredecessorsMapping: type[SpecialMapping]

	@cached_property
	def _pred(self) -> PortalPredecessorsMapping:
		return self.PortalPredecessorsMapping(self)

	preportal: PortalPredecessorsMapping = getatt("_pred")
	pred: PortalPredecessorsMapping = getatt("_pred")

	UnitGraphMapping: type[SpecialMapping]

	@cached_property
	def unit(self) -> UnitGraphMapping:
		return self.UnitGraphMapping(self)

	stat: GraphMapping = getatt("graph")

	def units(self):
		for units in self.unit.values():
			yield from units.values()

	def historical(self, stat: Stat):
		from .query import EntityStatAlias

		return EntityStatAlias(entity=self.stat, stat=stat)

	def do(self, func: Callable | str, *args, **kwargs) -> AbstractCharacter:
		"""Apply the function to myself, and return myself.

		Look up the function in the method store if needed. Pass it any
		arguments given, keyword or positional.

		Useful chiefly when chaining.

		"""
		if not callable(func):
			func = getattr(self.engine.method, func)
		func(self, *args, **kwargs)
		return self

	def copy_from(self, g: AbstractCharacter) -> AbstractCharacter:
		"""Copy all nodes and edges from the given graph into this.

		Return myself.

		"""
		renamed = {}
		for k in g.nodes:
			ok = k
			if k in self.place:
				n = 0
				while k in self.place:
					k = ok + (n,) if isinstance(ok, tuple) else (ok, n)
					n += 1
			renamed[ok] = k
			self.place[k] = g.nodes[k]
		if type(g) is nx.MultiDiGraph:
			g = nx.DiGraph(g)
		elif type(g) is nx.MultiGraph:
			g = nx.Graph(g)
		if type(g) is nx.DiGraph:
			for u, v in g.edges:
				self.edge[renamed[u]][renamed[v]] = g.adj[u][v]
		else:
			assert type(g) is nx.Graph
			for u, v, d in g.edges.data():
				self.add_portal(renamed[u], renamed[v], symmetrical=True, **d)
		return self

	def become(self, g: AbstractCharacter) -> AbstractCharacter:
		"""Erase all my nodes and edges. Replace them with a copy of the graph
		provided.

		Return myself.

		"""
		self.clear()
		self.place.update(g.nodes)
		self.adj.update(g.adj)
		return self

	def clear(self) -> None:
		self.node.clear()
		self.portal.clear()
		self.stat.clear()

	def _lookup_comparator(self, comparator: Callable | str) -> Callable:
		if callable(comparator):
			return comparator
		ops = {"ge": ge, "gt": gt, "le": le, "lt": lt, "eq": eq}
		if comparator in ops:
			return ops[comparator]
		return getattr(self.engine.function, comparator)

	def cull_nodes(
		self,
		stat: Stat,
		threshold: float = 0.5,
		comparator: Callable | str = ge,
	) -> AbstractCharacter:
		"""Delete nodes whose stat >= ``threshold`` (default 0.5).

		Optional argument ``comparator`` will replace >= as the test
		for whether to cull. You can use the name of a stored function.

		"""
		comparator = self._lookup_comparator(comparator)
		dead = [
			name
			for name, node in self.node.items()
			if stat in node and comparator(node[stat], threshold)
		]
		self.remove_nodes_from(dead)
		return self

	def cull_portals(
		self,
		stat: Stat,
		threshold: float = 0.5,
		comparator: Callable | str = ge,
	):
		"""Delete portals whose stat >= ``threshold`` (default 0.5).

		Optional argument ``comparator`` will replace >= as the test
		for whether to cull. You can use the name of a stored function.

		"""
		comparator = self._lookup_comparator(comparator)
		dead = []
		for u in self.portal:
			for v in self.portal[u]:
				if stat in self.portal[u][v] and comparator(
					self.portal[u][v][stat], threshold
				):
					dead.append((u, v))
		self.remove_edges_from(dead)
		return self

	cull_edges = cull_portals


class AbstractThing(MutableMapping):
	character: AbstractCharacter
	engine: AbstractEngine
	name: NodeName

	@property
	def location(self) -> Node:
		"""The ``Thing`` or ``Place`` I'm in."""
		locn = self["location"]
		if locn is None:
			raise AttributeError("Not really a Thing")
		try:
			return self.engine._get_node(self.character, locn)
		except KeyError as ex:
			raise AttributeError("Doesn't really exist") from ex

	@location.setter
	def location(self, v: Node | NodeName):
		if hasattr(v, "name"):
			v = v.name
		self["location"] = v

	def go_to_place(
		self, place: Node | NodeName, weight: Optional[Stat] = None
	) -> int:
		"""Assuming I'm in a node that has a :class:`Portal` direct
		to the given node, schedule myself to travel to the
		given :class:`Place`, taking an amount of time indicated by
		the ``weight`` stat on the :class:`Portal`, if given; else 1
		turn.

		Return the number of turns the travel will take.

		"""
		if hasattr(place, "name"):
			placen = place.name
		else:
			placen = place
		curloc = self["location"]
		orm = self.character.engine
		turns = (
			1
			if weight is None
			else self.engine._portal_objs[
				(self.character.name, curloc, place)
			].get(weight, 1)
		)
		with self.engine.plan():
			orm.turn += turns
			self["location"] = placen
		return turns

	def follow_path(
		self,
		path: list[_Key],
		weight: Optional[_Key] = None,
		check: bool = True,
	) -> int:
		"""Go to several nodes in succession, deciding how long to
		spend in each by consulting the ``weight`` stat of the
		:class:`Portal` connecting the one node to the next,
		default 1 turn.

		Return the total number of turns the travel will take. Raise
		:class:`TravelException` if I can't follow the whole path,
		either because some of its nodes don't exist, or because I'm
		scheduled to be somewhere else. Set ``check=False`` if
		you're really sure the path is correct, and this function
		will be faster.

		"""
		if len(path) < 2:
			raise ValueError("Paths need at least 2 nodes")
		eng = self.character.engine
		if check:
			prevplace = path.pop(0)
			if prevplace != self["location"]:
				raise ValueError("Path does not start at my present location")
			subpath = [prevplace]
			for place in path:
				if (
					prevplace not in self.character.portal
					or place not in self.character.portal[prevplace]
				):
					raise exc.TravelException(
						"Couldn't follow portal from {} to {}".format(
							prevplace, place
						),
						path=subpath,
						traveller=self,
					)
				subpath.append(place)
				prevplace = place
		else:
			subpath = path.copy()
		turns_total = 0
		prevsubplace = subpath.pop(0)
		turn_incs = []
		branch, turn, tick = eng._btt()
		for subplace in subpath:
			if weight is not None:
				turn_incs.append(
					self.engine._edge_val_cache.retrieve(
						self.character.name,
						prevsubplace,
						subplace,
						0,
						branch,
						turn,
						tick,
					)
				)
			else:
				turn_incs.append(1)
			turns_total += turn_incs[-1]
			turn += turn_incs[-1]
			tick = eng._turn_end_plan.get((branch, turn), 0)
		with eng.plan():
			for subplace, turn_inc in zip(subpath, turn_incs):
				eng.turn += turn_inc
				self["location"] = subplace
		return turns_total

	def travel_to(
		self,
		dest: Node | _Key,
		weight: Optional[_Key] = None,
		graph: nx.DiGraph = None,
	) -> int:
		"""Find the shortest path to the given node from where I am
		now, and follow it.

		If supplied, the ``weight`` stat of each :class:`Portal` along
		the path will be used in pathfinding, and for deciding how
		long to stay in each Place along the way. Otherwise, I will stay
		in each :class:`Place` for 1 turn.

		The ``graph`` argument may be any NetworkX-style graph. It
		will be used for pathfinding if supplied, otherwise I'll use
		my :class:`Character`. In either case, however, I will attempt
		to actually follow the path using my :class:`Character`, which
		might not be possible if the supplied ``graph`` and my
		:class:`Character` are too different. If it's not possible,
		I'll raise a :class:`TravelException`, whose ``subpath``
		attribute holds the part of the path that I *can* follow. To
		make me follow it, pass it to my ``follow_path`` method.

		Return value is the number of turns the travel will take.

		"""
		destn = dest.name if hasattr(dest, "name") else dest
		if destn == self.location.name:
			raise ValueError("I'm already there", self.name, destn)
		graph = self.character if graph is None else graph
		path = nx.shortest_path(graph, self["location"], destn, weight)
		return self.follow_path(path, weight)


class TimeSignal(Signal, Sequence):
	"""Acts like a tuple of ``(branch, turn, tick)`` for the most part.

	This is a ``Signal``. To set a function to be called whenever the
	branch or turn changes, pass it to my ``connect`` method.

	"""

	engine: AbstractEngine

	def __init__(self, engine: AbstractEngine):
		super().__init__()
		self.engine = engine

	def __iter__(self):
		yield self.engine.branch
		yield self.engine.turn
		yield self.engine.tick

	def __len__(self):
		return 3

	def __getitem__(self, i: str | int) -> Branch | Turn | Tick:
		if i in ("branch", 0):
			return self.engine.branch
		if i in ("turn", 1):
			return self.engine.turn
		if i in ("tick", 2):
			return self.engine.tick
		if isinstance(i, int):
			raise IndexError(i)
		else:
			raise KeyError(i)

	def __setitem__(self, i: str | int, v: str | int) -> None:
		if i in ("branch", 0):
			self.engine.branch = Branch(v)
		elif i in ("turn", 1):
			self.engine.turn = Turn(v)
		elif i in ("tick", 2):
			self.engine.tick = Tick(v)
		else:
			exctyp = KeyError if isinstance(i, str) else IndexError
			raise exctyp(i)

	def __str__(self):
		return str(tuple(self))

	def __eq__(self, other):
		return tuple(self) == other

	def __ne__(self, other):
		return tuple(self) != other

	def __gt__(self, other):
		return tuple(self) > other

	def __ge__(self, other):
		return tuple(self) >= other

	def __lt__(self, other):
		return tuple(self) < other

	def __le__(self, other):
		return tuple(self) <= other


class TimeSignalDescriptor:
	__doc__ = TimeSignal.__doc__

	def __get__(self, inst, cls):
		if not hasattr(inst, "_time_signal"):
			inst._time_signal = TimeSignal(inst)
		return inst._time_signal

	def __set__(self, inst: AbstractEngine, val: Time):
		if getattr(inst, "_worker", False):
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		if not hasattr(inst, "_time_signal"):
			inst._time_signal = TimeSignal(inst)
		sig = inst._time_signal
		branch_then, turn_then, tick_then = inst._btt()
		branch_now, turn_now, tick_now = val
		if (branch_then, turn_then, tick_then) == (
			branch_now,
			turn_now,
			tick_now,
		):
			return
		e = inst
		# enforce the arrow of time, if it's in effect
		if (
			hasattr(e, "_forward")
			and e._forward
			and hasattr(e, "_planning")
			and not e._planning
		):
			if branch_now != branch_then:
				raise TimeError("Can't change branches in a forward context")
			if turn_now < turn_then:
				raise TimeError(
					"Can't time travel backward in a forward context"
				)
			if turn_now > turn_then + 1:
				raise TimeError("Can't skip turns in a forward context")
		# make sure I'll end up within the revision range of the
		# destination branch

		if branch_now in e.branches():
			e._extend_branch(branch_now, turn_now, tick_now)
			e.load_at(branch_now, turn_now, tick_now)
		else:
			e._start_branch(branch_then, branch_now, turn_now, tick_now)
		e._obranch = e.eternal["branch"] = branch_now
		e._oturn = e.eternal["turn"] = turn_now
		e._otick = e.eternal["tick"] = tick_now
		sig.send(
			e,
			branch_then=branch_then,
			turn_then=turn_then,
			tick_then=tick_then,
			branch_now=branch_now,
			turn_now=turn_now,
			tick_now=tick_now,
		)


def hash_loaded_dict(data: LoadedDict) -> dict[str, dict[str, int] | int]:
	def unlist(o):
		if isinstance(o, (list, tuple)):
			return tuple(map(unlist, o))
		return o

	hashes = {}
	for k, v in data.items():
		if isinstance(v, list):
			the_hash = 0
			for elem in v:
				the_hash |= hash(tuple(map(unlist, elem)))
			hashes[k] = the_hash
		elif isinstance(v, dict):
			hasheses = hashes[k] = {}
			for k, v in v.items():
				the_hash = 0
				for elem in v:
					the_hash |= hash(tuple(map(unlist, elem)))
				hasheses[k] = the_hash
		else:
			raise TypeError("Invalid loaded dictionary")
	return hashes


def sort_set(s: Set[_T]) -> list[_T]:
	"""Return a sorted list of the contents of a set

	This is intended to be used to iterate over world state.

	Works by converting everything to bytes before comparison. Tuples get
	their contents converted and concatenated. ``frozenset``s in the given set
	are not supported.

	This is memoized.

	"""

	def sort_set_key(v) -> bytes:
		if isinstance(v, bytes):
			return v
		elif isinstance(v, tuple):
			return b"".join(map(sort_set_key, v))
		elif isinstance(v, str):
			return v.encode()
		elif isinstance(v, int):
			return v.to_bytes(8)
		elif isinstance(v, float):
			return b"".join(i.to_bytes() for i in v.as_integer_ratio())
		else:
			raise TypeError(v)

	if not isinstance(s, Set):
		raise TypeError("sets only")
	s = frozenset(s)
	if s not in sort_set.memo:
		sort_set.memo[s] = sorted(s, key=sort_set_key)
	return sort_set.memo[s].copy()


sort_set.memo = SizedDict()


def root_type(t: type) -> type:
	if t is Key or t is Value:
		return t
	elif t is Turn or t is Tick or t is Plan or t is RuleNeighborhood:
		return int
	elif hasattr(t, "__supertype__"):
		return root_type(t.__supertype__)
	elif hasattr(t, "__origin__"):
		orig = get_origin(t)
		if orig is None:
			return t
		ret = root_type(orig)
		if ret is Literal:
			for arg in get_args(orig):
				if not isinstance(arg, str):
					raise TypeError("Literal not storeable", arg)
			return str
		return ret
	return t


def deannotate(annotation):
	if "|" in annotation:
		for a in annotation.split("|"):
			yield from deannotate(a.strip())
		return
	if "Literal" == annotation[:7]:
		for a in annotation[7:].strip("[]").split(", "):
			yield from deannotate(a)
		return
	elif "[" in annotation:
		annotation = annotation[: annotation.index("[")]
	if hasattr(builtins, annotation):
		typ = getattr(builtins, annotation)
		if not isinstance(typ, type):
			typ = type(typ)
	elif annotation in ("type(...)", "..."):
		yield type(...)
		return
	else:
		typ = eval(annotation)
	if hasattr(typ, "__supertype__"):
		typ = typ.__supertype__
	if hasattr(typ, "__origin__"):
		if typ.__origin__ is Union:
			for arg in typ.__args__:
				yield getattr(arg, "__origin__", arg)
		elif typ.__origin__ is Literal:
			yield from map(type, typ.__args__)
		else:
			yield typ.__origin__
	else:
		yield typ


class AbstractFunctionStore(ABC):
	@abstractmethod
	def save(self, reimport: bool = True) -> None: ...

	@abstractmethod
	def reimport(self) -> None: ...

	@abstractmethod
	def iterplain(self) -> Iterator[tuple[str, str]]: ...

	def store_source(self, v: str, name: str | None = None) -> None: ...

	@abstractmethod
	def get_source(self, name: str) -> str: ...

	@staticmethod
	def truth(*args):
		return True
