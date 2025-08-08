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

from collections.abc import Iterable
from types import GenericAlias
from typing import (
	Annotated,
	Any,
	Callable,
	Literal,
	NewType,
	TypeAlias,
	TypeGuard,
)

import networkx as nx
from annotated_types import Ge

from .wrap import DictWrapper, ListWrapper, SetWrapper

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
	| type(...)
)


def is_valid_value(obj: _Value) -> TypeGuard[Value]:
	"""Is this an object that Lisien can serialize as a value?"""
	return (
		obj is ...
		or is_valid_key(obj)
		or (
			isinstance(obj, (dict, DictWrapper))
			and all(map(is_valid_key, obj.keys()))
			and all(map(is_valid_value, obj.values()))
		)
		or (
			isinstance(
				obj, (tuple, list, set, frozenset, ListWrapper, SetWrapper)
			)
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
UniversalKeyframe = NewType("UniversalKeyframe", dict)
RuleKeyframe = NewType("RuleKeyframe", dict)
RulebookKeyframe = NewType("RulebookKeyframe", dict)
NodeKeyframe = NewType("NodeKeyframe", dict)
EdgeKeyframe = NewType("EdgeKeyframe", dict)
NodeRowType: TypeAlias = tuple[CharName, NodeName, Branch, Turn, Tick, bool]
EdgeRowType: TypeAlias = tuple[
	CharName, NodeName, NodeName, Branch, Turn, Tick, bool
]
GraphValRowType: TypeAlias = tuple[CharName, Key, Branch, Turn, Tick, Value]
NodeValRowType: TypeAlias = tuple[
	CharName, NodeName, Key, Branch, Turn, Tick, Value
]
EdgeValRowType: TypeAlias = tuple[
	CharName, NodeName, NodeName, Key, Branch, Turn, Tick, Value
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
GraphNodeValKeyframe: TypeAlias = dict[CharName, NodeValDict]
EdgeValDict: TypeAlias = dict[NodeName, dict[NodeName, StatDict]]
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
	| dict[RuleName, int]
	| dict[RuleName, bool]
	| dict[RulebookName, RulebookKeyframe],
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
