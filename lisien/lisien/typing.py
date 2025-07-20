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

from typing import Literal, NewType, TypeAlias, TypeGuard, TypeVar, Generic

from .wrap import DictWrapper, ListWrapper, SetWrapper


def is_valid_key(obj) -> TypeGuard[Key]:
	"""Is this an object that Lisien can serialize as a key?"""
	return (
		obj is None
		or isinstance(obj, (str, int, float))
		or (
			isinstance(obj, (tuple, frozenset))
			and all(is_valid_key(elem) for elem in obj)
		)
	)


_Key: TypeAlias = (
	str | int | float | None | tuple["_Key", ...] | frozenset["_Key"]
)


KeyType = TypeVar(
	"KeyType",
	bound=_Key,
	covariant=True,
)


class Key(Generic[KeyType]):
	def __new__(cls, object) -> KeyType:
		if not isinstance(object, cls):
			raise TypeError("Not a valid key")

	def __instancecheck__(self, instance) -> bool:
		return is_valid_key(instance)


_Value: TypeAlias = (
	_Key
	| dict[_Key, "_Value"]
	| tuple["_Value", ...]
	| list["_Value"]
	| set["_Value"]
	| frozenset["_Value"]
)
_Value |= DictWrapper[_Key, _Value] | ListWrapper[_Value] | SetWrapper[_Value]


def is_valid_value(obj) -> TypeGuard[Value]:
	"""Is this an object that Lisien can serialize as a value?"""
	return (
		is_valid_key(obj)
		or (
			isinstance(obj, (dict, DictWrapper))
			and all(map(is_valid_key, obj.keys()))
			and all(map(is_valid_value, obj.values()))
		)
		or (
			isinstance(
				obj, (tuple, list, set, frozenset, ListWrapper, SetWrapper)
			)
			and all(map(is_valid_value, obj))
		)
	)


ValueType = TypeVar("ValueType", bound=_Value, covariant=True)


class Value(Generic[ValueType]):
	def __new__(cls, obj) -> ValueType:
		if not isinstance(obj, cls):
			raise TypeError("Not a valid value")
		return obj

	def __instancecheck__(self, instance) -> bool:
		return is_valid_value(instance)


Stat = NewType("Stat", Key)
EternalKey = NewType("EternalKey", Key)
UniversalKey = NewType("UniversalKey", Key)
Branch = NewType("Branch", str)
Turn = NewType("Turn", int)
Tick = NewType("Tick", int)
Time: TypeAlias = tuple[Branch, Turn, Tick]
LinearTime: TypeAlias = tuple[Turn, Tick]
TimeWindow: TypeAlias = tuple[Branch, Turn, Tick, Turn, Tick]
Plan = NewType("Plan", int)
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
RuleNeighborhood = NewType("RuleNeighborhood", int)
RuleBig = NewType("RuleBig", bool)
FuncName = NewType("FuncName", str)
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
	CharName, NodeName, NodeName, int, Branch, Turn, Tick, bool
]
GraphValRowType: TypeAlias = tuple[CharName, Key, Branch, Turn, Tick, Value]
NodeValRowType: TypeAlias = tuple[
	CharName, NodeName, Key, Branch, Turn, Tick, Value
]
EdgeValRowType: TypeAlias = tuple[
	CharName, NodeName, NodeName, int, Key, Branch, Turn, Tick, Value
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
DeltaDict: TypeAlias = dict[
	CharName,
	dict[
		Stat
		| Literal[
			"nodes", "node_val", "edges", "edge_val", "rulebook", "units"
		],
		StatDict
		| NodesDict
		| NodeValDict
		| EdgesDict
		| EdgeValDict
		| RulebookName,
	]
	| None,
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
