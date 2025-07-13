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

from typing import Literal, NewType, TypeAlias, TypeGuard

from .wrap import DictWrapper, ListWrapper, SetWrapper

Key = str | int | float | None | tuple["Key", ...] | frozenset["Key"]


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


class KeyClass:
	"""Fake class for things Lisien can use as keys

	They have to be serializable using lisien's particular msgpack schema,
	as well as hashable.

	"""

	def __new__(cls, that: Key) -> Key:
		return that

	def __instancecheck__(self, instance) -> bool:
		return is_valid_key(instance)


Value: TypeAlias = (
	Key
	| dict[Key, "Value"]
	| tuple["Value", ...]
	| list["Value"]
	| set["Value"]
	| frozenset["Value"]
)
Value |= DictWrapper[Key, Value] | ListWrapper[Value] | SetWrapper[Value]


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


class ValueClass:
	"""Fake class for things Lisien can use as values

	They have to be serializable using Lisien's msgpack schema.

	"""

	def __new__(cls, that: Value) -> Value:
		return that

	def __instancecheck__(self, instance) -> bool:
		return is_valid_value(instance)


Stat = NewType("Stat", KeyClass)
EternalKey = NewType("EternalKey", KeyClass)
UniversalKey = NewType("UniversalKey", KeyClass)
Branch = NewType("Branch", str)
Turn = NewType("Turn", int)
Tick = NewType("Tick", int)
Time: TypeAlias = tuple[Branch, Turn, Tick]
LinearTime: TypeAlias = tuple[Turn, Tick]
TimeWindow: TypeAlias = tuple[Branch, Turn, Tick, Turn, Tick]
Plan = NewType("Plan", int)
CharName = NewType("CharName", KeyClass)
NodeName = NewType("NodeName", KeyClass)
EntityKey: TypeAlias = (
	tuple[CharName]
	| tuple[CharName, NodeName]
	| tuple[CharName, NodeName, NodeName]
)
RulebookName = NewType("RulebookName", KeyClass)
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
StatDict: TypeAlias = dict[Stat | Literal["rulebook"], ValueClass]
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
	ValueClass,
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
		Stat | Literal["nodes", "node_val", "edges", "edge_val", "rulebook"],
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
	CharName
	| Literal[
		"universal",
		"triggers",
		"prereqs",
		"actions",
		"neighborhood",
		"big",
		"rulebook",
	],
	dict[
		Literal["graph_val", "nodes", "node_val", "edges", "edge_val"],
		GraphValKeyframe
		| GraphNodesKeyframe
		| GraphNodeValKeyframe
		| GraphEdgesKeyframe
		| GraphEdgeValKeyframe,
	]
	| dict[UniversalKey, ValueClass]
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
