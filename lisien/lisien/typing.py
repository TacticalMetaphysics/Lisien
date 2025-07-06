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

from typing import Any, Hashable, Literal, NewType, TypeGuard

_Key = str | int | float | None | tuple["Key", ...] | frozenset["Key"]


class Key(Hashable):
	"""Fake class for things lisien can use as keys

	They have to be serializable using lisien's particular msgpack schema,
	as well as hashable.

	"""

	def __new__(cls, that: Key) -> Key:
		return that

	def __instancecheck__(cls, instance) -> TypeGuard[Key]:
		return isinstance(instance, (str, int, float)) or (
			(isinstance(instance, tuple) or isinstance(instance, frozenset))
			and all(isinstance(elem, cls) for elem in instance)
		)


Key.register(str)
Key.register(int)
Key.register(float)
Key.register(type(None))


Stat = NewType("Stat", Key)
EternalKey = NewType("EternalKey", Key)
UniversalKey = NewType("UniversalKey", Key)
Branch = NewType("Branch", str)
Turn = NewType("Turn", int)
Tick = NewType("Tick", int)
Time = tuple[Branch, Turn, Tick]
TimeWindow = tuple[Branch, Turn, Tick, Turn, Tick]
Plan = NewType("Plan", int)
CharName = NewType("CharName", Key)
NodeName = NewType("NodeName", Key)
EntityKey = (
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
RuleFuncName = TriggerFuncName | PrereqFuncName | ActionFuncName
UniversalKeyframe = NewType("UniversalKeyframe", dict)
RuleKeyframe = NewType("RuleKeyframe", dict)
RulebookKeyframe = NewType("RulebookKeyframe", dict)
NodeKeyframe = NewType("NodeKeyframe", dict)
EdgeKeyframe = NewType("EdgeKeyframe", dict)
NodeRowType = tuple[CharName, NodeName, Branch, Turn, Tick, bool]
EdgeRowType = tuple[
	CharName, NodeName, NodeName, int, Branch, Turn, Tick, bool
]
GraphValRowType = tuple[CharName, Key, Branch, Turn, Tick, Any]
NodeValRowType = tuple[CharName, NodeName, Key, Branch, Turn, Tick, Any]
EdgeValRowType = tuple[
	CharName, NodeName, NodeName, int, Key, Branch, Turn, Tick, Any
]
StatDict = dict[Stat | Literal["rulebook"], Any]
CharDict = dict[
	Stat
	| Literal[
		"units",
		"character_rulebook",
		"unit_rulebook",
		"character_thing_rulebook",
		"character_place_rulebook",
		"character_portal_rulebook",
	],
	Any,
]
GraphValKeyframe = NewType("GraphValKeyframe", dict[CharName, CharDict])
NodeValDict = dict[NodeName, StatDict]
GraphNodeValKeyframe = dict[CharName, NodeValDict]
EdgeValDict = dict[NodeName, dict[NodeName, StatDict]]
GraphEdgeValKeyframe = dict[CharName, EdgeValDict]
NodesDict = dict[NodeName, bool]
GraphNodesKeyframe = dict[CharName, NodesDict]
EdgesDict = dict[NodeName, dict[NodeName, bool]]
GraphEdgesKeyframe = dict[CharName, EdgesDict]
DeltaDict = dict[
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
KeyframeTuple = tuple[
	CharName,
	Branch,
	Turn,
	Tick,
	GraphNodeValKeyframe,
	GraphEdgeValKeyframe,
	GraphValKeyframe,
]
Keyframe = dict[
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
	| dict[UniversalKey, Any]
	| dict[RuleName, list[TriggerFuncName]]
	| dict[RuleName, list[PrereqFuncName]]
	| dict[RuleName, list[ActionFuncName]]
	| dict[RuleName, int]
	| dict[RuleName, bool]
	| dict[RulebookName, RulebookKeyframe],
]
SlightlyPackedDeltaType = dict[
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
