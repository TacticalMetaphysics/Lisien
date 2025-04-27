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

from typing import Any, Hashable, NewType, TypeGuard

_Key = str | int | float | None | tuple["Key", ...] | frozenset["Key"]


# noinspection PyRedeclaration
class Key(Hashable):
	"""Fake class for things lisien can use as keys

	They have to be serializable using lisien's particular msgpack schema,
	as well as hashable.

	"""

	def __new__(cls, that: _Key) -> _Key:
		return that

	def __instancecheck__(cls, instance) -> TypeGuard[_Key]:
		return isinstance(instance, (str, int, float)) or (
			(isinstance(instance, tuple) or isinstance(instance, frozenset))
			and all(isinstance(elem, cls) for elem in instance)
		)


Key.register(str)
Key.register(int)
Key.register(float)
Key.register(type(None))

KeyHint = Key | str | int | float | None
KeyHint |= tuple[KeyHint, ...]
KeyHint |= frozenset[Key]


Branch = NewType("Branch", str)
Turn = NewType("Turn", int)
Tick = NewType("Tick", int)
Time = tuple[Branch, Turn, Tick]
TimeWindow = tuple[Branch, Turn, Tick, Turn, Tick]
Plan = NewType("Plan", int)
CharName = NewType("CharName", Key)
NodeName = NewType("NodeName", Key)
RulebookName = NewType("RulebookName", Key)
RulebookPriority = NewType("RulebookPriority", float)
RuleName = NewType("RuleName", str)
RuleNeighborhood = NewType("RuleNeighborhood", int)
RuleBig = NewType("RuleBig", bool)
FuncName = NewType("FuncName", str)
TriggerFuncName = NewType("TriggerFuncName", FuncName)
PrereqFuncName = NewType("PrereqFuncName", FuncName)
ActionFuncName = NewType("ActionFuncName", FuncName)
UniversalKeyframe = NewType("UniversalKeyframe", dict)
RuleKeyframe = NewType("RuleKeyframe", dict)
RulebookKeyframe = NewType("RulebookKeyframe", dict)
NodeKeyframe = NewType("NodeKeyframe", dict)
EdgeKeyframe = NewType("EdgeKeyframe", dict)
GraphValKeyframe = NewType("GraphValKeyframe", dict)
NodeRowType = tuple[CharName, NodeName, Branch, Turn, Tick, bool]
EdgeRowType = tuple[
	CharName, NodeName, NodeName, int, Branch, Turn, Tick, bool
]
GraphValRowType = tuple[CharName, Key, Branch, Turn, Tick, Any]
NodeValRowType = tuple[CharName, NodeName, Key, Branch, Turn, Tick, Any]
EdgeValRowType = tuple[
	CharName, NodeName, NodeName, int, Key, Branch, Turn, Tick, Any
]
StatDict = dict[Key, Any]
GraphValDict = dict[Key, StatDict]
NodeValDict = dict[Key, StatDict]
GraphNodeValDict = dict[Key, NodeValDict]
EdgeValDict = dict[Key, dict[Key, StatDict]]
GraphEdgeValDict = dict[Key, EdgeValDict]
DeltaDict = dict[
	Key, GraphValDict | GraphNodeValDict | GraphEdgeValDict | StatDict | None
]
KeyframeTuple = tuple[
	Key,
	Branch,
	Turn,
	Tick,
	GraphNodeValDict,
	GraphEdgeValDict,
	GraphValDict,
]
NodesDict = dict[Key, bool]
GraphNodesDict = dict[Key, NodesDict]
EdgesDict = dict[Key, dict[Key, bool]]
GraphEdgesDict = dict[Key, EdgesDict]
