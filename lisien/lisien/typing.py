from __future__ import annotations

from collections.abc import Hashable

from typing import Any, Self


class Key(Hashable):
	"""Type hint for things lisien can use as keys

	They have to be serializable using lisien's particular msgpack schema,
	as well as hashable.

	"""

	def __new__(cls, that: Self) -> Self:
		return that

	def __instancecheck__(cls, instance: Self) -> bool:
		return isinstance(instance, (str, int, float)) or (
			(isinstance(instance, tuple) or isinstance(instance, frozenset))
			and all(isinstance(elem, cls) for elem in instance)
		)


NodeRowType = tuple[Hashable, Hashable, str, int, int, bool]
EdgeRowType = tuple[Hashable, Hashable, Hashable, int, str, int, int, bool]
GraphValRowType = tuple[Hashable, Hashable, str, int, int, Any]
NodeValRowType = tuple[Hashable, Hashable, Hashable, str, int, int, Any]
EdgeValRowType = tuple[Hashable, Hashable, Hashable, int, str, int, int, Any]
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
	str,
	int,
	int,
	GraphNodeValDict,
	GraphEdgeValDict,
	GraphValDict,
]
NodesDict = dict[Key, bool]
GraphNodesDict = dict[Key, NodesDict]
EdgesDict = dict[Key, dict[Key, bool]]
GraphEdgesDict = dict[Key, EdgesDict]
