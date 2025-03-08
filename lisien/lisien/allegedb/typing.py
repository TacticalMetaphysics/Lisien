from typing import Hashable, Any

Key = str | int | float | tuple["Key", ...] | frozenset["Key"]
"""Type hint for things lisien can use as keys

They have to be serializable using lisien's particular msgpack schema,
as well as hashable.

"""
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
