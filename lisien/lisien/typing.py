from __future__ import annotations

from typing import Any

Key = str | int | float | None | tuple["Key", ...] | frozenset["Key"]
NodeRowType = tuple[Key, Key, str, int, int, bool]
EdgeRowType = tuple[Key, Key, Key, int, str, int, int, bool]
GraphValRowType = tuple[Key, Key, str, int, int, Any]
NodeValRowType = tuple[Key, Key, Key, str, int, int, Any]
EdgeValRowType = tuple[Key, Key, Key, int, str, int, int, Any]
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
