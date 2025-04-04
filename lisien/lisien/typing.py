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
