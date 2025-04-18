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

from typing import Any, NewType

Branch = NewType("Branch", str)
Turn = NewType("Turn", int)
Tick = NewType("Tick", int)
Time = tuple[Branch, Turn, Tick]
Plan = NewType("Plan", int)
Key = str | int | float | None | tuple["Key", ...] | frozenset["Key"]
NodeRowType = tuple[Key, Key, Branch, Turn, Tick, bool]
EdgeRowType = tuple[Key, Key, Key, int, Branch, Turn, Tick, bool]
GraphValRowType = tuple[Key, Key, Branch, Turn, Tick, Any]
NodeValRowType = tuple[Key, Key, Key, Branch, Turn, Tick, Any]
EdgeValRowType = tuple[Key, Key, Key, int, Branch, Turn, Tick, Any]
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
