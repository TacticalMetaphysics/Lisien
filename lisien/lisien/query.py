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
"""Database access and query builder

The main class here is :class:`QueryEngine`, which mostly just runs
SQL on demand -- but, for the most common insert commands, it keeps
a queue of data to insert, which is then serialized and inserted
with a call to ``flush``.

Sometimes you want to know when some stat of a lisien entity had a particular
value. To find out, construct a historical query and pass it to
``Engine.turns_when``, like this::

	physical = engine.character['physical']
	that = physical.thing['that']
	hist_loc = that.historical('location')
	print(list(engine.turns_when(hist_loc == 'there')))


You'll get the turns when ``that`` was ``there``.

Other comparison operators like ``>`` and ``<`` work as well.

"""

from __future__ import annotations

from sqlalchemy import MetaData, Table, and_, select
from sqlalchemy.sql.functions import func

from .types import (
	EntityStatAlias,
	EqQuery,
	GeQuery,
	GraphMapping,
	GtQuery,
	LeQuery,
	LtQuery,
	NeQuery,
	intersect2,
)


def windows_union(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
	"""Given a list of (beginning, ending), return a minimal version that
	contains the same ranges.

	:rtype: list

	"""

	def fix_overlap(left, right):
		if left == right:
			return [left]
		assert left[0] < right[0]
		if left[1] >= right[0]:
			if right[1] > left[1]:
				return [(left[0], right[1])]
			else:
				return [left]
		return [left, right]

	if len(windows) == 1:
		return windows
	none_left = []
	none_right = []
	otherwise = []
	for window in windows:
		if window[0] is None:
			none_left.append(window)
		elif window[1] is None:
			none_right.append(window)
		else:
			otherwise.append(window)

	res = []
	otherwise.sort()
	for window in none_left:
		if not res:
			res.append(window)
			continue
		res.extend(fix_overlap(res.pop(), window))
	while otherwise:
		window = otherwise.pop(0)
		if not res:
			res.append(window)
			continue
		res.extend(fix_overlap(res.pop(), window))
	for window in none_right:
		if not res:
			res.append(window)
			continue
		res.extend(fix_overlap(res.pop(), window))
	return res


def windows_intersection(
	windows: list[tuple[int, int]],
) -> list[tuple[int, int]]:
	"""Given a list of (beginning, ending), describe where they overlap.

	Only ever returns one item, but puts it in a list anyway, to be like
	``windows_union``.

	:rtype: list
	"""
	if len(windows) == 0:
		return []
	elif len(windows) == 1:
		return list(windows)

	done = [windows[0]]
	for window in windows[1:]:
		res = intersect2(done.pop(), window)
		if res:
			done.append(res)
		else:
			return done
	return done


def _the_select(tab: Table, val_col="value"):
	return select(
		tab.c.turn.label("turn_from"),
		tab.c.tick.label("tick_from"),
		func.lead(tab.c.turn)
		.over(order_by=(tab.c.turn, tab.c.tick))
		.label("turn_to"),
		func.lead(tab.c.tick)
		.over(order_by=(tab.c.turn, tab.c.tick))
		.label("tick_to"),
		tab.c[val_col],
	)


def _make_graph_val_select(
	meta: MetaData,
	graph: bytes,
	stat: bytes,
	branches: list[str],
	mid_turn: bool,
):
	tab: Table = meta.tables["graph_val"]
	if mid_turn:
		return _the_select(tab).where(
			and_(
				tab.c.graph == graph,
				tab.c.key == stat,
				tab.c.branch.in_(branches),
			)
		)
	ticksel = (
		select(
			tab.c.graph,
			tab.c.key,
			tab.c.branch,
			tab.c.turn,
			func.max(tab.c.tick).label("tick"),
		)
		.group_by(tab.c.graph, tab.c.key, tab.c.branch, tab.c.turn)
		.where(
			and_(
				tab.c.graph == graph,
				tab.c.key == stat,
				tab.c.branch.in_(branches),
			)
		)
		.subquery()
	)
	return _the_select(tab).select_from(
		tab.join(
			ticksel,
			and_(
				tab.c.graph == ticksel.c.graph,
				tab.c.key == ticksel.c.key,
				tab.c.branch == ticksel.c.branch,
				tab.c.turn == ticksel.c.turn,
				tab.c.tick == ticksel.c.tick,
			),
		)
	)


def _make_node_val_select(
	meta: MetaData,
	graph: bytes,
	node: bytes,
	stat: bytes,
	branches: list[str],
	mid_turn: bool,
):
	tab: Table = meta.tables["node_val"]
	if mid_turn:
		return _the_select(tab).where(
			and_(
				tab.c.graph == graph,
				tab.c.node == node,
				tab.c.key == stat,
				tab.c.branch.in_(branches),
			)
		)
	ticksel = (
		select(
			tab.c.graph,
			tab.c.node,
			tab.c.key,
			tab.c.branch,
			tab.c.turn,
			func.max(tab.c.tick).label("tick"),
		)
		.where(
			and_(
				tab.c.graph == graph,
				tab.c.node == node,
				tab.c.key == stat,
				tab.c.branch.in_(branches),
			)
		)
		.group_by(tab.c.graph, tab.c.node, tab.c.key, tab.c.branch, tab.c.turn)
		.subquery()
	)
	return _the_select(tab).select_from(
		tab.join(
			ticksel,
			and_(
				tab.c.graph == ticksel.c.graph,
				tab.c.node == ticksel.c.node,
				tab.c.key == ticksel.c.key,
				tab.c.branch == ticksel.c.branch,
				tab.c.turn == ticksel.c.turn,
				tab.c.tick == ticksel.c.tick,
			),
		)
	)


def _make_location_select(
	meta: MetaData,
	graph: bytes,
	thing: bytes,
	branches: list[str],
	mid_turn: bool,
):
	tab: Table = meta.tables["things"]
	if mid_turn:
		return _the_select(tab, val_col="location").where(
			and_(
				tab.c.character == graph,
				tab.c.thing == thing,
				tab.c.branch.in_(branches),
			)
		)
	ticksel = (
		select(
			tab.c.character,
			tab.c.thing,
			tab.c.branch,
			tab.c.turn,
			func.max(tab.c.tick).label("tick"),
		)
		.where(
			and_(
				tab.c.character == graph,
				tab.c.thing == thing,
				tab.c.branch.in_(branches),
			)
		)
		.group_by(tab.c.character, tab.c.thing, tab.c.branch, tab.c.turn)
		.subquery()
	)
	return _the_select(tab, val_col="location").select_from(
		tab.join(
			ticksel,
			and_(
				tab.c.character == ticksel.c.character,
				tab.c.thing == ticksel.c.thing,
				tab.c.branch == ticksel.c.branch,
				tab.c.turn == ticksel.c.turn,
				tab.c.tick == ticksel.c.tick,
			),
		)
	)


def _make_edge_val_select(
	meta: MetaData,
	graph: bytes,
	orig: bytes,
	dest: bytes,
	stat: bytes,
	branches: list[str],
	mid_turn: bool,
):
	tab: Table = meta.tables["edge_val"]
	if mid_turn:
		return _the_select(tab).where(
			and_(
				tab.c.graph == graph,
				tab.c.orig == orig,
				tab.c.dest == dest,
				tab.c.key == stat,
				tab.c.branches.in_(branches),
			)
		)
	ticksel = (
		select(
			tab.c.graph,
			tab.c.orig,
			tab.c.dest,
			tab.c.key,
			tab.c.branch,
			tab.c.turn,
			tab.c.tick if mid_turn else func.max(tab.c.tick).label("tick"),
		)
		.where(
			and_(
				tab.c.graph == graph,
				tab.c.orig == orig,
				tab.c.dest == dest,
				tab.c.key == stat,
				tab.c.branch.in_(branches),
			)
		)
		.group_by(
			tab.c.graph,
			tab.c.orig,
			tab.c.dest,
			tab.c.key,
			tab.c.branch,
			tab.c.turn,
		)
		.subquery()
	)
	return _the_select(tab).select_from(
		tab.join(
			ticksel,
			and_(
				tab.c.graph == ticksel.c.graph,
				tab.c.orig == ticksel.c.orig,
				tab.c.dest == ticksel.c.dest,
				tab.c.key == ticksel.c.key,
				tab.c.branch == ticksel.c.branch,
				tab.c.turn == ticksel.c.turn,
				tab.c.tick == ticksel.c.tick,
			),
		)
	)


def _make_side_sel(
	meta: MetaData,
	entity,
	stat,
	branches: list[str],
	pack: callable,
	mid_turn: bool,
):
	from .node import Place, Thing
	from .portal import Portal

	if isinstance(entity, GraphMapping):
		return _make_graph_val_select(
			meta, pack(entity.character.name), pack(stat), branches, mid_turn
		)
	elif isinstance(entity, Place):
		return _make_node_val_select(
			meta,
			pack(entity.character.name),
			pack(entity.name),
			pack(stat),
			branches,
			mid_turn,
		)
	elif isinstance(entity, Thing):
		if stat == "location":
			return _make_location_select(
				meta,
				pack(entity.character.name),
				pack(entity.name),
				branches,
				mid_turn,
			)
		else:
			return _make_node_val_select(
				meta,
				pack(entity.character.name),
				pack(entity.name),
				pack(stat),
				branches,
				mid_turn,
			)
	elif isinstance(entity, Portal):
		return _make_edge_val_select(
			meta,
			pack(entity.character.name),
			pack(entity.origin.name),
			pack(entity.destination.name),
			pack(stat),
			branches,
			mid_turn,
		)
	else:
		raise TypeError(f"Unknown entity type {type(entity)}")


def _getcol(alias: "EntityStatAlias"):
	from .node import Thing

	if isinstance(alias.entity, Thing) and alias.stat == "location":
		return "location"
	return "value"


comparisons = {
	"eq": EqQuery,
	"ne": NeQuery,
	"gt": GtQuery,
	"lt": LtQuery,
	"ge": GeQuery,
	"le": LeQuery,
}
