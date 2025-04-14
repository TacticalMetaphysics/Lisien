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

import operator
from collections.abc import Sequence, Set
from itertools import chain
from operator import eq, ge, gt, le, lt, ne
from typing import Any, Callable

from sqlalchemy import Table, and_, select
from sqlalchemy.sql.functions import func

from .alchemy import meta
from .util import EntityStatAccessor


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


def intersect2(left, right):
	"""Return intersection of 2 windows of time"""
	if left == right:
		return left
	elif left == (None, None) or left == ((None, None), (None, None)):
		return right
	elif right == (None, None) or right == ((None, None), (None, None)):
		return left
	elif left[0] is None or left[0] == (None, None):
		if right[0] is None or right[0] == (None, None):
			return None, min((left[1], right[1]))
		elif right[1] is None or right[1] == (None, None):
			if left[1] <= right[0]:
				return left[1], right[0]
			else:
				return None
		elif right[0] <= left[1]:
			return right[0], left[1]
		else:
			return None
	elif left[1] is None or left[1] == (None, None):
		if right[0] is None or right[0] == (None, None):
			return left[0], right[1]
		elif left[0] <= right[0]:
			return right
		elif right[1] is None or right[1] == (None, None):
			return max((left[0], right[0])), (None, None) if isinstance(
				left[0], tuple
			) else None
		elif left[0] <= right[1]:
			return left[0], right[1]
		else:
			return None
	# None not in left
	elif right[0] is None or right[0] == (None, None):
		return left[0], min((left[1], right[1]))
	elif right[1] is None or right[1] == (None, None):
		if left[1] >= right[0]:
			return right[0], left[1]
		else:
			return None
	if left > right:
		(left, right) = (right, left)
	if left[1] >= right[0]:
		if right[1] > left[1]:
			return right[0], left[1]
		else:
			return right
	return None


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
	graph: bytes, stat: bytes, branches: list[str], mid_turn: bool
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
	graph: bytes, node: bytes, stat: bytes, branches: list[str], mid_turn: bool
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
	graph: bytes, thing: bytes, branches: list[str], mid_turn: bool
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
	graph: bytes,
	orig: bytes,
	dest: bytes,
	idx: int,
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
				tab.c.idx == idx,
				tab.c.key == stat,
				tab.c.branches.in_(branches),
			)
		)
	ticksel = (
		select(
			tab.c.graph,
			tab.c.orig,
			tab.c.dest,
			tab.c.idx,
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
				tab.c.idx == idx,
				tab.c.key == stat,
				tab.c.branch.in_(branches),
			)
		)
		.group_by(
			tab.c.graph,
			tab.c.orig,
			tab.c.dest,
			tab.c.idx,
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
				tab.c.idx == ticksel.c.idx,
				tab.c.key == ticksel.c.key,
				tab.c.branch == ticksel.c.branch,
				tab.c.turn == ticksel.c.turn,
				tab.c.tick == ticksel.c.tick,
			),
		)
	)


def _make_side_sel(
	entity, stat, branches: list[str], pack: callable, mid_turn: bool
):
	from .character import AbstractCharacter
	from .node import Place, Thing
	from .portal import Portal

	if isinstance(entity, AbstractCharacter):
		return _make_graph_val_select(
			pack(entity.name), pack(stat), branches, mid_turn
		)
	elif isinstance(entity, Place):
		return _make_node_val_select(
			pack(entity.character.name),
			pack(entity.name),
			pack(stat),
			branches,
			mid_turn,
		)
	elif isinstance(entity, Thing):
		if stat == "location":
			return _make_location_select(
				pack(entity.character.name),
				pack(entity.name),
				branches,
				mid_turn,
			)
		else:
			return _make_node_val_select(
				pack(entity.character.name),
				pack(entity.name),
				pack(stat),
				branches,
				mid_turn,
			)
	elif isinstance(entity, Portal):
		return _make_edge_val_select(
			pack(entity.character.name),
			pack(entity.origin.name),
			pack(entity.destination.name),
			0,
			pack(stat),
			branches,
			mid_turn,
		)
	else:
		raise TypeError(f"Unknown entity type {type(entity)}")


def _getcol(alias: "StatusAlias"):
	from .node import Thing

	if isinstance(alias.entity, Thing) and alias.stat == "location":
		return "location"
	return "value"


class QueryResult(Sequence, Set):
	"""A slightly lazy tuple-like object holding a history query's results

	Testing for membership of a turn number in a QueryResult only evaluates
	the predicate for that turn number, and testing for membership of nearby
	turns is fast. Accessing the start or the end of the QueryResult only
	evaluates the initial or final item. Other forms of access cause the whole
	query to be evaluated in parallel.

	"""

	def __init__(self, windows_l, windows_r, oper, end_of_time):
		self._past_l = windows_l
		self._future_l = []
		self._past_r = windows_r
		self._future_r = []
		self._oper = oper
		self._list = None
		self._trues = set()
		self._falses = set()
		self._end_of_time = end_of_time

	def __iter__(self):
		if self._list is None:
			self._generate()
		return iter(self._list)

	def __reversed__(self):
		if self._list is None:
			self._generate()
		return reversed(self._list)

	def __len__(self):
		if not self._list:
			self._generate()
		return len(self._list)

	def __getitem__(self, item):
		if not self._list:
			if item == 0:
				return self._first()
			elif item == -1:
				return self._last()
			self._generate()
		return self._list[item]

	def _generate(self):
		raise NotImplementedError("_generate")

	def _first(self):
		raise NotImplementedError("_first")

	def _last(self):
		raise NotImplementedError("_last")

	def __str__(self):
		return f"<{self.__class__.__name__} containing {list(self)}>"

	def __repr__(self):
		return (
			f"<{self.__class__.__name__}({self._past_l}, {self._past_r},"
			f"{self._oper}, {self._end_of_time})>"
		)


class QueryResultEndTurn(QueryResult):
	def _generate(self):
		spans = []
		left = []
		right = []
		for turn_from, turn_to, l_v, r_v in _yield_intersections(
			chain(iter(self._past_l), reversed(self._future_l)),
			chain(iter(self._past_r), reversed(self._future_r)),
			until=self._end_of_time,
		):
			spans.append((turn_from, turn_to))
			left.append(l_v)
			right.append(r_v)
		try:
			import numpy as np

			bools = self._oper(np.array(left), np.array(right))
		except ImportError:
			bools = [self._oper(l, r) for (l, r) in zip(left, right)]
		self._list = _list = []
		append = _list.append
		add = self._trues.add
		for span, buul in zip(spans, bools):
			if buul:
				for turn in range(*span):
					append(turn)
					add(turn)

	def __contains__(self, item):
		if self._list is not None:
			return item in self._trues
		elif item in self._trues:
			return True
		elif item in self._falses:
			return False
		future_l = self._future_l
		past_l = self._past_l
		future_r = self._future_r
		past_r = self._past_r
		if not past_l:
			if not future_l:
				return False
			past_l.append((future_l.pop()))
		if not past_r:
			if not future_r:
				return False
			past_r.append((future_r.pop()))
		while past_l and past_l[-1][0] > item:
			future_l.append(past_l.pop())
		while future_l and future_l[-1][0] <= item:
			past_l.append(future_l.pop())
		while past_r and past_r[-1][0] > item:
			future_r.append(past_r.pop())
		while future_r and future_r[-1][0] <= item:
			past_r.append(future_r.pop())
		ret = self._oper(past_l[-1][2], past_r[-1][2])
		if ret:
			self._trues.add(item)
		else:
			self._falses.add(item)
		return ret

	def _last(self):
		"""Get the last turn on which the predicate held true"""
		past_l = self._past_l
		future_l = self._future_l
		while future_l:
			past_l.append(future_l.pop())
		past_r = self._past_r
		future_r = self._future_r
		while future_r:
			past_r.append(future_r)
		oper = self._oper
		while past_l and past_r:
			l_from, l_to, l_v = past_l[-1]
			r_from, r_to, r_v = past_r[-1]
			inter = intersect2((l_from, l_to), (r_from, r_to))
			if not inter:
				if l_from < r_from:
					future_r.append(past_r.pop())
				else:
					future_l.append(past_l.pop())
				continue
			if oper(l_v, r_v):
				# SQL results are exclusive on the right
				if inter[1] is None:
					return self._end_of_time - 1
				return inter[1] - 1

	def _first(self):
		"""Get the first turn on which the predicate held true"""
		if self._list is not None:
			if not self._list:
				return
			return self._list[0]
		oper = self._oper
		for turn_from, turn_to, l_v, r_v in _yield_intersections(
			chain(iter(self._past_l), reversed(self._future_l)),
			chain(iter(self._past_r), reversed(self._future_r)),
			until=self._end_of_time,
		):
			if oper(l_v, r_v):
				return turn_from


def _yield_intersections(iter_l, iter_r, until=None):
	try:
		l_from, l_to, l_v = next(iter_l)
	except StopIteration:
		return
	try:
		r_from, r_to, r_v = next(iter_r)
	except StopIteration:
		return
	while True:
		if l_to in (None, (None, None)):
			l_to = until
		if r_to in (None, (None, None)):
			r_to = until
		intersection = intersect2((l_from, l_to), (r_from, r_to))
		if intersection and intersection[0] != intersection[1]:
			yield intersection + (l_v, r_v)
			if intersection[1] is None or (
				isinstance(intersection[1], tuple) and intersection[1] is None
			):
				return
		if (
			l_to is None
			or r_to is None
			or (isinstance(l_to, tuple) and l_to[1] is None)
			or (isinstance(r_to, tuple) and r_to[1] is None)
		):
			break
		elif l_to <= r_to:
			try:
				l_from, l_to, l_v = next(iter_l)
			except StopIteration:
				break
		else:
			try:
				r_from, r_to, r_v = next(iter_r)
			except StopIteration:
				break
	if l_to is None:
		while True:
			try:
				r_from, r_to, r_v = next(iter_r)
			except StopIteration:
				if until:
					yield intersect2((l_from, l_to), (r_to, until)) + (
						l_v,
						r_v,
					)
				return
			yield intersect2((l_from, l_to), (r_from, r_to)) + (l_v, r_v)
	if r_to is None:
		while True:
			try:
				l_from, l_to, l_v = next(iter_l)
			except StopIteration:
				if until:
					yield intersect2((l_to, until), (r_from, r_to)) + (
						l_v,
						r_v,
					)
				return
			yield intersect2((l_from, l_to), (r_from, r_to)) + (l_v, r_v)


class QueryResultMidTurn(QueryResult):
	def _generate(self):
		spans = []
		left = []
		right = []
		for time_from, time_to, l_v, r_v in _yield_intersections(
			chain(iter(self._past_l), reversed(self._future_l)),
			chain(iter(self._past_r), reversed(self._future_r)),
			until=(self._end_of_time, 0),
		):
			spans.append((time_from, time_to))
			left.append(l_v)
			right.append(r_v)
		try:
			import numpy as np

			bools = self._oper(np.array(left), np.array(right))
		except ImportError:
			bools = [self._oper(l, r) for (l, r) in zip(left, right)]
		trues = self._trues
		_list = self._list = []
		for span, buul in zip(spans, bools):
			if buul:
				for turn in range(
					span[0][0], span[1][0] + (1 if span[1][1] else 0)
				):
					if turn in trues:
						continue
					trues.add(turn)
					_list.append(turn)

	def __contains__(self, item):
		if self._list is not None:
			return item in self._trues
		if item in self._trues:
			return True
		if item in self._falses:
			return False
		future_l = self._future_l
		past_l = self._past_l
		future_r = self._future_r
		past_r = self._past_r
		if not past_l:
			if not future_l:
				return False
			past_l.append(future_l.pop())
		if not past_r:
			if not future_r:
				return False
			past_r.append(future_r.pop())
		while past_l and past_l[-1][0][0] >= item:
			future_l.append(past_l.pop())
		while future_l and not (
			past_l
			and past_l[-1][0][0] <= item
			and (past_l[-1][1][0] is None or item <= past_l[-1][1][0])
		):
			past_l.append(future_l.pop())
		left_candidates = [past_l[-1]]
		while (
			future_l
			and future_l[-1][0][0] <= item
			and (future_l[-1][1][0] is None or item <= future_l[-1][1][0])
		):
			past_l.append(future_l.pop())
			left_candidates.append(past_l[-1])
		while past_r and past_r[-1][0][0] >= item:
			future_r.append(past_r.pop())
		while future_r and not (
			past_r and past_r[-1][0][0] <= item <= past_r[-1][1][0]
		):
			past_r.append(future_r.pop())
		right_candidates = [past_r[-1]]
		while (
			future_r
			and future_r[-1][0][0] <= item
			and (future_r[-1][1][0] is None or item <= future_r[-1][1][0])
		):
			past_r.append(future_r.pop())
			right_candidates.append(past_r[-1])
		oper = self._oper
		while left_candidates and right_candidates:
			if intersect2(left_candidates[-1][:2], right_candidates[-1][:2]):
				if oper(left_candidates[-1][2], right_candidates[-1][2]):
					return True
			if left_candidates[-1][0] < right_candidates[-1][0]:
				right_candidates.pop()
			else:
				left_candidates.pop()
		return False

	def _last(self):
		"""Get the last turn on which the predicate held true"""
		past_l = self._past_l
		future_l = self._future_l
		while future_l:
			past_l.append(future_l.pop())
		past_r = self._past_r
		future_r = self._future_r
		while future_r:
			past_r.append(future_r)
		oper = self._oper
		while past_l and past_r:
			l_from, l_to, l_v = past_l[-1]
			r_from, r_to, r_v = past_r[-1]
			inter = intersect2((l_from, l_to), (r_from, r_to))
			if not inter:
				if l_from < r_from:
					future_r.append(past_r.pop())
				else:
					future_l.append(past_l.pop())
				continue
			if oper(l_v, r_v):
				if inter[1] == (None, None):
					return self._end_of_time - 1
				return inter[1][0] - (0 if inter[1][1] else 1)

	def _first(self):
		"""Get the first turn on which the predicate held true"""
		oper = self._oper
		for time_from, time_to, l_v, r_v in _yield_intersections(
			chain(iter(self._past_l), reversed(self._future_l)),
			chain(iter(self._past_r), reversed(self._future_r)),
			until=(self._end_of_time, 0),
		):
			if oper(l_v, r_v):
				return time_from[0]


class CombinedQueryResult(QueryResult):
	def __init__(self, left: QueryResult, right: QueryResult, oper):
		self._left = left
		self._right = right
		self._oper = oper

	def _genset(self):
		if not hasattr(self, "_set"):
			self._set = self._oper(set(self._left), set(self._right))

	def __iter__(self):
		self._genset()
		return iter(self._set)

	def __len__(self):
		self._genset()
		return len(self._set)

	def __contains__(self, item):
		if hasattr(self, "_set"):
			return item in self._set
		return self._oper(item in self._left, item in self._right)


class Query(object):
	oper: Callable[[Any, Any], Any] = lambda x, y: NotImplemented

	def __new__(cls, engine, leftside, rightside=None, **kwargs):
		if rightside is None:
			if not isinstance(leftside, cls):
				raise TypeError("You can't make a query with only one side")
			me = leftside
		else:
			me = super().__new__(cls)
			me.leftside = leftside
			me.rightside = rightside
		me.engine = engine
		return me

	def _iter_times(self):
		raise NotImplementedError

	def _iter_ticks(self, turn):
		raise NotImplementedError

	def _iter_btts(self):
		raise NotImplementedError

	def __eq__(self, other):
		return EqQuery(self.engine, self, other)

	def __gt__(self, other):
		return GtQuery(self.engine, self, other)

	def __ge__(self, other):
		return GeQuery(self.engine, self, other)

	def __lt__(self, other):
		return LtQuery(self.engine, self, other)

	def __le__(self, other):
		return LeQuery(self.engine, self, other)

	def __ne__(self, other):
		return NeQuery(self.engine, self, other)


class ComparisonQuery(Query):
	oper: Callable[[Any, Any], bool] = lambda x, y: NotImplemented

	def _iter_times(self):
		return slow_iter_turns_eval_cmp(self, self.oper, engine=self.engine)

	def _iter_btts(self):
		return slow_iter_btts_eval_cmp(self, self.oper, engine=self.engine)

	def __and__(self, other):
		return IntersectionQuery(self.engine, self, other)

	def __or__(self, other):
		return UnionQuery(self.engine, self, other)

	def __sub__(self, other):
		return MinusQuery(self.engine, self, other)


class EqQuery(ComparisonQuery):
	oper = eq


class NeQuery(ComparisonQuery):
	oper = ne


class GtQuery(ComparisonQuery):
	oper = gt


class LtQuery(ComparisonQuery):
	oper = lt


class GeQuery(ComparisonQuery):
	oper = ge


class LeQuery(ComparisonQuery):
	oper = le


class CompoundQuery(Query):
	oper: Callable[[Any, Any], set] = lambda x, y: NotImplemented


class UnionQuery(CompoundQuery):
	oper = operator.or_


class IntersectionQuery(CompoundQuery):
	oper = operator.and_


class MinusQuery(CompoundQuery):
	oper = operator.sub


comparisons = {
	"eq": EqQuery,
	"ne": NeQuery,
	"gt": GtQuery,
	"lt": LtQuery,
	"ge": GeQuery,
	"le": LeQuery,
}


class StatusAlias(EntityStatAccessor):
	def __eq__(self, other):
		return EqQuery(self.engine, self, other)

	def __ne__(self, other):
		return NeQuery(self.engine, self, other)

	def __gt__(self, other):
		return GtQuery(self.engine, self, other)

	def __lt__(self, other):
		return LtQuery(self.engine, self, other)

	def __ge__(self, other):
		return GeQuery(self.engine, self, other)

	def __le__(self, other):
		return LeQuery(self.engine, self, other)


def _mungeside(side):
	if isinstance(side, Query):
		return side._iter_times
	elif isinstance(side, StatusAlias):
		return EntityStatAccessor(
			side.entity,
			side.stat,
			side.engine,
			side.branch,
			side.turn,
			side.tick,
			side.current,
			side.mungers,
		)
	elif isinstance(side, EntityStatAccessor):
		return side
	else:
		return lambda: side


def slow_iter_turns_eval_cmp(qry, oper, start_branch=None, engine=None):
	"""Iterate over all turns on which a comparison holds.

	This is expensive. It evaluates the query for every turn in history.

	"""
	leftside = _mungeside(qry.leftside)
	rightside = _mungeside(qry.rightside)
	engine = engine or leftside.engine or rightside.engine

	for branch, fork_turn, fork_tick in engine._iter_parent_btt(
		start_branch or engine.branch
	):
		if branch is None:
			return
		parent = engine.branch_parent(branch)
		turn_start, tick_start = engine._branch_start(branch)
		turn_end, tick_end = engine._branch_end(branch)
		for turn in range(turn_start, fork_turn + 1):
			if oper(leftside(branch, turn), rightside(branch, turn)):
				yield branch, turn


def slow_iter_btts_eval_cmp(qry, oper, start_branch=None, engine=None):
	leftside = _mungeside(qry.leftside)
	rightside = _mungeside(qry.rightside)
	engine = engine or leftside.engine or rightside.engine
	assert engine is not None

	for branch, fork_turn, fork_tick in engine._iter_parent_btt(
		start_branch or engine.branch
	):
		if branch is None:
			return
		turn_start = engine._branch_start(branch)[0]
		for turn in range(turn_start, fork_turn + 1):
			if turn == fork_turn:
				local_turn_end = fork_tick
			else:
				local_turn_end = engine._turn_end_plan[branch, turn]
			for tick in range(0, local_turn_end + 1):
				try:
					val = oper(
						leftside(branch, turn, tick),
						rightside(branch, turn, tick),
					)
				except KeyError:
					continue
				if val:
					yield branch, turn, tick
