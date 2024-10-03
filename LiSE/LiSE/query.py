# This file is part of LiSE, a framework for life simulation games.
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

Sometimes you want to know when some stat of a LiSE entity had a particular
value. To find out, construct a historical query and pass it to
``Engine.turns_when``, like this::

	physical = engine.character['physical']
	that = physical.thing['that']
	hist_loc = that.historical('location')
	print(list(engine.turns_when(hist_loc == 'there')))


You'll get the turns when ``that`` was ``there``.

Other comparison operators like ``>`` and ``<`` work as well.

"""

import operator
from collections.abc import MutableMapping, Sequence, Set
from itertools import chain
from operator import gt, lt, eq, ne, le, ge
from functools import partialmethod
from time import monotonic
from queue import Queue
from threading import Thread
from typing import Any, List, Callable, Tuple

from sqlalchemy import select, and_, Table
from sqlalchemy.sql.functions import func
import msgpack

from .alchemy import meta, gather_sql
from .allegedb import query
from .exc import IntegrityError, OperationalError
from .util import EntityStatAccessor
import LiSE


NONE = msgpack.packb(None)


def windows_union(windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
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
	windows: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
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
	graph: bytes, stat: bytes, branches: List[str], mid_turn: bool
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
	graph: bytes, node: bytes, stat: bytes, branches: List[str], mid_turn: bool
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
	graph: bytes, thing: bytes, branches: List[str], mid_turn: bool
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
	branches: List[str],
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
	entity, stat, branches: List[str], pack: callable, mid_turn: bool
):
	from .character import AbstractCharacter
	from .node import Place
	from .node import Thing
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
		parent, turn_start, tick_start, turn_end, tick_end = engine._branches[
			branch
		]
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
		parent, turn_start, tick_start, turn_end, tick_end = engine._branches[
			branch
		]
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


class ConnectionHolder(query.ConnectionHolder):
	def gather(self, meta):
		return gather_sql(meta)

	def initdb(self):
		"""Set up the database schema, both for allegedb and the special
		extensions for LiSE

		"""
		super().initdb()
		init_table = self.init_table
		for table in (
			"universals",
			"rules",
			"rulebooks",
			"things",
			"character_rulebook",
			"unit_rulebook",
			"character_thing_rulebook",
			"character_place_rulebook",
			"character_portal_rulebook",
			"node_rulebook",
			"portal_rulebook",
			"units",
			"character_rules_handled",
			"unit_rules_handled",
			"character_thing_rules_handled",
			"character_place_rules_handled",
			"character_portal_rules_handled",
			"node_rules_handled",
			"portal_rules_handled",
			"rule_triggers",
			"rule_prereqs",
			"rule_actions",
			"rule_neighborhood",
			"turns_completed",
		):
			try:
				init_table(table)
			except OperationalError:
				pass
			except Exception as ex:
				return ex
		schemaver_b = b"\xb4_lise_schema_version"
		ver = self.call_one("global_get", schemaver_b).fetchone()
		if ver is None:
			self.call_one("global_insert", schemaver_b, b"\x00")
		elif ver[0] != b"\x00":
			return ValueError(
				f"Unsupported database schema version: {ver}", ver
			)


class QueryEngine(query.QueryEngine):
	exist_edge_t = 0
	path = LiSE.__path__[0]
	IntegrityError = IntegrityError
	OperationalError = OperationalError
	holder_cls = ConnectionHolder
	tables = (
		"global",
		"branches",
		"turns",
		"graphs",
		"keyframes",
		"graph_val",
		"nodes",
		"node_val",
		"edges",
		"edge_val",
		"plans",
		"plan_ticks",
		"universals",
		"rules",
		"rulebooks",
		"rule_triggers",
		"rule_prereqs",
		"rule_actions",
		"character_rulebook",
		"unit_rulebook",
		"character_thing_rulebook",
		"character_place_rulebook",
		"character_portal_rulebook",
		"node_rules_handled",
		"portal_rules_handled",
		"things",
		"node_rulebook",
		"portal_rulebook",
		"units",
		"character_rules_handled",
		"unit_rules_handled",
		"character_thing_rules_handled",
		"character_place_rules_handled",
		"character_portal_rules_handled",
		"turns_completed",
	)
	kf_interval_override: callable

	def __init__(self, dbstring, connect_args, pack=None, unpack=None):
		super().__init__(
			dbstring, connect_args, pack, unpack, gather=gather_sql
		)

		self._records = 0
		self.keyframe_interval = None
		self.snap_keyframe = lambda: None
		self._char_rules_handled = []
		self._unit_rules_handled = []
		self._char_thing_rules_handled = []
		self._char_place_rules_handled = []
		self._char_portal_rules_handled = []
		self._node_rules_handled = []
		self._portal_rules_handled = []
		self._unitness = []
		self._location = []

	def _increc(self):
		self._records += 1
		override = self.kf_interval_override()
		if override is True:
			return
		if override is False or (
			self.keyframe_interval is not None
			and self._records % self.keyframe_interval == 0
		):
			self.snap_keyframe()

	def graph_val_set(self, graph, key, branch, turn, tick, value):
		super().graph_val_set(graph, key, branch, turn, tick, value)
		self._increc()

	def node_val_set(self, graph, node, key, branch, turn, tick, value):
		super().node_val_set(graph, node, key, branch, turn, tick, value)
		self._increc()

	def edge_val_set(
		self, graph, orig, dest, idx, key, branch, turn, tick, value
	):
		super().edge_val_set(
			graph, orig, dest, idx, key, branch, turn, tick, value
		)
		self._increc()

	def flush(self):
		super().flush()
		put = self._inq.put
		if self._unitness:
			put(
				(
					"silent",
					"many",
					"del_units_after",
					[
						(character, graph, node, branch, turn, turn, tick)
						for (
							character,
							graph,
							node,
							branch,
							turn,
							tick,
							_,
						) in self._unitness
					],
				)
			)
			put(("silent", "many", "units_insert", self._unitness))
			self._unitness = []
		if self._location:
			put(
				(
					"silent",
					"many",
					"del_things_after",
					[
						(character, thing, branch, turn, turn, tick)
						for (
							character,
							thing,
							branch,
							turn,
							tick,
							_,
						) in self._location
					],
				)
			)
			put(("silent", "many", "things_insert", self._location))
			self._location = []
		for attr, cmd in [
			("_char_rules_handled", "character_rules_handled_insert"),
			("_unit_rules_handled", "unit_rules_handled_insert"),
			(
				"_char_thing_rules_handled",
				"character_thing_rules_handled_insert",
			),
			(
				"_char_place_rules_handled",
				"character_place_rules_handled_insert",
			),
			(
				"_char_portal_rules_handled",
				"character_portal_rules_handled_insert",
			),
			("_node_rules_handled", "_node_rules_handled_insert"),
			("_portal_rules_handled", "portal_rules_handled_insert"),
		]:
			if getattr(self, attr):
				put(("silent", "many", cmd, getattr(self, attr)))
			setattr(self, attr, [])
		assert self.echo("flushed") == "flushed"

	def universals_dump(self):
		unpack = self.unpack
		for key, branch, turn, tick, value in self.call_one("universals_dump"):
			yield unpack(key), branch, turn, tick, unpack(value)

	def rulebooks_dump(self):
		unpack = self.unpack
		for rulebook, branch, turn, tick, rules, prio in self.call_one(
			"rulebooks_dump"
		):
			yield unpack(rulebook), branch, turn, tick, (unpack(rules), prio)

	def _rule_dump(self, typ):
		unpack = self.unpack
		for rule, branch, turn, tick, lst in self.call_one(
			"rule_{}_dump".format(typ)
		):
			yield rule, branch, turn, tick, unpack(lst)

	def rule_triggers_dump(self):
		return self._rule_dump("triggers")

	def rule_prereqs_dump(self):
		return self._rule_dump("prereqs")

	def rule_actions_dump(self):
		return self._rule_dump("actions")

	def rule_neighborhood_dump(self):
		return self._rule_dump("neighborhood")

	characters = characters_dump = query.QueryEngine.graphs_dump

	def node_rulebook_dump(self):
		unpack = self.unpack
		for character, node, branch, turn, tick, rulebook in self.call_one(
			"node_rulebook_dump"
		):
			yield (
				unpack(character),
				unpack(node),
				branch,
				turn,
				tick,
				unpack(rulebook),
			)

	def portal_rulebook_dump(self):
		unpack = self.unpack
		for (
			character,
			orig,
			dest,
			branch,
			turn,
			tick,
			rulebook,
		) in self.call_one("portal_rulebook_dump"):
			yield (
				unpack(character),
				unpack(orig),
				unpack(dest),
				branch,
				turn,
				tick,
				unpack(rulebook),
			)

	def _charactery_rulebook_dump(self, qry):
		unpack = self.unpack
		for character, branch, turn, tick, rulebook in self.call_one(
			qry + "_rulebook_dump"
		):
			yield unpack(character), branch, turn, tick, unpack(rulebook)

	character_rulebook_dump = partialmethod(
		_charactery_rulebook_dump, "character"
	)
	unit_rulebook_dump = partialmethod(_charactery_rulebook_dump, "unit")
	character_thing_rulebook_dump = partialmethod(
		_charactery_rulebook_dump, "character_thing"
	)
	character_place_rulebook_dump = partialmethod(
		_charactery_rulebook_dump, "character_place"
	)
	character_portal_rulebook_dump = partialmethod(
		_charactery_rulebook_dump, "character_portal"
	)

	def character_rules_handled_dump(self):
		unpack = self.unpack
		for character, rulebook, rule, branch, turn, tick in self.call_one(
			"character_rules_handled_dump"
		):
			yield unpack(character), unpack(rulebook), rule, branch, turn, tick

	def character_rules_changes_dump(self):
		unpack = self.unpack
		for (
			character,
			rulebook,
			rule,
			branch,
			turn,
			tick,
			handled_branch,
			handled_turn,
		) in self.call_one("character_rules_changes_dump"):
			yield (
				unpack(character),
				unpack(rulebook),
				rule,
				branch,
				turn,
				tick,
				handled_branch,
				handled_turn,
			)

	def unit_rules_handled_dump(self):
		unpack = self.unpack
		for (
			character,
			graph,
			unit,
			rulebook,
			rule,
			branch,
			turn,
			tick,
		) in self.call_one("unit_rules_handled_dump"):
			yield (
				unpack(character),
				unpack(graph),
				unpack(unit),
				unpack(rulebook),
				rule,
				branch,
				turn,
				tick,
			)

	def unit_rules_changes_dump(self):
		jl = self.unpack
		for (
			character,
			rulebook,
			rule,
			graph,
			unit,
			branch,
			turn,
			tick,
			handled_branch,
			handled_turn,
		) in self.call_one("unit_rules_changes_dump"):
			yield (
				jl(character),
				jl(rulebook),
				rule,
				jl(graph),
				jl(unit),
				branch,
				turn,
				tick,
				handled_branch,
				handled_turn,
			)

	def character_thing_rules_handled_dump(self):
		unpack = self.unpack
		for (
			character,
			thing,
			rulebook,
			rule,
			branch,
			turn,
			tick,
		) in self.call_one("character_thing_rules_handled_dump"):
			yield (
				unpack(character),
				unpack(thing),
				unpack(rulebook),
				rule,
				branch,
				turn,
				tick,
			)

	def character_thing_rules_changes_dump(self):
		jl = self.unpack
		for (
			character,
			thing,
			rulebook,
			rule,
			branch,
			turn,
			tick,
			handled_branch,
			handled_turn,
		) in self.call_one("character_thing_rules_changes_dump"):
			yield (
				jl(character),
				jl(thing),
				jl(rulebook),
				rule,
				branch,
				turn,
				tick,
				handled_branch,
				handled_turn,
			)

	def character_place_rules_handled_dump(self):
		unpack = self.unpack
		for (
			character,
			place,
			rulebook,
			rule,
			branch,
			turn,
			tick,
		) in self.call_one("character_place_rules_handled_dump"):
			yield (
				unpack(character),
				unpack(place),
				unpack(rulebook),
				rule,
				branch,
				turn,
				tick,
			)

	def character_place_rules_changes_dump(self):
		jl = self.unpack
		for (
			character,
			rulebook,
			rule,
			place,
			branch,
			turn,
			tick,
			handled_branch,
			handled_turn,
		) in self.call_one("character_place_rules_changes_dump"):
			yield (
				jl(character),
				jl(rulebook),
				rule,
				jl(place),
				branch,
				turn,
				tick,
				handled_branch,
				handled_turn,
			)

	def character_portal_rules_handled_dump(self):
		unpack = self.unpack
		for (
			character,
			rulebook,
			rule,
			orig,
			dest,
			branch,
			turn,
			tick,
		) in self.call_one("character_portal_rules_handled_dump"):
			yield (
				unpack(character),
				unpack(rulebook),
				unpack(orig),
				unpack(dest),
				rule,
				branch,
				turn,
				tick,
			)

	def character_portal_rules_changes_dump(self):
		jl = self.unpack
		for (
			character,
			rulebook,
			rule,
			orig,
			dest,
			branch,
			turn,
			tick,
			handled_branch,
			handled_turn,
		) in self.call_one("character_portal_rules_changes_dump"):
			yield (
				jl(character),
				jl(rulebook),
				rule,
				jl(orig),
				jl(dest),
				branch,
				turn,
				tick,
				handled_branch,
				handled_turn,
			)

	def node_rules_handled_dump(self):
		for (
			character,
			node,
			rulebook,
			rule,
			branch,
			turn,
			tick,
		) in self.call_one("node_rules_handled_dump"):
			yield (
				self.unpack(character),
				self.unpack(node),
				self.unpack(rulebook),
				rule,
				branch,
				turn,
				tick,
			)

	def node_rules_changes_dump(self):
		jl = self.unpack
		for (
			character,
			node,
			rulebook,
			rule,
			branch,
			turn,
			tick,
			handled_branch,
			handled_turn,
		) in self.call_one("node_rules_changes_dump"):
			yield (
				jl(character),
				jl(node),
				jl(rulebook),
				rule,
				branch,
				turn,
				tick,
				handled_branch,
				handled_turn,
			)

	def portal_rules_handled_dump(self):
		unpack = self.unpack
		for (
			character,
			orig,
			dest,
			rulebook,
			rule,
			branch,
			turn,
			tick,
		) in self.call_one("portal_rules_handled_dump"):
			yield (
				unpack(character),
				unpack(orig),
				unpack(dest),
				unpack(rulebook),
				rule,
				branch,
				turn,
				tick,
			)

	def portal_rules_changes_dump(self):
		jl = self.unpack
		for (
			character,
			orig,
			dest,
			rulebook,
			rule,
			branch,
			turn,
			tick,
			handled_branch,
			handled_turn,
		) in self.call_one("portal_rules_changes_dump"):
			yield (
				jl(character),
				jl(orig),
				jl(dest),
				jl(rulebook),
				rule,
				branch,
				turn,
				tick,
				handled_branch,
				handled_turn,
			)

	def senses_dump(self):
		unpack = self.unpack
		for character, sense, branch, turn, tick, function in self.call_one(
			"senses_dump"
		):
			yield unpack(character), sense, branch, turn, tick, function

	def things_dump(self):
		unpack = self.unpack
		for character, thing, branch, turn, tick, location in self.call_one(
			"things_dump"
		):
			yield (
				unpack(character),
				unpack(thing),
				branch,
				turn,
				tick,
				unpack(location),
			)

	def load_things(
		self,
		character,
		branch,
		turn_from,
		tick_from,
		turn_to=None,
		tick_to=None,
	):
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			if tick_to is not None:
				raise ValueError("Need both or neither of turn_to, tick_to")
			for thing, turn, tick, location in self.call_one(
				"load_things_tick_to_end",
				pack(character),
				branch,
				turn_from,
				turn_from,
				tick_from,
			):
				yield (
					character,
					unpack(thing),
					branch,
					turn,
					tick,
					unpack(location),
				)
		else:
			if tick_to is None:
				raise ValueError("Need both or neither of turn_to, tick_to")
			for thing, turn, tick, location in self.call_one(
				"load_things_tick_to_tick",
				pack(character),
				branch,
				turn_from,
				turn_from,
				tick_from,
				turn_to,
				turn_to,
				tick_to,
			):
				yield (
					character,
					unpack(thing),
					branch,
					turn,
					tick,
					unpack(location),
				)

	def units_dump(self):
		unpack = self.unpack
		for (
			character_graph,
			unit_graph,
			unit_node,
			branch,
			turn,
			tick,
			is_av,
		) in self.call_one("units_dump"):
			yield (
				unpack(character_graph),
				unpack(unit_graph),
				unpack(unit_node),
				branch,
				turn,
				tick,
				is_av,
			)

	def universal_set(self, key, branch, turn, tick, val):
		key, val = map(self.pack, (key, val))
		self.call_one("universals_insert", key, branch, turn, tick, val)
		self._increc()

	def universal_del(self, key, branch, turn, tick):
		key = self.pack(key)
		self.call_one("universals_insert", key, branch, turn, tick, NONE)
		self._increc()

	def comparison(
		self,
		entity0,
		stat0,
		entity1,
		stat1=None,
		oper="eq",
		windows: list = None,
	):
		if windows is None:
			windows = []
		stat1 = stat1 or stat0
		return comparisons[oper](
			leftside=entity0.status(stat0),
			rightside=entity1.status(stat1),
			windows=windows,
		)

	def count_all_table(self, tbl):
		return self.call_one("{}_count".format(tbl)).fetchone()[0]

	def rules_dump(self):
		for (name,) in self.call_one("rules_dump"):
			yield name

	def _set_rule_something(self, what, rule, branch, turn, tick, flist):
		flist = self.pack(flist)
		self.call_one(
			"rule_{}_insert".format(what), rule, branch, turn, tick, flist
		)
		self._increc()

	set_rule_triggers = partialmethod(_set_rule_something, "triggers")
	set_rule_prereqs = partialmethod(_set_rule_something, "prereqs")
	set_rule_actions = partialmethod(_set_rule_something, "actions")
	set_rule_neighborhood = partialmethod(_set_rule_something, "neighborhood")

	def set_rule(
		self,
		rule,
		branch,
		turn,
		tick,
		triggers=None,
		prereqs=None,
		actions=None,
		neighborhood=None,
	):
		try:
			self.call_one("rules_insert", rule)
			self._increc()
		except IntegrityError:
			pass
		self.set_rule_triggers(rule, branch, turn, tick, triggers or [])
		self.set_rule_prereqs(rule, branch, turn, tick, prereqs or [])
		self.set_rule_actions(rule, branch, turn, tick, actions or [])
		self.set_rule_neighborhood(rule, branch, turn, tick, neighborhood)

	def set_rulebook(self, name, branch, turn, tick, rules=None, prio=0.0):
		name, rules = map(self.pack, (name, rules or []))
		self.call_one(
			"rulebooks_insert", name, branch, turn, tick, rules, float(prio)
		)
		self._increc()

	def _set_rulebook_on_character(self, rbtyp, char, branch, turn, tick, rb):
		char, rb = map(self.pack, (char, rb))
		self.call_one(rbtyp + "_rulebook_insert", char, branch, turn, tick, rb)
		self._increc()

	set_character_rulebook = partialmethod(
		_set_rulebook_on_character, "character"
	)
	set_unit_rulebook = partialmethod(_set_rulebook_on_character, "unit")
	set_character_thing_rulebook = partialmethod(
		_set_rulebook_on_character, "character_thing"
	)
	set_character_place_rulebook = partialmethod(
		_set_rulebook_on_character, "character_place"
	)
	set_character_portal_rulebook = partialmethod(
		_set_rulebook_on_character, "character_portal"
	)

	def rulebooks(self):
		for book in self.call_one("rulebooks"):
			yield self.unpack(book)

	def exist_node(self, character, node, branch, turn, tick, extant):
		super().exist_node(character, node, branch, turn, tick, extant)
		self._increc()

	def exist_edge(
		self, character, orig, dest, idx, branch, turn, tick, extant=None
	):
		start = monotonic()
		if extant is None:
			branch, turn, tick, extant = idx, branch, turn, tick
			idx = 0
		super().exist_edge(
			character, orig, dest, idx, branch, turn, tick, extant
		)
		QueryEngine.exist_edge_t += monotonic() - start
		self._increc()

	def set_node_rulebook(self, character, node, branch, turn, tick, rulebook):
		(character, node, rulebook) = map(
			self.pack, (character, node, rulebook)
		)
		self.call_one(
			"node_rulebook_insert",
			character,
			node,
			branch,
			turn,
			tick,
			rulebook,
		)
		self._increc()

	def set_portal_rulebook(
		self, character, orig, dest, branch, turn, tick, rulebook
	):
		(character, orig, dest, rulebook) = map(
			self.pack, (character, orig, dest, rulebook)
		)
		self.call_one(
			"portal_rulebook_insert",
			character,
			orig,
			dest,
			branch,
			turn,
			tick,
			rulebook,
		)
		self._increc()

	def handled_character_rule(
		self, character, rulebook, rule, branch, turn, tick
	):
		(character, rulebook) = map(self.pack, (character, rulebook))
		self._char_rules_handled.append(
			(character, rulebook, rule, branch, turn, tick)
		)
		self._increc()

	def _flush_char_rules_handled(self):
		if not self._char_rules_handled:
			return
		self.call_many(
			"character_rules_handled_insert", self._char_rules_handled
		)
		self._char_rules_handled = []

	def handled_unit_rule(
		self, character, rulebook, rule, graph, unit, branch, turn, tick
	):
		character, graph, unit, rulebook = map(
			self.pack, (character, graph, unit, rulebook)
		)
		self._unit_rules_handled.append(
			(character, graph, unit, rulebook, rule, branch, turn, tick)
		)
		self._increc()

	def _flush_unit_rules_handled(self):
		if not self._unit_rules_handled:
			return
		self.call_many("unit_rules_handled_insert", self._unit_rules_handled)
		self._unit_rules_handled = []

	def handled_character_thing_rule(
		self, character, rulebook, rule, thing, branch, turn, tick
	):
		character, thing, rulebook = map(
			self.pack, (character, thing, rulebook)
		)
		self._char_thing_rules_handled.append(
			(character, thing, rulebook, rule, branch, turn, tick)
		)
		self._increc()

	def _flush_char_thing_rules_handled(self):
		if not self._char_thing_rules_handled:
			return
		self.call_many(
			"character_thing_rules_handled_insert",
			self._char_thing_rules_handled,
		)
		self._char_thing_rules_handled = []

	def handled_character_place_rule(
		self, character, rulebook, rule, place, branch, turn, tick
	):
		character, rulebook, place = map(
			self.pack, (character, rulebook, place)
		)
		self._char_place_rules_handled.append(
			(character, place, rulebook, rule, branch, turn, tick)
		)
		self._increc()

	def _flush_char_place_rules_handled(self):
		if not self._char_place_rules_handled:
			return
		self.call_many(
			"character_place_rules_handled_insert",
			self._char_place_rules_handled,
		)
		self._char_place_rules_handled = []

	def handled_character_portal_rule(
		self, character, rulebook, rule, orig, dest, branch, turn, tick
	):
		character, rulebook, orig, dest = map(
			self.pack, (character, rulebook, orig, dest)
		)
		self._char_portal_rules_handled.append(
			(character, orig, dest, rulebook, rule, branch, turn, tick)
		)
		self._increc()

	def _flush_char_portal_rules_handled(self):
		if not self._char_portal_rules_handled:
			return
		self.call_many(
			"character_portal_rules_handled_insert",
			self._char_portal_rules_handled,
		)
		self._char_portal_rules_handled = []

	def handled_node_rule(
		self, character, node, rulebook, rule, branch, turn, tick
	):
		(character, node, rulebook) = map(
			self.pack, (character, node, rulebook)
		)
		self._node_rules_handled.append(
			(character, node, rulebook, rule, branch, turn, tick)
		)
		self._increc()

	def _flush_node_rules_handled(self):
		if not self._node_rules_handled:
			return
		self.call_many("node_rules_handled_insert", self._node_rules_handled)
		self._node_rules_handled = []

	def handled_portal_rule(
		self, character, orig, dest, rulebook, rule, branch, turn, tick
	):
		(character, orig, dest, rulebook) = map(
			self.pack, (character, orig, dest, rulebook)
		)
		self._portal_rules_handled.append(
			(character, orig, dest, rulebook, rule, branch, turn, tick)
		)
		self._increc()

	def _flush_portal_rules_handled(self):
		if not self._portal_rules_handled:
			return
		self.call_many(
			"portal_rules_handled_insert", self._portal_rules_handled
		)
		self._portal_rules_handled = []

	def get_rulebook_char(self, rulemap, character):
		character = self.pack(character)
		for (book,) in self.call_one(
			"rulebook_get_{}".format(rulemap), character
		):
			return self.unpack(book)
		raise KeyError("No rulebook")

	def set_thing_loc(self, character, thing, branch, turn, tick, loc):
		(character, thing) = map(self.pack, (character, thing))
		loc = self.pack(loc)
		self._location.append((character, thing, branch, turn, tick, loc))
		self._increc()

	def unit_set(self, character, graph, node, branch, turn, tick, isav):
		(character, graph, node) = map(self.pack, (character, graph, node))
		self._unitness.append(
			(character, graph, node, branch, turn, tick, isav)
		)
		self._increc()

	def rulebooks_rules(self):
		for rulebook, rule in self.call_one("rulebooks_rules"):
			yield map(self.unpack, (rulebook, rule))

	def rulebook_get(self, rulebook, idx):
		return self.unpack(
			self.call_one("rulebook_get", self.pack(rulebook), idx).fetchone()[
				0
			]
		)

	def rulebook_set(self, rulebook, branch, turn, tick, rules):
		# what if the rulebook has other values set afterward? wipe them out, right?
		# should that happen in the query engine or elsewhere?
		rulebook, rules = map(self.pack, (rulebook, rules))
		try:
			self.call_one(
				"rulebooks_insert", rulebook, branch, turn, tick, rules
			)
			self._increc()
		except IntegrityError:
			self.call_one(
				"rulebooks_update", rules, rulebook, branch, turn, tick
			)

	def rulebook_del_time(self, branch, turn, tick):
		self.call_one("rulebooks_del_time", branch, turn, tick)

	def branch_descendants(self, branch):
		for child in self.call_one("branch_children", branch):
			yield child
			yield from self.branch_descendants(child)

	def turns_completed_dump(self):
		return self.call_one("turns_completed_dump")

	def complete_turn(self, branch, turn, discard_rules=False):
		try:
			self.call_one("turns_completed_insert", branch, turn)
		except IntegrityError:
			self.call_one("turns_completed_update", turn, branch)
		self._increc()
		if discard_rules:
			self._char_rules_handled = []
			self._unit_rules_handled = []
			self._char_thing_rules_handled = []
			self._char_place_rules_handled = []
			self._char_portal_rules_handled = []
			self._node_rules_handled = []
			self._portal_rules_handled = []


class QueryEngineProxy:
	def __init__(
		self, dbstring, connect_args, alchemy, pack=None, unpack=None
	):
		self._inq = Queue()
		self._outq = Queue()
		self._dbstring = dbstring
		self._connect_args = connect_args
		self._alchemy = alchemy
		self._pack = pack
		self._unpack = unpack
		self.globl = self.GlobalsProxy(self._inq, self._outq)
		self._thread = Thread(target=self._subthread, daemon=True)
		self._thread.start()

	def _subthread(self):
		real = QueryEngine(
			self._dbstring,
			self._connect_args,
			self._alchemy,
			pack=self._pack,
			unpack=self._unpack,
		)
		while True:
			func, args, kwargs = self._inq.get()
			if func == "get_global":
				if args not in real.globl:
					output = KeyError()
				else:
					output = real.globl[args]
			elif func == "set_global":
				k, v = args
				real.globl[k] = v
				continue
			elif func == "list_globals":
				output = list(real.globl)
			elif func == "len_globals":
				output = len(real.globl)
			elif func == "del_global":
				if args in real.globl:
					del real.globl[args]
					output = None
				else:
					output = KeyError()
			else:
				try:
					output = getattr(real, func)(*args, **kwargs)
				except Exception as ex:
					output = ex
			if hasattr(output, "__next__"):
				output = list(output)
			self._outq.put(output)
			if func == "close":
				return

	class Caller:
		def __init__(self, func, inq, outq):
			self._func = func
			self._inq = inq
			self._outq = outq

		def __call__(self, *args, **kwargs):
			self._inq.put((self._func, args, kwargs))
			ret = self._outq.get()
			if isinstance(ret, Exception):
				raise ret
			return ret

	class GlobalsProxy(MutableMapping):
		def __init__(self, inq, outq):
			self._inq = inq
			self._outq = outq

		def __iter__(self):
			self._inq.put(("list_globals", None, None))
			return iter(self._outq.get())

		def __len__(self):
			self._inq.put(("len_globals", None, None))
			return self._outq.get()

		def __getitem__(self, item):
			self._inq.put(("get_global", item, None))
			ret = self._outq.get()
			if isinstance(ret, Exception):
				raise ret
			return ret

		def __setitem__(self, key, value):
			self._inq.put(("set_global", (key, value), None))

		def __delitem__(self, key):
			self._inq.put(("del_global", key, None))
			ret = self._outq.get()
			if isinstance(ret, Exception):
				raise ret

	def __getattr__(self, item):
		if hasattr(QueryEngine, item) and callable(getattr(QueryEngine, item)):
			return self.Caller(item, self._inq, self._outq)
