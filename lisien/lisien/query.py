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

import operator
from collections import defaultdict
from collections.abc import Sequence, Set
from abc import abstractmethod
from collections.abc import MutableMapping, Sequence, Set
from operator import gt, lt, eq, ne, le, ge
from functools import partialmethod
from itertools import chain
from time import monotonic
from queue import Queue
from threading import Thread, Lock, RLock
from typing import Any, List, Callable, Tuple, Iterator, Optional

import pyarrow as pa
from pyarrow.lib import ArrowInvalid
import pyarrow.compute as pc
from parquetdb import ParquetDB
from sqlalchemy import select, and_, Table
from sqlalchemy.sql.functions import func
import msgpack

import lisien

from .alchemy import meta, gather_sql
from .allegedb import (
	query,
	Key,
	EdgeValRowType,
	EdgeRowType,
	NodeValRowType,
	NodeRowType,
	GraphValRowType,
)
from .allegedb.query import AbstractQueryEngine
from .exc import OperationalError
from .util import EntityStatAccessor

IntegrityError = query.IntegrityError


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


class ParquetDBHolder:
	schema = {
		"branches": [
			("branch", pa.string()),
			("parent", pa.string()),
			("parent_turn", pa.uint64()),
			("parent_tick", pa.uint64()),
			("end_turn", pa.uint64()),
			("end_tick", pa.uint64()),
		],
		"global": [("key", pa.binary()), ("value", pa.binary())],
		"turns": [
			("branch", pa.string()),
			("turn", pa.uint64()),
			("end_tick", pa.uint64()),
			("plan_end_tick", pa.uint64()),
		],
		"graphs": [
			("graph", pa.binary()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("type", pa.string()),
		],
		"keyframes": [
			("graph", pa.binary()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("nodes", pa.large_binary()),
			("edges", pa.large_binary()),
			("graph_val", pa.large_binary()),
		],
		"graph_val": [
			("graph", pa.binary()),
			("key", pa.binary()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("value", pa.binary()),
		],
		"nodes": [
			("graph", pa.binary()),
			("node", pa.binary()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("extant", pa.bool_()),
		],
		"node_val": [
			("graph", pa.binary()),
			("node", pa.binary()),
			("key", pa.binary()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("value", pa.binary()),
		],
		"edges": [
			("graph", pa.binary()),
			("orig", pa.binary()),
			("dest", pa.binary()),
			("idx", pa.uint64()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("extant", pa.bool_()),
		],
		"edge_val": [
			("graph", pa.binary()),
			("orig", pa.binary()),
			("dest", pa.binary()),
			("idx", pa.uint64()),
			("key", pa.binary()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("value", pa.binary()),
		],
		"plans": [
			("plan_id", pa.uint64()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
		],
		"plan_ticks": [
			("plan_id", pa.uint64()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
		],
		"universals": [
			("key", pa.binary()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("value", pa.binary()),
		],
		"rules": [("rule", pa.string())],
		"rulebooks": [
			("rulebook", pa.binary()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("rules", pa.binary()),
			("priority", pa.float64()),
		],
		"rule_triggers": [
			("rule", pa.string()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("triggers", pa.binary()),
		],
		"rule_neighborhood": [
			("rule", pa.string()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("neighborhood", pa.binary()),
		],
		"rule_prereqs": [
			("rule", pa.string()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("prereqs", pa.binary()),
		],
		"rule_actions": [
			("rule", pa.string()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("actions", pa.binary()),
		],
		"character_rulebook": [
			("character", pa.binary()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("rulebook", pa.binary()),
		],
		"unit_rulebook": [
			("character", pa.binary()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("rulebook", pa.binary()),
		],
		"character_thing_rulebook": [
			("character", pa.binary()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("rulebook", pa.binary()),
		],
		"character_place_rulebook": [
			("character", pa.binary()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("rulebook", pa.binary()),
		],
		"character_portal_rulebook": [
			("character", pa.binary()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("rulebook", pa.binary()),
		],
		"character_rules_handled": [
			("character", pa.binary()),
			("rulebook", pa.binary()),
			("rule", pa.string()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
		],
		"unit_rules_handled": [
			("character", pa.binary()),
			("graph", pa.binary()),
			("unit", pa.binary()),
			("rulebook", pa.binary()),
			("rule", pa.string()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
		],
		"character_thing_rules_handled": [
			("character", pa.binary()),
			("thing", pa.binary()),
			("rulebook", pa.binary()),
			("rule", pa.string()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
		],
		"character_place_rules_handled": [
			("character", pa.binary()),
			("place", pa.binary()),
			("rulebook", pa.binary()),
			("rule", pa.string()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
		],
		"character_portal_rules_handled": [
			("character", pa.binary()),
			("orig", pa.binary()),
			("dest", pa.binary()),
			("rulebook", pa.binary()),
			("rule", pa.string()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
		],
		"node_rules_handled": [
			("character", pa.binary()),
			("node", pa.binary()),
			("rulebook", pa.binary()),
			("rule", pa.string()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
		],
		"portal_rules_handled": [
			("character", pa.binary()),
			("orig", pa.binary()),
			("dest", pa.binary()),
			("rulebook", pa.binary()),
			("rule", pa.string()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
		],
		"things": [
			("character", pa.binary()),
			("thing", pa.binary()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("location", pa.binary()),
		],
		"node_rulebook": [
			("character", pa.binary()),
			("node", pa.binary()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("rulebook", pa.binary()),
		],
		"portal_rulebook": [
			("character", pa.binary()),
			("orig", pa.binary()),
			("dest", pa.binary()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("rulebook", pa.binary()),
		],
		"units": [
			("character_graph", pa.binary()),
			("unit_graph", pa.binary()),
			("unit_node", pa.binary()),
			("branch", pa.string()),
			("turn", pa.uint64()),
			("tick", pa.uint64()),
			("is_unit", pa.bool_()),
		],
		"turns_completed": [("branch", pa.string()), ("turn", pa.uint64())],
	}
	initial = {
		"global": [
			{
				"key": b"\xb4_lise_schema_version",
				"value": b"\x00",
			},
			{"key": b"\xabmain_branch", "value": b"\xa5trunk"},
			{"key": b"\xa6branch", "value": b"\xa5trunk"},
			{"key": b"\xa4turn", "value": b"\x00"},
			{"key": b"\xa4tick", "value": b"\x00"},
			{"key": b"\xa8language", "value": b"\xa3eng"},
		],
		"branches": [
			{
				"branch": "trunk",
				"parent": None,
				"parent_turn": 0,
				"parent_tick": 0,
				"end_turn": 0,
				"end_tick": 0,
			}
		],
	}

	def __init__(self, path, inq, outq):
		self._inq = inq
		self._outq = outq
		self._schema = {}
		self._path = path
		self.lock = RLock()
		self.existence_lock = Lock()
		self.existence_lock.acquire()

	def commit(self):
		pass

	def close(self):
		self.existence_lock.release()

	def initdb(self):
		initial = self.initial
		for table, schema in self.schema.items():
			schema = self._schema[table] = pa.schema(schema)
			db = ParquetDB(table, self._path)
			if db.dataset_exists():
				continue
			if table in initial:
				db.create(
					initial[table],
					schema=schema,
				)

	def insert(self, table: str, data: list) -> None:
		ParquetDB(table, self._path).create(data, schema=self._schema[table])

	def truncate_all(self):
		for table in self.schema:
			db = ParquetDB(table, self._path)
			if db.dataset_exists():
				db.drop_dataset()

	def del_units_after(self, many):
		db = ParquetDB("units", self._path)
		ids = []
		for character, graph, node, branch, turn, tick in many:
			for d in db.read(
				filters=[
					pc.field("character_graph") == character,
					pc.field("unit_graph") == graph,
					pc.field("unit_node") == node,
					pc.field("branch") == branch,
					pc.field("turn") >= turn,
				],
				columns=["id", "turn", "tick"],
			).to_pylist():
				if d["turn"] == turn:
					if d["tick"] >= tick:
						ids.append(d["id"])
				else:
					ids.append(d["id"])
		if ids:
			db.delete(ids)

	def del_things_after(self, many):
		db = ParquetDB("things", self._path)
		ids = []
		for character, thing, branch, turn, tick in many:
			for d in db.read(
				"things",
				filters=[
					pc.field("character") == character,
					pc.field("thing") == thing,
					pc.field("branch") == branch,
					pc.field("turn") >= turn,
				],
				columns=["id", "turn", "tick"],
			).to_pylist():
				if d["turn"] == turn:
					if d["tick"] >= tick:
						ids.append(d["id"])
				else:
					ids.append(d["id"])
		if ids:
			db.delete(ids)

	def dump(self, table: str) -> list:
		return ParquetDB(table, self._path).read().to_pylist()

	def rowcount(self, table: str) -> int:
		return ParquetDB(table, self._path).read().rowcount

	def rulebooks(self):
		return set(
			ParquetDB("rulebooks", self._path).read(columns=["rulebook"])[
				"rulebook"
			]
		)

	def graphs(self):
		return set(
			name.as_py()
			for name in ParquetDB("graphs", self._path).read(
				columns=["graph"]
			)["graph"]
		)

	def list_keyframes(self) -> list:
		return (
			ParquetDB("keyframes", self._path)
			.read(
				columns=["graph", "branch", "turn", "tick"],
			)
			.to_pylist()
		)

	def get_keyframe(
		self, graph: bytes, branch: str, turn: int, tick: int
	) -> Optional[Tuple[bytes, bytes, bytes]]:
		rec = ParquetDB("keyframes", self._path).read(
			"keyframes",
			filters=[
				pc.field("graph") == pc.scalar(graph),
				pc.field("branch") == pc.scalar(branch),
				pc.field("turn") == pc.scalar(turn),
				pc.field("tick") == pc.scalar(tick),
			],
			columns=["nodes", "edges", "graph_val"],
		)
		if not rec.num_rows:
			return None
		if rec.num_rows > 1:
			raise ValueError("Ambiguous keyframe, probably corrupt table")
		return (
			rec["nodes"][0].as_py(),
			rec["edges"][0].as_py(),
			rec["graph_val"][0].as_py(),
		)

	class InsertException(ValueError):
		pass

	def insert1(self, table: str, data: dict):
		try:
			return self.insert(table, [data])
		except Exception as ex:
			return ex

	def set_rulebook_on_character(self, rbtyp, char, branch, turn, tick, rb):
		self.insert1(
			f"{rbtyp}_rulebook",
			{
				"character": char,
				"branch": branch,
				"turn": turn,
				"tick": tick,
				"rulebook": rb,
			},
		)

	def graph_exists(self, graph: bytes) -> bool:
		return bool(
			ParquetDB("graphs", self._path)
			.read(
				filters=[pc.field("graph") == pc.scalar(graph)], columns=["id"]
			)
			.num_rows
		)

	def get_global(self, key: bytes) -> bytes:
		try:
			return (
				ParquetDB("global", self._path)
				.read(filters=[pc.field("key") == key])["value"][0]
				.as_py()
			)
		except IndexError as ex:
			raise KeyError(f"No such global: {key}") from ex

	def set_global(self, key: bytes, value: bytes):
		try:
			id_ = self.field_get_id("global", "key", key)
			return ParquetDB("global", self._path).update(
				{"id": id_, "key": key, "value": value}
			)
		except KeyError:
			return ParquetDB("global", self._path).create(
				{"key": key, "value": value}
			)

	def global_keys(self):
		return [
			d["key"]
			for d in ParquetDB("global", self._path)
			.read("global", columns=["key"])
			.to_pylist()
		]

	def field_get_id(self, table, keyfield, value):
		return self.filter_get_id(table, filters=[pc.field(keyfield) == value])

	def filter_get_id(self, table, filters):
		try:
			return (
				ParquetDB(table, self._path)
				.read(filters=filters, columns=["id"])["id"][0]
				.as_py()
			)
		except KeyError:
			return None

	def update_branch(
		self,
		branch: str,
		parent: str,
		parent_turn: int,
		parent_tick: int,
		end_turn: int,
		end_tick: int,
	) -> None:
		id_ = self.field_get_id("branches", "branch", branch)
		if id_ is None:
			raise KeyError(f"No branch: {branch}")
		return ParquetDB("branch", self._path).update(
			{
				"id": id_,
				"parent": parent,
				"parent_turn": parent_turn,
				"parent_tick": parent_tick,
				"end_turn": end_turn,
				"end_tick": end_tick,
			},
		)

	def set_branch(
		self,
		branch: str,
		parent: str,
		parent_turn: int,
		parent_tick: int,
		end_turn: int,
		end_tick: int,
	) -> None:
		try:
			self.update_branch(
				branch, parent, parent_turn, parent_tick, end_turn, end_tick
			)
		except KeyError:
			self.insert1(
				"branches",
				{
					"branch": branch,
					"parent": parent,
					"parent_turn": parent_turn,
					"parent_tick": parent_tick,
					"end_turn": end_turn,
					"end_tick": end_tick,
				},
			)

	def have_branch(self, branch: str) -> bool:
		return bool(
			ParquetDB("branches", self._path)
			.read("branches", filters=[pc.field("branch") == branch])
			.rowcount
		)

	def update_turn(
		self, branch: str, turn: int, end_tick: int, plan_end_tick: int
	):
		id_ = self.filter_get_id(
			"turns", [pc.field("branch") == branch, pc.field("turn") == turn]
		)
		if id_ is None:
			return ParquetDB("turns", self._path).create(
				{
					"branch": branch,
					"turn": turn,
					"end_tick": end_tick,
					"plan_end_tick": plan_end_tick,
				},
			)
		return ParquetDB("turns", self._path).update(
			{
				"id": id_,
				"branch": branch,
				"turn": turn,
				"end_tick": end_tick,
				"plan_end_tick": plan_end_tick,
			},
		)

	def set_turn(
		self, branch: str, turn: int, end_tick: int, plan_end_tick: int
	) -> None:
		try:
			self.update_turn(branch, turn, end_tick, plan_end_tick)
		except (ArrowInvalid, IndexError):
			ParquetDB("turns", self._path).create(
				{
					"branch": branch,
					"turn": turn,
					"end_tick": end_tick,
					"plan_end_tick": plan_end_tick,
				}
			)

	def set_turn_completed(self, branch: str, turn: int) -> None:
		db = ParquetDB("turns_completed", self._path)
		try:
			id_ = db.read(
				filters=[
					pc.field("branch") == branch,
					pc.field("turn") == turn,
				],
			)["id"][0]
			db.update({"id": id_, "branch": branch, "turn": turn})
		except IndexError:
			db.create({"branch": branch, "turn": turn})

	def load_things_tick_to_end(
		self, character: bytes, branch: str, turn_from: int, tick_from: int
	) -> List[Tuple[bytes, int, int, bytes]]:
		return list(
			self._iter_things_tick_to_end(
				character, branch, turn_from, tick_from
			)
		)

	def _iter_things_tick_to_end(
		self, character: bytes, branch: str, turn_from: int, tick_from: int
	) -> Iterator[Tuple[bytes, int, int, bytes]]:
		for d in (
			ParquetDB("things", self._path)
			.read(
				filters=[
					pc.field("character") == character,
					pc.field("branch") == branch,
					pc.field("turn") >= turn_from,
				],
			)
			.to_pylist()
		):
			if d["turn"] == turn_from:
				if d["tick"] >= tick_from:
					yield d["thing"], d["turn"], d["tick"], d["location"]
			else:
				yield d["thing"], d["turn"], d["tick"], d["location"]

	def load_things_tick_to_tick(
		self,
		character: bytes,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> List[Tuple[bytes, int, int, bytes]]:
		return list(
			self._iter_things_tick_to_tick(
				character, branch, turn_from, tick_from, turn_to, tick_to
			)
		)

	def _iter_things_tick_to_tick(
		self,
		character: bytes,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	):
		db = ParquetDB("things", self._path)
		if turn_from == turn_to:
			for d in db.read(
				filters=[
					pc.field("character") == character,
					pc.field("branch") == branch,
					pc.field("turn") == turn_from,
					pc.field("tick") >= tick_from,
					pc.field("tick") <= tick_to,
				],
			).to_pylist():
				yield d["thing"], d["turn"], d["tick"], d["location"]
		else:
			for d in db.read(
				filters=[
					pc.field("character") == character,
					pc.field("branch") == branch,
					pc.field("turn_from") >= turn_from,
					pc.field("turn_to") <= turn_to,
				],
			).to_pylist():
				if d["turn"] == turn_from:
					if d["tick"] >= tick_from:
						yield d["thing"], d["turn"], d["tick"], d["location"]
				elif d["turn"] == turn_to:
					if d["tick"] <= tick_to:
						yield d["thing"], d["turn"], d["tick"], d["location"]
				else:
					yield d["thing"], d["turn"], d["tick"], d["location"]

	def load_graph_val_tick_to_end(
		self, graph: bytes, branch: str, turn_from: int, tick_from: int
	) -> List[Tuple[bytes, int, int, bytes]]:
		return list(
			self._iter_graph_val_tick_to_end(
				graph, branch, turn_from, tick_from
			)
		)

	def _iter_graph_val_tick_to_end(
		self, graph: bytes, branch: str, turn_from: int, tick_from: int
	) -> Iterator[Tuple[bytes, int, int, bytes]]:
		for d in (
			ParquetDB("graph_val", self._path)
			.read(
				"graph_val",
				filters=[
					pc.field("graph") == graph,
					pc.field("branch") == branch,
					pc.field("turn") >= turn_from,
				],
			)
			.to_pylist()
		):
			if d["turn"] == turn_from:
				if d["tick"] >= tick_from:
					yield d["key"], d["turn"], d["tick"], d["value"]
			else:
				yield d["key"], d["turn"], d["tick"], d["value"]

	def load_graph_val_tick_to_tick(
		self,
		graph: bytes,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> List[Tuple[bytes, int, int, bytes]]:
		return list(
			self._iter_graph_val_tick_to_tick(
				graph, branch, turn_from, tick_from, turn_to, tick_to
			)
		)

	def _iter_graph_val_tick_to_tick(
		self,
		graph: bytes,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> Iterator[Tuple[bytes, int, int, bytes]]:
		db = ParquetDB("graph_val", self._path)
		if turn_from == tick_from:
			for d in db.read(
				filters=[
					pc.field("graph") == graph,
					pc.field("branch") == branch,
					pc.field("turn") == turn_from,
					pc.field("tick") >= tick_from,
					pc.field("tick") <= tick_to,
				],
			).to_pylist():
				yield d["key"], d["turn"], d["tick"], d["value"]
		else:
			for d in db.read(
				filters=[
					pc.field("graph") == graph,
					pc.field("branch") == branch,
					pc.field("turn") >= turn_from,
					pc.field("turn") <= turn_to,
				],
			).to_pylist():
				if d["turn"] == turn_from:
					if d["tick"] >= tick_from:
						yield d["key"], d["turn"], d["tick"], d["value"]
				elif d["turn"] == turn_to:
					if d["tick"] <= tick_from:
						yield d["key"], d["turn"], d["tick"], d["value"]
				else:
					yield d["key"], d["turn"], d["tick"], d["value"]

	def load_nodes_tick_to_end(
		self, graph: bytes, branch: str, turn_from: int, tick_from: int
	) -> List[Tuple[bytes, int, int, bool]]:
		return list(
			self._iter_nodes_tick_to_end(graph, branch, turn_from, tick_from)
		)

	def _iter_nodes_tick_to_end(
		self, graph: bytes, branch: str, turn_from: int, tick_from: int
	) -> Iterator[Tuple[bytes, int, int, bool]]:
		for d in (
			ParquetDB("nodes", self._path)
			.read(
				"nodes",
				filters=[
					pc.field("graph") == graph,
					pc.field("branch") == branch,
					pc.field("turn") >= turn_from,
				],
			)
			.to_pylist()
		):
			if d["turn"] == turn_from:
				if d["tick"] >= tick_from:
					yield (
						d["node"],
						d["turn"],
						d["tick"],
						d["extant"],
					)
			else:
				yield (
					d["node"],
					d["turn"],
					d["tick"],
					d["extant"],
				)

	def load_nodes_tick_to_tick(
		self,
		graph: bytes,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> List[Tuple[bytes, int, int, bool]]:
		return list(
			self._iter_nodes_tick_to_tick(
				graph, branch, turn_from, tick_from, turn_to, tick_to
			)
		)

	def _iter_nodes_tick_to_tick(
		self,
		graph: bytes,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> Iterator[Tuple[bytes, int, int, bool]]:
		db = ParquetDB("nodes", self._path)
		if turn_from == turn_to:
			for d in db.read(
				"nodes",
				filters=[
					pc.field("graph") == graph,
					pc.field("branch") == branch,
					pc.field("turn") == turn_from,
					pc.field("tick") >= tick_from,
					pc.field("tick") <= tick_to,
				],
			).to_pylist():
				yield (
					d["node"],
					d["turn"],
					d["tick"],
					d["extant"],
				)
		else:
			for d in db.read(
				"nodes",
				filters=[
					pc.field("graph") == graph,
					pc.field("branch") == branch,
					pc.field("turn") >= turn_from,
					pc.field("turn") <= turn_to,
				],
			).to_pylist():
				if d["turn"] == turn_from:
					if d["tick"] >= tick_from:
						yield (
							d["node"],
							d["turn"],
							d["tick"],
							d["extant"],
						)
				elif d["turn"] == turn_to:
					if d["tick"] <= tick_to:
						yield (
							d["node"],
							d["turn"],
							d["tick"],
							d["extant"],
						)
				else:
					yield (
						d["node"],
						d["turn"],
						d["tick"],
						d["extant"],
					)

	def load_node_val_tick_to_end(
		self, graph: bytes, branch: str, turn_from: int, tick_from: int
	) -> List[Tuple[bytes, bytes, int, int, bytes]]:
		return list(
			self._iter_node_val_tick_to_end(
				graph, branch, turn_from, tick_from
			)
		)

	def _iter_node_val_tick_to_end(
		self, graph: bytes, branch: str, turn_from: int, tick_from: int
	) -> Iterator[Tuple[bytes, bytes, int, int, bytes]]:
		for d in (
			ParquetDB("node_val", self._path)
			.read(
				filters=[
					pc.field("graph") == graph,
					pc.field("branch") == branch,
					pc.field("turn_from") >= turn_from,
				],
			)
			.to_pylist()
		):
			if d["turn"] == turn_from:
				if d["tick"] >= tick_from:
					yield (
						d["node"],
						d["key"],
						d["turn"],
						d["tick"],
					)
			else:
				yield d["node"], d["key"], d["turn"], d["tick"]

	def load_node_val_tick_to_tick(
		self,
		graph: bytes,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> List[Tuple[bytes, bytes, int, int, bytes]]:
		return list(
			self._iter_node_val_tick_to_tick(
				graph, branch, turn_from, tick_from, turn_to, tick_to
			)
		)

	def _iter_node_val_tick_to_tick(
		self,
		graph: bytes,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> Iterator[Tuple[bytes, bytes, int, int, bytes]]:
		db = ParquetDB("node_val", self._path)
		if turn_from == turn_to:
			for d in db.read(
				"node_val",
				filters=[
					pc.field("graph") == graph,
					pc.field("branch") == branch,
					pc.field("turn") == turn_from,
					pc.field("tick") >= tick_from,
					pc.field("tick") <= tick_to,
				],
			).to_pylist():
				yield (
					d["node"],
					d["key"],
					d["turn"],
					d["tick"],
					d["value"],
				)
		else:
			for d in db.read(
				"node_val",
				filters=[
					pc.field("graph") == graph,
					pc.field("branch") == branch,
					pc.field("turn") >= turn_from,
					pc.field("turn") <= turn_to,
				],
			).to_pylist():
				if d["turn"] == turn_from:
					if d["tick"] >= tick_from:
						yield (
							d["node"],
							d["key"],
							d["turn"],
							d["tick"],
							d["value"],
						)
				elif d["turn"] == turn_to:
					if d["tick"] <= tick_to:
						yield (
							d["node"],
							d["key"],
							d["turn"],
							d["tick"],
							d["value"],
						)
				else:
					yield (
						d["node"],
						d["key"],
						d["turn"],
						d["tick"],
						d["value"],
					)

	def load_edges_tick_to_end(
		self, graph: bytes, branch: str, turn_from: int, tick_from: int
	) -> List[Tuple[bytes, bytes, int, int, int, bool]]:
		return list(
			self._iter_edges_tick_to_end(graph, branch, turn_from, tick_from)
		)

	def _iter_edges_tick_to_end(
		self, graph: bytes, branch: str, turn_from: int, tick_from: int
	) -> Iterator[Tuple[bytes, bytes, int, int, int, bool]]:
		for d in (
			ParquetDB("edges", self._path)
			.read(
				"edges",
				filters=[
					pc.field("graph") == graph,
					pc.field("branch") == branch,
					pc.field("turn_from") >= turn_from,
				],
			)
			.to_pylist()
		):
			if d["turn"] == turn_from:
				if d["tick"] >= tick_from:
					yield (
						d["orig"],
						d["dest"],
						d["idx"],
						d["turn"],
						d["tick"],
						d["extant"],
					)
			else:
				yield (
					d["orig"],
					d["dest"],
					d["idx"],
					d["turn"],
					d["tick"],
					d["extant"],
				)

	def load_edges_tick_to_tick(
		self,
		graph: bytes,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> List[Tuple[bytes, bytes, bytes, int, int, int, bytes]]:
		return list(
			self._iter_edges_tick_to_tick(
				graph, branch, turn_from, tick_from, turn_to, tick_to
			)
		)

	def _iter_edges_tick_to_tick(
		self,
		graph: bytes,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> Iterator[Tuple[bytes, bytes, bytes, int, int, int, bytes]]:
		db = ParquetDB("edges", self._path)
		if turn_from == turn_to:
			for d in db.read(
				"edges",
				filters=[
					pc.field("graph") == graph,
					pc.field("branch") == branch,
					pc.field("turn") == turn_from,
					pc.field("tick") >= tick_from,
					pc.field("tick") <= tick_to,
				],
			).to_pylist():
				yield (
					d["orig"],
					d["dest"],
					d["idx"],
					d["turn"],
					d["tick"],
					d["extant"],
				)
		else:
			for d in db.read(
				"edges",
				filters=[
					pc.field("graph") == graph,
					pc.field("branch") == branch,
					pc.field("turn") >= turn_from,
					pc.field("turn") <= turn_to,
				],
			).to_pylist():
				if d["turn"] == turn_from:
					if d["tick"] >= tick_from:
						yield (
							d["orig"],
							d["dest"],
							d["idx"],
							d["turn"],
							d["tick"],
							d["extant"],
						)
				elif d["turn"] == turn_to:
					if d["tick"] <= tick_to:
						yield (
							d["orig"],
							d["dest"],
							d["idx"],
							d["turn"],
							d["tick"],
							d["extant"],
						)
				else:
					yield (
						d["orig"],
						d["dest"],
						d["idx"],
						d["turn"],
						d["tick"],
						d["extant"],
					)

	def load_edge_val_tick_to_end(
		self, graph: bytes, branch: str, turn_from: int, tick_from: int
	) -> List[Tuple[bytes, bytes, bytes, int, int, int, bytes]]:
		return list(
			self._iter_edge_val_tick_to_end(
				graph, branch, turn_from, tick_from
			)
		)

	def _iter_edge_val_tick_to_end(
		self, graph: bytes, branch: str, turn_from: int, tick_from: int
	) -> Iterator[Tuple[bytes, bytes, int, bytes, int, int, bytes]]:
		for d in (
			ParquetDB("edge_val", self._path)
			.read(
				filters=[
					pc.field("graph") == graph,
					pc.field("branch") == branch,
					pc.field("turn") >= turn_from,
				],
			)
			.to_pylist()
		):
			if d["turn"] == turn_from:
				if d["tick"] >= tick_from:
					yield (
						d["orig"],
						d["dest"],
						d["idx"],
						d["key"],
						d["turn"],
						d["tick"],
						d["value"],
					)
			else:
				yield (
					d["orig"],
					d["dest"],
					d["idx"],
					d["key"],
					d["turn"],
					d["tick"],
					d["value"],
				)

	def load_edge_val_tick_to_tick(
		self,
		graph: bytes,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> List[Tuple[bytes, bytes, bytes, int, int, int, bytes]]:
		return list(
			self._iter_edge_val_tick_to_tick(
				graph, branch, turn_from, tick_from, turn_to, tick_to
			)
		)

	def _iter_edge_val_tick_to_tick(
		self,
		graph: bytes,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> Iterator[Tuple[bytes, bytes, bytes, int, int, int, bytes]]:
		db = ParquetDB("edge_val", self._path)
		if turn_from == turn_to:
			for d in db.read(
				"edge_val",
				filters=[
					pc.field("graph") == graph,
					pc.field("branch") == branch,
					pc.field("turn") == turn_from,
					pc.field("tick") >= tick_from,
					pc.field("tick") <= tick_to,
				],
			).to_pylist():
				yield (
					d["orig"],
					d["dest"],
					d["idx"],
					d["key"],
					d["turn"],
					d["tick"],
					d["value"],
				)
		else:
			for d in db.read(
				"edge_val",
				filters=[
					pc.field("graph") == graph,
					pc.field("branch") == branch,
					pc.field("turn") >= turn_from,
					pc.field("turn") <= turn_to,
				],
			).to_pylist():
				if d["turn"] == turn_from:
					if d["tick"] >= tick_from:
						yield (
							d["orig"],
							d["dest"],
							d["idx"],
							d["key"],
							d["turn"],
							d["tick"],
							d["value"],
						)
				elif d["turn"] == turn_to:
					if d["tick"] <= tick_to:
						yield (
							d["orig"],
							d["dest"],
							d["idx"],
							d["key"],
							d["turn"],
							d["tick"],
							d["value"],
						)
				else:
					yield (
						d["orig"],
						d["dest"],
						d["idx"],
						d["key"],
						d["turn"],
						d["tick"],
						d["value"],
					)

	def _del_time(self, table: str, branch: str, turn: int, tick: int):
		id_ = self.filter_get_id(
			table,
			filters=[
				pc.field("branch") == branch,
				pc.field("turn") == turn,
				pc.field("tick") == tick,
			],
		)
		if id_ is None:
			raise KeyError("No value at this time")
		ParquetDB(table, self._path).delete([id_])

	def nodes_del_time(self, branch: str, turn: int, tick: int):
		self._del_time("nodes", branch, turn, tick)

	def edges_del_time(self, branch: str, turn: int, tick: int):
		self._del_time("edges", branch, turn, tick)

	def graph_val_del_time(self, branch: str, turn: int, tick: int):
		self._del_time("graph_val", branch, turn, tick)

	def node_val_del_time(self, branch: str, turn: int, tick: int):
		self._del_time("node_val", branch, turn, tick)

	def edge_val_del_time(self, branch: str, turn: int, tick: int):
		self._del_time("edge_val", branch, turn, tick)

	@staticmethod
	def echo(it):
		return it

	def run(self):
		inq = self._inq
		outq = self._outq
		while True:
			inst = inq.get()
			if inst == "shutdown":
				self.existence_lock.release()
				return
			if inst == "commit":
				continue
			if not isinstance(inst, (str, tuple)):
				raise TypeError("Can't use SQLAlchemy with ParquetDB")
			silent = False
			if inst[0] == "silent":
				silent = True
				inst = inst[1:]
			try:
				res = getattr(self, inst[0])(*inst[1], **inst[2])
			except Exception as ex:
				assert not silent, f"Got exception while silenced: {ex}"
				res = ex
			if not silent:
				outq.put(res)


class AbstractLiSEQueryEngine(AbstractQueryEngine):
	def comparison(
		self,
		entity0: Key,
		stat0: Key,
		entity1: Key,
		stat1: Key = None,
		oper: str = "eq",
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

	_records: int
	kf_interval_override: callable
	keyframe_interval: Optional[int]
	snap_keyframe: callable

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

	@abstractmethod
	def universals_dump(self) -> Iterator[Tuple[Key, str, int, int, Any]]:
		pass

	@abstractmethod
	def rulebooks_dump(
		self,
	) -> Iterator[Tuple[Key, str, int, int, Tuple[List[Key], float]]]:
		pass

	@abstractmethod
	def rules_dump(self) -> Iterator[str]:
		pass

	@abstractmethod
	def rule_triggers_dump(
		self,
	) -> Iterator[Tuple[Key, str, int, int, List[Key]]]:
		pass

	@abstractmethod
	def rule_prereqs_dump(
		self,
	) -> Iterator[Tuple[Key, str, int, int, List[Key]]]:
		pass

	@abstractmethod
	def rule_actions_dump(
		self,
	) -> Iterator[Tuple[Key, str, int, int, List[Key]]]:
		pass

	@abstractmethod
	def rule_neighborhood_dump(
		self,
	) -> Iterator[Tuple[Key, str, int, int, int]]:
		pass

	@abstractmethod
	def node_rulebook_dump(
		self,
	) -> Iterator[Tuple[Key, Key, str, int, int, Key]]:
		pass

	@abstractmethod
	def portal_rulebook_dump(
		self,
	) -> Iterator[Tuple[Key, Key, Key, str, int, int, Key]]:
		pass

	@abstractmethod
	def character_rulebook_dump(
		self,
	) -> Iterator[Tuple[Key, str, int, int, Key]]:
		pass

	@abstractmethod
	def unit_rulebook_dump(self) -> Iterator[Tuple[Key, str, int, int, Key]]:
		pass

	@abstractmethod
	def character_thing_rulebook_dump(
		self,
	) -> Iterator[Tuple[Key, str, int, int, Key]]:
		pass

	@abstractmethod
	def character_place_rulebook_dump(
		self,
	) -> Iterator[Tuple[Key, str, int, int, Key]]:
		pass

	@abstractmethod
	def character_portal_rulebook_dump(
		self,
	) -> Iterator[Tuple[Key, str, int, int, Key]]:
		pass

	@abstractmethod
	def character_rules_handled_dump(
		self,
	) -> Iterator[Tuple[Key, Key, str, str, int, int]]:
		pass

	@abstractmethod
	def character_rules_changed_dump(
		self,
	) -> Iterator[Tuple[Key, Key, str, str, int, int, int, int]]:
		pass

	@abstractmethod
	def unit_rules_handled_dump(
		self,
	) -> Iterator[Tuple[Key, Key, Key, Key, str, str, int, int]]:
		pass

	@abstractmethod
	def character_thing_rules_handled_dump(
		self,
	) -> Iterator[Tuple[Key, Key, Key, str, str, int, int]]:
		pass

	@abstractmethod
	def character_place_rules_handled_dump(
		self,
	) -> Iterator[Tuple[Key, Key, Key, str, str, int, int]]:
		pass

	@abstractmethod
	def character_portal_rules_handled_dump(
		self,
	) -> Iterator[Tuple[Key, Key, Key, Key, str, str, int, int]]:
		pass

	@abstractmethod
	def node_rules_handled_dump(
		self,
	) -> Iterator[Tuple[Key, Key, Key, str, str, int, int]]:
		pass

	@abstractmethod
	def portal_rules_handled_dump(
		self,
	) -> Iterator[Tuple[Key, Key, Key, Key, str, str, int, int]]:
		pass

	@abstractmethod
	def things_dump(self) -> Iterator[Tuple[Key, Key, str, int, int, Key]]:
		pass

	@abstractmethod
	def load_things(
		self,
		character: Key,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int = None,
		tick_to: int = None,
	) -> Iterator[Tuple[Key, Key, str, int, int, Key]]:
		pass

	@abstractmethod
	def units_dump(
		self,
	) -> Iterator[Tuple[Key, Key, Key, str, int, int, bool]]:
		pass

	@abstractmethod
	def universal_set(
		self, key: Key, branch: str, turn: int, tick: int, val: Any
	):
		pass

	@abstractmethod
	def universal_del(self, key: Key, branch: str, turn: int, tick: int):
		pass

	@abstractmethod
	def comparison(
		self,
		entity0: Key,
		stat0: Key,
		entity1: Key,
		stat1: Key = None,
		oper: str = "eq",
		windows: list = None,
	):
		pass

	@abstractmethod
	def count_all_table(self, tbl: str) -> int:
		pass

	@abstractmethod
	def set_rule_triggers(
		self, rule: str, branch: str, turn: int, tick: int, flist: List[str]
	):
		pass

	@abstractmethod
	def set_rule_prereqs(
		self, rule: str, branch: str, turn: int, tick: int, flist: List[str]
	):
		pass

	@abstractmethod
	def set_rule_actions(
		self, rule: str, branch: str, turn: int, tick: int, flist: List[str]
	):
		pass

	@abstractmethod
	def set_rule_neighborhood(
		self, rule: str, branch: str, turn: int, tick: int, neighborhood: int
	):
		pass

	@abstractmethod
	def set_rule(
		self,
		rule: str,
		branch: str,
		turn: int,
		tick: int,
		triggers: List[str],
		prereqs: List[str],
		actions: List[str],
		neighborhood: int,
	):
		pass

	@abstractmethod
	def set_rulebook(
		self,
		name: Key,
		branch: str,
		turn: int,
		tick: int,
		rules: List[str] = None,
		prio: float = 0.0,
	):
		pass

	@abstractmethod
	def set_character_rulebook(
		self, char: Key, branch: str, turn: int, tick: int, rb: Key
	):
		pass

	@abstractmethod
	def set_unit_rulebook(
		self, char: Key, branch: str, turn: int, tick: int, rb: Key
	):
		pass

	@abstractmethod
	def set_character_thing_rulebook(
		self, char: Key, branch: str, turn: int, tick: int, rb: Key
	):
		pass

	@abstractmethod
	def set_character_place_rulebook(
		self, char: Key, branch: str, turn: int, tick: int, rb: Key
	):
		pass

	@abstractmethod
	def set_character_portal_rulebook(
		self, char: Key, branch: str, turn: int, tick: int, rb: Key
	):
		pass

	@abstractmethod
	def rulebooks(self) -> Iterator[Key]:
		pass

	@abstractmethod
	def set_node_rulebook(
		self,
		character: Key,
		node: Key,
		branch: str,
		turn: int,
		tick: int,
		rulebook: Key,
	):
		pass

	@abstractmethod
	def set_portal_rulebook(
		self,
		character: Key,
		orig: Key,
		dest: Key,
		branch: str,
		turn: int,
		tick: int,
		rulebook: Key,
	):
		pass

	@abstractmethod
	def handled_character_rule(
		self,
		character: Key,
		rulebook: Key,
		rule: str,
		branch: str,
		turn: int,
		tick: int,
	):
		pass

	@abstractmethod
	def handled_unit_rule(
		self,
		character: Key,
		rulebook: Key,
		rule: str,
		graph: Key,
		unit: Key,
		branch: str,
		turn: int,
		tick: int,
	):
		pass

	@abstractmethod
	def handled_character_thing_rule(
		self,
		character: Key,
		rulebook: Key,
		rule: str,
		thing: Key,
		branch: str,
		turn: int,
		tick: int,
	):
		pass

	@abstractmethod
	def handled_character_place_rule(
		self,
		character: Key,
		rulebook: Key,
		rule: str,
		place: Key,
		branch: str,
		turn: int,
		tick: int,
	):
		pass

	@abstractmethod
	def handled_character_portal_rule(
		self,
		character: Key,
		rulebook: Key,
		rule: str,
		orig: Key,
		dest: Key,
		branch: str,
		turn: int,
		tick: int,
	):
		pass

	@abstractmethod
	def handled_node_rule(
		self,
		character: Key,
		node: Key,
		rulebook: Key,
		rule: str,
		branch: str,
		turn: int,
		tick: int,
	):
		pass

	@abstractmethod
	def handled_portal_rule(
		self,
		character: Key,
		orig: Key,
		dest: Key,
		rulebook: Key,
		rule: str,
		branch: str,
		turn: int,
		tick: int,
	):
		pass

	@abstractmethod
	def set_thing_loc(
		self,
		character: Key,
		thing: Key,
		branch: str,
		turn: int,
		tick: int,
		loc: Key,
	):
		pass

	@abstractmethod
	def unit_set(
		self,
		character: Key,
		graph: Key,
		node: Key,
		branch: str,
		turn: int,
		tick: int,
		isav: bool,
	):
		pass

	@abstractmethod
	def rulebook_set(
		self,
		rulebook: Key,
		branch: str,
		turn: int,
		tick: int,
		rules: List[str],
	):
		pass

	@abstractmethod
	def turns_completed_dump(self) -> Iterator[Tuple[str, int]]:
		pass

	@abstractmethod
	def complete_turn(
		self, branch: str, turn: int, discard_rules: bool = False
	):
		pass

	@abstractmethod
	def set_rulebook_on_character(
		self, rbtyp: str, char: Key, branch: str, turn: int, tick: int, rb: Key
	):
		pass


class ParquetGlobalMapping(MutableMapping):
	def __init__(self, query_engine: "ParquetQueryEngine"):
		self._qe = query_engine

	def __iter__(self):
		return self._qe.global_keys()

	def __len__(self):
		return self._qe.count_all_table("global")

	def __getitem__(self, item):
		return self._qe.global_get(item)

	def __setitem__(self, key, value):
		self._qe.global_set(key, value)

	def __delitem__(self, key):
		self._qe.global_del(key)


class ParquetQueryEngine(AbstractLiSEQueryEngine):
	holder_cls = ParquetDBHolder

	def __init__(self, path, _, pack=None, unpack=None):
		self._inq = Queue()
		self._outq = Queue()
		self._holder = self.holder_cls(path, self._inq, self._outq)
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

		if pack is None:

			def pack(s: Any) -> bytes:
				return repr(s).encode()

		if unpack is None:
			from ast import literal_eval

			def unpack(b: bytes) -> Any:
				return literal_eval(b.decode())

		self.pack = pack
		self.unpack = unpack
		self._branches = {}
		self._nodevals2set = []
		self._edgevals2set = []
		self._graphvals2set = []
		self._nodes2set = []
		self._edges2set = []
		self._char_rules_handled = []
		self._unit_rules_handled = []
		self._char_thing_rules_handled = []
		self._char_place_rules_handled = []
		self._char_portal_rules_handled = []
		self._node_rules_handled = []
		self._portal_rules_handled = []
		self._btts = set()
		self.globl = ParquetGlobalMapping(self)
		self._t = Thread(target=self._holder.run, daemon=True)
		self._t.start()

	def echo(self, something: Any) -> Any:
		self._inq.put(("echo", something))
		return self._outq.get()

	def call(self, method, *args, **kwargs):
		with self._holder.lock:
			self._inq.put((method, args, kwargs))
			ret = self._outq.get()
			if isinstance(ret, Exception):
				raise ret
			return ret

	def call_silent(self, method, *args, **kwargs):
		self._inq.put(("silent", method, args, kwargs))

	def global_keys(self):
		unpack = self.unpack
		for key in self.call("global_keys"):
			yield unpack(key)

	def new_graph(
		self, graph: Key, branch: str, turn: int, tick: int, typ: str
	) -> None:
		graph = self.pack(graph)
		self.call(
			"insert1",
			"graphs",
			{
				"graph": graph,
				"branch": branch,
				"turn": turn,
				"tick": tick,
				"type": typ,
			},
		)

	new_character = graphs_insert = new_graph

	def set_rulebook_on_character(
		self, rbtyp: str, char: Key, branch: str, turn: int, tick: int, rb: Key
	):
		pack = self.pack
		self.call(
			"set_rulebook_on_character",
			rbtyp,
			pack(char),
			branch,
			turn,
			tick,
			pack(rb),
		)

	def keyframes_insert_many(self, many):
		pack = self.pack
		return self.call(
			"insert",
			"keyframes",
			[
				{
					"graph": pack(graph),
					"branch": branch,
					"turn": turn,
					"tick": tick,
					"nodes": pack(nodes),
					"edges": pack(edges),
					"graph_val": pack(graph_val),
				}
				for (
					graph,
					branch,
					turn,
					tick,
					nodes,
					edges,
					graph_val,
				) in many
			],
		)

	def keyframes_dump(self):
		unpack = self.unpack
		for graph, branch, turn, tick, nodes, edges, graph_val in self.call(
			"dump", "keyframes"
		):
			yield (
				unpack(graph),
				branch,
				turn,
				tick,
				unpack(nodes),
				unpack(edges),
				unpack(graph_val),
			)

	def keyframes_list(self) -> Iterator:  # change name pls
		unpack = self.unpack
		for d in self.call("list_keyframes"):
			yield unpack(d["graph"]), d["branch"], d["turn"], d["tick"]

	def get_keyframe(
		self, graph: Key, branch: str, turn: int, tick: int
	) -> Optional[Tuple[dict, dict, dict]]:
		unpack = self.unpack
		stuff = self.call("get_keyframe", self.pack(graph), branch, turn, tick)
		if not stuff:
			return
		nodes, edges, val = stuff
		return unpack(nodes), unpack(edges), unpack(val)

	def have_branch(self, branch: str) -> bool:
		return self.call("have_branch", branch)

	def all_branches(self) -> Iterator[Tuple[str, str, int, int, int, int]]:
		for d in self.call("dump", "branches"):
			yield (
				d["branch"],
				d["parent"],
				d["parent_turn"],
				d["parent_tick"],
				d["end_turn"],
				d["end_tick"],
			)

	def global_get(self, key: Key) -> Any:
		return self.unpack(self.call("get_global", self.pack(key)))

	def global_set(self, key, value):
		pack = self.pack
		return self.call("set_global", pack(key), pack(value))

	def global_del(self, key):
		return self.call("del_global", self.pack(key))

	def global_items(self):
		unpack = self.unpack
		for d in self.call("dump", "global"):
			yield unpack(d["key"]), unpack(d["value"])

	def get_branch(self):
		return self.unpack(self.call("get_global", b"\xa6branch"))

	def get_turn(self):
		return self.unpack(self.call("get_global", b"\xa4turn"))

	def get_tick(self):
		return self.unpack(self.call("get_global", b"\xa4tick"))

	def new_branch(self, branch, parent, parent_turn, parent_tick):
		return self.call(
			"insert1",
			"branches",
			{
				"branch": branch,
				"parent": parent,
				"parent_turn": parent_turn,
				"parent_tick": parent_tick,
				"end_turn": parent_turn,
				"end_tick": parent_tick,
			},
		)

	def update_branch(
		self, branch, parent, parent_turn, parent_tick, end_turn, end_tick
	):
		return self.call(
			"update_branch",
			branch,
			parent,
			parent_turn,
			parent_tick,
			end_turn,
			end_tick,
		)

	def set_branch(
		self, branch, parent, parent_turn, parent_tick, end_turn, end_tick
	):
		return self.call(
			"set_branch",
			branch,
			parent,
			parent_turn,
			parent_tick,
			end_turn,
			end_tick,
		)

	def update_turn(self, branch, turn, end_tick, plan_end_tick):
		return self.call("update_turn", branch, turn, end_tick, plan_end_tick)

	def set_turn(self, branch, turn, end_tick, plan_end_tick):
		return self.call("set_turn", branch, turn, end_tick, plan_end_tick)

	def turns_dump(self):
		for d in self.call("dump", "turns"):
			yield d["branch"], d["turn"], d["end_tick"], d["plan_end_tick"]

	def flush(self):
		with self._holder.lock:
			put = self._inq.put
			pack = self.pack
			records = 0
			if self._graphvals2set:
				records += len(self._graphvals2set)
				put(
					(
						"silent",
						"insert",
						(
							"graph_val",
							[
								{
									"graph": pack(graph),
									"key": pack(key),
									"branch": branch,
									"turn": turn,
									"tick": tick,
									"value": pack(value),
								}
								for (
									graph,
									key,
									branch,
									turn,
									tick,
									value,
								) in self._graphvals2set
							],
						),
						{},
					)
				)
				self._graphvals2set = []
			if self._nodes2set:
				records += len(self._nodes2set)
				put(
					(
						"silent",
						"insert",
						(
							"nodes",
							[
								{
									"graph": pack(graph),
									"key": pack(key),
									"branch": branch,
									"turn": turn,
									"tick": tick,
									"extant": bool(extant),
								}
								for (
									graph,
									key,
									branch,
									turn,
									tick,
									extant,
								) in self._nodes2set
							],
						),
						{},
					)
				)
				self._nodes2set = []
			if self._nodevals2set:
				records += len(self._nodevals2set)
				put(
					(
						"silent",
						"insert",
						(
							"node_val",
							[
								{
									"graph": pack(graph),
									"node": pack(node),
									"key": pack(key),
									"branch": branch,
									"turn": turn,
									"tick": tick,
									"value": pack(value),
								}
								for (
									graph,
									node,
									key,
									branch,
									turn,
									tick,
									value,
								) in self._nodevals2set
							],
						),
						{},
					)
				)
				self._nodevals2set = []
			if self._edges2set:
				records += len(self._edges2set)
				put(
					(
						"silent",
						"insert",
						(
							"edges",
							[
								{
									"graph": pack(graph),
									"orig": pack(orig),
									"dest": pack(dest),
									"idx": 0,
									"branch": branch,
									"turn": turn,
									"tick": tick,
									"extant": bool(extant),
								}
								for (
									graph,
									orig,
									dest,
									branch,
									turn,
									tick,
									extant,
								) in self._edges2set
							],
						),
						{},
					)
				)
				self._edges2set = []
			if self._edgevals2set:
				records += len(self._edgevals2set)
				put(
					(
						"silent",
						"insert",
						(
							"edge_val",
							[
								{
									"graph": pack(graph),
									"orig": pack(orig),
									"dest": pack(dest),
									"idx": 0,
									"key": pack(key),
									"branch": branch,
									"turn": turn,
									"tick": tick,
									"value": pack(value),
								}
								for (
									graph,
									orig,
									dest,
									key,
									branch,
									turn,
									tick,
									value,
								) in self._edgevals2set
							],
						),
						{},
					)
				)
				self._edgevals2set = []
			if self._unitness:
				records += len(self._unitness)
				put(
					(
						"silent",
						"del_units_after",
						(
							[
								(character, graph, node, branch, turn, tick)
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
						),
						{},
					)
				)
				put(("silent", "insert", "units", {
					"character_graph": character,
					"unit_graph": graph,
					"node": node,
					"branch": branch,
					"turn": turn,
					"tick": tick,
					"is_unit": isunit
				} for (character, graph, node, branch, turn, tick, isunit) in self._unitness))
				self._unitness = []
			if self._location:
				records += len(self._location)
				put(
					(
						"silent",
						"del_things_after",
						(
							[
								(character, thing, branch, turn, tick)
								for (
									character,
									thing,
									branch,
									turn,
									tick,
									_,
								) in self._location
							],
						),
						{},
					)
				)
				put(
					(
						"silent",
						"insert",
						(
							"things",
							[
								{
									"character": character,
									"thing": thing,
									"branch": branch,
									"turn": turn,
									"tick": tick,
									"location": loc,
								}
								for (
									character,
									thing,
									branch,
									turn,
									tick,
									loc,
								) in self._location
							],
						),
						{},
					)
				)
				self._location = []
			for attr, cmd in [
				("_char_rules_handled", "character_rules_handled"),
				("_unit_rules_handled", "unit_rules_handled"),
				(
					"_char_thing_rules_handled",
					"character_thing_rules_handled",
				),
				(
					"_char_place_rules_handled",
					"character_place_rules_handled",
				),
				(
					"_char_portal_rules_handled",
					"character_portal_rules_handled",
				),
				("_node_rules_handled", "_node_rules_handled"),
				("_portal_rules_handled", "portal_rules_handled"),
			]:
				if getattr(self, attr):
					records += len(getattr(self, attr))
					put(("silent", "insert", (cmd, getattr(self, attr)), {}))
				setattr(self, attr, [])
		assert self.call("echo", "flushed") == "flushed"
		override = self.kf_interval_override()
		if override is False or (
			self.keyframe_interval is not None
			and self._records + records > self.keyframe_interval
		):
			self.snap_keyframe()
		if self.keyframe_interval:
			self._records = (self._records + records) % self.keyframe_interval
		else:
			self._records += records

	def universals_dump(self) -> Iterator[Tuple[Key, str, int, int, Any]]:
		unpack = self.unpack
		for d in self.call("dump", "universals"):
			yield (
				unpack(d["key"]),
				d["branch"],
				d["turn"],
				d["tick"],
				unpack(d["value"]),
			)

	def rulebooks_dump(
		self,
	) -> Iterator[Tuple[Key, str, int, int, Tuple[List[Key], float]]]:
		unpack = self.unpack
		for d in self.call("dump", "rulebooks"):
			yield (
				unpack(d["rulebook"]),
				d["branch"],
				d["turn"],
				d["tick"],
				unpack(d["rules"]),
				d["priority"],
			)

	def rules_dump(self) -> Iterator[str]:
		for d in self.call("dump", "rules"):
			yield d["rule"]

	def _rule_dump(self, typ):
		unpack = self.unpack
		for d in self.call("dump", "rule_" + typ):
			yield (
				d["rule"],
				d["branch"],
				d["turn"],
				d["tick"],
				unpack(d[typ]),
			)

	def rule_triggers_dump(
		self,
	) -> Iterator[Tuple[Key, str, int, int, List[Key]]]:
		return self._rule_dump("triggers")

	def rule_prereqs_dump(
		self,
	) -> Iterator[Tuple[Key, str, int, int, List[Key]]]:
		return self._rule_dump("prereqs")

	def rule_actions_dump(
		self,
	) -> Iterator[Tuple[Key, str, int, int, List[Key]]]:
		return self._rule_dump("actions")

	def rule_neighborhood_dump(
		self,
	) -> Iterator[Tuple[Key, str, int, int, int]]:
		for d in self.call("dump", "rule_neighborhood"):
			yield (
				d["rule"],
				d["branch"],
				d["turn"],
				d["tick"],
				d["neighborhood"],
			)

	def node_rulebook_dump(
		self,
	) -> Iterator[Tuple[Key, Key, str, int, int, Key]]:
		unpack = self.unpack
		for d in self.call("dump", "node_rulebook"):
			yield (
				unpack(d["character"]),
				unpack(d["node"]),
				d["branch"],
				d["turn"],
				d["tick"],
				unpack(d["rulebook"]),
			)

	def portal_rulebook_dump(
		self,
	) -> Iterator[Tuple[Key, Key, Key, str, int, int, Key]]:
		unpack = self.unpack
		for d in self.call("dump", "portal_rulebook"):
			yield (
				unpack(d["character"]),
				unpack(d["orig"]),
				unpack(d["dest"]),
				d["branch"],
				d["turn"],
				d["tick"],
				unpack(d["rulebook"]),
			)

	def _character_rulebook_dump(self, typ):
		unpack = self.unpack
		for d in self.call("dump", f"{typ}_rulebook"):
			yield (
				unpack(d["character"]),
				d["branch"],
				d["turn"],
				d["tick"],
				unpack(d["rulebook"]),
			)

	def character_rulebook_dump(
		self,
	) -> Iterator[Tuple[Key, str, int, int, Key]]:
		return self._character_rulebook_dump("character")

	def unit_rulebook_dump(self) -> Iterator[Tuple[Key, str, int, int, Key]]:
		return self._character_rulebook_dump("unit")

	def character_thing_rulebook_dump(
		self,
	) -> Iterator[Tuple[Key, str, int, int, Key]]:
		return self._character_rulebook_dump("character_thing")

	def character_place_rulebook_dump(
		self,
	) -> Iterator[Tuple[Key, str, int, int, Key]]:
		return self._character_rulebook_dump("character_place")

	def character_portal_rulebook_dump(
		self,
	) -> Iterator[Tuple[Key, str, int, int, Key]]:
		return self._character_rulebook_dump("character_portal")

	def character_rules_handled_dump(
		self,
	) -> Iterator[Tuple[Key, Key, str, str, int, int]]:
		unpack = self.unpack
		for d in self.call("dump", "character_rules_handled"):
			yield (
				unpack(d["character"]),
				unpack(d["rulebook"]),
				unpack(d["rule"]),
				d["branch"],
				d["turn"],
				d["tick"],
			)

	def unit_rules_handled_dump(
		self,
	) -> Iterator[Tuple[Key, Key, Key, Key, str, str, int, int]]:
		unpack = self.unpack
		for d in self.call("dump", "unit_rules_handled"):
			yield (
				unpack(d["character"]),
				unpack(d["graph"]),
				unpack(d["unit"]),
				unpack(d["rulebook"]),
				d["rule"],
				d["branch"],
				d["turn"],
				d["tick"],
			)

	def character_thing_rules_handled_dump(
		self,
	) -> Iterator[Tuple[Key, Key, Key, str, str, int, int]]:
		unpack = self.unpack
		for d in self.call("dump", "character_thing_rules_handled"):
			yield (
				unpack(d["character"]),
				unpack(d["thing"]),
				unpack(d["rulebook"]),
				d["rule"],
				d["branch"],
				d["turn"],
				d["tick"],
			)

	def character_place_rules_handled_dump(
		self,
	) -> Iterator[Tuple[Key, Key, Key, str, str, int, int]]:
		unpack = self.unpack
		for d in self.call("dump", "character_place_rules_handled"):
			yield (
				unpack(d["character"]),
				unpack(d["place"]),
				unpack(d["rulebook"]),
				d["rule"],
				d["branch"],
				d["turn"],
				d["tick"],
			)

	def character_portal_rules_handled_dump(
		self,
	) -> Iterator[Tuple[Key, Key, Key, Key, str, str, int, int]]:
		unpack = self.unpack
		for d in self.call("dump", "character_portal_rules_handled"):
			yield (
				unpack(d["character"]),
				unpack(d["orig"]),
				unpack(d["dest"]),
				unpack(d["rulebook"]),
				d["rule"],
				d["branch"],
				d["turn"],
				d["tick"],
			)

	def node_rules_handled_dump(
		self,
	) -> Iterator[Tuple[Key, Key, Key, str, str, int, int]]:
		unpack = self.unpack
		for d in self.call("dump", "node_rules_handled"):
			yield (
				unpack(d["character"]),
				unpack(d["node"]),
				unpack(d["rulebook"]),
				d["rule"],
				d["branch"],
				d["turn"],
				d["tick"],
			)

	def portal_rules_handled_dump(
		self,
	) -> Iterator[Tuple[Key, Key, Key, Key, str, str, int, int]]:
		unpack = self.unpack
		for d in self.call("dump", "portal_rules_handled"):
			yield (
				unpack(d["character"]),
				unpack(d["orig"]),
				unpack(d["dest"]),
				unpack(d["rulebook"]),
				d["rule"],
				d["branch"],
				d["turn"],
				d["tick"],
			)

	def things_dump(self) -> Iterator[Tuple[Key, Key, str, int, int, Key]]:
		unpack = self.unpack
		for d in self.call("dump", "things"):
			yield (
				unpack(d["character"]),
				unpack(d["thing"]),
				d["branch"],
				d["turn"],
				d["tick"],
				unpack(d["location"]),
			)

	def load_things(
		self,
		character: Key,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int = None,
		tick_to: int = None,
	) -> Iterator[Tuple[Key, Key, str, int, int, Key]]:
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			if tick_to is not None:
				raise ValueError("Need both or neither of turn_to, tick_to")
			for thing, turn, tick, location in self.call(
				"load_things_tick_to_end",
				pack(character),
				branch,
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
			for thing, turn, tick, location in self.call(
				"load_things_tick_to_tick",
				pack(character),
				branch,
				turn_from,
				tick_from,
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

	def units_dump(
		self,
	) -> Iterator[Tuple[Key, Key, Key, str, int, int, bool]]:
		unpack = self.unpack
		for d in self.call("dump", "units"):
			yield (
				unpack(d["character_graph"]),
				unpack(d["unit_graph"]),
				unpack(d["unit_node"]),
				d["branch"],
				d["turn"],
				d["tick"],
				d["is_unit"],
			)

	def universal_set(
		self, key: Key, branch: str, turn: int, tick: int, val: Any
	):
		pack = self.pack
		self.call(
			"insert1",
			"universals",
			{
				"key": pack(key),
				"branch": branch,
				"turn": turn,
				"tick": tick,
				"value": pack(val),
			},
		)

	def universal_del(self, key: Key, branch: str, turn: int, tick: int):
		self.universal_set(key, branch, turn, tick, b"\xc0")

	def count_all_table(self, tbl: str) -> int:
		return self.call("rowcount", tbl)

	def set_rule_triggers(
		self, rule: str, branch: str, turn: int, tick: int, flist: List[str]
	):
		self.call(
			"insert1",
			"rule_triggers",
			{
				"rule": rule,
				"branch": branch,
				"turn": turn,
				"tick": tick,
				"triggers": self.pack(flist),
			},
		)

	def set_rule_prereqs(
		self, rule: str, branch: str, turn: int, tick: int, flist: List[str]
	):
		self.call(
			"insert1",
			"rule_prereqs",
			{
				"rule": rule,
				"branch": branch,
				"turn": turn,
				"tick": tick,
				"prereqs": self.pack(flist),
			},
		)

	def set_rule_actions(
		self, rule: str, branch: str, turn: int, tick: int, flist: List[str]
	):
		self.call(
			"insert1",
			"rule_actions",
			{
				"rule": rule,
				"branch": branch,
				"turn": turn,
				"tick": tick,
				"actions": self.pack(flist),
			},
		)

	def set_rule_neighborhood(
		self,
		rule: str,
		branch: str,
		turn: int,
		tick: int,
		neighborhood: Optional[int],
	):
		self.call(
			"insert1",
			"rule_neighborhood",
			{
				"rule": rule,
				"branch": branch,
				"turn": turn,
				"tick": tick,
				"neighborhood": self.pack(neighborhood),
			},
		)

	def set_rule(
		self,
		rule: str,
		branch: str,
		turn: int,
		tick: int,
		triggers: List[str],
		prereqs: List[str],
		actions: List[str],
		neighborhood: int,
	):
		with self._holder.lock:
			self.set_rule_triggers(rule, branch, turn, tick, triggers)
			self.set_rule_prereqs(rule, branch, turn, tick, prereqs)
			self.set_rule_actions(rule, branch, turn, tick, actions)
			self.set_rule_neighborhood(rule, branch, turn, tick, neighborhood)

	def set_rulebook(
		self,
		name: Key,
		branch: str,
		turn: int,
		tick: int,
		rules: List[str] = None,
		prio: float = 0.0,
	):
		pack = self.pack
		self.call(
			"insert1",
			"rulebooks",
			{
				"rulebook": pack(name),
				"branch": branch,
				"turn": turn,
				"tick": tick,
				"rules": pack(rules),
				"priority": prio,
			},
		)

	def _set_character_something_rulebook(
		self, tab: str, char: Key, branch: str, turn: int, tick: int, rb: Key
	):
		pack = self.pack
		self.call(
			"insert1",
			tab,
			{
				"character": pack(char),
				"branch": branch,
				"turn": turn,
				"tick": tick,
				"rulebook": pack(rb),
			},
		)

	def set_character_rulebook(
		self, char: Key, branch: str, turn: int, tick: int, rb: Key
	):
		self._set_character_something_rulebook(
			"character_rulebook", char, branch, turn, tick, rb
		)

	def set_unit_rulebook(
		self, char: Key, branch: str, turn: int, tick: int, rb: Key
	):
		self._set_character_something_rulebook(
			"unit_rulebook", char, branch, turn, tick, rb
		)

	def set_character_thing_rulebook(
		self, char: Key, branch: str, turn: int, tick: int, rb: Key
	):
		self._set_character_something_rulebook(
			"character_thing_rulebook", char, branch, turn, tick, rb
		)

	def set_character_place_rulebook(
		self, char: Key, branch: str, turn: int, tick: int, rb: Key
	):
		self._set_character_something_rulebook(
			"character_place_rulebook", char, branch, turn, tick, rb
		)

	def set_character_portal_rulebook(
		self, char: Key, branch: str, turn: int, tick: int, rb: Key
	):
		self._set_character_something_rulebook(
			"character_portal_rulebook", char, branch, turn, tick, rb
		)

	def rulebooks(self) -> Iterator[Key]:
		return map(self.pack, self.call("rulebooks"))

	def set_node_rulebook(
		self,
		character: Key,
		node: Key,
		branch: str,
		turn: int,
		tick: int,
		rulebook: Key,
	):
		pack = self.pack
		self.call(
			"insert1",
			"node_rulebook",
			{
				"character": pack(character),
				"node": pack(node),
				"branch": branch,
				"turn": turn,
				"tick": tick,
				"rulebook": pack(rulebook),
			},
		)

	def set_portal_rulebook(
		self,
		character: Key,
		orig: Key,
		dest: Key,
		branch: str,
		turn: int,
		tick: int,
		rulebook: Key,
	):
		pack = self.pack
		self.call(
			"insert1",
			"portal_rulebook",
			{
				"character": pack(character),
				"orig": pack(orig),
				"dest": pack(dest),
				"branch": branch,
				"turn": turn,
				"tick": tick,
				"rulebook": pack(rulebook),
			},
		)

	def handled_character_rule(
		self,
		character: Key,
		rulebook: Key,
		rule: str,
		branch: str,
		turn: int,
		tick: int,
	):
		pack = self.pack
		self._char_rules_handled.append(
			(pack(character), pack(rulebook), rule, branch, turn, tick)
		)

	def handled_unit_rule(
		self,
		character: Key,
		rulebook: Key,
		rule: str,
		graph: Key,
		unit: Key,
		branch: str,
		turn: int,
		tick: int,
	):
		pack = self.pack
		self._unit_rules_handled.append(
			(
				pack(character),
				pack(rulebook),
				rule,
				pack(graph),
				pack(unit),
				branch,
				turn,
				tick,
			)
		)

	def handled_character_thing_rule(
		self,
		character: Key,
		rulebook: Key,
		rule: str,
		thing: Key,
		branch: str,
		turn: int,
		tick: int,
	):
		pack = self.pack
		self._char_thing_rules_handled.append(
			(
				pack(character),
				pack(rulebook),
				rule,
				pack(thing),
				branch,
				turn,
				tick,
			)
		)

	def handled_character_place_rule(
		self,
		character: Key,
		rulebook: Key,
		rule: str,
		place: Key,
		branch: str,
		turn: int,
		tick: int,
	):
		pack = self.pack
		self._char_place_rules_handled.append(
			(
				pack(character),
				pack(rulebook),
				rule,
				pack(place),
				branch,
				turn,
				tick,
			)
		)

	def handled_character_portal_rule(
		self,
		character: Key,
		rulebook: Key,
		rule: str,
		orig: Key,
		dest: Key,
		branch: str,
		turn: int,
		tick: int,
	):
		pack = self.pack
		self._char_portal_rules_handled.append(
			(
				pack(character),
				pack(rulebook),
				rule,
				pack(orig),
				pack(dest),
				branch,
				turn,
				tick,
			)
		)

	def handled_node_rule(
		self,
		character: Key,
		node: Key,
		rulebook: Key,
		rule: str,
		branch: str,
		turn: int,
		tick: int,
	):
		pack = self.pack
		self._node_rules_handled.append(
			(
				pack(character),
				pack(node),
				pack(rulebook),
				rule,
				branch,
				turn,
				tick,
			)
		)

	def handled_portal_rule(
		self,
		character: Key,
		orig: Key,
		dest: Key,
		rulebook: Key,
		rule: str,
		branch: str,
		turn: int,
		tick: int,
	):
		pack = self.pack
		self._portal_rules_handled.append(
			(
				pack(character),
				pack(orig),
				pack(dest),
				pack(rulebook),
				rule,
				branch,
				turn,
				tick,
			)
		)

	def set_thing_loc(
		self,
		character: Key,
		thing: Key,
		branch: str,
		turn: int,
		tick: int,
		loc: Key,
	):
		pack = self.pack
		self._location.append(
			(pack(character), pack(thing), branch, turn, tick, pack(loc))
		)

	def unit_set(
		self,
		character: Key,
		graph: Key,
		node: Key,
		branch: str,
		turn: int,
		tick: int,
		isav: bool,
	):
		pack = self.pack
		self._unitness.append((pack(character), pack(graph), pack(node), branch, turn, tick, pack(isav)))

	def rulebook_set(
		self,
		rulebook: Key,
		branch: str,
		turn: int,
		tick: int,
		rules: List[str],
	):
		pack = self.pack
		self.call(
			"insert1",
			"rulebooks",
			dict(
				rulebook=pack(rulebook),
				branch=branch,
				turn=turn,
				tick=tick,
				rules=pack(rules),
			),
		)

	def turns_completed_dump(self) -> Iterator[Tuple[str, int]]:
		for d in self.call("dump", "turns_completed"):
			yield d["branch"], d["turn"]

	def complete_turn(
		self, branch: str, turn: int, discard_rules: bool = False
	):
		self.call("insert1", "turns_completed", dict(branch=branch, turn=turn))
		if discard_rules:
			self._char_rules_handled = []
			self._unit_rules_handled = []
			self._char_thing_rules_handled = []
			self._char_place_rules_handled = []
			self._char_portal_rules_handled = []
			self._node_rules_handled = []
			self._portal_rules_handled = []

	def new_turn(
		self, branch: str, turn: int, end_tick: int = 0, plan_end_tick: int = 0
	):
		self.call(
			"insert1",
			"turns",
			dict(
				branch=branch,
				turn=turn,
				end_tick=end_tick,
				plan_end_tick=plan_end_tick,
			),
		)

	def set_turn_completed(self, branch: str, turn: int):
		try:
			self.call(
				"insert1", "turns_completed", dict(branch=branch, turn=turn)
			)
		except ArrowInvalid:
			pass

	def graph_val_dump(self) -> Iterator[GraphValRowType]:
		unpack = self.unpack
		for d in self.call("dump", "graph_val"):
			yield (
				unpack(d["character"]),
				unpack(d["key"]),
				d["branch"],
				d["turn"],
				d["tick"],
				unpack(d["value"]),
			)

	def _flush_graph_val(self):
		if not self._graphvals2set:
			return
		todo = self._graphvals2set
		self._graphvals2set = []
		pack = self.pack
		self.call(
			"insert",
			"graph_val",
			[
				{
					"graph": pack(graph),
					"key": pack(key),
					"branch": branch,
					"turn": turn,
					"tick": tick,
					"value": pack(value),
				}
				for (graph, key, branch, turn, tick, value) in todo
			],
		)

	def load_graph_val(
		self,
		graph: Key,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int = None,
		tick_to: int = None,
	):
		if (turn_to is None) ^ (tick_to is None):
			raise ValueError("I need both or neither of turn_to and tick_to")
		self._flush_graph_val()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.call(
				"load_graph_val_tick_to_end",
				pack(graph),
				branch,
				turn_from,
				tick_from,
			)
		else:
			it = self.call(
				"load_graph_val_tick_to_tick",
				pack(graph),
				branch,
				turn_from,
				tick_from,
				turn_to,
				tick_to,
			)
		for key, turn, tick, value in it:
			yield graph, unpack(key), branch, turn, tick, unpack(value)

	def graph_val_set(
		self, graph: Key, key: Key, branch: str, turn: int, tick: int, val: Any
	):
		self._graphvals2set.append((graph, key, branch, turn, tick, val))

	def graph_val_del_time(self, branch: str, turn: int, tick: int):
		self.call("graph_val_del_time", branch, turn, tick)

	def graphs_dump(self) -> Iterator[Tuple[Key, str, int, int, str]]:
		unpack = self.unpack
		for d in self.call("dump", "graphs"):
			yield (
				unpack(d["graph"]),
				d["branch"],
				d["turn"],
				d["tick"],
				d["type"],
			)

	characters = characters_dump = graphs_dump

	def exist_node(
		self,
		graph: Key,
		node: Key,
		branch: str,
		turn: int,
		tick: int,
		extant: bool,
	):
		pack = self.pack
		self.call(
			"insert1",
			"nodes",
			dict(
				graph=pack(graph),
				node=pack(node),
				branch=branch,
				turn=turn,
				tick=tick,
				extant=extant,
			),
		)

	def nodes_del_time(self, branch: str, turn: int, tick: int):
		self.call("nodes_del_time", branch, turn, tick)

	def nodes_dump(self) -> Iterator[NodeRowType]:
		unpack = self.unpack
		for d in self.call("dump", "nodes"):
			yield (
				unpack(d["graph"]),
				unpack(d["node"]),
				d["branch"],
				d["turn"],
				d["tick"],
				d["extant"],
			)

	def _flush_nodes(self):
		if not self._nodes2set:
			return
		pack = self.pack
		nodes2set = self._nodes2set
		self._nodes2set = []
		self.call(
			"insert",
			"nodes",
			[
				{
					"graph": pack(graph),
					"node": pack(node),
					"branch": branch,
					"turn": turn,
					"tick": tick,
					"extant": bool(extant),
				}
				for (graph, node, branch, turn, tick, extant) in nodes2set
			],
		)

	def load_nodes(
		self,
		graph: str,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int = None,
		tick_to: int = None,
	):
		if (turn_to is None) ^ (tick_to is None):
			raise TypeError("I need both or neither of turn_to and tick_to")
		self._flush_nodes()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.call(
				"load_nodes_tick_to_end",
				pack(graph),
				branch,
				turn_from,
				tick_from,
			)
		else:
			it = self.call(
				"load_nodes_tick_to_tick",
				pack(graph),
				branch,
				turn_from,
				tick_from,
				turn_to,
				tick_to,
			)
		for node, turn, tick, extant in it:
			yield graph, unpack(node), branch, turn, tick, extant

	def node_val_dump(self) -> Iterator[NodeValRowType]:
		unpack = self.unpack
		for d in self.call("dump", "node_val"):
			yield (
				unpack(d["graph"]),
				unpack(d["node"]),
				unpack(d["key"]),
				d["branch"],
				d["turn"],
				d["tick"],
				unpack(d["value"]),
			)

	def _flush_node_val(self):
		if not self._nodevals2set:
			return
		nodevals2set = self._nodevals2set
		self._nodevals2set = []
		pack = self.pack
		self.call(
			"insert",
			"node_val",
			[
				{
					"graph": pack(graph),
					"node": pack(node),
					"key": pack(key),
					"branch": branch,
					"turn": turn,
					"tick": tick,
					"value": pack(value),
				}
				for (
					graph,
					node,
					key,
					branch,
					turn,
					tick,
					value,
				) in nodevals2set
			],
		)

	def load_node_val(
		self,
		graph: Key,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int = None,
		tick_to: int = None,
	) -> Iterator[NodeValRowType]:
		if (turn_to is None) ^ (tick_to is None):
			raise TypeError("I need both or neither of turn_to and tick_to")
		self._flush_node_val()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.call(
				"load_node_val_tick_to_end",
				pack(graph),
				branch,
				turn_from,
				tick_from,
			)
		else:
			it = self.call(
				"load_node_val_tick_to_tick",
				pack(graph),
				branch,
				turn_from,
				tick_from,
				turn_to,
				tick_to,
			)
		for node, key, turn, tick, value in it:
			yield (
				graph,
				unpack(node),
				unpack(key),
				branch,
				turn,
				tick,
				unpack(value),
			)

	def node_val_set(
		self,
		graph: Key,
		node: Key,
		key: Key,
		branch: str,
		turn: int,
		tick: int,
		value: Any,
	):
		self._nodevals2set.append(
			(graph, node, key, branch, turn, tick, value)
		)

	def node_val_del_time(self, branch: str, turn: int, tick: int):
		self.call("node_val_del_time", branch, turn, tick)

	def edges_dump(self) -> Iterator[EdgeRowType]:
		unpack = self.unpack
		for d in self.call("dump", "edges"):
			yield (
				unpack(d["graph"]),
				unpack(d["orig"]),
				unpack(d["dest"]),
				d["idx"],
				d["branch"],
				d["turn"],
				d["tick"],
				d["extant"],
			)

	def _flush_edges(self):
		if not self._edges2set:
			return
		todo = self._edges2set
		self._edges2set = []
		pack = self.pack
		self.call(
			"insert",
			"edges",
			[
				{
					"graph": pack(graph),
					"orig": pack(orig),
					"dest": pack(dest),
					"idx": 0,
					"branch": branch,
					"turn": turn,
					"tick": tick,
					"extant": bool(extant),
				}
				for (graph, orig, dest, branch, turn, tick, extant) in todo
			],
		)

	def load_edges(
		self,
		graph: Key,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int = None,
		tick_to: int = None,
	) -> Iterator[EdgeRowType]:
		if (turn_to is None) ^ (tick_to is None):
			raise ValueError("I need both or neither of turn_to and tick_to")
		self._flush_edges()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.call(
				"load_edges_tick_to_end",
				pack(graph),
				branch,
				turn_from,
				tick_from,
			)
		else:
			it = self.call(
				"load_edges_tick_to_tick",
				pack(graph),
				branch,
				turn_from,
				tick_from,
				turn_to,
				tick_to,
			)
		for orig, dest, idx, turn, tick, extant in it:
			yield (
				graph,
				unpack(orig),
				unpack(dest),
				idx,
				branch,
				turn,
				tick,
				extant,
			)

	def exist_edge(
		self,
		graph: Key,
		orig: Key,
		dest: Key,
		idx: int,
		branch: str,
		turn: int,
		tick: int,
		extant: bool,
	):
		self._edges2set.append((graph, orig, dest, branch, turn, tick, extant))

	def edges_del_time(self, branch: str, turn: int, tick: int):
		self.call("edges_del_time", branch, turn, tick)

	def edge_val_dump(self) -> Iterator[EdgeValRowType]:
		unpack = self.unpack
		for d in self.call("dump", "edge_val"):
			yield (
				unpack(d["character"]),
				unpack(d["orig"]),
				unpack(d["dest"]),
				d["idx"],
				d["branch"],
				d["turn"],
				d["tick"],
				unpack(d["value"]),
			)

	def _flush_edge_val(self):
		if not self._edgevals2set:
			return
		todo = self._edgevals2set
		self._edgevals2set = []
		pack = self.pack
		self.call(
			"insert",
			"edge_val",
			[
				{
					"graph": pack(graph),
					"orig": pack(orig),
					"dest": pack(dest),
					"idx": 0,
					"key": pack(key),
					"branch": branch,
					"turn": turn,
					"tick": tick,
					"value": pack(value),
				}
				for (graph, orig, dest, key, branch, turn, tick, value) in todo
			],
		)

	def load_edge_val(
		self,
		graph: Key,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int = None,
		tick_to: int = None,
	) -> Iterator[EdgeValRowType]:
		if (turn_to is None) ^ (tick_to is None):
			raise TypeError("I need both or neither of turn_to and tick_to")
		self._flush_edge_val()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.call(
				"load_edge_val_tick_to_end",
				pack(graph),
				branch,
				turn_from,
				tick_from,
			)
		else:
			it = self.call(
				"load_edge_val_tick_to_tick",
				pack(graph),
				branch,
				turn_from,
				tick_from,
				turn_to,
				tick_to,
			)
		for orig, dest, idx, key, turn, tick, value in it:
			yield (
				graph,
				unpack(orig),
				unpack(dest),
				idx,
				unpack(key),
				branch,
				turn,
				tick,
				unpack(value),
			)

	def edge_val_set(
		self,
		graph: Key,
		orig: Key,
		dest: Key,
		idx: int,
		key: Key,
		branch: str,
		turn: int,
		tick: int,
		value: Any,
	):
		self._edgevals2set.append(
			(graph, orig, dest, key, branch, turn, tick, value)
		)

	def edge_val_del_time(self, branch: str, turn: int, tick: int):
		self.call("edge_val_del_time", branch, turn, tick)

	def plans_dump(self) -> Iterator:
		for d in self.call("dump", "plans"):
			yield d["plan_id"], d["branch"], d["turn"], d["tick"]

	def plans_insert(self, plan_id: int, branch: str, turn: int, tick: int):
		self.call(
			"insert1",
			"plans",
			dict(plan_id=plan_id, branch=branch, turn=turn, tick=tick),
		)

	def plans_insert_many(self, many: List[Tuple[int, str, int, int]]):
		self.call(
			"insert",
			"plans",
			[
				dict(zip(("plan_id", "branch", "turn", "tick"), plan))
				for plan in many
			],
		)

	def plan_ticks_insert(self, plan_id: int, turn: int, tick: int):
		self.call(
			"insert1",
			"plan_ticks",
			dict(plan_id=plan_id, turn=turn, tick=tick),
		)

	def plan_ticks_insert_many(self, many: List[Tuple[int, int, int]]):
		self.call(
			"insert",
			"plan_ticks",
			[
				dict(zip(("plan_id", "turn", "tick"), plan_tick))
				for plan_tick in many
			],
		)

	def plan_ticks_dump(self) -> Iterator:
		for d in self.call("dump", "plan_ticks"):
			yield d["plan_id"], d["turn"], d["tick"]

	def truncate_all(self):
		self.call("truncate_all")

	def close(self):
		self.call("close")

	def commit(self):
		self.flush()
		self.call("commit")

	def initdb(self):
		self.call("initdb")


class ConnectionHolder(query.ConnectionHolder):
	def gather(self, meta):
		return gather_sql(meta)

	def initdb(self):
		"""Set up the database schema, both for allegedb and the special
		extensions for lisien

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
			"rule_big",
			"turns_completed",
			"keyframe_extensions",
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


class QueryEngine(query.QueryEngine, AbstractLiSEQueryEngine):
	exist_edge_t = 0
	path = lisien.__path__[0]
	IntegrityError = IntegrityError
	OperationalError = OperationalError
	holder_cls = ConnectionHolder
	tables = (
		"global",
		"branches",
		"turns",
		"graphs",
		"keyframes",
		"keyframe_extensions",
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
		self._new_keyframe_extensions = []
		self._char_rules_handled = []
		self._unit_rules_handled = []
		self._char_thing_rules_handled = []
		self._char_place_rules_handled = []
		self._char_portal_rules_handled = []
		self._node_rules_handled = []
		self._portal_rules_handled = []
		self._unitness = []
		self._location = []

	_infixes2load = [
		"nodes",
		"edges",
		"graph_val",
		"node_val",
		"edge_val",
		"things",
		"character_rulebook",
		"unit_rulebook",
		"character_thing_rulebook",
		"character_place_rulebook",
		"character_portal_rulebook",
		"node_rulebook",
		"portal_rulebook",
		"universals",
		"rulebooks",
		"rule_triggers",
		"rule_prereqs",
		"rule_actions",
		"rule_neighborhoods",
		"rule_big",
	]

	def load_windows(self, windows: list) -> dict:
		def empty_char():
			return {
				"nodes": [],
				"edges": [],
				"graph_val": [],
				"node_val": [],
				"edge_val": [],
				"things": [],
				"character_rulebook": [],
				"unit_rulebook": [],
				"character_thing_rulebook": [],
				"character_place_rulebook": [],
				"character_portal_rulebook": [],
				"node_rulebook": [],
				"portal_rulebook": [],
			}

		ret = defaultdict(empty_char)
		self._load_windows_into(ret, windows)
		return ret

	def _get_one_window(
		self, ret, branch, turn_from, tick_from, turn_to, tick_to
	):
		super()._get_one_window(
			ret, branch, turn_from, tick_from, turn_to, tick_to
		)
		unpack = self.unpack
		assert self._outq.get() == (
			"begin",
			"things",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)
		while isinstance(got := self._outq.get(), list):
			for graph, node, turn, tick, loc in got:
				(graph, node, loc) = map(unpack, (graph, node, loc))
				ret[graph]["things"].append(
					(graph, node, branch, turn, tick, loc)
				)
		assert got == (
			"end",
			"things",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		), got
		assert self._outq.get() == (
			"begin",
			"character_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)
		while isinstance(got := self._outq.get(), list):
			for graph, turn, tick, rb in got:
				(graph, rb) = map(unpack, (graph, rb))
				ret[graph]["character_rulebook"].append(
					(graph, branch, turn, tick, rb)
				)
		assert got == (
			"end",
			"character_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		), got
		assert self._outq.get() == (
			"begin",
			"unit_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)
		while isinstance(got := self._outq.get(), list):
			for graph, turn, tick, rb in got:
				(graph, rb) = map(unpack, (graph, rb))
				ret[graph]["unit_rulebook"].append(
					(graph, branch, turn, tick, rb)
				)
		assert got == (
			"end",
			"unit_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		), got
		assert self._outq.get() == (
			"begin",
			"character_thing_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)
		while isinstance(got := self._outq.get(), list):
			for graph, turn, tick, rb in got:
				(graph, rb) = map(unpack, (graph, rb))
				ret[graph]["character_thing_rulebook"].append(
					(graph, branch, turn, tick, rb)
				)
		assert got == (
			"end",
			"character_thing_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		), got
		assert self._outq.get() == (
			"begin",
			"character_place_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)
		while isinstance(got := self._outq.get(), list):
			for graph, turn, tick, rb in got:
				(graph, rb) = map(unpack, (graph, rb))
				ret[graph]["character_place_rulebook"].append(
					(graph, branch, turn, tick, rb)
				)
		assert got == (
			"end",
			"character_place_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		), got
		assert self._outq.get() == (
			"begin",
			"character_portal_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)
		while isinstance(got := self._outq.get(), list):
			for graph, turn, tick, rb in got:
				(graph, rb) = map(unpack, (graph, rb))
				ret[graph]["character_portal_rulebook"].append(
					(graph, branch, turn, tick, rb)
				)
		assert got == (
			"end",
			"character_portal_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		), got
		assert self._outq.get() == (
			"begin",
			"node_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)
		while isinstance(got := self._outq.get(), list):
			for graph, node, turn, tick, rb in got:
				(graph, node, rb) = map(unpack, (graph, node, rb))
				ret[graph]["node_rulebook"].append(
					(graph, node, branch, turn, tick, rb)
				)
		assert got == (
			"end",
			"node_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		), got
		assert self._outq.get() == (
			"begin",
			"portal_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)
		while isinstance(got := self._outq.get(), list):
			for graph, orig, dest, turn, tick, rb in got:
				(graph, orig, dest, rb) = map(unpack, (graph, orig, dest, rb))
				ret[graph]["portal_rulebook"].append(
					(graph, orig, dest, branch, turn, tick, rb)
				)
		assert got == (
			"end",
			"portal_rulebook",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		), got
		assert self._outq.get() == (
			"begin",
			"universals",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)
		while isinstance(got := self._outq.get(), list):
			for key, branch, turn, tick, val in got:
				(key, val) = map(unpack, (key, val))
				if "universals" in ret:
					ret["universals"].append((key, branch, turn, tick, val))
				else:
					ret["universals"] = [(key, branch, turn, tick, val)]
		assert got == (
			"end",
			"universals",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		), got
		assert self._outq.get() == (
			"begin",
			"rulebooks",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)
		while isinstance(got := self._outq.get(), list):
			for rulebook, branch, turn, tick, rules, priority in got:
				(rulebook, rules) = map(unpack, (rulebook, rules))
				if "rulebooks" in ret:
					ret["rulebooks"].append(
						(rulebook, branch, turn, tick, (rules, priority))
					)
				else:
					ret["rulebooks"] = [
						(rulebook, branch, turn, tick, (rules, priority))
					]
		assert got == (
			"end",
			"rulebooks",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		), got
		assert self._outq.get() == (
			"begin",
			"rule_triggers",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)
		while isinstance(got := self._outq.get(), list):
			for rule, branch, turn, tick, triggers in got:
				triggers = unpack(triggers)
				if "rule_triggers" in ret:
					ret["rule_triggers"].append(
						(rule, branch, turn, tick, triggers)
					)
				else:
					ret["rule_triggers"] = [
						(rule, branch, turn, tick, triggers)
					]
		assert got == (
			"end",
			"rule_triggers",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		), got
		assert self._outq.get() == (
			"begin",
			"rule_prereqs",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)
		while isinstance(got := self._outq.get(), list):
			for rule, branch, turn, tick, prereqs in got:
				prereqs = unpack(prereqs)
				if "rule_prereqs" in ret:
					ret["rule_prereqs"].append(
						(rule, branch, turn, tick, prereqs)
					)
				else:
					ret["rule_prereqs"] = [(rule, branch, turn, tick, prereqs)]
		assert got == (
			"end",
			"rule_prereqs",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		), got
		assert self._outq.get() == (
			"begin",
			"rule_actions",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)
		while isinstance(got := self._outq.get(), list):
			for rule, branch, turn, tick, actions in got:
				actions = unpack(actions)
				if "rule_actions" in ret:
					ret["rule_actions"].append(
						(rule, branch, turn, tick, actions)
					)
				else:
					ret["rule_actions"] = [(rule, branch, turn, tick, actions)]
		assert got == (
			"end",
			"rule_actions",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		), got
		assert self._outq.get() == (
			"begin",
			"rule_neighborhoods",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)
		while isinstance(got := self._outq.get(), list):
			for rule, branch, turn, tick, neighborhoods in got:
				neighborhoods = unpack(neighborhoods)
				if "rule_neighborhoods" in ret:
					ret["rule_neighborhoods"].append(
						(rule, branch, turn, tick, neighborhoods)
					)
				else:
					ret["rule_neighborhoods"] = [
						(rule, branch, turn, tick, neighborhoods)
					]
		assert got == (
			"end",
			"rule_neighborhoods",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		), got
		assert self._outq.get() == (
			"begin",
			"rule_big",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)
		while isinstance(got := self._outq.get(), list):
			for rule, branch, turn, tick, big in got:
				if "rule_big" in ret:
					ret["rule_big"].append((rule, branch, turn, tick, big))
				else:
					ret["rule_big"] = [(rule, branch, turn, tick, big)]
		assert got == (
			"end",
			"rule_big",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)

	def keyframe_extension_insert(
		self,
		branch,
		turn,
		tick,
		universal,
		rules,
		rulebooks,
	):
		pack = self.pack
		self._new_keyframe_extensions.append(
			(
				branch,
				turn,
				tick,
				pack(universal),
				pack(rules),
				pack(rulebooks),
			)
		)

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

	def _flush(self):
		super()._flush()
		put = self._inq.put
		if self._new_keyframe_extensions:
			put(
				(
					"silent",
					"many",
					"keyframe_extensions_insert",
					self._new_keyframe_extensions,
				)
			)
			self._new_keyframe_extensions = []
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

	def keyframe_extensions_dump(self):
		unpack = self.unpack
		for (
			branch,
			turn,
			tick,
			universal,
			rule,
			rulebook,
			neighborhood,
			big,
		) in self.call_one("keyframe_extensions_dump"):
			yield (
				branch,
				turn,
				tick,
				unpack(universal),
				unpack(rule),
				unpack(rulebook),
				unpack(neighborhood),
				unpack(big),
			)

	def get_keyframe_extensions(self, branch: str, turn: int, tick: int):
		self.flush()
		unpack = self.unpack
		try:
			universal, rule, rulebook = next(
				iter(
					self.call_one(
						"get_keyframe_extensions", branch, turn, tick
					)
				)
			)
		except StopIteration:
			raise KeyError("No keyframe", branch, turn, tick)
		return (
			unpack(universal),
			unpack(rule),
			unpack(rulebook),
		)

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
				tick_from,
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

	def set_rule_big(
		self, rule: str, branch: str, turn: int, tick: int, big: bool
	):
		self.call_one("rule_big_insert", rule, branch, turn, tick, big)
		self._increc()

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
		big=False,
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
		self.set_rule_big(rule, branch, turn, tick, big)

	def set_rulebook(self, name, branch, turn, tick, rules=None, prio=0.0):
		name, rules = map(self.pack, (name, rules or []))
		self.call_one(
			"rulebooks_insert", name, branch, turn, tick, rules, float(prio)
		)
		self._increc()

	def set_rulebook_on_character(self, rbtyp, char, branch, turn, tick, rb):
		char, rb = map(self.pack, (char, rb))
		self.call_one(rbtyp + "_insert", char, branch, turn, tick, rb)
		self._increc()

	set_character_rulebook = partialmethod(
		set_rulebook_on_character, "character_rulebook"
	)
	set_unit_rulebook = partialmethod(
		set_rulebook_on_character, "unit_rulebook"
	)
	set_character_thing_rulebook = partialmethod(
		set_rulebook_on_character, "character_thing_rulebook"
	)
	set_character_place_rulebook = partialmethod(
		set_rulebook_on_character, "character_place_rulebook"
	)
	set_character_portal_rulebook = partialmethod(
		set_rulebook_on_character, "character_portal_rulebook"
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
