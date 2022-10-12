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
"""The query engine provides Pythonic methods to access the database.

This module also contains a notably unfinished implementation of a query
language specific to LiSE. Access some stats using entities' method
``historical``, and do comparisons on those, and instead of a boolean
result you'll get a callable object that will return an iterator over
turn numbers in which the comparison evaluated to ``True``.

"""
import operator
from collections.abc import MutableMapping, Set
from itertools import chain
from operator import gt, lt, eq, ne, le, ge
from functools import partialmethod
from time import monotonic
from queue import Queue
from threading import Thread
from typing import Any, List, Callable, Tuple, Union

from sqlalchemy import select, and_, or_, not_, literal, Table
from sqlalchemy.sql.functions import func
from .alchemy import meta, gather_sql

from .allegedb import query
from .exc import (IntegrityError, OperationalError)
from .util import EntityStatAccessor
import LiSE


def windows_union(windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
	"""Given a list of (beginning, ending), return a minimal version that contains the same ranges.

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
		windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
	"""Given a list of (beginning, ending), return another describing where they overlap.

	Only ever returns one item, but puts it in a list anyway, to be like
	``windows_union``.

	:rtype: list
	"""

	def intersect2(left, right):
		if left == right:
			return left
		elif left == (None, None):
			return right
		elif right == (None, None):
			return left
		elif left[0] is None:
			if right[0] is None:
				return None, min((left[1], right[1]))
			elif right[1] is None:
				if left[1] <= right[0]:
					return left[1], right[0]
				else:
					return None
			elif right[0] <= left[1]:
				return right[0], left[1]
			else:
				return None
		elif left[1] is None:
			if right[0] is None:
				return left[0], right[1]
			elif left[0] <= right[0]:
				return right
			elif right[1] is None:
				return max((left[0], right[0])), None
			elif left[0] <= right[1]:
				return left[0], right[1]
			else:
				return None
		# None not in left
		elif right[0] is None:
			return left[0], min((left[1], right[1]))
		elif right[1] is None:
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

	if len(windows) == 0:
		return []
	elif len(windows) == 1:
		return windows

	done = []
	for window in windows:
		if not done:
			done.append(window)
			continue
		res = intersect2(done.pop(), window)
		if res:
			done.append(res)
	return done


def the_select(tab: Table, val_col='value'):
	return select(
		tab.c.turn.label('turn_from'), tab.c.tick.label('tick_from'),
		func.lead(tab.c.turn).over(order_by=(tab.c.turn,
												tab.c.tick)).label('turn_to'),
		func.lead(tab.c.tick).over(order_by=(tab.c.turn,
												tab.c.tick)).label('tick_to'),
		tab.c[val_col])


def make_graph_val_select(graph: bytes, stat: bytes, branches: List[str],
							mid_turn: bool):
	tab: Table = meta.tables['graph_val']
	ticksel = select(
		tab.c.graph, tab.c.key, tab.c.branch, tab.c.turn,
		tab.c.tick if mid_turn else func.max(tab.c.tick).label('tick')).where(
			and_(tab.c.graph == graph, tab.c.key == stat,
					tab.c.branch.in_(branches)))
	if not mid_turn:
		ticksel = ticksel.group_by(tab.c.graph, tab.c.key, tab.c.branch,
									tab.c.turn)
	return the_select(tab).select_from(
		tab.join(
			ticksel,
			and_(tab.c.graph == ticksel.c.graph, tab.c.key == ticksel.c.key,
					tab.c.branch == ticksel.c.branch,
					tab.c.turn == ticksel.c.turn,
					tab.c.tick == ticksel.c.tick)))


def make_node_val_select(graph: bytes, node: bytes, stat: bytes,
							branches: List[str], mid_turn: bool):
	tab: Table = meta.tables['node_val']
	ticksel = select(
		tab.c.graph, tab.c.node, tab.c.stat, tab.c.branch, tab.c.turn,
		tab.c.tick if mid_turn else func.max(tab.c.tick).label('tick')).where(
			and_(tab.c.graph == graph, tab.c.node == node, tab.c.stat == stat,
					tab.c.branch.in_(branches)))
	if not mid_turn:
		ticksel = ticksel.group_by(tab.c.graph, tab.c.node, tab.c.stat,
									tab.c.branch, tab.c.turn)
	return the_select(tab).select_from(
		tab.join(
			ticksel,
			and_(tab.c.graph == ticksel.c.graph, tab.c.node == ticksel.c.node,
					tab.c.stat == ticksel.c.stat,
					tab.c.branch == ticksel.c.branch,
					tab.c.turn == ticksel.c.turn,
					tab.c.tick == ticksel.c.tick)))


def make_location_select(graph: bytes, thing: bytes, branches: List[str],
							mid_turn: bool):
	tab: Table = meta.tables['things']
	ticksel = select(
		tab.c.character, tab.c.thing, tab.c.branch, tab.c.turn,
		tab.c.tick if mid_turn else func.max(tab.c.tick).label('tick')).where(
			and_(tab.c.character == graph, tab.c.thing == thing,
					tab.c.branch.in_(branches)))
	if not mid_turn:
		ticksel = ticksel.group_by(tab.c.character, tab.c.thing, tab.c.branch,
									tab.c.turn)
	return the_select(tab, val_col='location').select_from(
		tab.join(
			ticksel,
			and_(tab.c.character == ticksel.c.character,
					tab.c.thing == ticksel.c.thing,
					tab.c.branch == ticksel.c.branch,
					tab.c.turn == ticksel.c.turn,
					tab.c.tick == ticksel.c.tick)))


def make_edge_val_select(graph: bytes, orig: bytes, dest: bytes, idx: int,
							stat: bytes, branches: List[str], mid_turn: bool):
	tab: Table = meta.tables['edge_val']
	ticksel = select(
		tab.c.graph, tab.c.orig, tab.c.dest, tab.c.idx, tab.c.stat,
		tab.c.branch, tab.c.turn,
		tab.c.tick if mid_turn else func.max(tab.c.tick).label('tick')).where(
			and_(tab.c.graph == graph, tab.c.orig == orig, tab.c.dest == dest,
					tab.c.idx == idx, tab.c.stat == stat,
					tab.c.branch.in_(branches)))
	if not mid_turn:
		ticksel = ticksel.group_by(tab.c.graph, tab.c.orig, tab.c.dest,
									tab.c.idx, tab.c.stat, tab.c.branch,
									tab.c.turn)
	return the_select(tab).select_from(
		tab.join(
			ticksel,
			and_(tab.c.graph == ticksel.c.graph, tab.c.orig == ticksel.c.orig,
					tab.c.dest == ticksel.c.dest, tab.c.idx == ticksel.c.idx,
					tab.c.stat == ticksel.c.stat,
					tab.c.branch == ticksel.c.branch,
					tab.c.turn == ticksel.c.turn,
					tab.c.tick == ticksel.c.tick)))


def make_side_sel(entity, stat, branches: List[str], pack: callable,
					mid_turn: bool):
	from .character import AbstractCharacter
	from .place import Place
	from .thing import Thing
	from .portal import Portal
	if isinstance(entity, AbstractCharacter):
		return make_graph_val_select(pack(entity.name), pack(stat), branches,
										mid_turn)
	elif isinstance(entity, Place):
		return make_node_val_select(pack(entity.character.name),
									pack(entity.name), pack(stat), branches,
									mid_turn)
	elif isinstance(entity, Thing):
		if stat == 'location':
			return make_location_select(pack(entity.character.name),
										pack(entity.name), branches, mid_turn)
		else:
			return make_node_val_select(pack(entity.character.name),
										pack(entity.name), pack(stat),
										branches, mid_turn)
	elif isinstance(entity, Portal):
		return make_edge_val_select(pack(entity.character.name),
									pack(entity.origin.name),
									pack(entity.destination.name), 0,
									pack(stat), branches, mid_turn)
	else:
		raise TypeError(f"Unknown entity type {type(entity)}")


def _msfq_mid_turn(qry,
					left_sel,
					right_sel,
					left_col='value',
					right_col='value'):
	# figure whether there is *no* overlap between the time ranges
	left_time_to_lt_right_time_from = and_(
		left_sel.c.turn_to != None, right_sel.c.turn_from != None,
		or_(
			left_sel.c.turn_to - 1 < right_sel.c.turn_from,
			and_(left_sel.c.turn_to == right_sel.c.turn_from,
					left_sel.c.tick_to - 1 < right_sel.c.tick_from)))
	right_time_to_lt_left_time_from = and_(
		right_sel.c.turn_to != None, left_sel.c.turn_from != None,
		or_(
			right_sel.c.turn_to - 1 < left_sel.c.turn_from,
			and_(right_sel.c.turn_to == right_sel.c.turn_from,
					right_sel.c.tick_to - 1 < left_sel.c.tick_from)))
	# then invert it
	join = left_sel.alias().join(
		right_sel.alias(),
		not_(
			or_(left_time_to_lt_right_time_from,
				right_time_to_lt_left_time_from)))
	return select(left_sel.c.turn_from, left_sel.c.tick_from,
					left_sel.c.turn_to, left_sel.c.tick_to,
					right_sel.c.turn_from, right_sel.c.tick_from,
					right_sel.c.turn_to,
					right_sel.c.tick_to).distinct().select_from(join).where(
						qry.oper(left_sel.c[left_col], right_sel.c[right_col]))


def _msfq_end_turn(qry,
					left_sel,
					right_sel,
					left_col='value',
					right_col='value'):
	# intervals in LiSE that are open on the left are always also open on the
	# right; therefore, when either interval is open on the left, there is an
	# overlap
	left_time_to_lt_right_time_from = and_(
		left_sel.c.turn_to != None, right_sel.c.turn_from != None,
		left_sel.c.turn_to - 1 < right_sel.c.turn_from)
	right_time_to_lt_left_time_from = and_(
		right_sel.c.turn_to != None, left_sel.c.turn_from != None,
		right_sel.c.turn_to - 1 < left_sel.c.turn_from)
	join = left_sel.alias().join(
		right_sel.alias(),
		not_(
			or_(left_time_to_lt_right_time_from,
				right_time_to_lt_left_time_from)))
	return select(left_sel.c.turn_from, left_sel.c.turn_to,
					right_sel.c.turn_from,
					right_sel.c.turn_to).distinct().select_from(join).where(
						qry.oper(left_sel.c[left_col], right_sel.c[right_col]))


def _getcol(alias: "StatusAlias"):
	from .thing import Thing
	if isinstance(alias.entity, Thing) and alias.stat == 'location':
		return 'location'
	return 'value'


def make_select_from_eq_query(qry: Union["EqQuery",
											"NeQuery"], branches: List[str],
								pack: callable, mid_turn: bool):
	left = qry.leftside
	right = qry.rightside
	if isinstance(left, StatusAlias) and isinstance(right, StatusAlias):
		left_sel = make_side_sel(left.entity, left.stat, branches, pack,
									mid_turn)
		right_sel = make_side_sel(right.entity, right.stat, branches, pack,
									mid_turn)
		if mid_turn:
			return _msfq_mid_turn(qry, left_sel, right_sel, _getcol(left),
									_getcol(right))
		else:
			return _msfq_end_turn(qry, left_sel, right_sel, _getcol(left),
									_getcol(right))

	elif isinstance(right, StatusAlias):
		right_sel = make_side_sel(right.entity, right.stat, branches, pack,
									mid_turn)
		if mid_turn:
			return select(right_sel.c.turn_from, right_sel.c.tick_from,
							right_sel.c.turn_to, right_sel.c.tick_to).where(
								qry.oper(pack(left),
											right_sel.c[_getcol(right)]))
		else:
			return select(right_sel.c.turn_from, right_sel.c.turn_to).where(
				qry.oper(pack(left), right_sel.c[_getcol(right)]))

	elif isinstance(left, StatusAlias):
		left_sel = make_side_sel(left.entity, left.stat, branches, pack,
									mid_turn)
		if mid_turn:
			return select(left_sel.c.turn_from, left_sel.c.tick_from,
							left_sel.c.turn_to, left_sel.c.tick_to).where(
								qry.oper(left_sel.c[_getcol(left)],
											pack(right)))
		else:
			return select(left_sel.c.turn_from, left_sel.c.turn_to).where(
				qry.oper(left_sel.c[_getcol(left)], pack(right)))
	else:
		return select(literal(left) == literal(right))


class QueryResult(Set):
	pass


class EqNeQueryResultEndTurn(QueryResult):

	def __init__(self, windows):
		self._past = windows
		self._future = []
		self._trues = set()
		self._falses = set()
		self._iterated = False

	def __iter__(self):
		if self._iterated:
			yield from self._trues
			return
		add = self._trues.add
		for turn_from, turn_to in self._past:
			for turn in range(turn_from, turn_to):
				add(turn)
				yield turn
		for turn_from, turn_to in reversed(self._future):
			for turn in range(turn_from, turn_to):
				add(turn)
				yield turn
		self._iterated = True
		del self._falses

	def __len__(self):
		if self._iterated:
			return len(self._trues)
		n = 0
		for _ in self:
			n += 1
		return n

	def __contains__(self, item):
		if self._iterated:
			return item in self._trues
		elif item in self._trues:
			return True
		elif item in self._falses:
			return False
		past = self._past
		future = self._future
		if not past:
			if not future:
				return False
			past.append(future.pop())
		while past and item < past[0][0]:
			future.append(past.pop())
		while future and item >= past[0][1]:
			past.append(future.pop())
		if not past:
			return False
		ret = past[0][0] <= item < past[0][1]
		if ret:
			self._trues.add(item)
		else:
			self._falses.add(item)
		return ret

	def __repr__(self):
		return f"{self.__class__}({self._past + list(reversed(self._future))})"


class EqNeQueryResultMidTurn(QueryResult):

	def __init__(self, windows):
		self._past = windows
		self._future = []
		self._trues = set()
		self._falses = set()
		self._iterated = False

	def __iter__(self):
		if self._iterated:
			yield from self._trues
			return
		seen = set()
		add = self._trues.add
		for seq in (self._past, reversed(self._future)):
			for (turn_from, tick_from), (turn_to, tick_to) in seq:
				if turn_from == turn_to:
					if turn_from not in seen:
						add(turn_from)
						yield turn_from
						seen.add(turn_from)
				else:
					for turn in range(turn_from, turn_to):
						if turn not in seen:
							add(turn)
							yield turn
							seen.add(turn)
		self._iterated = True
		del self._falses

	def __len__(self):
		if self._iterated:
			return len(self._trues)
		n = 0
		for _ in self:
			n += 1
		return n

	def __contains__(self, item):
		if self._iterated:
			return item in self._trues
		elif item in self._trues:
			return True
		elif item in self._falses:
			return False
		past = self._past
		future = self._future
		if not past:
			if not future:
				return False
			future.append(past.pop())
		while item < past[0][0][0]:
			if not past:
				return False
			future.append(past.pop())
		while item >= past[0][1][0]:
			if not future:
				return False
			past.append(future.pop())
		ret = past[0][0][0] <= item < past[0][1][0]
		if ret:
			self._trues.add(item)
		else:
			self._falses.add(item)
		return ret


class GtLtQueryResultEndTurn(QueryResult):

	def __init__(self, windows_l, windows_r, oper):
		self._past_l = windows_l
		self._future_l = []
		self._past_r = windows_r
		self._future_r = []
		self._oper = oper
		self._iterated = False
		self._trues = set()
		self._falses = set()

	def __iter__(self):
		if self._iterated:
			yield from self._trues
			return
		if not ((self._past_l or self._future_l) and
				(self._past_r or self._future_r)):
			return
		if self._future_l:
			end_l = self._future_l[0][1]
			if end_l in (None, (None, None)):
				end_l = self._future_l[0][0]
		else:
			end_l = self._past_l[-1][1]
			if end_l in (None, (None, None)):
				end_l = self._past_l[-1][0]
		if self._future_r:
			end_r = self._future_r[0][1]
			if end_r in (None, (None, None)):
				end_r = self._future_r[0][0]
		else:
			end_r = self._past_r[-1][1]
			if end_r in (None, (None, None)):
				end_r = self._past_r[-1][0]
		end = max((end_l, end_r))
		if isinstance(end, tuple):
			end = end[0]
		oper = self._oper
		trues = self._trues
		left = chain(iter(self._past_l), reversed(self._future_l))
		right = chain(iter(self._past_r), reversed(self._future_r))
		(l_from, _), (l_to, _), l_v = next(left)
		(r_from, _), (r_to, _), r_v = next(right)
		for turn in range(0, end + 1):
			while not (l_from <= turn and (l_to is None or turn < l_to)):
				try:
					(l_from, _), (l_to, _), l_v = next(left)
				except StopIteration:
					return
			while not (r_from <= turn and (r_to is None or turn < r_to)):
				try:
					(r_from, _), (r_to, _), r_v = next(right)
				except StopIteration:
					return
			if oper(l_v, r_v):
				trues.add(turn)
				yield turn
		self._iterated = True
		del self._falses

	def __len__(self):
		if self._iterated:
			return len(self._trues)
		n = 0
		for _ in self:
			n += 1
		return n

	def __contains__(self, item):
		if self._iterated:
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
		while past_l and past_l[-1][0] >= item:
			future_l.append(past_l.pop())
		while future_l and future_l[-1][0] < item:
			past_l.append(future_l.pop())
		while past_r and past_r[-1][0] >= item:
			future_r.append(past_r.pop())
		while future_r and future_r[-1][0] < item:
			past_r.append(future_r.pop())
		ret = self._oper(past_l[-1][2], past_r[-1][2])
		if ret:
			self._trues.add(item)
		else:
			self._falses.add(item)
		return ret


class GtLtQueryResultMidTurn(QueryResult):

	def __init__(self, windows_l, windows_r, oper):
		self._past_l = windows_l
		self._future_l = []
		self._past_r = windows_r
		self._future_r = []
		self._oper = oper
		self._iterated = False
		self._trues = set()
		self._falses = set()

	def __iter__(self):
		if self._iterated:
			yield from self._trues
			return
		if not ((self._past_l or self._future_l) and
				(self._past_r or self._future_r)):
			return
		add = self._trues.add
		future_l = self._future_l
		future_l.extend(reversed(self._past_l))
		self._past_l = past_l = []
		future_r = self._future_r
		future_r.extend(reversed(self._past_r))
		self._past_r = past_r = []
		oper = self._oper
		end = max((future_r[0][1][0], future_l[0][1][0]))
		past_l.append(future_l.pop())
		past_r.append(future_r.pop())
		for i in range(0, end):
			while not (past_l[-1][0][0] <= i < past_l[-1][1][0]
						or past_l[-1][0][0] == i == past_l[-1][1][0]):
				past_l.append(future_l.pop())
			while not (past_r[-1][0][0] <= i < past_r[-1][1][0]
						or past_r[-1][0][0] == i == past_r[-1][1][0]):
				past_r.append(future_r.pop())
			v_l = past_l[-1][-1]
			v_r = past_r[-1][-1]
			if oper(v_l, v_r):
				add(i)
				yield i
		self._iterated = True
		del self._falses

	def __len__(self):
		if self._iterated:
			return len(self._trues)
		n = 0
		for _ in self:
			n += 1
		self._len = n
		return n

	def __contains__(self, item):
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
		oper = self._oper
		while past_l and past_l[-1][0][0] >= item:
			future_l.append(past_l.pop())
		while future_l and not (past_l[-1][0][0] <= item < past_l[-1][1][0] or
								past_l[-1][0][0] == item == past_l[-1][1][0]):
			past_l.append(future_l.pop())
		left_candidates = [past_l[-1]]
		while future_l and (past_l[-1][0][0] <= item < past_l[-1][1][0]
							or past_l[-1][0][0] == item == past_l[-1][1][0]):
			past_l.append(future_l.pop())
			left_candidates.append(past_l[-1])
		while past_r and past_r[-1][0][0] >= item:
			future_r.append(past_r.pop())
		while future_r and not (past_r[-1][0][0] <= item < past_r[-1][1][0] or
								past_r[-1][0][0] == item == past_r[-1][1][0]):
			past_r.append(future_r.pop())
		right_candidates = [past_r[-1]]
		while future_r and (past_r[-1][0][0] <= item < past_r[-1][1][0]
							or past_r[-1][0][0] == item == past_r[-1][1][0]):
			past_r.append(future_r.pop())
			right_candidates.append(past_r[-1])
		for l_time_from, l_time_to, l_v in left_candidates:
			for r_time_from, r_time_to, r_v in right_candidates:
				if not (r_time_to < l_time_from or r_time_from > l_time_to):
					if oper(l_v, r_v):
						return True
		return False


class CombinedQueryResult(QueryResult):

	def __init__(self, left: QueryResult, right: QueryResult, oper):
		self._left = left
		self._right = right
		self._oper = oper

	def _genset(self):
		if not hasattr(self, '_set'):
			self._set = self._oper(set(self._left), set(self._right))

	def __iter__(self):
		self._genset()
		return iter(self._set)

	def __len__(self):
		self._genset()
		return len(self._set)

	def __contains__(self, item):
		if hasattr(self, '_set'):
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

	def iter_times(self):
		raise NotImplementedError

	def iter_ticks(self, turn):
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

	def iter_times(self):
		return slow_iter_turns_eval_cmp(self, self.oper, engine=self.engine)


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
	'eq': EqQuery,
	'ne': NeQuery,
	'gt': GtQuery,
	'lt': LtQuery,
	'ge': GeQuery,
	'le': LeQuery
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


def slow_iter_turns_eval_cmp(qry, oper, start_branch=None, engine=None):
	"""Iterate over all turns on which a comparison holds.

	This is expensive. It evaluates the query for every turn in history.

	"""

	def mungeside(side):
		if isinstance(side, Query):
			return side.iter_times
		elif isinstance(side, StatusAlias):
			return EntityStatAccessor(side.entity, side.stat, side.engine,
										side.branch, side.turn, side.tick,
										side.current, side.mungers)
		elif isinstance(side, EntityStatAccessor):
			return side
		else:
			return lambda: side

	leftside = mungeside(qry.leftside)
	rightside = mungeside(qry.rightside)
	engine = engine or leftside.engine or rightside.engine

	for (branch, fork_turn,
			fork_tick) in engine._iter_parent_btt(start_branch
													or engine.branch):
		if branch is None:
			return
		parent, turn_start, tick_start, turn_end, tick_end = engine._branches[
			branch]
		for turn in range(turn_start, fork_turn + 1):
			if oper(leftside(branch, turn), rightside(branch, turn)):
				yield branch, turn


class ConnectionHolder(query.ConnectionHolder):

	def gather(self, meta):
		return gather_sql(meta)

	def initdb(self):
		"""Set up the database schema, both for allegedb and the special
		extensions for LiSE

		"""
		super().initdb()
		init_table = self.init_table
		for table in ('universals', 'rules', 'rulebooks', 'things',
						'character_rulebook', 'unit_rulebook',
						'character_thing_rulebook', 'character_place_rulebook',
						'character_portal_rulebook', 'node_rulebook',
						'portal_rulebook', 'units', 'character_rules_handled',
						'unit_rules_handled', 'character_thing_rules_handled',
						'character_place_rules_handled',
						'character_portal_rules_handled', 'node_rules_handled',
						'portal_rules_handled', 'rule_triggers',
						'rule_prereqs', 'rule_actions', 'turns_completed'):
			try:
				init_table(table)
			except OperationalError:
				pass
			except Exception as ex:
				return ex


class QueryEngine(query.QueryEngine):
	exist_edge_t = 0
	path = LiSE.__path__[0]
	IntegrityError = IntegrityError
	OperationalError = OperationalError
	holder_cls = ConnectionHolder
	tables = ('global', 'branches', 'turns', 'graphs', 'keyframes',
				'graph_val', 'nodes', 'node_val', 'edges', 'edge_val', 'plans',
				'plan_ticks', 'universals', 'rules', 'rulebooks',
				'rule_triggers', 'rule_prereqs', 'rule_actions',
				'character_rulebook', 'unit_rulebook',
				'character_thing_rulebook', 'character_place_rulebook',
				'character_portal_rulebook', 'node_rules_handled',
				'portal_rules_handled', 'things', 'node_rulebook',
				'portal_rulebook', 'units', 'character_rules_handled',
				'unit_rules_handled', 'character_thing_rules_handled',
				'character_place_rules_handled',
				'character_portal_rules_handled', 'turns_completed')

	def __init__(self, dbstring, connect_args, pack=None, unpack=None):
		super().__init__(dbstring,
							connect_args,
							pack,
							unpack,
							gather=gather_sql)
		self._char_rules_handled = []
		self._unit_rules_handled = []
		self._char_thing_rules_handled = []
		self._char_place_rules_handled = []
		self._char_portal_rules_handled = []
		self._node_rules_handled = []
		self._portal_rules_handled = []

	def flush(self):
		super().flush()
		put = self._inq.put
		if self._char_rules_handled:
			put(('silent', 'many', 'character_rules_handled_insert',
					self._char_rules_handled))
			self._char_rules_handled = []
		if self._unit_rules_handled:
			put(('silent', 'many', 'unit_rules_handled_insert',
					self._unit_rules_handled))
			self._unit_rules_handled = []
		if self._char_thing_rules_handled:
			put(('silent', 'many', 'character_thing_rules_handled_insert',
					self._char_thing_rules_handled))
			self._char_thing_rules_handled = []
		if self._char_place_rules_handled:
			put(('silent', 'many', 'character_place_rules_handled_insert',
					self._char_place_rules_handled))
			self._char_place_rules_handled = []
		if self._char_portal_rules_handled:
			put(('silent', 'many', 'character_portal_rules_handled_insert',
					self._char_portal_rules_handled))
			self._char_portal_rules_handled = []
		if self._node_rules_handled:
			put(('silent', 'many', 'node_rules_handled_insert',
					self._node_rules_handled))
			self._node_rules_handled = []
		if self._portal_rules_handled:
			put(('silent', 'many', 'portal_rules_handled_insert',
					self._portal_rules_handled))
			self._portal_rules_handled = []

	def universals_dump(self):
		unpack = self.unpack
		for key, branch, turn, tick, value in self.call_one('universals_dump'):
			yield unpack(key), branch, turn, tick, unpack(value)

	def rulebooks_dump(self):
		unpack = self.unpack
		for rulebook, branch, turn, tick, rules in self.call_one(
			'rulebooks_dump'):
			yield unpack(rulebook), branch, turn, tick, unpack(rules)

	def _rule_dump(self, typ):
		unpack = self.unpack
		for rule, branch, turn, tick, lst in self.call_one(
			'rule_{}_dump'.format(typ)):
			yield rule, branch, turn, tick, unpack(lst)

	def rule_triggers_dump(self):
		return self._rule_dump('triggers')

	def rule_prereqs_dump(self):
		return self._rule_dump('prereqs')

	def rule_actions_dump(self):
		return self._rule_dump('actions')

	def characters_dump(self):
		unpack = self.unpack
		for graph, typ in self.call_one('graphs_dump'):
			if typ == 'DiGraph':
				yield unpack(graph)

	characters = characters_dump

	def node_rulebook_dump(self):
		unpack = self.unpack
		for character, node, branch, turn, tick, rulebook in self.call_one(
			'node_rulebook_dump'):
			yield unpack(character), unpack(node), branch, turn, tick, unpack(
				rulebook)

	def portal_rulebook_dump(self):
		unpack = self.unpack
		for character, orig, dest, branch, turn, tick, rulebook in self.call_one(
			'portal_rulebook_dump'):
			yield (unpack(character), unpack(orig), unpack(dest), branch, turn,
					tick, unpack(rulebook))

	def _charactery_rulebook_dump(self, qry):
		unpack = self.unpack
		for character, branch, turn, tick, rulebook in self.call_one(
			qry + '_rulebook_dump'):
			yield unpack(character), branch, turn, tick, unpack(rulebook)

	character_rulebook_dump = partialmethod(_charactery_rulebook_dump,
											'character')
	unit_rulebook_dump = partialmethod(_charactery_rulebook_dump, 'unit')
	character_thing_rulebook_dump = partialmethod(_charactery_rulebook_dump,
													'character_thing')
	character_place_rulebook_dump = partialmethod(_charactery_rulebook_dump,
													'character_place')
	character_portal_rulebook_dump = partialmethod(_charactery_rulebook_dump,
													'character_portal')

	def character_rules_handled_dump(self):
		unpack = self.unpack
		for character, rulebook, rule, branch, turn, tick in self.call_one(
			'character_rules_handled_dump'):
			yield unpack(character), unpack(rulebook), rule, branch, turn, tick

	def character_rules_changes_dump(self):
		unpack = self.unpack
		for (character, rulebook, rule, branch, turn, tick, handled_branch,
				handled_turn) in self.call_one('character_rules_changes_dump'):
			yield (unpack(character), unpack(rulebook), rule, branch, turn,
					tick, handled_branch, handled_turn)

	def unit_rules_handled_dump(self):
		unpack = self.unpack
		for character, graph, unit, rulebook, rule, branch, turn, tick in self.call_one(
			'unit_rules_handled_dump'):
			yield (unpack(character), unpack(graph), unpack(unit),
					unpack(rulebook), rule, branch, turn, tick)

	def unit_rules_changes_dump(self):
		jl = self.unpack
		for (character, rulebook, rule, graph, unit, branch, turn, tick,
				handled_branch,
				handled_turn) in self.call_one('unit_rules_changes_dump'):
			yield (jl(character), jl(rulebook), rule, jl(graph), jl(unit),
					branch, turn, tick, handled_branch, handled_turn)

	def character_thing_rules_handled_dump(self):
		unpack = self.unpack
		for character, thing, rulebook, rule, branch, turn, tick in self.call_one(
			'character_thing_rules_handled_dump'):
			yield unpack(character), unpack(thing), unpack(
				rulebook), rule, branch, turn, tick

	def character_thing_rules_changes_dump(self):
		jl = self.unpack
		for (character, thing, rulebook, rule, branch, turn, tick,
				handled_branch, handled_turn
				) in self.call_one('character_thing_rules_changes_dump'):
			yield (jl(character), jl(thing), jl(rulebook), rule, branch, turn,
					tick, handled_branch, handled_turn)

	def character_place_rules_handled_dump(self):
		unpack = self.unpack
		for character, place, rulebook, rule, branch, turn, tick in self.call_one(
			'character_place_rules_handled_dump'):
			yield unpack(character), unpack(place), unpack(
				rulebook), rule, branch, turn, tick

	def character_place_rules_changes_dump(self):
		jl = self.unpack
		for (character, rulebook, rule, place, branch, turn, tick,
				handled_branch, handled_turn
				) in self.call_one('character_place_rules_changes_dump'):
			yield (jl(character), jl(rulebook), rule, jl(place), branch, turn,
					tick, handled_branch, handled_turn)

	def character_portal_rules_handled_dump(self):
		unpack = self.unpack
		for character, rulebook, rule, orig, dest, branch, turn, tick in self.call_one(
			'character_portal_rules_handled_dump'):
			yield (unpack(character), unpack(rulebook), unpack(orig),
					unpack(dest), rule, branch, turn, tick)

	def character_portal_rules_changes_dump(self):
		jl = self.unpack
		for (character, rulebook, rule, orig, dest, branch, turn, tick,
				handled_branch, handled_turn
				) in self.call_one('character_portal_rules_changes_dump'):
			yield (jl(character), jl(rulebook), rule, jl(orig), jl(dest),
					branch, turn, tick, handled_branch, handled_turn)

	def node_rules_handled_dump(self):
		for character, node, rulebook, rule, branch, turn, tick in self.call_one(
			'node_rules_handled_dump'):
			yield self.unpack(character), self.unpack(node), self.unpack(
				rulebook), rule, branch, turn, tick

	def node_rules_changes_dump(self):
		jl = self.unpack
		for (character, node, rulebook, rule, branch, turn, tick,
				handled_branch,
				handled_turn) in self.call_one('node_rules_changes_dump'):
			yield (jl(character), jl(node), jl(rulebook), rule, branch, turn,
					tick, handled_branch, handled_turn)

	def portal_rules_handled_dump(self):
		unpack = self.unpack
		for character, orig, dest, rulebook, rule, branch, turn, tick in self.call_one(
			'portal_rules_handled_dump'):
			yield (unpack(character), unpack(orig), unpack(dest),
					unpack(rulebook), rule, branch, turn, tick)

	def portal_rules_changes_dump(self):
		jl = self.unpack
		for (character, orig, dest, rulebook, rule, branch, turn, tick,
				handled_branch,
				handled_turn) in self.call_one('portal_rules_changes_dump'):
			yield (jl(character), jl(orig), jl(dest), jl(rulebook), rule,
					branch, turn, tick, handled_branch, handled_turn)

	def senses_dump(self):
		unpack = self.unpack
		for character, sense, branch, turn, tick, function in self.call_one(
			'senses_dump'):
			yield unpack(character), sense, branch, turn, tick, function

	def things_dump(self):
		unpack = self.unpack
		for character, thing, branch, turn, tick, location in self.call_one(
			'things_dump'):
			yield (unpack(character), unpack(thing), branch, turn, tick,
					unpack(location))

	def load_things(self,
					character,
					branch,
					turn_from,
					tick_from,
					turn_to=None,
					tick_to=None):
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			if tick_to is not None:
				raise ValueError("Need both or neither of turn_to, tick_to")
			for thing, turn, tick, location in self.call_one(
				'load_things_tick_to_end', pack(character), branch, turn_from,
				turn_from, tick_from):
				yield character, unpack(thing), branch, turn, tick, unpack(
					location)
		else:
			if tick_to is None:
				raise ValueError("Need both or neither of turn_to, tick_to")
			for thing, turn, tick, location in self.call_one(
				'load_things_tick_to_tick', pack(character), branch, turn_from,
				turn_from, tick_from, turn_to, turn_to, tick_to):
				yield character, unpack(thing), branch, turn, tick, unpack(
					location)

	def units_dump(self):
		unpack = self.unpack
		for character_graph, unit_graph, unit_node, branch, turn, tick, is_av in self.call_one(
			'units_dump'):
			yield (unpack(character_graph), unpack(unit_graph),
					unpack(unit_node), branch, turn, tick, is_av)

	def universal_set(self, key, branch, turn, tick, val):
		key, val = map(self.pack, (key, val))
		self.call_one('universals_insert', key, branch, turn, tick, val)

	def universal_del(self, key, branch, turn, tick):
		key = self.pack(key)
		self.call_one('universals_insert', key, branch, turn, tick, None)

	def comparison(self,
					entity0,
					stat0,
					entity1,
					stat1=None,
					oper='eq',
					windows: list = None):
		if windows is None:
			windows = []
		stat1 = stat1 or stat0
		return comparisons[oper](leftside=entity0.status(stat0),
									rightside=entity1.status(stat1),
									windows=windows)

	def count_all_table(self, tbl):
		return self.call_one('{}_count'.format(tbl)).fetchone()[0]

	def rules_dump(self):
		for (name, ) in self.call_one('rules_dump'):
			yield name

	def _set_rule_something(self, what, rule, branch, turn, tick, flist):
		flist = self.pack(flist)
		return self.call_one('rule_{}_insert'.format(what), rule, branch, turn,
								tick, flist)

	set_rule_triggers = partialmethod(_set_rule_something, 'triggers')
	set_rule_prereqs = partialmethod(_set_rule_something, 'prereqs')
	set_rule_actions = partialmethod(_set_rule_something, 'actions')

	def set_rule(self,
					rule,
					branch,
					turn,
					tick,
					triggers=None,
					prereqs=None,
					actions=None):
		self.call_one('rules_insert', rule)
		self.set_rule_triggers(rule, branch, turn, tick, triggers or [])
		self.set_rule_prereqs(rule, branch, turn, tick, prereqs or [])
		self.set_rule_actions(rule, branch, turn, tick, actions or [])

	def set_rulebook(self, name, branch, turn, tick, rules=None):
		name, rules = map(self.pack, (name, rules or []))
		self.call_one('rulebooks_insert', name, branch, turn, tick, rules)

	def _set_rulebook_on_character(self, rbtyp, char, branch, turn, tick, rb):
		char, rb = map(self.pack, (char, rb))
		self.call_one(rbtyp + '_rulebook_insert', char, branch, turn, tick, rb)

	set_character_rulebook = partialmethod(_set_rulebook_on_character,
											'character')
	set_unit_rulebook = partialmethod(_set_rulebook_on_character, 'unit')
	set_character_thing_rulebook = partialmethod(_set_rulebook_on_character,
													'character_thing')
	set_character_place_rulebook = partialmethod(_set_rulebook_on_character,
													'character_place')
	set_character_portal_rulebook = partialmethod(_set_rulebook_on_character,
													'character_portal')

	def rulebooks(self):
		for book in self.call_one('rulebooks'):
			yield self.unpack(book)

	def exist_node(self, character, node, branch, turn, tick, extant):
		super().exist_node(character, node, branch, turn, tick, extant)

	def exist_edge(self,
					character,
					orig,
					dest,
					idx,
					branch,
					turn,
					tick,
					extant=None):
		start = monotonic()
		if extant is None:
			branch, turn, tick, extant = idx, branch, turn, tick
			idx = 0
		super().exist_edge(character, orig, dest, idx, branch, turn, tick,
							extant)
		QueryEngine.exist_edge_t += monotonic() - start

	def set_node_rulebook(self, character, node, branch, turn, tick, rulebook):
		(character, node, rulebook) = map(self.pack,
											(character, node, rulebook))
		return self.call_one('node_rulebook_insert', character, node, branch,
								turn, tick, rulebook)

	def set_portal_rulebook(self, character, orig, dest, branch, turn, tick,
							rulebook):
		(character, orig, dest,
			rulebook) = map(self.pack, (character, orig, dest, rulebook))
		return self.call_one('portal_rulebook_insert', character, orig, dest,
								branch, turn, tick, rulebook)

	def handled_character_rule(self, character, rulebook, rule, branch, turn,
								tick):
		(character, rulebook) = map(self.pack, (character, rulebook))
		self._char_rules_handled.append(
			(character, rulebook, rule, branch, turn, tick))

	def _flush_char_rules_handled(self):
		if not self._char_rules_handled:
			return
		self.call_many('character_rules_handled_insert',
						self._char_rules_handled)
		self._char_rules_handled = []

	def handled_unit_rule(self, character, rulebook, rule, graph, unit, branch,
							turn, tick):
		character, graph, unit, rulebook = map(
			self.pack, (character, graph, unit, rulebook))
		self._unit_rules_handled.append(
			(character, graph, unit, rulebook, rule, branch, turn, tick))

	def _flush_unit_rules_handled(self):
		if not self._unit_rules_handled:
			return
		self.call_many('unit_rules_handled_insert', self._unit_rules_handled)
		self._unit_rules_handled = []

	def handled_character_thing_rule(self, character, rulebook, rule, thing,
										branch, turn, tick):
		character, thing, rulebook = map(self.pack,
											(character, thing, rulebook))
		self._char_thing_rules_handled.append(
			(character, thing, rulebook, rule, branch, turn, tick))

	def _flush_char_thing_rules_handled(self):
		if not self._char_thing_rules_handled:
			return
		self.call_many('character_thing_rules_handled_insert',
						self._char_thing_rules_handled)
		self._char_thing_rules_handled = []

	def handled_character_place_rule(self, character, rulebook, rule, place,
										branch, turn, tick):
		character, rulebook, place = map(self.pack,
											(character, rulebook, place))
		self._char_place_rules_handled.append(
			(character, place, rulebook, rule, branch, turn, tick))

	def _flush_char_place_rules_handled(self):
		if not self._char_place_rules_handled:
			return
		self.call_many('character_place_rules_handled_insert',
						self._char_place_rules_handled)
		self._char_place_rules_handled = []

	def handled_character_portal_rule(self, character, rulebook, rule, orig,
										dest, branch, turn, tick):
		character, rulebook, orig, dest = map(
			self.pack, (character, rulebook, orig, dest))
		self._char_portal_rules_handled.append(
			(character, orig, dest, rulebook, rule, branch, turn, tick))

	def _flush_char_portal_rules_handled(self):
		if not self._char_portal_rules_handled:
			return
		self.call_many('character_portal_rules_handled_insert',
						self._char_portal_rules_handled)
		self._char_portal_rules_handled = []

	def handled_node_rule(self, character, node, rulebook, rule, branch, turn,
							tick):
		(character, node, rulebook) = map(self.pack,
											(character, node, rulebook))
		self._node_rules_handled.append(
			(character, node, rulebook, rule, branch, turn, tick))

	def _flush_node_rules_handled(self):
		if not self._node_rules_handled:
			return
		self.call_many('node_rules_handled_insert', self._node_rules_handled)
		self._node_rules_handled = []

	def handled_portal_rule(self, character, orig, dest, rulebook, rule,
							branch, turn, tick):
		(character, orig, dest,
			rulebook) = map(self.pack, (character, orig, dest, rulebook))
		self._portal_rules_handled.append(
			(character, orig, dest, rulebook, rule, branch, turn, tick))

	def _flush_portal_rules_handled(self):
		if not self._portal_rules_handled:
			return
		self.call_many('portal_rules_handled_insert',
						self._portal_rules_handled)
		self._portal_rules_handled = []

	def get_rulebook_char(self, rulemap, character):
		character = self.pack(character)
		for (book, ) in self.call_one('rulebook_get_{}'.format(rulemap),
										character):
			return self.unpack(book)
		raise KeyError("No rulebook")

	def set_thing_loc(self, character, thing, branch, turn, tick, loc):
		(character, thing) = map(self.pack, (character, thing))
		loc = self.pack(loc)
		self.call_one('del_things_after', character, thing, branch, turn, turn,
						tick)
		self.call_one('things_insert', character, thing, branch, turn, tick,
						loc)

	def unit_set(self, character, graph, node, branch, turn, tick, isav):
		(character, graph, node) = map(self.pack, (character, graph, node))
		self.call_one('del_units_after', character, graph, node, branch, turn,
						turn, tick)
		self.call_one('units_insert', character, graph, node, branch, turn,
						tick, isav)

	def rulebooks_rules(self):
		for (rulebook, rule) in self.call_one('rulebooks_rules'):
			yield map(self.unpack, (rulebook, rule))

	def rulebook_get(self, rulebook, idx):
		return self.unpack(
			self.call_one('rulebook_get', self.pack(rulebook),
							idx).fetchone()[0])

	def rulebook_set(self, rulebook, branch, turn, tick, rules):
		# what if the rulebook has other values set afterward? wipe them out, right?
		# should that happen in the query engine or elsewhere?
		rulebook, rules = map(self.pack, (rulebook, rules))
		try:
			self.call_one('rulebooks_insert', rulebook, branch, turn, tick,
							rules)
		except IntegrityError:
			self.call_one('rulebooks_update', rules, rulebook, branch, turn,
							tick)

	def rulebook_del_time(self, branch, turn, tick):
		self.call_one('rulebooks_del_time', branch, turn, tick)

	def branch_descendants(self, branch):
		for child in self.call_one('branch_children', branch):
			yield child
			yield from self.branch_descendants(child)

	def turns_completed_dump(self):
		return self.call_one('turns_completed_dump')

	def complete_turn(self, branch, turn, discard_rules=False):
		try:
			self.call_one('turns_completed_insert', branch, turn)
		except IntegrityError:
			self.call_one('turns_completed_update', turn, branch)
		if discard_rules:
			self._char_rules_handled = []
			self._unit_rules_handled = []
			self._char_thing_rules_handled = []
			self._char_place_rules_handled = []
			self._char_portal_rules_handled = []
			self._node_rules_handled = []
			self._portal_rules_handled = []


class QueryEngineProxy:

	def __init__(self,
					dbstring,
					connect_args,
					alchemy,
					pack=None,
					unpack=None):
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
		real = QueryEngine(self._dbstring,
							self._connect_args,
							self._alchemy,
							pack=self._pack,
							unpack=self._unpack)
		while True:
			func, args, kwargs = self._inq.get()
			if func == 'get_global':
				if args not in real.globl:
					output = KeyError()
				else:
					output = real.globl[args]
			elif func == 'set_global':
				k, v = args
				real.globl[k] = v
				continue
			elif func == 'list_globals':
				output = list(real.globl)
			elif func == 'len_globals':
				output = len(real.globl)
			elif func == 'del_global':
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
			if hasattr(output, '__next__'):
				output = list(output)
			self._outq.put(output)
			if func == 'close':
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
			self._inq.put(('list_globals', None, None))
			return iter(self._outq.get())

		def __len__(self):
			self._inq.put(('len_globals', None, None))
			return self._outq.get()

		def __getitem__(self, item):
			self._inq.put(('get_global', item, None))
			ret = self._outq.get()
			if isinstance(ret, Exception):
				raise ret
			return ret

		def __setitem__(self, key, value):
			self._inq.put(('set_global', (key, value), None))

		def __delitem__(self, key):
			self._inq.put(('del_global', key, None))
			ret = self._outq.get()
			if isinstance(ret, Exception):
				raise ret

	def __getattr__(self, item):
		if hasattr(QueryEngine, item) and callable(getattr(QueryEngine, item)):
			return self.Caller(item, self._inq, self._outq)
