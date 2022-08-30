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
from collections.abc import MutableMapping
from operator import gt, lt, eq, ne, le, ge
from functools import partialmethod
from time import monotonic
from queue import Queue
from threading import Thread
from typing import Any, List, Callable

from .allegedb import query
from .exc import (IntegrityError, OperationalError)
from .util import EntityStatAccessor
import LiSE


def windows_union(windows):
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


def windows_intersection(windows):
	"""Given a list of (beginning, ending), return another describing where they overlap.

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
			else:
				return right  # assumes left[0] <= right[0]
		# None not in left
		elif right[0] is None:
			return left[0], min((left[1], right[1]))
		elif right[1] is None:
			if left[1] >= right[0]:
				return right[0], left[1]
			else:
				return None
		assert None not in left and None not in right and left[0] < right[1]
		if left[1] >= right[0]:
			if right[1] > left[1]:
				return right[0], left[1]
			else:
				return right
		return None

	if len(windows) == 1:
		return windows
	left_none = []
	right_none = []
	otherwise = []
	for window in windows:
		assert window is not None, None
		if window[0] is None:
			left_none.append(window)
		elif window[1] is None:
			right_none.append(window)
		else:
			otherwise.append(window)

	done = []
	todo = left_none + sorted(otherwise)
	for window in todo:
		if not done:
			done.append(window)
			continue
		res = intersect2(done.pop(), window)
		if res:
			done.append(res)
	return done


def the_select(tab):
	from sqlalchemy import select, Table
	from sqlalchemy.sql.functions import func
	tab: Table
	return select(
		tab.c.turn.label('turn_from'), tab.c.tick.label('tick_from'),
		func.lead(tab.c.turn).over(order_by=(tab.c.turn,
												tab.c.tick)).label('turn_to'),
		func.lead(tab.c.tick).over(order_by=(tab.c.turn,
												tab.c.tick)).label('tick_to'),
		tab.c.value)


def make_graph_val_select(graph: bytes, stat: bytes, branches: List[str],
							mid_turn: bool):
	from sqlalchemy import select, and_, Table
	from sqlalchemy.sql.functions import func
	from .alchemy import meta
	tab: Table = meta.tables['graph_val']
	ticksel = select(
		tab.c.graph, tab.c.stat, tab.c.branch, tab.c.turn,
		tab.c.tick if mid_turn else func.max(tab.c.tick).label('tick')).where(
			and_(tab.c.graph == graph, tab.c.stat == stat,
					tab.c.branch.in_(branches)))
	if not mid_turn:
		ticksel = ticksel.group_by(tab.c.graph, tab.c.stat, tab.c.branch,
									tab.c.turn)
	return the_select(tab).select_from(
		tab.join(
			ticksel,
			and_(tab.c.graph == ticksel.c.graph, tab.c.stat == ticksel.c.stat,
					tab.c.branch == ticksel.c.branch,
					tab.c.turn == ticksel.c.turn,
					tab.c.tick == ticksel.c.tick)))


def make_node_val_select(graph: bytes, node: bytes, stat: bytes,
							branches: List[str], mid_turn: bool):
	from sqlalchemy import select, and_, Table
	from sqlalchemy.sql.functions import func
	from .alchemy import meta
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
	from sqlalchemy import select, and_, Table
	from sqlalchemy.sql.functions import func
	from .alchemy import meta
	tab: Table = meta.tables['things']
	ticksel = select(
		tab.c.character, tab.c.thing, tab.c.branch, tab.c.turn,
		tab.c.tick if mid_turn else func.max(tab.c.tick).label('tick')).where(
			and_(tab.c.character == graph, tab.c.thing == thing,
					tab.c.branch.in_(branches)))
	if not mid_turn:
		ticksel = ticksel.group_by(tab.c.character, tab.c.thing, tab.c.branch,
									tab.c.turn)
	return the_select(tab).select_from(
		tab.join(
			ticksel,
			and_(tab.c.character == ticksel.c.character,
					tab.c.thing == ticksel.c.thing,
					tab.c.branch == ticksel.c.branch,
					tab.c.turn == ticksel.c.turn,
					tab.c.tick == ticksel.c.tick)))


def make_edge_val_select(graph: bytes, orig: bytes, dest: bytes, idx: int,
							stat: bytes, branches: List[str], mid_turn: bool):
	from sqlalchemy import select, and_, Table
	from sqlalchemy.sql.functions import func
	from .alchemy import meta
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


def make_select_from_query(qry: "Query", branches: List[str], pack: callable,
							mid_turn: bool):
	from sqlalchemy import select, and_, or_
	left = qry.leftside
	right = qry.rightside
	if isinstance(left, StatusAlias) and isinstance(
		right, StatusAlias) and isinstance(qry, ComparisonQuery):
		left_sel = make_side_sel(left.entity, left.stat, branches, pack,
									mid_turn)
		right_sel = make_side_sel(right.entity, right.stat, branches, pack,
									mid_turn)
		# figure whether there is overlap between the time ranges
		left_time_from_lte_right_time_from = or_(
			left_sel.c.turn_from < right_sel.c.turn_from,
			and_(left_sel.c.turn_from == right_sel.c.turn_from,
					left_sel.c.tick_from <= right_sel.c.tick_from))
		right_time_to_lte_left_time_to = or_(
			left_sel.c.turn_to == None,
			right_sel.c.turn_to < left_sel.c.turn_to,
			and_(right_sel.c.turn_to == left_sel.c.turn_to,
					right_sel.c.tick_to <= left_sel.c.tick_to))
		right_time_from_lte_left_time_from = or_(
			right_sel.c.turn_from < left_sel.c.turn_from,
			and_(right_sel.c.turn_from == left_sel.c.turn_from,
					right_sel.c.tick_from <= left_sel.c.tick_from))
		left_time_to_lte_right_time_to = or_(
			right_sel.c.turn_to == None,
			left_sel.c.turn_to < right_sel.c.turn_to,
			and_(left_sel.c.turn_to == right_sel.c.turn_to,
					left_sel.c.tick_to <= right_sel.c.tick_to))
		join_cond = or_(
			# left contains right
			and_(left_time_from_lte_right_time_from,
					right_time_to_lte_left_time_to),
			# right contains left
			and_(right_time_from_lte_left_time_from,
					left_time_to_lte_right_time_to),
			# left overlaps right on the beginning
			and_(left_time_from_lte_right_time_from,
					left_time_to_lte_right_time_to),
			# left overlaps right on the ending
			and_(right_time_from_lte_left_time_from,
					right_time_to_lte_left_time_to))
		return select(left_sel.c.turn_from, left_sel.c.tick_from,
						left_sel.c.turn_to, left_sel.c.tick_to,
						right_sel.c.turn_from, right_sel.c.tick_from,
						right_sel.c.turn_to, right_sel.c.tick_to).where(
							and_(qry.oper(left_sel.c.value, right_sel.c.value),
									join_cond)), True

	elif isinstance(right, StatusAlias) and isinstance(qry, ComparisonQuery):
		right_sel = make_side_sel(right.entity, right.stat, branches, pack,
									mid_turn)
		return select(right_sel.c.turn_from, right_sel.c.tick_from,
						right_sel.c.turn_to, right_sel.c.tick_to).where(
							qry.oper(pack(left), right_sel.c.value)), True

	elif isinstance(left, StatusAlias) and isinstance(qry, ComparisonQuery):
		left_sel = make_side_sel(left.entity, left.stat, branches, pack,
									mid_turn)
		return select(left_sel.c.turn_from, left_sel.c.tick_from,
						left_sel.c.turn_to, left_sel.c.tick_to).where(
							qry.oper(left_sel.c.value, pack(right))), True
	else:
		raise NotImplementedError("oh no, can't do that")


class Query(object):

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

	def iter_turns(self):
		raise NotImplementedError

	def iter_ticks(self, turn):
		raise NotImplementedError

	def __eq__(self, other):
		return EqQuery(self.engine, self, self.engine._entityfy(other))

	def __gt__(self, other):
		return GtQuery(self.engine, self, self.engine._entityfy(other))

	def __ge__(self, other):
		return GeQuery(self.engine, self, self.engine._entityfy(other))

	def __lt__(self, other):
		return LtQuery(self.engine, self, self.engine._entityfy(other))

	def __le__(self, other):
		return LeQuery(self.engine, self, self.engine._entityfy(other))

	def __ne__(self, other):
		return NeQuery(self.engine, self, self.engine._entityfy(other))


class Union(Query):
	pass


class ComparisonQuery(Query):
	oper: Callable[[Any, Any], bool] = lambda x, y: NotImplemented

	def iter_turns(self):
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
			return side.iter_turns
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
	last_turn = engine._branches[engine.branch][-1]

	for (branch, _, _) in engine._iter_parent_btt(start_branch
													or engine.branch):
		if branch is None:
			return
		parent, turn_start, tick_start, turn_end, tick_end = engine._branches[
			branch]
		for turn in range(turn_start, last_turn + 1):
			if oper(leftside(branch, turn), rightside(branch, turn)):
				yield branch, turn


class ConnectionHolder(query.ConnectionHolder):

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

	def __init__(self,
					dbstring,
					connect_args,
					alchemy,
					strings_filename: str = None,
					pack=None,
					unpack=None):
		if alchemy:
			try:
				from .alchemy import gather_sql
			except ImportError:
				gather_sql = None
		else:
			gather_sql = None
		super().__init__(dbstring,
							connect_args,
							alchemy,
							strings_filename,
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
		for key, branch, turn, tick, value in self.sql('universals_dump'):
			yield unpack(key), branch, turn, tick, unpack(value)

	def rulebooks_dump(self):
		unpack = self.unpack
		for rulebook, branch, turn, tick, rules in self.sql('rulebooks_dump'):
			yield unpack(rulebook), branch, turn, tick, unpack(rules)

	def _rule_dump(self, typ):
		unpack = self.unpack
		for rule, branch, turn, tick, lst in self.sql(
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
		for graph, typ in self.sql('graphs_dump'):
			if typ == 'DiGraph':
				yield unpack(graph)

	characters = characters_dump

	def node_rulebook_dump(self):
		unpack = self.unpack
		for character, node, branch, turn, tick, rulebook in self.sql(
			'node_rulebook_dump'):
			yield unpack(character), unpack(node), branch, turn, tick, unpack(
				rulebook)

	def portal_rulebook_dump(self):
		unpack = self.unpack
		for character, orig, dest, branch, turn, tick, rulebook in self.sql(
			'portal_rulebook_dump'):
			yield (unpack(character), unpack(orig), unpack(dest), branch, turn,
					tick, unpack(rulebook))

	def _charactery_rulebook_dump(self, qry):
		unpack = self.unpack
		for character, branch, turn, tick, rulebook in self.sql(
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
		for character, rulebook, rule, branch, turn, tick in self.sql(
			'character_rules_handled_dump'):
			yield unpack(character), unpack(rulebook), rule, branch, turn, tick

	def character_rules_changes_dump(self):
		unpack = self.unpack
		for (character, rulebook, rule, branch, turn, tick, handled_branch,
				handled_turn) in self.sql('character_rules_changes_dump'):
			yield (unpack(character), unpack(rulebook), rule, branch, turn,
					tick, handled_branch, handled_turn)

	def unit_rules_handled_dump(self):
		unpack = self.unpack
		for character, graph, unit, rulebook, rule, branch, turn, tick in self.sql(
			'unit_rules_handled_dump'):
			yield (unpack(character), unpack(graph), unpack(unit),
					unpack(rulebook), rule, branch, turn, tick)

	def unit_rules_changes_dump(self):
		jl = self.unpack
		for (character, rulebook, rule, graph, unit, branch, turn, tick,
				handled_branch,
				handled_turn) in self.sql('unit_rules_changes_dump'):
			yield (jl(character), jl(rulebook), rule, jl(graph), jl(unit),
					branch, turn, tick, handled_branch, handled_turn)

	def character_thing_rules_handled_dump(self):
		unpack = self.unpack
		for character, thing, rulebook, rule, branch, turn, tick in self.sql(
			'character_thing_rules_handled_dump'):
			yield unpack(character), unpack(thing), unpack(
				rulebook), rule, branch, turn, tick

	def character_thing_rules_changes_dump(self):
		jl = self.unpack
		for (
			character, thing, rulebook, rule, branch, turn, tick,
			handled_branch,
			handled_turn) in self.sql('character_thing_rules_changes_dump'):
			yield (jl(character), jl(thing), jl(rulebook), rule, branch, turn,
					tick, handled_branch, handled_turn)

	def character_place_rules_handled_dump(self):
		unpack = self.unpack
		for character, place, rulebook, rule, branch, turn, tick in self.sql(
			'character_place_rules_handled_dump'):
			yield unpack(character), unpack(place), unpack(
				rulebook), rule, branch, turn, tick

	def character_place_rules_changes_dump(self):
		jl = self.unpack
		for (
			character, rulebook, rule, place, branch, turn, tick,
			handled_branch,
			handled_turn) in self.sql('character_place_rules_changes_dump'):
			yield (jl(character), jl(rulebook), rule, jl(place), branch, turn,
					tick, handled_branch, handled_turn)

	def character_portal_rules_handled_dump(self):
		unpack = self.unpack
		for character, rulebook, rule, orig, dest, branch, turn, tick in self.sql(
			'character_portal_rules_handled_dump'):
			yield (unpack(character), unpack(rulebook), unpack(orig),
					unpack(dest), rule, branch, turn, tick)

	def character_portal_rules_changes_dump(self):
		jl = self.unpack
		for (
			character, rulebook, rule, orig, dest, branch, turn, tick,
			handled_branch,
			handled_turn) in self.sql('character_portal_rules_changes_dump'):
			yield (jl(character), jl(rulebook), rule, jl(orig), jl(dest),
					branch, turn, tick, handled_branch, handled_turn)

	def node_rules_handled_dump(self):
		for character, node, rulebook, rule, branch, turn, tick in self.sql(
			'node_rules_handled_dump'):
			yield self.unpack(character), self.unpack(node), self.unpack(
				rulebook), rule, branch, turn, tick

	def node_rules_changes_dump(self):
		jl = self.unpack
		for (character, node, rulebook, rule, branch, turn, tick,
				handled_branch,
				handled_turn) in self.sql('node_rules_changes_dump'):
			yield (jl(character), jl(node), jl(rulebook), rule, branch, turn,
					tick, handled_branch, handled_turn)

	def portal_rules_handled_dump(self):
		unpack = self.unpack
		for character, orig, dest, rulebook, rule, branch, turn, tick in self.sql(
			'portal_rules_handled_dump'):
			yield (unpack(character), unpack(orig), unpack(dest),
					unpack(rulebook), rule, branch, turn, tick)

	def portal_rules_changes_dump(self):
		jl = self.unpack
		for (character, orig, dest, rulebook, rule, branch, turn, tick,
				handled_branch,
				handled_turn) in self.sql('portal_rules_changes_dump'):
			yield (jl(character), jl(orig), jl(dest), jl(rulebook), rule,
					branch, turn, tick, handled_branch, handled_turn)

	def senses_dump(self):
		unpack = self.unpack
		for character, sense, branch, turn, tick, function in self.sql(
			'senses_dump'):
			yield unpack(character), sense, branch, turn, tick, function

	def things_dump(self):
		unpack = self.unpack
		for character, thing, branch, turn, tick, location in self.sql(
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
			for thing, turn, tick, location in self.sql(
				'load_things_tick_to_end', pack(character), branch, turn_from,
				turn_from, tick_from):
				yield character, unpack(thing), branch, turn, tick, unpack(
					location)
		else:
			if tick_to is None:
				raise ValueError("Need both or neither of turn_to, tick_to")
			for thing, turn, tick, location in self.sql(
				'load_things_tick_to_tick', pack(character), branch, turn_from,
				turn_from, tick_from, turn_to, turn_to, tick_to):
				yield character, unpack(thing), branch, turn, tick, unpack(
					location)

	def units_dump(self):
		unpack = self.unpack
		for character_graph, unit_graph, unit_node, branch, turn, tick, is_av in self.sql(
			'units_dump'):
			yield (unpack(character_graph), unpack(unit_graph),
					unpack(unit_node), branch, turn, tick, is_av)

	def universal_set(self, key, branch, turn, tick, val):
		key, val = map(self.pack, (key, val))
		self.sql('universals_insert', key, branch, turn, tick, val)

	def universal_del(self, key, branch, turn, tick):
		key = self.pack(key)
		self.sql('universals_insert', key, branch, turn, tick, None)

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
		return self.sql('{}_count'.format(tbl)).fetchone()[0]

	def rules_dump(self):
		for (name, ) in self.sql('rules_dump'):
			yield name

	def _set_rule_something(self, what, rule, branch, turn, tick, flist):
		flist = self.pack(flist)
		return self.sql('rule_{}_insert'.format(what), rule, branch, turn,
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
		self.sql('rules_insert', rule)
		self.set_rule_triggers(rule, branch, turn, tick, triggers or [])
		self.set_rule_prereqs(rule, branch, turn, tick, prereqs or [])
		self.set_rule_actions(rule, branch, turn, tick, actions or [])

	def set_rulebook(self, name, branch, turn, tick, rules=None):
		name, rules = map(self.pack, (name, rules or []))
		self.sql('rulebooks_insert', name, branch, turn, tick, rules)

	def _set_rulebook_on_character(self, rbtyp, char, branch, turn, tick, rb):
		char, rb = map(self.pack, (char, rb))
		self.sql(rbtyp + '_rulebook_insert', char, branch, turn, tick, rb)

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
		for book in self.sql('rulebooks'):
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
		return self.sql('node_rulebook_insert', character, node, branch, turn,
						tick, rulebook)

	def set_portal_rulebook(self, character, orig, dest, branch, turn, tick,
							rulebook):
		(character, orig, dest,
			rulebook) = map(self.pack, (character, orig, dest, rulebook))
		return self.sql('portal_rulebook_insert', character, orig, dest,
						branch, turn, tick, rulebook)

	def handled_character_rule(self, character, rulebook, rule, branch, turn,
								tick):
		(character, rulebook) = map(self.pack, (character, rulebook))
		self._char_rules_handled.append(
			(character, rulebook, rule, branch, turn, tick))

	def _flush_char_rules_handled(self):
		if not self._char_rules_handled:
			return
		self.sqlmany('character_rules_handled_insert',
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
		self.sqlmany('unit_rules_handled_insert', self._unit_rules_handled)
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
		self.sqlmany('character_thing_rules_handled_insert',
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
		self.sqlmany('character_place_rules_handled_insert',
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
		self.sqlmany('character_portal_rules_handled_insert',
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
		self.sqlmany('node_rules_handled_insert', self._node_rules_handled)
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
		self.sqlmany('portal_rules_handled_insert', self._portal_rules_handled)
		self._portal_rules_handled = []

	def get_rulebook_char(self, rulemap, character):
		character = self.pack(character)
		for (book, ) in self.sql('rulebook_get_{}'.format(rulemap), character):
			return self.unpack(book)
		raise KeyError("No rulebook")

	def set_thing_loc(self, character, thing, branch, turn, tick, loc):
		(character, thing) = map(self.pack, (character, thing))
		loc = self.pack(loc)
		self.sql('del_things_after', character, thing, branch, turn, turn,
					tick)
		self.sql('things_insert', character, thing, branch, turn, tick, loc)

	def unit_set(self, character, graph, node, branch, turn, tick, isav):
		(character, graph, node) = map(self.pack, (character, graph, node))
		self.sql('del_units_after', character, graph, node, branch, turn, turn,
					tick)
		self.sql('units_insert', character, graph, node, branch, turn, tick,
					isav)

	def rulebooks_rules(self):
		for (rulebook, rule) in self.sql('rulebooks_rules'):
			yield map(self.unpack, (rulebook, rule))

	def rulebook_get(self, rulebook, idx):
		return self.unpack(
			self.sql('rulebook_get', self.pack(rulebook), idx).fetchone()[0])

	def rulebook_set(self, rulebook, branch, turn, tick, rules):
		# what if the rulebook has other values set afterward? wipe them out, right?
		# should that happen in the query engine or elsewhere?
		rulebook, rules = map(self.pack, (rulebook, rules))
		try:
			self.sql('rulebooks_insert', rulebook, branch, turn, tick, rules)
		except IntegrityError:
			self.sql('rulebooks_update', rules, rulebook, branch, turn, tick)

	def rulebook_del_time(self, branch, turn, tick):
		self.sql('rulebooks_del_time', branch, turn, tick)

	def branch_descendants(self, branch):
		for child in self.sql('branch_children', branch):
			yield child
			yield from self.branch_descendants(child)

	def turns_completed_dump(self):
		return self.sql('turns_completed_dump')

	def complete_turn(self, branch, turn, discard_rules=False):
		try:
			self.sql('turns_completed_insert', branch, turn)
		except IntegrityError:
			self.sql('turns_completed_update', turn, branch)
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
