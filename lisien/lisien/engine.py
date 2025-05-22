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
"""The "engine" of lisien is an object relational mapper with special
stores for game data and entities, as well as properties for manipulating the
flow of time.

"""

from __future__ import annotations

import os
import pickle
import shutil
import sys
import zlib
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from concurrent.futures import wait as futwait
from contextlib import ContextDecorator, contextmanager
from functools import cached_property, partial, wraps
from itertools import chain, pairwise
from multiprocessing import Pipe, Process, Queue
from operator import itemgetter, lt
from os import PathLike
from queue import Empty, SimpleQueue
from random import Random
from threading import Lock, RLock, Thread
from time import sleep
from types import FunctionType, MethodType, ModuleType
from typing import Any, Callable, Iterable, Iterator, Optional, Type

import msgpack
import networkx as nx
import numpy as np
from blinker import Signal
from networkx import (
	Graph,
	NetworkXError,
	from_dict_of_dicts,
	from_dict_of_lists,
	spring_layout,
)

from . import exc
from .cache import (
	CharacterPlaceRulesHandledCache,
	CharacterPortalRulesHandledCache,
	CharacterRulesHandledCache,
	CharactersRulebooksCache,
	CharacterThingRulesHandledCache,
	EdgesCache,
	EdgeValCache,
	EntitylessCache,
	GraphValCache,
	InitializedEntitylessCache,
	NodeContentsCache,
	NodeRulesHandledCache,
	NodesCache,
	NodesRulebooksCache,
	NodeValCache,
	PickyDefaultDict,
	PortalRulesHandledCache,
	PortalsRulebooksCache,
	StructuredDefaultDict,
	ThingsCache,
	TurnEndDict,
	TurnEndPlanDict,
	UnitnessCache,
	UnitRulesHandledCache,
)
from .character import Character
from .db import NullQueryEngine, ParquetQueryEngine, SQLAlchemyQueryEngine
from .exc import (
	GraphNameError,
	HistoricKeyError,
	KeyframeError,
	OutOfTimelineError,
)
from .facade import CharacterFacade
from .graph import DiGraph
from .node import Place, Thing
from .portal import Portal
from .proxy import worker_subprocess
from .query import (
	CombinedQueryResult,
	ComparisonQuery,
	CompoundQuery,
	Query,
	QueryResult,
	QueryResultEndTurn,
	QueryResultMidTurn,
	StatusAlias,
	_make_side_sel,
)
from .rule import AllRuleBooks, AllRules, Rule
from .typing import (
	Branch,
	CharName,
	DeltaDict,
	EdgeValDict,
	GraphEdgesDict,
	GraphEdgeValDict,
	GraphNodesDict,
	GraphNodeValDict,
	GraphValDict,
	Key,
	KeyframeTuple,
	NodeName,
	NodeValDict,
	Plan,
	StatDict,
	Tick,
	Time,
	Turn,
)
from .util import (
	AbstractCharacter,
	AbstractEngine,
	SizedDict,
	TimeSignalDescriptor,
	fake_submit,
	garbage,
	normalize_layout,
	sort_set,
	world_locked,
)
from .window import WindowDict, update_backward_window, update_window
from .xcollections import (
	CharacterMapping,
	FunctionStore,
	StringStore,
	UniversalMapping,
)

SlightlyPackedDeltaType = dict[
	bytes,
	dict[
		bytes,
		bytes
		| dict[
			bytes,
			bytes | dict[bytes, bytes | dict[bytes, bytes]],
		],
	],
]

TRUE: bytes = msgpack.packb(True)
FALSE: bytes = msgpack.packb(False)
NONE: bytes = msgpack.packb(None)
NAME: bytes = msgpack.packb("name")
NODES: bytes = msgpack.packb("nodes")
EDGES: bytes = msgpack.packb("edges")
UNITS: bytes = msgpack.packb("units")
RULEBOOK: bytes = msgpack.packb("rulebook")
RULEBOOKS: bytes = msgpack.packb("rulebooks")
NODE_VAL: bytes = msgpack.packb("node_val")
EDGE_VAL: bytes = msgpack.packb("edge_val")
ETERNAL: bytes = msgpack.packb("eternal")
UNIVERSAL: bytes = msgpack.packb("universal")
STRINGS: bytes = msgpack.packb("strings")
RULES: bytes = msgpack.packb("rules")
TRIGGERS: bytes = msgpack.packb("triggers")
PREREQS: bytes = msgpack.packb("prereqs")
ACTIONS: bytes = msgpack.packb("actions")
NEIGHBORHOOD: bytes = msgpack.packb("neighborhood")
BIG: bytes = msgpack.packb("big")
LOCATION: bytes = msgpack.packb("location")
BRANCH: bytes = msgpack.packb("branch")


class InnerStopIteration(StopIteration):
	pass


class DummyEntity(dict):
	"""Something to use in place of a node or edge"""

	__slots__ = ["engine"]

	def __init__(self, engine: AbstractEngine):
		super().__init__()
		self.engine = engine


class PlanningContext(ContextDecorator):
	"""A context manager for 'hypothetical' edits.

	Start a block of code like::

		with orm.plan():
			...


	and any changes you make to the world state within that block will be
	'plans,' meaning that they are used as defaults. The world will
	obey your plan unless you make changes to the same entities outside
	the plan, in which case the world will obey those, and cancel any
	future plan.

	Plans are *not* canceled when concerned entities are deleted, although
	they are unlikely to be followed.

	New branches cannot be started within plans. The ``with orm.forward():``
	optimization is disabled within a ``with orm.plan():`` block, so
	consider another approach instead of making a very large plan.

	With ``reset=True`` (the default), when the plan block closes,
	the time will reset to when it began.

	"""

	__slots__ = ["orm", "id", "forward", "reset"]

	def __init__(self, orm: "Engine", reset=True):
		self.orm = orm
		if reset:
			self.reset = orm._btt()
		else:
			self.reset = None

	def __enter__(self):
		orm = self.orm
		if orm._planning:
			raise ValueError("Already planning")
		orm._planning = True
		branch, turn, tick = orm._btt()
		self.id = myid = orm._last_plan = orm._last_plan + 1
		self.forward = orm._forward
		if orm._forward:
			orm._forward = False
		orm._plans[myid] = branch, turn, tick
		orm.query.plans_insert(myid, branch, turn, tick)
		orm._branches_plans[branch].add(myid)
		return myid

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.orm._planning = False
		if self.reset is not None:
			self.orm._set_btt(*self.reset)
		if self.forward:
			self.orm._forward = True


class NextTurn(Signal):
	"""Make time move forward in the simulation.

	Calls ``advance`` repeatedly, returning a list of the rules' return values.

	I am also a ``Signal``, so you can register functions to be
	called when the simulation runs. Pass them to my ``connect``
	method.

	"""

	def __init__(self, engine: Engine):
		super().__init__()
		self.engine = engine

	def __call__(self) -> tuple[list, DeltaDict]:
		engine = self.engine
		for store in engine.stores:
			if getattr(store, "_need_save", None):
				store.save()
		if hasattr(engine, "_worker_processes"):
			engine._update_all_worker_process_states()
		start_branch, start_turn, start_tick = engine._btt()
		latest_turn = engine._get_last_completed_turn(start_branch)
		if latest_turn is None or start_turn == latest_turn:
			# Pre-emptively nudge the loadedness and branch tracking,
			# so that lisien does not try to load an empty turn before every
			# loop of the rules engine
			engine._extend_branch(start_branch, start_turn + 1, 0)
			engine.turn += 1
			engine.tick = engine.turn_end_plan()
		elif start_turn < latest_turn:
			engine.turn += 1
			engine.tick = engine.turn_end_plan()
			self.send(
				engine,
				branch=engine.branch,
				turn=engine.turn,
				tick=engine.tick,
			)
			return [], engine._get_branch_delta(
				branch=start_branch,
				turn_from=start_turn,
				turn_to=engine.turn,
				tick_from=start_tick,
				tick_to=engine.tick,
			)
		elif start_turn > latest_turn + 1:
			raise exc.RulesEngineError(
				"Can't run the rules engine on any turn but the latest"
			)
		results = []
		if hasattr(engine, "_rules_iter"):
			it = engine._rules_iter
		else:
			todo = engine._eval_triggers()
			it = engine._rules_iter = engine._follow_rules(todo)
		with engine.advancing():
			for res in it:
				if isinstance(res, InnerStopIteration):
					del engine._rules_iter
					raise StopIteration from res
				elif res:
					if isinstance(res, tuple) and res[0] == "stop":
						engine.universal["last_result"] = res
						engine.universal["last_result_idx"] = 0
						branch, turn, tick = engine._btt()
						self.send(engine, branch=branch, turn=turn, tick=tick)
						return list(res), engine._get_branch_delta(
							branch=start_branch,
							turn_from=start_turn,
							turn_to=turn,
							tick_from=start_tick,
							tick_to=tick,
						)
					else:
						results.extend(res)
		del engine._rules_iter
		# accept any new plans
		engine.tick = engine.turn_end_plan()
		engine._complete_turn(
			start_branch,
			engine.turn,
		)
		if (
			engine.flush_interval is not None
			and engine.turn % engine.flush_interval == 0
		):
			engine.flush()
		if (
			engine.commit_interval is not None
			and engine.turn % engine.commit_interval == 0
		):
			engine.commit()
		self.send(
			self.engine,
			branch=engine.branch,
			turn=engine.turn,
			tick=engine.tick,
		)
		delta = engine._get_branch_delta(
			branch=engine.branch,
			turn_from=start_turn,
			turn_to=engine.turn,
			tick_from=start_tick,
			tick_to=engine.tick,
		)
		if results:
			engine.universal["last_result"] = results
			engine.universal["last_result_idx"] = 0
		return results, delta


class AbstractSchema(ABC):
	"""Base class for schemas describing what changes are permitted to the game world"""

	def __init__(self, engine: AbstractEngine):
		self.engine = engine

	@abstractmethod
	def entity_permitted(self, entity):
		raise NotImplementedError

	@abstractmethod
	def stat_permitted(self, turn, entity, key, value):
		raise NotImplementedError


class NullSchema(AbstractSchema):
	"""Schema that permits all changes to the game world"""

	def entity_permitted(self, entity):
		return True

	def stat_permitted(self, turn, entity, key, value):
		return True


class Engine(AbstractEngine, Executor):
	"""Lisien, the Life Simulator Engine.

	:param prefix: directory containing the simulation and its code;
		defaults to the working directory. If ``None``, Lisien won't save
		any rules code to disk, and won't save world data unless you supply
		:param connect_string:.
	:param string: module storing strings to be used in the game; if absent,
		we'll use a :class:`lisien.xcollections.StringStore` to keep them in a
		JSON file in the ``prefix``.
	:param function: module containing utility functions; if absent, we'll
		use a :class:`lisien.xcollections.FunctionStore` to keep them in a .py
		file in the ``prefix``
	:param method: module containing functions taking this engine as
		first arg; if absent, we'll
		use a :class:`lisien.xcollections.FunctionStore` to keep them in a .py
		file in the ``prefix``.
	:param trigger: module containing trigger functions, taking a lisien
		entity and returning a boolean for whether to run a rule; if absent, we'll
		use a :class:`lisien.xcollections.FunctionStore` to keep them in a .py
		file in the ``prefix``.
	:param prereq: module containing prereq functions, taking a lisien entity and
		returning a boolean for whether to permit a rule to run; if absent, we'll
		use a :class:`lisien.xcollections.FunctionStore` to keep them in a .py
		file in the ``prefix``.
	:param action: module containing action functions, taking a lisien entity and
		mutating it (and possibly the rest of the world); if absent, we'll
		use a :class:`lisien.xcollections.FunctionStore` to keep them in a .py
		file in the ``prefix``.
	:param main_branch: the string name of the branch to start from. Defaults
		to "trunk" if not set in some prior session. You should only change
		this if your game generates new initial conditions for each
		playthrough.
	:param connect_string: a rfc1738 URI for a database to connect to. Leave
		``None`` to use the ParquetDB database in the ``prefix``.
	:param connect_args: Dictionary of keyword arguments for the
		database connection
	:param schema: A Schema class that determines which changes to allow to
		the world; used when a player should not be able to change just
		anything. Defaults to :class:`NullSchema`, which allows all changes.
	:param flush_interval: lisien will put pending changes into the database
		transaction every ``flush_interval`` turns. If ``None``, only flush
		on commit. Default ``None``.
	:param keyframe_interval: How many records to let through before automatically
		snapping a keyframe, default ``1000``. If ``None``, you'll need
		to call ``snap_keyframe`` yourself.
	:param commit_interval: Lisien will commit changes to disk every
		``commit_interval`` turns. If ``None`` (the default), only commit
		on close or manual call to ``commit``.
	:param random_seed: A number to initialize the randomizer.
	:param logfun: An optional function taking arguments
		``level, message``, which should log `message` somehow.
	:param clear: Whether to delete *any and all* existing data
		and code in ``prefix`` and the database. Use with caution!
	:param keep_rules_journal: Boolean; if ``True`` (the default), keep
		information on the behavior of the rules engine in the database.
		Makes the database rather large, but useful for debugging.
	:param keyframe_on_close: Whether to snap a keyframe when closing the
		engine, default ``True``. This is usually what you want, as it will
		make future startups faster, but could cause database bloat if
		your game runs few turns per session.
	:param enforce_end_of_time: Whether to raise an exception when
		time travelling to a point after the time that's been simulated.
		Default ``True``. You normally want this, but it could cause problems
		if you're not using the rules engine.
	:param workers: How many subprocesses to use as workers for
		parallel processing. When ``None`` (the default), use as many
		subprocesses as we have CPU cores. When ``0``, parallel processing
		is disabled. Note that ``workers=0`` implies that trigger
		functions operate on bare lisien objects, and can therefore have
		side effects. If you don't want this, instead use
		``workers=1``, which *does* disable parallelism in the case
		of trigger functions.

	"""

	char_cls = Character
	thing_cls = Thing
	place_cls = node_cls = Place
	portal_cls = edge_cls = Portal
	entity_cls = char_cls | thing_cls | place_cls | portal_cls
	illegal_graph_names = {
		"global",
		"eternal",
		"universal",
		"rulebooks",
		"rules",
	}
	illegal_node_names = {"nodes", "node_val", "edges", "edge_val", "things"}
	time = TimeSignalDescriptor()

	@property
	def eternal(self):
		return self.query.globl

	@property
	def branch(self) -> Branch:
		return Branch(self._obranch)

	@branch.setter
	@world_locked
	def branch(self, v: str):
		if not isinstance(v, str):
			raise TypeError("branch must be str")
		if self._planning:
			raise ValueError("Don't change branches while planning")
		curbranch, curturn, curtick = self._btt()
		if curbranch == v:
			return
		# make sure I'll end up within the revision range of the
		# destination branch
		v = Branch(v)
		if v != self.main_branch and v in self.branches():
			parturn = self._branch_start(v)[0]
			if curturn < parturn:
				raise OutOfTimelineError(
					"Tried to jump to branch {br} at turn {tr}, "
					"but {br} starts at turn {rv}. "
					"Go to turn {rv} or later to use this branch.".format(
						br=v, tr=self.turn, rv=parturn
					),
					self.branch,
					self.turn,
					self.tick,
					v,
					self.turn,
					self.tick,
				)
		then = self._btt()
		branch_is_new = v not in self.branches()
		if branch_is_new:
			# assumes the present turn in the parent branch has
			# been finalized.
			self._start_branch(curbranch, v, self.turn, self.tick)
			tick = self.tick
		else:
			self._otick = tick = self.turn_end(v, self.turn)
		parent = self._obranch
		self._obranch = v
		if branch_is_new:
			self._copy_plans(parent, self.turn, tick)
			self.snap_keyframe(silent=True)
			return
		self.load_at(v, curturn, tick)
		self.time.send(self.time, then=then, now=self._btt())

	@property
	def main_branch(self):
		return self.query.globl["main_branch"]

	def switch_main_branch(self, branch: str) -> None:
		if self.branch != self.main_branch or self.turn != 0 or self.tick != 0:
			raise ValueError("Go to the start of time first")
		branch = Branch(branch)
		if (
			branch in self.branches()
			and self.branch_parent(branch) is not None
		):
			raise ValueError("Not a main branch")
		then = self._btt()
		self.query.globl["main_branch"] = self.branch = branch
		self.time.send(self, then=then, now=self._btt())

	@property
	def turn(self) -> Turn:
		return Turn(self._oturn)

	@turn.setter
	@world_locked
	def turn(self, v: int):
		if not isinstance(v, int):
			raise TypeError("Turns must be integers")
		if v < 0:
			raise ValueError("Turns can't be negative")
		if v == self.turn:
			return
		turn_end, tick_end = self._branch_end()
		if self._enforce_end_of_time and not self._planning and v > turn_end:
			raise OutOfTimelineError(
				f"The turn {v} is after the end of the branch {self.branch}. "
				f"Go to turn {turn_end} and simulate with `next_turn`.",
				self.branch,
				self.turn,
				self.tick,
				self.branch,
				v,
				self.tick,
			)
		# enforce the arrow of time, if it's in effect
		if self._forward and v < self._oturn:
			raise ValueError("Can't time travel backward in a forward context")
		v = Turn(v)
		oldrando = self.universal.get("rando_state")
		branch = self.branch
		if self._planning:
			tick = self._turn_end_plan[branch, v]
		else:
			tick = self._turn_end[branch, v]
		self.load_at(branch, v, tick)
		then = self._btt()
		self._otick = tick
		self._oturn = v
		newrando = self.universal.get("rando_state")
		if newrando and newrando != oldrando:
			self._rando.setstate(newrando)
		self.time.send(self, then=then, now=self._btt())

	@property
	def tick(self) -> Tick:
		"""A counter of how many changes have occurred this turn.

		Can be set manually, but is more often set to the last tick in a turn
		as a side effect of setting ``turn``.

		"""
		return Tick(self._otick)

	@tick.setter
	@world_locked
	def tick(self, v: int):
		if not isinstance(v, int):
			raise TypeError("Ticks must be integers")
		if v < 0:
			raise ValueError("Ticks can't be negative")
		# enforce the arrow of time, if it's in effect
		if self._forward and v < self._otick:
			raise ValueError("Can't time travel backward in a forward context")
		if v == self.tick:
			return
		tick_end = self._turn_end_plan[self.branch, self.turn]
		if v > tick_end + 1:
			raise OutOfTimelineError(
				f"The tick {v} is after the end of the turn {self.turn}. "
				f"Go to tick {tick_end + 1} and simulate with `next_turn`.",
				self.branch,
				self.turn,
				self.tick,
				self.branch,
				self.turn,
				v,
			)
		oldrando = self.universal.get("rando_state")
		v = Tick(v)
		self.load_at(self.branch, self.turn, v)
		if not self._planning:
			self._extend_branch(self.branch, self.turn, v)
		old_tick = self._otick
		self._otick = v
		newrando = self.universal.get("rando_state")
		if newrando and newrando != oldrando:
			self._rando.setstate(newrando)
		self.time.send(
			self,
			then=(self.branch, self.turn, old_tick),
			now=(self.branch, self.turn, v),
		)

	@cached_property
	def _where_cached(self) -> dict[Time, list]:
		return defaultdict(list)

	@cached_property
	def _node_objs(self) -> SizedDict:
		return SizedDict()

	@cached_property
	def _edge_objs(self) -> SizedDict:
		return SizedDict()

	@cached_property
	def _nbtt_stuff(self):
		return (
			self._btt,
			self._turn_end_plan,
			self._turn_end,
			self._plan_ticks,
			self._time_plan,
		)

	@cached_property
	def _node_exists_stuff(
		self,
	) -> tuple[
		Callable[[tuple[CharName, NodeName, Branch, Turn, Tick]], Any],
		Callable[[], Time],
	]:
		return (self._nodes_cache._base_retrieve, self._btt)

	@cached_property
	def _exist_node_stuff(
		self,
	) -> tuple[
		Callable[[], Time],
		Callable[[CharName, NodeName, Branch, Turn, Tick, bool], None],
		Callable[[CharName, NodeName, Branch, Turn, Tick, Any], None],
	]:
		return (self._nbtt, self.query.exist_node, self._nodes_cache.store)

	@cached_property
	def _edge_exists_stuff(
		self,
	) -> tuple[
		Callable[
			[tuple[CharName, NodeName, NodeName, int, Branch, Turn, Tick]],
			bool,
		],
		Callable[[], Time],
	]:
		return (self._edges_cache._base_retrieve, self._btt)

	@cached_property
	def _exist_edge_stuff(
		self,
	) -> tuple[
		Callable[[], Time],
		Callable[
			[CharName, NodeName, NodeName, int, Branch, Turn, Tick, bool], None
		],
		Callable[
			[CharName, NodeName, NodeName, int, Branch, Turn, Tick, Any], None
		],
	]:
		return (self._nbtt, self.query.exist_edge, self._edges_cache.store)

	@cached_property
	def _loaded(
		self,
	) -> dict[Branch, tuple[Turn, Tick, Optional[Turn], Optional[Tick]]]:
		"""Slices of time that are currently in memory

		{branch: (turn_from, tick_from, turn_to, tick_to)}

		"""
		return {}

	@cached_property
	def _get_node_stuff(
		self,
	) -> tuple[
		dict, Callable[[Key, Key], bool], Callable[[Key, Key], node_cls]
	]:
		return (self._node_objs, self._node_exists, self._make_node)

	@cached_property
	def _get_edge_stuff(
		self,
	) -> tuple[
		dict,
		Callable[[Key, Key, Key, int], bool],
		Callable[[Key, Key, Key, int], edge_cls],
	]:
		return (self._edge_objs, self._edge_exists, self._make_edge)

	@cached_property
	def _childbranch(self) -> dict[str, set[str]]:
		"""Immediate children of a branch"""
		return defaultdict(set)

	@cached_property
	def _branches_d(
		self,
	) -> dict[Branch, tuple[Branch | None, Turn, Tick, Turn, Tick]]:
		"""Parent, start time, and end time of each branch. Includes plans."""
		return {}

	@cached_property
	def _branch_parents(self) -> dict[Branch, set[Branch]]:
		"""Parents of a branch at any remove"""
		return defaultdict(set)

	@cached_property
	def _turn_end(self) -> dict[tuple[Branch, Turn], Tick]:
		return TurnEndDict(self)

	@cached_property
	def _turn_end_plan(self) -> dict[tuple[Branch, Turn], Tick]:
		return TurnEndPlanDict(self)

	@cached_property
	def _graph_objs(self) -> dict[Key, AbstractCharacter]:
		return {}

	@cached_property
	def _plans(self) -> dict[Plan, Time]:
		return {}

	@cached_property
	def _branches_plans(self) -> dict[Branch, set[Plan]]:
		return defaultdict(set)

	@cached_property
	def _plan_ticks(self) -> dict[Plan, dict[Turn, set[Tick]]]:
		return defaultdict(lambda: defaultdict(set))

	@cached_property
	def _time_plan(self) -> dict[Time, Plan]:
		return {}

	@cached_property
	def _graph_cache(self) -> EntitylessCache:
		return EntitylessCache(self, name="graph cache")

	@cached_property
	def _graph_val_cache(self) -> GraphValCache:
		ret = GraphValCache(self, name="graph val cache")
		ret.setdb = self.query.graph_val_set
		ret.deldb = self.query.graph_val_del_time
		return ret

	@cached_property
	def _nodes_cache(self) -> NodesCache:
		ret = NodesCache(self, name="nodes cache")
		ret.setdb = self.query.exist_node
		ret.deldb = self.query.nodes_del_time
		return ret

	@cached_property
	def _edges_cache(self) -> EdgesCache:
		ret = EdgesCache(self, name="edges cache")
		ret.setdb = self.query.exist_edge
		ret.deldb = self.query.edges_del_time
		return ret

	@cached_property
	def _node_val_cache(self) -> NodeValCache:
		ret = NodeValCache(self, name="node val cache")
		ret.setdb = self.query.node_val_set
		ret.deldb = self.query.node_val_del_time
		return ret

	@cached_property
	def _edge_val_cache(self) -> EdgeValCache:
		ret = EdgeValCache(self, name="edge val cache")
		ret.setdb = self.query.edge_val_set
		ret.deldb = self.query.edge_val_del_time
		return ret

	@cached_property
	def _things_cache(self) -> ThingsCache:
		ret = ThingsCache(self, name="things cache")
		ret.setdb = self.query.set_thing_loc
		return ret

	@cached_property
	def _node_contents_cache(self) -> NodeContentsCache:
		return NodeContentsCache(self, name="node contents cache")

	@cached_property
	def _neighbors_cache(self) -> SizedDict:
		return SizedDict()

	@cached_property
	def _universal_cache(self) -> EntitylessCache:
		ret = EntitylessCache(self, name="universal cache")
		ret.setdb = self.query.universal_set
		return ret

	@cached_property
	def _rulebooks_cache(self) -> InitializedEntitylessCache:
		ret = InitializedEntitylessCache(self, name="rulebooks cache")
		ret.setdb = self.query.rulebook_set
		return ret

	@cached_property
	def _characters_rulebooks_cache(self) -> CharactersRulebooksCache:
		return CharactersRulebooksCache(
			self, name="characters rulebooks cache"
		)

	@cached_property
	def _units_rulebooks_cache(self) -> CharactersRulebooksCache:
		return CharactersRulebooksCache(self, name="units rulebooks cache")

	@cached_property
	def _characters_things_rulebooks_cache(self) -> CharactersRulebooksCache:
		return CharactersRulebooksCache(
			self, name="characters things rulebooks cache"
		)

	@cached_property
	def _characters_places_rulebooks_cache(self) -> CharactersRulebooksCache:
		return CharactersRulebooksCache(
			self, name="characters places rulebooks cache"
		)

	@cached_property
	def _characters_portals_rulebooks_cache(self) -> CharactersRulebooksCache:
		return CharactersRulebooksCache(
			self, name="characters portals rulebooks cache"
		)

	@cached_property
	def _nodes_rulebooks_cache(self) -> NodesRulebooksCache:
		return NodesRulebooksCache(self, name="nodes rulebooks cache")

	@cached_property
	def _portals_rulebooks_cache(self) -> PortalsRulebooksCache:
		return PortalsRulebooksCache(self, name="portals rulebooks_ ache")

	@cached_property
	def _triggers_cache(self) -> InitializedEntitylessCache:
		return InitializedEntitylessCache(self, name="triggers cache")

	@cached_property
	def _prereqs_cache(self) -> InitializedEntitylessCache:
		return InitializedEntitylessCache(self, name="prereqs cache")

	@cached_property
	def _actions_cache(self) -> InitializedEntitylessCache:
		return InitializedEntitylessCache(self, name="actions cache")

	@cached_property
	def _neighborhoods_cache(self) -> InitializedEntitylessCache:
		return InitializedEntitylessCache(self, name="neighborhoods cache")

	@cached_property
	def _rule_bigness_cache(self) -> InitializedEntitylessCache:
		return InitializedEntitylessCache(self, name="rule bigness cache")

	@cached_property
	def _node_rules_handled_cache(self) -> NodeRulesHandledCache:
		return NodeRulesHandledCache(self, name="node rules handled cache")

	@cached_property
	def _portal_rules_handled_cache(self) -> PortalRulesHandledCache:
		return PortalRulesHandledCache(self, name="portal rules handled cache")

	@cached_property
	def _character_rules_handled_cache(self) -> CharacterRulesHandledCache:
		return CharacterRulesHandledCache(
			self, name="character rules handled cache"
		)

	@cached_property
	def _unit_rules_handled_cache(self) -> UnitRulesHandledCache:
		return UnitRulesHandledCache(self, name="unit rules handled cache")

	@cached_property
	def _character_thing_rules_handled_cache(
		self,
	) -> CharacterThingRulesHandledCache:
		return CharacterThingRulesHandledCache(
			self, name="character thing rules handled cache"
		)

	@cached_property
	def _character_place_rules_handled_cache(
		self,
	) -> CharacterPlaceRulesHandledCache:
		return CharacterPlaceRulesHandledCache(
			self, name="character place rules handled cache"
		)

	@cached_property
	def _character_portal_rules_handled_cache(
		self,
	) -> CharacterPortalRulesHandledCache:
		return CharacterPortalRulesHandledCache(
			self, name="character portal rules handled cache"
		)

	@cached_property
	def _unitness_cache(self) -> UnitnessCache:
		return UnitnessCache(self, name="unitness cache")

	@cached_property
	def _turns_completed_d(self) -> dict[Branch, Turn]:
		return {}

	@cached_property
	def universal(self) -> UniversalMapping:
		return UniversalMapping(self)

	@cached_property
	def rule(self) -> AllRules:
		return AllRules(self)

	@cached_property
	def rulebook(self) -> AllRuleBooks:
		return AllRuleBooks(self)

	@cached_property
	def _keyframes_dict(self) -> dict[Branch, dict[Turn, set[Tick]]]:
		return PickyDefaultDict(WindowDict)

	@cached_property
	def _keyframes_times(self) -> set[Time]:
		return set()

	@cached_property
	def _keyframes_loaded(self) -> set[Time]:
		return set()

	@cached_property
	def _caches(self) -> list:
		return [
			self._things_cache,
			self._node_contents_cache,
			self._universal_cache,
			self._rulebooks_cache,
			self._characters_rulebooks_cache,
			self._units_rulebooks_cache,
			self._characters_things_rulebooks_cache,
			self._characters_places_rulebooks_cache,
			self._characters_portals_rulebooks_cache,
			self._nodes_rulebooks_cache,
			self._portals_rulebooks_cache,
			self._triggers_cache,
			self._prereqs_cache,
			self._actions_cache,
			self._character_rules_handled_cache,
			self._unit_rules_handled_cache,
			self._character_thing_rules_handled_cache,
			self._character_place_rules_handled_cache,
			self._character_portal_rules_handled_cache,
			self._node_rules_handled_cache,
			self._portal_rules_handled_cache,
			self._unitness_cache,
			self._graph_val_cache,
			self._nodes_cache,
			self._edges_cache,
			self._node_val_cache,
			self._edge_val_cache,
		]

	@cached_property
	def character(self) -> CharacterMapping:
		return CharacterMapping(self)

	def _btt(self) -> Time:
		"""Return the branch, turn, and tick."""
		return Branch(self._obranch), Turn(self._oturn), Tick(self._otick)

	def _set_btt(self, branch: Branch, turn: Turn, tick: Tick):
		(self._obranch, self._oturn, self._otick) = (branch, turn, tick)

	@world_locked
	def _nbtt(self) -> Time:
		"""Increment the tick and return branch, turn, tick

		Unless we're viewing the past, in which case raise HistoryError.

		Idea is you use this when you want to advance time, which you
		can only do once per branch, turn, tick.

		"""
		(
			btt,
			turn_end_plan,
			turn_end,
			plan_ticks,
			time_plan,
		) = self._nbtt_stuff
		branch, turn, tick = btt()
		branch_turn = (branch, turn)
		tick += 1
		if branch_turn in turn_end_plan and tick <= turn_end_plan[branch_turn]:
			tick = turn_end_plan[branch_turn] + 1
		if branch_turn in turn_end and turn_end[branch_turn] > tick:
			raise HistoricKeyError(
				"You're not at the end of turn {}. "
				"Go to tick {} to change things".format(
					turn, turn_end[branch_turn]
				)
			)
		if self._planning:
			last_plan = self._last_plan
			if (turn, tick) in plan_ticks[last_plan]:
				raise OutOfTimelineError(
					"Trying to make a plan at {}, "
					"but that time already happened".format(
						(branch, turn, tick)
					),
					self.branch,
					self.turn,
					self.tick,
					self.branch,
					self.turn,
					tick,
				)
			plan_ticks[last_plan][turn].add(tick)
			self.query.plan_ticks_insert(last_plan, turn, tick)
			time_plan[branch, turn, tick] = last_plan
		else:
			end_turn, _ = self._branch_end(branch)
			if turn < end_turn:
				raise OutOfTimelineError(
					"You're in the past. Go to turn {} to change things"
					" -- or start a new branch".format(end_turn),
					*btt(),
					branch,
					turn,
					tick,
				)
			elif turn == end_turn and (branch, turn) in turn_end_plan:
				# Accept any plans made for this turn
				tick = turn_end_plan[branch, turn] + 1
			if tick > turn_end[branch_turn]:
				turn_end[branch_turn] = tick
		loaded = self._loaded
		if branch in loaded:
			(early_turn, early_tick, late_turn, late_tick) = loaded[branch]
			if late_turn is not None:
				if turn > late_turn:
					(late_turn, late_tick) = (turn, tick)
				elif turn == late_turn and tick > late_tick:
					late_tick = tick
			loaded[branch] = (early_turn, early_tick, late_turn, late_tick)
		else:
			loaded[branch] = (turn, tick, turn, tick)
		self._extend_branch(branch, turn, tick)
		then = self._btt()
		self._otick = tick
		self.time.send(self, then=then, now=self._btt())
		return branch, turn, tick

	def is_ancestor_of(self, parent: Branch, child: Branch) -> bool:
		"""Return whether ``child`` is a branch descended from ``parent``

		At any remove.

		"""
		branches = self.branches()
		if parent not in branches:
			raise ValueError("Not a branch", parent)
		if child not in branches:
			raise ValueError("Not a branch", child)
		if parent is None or parent == child or parent == self.main_branch:
			return True
		if child == self.main_branch:
			return False
		if self.branch_parent(child) == parent:
			return True
		return self.is_ancestor_of(parent, self.branch_parent(child))

	def __getattr__(self, item):
		try:
			return MethodType(
				getattr(super().__getattribute__("method"), item), self
			)
		except AttributeError:
			raise AttributeError("No such attribute", item)

	def __hasattr__(self, item):
		return hasattr(super().__getattribute__("method"), item)

	def _graph_state_hash(
		self, nodes: NodeValDict, edges: EdgeValDict, vals: StatDict
	) -> bytes:
		from hashlib import blake2b

		qpac = self.query.pack

		if isinstance(qpac(" "), str):

			def pack(x):
				return qpac(x).encode()
		else:
			pack = qpac
		nodes_hash = 0
		for name, val in nodes.items():
			hash = blake2b(pack(name))
			hash.update(pack(val))
			nodes_hash ^= int.from_bytes(hash.digest(), "little")
		edges_hash = 0
		for orig, dests in edges.items():
			for dest, idxs in dests.items():
				for idx, val in idxs.items():
					hash = blake2b(pack(orig))
					hash.update(pack(dest))
					hash.update(pack(idx))
					hash.update(pack(val))
					edges_hash ^= int.from_bytes(hash.digest(), "little")
		val_hash = 0
		for key, val in vals.items():
			hash = blake2b(pack(key))
			hash.update(pack(val))
			val_hash ^= int.from_bytes(hash.digest(), "little")
		total_hash = blake2b(nodes_hash.to_bytes(64, "little"))
		total_hash.update(edges_hash.to_bytes(64, "little"))
		total_hash.update(val_hash.to_bytes(64, "little"))
		return total_hash.digest()

	def _kfhash(
		self,
		graphn: Key,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		nodes: NodeValDict,
		edges: EdgeValDict,
		vals: StatDict,
	) -> bytes:
		"""Return a hash digest of a keyframe"""
		from hashlib import blake2b

		qpac = self.query.pack

		if isinstance(qpac(" "), str):

			def pack(x):
				return qpac(x).encode()
		else:
			pack = qpac
		total_hash = blake2b(pack(graphn))
		total_hash.update(pack(branch))
		total_hash.update(pack(turn))
		total_hash.update(pack(tick))
		total_hash.update(self._graph_state_hash(nodes, edges, vals))
		return total_hash.digest()

	def _get_node(self, graph: Key | Graph, node: Key):
		node_objs, node_exists, make_node = self._get_node_stuff
		if type(graph) is str:
			graphn = graph
			graph = self.character[graphn]
		else:
			graphn = graph.name
		key = (graphn, node)
		if key in node_objs:
			ret = node_objs[key]
			if ret._validate_node_type():
				return ret
			else:
				del node_objs[key]
		if not node_exists(graphn, node):
			raise KeyError("No such node: {} in {}".format(node, graphn))
		ret = make_node(graph, node)
		node_objs[key] = ret
		return ret

	def _get_edge(self, graph, orig, dest, idx=0):
		edge_objs, edge_exists, make_edge = self._get_edge_stuff
		if type(graph) is str:
			graphn = graph
			graph = self.character[graphn]
		else:
			graphn = graph.name
		key = (graphn, orig, dest, idx)
		if key in edge_objs:
			return edge_objs[key]
		if not edge_exists(graphn, orig, dest, idx):
			raise KeyError(
				"No such edge: {}->{}[{}] in {}".format(
					orig, dest, idx, graphn
				)
			)
		ret = make_edge(graph, orig, dest, idx)
		edge_objs[key] = ret
		return ret

	def plan(self, reset=True) -> PlanningContext:
		__doc__ = PlanningContext.__doc__
		return PlanningContext(self, reset)

	@world_locked
	def _copy_plans(
		self, branch_from: Branch, turn_from: Turn, tick_from: Tick
	) -> None:
		"""Copy all plans active at the given time to the current branch"""
		plan_ticks = self._plan_ticks
		time_plan = self._time_plan
		plans = self._plans
		branch = self.branch
		where_cached = self._where_cached
		turn_end_plan = self._turn_end_plan
		was_planning = self._planning
		self._planning = True
		for plan_id in self._branches_plans[branch_from]:
			_, start_turn, start_tick = plans[plan_id]
			if (
				branch_from,
				start_turn,
			) not in turn_end_plan or start_tick > turn_end_plan[
				branch_from, start_turn
			]:
				turn_end_plan[branch_from, start_turn] = start_tick
			if (start_turn, start_tick) > (turn_from, tick_from):
				continue
			incremented = False
			for turn, ticks in list(plan_ticks[plan_id].items()):
				if turn < turn_from:
					continue
				for tick in ticks:
					if (turn, tick) < (turn_from, tick_from):
						continue
					if not incremented:
						self._last_plan += 1
						incremented = True
						plans[self._last_plan] = branch, turn, tick
					if (
						branch,
						turn,
					) not in turn_end_plan or tick > turn_end_plan[
						branch, turn
					]:
						turn_end_plan[branch, turn] = tick
					plan_ticks[self._last_plan][turn].add(tick)
					self.query.plan_ticks_insert(self._last_plan, turn, tick)
					for cache in where_cached[branch_from, turn, tick]:
						data = cache.settings[branch_from][turn][tick]
						value = data[-1]
						key = data[:-1]
						args = key + (branch, turn, tick, value)
						if hasattr(cache, "setdb"):
							cache.setdb(*args)
						cache.store(*args, planning=True)
						time_plan[branch, turn, tick] = self._last_plan
		self._planning = was_planning

	@world_locked
	def delete_plan(self, plan: Plan) -> None:
		"""Delete the portion of a plan that has yet to occur.

		:arg plan: integer ID of a plan, as given by
				   ``with self.plan() as plan:``

		"""
		branch, turn, tick = self._btt()
		to_delete = []
		plan_ticks = self._plan_ticks[plan]
		for (
			trn,
			tcks,
		) in (
			plan_ticks.items()
		):  # might improve performance to use a WindowDict for plan_ticks
			if turn == trn:
				for tck in tcks:
					if tck >= tick:
						to_delete.append((trn, tck))
			elif trn > turn:
				to_delete.extend((trn, tck) for tck in tcks)
		# Delete stuff that happened at contradicted times,
		# and then delete the times from the plan
		where_cached = self._where_cached
		time_plan = self._time_plan
		for trn, tck in to_delete:
			for cache in where_cached[branch, trn, tck]:
				cache.remove(branch, trn, tck)
				if hasattr(cache, "deldb"):
					cache.deldb(branch, trn, tck)
			del where_cached[branch, trn, tck]
			plan_ticks[trn].remove(tck)
			if not plan_ticks[trn]:
				del plan_ticks[trn]
			del time_plan[branch, trn, tck]

	@contextmanager
	def advancing(self):
		"""A context manager for when time is moving forward one turn at a time.

		When used in lisien, this means that the game is being simulated.
		It changes how the caching works, making it more efficient.

		"""
		if self._forward:
			raise ValueError("Already advancing")
		self._forward = True
		yield
		self._forward = False

	@contextmanager
	def batch(self):
		"""A context manager for when you're creating lots of state.

		Reads will be much slower in a batch, but writes will be faster.

		You *can* combine this with ``advancing`` but it isn't any faster.

		"""
		if self._no_kc:
			yield
			return
		self._no_kc = True
		with garbage():
			yield
		self._no_kc = False

	def _set_graph_in_delta(
		self,
		branch: Branch,
		turn_from: Turn,
		tick_from: Tick,
		turn_to: Turn,
		tick_to: Tick,
		delta: DeltaDict,
		_: None,
		graph: Key,
		val: Any,
	) -> None:
		"""Change a delta to say that a graph was deleted or not"""
		if val in (None, "Deleted"):
			delta[graph] = None
		elif graph not in delta or delta[graph] is None:
			# If the graph was *created* within our window,
			# include its whole initial keyframe
			delta[graph] = {}
			kf_time = None
			the_kf = None
			graph_kf = self._graph_cache.keyframe[None,]
			if branch in graph_kf:
				kfb = graph_kf[branch]
				if turn_from == turn_to:
					# past view is reverse chronological
					for t in kfb[turn_from].past(tick_to):
						if tick_from <= t:
							break
						elif t < tick_from:
							return
					else:
						return
					kf_time = branch, turn_from, t
					the_kf = graph_kf[branch][turn_from][t]
				elif (
					turn_from in kfb
					and kfb[turn_from].end > tick_from
					and graph
					in (
						the_kf := graph_kf[branch][turn_from][
							kfb[turn_from].end
						]
					)
				):
					kf_time = branch, turn_from, kfb[turn_from].end
					the_kf = graph_kf[branch][turn_from][kf_time[2]]
				elif (
					kfb.rev_after(turn_from) is not None
					and kfb.rev_before(turn_to) is not None
					and kfb.rev_after(turn_from)
					<= (r := kfb.rev_before(turn_to))
				):
					if r == turn_to:
						if (
							kfb[r].end < tick_to
							and graph in graph_kf[branch][r][kfb[r].end]
						):
							kf_time = branch, r, kfb[r].end
							the_kf = graph_kf[branch][r][kf_time[2]]
					else:
						the_kf = graph_kf[branch][r][kfb[r].end]
						if graph in the_kf:
							kf_time = branch, r, kfb[r].end
			if kf_time is not None:
				assert isinstance(the_kf, dict)
				# Well, we have *a keyframe* attesting the graph's existence,
				# but we don't know it was *created* at that time.
				# Check the presettings; if there was no type set for the
				# graph before this keyframe, then it's the keyframe
				# in which the graph was created.
				# (An absence of presettings data indicates that the graph
				# existed prior to the current branch.)
				preset = self._graph_cache.presettings
				b, r, t = kf_time
				assert b == branch
				if (
					b not in preset
					or r not in preset[b]
					or t not in preset[b][r]
					or preset[b][r][t][2] is None
				):
					return
				delta[graph] = {}

	def _get_branch_delta(
		self,
		branch: Branch,
		turn_from: Turn,
		tick_from: Tick,
		turn_to: Turn,
		tick_to: Tick,
	) -> DeltaDict:
		"""Get a dictionary describing changes to all graphs.

		The keys are graph names. Their values are dictionaries of the
		graphs' attributes' new values, with ``None`` for deleted keys. Also
		in those graph dictionaries are special keys 'node_val' and
		'edge_val' describing changes to node and edge attributes,
		and 'nodes' and 'edges' full of booleans indicating whether a node
		or edge exists.

		"""

		setgraph = partial(
			self._set_graph_in_delta,
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)

		def setgraphval(
			delta: DeltaDict, graph: Key, key: Key, val: Any
		) -> None:
			"""Change a delta to say that a graph stat was set to a certain value"""
			if (graphstat := delta.setdefault(graph, {})) is not None:
				graphstat[key] = val

		def setnode(
			delta: DeltaDict, graph: Key, node: Key, exists: bool | None
		) -> None:
			"""Change a delta to say that a node was created or deleted"""
			if (graphstat := delta.setdefault(graph, {})) is not None:
				graphstat.setdefault("nodes", {})[node] = bool(exists)

		def setnodeval(
			delta: DeltaDict, graph: Key, node: Key, key: Key, value: Any
		) -> None:
			"""Change a delta to say that a node stat was set to a certain value"""
			if (graphstat := delta.setdefault(graph, {})) is not None:
				if (
					graph in delta
					and "nodes" in delta[graph]
					and node in delta[graph]["nodes"]
					and not delta[graph]["nodes"][node]
				):
					return
				graphstat.setdefault("node_val", {}).setdefault(node, {})[
					key
				] = value

		def setedge(
			delta: DeltaDict,
			is_multigraph: callable,
			graph: Key,
			orig: Key,
			dest: Key,
			idx: int,
			exists: bool | None,
		) -> None:
			"""Change a delta to say that an edge was created or deleted"""
			if (graphstat := delta.setdefault(graph, {})) is not None:
				if is_multigraph(graph):
					raise NotImplementedError("Only digraphs for now")
				if "edges" in graphstat:
					es = graphstat["edges"]
					if orig in es:
						es[orig][dest] = exists
					else:
						es[orig] = {dest: exists}
				else:
					graphstat["edges"] = {orig: {dest: exists}}

		def setedgeval(
			delta: DeltaDict,
			is_multigraph: callable,
			graph: Key,
			orig: Key,
			dest: Key,
			idx: int,
			key: Key,
			value: Any,
		) -> None:
			"""Change a delta to say that an edge stat was set to a certain value"""
			if (graphstat := delta.setdefault(graph, {})) is not None:
				if is_multigraph(graph):
					raise NotImplementedError("Only digraphs for now")
				if (
					"edges" in graphstat
					and orig in graphstat["edges"]
					and dest in graphstat["edges"][orig]
					and not graphstat["edges"][orig][dest]
				):
					return
				graphstat.setdefault("edge_val", {}).setdefault(
					orig, {}
				).setdefault(dest, {})[key] = value

		if not isinstance(branch, str):
			raise TypeError("branch must be str")
		for arg in (turn_from, tick_from, turn_to, tick_to):
			if not isinstance(arg, int):
				raise TypeError("turn and tick must be int")
		self.load_between(branch, turn_from, tick_from, turn_to, tick_to)
		if turn_from == turn_to:
			if tick_from == tick_to:
				return {}
			return self._get_turn_delta(branch, turn_to, tick_from, tick_to)
		delta = {}
		graph_objs = self._graph_objs

		if turn_from < turn_to:
			updater = partial(
				update_window, turn_from, tick_from, turn_to, tick_to
			)
			attribute = "settings"
			tick_to += 1
		else:
			updater = partial(
				update_backward_window, turn_from, tick_from, turn_to, tick_to
			)
			attribute = "presettings"
		gbranches = getattr(self._graph_cache, attribute)
		gvbranches = getattr(self._graph_val_cache, attribute)
		nbranches = getattr(self._nodes_cache, attribute)
		nvbranches = getattr(self._node_val_cache, attribute)
		ebranches = getattr(self._edges_cache, attribute)
		evbranches = getattr(self._edge_val_cache, attribute)
		univbranches = getattr(self._universal_cache, attribute)
		unitbranches = getattr(self._unitness_cache, attribute)
		thbranches = getattr(self._things_cache, attribute)
		rbbranches = getattr(self._rulebooks_cache, attribute)
		trigbranches = getattr(self._triggers_cache, attribute)
		preqbranches = getattr(self._prereqs_cache, attribute)
		actbranches = getattr(self._actions_cache, attribute)
		nbrbranches = getattr(self._neighborhoods_cache, attribute)
		bigbranches = getattr(self._rule_bigness_cache, attribute)
		charrbbranches = getattr(self._characters_rulebooks_cache, attribute)
		avrbbranches = getattr(self._units_rulebooks_cache, attribute)
		charthrbbranches = getattr(
			self._characters_things_rulebooks_cache, attribute
		)
		charplrbbranches = getattr(
			self._characters_places_rulebooks_cache, attribute
		)
		charporbbranches = getattr(
			self._characters_portals_rulebooks_cache, attribute
		)
		noderbbranches = getattr(self._nodes_rulebooks_cache, attribute)
		edgerbbranches = getattr(self._portals_rulebooks_cache, attribute)

		if branch in gbranches:
			updater(partial(setgraph, delta), gbranches[branch])

		if branch in gvbranches:
			updater(partial(setgraphval, delta), gvbranches[branch])

		if branch in nbranches:
			updater(partial(setnode, delta), nbranches[branch])

		if branch in nvbranches:
			updater(partial(setnodeval, delta), nvbranches[branch])

		if branch in ebranches:
			updater(
				partial(
					setedge, delta, lambda g: graph_objs[g].is_multigraph()
				),
				ebranches[branch],
			)

		if branch in evbranches:
			updater(
				partial(
					setedgeval, delta, lambda g: graph_objs[g].is_multigraph()
				),
				evbranches[branch],
			)

		def upduniv(_, key, val):
			delta.setdefault("universal", {})[key] = val

		if branch in univbranches:
			updater(upduniv, univbranches[branch])

		def updunit(char, graph, units_d):
			if units_d is None:
				if char not in delta:
					delta[char] = {"units": {graph: None}}
				elif delta[char] is None:
					return
				elif "units" not in delta[char]:
					delta[char]["units"] = {graph: None}
				else:
					delta[char]["units"][graph] = None
				return
			elif isinstance(units_d, tuple):
				node, av = units_d
				delta.setdefault(char, {}).setdefault("units", {}).setdefault(
					graph, {}
				)[node] = bool(av)
				return
			if char not in delta:
				delta[char] = {"units": {graph: units_d.copy()}}
			elif "units" not in delta[char]:
				delta[char]["units"] = {graph: units_d.copy()}
			elif graph not in delta[char]["units"]:
				delta[char]["units"][graph] = units_d.copy()
			else:
				delta[char]["units"][graph].update(units_d)

		if branch in unitbranches:
			updater(updunit, unitbranches[branch])

		def updthing(char, thing, loc):
			if char in delta and (
				delta[char] is None
				or (
					"nodes" in delta[char]
					and thing in delta[char]["nodes"]
					and not delta[char]["nodes"][thing]
				)
			):
				return
			thingd = (
				delta.setdefault(char, {})
				.setdefault("node_val", {})
				.setdefault(thing, {})
			)
			thingd["location"] = loc

		if branch in thbranches:
			updater(updthing, thbranches[branch])

		def updrb(_, rulebook, rules):
			delta.setdefault("rulebooks", {})[rulebook] = rules

		if branch in rbbranches:
			updater(updrb, rbbranches[branch])

		def updru(key, _, rule, funs):
			delta.setdefault("rules", {}).setdefault(rule, {})[key] = funs

		if branch in trigbranches:
			updater(partial(updru, "triggers"), trigbranches[branch])

		if branch in preqbranches:
			updater(partial(updru, "prereqs"), preqbranches[branch])

		if branch in actbranches:
			updater(partial(updru, "actions"), actbranches[branch])

		if branch in nbrbranches:
			updater(partial(updru, "neighborhood"), nbrbranches[branch])

		if branch in bigbranches:
			updater(partial(updru, "big"), bigbranches[branch])

		def updcrb(key, _, character, rulebook):
			if character in delta and delta[character] is None:
				return
			delta.setdefault(character, {})[key] = rulebook

		if branch in charrbbranches:
			updater(
				partial(updcrb, "character_rulebook"), charrbbranches[branch]
			)

		if branch in avrbbranches:
			updater(partial(updcrb, "unit_rulebook"), avrbbranches[branch])

		if branch in charthrbbranches:
			updater(
				partial(updcrb, "character_thing_rulebook"),
				charthrbbranches[branch],
			)

		if branch in charplrbbranches:
			updater(
				partial(updcrb, "character_place_rulebook"),
				charplrbbranches[branch],
			)

		if branch in charporbbranches:
			updater(
				partial(updcrb, "character_portal_rulebook"),
				charporbbranches[branch],
			)

		def updnoderb(character, node, rulebook):
			if (character in delta) and (
				delta[character] is None
				or (
					"nodes" in delta[character]
					and node in delta[character]["nodes"]
					and not delta[character]["nodes"][node]
				)
			):
				return
			delta.setdefault(character, {}).setdefault(
				"node_val", {}
			).setdefault(node, {})["rulebook"] = rulebook

		if branch in noderbbranches:
			updater(updnoderb, noderbbranches[branch])

		def updedgerb(character, orig, dest, rulebook=None):
			if rulebook is None:
				# It's one of those updates that stores all the rulebooks from
				# some origin. Not relevant to deltas.
				return
			if character in delta and (
				delta[character] is None
				or (
					"edges" in delta[character]
					and orig in delta[character]["edges"]
					and dest in delta[character]["edges"][orig]
					and not delta[character]["edges"][orig][dest]
				)
			):
				return
			delta.setdefault(character, {}).setdefault(
				"edge_val", {}
			).setdefault(orig, {}).setdefault(dest, {})["rulebook"] = rulebook

		if branch in edgerbbranches:
			updater(updedgerb, edgerbbranches[branch])

		return delta

	def _get_turn_delta(
		self,
		branch: Branch = None,
		turn: Turn = None,
		tick_from: Tick = 0,
		tick_to: Tick = None,
	) -> DeltaDict:
		"""Get a dictionary describing changes made on a given turn.

		If ``tick_to`` is not supplied, report all changes after ``tick_from``
		(default 0).

		The keys are graph names. Their values are dictionaries of the
		graphs' attributes' new values, with ``None`` for deleted keys. Also
		in those graph dictionaries are special keys 'node_val' and
		'edge_val' describing changes to node and edge attributes,
		and 'nodes' and 'edges' full of booleans indicating whether a node
		or edge exists.

		:arg branch: A branch of history; defaults to the current branch
		:arg turn: The turn in the branch; defaults to the current turn
		:arg tick_from: Starting tick; defaults to 0
		:arg tick_to: tick at which to stop the delta, defaulting to the
				   present tick if it's the present turn, or the end
				   tick if it's any other turn

		"""
		branch = branch or self.branch
		turn = turn or self.turn
		if tick_to is None:
			if turn == self.turn:
				tick_to = self.tick
			else:
				tick_to = self._turn_end[branch, turn]
		branch = branch or self.branch
		turn = turn or self.turn
		tick_to = tick_to or self.tick
		self.load_between(branch, turn, tick_from, turn, tick_to)
		delta = {}
		if tick_from < tick_to:
			attribute = "settings"
			tick_to += 1
		else:
			attribute = "presettings"
		gbranches = getattr(self._graph_cache, attribute)
		gvbranches = getattr(self._graph_val_cache, attribute)
		nbranches = getattr(self._nodes_cache, attribute)
		nvbranches = getattr(self._node_val_cache, attribute)
		ebranches = getattr(self._edges_cache, attribute)
		evbranches = getattr(self._edge_val_cache, attribute)
		universals_settings = getattr(self._universal_cache, attribute)
		unitness_settings = getattr(self._unitness_cache, attribute)
		things_settings = getattr(self._things_cache, attribute)
		rulebooks_settings = getattr(self._rulebooks_cache, attribute)
		triggers_settings = getattr(self._triggers_cache, attribute)
		prereqs_settings = getattr(self._prereqs_cache, attribute)
		actions_settings = getattr(self._actions_cache, attribute)
		neighborhood_settings = getattr(self._neighborhoods_cache, attribute)
		big_settings = getattr(self._rule_bigness_cache, attribute)
		character_rulebooks_settings = getattr(
			self._characters_rulebooks_cache, attribute
		)
		avatar_rulebooks_settings = getattr(
			self._units_rulebooks_cache, attribute
		)
		character_thing_rulebooks_settings = getattr(
			self._characters_things_rulebooks_cache, attribute
		)
		character_place_rulebooks_settings = getattr(
			self._characters_places_rulebooks_cache, attribute
		)
		character_portal_rulebooks_settings = getattr(
			self._characters_portals_rulebooks_cache, attribute
		)
		node_rulebooks_settings = getattr(
			self._nodes_rulebooks_cache, attribute
		)
		portal_rulebooks_settings = getattr(
			self._portals_rulebooks_cache, attribute
		)

		if branch in gbranches and turn in gbranches[branch]:
			for _, graph, typ in gbranches[branch][turn][tick_from:tick_to]:
				# typ may be None if the graph was never deleted, but we're
				# traveling back to before it was created
				self._set_graph_in_delta(
					branch,
					turn,
					tick_from,
					turn,
					tick_to,
					delta,
					None,
					graph,
					typ,
				)

		if branch in gvbranches and turn in gvbranches[branch]:
			for graph, key, value in gvbranches[branch][turn][
				tick_from:tick_to
			]:
				if graph in delta:
					if delta[graph] is None:
						continue
					delta[graph][key] = value
				else:
					delta[graph] = {key: value}

		if branch in nbranches and turn in nbranches[branch]:
			for graph, node, exists in nbranches[branch][turn][
				tick_from:tick_to
			]:
				if graph in delta and delta[graph] is None:
					continue
				delta.setdefault(graph, {}).setdefault("nodes", {})[node] = (
					bool(exists)
				)

		if branch in nvbranches and turn in nvbranches[branch]:
			for graph, node, key, value in nvbranches[branch][turn][
				tick_from:tick_to
			]:
				if graph in delta and (
					delta[graph] is None
					or (
						"nodes" in delta[graph]
						and node in delta[graph]["nodes"]
						and not delta[graph]["nodes"][node]
					)
				):
					continue
				nodevd = delta.setdefault(graph, {}).setdefault("node_val", {})
				if node in nodevd:
					nodevd[node][key] = value
				else:
					nodevd[node] = {key: value}

		graph_objs = self._graph_objs
		if branch in ebranches and turn in ebranches[branch]:
			for graph, orig, dest, idx, exists in ebranches[branch][turn][
				tick_from:tick_to
			]:
				if graph_objs[graph].is_multigraph():
					if graph in delta and (
						delta[graph] is None
						or (
							"edges" in delta[graph]
							and orig in delta[graph]["edges"]
							and dest in delta[graph]["edges"][orig]
							and idx in delta[graph]["edges"][orig][dest]
							and not delta[graph]["edges"][orig][dest][idx]
						)
					):
						continue
					delta.setdefault(graph, {}).setdefault(
						"edges", {}
					).setdefault(orig, {})[dest] = bool(exists)
				else:
					if graph in delta and (
						delta[graph] is None
						or (
							"edges" in delta[graph]
							and orig in delta[graph]["edges"]
							and dest in delta[graph]["edges"][orig]
							and not delta[graph]["edges"][orig][dest]
						)
					):
						continue
					delta.setdefault(graph, {}).setdefault(
						"edges", {}
					).setdefault(orig, {})[dest] = bool(exists)
		pass
		if branch in evbranches and turn in evbranches[branch]:
			for graph, orig, dest, idx, key, value in evbranches[branch][turn][
				tick_from:tick_to
			]:
				if graph in delta and delta[graph] is None:
					continue
				edgevd = (
					delta.setdefault(graph, {})
					.setdefault("edge_val", {})
					.setdefault(orig, {})
					.setdefault(dest, {})
				)
				if graph_objs[graph].is_multigraph():
					if idx in edgevd:
						edgevd[idx][key] = value
					else:
						edgevd[idx] = {key: value}
				else:
					edgevd[key] = value
		if (
			branch in universals_settings
			and turn in universals_settings[branch]
		):
			for _, key, val in universals_settings[branch][turn][
				tick_from:tick_to
			]:
				delta.setdefault("universal", {})[key] = val
		if branch in unitness_settings and turn in unitness_settings[branch]:
			for chara, graph, *stuff in unitness_settings[branch][turn][
				tick_from:tick_to
			]:
				if (graph in delta and delta[graph] is None) or (
					not isinstance(stuff, list)
					or len(stuff) != 1
					or not isinstance(stuff[0], dict)
				):
					continue
				chardelt = delta.setdefault(chara, {})
				if chardelt is None:
					continue
				for node, is_av in stuff[0].items():
					chardelt.setdefault("units", {}).setdefault(graph, {})[
						node
					] = is_av
		if branch in things_settings and turn in things_settings[branch]:
			for chara, thing, location in things_settings[branch][turn][
				tick_from:tick_to
			]:
				if chara in delta and delta[chara] is None:
					continue
				thingd = (
					delta.setdefault(chara, {})
					.setdefault("node_val", {})
					.setdefault(thing, {})
				)
				thingd["location"] = location
		delta["rulebooks"] = rbdif = {}
		if branch in rulebooks_settings and turn in rulebooks_settings[branch]:
			for _, rulebook, rules in rulebooks_settings[branch][turn][
				tick_from:tick_to
			]:
				rbdif[rulebook] = rules
		delta["rules"] = rdif = {}
		if branch in triggers_settings and turn in triggers_settings[branch]:
			for _, rule, funs in triggers_settings[branch][turn][
				tick_from:tick_to
			]:
				rdif.setdefault(rule, {})["triggers"] = funs
		if branch in prereqs_settings and turn in prereqs_settings[branch]:
			for _, rule, funs in prereqs_settings[branch][turn][
				tick_from:tick_to
			]:
				rdif.setdefault(rule, {})["prereqs"] = funs
		if branch in actions_settings and turn in actions_settings[branch]:
			for _, rule, funs in actions_settings[branch][turn][
				tick_from:tick_to
			]:
				rdif.setdefault(rule, {})["actions"] = funs
		if (
			branch in neighborhood_settings
			and turn in neighborhood_settings[branch]
		):
			for _, rule, neighborhood in neighborhood_settings[branch][turn][
				tick_from:tick_to
			]:
				rdif.setdefault(rule, {})["neighborhood"] = neighborhood
		if branch in big_settings and turn in big_settings[branch]:
			for _, rule, big in big_settings[branch][turn][tick_from:tick_to]:
				rdif.setdefault(rule, {})["big"] = big

		if (
			branch in character_rulebooks_settings
			and turn in character_rulebooks_settings[branch]
		):
			for _, character, rulebook in character_rulebooks_settings[branch][
				turn
			][tick_from:tick_to]:
				chardelt = delta.setdefault(character, {})
				if chardelt is None:
					continue
				chardelt["character_rulebook"] = rulebook
		if (
			branch in avatar_rulebooks_settings
			and turn in avatar_rulebooks_settings[branch]
		):
			for _, character, rulebook in avatar_rulebooks_settings[branch][
				turn
			][tick_from:tick_to]:
				chardelt = delta.setdefault(character, {})
				if chardelt is None:
					continue
				chardelt["unit_rulebook"] = rulebook
		if (
			branch in character_thing_rulebooks_settings
			and turn in character_thing_rulebooks_settings[branch]
		):
			for _, character, rulebook in character_thing_rulebooks_settings[
				branch
			][turn][tick_from:tick_to]:
				chardelt = delta.setdefault(character, {})
				if chardelt is None:
					continue
				chardelt["character_thing_rulebook"] = rulebook
		if (
			branch in character_place_rulebooks_settings
			and turn in character_place_rulebooks_settings[branch]
		):
			for _, character, rulebook in character_place_rulebooks_settings[
				branch
			][turn][tick_from:tick_to]:
				chardelt = delta.setdefault(character, {})
				if chardelt is None:
					continue
				chardelt["character_place_rulebook"] = rulebook
		if (
			branch in character_portal_rulebooks_settings
			and turn in character_portal_rulebooks_settings[branch]
		):
			for _, character, rulebook in character_portal_rulebooks_settings[
				branch
			][turn][tick_from:tick_to]:
				chardelt = delta.setdefault(character, {})
				if chardelt is None:
					continue
				chardelt["character_portal_rulebook"] = rulebook

		if (
			branch in node_rulebooks_settings
			and turn in node_rulebooks_settings[branch]
		):
			for character, node, rulebook in node_rulebooks_settings[branch][
				turn
			][tick_from:tick_to]:
				chardelt = delta.setdefault(character, {})
				if chardelt is None:
					continue
				chardelt.setdefault("node_val", {}).setdefault(node, {})[
					"rulebook"
				] = rulebook
		if (
			branch in portal_rulebooks_settings
			and turn in portal_rulebooks_settings[branch]
		):
			for setting in portal_rulebooks_settings[branch][turn][
				tick_from:tick_to
			]:
				try:
					character, orig, dest, rulebook = setting
				except ValueError:
					continue
				chardelt = delta.setdefault(character, {})
				if chardelt is None:
					continue
				chardelt.setdefault("edge_val", {}).setdefault(
					orig, {}
				).setdefault(dest, {})["rulebook"] = rulebook
		return delta

	@cached_property
	def next_turn(self) -> NextTurn:
		return NextTurn(self)

	@cached_property
	def world_lock(self):
		return RLock()

	@world_locked
	def __init__(
		self,
		prefix: PathLike | str | None = ".",
		*,
		string: StringStore | dict = None,
		trigger: FunctionStore | ModuleType = None,
		prereq: FunctionStore | ModuleType = None,
		action: FunctionStore | ModuleType = None,
		function: FunctionStore | ModuleType = None,
		method: FunctionStore | ModuleType = None,
		main_branch: Branch = None,
		connect_string: str = None,
		connect_args: dict = None,
		schema_cls: Type[AbstractSchema] = NullSchema,
		flush_interval: int = None,
		keyframe_interval: int | None = 1000,
		commit_interval: int = None,
		random_seed: int = None,
		logfun: callable = None,
		clear: bool = False,
		keep_rules_journal: bool = True,
		keyframe_on_close: bool = True,
		enforce_end_of_time: bool = True,
		workers: int = None,
	):
		if workers is None:
			workers = os.cpu_count()
		self._planning = False
		self._forward = False
		self._no_kc = False
		self._enforce_end_of_time = enforce_end_of_time
		self._keyframe_on_close = keyframe_on_close
		self._prefix = prefix
		self.keep_rules_journal = keep_rules_journal
		self.flush_interval = flush_interval
		self.commit_interval = commit_interval
		self.schema = schema_cls(self)
		# in case this is the first startup
		self._obranch = main_branch or "trunk"
		self._otick = self._oturn = 0
		self._init_log(logfun)
		self._init_func_stores(
			prefix, function, method, trigger, prereq, action, clear
		)
		self._init_load(
			prefix,
			connect_string,
			connect_args,
			keyframe_interval,
			main_branch,
			clear,
		)
		self._init_random(random_seed)
		self._init_string(prefix, string, clear)
		self._top_uid = 0
		if workers > 0:
			self._start_workers(prefix, workers)

	def _init_log(self, logfun: Optional[callable]):
		if logfun is None:
			from logging import getLogger

			logger = getLogger("lisien")

			def logfun(level, msg):
				if isinstance(level, int):
					logger.log(level, msg)
				else:
					getattr(logger, level)(msg)

		self.log = logfun

	def _init_func_stores(
		self,
		prefix: str | os.PathLike | None,
		function: ModuleType | FunctionStore,
		method: ModuleType | FunctionStore,
		trigger: ModuleType | FunctionStore,
		prereq: ModuleType | FunctionStore,
		action: ModuleType | FunctionStore,
		clear: bool,
	):
		for module, name in (
			(function, "function"),
			(method, "method"),
			(trigger, "trigger"),
			(prereq, "prereq"),
			(action, "action"),
		):
			if isinstance(module, ModuleType):
				setattr(self, name, module)
			elif prefix is None:
				setattr(self, name, FunctionStore(None, module=name))
			else:
				fn = os.path.join(prefix, f"{name}.py")
				setattr(self, name, FunctionStore(fn, module=name))
				if clear and os.path.exists(fn):
					os.remove(fn)

	def _init_load(
		self,
		prefix: str | os.PathLike | None,
		connect_string: str | None,
		connect_args: dict | None,
		keyframe_interval: int | None,
		main_branch: Branch,
		clear: bool,
	):
		if prefix is None:
			if connect_string is None:
				self.query = NullQueryEngine()
			else:
				self.query = SQLAlchemyQueryEngine(
					connect_string, connect_args or {}, self.pack, self.unpack
				)
		else:
			if not os.path.exists(prefix):
				os.mkdir(prefix)
			if not os.path.isdir(prefix):
				raise FileExistsError("Need a directory")
			if connect_string is None:
				self.query = ParquetQueryEngine(
					os.path.join(prefix, "world"),
					self.pack,
					self.unpack,
				)
			else:
				self.query = SQLAlchemyQueryEngine(
					connect_string, connect_args or {}, self.pack, self.unpack
				)
			if clear:
				self.query.truncate_all()

		self.query.keyframe_interval = keyframe_interval
		self._load_keyframe_times()
		if main_branch is not None:
			self.query.globl["main_branch"] = main_branch
		elif "main_branch" not in self.query.globl:
			main_branch = self.query.globl["main_branch"] = "trunk"
		else:
			main_branch = self.query.globl["main_branch"]
		assert main_branch is not None
		assert main_branch == self.query.globl["main_branch"]
		self._obranch = self.query.get_branch()
		self._oturn = self.query.get_turn()
		self._otick = self.query.get_tick()
		for (
			branch,
			parent,
			parent_turn,
			parent_tick,
			end_turn,
			end_tick,
		) in self.query.all_branches():
			self._branches_d[branch] = (
				parent,
				parent_turn,
				parent_tick,
				end_turn,
				end_tick,
			)
			self._upd_branch_parentage(parent, branch)
		for branch, turn, end_tick, plan_end_tick in self.query.turns_dump():
			self._turn_end[branch, turn] = max(
				self._turn_end[branch, turn], end_tick
			)
			self._turn_end_plan[branch, turn] = max(
				(self._turn_end_plan[branch, turn], plan_end_tick)
			)
		if main_branch not in self._branches_d:
			self._branches_d[main_branch] = None, 0, 0, 0, 0
		self._load_graphs()
		self._load_plans()
		self._load_rules_handled()
		self._turns_completed_d.update(self.query.turns_completed_dump())
		self._rules_cache = {
			name: Rule(self, name, create=False)
			for name in self.query.rules_dump()
		}
		with garbage():
			self._load(*self._read_at(*self._btt()))
		self.query.snap_keyframe = self.snap_keyframe
		self.query.kf_interval_override = self._detect_kf_interval_override
		if not self._keyframes_times:
			self._snap_keyframe_de_novo(*self._btt())

	def _init_random(self, random_seed: int | None):
		self._rando = Random()
		if "rando_state" in self.universal:
			self._rando.setstate(self.universal["rando_state"])
		else:
			self._rando.seed(random_seed)
			rando_state = self._rando.getstate()
			if self._oturn == self._otick == 0:
				now = self._btt()
				if now in self._keyframes_times:
					assert now in self._keyframes_loaded
					kf = self._universal_cache.get_keyframe(*now, copy=False)
					kf["rando_state"] = rando_state
				self._universal_cache.store(
					"rando_state",
					self.branch,
					0,
					0,
					rando_state,
					loading=True,
				)
				self.query.universal_set(
					"rando_state", self.branch, 0, 0, rando_state
				)
			else:
				self.universal["rando_state"] = rando_state

	def _init_string(
		self,
		prefix: str | os.PathLike | None,
		string: StringStore | dict | None,
		clear: bool,
	):
		if string:
			self.string = string
		elif prefix is None:
			self.string = StringStore(
				self,
				None,
				self.eternal.setdefault("language", "eng"),
			)
		elif isinstance(string, dict):
			self.string = StringStore(
				string, None, self.eternal.setdefault("language", "eng")
			)
		else:
			string_prefix = os.path.join(prefix, "strings")
			if clear and os.path.isdir(string_prefix):
				shutil.rmtree(string_prefix)
			if not os.path.exists(string_prefix):
				os.mkdir(string_prefix)
			self.string = StringStore(
				self,
				string_prefix,
				self.eternal.setdefault("language", "eng"),
			)

	def _start_workers(self, prefix: str | os.PathLike | None, workers: int):
		def sync_log_forever(q):
			while True:
				self.log(*q.get())

		for store in self.stores:
			if hasattr(store, "save"):
				store.save(reimport=False)

		self._trigger_pool = ThreadPoolExecutor()
		self._worker_last_eternal = dict(self.eternal.items())
		initial_payload = self._get_worker_kf_payload()

		self._worker_processes = wp = []
		self._worker_inputs = wi = []
		self._worker_outputs = wo = []
		self._worker_locks = wlk = []
		self._worker_log_queues = wl = []
		self._worker_log_threads = wlt = []
		for i in range(workers):
			inpipe_there, inpipe_here = Pipe(duplex=False)
			outpipe_here, outpipe_there = Pipe(duplex=False)
			logq = Queue()
			logthread = Thread(
				target=sync_log_forever, args=(logq,), daemon=True
			)
			worker_args = [
				i,
				prefix,
				self._branches_d,
				dict(self.eternal),
				inpipe_there,
				outpipe_there,
				logq,
			]
			for store in (
				self.function,
				self.method,
				self.trigger,
				self.prereq,
				self.action,
			):
				if hasattr(store, "_locl"):
					worker_args.append(store._locl)
				elif hasattr(store, "__dict__"):
					worker_args.append(
						{
							k: v
							for (k, v) in store.__dict__.items()
							if callable(v)
						}
					)
				else:
					funcs = {}
					for name in dir(store):
						value = getattr(store, name)
						if callable(value):
							funcs[name] = value
					worker_args.append(funcs)
			proc = Process(
				target=worker_subprocess,
				args=worker_args,
			)
			wi.append(inpipe_here)
			wo.append(outpipe_here)
			wl.append(logq)
			wlk.append(Lock())
			wlt.append(logthread)
			wp.append(proc)
			logthread.start()
			proc.start()
			with wlk[-1]:
				inpipe_here.send_bytes(initial_payload)
		self._how_many_futs_running = 0
		self._fut_manager_thread = Thread(
			target=self._manage_futs, daemon=True
		)
		self._futs_to_start: SimpleQueue[Future] = SimpleQueue()
		self._uid_to_fut: dict[int, Future] = {}
		self._fut_manager_thread.start()
		if hasattr(self.trigger, "connect"):
			self.trigger.connect(self._reimport_trigger_functions)
		if hasattr(self.function, "connect"):
			self.function.connect(self._reimport_worker_functions)
		if hasattr(self.method, "connect"):
			self.method.connect(self._reimport_worker_methods)
		self._worker_updated_btts = [self._btt()] * workers

	def _start_branch(
		self, parent: str, branch: str, turn: int, tick: int
	) -> None:
		"""Record the point at which a new branch forks off from its parent"""
		_, start_turn, start_tick, end_turn, end_tick = self._branches_d[
			parent
		]
		if not (
			(start_turn, start_tick) <= (turn, tick) <= (end_turn, end_tick)
		):
			raise OutOfTimelineError(
				"The parent branch does not cover that time",
				parent,
				turn,
				tick,
			)
		self._branches_d[branch] = (parent, turn, tick, turn, tick)
		self._turn_end[branch, turn] = self._turn_end_plan[branch, turn] = tick
		self._loaded[branch] = (turn, tick, None, None)
		self._upd_branch_parentage(parent, branch)
		self.query.new_branch(branch, parent, turn, tick)

	def _extend_branch(self, branch: str, turn: int, tick: int) -> None:
		"""Record a change in the span of time that a branch includes"""
		parent, start_turn, start_tick, end_turn, end_tick = self._branches_d[
			branch
		]
		if (turn, tick) < (start_turn, start_tick):
			raise OutOfTimelineError(
				"Can't extend branch backwards", branch, turn, tick
			)
		if (turn, tick) < (end_turn, end_tick):
			return
		if (branch, turn) in self._turn_end_plan:
			if tick > self._turn_end_plan[branch, turn]:
				self._turn_end_plan[branch, turn] = tick
		else:
			self._turn_end_plan[branch, turn] = tick
		self._updload(branch, turn, tick)
		if not self._planning:
			self._branches_d[branch] = (
				parent,
				start_turn,
				start_tick,
				turn,
				tick,
			)
			if (branch, turn) in self._turn_end:
				if tick > self._turn_end[branch, turn]:
					self._turn_end[branch, turn] = tick
			else:
				self._turn_end[branch, turn] = tick

	def _get_keyframe(
		self,
		branch: str,
		turn: int,
		tick: int,
		copy=True,
		rulebooks=True,
		silent=False,
	):
		"""Load the keyframe if it's not loaded, and return it"""
		if (branch, turn, tick) in self._keyframes_loaded:
			if silent:
				return
			return self._get_kf(
				branch, turn, tick, copy=copy, rulebooks=rulebooks
			)
		univ, rule, rulebook = self.query.get_keyframe_extensions(
			branch, turn, tick
		)
		self._universal_cache.set_keyframe(branch, turn, tick, univ)
		self._triggers_cache.set_keyframe(
			branch, turn, tick, rule.get("triggers", {})
		)
		self._prereqs_cache.set_keyframe(
			branch, turn, tick, rule.get("prereqs", {})
		)
		self._actions_cache.set_keyframe(
			branch, turn, tick, rule.get("actions", {})
		)
		self._neighborhoods_cache.set_keyframe(
			branch, turn, tick, rule.get("neighborhood", {})
		)
		self._rule_bigness_cache.set_keyframe(
			branch, turn, tick, rule.get("big", {})
		)
		self._rulebooks_cache.set_keyframe(branch, turn, tick, rulebook)
		with (
			self.batch()
		):  # so that iter_keys doesn't try fetching the kf we're about to make
			keyframe_graphs = list(
				self.query.get_all_keyframe_graphs(branch, turn, tick)
			)
			self._graph_cache.set_keyframe(
				branch,
				turn,
				tick,
				{graph: "DiGraph" for (graph, _, _, _) in keyframe_graphs},
			)
			for (
				graph,
				nodes,
				edges,
				graph_val,
			) in keyframe_graphs:
				self._snap_keyframe_de_novo_graph(
					graph, branch, turn, tick, nodes, edges, graph_val
				)
		if not keyframe_graphs:
			for cache in (
				self._characters_rulebooks_cache,
				self._units_rulebooks_cache,
				self._characters_things_rulebooks_cache,
				self._characters_places_rulebooks_cache,
				self._characters_portals_rulebooks_cache,
			):
				cache.set_keyframe(branch, turn, tick, {})
		self._updload(branch, turn, tick)
		if branch in self._keyframes_dict:
			if turn in self._keyframes_dict[branch]:
				self._keyframes_dict[branch][turn].add(tick)
			else:
				self._keyframes_dict[branch][turn] = {tick}
		else:
			self._keyframes_dict[branch] = {turn: {tick}}
		self._keyframes_times.add((branch, turn, tick))
		self._keyframes_loaded.add((branch, turn, tick))
		ret = self._get_kf(branch, turn, tick, copy=copy)
		charrbkf = {}
		unitrbkf = {}
		charthingrbkf = {}
		charplacerbkf = {}
		charportrbkf = {}
		for graph, graphval in ret["graph_val"].items():
			charrbkf[graph] = graphval.get(
				"character_rulebook", ("character", graph)
			)
			unitrbkf[graph] = graphval.get("unit_rulebook", ("unit", graph))
			charthingrbkf[graph] = graphval.get(
				"character_thing_rulebook", ("character_thing", graph)
			)
			charplacerbkf[graph] = graphval.get(
				"character_place_rulebook", ("character_place", graph)
			)
			charportrbkf[graph] = graphval.get(
				"character_portal_rulebook", ("character_portal", graph)
			)
			self._unitness_cache.set_keyframe(
				graph, branch, turn, tick, graphval.get("units", {})
			)
			if graph in ret["node_val"]:
				locs = {}
				conts = {}
				noderbkf = {}
				for node, val in ret["node_val"][graph].items():
					noderbkf[node] = val.get("rulebook", (graph, node))
					if "location" not in val:
						continue
					locs[node] = location = val["location"]
					if location in conts:
						conts[location].add(node)
					else:
						conts[location] = {node}
				self._things_cache.set_keyframe(
					graph, branch, turn, tick, locs
				)
				self._node_contents_cache.set_keyframe(
					graph,
					branch,
					turn,
					tick,
					{k: frozenset(v) for (k, v) in conts.items()},
				)
				self._nodes_rulebooks_cache.set_keyframe(
					graph, branch, turn, tick, noderbkf
				)
			else:
				self._things_cache.set_keyframe(graph, branch, turn, tick, {})
				self._node_contents_cache.set_keyframe(
					graph, branch, turn, tick, {}
				)
				self._nodes_rulebooks_cache.set_keyframe(
					graph, branch, turn, tick, {}
				)
			if graph in ret["edge_val"]:
				edgerbkf = {}
				for orig, dests in ret["edge_val"][graph].items():
					if not dests:
						continue
					origrbkf = edgerbkf[orig] = {}
					for dest, val in dests.items():
						origrbkf[dest] = val.get(
							"rulebook", (graph, orig, dest)
						)
				self._portals_rulebooks_cache.set_keyframe(
					graph, branch, turn, tick, edgerbkf
				)
			else:
				self._portals_rulebooks_cache.set_keyframe(
					graph, branch, turn, tick, {}
				)
		self._characters_rulebooks_cache.set_keyframe(
			branch, turn, tick, charrbkf
		)
		self._units_rulebooks_cache.set_keyframe(branch, turn, tick, unitrbkf)
		self._characters_things_rulebooks_cache.set_keyframe(
			branch, turn, tick, charthingrbkf
		)
		self._characters_places_rulebooks_cache.set_keyframe(
			branch, turn, tick, charplacerbkf
		)
		self._characters_portals_rulebooks_cache.set_keyframe(
			branch, turn, tick, charportrbkf
		)
		if silent:
			return  # not that it helps performance any, in this case
		return ret

	def _iter_parent_btt(
		self,
		branch: str = None,
		turn: int = None,
		tick: int = None,
		*,
		stoptime: tuple[str, int, int] = None,
	) -> Iterator[tuple[str, int, int]]:
		"""Private use.

		Iterate over (branch, turn, tick), where the branch is
		a descendant of the previous (starting with whatever branch is
		presently active and ending at the main branch), and the turn is the
		latest revision in the branch that matters.

		:arg stoptime: a triple, ``(branch, turn, tick)``. Iteration will
		stop instead of yielding that time or any before it. The tick may be
		``None``, in which case, iteration will stop instead of yielding the
		turn.

		"""
		branch = branch or self.branch
		trn = self.turn if turn is None else turn
		tck = self.tick if tick is None else tick
		yield branch, trn, tck
		branches = self.branches()
		if stoptime:
			stopbranch, stopturn, stoptick = stoptime
			stopping = stopbranch == branch
			while branch in branches and not stopping:
				trn, tck = self._branch_start(branch)
				branch = self.branch_parent(branch)
				if branch is None:
					return
				if branch == stopbranch:
					stopping = True
					if trn < stopturn or (
						trn == stopturn
						and (stoptick is None or tck <= stoptick)
					):
						return
				yield branch, trn, tck
		else:
			while branch in branches:
				trn, tck = self._branch_start(branch)
				branch = self.branch_parent(branch)
				if branch is None:
					yield "trunk", 0, 0
					return
				yield branch, trn, tck

	def _iter_keyframes(
		self,
		branch: str,
		turn: int,
		tick: int,
		*,
		loaded=False,
		with_fork_points=False,
		stoptime: tuple[str, int, int] = None,
	):
		"""Iterate back over (branch, turn, tick) at which there is a keyframe

		Follows the timestream, like :method:`_iter_parent_btt`, but yields more times.
		We may have any number of keyframes in the same branch, and will yield
		them all.

		With ``loaded=True``, only yield keyframes that are in memory now.

		Use ``with_fork_points=True`` to also include all the times that the
		timeline branched.

		``stoptime`` is as in :method:`_iter_parent_btt`.

		"""
		kfd = self._keyframes_dict
		kfs = self._keyframes_times
		kfl = self._keyframes_loaded
		it = pairwise(
			self._iter_parent_btt(branch, turn, tick, stoptime=stoptime)
		)
		try:
			a, b = next(it)
		except StopIteration:
			assert branch in self.branches() and self._branch_start(
				branch
			) == (
				0,
				0,
			)
			a = (branch, turn, tick)
			b = (branch, 0, 0)
			if a == b:
				if (loaded and a in kfl) or (not loaded and a in kfs):
					yield a
				return
		for (b0, r0, t0), (b1, r1, t1) in chain([(a, b)], it):
			# we're going up the timestream, meaning that b1, r1, t1
			# is *before* b0, r0, t0
			if loaded:
				if (b0, r0, t0) in kfl:
					yield b0, r0, t0
			elif (b0, r0, t0) in kfs:
				yield b0, r0, t0
			if b0 not in kfd:
				continue
			assert b0 in self.branches()
			kfdb = kfd[b0]
			if r0 in kfdb:
				tcks = sorted(kfdb[r0])
				while tcks and tcks[-1] > t0:
					tcks.pop()
				if not tcks:
					if with_fork_points:
						yield b0, r0, t0
					continue
				if loaded:
					for tck in reversed(tcks):
						if r0 == r1 and tck <= t1:
							break
						if (b0, r0, tck) != (b0, r0, t0) and (
							b0,
							r0,
							tck,
						) in kfl:
							yield b0, r0, tck
				else:
					for tck in reversed(tcks):
						if tck < t0:
							break
						yield b0, r0, tck
			for r_between in range(r0 - 1, r1, -1):  # too much iteration?
				if r_between in kfdb:
					tcks = sorted(kfdb[r_between], reverse=True)
					if loaded:
						for tck in tcks:
							if (b0, r_between, tck) in kfl:
								yield b0, r_between, tck
					else:
						for tck in tcks:
							yield b0, r_between, tck
			if r1 in kfdb:
				tcks = sorted(kfdb[r1], reverse=True)
				if loaded:
					for tck in tcks:
						if tck <= t1:
							break
						if (b0, r1, tck) in kfl:
							yield b0, r1, tck
				else:
					for tck in tcks:
						if tck <= t1:
							break
						yield b0, r1, tck
		if b1 in kfd:
			kfdb = kfd[b1]
			tcks = sorted(kfdb[r1], reverse=True)
			while tcks[-1] > t1:
				tcks.pop()
			if not tcks:
				if with_fork_points:
					yield b1, r1, t1
				return
			if loaded:
				for tck in tcks:
					if (b1, r1, tck) in kfl:
						yield b1, r1, tck
			else:
				for tck in tcks:
					yield b1, r1, tck
			if with_fork_points and tcks[-1] == t1:
				return
		if with_fork_points:
			yield b1, r1, t1

	def _has_graph(self, graph, branch=None, turn=None, tick=None):
		if branch is None:
			branch = self.branch
		if turn is None:
			turn = self.turn
		if tick is None:
			tick = self.tick
		try:
			return (
				self._graph_cache.retrieve(graph, branch, turn, tick)
				!= "Deleted"
			)
		except KeyError:
			return False

	def _get_kf(
		self, branch: str, turn: int, tick: int, copy=True, rulebooks=True
	) -> dict[
		Key,
		GraphNodesDict
		| GraphNodeValDict
		| GraphEdgesDict
		| GraphEdgeValDict
		| GraphValDict,
	]:
		"""Get a keyframe that's already in memory"""
		assert (branch, turn, tick) in self._keyframes_loaded
		graph_val: GraphValDict = {}
		nodes: GraphNodesDict = {}
		node_val: GraphNodeValDict = {}
		edges: GraphEdgesDict = {}
		edge_val: GraphEdgeValDict = {}
		kf = {
			"graph_val": graph_val,
			"nodes": nodes,
			"node_val": node_val,
			"edges": edges,
			"edge_val": edge_val,
		}
		for graph in self._graph_cache.iter_keys(branch, turn, tick):
			try:
				if (
					self._graph_cache.retrieve(graph, branch, turn, tick)
					== "Deleted"
				):
					continue
			except KeyError:
				continue
			graph_val[graph] = {}
		for k in graph_val:
			try:
				graph_val[k] = self._graph_val_cache.get_keyframe(
					k, branch, turn, tick, copy
				)
			except KeyframeError:
				pass
			try:
				nodes[k] = self._nodes_cache.get_keyframe(
					k, branch, turn, tick, copy
				)
			except KeyframeError:
				pass
			try:
				node_val[k] = self._node_val_cache.get_keyframe(
					k, branch, turn, tick, copy
				)
			except KeyframeError:
				pass
			try:
				edges[k] = self._edges_cache.get_keyframe(
					k, branch, turn, tick, copy
				)
			except KeyframeError:
				pass
			try:
				edge_val[k] = self._edge_val_cache.get_keyframe(
					k, branch, turn, tick, copy
				)
			except KeyframeError:
				pass

		if rulebooks:
			for graph, vals in kf["graph_val"].items():
				try:
					vals["units"] = self._unitness_cache.get_keyframe(
						graph, branch, turn, tick
					)
				except KeyError:
					pass
				try:
					vals["character_rulebook"] = (
						self._characters_rulebooks_cache.retrieve(
							graph, branch, turn, tick
						)
					)
				except KeyError:
					pass
				try:
					vals["unit_rulebook"] = (
						self._units_rulebooks_cache.retrieve(
							graph, branch, turn, tick
						)
					)
				except KeyError:
					pass
				try:
					vals["character_thing_rulebook"] = (
						self._characters_things_rulebooks_cache.retrieve(
							graph, branch, turn, tick
						)
					)
				except KeyError:
					pass
				try:
					vals["character_place_rulebook"] = (
						self._characters_places_rulebooks_cache.retrieve(
							graph, branch, turn, tick
						)
					)
				except KeyError:
					pass
				try:
					vals["character_portal_rulebook"] = (
						self._characters_portals_rulebooks_cache.retrieve(
							graph, branch, turn, tick
						)
					)
				except KeyError:
					pass
				if graph in kf["nodes"] and kf["nodes"][graph]:
					try:
						node_rb_kf = self._nodes_rulebooks_cache.get_keyframe(
							graph, branch, turn, tick
						)
					except KeyframeError:
						node_rb_kf = {}
					for node in kf["nodes"][graph]:
						kf.setdefault("node_val", {}).setdefault(
							graph, {}
						).setdefault(node, {})["rulebook"] = node_rb_kf.get(
							node, (graph, node)
						)
				if graph in kf["edges"] and kf["edges"][graph]:
					try:
						port_rb_kf = (
							self._portals_rulebooks_cache.get_keyframe(
								graph, branch, turn, tick
							)
						)
					except KeyframeError:
						port_rb_kf = {}
					if graph not in kf["edge_val"]:
						kf["edge_val"][graph] = {}
					kf_graph_edge_val = kf["edge_val"][graph]
					for orig in kf["edges"][graph]:
						if orig not in kf_graph_edge_val:
							kf_graph_edge_val[orig] = {}
						kf_graph_orig_edge_val = kf_graph_edge_val[orig]
						if orig not in port_rb_kf:
							port_rb_kf[orig] = {}
						port_rb_kf_dests = port_rb_kf[orig]
						for dest in kf["edges"][graph][orig]:
							if dest not in kf_graph_orig_edge_val:
								kf_graph_orig_edge_val[dest] = {}
							kf_graph_dest_edge_val = kf_graph_orig_edge_val[
								dest
							]
							rulebook = port_rb_kf_dests.get(
								dest, (graph, orig, dest)
							)
							kf_graph_dest_edge_val["rulebook"] = rulebook
		for graph in kf["graph_val"]:
			try:
				locs_kf = self._things_cache.get_keyframe(
					graph, branch, turn, tick
				)
			except KeyframeError:
				locs_kf = {}
				for thing in list(
					self._things_cache.iter_keys(graph, branch, turn, tick)
				):
					locs_kf[thing] = self._things_cache.retrieve(
						graph, thing, branch, turn, tick
					)
			if "node_val" not in kf:
				kf["node_val"] = {
					graph: {
						thing: {"location": loc}
						for (thing, loc) in locs_kf.items()
					}
				}
			elif graph not in kf["node_val"]:
				kf["node_val"][graph] = {
					thing: {"location": loc}
					for (thing, loc) in locs_kf.items()
				}
			else:
				for thing, loc in locs_kf.items():
					if thing in kf["node_val"][graph]:
						kf["node_val"][graph][thing]["location"] = loc
					else:
						kf["node_val"][graph][thing] = {"location": loc}
		kf["universal"] = self._universal_cache.get_keyframe(
			branch, turn, tick
		)
		kf["triggers"] = self._triggers_cache.get_keyframe(branch, turn, tick)
		kf["prereqs"] = self._prereqs_cache.get_keyframe(branch, turn, tick)
		kf["actions"] = self._actions_cache.get_keyframe(branch, turn, tick)
		kf["neighborhood"] = self._neighborhoods_cache.get_keyframe(
			branch, turn, tick
		)
		kf["big"] = self._rule_bigness_cache.get_keyframe(branch, turn, tick)
		kf["rulebook"] = self._rulebooks_cache.get_keyframe(branch, turn, tick)
		return kf

	def _load_keyframe_times(self):
		keyframes_dict = self._keyframes_dict
		keyframes_times = self._keyframes_times
		q = self.query
		for branch, turn, tick in q.keyframes_dump():
			if branch not in keyframes_dict:
				keyframes_dict[branch] = {turn: {tick}}
			else:
				keyframes_dict_branch = keyframes_dict[branch]
				if turn not in keyframes_dict_branch:
					keyframes_dict_branch[turn] = {tick}
				else:
					keyframes_dict_branch[turn].add(tick)
			keyframes_times.add((branch, turn, tick))

	def _load_plans(self) -> None:
		q = self.query

		last_plan = -1
		plans = self._plans
		branches_plans = self._branches_plans
		for plan, branch, turn, tick in q.plans_dump():
			plans[plan] = branch, turn, tick
			branches_plans[branch].add(plan)
			if plan > last_plan:
				last_plan = plan
		self._last_plan = last_plan
		plan_ticks = self._plan_ticks
		time_plan = self._time_plan
		turn_end_plan = self._turn_end_plan
		for plan, turn, tick in q.plan_ticks_dump():
			plan_ticks[plan][turn].add(tick)
			branch = plans[plan][0]
			turn_end_plan[branch, turn] = max(
				(turn_end_plan[branch, turn], tick)
			)
			time_plan[branch, turn, tick] = plan

	def _load_rules_handled(self):
		q = self.query
		store_crh = self._character_rules_handled_cache.store
		for row in q.character_rules_handled_dump():
			store_crh(*row, loading=True)
		store_arh = self._unit_rules_handled_cache.store
		for row in q.unit_rules_handled_dump():
			store_arh(*row, loading=True)
		store_ctrh = self._character_thing_rules_handled_cache.store
		for row in q.character_thing_rules_handled_dump():
			store_ctrh(*row, loading=True)
		store_cprh = self._character_place_rules_handled_cache.store
		for row in q.character_place_rules_handled_dump():
			store_cprh(*row, loading=True)
		store_cporh = self._character_portal_rules_handled_cache.store
		for row in q.character_portal_rules_handled_dump():
			store_cporh(*row, loading=True)
		store_cnrh = self._node_rules_handled_cache.store
		for row in q.node_rules_handled_dump():
			store_cnrh(*row, loading=True)
		store_porh = self._portal_rules_handled_cache.store
		for row in q.portal_rules_handled_dump():
			store_porh(*row, loading=True)

	def _upd_branch_parentage(self, parent: str, child: str) -> None:
		self._childbranch[parent].add(child)
		self._branch_parents[child].add(parent)
		while (parent := self.branch_parent(parent)) is not None:
			self._branch_parents[child].add(parent)

	def _alias_kf(self, branch_from, branch_to, turn, tick):
		"""Copy a keyframe from one branch to another

		This aliases the data, rather than really copying. Keyframes don't
		change, so it should be fine.

		This does *not* save a new keyframe to disk.

		"""
		try:
			graph_keyframe = self._graph_cache.get_keyframe(
				branch_from, turn, tick, copy=False
			)
		except KeyframeError:
			graph_keyframe = {}
			for graph in self._graph_cache.iter_entities(
				branch_from, turn, tick
			):
				try:
					graph_keyframe[graph] = self._graph_cache.retrieve(
						graph, branch_from, turn, tick
					)
				except KeyError:
					pass
		self._graph_cache.set_keyframe(
			branch_to,
			turn,
			tick,
			graph_keyframe,
		)
		for graph in graph_keyframe:
			try:
				graph_vals = self._graph_val_cache.get_keyframe(
					graph, branch_from, turn, tick, copy=False
				)
			except KeyframeError:
				graph_vals = {}
			self._graph_val_cache.set_keyframe(
				graph, branch_to, turn, tick, graph_vals
			)
			try:
				nodes = self._nodes_cache.get_keyframe(
					graph, branch_from, turn, tick, copy=False
				)
			except KeyframeError:
				nodes = {}
			self._nodes_cache.set_keyframe(graph, branch_to, turn, tick, nodes)
			try:
				node_val = self._node_val_cache.get_keyframe(
					graph, branch_from, turn, tick, copy=False
				)
			except KeyframeError:
				node_val = {}
			self._node_val_cache.set_keyframe(
				graph, branch_to, turn, tick, node_val
			)
			try:
				edges = self._edges_cache.get_keyframe(
					graph, branch_from, turn, tick, copy=False
				)
			except KeyframeError:
				edges = {}
			self._edges_cache.set_keyframe(graph, branch_to, turn, tick, edges)
			try:
				edge_val = self._edge_val_cache.get_keyframe(
					graph, branch_from, turn, tick, copy=False
				)
			except KeyframeError:
				edge_val = {}
			self._edge_val_cache.set_keyframe(
				graph, branch_to, turn, tick, edge_val
			)
		for cache in (
			self._universal_cache,
			self._triggers_cache,
			self._prereqs_cache,
			self._actions_cache,
			self._rulebooks_cache,
			self._unitness_cache,
			self._characters_rulebooks_cache,
			self._units_rulebooks_cache,
			self._characters_things_rulebooks_cache,
			self._characters_places_rulebooks_cache,
			self._characters_portals_rulebooks_cache,
			self._neighborhoods_cache,
			self._rule_bigness_cache,
		):
			cache.alias_keyframe(branch_from, branch_to, turn, tick)

		for character in self._graph_cache.iter_entities(
			branch_from, turn, tick
		):
			loc_kf = self._things_cache.get_keyframe(
				character, branch_from, turn, tick, copy=False
			)
			conts_kf = self._node_contents_cache.get_keyframe(
				character, branch_from, turn, tick, copy=False
			)
			units_kf = self._unitness_cache.get_keyframe(
				character, branch_from, turn, tick, copy=False
			)
			self._things_cache.set_keyframe(
				character, branch_to, turn, tick, loc_kf
			)
			self._node_contents_cache.set_keyframe(
				character, branch_to, turn, tick, conts_kf
			)
			self._unitness_cache.set_keyframe(
				character, branch_to, turn, tick, units_kf
			)
			self._nodes_rulebooks_cache.set_keyframe(
				character,
				branch_to,
				turn,
				tick,
				self._nodes_rulebooks_cache.get_keyframe(
					character, branch_from, turn, tick, copy=False
				),
			)
			self._portals_rulebooks_cache.set_keyframe(
				character,
				branch_to,
				turn,
				tick,
				self._portals_rulebooks_cache.get_keyframe(
					character, branch_from, turn, tick, copy=False
				),
			)
		self._keyframes_times.add((branch_to, turn, tick))
		self._keyframes_loaded.add((branch_to, turn, tick))
		if branch_to in self._keyframes_dict:
			kdb = self._keyframes_dict[branch_to]
			if turn in kdb:
				kdb[turn].add(tick)
			else:
				kdb[turn] = {tick}
		else:
			self._keyframes_dict[branch_to] = {turn: {tick}}

	@staticmethod
	def _apply_unit_delta(keyframe: dict, delta: dict) -> None:
		for graf, units in delta.items():
			if graf in keyframe:
				for unit, ex in units.items():
					if ex:
						keyframe[graf][unit] = True
					elif unit in keyframe[graf]:
						del keyframe[graf][unit]
			else:
				keyframe[graf] = {
					unit: True for (unit, ex) in units.items() if ex
				}

	@staticmethod
	def _apply_graph_val_delta(
		graph: Key,
		graph_val_keyframe: dict,
		character_rulebook_keyframe: dict,
		unit_rulebook_keyframe: dict,
		character_thing_rulebook_keyframe: dict,
		character_place_rulebook_keyframe: dict,
		character_portal_rulebook_keyframe: dict,
		graph_val_delta: dict,
	):
		for key, kf in [
			("character_rulebook", character_rulebook_keyframe),
			("unit_rulebook", unit_rulebook_keyframe),
			("character_thing_rulebook", character_thing_rulebook_keyframe),
			("character_place_rulebook", character_place_rulebook_keyframe),
			("character_portal_rulebook", character_portal_rulebook_keyframe),
		]:
			if key in graph_val_delta:
				kf[graph] = graph_val_delta.pop(key)
			elif graph not in kf:
				kf[graph] = (key, graph)
		for k, v in graph_val_delta.items():
			if v is None:
				if k in graph_val_keyframe:
					del graph_val_keyframe[k]
			else:
				graph_val_keyframe[k] = v

	@staticmethod
	def _apply_node_delta(
		charname: Key,
		node_val_keyframe: dict,
		nodes_keyframe: dict,
		node_rulebook_keyframe: dict,
		thing_location_keyframe: dict,
		place_contents_keyframe: dict,
		node_val_delta: dict,
		nodes_delta: dict,
	) -> None:
		for node, ex in nodes_delta.items():
			if ex:
				nodes_keyframe[node] = True
				if node not in node_val_keyframe:
					node_val_keyframe[node] = {}
					node_rulebook_keyframe[node] = (charname, node)
			else:
				if node in nodes_keyframe:
					del nodes_keyframe[node]
				if node in node_val_keyframe:
					del node_val_keyframe[node]
				if node in thing_location_keyframe:
					del thing_location_keyframe[node]
				if node in place_contents_keyframe:
					del place_contents_keyframe[node]
		for node, upd in node_val_delta.items():
			if node in nodes_delta and not nodes_delta[node]:
				continue
			upd = upd.copy()
			if "location" in upd and upd["location"] is not None:
				loc = upd.pop("location")
				thing_location_keyframe[node] = loc
				if loc in place_contents_keyframe:
					place_contents_keyframe[loc].add(node)
				else:
					place_contents_keyframe[loc] = {node}
			if "rulebook" in upd:
				node_rulebook_keyframe[node] = upd.pop("rulebook")
			elif (
				node in node_val_keyframe
				and "rulebook" in node_val_keyframe[node]
			):
				node_rulebook_keyframe[node] = node_val_keyframe[node].pop(
					"rulebook"
				)
			else:
				assert node in node_rulebook_keyframe, (
					f"No rulebook for {node}"
				)
			if upd and node in node_val_keyframe:
				kv = node_val_keyframe[node]
				for key, value in upd.items():
					if value is None:
						if key in kv:
							del kv[key]
					else:
						kv[key] = value

	@staticmethod
	def _apply_edge_delta(
		charname: Key,
		edge_val_keyframe: dict,
		edges_keyframe: dict,
		portal_rulebook_keyframe: dict,
		edge_val_delta: dict,
		edges_delta: dict,
	) -> None:
		for orig, dests in edges_delta.items():
			for dest, ex in dests.items():
				if ex:
					edge_val_keyframe.setdefault(orig, {}).setdefault(dest, {})
					edges_keyframe.setdefault(orig, {})[dest] = True
					portal_rulebook_keyframe.setdefault(orig, {})[dest] = (
						charname,
						orig,
						dest,
					)
				elif orig in edges_keyframe and dest in edges_keyframe[orig]:
					del edges_keyframe[orig][dest]
					if not edges_keyframe[orig]:
						del edges_keyframe[orig]
					if orig in edge_val_keyframe:
						if dest in edge_val_keyframe[orig]:
							del edge_val_keyframe[orig][dest]
						if not edge_val_keyframe[orig]:
							del edge_val_keyframe[orig]
		for orig, dests in edge_val_delta.items():
			for dest, upd in dests.items():
				if (
					orig in edges_delta
					and dest in edges_delta[orig]
					and not edges_delta[orig][dest]
				):
					continue
				upd = upd.copy()
				if "rulebook" in upd:
					portal_rulebook_keyframe.setdefault(orig, {})[dest] = (
						upd.pop("rulebook")
					)
				elif (
					orig in edge_val_keyframe
					and dest in edge_val_keyframe[orig]
					and "rulebook" in edge_val_keyframe[orig][dest]
				):
					portal_rulebook_keyframe.setdefault(orig, {})[dest] = (
						edge_val_keyframe[orig][dest].pop("rulebook")
					)
				else:
					assert (
						orig in portal_rulebook_keyframe
						and dest in portal_rulebook_keyframe[orig]
					), f"No rulebook for {orig}->{dest}"
				if upd:
					kv = edge_val_keyframe.setdefault(orig, {}).setdefault(
						dest, {}
					)
					for key, value in upd.items():
						if value is None:
							del kv[key]
						else:
							kv[key] = value
		for orig, dests in list(edges_keyframe.items()):
			if not dests:
				del edges_keyframe[orig]
				if orig in edge_val_keyframe:
					del edge_val_keyframe[orig]

	def _snap_keyframe_from_delta(
		self,
		then: tuple[str, int, int],
		now: tuple[str, int, int],
		delta: DeltaDict,
	) -> None:
		assert then[0] == now[0]
		if then == now:
			return
		keyframe = self._get_keyframe(*then, rulebooks=False)
		graph_val_keyframe: dict[Key, GraphValDict] = keyframe["graph_val"]
		nodes_keyframe: dict[Key, GraphNodesDict] = keyframe["nodes"]
		node_val_keyframe: dict[Key, GraphNodeValDict] = keyframe["node_val"]
		edges_keyframe: dict[Key, GraphEdgesDict] = keyframe["edges"]
		edge_val_keyframe: dict[Key, GraphEdgeValDict] = keyframe["edge_val"]
		universal_keyframe = keyframe["universal"]
		rulebooks_keyframe = keyframe["rulebook"]
		triggers_keyframe = keyframe["triggers"]
		prereqs_keyframe = keyframe["prereqs"]
		actions_keyframe = keyframe["actions"]
		neighborhoods_keyframe = keyframe["neighborhood"]
		bigs = keyframe["big"]
		characters_rulebooks_keyframe = (
			self._characters_rulebooks_cache.get_keyframe(*then)
		)
		units_rulebooks_keyframe = self._units_rulebooks_cache.get_keyframe(
			*then
		)
		characters_things_rulebooks_keyframe = (
			self._characters_things_rulebooks_cache.get_keyframe(*then)
		)
		characters_places_rulebooks_keyframe = (
			self._characters_places_rulebooks_cache.get_keyframe(*then)
		)
		characters_portals_rulebooks_keyframe = (
			self._characters_portals_rulebooks_cache.get_keyframe(*then)
		)
		for k, v in delta.get("universal", {}).items():
			if v is None:
				if k in universal_keyframe:
					del universal_keyframe[k]
			else:
				universal_keyframe[k] = v
		if "rulebooks" in delta:
			rulebooks_keyframe.update(delta["rulebooks"])
		for rule, funcs in delta.get("rules", {}).items():
			triggers_keyframe[rule] = funcs.get(
				"triggers", triggers_keyframe.get(rule, [])
			)
			prereqs_keyframe[rule] = funcs.get(
				"prereqs", prereqs_keyframe.get(rule, [])
			)
			actions_keyframe[rule] = funcs.get(
				"actions", actions_keyframe.get(rule, [])
			)
			if "neighborhood" in funcs:
				neighborhoods_keyframe[rule] = funcs["neighborhood"]
			if "big" in funcs:
				bigs[rule] = funcs["big"]
		things_keyframe = {}
		nodes_rulebooks_keyframe = {}
		portals_rulebooks_keyframe = {}
		units_keyframe = {}
		for graph in (
			set(self._graph_cache.iter_keys(*then)).union(delta.keys())
			- self.illegal_graph_names
		):
			delt = delta.get(graph, {})
			if delt is None:
				continue
			try:
				noderbs = nodes_rulebooks_keyframe[graph] = (
					self._nodes_rulebooks_cache.get_keyframe(graph, *then)
				)
			except KeyframeError:
				noderbs = nodes_rulebooks_keyframe[graph] = {}
			try:
				portrbs = portals_rulebooks_keyframe[graph] = (
					self._portals_rulebooks_cache.get_keyframe(graph, *then)
				)
			except KeyframeError:
				portrbs = portals_rulebooks_keyframe[graph] = {}
			try:
				charunit = units_keyframe[graph] = (
					self._unitness_cache.get_keyframe(graph, *then, copy=True)
				)
			except KeyframeError:
				charunit = units_keyframe[graph] = {}
			try:
				locs = things_keyframe[graph] = (
					self._things_cache.get_keyframe(graph, *then, copy=True)
				)
			except KeyframeError:
				locs = things_keyframe[graph] = {}
			try:
				conts = {
					key: set(value)
					for (key, value) in self._node_contents_cache.get_keyframe(
						graph, *then, copy=True
					).items()
				}
			except KeyframeError:
				conts = {}
			if graph not in node_val_keyframe:
				node_val_keyframe[graph] = {}
			if graph not in nodes_keyframe:
				nodes_keyframe[graph] = {}
			self._apply_unit_delta(charunit, delt.pop("units", {}) or {})
			self._apply_node_delta(
				graph,
				node_val_keyframe.setdefault(graph, {}),
				nodes_keyframe.setdefault(graph, {}),
				noderbs,
				locs,
				conts,
				delt.pop("node_val", {}) or {},
				delt.pop("nodes", {}) or {},
			)
			self._apply_edge_delta(
				graph,
				edge_val_keyframe.setdefault(graph, {}),
				edges_keyframe.setdefault(graph, {}),
				portrbs,
				delt.pop("edge_val", {}) or {},
				delt.pop("edges", {}) or {},
			)
			self._apply_graph_val_delta(
				graph,
				graph_val_keyframe.setdefault(graph, {}),
				characters_rulebooks_keyframe,
				units_rulebooks_keyframe,
				characters_things_rulebooks_keyframe,
				characters_places_rulebooks_keyframe,
				characters_portals_rulebooks_keyframe,
				delt,
			)
			if graph not in edge_val_keyframe:
				edge_val_keyframe[graph] = {}
			if graph not in edges_keyframe:
				edges_keyframe[graph] = {}
			self._unitness_cache.set_keyframe(graph, *now, charunit)
			self._things_cache.set_keyframe(graph, *now, locs)
			self._node_contents_cache.set_keyframe(
				graph,
				*now,
				{key: frozenset(value) for (key, value) in conts.items()},
			)
			self._nodes_rulebooks_cache.set_keyframe(graph, *now, noderbs)
			self._portals_rulebooks_cache.set_keyframe(graph, *now, portrbs)
			self._nodes_cache.set_keyframe(graph, *now, nodes_keyframe[graph])
			self._node_val_cache.set_keyframe(
				graph, *now, node_val_keyframe[graph]
			)
			self._edges_cache.set_keyframe(graph, *now, edges_keyframe[graph])
			self._edge_val_cache.set_keyframe(
				graph, *now, edge_val_keyframe[graph]
			)
			self._graph_val_cache.set_keyframe(
				graph, *now, graph_val_keyframe[graph]
			)
		self._characters_rulebooks_cache.set_keyframe(
			*now, characters_rulebooks_keyframe
		)
		self._units_rulebooks_cache.set_keyframe(
			*now, units_rulebooks_keyframe
		)
		self._characters_things_rulebooks_cache.set_keyframe(
			*now, characters_things_rulebooks_keyframe
		)
		self._characters_places_rulebooks_cache.set_keyframe(
			*now, characters_places_rulebooks_keyframe
		)
		self._characters_portals_rulebooks_cache.set_keyframe(
			*now, characters_portals_rulebooks_keyframe
		)
		self._universal_cache.set_keyframe(*now, universal_keyframe)
		self._triggers_cache.set_keyframe(*now, triggers_keyframe)
		self._prereqs_cache.set_keyframe(*now, prereqs_keyframe)
		self._actions_cache.set_keyframe(*now, actions_keyframe)
		self._neighborhoods_cache.set_keyframe(*now, neighborhoods_keyframe)
		self._rule_bigness_cache.set_keyframe(*now, bigs)
		self._rulebooks_cache.set_keyframe(*now, rulebooks_keyframe)
		self.query.keyframe_extension_insert(
			*now,
			universal_keyframe,
			{
				"triggers": triggers_keyframe,
				"prereqs": prereqs_keyframe,
				"actions": actions_keyframe,
				"neighborhood": neighborhoods_keyframe,
				"big": bigs,
			},
			rulebooks_keyframe,
		)
		kfd = self._keyframes_dict
		kfs = self._keyframes_times
		kfsl = self._keyframes_loaded
		kfs.add(now)
		kfsl.add(now)
		self.query.keyframe_insert(*now)
		branch, turn, tick = now
		if branch not in kfd:
			kfd[branch] = {
				turn: {
					tick,
				}
			}
		elif turn not in kfd[branch]:
			kfd[branch][turn] = {
				tick,
			}
		else:
			kfd[branch][turn].add(tick)
		inskf = self.query.keyframe_graph_insert
		graphs_keyframe = {g: "DiGraph" for g in graph_val_keyframe}
		for graph in graphs_keyframe.keys() - self.illegal_graph_names:
			deltg = delta.get(graph, {})
			if deltg is None:
				del graphs_keyframe[graph]
				continue
			combined_node_val_keyframe = {
				node: val.copy()
				for (node, val) in node_val_keyframe.get(graph, {}).items()
			}
			for node, loc in things_keyframe.get(graph, {}).items():
				if loc is None:
					continue
				if node in combined_node_val_keyframe:
					combined_node_val_keyframe[node]["location"] = loc
				else:
					combined_node_val_keyframe[node] = {"location": loc}
			for node, rb in nodes_rulebooks_keyframe.get(graph, {}).items():
				if node in combined_node_val_keyframe:
					combined_node_val_keyframe[node]["rulebook"] = rb
				elif node in nodes_keyframe[graph]:
					combined_node_val_keyframe[node] = {"rulebook": rb}
			for node, ex in nodes_keyframe.get(graph, {}).items():
				if ex and node not in combined_node_val_keyframe:
					combined_node_val_keyframe[node] = {
						"rulebook": (graph, node)
					}
			combined_edge_val_keyframe = {
				orig: {dest: val.copy() for (dest, val) in dests.items()}
				for (orig, dests) in edge_val_keyframe.get(graph, {}).items()
			}
			for orig, dests in portals_rulebooks_keyframe.get(
				graph, {}
			).items():
				for dest, rb in dests.items():
					if (
						orig not in edges_keyframe[graph]
						or dest not in edges_keyframe[graph][orig]
					):
						continue
					combined_edge_val_keyframe.setdefault(orig, {}).setdefault(
						dest, {}
					)["rulebook"] = rb
			for orig, dests in edges_keyframe.get(graph, {}).items():
				for dest, ex in dests.items():
					if ex and (
						orig not in combined_edge_val_keyframe
						or dest not in combined_edge_val_keyframe[orig]
					):
						combined_edge_val_keyframe.setdefault(
							orig, {}
						).setdefault(dest, {})
			combined_graph_val_keyframe = graph_val_keyframe.get(
				graph, {}
			).copy()
			combined_graph_val_keyframe["character_rulebook"] = (
				characters_rulebooks_keyframe[graph]
			)
			combined_graph_val_keyframe["unit_rulebook"] = (
				units_rulebooks_keyframe[graph]
			)
			combined_graph_val_keyframe["character_thing_rulebook"] = (
				characters_things_rulebooks_keyframe[graph]
			)
			combined_graph_val_keyframe["character_place_rulebook"] = (
				characters_places_rulebooks_keyframe[graph]
			)
			combined_graph_val_keyframe["character_portal_rulebook"] = (
				characters_portals_rulebooks_keyframe[graph]
			)
			combined_graph_val_keyframe["units"] = units_keyframe[graph]
			inskf(
				graph,
				*now,
				combined_node_val_keyframe,
				combined_edge_val_keyframe,
				combined_graph_val_keyframe,
			)
		self._graph_cache.set_keyframe(*now, graphs_keyframe)

	def _recurse_delta_keyframes(self, branch, turn, tick):
		"""Make keyframes until we have one in the current branch"""
		time_from = branch, turn, tick
		kfd = self._keyframes_dict
		if time_from[0] in kfd:
			# could probably avoid these sorts by restructuring kfd
			for turn in sorted(kfd[time_from[0]].keys(), reverse=True):
				if turn < time_from[1]:
					return time_from[0], turn, max(kfd[time_from[0]][turn])
				elif turn == time_from[1]:
					for tick in sorted(kfd[time_from[0]][turn], reverse=True):
						if time_from[2] <= tick:
							return time_from[0], turn, tick
		parent, branched_turn_from, branched_tick_from, turn_to, tick_to = (
			self._branches_d[time_from[0]]
		)
		if parent is None:
			if (
				branch,
				branched_turn_from,
				branched_tick_from,
			) in self._keyframes_times:
				self._get_keyframe(
					branch, branched_turn_from, branched_tick_from, silent=True
				)
				return branch, branched_turn_from, branched_tick_from
			elif branch in self._keyframes_dict:
				for r in sorted(self._keyframes_dict[branch], reverse=True):
					if r <= turn:
						t = max(self._keyframes_dict[branch][r])
						self._get_keyframe(branch, r, t, silent=True)
						return branch, r, t
			self._snap_keyframe_de_novo(*time_from)
			return time_from
		else:
			(parent, turn_from, tick_from) = self._recurse_delta_keyframes(
				parent, branched_turn_from, branched_tick_from
			)
			if (
				parent,
				branched_turn_from,
				branched_tick_from,
			) not in self._keyframes_times:
				self._get_keyframe(parent, turn_from, tick_from)
				self._snap_keyframe_from_delta(
					(parent, turn_from, tick_from),
					(parent, branched_turn_from, branched_tick_from),
					self._get_branch_delta(
						parent,
						turn_from,
						tick_from,
						branched_turn_from,
						branched_tick_from,
					),
				)
			if (
				time_from[0],
				branched_turn_from,
				branched_tick_from,
			) not in self._keyframes_times:
				self._get_keyframe(
					parent, branched_turn_from, branched_tick_from, silent=True
				)
				assert (
					parent,
					branched_turn_from,
					branched_tick_from,
				) in self._keyframes_loaded
				self._alias_kf(
					parent,
					time_from[0],
					branched_turn_from,
					branched_tick_from,
				)
		return time_from[0], branched_turn_from, branched_tick_from

	def _node_exists(self, character: CharName, node: NodeName) -> bool:
		retrieve, btt = self._node_exists_stuff
		args = (character, node) + btt()
		retrieved = retrieve(args)
		return retrieved is not None and not isinstance(retrieved, Exception)

	@world_locked
	def _exist_node(self, character: Key, node: Key, exist=True) -> None:
		nbtt, exist_node, store = self._exist_node_stuff
		branch, turn, tick = nbtt()
		exist_node(character, node, branch, turn, tick, exist)
		store(character, node, branch, turn, tick, exist)
		if exist:
			self._nodes_rulebooks_cache.store(
				character, node, branch, turn, tick, (character, node)
			)
			self.query.set_node_rulebook(
				character, node, branch, turn, tick, (character, node)
			)

	def _edge_exists(
		self, character: Key, orig: Key, dest: Key, idx=0
	) -> bool:
		retrieve, btt = self._edge_exists_stuff
		args = (character, orig, dest, idx) + btt()
		retrieved = retrieve(args)
		return retrieved is not None and not isinstance(retrieved, Exception)

	@world_locked
	def _exist_edge(
		self, character: Key, orig: Key, dest: Key, idx=0, exist=True
	) -> None:
		nbtt, exist_edge, store = self._exist_edge_stuff
		branch, turn, tick = nbtt()
		exist_edge(
			character, orig, dest, idx, branch, turn, tick, exist or False
		)
		store(character, orig, dest, idx, branch, turn, tick, exist)
		if (character, orig, dest) in self._edge_objs:
			del self._edge_objs[character, orig, dest]
		if exist:
			self._portals_rulebooks_cache.store(
				character,
				orig,
				dest,
				branch,
				turn,
				tick,
				(character, orig, dest),
			)
			self.query.set_portal_rulebook(
				character,
				orig,
				dest,
				branch,
				turn,
				tick,
				(character, orig, dest),
			)

	def _call_in_subprocess(
		self,
		uid,
		method,
		future: Future,
		*args,
		update=True,
		**kwargs,
	):
		i = uid % len(self._worker_inputs)
		uidbytes = uid.to_bytes(8, "little")
		argbytes = zlib.compress(self.pack((method, args, kwargs)))
		with self._worker_locks[i]:
			if update:
				self._update_worker_process_state(i, lock=False)
			self._worker_inputs[i].send_bytes(uidbytes + argbytes)
			output = self._worker_outputs[i].recv_bytes()
		got_uid = int.from_bytes(output[:8], "little")
		result = self.unpack(zlib.decompress(output[8:]))
		assert got_uid == uid
		self._how_many_futs_running -= 1
		del self._uid_to_fut[uid]
		if isinstance(result, Exception):
			future.set_exception(result)
		else:
			future.set_result(result)

	def _build_loading_windows(
		self,
		branch_from: str,
		turn_from: int,
		tick_from: int,
		branch_to: str,
		turn_to: int | None,
		tick_to: int | None,
	) -> list[tuple[str, int, int, int, int]]:
		"""Return windows of time I've got to load

		In order to have a complete timeline between these points.

		Returned windows are in reverse chronological order.

		"""
		if branch_from == branch_to:
			return [(branch_from, turn_from, tick_from, turn_to, tick_to)]
		windows = []
		if turn_to is None:
			branch1 = self.branch_parent(branch_to)
			turn1, tick1 = self._branch_start(branch_to)
			windows.append(
				(
					branch_to,
					turn1,
					tick1,
					None,
					None,
				)
			)
			parentage_iter = self._iter_parent_btt(branch1, turn1, tick1)
		else:
			parentage_iter = self._iter_parent_btt(branch_to, turn_to, tick_to)
			branch1, turn1, tick1 = next(parentage_iter)
		for branch0, turn0, tick0 in parentage_iter:
			windows.append((branch1, turn0, tick0, turn1, tick1))
			(branch1, turn1, tick1) = (branch0, turn0, tick0)
			if branch0 == branch_from:
				windows.append((branch0, turn_from, tick_from, turn0, tick0))
				break
		else:
			raise HistoricKeyError("Couldn't build sensible loading windows")
		return windows

	def _keyframe_after(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> Optional[Time]:
		if branch not in self._keyframes_dict:
			return None
		kfdb = self._keyframes_dict[branch]
		if turn in kfdb:
			ticks: set[Tick] = set(filter(partial(lt, tick), kfdb[turn]))
			if ticks:
				return branch, turn, min(ticks)
		turns: set[Turn] = set(filter(partial(lt, turn), kfdb.keys()))
		if turns:
			r = min(turns)
			return branch, r, min(kfdb[r])
		return None

	def _updload(self, branch, turn, tick):
		loaded = self._loaded
		if branch not in loaded:
			latekf = self._keyframe_after(branch, turn, tick)
			if latekf is None or latekf == (branch, turn, tick):
				loaded[branch] = (turn, tick, None, None)
			else:
				_, r, t = latekf
				loaded[branch] = (turn, tick, r, t)
			return
		(early_turn, early_tick, late_turn, late_tick) = loaded[branch]
		if None in (late_turn, late_tick):
			assert late_turn is late_tick is None
			if (turn, tick) < (early_turn, early_tick):
				(early_turn, early_tick) = (turn, tick)
		else:
			if (turn, tick) < (early_turn, early_tick):
				(early_turn, early_tick) = (turn, tick)
			if (late_turn, late_tick) < (turn, tick):
				(late_turn, late_tick) = (turn, tick)
		loaded[branch] = (early_turn, early_tick, late_turn, late_tick)

	@world_locked
	def load_between(
		self,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	):
		if self._time_is_loaded_between(
			branch, turn_from, tick_from, turn_to, tick_to
		):
			return
		self._get_keyframe(branch, turn_from, tick_from, silent=True)
		noderows = []
		nodevalrows = []
		edgerows = []
		edgevalrows = []
		graphvalrows = []
		graphsrows = list(
			self.query.graphs_types(
				branch, turn_from, tick_from, turn_to, tick_to
			)
		)
		self._graph_cache.load(graphsrows)
		loaded_graphs = self.query.load_windows(
			[(branch, turn_from, tick_from, turn_to, tick_to)]
		)
		for graph, loaded in loaded_graphs.items():
			match graph:
				case "universals":
					self._universal_cache.load(loaded)
				case "rulebooks":
					self._rulebooks_cache.load(loaded)
				case "rule_triggers":
					self._triggers_cache.load(loaded)
				case "rule_prereqs":
					self._prereqs_cache.load(loaded)
				case "rule_actions":
					self._actions_cache.load(loaded)
				case "rule_neighborhoods":
					self._neighborhoods_cache.load(loaded)
				case "rule_big":
					self._rule_bigness_cache.load(loaded)
				case _:
					noderows.extend(loaded["nodes"])
					edgerows.extend(loaded["edges"])
					nodevalrows.extend(loaded["node_val"])
					edgevalrows.extend(loaded["edge_val"])
					graphvalrows.extend(loaded["graph_val"])
					if things := loaded.get("things"):
						self._things_cache.load(things)
					if crb := loaded.get("character_rulebook"):
						self._characters_rulebooks_cache.load(crb)
					if urb := loaded.get("unit_rulebook"):
						self._units_rulebooks_cache.load(urb)
					if cthrb := loaded.get("character_thing_rulebook"):
						self._characters_things_rulebooks_cache.load(cthrb)
					if cprb := loaded.get("character_place_rulebook"):
						self._characters_places_rulebooks_cache.load(cprb)
					if cporb := loaded.get("character_portal_rulebook"):
						self._characters_portals_rulebooks_cache.load(cporb)
					if nrb := loaded.get("node_rulebook"):
						self._nodes_rulebooks_cache.load(nrb)
					if porb := loaded.get("portal_rulebook"):
						self._portals_rulebooks_cache.load(porb)
					loaded_graphs[graph] = loaded
		self._nodes_cache.load(noderows)
		self._node_val_cache.load(nodevalrows)
		self._edges_cache.load(edgerows)
		self._edge_val_cache.load(edgevalrows)
		self._graph_val_cache.load(graphvalrows)
		return loaded_graphs

	@world_locked
	def unload(self) -> None:
		"""Remove everything from memory that can be removed."""
		# If we're not connected to some database, we can't unload anything
		# without losing data
		if isinstance(self.query, NullQueryEngine):
			return
		# find the slices of time that need to stay loaded
		branch, turn, tick = self._btt()
		iter_parent_btt = self._iter_parent_btt
		kfd = self._keyframes_dict
		if not kfd:
			return
		loaded = self._loaded
		to_keep = {}
		# Find a path to the latest past keyframe we can use. Keep things
		# loaded from there to here.
		for past_branch, past_turn, past_tick in iter_parent_btt(
			branch, turn, tick
		):
			if past_branch not in loaded:
				continue  # nothing happened in this branch i guess
			early_turn, early_tick, late_turn, late_tick = loaded[past_branch]
			if None in (late_turn, late_tick):
				assert late_turn is late_tick is None
				late_turn, late_tick = self._branch_end(past_branch)
			if past_branch in kfd:
				for kfturn, kfticks in kfd[past_branch].items():
					# this can't possibly perform very well.
					# Maybe I need another loadedness dict that gives the two
					# keyframes I am between and gets upkept upon time travel
					for kftick in kfticks:
						if (
							(early_turn, early_tick)
							<= (kfturn, kftick)
							<= (late_turn, late_tick)
						):
							if (
								kfturn < turn
								or (kfturn == turn and kftick < tick)
							) and (
								kfturn > early_turn
								or (
									kfturn == early_turn
									and kftick > early_tick
								)
							):
								early_turn, early_tick = kfturn, kftick
							elif (
								kfturn > turn
								or (kfturn == turn and kftick >= tick)
							) and (
								kfturn < late_turn
								or (kfturn == late_turn and kftick < late_tick)
							):
								late_turn, late_tick = kfturn, kftick
				to_keep[past_branch] = (
					early_turn,
					early_tick,
					*max(((past_turn, past_tick), (late_turn, late_tick))),
				)
				break
			else:
				to_keep[past_branch] = (
					early_turn,
					early_tick,
					late_turn,
					late_tick,
				)
		if not to_keep:
			# unloading literally everything would make the game unplayable,
			# so don't
			if hasattr(self, "warning"):
				self.warning("Not unloading, due to lack of keyframes")
			return
		caches = self._caches
		kf_to_keep = set()
		times_unloaded = set()
		for past_branch, (
			early_turn,
			early_tick,
			late_turn,
			late_tick,
		) in to_keep.items():
			# I could optimize this with windowdicts
			if early_turn == late_turn:
				if (
					past_branch in self._keyframes_dict
					and early_turn in self._keyframes_dict[past_branch]
				):
					for tick in self._keyframes_dict[past_branch][early_turn]:
						if early_tick <= tick <= late_tick:
							kf_to_keep.add((past_branch, early_turn, tick))
			else:
				if past_branch in self._keyframes_dict:
					for turn, ticks in self._keyframes_dict[
						past_branch
					].items():
						if turn < early_turn or late_turn < turn:
							continue
						elif early_turn == turn:
							for tick in ticks:
								if early_tick <= tick:
									kf_to_keep.add((past_branch, turn, tick))
						elif turn == late_turn:
							for tick in ticks:
								if tick <= late_tick:
									kf_to_keep.add((past_branch, turn, tick))
						else:
							kf_to_keep.update(
								(past_branch, turn, tick) for tick in ticks
							)
			kf_to_keep &= self._keyframes_loaded
			for cache in caches:
				cache.truncate(past_branch, early_turn, early_tick, "backward")
				cache.truncate(past_branch, late_turn, late_tick, "forward")
				if not hasattr(cache, "keyframe"):
					continue
				for graph, branches in cache.keyframe.items():
					turns = branches[past_branch]
					turns_truncated = turns.truncate(late_turn, "forward")
					if late_turn in turns:
						late = turns[late_turn]
						times_unloaded.update(
							(past_branch, late_turn, t)
							for t in late.truncate(late_tick, "forward")
						)
					turns_truncated.update(
						turns.truncate(early_turn, "backward")
					)
					times_unloaded.update(
						(past_branch, turn_deleted, tick_deleted)
						for turn_deleted in self._keyframes_dict[
							past_branch
						].keys()
						& turns_truncated
						for tick_deleted in self._keyframes_dict[past_branch][
							turn_deleted
						]
					)
					if early_turn in turns:
						early = turns[early_turn]
						times_unloaded.update(
							(past_branch, early_turn, t)
							for t in early.truncate(early_tick, "backward")
						)
					unloaded_wrongly = times_unloaded & kf_to_keep
					assert not unloaded_wrongly, unloaded_wrongly
		self._keyframes_loaded.clear()
		self._keyframes_loaded.update(kf_to_keep)
		loaded.update(to_keep)
		for branch in set(loaded).difference(to_keep):
			for cache in caches:
				cache.remove_branch(branch)
			del loaded[branch]

	def _time_is_loaded(
		self, branch: Branch, turn: Turn = None, tick: Tick = None
	) -> bool:
		loaded = self._loaded
		if branch not in loaded:
			return False
		if turn is None:
			if tick is not None:
				raise ValueError("Need both or neither of turn and tick")
			return True
		if tick is None:
			(past_turn, _, future_turn, _) = loaded[branch]
			if future_turn is None:
				return past_turn <= turn
			return past_turn <= turn <= future_turn
		else:
			early_turn, early_tick, late_turn, late_tick = loaded[branch]
			if late_turn is None:
				assert late_tick is None
				return (early_turn, early_tick) <= (turn, tick)
			return (
				(early_turn, early_tick)
				<= (turn, tick)
				<= (late_turn, late_tick)
			)

	def _time_is_loaded_between(
		self,
		branch: Branch,
		turn_from: Turn,
		tick_from: Tick,
		turn_to: Turn,
		tick_to: Tick,
	) -> bool:
		"""Return whether we have all time in this window in memory"""
		loaded = self._loaded
		if branch not in loaded:
			return False
		early_turn, early_tick, late_turn, late_tick = loaded[branch]
		if (turn_from, tick_from) < (early_turn, early_tick):
			return False
		if None in (late_turn, late_tick):
			return True
		elif (late_turn, late_tick) < (turn_to, tick_to):
			return False
		return True

	def _build_keyframe_window(
		self, branch: Branch, turn: Turn, tick: Tick, loading=False
	) -> tuple[Time | None, Time | None]:
		"""Return a pair of keyframes that contain the given moment

		They give the smallest contiguous span of time I can reasonably load.

		"""
		branch_now = branch
		turn_now = turn
		tick_now = tick
		latest_past_keyframe: Optional[Time] = None
		earliest_future_keyframe: Optional[Time] = None
		branch_parents = self._branch_parents
		cache = self._keyframes_times if loading else self._keyframes_loaded
		for branch, turn, tick in cache:
			# Figure out the latest keyframe that is earlier than the present
			# moment, and the earliest keyframe that is later than the
			# present moment, for each graph. Can I avoid iterating over the
			# entire keyframes table, somehow?
			if branch == branch_now:
				if turn < turn_now:
					if latest_past_keyframe:
						(late_branch, late_turn, late_tick) = (
							latest_past_keyframe
						)
						if (
							late_branch != branch
							or late_turn < turn
							or (late_turn == turn and late_tick < tick)
						):
							latest_past_keyframe = (branch, turn, tick)
					else:
						latest_past_keyframe = (branch, turn, tick)
				elif turn > turn_now:
					if earliest_future_keyframe:
						(early_branch, early_turn, early_tick) = (
							earliest_future_keyframe
						)
						if (
							early_branch != branch
							or early_turn > turn
							or (early_turn == turn and early_tick > tick)
						):
							earliest_future_keyframe = (branch, turn, tick)
					else:
						earliest_future_keyframe = (branch, turn, tick)
				elif tick < tick_now:
					if latest_past_keyframe:
						(late_branch, late_turn, late_tick) = (
							latest_past_keyframe
						)
						if (
							late_branch != branch
							or late_turn < turn
							or (late_turn == turn and late_tick < tick)
						):
							latest_past_keyframe = (branch, turn, tick)
					else:
						latest_past_keyframe = (branch, turn, tick)
				elif tick > tick_now:
					if earliest_future_keyframe:
						(early_branch, early_turn, early_tick) = (
							earliest_future_keyframe
						)
						if (
							early_branch != branch
							or early_turn > turn
							or (early_turn == turn and early_tick > tick)
						):
							earliest_future_keyframe = (branch, turn, tick)
					else:
						earliest_future_keyframe = (branch, turn, tick)
				else:
					latest_past_keyframe = (branch, turn, tick)
			elif branch in branch_parents[branch_now]:
				if latest_past_keyframe:
					(late_branch, late_turn, late_tick) = latest_past_keyframe
					if branch == late_branch:
						if turn > late_turn or (
							turn == late_turn and tick > late_tick
						):
							latest_past_keyframe = (branch, turn, tick)
					elif late_branch in branch_parents[branch]:
						latest_past_keyframe = (branch, turn, tick)
				else:
					latest_past_keyframe = (branch, turn, tick)
		(branch, turn, tick) = (branch_now, turn_now, tick_now)
		if not loading or branch not in self._loaded:
			return latest_past_keyframe, earliest_future_keyframe
		if (
			earliest_future_keyframe
			and earliest_future_keyframe[1:] < self._loaded[branch][:2]
		):
			earliest_future_keyframe = (branch, *self._loaded[branch][:2])
		if (
			latest_past_keyframe
			and None not in self._loaded[branch][2:]
			and self._loaded[branch][2:] < latest_past_keyframe[1:]
		):
			latest_past_keyframe = (branch, *self._loaded[branch][2:])
		return latest_past_keyframe, earliest_future_keyframe

	@world_locked
	def snap_keyframe(
		self, silent=False, update_worker_processes=True
	) -> dict | None:
		"""Make a copy of the complete state of the world.

		You need to do this occasionally in order to keep time travel
		performant.

		The keyframe will be saved to the database at the next call to
		``flush``.

		Return the keyframe by default. With ``silent=True``,
		return ``None``. This is a little faster, and uses a little less
		memory.

		"""
		branch, turn, tick = self._btt()
		if (branch, turn, tick) in self._keyframes_times:
			if silent:
				return None
			return self._get_keyframe(branch, turn, tick)
		if not (self._branch_start() <= (turn, tick) <= self._branch_end()):
			raise OutOfTimelineError("Don't snap keyframes in plans")
		kfd = self._keyframes_dict
		the_kf: tuple[str, int, int] = None
		if branch in kfd:
			# I could probably avoid sorting these by using windowdicts
			for trn in sorted(kfd[branch].keys(), reverse=True):
				if trn < turn:
					the_kf = (branch, trn, max(kfd[branch][trn]))
					break
				elif trn == turn:
					for tck in sorted(kfd[branch][trn], reverse=True):
						if tck <= tick:
							the_kf = (branch, trn, tck)
							break
				if the_kf is not None:
					break
		if the_kf is None:
			parent = self.branch_parent(branch)
			if parent is None:
				self._snap_keyframe_de_novo(branch, turn, tick)
				if silent:
					return None
				else:
					return self._get_kf(branch, turn, tick)
			the_kf = self._recurse_delta_keyframes(branch, turn, tick)
		if the_kf not in self._keyframes_loaded:
			self._get_keyframe(*the_kf, silent=True)
		if the_kf != (branch, turn, tick):
			if the_kf[0] != branch:
				self._alias_kf(the_kf[0], branch, turn, tick)
			self._snap_keyframe_from_delta(
				the_kf,
				(branch, turn, tick),
				self._get_branch_delta(*the_kf, turn, tick),
			)
		if silent:
			return None
		ret = self._get_kf(branch, turn, tick)
		if hasattr(self, "_worker_processes") and update_worker_processes:
			self._update_all_worker_process_states(clobber=True)
		return ret

	@world_locked
	def _read_at(
		self, branch: Branch, turn: Branch, tick: Branch
	) -> tuple[
		Time | None,
		Time | None,
		list,
		dict,
	]:
		latest_past_keyframe: Time | None
		earliest_future_keyframe: Time | None
		branch_now, turn_now, tick_now = branch, turn, tick
		(latest_past_keyframe, earliest_future_keyframe) = (
			self._build_keyframe_window(
				branch_now,
				turn_now,
				tick_now,
				loading=True,
			)
		)
		# If branch is a descendant of branch_now, don't load the keyframe
		# there, because then we'd potentially be loading keyframes from any
		# number of possible futures, and we're trying to be conservative
		# about what we load. If neither branch is an ancestor of the other,
		# we can't use the keyframe for this load

		if latest_past_keyframe is None:
			if earliest_future_keyframe is None:
				return (
					None,
					None,
					list(
						self.query.graphs_types(
							self.query.globl["main_branch"], 0, 0
						)
					),
					self.query.load_windows(
						[(self.query.globl["main_branch"], 0, 0, None, None)]
					),
				)
			else:
				windows = self._build_loading_windows(
					self.query.globl["main_branch"], 0, 0, branch, turn, tick
				)
		else:
			past_branch, past_turn, past_tick = latest_past_keyframe
			if earliest_future_keyframe is None:
				# Load data from the keyframe to now
				windows = self._build_loading_windows(
					past_branch,
					past_turn,
					past_tick,
					branch,
					None,
					None,
				)
			else:
				# Load data between the two keyframes
				(future_branch, future_turn, future_tick) = (
					earliest_future_keyframe
				)
				windows = self._build_loading_windows(
					past_branch,
					past_turn,
					past_tick,
					future_branch,
					future_turn,
					future_tick,
				)
		graphs_types = []
		for window in windows:
			graphs_types.extend(self.query.graphs_types(*window))
		return (
			latest_past_keyframe,
			earliest_future_keyframe,
			graphs_types,
			self.query.load_windows(windows),
		)

	@world_locked
	def load_at(
		self, branch: Branch, turn: Turn, tick: Tick | None = None
	) -> None:
		if tick is None:
			tick = self._turn_end[branch, turn]
		if self._time_is_loaded(branch, turn, tick):
			return
		with garbage():
			self._load(*self._read_at(branch, turn, tick))

	def _load(
		self,
		latest_past_keyframe: Time | None,
		earliest_future_keyframe: tuple[str, int, int] | None,
		graphs_rows: list,
		loaded: dict,
	):
		"""Load history data at the given time

		Will load the keyframe prior to that time, and all history
		data following, up to (but not including) the keyframe thereafter.

		"""
		if latest_past_keyframe:
			self._get_keyframe(*latest_past_keyframe, silent=True)

		if universals := loaded.pop("universals", None):
			self._universal_cache.load(universals)
		if rulebooks := loaded.pop("rulebooks", None):
			self._rulebooks_cache.load(rulebooks)
		if rule_triggers := loaded.pop("rule_triggers", None):
			self._triggers_cache.load(rule_triggers)
		if rule_prereqs := loaded.pop("rule_prereqs", None):
			self._prereqs_cache.load(rule_prereqs)
		if rule_actions := loaded.pop("rule_actions", None):
			self._actions_cache.load(rule_actions)
		if rule_neighborhoods := loaded.pop("rule_neighborhoods", None):
			self._neighborhoods_cache.load(rule_neighborhoods)
		if rule_big := loaded.pop("rule_big", None):
			self._rule_bigness_cache.load(rule_big)
		for loaded_graph, data in loaded.items():
			assert isinstance(data, dict)
			if data.get("things"):
				self._things_cache.load(data["things"])
			if data.get("character_rulebook"):
				self._characters_rulebooks_cache.load(
					data["character_rulebook"]
				)
			if data.get("unit_rulebook"):
				self._units_rulebooks_cache.load(data["unit_rulebook"])
			if data.get("character_thing_rulebook"):
				self._characters_things_rulebooks_cache.load(
					data["character_thing_rulebook"]
				)
			if data.get("character_place_rulebook"):
				self._characters_places_rulebooks_cache.load(
					data["character_place_rulebook"]
				)
			if data.get("character_portal_rulebook"):
				self._characters_portals_rulebooks_cache.load(
					data["character_portal_rulebook"]
				)
			if data.get("node_rulebook"):
				self._nodes_rulebooks_cache.load(data["node_rulebook"])
			if data.get("portal_rulebook"):
				self._portals_rulebooks_cache.load(data["portal_rulebook"])

		self._graph_cache.load(graphs_rows)
		noderows = []
		edgerows = []
		nodevalrows = []
		edgevalrows = []
		graphvalrows = []
		for graph, graph_loaded in loaded.items():
			noderows.extend(graph_loaded["nodes"])
			edgerows.extend(graph_loaded["edges"])
			nodevalrows.extend(graph_loaded["node_val"])
			edgevalrows.extend(graph_loaded["edge_val"])
			graphvalrows.extend(graph_loaded["graph_val"])

		self._graph_cache.load(graphs_rows)
		self._nodes_cache.load(noderows)
		self._edges_cache.load(edgerows)
		self._graph_val_cache.load(graphvalrows)
		self._node_val_cache.load(nodevalrows)
		self._edge_val_cache.load(edgevalrows)

	def turn_end(self, branch: Branch = None, turn: Turn = None) -> int:
		branch = branch or self._obranch
		turn = turn or self._oturn
		return self._turn_end[branch, turn]

	def turn_end_plan(self, branch: Branch = None, turn: Turn = None):
		branch = branch or self._obranch
		turn = turn or self._oturn
		return self._turn_end_plan[branch, turn]

	def submit(
		self, fn: FunctionType | MethodType, /, *args, **kwargs
	) -> Future:
		if fn.__module__ not in {
			"function",
			"method",
			"trigger",
			"prereq",
			"action",
		}:
			raise ValueError(
				"Function is not stored in this lisien engine. "
				"Use, eg., the engine's attribute `function` to store it."
			)
		uid = self._top_uid
		if hasattr(self, "_worker_processes"):
			ret = Future()
			ret._t = Thread(
				target=self._call_in_subprocess,
				args=(uid, fn, ret, *args),
				kwargs=kwargs,
			)
			self._uid_to_fut[uid] = ret
			self._futs_to_start.put(ret)
		else:
			ret = fake_submit(fn, *args, **kwargs)
		ret.uid = uid
		self._top_uid += 1
		return ret

	def _manage_futs(self):
		while True:
			while self._how_many_futs_running < len(self._worker_processes):
				try:
					fut = self._futs_to_start.get()
				except Empty:
					break
				if not fut.running() and fut.set_running_or_notify_cancel():
					fut._t.start()
					self._how_many_futs_running += 1
			sleep(0.001)

	def shutdown(self, wait=True, *, cancel_futures=False) -> None:
		if not hasattr(self, "_worker_processes"):
			return
		if cancel_futures:
			for fut in self._uid_to_fut.values():
				fut.cancel()
		if wait:
			futwait(self._uid_to_fut.values())
		self._uid_to_fut = {}
		for i, (lock, pipein, pipeout, proc) in enumerate(
			zip(
				self._worker_locks,
				self._worker_inputs,
				self._worker_outputs,
				self._worker_processes,
			)
		):
			with lock:
				pipein.send_bytes(b"shutdown")
				recvd = pipeout.recv_bytes()
				if recvd != b"done":
					raise RuntimeError("Worker process wasn't done", i)
				pipein.close()
				pipeout.close()
				proc.join(timeout=5)
				if proc.exitcode is None:
					raise RuntimeError("Worker process didn't exit", i)
				if proc.exitcode != 0:
					raise RuntimeError(
						"Worker process didn't exit normally", i, proc.exitcode
					)
				proc.close()
				pipein.close()
				pipeout.close()

	def _detect_kf_interval_override(self):
		if self._planning:
			self._kf_overridden = True
			return True
		if getattr(self, "_no_kc", False):
			self._kf_overridden = True
			return True
		if getattr(self, "_kf_overridden", False):
			self._kf_overridden = False
			return False

	def _reimport_some_functions(self, some):
		if getattr(self, "_prefix", None) is not None:
			self._call_every_subprocess(f"_reimport_{some}")
		else:
			callables = {}
			for att in dir(getattr(self, some)):
				v = getattr(getattr(self.some), att)
				if callable(v):
					callables[att] = v
			self._call_every_subprocess(
				f"_replace_{some}_pkl", pickle.dumps(callables)
			)

	def _reimport_trigger_functions(self, *args, attr, **kwargs):
		if attr is not None:
			return
		self._reimport_some_functions("triggers")

	def _reimport_worker_functions(self, *args, attr, **kwargs):
		if attr is not None:
			return
		self._reimport_some_functions("functions")

	def _reimport_worker_methods(self, *args, attr, **kwargs):
		if attr is not None:
			return
		self._reimport_some_functions("methods")

	def _get_worker_kf_payload(self, uid: int = sys.maxsize) -> bytes:
		# I'm not using the uid at the moment, because this doesn't return anything
		plainstored = {}
		pklstored = {}
		for name, store in [
			("function", self.function),
			("method", self.method),
			("trigger", self.trigger),
			("prereq", self.prereq),
			("action", self.action),
		]:
			if hasattr(store, "iterplain") and callable(store.iterplain):
				plainstored[name] = dict(store.iterplain())
			else:
				pklstored[name] = pickle.dumps(store)
		return uid.to_bytes(8, "little") + zlib.compress(
			self.pack(
				(
					"_upd_from_game_start",
					(
						None,
						*self._btt(),
						(
							self.snap_keyframe(update_worker_processes=False),
							dict(self.eternal.items()),
							plainstored,
							pklstored,
						),
					),
					{},
				)
			)
		)

	def _call_any_subprocess(self, method: str | callable, *args, **kwargs):
		uid = self._top_uid
		self._top_uid += 1
		return self._call_in_subprocess(uid, method, *args, **kwargs)

	@contextmanager
	def _all_worker_locks_ctx(self):
		for lock in self._worker_locks:
			lock.acquire()
		yield
		for lock in self._worker_locks:
			lock.release()

	@staticmethod
	def _all_worker_locks(fn):
		@wraps(fn)
		def call_with_all_worker_locks(self, *args, **kwargs):
			with self._all_worker_locks_ctx():
				return fn(self, *args, **kwargs)

		return call_with_all_worker_locks

	@_all_worker_locks
	def _call_every_subprocess(self, method: str, *args, **kwargs):
		ret = []
		uids = []
		for _ in range(len(self._worker_processes)):
			uids.append(self._top_uid)
			uidbytes = self._top_uid.to_bytes(8, "little")
			argbytes = zlib.compress(self.pack((method, args, kwargs)))
			i = self._top_uid % len(self._worker_processes)
			self._top_uid += 1
			self._worker_inputs[i].send_bytes(uidbytes + argbytes)
		for uid in uids:
			i = uid % len(self._worker_processes)
			outbytes = self._worker_outputs[i].recv_bytes()
			got_uid = int.from_bytes(outbytes[:8], "little")
			assert got_uid == uid
			retval = self.unpack(zlib.decompress(outbytes[8:]))
			if isinstance(retval, Exception):
				raise retval
			ret.append(retval)
		return ret

	@world_locked
	def _init_graph(
		self,
		name: Key,
		type_s: str = "DiGraph",
		data: CharacterFacade | Graph | nx.Graph | dict | KeyframeTuple = None,
	) -> None:
		if name in self.illegal_graph_names:
			raise GraphNameError("Illegal name")
		now = self._btt()
		for rbcache, rbname in [
			(self._characters_rulebooks_cache, "character_rulebook"),
			(self._units_rulebooks_cache, "unit_rulebook"),
			(
				self._characters_things_rulebooks_cache,
				"character_thing_rulebook",
			),
			(
				self._characters_places_rulebooks_cache,
				"character_place_rulebook",
			),
			(
				self._characters_portals_rulebooks_cache,
				"character_portal_rulebook",
			),
		]:
			try:
				kf = rbcache.get_keyframe(*now)
			except KeyframeError:
				kf = {}
				for ch in self._graph_cache.iter_entities(*now):
					# may yield this very character
					try:
						kf[ch] = rbcache.retrieve(ch, *now)
					except KeyError:
						kf[ch] = (rbname, ch)
			kf[name] = (rbname, name)
			rbcache.set_keyframe(*now, kf)
		branch, turn, tick = self._btt()
		self._graph_cache.store(name, branch, turn, tick, type_s)
		self.snap_keyframe(silent=True, update_worker_processes=False)
		self.query.graphs_insert(name, branch, turn, tick, type_s)
		self._extend_branch(branch, turn, tick)
		if isinstance(data, DiGraph):
			nodes = data._nodes_state()
			edges = data._edges_state()
			val = data._val_state()
			self._snap_keyframe_de_novo_graph(
				name, branch, turn, tick, nodes, edges, val
			)
			self.query.keyframe_graph_insert(
				name, branch, turn, tick, nodes, edges, val
			)
		elif isinstance(data, nx.Graph):
			nodes = {k: v.copy() for (k, v) in data.nodes.items()}
			edges = {}
			for orig in data.adj:
				succs = edges[orig] = {}
				for dest, stats in data.adj[orig].items():
					succs[dest] = stats.copy()
			self._snap_keyframe_de_novo_graph(
				name,
				branch,
				turn,
				tick,
				nodes,
				edges,
				data.graph,
			)
			self.query.keyframe_graph_insert(
				name,
				branch,
				turn,
				tick,
				nodes,
				edges,
				data.graph,
			)
		elif isinstance(data, dict):
			try:
				data = nx.from_dict_of_dicts(data)
			except AttributeError:
				data = nx.from_dict_of_lists(data)
			nodes = {k: v.copy() for (k, v) in data.nodes.items()}
			edges = {}
			for orig in data.adj:
				succs = edges[orig] = {}
				for dest, stats in data.adj[orig].items():
					succs[dest] = stats.copy()
			self._snap_keyframe_de_novo_graph(
				name, branch, turn, tick, nodes, edges, {}
			)
			self.query.keyframe_graph_insert(
				name,
				branch,
				turn,
				tick,
				nodes,
				edges,
				{},
			)
		elif data is None:
			self._snap_keyframe_de_novo_graph(
				name, branch, turn, tick, {}, {}, {}
			)
			self.query.keyframe_graph_insert(
				name, branch, turn, tick, {}, {}, {}
			)
		else:
			if len(data) != 3 or not all(isinstance(d, dict) for d in data):
				raise TypeError("Invalid graph data")
			self._snap_keyframe_de_novo_graph(name, branch, turn, tick, *data)
			self.query.keyframe_graph_insert(name, branch, turn, tick, *data)
		if hasattr(self, "_worker_processes"):
			self._call_every_subprocess("_add_character", name, data)

	@world_locked
	def _complete_turn(self, branch: str, turn: int) -> None:
		self._extend_branch(branch, turn, self.turn_end_plan(branch, turn))
		self._turns_completed_d[branch] = turn
		self.query.complete_turn(
			branch, turn, discard_rules=not self.keep_rules_journal
		)

	def _get_last_completed_turn(self, branch: Branch) -> Turn | None:
		if branch not in self._turns_completed_d:
			return None
		return self._turns_completed_d[branch]

	def _load_graphs(self) -> None:
		for charn, branch, turn, tick, typ in self.query.characters():
			self._graph_cache.store(
				charn,
				branch,
				turn,
				tick,
				(typ if typ != "Deleted" else None),
			)
			self._graph_objs[charn] = self.char_cls(
				self, charn, init_rulebooks=False
			)

	def _make_node(self, graph: Character, node: Key) -> thing_cls | place_cls:
		if self._is_thing(graph.name, node):
			return self.thing_cls(graph, node)
		else:
			return self.place_cls(graph, node)

	def _make_edge(
		self,
		graph: Character,
		orig: Key,
		dest: Key,
		idx=0,
	) -> portal_cls:
		return self.portal_cls(graph, orig, dest)

	def _is_timespan_too_big(
		self, branch: str, turn_from: int, turn_to: int
	) -> bool:
		"""Return whether the changes between these turns are numerous enough that you might as well use the slow delta

		Somewhat imprecise.

		"""
		kfint = self.query.keyframe_interval
		if kfint is None:
			return False
		if turn_from == turn_to:
			return self._turn_end_plan[branch, turn_from] > kfint
		acc = 0
		for r in range(
			min((turn_from, turn_to)),
			max((turn_from, turn_to)),
		):
			acc += self._turn_end_plan[branch, r]
			if acc > kfint:
				return True
		return False

	def get_delta(
		self,
		time_from: Time | tuple[Branch, Turn],
		time_to: Time | tuple[Branch, Turn],
		slow: bool = False,
	) -> DeltaDict:
		"""Get a dictionary describing changes to the world.

		Most keys will be character names, and their values will be
		dictionaries of the character's stats' new values, with ``None``
		for deleted keys. Characters' dictionaries have special keys
		'nodes' and 'edges' which contain booleans indicating whether
		the node or edge has been created (True) or deleted (False), and 'node_val' and
		'edge_val' for the stats of those entities. For edges (also
		called portals) these dictionaries are two layers deep, keyed
		first by the origin, then by the destination.

		Characters also have special keys for the various rulebooks
		they have:

		* ``'character_rulebook'``
		* ``'unit_rulebook'``
		* ``'character_thing_rulebook'``
		* ``'character_place_rulebook'``
		* ``'character_portal_rulebook'``

		And each node and edge may have a 'rulebook' stat of its own.
		If a node is a thing, it gets a 'location'; when the 'location'
		is deleted, that means it's back to being a place.

		Keys at the top level that are not character names:

		* ``'rulebooks'``, a dictionary keyed by the name of each changed
		  rulebook, the value being a list of rule names
		* ``'rules'``, a dictionary keyed by the name of each changed rule,
		  containing any of the lists ``'triggers'``, ``'prereqs'``,
		  and ``'actions'``


		:param slow: Whether to compare entire keyframes. Default ``False``,
			but we may take that approach anyway, if comparing between branches,
			or between times that are far enough apart that a delta assuming
			linear time would require *more* comparisons than comparing keyframes.

		"""
		if len(time_from) < 3 or time_from[2] is None:
			time_from = (*time_from[:2], self._turn_end_plan[time_from[:2]])
		if len(time_to) < 3 or time_to[2] is None:
			time_to = (*time_to[:2], self._turn_end_plan[time_to[:2]])
		if time_from == time_to:
			return {}
		if time_from[0] == time_to[0]:
			if slow or self._is_timespan_too_big(
				time_from[0], time_from[1], time_to[1]
			):
				return self._unpack_slightly_packed_delta(
					self._get_slow_delta(time_from, time_to)
				)
			else:
				return self._get_branch_delta(
					*time_from, time_to[1], time_to[2]
				)
		return self._unpack_slightly_packed_delta(
			self._get_slow_delta(time_from, time_to)
		)

	def _unpack_slightly_packed_delta(
		self, delta: SlightlyPackedDeltaType
	) -> DeltaDict:
		unpack = self.unpack
		delta = delta.copy()
		delt = {}
		if UNIVERSAL in delta:
			universal = delt["universal"] = {}
			for k, v in delta.pop(UNIVERSAL).items():
				universal[unpack(k)] = unpack(v)
		if RULES in delta:
			rules = delt["rules"] = {}
			for rule_name, funclists in delta.pop(RULES).items():
				rules[unpack(rule_name)] = {
					"triggers": unpack(funclists[TRIGGERS]),
					"prereqs": unpack(funclists[PREREQS]),
					"actions": unpack(funclists[ACTIONS]),
				}
		if RULEBOOK in delta:
			rulebook = delt["rulebook"] = {}
			for rulebok, rules in delta.pop(RULEBOOK).items():
				rulebook[unpack(rulebok)] = unpack(rules)
		for char, chardeltpacked in delta.items():
			if chardeltpacked == b"\xc0":
				delt[unpack(char)] = None
				continue
			chardelt = delt[unpack(char)] = {}
			if NODES in chardeltpacked:
				chardelt["nodes"] = {
					unpack(node): extant == TRUE
					for (node, extant) in chardeltpacked.pop(NODES).items()
				}
			if EDGES in chardeltpacked:
				edges = chardelt["edges"] = {}
				for ab, ex in chardeltpacked.pop(EDGES).items():
					a, b = unpack(ab)
					if a not in edges:
						edges[a] = {}
					edges[a][b] = ex == TRUE
			if NODE_VAL in chardeltpacked:
				node_val = chardelt["node_val"] = {}
				for node, stats in chardeltpacked.pop(NODE_VAL).items():
					node_val[unpack(node)] = {
						unpack(k): unpack(v) for (k, v) in stats.items()
					}
			if EDGE_VAL in chardeltpacked:
				edge_val = chardelt["edge_val"] = {}
				for a, bs in chardeltpacked.pop(EDGE_VAL).items():
					aA = unpack(a)
					if aA not in edge_val:
						edge_val[aA] = {}
					for b, stats in bs.items():
						edge_val[aA][unpack(b)] = {
							unpack(k): unpack(v) for (k, v) in stats.items()
						}
			for k, v in chardeltpacked.items():
				chardelt[unpack(k)] = unpack(v)
		return delt

	def _get_slow_delta(
		self, btt_from: Time, btt_to: Time
	) -> SlightlyPackedDeltaType:
		def newgraph():
			return {
				# null mungers mean KeyError, which is correct
				NODES: PickyDefaultDict(
					bytes, args_munger=None, kwargs_munger=None
				),
				EDGES: PickyDefaultDict(
					bytes, args_munger=None, kwargs_munger=None
				),
				NODE_VAL: StructuredDefaultDict(
					1, bytes, args_munger=None, kwargs_munger=None
				),
				EDGE_VAL: StructuredDefaultDict(
					2, bytes, args_munger=None, kwargs_munger=None
				),
			}

		delta: dict[bytes, Any] = {
			UNIVERSAL: PickyDefaultDict(bytes),
			RULES: StructuredDefaultDict(1, bytes),
			RULEBOOK: PickyDefaultDict(bytes),
		}
		pack = self.pack
		now = self._btt()
		self._set_btt(*btt_from)
		kf_from = self.snap_keyframe()
		self._set_btt(*btt_to)
		kf_to = self.snap_keyframe()
		self._set_btt(*now)
		keys = []
		ids_from = []
		ids_to = []
		values_from = []
		values_to = []
		# Comparing object IDs is guaranteed never to give a false equality,
		# because of the way keyframes are constructed.
		# It may give a false inequality.
		non_graph_kf_keys = [
			"universal",
			"triggers",
			"prereqs",
			"actions",
			"neighborhood",
			"big",
			"rulebook",
		]
		for kfkey in non_graph_kf_keys:
			for k in (
				kf_from.get(kfkey, {}).keys() | kf_to.get(kfkey, {}).keys()
			):
				keys.append((kfkey, k))
				va = kf_from[kfkey].get(k)
				vb = kf_to[kfkey].get(k)
				ids_from.append(id(va))
				ids_to.append(id(vb))
				values_from.append(va)
				values_to.append(vb)
		for graph in kf_from["graph_val"].keys() | kf_to["graph_val"].keys():
			a = kf_from["graph_val"].get(graph, {})
			b = kf_to["graph_val"].get(graph, {})
			for k in a.keys() | b.keys():
				keys.append(("graph", graph, k))
				va = a.get(k)
				vb = b.get(k)
				ids_from.append(id(va))
				ids_to.append(id(vb))
				values_from.append(va)
				values_to.append(vb)
		for graph in kf_from["node_val"].keys() | kf_to["node_val"].keys():
			nodes = set()
			if graph in kf_from["node_val"]:
				nodes.update(kf_from["node_val"][graph].keys())
			if graph in kf_to["node_val"]:
				nodes.update(kf_to["node_val"][graph].keys())
			for node in nodes:
				a = kf_from["node_val"].get(graph, {}).get(node, {})
				b = kf_to["node_val"].get(graph, {}).get(node, {})
				for k in a.keys() | b.keys():
					keys.append(("node", graph, node, k))
					va = a.get(k)
					vb = b.get(k)
					ids_from.append(id(va))
					ids_to.append(id(vb))
					values_from.append(va)
					values_to.append(vb)
		for graph in kf_from["edge_val"].keys() | kf_to["edge_val"].keys():
			edges = set()
			if graph in kf_from["edge_val"]:
				for orig in kf_from["edge_val"][graph]:
					for dest in kf_from["edge_val"][graph][orig]:
						edges.add((orig, dest))
			if graph in kf_to["edge_val"]:
				for orig in kf_to["edge_val"][graph]:
					for dest in kf_to["edge_val"][graph][orig]:
						edges.add((orig, dest))
			for orig, dest in edges:
				a = (
					kf_from["edge_val"]
					.get(graph, {})
					.get(orig, {})
					.get(dest, {})
				)
				b = (
					kf_to["edge_val"]
					.get(graph, {})
					.get(orig, {})
					.get(dest, {})
				)
				for k in a.keys() | b.keys():
					keys.append(("edge", graph, orig, dest, k))
					va = a.get(k)
					vb = b.get(k)
					ids_from.append(id(va))
					ids_to.append(id(vb))
					values_from.append(va)
					values_to.append(vb)

		def pack_one(k, va, vb, deleted_nodes, deleted_edges):
			if va == vb:
				return
			v = pack(vb)
			if k[0] == "universal":
				key = pack(k[1])
				delta[UNIVERSAL][key] = v
			elif k[0] == "triggers":
				rule = pack(k[1])
				delta[RULES][rule][TRIGGERS] = v
			elif k[0] == "prereqs":
				rule = pack(k[1])
				delta[RULES][rule][PREREQS] = v
			elif k[0] == "actions":
				rule = pack(k[1])
				delta[RULES][rule][ACTIONS] = v
			elif k[0] == "neighborhood":
				rule = pack(k[1])
				delta[RULES][rule][NEIGHBORHOOD] = v
			elif k[0] == "big":
				rule = pack(k[1])
				delta[RULES][rule][BIG] = v
			elif k[0] == "rulebook":
				rulebook = pack(k[1])
				delta[RULEBOOK][rulebook] = v
			elif k[0] == "node":
				_, graph, node, key = k
				if graph in deleted_nodes and node in deleted_nodes[graph]:
					return
				graph, node, key = map(pack, (graph, node, key))
				if graph not in delta:
					delta[graph] = newgraph()
				delta[graph][NODE_VAL][node][key] = v
			elif k[0] == "edge":
				_, graph, orig, dest, key = k
				if (graph, orig, dest) in deleted_edges:
					return
				graph, orig, dest, key = map(pack, (graph, orig, dest, key))
				if graph not in delta:
					delta[graph] = newgraph()
				delta[graph][EDGE_VAL][orig][dest][key] = v
			else:
				assert k[0] == "graph"
				_, graph, key = k
				graph, key = map(pack, (graph, key))
				if graph not in delta:
					delta[graph] = newgraph()
				delta[graph][key] = v

		def pack_node(graph, node, existence):
			grap, node = map(pack, (graph, node))
			if grap not in delta:
				delta[grap] = newgraph()
			delta[grap][NODES][node] = existence

		def pack_edge(graph, orig, dest, existence):
			graph, origdest = map(pack, (graph, (orig, dest)))
			if graph not in delta:
				delta[graph] = newgraph()
			delta[graph][EDGES][origdest] = existence

		futs = []
		with ThreadPoolExecutor() as pool:
			nodes_intersection = (
				kf_from["nodes"].keys() & kf_to["nodes"].keys()
			)
			deleted_nodes = {}
			for graph in nodes_intersection:
				deleted_nodes_here = deleted_nodes[graph] = (
					kf_from["nodes"][graph].keys()
					- kf_to["nodes"][graph].keys()
				)
				for node in deleted_nodes_here:
					futs.append(pool.submit(pack_node, graph, node, FALSE))
			deleted_edges = set()
			for graph in kf_from["edges"]:
				for orig in kf_from["edges"][graph]:
					for dest, ex in kf_from["edges"][graph][orig].items():
						deleted_edges.add((graph, orig, dest))
			for graph in kf_to["edges"]:
				for orig in kf_to["edges"][graph]:
					for dest, ex in kf_to["edges"][graph][orig].items():
						deleted_edges.discard((graph, orig, dest))
			values_changed = np.array(ids_from) != np.array(ids_to)
			for k, va, vb, _ in filter(
				itemgetter(3),
				zip(keys, values_from, values_to, values_changed),
			):
				futs.append(
					pool.submit(
						pack_one, k, va, vb, deleted_nodes, deleted_edges
					)
				)
			for graf in (
				kf_from["graph_val"].keys() - kf_to["graph_val"].keys()
			):
				delta[self.pack(graf)] = NONE
			for graph in nodes_intersection:
				for node in (
					kf_to["nodes"][graph].keys()
					- kf_from["nodes"][graph].keys()
				):
					futs.append(pool.submit(pack_node, graph, node, TRUE))
			for graph, orig, dest in deleted_edges:
				futs.append(pool.submit(pack_edge, graph, orig, dest, FALSE))
			edges_to = {
				(graph, orig, dest)
				for graph in kf_to["edges"]
				for orig in kf_to["edges"][graph]
				for dest in kf_to["edges"][graph][orig]
			}
			edges_from = {
				(graph, orig, dest)
				for graph in kf_from["edges"]
				for orig in kf_from["edges"][graph]
				for dest in kf_from["edges"][graph][orig]
			}
			for graph, orig, dest in edges_to - edges_from:
				futs.append(pool.submit(pack_edge, graph, orig, dest, TRUE))
			for deleted in (
				kf_from["graph_val"].keys() - kf_to["graph_val"].keys()
			):
				delta[pack(deleted)] = NONE
			futwait(futs)
		if not delta[UNIVERSAL]:
			del delta[UNIVERSAL]
		if not delta[RULEBOOK]:
			del delta[RULEBOOK]
		todel = []
		for rule_name, rule in delta[RULES].items():
			if not rule[TRIGGERS]:
				del rule[TRIGGERS]
			if not rule[PREREQS]:
				del rule[PREREQS]
			if not rule[ACTIONS]:
				del rule[ACTIONS]
			if not rule:
				todel.append(rule_name)
		for deleterule in todel:
			del delta[deleterule]
		if not delta[RULES]:
			del delta[RULES]
		for key, mapp in delta.items():
			if key in {RULES, RULEBOOKS, ETERNAL, UNIVERSAL} or mapp == NONE:
				continue
			todel = []
			for keey, mappp in mapp.items():
				if not mappp:
					todel.append(keey)
			for todo in todel:
				del mapp[todo]
		for added in kf_to["graph_val"].keys() - kf_from["graph_val"].keys():
			graphn = pack(added)
			if graphn not in delta:
				delta[graphn] = {}
		return delta

	def _del_rulebook(self, rulebook):
		raise NotImplementedError("Can't delete rulebooks yet")

	@property
	def stores(self):
		return (
			self.action,
			self.prereq,
			self.trigger,
			self.function,
			self.method,
			self.string,
		)

	def debug(self, msg: str) -> None:
		"""Log a message at level 'debug'"""
		self.log("debug", msg)

	def info(self, msg: str) -> None:
		"""Log a message at level 'info'"""
		self.log("info", msg)

	def warning(self, msg: str) -> None:
		"""Log a message at level 'warning'"""
		self.log("warning", msg)

	def error(self, msg: str) -> None:
		"""Log a message at level 'error'"""
		self.log("error", msg)

	def critical(self, msg: str) -> None:
		"""Log a message at level 'critical'"""
		self.log("critical", msg)

	def close(self) -> None:
		"""Commit changes and close the database

		This will be useless thereafter.

		"""
		if hasattr(self, "_closed"):
			raise RuntimeError("Already closed")
		time_was = (self.turn, self.tick)
		if time_was > self._branch_end():
			(self.turn, self.tick) = self._branch_end()
		if (
			self._keyframe_on_close
			and self._btt() not in self._keyframes_times
		):
			self.snap_keyframe(silent=True, update_worker_processes=False)
		(self.turn, self.tick) = time_was
		for store in self.stores:
			if hasattr(store, "save"):
				store.save(reimport=False)
			if not hasattr(store, "_filename") or store._filename is None:
				continue
			path, filename = os.path.split(store._filename)
			modname = filename[:-3]
			if modname in sys.modules:
				del sys.modules[modname]
		self.commit()
		self.query.close()
		self.shutdown()
		self._closed = True

	def __enter__(self):
		"""Return myself. For compatibility with ``with`` semantics."""
		return self

	def __exit__(self, *args):
		"""Close on exit."""
		self.close()

	def _handled_char(
		self,
		charn: Key,
		rulebook: Key,
		rulen: Key,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> None:
		try:
			self._character_rules_handled_cache.store(
				charn, rulebook, rulen, branch, turn, tick
			)
		except ValueError:
			assert (
				rulen
				in self._character_rules_handled_cache.handled[
					charn, rulebook, branch, turn
				]
			)
			return
		self.query.handled_character_rule(
			charn, rulebook, rulen, branch, turn, tick
		)

	def _handled_av(
		self,
		character: Key,
		graph: Key,
		avatar: Key,
		rulebook: Key,
		rule: Key,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> None:
		try:
			self._unit_rules_handled_cache.store(
				character, graph, avatar, rulebook, rule, branch, turn, tick
			)
		except ValueError:
			assert (
				rule
				in self._unit_rules_handled_cache.handled[
					character, graph, avatar, rulebook, branch, turn
				]
			)
			return
		self.query.handled_unit_rule(
			character, rulebook, rule, graph, avatar, branch, turn, tick
		)

	def _handled_char_thing(
		self,
		character: Key,
		thing: Key,
		rulebook: Key,
		rule: Key,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> None:
		try:
			self._character_thing_rules_handled_cache.store(
				character, thing, rulebook, rule, branch, turn, tick
			)
		except ValueError:
			assert (
				rule
				in self._character_thing_rules_handled_cache.handled[
					character, thing, rulebook, branch, turn
				]
			)
			return
		self.query.handled_character_thing_rule(
			character, rulebook, rule, thing, branch, turn, tick
		)

	def _handled_char_place(
		self,
		character: Key,
		place: Key,
		rulebook: Key,
		rule: Key,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> None:
		try:
			self._character_place_rules_handled_cache.store(
				character, place, rulebook, rule, branch, turn, tick
			)
		except ValueError:
			assert (
				rule
				in self._character_place_rules_handled_cache.handled[
					character, place, rulebook, branch, turn
				]
			)
			return
		self.query.handled_character_place_rule(
			character, rulebook, rule, place, branch, turn, tick
		)

	def _handled_char_port(
		self,
		character: Key,
		orig: Key,
		dest: Key,
		rulebook: Key,
		rule: Key,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> None:
		try:
			self._character_portal_rules_handled_cache.store(
				character, orig, dest, rulebook, rule, branch, turn, tick
			)
		except ValueError:
			assert (
				rule
				in self._character_portal_rules_handled_cache.handled[
					character, orig, dest, rulebook, branch, turn
				]
			)
			return
		self.query.handled_character_portal_rule(
			character, orig, dest, rulebook, rule, branch, turn, tick
		)

	def _handled_node(
		self,
		character: Key,
		node: Key,
		rulebook: Key,
		rule: Key,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> None:
		try:
			self._node_rules_handled_cache.store(
				character, node, rulebook, rule, branch, turn, tick
			)
		except ValueError:
			assert (
				rule
				in self._node_rules_handled_cache.handled[
					character, node, rulebook, branch, turn
				]
			)
			return
		self.query.handled_node_rule(
			character, node, rulebook, rule, branch, turn, tick
		)

	def _handled_portal(
		self,
		character: Key,
		orig: Key,
		dest: Key,
		rulebook: Key,
		rule: Key,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> None:
		try:
			self._portal_rules_handled_cache.store(
				character, orig, dest, rulebook, rule, branch, turn, tick
			)
		except ValueError:
			assert (
				rule
				in self._portal_rules_handled_cache.handled[
					character, orig, dest, rulebook, branch, turn
				]
			)
			return
		self.query.handled_portal_rule(
			character, orig, dest, rulebook, rule, branch, turn, tick
		)

	@world_locked
	@_all_worker_locks
	def _update_all_worker_process_states(self, clobber=False):
		for store in self.stores:
			store.save(reimport=False)
		kf_payload = None
		deltas = {}
		for i in range(len(self._worker_processes)):
			branch_from, turn_from, tick_from = self._worker_updated_btts[i]
			if (branch_from, turn_from, tick_from) == self._btt():
				continue
			if not clobber and branch_from == self.branch:
				old_eternal = self._worker_last_eternal
				new_eternal = self._worker_last_eternal = dict(
					self.eternal.items()
				)
				eternal_delta = {
					k: new_eternal.get(k)
					for k in old_eternal.keys() | new_eternal.keys()
					if old_eternal.get(k) != new_eternal.get(k)
				}
				if (branch_from, turn_from, tick_from) in deltas:
					delt = deltas[branch_from, turn_from, tick_from]
				else:
					delt = deltas[branch_from, turn_from, tick_from] = (
						self._get_branch_delta(
							branch_from,
							turn_from,
							tick_from,
							self.turn,
							self.tick,
						)
					)
				if eternal_delta:
					delt["eternal"] = eternal_delta
				kwargs = {}
				if self._prefix is None:
					kwargs["_replace_funcs_plain"] = plain = {}
					kwargs["_replace_funcs_pkl"] = pkl = {}
					for name, store in [
						("function", self.function),
						("method", self.method),
						("trigger", self.trigger),
						("prereq", self.prereq),
						("action", self.action),
					]:
						if hasattr(store, "iterplain") and callable(
							store.iterplain
						):
							plain[name] = dict(store.iterplain())
							continue
						else:
							pkl[name] = pickle.dumps(store)
				argbytes = sys.maxsize.to_bytes(8, "little") + zlib.compress(
					self.pack(
						(
							"_upd",
							(
								None,
								self.branch,
								self.turn,
								self.tick,
								(None, delt),
							),
							kwargs,
						)
					)
				)
				self._worker_inputs[i].send_bytes(argbytes)
			else:
				if kf_payload is None:
					kf_payload = self._get_worker_kf_payload()
				self._worker_inputs[i].send_bytes(kf_payload)
			self._worker_updated_btts[i] = self._btt()

	@world_locked
	def _update_worker_process_state(self, i, lock=True):
		branch_from, turn_from, tick_from = self._worker_updated_btts[i]
		if (branch_from, turn_from, tick_from) == self._btt():
			return
		old_eternal = self._worker_last_eternal
		new_eternal = self._worker_last_eternal = dict(self.eternal.items())
		eternal_delta = {
			k: new_eternal.get(k)
			for k in old_eternal.keys() | new_eternal.keys()
			if old_eternal.get(k) != new_eternal.get(k)
		}
		if branch_from == self.branch:
			delt = self._get_branch_delta(
				branch_from, turn_from, tick_from, self.turn, self.tick
			)
			delt["eternal"] = eternal_delta
			argbytes = sys.maxsize.to_bytes(8, "little") + zlib.compress(
				self.pack(
					(
						"_upd",
						(
							None,
							self.branch,
							self.turn,
							self.tick,
							(None, delt),
						),
						{},
					)
				)
			)
		else:
			argbytes = self._get_worker_kf_payload()
		if lock:
			with self._worker_locks[i]:
				self._worker_inputs[i].send_bytes(argbytes)
				self._worker_updated_btts[i] = self._btt()
		else:
			self._worker_inputs[i].send_bytes(argbytes)
			self._worker_updated_btts[i] = self._btt()

	def _changed(self, charn, entity: tuple) -> bool:
		if len(entity) == 1:
			vbranches = self._node_val_cache.settings
			entikey = (charn, entity[0])
		elif len(entity) != 2:
			raise TypeError("Unknown entity type")
		else:
			vbranches = self._edge_val_cache.settings
			entikey = (
				charn,
				*entity,
				0,
			)
		branch, turn, _ = self._btt()
		turn -= 1
		if turn <= self.branch_start_turn():
			branch = self.branch_parent(branch)
			assert branch is not None
		if branch not in vbranches:
			return False
		vbranchesb = vbranches[branch]
		if turn not in vbranchesb:
			return False
		return entikey in vbranchesb[turn].entikeys

	def _iter_submit_triggers(
		self,
		prio: float,
		rulebook: Key,
		rule: Rule,
		handled_fun: callable,
		entity,
		neighbors: Iterable = None,
	):
		changed = self._changed
		charn = entity.character.name
		if neighbors is not None and not (
			any(changed(charn, neighbor) for neighbor in neighbors)
		):
			return
		if self.trigger.truth in rule.triggers:
			fut = fake_submit(self.trigger.truth)
			fut.rule = rule
			fut.prio = prio
			fut.entity = entity
			fut.rulebook = rulebook
			fut.handled = handled_fun
			yield fut
			return
		for trigger in rule.triggers:
			fut = self.submit(trigger, entity)
			fut.rule = rule
			fut.prio = prio
			fut.entity = entity
			fut.rulebook = rulebook
			fut.handled = handled_fun
			yield fut

	def _check_prereqs(self, rule: Rule, handled_fun: callable, entity):
		if not entity:
			return False
		for prereq in rule.prereqs:
			res = prereq(entity)
			if not res:
				handled_fun(self.tick)
				return False
		return True

	def _do_actions(self, rule: Rule, handled_fun: callable, entity):
		if rule.big:
			entity = entity.facade()
		actres = []
		for action in rule.actions:
			res = action(entity)
			if res:
				actres.append(res)
			if not entity:
				break
		if rule.big:
			with self.batch():
				entity.engine.apply()
		handled_fun(self.tick)
		return actres

	def _get_place_neighbors(self, charn: Key, name: Key) -> set[Key]:
		seen: set[Key] = set()
		for succ in self._edges_cache.iter_successors(
			charn, name, *self._btt()
		):
			seen.add(succ)
		for pred in self._edges_cache.iter_predecessors(
			charn, name, *self._btt()
		):
			seen.add(pred)
		return seen

	def _get_place_contents(self, charn: Key, name: Key) -> set[Key]:
		try:
			return self._node_contents_cache.retrieve(
				charn, name, *self._btt()
			)
		except KeyError:
			return set()

	def _iter_place_portals(
		self, charn: Key, name: Key
	) -> Iterator[tuple[Key, Key]]:
		now = self._btt()
		for dest in self._edges_cache.iter_successors(charn, name, *now):
			yield (name, dest)
		for orig in self._edges_cache.iter_predecessors(charn, name, *now):
			yield (orig, name)

	def _get_thing_location_tup(
		self, charn: Key, name: Key
	) -> tuple[Key, Key] | ():
		try:
			return (self._things_cache.retrieve(charn, name, *self._btt()),)
		except KeyError:
			return ()

	def _get_neighbors(
		self,
		entity: place_cls | thing_cls | portal_cls,
		neighborhood: int | None,
	) -> list[tuple[Key] | tuple[Key, Key]] | None:
		"""Get a list of neighbors within the neighborhood

		Neighbors are given by a tuple containing only their name,
		if they are Places or Things, or their origin's and destination's
		names, if they are Portals.

		"""
		charn = entity.character.name
		btt = self._btt()

		if neighborhood is None:
			return None
		if hasattr(entity, "name"):
			cache_key = (charn, entity.name, *btt)
		else:
			cache_key = (
				charn,
				entity.origin.name,
				entity.destination.name,
				*btt,
			)
		if cache_key in self._neighbors_cache:
			return self._neighbors_cache[cache_key]
		if hasattr(entity, "name"):
			neighbors = [(entity.name,)]
			while hasattr(entity, "location"):
				entity = entity.location
				neighbors.append((entity.name,))
		else:
			neighbors = [(entity.origin.name, entity.destination.name)]
		seen = set(neighbors)
		i = 0
		for _ in range(neighborhood):
			j = len(neighbors)
			for neighbor in neighbors[i:]:
				if len(neighbor) == 2:
					orign, destn = neighbor
					for placen in (orign, destn):
						for neighbor_place in chain(
							self._get_place_neighbors(charn, placen),
							self._get_place_contents(charn, placen),
							self._get_thing_location_tup(charn, placen),
						):
							if neighbor_place not in seen:
								neighbors.append((neighbor_place,))
								seen.add(neighbor_place)
							for neighbor_thing in self._get_place_contents(
								charn, neighbor_place
							):
								if neighbor_thing not in seen:
									neighbors.append((neighbor_thing,))
									seen.add(neighbor_thing)
						for neighbor_portal in self._iter_place_portals(
							charn, placen
						):
							if neighbor_portal not in seen:
								neighbors.append(neighbor_portal)
								seen.add(neighbor_portal)
				else:
					(neighbor,) = neighbor
					for neighbor_place in chain(
						self._get_place_neighbors(charn, neighbor),
						self._get_place_contents(charn, neighbor),
						self._get_thing_location_tup(charn, neighbor),
					):
						if neighbor_place not in seen:
							neighbors.append((neighbor_place,))
							seen.add(neighbor_place)
						for neighbor_thing in self._get_place_contents(
							charn, neighbor_place
						):
							if neighbor_thing not in seen:
								neighbors.append((neighbor_thing,))
								seen.add(neighbor_thing)
					for neighbor_portal in self._iter_place_portals(
						charn, neighbor
					):
						if neighbor_portal not in seen:
							neighbors.append(neighbor_portal)
							seen.add(neighbor_portal)
			i = j
		self._neighbors_cache[cache_key] = neighbors
		return neighbors

	def _get_effective_neighbors(
		self,
		entity: place_cls | thing_cls | portal_cls,
		neighborhood: Optional[int],
	):
		"""Get neighbors unless that's a different set of entities since last turn

		In which case return None

		"""
		if neighborhood is None:
			return None

		branch_now, turn_now, tick_now = self._btt()
		if turn_now <= 1:
			# everything's "created" at the start of the game,
			# and therefore, there's been a "change" to the neighborhood
			return None
		with self.world_lock:
			self.load_at(branch_now, turn_now - 1, 0)
			self._oturn -= 1
			self._otick = 0
			last_turn_neighbors = self._get_neighbors(entity, neighborhood)
			self._set_btt(branch_now, turn_now, tick_now)
			this_turn_neighbors = self._get_neighbors(entity, neighborhood)
		if set(last_turn_neighbors) != set(this_turn_neighbors):
			return None
		return this_turn_neighbors

	def _get_node_mini(self, graphn: Key, noden: Key):
		node_objs = self._node_objs
		key = (graphn, noden)
		if key not in node_objs:
			node_objs[key] = self._make_node(self.character[graphn], noden)
		return node_objs[key]

	def _get_thing(self, graphn: Key, thingn: Key):
		node_objs = self._node_objs
		key = (graphn, thingn)
		if key not in node_objs:
			node_objs[key] = self.thing_cls(self.character[graphn], thingn)
		return node_objs[key]

	def _get_place(self, graphn: Key, placen: Key):
		node_objs = self._node_objs
		key = (graphn, placen)
		if key not in node_objs:
			node_objs[key] = self.place_cls(self.character[graphn], placen)
		return node_objs[key]

	def _eval_triggers(self):
		branch, turn, tick = self._btt()
		charmap = self.character
		rulemap = self.rule
		todo = defaultdict(list)
		trig_futs = []

		for (
			prio,
			charactername,
			rulebook,
			rulename,
		) in self._character_rules_handled_cache.iter_unhandled_rules(
			branch, turn, tick
		):
			if charactername not in charmap:
				continue
			rule = rulemap[rulename]
			handled = partial(
				self._handled_char,
				charactername,
				rulebook,
				rulename,
				branch,
				turn,
			)
			entity = charmap[charactername]
			trig_futs.extend(
				self._iter_submit_triggers(
					prio,
					rulebook,
					rule,
					handled,
					entity,
					None,
				)
			)

		avcache_retr = self._unitness_cache._base_retrieve
		node_exists = self._node_exists
		get_node = self._get_node_mini
		get_thing = self._get_thing
		get_place = self._get_place

		for (
			prio,
			charn,
			graphn,
			avn,
			rulebook,
			rulen,
		) in self._unit_rules_handled_cache.iter_unhandled_rules(
			branch, turn, tick
		):
			if not node_exists(graphn, avn) or avcache_retr(
				(charn, graphn, avn, branch, turn, tick)
			) in (KeyError, None):
				continue
			rule = rulemap[rulen]
			handled = partial(
				self._handled_av,
				charn,
				graphn,
				avn,
				rulebook,
				rulen,
				branch,
				turn,
			)
			entity = get_node(graphn, avn)
			trig_futs.extend(
				self._iter_submit_triggers(
					prio,
					rulebook,
					rule,
					handled,
					entity,
					self._get_effective_neighbors(entity, rule.neighborhood),
				)
			)
		is_thing = self._is_thing
		handled_char_thing = self._handled_char_thing
		for (
			prio,
			charn,
			thingn,
			rulebook,
			rulen,
		) in self._character_thing_rules_handled_cache.iter_unhandled_rules(
			branch, turn, tick
		):
			if not node_exists(charn, thingn) or not is_thing(charn, thingn):
				continue
			rule = rulemap[rulen]
			handled = partial(
				handled_char_thing,
				charn,
				thingn,
				rulebook,
				rulen,
				branch,
				turn,
			)
			entity = get_thing(charn, thingn)
			trig_futs.extend(
				self._iter_submit_triggers(
					prio,
					rulebook,
					rule,
					handled,
					entity,
					self._get_effective_neighbors(entity, rule.neighborhood),
				)
			)
		handled_char_place = self._handled_char_place
		for (
			prio,
			charn,
			placen,
			rulebook,
			rulen,
		) in self._character_place_rules_handled_cache.iter_unhandled_rules(
			branch, turn, tick
		):
			if not node_exists(charn, placen) or is_thing(charn, placen):
				continue
			rule = rulemap[rulen]
			handled = partial(
				handled_char_place,
				charn,
				placen,
				rulebook,
				rulen,
				branch,
				turn,
			)
			entity = get_place(charn, placen)
			trig_futs.extend(
				self._iter_submit_triggers(
					prio,
					rulebook,
					rule,
					handled,
					entity,
					self._get_effective_neighbors(entity, rule.neighborhood),
				)
			)
		edge_exists = self._edge_exists
		get_edge = self._get_edge
		handled_char_port = self._handled_char_port
		for (
			prio,
			charn,
			orign,
			destn,
			rulebook,
			rulen,
		) in self._character_portal_rules_handled_cache.iter_unhandled_rules(
			branch, turn, tick
		):
			if not edge_exists(charn, orign, destn):
				continue
			rule = rulemap[rulen]
			handled = partial(
				handled_char_port,
				charn,
				orign,
				destn,
				rulebook,
				rulen,
				branch,
				turn,
			)
			entity = get_edge(charn, orign, destn)
			trig_futs.extend(
				self._iter_submit_triggers(
					prio,
					rulebook,
					rule,
					handled,
					entity,
					self._get_effective_neighbors(entity, rule.neighborhood),
				)
			)
		handled_node = self._handled_node
		for (
			prio,
			charn,
			noden,
			rulebook,
			rulen,
		) in self._node_rules_handled_cache.iter_unhandled_rules(
			branch, turn, tick
		):
			if not node_exists(charn, noden):
				continue
			rule = rulemap[rulen]
			handled = partial(
				handled_node, charn, noden, rulebook, rulen, branch, turn
			)
			entity = get_node(charn, noden)
			trig_futs.extend(
				self._iter_submit_triggers(
					prio,
					rulebook,
					rule,
					handled,
					entity,
					self._get_effective_neighbors(entity, rule.neighborhood),
				)
			)
		handled_portal = self._handled_portal
		for (
			prio,
			charn,
			orign,
			destn,
			rulebook,
			rulen,
		) in self._portal_rules_handled_cache.iter_unhandled_rules(
			branch, turn, tick
		):
			if not edge_exists(charn, orign, destn):
				continue
			rule = rulemap[rulen]
			handled = partial(
				handled_portal,
				charn,
				orign,
				destn,
				rulebook,
				rulen,
				branch,
				turn,
			)
			entity = get_edge(charn, orign, destn)
			trig_futs.extend(
				self._iter_submit_triggers(
					prio,
					rulebook,
					rule,
					handled,
					entity,
					self._get_effective_neighbors(entity, rule.neighborhood),
				)
			)

		for fut in trig_futs:
			if fut.result():
				todo[fut.prio, fut.rulebook].append(
					(
						fut.rule,
						fut.handled,
						fut.entity,
					)
				)
			else:
				fut.handled(self.tick)

		return todo

	def _fmtent(self, entity):
		if isinstance(entity, self.char_cls):
			return entity.name
		elif hasattr(entity, "name"):
			return f"{entity.character.name}.node[{entity.name}]"
		else:
			return (
				f"{entity.character.name}.portal"
				f"[{entity.origin.name}][{entity.destination.name}]"
			)

	def _follow_one_rule(self, rule, handled, entity):
		check_prereqs = self._check_prereqs
		do_actions = self._do_actions

		if not entity:
			self.debug(
				f"not checking prereqs for rule {rule.name} "
				f"on nonexistent entity {self._fmtent(entity)}"
			)
			return
		self.debug(
			f"checking prereqs for rule {rule.name} on entity {self._fmtent(entity)}"
		)
		if check_prereqs(rule, handled, entity):
			self.debug(
				f"prereqs for rule {rule.name} on entity "
				f"{self._fmtent(entity)} satisfied, will run actions"
			)
			try:
				ret = do_actions(rule, handled, entity)
				self.debug(
					f"actions for rule {rule.name} on entity "
					f"{self._fmtent(entity)} have run without incident"
				)
				return ret
			except StopIteration as ex:
				raise InnerStopIteration from ex

	def _follow_rules(self, todo):
		# TODO: roll back changes done by rules that raise an exception
		# TODO: if there's a paradox while following some rule,
		#  start a new branch, copying handled rules
		for prio_rulebook in sort_set(todo.keys()):
			for rule, handled, entity in todo[prio_rulebook]:
				yield self._follow_one_rule(rule, handled, entity)

	def new_character(
		self,
		name: Key,
		data: Graph = None,
		layout: bool = False,
		node: dict = None,
		edge: dict = None,
		**kwargs,
	) -> Character:
		"""Create and return a new :class:`Character`.

		See :meth:`add_character` for details.

		"""
		self.add_character(name, data, layout, node=node, edge=edge, **kwargs)
		return self.character[name]

	def add_character(
		self,
		name: Key,
		data: Graph | DiGraph = None,
		layout: bool = False,
		node: dict = None,
		edge: dict = None,
		**kwargs,
	) -> None:
		"""Create a new character.

		You'll be able to access it as a :class:`Character` object by
		looking up ``name`` in my ``character`` property.

		``data``, if provided, should be a :class:`networkx.Graph`
		or :class:`networkx.DiGraph` object. The character will be
		a copy of it.

		``node`` may be a dictionary of dictionaries representing either
		``Thing`` objects, if they have a ``"location"`` key, or else
		``Place`` objects.

		``edge`` may be a 3-layer dictionary representing ``Portal`` objects,
		connecting mainly ``Place`` objects together.

		With ``layout=True``, compute a layout to make the
		graph show up nicely in elide.

		Any keyword arguments will be set as stats of the new character.

		"""
		if name in self.character:
			raise KeyError("Already have that character", name)
		if layout and (data or node or edge):
			if data is None:
				data = nx.DiGraph()
			if node:
				for name, nvs in node.items():
					data.add_node(name, **nvs)
			if edge:
				for orig, dests in edge.items():
					for dest, evs in dests.items():
						data.add_edge(orig, dest, **evs)
			nodes = data.nodes
			try:
				layout = normalize_layout(
					{
						name: name
						for name, node in nodes.items()
						if "location" not in node
					}
				)
			except (TypeError, ValueError):
				layout = normalize_layout(
					spring_layout(
						[
							name
							for name, node in nodes.items()
							if "location" not in node
						]
					)
				)
			for k, (x, y) in layout.items():
				nodes[k]["_x"] = x
				nodes[k]["_y"] = y
		if kwargs:
			if not data:
				data = nx.DiGraph()
			if not isinstance(data, Graph):
				try:
					data = from_dict_of_lists(data)
				except NetworkXError:
					data = from_dict_of_dicts(data)
			if node:
				for k, v in node.items():
					data.add_node(k, **v)
			if edge:
				for orig, dests in edge.items():
					for dest, v in dests.items():
						data.add_edge(orig, dest, **v)
			data.graph.update(kwargs)
		# When initializing the world state, we don't have to worry about deltas;
		# it's OK to make multiple characters at ('trunk', 0, 0).
		# At any time past the start, we have to advance the tick.
		if self.branch != self.main_branch or self.turn != 0 or self.tick != 0:
			self._nbtt()
		self._init_graph(name, "DiGraph", data)
		if self._btt() not in self._keyframes_times:
			self.snap_keyframe(silent=True, update_worker_processes=False)
		if hasattr(self, "_worker_processes"):
			self._update_all_worker_process_states(clobber=True)
		self._graph_objs[name] = self.char_cls(self, name)

	@world_locked
	def del_character(self, name: Key) -> None:
		"""Mark a graph as deleted

		:arg name: name of an existing graph

		"""
		# make sure the graph exists before deleting
		graph = self.character[name]
		for thing in list(graph.thing):
			del graph.thing[thing]
		for orig in list(graph.adj):
			for dest in list(graph.adj[orig]):
				del graph.adj[orig][dest]
		for node in list(graph.node):
			del graph.node[node]
		for stat in set(graph.graph) - {"name", "units"}:
			del graph.graph[stat]
		branch, turn, tick = self._nbtt()
		self.query.graphs_insert(name, branch, turn, tick, "Deleted")
		self._graph_cache.store(name, branch, turn, tick, None)
		self._graph_cache.keycache.clear()
		if hasattr(self, "_worker_processes"):
			self._call_every_subprocess("_del_character", name)

	def _is_thing(self, character: Key, node: Key) -> bool:
		return self._things_cache.contains_entity(
			character, node, *self._btt()
		)

	def _set_thing_loc(self, character: Key, node: Key, loc: Key) -> None:
		branch, turn, tick = self._nbtt()
		# make sure the location really exists now
		if loc is not None:
			self._nodes_cache.retrieve(character, loc, branch, turn, tick)
		self._things_cache.store(character, node, branch, turn, tick, loc)
		self.query.set_thing_loc(character, node, branch, turn, tick, loc)

	def _snap_keyframe_de_novo(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		universal = dict(self.universal.items())
		self._universal_cache.set_keyframe(branch, turn, tick, universal)
		all_graphs = {
			graph: self._graph_cache.retrieve(graph, branch, turn, tick)
			for graph in self._graph_cache.iter_keys(branch, turn, tick)
		}
		self._graph_cache.set_keyframe(branch, turn, tick, all_graphs)
		for char in all_graphs:
			char_kf = {}
			for graph in self._unitness_cache.iter_keys(
				char, branch, turn, tick
			):
				char_kf[graph] = {
					unit: self._unitness_cache.retrieve(
						char, graph, unit, branch, turn, tick
					)
					for unit in self._unitness_cache.iter_keys(
						char, graph, branch, turn, tick
					)
				}

			self._unitness_cache.set_keyframe(
				char, branch, turn, tick, char_kf
			)
		rbnames = list(self._rulebooks_cache.iter_keys(branch, turn, tick))
		rbs = {}
		for rbname in rbnames:
			try:
				rbs[rbname] = self._rulebooks_cache.retrieve(
					rbname, branch, turn, tick
				)
			except KeyError:
				rbs[rbname] = (tuple(), 0.0)
		self._rulebooks_cache.set_keyframe(branch, turn, tick, rbs)
		rulenames = list(self._rules_cache)
		trigs = {}
		preqs = {}
		acts = {}
		nbrs = {}
		bigs = {}
		for rule in rulenames:
			try:
				trigs[rule] = self._triggers_cache.retrieve(
					rule, branch, turn, tick
				)
			except KeyError:
				trigs[rule] = tuple()
			try:
				preqs[rule] = self._prereqs_cache.retrieve(
					rule, branch, turn, tick
				)
			except KeyError:
				preqs[rule] = tuple()
			try:
				acts[rule] = self._actions_cache.retrieve(
					rule, branch, turn, tick
				)
			except KeyError:
				acts[rule] = tuple()
			try:
				nbrs[rule] = self._neighborhoods_cache.retrieve(
					rule, branch, turn, tick
				)
			except KeyError:
				nbrs[rule] = None
			try:
				bigs[rule] = self._rule_bigness_cache.retrieve(
					rule, branch, turn, tick
				)
			except KeyError:
				bigs[rule] = False
		self._triggers_cache.set_keyframe(branch, turn, tick, trigs)
		self._prereqs_cache.set_keyframe(branch, turn, tick, preqs)
		self._actions_cache.set_keyframe(branch, turn, tick, acts)
		self._neighborhoods_cache.set_keyframe(branch, turn, tick, nbrs)
		self._rule_bigness_cache.set_keyframe(branch, turn, tick, bigs)
		for charname in all_graphs:
			locs = {}
			conts_mut = {}
			for thingname in self._things_cache.iter_keys(
				charname, branch, turn, tick
			):
				try:
					locname = self._things_cache.retrieve(
						charname, thingname, branch, turn, tick
					)
				except KeyError:
					locname = None
				locs[thingname] = locname
				if locname in conts_mut:
					conts_mut[locname].add(thingname)
				else:
					conts_mut[locname] = {thingname}
			try:
				units = self._graph_val_cache.retrieve(
					charname, "units", branch, turn, tick
				)
			except KeyError:
				units = {}
			conts = {k: frozenset(v) for (k, v) in conts_mut.items()}
			self._things_cache.set_keyframe(charname, branch, turn, tick, locs)
			self._node_contents_cache.set_keyframe(
				charname, branch, turn, tick, conts
			)
			self._unitness_cache.set_keyframe(
				charname, branch, turn, tick, units
			)
		for rbcache in (
			self._characters_rulebooks_cache,
			self._units_rulebooks_cache,
			self._characters_things_rulebooks_cache,
			self._characters_places_rulebooks_cache,
			self._characters_portals_rulebooks_cache,
		):
			kf = {
				ch: rbcache.retrieve(ch, branch, turn, tick)
				for ch in rbcache.iter_entities(branch, turn, tick)
			}
			rbcache.set_keyframe(branch, turn, tick, kf)
		self.query.keyframe_extension_insert(
			branch,
			turn,
			tick,
			universal,
			{
				"triggers": trigs,
				"prereqs": preqs,
				"actions": acts,
				"neighborhood": nbrs,
				"big": bigs,
			},
			rbs,
		)
		kfd = self._keyframes_dict
		self._keyframes_times.add((branch, turn, tick))
		self._keyframes_loaded.add((branch, turn, tick))
		inskf = self.query.keyframe_graph_insert
		self.query.keyframe_insert(branch, turn, tick)
		nrbcache = self._nodes_rulebooks_cache
		porbcache = self._portals_rulebooks_cache
		for graphn in all_graphs:
			graph = self.graph[graphn]
			nodes = graph._nodes_state()
			edges = graph._edges_state()
			val = graph._val_state()
			nrbkf = {
				node: nrbcache.retrieve(graphn, node, branch, turn, tick)
				for node in nodes
			}
			for node, rb in nrbkf.items():
				nodes[node]["rulebook"] = rb
			nrbcache.set_keyframe(
				(graphn,),
				branch,
				turn,
				tick,
				nrbkf,
			)
			porbkf = {
				orig: {
					dest: porbcache.retrieve(
						graphn, orig, dest, branch, turn, tick
					)
					for dest in edges[orig]
				}
				for orig in edges
			}
			for orig, dests in porbkf.items():
				for dest, rb in dests.items():
					edges[orig][dest]["rulebook"] = rb
			porbcache.set_keyframe(
				graphn,
				branch,
				turn,
				tick,
				porbkf,
			)
			inskf(graphn, branch, turn, tick, nodes, edges, val)
		if branch not in kfd:
			kfd[branch] = {
				turn: {
					tick,
				}
			}
		elif turn not in kfd[branch]:
			kfd[branch][turn] = {
				tick,
			}
		else:
			kfd[branch][turn].add(tick)

	def _snap_keyframe_de_novo_graph(
		self,
		graph: Key,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		nodes: NodeValDict,
		edges: EdgeValDict,
		graph_val: StatDict,
	) -> None:
		for rb_kf_type, rb_kf_cache in [
			("character_rulebook", self._characters_rulebooks_cache),
			("unit_rulebook", self._units_rulebooks_cache),
			(
				"character_thing_rulebook",
				self._characters_things_rulebooks_cache,
			),
			(
				"character_place_rulebook",
				self._characters_places_rulebooks_cache,
			),
			(
				"character_portal_rulebook",
				self._characters_portals_rulebooks_cache,
			),
		]:
			try:
				kf = rb_kf_cache.get_keyframe(branch, turn, tick)
			except KeyError:
				kf = {}
			kf[graph] = graph_val.pop(rb_kf_type, (rb_kf_type, graph))
			rb_kf_cache.set_keyframe(branch, turn, tick, kf)
		self._unitness_cache.set_keyframe(
			graph, branch, turn, tick, graph_val.pop("units", {})
		)
		node_rb_kf = {}
		locs_kf = {}
		conts_kf = {}
		for node, val in nodes.items():
			node_rb_kf[node] = val.pop("rulebook", (graph, node))
			if "location" not in val:
				continue
			locs_kf[node] = location = val["location"]
			if location in conts_kf:
				conts_kf[location].add(node)
			else:
				conts_kf[location] = {node}
		self._nodes_rulebooks_cache.set_keyframe(
			graph, branch, turn, tick, node_rb_kf
		)
		self._things_cache.set_keyframe(graph, branch, turn, tick, locs_kf)
		self._node_contents_cache.set_keyframe(
			graph,
			branch,
			turn,
			tick,
			{n: frozenset(conts) for (n, conts) in conts_kf.items()},
		)
		port_rb_kf = {}
		for orig, dests in edges.items():
			if not dests:
				continue
			port_rb_kf[orig] = rbs = {}
			for dest, port in dests.items():
				rbs[dest] = port.pop("rulebook", (graph, orig, dest))
		self._portals_rulebooks_cache.set_keyframe(
			graph,
			branch,
			turn,
			tick,
			port_rb_kf,
		)
		try:
			graphs_keyframe = self._graph_cache.get_keyframe(
				branch, turn, tick
			)
		except KeyframeError:
			graphs_keyframe = {
				g: "DiGraph"
				for g in self._graph_cache.iter_keys(branch, turn, tick)
			}
		graphs_keyframe[graph] = "DiGraph"
		self._graph_cache.set_keyframe(branch, turn, tick, graphs_keyframe)
		self._graph_cache.keycache.clear()
		self._nodes_cache.set_keyframe(
			graph, branch, turn, tick, {node: True for node in nodes}
		)
		self._node_val_cache.set_keyframe(graph, branch, turn, tick, nodes)
		self._edges_cache.set_keyframe(
			graph,
			branch,
			turn,
			tick,
			{
				orig: {dest: True for dest in edges[orig]}
				for orig in edges
				if edges[orig]
			},
		)
		self._edge_val_cache.set_keyframe(graph, branch, turn, tick, edges)
		self._graph_val_cache.set_keyframe(
			graph, branch, turn, tick, graph_val
		)
		if (branch, turn, tick) not in self._keyframes_times:
			self._keyframes_times.add((branch, turn, tick))
			self._keyframes_loaded.add((branch, turn, tick))
			if branch in self._keyframes_dict:
				turns = self._keyframes_dict[branch]
				if turn in turns:
					turns[turn].add(tick)
				else:
					turns[turn] = {tick}
			else:
				self._keyframes_dict[branch] = {turn: {tick}}
		assert (
			(graph,) in self._things_cache.keyframe
			and branch in self._things_cache.keyframe[graph,]
			and turn in self._things_cache.keyframe[graph,][branch]
			and tick in self._things_cache.keyframe[graph,][branch][turn]
		)

	def flush(self) -> None:
		"""Write pending changes to disk.

		You can set a ``flush_interval`` when you instantiate ``Engine``
		to call this every so many turns. However, this may cause your game to
		hitch up sometimes, so it's better to call ``flush`` when you know the
		player won't be running the simulation for a while.

		"""
		turn_end = self._turn_end
		set_turn = self.query.set_turn
		for (branch, turn), plan_end_tick in self._turn_end_plan.items():
			set_turn(branch, turn, turn_end[branch, turn], plan_end_tick)
		set_branch = self.query.set_branch
		for branch, (
			parent,
			turn_start,
			tick_start,
			turn_end,
			tick_end,
		) in self._branches_d.items():
			set_branch(
				branch, parent, turn_start, tick_start, turn_end, tick_end
			)
		self.query.flush()

	@world_locked
	def commit(self, unload=True) -> None:
		"""Write the state of all graphs and commit the transaction.

		Also saves the current branch, turn, and tick.

		Call with ``unload=False`` if you want to keep the written state in memory.

		"""
		self.query.globl["branch"] = self._obranch
		self.query.globl["turn"] = self._oturn
		self.query.globl["tick"] = self._otick
		self.flush()
		self.query.commit()
		if unload:
			self.unload()

	def turns_when(self, qry: Query, mid_turn=False) -> QueryResult | set:
		"""Return the turns when the query held true

		Only the state of the world at the end of the turn is considered.
		To include turns where the query held true at some tick, but
		became false, set ``mid_turn=True``

		:arg qry: a Query, likely constructed by comparing the result
				  of a call to an entity's ``historical`` method with
				  the output of ``self.alias(..)`` or another
				  ``historical(..)``

		"""
		unpack = self.unpack
		end = self._branch_end()[0] + 1

		def unpack_data_mid(data):
			return [
				((turn_from, tick_from), (turn_to, tick_to), unpack(v))
				for (turn_from, tick_from, turn_to, tick_to, v) in data
			]

		def unpack_data_end(data):
			return [
				(turn_from, turn_to, unpack(v))
				for (turn_from, _, turn_to, _, v) in data
			]

		if not isinstance(qry, ComparisonQuery):
			if not isinstance(qry, CompoundQuery):
				raise TypeError("Unsupported query type: " + repr(type(qry)))
			return CombinedQueryResult(
				self.turns_when(qry.leftside, mid_turn),
				self.turns_when(qry.rightside, mid_turn),
				qry.oper,
			)
		self.flush()
		branches = list({branch for branch, _, _ in self._iter_parent_btt()})
		left = qry.leftside
		right = qry.rightside
		if isinstance(left, StatusAlias) and isinstance(right, StatusAlias):
			left_sel = _make_side_sel(
				left.entity, left.stat, branches, self.pack, mid_turn
			)
			right_sel = _make_side_sel(
				right.entity, right.stat, branches, self.pack, mid_turn
			)
			left_data = self.query.execute(left_sel)
			right_data = self.query.execute(right_sel)
			if mid_turn:
				return QueryResultMidTurn(
					unpack_data_mid(left_data),
					unpack_data_mid(right_data),
					qry.oper,
					end,
				)
			else:
				return QueryResultEndTurn(
					unpack_data_end(left_data),
					unpack_data_end(right_data),
					qry.oper,
					end,
				)
		elif isinstance(left, StatusAlias):
			left_sel = _make_side_sel(
				left.entity, left.stat, branches, self.pack, mid_turn
			)
			left_data = self.query.execute(left_sel)
			if mid_turn:
				return QueryResultMidTurn(
					unpack_data_mid(left_data),
					[(0, 0, None, None, right)],
					qry.oper,
					end,
				)
			else:
				return QueryResultEndTurn(
					unpack_data_end(left_data),
					[(0, None, right)],
					qry.oper,
					end,
				)
		elif isinstance(right, StatusAlias):
			right_sel = _make_side_sel(
				right.entity, right.stat, branches, self.pack, mid_turn
			)
			right_data = self.query.execute(right_sel)
			if mid_turn:
				return QueryResultMidTurn(
					[(0, 0, None, None, left)],
					unpack_data_mid(right_data),
					qry.oper,
					end,
				)
			else:
				return QueryResultEndTurn(
					[(0, None, left)],
					unpack_data_end(right_data),
					qry.oper,
					end,
				)
		else:
			if qry.oper(left, right):
				return set(range(0, self.turn))
			else:
				return set()

	def _node_contents(self, character: Key, node: Key) -> set:
		return self._node_contents_cache.retrieve(
			character, node, *self._btt()
		)

	def apply_choices(
		self, choices: list[dict], dry_run=False, perfectionist=False
	) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
		"""Validate changes a player wants to make, and apply if acceptable.

		Argument ``choices`` is a list of dictionaries, of which each must
		have values for ``"entity"`` (a lisien entity) and ``"changes"``
		-- the later being a list of lists of pairs. Each change list
		is applied on a successive turn, and each pair ``(key, value)``
		sets a key on the entity to a value on that turn.

		Returns a pair of lists containing acceptance and rejection messages,
		which the UI may present as it sees fit. They are always in a pair
		with the change request as the zeroth item. The message may be None
		or a string.

		Validator functions may return only a boolean indicating acceptance.
		If they instead return a pair, the initial boolean indicates
		acceptance and the following item is the message.

		This function will not actually result in any simulation happening.
		It creates a plan. See my ``plan`` context manager for the precise
		meaning of this.

		With ``dry_run=True`` just return the acceptances and rejections
		without really planning anything. With ``perfectionist=True`` apply
		changes if and only if all of them are accepted.

		"""
		schema = self.schema
		todo = defaultdict(list)
		acceptances = []
		rejections = []
		for track in choices:
			entity = track["entity"]
			permissible = schema.entity_permitted(entity)
			if isinstance(permissible, tuple):
				permissible, msg = permissible
			else:
				msg = ""
			if not permissible:
				for turn, changes in enumerate(
					track["changes"], start=self.turn + 1
				):
					rejections.extend(
						((turn, entity, k, v), msg) for (k, v) in changes
					)
				continue
			for turn, changes in enumerate(
				track["changes"], start=self.turn + 1
			):
				for k, v in changes:
					ekv = (entity, k, v)
					parcel = (turn, entity, k, v)
					val = schema.stat_permitted(*parcel)
					if type(val) is tuple:
						accept, message = val
						if accept:
							todo[turn].append(ekv)
							l = acceptances
						else:
							l = rejections
						l.append((parcel, message))
					elif val:
						todo[turn].append(ekv)
						acceptances.append((parcel, None))
					else:
						rejections.append((parcel, None))
		if dry_run or (perfectionist and rejections):
			return acceptances, rejections
		now = self.turn
		with self.plan():
			for turn in sorted(todo):
				self.turn = turn
				for entity, key, value in todo[turn]:
					if isinstance(entity, self.char_cls):
						entity.stat[key] = value
					else:
						entity[key] = value
		self.turn = now
		return acceptances, rejections

	def game_start(self):
		import importlib.machinery
		import importlib.util

		loader = importlib.machinery.SourceFileLoader(
			"game_start", os.path.join(self._prefix, "game_start.py")
		)
		spec = importlib.util.spec_from_loader("game_start", loader)
		game_start = importlib.util.module_from_spec(spec)
		loader.exec_module(game_start)
		game_start.game_start(self)
