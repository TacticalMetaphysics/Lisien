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

import gc
import os
import pickle
import shutil
import signal
import sys
import zlib
from abc import ABC, abstractmethod
from collections import UserDict, defaultdict
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from concurrent.futures import wait as futwait
from contextlib import ContextDecorator, contextmanager
from functools import cached_property, partial, wraps
from io import TextIOWrapper
from itertools import chain, pairwise
from logging import DEBUG, Formatter, Logger, LogRecord, StreamHandler
from operator import itemgetter, lt
from os import PathLike
from queue import Empty, SimpleQueue
from random import Random
from threading import Lock, RLock, Thread
from time import sleep
from types import FunctionType, MethodType, ModuleType
from typing import Any, Callable, Iterable, Iterator, Literal, Optional, Type
from zipfile import ZIP_DEFLATED, ZipFile

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
	BignessCache,
	CharacterPlaceRulesHandledCache,
	CharacterPortalRulesHandledCache,
	CharacterRulesHandledCache,
	CharactersRulebooksCache,
	CharacterThingRulesHandledCache,
	EdgesCache,
	EdgeValCache,
	EntitylessCache,
	FuncListCache,
	GraphCache,
	GraphValCache,
	NeighborhoodsCache,
	NodeContentsCache,
	NodeRulesHandledCache,
	NodesCache,
	NodesRulebooksCache,
	NodeValCache,
	PickyDefaultDict,
	PortalRulesHandledCache,
	PortalsRulebooksCache,
	RulebooksCache,
	StructuredDefaultDict,
	ThingsCache,
	TurnEndDict,
	TurnEndPlanDict,
	UnitnessCache,
	UnitRulesHandledCache,
)
from .character import Character
from .collections import (
	ChangeTrackingDict,
	CharacterMapping,
	FunctionStore,
	StringStore,
	UniversalMapping,
)
from .db import (
	NullDatabaseConnector,
	ParquetDatabaseConnector,
	SQLAlchemyDatabaseConnector,
)
from .exc import (
	GraphNameError,
	HistoricKeyError,
	KeyframeError,
	OutOfTimelineError,
	RedundantRuleError,
)
from .exporter import Exporter
from .facade import CharacterFacade
from .importer import xml_to_pqdb, xml_to_sqlite
from .node import Place, Thing
from .portal import Portal
from .proxy import worker_subprocess
from .query import (
	CharacterStatAccessor,
	CombinedQueryResult,
	ComparisonQuery,
	CompoundQuery,
	EntityStatAccessor,
	Query,
	QueryResult,
	QueryResultEndTurn,
	QueryResultMidTurn,
	_make_side_sel,
)
from .rule import AllRuleBooks, AllRules, Rule
from .types import (
	Branch,
	CharacterRulebookTypeStr,
	CharDelta,
	CharName,
	DeltaDict,
	DiGraph,
	EdgesDict,
	EdgeValDict,
	EntityKey,
	GraphEdgesKeyframe,
	GraphEdgeValKeyframe,
	GraphNodesKeyframe,
	GraphNodeValKeyframe,
	GraphValKeyframe,
	Key,
	Keyframe,
	KeyframeTuple,
	LinearTime,
	NodeName,
	NodesDict,
	NodeValDict,
	Plan,
	RuleBig,
	RulebookName,
	RulebookPriority,
	RuleFuncName,
	RuleName,
	RuleNeighborhood,
	SlightlyPackedDeltaType,
	Stat,
	StatDict,
	Tick,
	Time,
	Turn,
	UniversalKey,
	Value,
)
from .util import (
	ACTIONS,
	BIG,
	EDGE_VAL,
	EDGES,
	ELLIPSIS,
	ETERNAL,
	FALSE,
	NEIGHBORHOOD,
	NODE_VAL,
	NODES,
	PREREQS,
	RULEBOOK,
	RULEBOOKS,
	RULES,
	TRIGGERS,
	TRUE,
	UNITS,
	UNIVERSAL,
	AbstractBookmarkMapping,
	AbstractCharacter,
	AbstractEngine,
	SizedDict,
	TimeSignalDescriptor,
	fake_submit,
	garbage,
	normalize_layout,
	sort_set,
	timer,
	world_locked,
)
from .window import (
	LinearTimeListDict,
	WindowDict,
	update_backward_window,
	update_window,
)
from .wrap import OrderlyFrozenSet, OrderlySet

SUBPROCESS_TIMEOUT = 30
if "LISIEN_SUBPROCESS_TIMEOUT" in os.environ:
	try:
		SUBPROCESS_TIMEOUT = int(os.environ["LISIEN_SUBPROCESS_TIMEOUT"])
	except ValueError:
		SUBPROCESS_TIMEOUT = None
KILL_SUBPROCESS = False
if "LISIEN_KILL_SUBPROCESS" in os.environ:
	KILL_SUBPROCESS = bool(os.environ["LISIEN_KILL_SUBPROCESS"])


class InnerStopIteration(StopIteration):
	pass


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
			engine._extend_branch(start_branch, Turn(start_turn + 1), Tick(0))
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
		with (
			timer("seconds to run the rules engine", engine.debug),
			engine.advancing(),
		):
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
		if results:
			engine.universal["last_result"] = results
			engine.universal["last_result_idx"] = 0
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


class WorkerFormatter(Formatter):
	def formatMessage(self, record: LogRecord) -> str:
		if not hasattr(record, "worker_idx"):
			raise RuntimeError(
				"WorkerFormatter received a LogRecord from a non-worker",
				record,
			)
		return f"worker {getattr(record, 'worker_idx', '???')}: {super().formatMessage(record)}"


class BookmarkMapping(AbstractBookmarkMapping, UserDict):
	"""Points in time you might want to return to.

	Call this with a valid key, like a string, to place a bookmark at the
	current time, or, if there is already a bookmark by the given name,
	then return to it.

	The times stored here are triples of (branch, turn, tick). If you wish,
	you can set the engine's `time` property to one of those triples yourself,
	and time travel all the same.

	"""

	def __init__(self, eng: Engine):
		self.eng = eng
		super().__init__(eng.query.bookmark_items())

	def __setitem__(self, key, value):
		if not (
			isinstance(value, tuple)
			and len(value) == 3
			and isinstance(value[0], str)
			and isinstance(value[1], int)
			and isinstance(value[2], int)
		):
			raise TypeError("Not a valid time", value)
		super().__setitem__(key, value)
		self.eng.query.set_bookmark(key, *value)

	def __delitem__(self, key):
		super().__delitem__(key)
		self.eng.query.del_bookmark(key)

	def __call__(self, key: Key):
		if key in self:
			self.eng.time = self[key]
		else:
			self[key] = tuple(self.eng.time)


class Engine(AbstractEngine, Executor):
	"""Lisien, the Life Simulator Engine.

	:param prefix: directory containing the simulation and its code;
		defaults to the working directory. If ``None``, Lisien won't save
		any rules code to disk, and won't save world data unless you supply
		:param connect_string:. This is the only positional argument;
		all others require keywords.
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
	:param trunk: the string name of the branch to start games from. Defaults
		to "trunk" if not set in some prior session. You should only change
		this if your game generated a new initial world state for a new
		playthrough.
	:param connect_string: a URL for a database to connect to. Leave
		it as the default, ``None``, to use the ParquetDB database in the
		``prefix``. This uses SQLAlchemy's URL structure:
		https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls
	:param connect_args: Dictionary of keyword arguments for the
		database connection. Only meaningful when combined with
		``connect_string``. For details, see:
		https://docs.sqlalchemy.org/en/20/core/engines.html#custom-dbapi-args
	:param schema: A Schema class that determines which changes to allow to
		the world; used when a player should not be able to change just
		anything. Defaults to :class:`NullSchema`, which allows all changes.
	:param flush_interval: Lisien will put pending changes into the database
		transaction every ``flush_interval`` turns. If ``None``, only flush
		on commit. Default ``None``.
	:param keyframe_interval: How many records to let through before automatically
		snapping a keyframe, default ``1000``. If ``None``, you'll need
		to call ``snap_keyframe`` yourself.
	:param commit_interval: Lisien will commit changes to disk every
		``commit_interval`` turns. If ``None`` (the default), only commit
		on close or manual call to ``commit``.
	:param random_seed: A number to initialize the randomizer. Lisien saves
		the state of the randomizer, so you only need to supply this when
		starting a game for the first time.
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
		if you use something other than Lisien's rules engine for game
		logic.
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
	trigger: FunctionStore
	prereq: FunctionStore
	action: FunctionStore
	function: FunctionStore
	method: FunctionStore

	@property
	def eternal(self):
		return self.query.eternal

	@property
	def branch(self) -> Branch:
		return self._obranch

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
		if v != self.trunk and v in self.branches():
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
	def trunk(self):
		return self.query.eternal["trunk"]

	@trunk.setter
	def trunk(self, branch: Branch) -> None:
		if self.branch != self.trunk or self.turn != 0 or self.tick != 0:
			raise AttributeError("Go to the start of time first")
		if (
			branch in self.branches()
			and self.branch_parent(branch) is not None
		):
			raise AttributeError("Not a trunk branch")
		then = self._btt()
		self.query.eternal["trunk"] = self.branch = branch
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
		self._extend_branch(branch, v, tick)
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
		if self.turn == self.branch_end_turn():
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
	def bookmark(self) -> BookmarkMapping:
		return BookmarkMapping(self)

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
			[tuple[CharName, NodeName, NodeName, Branch, Turn, Tick]],
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
			[CharName, NodeName, NodeName, Branch, Turn, Tick, bool], None
		],
		Callable[
			[CharName, NodeName, NodeName, Branch, Turn, Tick, Any], None
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
		SizedDict,
		Callable[[tuple], Any],
		Callable[[], tuple[Branch, Turn, Tick]],
		Callable[[Character, NodeName], Thing | Place],
	]:
		return (
			self._node_objs,
			self._nodes_cache._base_retrieve,
			self._btt,
			self._make_node,
		)

	@cached_property
	def _get_edge_stuff(
		self,
	) -> tuple[
		SizedDict,
		Callable[[CharName, NodeName, NodeName], bool],
		Callable[[Character, NodeName, NodeName], Portal],
	]:
		return self._edge_objs, self._edge_exists, self._make_edge

	@cached_property
	def _childbranch(self) -> dict[Branch, set[Branch]]:
		"""Immediate children of a branch"""
		return defaultdict(set)

	@cached_property
	def _branches_d(
		self,
	) -> ChangeTrackingDict[
		Branch, tuple[Branch | None, Turn, Tick, Turn, Tick]
	]:
		"""Parent, start time, and end time of each branch. Plans not included."""
		return ChangeTrackingDict()

	@cached_property
	def _branch_parents(self) -> dict[Branch, set[Branch]]:
		"""Parents of a branch at any remove"""
		return defaultdict(set)

	@cached_property
	def _turn_end(self) -> TurnEndDict[tuple[Branch, Turn], Tick]:
		return TurnEndDict(self)

	@cached_property
	def _turn_end_plan(self) -> TurnEndPlanDict[tuple[Branch, Turn], Tick]:
		return TurnEndPlanDict(self)

	@cached_property
	def _graph_objs(self) -> dict[CharName, AbstractCharacter]:
		return {}

	@cached_property
	def _plans(self) -> dict[Plan, Time]:
		return {}

	@cached_property
	def _branches_plans(self) -> dict[Branch, set[Plan]]:
		return defaultdict(set)

	@cached_property
	def _plan_ticks(
		self,
	) -> dict[Plan, dict[Branch, LinearTimeListDict]]:
		return defaultdict(lambda: defaultdict(LinearTimeListDict))

	@cached_property
	def _time_plan(self) -> dict[Time, Plan]:
		return {}

	@cached_property
	def _graph_cache(self) -> GraphCache:
		return GraphCache(self, name="graph cache")

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
		ret.deldb = self.query.things_del_time
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
	def _rulebooks_cache(self) -> RulebooksCache:
		ret = RulebooksCache(self, name="rulebooks cache")
		ret.setdb = self.query.rulebook_set
		return ret

	@cached_property
	def _characters_rulebooks_cache(self) -> CharactersRulebooksCache:
		return CharactersRulebooksCache(self, name="character_rulebook")

	@cached_property
	def _units_rulebooks_cache(self) -> CharactersRulebooksCache:
		return CharactersRulebooksCache(self, name="unit_ulebook")

	@cached_property
	def _characters_things_rulebooks_cache(self) -> CharactersRulebooksCache:
		return CharactersRulebooksCache(self, name="character_thing_rulebook")

	@cached_property
	def _characters_places_rulebooks_cache(self) -> CharactersRulebooksCache:
		return CharactersRulebooksCache(self, name="character_place_rulebook")

	@cached_property
	def _characters_portals_rulebooks_cache(self) -> CharactersRulebooksCache:
		return CharactersRulebooksCache(
			self, name="character_portals_rulebook"
		)

	@cached_property
	def _nodes_rulebooks_cache(self) -> NodesRulebooksCache:
		return NodesRulebooksCache(self, name="nodes rulebooks cache")

	@cached_property
	def _portals_rulebooks_cache(self) -> PortalsRulebooksCache:
		return PortalsRulebooksCache(self, name="portals rulebooks cache")

	@cached_property
	def _triggers_cache(self) -> FuncListCache:
		return FuncListCache(self, name="triggers cache")

	@cached_property
	def _prereqs_cache(self) -> FuncListCache:
		return FuncListCache(self, name="prereqs cache")

	@cached_property
	def _actions_cache(self) -> FuncListCache:
		return FuncListCache(self, name="actions cache")

	@cached_property
	def _neighborhoods_cache(self) -> NeighborhoodsCache:
		return NeighborhoodsCache(self, name="neighborhoods cache")

	@cached_property
	def _rule_bigness_cache(self) -> BignessCache:
		return BignessCache(self, name="rule bigness cache")

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
	def _keyframes_dict(self) -> dict[Branch, WindowDict[Turn, set[Tick]]]:
		return PickyDefaultDict(WindowDict)

	@cached_property
	def _keyframes_times(self) -> set[Time]:
		return set()

	@cached_property
	def _keyframes_loaded(self) -> set[Time]:
		return set()

	@cached_property
	def _caches(self) -> tuple[Cache, ...]:
		return (
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
		)

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
			this_plan = plan_ticks[last_plan][branch]
			if turn in this_plan:
				ticks = this_plan[turn]
				ticks.append(tick)
				this_plan[turn] = ticks
			else:
				this_plan[turn] = [tick]
			self.query.plans_insert(last_plan, branch, turn, tick)
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
			for dest, val in dests.items():
				hash = blake2b(pack(orig))
				hash.update(pack(dest))
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

	def _get_node(
		self, graph: CharName | Character, node: NodeName
	) -> place_cls | thing_cls:
		node_objs, retrieve, btt, make_node = self._get_node_stuff
		if hasattr(graph, "name"):
			graphn = graph.name
		else:
			graphn = graph
			graph = self.character[graphn]
		key = (graphn, node)
		if key in node_objs:
			ret = node_objs[key]
			if ret._validate_node_type():
				return ret
			else:
				del node_objs[key]
		ex = retrieve((graphn, node, *btt()))
		if isinstance(ex, Exception):
			raise ex
		if not ex:
			raise KeyError("No such node: {} in {}".format(node, graphn))
		ret = make_node(graph, node)
		node_objs[key] = ret
		return ret

	def _get_edge(
		self, graph: Character | CharName, orig: NodeName, dest: NodeName
	):
		edge_objs, edge_exists, make_edge = self._get_edge_stuff
		if type(graph) is str:
			graphn = graph
			graph = self.character[graphn]
		else:
			graphn = graph.name
		key = (graphn, orig, dest)
		if key in edge_objs:
			return edge_objs[key]
		if not edge_exists(graphn, orig, dest):
			raise KeyError(
				"No such edge: {}->{} in {}".format(orig, dest, graphn)
			)
		ret = make_edge(graph, orig, dest)
		edge_objs[key] = ret
		return ret

	def plan(self, reset: bool = True) -> PlanningContext:
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
			times = plan_ticks[plan_id][branch_from].iter_times()
			for turn, tick in times:
				if (turn_from, tick_from) <= (turn, tick):
					break
			else:
				continue
			self._last_plan += 1
			plans[self._last_plan] = branch, turn, tick
			if (
				branch,
				turn,
			) not in turn_end_plan or tick > turn_end_plan[branch, turn]:
				turn_end_plan[branch, turn] = tick
			for turn, tick in times:
				ticks = plan_ticks[self._last_plan][branch][turn]
				ticks.append(tick)
				plan_ticks[self._last_plan][branch][turn] = ticks
				self.query.plans_insert(self._last_plan, branch, turn, tick)
				for cache in self._caches:
					if not hasattr(cache, "settings"):
						continue
					try:
						data = cache.settings[branch_from][turn][tick]
					except KeyError:
						continue
					value = data[-1]
					key = data[:-1]
					if key[0] is None:
						key = key[1:]
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
		plan_ticks = self._plan_ticks[plan]
		for branch in plan_ticks:
			plan_times = plan_ticks[branch].iter_times()
			for start_turn, start_tick in plan_times:
				if self._branch_end(branch) < (start_turn, start_tick):
					break
			else:
				continue
			time_plan = self._time_plan
			for trn, tck in ((start_turn, start_tick), *plan_times):
				for cache in self._caches:
					if hasattr(cache, "discard"):
						cache.discard(branch, trn, tck)
					if hasattr(cache, "deldb"):
						cache.deldb(branch, trn, tck)
				plan_ticks[branch][trn].remove(tck)
				if not plan_ticks[branch][trn]:
					del plan_ticks[branch][trn]
				if (branch, trn, tck) in time_plan:
					del time_plan[branch, trn, tck]
			# Delete keyframes on or after the start of the plan
			kf2del = []
			for r, ts in self._keyframes_dict[branch].items():
				if r > start_turn:
					kf2del.extend((r, t) for t in ts)
				elif r == start_turn:
					kf2del.extend((r, t) for t in ts if t >= start_tick)
			for kf_turn, kf_tick in kf2del:
				self._delete_keyframe(branch, kf_turn, kf_tick)

	def _delete_keyframe(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		if (branch, turn, tick) not in self._keyframes_times:
			raise KeyframeError("No keyframe at that time", branch, turn, tick)
		self.query.delete_keyframe(branch, turn, tick)
		self._keyframes_times.remove((branch, turn, tick))
		self._keyframes_loaded.remove((branch, turn, tick))
		self._keyframes_dict[branch][turn].remove(tick)
		if not self._keyframes_dict[branch][turn]:
			del self._keyframes_dict[branch][turn]
		if not self._keyframes_dict[branch]:
			del self._keyframes_dict[branch]
		for cache in self._caches:
			if hasattr(cache, "delete_keyframe"):
				cache.discard_keyframe(branch, turn, tick)
			if hasattr(cache, "shallowest"):
				cache.shallowest.clear()

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
		__: Turn,
		___: Tick,
		_: None,
		graph: CharName,
		val: Value,
	) -> None:
		"""Change a delta to say that a graph was deleted or not"""
		if val in (..., None, "Deleted"):
			delta[graph] = ...
		elif graph not in delta or delta[graph] is ...:
			# If the graph was *created* within our window,
			# include its whole initial keyframe
			delta[graph] = {
				"character_rulebook": ("character_rulebook", graph),
				"unit_rulebook": ("unit_rulebook", graph),
				"character_thing_rulebook": (
					"character_thing_rulebook",
					graph,
				),
				"character_place_rulebook": (
					"character_place_rulebook",
					graph,
				),
				"character_portal_rulebook": (
					"character_portal_rulebook",
					graph,
				),
			}
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
					or preset[b][r][t][2] is ...
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
		self.debug(
			"Getting delta in branch %s from %d, %d to %d, %d",
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)

		setgraph = partial(
			self._set_graph_in_delta,
			branch,
			turn_from,
			tick_from,
			turn_to,
			tick_to,
		)

		def setgraphval(
			delta: DeltaDict,
			_: Turn,
			__: Tick,
			graph: CharName,
			key: Stat,
			val: Value,
		) -> None:
			"""Change a delta to say that a graph stat was set to a certain value"""
			if graph not in delta:
				delta[graph] = {}
			if delta[graph] is not ...:
				graph_stats: CharDelta = delta[graph]
				graph_stats[key] = val

		def setnode(
			delta: DeltaDict,
			turn: Turn,
			tick: Tick,
			graph: CharName,
			node: NodeName,
			exists: bool | None,
		) -> None:
			"""Change a delta to say that a node was created or deleted"""
			if graph not in delta:
				delta[graph] = {}
			if delta[graph] is ...:
				return
			graph_nodes: NodesDict = delta[graph].setdefault("nodes", {})
			graph_nodes[node] = bool(exists)
			try:
				contents: frozenset[NodeName] = (
					self._node_contents_cache.retrieve(
						graph, node, branch, turn, tick
					)
				)
			except KeyError:
				return
			if not contents:
				return
			for thing in contents:
				graph_nodes[thing] = bool(exists)

		def setnodeval(
			delta: DeltaDict,
			_: Turn,
			__: Tick,
			graph: CharName,
			node: NodeName,
			key: Stat,
			value: Value,
		) -> None:
			"""Change a delta to say that a node stat was set to a certain value"""
			if graph not in delta:
				delta[graph] = {}
			if delta[graph] is ...:
				return
			if (
				"nodes" in delta[graph]
				and node in delta[graph]["nodes"]
				and not delta[graph]["nodes"][node]
			):
				return
			graphstats: CharDelta = delta[graph]
			nodestats: NodeValDict = graphstats.setdefault("node_val", {})
			if node in nodestats:
				nodestats[node][key] = value
			else:
				nodestats[node] = {key: value}

		def setedge(
			delta: DeltaDict,
			is_multigraph: callable,
			_: Turn,
			__: Tick,
			graph: CharName,
			orig: NodeName,
			dest: NodeName,
			exists: bool | None,
		) -> None:
			"""Change a delta to say that an edge was created or deleted"""
			if graph not in delta:
				delta[graph] = {}
			if delta[graph] is ...:
				return
			graphstat: CharDelta = delta[graph]
			if "edges" in graphstat:
				es: EdgesDict = graphstat["edges"]
				if orig in es:
					es[orig][dest] = bool(exists)
				else:
					es[orig] = {dest: bool(exists)}
			else:
				graphstat["edges"] = {orig: {dest: bool(exists)}}

		def setedgeval(
			delta: DeltaDict,
			is_multigraph: callable,
			turn: Turn,
			tick: Tick,
			graph: CharName,
			orig: NodeName,
			dest: NodeName,
			key: Stat,
			value: Value,
		) -> None:
			if graph not in delta:
				delta[graph] = {}
			if delta[graph] is ...:
				return
			graphstat: CharDelta = delta[graph]
			if (
				"edges" in graphstat
				and orig in graphstat["edges"]
				and dest in graphstat["edges"][orig]
				and not graphstat["edges"][orig][dest]
			):
				return
			if "edge_val" in graphstat:
				evs: EdgeValDict = graphstat["edge_val"]
				if orig in evs:
					if dest in evs[orig]:
						evs[orig][dest][key] = value
					else:
						evs[orig][dest] = {key: value}
				else:
					evs[orig] = {dest: {key: value}}
			else:
				graphstat["edge_val"] = {orig: {dest: {key: value}}}

		if not isinstance(branch, str):
			raise TypeError("branch must be str")
		for arg in (turn_from, tick_from, turn_to, tick_to):
			if not isinstance(arg, int):
				raise TypeError("turn and tick must be int")
		self.load_between(branch, turn_from, tick_from, turn_to, tick_to)
		a: LinearTime = turn_from, tick_from
		b: LinearTime = turn_to, tick_to
		if a == b:
			return {}
		delta = {}
		graph_objs = self._graph_objs

		if a < b:
			updater = partial(
				update_window, turn_from, tick_from, turn_to, tick_to
			)
			attribute = "settings"
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

		def upduniv(
			_: Turn, __: Tick, ___: None, key: UniversalKey, val: Value
		):
			delta.setdefault("universal", {})[key] = val

		if branch in univbranches:
			updater(upduniv, univbranches[branch])

		def updunit(
			_: Turn,
			__: Tick,
			char: CharName,
			graph: CharName,
			node: NodeName,
			is_unit: bool,
		):
			if char in delta and delta[char] is ...:
				return
			delta.setdefault(char, {}).setdefault("units", {}).setdefault(
				graph, {}
			)[node] = bool(is_unit)

		if branch in unitbranches:
			updater(updunit, unitbranches[branch])

		def updthing(
			_: Turn,
			__: Tick,
			char: CharName,
			thing: NodeName,
			loc: NodeName,
		):
			if char in delta and (
				delta[char] is ...
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

		def updrb(
			__: Turn,
			___: Tick,
			_: None,
			rulebook: RulebookName,
			rules: list[RuleName],
		):
			delta.setdefault("rulebooks", {})[rulebook] = rules or []

		if branch in rbbranches:
			updater(updrb, rbbranches[branch])

		def updru(
			key: Literal["triggers", "prereqs", "actions"],
			__: Turn,
			___: Tick,
			_: None,
			rule: RuleName,
			funs: list[RuleFuncName],
		):
			delta.setdefault("rules", {}).setdefault(rule, {})[key] = (
				funs or []
			)

		if branch in trigbranches:
			updater(partial(updru, "triggers"), trigbranches[branch])

		if branch in preqbranches:
			updater(partial(updru, "prereqs"), preqbranches[branch])

		if branch in actbranches:
			updater(partial(updru, "actions"), actbranches[branch])

		def updnbr(
			__: Turn,
			___: Tick,
			_: None,
			rule: RuleName,
			neighborhood: RuleNeighborhood,
		):
			if neighborhood is not None:
				if not isinstance(neighborhood, int):
					raise TypeError(
						"Neighborhood must be int or None", neighborhood
					)
				if neighborhood < 0:
					raise ValueError(
						"Neighborhood must not be negative", neighborhood
					)
			delta.setdefault("rules", {}).setdefault(rule, {})[
				"neighborhood"
			] = neighborhood

		if branch in nbrbranches:
			updater(updnbr, nbrbranches[branch])

		def updbig(__: Turn, ___: Tick, _: None, rule: RuleName, big: RuleBig):
			if big is not None and not isinstance(big, bool):
				raise TypeError("big must be boolean", big)
			delta.setdefault("rules", {}).setdefault(rule, {})["big"] = big

		if branch in bigbranches:
			updater(updbig, bigbranches[branch])

		def updcrb(
			key: CharacterRulebookTypeStr,
			__: Turn,
			___: Tick,
			_: None,
			character: CharName,
			rulebook: RulebookName,
		):
			if character in delta and delta[character] is ...:
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

		def updnoderb(
			_: Turn,
			__: Tick,
			character: CharName,
			node: NodeName,
			rulebook: RulebookName,
		):
			if (character in delta) and (
				delta[character] is ...
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

		def updedgerb(
			_: Turn,
			__: Tick,
			character: CharName,
			orig: NodeName,
			dest: NodeName,
			rulebook: RulebookName | None = None,
		):
			if rulebook is None:
				# It's one of those updates that stores all the rulebooks from
				# some origin. Not relevant to deltas.
				return
			if character in delta and (
				delta[character] is ...
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
		string: StringStore | dict | None = None,
		trigger: FunctionStore | ModuleType | None = None,
		prereq: FunctionStore | ModuleType | None = None,
		action: FunctionStore | ModuleType | None = None,
		function: FunctionStore | ModuleType | None = None,
		method: FunctionStore | ModuleType | None = None,
		trunk: Branch | None = None,
		connect_string: str | None = None,
		connect_args: dict | None = None,
		schema_cls: Type[AbstractSchema] = NullSchema,
		flush_interval: int | None = None,
		keyframe_interval: int | None = 1000,
		commit_interval: int | None = None,
		random_seed: int | None = None,
		clear: bool = False,
		keep_rules_journal: bool = True,
		keyframe_on_close: bool = True,
		enforce_end_of_time: bool = True,
		logger: Logger | None = None,
		workers: int | None = None,
	):
		if workers is None:
			workers = os.cpu_count()
		if prefix:
			os.makedirs(prefix, exist_ok=True)
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
		self._obranch = trunk or "trunk"
		self._otick = self._oturn = 0
		if logger is not None:
			self._logger = logger
		worker_handler = StreamHandler()
		worker_handler.addFilter(lambda rec: hasattr(rec, "worker_idx"))
		worker_handler.setLevel(DEBUG)
		worker_handler.setFormatter(WorkerFormatter())
		self.logger.addHandler(worker_handler)
		self._init_func_stores(
			prefix, function, method, trigger, prereq, action, clear
		)
		self._init_load(
			prefix,
			connect_string,
			connect_args,
			keyframe_interval,
			trunk,
			clear,
		)
		if not self._turn_end or not self._turn_end_plan:
			self.query.set_turn(
				"trunk",
				0,
				self._turn_end.setdefault(("trunk", 0), 0),
				self._turn_end_plan.setdefault(("trunk", 0), 0),
			)
		self._init_random(random_seed)
		self._init_string(prefix, string, clear)
		self._top_uid = 0
		if workers != 0:
			self._start_worker_processes(prefix, workers)

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
		if not hasattr(self, "query"):
			if prefix is None:
				if connect_string is None:
					self.query = NullDatabaseConnector()
				else:
					self.query = SQLAlchemyDatabaseConnector(
						connect_string,
						connect_args or {},
						self.pack,
						self.unpack,
						clear=clear,
					)
			else:
				if not os.path.exists(prefix):
					os.mkdir(prefix)
				if not os.path.isdir(prefix):
					raise FileExistsError("Need a directory")
				if connect_string is None:
					self.query = ParquetDatabaseConnector(
						os.path.join(prefix, "world"),
						self.pack,
						self.unpack,
						clear=clear,
					)
				else:
					self.query = SQLAlchemyDatabaseConnector(
						connect_string,
						connect_args or {},
						self.pack,
						self.unpack,
						clear=clear,
					)

		self.query.keyframe_interval = keyframe_interval
		self._load_keyframe_times()
		if main_branch is not None:
			self.query.eternal["trunk"] = main_branch
		elif "trunk" not in self.query.eternal:
			main_branch = self.query.eternal["trunk"] = "trunk"
		else:
			main_branch = self.query.eternal["trunk"]
		assert main_branch is not None
		assert main_branch == self.query.eternal["trunk"]
		if "branch" in self.query.eternal:
			self._obranch = self.query.eternal["branch"]
		if "turn" in self.query.eternal:
			self._oturn = self.query.eternal["turn"]
		if "tick" in self.query.eternal:
			self._otick = self.query.eternal["tick"]
		for (
			branch,
			parent,
			parent_turn,
			parent_tick,
			end_turn,
			end_tick,
		) in self.query.branches_dump():
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
					Turn(0),
					Tick(0),
					rando_state,
					loading=True,
				)
				self.query.universal_set(
					"rando_state",
					self.branch,
					Turn(0),
					Tick(0),
					rando_state,
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

	def _sync_log_forever(self, q: SimpleQueue[LogRecord]) -> None:
		while not hasattr(self, "_closed"):
			recs: list[LogRecord] = []
			while True:
				try:
					recs.append(q.get())
				except Empty:
					break
			for rec in recs:
				self.logger.handle(rec)
			sleep(0.5)

	def _start_worker_processes(
		self, prefix: str | os.PathLike | None, workers: int
	):
		from multiprocessing import get_context
		from multiprocessing.connection import Connection
		from multiprocessing.process import BaseProcess

		for store in self.stores:
			if hasattr(store, "save"):
				store.save(reimport=False)

		self._trigger_pool = ThreadPoolExecutor()
		self._worker_last_eternal = dict(self.eternal.items())
		initial_payload = self._get_worker_kf_payload()

		self._worker_processes: list[BaseProcess] = []
		wp = self._worker_processes
		self._mp_ctx = ctx = get_context("spawn")
		self._worker_inputs: list[Connection] = []
		wi = self._worker_inputs
		self._worker_outputs: list[Connection] = []
		wo = self._worker_outputs
		self._worker_locks: list[Lock] = []
		wlk = self._worker_locks
		self._worker_log_queues: list[SimpleQueue] = []
		wl = self._worker_log_queues
		self._worker_log_threads: list[Thread] = []
		wlt = self._worker_log_threads
		for i in range(workers):
			inpipe_there, inpipe_here = ctx.Pipe(duplex=False)
			outpipe_here, outpipe_there = ctx.Pipe(duplex=False)
			logq = ctx.SimpleQueue()
			logthread = Thread(
				target=self._sync_log_forever, args=(logq,), daemon=True
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
			proc = ctx.Process(
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
		if hasattr(self.trigger, "connect"):
			self.trigger.connect(self._reimport_trigger_functions)
		if hasattr(self.function, "connect"):
			self.function.connect(self._reimport_worker_functions)
		if hasattr(self.method, "connect"):
			self.method.connect(self._reimport_worker_methods)
		self._setup_fut_manager(workers)

	def _setup_fut_manager(self, workers: int):
		self._workers = workers
		self._worker_updated_btts: list[Time] = [self._btt()] * workers
		self._uid_to_fut: dict[int, Future] = {}
		self._futs_to_start: SimpleQueue[Future] = SimpleQueue()
		self._how_many_futs_running = 0
		self._fut_manager_thread = Thread(
			target=self._manage_futs, daemon=True
		)
		self._fut_manager_thread.start()

	def _start_branch(
		self, parent: Branch, branch: Branch, turn: Turn, tick: Tick
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
		self.query.set_branch(branch, parent, turn, tick, turn, tick)

	def _extend_branch(self, branch: Branch, turn: Turn, tick: Tick) -> None:
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
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rulebooks: bool = True,
		silent: bool = False,
	) -> Keyframe | None:
		"""Load the keyframe if it's not loaded, and return it"""
		if (branch, turn, tick) in self._keyframes_loaded:
			self.debug(
				"Keyframe already loaded, returning %s, %d, %d from memory",
				branch,
				turn,
				tick,
			)
			if silent:
				return
			return self._get_kf(branch, turn, tick, rulebooks=rulebooks)
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
		keyframe_graphs = list(
			self.query.get_all_keyframe_graphs(branch, turn, tick)
		)
		with (
			self.batch()
		):  # so that iter_keys doesn't try fetching the kf we're about to make
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
		self._mark_keyframe_loaded(branch, turn, tick)
		ret = self._get_kf(branch, turn, tick)
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
				dests: dict[NodeName, dict[Stat | Literal["rulebook"], Value]]
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
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		loaded: bool = False,
		with_fork_points: bool = False,
		stoptime: Optional[Time] = None,
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
			if r0 == r1:
				if r0 in kfdb:
					tcks = sorted(kfdb[r0], reverse=True)
					if loaded:
						for tck in tcks:
							if (b0, r0, tck) in kfl:
								yield b0, r0, tck
					else:
						for tck in tcks:
							yield b0, r0, tck
			else:
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
		if b1 in kfd and r1 in kfd[b1]:
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
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rulebooks: bool = True,
	) -> Keyframe:
		"""Get a keyframe that's already in memory"""
		assert (branch, turn, tick) in self._keyframes_loaded
		graph_val: GraphValKeyframe = {}
		nodes: GraphNodesKeyframe = {}
		node_val: GraphNodeValKeyframe = {}
		edges: GraphEdgesKeyframe = {}
		edge_val: GraphEdgeValKeyframe = {}
		kf: Keyframe = {
			"graph_val": graph_val,
			"nodes": nodes,
			"node_val": node_val,
			"edges": edges,
			"edge_val": edge_val,
		}
		for k in self._graph_cache.iter_keys(branch, turn, tick):
			try:
				if (
					self._graph_cache.retrieve(k, branch, turn, tick)
					== "Deleted"
				):
					continue
			except KeyError:
				continue
			try:
				graph_val[k] = self._graph_val_cache.get_keyframe(
					k, branch, turn, tick
				)
			except KeyframeError:
				graph_val[k] = {}
			try:
				graph_val[k]["units"] = (
					self._unitness_cache.dict_cache.get_keyframe(
						k, branch, turn, tick
					)
				)
			except KeyframeError:
				pass
			try:
				nodes[k] = self._nodes_cache.get_keyframe(
					k, branch, turn, tick
				)
			except KeyframeError:
				pass
			try:
				node_val[k] = self._node_val_cache.get_keyframe(
					k, branch, turn, tick
				)
			except KeyframeError:
				pass
			try:
				edges[k] = self._edges_cache.get_keyframe(
					k, branch, turn, tick
				)
			except KeyframeError:
				pass
			try:
				edge_val[k] = self._edge_val_cache.get_keyframe(
					k, branch, turn, tick
				)
			except KeyframeError:
				pass
			try:
				locs_kf = self._things_cache.get_keyframe(
					k, branch, turn, tick
				)
			except KeyframeError:
				locs_kf = {}
				for thing in list(
					self._things_cache.iter_entities(k, branch, turn, tick)
				):
					locs_kf[thing] = self._things_cache.retrieve(
						k, thing, branch, turn, tick
					)
			if k not in node_val:
				node_val[k] = {
					thing: {"location": loc}
					for (thing, loc) in locs_kf.items()
				}
			else:
				for thing, loc in locs_kf.items():
					if thing in node_val[k]:
						node_val[k][thing]["location"] = loc
					else:
						node_val[k][thing] = {"location": loc}

		if rulebooks:
			for graph, vals in graph_val.items():
				try:
					vals["units"] = (
						self._unitness_cache.dict_cache.get_keyframe(
							graph, branch, turn, tick
						)
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
				if graph in nodes and nodes[graph]:
					try:
						node_rb_kf = self._nodes_rulebooks_cache.get_keyframe(
							graph, branch, turn, tick
						)
					except KeyframeError:
						node_rb_kf = {}
					for node in nodes[graph]:
						node_val.setdefault(graph, {}).setdefault(node, {})[
							"rulebook"
						] = node_rb_kf.get(node, (graph, node))
				if graph in kf["edges"] and kf["edges"][graph]:
					try:
						port_rb_kf = (
							self._portals_rulebooks_cache.get_keyframe(
								graph, branch, turn, tick
							)
						)
					except KeyframeError:
						port_rb_kf = {}
					if graph not in edge_val:
						edge_val[graph] = {}
					kf_graph_edge_val = edge_val[graph]
					for orig in edges[graph]:
						if orig not in kf_graph_edge_val:
							kf_graph_edge_val[orig] = {}
						kf_graph_orig_edge_val = kf_graph_edge_val[orig]
						if orig not in port_rb_kf:
							port_rb_kf[orig] = {}
						port_rb_kf_dests = port_rb_kf[orig]
						for dest in edges[graph][orig]:
							if dest not in kf_graph_orig_edge_val:
								kf_graph_orig_edge_val[dest] = {}
							kf_graph_dest_edge_val = kf_graph_orig_edge_val[
								dest
							]
							rulebook = port_rb_kf_dests.get(
								dest, (graph, orig, dest)
							)
							kf_graph_dest_edge_val["rulebook"] = rulebook
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
		plan_ticks = self._plan_ticks
		time_plan = self._time_plan
		turn_end_plan = self._turn_end_plan
		for plan, branch, turn, tick in q.plan_ticks_dump():
			plans[plan] = branch, turn, tick
			branches_plans[branch].add(plan)
			if plan > last_plan:
				last_plan = plan
			ticks = plan_ticks[plan][branch][turn]
			ticks.append(tick)
			plan_ticks[plan][branch][turn] = ticks
			turn_end_plan[branch, turn] = max(
				(turn_end_plan[branch, turn], tick)
			)
			time_plan[branch, turn, tick] = plan
		self._last_plan = last_plan

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

	def _upd_branch_parentage(self, parent: Branch, child: Branch) -> None:
		self._childbranch[parent].add(child)
		self._branch_parents[child].add(parent)
		while (parent := self.branch_parent(parent)) is not None:
			self._branch_parents[child].add(parent)

	def _alias_kf(
		self, branch_from: Branch, branch_to: Branch, turn: Turn, tick: Tick
	):
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
		for cache in (
			self._graph_val_cache,
			self._nodes_cache,
			self._node_val_cache,
			self._edges_cache,
			self._edge_val_cache,
			self._things_cache,
			self._node_contents_cache,
		):
			cache.alias_keyframe(
				branch_from,
				branch_to,
				turn,
				tick,
				{},
			)
		for cache in (
			self._universal_cache,
			self._triggers_cache,
			self._prereqs_cache,
			self._actions_cache,
			self._rulebooks_cache,
			self._unitness_cache,
			self._unitness_cache.leader_cache,
			self._unitness_cache.dict_cache,
			self._characters_rulebooks_cache,
			self._units_rulebooks_cache,
			self._characters_things_rulebooks_cache,
			self._characters_places_rulebooks_cache,
			self._characters_portals_rulebooks_cache,
			self._nodes_rulebooks_cache,
			self._portals_rulebooks_cache,
			self._neighborhoods_cache,
			self._rule_bigness_cache,
		):
			cache.alias_keyframe(branch_from, branch_to, turn, tick)
		self._mark_keyframe_loaded(branch_to, turn, tick)

	def _mark_keyframe_loaded(self, branch, turn, tick):
		self._keyframes_times.add((branch, turn, tick))
		self._keyframes_loaded.add((branch, turn, tick))
		if branch in self._keyframes_dict:
			kdb = self._keyframes_dict[branch]
			if turn in kdb:
				kdb[turn].add(tick)
			else:
				kdb[turn] = {tick}
		else:
			self._keyframes_dict[branch] = {turn: {tick}}

	@staticmethod
	def _apply_unit_delta(
		char: CharName,
		char_unit_kf: dict[CharName, dict[NodeName, bool]],
		user_set_kf: dict[CharName, dict[NodeName, frozenset[CharName]]],
		delta: dict[CharName, DeltaDict],
	) -> None:
		singlechar = frozenset([char])
		for graf, stuff in delta.items():
			if stuff is ...:
				if graf in char_unit_kf:
					del char_unit_kf[graf]
				if char in user_set_kf:
					singlegraf = frozenset([graf])
					for garph in user_set_kf[char]:
						user_set_kf[char][garph] -= singlegraf
			elif "nodes" in stuff:
				for node, ex in stuff["nodes"].items():
					if ex:
						continue
					if graf in char_unit_kf and node in char_unit_kf[graf]:
						del char_unit_kf[graf][node]
						if not char_unit_kf[graf]:
							del char_unit_kf[graf]
					if char in user_set_kf:
						singlegraf = frozenset([graf])
						for graaf in user_set_kf[char]:
							user_set_kf[char][graaf] -= singlegraf
		if char not in delta or "units" not in delta[char]:
			return
		for graf, units in delta[char]["units"].items():
			for unit, ex in units.items():
				if ex:
					if graf in char_unit_kf:
						char_unit_kf[graf][unit] = True
					else:
						char_unit_kf[graf] = {unit: True}
					if graf in user_set_kf:
						if unit in user_set_kf[graf]:
							user_set_kf[graf][unit] |= singlechar
						else:
							user_set_kf[graf][unit] = singlechar
					else:
						user_set_kf[graf] = {unit: singlechar}
				else:
					if graf in char_unit_kf and unit in char_unit_kf[graf]:
						del char_unit_kf[graf][unit]
						if not char_unit_kf[graf]:
							del char_unit_kf[graf]
					if graf in user_set_kf:
						if unit in user_set_kf[graf]:
							user_set_kf[graf][unit] -= singlechar
							if not user_set_kf[graf][unit]:
								del user_set_kf[graf][unit]
							if not user_set_kf[graf]:
								del user_set_kf[graf]

	@staticmethod
	def _apply_graph_val_delta(
		graph: CharName,
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
		for k in graph_val_delta.keys() - {
			"nodes",
			"node_val",
			"edges",
			"edge_val",
			"units",
			"character_rulebook",
			"unit_rulebook",
			"character_thing_rulebook",
			"character_place_rulebook",
			"character_portal_rulebook",
		}:
			v = graph_val_delta[k]
			if v is ...:
				if k in graph_val_keyframe:
					del graph_val_keyframe[k]
			else:
				graph_val_keyframe[k] = v

	@staticmethod
	def _apply_node_delta(
		charname: CharName,
		node_val_keyframe: dict,
		nodes_keyframe: dict,
		node_rulebook_keyframe: dict,
		thing_location_keyframe: dict,
		node_contents_keyframe: dict,
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
					if (
						"location" in node_val_keyframe[node]
						and node_val_keyframe[node]["location"]
						in node_contents_keyframe
					):
						node_contents_keyframe[
							node_val_keyframe[node]["location"]
						].remove(node)
					del node_val_keyframe[node]
				if node in thing_location_keyframe:
					del thing_location_keyframe[node]
				if node in node_contents_keyframe:
					for contained in node_contents_keyframe[node]:
						del nodes_keyframe[contained]
					del node_contents_keyframe[node]
		for node, upd in node_val_delta.items():
			if node in nodes_delta and not nodes_delta[node]:
				continue
			upd = upd.copy()
			if "location" in upd:
				loc = upd.pop("location")
				thing_location_keyframe[node] = loc
				if loc in node_contents_keyframe:
					if loc is ...:
						node_contents_keyframe[loc].remove(node)
					else:
						node_contents_keyframe[loc].add(node)
				elif loc is not ...:
					node_contents_keyframe[loc] = {node}
			if "rulebook" in upd:
				node_rulebook_keyframe[node] = upd.pop("rulebook")
			elif (
				node in node_val_keyframe
				and "rulebook" in node_val_keyframe[node]
			):
				node_rulebook_keyframe[node] = node_val_keyframe[node].pop(
					"rulebook"
				)
			elif node in nodes_keyframe:
				assert node in node_rulebook_keyframe, (
					f"No rulebook for {node}"
				)
			else:
				node_rulebook_keyframe[node] = (charname, node)
			if upd and node in node_val_keyframe:
				kv = node_val_keyframe[node]
				for key, value in upd.items():
					if value is ...:
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
						if value is ...:
							if key in kv:
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
		then: Time,
		now: Time,
		delta: DeltaDict,
	) -> None:
		if then[0] != now[0]:
			raise RuntimeError(
				"Tried to snap a keyframe from delta between branches"
			)
		if then == now:
			self.debug("Redundant keyframe snap at %s, %d, %d", *now)
			return
		if not self._time_is_loaded_between(*then, *now[1:]):
			raise RuntimeError(
				"Tried to snap a delta of time not loaded", *then, *now[1:]
			)
		self.debug(
			"Snapping keyframe from delta in branch %s between times "
			"%d, %d and %d, %d",
			*then,
			*now[1:],
		)
		keyframe = self._get_keyframe(*then, rulebooks=False)
		graph_val_keyframe: GraphValKeyframe = keyframe["graph_val"]
		nodes_keyframe: GraphNodesKeyframe = keyframe["nodes"]
		node_val_keyframe: GraphNodeValKeyframe = keyframe["node_val"]
		edges_keyframe: GraphEdgesKeyframe = keyframe["edges"]
		edge_val_keyframe: GraphEdgeValKeyframe = keyframe["edge_val"]
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
			if v is ...:
				if k in universal_keyframe:
					del universal_keyframe[k]
			else:
				universal_keyframe[k] = v
		if "rulebooks" in delta:
			rulebooks_keyframe.update(delta["rulebooks"])
		for rule, funcs in delta.get("rules", {}).items():
			if "triggers" in funcs and funcs["triggers"]:
				triggers_keyframe[rule] = funcs["triggers"]
			if "prereqs" in funcs and funcs["prereqs"]:
				prereqs_keyframe[rule] = funcs["prereqs"]
			if "actions" in funcs and funcs["actions"]:
				actions_keyframe[rule] = funcs["actions"]
			if "neighborhood" in funcs and funcs["neighborhood"] is not None:
				neighborhoods_keyframe[rule] = funcs["neighborhood"]
			if "big" in funcs and funcs["big"]:
				bigs[rule] = funcs["big"]
		things_keyframe = {}
		nodes_rulebooks_keyframe = {}
		portals_rulebooks_keyframe = {}
		units_keyframe = {}
		for k, vs in graph_val_keyframe.items():
			if "units" in vs:
				units_keyframe[k] = vs.pop("units")
		user_set_keyframe = {}
		graphs: set[CharName] = (
			set(self._graph_cache.iter_keys(*then)).union(delta.keys())
			- self.illegal_graph_names
		)
		for graph in graphs:
			try:
				user_set_keyframe[graph] = (
					self._unitness_cache.leader_cache.get_keyframe(
						graph, *then, copy=True
					)
				)
			except KeyframeError:
				user_set_keyframe[graph] = {}
		for graph in graphs:
			delt = delta.get(graph, {})
			if delt is ...:
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
					self._unitness_cache.dict_cache.get_keyframe(
						graph, *then, copy=True
					)
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
			self._apply_unit_delta(
				graph,
				charunit,
				user_set_keyframe,
				delta,
			)
			self._apply_node_delta(
				graph,
				node_val_keyframe.setdefault(graph, {}),
				nodes_keyframe.setdefault(graph, {}),
				noderbs,
				locs,
				conts,
				delt.get("node_val", {}),
				delt.get("nodes", {}),
			)
			self._apply_edge_delta(
				graph,
				edge_val_keyframe.setdefault(graph, {}),
				edges_keyframe.setdefault(graph, {}),
				portrbs,
				delt.get("edge_val", {}),
				delt.get("edges", {}),
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
		for char, kf in user_set_keyframe.items():
			self._unitness_cache.leader_cache.set_keyframe(char, *now, kf)
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
			if deltg is ...:
				del graphs_keyframe[graph]
				continue
			combined_node_val_keyframe = {
				node: val.copy()
				for (node, val) in node_val_keyframe.get(graph, {}).items()
			}
			for node, loc in things_keyframe.get(graph, {}).items():
				if loc is ...:
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
			if units_keyframe[graph]:
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
			self._mark_keyframe_loaded(*time_from)
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
				self._mark_keyframe_loaded(
					parent, branched_turn_from, branched_tick_from
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
		return retrieved and not isinstance(retrieved, Exception)

	@world_locked
	def _exist_node(
		self,
		character: CharName,
		node: NodeName,
		exist: bool = True,
		*,
		now: Optional[Time] = None,
	) -> None:
		nbtt, exist_node, store = self._exist_node_stuff
		if now:
			branch, turn, tick = now
		else:
			branch, turn, tick = nbtt()
		store(character, node, branch, turn, tick, exist)
		if exist:
			self._nodes_rulebooks_cache.store(
				character, node, branch, turn, tick, (character, node)
			)
			self.query.set_node_rulebook(
				character, node, branch, turn, tick, (character, node)
			)
		exist_node(character, node, branch, turn, tick, exist)

	def _edge_exists(
		self, character: CharName, orig: NodeName, dest: NodeName
	) -> bool:
		retrieve, btt = self._edge_exists_stuff
		args = (character, orig, dest, *btt())
		retrieved = retrieve(args)
		return retrieved is not None and not isinstance(retrieved, Exception)

	@world_locked
	def _exist_edge(
		self,
		character: Key,
		orig: Key,
		dest: Key,
		exist: bool = True,
		*,
		now: Optional[Time] = None,
	) -> None:
		nbtt, exist_edge, store = self._exist_edge_stuff
		if now:
			branch, turn, tick = now
		else:
			branch, turn, tick = nbtt()
		store(character, orig, dest, branch, turn, tick, exist)
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
		exist_edge(character, orig, dest, branch, turn, tick, exist or False)

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
		branch: Branch,
		turn_from: Turn,
		tick_from: Tick,
		turn_to: Turn,
		tick_to: Tick,
	):
		if self._time_is_loaded_between(
			branch, turn_from, tick_from, turn_to, tick_to
		):
			return
		try:
			self._get_keyframe(branch, turn_from, tick_from, silent=True)
		except KeyframeError:
			self._recurse_delta_keyframes(branch, turn_from, tick_from)
		noderows = []
		nodevalrows = []
		edgerows = []
		edgevalrows = []
		graphvalrows = []
		loaded_graphs = self.query.load_windows(
			[(branch, turn_from, tick_from, turn_to, tick_to)]
		)
		for graph, loaded in loaded_graphs.items():
			match graph:
				case "graphs":
					self._graph_cache.load(loaded)
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
				case "rule_neighborhood":
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
		if not self._time_is_loaded_between(
			branch, turn_from, tick_from, turn_to, tick_to
		):
			self.warning(
				"Didn't completely fill the time window: %s (%d,%d)→(%d,%d)",
				branch,
				turn_from,
				tick_from,
				turn_to,
				tick_to,
			)
			self._extend_loaded_window(
				branch, turn_from, tick_from, turn_to, tick_to
			)
		return loaded_graphs

	def _extend_loaded_window(
		self,
		branch: Branch,
		turn_from: Turn,
		tick_from: Tick,
		turn_to: Turn | None,
		tick_to: Tick | None,
	):
		loaded = self._loaded
		if branch in loaded:
			a, b, x, y = loaded[branch]
			if None in (x, y) or (x, y) < (turn_to, tick_to):
				(x, y) = (turn_to, tick_to)
			if (turn_from, tick_from) < (a, b):
				(a, b) = (turn_from, tick_from)
			loaded[branch] = (a, b, x, y)
		else:
			loaded[branch] = turn_from, tick_from, turn_to, tick_to

	@world_locked
	def unload(self) -> None:
		"""Remove everything from memory that can be removed."""
		# If we're not connected to some database, we can't unload anything
		# without losing data
		if isinstance(self.query, NullDatabaseConnector):
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

	def _iter_linear_time(self, branch: Branch):
		for turn, ticks in reversed(self._keyframes_dict[branch].items()):
			for tick in sorted(ticks, reverse=True):
				yield turn, tick

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
		for branch, turn, tick in self._iter_keyframes(
			branch_now, turn_now, tick_now, loaded=not loading
		):
			if (turn, tick) <= (turn_now, tick_now):
				latest_past_keyframe = branch, turn, tick
				break
		for turn, ticks in (
			self._keyframes_dict[branch_now]
			.future(turn, include_same_rev=True)
			.items()
		):
			for tick in sorted(ticks):
				if (turn, tick) <= (turn_now, tick_now):
					continue
				if (branch_now, turn, tick) in cache:
					earliest_future_keyframe = (branch_now, turn, tick)
					break
			if earliest_future_keyframe is not None:
				break
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
		self, silent: bool = False, update_worker_processes: bool = True
	) -> Keyframe | None:
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
		self.debug("Snapping keyframe at %s, %d, %d", branch, turn, tick)
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
				self.debug(
					"Fresh keyframe, snapping de novo at %s, %d, %d",
					branch,
					turn,
					tick,
				)
				self._snap_keyframe_de_novo(branch, turn, tick)
				if silent:
					return None
				else:
					return self._get_kf(branch, turn, tick)
			self.debug(
				"Swimming up the timestream from %s, %d, %d",
				branch,
				turn,
				tick,
			)
			the_kf = self._recurse_delta_keyframes(branch, turn, tick)
		if the_kf not in self._keyframes_loaded:
			self._get_keyframe(*the_kf, silent=True)
		if the_kf != (branch, turn, tick):
			self.load_between(*the_kf, turn, tick)
			if the_kf[0] != branch:
				self.debug(
					"Aliasing keyframe from branch %s to %s, %d, %d",
					the_kf[0],
					branch,
					turn,
					tick,
				)
				self._alias_kf(the_kf[0], branch, turn, tick)
			self._snap_keyframe_from_delta(
				the_kf,
				(branch, turn, tick),
				self._get_branch_delta(*the_kf, turn, tick),
			)
			self._mark_keyframe_loaded(branch, turn, tick)
		if silent:
			return None
		ret = self._get_kf(branch, turn, tick)
		if hasattr(self, "_worker_processes") and update_worker_processes:
			self._update_all_worker_process_states(clobber=True)
		return ret

	@world_locked
	def _read_at(
		self, branch: Branch, turn: Turn, tick: Tick
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
							self.query.eternal["trunk"], 0, 0
						)
					),
					self.query.load_windows(
						[(self.query.eternal["trunk"], 0, 0, None, None)]
					),
				)
			else:
				windows = self._build_loading_windows(
					self.query.eternal["trunk"], 0, 0, branch, turn, tick
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
			if hasattr(self, "_validate_initial_keyframe_load"):
				self._validate_initial_keyframe_load(
					self._get_keyframe(*latest_past_keyframe)
				)
			else:
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
		if rule_neighborhoods := loaded.pop("rule_neighborhood", None):
			self._neighborhoods_cache.load(rule_neighborhoods)
		if rule_big := loaded.pop("rule_big", None):
			self._rule_bigness_cache.load(rule_big)
		if graphs := loaded.pop("graphs", None):
			self._graph_cache.load(graphs)
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
			if data.get("units"):
				self._unitness_cache.load(data["units"])

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

	def turn_end(self, branch: Branch = None, turn: Turn = None) -> Tick:
		if branch is None:
			branch = self._obranch
		if turn is None:
			turn = self._oturn
		return self._turn_end[branch, turn]

	def turn_end_plan(self, branch: Branch = None, turn: Turn = None) -> Tick:
		if branch is None:
			branch = self._obranch
		if turn is None:
			turn = self._oturn
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
			builtins = __builtins__.copy()
			builtins["set"] = OrderlySet
			builtins["frozenset"] = OrderlyFrozenSet
			globls = globals().copy()
			ret = eval(
				"fake_submit(fn, *args, **kwargs)",
				globls,
				{
					"fake_submit": fake_submit,
					"fn": fn,
					"args": args,
					"kwargs": kwargs,
				},
			)
		ret.uid = uid
		self._top_uid += 1
		return ret

	def _manage_futs(self):
		while not hasattr(self, "_closed"):
			while self._how_many_futs_running < self._workers:
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

		if hasattr(self, "_worker_processes"):
			for i, (lock, pipein, pipeout, proc) in enumerate(
				zip(
					self._worker_locks,
					self._worker_inputs,
					self._worker_outputs,
					self._worker_processes,
				)
			):
				with lock:
					if not proc.is_alive():
						continue
					pipein.send_bytes(b"shutdown")
					proc.join(timeout=SUBPROCESS_TIMEOUT)
					if proc.exitcode is None:
						if KILL_SUBPROCESS:
							os.kill(proc.pid, signal.SIGKILL)
						else:
							raise RuntimeError("Worker process didn't exit", i)
					if not KILL_SUBPROCESS and proc.exitcode != 0:
						raise RuntimeError(
							"Worker process didn't exit normally",
							i,
							proc.exitcode,
						)
					proc.close()
					pipein.close()
					pipeout.close()
			del self._worker_processes

	def _detect_kf_interval_override(self):
		if self._planning:
			return True
		if getattr(self, "_no_kc", False):
			return True

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
		name: CharName,
		type_s: str = "DiGraph",
		data: CharacterFacade | Graph | nx.Graph | dict | KeyframeTuple = None,
	) -> None:
		if name in self.illegal_graph_names:
			raise GraphNameError("Illegal name")
		branch, turn, tick = self._btt()
		if (turn, tick) != (0, 0):
			branch, turn, tick = self._nbtt()
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
				kf = rbcache.get_keyframe(branch, turn, tick)
			except KeyframeError:
				kf = {}
				for ch in self._graph_cache.iter_entities(branch, turn, tick):
					# may yield this very character
					try:
						kf[ch] = rbcache.retrieve(ch, branch, turn, tick)
					except KeyError:
						kf[ch] = (rbname, ch)
			kf[name] = (rbname, name)
			rbcache.set_keyframe(branch, turn, tick, kf)
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
	def _complete_turn(self, branch: Branch, turn: Turn) -> None:
		self._extend_branch(branch, turn, self.turn_end_plan(branch, turn))
		self._turns_completed_d[branch] = turn
		self.query.complete_turn(
			branch, turn, discard_rules=not self.keep_rules_journal
		)

	def _get_last_completed_turn(self, branch: Branch) -> Turn | None:
		if branch not in self._turns_completed_d:
			return None
		return self._turns_completed_d[branch]

	def _make_node(
		self, graph: Character, node: NodeName
	) -> thing_cls | place_cls:
		if self._is_thing(graph.name, node):
			return self.thing_cls(graph, node)
		else:
			return self.place_cls(graph, node)

	def _make_edge(
		self,
		graph: Character,
		orig: NodeName,
		dest: NodeName,
	) -> portal_cls:
		return self.portal_cls(graph, orig, dest)

	def _is_timespan_too_big(
		self, branch: Branch, turn_from: Turn, turn_to: Turn
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
		r: Turn
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
			if chardeltpacked == ELLIPSIS:
				delt[unpack(char)] = ...
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
				va = kf_from[kfkey].get(k, ...)
				vb = kf_to[kfkey].get(k, ...)
				ids_from.append(id(va))
				ids_to.append(id(vb))
				values_from.append(va)
				values_to.append(vb)
		for graph in kf_from["graph_val"].keys() | kf_to["graph_val"].keys():
			a = kf_from["graph_val"].get(graph, {})
			b = kf_to["graph_val"].get(graph, {})
			key_union = a.keys() | b.keys()
			if "units" in key_union:
				units_a = a.get("units", {})
				units_b = b.get("units", {})
				for g in units_a.keys() | units_b.keys():
					keys.append(("units", graph, g))
					va = frozenset(units_a.get(g, {}).keys())
					vb = frozenset(units_b.get(g, {}).keys())
					ids_from.append(id(va))
					ids_to.append(id(vb))
					values_from.append(va)
					values_to.append(vb)
			for k in (a.keys() | b.keys()) - {"units"}:
				keys.append(("graph", graph, k))
				va = a.get(k, ...)
				vb = b.get(k, ...)
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
					va = a.get(k, ...)
					vb = b.get(k, ...)
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
					va = a.get(k, ...)
					vb = b.get(k, ...)
					ids_from.append(id(va))
					ids_to.append(id(vb))
					values_from.append(va)
					values_to.append(vb)

		def pack_one(k, va, vb, deleted_nodes, deleted_edges):
			if va == vb:
				return
			match k:
				case "universal", key:
					key = pack(key)
					delta[UNIVERSAL][key] = pack(vb)
				case "triggers", rule:
					delta[RULES][pack(rule)][TRIGGERS] = pack(vb)
				case "prereqs", rule:
					delta[RULES][pack(rule)][PREREQS] = pack(vb)
				case "actions", rule:
					delta[RULES][pack(rule)][ACTIONS] = pack(vb)
				case "neighborhood", rule:
					delta[RULES][pack(rule)][NEIGHBORHOOD] = pack(vb)
				case "big", rule:
					delta[RULES][pack(rule)][BIG] = pack(vb)
				case "rulebook", rulebook:
					delta[RULEBOOK][pack(rulebook)] = pack(vb)
				case "units", char, graph:
					va: frozenset[NodeName]
					vb: frozenset[NodeName]
					charpacked = pack(char)
					unit_delta = {}
					for k in vb - va:
						unit_delta[k] = True
					for k in va - vb:
						unit_delta[k] = False
					if charpacked in delta:
						if UNITS in delta[charpacked]:
							delta[charpacked][UNITS][graph] = unit_delta
						else:
							delta[charpacked][UNITS] = {graph: unit_delta}
					else:
						delta[charpacked] = {UNITS: {graph: unit_delta}}
				case "node", graph, node, key:
					if graph in deleted_nodes and node in deleted_nodes[graph]:
						return
					graph, node, key = map(pack, (graph, node, key))
					if graph not in delta:
						delta[graph] = newgraph()
					delta[graph][NODE_VAL][node][key] = pack(vb)
				case "edge", graph, orig, dest, key:
					if (graph, orig, dest) in deleted_edges:
						return
					graph, orig, dest, key = map(
						pack, (graph, orig, dest, key)
					)
					if graph not in delta:
						delta[graph] = newgraph()
					delta[graph][EDGE_VAL][orig][dest][key] = pack(vb)
				case "graph", graph, key:
					graph, key = map(pack, (graph, key))
					if graph not in delta:
						delta[graph] = newgraph()
					delta[graph][key] = pack(vb)

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
			values_changed: np.array = np.array(ids_from) != np.array(ids_to)
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
				delta[self.pack(graf)] = ELLIPSIS
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
				delta[pack(deleted)] = ELLIPSIS
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
			if (
				key in {RULES, RULEBOOKS, ETERNAL, UNIVERSAL}
				or mapp == ELLIPSIS
			):
				continue
			todel = []
			if UNITS in mapp:
				mapp[UNITS] = pack(mapp[UNITS])
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

	def close(self) -> None:
		"""Commit changes and close the database

		This will be useless thereafter.

		"""
		if hasattr(self, "_closed"):
			raise RuntimeError("Already closed")
		time_was: tuple[Turn, Tick] = (self.turn, self.tick)
		if time_was > self._branch_end():
			(self.turn, self.tick) = self._branch_end()
		if (
			self._keyframe_on_close
			and self._btt() not in self._keyframes_times
		):
			if hasattr(self, "_validate_final_keyframe"):
				self._validate_final_keyframe(
					self.snap_keyframe(update_worker_processes=False)
				)
			else:
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
		self.shutdown()
		self.query.close()
		for cache in self._caches:
			if hasattr(cache, "clear"):
				cache.clear()
		gc.collect()
		self._closed = True

	def __enter__(self):
		"""Return myself. For compatibility with ``with`` semantics."""
		return self

	def __exit__(self, *args):
		"""Close on exit."""
		self.close()

	def _handled_char(
		self,
		charn: CharName,
		rulebook: RulebookName,
		rulen: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> None:
		self._character_rules_handled_cache.store(
			charn, rulebook, rulen, branch, turn, tick
		)
		self.query.handled_character_rule(
			charn, rulebook, rulen, branch, turn, tick
		)

	def _handled_av(
		self,
		character: CharName,
		graph: CharName,
		avatar: NodeName,
		rulebook: RulebookName,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> None:
		self._unit_rules_handled_cache.store(
			character, graph, avatar, rulebook, rule, branch, turn, tick
		)
		self.query.handled_unit_rule(
			character, rulebook, rule, graph, avatar, branch, turn, tick
		)

	def _handled_char_thing(
		self,
		character: CharName,
		rulebook: RulebookName,
		rule: RuleName,
		thing: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> None:
		self._character_thing_rules_handled_cache.store(
			character, thing, rulebook, rule, branch, turn, tick
		)
		self.query.handled_character_thing_rule(
			character, rulebook, rule, thing, branch, turn, tick
		)

	def _handled_char_place(
		self,
		character: CharName,
		place: NodeName,
		rulebook: RulebookName,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> None:
		self._character_place_rules_handled_cache.store(
			character, place, rulebook, rule, branch, turn, tick
		)
		self.query.handled_character_place_rule(
			character, rulebook, rule, place, branch, turn, tick
		)

	def _handled_char_port(
		self,
		character: CharName,
		orig: NodeName,
		dest: NodeName,
		rulebook: RulebookName,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> None:
		self._character_portal_rules_handled_cache.store(
			character, orig, dest, rulebook, rule, branch, turn, tick
		)
		self.query.handled_character_portal_rule(
			character, rulebook, rule, orig, dest, branch, turn, tick
		)

	def _handled_node(
		self,
		character: CharName,
		node: NodeName,
		rulebook: RulebookName,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> None:
		self._node_rules_handled_cache.store(
			character, node, rulebook, rule, branch, turn, tick
		)
		self.query.handled_node_rule(
			character, node, rulebook, rule, branch, turn, tick
		)

	def _handled_portal(
		self,
		character: CharName,
		orig: NodeName,
		dest: NodeName,
		rulebook: RulebookName,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> None:
		self._portal_rules_handled_cache.store(
			character, orig, dest, rulebook, rule, branch, turn, tick
		)
		self.query.handled_portal_rule(
			character, orig, dest, rulebook, rule, branch, turn, tick
		)

	@world_locked
	@_all_worker_locks
	def _update_all_worker_process_states(self, clobber: bool = False):
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
					k: new_eternal.get(k, ...)
					for k in old_eternal.keys() | new_eternal.keys()
					if old_eternal.get(k, ...) != new_eternal.get(k, ...)
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
			self.debug(
				"Updated all worker process states at "
				+ repr(self._worker_updated_btts[i])
				+ f" ({len(deltas)} distinct deltas)"
			)

	@world_locked
	def _update_worker_process_state(self, i, lock=True):
		branch_from, turn_from, tick_from = self._worker_updated_btts[i]
		if (branch_from, turn_from, tick_from) == self._btt():
			return
		old_eternal = self._worker_last_eternal
		new_eternal = self._worker_last_eternal = dict(self.eternal.items())
		eternal_delta = {
			k: new_eternal.get(k, ...)
			for k in old_eternal.keys() | new_eternal.keys()
			if old_eternal.get(k, ...) != new_eternal.get(k, ...)
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
		self.debug(f"Updated worker {i} at {self._worker_updated_btts[i]}")

	def _changed(self, charn: CharName, entity: tuple) -> bool:
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
		rulebook: RulebookName,
		rule: Rule,
		check_handled: Callable[[], bool],
		mark_handled: Callable[[Tick], None],
		entity: char_cls | thing_cls | place_cls | portal_cls,
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
			fut.check_handled = check_handled
			fut.mark_handled = mark_handled
			yield fut
			return
		for trigger in rule.triggers:
			fut = self.submit(trigger, entity)
			fut.rule = rule
			fut.prio = prio
			fut.entity = entity
			fut.rulebook = rulebook
			fut.check_handled = check_handled
			fut.mark_handled = mark_handled
			yield fut

	def _check_prereqs(
		self, rule: Rule, mark_handled: Callable[[Tick], None], entity
	):
		if not entity:
			return False
		for prereq in rule.prereqs:
			res = prereq(entity)
			if not res:
				mark_handled(self.tick)
				return False
		return True

	def _do_actions(
		self, rule: Rule, mark_handled: Callable[[Tick], None], entity
	):
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
		mark_handled(self.tick)
		return actres

	def _get_place_neighbors(
		self, charn: CharName, name: NodeName
	) -> set[Key]:
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

	def _get_place_contents(self, charn: CharName, name: NodeName) -> set[Key]:
		try:
			return self._node_contents_cache.retrieve(
				charn, name, *self._btt()
			)
		except KeyError:
			return set()

	def _iter_place_portals(
		self, charn: CharName, name: NodeName
	) -> Iterator[tuple[Key, Key]]:
		now = self._btt()
		for dest in self._edges_cache.iter_successors(charn, name, *now):
			yield (name, dest)
		for orig in self._edges_cache.iter_predecessors(charn, name, *now):
			yield (orig, name)

	def _get_thing_location_tup(
		self, charn: CharName, name: NodeName
	) -> tuple[Key, Key] | ():
		try:
			return (self._things_cache.retrieve(charn, name, *self._btt()),)
		except KeyError:
			return ()

	def _get_neighbors(
		self,
		entity: place_cls | thing_cls | portal_cls,
		neighborhood: int | None,
	) -> list[tuple[NodeName] | tuple[NodeName, NodeName]] | None:
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
	) -> list[tuple[NodeName] | tuple[NodeName, NodeName]] | None:
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
			self.load_at(branch_now, Turn(turn_now - 1), 0)
			self._oturn -= 1
			self._otick = 0
			last_turn_neighbors = self._get_neighbors(entity, neighborhood)
			self._set_btt(branch_now, turn_now, tick_now)
			this_turn_neighbors = self._get_neighbors(entity, neighborhood)
		if set(last_turn_neighbors) != set(this_turn_neighbors):
			return None
		return this_turn_neighbors

	def _get_thing(self, graphn: CharName, thingn: NodeName):
		node_objs = self._node_objs
		key = (graphn, thingn)
		if key not in node_objs:
			node_objs[key] = self.thing_cls(self.character[graphn], thingn)
		return node_objs[key]

	def _get_place(self, graphn: CharName, placen: NodeName):
		node_objs = self._node_objs
		key = (graphn, placen)
		if key not in node_objs:
			node_objs[key] = self.place_cls(self.character[graphn], placen)
		return node_objs[key]

	RulesTodoType = dict[
		tuple[float, RulebookName],
		list[
			tuple[
				Rule,
				list[tuple[Callable[..., bool], Callable, EntityKey]],
			]
		],
	]

	def _eval_triggers(self) -> RulesTodoType:
		branch, turn, tick = self._btt()
		charmap = self.character
		rulemap = self.rule
		todo: dict[
			tuple[RulebookPriority, RulebookName],
			list[
				tuple[
					Rule,
					list[
						tuple[
							Callable[[], bool],
							Callable[[Tick], None],
							EntityKey,
						],
					]
					| None,
				],
			],
		] = {}
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
			check_handled = partial(
				self._character_rules_handled_cache.was_handled,
				branch,
				turn,
				rulebook,
				rulename,
				charactername,
			)
			mark_handled = partial(
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
					check_handled,
					mark_handled,
					entity,
					None,
				)
			)

		avcache_retr = self._unitness_cache._base_retrieve
		node_exists = self._node_exists
		get_node = self._get_node
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
			check_handled = partial(
				self._unit_rules_handled_cache.was_handled,
				branch,
				turn,
				rulebook,
				rulen,
				charn,
				graphn,
				avn,
			)
			mark_handled = partial(
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
					check_handled,
					mark_handled,
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
			check_handled = partial(
				self._character_thing_rules_handled_cache.was_handled,
				branch,
				turn,
				rulebook,
				rulen,
				charn,
				thingn,
			)
			mark_handled = partial(
				handled_char_thing,
				charn,
				rulebook,
				rulen,
				thingn,
				branch,
				turn,
			)
			entity = get_thing(charn, thingn)
			trig_futs.extend(
				self._iter_submit_triggers(
					prio,
					rulebook,
					rule,
					check_handled,
					mark_handled,
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
			check_handled = partial(
				self._character_place_rules_handled_cache.was_handled,
				branch,
				turn,
				rulebook,
				rulen,
				charn,
				placen,
			)
			mark_handled = partial(
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
					check_handled,
					mark_handled,
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
			check_handled = partial(
				self._character_portal_rules_handled_cache.was_handled,
				branch,
				turn,
				rulebook,
				rulen,
				charn,
				orign,
				destn,
			)
			mark_handled = partial(
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
					check_handled,
					mark_handled,
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
			check_handled = partial(
				self._node_rules_handled_cache.was_handled,
				branch,
				turn,
				rulebook,
				rulen,
				charn,
				noden,
			)
			mark_handled = partial(
				handled_node, charn, noden, rulebook, rulen, branch, turn
			)
			entity = get_node(charn, noden)
			trig_futs.extend(
				self._iter_submit_triggers(
					prio,
					rulebook,
					rule,
					check_handled,
					mark_handled,
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
			check_handled = partial(
				self._portal_rules_handled_cache.was_handled,
				branch,
				turn,
				rulebook,
				rulen,
				charn,
				orign,
				destn,
			)
			mark_handled = partial(
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
					check_handled,
					mark_handled,
					entity,
					self._get_effective_neighbors(entity, rule.neighborhood),
				)
			)

		to_done: set[tuple[RulebookName, RuleName, EntityKey]] = set()
		for fut in trig_futs:
			entity_key = self._get_entity_key(fut.entity)
			if (
				fut.rulebook,
				fut.rule.name,
				entity_key,
			) not in to_done and fut.result():
				to_done.add((fut.rulebook, fut.rule.name, entity_key))
				todo_key = (fut.prio, fut.rulebook)
				rulebook = self.rulebook[fut.rulebook]
				rbidx = rulebook.index(fut.rule)
				what_do = (fut.check_handled, fut.mark_handled)
				if todo_key in todo:
					rulez = todo[todo_key]
					if rulez[rbidx] is None:
						rulez[rbidx] = (fut.rule, {entity_key: what_do})
					else:
						rule, applications = rulez[rulebook.index(fut.rule)]
						applications[entity_key] = what_do
				else:
					rulez = [None] * len(rulebook)
					todo[todo_key] = rulez
					rulez[rbidx] = (
						fut.rule,
						{entity_key: what_do},
					)

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

	def _follow_one_rule(
		self,
		rule: Rule,
		check_handled: Callable[[], bool],
		mark_handled: Callable[[Tick], None],
		entity: char_cls | thing_cls | place_cls | portal_cls,
	):
		check_prereqs = self._check_prereqs
		do_actions = self._do_actions

		if not entity:
			self.debug(
				f"not checking prereqs for rule {rule.name} "
				f"on nonexistent entity {self._fmtent(entity)}"
			)
			return
		if check_handled():
			msg = (
				f"Tried to run rule {rule.name} on {entity}, but "
				"it's already been run this turn, in the same rulebook"
			)
			self.error(msg)
			raise RedundantRuleError(msg, rule, entity)
		self.debug(
			f"checking prereqs for rule {rule.name} on entity {self._fmtent(entity)}"
		)
		builtins = __builtins__.copy()
		builtins["set"] = OrderlySet
		builtins["frozenset"] = OrderlyFrozenSet
		globls = globals().copy()
		globls["__builtins__"] = builtins
		if eval(
			"check_prereqs(rule, mark_handled, entity)",
			globls,
			{
				"check_prereqs": check_prereqs,
				"rule": rule,
				"mark_handled": mark_handled,
				"entity": entity,
			},
		):
			self.debug(
				f"prereqs for rule {rule.name} on entity "
				f"{self._fmtent(entity)} satisfied, will run actions"
			)
			try:
				ret = eval(
					"do_actions(rule, mark_handled, entity)",
					globls,
					{
						"do_actions": do_actions,
						"rule": rule,
						"mark_handled": mark_handled,
						"entity": entity,
					},
				)
				self.debug(
					f"actions for rule {rule.name} on entity "
					f"{self._fmtent(entity)} have run without incident"
				)
				return ret
			except StopIteration as ex:
				raise InnerStopIteration from ex

	@classmethod
	def _get_entity_key(cls, ent: entity_cls) -> EntityKey:
		if isinstance(ent, cls.char_cls):
			return (ent.name,)
		elif isinstance(ent, (cls.place_cls, cls.thing_cls)):
			return (ent.character.name, ent.name)
		else:
			assert isinstance(ent, cls.edge_cls)
			return (ent.character.name, ent.orig, ent.dest)

	def _get_entity_from_key(self, ent_key: EntityKey) -> entity_cls:
		if len(ent_key) == 1:
			(name,) = ent_key
			return self.character[name]
		elif len(ent_key) == 2:
			(char, name) = ent_key
			return self.character[char].node[name]
		else:
			(char, orig, dest) = ent_key
			return self.character[char].portal[orig][dest]

	def _follow_rules(
		self,
		todo: RulesTodoType,
	):
		# TODO: roll back changes done by rules that raise an exception
		# TODO: if there's a paradox while following some rule,
		#  start a new branch, copying handled rules
		for prio, rulebook in sort_set(todo.keys()):
			for rule, applications in filter(None, todo[prio, rulebook]):
				for entity_key in sort_set(applications.keys()):
					try:
						entity = self._get_entity_from_key(entity_key)
					except KeyError:
						# entity was deleted by some earlier rule application
						continue
					check_handled, mark_handled = applications[entity_key]
					yield self._follow_one_rule(
						rule,
						check_handled,
						mark_handled,
						entity,
					)

	def new_character(
		self,
		name: Key | str | int | float | tuple[Key, ...] | frozenset[Key],
		data: Optional[Graph] = None,
		layout: bool = False,
		node: Optional[NodeValDict] = None,
		edge: Optional[EdgeValDict] = None,
		**kwargs,
	) -> Character:
		"""Create and return a new :class:`Character`.

		See :meth:`add_character` for details.

		"""
		self.add_character(name, data, layout, node=node, edge=edge, **kwargs)
		return self.character[CharName(name)]

	def add_character(
		self,
		name: Key | str | int | float | tuple[Key, ...] | frozenset[Key],
		data: Optional[Graph | DiGraph] = None,
		layout: bool = False,
		node: Optional[NodeValDict] = None,
		edge: Optional[EdgeValDict] = None,
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
		if self.branch != self.trunk or self.turn != 0 or self.tick != 0:
			self._nbtt()
		self._init_graph(name, "DiGraph", data)
		if self._btt() not in self._keyframes_times:
			self.snap_keyframe(silent=True, update_worker_processes=False)
		if hasattr(self, "_worker_processes"):
			self._update_all_worker_process_states(clobber=True)

	@world_locked
	def del_character(self, name: CharName) -> None:
		"""Mark a graph as deleted

		:arg name: name of an existing graph

		"""
		# make sure the graph exists before deleting
		graph = self.character[name]
		with self.batch(), self._graph_val_cache.overwriting():
			now = self._nbtt()
			for orig in list(graph.adj):
				for dest in list(graph.adj[orig]):
					now = graph.adj[orig][dest]._delete(now=now)
			for node in list(graph.node):
				if node in graph.node:
					now = graph.node[node]._delete(now=now)
			for stat in set(graph.graph) - {"name", "units"}:
				self._graph_val_cache.store(name, stat, *now, None)
				self.query.graph_val_set(name, stat, *now, None)
			self._graph_cache.store(name, *now, ...)
			self.query.graphs_insert(name, *now, "Deleted")
			self._graph_cache.keycache.clear()
		if hasattr(self, "_worker_processes"):
			self._call_every_subprocess("_del_character", name)

	def _is_thing(self, character: CharName, node: NodeName) -> bool:
		return self._things_cache.contains_entity(
			character, node, *self._btt()
		)

	@world_locked
	def _set_thing_loc(
		self, character: CharName, node: NodeName, loc: NodeName
	) -> None:
		if loc is not None:
			# make sure the location really exists now
			self._nodes_cache.retrieve(character, loc, *self.time)
		branch, turn, tick = self._nbtt()
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
		user_kf = {}
		for char in all_graphs:
			char_kf = {}
			for graph in self._unitness_cache.iter_char_graphs(
				char, branch, turn, tick
			):
				for unit in self._unitness_cache.iter_entities(
					char, graph, branch, turn, tick
				):
					char_kf[graph] = {
						unit: self._unitness_cache.retrieve(
							char, graph, unit, branch, turn, tick
						)
					}
					if graph in user_kf:
						if unit in user_kf[graph]:
							user_kf[graph][unit] |= frozenset([char])
						else:
							user_kf[graph][unit] = frozenset([char])
					else:
						user_kf[graph] = {unit: frozenset([char])}

			self._unitness_cache.set_keyframe(
				char, branch, turn, tick, char_kf
			)
		for char, kf in user_kf.items():
			self._unitness_cache.leader_cache.set_keyframe(
				char, branch, turn, tick, user_kf
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
				triggers = self._triggers_cache.retrieve(
					rule, branch, turn, tick
				)
				if triggers:
					trigs[rule] = triggers
			except KeyError:
				pass
			try:
				prereqs = self._prereqs_cache.retrieve(
					rule, branch, turn, tick
				)
				if prereqs:
					preqs[rule] = prereqs
			except KeyError:
				pass
			try:
				actions = self._actions_cache.retrieve(
					rule, branch, turn, tick
				)
				if actions:
					acts[rule] = actions
			except KeyError:
				pass
			try:
				neighbors = self._neighborhoods_cache.retrieve(
					rule, branch, turn, tick
				)
				if neighbors is not None:
					nbrs[rule] = neighbors
			except KeyError:
				pass
			try:
				big = self._rule_bigness_cache.retrieve(
					rule, branch, turn, tick
				)
				if big:
					bigs[rule] = big
			except KeyError:
				pass
		self._triggers_cache.set_keyframe(branch, turn, tick, trigs)
		self._prereqs_cache.set_keyframe(branch, turn, tick, preqs)
		self._actions_cache.set_keyframe(branch, turn, tick, acts)
		self._neighborhoods_cache.set_keyframe(branch, turn, tick, nbrs)
		self._rule_bigness_cache.set_keyframe(branch, turn, tick, bigs)
		for charname in all_graphs:
			locs = {}
			conts_mut = {}
			for thingname in self._things_cache.iter_things(
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
			kf = {}
			for ch in all_graphs:
				try:
					kf[ch] = rbcache.retrieve(ch, branch, turn, tick)
				except KeyError:
					kf[ch] = (rbcache.name, ch)
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
			graph = self.character[graphn]
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
		graph: CharName,
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
		units_kf = graph_val.pop("units", {})
		self._unitness_cache.set_keyframe(graph, branch, turn, tick, units_kf)
		for char, units in units_kf.items():
			try:
				user_kf = self._unitness_cache.leader_cache.get_keyframe(
					char, branch, turn, tick, copy=True
				)
			except KeyframeError:
				user_kf = {}
			for unit in units:
				if unit in user_kf:
					user_kf[unit] |= frozenset([graph])
				else:
					user_kf[unit] = frozenset([graph])
			self._unitness_cache.leader_cache.set_keyframe(
				char, branch, turn, tick, user_kf
			)
		node_rb_kf = {}
		locs_kf = {}
		conts_kf = {}
		for unit, val in nodes.items():
			node_rb_kf[unit] = val.pop("rulebook", (graph, unit))
			if "location" not in val:
				continue
			locs_kf[unit] = location = val["location"]
			if location in conts_kf:
				conts_kf[location].add(unit)
			else:
				conts_kf[location] = {unit}
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
		if self._turn_end_plan.changed:
			for (
				branch,
				turn,
			), plan_end_tick in self._turn_end_plan.changed.items():
				set_turn(branch, turn, turn_end[branch, turn], plan_end_tick)
			self._turn_end_plan.apply_changes()
		set_branch = self.query.set_branch
		if self._branches_d.changed:
			for branch, (
				parent,
				turn_start,
				tick_start,
				turn_end,
				tick_end,
			) in self._branches_d.changed.items():
				set_branch(
					branch, parent, turn_start, tick_start, turn_end, tick_end
				)
			self._branches_d.apply_changes()
		self.query.flush()

	@world_locked
	def commit(self, unload: bool = True) -> None:
		"""Write the state of all graphs and commit the transaction.

		Also saves the current branch, turn, and tick.

		Call with ``unload=False`` if you want to keep the written state in memory.

		"""
		self.query.eternal["branch"] = self._obranch
		self.query.eternal["turn"] = self._oturn
		self.query.eternal["tick"] = self._otick
		self.flush()
		self.query.commit()
		if unload:
			self.unload()

	def turns_when(
		self, qry: Query, mid_turn: bool = False
	) -> QueryResult | set:
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
		if isinstance(
			left, (EntityStatAccessor, CharacterStatAccessor)
		) and isinstance(right, (EntityStatAccessor, CharacterStatAccessor)):
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
		elif isinstance(left, (EntityStatAccessor, CharacterStatAccessor)):
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
		elif isinstance(right, (EntityStatAccessor, CharacterStatAccessor)):
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

	def _node_contents(self, character: CharName, node: NodeName) -> set:
		return self._node_contents_cache.retrieve(
			character, node, *self._btt()
		)

	def apply_choices(
		self,
		choices: list[dict],
		dry_run: bool = False,
		perfectionist: bool = False,
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

	def game_start(self) -> None:
		import importlib.machinery
		import importlib.util

		loader = importlib.machinery.SourceFileLoader(
			"game_start", os.path.join(self._prefix, "game_start.py")
		)
		spec = importlib.util.spec_from_loader("game_start", loader)
		game_start = importlib.util.module_from_spec(spec)
		loader.exec_module(game_start)
		game_start.game_start(self)

	@classmethod
	def from_archive(
		cls,
		archive_path: str | os.PathLike,
		prefix: str | os.PathLike = ".",
		**kwargs,
	) -> Engine:
		"""Make a new Lisien engine out of an archive exported from another engine"""
		shutil.unpack_archive(archive_path, prefix, "zip")
		extracted = os.listdir(prefix)
		if "world.xml" in extracted:
			xml_path = os.path.join(prefix, "world.xml")
			if "connect_string" in kwargs and kwargs["connect_string"]:
				xml_to_sqlite(xml_path, os.path.join(prefix, "world.sqlite3"))
			else:
				os.makedirs(os.path.join(prefix, "world"), exist_ok=True)
				xml_to_pqdb(xml_path, os.path.join(prefix, "world"))
		return Engine(prefix, **kwargs)

	def export(
		self,
		name: str | None = None,
		path: str | os.PathLike | None = None,
		indent: bool = True,
	) -> str | os.PathLike:
		if path is None:
			if name is None:
				raise ValueError(
					"Need a path to export to, or at least a name"
				)
			path = os.path.join(os.getcwd(), f"{name}.lisien")
		self.commit()
		with ZipFile(path, "w", ZIP_DEFLATED) as zf:
			with zf.open("world.xml", "w") as f:
				Exporter(self.query, self).write_xml(f, name, indent)
			if isinstance(self.string, StringStore):
				self.string.save()
				for lang in os.listdir(self.string._prefix):
					with (
						open(
							os.path.join(self.string._prefix, lang), "rb"
						) as inf,
						zf.open(os.path.join("strings", lang), "w") as outf,
					):
						outf.writelines(inf)
			elif isinstance(self.string, dict):
				import json

				for lang, strings in self.string.items():
					with (
						zf.open(f"strings/{lang}.json", "w") as outfb,
						TextIOWrapper(outfb) as outf,
					):
						json.dump(strings, outf, indent=2)
			else:
				self.error(
					f"Couldn't save strings; don't know how to save {type(self.string)}"
				)
			for store in self.stores:
				if store is self.string:
					continue
				if isinstance(store, FunctionStore):
					with (
						zf.open(store.__name__ + ".py", "w") as outfb,
						TextIOWrapper(outfb) as outf,
					):
						for _, function in store.iterplain():
							outf.write(function + "\n\n")
				elif hasattr(store, "__file__"):
					with (
						open(store.__file__, "rb") as inf,
						zf.open(os.path.basename(store.__file__), "w") as outf,
					):
						outf.writelines(inf)
				else:
					self.error(
						f"Couldn't export {store}, because we don't know what file it's in"
					)

		try:
			from androidstorage4kivy import SharedStorage
		except ModuleNotFoundError:
			return path
		if not hasattr(self, "_shared_storage"):
			self._shared_storage = SharedStorage()
		ss = self._shared_storage
		return ss.copy_to_shared(path)
