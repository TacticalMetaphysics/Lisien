# This file is part of Elide, frontend to Lisien, a framework for life simulation games.
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

import shutil
import sys
import os
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from concurrent.futures import wait as futwait
from functools import partial
from multiprocessing import Process, Pipe, Queue
from operator import itemgetter
from collections import defaultdict
from itertools import chain
from queue import SimpleQueue, Empty
from threading import Thread, Lock
from time import sleep
from types import FunctionType, ModuleType, MethodType
from typing import Dict, Union, Tuple, Any, Set, List, Type, Optional
from os import PathLike
from abc import ABC, abstractmethod
from random import Random
import zlib

import msgpack
import numpy as np
import networkx as nx
from networkx import (
	Graph,
	DiGraph,
	spring_layout,
	from_dict_of_dicts,
	from_dict_of_lists,
	NetworkXError,
)
from blinker import Signal

from .allegedb import ORM as gORM, KeyframeTuple, Key
from .allegedb import (
	StatDict,
	NodeValDict,
	EdgeValDict,
	DeltaDict,
	world_locked,
	OutOfTimelineError,
)
from .allegedb.cache import (
	KeyframeError,
	PickyDefaultDict,
	StructuredDefaultDict,
)
from .allegedb.window import update_window, update_backward_window
from .cache import PortalsRulebooksCache, CharactersRulebooksCache
from .util import sort_set, AbstractEngine, final_rule, normalize_layout
from .xcollections import (
	StringStore,
	FunctionStore,
	MethodStore,
	UniversalMapping,
)
from .query import (
	Query,
	_make_side_sel,
	StatusAlias,
	ComparisonQuery,
	CompoundQuery,
	QueryResultMidTurn,
	QueryResult,
	QueryResultEndTurn,
	CombinedQueryResult,
)
from .proxy import worker_subprocess
from .character import Character
from .facade import CharacterFacade
from .node import Place, Thing
from .portal import Portal
from .query import QueryEngine
from . import exc

SlightlyPackedDeltaType = Dict[
	bytes,
	Dict[
		bytes,
		Union[
			bytes,
			Dict[
				bytes,
				Union[bytes, Dict[bytes, Union[bytes, Dict[bytes, bytes]]]],
			],
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

	def __init__(self, engine: "AbstractEngine"):
		super().__init__()
		self.engine = engine


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

	def __call__(self) -> Tuple[List, DeltaDict]:
		engine = self.engine
		for store in engine.stores:
			if getattr(store, "_need_save", None):
				store.save()
		start_branch, start_turn, start_tick = engine._btt()
		latest_turn = engine._turns_completed[start_branch]
		if start_turn < latest_turn:
			engine.turn += 1
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
		if start_turn == latest_turn:
			# Pre-emptively nudge the loadedness and branch tracking,
			# so that lisien does not try to load an empty turn before every
			# loop of the rules engine
			turn0, tick0, turn1, tick1 = engine._loaded[start_branch]
			engine._loaded[start_branch] = (turn0, tick0, start_turn + 1, 0)
			parent, turn_from, tick_from, turn_to, tick_to = engine._branches[
				start_branch
			]
			engine._branches[start_branch] = (
				parent,
				turn_from,
				tick_from,
				start_turn + 1,
				0,
			)
			engine.turn += 1
		results = []
		with engine.advancing():
			for res in iter(engine._advance, final_rule):
				if res:
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
		engine._turns_completed[start_branch] = engine.turn
		engine.query.complete_turn(
			start_branch,
			engine.turn,
			discard_rules=not engine.keep_rules_journal,
		)
		if (
			engine.flush_interval is not None
			and engine.turn % engine.flush_interval == 0
		):
			engine.query.flush()
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


class Engine(AbstractEngine, gORM, Executor):
	"""lisien, the Life Simulator Engine.

	:param prefix: directory containing the simulation and its code;
		defaults to the working directory.
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
		``None`` to use the SQLite database in the ``prefix``.
	:param connect_args: dictionary of keyword arguments for the
		database connection
	:param schema: a Schema class that determines which changes to allow to
		the world; used when a player should not be able to change just
		anything. Defaults to :class:`NullSchema`.
	:param flush_interval: lisien will put pending changes into the database
		transaction every ``flush_interval`` turns. If ``None``, only flush
		on commit. Default ``None``.
	:param keyframe_interval: How many records to let through before automatically
		snapping a keyframe, default ``1000``. If ``None``, you'll need
		to call ``snap_keyframe`` yourself.
	:param commit_interval: lisien will commit changes to disk every
		``commit_interval`` turns. If ``None`` (the default), only commit
		on close or manual call to ``commit``.
	:param random_seed: a number to initialize the randomizer.
	:param logfun: an optional function taking arguments
		``level, message``, which should log `message` somehow.
	:param clear: whether to delete *any and all* existing data
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
	:param threaded_triggers: Whether to evaluate trigger functions in threads.
		This has performance benefits if you are using a free-threaded build of
		Python (without a GIL). Defaults to ``True`` when there are workers
		(see below), ``False`` otherwise.
	:param workers: How many subprocesses to use as workers for
		parallel processing. When ``None`` (the default), use as many
		subprocesses as we have CPU cores. When ``0``, parallel processing
		is not necessarily disabled; threads may still be used, which
		may or may not run in parallel, depending on your Python interpreter.
		However, note that ``workerss=0`` implies that trigger
		functions operate on bare lisien objects, and can therefore have
		side effects. If you don't want this, instead use
		``workers=1``, which *does* disable parallelism in the case
		of trigger functions.

	"""

	char_cls = Character
	thing_cls = Thing
	place_cls = node_cls = Place
	portal_cls = edge_cls = Portal
	query_engine_cls = QueryEngine
	illegal_graph_names = {
		"global",
		"eternal",
		"universal",
		"rulebooks",
		"rules",
	}
	illegal_node_names = {"nodes", "node_val", "edges", "edge_val", "things"}

	@property
	def eternal(self):
		return self.query.globl

	def __getattr__(self, item):
		meth = super().__getattribute__("method").__getattr__(item)
		return MethodType(meth, self)

	def __hasattr__(self, item):
		return hasattr(super().__getattribute__("method"), item)

	def __init__(
		self,
		prefix: Union[PathLike, str] = ".",
		*,
		string: Union[StringStore, dict] = None,
		trigger: Union[FunctionStore, ModuleType] = None,
		prereq: Union[FunctionStore, ModuleType] = None,
		action: Union[FunctionStore, ModuleType] = None,
		function: Union[FunctionStore, ModuleType] = None,
		method: Union[MethodStore, ModuleType] = None,
		main_branch: str = None,
		connect_string: str = None,
		connect_args: dict = None,
		schema_cls: Type[AbstractSchema] = NullSchema,
		flush_interval: int = None,
		keyframe_interval: Optional[int] = 1000,
		commit_interval: int = None,
		random_seed: int = None,
		logfun: callable = None,
		clear: bool = False,
		keep_rules_journal: bool = True,
		keyframe_on_close: bool = True,
		enforce_end_of_time: bool = True,
		threaded_triggers: bool = None,
		workers: int = None,
	):
		if logfun is None:
			from logging import getLogger

			logger = getLogger("lisien")

			def logfun(level, msg):
				if isinstance(level, int):
					logger.log(level, msg)
				else:
					getattr(logger, level)(msg)

		self.log = logfun
		self._prefix = prefix
		if connect_args is None:
			connect_args = {}
		if not os.path.exists(prefix):
			os.mkdir(prefix)
		if not os.path.isdir(prefix):
			raise FileExistsError("Need a directory")
		self.keep_rules_journal = keep_rules_journal
		self._keyframe_on_close = keyframe_on_close
		if string:
			self.string = string
		else:
			self._string_prefix = os.path.join(prefix, "strings")
			if clear and os.path.isdir(self._string_prefix):
				shutil.rmtree(self._string_prefix)
			if not os.path.exists(self._string_prefix):
				os.mkdir(self._string_prefix)
		for module, name in (
			(function, "function"),
			(method, "method"),
			(trigger, "trigger"),
			(prereq, "prereq"),
			(action, "action"),
		):
			if module:
				setattr(self, name, module)
			else:
				fn = os.path.join(prefix, f"{name}.py")
				setattr(self, f"_{name}_file", fn)
				if clear and os.path.exists(fn):
					os.remove(fn)
		self.schema = schema_cls(self)
		if connect_string:
			connect_string = connect_string.split("sqlite:///")[-1]
		super().__init__(
			connect_string or os.path.join(prefix, "world.db"),
			clear=clear,
			connect_args=connect_args,
			main_branch=main_branch,
			enforce_end_of_time=enforce_end_of_time,
		)
		self._things_cache.setdb = self.query.set_thing_loc
		self._universal_cache.setdb = self.query.universal_set
		self._rulebooks_cache.setdb = self.query.rulebook_set
		if hasattr(self, "_string_prefix"):
			self.string = StringStore(
				self.query,
				self._string_prefix,
				self.eternal.setdefault("language", "eng"),
			)
		self.next_turn = NextTurn(self)
		self.commit_interval = commit_interval
		self.query.keyframe_interval = keyframe_interval
		self.query.snap_keyframe = self.snap_keyframe
		self.query.kf_interval_override = self._detect_kf_interval_override
		self.flush_interval = flush_interval
		self._rando = Random()
		if "rando_state" in self.universal:
			self._rando.setstate(self.universal["rando_state"])
		else:
			self._rando.seed(random_seed)
			rando_state = self._rando.getstate()
			if self._oturn == self._otick == 0:
				self._universal_cache.store(
					"rando_state", self.branch, 0, 0, rando_state, loading=True
				)
				self.query.universal_set(
					"rando_state", self.branch, 0, 0, rando_state
				)
			else:
				self.universal["rando_state"] = rando_state
		if not self._keyframes_times:
			self._snap_keyframe_de_novo(*self._btt())
		if threaded_triggers is None:
			threaded_triggers = workers is not None and workers != 0
		if threaded_triggers:
			self._trigger_pool = ThreadPoolExecutor()
		if workers is None:
			workers = os.cpu_count() or 0
		if workers > 0:

			def sync_log_forever(q):
				while True:
					self.log(*q.get())

			for store in self.stores:
				store.save(reimport=False)

			branches_payload = zlib.compress(self.pack(self._branches))
			self._worker_last_eternal = dict(self.eternal.items())
			initial_payload = self._get_worker_kf_payload(-1)

			self._worker_processes = wp = []
			self._worker_inputs = wi = []
			self._worker_outputs = wo = []
			self._worker_locks = wlk = []
			self._worker_log_queues = wl = []
			self._worker_log_threads = wlt = []
			self._top_uid = 0
			for i in range(workers):
				inpipe_there, inpipe_here = Pipe(duplex=False)
				outpipe_here, outpipe_there = Pipe(duplex=False)
				logq = Queue()
				logthread = Thread(
					target=sync_log_forever, args=(logq,), daemon=True
				)
				proc = Process(
					target=worker_subprocess,
					args=(prefix, inpipe_there, outpipe_there, logq),
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
					inpipe_here.send_bytes(branches_payload)
					inpipe_here.send_bytes(initial_payload)
			self._how_many_futs_running = 0
			self._fut_manager_thread = Thread(
				target=self._manage_futs, daemon=True
			)
			self._futs_to_start: SimpleQueue[Future] = SimpleQueue()
			self._uid_to_fut: dict[int, Future] = {}
			self._fut_manager_thread.start()
			self.trigger.connect(self._reimport_trigger_functions)
			self.function.connect(self._reimport_worker_functions)
			self.method.connect(self._reimport_worker_methods)
			self._worker_updated_btts = [self._btt()] * workers
		self._rules_iter = self._follow_rules()

	def _call_in_subprocess(
		self, uid, method, func_name, future: Future, *args, **kwargs
	):
		i = uid % len(self._worker_inputs)
		argbytes = zlib.compress(
			self.pack((uid, method, (func_name, *args), kwargs))
		)
		with self._worker_locks[i]:
			self._update_worker_process_state(i)
			self._worker_inputs[i].send_bytes(argbytes)
			output = self._worker_outputs[i].recv_bytes()
		got_uid, result = self.unpack(zlib.decompress(output))
		assert got_uid == uid
		self._how_many_futs_running -= 1
		del self._uid_to_fut[uid]
		if isinstance(result, Exception):
			future.set_exception(result)
		else:
			future.set_result(result)

	def snap_keyframe(
		self, silent=False, update_worker_processes=True
	) -> Optional[dict]:
		ret = super().snap_keyframe(silent)
		if hasattr(self, "_worker_processes") and update_worker_processes:
			self._update_all_worker_process_states(clobber=True)
		return ret

	def submit(
		self, fn: Union[FunctionType, MethodType], /, *args, **kwargs
	) -> Future:
		if not hasattr(self, "_worker_processes"):
			raise RuntimeError("lisien was launched with no worker processes")
		if fn.__module__ == "function":
			method = "_call_function"
		elif fn.__module__ == "method":
			method = "_call_method"
		else:
			raise ValueError(
				"Function is not stored in this lisien engine. "
				"Use the engine's attributes `function` and `method` to store it."
			)
		uid = self._top_uid
		ret = Future()
		ret.uid = uid
		ret._t = Thread(
			target=self._call_in_subprocess,
			args=(uid, method, fn.__name__, ret, *args),
			kwargs=kwargs,
		)
		self._top_uid += 1
		self._uid_to_fut[uid] = ret
		self._futs_to_start.put(ret)
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
			with lock:  # deadlock here.
				pipein.send_bytes(b"shutdown")
				recvd = pipeout.recv_bytes()
				assert recvd == b"done", (
					f"expected 'done', got {self.unpack(zlib.decompress(recvd))}"
				)
				proc.join()
				proc.close()
				pipein.close()
				pipeout.close()

	def _detect_kf_interval_override(self):
		if getattr(self, "_no_kc", False):
			self._kf_overridden = True
			return True
		if getattr(self, "_kf_overridden", False):
			self._kf_overridden = False
			return False

	def _reimport_trigger_functions(self, *args, attr, **kwargs):
		if attr is not None:
			return
		payload = zlib.compress(self.pack((-1, "_reimport_triggers", (), {})))
		for lock, pipe in zip(self._worker_locks, self._worker_inputs):
			with lock:
				pipe.send_bytes(payload)

	def _reimport_worker_functions(self, *args, attr, **kwargs):
		if attr is not None:
			return
		payload = zlib.compress(self.pack((-1, "_reimport_functions", (), {})))
		for lock, pipe in zip(self._worker_locks, self._worker_inputs):
			with lock:
				pipe.send_bytes(payload)

	def _reimport_worker_methods(self, *args, attr, **kwargs):
		if attr is not None:
			return
		payload = zlib.compress(self.pack((-1, "_reimport_methods", (), {})))
		for lock, pipe in zip(self._worker_locks, self._worker_inputs):
			with lock:
				pipe.send_bytes(payload)

	def _get_worker_kf_payload(self, uid: int = -1) -> bytes:
		# I'm not using the uid at the moment, because this doesn't return anything
		return zlib.compress(
			self.pack(
				(
					uid,
					"_upd_from_game_start",
					(
						None,
						None,
						None,
						None,
						(
							self.snap_keyframe(update_worker_processes=False),
							dict(self.eternal.items()),
							dict(self.function.iterplain()),
							dict(self.method.iterplain()),
							dict(self.trigger.iterplain()),
							dict(self.prereq.iterplain()),
							dict(self.action.iterplain()),
						),
					),
					{},
				)
			)
		)

	def _call_a_subproxy(self, uid, method: str, *args, **kwargs):
		argbytes = zlib.compress(self.pack((uid, method, args, kwargs)))
		i = uid % len(self._worker_inputs)
		with self._worker_locks[i]:
			self._worker_inputs[i].send_bytes(argbytes)
			output = self._worker_outputs[i].recv_bytes()
		got_uid, ret = self.unpack(zlib.decompress(output))
		assert got_uid == uid
		if isinstance(ret, Exception):
			raise ret
		return ret

	def _call_any_subproxy(self, method: str, *args, **kwargs):
		uid = self._top_uid
		self._top_uid += 1
		return self._call_a_subproxy(uid, method, *args, **kwargs)

	def _call_every_subproxy(self, method: str, *args, **kwargs):
		ret = []
		for lock in self._worker_locks:
			lock.acquire()
		uids = []
		for _ in range(len(self._worker_processes)):
			uids.append(self._top_uid)
			argbytes = zlib.compress(
				self.pack((self._top_uid, method, args, kwargs))
			)
			i = self._top_uid % len(self._worker_processes)
			self._top_uid += 1
			self._worker_inputs[i].send_bytes(argbytes)
		for uid in uids:
			i = uid % len(self._worker_processes)
			outbytes = self._worker_outputs[i].recv_bytes()
			got_uid, retval = self.unpack(zlib.decompress(outbytes))
			assert got_uid == uid
			if isinstance(retval, Exception):
				raise retval
			ret.append(retval)
		for lock in self._worker_locks:
			lock.release()
		return ret

	def _init_graph(
		self,
		name: Key,
		type_s="DiGraph",
		data: Union[
			CharacterFacade, Graph, nx.Graph, dict, KeyframeTuple
		] = None,
	) -> None:
		if hasattr(data, "stat"):
			if not hasattr(data, "thing"):
				raise TypeError(
					"Need a graph like object or a keyframe tuple", type(data)
				)
			things = {
				thing: thing["location"] for thing in data.thing.values()
			}
			units = data.stat.get("units", {})
		elif hasattr(data, "graph"):
			things = {
				key: val["location"]
				for (key, val) in data.nodes.items()
				if "location" in val
			}
			units = data.graph.get("units", {})
		elif isinstance(data, tuple):
			things = {
				name: thing["location"]
				for name, thing in data[-3].items()
				if "location" in thing
			}
			units = data[-1].get("units", {})
		elif data is None:
			things = {}
			units = {}
		elif not isinstance(data, dict):
			raise TypeError("Need a graph like object or a keyframe tuple")
		else:
			things = {
				thing: thing["location"]
				for thing in data.get("node_val", {}).values()
				if "location" in thing
			}
			units = data.get("units", {})
		now = self._btt()
		conts = {}
		for thing, loc in things.items():
			if loc in conts:
				conts[loc].add(thing)
			else:
				conts[loc] = {thing}
		self._node_contents_cache.set_keyframe(
			(name,), *now, {k: frozenset(v) for (k, v) in conts.items()}
		)
		self._things_cache.set_keyframe((name,), *now, things)
		self._unitness_cache.set_keyframe((name,), *now, units)
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
		self.snap_keyframe(silent=True, update_worker_processes=False)
		super()._init_graph(name, type_s, data)
		if hasattr(self, "_worker_processes"):
			self._call_every_subproxy("_add_character", name, data)

	def _load_plans(self) -> None:
		from .rule import Rule

		q = self.query
		super()._load_plans()
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
		self._turns_completed.update(q.turns_completed_dump())
		self._rules_cache = {
			name: Rule(self, name, create=False) for name in q.rules_dump()
		}

	@world_locked
	def _load_between(
		self,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> None:
		"""Load time between two keyframes."""
		loaded = super()._load_between(
			branch, turn_from, tick_from, turn_to, tick_to
		)
		if universals := loaded.pop("universals"):
			self._universal_cache.load(universals)
		if rulebooks := loaded.pop("rulebooks"):
			self._rulebooks_cache.load(rulebooks)
		if rule_triggers := loaded.pop("rule_triggers"):
			self._triggers_cache.load(rule_triggers)
		if rule_prereqs := loaded.pop("rule_prereqs"):
			self._prereqs_cache.load(rule_prereqs)
		if rule_actions := loaded.pop("rule_actions"):
			self._actions_cache.load(rule_actions)
		if rule_neighborhoods := loaded.pop("rule_neighborhoods"):
			self._neighborhoods_cache.load(rule_neighborhoods)
		if rule_big := loaded.pop("rule_big"):
			self._rule_bigness_cache.load(rule_big)
		for graph, rowdict in loaded.items():
			if rowdict.get("things"):
				self._things_cache.load(rowdict["things"])
			if rowdict.get("character_rulebook"):
				self._characters_rulebooks_cache.load(
					rowdict["character_rulebook"]
				)
			if rowdict.get("unit_rulebook"):
				self._units_rulebooks_cache.load(rowdict["unit_rulebook"])
			if rowdict.get("character_thing_rulebook"):
				self._characters_things_rulebooks_cache.load(
					rowdict["character_thing_rulebook"]
				)
			if rowdict.get("character_place_rulebook"):
				self._characters_places_rulebooks_cache.load(
					rowdict["character_place_rulebook"]
				)
			if rowdict.get("character_portal_rulebook"):
				self._characters_portals_rulebooks_cache.load(
					rowdict["character_portal_rulebook"]
				)
			if rowdict.get("node_rulebook"):
				self._nodes_rulebooks_cache.load(rowdict["node_rulebook"])
			if rowdict.get("portal_rulebook"):
				self._portals_rulebooks_cache.load(rowdict["portal_rulebook"])
		return loaded

	def _load(
		self,
		latest_past_keyframe: Optional[Tuple[str, int, int]],
		earliest_future_keyframe: Optional[Tuple[str, int, int]],
		graphs_rows: list,
		loaded: dict,
	) -> None:
		"""Load history data at the given time

		Will load the keyframe prior to that time, and all history
		data following, up to (but not including) the keyframe thereafter.

		"""
		if latest_past_keyframe:
			self._get_keyframe(*latest_past_keyframe)

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
		super()._load(
			latest_past_keyframe, earliest_future_keyframe, graphs_rows, loaded
		)

	def _init_caches(self) -> None:
		from .xcollections import (
			FunctionStore,
			CharacterMapping,
		)
		from .cache import (
			NodeContentsCache,
			InitializedCache,
			InitializedEntitylessCache,
			UnitnessCache,
			UnitRulesHandledCache,
			CharacterThingRulesHandledCache,
			CharacterPlaceRulesHandledCache,
			CharacterPortalRulesHandledCache,
			NodeRulesHandledCache,
			PortalRulesHandledCache,
			CharacterRulesHandledCache,
			ThingsCache,
		)
		from .allegedb.cache import EntitylessCache
		from .rule import AllRuleBooks, AllRules

		super()._init_caches()
		self._neighbors_cache = {}
		self._things_cache = ThingsCache(self)
		self._node_contents_cache = NodeContentsCache(self)
		self.character = self.graph = CharacterMapping(self)
		self._universal_cache = EntitylessCache(self, "universal_cache")
		self._rulebooks_cache = InitializedEntitylessCache(
			self, "rulebooks_cache"
		)
		self._characters_rulebooks_cache = CharactersRulebooksCache(
			self, "characters_rulebooks_cache"
		)
		self._units_rulebooks_cache = CharactersRulebooksCache(
			self, "units_rulebooks_cache"
		)
		self._characters_things_rulebooks_cache = CharactersRulebooksCache(
			self, "characters_things_rulebooks_cache"
		)
		self._characters_places_rulebooks_cache = CharactersRulebooksCache(
			self, "characters_places_rulebooks_cache"
		)
		self._characters_portals_rulebooks_cache = CharactersRulebooksCache(
			self, "characters_portals_rulebooks_cache"
		)
		self._nodes_rulebooks_cache = InitializedCache(
			self, "nodes_rulebooks_cache"
		)
		self._portals_rulebooks_cache = PortalsRulebooksCache(
			self, "portals_rulebooks_cache"
		)
		self._triggers_cache = InitializedEntitylessCache(
			self, "triggers_cache"
		)
		self._prereqs_cache = InitializedEntitylessCache(self, "prereqs_cache")
		self._actions_cache = InitializedEntitylessCache(self, "actions_cache")
		self._neighborhoods_cache = InitializedEntitylessCache(
			self, "neighborhoods_cache"
		)
		self._rule_bigness_cache = InitializedEntitylessCache(
			self, "rule_bigness_cache"
		)
		self._node_rules_handled_cache = NodeRulesHandledCache(self)
		self._portal_rules_handled_cache = PortalRulesHandledCache(self)
		self._character_rules_handled_cache = CharacterRulesHandledCache(
			self, "character_rules_handled_cache"
		)
		self._unit_rules_handled_cache = UnitRulesHandledCache(
			self, "unit_rules_handled_cache"
		)
		self._character_thing_rules_handled_cache = (
			CharacterThingRulesHandledCache(
				self, "character_thing_rules_handled_cache"
			)
		)
		self._character_place_rules_handled_cache = (
			CharacterPlaceRulesHandledCache(
				self, "character_place_rules_handled_cache"
			)
		)
		self._character_portal_rules_handled_cache = (
			CharacterPortalRulesHandledCache(
				self, "character_portal_rules_handled_cache"
			)
		)
		self._unitness_cache = UnitnessCache(self)
		self._turns_completed = defaultdict(lambda: max((0, self.turn - 1)))
		self._turns_completed_previous = self._turns_completed.copy()
		"""The last turn when the rules engine ran in each branch"""
		self.universal = UniversalMapping(self)
		if hasattr(self, "_action_file"):
			self.action = FunctionStore(self._action_file)
		if hasattr(self, "_prereq_file"):
			self.prereq = FunctionStore(self._prereq_file)
		if hasattr(self, "_trigger_file"):
			self.trigger = FunctionStore(self._trigger_file)
		if hasattr(self, "_function_file"):
			self.function = FunctionStore(self._function_file)
		if hasattr(self, "_method_file"):
			self.method = FunctionStore(self._method_file)
		self.rule = AllRules(self)
		self.rulebook = AllRuleBooks(self)
		self._caches += [
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
			# rules handled caches not here because they don't really unload
			self._unitness_cache,
		]

	def _load_graphs(self) -> None:
		for charn, branch, turn, tick, typ in self.query.characters():
			self._graph_cache.store(
				charn, branch, turn, tick, (typ if typ != "Deleted" else None)
			)
			self._graph_objs[charn] = self.char_cls(
				self, charn, init_rulebooks=False
			)

	def _make_node(
		self, graph: Character, node: Key
	) -> Union[thing_cls, place_cls]:
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

	def _copy_kf(self, branch_from, branch_to, turn, tick):
		super()._copy_kf(branch_from, branch_to, turn, tick)
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
			self._nodes_rulebooks_cache,
			self._portals_rulebooks_cache,
			self._neighborhoods_cache,
			self._rule_bigness_cache,
		):
			cache.copy_keyframe(branch_from, branch_to, turn, tick)

		for character in self._graph_cache.iter_entities(
			branch_from, turn, tick
		):
			loc_kf = self._things_cache.get_keyframe(
				(character,), branch_from, turn, tick, copy=True
			)
			conts_kf = self._node_contents_cache.get_keyframe(
				(character,), branch_from, turn, tick, copy=True
			)
			units_kf = self._unitness_cache.get_keyframe(
				(character,), branch_from, turn, tick, copy=True
			)
			self._things_cache.set_keyframe(
				(character,), branch_to, turn, tick, loc_kf
			)
			self._node_contents_cache.set_keyframe(
				(character,), branch_to, turn, tick, conts_kf
			)
			self._unitness_cache.set_keyframe(
				(character,), branch_to, turn, tick, units_kf
			)
			self._nodes_rulebooks_cache.set_keyframe(
				(character,),
				branch_to,
				turn,
				tick,
				self._nodes_rulebooks_cache.get_keyframe(
					(character,), branch_from, turn, tick
				),
			)
		self.query.keyframe_extension_insert(
			branch_to,
			turn,
			tick,
			self._universal_cache.get_keyframe(branch_to, turn, tick),
			{
				"triggers": self._triggers_cache.get_keyframe(
					branch_to, turn, tick
				),
				"prereqs": self._prereqs_cache.get_keyframe(
					branch_to, turn, tick
				),
				"actions": self._actions_cache.get_keyframe(
					branch_to, turn, tick
				),
				"neighborhood": self._neighborhoods_cache.get_keyframe(
					branch_to, turn, tick
				),
				"big": self._rule_bigness_cache.get_keyframe(
					branch_to, turn, tick
				),
			},
			self._rulebooks_cache.get_keyframe(branch_to, turn, tick),
		)

	def _get_kf(
		self, branch: str, turn: int, tick: int, copy: bool = True
	) -> dict:
		kf = super()._get_kf(branch, turn, tick, copy=copy)
		for graph, vals in kf["graph_val"].items():
			vals["units"] = self._unitness_cache.get_keyframe(
				(graph,), branch, turn, tick
			)
			vals["character_rulebook"] = (
				self._characters_rulebooks_cache.retrieve(
					graph, branch, turn, tick
				)
			)
			vals["unit_rulebook"] = self._units_rulebooks_cache.retrieve(
				graph, branch, turn, tick
			)
			vals["character_thing_rulebook"] = (
				self._characters_things_rulebooks_cache.retrieve(
					graph, branch, turn, tick
				)
			)
			vals["character_place_rulebook"] = (
				self._characters_places_rulebooks_cache.retrieve(
					graph, branch, turn, tick
				)
			)
			vals["character_portal_rulebook"] = (
				self._characters_portals_rulebooks_cache.retrieve(
					graph, branch, turn, tick
				)
			)
			try:
				node_rb_kf = self._nodes_rulebooks_cache.get_keyframe(
					(graph,), branch, turn, tick
				)
			except KeyframeError:
				node_rb_kf = {
					node: (graph, node) for node in kf["nodes"][graph]
				}
			for node, rb in node_rb_kf.items():
				if graph in kf["node_val"]:
					if node in kf["node_val"][graph]:
						kf["node_val"][graph][node]["rulebook"] = rb
					else:
						kf["node_val"][graph][node] = {"rulebook": rb}
				else:
					kf["node_val"] = {graph: {node: {"rulebook": rb}}}
			try:
				port_rb_kf = self._portals_rulebooks_cache.get_keyframe(
					(graph,), branch, turn, tick
				)
			except KeyframeError:
				port_rb_kf = {}
				for orig, dests in kf["edges"][graph].items():
					destrbs = port_rb_kf[orig] = {}
					for dest, ex in dests.items():
						if ex:
							destrbs[dest] = (graph, orig, dest)
			for orig, dests in port_rb_kf.items():
				for dest, rb in dests.items():
					if graph in kf["edge_val"]:
						if orig in kf["edge_val"][graph]:
							if dest in kf["edge_val"][graph][orig]:
								assert (
									graph in kf["edges"]
									and orig in kf["edges"][graph]
									and dest in kf["edges"][graph][orig]
								)
								kf["edge_val"][graph][orig][dest][
									"rulebook"
								] = rb
							elif (
								graph in kf["edges"]
								and orig in kf["edges"][graph]
								and dest in kf["edges"][graph][orig]
							):
								kf["edge_val"][graph][orig][dest] = {
									"rulebook": rb
								}
						elif (
							graph in kf["edges"] and orig in kf["edges"][graph]
						):
							kf["edge_val"][graph][orig] = {
								dest: {"rulebook": rb}
							}
					elif (
						graph in kf["edges"]
						and orig in kf["edges"][graph]
						and dest in kf["edges"][graph][orig]
					):
						kf["edge_val"][graph] = {
							orig: {dest: {"rulebook": rb}}
						}
			try:
				locs_kf = self._things_cache.get_keyframe(
					(graph,), branch, turn, tick
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
				kf["node_val"] = {graph: locs_kf}
			elif graph not in kf["node_val"]:
				kf["node_val"][graph] = locs_kf
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

	def _get_keyframe(
		self, branch: str, turn: int, tick: int, copy=True, silent=False
	):
		if (branch, turn, tick) in self._keyframes_loaded:
			if silent:
				return
			return self._get_kf(branch, turn, tick, copy=copy)
		univ, rule, rulebook = self.query.get_keyframe_extensions(
			branch, turn, tick
		)
		self._universal_cache.set_keyframe(branch, turn, tick, univ)
		self._triggers_cache.set_keyframe(branch, turn, tick, rule["triggers"])
		self._prereqs_cache.set_keyframe(branch, turn, tick, rule["prereqs"])
		self._actions_cache.set_keyframe(branch, turn, tick, rule["actions"])
		self._neighborhoods_cache.set_keyframe(
			branch, turn, tick, rule.get("neighborhood", {})
		)
		self._rule_bigness_cache.set_keyframe(branch, turn, tick, rule["big"])
		self._rulebooks_cache.set_keyframe(branch, turn, tick, rulebook)

		ret = super()._get_keyframe(
			branch, turn, tick, copy=copy, silent=False
		)

		charrbkf = {}
		unitrbkf = {}
		charthingrbkf = {}
		charplacerbkf = {}
		charportrbkf = {}
		for graph, graphval in ret["graph_val"].items():
			if "character_rulebook" in graphval:
				charrbkf[graph] = graphval["character_rulebook"]
			if "unit_rulebook" in graphval:
				unitrbkf[graph] = graphval["unit_rulebook"]
			if "character_thing_rulebook" in graphval:
				charthingrbkf[graph] = graphval["character_thing_rulebook"]
			if "character_place_rulebook" in graphval:
				charplacerbkf[graph] = graphval["character_place_rulebook"]
			if "character_portal_rulebook" in graphval:
				charportrbkf[graph] = graphval["character_portal_rulebook"]
			if "units" in graphval:
				self._unitness_cache.set_keyframe(
					(graph,), branch, turn, tick, graphval["units"]
				)
			else:
				self._unitness_cache.set_keyframe(
					(graph,), branch, turn, tick, {}
				)
			if graph in ret["node_val"]:
				locs = {}
				conts = {}
				for node, val in ret["node_val"][graph].items():
					if "location" not in val:
						continue
					locs[node] = location = val["location"]
					if location in conts:
						conts[location].add(node)
					else:
						conts[location] = {node}
				self._things_cache.set_keyframe(
					(graph,), branch, turn, tick, locs
				)
				self._node_contents_cache.set_keyframe(
					(graph,),
					branch,
					turn,
					tick,
					{k: frozenset(v) for (k, v) in conts.items()},
				)
			else:
				self._things_cache.set_keyframe(
					(graph,), branch, turn, tick, {}
				)
				self._node_contents_cache.set_keyframe(
					(graph,), branch, turn, tick, {}
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
		time_from: Tuple[str, int, int],
		time_to: Tuple[str, int, int],
		slow=False,
	) -> DeltaDict:
		"""Get a dictionary describing changes to the world.

		Most keys will be character names, and their values will be
		dictionaries of the character's stats' new values, with ``None``
		for deleted keys. Characters' dictionaries have special keys
		'nodes' and 'edges' which contain booleans indicating whether
		the node or edge exists at the moment, and 'node_val' and
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
		self, btt_from: Tuple[str, int, int], btt_to: Tuple[str, int, int]
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

		delta: Dict[bytes, Any] = {
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

	def _get_branch_delta(
		self,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> DeltaDict:
		if not isinstance(branch, str):
			raise TypeError("branch must be str")
		for arg in (turn_from, tick_from, turn_to, tick_to):
			if not isinstance(arg, int):
				raise TypeError("turn and tick must be int")
		if turn_from == turn_to:
			if tick_from == tick_to:
				return {}
			return self._get_turn_delta(branch, turn_to, tick_from, tick_to)
		delta = super()._get_branch_delta(
			branch, turn_from, tick_from, turn_to, tick_to
		)
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
		branch: str = None,
		turn: int = None,
		tick_from=0,
		tick_to: int = None,
	) -> DeltaDict:
		"""Get a dictionary of changes to the world within a given turn

		Defaults to the present turn, and stops at the present tick
		unless specified.

		See the documentation for ``get_delta`` for a detailed
		description of the delta format.

		:arg branch: branch of history, defaulting to the present branch
		:arg turn: turn within the branch, defaulting to the present
				   turn
		:arg tick_from: tick at which to start the delta, default 0
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
		delta = super()._get_turn_delta(branch, turn, tick_from, tick_to)
		if tick_from < tick_to:
			attribute = "settings"
			tick_to += 1
		else:
			attribute = "presettings"
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

	def flush(self):
		__doc__ = gORM.flush.__doc__
		super().flush()
		turns_completed_previous = self._turns_completed_previous
		turns_completed = self._turns_completed
		set_turn_completed = self.query.set_turn_completed
		for branch, turn_late in turns_completed.items():
			turn_early = turns_completed_previous.get(branch)
			if turn_late != turn_early:
				if turn_early is not None and turn_late <= turn_early:
					raise RuntimeError("Incoherent turns_completed cache")
				set_turn_completed(branch, turn_late)
		self._turns_completed_previous = turns_completed.copy()

	def close(self) -> None:
		"""Commit changes and close the database

		This will be useless thereafter.

		"""
		if hasattr(self, "_closed"):
			raise RuntimeError("Already closed")
		if (
			self._keyframe_on_close
			and self._btt() not in self._keyframes_times
		):
			self.snap_keyframe(silent=True, update_worker_processes=False)
		for store in self.stores:
			if hasattr(store, "save"):
				store.save(reimport=False)
			if not hasattr(store, "_filename"):
				continue
			path, filename = os.path.split(store._filename)
			modname = filename[:-3]
			if modname in sys.modules:
				del sys.modules[modname]
		self.commit()
		self.query.close()
		self.shutdown()
		self._closed = True

	def _exist_node(self, character: Key, node: Key, exist=True) -> None:
		super()._exist_node(character, node, exist)
		if exist:
			btt = self._btt()
			self._nodes_rulebooks_cache.store(
				character, node, *btt, (character, node)
			)
			self.query.set_node_rulebook(
				character, node, *btt, (character, node)
			)

	def _exist_edge(
		self, character: Key, orig: Key, dest: Key, idx=0, exist=True
	) -> None:
		super()._exist_edge(character, orig, dest, idx, exist)
		if exist:
			btt = self._btt()
			self._portals_rulebooks_cache.store(
				character, orig, dest, *btt, (character, orig, dest)
			)
			self.query.set_portal_rulebook(
				character, orig, dest, *btt, (character, orig, dest)
			)

	def _snap_keyframe_from_delta(
		self,
		then: Tuple[str, int, int],
		now: Tuple[str, int, int],
		delta: DeltaDict,
	) -> None:
		if then == now:
			return
		b, r, t = then
		branch, turn, tick = now
		univ = self._universal_cache.get_keyframe(b, r, t, copy=True)
		rbs = self._rulebooks_cache.get_keyframe(b, r, t, copy=True)
		for k, v in delta.pop("universal", {}).items():
			if v is None:
				if k in univ:
					del univ[k]
			else:
				univ[k] = v
		rbs.update(delta.pop("rulebooks", {}))
		trigs = self._triggers_cache.get_keyframe(b, r, t, copy=True)
		preqs = self._prereqs_cache.get_keyframe(b, r, t, copy=True)
		acts = self._actions_cache.get_keyframe(b, r, t, copy=True)
		nbrs = self._neighborhoods_cache.get_keyframe(b, r, t, copy=True)
		bigs = self._rule_bigness_cache.get_keyframe(b, r, t, copy=True)
		for rule, funcs in delta.pop("rules", {}).items():
			trigs[rule] = funcs.get("triggers", trigs.get(rule, ()))
			preqs[rule] = funcs.get("prereqs", preqs.get(rule, ()))
			acts[rule] = funcs.get("actions", acts.get(rule, ()))
			if "neighborhood" in funcs:
				nbrs[rule] = funcs["neighborhood"]
			if "big" in funcs:
				bigs[rule] = funcs["big"]
		charrbs = self._characters_rulebooks_cache.get_keyframe(*then)
		unitrbs = self._units_rulebooks_cache.get_keyframe(*then)
		thingrbs = self._characters_things_rulebooks_cache.get_keyframe(*then)
		placerbs = self._characters_places_rulebooks_cache.get_keyframe(*then)
		charportrbs = self._characters_portals_rulebooks_cache.get_keyframe(
			*then
		)
		for graph in (
			set(self._graph_cache.iter_keys(b, r, t)).union(delta.keys())
			- self.illegal_graph_names
		):
			delt = delta.get(graph, {})
			if delt is None:
				continue
			noderbs = self._nodes_rulebooks_cache.get_keyframe((graph,), *then)
			portrbs = self._portals_rulebooks_cache.get_keyframe(
				(graph,), *then
			)
			charunit = self._unitness_cache.get_keyframe(
				(graph,), b, r, t, copy=True
			)
			if "units" in delt and delt["units"]:
				for graf, units in delt["units"].items():
					if graf in charunit:
						for unit, ex in units.items():
							if ex:
								charunit[graf][unit] = True
					else:
						charunit[graf] = {
							unit: True for (unit, ex) in units.items() if ex
						}
			self._unitness_cache.set_keyframe((graph,), *now, charunit)
			self._unitness_cache.get_keyframe((graph,), *now)
			if "character_rulebook" in delt:
				charrbs[graph] = delt["character_rulebook"]
			if "unit_rulebook" in delt:
				unitrbs[graph] = delt["unit_rulebook"]
			if "character_thing_rulebook" in delt:
				thingrbs[graph] = delt["character_thing_rulebook"]
			if "character_place_rulebook" in delt:
				placerbs[graph] = delt["character_place_rulebook"]
			if "character_portal_rulebook" in delt:
				charportrbs[graph] = delt["character_portal_rulebook"]
			locs = self._things_cache.get_keyframe(
				(graph,), b, r, t, copy=True
			)
			conts_kf = self._node_contents_cache.get_keyframe(
				(graph,), b, r, t, copy=True
			)
			conts = {key: set(value) for (key, value) in conts_kf.items()}
			if "node_val" in delt:
				node_kf = self._nodes_cache.get_keyframe((graph,), b, r, t)
				for node in node_kf.keys() | delt.get("node_val", {}).keys():
					if node in delt["node_val"]:
						delt_val = delt["node_val"][node]
						if "rulebook" in delt_val:
							noderbs[node] = delt_val.pop("rulebook")
						if "location" in delt_val:
							if node in locs:
								oldloc = locs[node]
								if oldloc in conts:
									conts[oldloc].discard(node)
							locs[node] = loc = delt_val["location"]
							if loc in conts:
								conts[loc].add(node)
							else:
								conts[loc] = {node}
			if "edge_val" in delt:
				port_kf = self._edges_cache.get_keyframe((graph,), b, r, t)

				def port_in_kf(orig, dest):
					return (
						orig in port_kf
						and dest in port_kf[orig]
						and port_kf[orig][dest]
					)

				def port_in_delt(orig, dest):
					return (
						"edges" in delt
						and orig in delt["edges"]
						and dest in delt["edges"][orig]
						and delt["edges"][orig][dest]
					)

				for orig in port_kf.keys() | delt["edge_val"].keys():
					if orig in port_kf:
						dests = port_kf[orig]
						for dest, val in dests.items():
							if "rulebook" in val:
								rulebook = val.pop("rulebook")
								if rulebook is None:
									continue
								elif orig in portrbs:
									portrbs[orig][dest] = rulebook
								else:
									portrbs[orig] = {dest: rulebook}
					else:
						portrbs[orig] = {
							dest: kvs["rulebook"]
							for dest, kvs in delt["edge_val"][orig].items()
							if (
								port_in_kf(orig, dest)
								or port_in_delt(orig, dest)
							)
							and "rulebook" in kvs
						}

			conts = {key: frozenset(value) for (key, value) in conts.items()}
			self._things_cache.set_keyframe((graph,), *now, locs)
			self._node_contents_cache.set_keyframe((graph,), *now, conts)
			self._nodes_rulebooks_cache.set_keyframe(
				(graph,), branch, turn, tick, noderbs
			)
			self._portals_rulebooks_cache.set_keyframe(
				(graph,), branch, turn, tick, portrbs
			)
		self._characters_rulebooks_cache.set_keyframe(
			branch, turn, tick, charrbs
		)
		self._units_rulebooks_cache.set_keyframe(branch, turn, tick, unitrbs)
		self._characters_things_rulebooks_cache.set_keyframe(
			branch, turn, tick, thingrbs
		)
		self._characters_places_rulebooks_cache.set_keyframe(
			branch, turn, tick, placerbs
		)
		self._characters_portals_rulebooks_cache.set_keyframe(
			branch, turn, tick, charportrbs
		)
		self._universal_cache.set_keyframe(branch, turn, tick, univ)
		self._triggers_cache.set_keyframe(branch, turn, tick, trigs)
		self._prereqs_cache.set_keyframe(branch, turn, tick, preqs)
		self._actions_cache.set_keyframe(branch, turn, tick, acts)
		self._neighborhoods_cache.set_keyframe(branch, turn, tick, nbrs)
		self._rule_bigness_cache.set_keyframe(branch, turn, tick, bigs)
		self._rulebooks_cache.set_keyframe(branch, turn, tick, rbs)
		self.query.keyframe_extension_insert(
			*now,
			univ,
			{
				"triggers": trigs,
				"prereqs": preqs,
				"actions": acts,
				"neighborhood": nbrs,
				"big": bigs,
			},
			rbs,
		)
		super()._snap_keyframe_from_delta(then, now, delta)

	def __enter__(self):
		"""Return myself. For compatibility with ``with`` semantics."""
		return self

	def __exit__(self, *args):
		"""Close on exit."""
		self.close()

	def _set_branch(self, v: str) -> None:
		if not isinstance(v, str):
			raise TypeError("Branch names must be strings")
		oldrando = self.universal.get("rando_state")
		super()._set_branch(v)
		if v not in self._turns_completed:
			self._turns_completed[v] = self.turn
		newrando = self.universal.get("rando_state")
		if newrando and newrando != oldrando:
			self._rando.setstate(newrando)
		self.time.send(self.time, branch=self._obranch, turn=self._oturn)

	def _set_turn(self, v: int) -> None:
		if not isinstance(v, int):
			raise TypeError("Turns must be integers")
		if v < 0:
			raise ValueError("Turns can't be negative")
		turn_end = self._branches[self.branch][3]
		if not self._planning and v > turn_end + 1:
			raise OutOfTimelineError(
				f"The turn {v} is after the end of the branch {self.branch}. "
				f"Go to turn {turn_end + 1} and simulate with `next_turn`.",
				self.branch,
				self.turn,
				self.tick,
				self.branch,
				v,
				self.tick,
			)
		oldrando = self.universal.get("rando_state")
		oldturn = self._oturn
		super()._set_turn(v)
		newrando = self.universal.get("rando_state")
		if v > oldturn and newrando and newrando != oldrando:
			self._rando.setstate(newrando)
		self.time.send(self.time, branch=self._obranch, turn=self._oturn)

	def _set_tick(self, v: int) -> None:
		if not isinstance(v, int):
			raise TypeError("Ticks must be integers")
		if v < 0:
			raise ValueError("Ticks can't be negative")
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
		oldtick = self._otick
		super()._set_tick(v)
		newrando = self.universal.get("rando_state")
		if v > oldtick and newrando and newrando != oldrando:
			self._rando.setstate(newrando)

	def _handled_char(
		self,
		charn: Key,
		rulebook: Key,
		rulen: Key,
		branch: str,
		turn: int,
		tick: int,
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
		branch: str,
		turn: int,
		tick: int,
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
		branch: str,
		turn: int,
		tick: int,
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
		branch: str,
		turn: int,
		tick: int,
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
		branch: str,
		turn: int,
		tick: int,
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
		branch: str,
		turn: int,
		tick: int,
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
		branch: str,
		turn: int,
		tick: int,
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

	def _update_all_worker_process_states(self, clobber=False):
		for lock in self._worker_locks:
			lock.acquire()
		kf_payload = None
		deltas = {}
		for i in range(len(self._worker_processes)):
			branch_from, turn_from, tick_from = self._worker_updated_btts[i]
			old_eternal = self._worker_last_eternal
			new_eternal = self._worker_last_eternal = dict(
				self.eternal.items()
			)
			eternal_delta = {
				k: new_eternal.get(k)
				for k in old_eternal.keys() | new_eternal.keys()
				if old_eternal.get(k) != new_eternal.get(k)
			}
			if not clobber and branch_from == self.branch:
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
				argbytes = zlib.compress(
					self.pack(
						(
							-1,
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
				self._worker_inputs[i].send_bytes(argbytes)
			else:
				if kf_payload is None:
					kf_payload = self._get_worker_kf_payload(-1)
				self._worker_inputs[i].send_bytes(kf_payload)
			self._worker_updated_btts[i] = self._btt()
		for lock in self._worker_locks:
			lock.release()

	def _update_worker_process_state(self, i):
		branch_from, turn_from, tick_from = self._worker_updated_btts[i]
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
			argbytes = zlib.compress(
				self.pack(
					(
						-1,
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
			self._worker_inputs[i].send_bytes(argbytes)
		else:
			self._worker_inputs[i].send_bytes(self._get_worker_kf_payload())
		self._worker_updated_btts[i] = self._btt()

	def _follow_rules(self):
		# TODO: roll back changes done by rules that raise an exception
		# TODO: if there's a paradox while following some rule,
		#  start a new branch, copying handled rules
		from collections import defaultdict

		thing_cls = self.thing_cls
		place_cls = self.place_cls
		portal_cls = self.portal_cls

		branch, turn, tick = self._btt()
		charmap = self.character
		rulemap = self.rule
		pool = getattr(self, "_trigger_pool", None)
		if pool:
			submit = pool.submit
		else:
			submit = partial
		todo = defaultdict(list)

		def changed(entity: tuple) -> bool:
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
			if turn <= self._branches[branch][1]:
				branch = self._branches[branch][0]
				assert branch is not None
			if branch not in vbranches:
				return False
			vbranchesb = vbranches[branch]
			if turn not in vbranchesb:
				return False
			return entikey in vbranchesb[turn].entikeys

		if hasattr(self, "_worker_processes") and self.turn > 0:
			self._update_all_worker_process_states()
			# Now we can evaluate trigger functions in the worker processes,
			# in parallel.

		def check_triggers(
			prio, rulebook, rule, handled_fun, entity, neighbors=None
		):
			if neighbors is not None and not (
				any(changed(neighbor) for neighbor in neighbors)
			):
				return False
			for trigger in rule.triggers:
				if hasattr(self, "_worker_processes"):
					res = self._call_any_subproxy(
						"_eval_trigger", trigger.__name__, entity
					)
				else:
					res = trigger(entity)
				if res:
					todo[prio, rulebook].append((rule, handled_fun, entity))
					return True
			else:
				handled_fun(self.tick)
				return False

		def check_prereqs(rule, handled_fun, entity):
			if not entity:
				return False
			for prereq in rule.prereqs:
				res = prereq(entity)
				if not res:
					handled_fun(self.tick)
					return False
			return True

		def do_actions(rule, handled_fun, entity):
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

		truthfun = self.trigger.truth

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
			if truthfun in self.rulebook[rulebook]:
				todo[prio, rulebook].append((rule, handled, entity))
				continue
			trig_futs.append(
				submit(
					check_triggers, prio, rulebook, rule, handled, entity, None
				)
			)

		avcache_retr = self._unitness_cache._base_retrieve
		node_exists = self._node_exists
		make_node = self._make_node
		node_objs = self._node_objs

		def get_neighbors(
			entity: Union[place_cls, thing_cls, portal_cls],
			neighborhood: Optional[int],
		) -> Optional[List[Union[Tuple[Key], Tuple[Key, Key]]]]:
			"""Get a list of neighbors within the neighborhood

			Neighbors are given by a tuple containing only their name,
			if they are Places or Things, or their origin's and destination's
			names, if they are Portals.

			"""
			charn = entity.character.name
			btt = self._btt()

			def get_place_neighbors(name: Key) -> Set[Key]:
				seen: Set[Key] = set()
				for succ in self._edges_cache.iter_successors(
					charn, name, *btt
				):
					seen.add(succ)
				for pred in self._edges_cache.iter_predecessors(
					charn, name, *btt
				):
					seen.add(pred)
				return seen

			def get_place_contents(name: Key) -> Set[Key]:
				try:
					return self._node_contents_cache.retrieve(
						charn, name, *btt
					)
				except KeyError:
					return set()

			def get_place_portals(name: Key) -> Set[Tuple[Key, Key]]:
				seen: Set[Tuple[Key, Key]] = set()
				seen.update(
					(name, dest)
					for dest in self._edges_cache.iter_successors(
						charn, name, *btt
					)
				)
				seen.update(
					(orig, name)
					for orig in self._edges_cache.iter_predecessors(
						charn, name, *btt
					)
				)
				return seen

			def get_thing_location_tup(name: Key) -> Union[(), Tuple[Key]]:
				try:
					return (self._things_cache.retrieve(charn, name, *btt),)
				except KeyError:
					return ()

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
								get_place_neighbors(placen),
								get_place_contents(placen),
								get_thing_location_tup(placen),
							):
								if neighbor_place not in seen:
									neighbors.append((neighbor_place,))
									seen.add(neighbor_place)
								for neighbor_thing in get_place_contents(
									neighbor_place
								):
									if neighbor_thing not in seen:
										neighbors.append((neighbor_thing,))
										seen.add(neighbor_thing)
							for neighbor_portal in get_place_portals(placen):
								if neighbor_portal not in seen:
									neighbors.append(neighbor_portal)
									seen.add(neighbor_portal)
					else:
						(neighbor,) = neighbor
						for neighbor_place in chain(
							get_place_neighbors(neighbor),
							get_place_contents(neighbor),
							get_thing_location_tup(neighbor),
						):
							if neighbor_place not in seen:
								neighbors.append((neighbor_place,))
								seen.add(neighbor_place)
							for neighbor_thing in get_place_contents(
								neighbor_place
							):
								if neighbor_thing not in seen:
									neighbors.append((neighbor_thing,))
									seen.add(neighbor_thing)
						for neighbor_portal in get_place_portals(neighbor):
							if neighbor_portal not in seen:
								neighbors.append(neighbor_portal)
								seen.add(neighbor_portal)
				i = j
			self._neighbors_cache[cache_key] = neighbors
			return neighbors

		def get_effective_neighbors(entity, neighborhood):
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
				self._load_at(branch_now, turn_now - 1, 0)
				self._oturn -= 1
				self._otick = 0
				last_turn_neighbors = get_neighbors(entity, neighborhood)
				self._set_btt(branch_now, turn_now, tick_now)
				this_turn_neighbors = get_neighbors(entity, neighborhood)
			if set(last_turn_neighbors) != set(this_turn_neighbors):
				return None
			return this_turn_neighbors

		def get_node(graphn, noden):
			key = (graphn, noden)
			if key not in node_objs:
				node_objs[key] = make_node(charmap[graphn], noden)
			return node_objs[key]

		def get_thing(graphn, thingn):
			key = (graphn, thingn)
			if key not in node_objs:
				node_objs[key] = thing_cls(charmap[graphn], thingn)
			return node_objs[key]

		def get_place(graphn, placen):
			key = (graphn, placen)
			if key not in node_objs:
				node_objs[key] = place_cls(charmap[graphn], placen)
			return node_objs[key]

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
			if truthfun in self.rulebook[rulebook]:
				todo[prio, rulebook].append((rule, handled, entity))
				continue
			trig_futs.append(
				submit(
					check_triggers,
					prio,
					rulebook,
					rule,
					handled,
					entity,
					get_effective_neighbors(entity, rule.neighborhood),
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
			if truthfun in self.rulebook[rulebook]:
				todo[prio, rulebook].append((rule, handled, entity))
				continue
			trig_futs.append(
				submit(
					check_triggers,
					prio,
					rulebook,
					rule,
					handled,
					entity,
					get_effective_neighbors(entity, rule.neighborhood),
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
			if truthfun in self.rulebook[rulebook]:
				todo[prio, rulebook].append((rule, handled, entity))
				continue
			trig_futs.append(
				submit(
					check_triggers,
					prio,
					rulebook,
					rule,
					handled,
					entity,
					get_effective_neighbors(entity, rule.neighborhood),
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
			if truthfun in self.rulebook[rulebook]:
				todo[prio, rulebook].append((rule, handled, entity))
				continue
			trig_futs.append(
				submit(
					check_triggers,
					prio,
					rulebook,
					rule,
					handled,
					entity,
					get_effective_neighbors(entity, rule.neighborhood),
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
			if truthfun in self.rulebook[rulebook]:
				todo[prio, rulebook].append((rule, handled, entity))
				continue
			trig_futs.append(
				submit(
					check_triggers,
					prio,
					rulebook,
					rule,
					handled,
					entity,
					get_effective_neighbors(entity, rule.neighborhood),
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
			if truthfun in self.rulebook[rulebook]:
				todo[prio, rulebook].append((rule, handled, entity))
				continue
			trig_futs.append(
				submit(
					check_triggers,
					prio,
					rulebook,
					rule,
					handled,
					entity,
					get_effective_neighbors(entity, rule.neighborhood),
				)
			)
		if pool:
			futwait(trig_futs)
		else:
			for part in trig_futs:
				part()

		def fmtent(entity):
			if isinstance(entity, self.char_cls):
				return entity.name
			elif hasattr(entity, "name"):
				return f"{entity.character.name}.node[{entity.name}]"
			else:
				return (
					f"{entity.character.name}.portal"
					f"[{entity.origin.name}][{entity.destination.name}]"
				)

		for prio_rulebook in sort_set(todo.keys()):
			for rule, handled, entity in todo[prio_rulebook]:
				if not entity:
					continue
				self.debug(
					f"checking prereqs for rule {rule.name} on entity {fmtent(entity)}"
				)
				if check_prereqs(rule, handled, entity):
					self.debug(
						f"prereqs for rule {rule.name} on entity "
						f"{fmtent(entity)} satisfied, will run actions"
					)
					try:
						yield do_actions(rule, handled, entity)
						self.debug(
							f"actions for rule {rule.name} on entity "
							f"{fmtent(entity)} have run without incident"
						)
					except StopIteration:
						raise InnerStopIteration

	def _advance(self) -> Any:
		"""Follow the next rule if available.

		If we've run out of rules, reset the rules iterator.

		"""
		assert self.turn > self._turns_completed[self.branch]
		try:
			return next(self._rules_iter)
		except InnerStopIteration:
			self._rules_iter = self._follow_rules()
			raise StopIteration()
		except StopIteration:
			self._rules_iter = self._follow_rules()
			return final_rule

	# except Exception as ex:
	# self._rules_iter = self._follow_rules()
	# return ex

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

	new_graph = new_character

	def add_character(
		self,
		name: Key,
		data: Union[Graph, DiGraph] = None,
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
				data = DiGraph()
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
		self._init_graph(name, "DiGraph", data)
		if self._btt() not in self._keyframes_times:
			self.snap_keyframe(silent=True)
		self._graph_objs[name] = self.char_cls(self, name)

	def del_graph(self, name: Key) -> None:
		graph = self.graph[name]
		for thing in list(graph.thing):
			del graph.thing[thing]
		super().del_graph(name)
		if hasattr(self, "_worker_processes"):
			self._call_every_subproxy("_del_character", name)

	del_character = del_graph

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
		self, branch: str, turn: int, tick: int
	) -> None:
		universal = dict(self.universal.items())
		self._universal_cache.set_keyframe(branch, turn, tick, universal)
		all_graphs = set(self._graph_cache.iter_keys(branch, turn, tick))
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
				(char,), branch, turn, tick, char_kf
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
		thing_graphs = all_graphs.copy()
		for charname in all_graphs:
			thing_graphs.discard(charname)
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
			self._things_cache.set_keyframe(
				(charname,), branch, turn, tick, locs
			)
			self._node_contents_cache.set_keyframe(
				(charname,), branch, turn, tick, conts
			)
			self._unitness_cache.set_keyframe(
				(charname,), branch, turn, tick, units
			)
		for graph in thing_graphs:
			self._things_cache.set_keyframe((graph,), branch, turn, tick, {})
			self._node_contents_cache.set_keyframe(
				(graph,), branch, turn, tick, {}
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
		super()._snap_keyframe_de_novo(branch, turn, tick)

	def _snap_keyframe_de_novo_graph(
		self,
		graph: Key,
		branch: str,
		turn: int,
		tick: int,
		nodes: NodeValDict,
		edges: EdgeValDict,
		graph_val: StatDict,
		copy_to_branch: str = None,
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
				kf[graph] = graph_val.pop(rb_kf_type, (rb_kf_type, graph))
			except KeyError:
				kf = {graph: graph_val.pop(rb_kf_type, (rb_kf_type, graph))}
				for char in self._graph_cache.iter_keys(branch, turn, tick):
					# seems like Python gets upset when it has to catch too many
					# nested KeyError?
					rb = rb_kf_cache._base_retrieve((char, branch, turn, tick))
					if isinstance(rb, KeyError):
						kf[char] = (rb_kf_type, char)
					else:
						kf[char] = rb_kf_cache.retrieve(
							char, branch, turn, tick
						)
			rb_kf_cache.set_keyframe(branch, turn, tick, kf)
		self._nodes_rulebooks_cache.set_keyframe(
			(graph,),
			branch,
			turn,
			tick,
			{n: nvs.pop("rulebook", (graph, n)) for (n, nvs) in nodes.items()},
		)
		port_rb_kf = {}
		for orig, dests in edges.items():
			port_rb_kf[orig] = rbs = {}
			for dest, port in dests.items():
				rbs[dest] = port.pop("rulebook", (graph, orig, dest))
		self._portals_rulebooks_cache.set_keyframe(
			(graph,),
			branch,
			turn,
			tick,
			port_rb_kf,
		)
		super()._snap_keyframe_de_novo_graph(
			graph, branch, turn, tick, nodes, edges, graph_val
		)
		if "units" in graph_val:
			self._unitness_cache.set_keyframe(
				(graph,), branch, turn, tick, graph_val["units"]
			)
		else:
			self._unitness_cache.set_keyframe((graph,), branch, turn, tick, {})
		newkf = {}
		contkf = {}
		for name, node in nodes.items():
			if not isinstance(node, dict):
				raise TypeError("nodes in keyframes must be dictionaries")
			if "location" not in node:
				continue
			locn = node["location"]
			newkf[name] = locn
			if locn in contkf:
				contkf[locn].add(name)
			else:
				contkf[locn] = {
					name,
				}
		contkf = {k: frozenset(v) for (k, v) in contkf.items()}
		self._node_contents_cache.set_keyframe(
			(graph,), branch, turn, tick, contkf
		)
		self._things_cache.set_keyframe((graph,), branch, turn, tick, newkf)
		assert (
			(graph,) in self._things_cache.keyframe
			and branch in self._things_cache.keyframe[graph,]
			and turn in self._things_cache.keyframe[graph,][branch]
			and tick in self._things_cache.keyframe[graph,][branch][turn]
		)

	def turns_when(
		self, qry: Query, mid_turn=False
	) -> Union[QueryResult, set]:
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
		end = self._branches[self.branch][3] + 1

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
		self.query.flush()
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

	def _node_contents(self, character: Key, node: Key) -> Set:
		return self._node_contents_cache.retrieve(
			character, node, *self._btt()
		)

	def apply_choices(
		self, choices: List[dict], dry_run=False, perfectionist=False
	) -> Tuple[List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
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
