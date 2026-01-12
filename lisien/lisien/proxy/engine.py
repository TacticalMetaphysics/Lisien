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

import ast
import io
import logging
import os
import pickle
from collections import UserDict
from contextlib import contextmanager
from functools import cached_property, partial
from inspect import getsource
from os import PathLike
from pathlib import Path
from random import Random
from threading import Lock
from time import monotonic
from types import MethodType
from typing import (
	Any,
	Callable,
	ClassVar,
	Iterable,
	Iterator,
	Literal,
	MutableMapping,
	Optional,
	Protocol,
	IO,
)

from attrs import Converter, define, field
import networkx as nx
from blinker import Signal

try:
	import msgpack
	import msgpack._cmsgpack

	if msgpack.Packer.__module__.endswith("cmsgpack"):
		Ext = msgpack.ExtType
	else:
		import umsgpack as msgpack

		Ext = msgpack.Ext
except ImportError:
	import umsgpack as msgpack

	Ext = msgpack.Ext

from ..collections import (
	FunctionStore,
	TriggerStore,
	GenericFunctionStore,
	MethodStore,
	ActionStore,
	PrereqStore,
)
from ..exc import OutOfTimelineError, WorkerProcessReadOnlyError
from ..types import (
	AbstractBookmarkMapping,
	AbstractCharacter,
	AbstractEngine,
	AbstractFunctionStore,
	ActionFuncName,
	Branch,
	CharDelta,
	CharName,
	DeltaDict,
	EdgesDict,
	EdgeValDict,
	EternalKey,
	FuncName,
	FuncStoreName,
	Key,
	Keyframe,
	KeyHint,
	LinearTime,
	MsgpackExtensionType,
	Node,
	NodeName,
	NodesDict,
	NodeValDict,
	PrereqFuncName,
	RulebookName,
	RulebookPriority,
	RuleName,
	RuleNeighborhood,
	StatDict,
	Tick,
	Time,
	TimeSignalDescriptor,
	TriggerFuncName,
	Turn,
	UniversalKey,
	Value,
	ValueHint,
	PickyDefaultDict,
	StructuredDefaultDict,
	AbstractLanguageDescriptor,
	Stat,
	CharacterRulebookTypeStr,
	AttrSignal,
)
from ..util import (
	dedent_source,
	format_call_sig,
	msgpack_array_header,
	msgpack_map_header,
)
from ..wrap import UnwrappingDict
from .abc import FuncListProxy, FuncProxy, RuleBookProxy, RuleProxy
from .character import CharacterProxy, PlaceProxy, PortalProxy, ThingProxy


@define
class FuncStoreProxy(AbstractFunctionStore, AttrSignal):
	_store: ClassVar[FuncStoreName]
	engine: EngineProxy
	_cache: dict[str, str] = field(alias="initial", factory=dict)
	_proxy_cache: dict[FuncName, FuncProxy] = field(init=False, factory=dict)

	def save(self, reimport: bool = True) -> None:
		self.engine.handle("save_code", reimport=reimport)

	def reimport(self) -> None:
		self.engine.handle("reimport_code")

	def iterplain(self) -> Iterator[tuple[str, str]]:
		return iter(self._cache.items())

	def store_source(self, v: str, name: str | None = None) -> None:
		self._cache[name] = v
		self.engine.handle("store_source", v=v, name=name, store=self._store)

	def _worker_check(self):
		self.engine._worker_check()

	def load(self):
		self._cache = self.engine.handle("source_copy", store=self._store)

	def __getattr__(self, k: str) -> FuncProxy:
		k = FuncName(k)
		if k in super().__getattribute__("_cache"):
			proxcache = super().__getattribute__("_proxy_cache")
			if k not in proxcache:
				proxcache[k] = FuncProxy(self, k)
			return proxcache[k]
		else:
			raise AttributeError(k)

	def __call__(self, func: Callable):
		src = getsource(func)
		self.engine.handle(
			"store_source", v=src, name=func.__name__, store=self._store
		)
		funcname = func.__name__
		self._cache[funcname] = src
		self._proxy_cache[funcname] = FuncProxy(self, funcname)

	def __setattr__(self, func_name: str, func: Callable | str):
		if func_name in ("_cache", "_proxy_cache"):
			if not isinstance(func, dict):
				raise TypeError("Invalid FuncStoreProxy cache", func)
			return super().__setattr__(func_name, func)
		if func_name == "engine":
			if not isinstance(func, EngineProxy):
				raise TypeError("EngineProxy only", func)
			return super().__setattr__(func_name, func)
		if func_name in (
			"_store",
			"receivers",
			"_by_sender",
			"_by_receiver",
			"_weak_senders",
			"is_muted",
		):
			super().__setattr__(func_name, func)
			return
		if callable(func):
			source = getsource(func)
		else:
			source = func
		self._set_source(func_name, source)

	def __delattr__(self, func_name: str):
		self.engine.handle(
			command="del_source", store=self._store, k=func_name
		)
		del self._cache[func_name]

	def get_source(self, func_name: str) -> str:
		return self.engine.handle(
			command="get_source", store=self._store, name=func_name
		)

	def _set_source(self, func_name: str, source: str) -> None:
		self._cache[func_name] = source
		self.engine.handle(
			command="store_source", store=self._store, name=func_name, v=source
		)


@define
class GenericFunctionStoreProxy(FuncStoreProxy):
	_store: ClassVar[FuncStoreName] = "function"


@define
class MethodStoreProxy(FuncStoreProxy):
	_store: ClassVar[FuncStoreName] = "method"


@define
class TriggerStoreProxy(FuncStoreProxy):
	_store: ClassVar[FuncStoreName] = "trigger"

	def __attrs_post_init__(self):
		self._cache["truth"] = dedent_source(
			"""
		def truth(*args):
			return True""".strip("\n")
		)


@define
class PrereqStoreProxy(FuncStoreProxy):
	_store: ClassVar[FuncStoreName] = "prereq"


@define
class ActionStoreProxy(FuncStoreProxy):
	_store: ClassVar[FuncStoreName] = "action"


class BookmarkMappingProxy(AbstractBookmarkMapping, UserDict):
	def __init__(self, engine: EngineProxy):
		self.engine = engine
		super().__init__(self.engine.handle("bookmarks_dump"))

	def __call__(self, key: Key) -> None:
		self.data[key] = self.engine.handle("set_bookmark", key=key)

	def __setitem__(self, key: KeyHint, value: Time):
		if not (
			isinstance(value, tuple)
			and len(tuple) == 3
			and isinstance(value[0], str)
			and isinstance(value[1], int)
			and isinstance(value[2], int)
		):
			raise TypeError("Not a valid time", value)
		self.data[key] = self.engine.handle(
			"set_bookmark", key=key, time=value
		)

	def __delitem__(self, key: Key):
		self.engine.handle("del_bookmark", key=key)
		del self.data[key]


EngineProxyCallback = Callable[
	[str, Branch, Turn, Tick, tuple[Value, ...]], Any
]


class SubmitProtocol(Protocol):
	def __call__(self, fn, /, *args, **kwargs): ...


@define
class EngineProxy(AbstractEngine):
	"""An engine-like object for controlling a lisien process

	Don't instantiate this directly. Use :class:`EngineProcessManager` instead.
	The ``start`` method will return an :class:`EngineProxy` instance.

	"""

	char_cls: ClassVar = CharacterProxy
	thing_cls: ClassVar = ThingProxy
	place_cls: ClassVar = PlaceProxy
	portal_cls: ClassVar = PortalProxy
	is_proxy: ClassVar = True
	_branch: Branch = field(
		init=False, default=Branch("trunk"), converter=Branch
	)
	_turn: Turn = field(init=False, default=Turn(0), converter=Turn)
	_tick: Tick = field(init=False, default=Tick(0), converter=Tick)
	_mutable_worker: bool = field(init=False, default=False)
	_get_input_bytes: Callable[[], bytes] | None
	_send_output_bytes: Callable[[bytes], None] | None
	logger: logging.Logger
	prefix: os.PathLike[str] | None = None
	_install_modules: list[str] | tuple[str] = ()
	_eternal: dict[EternalKey, Value] = field(
		factory=lambda: {"language": "eng"}
	)
	_random_seed: int | None = None
	enforce_end_of_time: bool = True
	_universal: dict[UniversalKey, Value] = field(factory=lambda: {})
	_branches_d: dict[Branch, tuple[Branch | None, Turn, Tick, Turn, Tick]] = (
		field(
			factory=lambda: {
				Branch("trunk"): (None, Turn(0), Tick(0), Turn(0), Tick(0))
			},
		)
	)
	i: int | None = field(alias="worker_index", default=None)

	@property
	def _worker(self) -> bool:
		return self.i is not None

	@staticmethod
	def _convert_function_store_proxy(cls, proxy_cls, src_d, self):
		if self._worker:
			return cls(
				self.prefix.joinpath(cls._store + ".py"), initial=src_d or {}
			)
		return proxy_cls(self, initial=src_d or {})

	function: GenericFunctionStoreProxy | GenericFunctionStore = field(
		converter=Converter(
			partial(
				_convert_function_store_proxy,
				GenericFunctionStore,
				GenericFunctionStoreProxy,
			),
			takes_self=True,
		),
		default=None,
	)
	method: MethodStoreProxy | MethodStore = field(
		converter=Converter(
			partial(
				_convert_function_store_proxy, MethodStore, MethodStoreProxy
			),
			takes_self=True,
		),
		default=None,
	)
	action: ActionStoreProxy | ActionStore = field(
		converter=Converter(
			partial(
				_convert_function_store_proxy, ActionStore, ActionStoreProxy
			),
			takes_self=True,
		),
		default=None,
	)
	prereq: PrereqStoreProxy | PrereqStore = field(
		converter=Converter(
			partial(
				_convert_function_store_proxy, PrereqStore, PrereqStoreProxy
			),
			takes_self=True,
		),
		default=None,
	)

	@staticmethod
	def _convert_trig_store_proxy(src_d: dict[str, str] | None, self):
		if src_d is None:
			src_d = {"truth": "def truth(obj):\n\treturn True"}
		elif "truth" not in src_d:
			src_d["truth"] = "def truth(obj):\n\treturn True"
		if self.i is None:
			return TriggerStoreProxy(self, initial=src_d)
		else:
			return TriggerStore(
				self.prefix.joinpath("trigger.py") if self.prefix else None,
				initial=src_d,
			)

	trigger: TriggerStoreProxy | TriggerStore = field(
		converter=Converter(_convert_trig_store_proxy, takes_self=True),
		default=None,
	)

	@staticmethod
	def _validate_strings(self, attr, val):
		if val is None:
			return True
		self.string._cache = val

	_strings: dict[str, str] | None = field(
		validator=_validate_strings, default=None
	)

	_planning: bool = field(alias="_planning", init=False, default=False)
	_initialized: bool = field(alias="_initialized", init=False, default=False)
	_forward: bool = field(alias="_forward", init=False, default=False)
	closed: bool = field(init=False, default=False)

	@staticmethod
	def _convert_replay_file(
		replay_file: Path | os.PathLike[str] | IO[str], self
	):
		if replay_file is None:
			return None
		if not isinstance(replay_file, io.TextIOBase):
			replay_file = Path(replay_file)
			if replay_file.exists():
				with open(replay_file, "rt") as rf:
					return rf.read().replace(
						"<lisien.proxy.EngineProxy>", "eng"
					)
			else:
				return open(replay_file, "wt")
		elif hasattr(replay_file, "mode"):
			if "w" in replay_file.mode or "a" in replay_file.mode:
				return replay_file
			elif "r" in replay_file.mode:
				return replay_file.read().replace(
					"<lisien.proxy.EngineProxy>", "eng"
				)
			else:
				self.error(
					"Can't open replay file, mode " + repr(replay_file.mode)
				)
				return None
		else:
			txt = replay_file.read()
			if not txt:
				self.warning("Recording replay to an io object with no mode")
				return replay_file
			else:
				return txt.replace("<lisien.proxy.EngineProxy>", "eng")

	replay_file: str | None = field(
		default=None,
		converter=Converter(_convert_replay_file, takes_self=True),
	)

	@cached_property
	def next_turn(self):
		if self.i and not getattr(self, "_mutable_worker", False):

			def no_next_turn():
				raise WorkerProcessReadOnlyError(
					"Can't advance time in a worker process"
				)

			return no_next_turn
		return NextTurnProxy(self)

	@cached_property
	def _pipe_out_lock(self):
		return Lock()

	@cached_property
	def _pipe_in_lock(self):
		return Lock()

	@cached_property
	def _round_trip_lock(self):
		return Lock()

	@cached_property
	def _commit_lock(self):
		return Lock()

	@cached_property
	def _rules_cache(
		self,
	) -> dict[
		RuleName,
		dict[
			Literal["triggers", "prereqs", "actions"],
			list[TriggerFuncName]
			| list[PrereqFuncName]
			| list[ActionFuncName],
		],
	]:
		return {}

	@cached_property
	def _neighborhood_cache(self) -> dict[RuleName, RuleNeighborhood]:
		return {}

	@cached_property
	def _rulebooks_cache(
		self,
	) -> dict[RulebookName, tuple[list[RuleName], RulebookPriority]]:
		return {}

	@cached_property
	def _character_rulebooks_cache(
		self,
	) -> dict[
		CharName,
		dict[
			Literal["character", "unit", "thing", "place", "portal"],
			RulebookName,
		],
	]:
		return StructuredDefaultDict(
			1, RulebookName, args_munger=lambda inst, k: ((inst.key, k),)
		)

	@cached_property
	def _char_node_rulebooks_cache(
		self,
	) -> dict[CharName, dict[NodeName, RulebookName]]:
		return StructuredDefaultDict(
			1, RulebookName, args_munger=lambda inst, k: ((inst.key, k),)
		)

	@cached_property
	def _char_port_rulebooks_cache(
		self,
	) -> dict[CharName, dict[NodeName, dict[NodeName, RulebookName]]]:
		return StructuredDefaultDict(
			2,
			RulebookName,
			args_munger=lambda inst, k: ((inst.parent.key, inst.key, k),),
		)

	@cached_property
	def _things_cache(self) -> dict[CharName, dict[NodeName, ThingProxy]]:
		return StructuredDefaultDict(1, ThingProxy)

	@cached_property
	def _character_places_cache(
		self,
	) -> dict[CharName, dict[NodeName, PlaceProxy]]:
		return StructuredDefaultDict(1, PlaceProxy)

	@cached_property
	def _character_portals_cache(self) -> PortalObjCache:
		return PortalObjCache()

	@cached_property
	def _char_cache(self) -> dict[CharName, CharacterProxy]:
		return {}

	@cached_property
	def _character_units_cache(
		self,
	) -> dict[CharName, dict[CharName, set[NodeName]]]:
		return StructuredDefaultDict(1, set)

	@cached_property
	def _unit_characters_cache(
		self,
	) -> dict[CharName, dict[NodeName, set[CharName]]]:
		return StructuredDefaultDict(1, set)

	@cached_property
	def _char_stat_cache(self) -> dict[CharName, dict[Stat, Value]]:
		return PickyDefaultDict(UnwrappingDict)

	@cached_property
	def _node_stat_cache(
		self,
	) -> dict[CharName, dict[NodeName, dict[Stat, Value]]]:
		return StructuredDefaultDict(1, UnwrappingDict)

	@cached_property
	def _portal_stat_cache(
		self,
	) -> dict[CharName, dict[NodeName, dict[NodeName, dict[Stat, Value]]]]:
		return StructuredDefaultDict(2, UnwrappingDict)

	@cached_property
	def _rule_obj_cache(self) -> dict[RuleName, RuleProxy]:
		return {}

	@cached_property
	def _rulebook_obj_cache(self) -> dict[RuleName, RuleBookProxy]:
		return {}

	def _get_node(
		self, char: AbstractCharacter | KeyHint, node: NodeName
	) -> Node:
		return self.character[char].node[node]

	def _worker_check(self):
		if self._worker and not getattr(self, "_mutable_worker", False):
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)

	def _set_btt(
		self,
		branch: Branch,
		turn: Turn,
		tick: Optional[Tick] = None,
		cb: Optional[Callable] = None,
	):
		return self.handle(
			"time_travel",
			branch=branch,
			turn=turn,
			tick=tick,
			cb=partial(self._upd_and_cb, cb=cb) if cb else self._upd_and_cb,
		)

	_time_warp = _set_btt

	def _start_branch(
		self, parent: Branch, branch: Branch, turn: Turn, tick: Tick
	) -> None:
		_, start_turn, start_tick, end_turn, end_tick = self._branches_d[
			parent
		]
		if not (
			LinearTime(start_turn, start_tick)
			<= LinearTime(turn, tick)
			<= LinearTime(end_turn, end_tick)
		):
			raise OutOfTimelineError(
				"The parent branch does not cover that time",
				parent,
				turn,
				tick,
			)
		self._branches_d[branch] = (parent, turn, tick, turn, tick)
		self.handle(
			"start_branch", parent=parent, branch=branch, turn=turn, tick=tick
		)

	def _extend_branch(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		parent, start_turn, start_tick, end_turn, end_tick = self._branches_d[
			branch
		]
		if LinearTime(turn, tick) < LinearTime(start_turn, start_tick):
			raise OutOfTimelineError(
				"Can't extend branch backwards", branch, turn, tick
			)
		if LinearTime(turn, tick) < LinearTime(end_turn, end_tick):
			return
		if not self._planning:
			self._branches_d[branch] = (
				parent,
				start_turn,
				start_tick,
				turn,
				tick,
			)

	def load_at(
		self, branch: str | Branch, turn: int | Turn, tick: int | Tick
	) -> None:
		self.handle("load_at", branch=branch, turn=turn, tick=tick)

	def branch_end(self, branch: str | Branch | None = None):
		return self.handle("branch_end", branch=branch)

	def turn_end(
		self,
		branch: str | Branch | None = None,
		turn: int | Turn | None = None,
	) -> Tick:
		if self._worker:
			raise NotImplementedError("Need to cache turn ends in workers")
		return self.handle("turn_end", branch=branch, turn=turn)

	def turn_end_plan(
		self,
		branch: str | Branch | None = None,
		turn: int | Turn | None = None,
	) -> Tick:
		if self._worker:
			raise NotImplementedError("Need to cache plans in workers")
		return self.handle("turn_end_plan", branch=branch, turn=turn)

	@property
	def trunk(self) -> Branch:
		return self.handle("main_branch")

	@trunk.setter
	def trunk(self, branch: str | Branch) -> None:
		self._worker_check()
		if self.branch != self.trunk or self.turn != 0 or self._tick != 0:
			raise AttributeError("Go to the start of time first")
		kf = self.handle(
			"switch_main_branch", branch=branch, cb=self._upd_time
		)
		assert self.branch == branch
		self._replace_state_with_kf(kf)

	@cached_property
	def bookmark(self) -> BookmarkMappingProxy:
		return BookmarkMappingProxy(self)

	def export(
		self,
		name: str | None = None,
		path: str | os.PathLike | None = None,
		indent: bool = True,
	) -> str | os.PathLike:
		if name is None and path is None:
			raise ValueError("Need name or path")
		return self.handle("export", name=name, path=path, indent=indent)

	@classmethod
	def from_archive(
		cls,
		path: str | os.PathLike,
		prefix: str | os.PathLike | None = ".",
		**kwargs,
	) -> AbstractEngine:
		raise TypeError(
			"You want the ``load_archive`` method of ``EngineProcessManager`` instead"
		)

	def snap_keyframe(self) -> Keyframe:
		if self._worker and not getattr(self, "_mutable_worker", False):
			raise WorkerProcessReadOnlyError(
				"Can't snap a keyframe in a worker process"
			)
		return self.handle("snap_keyframe")

	def game_init(self) -> None:
		self._worker_check()
		self.handle("game_init", cb=self._upd_from_game_start)

	def _node_exists(self, char: CharName, node: NodeName) -> bool:
		return self.handle("node_exists", char=char, node=node)

	def _upd_from_game_start(
		self,
		command: str,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		result: Value[
			tuple[
				Keyframe,
				dict[EternalKey, Value],
				dict[FuncStoreName, dict[str, str]],
				dict[FuncStoreName, bytes],
			]
		],
	):
		(start_kf, eternal, plainstored, pklstored) = result
		self._initialized = False
		self._eternal = eternal
		for name, store in [
			("function", self.function),
			("method", self.method),
			("trigger", self.trigger),
			("prereq", self.prereq),
			("action", self.action),
		]:
			if name in plainstored:
				if isinstance(store, FuncStoreProxy):
					store._cache = plainstored[name]
					self.debug(
						f"Replaced func store proxies for {name} in {store} in worker {self.i}"
					)
				elif isinstance(store, FunctionStore):
					for fname, source in plainstored[name].items():
						store._set_source(fname, source)
						self.debug(
							f"Replaced function {fname} in {store} with plain source in worker {self.i}"
						)
				else:
					self.error(
						f"Can't set {name} on {store} in worker {self.i}"
					)
			elif name in pklstored:
				replacement = pickle.loads(pklstored[name])
				setattr(self, name, replacement)
			elif hasattr(store, "reimport") and callable(store.reimport):
				store.reimport()
		self._replace_state_with_kf(start_kf)
		self._branch = branch
		self._turn = turn
		self._tick = tick
		self._initialized = True

	def switch_main_branch(self, branch: str) -> None:
		self._worker_check()
		if (
			self.branch != self.main_branch
			or self.turn != 0
			or self._tick != 0
		):
			raise ValueError("Go to the start of time first")
		kf = self.handle(
			"switch_main_branch", branch=branch, cb=self._upd_time
		)
		assert self.branch == branch
		self._replace_state_with_kf(kf)

	def _replace_state_with_kf(self, result: Keyframe | None, **kwargs):
		self.debug("EngineProxy: replacing state with a keyframe")
		things = self._things_cache
		places = self._character_places_cache

		portals = self._character_portals_cache
		if result is None:
			self.debug("EngineProxy: empty keyframe; clearing caches")
			self._char_cache.clear()
			self._universal.clear()
			return
		self._universal = result["universal"]
		self._rules_cache.clear()
		rc = self._rules_cache
		triggers: list[TriggerFuncName]
		for rule, triggers in result["triggers"].items():
			triglist = []
			if isinstance(self.trigger, FuncStoreProxy):
				for func in triggers:
					if not hasattr(self.trigger, func):
						self.trigger._proxy_cache[func] = FuncProxy(
							self.trigger, func
						)
					triglist.append(getattr(self.trigger, func))
			else:
				for func in triggers:
					if not hasattr(self.trigger, func):
						if isinstance(self.trigger, FuncStoreProxy):
							setattr(
								self.trigger,
								func,
								FuncProxy(self.trigger, func),
							)
						elif hasattr(self.trigger, "reimport"):
							self.trigger.reimport()
					if hasattr(self.trigger, func):
						triglist.append(getattr(self.trigger, func))
					else:
						self.warning(
							f"didn't find {func} in trigger file {self.trigger._filename}"
						)
			if rule in rc:
				rc[rule]["triggers"] = triglist
			else:
				rc[rule] = {
					"triggers": triglist,
					"prereqs": [],
					"actions": [],
				}
		self.debug("EngineProxy: replaced triggers with keyframe...")
		prereqs: list[PrereqFuncName]
		for rule, prereqs in result["prereqs"].items():
			preqlist = []
			if isinstance(self.prereq, FuncStoreProxy):
				for func in prereqs:
					if not hasattr(self.prereq, func):
						self.prereq._proxy_cache[func] = FuncProxy(
							self.prereq, func
						)
					preqlist.append(getattr(self.prereq, func))
			else:
				for func in prereqs:
					if not hasattr(self.prereq, func):
						if isinstance(self.prereq, FuncStoreProxy):
							setattr(
								self.prereq, func, FuncProxy(self.prereq, func)
							)
						elif hasattr(self.prereq, "reimport"):
							self.prereq.reimport()
					if hasattr(self.prereq, func):
						preqlist.append(getattr(self.prereq, func))
					else:
						self.warning(
							f"didn't find {func} in prereq file {self.prereq._filename}"
						)
			if rule in rc:
				rc[rule]["prereqs"] = preqlist
			else:
				rc[rule] = {
					"triggers": [],
					"prereqs": preqlist,
					"actions": [],
				}
		self.debug("EngineProxy: replaced prereqs with keyframe...")
		actions: list[ActionFuncName]
		for rule, actions in result["actions"].items():
			actlist = []
			if isinstance(self.action, FuncStoreProxy):
				for func in actions:
					if not hasattr(self.action, func):
						self.action._proxy_cache[func] = FuncProxy(
							self.action, func
						)
					actlist.append(getattr(self.action, func))
			else:
				for func in actions:
					if not hasattr(self.action, func) and hasattr(
						self.action, "reimport"
					):
						self.action.reimport()
					if hasattr(self.action, func):
						actlist.append(getattr(self.action, func))
					else:
						self.warning(
							f"didn't find {func} in action file {self.action._filename}"
						)
			if rule in rc:
				rc[rule]["actions"] = actlist
			else:
				rc[rule] = {
					"triggers": [],
					"prereqs": [],
					"actions": actlist,
				}
		self.debug("EngineProxy: replaced actions with keyframe...")
		self._rulebooks_cache = result["rulebook"]
		chars = self._char_cache
		chars.clear()
		for graph in (
			result["graph_val"].keys()
			| result["nodes"].keys()
			| result["node_val"].keys()
			| result["edges"].keys()
			| result["edge_val"].keys()
		):
			chars[graph] = CharacterProxy(self, graph)
		self.debug(f"EngineProxy: {len(chars)} characters in this keyframe...")
		graph: CharName
		graph_stats: dict[
			Stat | CharacterRulebookTypeStr | Literal["units"],
			Value | dict[CharName, dict[NodeName, bool]] | RulebookName,
		]
		for graph, graph_stats in result["graph_val"].items():
			if "character_rulebook" in graph_stats:
				self._character_rulebooks_cache[graph]["character"] = (
					graph_stats.pop("character_rulebook")
				)
			if "unit_rulebook" in graph_stats:
				self._character_rulebooks_cache[graph]["unit"] = (
					graph_stats.pop("unit_rulebook")
				)
			if "character_thing_rulebook" in graph_stats:
				self._character_rulebooks_cache[graph]["thing"] = (
					graph_stats.pop("character_thing_rulebook")
				)
			if "character_place_rulebook" in graph_stats:
				self._character_rulebooks_cache[graph]["place"] = (
					graph_stats.pop("character_place_rulebook")
				)
			if "character_portal_rulebook" in graph_stats:
				self._character_rulebooks_cache[graph]["portal"] = (
					graph_stats.pop("character_portal_rulebook")
				)
			if "units" in graph_stats:
				for char, units in graph_stats.pop("units").items():
					self._character_units_cache[graph][char] = set(
						units.keys()
					)
			else:
				del self._character_units_cache[graph]
			graph_stats: dict[Stat, Value]
			self._char_stat_cache[graph] = graph_stats
			self.debug(f"EngineProxy: got keyframed stats for {graph}...")
		nodes_to_delete = {
			(char, node)
			for char in things.keys() | places.keys()
			for node in things.get(char, {}).keys()
			| places.get(char, {}).keys()
		}
		self.debug(
			f"EngineProxy: deleting {len(nodes_to_delete)} nodes "
			"to match the keyframe..."
		)
		for char, nodes in result["nodes"].items():
			nodes: NodesDict
			for node, ex in nodes.items():
				if ex:
					if not (
						(char in things and node in things[char])
						or (char in places and node in places[char])
					):
						places[char][node] = PlaceProxy(chars[char], node)
					nodes_to_delete.discard((char, node))
				else:
					if char in things and node in things[char]:
						del things[char][node]
					if char in places and node in places[char]:
						del places[char][node]
		for char, node in nodes_to_delete:
			if node in things[char]:
				del things[char][node]
			else:
				del places[char][node]
		for char, nodestats in result["node_val"].items():
			self.debug(
				f"EngineProxy: Updating node stats for character {char} "
				"from a keyframe..."
			)
			node: NodeName
			node_stats: dict[Stat | Literal["location"], Value | NodeName]
			for node, node_stats in nodestats.items():
				if "location" in node_stats:
					if char not in things or node not in things[char]:
						things[char][node] = ThingProxy(
							chars[char],
							node,
							NodeName(node_stats.pop("location")),
						)
					else:
						things[char][node]._location = NodeName(
							node_stats.pop("location")
						)
					if char in places and node in places[char]:
						del places[char][node]
				else:
					if char not in places or node not in places[char]:
						places[char][node] = PlaceProxy(chars[char], node)
					if char in things and node in things[char]:
						del things[char][node]
				node_stats: dict[Stat, Value]
				self._node_stat_cache[char][node] = node_stats
		edges_to_delete = {
			(char, orig, dest)
			for char in portals.successors
			for orig in portals.successors[char]
			for dest in portals.successors[char][orig]
		}
		self.debug(
			f"EngineProxy: Deleting {len(edges_to_delete)} to match a keyframe..."
		)
		for char, origs in result["edges"].items():
			origs: EdgesDict
			for orig, dests in origs.items():
				for dest, exists in dests.items():
					if (
						char in portals.successors
						and orig in portals.successors[char]
						and dest in portals.successors[char][orig]
					):
						if exists:
							edges_to_delete.discard((char, orig, dest))
						else:
							del portals.successors[char][orig][dest]
							del portals.predecessors[char][dest][orig]
						continue
					if exists:
						edges_to_delete.discard((char, orig, dest))
					else:
						continue
					that = PortalProxy(chars[char], orig, dest)
					portals.store(char, orig, dest, that)
		for char, orig, dest in edges_to_delete:
			portals.delete(char, orig, dest)
		for char, origs in result["edge_val"].items():
			self.debug(
				f"EngineProxy: Updating portal stat values for character {char} "
				"to match a keyframe..."
			)
			origs: EdgeValDict
			for orig, dests in origs.items():
				for dest, port_stats in dests.items():
					self._portal_stat_cache[char][orig][dest] = port_stats

	def _pull_kf_now(self, *args, **kwargs):
		self._replace_state_with_kf(self.handle("snap_keyframe"))

	@property
	def branch(self) -> Branch:
		return self._branch

	@branch.setter
	def branch(self, v: Branch):
		if v not in self._branches_d:
			self._start_branch(self.branch, v, self.turn, self.tick)
		self._set_btt(v, self.turn)

	@property
	def turn(self) -> Turn:
		return self._turn

	@turn.setter
	def turn(self, v: int):
		if not isinstance(v, int):
			raise TypeError("Turns must be integers")
		if v < 0:
			raise ValueError("Turns can't be negative")
		if v == self.turn:
			return
		v = Turn(v)
		turn_end, tick_end = self._branch_end()
		if (
			self.enforce_end_of_time
			and not self._planning
			and (v, self.tick) > (turn_end, tick_end)
		):
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
		branch = self.branch
		if self._planning:
			tick = self.turn_end_plan()
		else:
			tick = self.turn_end()
		self._extend_branch(branch, v, tick)
		self._set_btt(branch, v)

	@property
	def tick(self) -> Tick:
		return self._tick

	@tick.setter
	def tick(self, v: Tick):
		if not isinstance(v, int):
			raise TypeError("Ticks must be integers")
		if v < 0:
			raise ValueError("Ticks can't be negative")
		if v == self.tick:
			return
		v = Tick(v)
		if self.enforce_end_of_time:
			if self._planning:
				end_turn, end_tick = self.handle(
					"branch_end_plan_turn_and_tick"
				)
			else:
				end_turn, end_tick = (
					self.branch_end_turn(),
					self.branch_end_tick(),
				)
			if (self.turn, v) > (end_turn, end_tick):
				raise OutOfTimelineError(
					f"Tick {v} of turn {self.turn} is after the end of the branch"
					f" {self.branch}. "
					f"Simulate with `next_turn`.",
					self.branch,
					self.turn,
					self.tick,
					self.branch,
					v,
					self.tick,
				)
		self._set_btt(self.branch, self.turn, v)

	def _btt(self) -> Time:
		return self._branch, self._turn, self._tick

	@cached_property
	def character(self):
		return CharacterMapProxy(self)

	@cached_property
	def eternal(self) -> MutableMapping[KeyHint | EternalKey, ValueHint]:
		return EternalVarProxy(self)

	@cached_property
	def universal(
		self,
	) -> MutableMapping[KeyHint | UniversalKey, ValueHint]:
		return GlobalVarProxy(self)

	@cached_property
	def rulebook(self):
		return AllRuleBooksProxy(self)

	@cached_property
	def rule(self):
		return AllRulesProxy(self)

	@property
	def logger(self):
		return self._logger

	@cached_property
	def _rando(self):
		if self._worker:
			return Random(self._random_seed)
		else:
			return RandoProxy(self, seed=self._random_seed)

	@cached_property
	def string(self):
		return StringStoreProxy(self)

	@property
	def _caches(self):
		return [
			self._node_stat_cache,
			self._portal_stat_cache,
			self._char_stat_cache,
			self._things_cache,
			self._character_places_cache,
			self._character_rulebooks_cache,
			self._char_node_rulebooks_cache,
			self._char_port_rulebooks_cache,
			self._character_portals_cache,
			self._character_units_cache,
			self._unit_characters_cache,
			self._rule_obj_cache,
			self._rulebook_obj_cache,
			self._char_cache,
		]

	def _restart(
		self,
		prefix: Path | PathLike[str] | None,
		time: Time,
		eternal: dict[EternalKey, Value],
		branches: dict[Branch, tuple[Branch | None, Turn, Tick, Turn, Tick]],
		rando: int | tuple | None,
	):
		if prefix is not None:
			prefix = Path(prefix)
		if self._worker:
			filez = os.listdir(prefix)
			for fs in ["function", "method", "trigger", "prereq", "action"]:
				py = fs + ".py"
				if py in filez:
					store = getattr(self, fs)
					if prefix:
						store._filename = prefix.joinpath(py)
						store.reimport()
					else:
						store._cache = {}
		for cache in self._caches:
			cache.clear()
		self._eternal = dict(eternal)
		self._branches_d = dict(branches)
		if isinstance(rando, tuple):
			self._rando.setstate(rando)
		else:
			self._rando.seed(rando)
		(self._branch, self._turn, self._tick) = time

	def _init_pull_from_core(self):
		self.debug("EngineProxy: Getting time...")
		self.send_bytes(GET_TIME)
		received = self.recv()
		self.debug(
			f"EngineProxy: Got time: {received}. Pulling initial keyframe..."
		)
		self._branch, self._turn, self._tick = received[-1]
		self._initialized = False
		self._pull_kf_now()
		self.debug(
			"EngineProxy: Initial keyframe pulled. "
			f"Installing {len(self._install_modules)} modules..."
		)
		self._initialized = True
		for module in self._install_modules:
			self.handle("install_module", module=module)
			self.debug(f"Installed module: {module}")
		self.debug("EngineProxy: All modules installed.")
		if hasattr(self, "_replay_txt"):
			self.debug("EngineProxy: Running a replay.")
			replay = ast.parse(self._replay_txt)
			expr: ast.Expression
			for expr in replay.body:
				if isinstance(expr.value, ast.Call):
					method = expr.value.func.id
					args = []
					kwargs = {}
					arg: ast.Expression
					for arg in expr.value.args:
						if isinstance(arg.value, ast.Subscript):
							whatmap = arg.value.value.attr
							key = arg.value.slice.value
							args.append(getattr(self, whatmap)[key])
						elif hasattr(arg.value, "value"):
							args.append(arg.value.value)
						else:
							args.append(ast.unparse(arg.value))
					for kw in expr.value.keywords:
						if isinstance(kw.value, ast.Subscript):
							whatmap = kw.value.value.attr
							key = kw.value.slice.value
							kwargs[kw.arg] = getattr(self, whatmap)[key]
						else:
							if hasattr(kw.value, "value"):
								kwargs[kw.arg] = kw.value.value
							else:
								kwargs[kw.arg] = ast.unparse(kw.value)
					self.handle(method, *args, **kwargs)
		self.debug("EngineProxy: Initial pull from core completed.")

	def __getattr__(self, item: str):
		try:
			return super().__getattr__(item)
		except AttributeError:
			pass
		method = super().__getattribute__("method")
		meth = method.__getattr__(item)
		return MethodType(meth, self)

	def _reimport_triggers(self) -> None:
		if hasattr(self.trigger, "reimport"):
			self.trigger.reimport()

	def _reimport_code(self, stores: list[str] | None = None) -> None:
		if stores is None:
			stores = ["function", "method", "trigger", "prereq", "action"]
		for store_name in stores:
			store = getattr(self, store_name)
			store.reimport()

	def _replace_triggers_pkl(self, replacement: bytes) -> None:
		assert self._worker, "Loaded replacement triggers outside a worker"
		self.trigger._locl = pickle.loads(replacement)

	def _eval_trigger(self, name, entity) -> None:
		return getattr(self.trigger, name)(entity)

	def _call_function(self, name: FuncName, *args, **kwargs):
		return getattr(self.function, name)(*args, **kwargs)

	def _reimport_functions(self) -> None:
		if hasattr(self.function, "reimport"):
			self.function.reimport()

	def _replace_functions_pkl(self, replacement: bytes) -> None:
		assert self._worker, "Loaded replacement functions outside a worker"
		self.function._locl = pickle.loads(replacement)

	def _call_method(self, name: FuncName, *args, **kwargs):
		return MethodType(getattr(self.method, name), self)(*args, **kwargs)

	def _reimport_methods(self) -> None:
		if hasattr(self.method, "reimport"):
			self.method.reimport()

	def _replace_methods_pkl(self, replacement: bytes) -> None:
		assert self._worker, "Loaded replacement methods outside a worker"
		self.method._locl = pickle.loads(replacement)

	def send(
		self, obj: Value, blocking: bool = True, timeout: int | float = 1
	) -> None:
		self.send_bytes(self.pack(obj), blocking=blocking, timeout=timeout)

	def send_bytes(self, obj: bytes, blocking=True, timeout=1) -> None:
		self._pipe_out_lock.acquire(blocking, timeout)
		self._send_output_bytes(obj)
		self._pipe_out_lock.release()

	def recv_bytes(self, blocking=True, timeout=1) -> bytes:
		self._pipe_in_lock.acquire(blocking, timeout)
		data = self._get_input_bytes()
		self._pipe_in_lock.release()
		return data

	def recv(self, blocking: bool = True, timeout: int | float = 1) -> Value:
		return self.unpack(self.recv_bytes(blocking=blocking, timeout=timeout))

	def debug(self, msg: str) -> None:
		self.logger.debug(msg)

	def info(self, msg: str) -> None:
		self.logger.info(msg)

	def warning(self, msg: str) -> None:
		self.logger.warning(msg)

	def error(self, msg: str) -> None:
		self.logger.error(msg)

	def critical(self, msg: str) -> None:
		self.logger.critical(msg)

	def handle(
		self,
		cmd: str | None = None,
		*,
		cb: EngineProxyCallback | None = None,
		**kwargs,
	):
		"""Send a command to the lisien core.

		The only positional argument should be the name of a
		method in :class:``EngineHandle``. All keyword arguments
		will be passed to it, except ``cb``.

		With a function ``cb``, I will call ``cb`` when I get
		a result.
		``cb`` will be called with positional arguments ``command``,
		the name of the command you called; ``result``, the present ``branch``,
		``turn``, and ``tick``, possibly different than when you called
		``handle``.`; and the value returned by the core, possibly ``None``.

		"""
		then = self._btt()
		if self._worker or getattr(
			super(EngineProxy, self), "_mutable_worker", False
		):
			return
		if self.closed:
			raise RuntimeError(f"Already closed: {id(self)}")
		if "command" in kwargs:
			cmd = kwargs["command"]
		elif cmd:
			kwargs["command"] = cmd
		else:
			raise TypeError("No command")
		if hasattr(super(EngineProxy, self), "_replay_file"):
			self._replay_file.write(format_call_sig(cmd, **kwargs) + "\n")
		start_ts = monotonic()
		with self._round_trip_lock:
			self.send(Value(kwargs))
			received = self.recv()
			command, branch, turn, tick, r = received
		self.debug(
			"EngineProxy: received {} in {:,.2f} seconds".format(
				(command, branch, turn, tick), monotonic() - start_ts
			)
		)
		if (branch, turn, tick) != then:
			self._upd_time(command, branch, turn, tick, r)
		if isinstance(r, Exception):
			raise r
		if cmd != command:
			raise RuntimeError(
				f"Sent command {cmd}, but received results for {command}"
			)
		if cb:
			cb(command, branch, turn, tick, r)
		return r

	def _unpack_recv(self) -> Value:
		ret = self.unpack(self.recv_bytes())
		return ret

	def _upd_caches(
		self,
		command: str,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		result: tuple[Value, ...],
	) -> None:
		result, deltas = result
		self.eternal._update_cache(deltas.pop("eternal", {}))
		self.universal._update_cache(deltas.pop("universal", {}))
		# I think if you travel back to before a rule was created
		# it'll show up empty.
		# That's ok I guess
		for rule, delta in deltas.pop("rules", {}).items():
			rule_cached = self._rules_cache.setdefault(rule, {})
			for funcl in ("triggers", "prereqs", "actions"):
				if funcl in delta:
					func_proxies = []
					store = getattr(self, funcl[:-1])
					if delta[funcl] is ...:
						rule_cached[funcl] = []
						continue
					for func in delta[funcl]:
						if hasattr(store, func):
							func_proxy = getattr(store, func)
						else:
							func_proxy = FuncProxy(store, func)
							if hasattr(store, "_proxy_cache"):
								store._proxy_cache[func] = func_proxy
							else:
								# why can this even happen??
								store._proxy_cache = {func: func_proxy}
							if hasattr(store, "_cache"):
								store._cache[func] = store.get_source(func)
							else:
								store._cache = {func: store.get_source(func)}
						func_proxies.append(func_proxy)
					rule_cached[funcl] = func_proxies
			if rule not in self._rule_obj_cache:
				self._rule_obj_cache[rule] = RuleProxy(self, rule)
			ruleproxy = self._rule_obj_cache[rule]
			ruleproxy.send(ruleproxy, **delta)
		rulebookdeltas = deltas.pop("rulebooks", {})
		self._rulebooks_cache.update(rulebookdeltas)
		for rulebook, delta in rulebookdeltas.items():
			if rulebook not in self._rulebook_obj_cache:
				self._rulebook_obj_cache[rulebook] = RuleBookProxy(
					self, rulebook
				)
			rulebookproxy = self._rulebook_obj_cache[rulebook]
			# the "delta" is just the rules list, for now
			rulebookproxy.send(rulebookproxy, rules=delta)
		to_delete = set()
		for char, chardelta in deltas.items():
			if chardelta in (None, ...):
				to_delete.add(char)
				continue
			if char not in self._char_cache:
				self._char_cache[char] = CharacterProxy(self, char)
			chara = self.character[char]
			chara._apply_delta(chardelta)
		for char in to_delete & self._char_cache.keys():
			del self._char_cache[char]

	def _upd_time(
		self,
		command: str,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		result: tuple[Value, ...],
	) -> None:
		then = self._btt()
		self._branch = branch
		self._turn = turn
		self._tick = tick
		if not self._planning:
			if branch not in self._branches_d:
				self._branches_d[branch] = (None, turn, tick, turn, tick)
			else:
				self._extend_branch(branch, turn, tick)
		self.time.send(self, then=then, now=(branch, turn, tick))

	@contextmanager
	def plan(self):
		self._planning = True
		yield self.handle("start_plan")
		self.handle("end_plan", cb=self._upd)
		self._planning = False

	def apply_choices(
		self,
		choices: list[dict],
		dry_run: bool = False,
		perfectionist: bool = False,
	) -> Value:
		if self._worker and not getattr(self, "_mutable_worker", False):
			raise WorkerProcessReadOnlyError(
				"Tried to change the world state in a worker process"
			)
		return self.handle(
			"apply_choices",
			choices=choices,
			dry_run=dry_run,
			perfectionist=perfectionist,
		)

	def _save_and_reimport_all_code(self) -> None:
		for store in (
			self.function,
			self.method,
			self.prereq,
			self.trigger,
			self.action,
		):
			if hasattr(store, "reimport"):
				store.save(reimport=True)

	def _replace_funcs_pkl(
		self,
		*,
		function: Optional[bytes] = None,
		method: Optional[bytes] = None,
		trigger: Optional[bytes] = None,
		prereq: Optional[bytes] = None,
		action: Optional[bytes] = None,
	) -> None:
		assert self._worker, "Replaced code outside of a worker"
		if function:
			self.function._cache = pickle.loads(function)
		if method:
			self.method._cache = pickle.loads(method)
		if trigger:
			self.trigger._cache = pickle.loads(trigger)
		if prereq:
			self.prereq._cache = pickle.loads(prereq)
		if action:
			self.action._cache = pickle.loads(action)

	def _replace_funcs_plain(
		self,
		*,
		function: Optional[dict[str, str]] = None,
		method: Optional[dict[str, str]] = None,
		trigger: Optional[dict[str, str]] = None,
		prereq: Optional[dict[str, str]] = None,
		action: Optional[dict[str, str]] = None,
	) -> None:
		assert self._worker, "Replaced code outside of a worker"
		for name, store, upd in [
			("function", self.function, function),
			("method", self.method, method),
			("trigger", self.trigger, trigger),
			("prereq", self.prereq, prereq),
			("action", self.action, action),
		]:
			if upd is None:
				continue
			for funcname, source in upd.items():
				store._set_source(funcname, source)

	def _upd(self, *args, **kwargs) -> None:
		to_replace_pkl = kwargs.pop("_replace_funcs_pkl", {})
		to_replace_plain = kwargs.pop("_replace_funcs_plain", {})
		if to_replace_pkl or to_replace_plain:
			assert self._worker, "Replaced code outside of a worker"
			if to_replace_pkl:
				self._replace_funcs_pkl(**to_replace_pkl)
			if to_replace_plain:
				self._replace_funcs_plain(**to_replace_plain)
		elif self._worker:
			self._reimport_code()
		else:
			self._save_and_reimport_all_code()
		self._upd_caches(*args, **kwargs)
		self._upd_time(*args, **kwargs)

	def _upd_and_cb(
		self, *args, cb: EngineProxyCallback | None = None, **kwargs
	) -> None:
		self._upd(*args, **kwargs)
		if cb:
			cb(*args, **kwargs)

	def _add_character(
		self,
		char: CharName,
		data: CharDelta | nx.Graph = None,
		layout: bool = False,
		node: NodeValDict | None = None,
		edge: EdgeValDict | None = None,
		**attr: StatDict,
	):
		if char in self._char_cache:
			raise KeyError("Character already exists")
		if isinstance(data, dict):
			try:
				data = nx.from_dict_of_lists(data)
			except nx.NetworkXException:
				data = nx.from_dict_of_dicts(data)
		data: nx.Graph
		if data is not None:
			if not isinstance(data, nx.Graph):
				raise TypeError("Need dict or graph", type(data))
			elif data.is_multigraph():
				raise TypeError("No multigraphs")
			if not data.is_directed():
				data = data.to_directed()
		self._char_cache[char] = character = CharacterProxy(self, char)
		self._char_stat_cache[char] = attr
		placedata = {}
		thingdata = {}
		if data:
			for k, v in data.nodes.items():
				if "location" in v:
					thingdata[k] = v
				else:
					placedata[k] = v
		if node:
			for n, v in node.items():
				if "location" in v:
					thingdata[n] = v
				else:
					placedata[n] = v
		for place, stats in placedata.items():
			if (
				char not in self._character_places_cache
				or place not in self._character_places_cache[char]
			):
				self._character_places_cache[char][place] = PlaceProxy(
					character, place
				)
			self._node_stat_cache[char][place] = stats
		for thing, stats in thingdata.items():
			if "location" not in stats:
				raise ValueError("Things must always have locations")
			loc = stats.pop("location")
			if (
				char not in self._things_cache
				or thing not in self._things_cache[char]
			):
				self._things_cache[char][thing] = ThingProxy(
					character, thing, loc
				)
			self._node_stat_cache[char][thing] = stats
		portdata = edge or {}
		if data:
			for orig, dest in data.edges:
				if orig in portdata:
					portdata[orig][dest] = {}
				else:
					portdata[orig] = {dest: {}}
		for orig, dests in portdata.items():
			for dest, stats in dests.items():
				self._character_portals_cache.store(
					char,
					orig,
					dest,
					PortalProxy(character, orig, dest),
				)
				self._portal_stat_cache[char][orig][dest] = stats
		self.handle(
			command="add_character",
			char=char,
			data=data,
			**attr,
			branching=True,
		)

	def add_character(
		self,
		char: KeyHint | CharName,
		data: CharDelta | None = None,
		layout: bool = False,
		node: NodeValDict | None = None,
		edge: EdgeValDict | None = None,
		**attr: StatDict,
	):
		if self._worker and not getattr(self, "_mutable_worker", False):
			raise WorkerProcessReadOnlyError(
				"Tried to change world state in a worker process"
			)
		self._add_character(
			CharName(Key(char)), data, layout, node, edge, **attr
		)

	def new_character(
		self,
		char: KeyHint | CharName,
		data: CharDelta | None = None,
		layout: bool = False,
		node: NodeValDict | None = None,
		edge: EdgeValDict | None = None,
		**attr: StatDict,
	):
		self.add_character(char, data, layout, node, edge, **attr)
		return self._char_cache[char]

	def _del_character(self, char: CharName) -> None:
		if char not in self._char_cache:
			raise KeyError("No such character")
		del self._char_cache[char]
		if char in self._char_stat_cache:
			del self._char_stat_cache[char]
		if char in self._character_places_cache:
			del self._character_places_cache[char]
		if char in self._things_cache:
			del self._things_cache[char]
		self._character_portals_cache.delete_char(char)
		self.handle(command="del_character", char=char, branching=True)

	def del_character(self, char: KeyHint | CharName) -> None:
		if self._worker and not getattr(self, "_mutable_worker", False):
			raise WorkerProcessReadOnlyError(
				"tried to change world state in a worker process"
			)
		self._del_character(char)

	del_graph = del_character

	def del_node(
		self, char: KeyHint | CharName, node: KeyHint | NodeName
	) -> None:
		char = CharName(Key(char))
		node = NodeName(Key(node))
		if char not in self._char_cache:
			raise KeyError("No such character")
		if (
			node not in self._character_places_cache[char]
			and node not in self._things_cache[char]
		):
			raise KeyError("No such node")
		cont = list(self._node_contents(char, node))
		if node in self._things_cache[char]:
			del self._things_cache[char][node]
		for contained in cont:
			del self._things_cache[char][contained]
		if node in self._character_places_cache[char]:  # just to be safe
			del self._character_places_cache[char][node]
		successors = self._character_portals_cache.successors
		predecessors = self._character_portals_cache.predecessors
		if char in successors and node in successors[char]:
			del successors[char][node]
			if not successors[char]:
				del successors[char]
		if char in predecessors and node in predecessors[char]:
			del predecessors[char][node]
			if not predecessors[char]:
				del predecessors[char]
		nv = self._node_stat_cache
		if char in nv and node in nv[char]:
			del nv[char][node]
			if not nv[char]:
				del nv[char]
		self.handle(command="del_node", char=char, node=node, branching=True)

	def del_portal(
		self,
		char: KeyHint | CharName,
		orig: KeyHint | NodeName,
		dest: KeyHint | NodeName,
	) -> None:
		if char not in self._char_cache:
			raise KeyError("No such character")
		char = CharName(Key(char))
		orig = NodeName(Key(orig))
		dest = NodeName(Key(dest))
		self._character_portals_cache.delete(char, orig, dest)
		ev = self._portal_stat_cache
		if char in ev and orig in ev[char] and dest in ev[char][orig]:
			del ev[char][orig][dest]
			if not ev[char][orig]:
				del ev[char][orig]
			if not ev[char]:
				del ev[char]
		self.handle(
			command="del_portal",
			char=char,
			orig=orig,
			dest=dest,
			branching=True,
		)

	def commit(self):
		self._commit_lock.acquire()
		self.handle("commit", cb=self._release_commit_lock)

	def _release_commit_lock(
		self,
		*,
		command: str,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		result: tuple[ValueHint, ...],
	):
		self._commit_lock.release()

	def close(self) -> None:
		if getattr(self, "closed", False):
			return
		self._commit_lock.acquire()
		self._commit_lock.release()
		self.handle("close")
		self.closed = True

	def _node_contents(
		self, character: KeyHint, node: NodeName
	) -> Iterator[NodeName]:
		# very slow. do better
		for thing in self.character[character].thing.values():
			if thing["location"] == node:
				yield thing.name


@define
class NextTurnProxy(AttrSignal):
	engine: EngineProxy

	def __call__(
		self, cb: Optional[callable] = None
	) -> tuple[list, DeltaDict]:
		return self.engine.handle(
			"next_turn",
			cb=partial(
				self.engine._upd_and_cb, cb=partial(self._send_and_cb, cb)
			),
		)

	def _send_and_cb(self, cb: Optional[callable] = None, *args, **kwargs):
		self.send(self)
		if cb is not None:
			cb(*args, **kwargs)


class RandoProxy(Random):
	"""Proxy to a randomizer"""

	def __init__(self, engine: EngineProxy, seed=None):
		self.engine = engine
		self._handle = engine.handle
		self.gauss_next = None
		if seed:
			self.seed(seed)

	def seed(self, a=None, version=2):
		self._handle(
			cmd="call_randomizer", method="seed", a=a, version=version
		)

	def getstate(self):
		return self._handle(cmd="call_randomizer", method="getstate")

	def setstate(self, state):
		return self._handle(
			cmd="call_randomizer", method="setstate", state=state
		)

	def _randbelow(
		self, n, int=int, maxsize=1, type=type, Method=None, BuiltinMethod=None
	):
		return self._handle(
			cmd="call_randomizer", method="_randbelow", n=n, maxsize=maxsize
		)

	def random(self):
		return self._handle(cmd="call_randomizer", method="random")


@define
class PortalObjCache:
	successors = field(init=False, factory=dict)
	predecessors = field(init=False, factory=dict)

	def store(
		self, char: CharName, u: NodeName, v: NodeName, obj: PortalProxy
	) -> None:
		succ = self.successors
		if char in succ:
			if u in succ[char]:
				succ[char][u][v] = obj
			else:
				succ[char][u] = {v: obj}
		else:
			succ[char] = {u: {v: obj}}
		pred = self.predecessors
		if char in pred:
			if v in pred[char]:
				pred[char][v][u] = obj
			else:
				pred[char][v] = {u: obj}
		else:
			pred[char] = {v: {u: obj}}

	def clear(self):
		self.successors.clear()
		self.predecessors.clear()

	def delete(self, char: CharName, u: NodeName, v: NodeName) -> None:
		succ = self.successors
		if char in succ:
			succ_us = succ[char]
			if u in succ_us and v in succ_us[u]:
				del succ_us[u][v]
			if not succ_us[u]:
				del succ_us[u]
			if not succ_us:
				del succ[char]
		pred = self.predecessors
		if char in pred:
			pred_vs = pred[char]
			if v in pred_vs and u in pred_vs[v]:
				del pred_vs[v][u]
			if not pred_vs[v]:
				del pred_vs[v]
			if not pred_vs:
				del pred[char]

	def delete_char(self, char: CharName) -> None:
		if char in self.successors:
			del self.successors[char]
			del self.predecessors[char]


@define
class CharacterMapProxy(MutableMapping, AttrSignal):
	engine: EngineProxy

	def _worker_check(self):
		self.engine._worker_check()

	def __iter__(self) -> Iterator[CharName]:
		return iter(self.engine._char_cache.keys())

	def __contains__(self, k: KeyHint | CharName):
		return k in self.engine._char_cache

	def __len__(self):
		return len(self.engine._char_cache)

	def __getitem__(self, k: KeyHint | CharName) -> CharacterProxy:
		return self.engine._char_cache[k]

	def __setitem__(self, k: KeyHint | CharName, v: CharDelta):
		self._worker_check()
		k = CharName(Key(k))
		self.engine.handle(
			command="set_character", char=k, data=v, branching=True
		)
		self.engine._char_cache[k] = CharacterProxy(self.engine, k)
		self.send(self, key=k, val=v)

	def __delitem__(self, k: KeyHint | CharName):
		self._worker_check()
		k = CharName(Key(k))
		self.engine.handle(command="del_character", char=k, branching=True)
		for graph, characters in self.engine._unit_characters_cache.items():
			if k in characters:
				del characters[k]
		successors = self.engine._character_portals_cache.successors
		predecessors = self.engine._character_portals_cache.predecessors
		if k in successors:
			del successors[k]
		if k in predecessors:
			del predecessors[k]
		for cache_name in (
			"_char_cache",
			"_node_stat_cache",
			"_portal_stat_cache",
			"_char_stat_cache",
			"_things_cache",
			"_character_places_cache",
			"_character_rulebooks_cache",
			"_char_node_rulebooks_cache",
			"_char_port_rulebooks_cache",
			"_character_units_cache",
		):
			cache = getattr(self.engine, cache_name)
			if k in cache:
				del cache[k]
		self.send(self, key=k, val=None)


class ProxyLanguageDescriptor(AbstractLanguageDescriptor):
	def _get_language(self, inst):
		if not hasattr(inst, "_language"):
			inst._language = inst.engine.handle(command="get_language")
		return inst._language

	def _set_language(self, inst, val):
		inst._language = val
		delta = inst.engine.handle(command="set_language", lang=val)
		cache = inst._cache
		for k, v in delta.items():
			if k in cache:
				if v is None:
					del cache[k]
				elif cache[k] != v:
					cache[k] = v
					inst.send(inst, key=k, string=v)
			elif v is not None:
				cache[k] = v
				inst.send(inst, key=k, string=v)


@define
class StringStoreProxy(AttrSignal):
	engine: EngineProxy
	_cache: dict[str, str] = field(factory=dict)
	_language: str = "eng"
	_store: ClassVar = "strings"
	language: ClassVar = ProxyLanguageDescriptor()

	def _worker_check(self):
		self.engine._worker_check()

	def load(self) -> None:
		self._cache = self.engine.handle("strings_copy")

	def __getattr__(self, k: str) -> str:
		cache = super().__getattribute__("_cache")
		if k in cache:
			return cache[k]
		raise AttributeError(k)

	def __setattr__(self, k: str, v: str):
		if k in (
			"_cache",
			"engine",
			"language",
			"_language",
			"receivers",
			"_by_receiver",
			"_by_sender",
			"_weak_senders",
			"is_muted",
		):
			super().__setattr__(k, v)
			return
		self._worker_check()
		self._cache[k] = v
		self.engine.handle(command="set_string", k=k, v=v)
		self.send(self, key=k, string=v)

	def __delattr__(self, k: str):
		self._worker_check()
		del self._cache[k]
		self.engine.handle(command="del_string", k=k)
		self.send(self, key=k, string=None)

	def lang_items(self, lang: Optional[str] = None):
		if lang is None or lang == self.language:
			yield from self._cache.items()
		else:
			yield from self.engine.handle(
				command="get_string_lang_items", lang=lang
			)


@define
class EternalVarProxy(MutableMapping):
	engine: EngineProxy

	@property
	def _cache(self) -> dict[EternalKey, Value]:
		return self.engine._eternal

	def _worker_check(self):
		self.engine._worker_check()

	def __contains__(self, k: KeyHint | EternalKey):
		return k in self._cache

	def __iter__(self) -> Iterator[EternalKey]:
		return iter(self._cache)

	def __len__(self):
		return len(self._cache)

	def __getitem__(self, k: KeyHint | EternalKey):
		return self._cache[EternalKey(Key(k))]

	def __setitem__(self, k: KeyHint | EternalKey, v: ValueHint | Value):
		self._worker_check()
		k = EternalKey(Key(k))
		v = Value(v)
		self._cache[k] = v
		self.engine.handle("set_eternal", k=k, v=v)

	def __delitem__(self, k: KeyHint | EternalKey):
		self._worker_check()
		k = EternalKey(Key(k))
		del self._cache[k]
		self.engine.handle(command="del_eternal", k=k)

	def _update_cache(self, data: dict[EternalKey, Value]):
		for k, v in data.items():
			if v is None:
				del self._cache[k]
			else:
				self._cache[k] = v


@define
class GlobalVarProxy(MutableMapping, AttrSignal):
	engine: EngineProxy

	@property
	def _cache(self) -> dict[UniversalKey, Value]:
		return self.engine._universal

	def _worker_check(self):
		self.engine._worker_check()

	def __iter__(self) -> Iterator[UniversalKey]:
		return iter(self._cache)

	def __len__(self):
		return len(self._cache)

	def __getitem__(self, k: KeyHint | UniversalKey):
		return self._cache[UniversalKey(Key(k))]

	def __setitem__(self, k: KeyHint | UniversalKey, v: ValueHint | Value):
		self._worker_check()
		k = UniversalKey(Key(k))
		v = Value(v)
		self._cache[k] = v
		self.engine.handle("set_universal", k=k, v=v, branching=True)
		self.send(self, key=k, value=v)

	def __delitem__(self, k: KeyHint | UniversalKey):
		self._worker_check()
		k = UniversalKey(Key(k))
		del self._cache[k]
		self.engine.handle("del_universal", k=k, branching=True)
		self.send(self, key=k, value=...)

	def _update_cache(self, data: dict[UniversalKey, Value]):
		for k, v in data.items():
			if v is ...:
				if k not in self._cache:
					continue
				del self._cache[k]
				self.send(self, key=k, value=...)
			else:
				self._cache[k] = v
				self.send(self, key=k, value=v)


@define
class AllRuleBooksProxy(MutableMapping):
	engine: EngineProxy
	_objs: dict[RulebookName, RuleBookProxy] = field(init=False, factory=dict)

	@property
	def _cache(
		self,
	) -> dict[RulebookName, tuple[list[RuleName], RulebookPriority]]:
		return self.engine._rulebooks_cache

	def __iter__(self) -> Iterator[RulebookName]:
		yield from self._cache

	def __len__(self):
		return len(self._cache)

	def __contains__(self, k: str | RulebookName):
		return k in self._cache

	def __getitem__(self, k: KeyHint | RulebookName):
		k = RulebookName(Key(k))
		if k not in self:
			self.engine.handle("new_empty_rulebook", rulebook=k)
			no_rules: list[RuleName] = []
			zero_prio = RulebookPriority(0.0)
			self._cache[k] = no_rules, zero_prio
		if k not in self._objs:
			self._objs[k] = RuleBookProxy(self.engine, k)
		return self._objs[k]

	def __setitem__(
		self,
		key: KeyHint | RulebookName,
		value: FuncListProxy | Iterable[RuleProxy | str | FuncProxy],
	):
		key = RulebookName(Key(key))
		rules: list[RuleName] = []
		for rule in value:
			if isinstance(rule, str):
				rule = RuleName(rule)
			elif hasattr(rule, "name"):
				rule = RuleName(rule.name)
			elif hasattr(rule, "__name__"):
				rule = RuleName(rule.__name__)
			else:
				raise TypeError("Not a rule", rule)
			rules.append(rule)
		self.engine.handle("set_rulebook_rules", rulebook=key, rules=rules)

	def __delitem__(self, key: KeyHint | RulebookName):
		key = RulebookName(Key(key))
		del self._cache[key]
		self.engine.handle("del_rulebook", rulebook=key)


@define
class AllRulesProxy(MutableMapping):
	engine: EngineProxy
	_proxy_cache: dict[RuleName, RuleProxy] = field(init=False, factory=dict)

	@property
	def _cache(
		self,
	) -> dict[
		RuleName,
		dict[
			Literal["triggers", "prereqs", "actions"],
			list[TriggerFuncName]
			| list[PrereqFuncName]
			| list[ActionFuncName],
		],
	]:
		return self.engine._rules_cache

	def _worker_check(self):
		self.engine._worker_check()

	def __iter__(self):
		return iter(self._cache)

	def __len__(self):
		return len(self._cache)

	def __contains__(self, k: str | RuleName):
		return k in self._cache

	def __getitem__(self, k: str | RuleName):
		k = RuleName(k)
		if k not in self:
			raise KeyError("No rule: {}".format(k))
		if k not in self._proxy_cache:
			self._proxy_cache[k] = RuleProxy(self.engine, k)
		return self._proxy_cache[k]

	def __setitem__(
		self, key: str | RuleName, value: RuleProxy | FuncProxy | str
	):
		key = RuleName(key)
		if isinstance(value, RuleProxy):
			self._proxy_cache[key] = value
		elif callable(value) or hasattr(self.engine.action, value):
			proxy = self._proxy_cache[key] = RuleProxy(self.engine, key)
			proxy.action(value)
		else:
			raise TypeError("Need RuleProxy or an action", type(value))

	def __delitem__(self, key: str | RuleName):
		key = RuleName(key)
		self.engine.handle("del_rule", rule=key)
		if key in self._proxy_cache:
			del self._proxy_cache[key]

	def __call__(
		self,
		action: Callable | FuncProxy | None = None,
		always: bool = False,
		neighborhood: Optional[int] = None,
	):
		if action is None:
			return partial(self, always=always, neighborhood=neighborhood)
		name = getattr(action, "__name__", action)
		self[name] = action
		ret = self[name]
		if always:
			ret.triggers.append(self.engine.trigger.truth)
		if neighborhood is not None:
			ret.neighborhood = neighborhood
		return ret

	def new_empty(self, k: str | RuleName) -> RuleProxy:
		self._worker_check()
		k = RuleName(k)
		self.engine.handle(command="new_empty_rule", rule=k)
		self._cache[k] = {"triggers": [], "prereqs": [], "actions": []}
		self._proxy_cache[k] = RuleProxy(self.engine, k)
		return self._proxy_cache[k]


def _finish_packing(pack, cmd, branch, turn, tick, mostly_bytes):
	r = mostly_bytes

	resp = (
		msgpack_array_header(5)
		+ pack(cmd)
		+ pack(branch)
		+ pack(turn)
		+ pack(tick)
	)
	if isinstance(r, dict):
		resp += msgpack_map_header(len(r))
		for k, v in r.items():
			resp += k + v
	elif isinstance(r, tuple):
		ext = Ext(
			MsgpackExtensionType.tuple.value,
			msgpack_array_header(len(r)) + b"".join(r),
		)

		resp += msgpack.packb(ext)
	elif isinstance(r, list):
		resp += msgpack_array_header(len(r)) + b"".join(r)
	else:
		resp += r
	return resp


class WorkerLogHandler(logging.Handler):
	def __init__(self, logq, level, i):
		super().__init__(level)
		self._logq = logq
		self._i = i

	def emit(self, record):
		record.worker_idx = self._i
		self._logq.put(record)


GET_TIME = b"\x81\xa7command\xa8get_time"
