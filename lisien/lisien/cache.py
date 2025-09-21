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
"""Classes for in-memory storage and retrieval of historical graph data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
from contextlib import contextmanager
from functools import cached_property, wraps
from itertools import chain, pairwise
from operator import itemgetter
from sys import getsizeof, stderr
from threading import RLock
from typing import (
	TYPE_CHECKING,
	Callable,
	Hashable,
	Iterator,
	Literal,
	Optional,
)

from . import engine
from .collections import ChangeTrackingDict
from .exc import (
	HistoricKeyError,
	KeyframeError,
	NotInKeyframeError,
	TotalKeyError,
)
from .types import (
	Branch,
	CharName,
	EntityKey,
	Key,
	LinearTime,
	NodeName,
	Plan,
	RuleBig,
	RulebookName,
	RulebookPriority,
	RuleFuncName,
	RuleName,
	RuleNeighborhood,
	Stat,
	Tick,
	Time,
	Turn,
	Value,
)
from .util import sort_set
from .window import (
	Direction,
	EntikeySettingsTurnDict,
	SettingsTurnDict,
	WindowDict,
)

if TYPE_CHECKING:
	from . import engine


class SizedDict(OrderedDict):
	"""A dictionary that discards old entries when it gets too big."""

	def __init__(self, max_entries: int = 1000):
		self._n = max_entries
		self._lock = RLock()
		super().__init__()

	def __setitem__(self, key, value):
		with self._lock:
			while len(self) > self._n:
				self.popitem(last=False)
			super().__setitem__(key, value)


def _default_args_munger(self, k):
	"""By default, `PickyDefaultDict`'s ``type`` takes no positional arguments."""
	return tuple()


def _default_kwargs_munger(self, k):
	"""By default, `PickyDefaultDict`'s ``type`` takes no keyword arguments."""
	return {}


class PickyDefaultDict(dict):
	"""A ``defaultdict`` alternative that requires values of a specific type.

	Pass some type object (such as a class) to the constructor to
	specify what type to use by default, which is the only type I will
	accept.

	Default values are constructed with no arguments by default;
	supply ``args_munger`` and/or ``kwargs_munger`` to override this.
	They take arguments ``self`` and the unused key being looked up.

	"""

	__slots__ = [
		"type",
		"args_munger",
		"kwargs_munger",
		"parent",
		"key",
		"_lock",
	]

	def __init__(
		self,
		type: type,
		args_munger: callable = _default_args_munger,
		kwargs_munger: callable = _default_kwargs_munger,
	):
		self._lock = RLock()
		self.type = type
		self.args_munger = args_munger
		self.kwargs_munger = kwargs_munger

	def __getitem__(self, k):
		with self._lock:
			if k in self:
				return super(PickyDefaultDict, self).__getitem__(k)
			try:
				ret = self[k] = self.type(
					*self.args_munger(self, k), **self.kwargs_munger(self, k)
				)
			except TypeError:
				raise KeyError(k)
			return ret

	def _create(self, v):
		return self.type(v)

	def __setitem__(self, k, v):
		with self._lock:
			if not isinstance(v, self.type):
				v = self._create(v)
			super(PickyDefaultDict, self).__setitem__(k, v)


class StructuredDefaultDict(dict):
	"""A `defaultdict`-like class with values stored at a specific depth.

	Requires an integer to tell it how many layers deep to go.
	The innermost layer will be ``PickyDefaultDict``, which will take the
	``type``, ``args_munger``, and ``kwargs_munger`` arguments supplied
	to my constructor.

	"""

	__slots__ = (
		"layer",
		"type",
		"args_munger",
		"kwargs_munger",
		"parent",
		"key",
		"_stuff",
		"_lock",
		"gettest",
		"settest",
	)

	def __init__(
		self,
		layers: int,
		type: type = None,
		args_munger: callable = _default_args_munger,
		kwargs_munger: callable = _default_kwargs_munger,
		gettest: callable = lambda k: None,
		settest: callable = lambda k, v: None,
	):
		if layers < 1:
			raise ValueError("Not enough layers")
		self._lock = RLock()
		self.layer = layers
		self.type = type
		self.args_munger = args_munger
		self.kwargs_munger = kwargs_munger
		self._stuff = (layers, type, args_munger, kwargs_munger)
		self.gettest = gettest
		self.settest = settest

	def __getitem__(self, k):
		with self._lock:
			self.gettest(k)
			if k in self:
				return dict.__getitem__(self, k)
			layer, typ, args_munger, kwargs_munger = self._stuff
			if layer == 1:
				if typ is None:
					ret = {}
				else:
					ret = PickyDefaultDict(typ, args_munger, kwargs_munger)
					ret.parent = self
					ret.key = k
			elif layer < 1:
				raise ValueError("Invalid layer")
			else:
				ret = StructuredDefaultDict(
					layer - 1, typ, args_munger, kwargs_munger
				)
				ret.parent = self
				ret.key = k
			dict.__setitem__(self, k, ret)
			return ret

	def __setitem__(self, k, v):
		with self._lock:
			self.settest(k, v)
			if type(v) is StructuredDefaultDict:
				layer, typ, args_munger, kwargs_munger = self._stuff
				if (
					v.layer == layer - 1
					and (typ is None or v.type is typ)
					and v.args_munger is args_munger
					and v.kwargs_munger is kwargs_munger
				):
					super().__setitem__(k, v)
					return
			elif type(v) is PickyDefaultDict:
				layer, typ, args_munger, kwargs_munger = self._stuff
				if (
					layer == 1
					and v.type is typ
					and v.args_munger is args_munger
					and v.kwargs_munger is kwargs_munger
				):
					super().__setitem__(k, v)
					return
			raise TypeError("Can't set layer {}".format(self.layer))


class TurnEndDict(ChangeTrackingDict):
	"""Tick on which a (branch, turn) ends, not including any plans"""

	engine: "engine.Engine"

	def __init__(self, engine):
		self.engine = engine
		super().__init__()

	@cached_property
	def other_d(self) -> TurnEndPlanDict[tuple[Branch, Turn], Tick]:
		return self.engine._turn_end_plan

	def __getitem__(self, item: tuple[Branch, Turn]) -> Tick:
		if item not in self:
			if item not in self.other_d:
				self.other_d[item] = 0
				super().__setitem__(item, 0)
				return Tick(0)
			else:
				ret = super(TurnEndPlanDict, self.other_d).__getitem__(item)
				super().__setitem__(item, ret)
				return ret
		return super().__getitem__(item)

	def __setitem__(self, key: tuple[Branch, Turn], value: Tick):
		super().__setitem__(key, value)
		if key not in self.other_d or self.other_d[key] < value:
			self.other_d[key] = value


class TurnEndPlanDict(TurnEndDict):
	"""Tick on which a (branch, turn) ends, including plans"""

	engine: "engine.Engine"

	@cached_property
	def other_d(self) -> TurnEndDict[tuple[Branch, Turn], Tick]:
		return self.engine._turn_end

	def __setitem__(self, key: tuple[Branch, Turn], value: Tick):
		if key in self.other_d:
			assert value >= self.other_d[key]
		else:
			self.other_d[key] = value
		super().__setitem__(key, value)


class Cache:
	"""A data store that's useful for tracking graph revisions."""

	name: str
	store: callable
	retrieve: callable
	get_keyframe: callable
	set_keyframe: callable
	initial_value = ...

	def __init__(
		self, db: "engine.Engine", name: str, keyframe_dict: dict | None = None
	):
		self.name = name
		self.db = db
		self._keyframe_dict = keyframe_dict

	@cached_property
	def parents(self):
		"""Entity data keyed by the entities' parents.

		An entity's parent is what it's contained in. When speaking of a node,
		this is its graph. When speaking of an edge, the parent is a pair of
		the graph and origin.

		Deeper layers of this cache are keyed by branch and revision.

		"""
		return StructuredDefaultDict(3, SettingsTurnDict)

	@cached_property
	def keys(self):
		"""Cache of entity data keyed by the entities themselves.

		That means the whole tuple identifying the entity is the
		top-level key in this cache here. The second-to-top level
		is the key within the entity.

		Deeper layers of this cache are keyed by branch, turn, and tick.

		"""
		return StructuredDefaultDict(2, SettingsTurnDict)

	@cached_property
	def keycache(self):
		"""Keys an entity has at a given turn and tick."""
		return PickyDefaultDict(SettingsTurnDict)

	@cached_property
	def branches(self):
		"""A less structured alternative to ``keys``.

		For when you already know the entity and the key within it,
		but still need to iterate through history to find the value.

		"""
		return StructuredDefaultDict(1, SettingsTurnDict)

	@cached_property
	def keyframe(self):
		"""Key-value dictionaries representing my state at a given time"""
		return StructuredDefaultDict(
			1, SettingsTurnDict, **(self._keyframe_dict or {})
		)

	@cached_property
	def shallowest(self):
		"""A dictionary for plain, unstructured hinting."""
		return OrderedDict()

	@cached_property
	def settings(self):
		"""All the ``entity[key] = value`` settings on some turn"""
		return PickyDefaultDict(EntikeySettingsTurnDict)

	@cached_property
	def presettings(self):
		"""The values prior to ``entity[key] = value`` settings on some turn"""
		return PickyDefaultDict(EntikeySettingsTurnDict)

	@cached_property
	def time_entity(self):
		return {}

	@cached_property
	def _kc_lru(self):
		return OrderedDict()

	@cached_property
	def _lock(self):
		return RLock()

	@cached_property
	def _store_stuff(self):
		db = self.db
		return (
			self._lock,
			self.parents,
			self.branches,
			self.keys,
			getattr(self, "delete_plan", db.delete_plan),
			db._time_plan,
			db._plan_ticks,
			self._iter_future_contradictions,
			db._extend_branch,
			self._store_journal,
			self.time_entity,
			self.keycache,
			db,
			self._update_keycache,
		)

	@cached_property
	def _remove_stuff(self):
		return (
			self._lock,
			self.time_entity,
			self.parents,
			self.branches,
			self.keys,
			self.settings,
			self.presettings,
			self._remove_keycache,
			self.keycache,
		)

	@cached_property
	def _truncate_stuff(self):
		return (
			self._lock,
			self.parents,
			self.branches,
			self.keys,
			self.settings,
			self.presettings,
			self.keycache,
		)

	@cached_property
	def _store_journal_stuff(
		self,
	) -> tuple[PickyDefaultDict, PickyDefaultDict, callable]:
		return (self.settings, self.presettings, self._base_retrieve)

	def clear(self):
		with self._lock:
			self.parents.clear()
			self.keys.clear()
			self.keycache.clear()
			self.branches.clear()
			self.keyframe.clear()
			self.shallowest.clear()
			self.settings.clear()
			self.time_entity.clear()
			self._kc_lru.clear()

	def total_size(self, handlers=(), verbose=False):
		"""Returns the approximate memory footprint an object and all of its contents.

		Automatically finds the contents of the following builtin containers and
		their subclasses:  tuple, list, deque, dict, set and frozenset.
		To search other containers, add handlers to iterate over their contents:

		    handlers = {SomeContainerClass: iter,
		                OtherContainerClass: OtherContainerClass.get_elements}

		From https://code.activestate.com/recipes/577504-compute-memory-footprint-of-an-object-and-its-cont/download/1/

		"""
		all_handlers = {
			tuple: iter,
			list: iter,
			deque: iter,
			WindowDict: lambda d: [d._past, d._future, d._keys],
			dict: lambda d: chain.from_iterable(d.items()),
			set: iter,
			frozenset: iter,
			Cache: lambda o: [
				o.branches,
				o.settings,
				o.presettings,
				o.keycache,
			],
		}
		all_handlers.update(handlers)
		seen = set()  # track which object id's have already been seen
		default_size = getsizeof(
			0
		)  # estimate sizeof object without __sizeof__

		def sizeof(o):
			if id(o) in seen:  # do not double count the same object
				return 0
			seen.add(id(o))
			s = getsizeof(o, default_size)

			if verbose:
				print(s, type(o), repr(o), file=stderr)

			for typ, handler in all_handlers.items():
				if isinstance(o, typ):
					s += sum(map(sizeof, handler(o)))
					break
			return s

		return sizeof(self)

	def _get_keyframe(
		self,
		graph_ent: tuple[Key, ...],
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> dict:
		if graph_ent not in self.keyframe:
			raise KeyframeError("Unknown graph-entity", graph_ent)
		g = self.keyframe[graph_ent]
		if branch not in g:
			raise KeyframeError("Unknown branch", branch)
		b = g[branch]
		if turn not in b:
			raise KeyframeError("Unknown turn", branch, turn)
		r = b[turn]
		if tick not in r:
			raise KeyframeError("Unknown tick", branch, turn, tick)
		ret = r[tick]
		return ret

	def discard_keyframe(self, branch: Branch, turn: Turn, tick: Tick) -> None:
		for entity in list(self.keyframe):
			if (
				branch in self.keyframe[entity]
				and turn in self.keyframe[entity][branch]
				and tick in self.keyframe[entity][branch][turn]
			):
				del self.keyframe[entity][branch][turn][tick]

	def _set_keyframe(
		self,
		graph_ent: tuple[Key, ...],
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict,
	) -> None:
		if not isinstance(graph_ent, tuple):
			raise TypeError(
				"Keyframes can only be set to tuples identifying graph entities"
			)
		if not isinstance(branch, str):
			raise TypeError("Branches must be strings")
		if not isinstance(turn, int):
			raise TypeError("Turns must be integers")
		if turn < 0:
			raise ValueError("Turns can't be negative")
		if not isinstance(tick, int):
			raise TypeError("Ticks must be integers")
		if tick < 0:
			raise ValueError("Ticks can't be negative")
		kfg = self.keyframe[graph_ent]
		if branch in kfg:
			kfgb = kfg[branch]
			if turn in kfgb:
				kfgb[turn][tick] = keyframe
			else:
				kfgb[turn] = {tick: keyframe}
		else:
			d = SettingsTurnDict()
			d[turn] = {tick: keyframe}
			kfg[branch] = d

	def alias_keyframe(
		self,
		branch_from: Branch,
		branch_to: Branch,
		turn: Turn,
		tick: Tick,
		default: Optional[dict] = None,
	):
		for graph_ent in self.keyframe:
			try:
				kf = self._get_keyframe(graph_ent, branch_from, turn, tick)
			except KeyframeError:
				if default is not None:
					kf = default
				else:
					continue
			self._set_keyframe(graph_ent, branch_to, turn, tick, kf)

	def load(self, data: list[tuple]):
		"""Add a bunch of data. Must be in chronological order.

		But it doesn't need to all be from the same branch, as long as
		each branch is chronological of itself.

		"""

		def sort_key(v):
			if isinstance(v, tuple):
				return (2,) + tuple(map(repr, v))
			if isinstance(v, str):
				return 1, v
			return 0, repr(v)

		branches: defaultdict[Branch, list[tuple]] = defaultdict(list)
		for row in data:
			branches[row[-4]].append(row)
		db = self.db
		# Make keycaches and valcaches. Must be done chronologically
		# to make forwarding work.
		childbranch = db._childbranch
		branch2do = deque([Branch("trunk")])

		store = self.store
		with self.overwriting():
			while branch2do:
				branch = branch2do.popleft()
				for row in sorted(branches[branch], key=sort_key):
					try:
						store(*row, planning=False, loading=True)
					except HistoricKeyError:
						continue
				if branch in childbranch:
					branch2do.extend(childbranch[branch])

	def _valcache_lookup(
		self, cache: dict, branch: Branch, turn: Turn, tick: Tick
	):
		"""Return the value at the given time in ``cache``"""
		for b, r, t in self.db._iter_parent_btt(branch, turn, tick):
			if b in cache:
				if r in cache[b] and cache[b][r].rev_gettable(t):
					try:
						return cache[b][r][t]
					except HistoricKeyError as ex:
						if ex.deleted:
							raise
				elif cache[b].rev_gettable(r - 1):
					cbr = cache[b][r - 1]
					try:
						return cbr.final()
					except HistoricKeyError as ex:
						if ex.deleted:
							raise

	def _get_keycachelike(
		self,
		keycache: dict,
		keys: dict,
		get_adds_dels: callable,
		parentity: tuple[Key, ...],
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool,
	):
		"""Try to retrieve a frozenset representing extant keys.

		If I can't, generate one, store it, and return it.

		"""
		keycache_key = parentity + (branch,)
		keycache2 = keycache3 = None
		if keycache_key in keycache:
			keycache2 = keycache[keycache_key]
			if turn in keycache2:
				keycache3 = keycache2[turn]
				if tick in keycache3:
					return keycache3[tick]
		with self._lock:
			if forward:
				# Take valid values from the past of a keycache and copy them
				# forward, into the present. Assumes that time is only moving
				# forward, never backward, never skipping any turns or ticks,
				# and any changes to the world state are happening through
				# allegedb proper, meaning they'll all get cached. In lisien this
				# means every change to the world state should happen inside of
				# a call to ``Engine.next_turn`` in a rule.
				if keycache2 and keycache2.rev_gettable(turn):
					# there's a keycache from a prior turn in this branch. Get it
					if turn not in keycache2:
						# since it's not this *exact* turn, there might be changes
						old_turn = keycache2.rev_before(turn)
						old_turn_kc = keycache2[turn]
						added, deleted = get_adds_dels(
							parentity,
							branch,
							turn,
							tick,
							stoptime=(branch, old_turn, old_turn_kc.end),
							cache=keys,
						)
						try:
							ret = (
								old_turn_kc.final()
								.union(added)
								.difference(deleted)
							)
						except KeyError:
							ret = frozenset()
						# assert ret == get_adds_dels(
						# keys[parentity], branch, turn, tick)[0]  # slow
						new_turn_kc = WindowDict()
						new_turn_kc[tick] = ret
						keycache2[turn] = new_turn_kc
						return ret
					if not keycache3:
						keycache3 = keycache2[turn]
					if tick not in keycache3:
						if keycache3.rev_gettable(tick):
							added, deleted = get_adds_dels(
								parentity,
								branch,
								turn,
								tick,
								stoptime=(
									branch,
									turn,
									keycache3.rev_before(tick),
								),
								cache=keys,
							)
							ret = (
								keycache3[tick]
								.union(added)
								.difference(deleted)
							)
							# assert ret == get_adds_dels(
							# keys[parentity], branch, turn, tick)[0]  # slow
							keycache3[tick] = ret
							return ret
						else:
							turn_before = keycache2.rev_before(turn)
							if turn_before == turn:
								tick_before = tick
								if keycache2[turn_before].rev_gettable(tick):
									keys_before = keycache2[turn_before][tick]
								else:
									keys_before = frozenset()
							else:
								tick_before = keycache2[turn_before].end
								keys_before = keycache2[turn_before][
									tick_before
								]
							added, deleted = get_adds_dels(
								parentity,
								branch,
								turn,
								tick,
								stoptime=(branch, turn_before, tick_before),
								cache=keys,
							)
							ret = keycache3[tick] = keys_before.union(
								added
							).difference(deleted)
							# assert ret == get_adds_dels(
							# keys[parentity], branch, turn, tick)[0]  # slow
							return ret
					# assert kcturn[tick] == get_adds_dels(
					# keys[parentity], branch, turn, tick)[0]  # slow
					return keycache3[tick]
			# still have to get a stoptime -- the time of the last keyframe
			stoptime, _ = self.db._build_keyframe_window(branch, turn, tick)
			if stoptime is None:
				ret = None
				if parentity in self.keyframe:
					keyframes = self.keyframe[parentity]
					if branch in keyframes:
						kfb = keyframes[branch]
						if turn in kfb:
							kfbr = kfb[turn]
							if tick in kfbr:
								ret = frozenset(kfbr[tick].keys())
				if ret is None:
					adds, _ = get_adds_dels(parentity, branch, turn, tick)
					ret = frozenset(adds)
			elif stoptime == (branch, turn, tick):
				try:
					kf = self._get_keyframe(parentity, branch, turn, tick)
					ret = frozenset(kf.keys())
				except KeyframeError:
					if tick == 0:
						stoptime, _ = self.db._build_keyframe_window(
							branch,
							Turn(turn - 1),
							self.db.turn_end_plan(branch, Turn(turn - 1)),
						)
					else:
						stoptime, _ = self.db._build_keyframe_window(
							branch, turn, Tick(tick - 1)
						)
					if stoptime is None:
						adds, _ = get_adds_dels(parentity, branch, turn, tick)
						ret = frozenset(adds)
					else:
						try:
							kf = self._get_keyframe(parentity, *stoptime)
							adds, dels = get_adds_dels(
								parentity,
								branch,
								turn,
								tick,
								stoptime=stoptime,
							)
							ret = frozenset((kf.keys() | adds) - dels)
						except KeyframeError:
							# entity absent from keyframe, means it was created after that
							adds, _ = get_adds_dels(
								parentity,
								branch,
								turn,
								tick,
								stoptime=stoptime,
							)
							ret = frozenset(adds)
			else:
				try:
					kf = self._get_keyframe(parentity, *stoptime)
					adds, dels = get_adds_dels(
						parentity, branch, turn, tick, stoptime=stoptime
					)
					ret = frozenset((kf.keys() | adds) - dels)
				except KeyframeError:
					adds, _ = get_adds_dels(
						parentity, branch, turn, tick, stoptime=stoptime
					)
					ret = frozenset(adds)
			if keycache2:
				if keycache3:
					keycache3[tick] = ret
				else:
					keycache2[turn] = {tick: ret}
			else:
				kcc = SettingsTurnDict()
				kcc[turn] = {tick: ret}
				keycache[keycache_key] = kcc
			return ret

	def _get_keycache(
		self,
		parentity: tuple[Key, ...],
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool,
	) -> frozenset[Key]:
		"""Get a frozenset of keys that exist in the entity at the moment.

		With ``forward=True``, enable an optimization that copies old key sets
		forward and updates them.

		"""
		return self._get_keycachelike(
			self.keycache,
			self.keys,
			self._get_adds_dels,
			parentity,
			branch,
			turn,
			tick,
			forward=forward,
		)

	def _truncate_keycache(self, parent, entity, branch, turn, tick):
		keycache = self.keycache
		keycache_key = (*parent, entity, branch)
		if keycache_key in keycache:
			thiskeycache = keycache[keycache_key]
			if turn in thiskeycache:
				thiskeycache[turn].truncate(tick)
				if not thiskeycache[turn]:
					del thiskeycache[turn]
			else:
				thiskeycache.truncate(turn)
			if not thiskeycache:
				del keycache[keycache_key]

	def _update_keycache(self, *args, forward: bool):
		"""Add or remove a key in the set describing the keys that exist."""
		entity: Key
		key: Key
		branch: Branch
		turn: Turn
		tick: Tick
		value: Value
		entity, key, branch, turn, tick, value = args[-6:]
		parent: tuple[Key, ...] = args[:-6]
		self._truncate_keycache(parent, entity, branch, turn, tick)
		if self.db._no_kc:
			return
		kc = self._get_keycache(
			parent + (entity,), branch, turn, tick, forward=forward
		)
		if self._count_as_deleted(value):
			kc = kc.difference((key,))
		else:
			kc = kc.union((key,))
		parentibranch = (*parent, entity, branch)
		if parentibranch not in self.keycache:
			self.keycache[parentibranch] = SettingsTurnDict({turn: {tick: kc}})
		else:
			self.keycache[parentibranch][turn][tick] = kc

	@staticmethod
	def _count_as_deleted(obj):
		return obj is ...

	def _get_adds_dels(
		self,
		entity: tuple[Key, ...],
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		stoptime: Time | None = None,
		cache: Optional[dict] = None,
	):
		"""Return a pair of sets describing changes to the entity's keys

		Returns a pair of sets: ``(added, deleted)``. These are the changes
		to the key set that occurred since ``stoptime``, which, if present,
		should be a triple ``(branch, turn, tick)``.

		With ``stoptime=None`` (the default), ``added`` will in fact be all
		keys, and ``deleted`` will be empty.

		"""
		# Not using the journal because that doesn't distinguish entities.
		# I think I might not want to use ``stoptime`` at all, now that
		# there is such a thing as keyframes...
		cache = cache or self.keys
		added = set()
		deleted = set()
		kf = self.keyframe.get(entity, None)
		for key, branches in cache.get(entity, {}).items():
			for branc, trn, tck in self.db._iter_parent_btt(
				branch, turn, tick, stoptime=stoptime
			):
				if branc not in branches or not branches[branc].rev_gettable(
					trn
				):
					continue
				turnd = branches[branc]
				if trn in turnd:
					if turnd[trn].rev_gettable(tck):
						if self._count_as_deleted(turnd[trn][tck]):
							deleted.add(key)
						else:
							added.add(key)
						break
					else:
						trn -= 1
				if not turnd.rev_gettable(trn):
					break
				tickd = turnd[trn]
				if self._count_as_deleted(tickd.final()):
					deleted.add(key)
				else:
					added.add(key)
				break
		if not kf:
			return added, deleted
		for branc, trn, tck in self.db._iter_parent_btt(
			branch, turn, tick, stoptime=stoptime
		):
			if branc not in kf or not kf[branc].rev_gettable(trn):
				continue
			kfb = kf[branc]
			if trn in kfb and kfb[trn].rev_gettable(tck):
				added.update(set(kfb[trn][tck]).difference(deleted))
			elif kfb.rev_gettable(trn):
				try:
					additions = set(kfb[trn].final())
				except KeyError:
					additions = set()
				added.update(additions.difference(deleted))
			else:
				continue
			break
		return added, deleted

	def _store(
		self,
		*args,
		planning: bool | None = None,
		forward: bool | None = None,
		loading: bool = False,
		contra: Optional[bool] = None,
	) -> None:
		"""Put a value in various dictionaries for later .retrieve(...).

		Needs at least five arguments, of which the -1th is the value
		to store, the -2th is the tick to store it at, the -3th
		is the turn to store it in, the -4th is the branch the
		revision is in, the -5th is the key the value is for,
		and the remaining arguments identify the entity that has
		the key, eg. a graph, node, or edge.

		With ``planning=True``, you will be permitted to alter
		"history" that takes place after the last non-planning
		moment of time, without much regard to consistency.
		Otherwise, contradictions will be handled by deleting
		everything in the contradicted plan after the present moment,
		unless you set ``contra=False``.

		``loading=True`` prevents me from updating the ORM's records
		of the ends of branches and turns.

		"""
		(
			lock,
			self_parents,
			self_branches,
			self_keys,
			delete_plan,
			time_plan,
			plan_ticks,
			self_iter_future_contradictions,
			db_extend_branch,
			self_store_journal,
			self_time_entity,
			keycache,
			db,
			update_keycache,
		) = self._store_stuff
		if planning is None:
			planning = db._planning
		if forward is None:
			forward = db._forward
		if contra is None:
			contra = not loading
		entity: Key
		key: Key
		branch: Branch
		turn: Turn
		tick: Tick
		value: Value
		entity, key, branch, turn, tick, value = args[-6:]
		if loading:
			self.db._updload(branch, turn, tick)
		parent: tuple[Key, ...] = args[:-6]
		entikey = (entity, key)
		parentikey = parent + (entity, key)
		contras: list[LinearTime]
		with lock:
			self_store_journal(*args)
			if parent:
				parentity = self_parents[parent][entity]
				if key in parentity:
					branches = parentity[key]
					turns = branches[branch]
				else:
					branches = self_branches[parentikey] = self_keys[
						parent + (entity,)
					][key] = parentity[key]
					turns = branches[branch]
			else:
				if entikey in self_branches:
					branches = self_branches[entikey]
					turns = branches[branch]
				else:
					branches = self_branches[entikey]
					self_keys[entity,][key] = branches
					turns = branches[branch]
			if contra:
				contras = list(
					self_iter_future_contradictions(
						entity, key, turns, branch, turn, tick, value
					)
				)
				if contras:
					self.shallowest = {}
				for contra_turn, contra_tick in contras:
					if (
						branch,
						contra_turn,
						contra_tick,
					) in time_plan:  # could've been deleted in this very loop
						delete_plan(
							time_plan[branch, contra_turn, contra_tick]
						)
			if planning:
				if (
					not isinstance(self, NodeContentsCache)
					and turn in turns
					and tick < turns[turn].end
				):
					raise HistoricKeyError(
						"Already have some ticks after {} in turn {} of branch {}".format(
							tick, turn, branch
						)
					)
				plan = time_plan[branch, turn, tick] = db._last_plan
				ticks = plan_ticks[plan][branch][turn]
				ticks.append(tick)
				plan_ticks[plan][branch][turn] = ticks
			branches[branch] = turns
			if not loading and not planning:
				db_extend_branch(branch, turn, tick)
			self.shallowest[(*parent, entity, key, branch, turn, tick)] = value
			turns.store_at(turn, tick, value)
			self_time_entity[branch, turn, tick] = parent, entity, key
			if not loading:
				update_keycache(*args, forward=forward)

	def remove_character(self, character):
		(
			lock,
			time_entity,
			parents,
			branches,
			keys,
			settings,
			presettings,
			remove_keycache,
			keycache,
		) = self._remove_stuff
		todel = {
			(branch, turn, tick, parent, entity, key)
			for (
				(branch, turn, tick),
				(parent, entity, key),
			) in time_entity.items()
			if (parent and parent[0] == character)
			or (not parent and entity == character)
		}
		todel_shallow = {k for k in self.shallowest if k[0] == character}
		with lock:
			for k in todel_shallow:
				del self.shallowest[k]
			for branch, turn, tick, parent, entity, key in todel:
				self._remove_btt_parentikey(
					branch, turn, tick, parent, entity, key
				)

	def remove_branch(self, branch: str):
		(
			lock,
			time_entity,
			parents,
			branches,
			keys,
			settings,
			presettings,
			remove_keycache,
			keycache,
		) = self._remove_stuff
		todel = {
			(branc, turn, tick, parent, entity, key)
			for (
				(branc, turn, tick),
				(parent, entity, key),
			) in time_entity.items()
			if branc == branch
		}
		todel_shallow = {k for k in self.shallowest if k[-2] == branch}
		with lock:
			for k in todel_shallow:
				del self.shallowest[k]
			for branc, turn, tick, parent, entity, key in todel:
				self._remove_btt_parentikey(
					branc, turn, tick, parent, entity, key
				)
				if (
					*parent,
					entity,
					key,
					branc,
					turn,
					tick,
				) in self.shallowest:
					del self.shallowest[
						(*parent, entity, key, branc, turn, tick)
					]

	def _remove_btt_parentikey(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		parent: tuple[Key, ...],
		entity: Key,
		key: Key,
	):
		(
			_,
			time_entity,
			parents,
			branches,
			keys,
			settings,
			presettings,
			remove_keycache,
			keycache,
		) = self._remove_stuff
		try:
			del time_entity[branch][turn][tick]
		except KeyError:
			pass
		branchkey = parent + (entity, key)
		keykey = parent + (entity,)
		if parent in parents:
			parentt = parents[parent]
			if entity in parentt:
				entty = parentt[entity]
				if key in entty:
					kee = entty[key]
					if branch in kee:
						del kee[branch]
					if not kee:
						del entty[key]
				if not entty:
					del parentt[entity]
			if not parentt:
				del parents[parent]
		if branchkey in branches:
			entty = branches[branchkey]
			if branch in entty:
				del entty[branch]
			if not entty:
				del branches[branchkey]
		if keykey in keys:
			entty = keys[keykey]
			if key in entty:
				kee = entty[key]
				if branch in kee:
					del kee[branch]
				if not kee:
					del entty[key]
			if not entty:
				del keys[keykey]

	def discard(self, branch: Branch, turn: Turn, tick: Tick):
		"""Delete all data from a specific tick, if present"""
		if (branch, turn, tick) not in self.time_entity:
			return
		self.remove(branch, turn, tick)

	def remove(self, branch: Branch, turn: Turn, tick: Tick):
		"""Delete all data from a specific tick"""
		(
			lock,
			time_entity,
			parents,
			branches,
			keys,
			settings,
			presettings,
			remove_keycache,
			keycache,
		) = self._remove_stuff
		parent, entity, key = time_entity[branch, turn, tick]
		branchkey = parent + (entity, key)
		keykey = parent + (entity,)
		with lock:
			if parent in parents:
				parentt = parents[parent]
				if entity in parentt:
					entty = parentt[entity]
					if key in entty:
						kee = entty[key]
						if branch in kee:
							branhc = kee[branch]
							if turn in branhc:
								trn = branhc[turn]
								del trn[tick]
								if not trn:
									del branhc[turn]
								if not branhc:
									del kee[branch]
						if not kee:
							del entty[key]
					if not entty:
						del parentt[entity]
				if not parentt:
					del parents[parent]
			if branchkey in branches:
				entty = branches[branchkey]
				if branch in entty:
					branhc = entty[branch]
					if turn in branhc:
						trn = branhc[turn]
						if tick in trn:
							del trn[tick]
						if not trn:
							del branhc[turn]
					if not branhc:
						del entty[branch]
				if not entty:
					del branches[branchkey]
			if keykey in keys:
				entty = keys[keykey]
				if key in entty:
					kee = entty[key]
					if branch in kee:
						branhc = kee[branch]
						if turn in branhc:
							trn = entty[turn]
							if tick in trn:
								del trn[tick]
							if not trn:
								del branhc[turn]
						if not branhc:
							del kee[branch]
					if not kee:
						del entty[key]
				if not entty:
					del keys[keykey]
			if branch in settings:
				branhc = settings[branch]
				if turn in branhc:
					trn = branhc[turn]
					if tick in trn:
						del trn[tick]
					if not trn:
						del branhc[turn]
				if not branhc:
					del settings[branch]
			if branch in presettings:
				pbranhc = presettings[branch]
				if turn in pbranhc:
					ptrn = pbranhc[turn]
					if tick in ptrn:
						del ptrn[tick]
					if not ptrn:
						del pbranhc[turn]
				if not pbranhc:
					del presettings[branch]
			self.shallowest = OrderedDict()
			remove_keycache((*parent, entity, branch), turn, tick)

	def _remove_keycache(
		self, entity_branch: tuple[Key, ..., Branch], turn: Turn, tick: Tick
	):
		"""Remove the future of a given entity from a branch in the keycache"""
		keycache = self.keycache
		if entity_branch in keycache:
			kc = keycache[entity_branch]
			if turn in kc:
				kcturn = kc[turn]
				if tick in kcturn:
					del kcturn[tick]
				kcturn.truncate(tick)
				if not kcturn:
					del kc[turn]
			kc.truncate(turn)
			if not kc:
				del keycache[entity_branch]

	def truncate(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		direction: Direction = Direction.FORWARD,
	):
		direction = Direction(direction)
		(lock, parents, branches, keys, settings, presettings, keycache) = (
			self._truncate_stuff
		)

		def truncate_branhc(branhc):
			if turn in branhc:
				trn = branhc[turn]
				trn.truncate(tick, direction)
				branhc.truncate(turn, direction)
				if turn in branhc and not branhc[turn]:
					del branhc[turn]
			else:
				branhc.truncate(turn, direction)

		with lock:
			for entities in parents.values():
				for keys in entities.values():
					for branches in keys.values():
						if branch not in branches:
							continue
						truncate_branhc(branches[branch])
			for branches in branches.values():
				if branch not in branches:
					continue
				truncate_branhc(branches[branch])
			for keys in keys.values():
				for branches in keys.values():
					if branch not in branches:
						continue
					truncate_branhc(branches[branch])
			truncate_branhc(settings[branch])
			truncate_branhc(presettings[branch])
			self.shallowest = OrderedDict()
			for entity_branch in keycache:
				if entity_branch[-1] == branch:
					truncate_branhc(keycache[entity_branch])

	@staticmethod
	def _iter_future_contradictions(
		entity: Key,
		key: Key,
		turns: WindowDict,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
	):
		"""Iterate over contradicted ``(turn, tick)`` if applicable"""
		# assumes that all future entries are in the plan
		if not turns:
			return
		if turn in turns:
			future_ticks = turns[turn].future(tick, include_same_rev=False)
			for tck, newval in future_ticks.items():
				if newval != value:
					yield turn, tck
			future_turns = turns.future(turn, include_same_rev=False)
		elif turns.rev_gettable(turn):
			future_turns = turns.future(turn, include_same_rev=True)
		else:
			future_turns = turns
		if not future_turns:
			return
		for trn, ticks in future_turns.items():
			for tick, newval in ticks.items():
				if newval != value:
					yield trn, tick

	@contextmanager
	def overwriting(self):
		if hasattr(self, "overwrite_journal"):
			yield
			return
		self.overwrite_journal = True
		yield
		del self.overwrite_journal

	def __repr__(self):
		return f"<{type(self)} named {self.name} at {hex(id(self))}>"

	def _store_journal(self, *args):
		# overridden in lisien.cache.InitializedCache
		(settings, presettings, base_retrieve) = self._store_journal_stuff
		entity: Key
		key: Key
		branch: Branch
		turn: Turn
		tick: Tick
		value: Value
		entity, key, branch, turn, tick, value = args[-6:]
		parent = args[:-6]
		settings_turns = settings[branch]
		presettings_turns = presettings[branch]
		prev = base_retrieve(args[:-1], store_hint=False)
		if isinstance(prev, KeyError):
			prev = self.initial_value
		if turn in settings_turns:
			# These assertions hold for most caches but not for the contents
			# caches, and are therefore commented out.
			# assert turn in presettings_turns \
			# or turn in presettings_turns.future()
			setticks = settings_turns[turn]
			if not hasattr(self, "overwrite_journal") and tick in setticks:
				raise KeyError(
					"Already have journal entry", self.name, branch, turn, tick
				)
			# assert tick not in setticks
			presetticks = presettings_turns[turn]
			# assert tick not in presetticks
			presetticks[tick] = parent + (entity, key, prev)
			setticks[tick] = parent + (entity, key, value)
		else:
			presettings_turns[turn] = {tick: parent + (entity, key, prev)}
			settings_turns[turn] = {tick: parent + (entity, key, value)}

	def _base_retrieve(
		self,
		args,
		store_hint: bool = True,
		retrieve_hint: bool = True,
		search: bool = False,
	):
		"""Hot code.

		Swim up the timestream trying to find a value for the
		key in the entity that applied at the given (branch, turn, tick).
		If we hit a keyframe, return the value therein, or KeyError if
		there is none.

		May *return* an exception, rather than raising it. This is to enable
		use outside try-blocks, which have some performance overhead.

		Memoized by default. Use ``store_hint=False`` to avoid making a memo,
		``retrieve_hint=False`` to avoid using one already made.

		With ``search=True``, use binary search. This isn't the default,
		because most retrievals are close to each other.

		"""
		shallowest = self.shallowest
		if retrieve_hint and args in shallowest:
			return shallowest[args]
		entity: tuple[Key, ...] = args[:-4]
		key: Key
		branch: Branch
		turn: Turn
		tick: Tick
		key, branch, turn, tick = args[-4:]
		keyframes = self.keyframe.get(entity, {})
		branches = self.branches
		entikey = entity + (key,)

		def get(d: WindowDict, k: int):
			if search:
				return d.search(k)
			else:
				return d[k]

		def hint(v):
			if store_hint:
				shallowest[args] = v
			return v

		if entikey in branches:
			branchentk = branches[entikey]
			# We have data for this entity and key,
			# but a keyframe might have more recent data.
			# Iterate over the keyframes in reverse chronological order
			# and return either the first value in a keyframe for this
			# entity and key, or the first value in our own
			# store, whichever took effect later.
			it = pairwise(
				self.db._iter_keyframes(
					branch, turn, tick, loaded=True, with_fork_points=True
				)
			)
			try:
				zero, one = next(it)
				if zero == (branch, turn, tick):
					it = chain([(zero, one)], it)
				else:
					it = chain([((branch, turn, tick), zero), (zero, one)], it)
			except StopIteration:
				# There is at most one keyframe in the past.
				# If branches has anything later than that, before the present,
				# use branches. Otherwise, defer to the keyframe.
				def get_chron(b, r, t):
					if b in branchentk:
						if r in branchentk[b]:
							if branchentk[b][r].rev_gettable(t):
								return hint(branchentk[b][r][t])
							elif (
								branchentk[b].rev_before(r - 1, search=search)
								is not None
							):
								return hint(branchentk[b][r - 1].final())
						elif (
							branchentk[b].rev_before(r, search=search)
							is not None
						):
							return hint(branchentk[b][r].final())
					return KeyError("Not in chron data", b, r, t)

				kfit = self.db._iter_keyframes(branch, turn, tick, loaded=True)
				try:
					stoptime = next(kfit)
					for b, r, t in self.db._iter_parent_btt(
						branch, turn, tick, stoptime=stoptime
					):
						ret = get_chron(b, r, t)
						if isinstance(ret, KeyError):
							continue
						return hint(ret)
					b, r, t = stoptime
					if (
						b in keyframes
						and r in keyframes[b]
						and t in keyframes[b][r]
					):
						kf = keyframes[b][r][t]
						if key in kf:
							return hint(kf[key])
						else:
							return hint(
								NotInKeyframeError(
									"No value", entikey, b, r, t
								)
							)
				except StopIteration:
					# There are no keyframes in the past at all.
					for b, r, t in self.db._iter_parent_btt(
						branch, turn, tick
					):
						ret = get_chron(b, r, t)
						if isinstance(ret, KeyError):
							continue
						return hint(ret)
					return TotalKeyError(
						"No keyframe loaded", entikey, branch, turn, tick
					)
				if (
					b in keyframes
					and r in keyframes[b]
					and t in keyframes[b][r]
				):
					kf = keyframes[b][r][t]
					if key in kf:
						return hint(kf[key])
					else:
						return NotInKeyframeError("No value", entikey, b, r, t)
				else:
					return TotalKeyError(
						"No keyframe loaded", entikey, b, r, t
					)
			for (b0, r0, t0), (b1, r1, t1) in it:
				if (b0, r0, t0) in self.db._keyframes_loaded and (
					b0 in keyframes
					and r0 in keyframes[b0]
					and t0 in keyframes[b0][r0]
				):
					# There's a keyframe at this exact moment. Use it.
					kf = keyframes[b0][r0][t0]
					if key in kf:
						return hint(kf[key])
					else:
						return hint(
							NotInKeyframeError("No value", entikey, b0, r0, t0)
						)
				if b0 in branchentk and (
					(
						r0 in branchentk[b0]
						and branchentk[b0][r0].rev_gettable(t0)
					)
					or branchentk[b0].rev_gettable(r0)
				):
					if r0 in branchentk[b0] and branchentk[b0][
						r0
					].rev_gettable(t0):
						return hint(branchentk[b0][r0][t0])
					else:
						return hint(branchentk[b0][r0].final())
				elif b0 in branchentk and (
					r0 != r1
					and branchentk[b0].rev_gettable(r0)
					and (
						(
							branchentk[b0].rev_before(r0, search=search) == r1
							and get(branchentk[b0], r0).end > t1
						)
						or branchentk[b0].rev_before(r0, search=search) > r1
					)
				):
					# branches does not have a value *this* turn,
					# but has one for a prior turn, and it's still between
					# the two keyframes.
					return hint(branchentk[b0][r0 - 1].final())
				elif (b1, r1, t1) in self.db._keyframes_loaded:
					# branches has no value between these two keyframes,
					# but we have the keyframe further back.
					# Which doesn't mean any of its data is stored in
					# this cache, though.
					if (
						b1 not in keyframes
						or r1 not in keyframes[b1]
						or t1 not in keyframes[b1][r1]
					):
						return hint(
							NotInKeyframeError("No value", entikey, b1, r1, t1)
						)
					brtk = keyframes[b1][r1][t1]
					if key in brtk:
						return hint(brtk[key])
					else:
						return hint(
							NotInKeyframeError("No value", entikey, b1, r1, t1)
						)
		elif keyframes:
			# We have no chronological data, just keyframes.
			# That makes things easy.
			for b0, r0, t0 in self.db._iter_keyframes(
				branch, turn, tick, loaded=True
			):
				if (
					b0 not in keyframes
					or r0 not in keyframes[b0]
					or t0 not in keyframes[b0][r0]
					or key not in keyframes[b0][r0][t0]
				):
					return hint(
						NotInKeyframeError("No value", entikey, b0, r0, t0)
					)
				return hint(keyframes[b0][r0][t0][key])
		return hint(TotalKeyError("No value, ever", entikey))

	def _retrieve(self, *args, search: bool = False) -> Value:
		"""Get a value previously .store(...)'d.

		Needs at least five arguments. The -1th is the tick
		within the turn you want,
		the -2th is that turn, the -3th is the branch,
		and the -4th is the key. All other arguments identify
		the entity that the key is in.

		With ``search=True``, use binary search; otherwise,
		seek back and forth like a tape head.

		"""
		ret = self._base_retrieve(args, search=search)
		if ret is ...:
			raise HistoricKeyError("Set, then deleted", deleted=True)
		elif isinstance(ret, Exception):
			ret.args = (*ret.args, args)
			raise ret
		return ret

	def _iter_entities_or_keys(
		self, *args, forward: bool | None = None
	) -> Iterator[Key]:
		"""Iterate over the keys an entity has, if you specify an entity.

		Otherwise, iterate over the entities themselves, or at any rate the
		tuple specifying which entity.

		"""
		if forward is None:
			forward = self.db._forward
		entity: tuple[Key, ...] = args[:-3]
		branch: Branch
		turn: Turn
		tick: Tick
		branch, turn, tick = args[-3:]
		if self.db._no_kc:
			kc = self._get_adds_dels(entity, branch, turn, tick)[0]
		else:
			try:
				kc = self._get_keycache(
					entity, branch, turn, tick, forward=forward
				)
			except KeyframeError:
				return
		for that in kc:
			if not self._contains_entity_or_key(
				*entity, that, branch, turn, tick
			):
				raise RuntimeError(
					"Bad keycache", entity, that, branch, turn, tick
				)
			yield that

	def _count_entities_or_keys(self, *args, forward: bool | None = None):
		"""Return the number of keys an entity has, if you specify an entity.

		Otherwise, return the number of entities.

		"""
		if forward is None:
			forward = self.db._forward
		entity: tuple[Key, ...] = args[:-3]
		branch: Branch
		turn: Turn
		tick: Tick
		branch, turn, tick = args[-3:]
		if self.db._no_kc:
			return len(self._get_adds_dels(entity, branch, turn, tick)[0])
		return len(
			self._get_keycache(entity, branch, turn, tick, forward=forward)
		)

	def _contains_entity_or_key(self, *args, search: bool = False):
		"""Check if an entity has a key at the given time, if entity specified.

		Otherwise, check if the entity exists.

		"""
		retr = self._base_retrieve(args, search=search)
		return not (
			self._count_as_deleted(retr) or isinstance(retr, Exception)
		)


class GraphValCache(Cache):
	def get_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		copy: bool = True,
	) -> dict[Stat, Value]:
		ret = self._get_keyframe((graph,), branch, turn, tick)
		if copy:
			ret = ret.copy()
		return ret

	def set_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict[Stat, Value],
	) -> None:
		self._set_keyframe((graph,), branch, turn, tick, keyframe)

	def store(
		self,
		graph: CharName,
		key: Stat,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
		*,
		planning: bool | None = None,
		forward: bool | None = None,
		loading: bool = False,
		contra: bool | None = None,
	) -> None:
		self._store(
			graph,
			key,
			branch,
			turn,
			tick,
			value,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)

	def retrieve(
		self,
		graph: CharName,
		key: Stat,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> Value:
		return self._retrieve(graph, key, branch, turn, tick, search=search)

	def contains_key(
		self,
		graph: CharName,
		key: Stat,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	) -> bool:
		return self._contains_entity_or_key(graph, key, branch, turn, tick)

	def count_keys(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> int:
		return self._count_entities_or_keys(
			graph, branch, turn, tick, forward=forward
		)

	def count_entities(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> int:
		return self._count_entities_or_keys(
			branch, turn, tick, forward=forward
		)

	def iter_keys(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> Iterator[Stat]:
		return self._iter_entities_or_keys(
			graph, branch, turn, tick, forward=forward
		)


class NodesCache(Cache):
	"""A cache for remembering whether nodes exist at a given time."""

	__slots__ = ()

	initial_value = False

	def store(
		self,
		graph: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		ex: bool,
		*,
		planning: bool | None = None,
		forward: bool | None = None,
		loading: bool = False,
		contra: bool | None = None,
	):
		self._store(
			graph,
			node,
			branch,
			turn,
			tick,
			ex,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)

	@staticmethod
	def _count_as_deleted(obj):
		return not obj

	def retrieve(
		self,
		graph: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> bool:
		return bool(
			self._retrieve(graph, node, branch, turn, tick, search=search)
		)

	def iter_nodes(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> Iterator[NodeName]:
		return self._iter_entities_or_keys(
			graph, branch, turn, tick, forward=forward
		)

	iter_entities = iter_nodes

	def count_nodes(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> int:
		return self._count_entities_or_keys(
			graph, branch, turn, tick, forward=forward
		)

	count_entities = count_nodes

	def node_exists(
		self,
		graph: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> bool:
		return self._contains_entity_or_key(
			graph, node, branch, turn, tick, search=search
		)

	contains_entity = node_exists

	def _update_keycache(self, *args, forward):
		graph: CharName
		node: NodeName
		branch: Branch
		turn: Turn
		tick: Tick
		ex: Optional[bool]
		graph, node, branch, turn, tick, ex = args
		if not ex:
			ex = ...
		super()._update_keycache(
			graph, node, branch, turn, tick, ex, forward=forward
		)

	def _iter_future_contradictions(
		self,
		entity: Key,
		key: Key,
		turns: WindowDict,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
	):
		yield from super()._iter_future_contradictions(
			entity, key, turns, branch, turn, tick, value
		)
		yield from self.db._edges_cache._slow_iter_node_contradicted_times(
			branch, turn, tick, entity, key
		)

	def get_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		copy: bool = True,
	) -> dict[NodeName, bool]:
		ret = self._get_keyframe((graph,), branch, turn, tick)
		if copy:
			ret = ret.copy()
		return ret

	def set_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict[NodeName, bool],
	) -> None:
		self._set_keyframe((graph,), branch, turn, tick, keyframe)


class NodeValCache(Cache):
	def get_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		copy=True,
	) -> dict[NodeName, dict[Stat, Value]]:
		ret = self._get_keyframe((graph,), branch, turn, tick)
		if copy:
			ret = ret.copy()
			for k, v in ret.items():
				ret[k] = v.copy()
		return ret

	def set_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict[NodeName, dict[Stat, Value]],
	):
		self._set_keyframe((graph,), branch, turn, tick, keyframe)
		for node, vals in keyframe.items():
			self._set_keyframe((graph, node), branch, turn, tick, vals)

	def store(
		self,
		graph: CharName,
		node: NodeName,
		key: Stat,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
		*,
		planning: bool | None = None,
		forward: bool | None = None,
		loading: bool = False,
		contra: bool | None = None,
	) -> None:
		self._store(
			graph,
			node,
			key,
			branch,
			turn,
			tick,
			value,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)

	def retrieve(
		self,
		graph: CharName,
		node: NodeName,
		key: Stat,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> Value:
		return self._retrieve(
			graph, node, key, branch, turn, tick, search=search
		)

	def iter_keys(
		self,
		graph: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> Iterator[Stat]:
		return self._iter_entities_or_keys(
			graph, node, branch, turn, tick, forward=forward
		)

	def count_keys(
		self,
		graph: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> int:
		return self._count_entities_or_keys(
			graph, node, branch, turn, tick, forward=forward
		)

	def contains_key(
		self,
		graph: CharName,
		node: NodeName,
		key: Stat,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> bool:
		return self._contains_entity_or_key(
			graph, node, key, branch, turn, tick, search=search
		)


class EdgesCache(Cache):
	"""A cache for remembering whether edges exist at a given time."""

	__slots__ = (
		"origcache",
		"predecessors",
		"_origcache_lru",
		"_get_origcache_stuff",
		"_additional_store_stuff",
	)

	initial_value = False

	@property
	def successors(self):
		return self.parents

	def __init__(
		self,
		db: "engine.Engine",
		name: str,
		keyframe_dict: Optional[dict] = None,
	):
		super().__init__(db, name, keyframe_dict)
		self.origcache = PickyDefaultDict(SettingsTurnDict)
		self.predecessors = StructuredDefaultDict(3, SettingsTurnDict)
		self._origcache_lru = OrderedDict()
		self._get_origcache_stuff: tuple[
			PickyDefaultDict,
			OrderedDict,
			callable,
			StructuredDefaultDict,
			callable,
		] = (
			self.origcache,
			self._origcache_lru,
			self._get_keycachelike,
			self.predecessors,
			self._adds_dels_predecessors,
		)
		self._additional_store_stuff = (
			self.db,
			self.predecessors,
			self.successors,
		)

	def total_size(
		self, handlers: Optional[dict] = None, verbose: bool = False
	):
		all_handlers = {
			EdgesCache: lambda e: [
				e.predecessors,
				e.successors,
				e.origcache,
			]
		}
		if handlers:
			all_handlers.update(handlers)
		return super().total_size(all_handlers, verbose)

	def get_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		copy: bool = True,
	) -> dict[NodeName, dict[NodeName, bool]]:
		ret = self._get_keyframe((graph,), branch, turn, tick)
		if copy:
			ret = ret.copy()
			for orig, dests in list(ret.items()):
				ret[orig] = dests.copy()
		return ret

	def set_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict[NodeName, dict[NodeName, bool]],
	) -> None:
		self._set_keyframe((graph,), branch, turn, tick, keyframe)
		for orig, dests in keyframe.items():
			self._set_keyframe((graph, orig), branch, turn, tick, dests)
			for dest, ex in dests.items():
				self._set_keyframe(
					(graph, orig, dest), branch, turn, tick, {0: ex}
				)

	@staticmethod
	def _count_as_deleted(obj):
		return not obj

	def _update_keycache(self, *args, forward: bool):
		super()._update_keycache(*args, forward=forward)
		dest: Hashable
		branch: str
		turn: int
		tick: int
		graph, orig, dest, branch, turn, tick, value = args
		origs = self._get_origcache(
			graph, dest, branch, turn, tick, forward=forward
		)
		if value:
			origs = origs.union((orig,))
		else:
			origs = origs.difference((orig,))
		self.origcache[graph, dest, branch][turn][tick] = origs

	def _slow_iter_node_contradicted_times(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		graph: CharName,
		node: NodeName,
	):
		# slow and bad.
		retrieve = self._base_retrieve
		for items in (
			self.successors[graph,][node].items(),
			self.predecessors[graph,][node].items(),
		):
			for dest, branches in items:  # dest might really be orig
				brnch = branches[branch]
				if turn in brnch:
					ticks = brnch[turn]
					for tck, present in ticks.future(tick).items():
						if tck > tick and present is not retrieve(
							(graph, node, dest, branch, turn, tick)
						):
							yield turn, tck
				for trn, ticks in brnch.future(turn).items():
					for tck, present in ticks.items():
						if present is not retrieve(
							(graph, node, dest, branch, turn, tick)
						):
							yield trn, tck

	def _adds_dels_successors(
		self,
		parentity: tuple[Key, ...],
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		stoptime: Optional[Time] = None,
		cache: Optional[dict] = None,
	):
		graph: CharName
		orig: NodeName
		graph, orig = parentity
		cache = cache or self.successors
		if (graph, orig) in cache and cache[graph, orig]:
			added, deleted = self._get_adds_dels(
				(graph, orig), branch, turn, tick, stoptime=stoptime
			)
		else:
			added = set()
			deleted = set()
		kf = self.keyframe
		itparbtt = self.db._iter_parent_btt
		its = [(ks, v) for (ks, v) in kf.items() if len(ks) == 3]
		for (grap, org, dest), kfg in its:  # too much iteration!
			if (grap, org) != (graph, orig):
				continue
			for branc, trn, tck in itparbtt(
				branch, turn, tick, stoptime=stoptime
			):
				if branc not in kfg:
					continue
				kfgb = kfg[branc]
				if trn in kfgb:
					kfgbr = kfgb[trn]
					if kfgbr.rev_gettable(tck):
						if kfgbr[tck] and dest not in deleted:
							added.add(dest)
						continue
				if kfgb.rev_gettable(trn):
					if kfgb[trn].final() and dest not in deleted:
						added.add(dest)
		return added, deleted

	def _adds_dels_predecessors(
		self,
		parentity: tuple[Key, ...],
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		stoptime: tuple[str, int, int] = None,
		cache: dict = None,
	):
		graph, dest = parentity
		added = set()
		deleted = set()
		cache = cache or self.predecessors
		if cache[graph, dest]:
			for orig in cache[graph, dest]:
				addidx, delidx = self._get_adds_dels(
					(graph, orig, dest), branch, turn, tick, stoptime=stoptime
				)
				if addidx and not delidx:
					added.add(orig)
				elif delidx and not addidx:
					deleted.add(orig)
		else:
			kf = self.keyframe
			itparbtt = self.db._iter_parent_btt
			for k, kfg in kf.items():  # too much iteration!
				if len(k) != 3:
					continue
				(grap, orig, dst) = k
				if (grap, dst) != (graph, dest):
					continue
				for branc, trn, tck in itparbtt(
					branch, turn, tick, stoptime=stoptime
				):
					if branc not in kfg:
						continue
					kfgb = kfg[branc]
					if trn in kfgb:
						kfgbr = kfgb[trn]
						if kfgbr.rev_gettable(tck):
							if kfgbr[tck]:
								added.add(orig)
							continue
					if kfgb.rev_gettable(trn):
						if kfgb[trn].final():
							added.add(orig)
		return added, deleted

	def _get_origcache(
		self,
		graph: CharName,
		dest: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool,
	):
		"""Return a set of origin nodes leading to ``dest``"""
		(
			origcache,
			origcache_lru,
			get_keycachelike,
			predecessors,
			adds_dels_sucpred,
		) = self._get_origcache_stuff
		return get_keycachelike(
			origcache,
			predecessors,
			adds_dels_sucpred,
			(graph, dest),
			branch,
			turn,
			tick,
			forward=forward,
		)

	def iter_successors(
		self,
		graph: CharName,
		orig: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: Optional[bool] = None,
	):
		"""Iterate over successors of a given origin node at a given time."""
		if self.db._no_kc:
			yield from self._adds_dels_successors(
				(graph, orig), branch, turn, tick
			)[0]
			return
		if forward is None:
			forward = self.db._forward
		yield from self._get_keycache(
			(graph, orig), branch, turn, tick, forward=forward
		)

	def iter_predecessors(
		self,
		graph: CharName,
		dest: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: Optional[bool] = None,
	):
		"""Iterate over predecessors to a destination node at a given time."""
		if self.db._no_kc:
			yield from self._adds_dels_predecessors(
				(graph, dest), branch, turn, tick
			)[0]
			return
		if forward is None:
			forward = self.db._forward
		yield from self._get_origcache(
			graph, dest, branch, turn, tick, forward=forward
		)

	def has_successor(
		self,
		graph: CharName,
		orig: NodeName,
		dest: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: Optional[bool] = None,
	):
		"""Return whether an edge connects the origin to the destination now"""
		# Use a keycache if we have it.
		# If we don't, only generate one if we're forwarding, and only
		# if it's no more than a turn ago.
		keycache_key = (graph, orig, dest, branch)
		if keycache_key in self.keycache:
			return dest in self._get_keycache(
				(graph, orig), branch, turn, tick, forward=forward
			)
		got = self._base_retrieve((graph, orig, dest, branch, turn, tick))
		return got is True

	def has_predecessor(
		self,
		graph: CharName,
		dest: NodeName,
		orig: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: Optional[bool] = None,
	):
		"""Return whether an edge connects the destination to the origin now"""
		got = self._base_retrieve((graph, orig, dest, branch, turn, tick))
		return got is True

	def store(
		self,
		graph: CharName,
		orig: NodeName,
		dest: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		ex: bool,
		*,
		planning: Optional[bool] = None,
		forward: Optional[bool] = None,
		loading: bool = False,
		contra: Optional[bool] = None,
	):
		if contra is None:
			contra = not loading
		db, predecessors, successors = self._additional_store_stuff
		if planning is None:
			planning = db._planning
		self._store(
			graph,
			orig,
			dest,
			branch,
			turn,
			tick,
			ex,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)
		for dest, branches in successors[graph,][orig].items():
			try:
				pred_ticks = branches[branch][turn]
			except KeyError:
				continue
			if not pred_ticks.rev_gettable(tick):
				continue
			predecessors[graph,][dest][orig][branch].store_at(
				turn, tick, pred_ticks[tick]
			)

	def retrieve(
		self,
		graph: CharName,
		orig: NodeName,
		dest: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> bool:
		return bool(
			self._retrieve(
				graph, orig, dest, branch, turn, tick, search=search
			)
		)


class EdgeValCache(Cache):
	def store(
		self,
		graph: CharName,
		orig: NodeName,
		dest: NodeName,
		key: Stat,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
		*,
		planning: bool | None = None,
		forward: bool | None = None,
		loading: bool = False,
		contra: bool | None = None,
	):
		self._store(
			graph,
			orig,
			dest,
			key,
			branch,
			turn,
			tick,
			value,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)

	def retrieve(
		self,
		graph: CharName,
		orig: NodeName,
		dest: NodeName,
		key: Stat,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> Value:
		return self._retrieve(
			graph, orig, dest, key, branch, turn, tick, search=search
		)

	def iter_keys(
		self,
		graph: CharName,
		orig: NodeName,
		dest: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> Iterator[Stat]:
		return self._iter_entities_or_keys(
			graph, orig, dest, branch, turn, tick, forward=forward
		)

	def count_keys(
		self,
		graph: CharName,
		orig: NodeName,
		dest: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> int:
		return self._count_entities_or_keys(
			graph, orig, dest, branch, turn, tick, forward=forward
		)

	def contains_key(
		self,
		graph: CharName,
		orig: NodeName,
		dest: NodeName,
		key: Stat,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> bool:
		return self._contains_entity_or_key(
			graph, orig, dest, key, branch, turn, tick, search=search
		)

	def get_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		copy: bool = True,
	) -> dict[NodeName, dict[NodeName, dict[Stat, Value]]]:
		ret = self._get_keyframe((graph,), branch, turn, tick)
		if copy:
			ret = ret.copy()
			for orig, dests in ret.items():
				redests = ret[orig] = {}
				for dest, val in dests.items():
					redests[dest] = val.copy()
		return ret

	def set_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict[NodeName, dict[NodeName, dict[Stat, Value]]],
	):
		self._set_keyframe((graph,), branch, turn, tick, keyframe)
		for orig, dests in keyframe.items():
			for dest, val in dests.items():
				self._set_keyframe(
					(graph, orig, dest), branch, turn, tick, val
				)


class EntitylessCache(Cache):
	__slots__ = ()

	def store(
		self,
		key: Key,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
		*,
		planning: Optional[bool] = None,
		forward: Optional[bool] = None,
		loading: bool = False,
		contra: Optional[bool] = None,
	):
		self._store(
			None,
			key,
			branch,
			turn,
			tick,
			value,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)

	def get_keyframe(
		self, branch: Branch, turn: Turn, tick: Tick, copy: bool = True
	) -> dict:
		ret = self._get_keyframe((Key(None),), branch, turn, tick)
		if copy:
			ret = ret.copy()
		return ret

	def set_keyframe(
		self, branch: Branch, turn: Turn, tick: Tick, keyframe: dict
	) -> None:
		self._set_keyframe((Key(None),), branch, turn, tick, keyframe)

	def iter_keys(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: Optional[bool] = None,
	) -> Iterator[Key]:
		return self._iter_entities_or_keys(
			None, branch, turn, tick, forward=forward
		)

	def contains_key(
		self, ke: Key, branch: Branch, turn: Turn, tick: Tick
	) -> bool:
		return self._contains_entity_or_key(None, ke, branch, turn, tick)

	def retrieve(
		self,
		key: Key,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> Value:
		return self._retrieve(None, key, branch, turn, tick, search=search)


class GraphCache(Cache):
	overwrite_journal = True
	initial_value = None

	def store(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		type: Literal["DiGraph", None],
		*,
		planning: bool | None = None,
		forward: bool | None = None,
		loading: bool = False,
		contra: bool | None = None,
	) -> None:
		self._store(
			None,
			graph,
			branch,
			turn,
			tick,
			type,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)

	def retrieve(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> Literal["DiGraph", None]:
		return self._retrieve(None, graph, branch, turn, tick, search=search)

	def iter_keys(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		forward: bool | None = None,
	) -> Iterator[CharName]:
		return self._iter_entities_or_keys(
			None, branch, turn, tick, forward=forward
		)

	iter_entities = iter_keys

	def count_keys(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		forward: bool | None = None,
	) -> int:
		kc = self._get_keycache(
			(Key(None),), branch, turn, tick, forward=forward
		)
		return len(kc)

	def get_keyframe(
		self, branch: Branch, turn: Turn, tick: Tick, *, copy: bool = True
	):
		ret = self._get_keyframe((Key(None),), branch, turn, tick)
		if copy:
			ret = ret.copy()
		return ret

	def set_keyframe(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict[CharName, Literal["DiGraph", None]],
	) -> None:
		self._set_keyframe((Key(None),), branch, turn, tick, keyframe)


class InitializedCache(Cache):
	__slots__ = ()

	def _retrieve_for_journal(self, args):
		return self._retrieve(*args[:-1])

	def _store_journal(self, *args):
		entity, key, branch, turn, tick, value = args[-6:]
		parent = args[:-6]
		settings_turns = self.settings[branch]
		presettings_turns = self.presettings[branch]
		try:
			prev = self._retrieve_for_journal(args)
		except KeyError:
			prev = self.initial_value
		if prev == value:
			return  # not much point reporting on a non-change in a diff
		if turn in settings_turns.future(turn, include_same_rev=True):
			assert (
				turn in presettings_turns or turn in presettings_turns.future()
			)
			setticks = settings_turns[turn]
			presetticks = presettings_turns[turn]
			presetticks[tick] = parent + (entity, key, prev)
			setticks[tick] = parent + (entity, key, value)
		else:
			presettings_turns[turn] = {tick: parent + (entity, key, prev)}
			settings_turns[turn] = {tick: parent + (entity, key, value)}


class RulebooksCache(Cache):
	def store(
		self,
		rulebook: RulebookName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rules_prio: tuple[list[RuleName], RulebookPriority],
		*,
		planning: bool | None = None,
		forward: bool | None = None,
		loading: bool = False,
		contra: bool | None = None,
	) -> None:
		self._store(
			None,
			rulebook,
			branch,
			turn,
			tick,
			rules_prio,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)

	def retrieve(
		self,
		rulebook: RulebookName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> tuple[list[RuleName], RulebookPriority]:
		rules, prio = self._retrieve(
			None, rulebook, branch, turn, tick, search=search
		)
		return list(rules), prio

	def get_keyframe(
		self, branch: Branch, turn: Turn, tick: Tick, copy: bool = True
	) -> dict[RulebookName, tuple[list[RuleName], RulebookPriority]]:
		ret = self._get_keyframe((Key(None),), branch, turn, tick)
		if copy:
			ret = ret.copy()
			for rulebook, (rules, prio) in ret.items():
				ret[rulebook] = (list(rules), prio)
		return ret

	def set_keyframe(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict[RulebookName, tuple[list[RuleName], RulebookPriority]],
	) -> None:
		self._set_keyframe((Key(None),), branch, turn, tick, keyframe)

	def iter_keys(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> Iterator[RulebookName]:
		return self._iter_entities_or_keys(
			None, branch, turn, tick, forward=forward
		)


class FuncListCache(InitializedCache):
	def store(
		self,
		rulebook: RulebookName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		funcs: list[RuleFuncName],
		*,
		planning: bool | None = None,
		forward: bool | None = None,
		loading: bool = False,
		contra: bool | None = None,
	) -> None:
		self._store(
			None,
			rulebook,
			branch,
			turn,
			tick,
			funcs,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)

	def retrieve(
		self,
		rulebook: RulebookName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> list[RuleFuncName]:
		return self._retrieve(
			None, rulebook, branch, turn, tick, search=search
		)

	def contains_rulebook(
		self,
		rulebook: RulebookName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool | None = None,
	) -> bool:
		return self._contains_entity_or_key(
			None, rulebook, branch, turn, tick, search=search
		)

	contains_key = contains_rulebook

	def get_keyframe(
		self, branch: Branch, turn: Turn, tick: Tick, *, copy: bool = True
	) -> dict[RulebookName, list[RuleFuncName]]:
		ret = self._get_keyframe((Key(None),), branch, turn, tick)
		if copy:
			ret = ret.copy()
			for rb, funcs in ret.items():
				ret[rb] = funcs.copy()
		return ret

	def set_keyframe(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict[RulebookName, list[RuleFuncName]],
	):
		self._set_keyframe((Key(None),), branch, turn, tick, keyframe)


class NeighborhoodsCache(InitializedCache):
	initial_value = None

	def store(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		neighborhood: RuleNeighborhood,
		*,
		planning: bool | None = None,
		forward: bool | None = None,
		loading: bool = False,
		contra: bool | None = None,
	):
		self._store(
			Key(None),
			rule,
			branch,
			turn,
			tick,
			neighborhood,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)

	def retrieve(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> RuleNeighborhood:
		return self._retrieve(None, rule, branch, turn, tick, search=search)

	def get_keyframe(
		self, branch: Branch, turn: Turn, tick: Tick, *, copy: bool = True
	) -> dict[RuleName, RuleNeighborhood]:
		ret = self._get_keyframe((Key(None),), branch, turn, tick)
		if copy:
			ret = ret.copy()
		return ret

	def set_keyframe(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict[RuleName, RuleNeighborhood],
	) -> None:
		self._set_keyframe((Key(None),), branch, turn, tick, keyframe)


class BignessCache(InitializedCache):
	initial_value = False

	def store(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		big: RuleBig,
		*,
		planning: bool | None = None,
		forward: bool | None = None,
		loading: bool = False,
		contra: bool | None = None,
	) -> None:
		self._store(
			None,
			rule,
			branch,
			turn,
			tick,
			big,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)

	def retrieve(
		self,
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> RuleBig:
		return self._retrieve(None, rule, branch, turn, tick, search=search)

	def get_keyframe(
		self, branch: Branch, turn: Turn, tick: Tick, *, copy: bool = True
	):
		ret = self._get_keyframe((Key(None),), branch, turn, tick)
		if copy:
			ret = ret.copy()
		return ret

	def set_keyframe(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict[RuleName, RuleBig],
	) -> None:
		self._set_keyframe((Key(None),), branch, turn, tick, keyframe)


class NodesRulebooksCache(InitializedCache):
	def get_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		copy: bool = True,
	) -> dict[NodeName, RulebookName]:
		ret = self._get_keyframe((graph,), branch, turn, tick)
		if copy:
			ret = ret.copy()
		return ret

	def set_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict[NodeName, RulebookName],
	) -> None:
		self._set_keyframe((graph,), branch, turn, tick, keyframe)

	def store(
		self,
		graph: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rulebook: RulebookName,
		*,
		planning: bool | None = None,
		forward: bool | None = None,
		loading: bool = False,
		contra: bool | None = None,
	) -> None:
		self._store(
			graph,
			node,
			branch,
			turn,
			tick,
			rulebook,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)

	def retrieve(
		self,
		graph: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> RulebookName:
		return self._retrieve(graph, node, branch, turn, tick, search=search)


class CharactersRulebooksCache(InitializedCache):
	def _retrieve_for_journal(self, args):
		return self.retrieve(*args[1:-1])

	def get_keyframe(
		self, branch: Branch, turn: Turn, tick: Tick, *, copy: bool = True
	):
		ret = self._get_keyframe((Key(None),), branch, turn, tick)
		if copy:
			ret = ret.copy()
		return ret

	def set_keyframe(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict[CharName, RulebookName],
	):
		self._set_keyframe((Key(None),), branch, turn, tick, keyframe)

	def store(
		self,
		character: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rulebook: RulebookName,
		*,
		planning: bool | None = None,
		forward: bool | None = None,
		loading: bool = False,
		contra: bool = None,
	) -> None:
		self._store(
			None,
			character,
			branch,
			turn,
			tick,
			rulebook,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)

	def retrieve(
		self,
		character: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> RulebookName:
		return self._retrieve(
			None, character, branch, turn, tick, search=search
		)


class PortalsRulebooksCache(InitializedCache):
	def store(
		self,
		char: CharName,
		orig: NodeName,
		dest: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		rb: RulebookName,
		planning: Optional[bool] = None,
		forward: Optional[bool] = None,
		loading: bool = False,
		contra: Optional[bool] = None,
	) -> None:
		try:
			destrbs = self.retrieve_successors(char, orig, branch, turn, tick)
			destrbs[dest] = rb
		except KeyError:
			destrbs = {dest: rb}
		self._store(
			char,
			orig,
			branch,
			turn,
			tick,
			destrbs,
			loading=loading,
			contra=contra,
			forward=forward,
			planning=planning,
		)
		# The former will be overwritten in the journal (but not elsewhere)
		# by the latter:
		self._store(
			char,
			orig,
			dest,
			branch,
			turn,
			tick,
			rb,
			loading=loading,
			contra=contra,
			forward=forward,
			planning=planning,
		)

	def retrieve(
		self,
		char: CharName,
		orig: NodeName,
		dest: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> RulebookName:
		return self._retrieve(
			char, orig, dest, branch, turn, tick, search=search
		)

	def retrieve_successors(
		self,
		char: CharName,
		orig: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> dict[NodeName, RulebookName]:
		return self._retrieve(char, orig, branch, turn, tick, search=search)

	def get_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		copy: bool = True,
	) -> dict[NodeName, dict[NodeName, RulebookName]]:
		ret = self._get_keyframe((graph,), branch, turn, tick)
		if copy:
			ret = ret.copy()
			for orig, dests in ret.items():
				ret[orig] = dests.copy()
		return ret

	def get_orig_keyframe(
		self,
		graph: CharName,
		orig: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		copy: bool = True,
	) -> dict[NodeName, RulebookName]:
		ret = self._get_keyframe((graph, orig), branch, turn, tick)
		if copy:
			ret = ret.copy()
		return ret

	def iter_keys(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> Iterator[NodeName]:
		return self._iter_entities_or_keys(
			graph, branch, turn, tick, forward=forward
		)

	def set_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict[NodeName, dict[NodeName, RulebookName]],
	):
		self._set_keyframe((graph,), branch, turn, tick, keyframe)
		for orig, dests in keyframe.items():
			for dest, rulebook in dests.items():
				try:
					subkf = self._get_keyframe(
						(graph, orig),
						branch,
						turn,
						tick,
					).copy()
					subkf[dest] = rulebook
				except KeyError:
					subkf = {dest: rulebook}
				self._set_keyframe((graph, orig), branch, turn, tick, subkf)


class LeaderSetCache(Cache):
	"""A cache for remembering what set of characters have certain nodes as units"""

	def store(
		self,
		character: CharName,
		graph: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		is_unit: bool,
		*,
		planning: bool | None = None,
		forward: bool | None = None,
		loading: bool = False,
		contra: bool | None = None,
	):
		if forward is None:
			forward = self.db._forward
		if is_unit:
			users = frozenset([character])
			try:
				users |= self.retrieve(
					graph, node, branch, turn, tick, search=not forward
				)
			except KeyError:
				pass
		else:
			try:
				users = self.retrieve(
					graph, node, branch, turn, tick, search=not forward
				)
			except KeyError:
				users = frozenset([])
			users -= {character}
		self._store(
			graph,
			node,
			branch,
			turn,
			tick,
			users,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)

	def retrieve(
		self,
		graph: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		search: bool = False,
	):
		return self._retrieve(graph, node, branch, turn, tick, search=search)

	def get_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		copy: bool = True,
	) -> dict[NodeName, frozenset[CharName]]:
		ret = self._get_keyframe((graph,), branch, turn, tick)
		if copy:
			ret = ret.copy()
		return ret

	def set_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict[NodeName, frozenset[CharName]],
	):
		self._set_keyframe((graph,), branch, turn, tick, keyframe)

	def alias_keyframe(
		self,
		branch_from: Branch,
		branch_to: Branch,
		turn: Turn,
		tick: Tick,
		default: Optional[frozenset] = None,
	):
		super().alias_keyframe(branch_from, branch_to, turn, tick, default)


class UnitDictCache(Cache):
	def store(
		self,
		character: CharName,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		d: dict[NodeName, bool],
		*,
		planning: bool | None = None,
		forward: bool | None = None,
		loading: bool = False,
		contra: bool | None = None,
	):
		for node, is_unit in d.items():
			d[node] = bool(is_unit)
		self._store(
			character,
			graph,
			branch,
			turn,
			tick,
			d,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)

	def iter_entities(
		self,
		character: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> Iterator[CharName]:
		return self._iter_entities_or_keys(
			character, branch, turn, tick, forward=forward
		)

	def count_entities(
		self,
		character: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> int:
		return self._count_entities_or_keys(
			character, branch, turn, tick, forward=forward
		)

	def retrieve(
		self,
		character: CharName,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		search: bool = False,
	) -> dict[NodeName, bool]:
		return self._retrieve(
			character, graph, branch, turn, tick, search=search
		)

	def contains_graph(
		self,
		character: CharName,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> bool:
		return self._contains_entity_or_key(
			character, graph, branch, turn, tick, search=search
		)

	contains_key = contains_graph

	def iter_graphs(
		self,
		character: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> Iterator[CharName]:
		return self._iter_entities_or_keys(
			character, branch, turn, tick, forward=forward
		)

	iter_keys = iter_graphs

	def get_keyframe(
		self,
		character: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		copy: bool = True,
	) -> dict[CharName, dict[NodeName, bool]]:
		ret = self._get_keyframe((character,), branch, turn, tick)
		if copy:
			ret = ret.copy()
			for k, v in ret.items():
				ret[k] = v.copy()
		return ret

	def set_keyframe(
		self,
		character: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict[CharName, dict[NodeName, bool]],
	):
		self._set_keyframe((character,), branch, turn, tick, keyframe)


class UnitnessCache(Cache):
	"""A cache for remembering when a node is a unit of a character."""

	initial_value = False

	@staticmethod
	def _count_as_deleted(obj):
		return not obj

	def __init__(
		self,
		db: "engine.Engine",
		name: str,
		keyframe_dict: Optional[dict] = None,
	):
		super().__init__(db, name, keyframe_dict)
		self.leader_cache = LeaderSetCache(db, "user cache")
		self.dict_cache = UnitDictCache(db, "unit dict cache")

	@contextmanager
	def overwriting(self):
		if hasattr(self, "overwrite_journal"):
			yield
			return
		self.overwrite_journal = True
		with self.leader_cache.overwriting(), self.dict_cache.overwriting():
			yield
		del self.overwrite_journal

	def store(
		self,
		character: CharName,
		graph: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		is_unit: bool,
		*,
		planning: Optional[bool] = None,
		forward: Optional[bool] = None,
		loading: bool = False,
		contra: Optional[bool] = None,
	):
		if forward is None:
			forward = self.db._forward
		self._store(
			character,
			graph,
			node,
			branch,
			turn,
			tick,
			is_unit,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)
		try:
			d = self.dict_cache.retrieve(
				character, graph, branch, turn, tick, search=not forward
			).copy()
		except KeyError:
			d = {}
		if is_unit:
			d[node] = True
		elif node in d:
			del d[node]
		self.dict_cache.store(
			character,
			graph,
			branch,
			turn,
			tick,
			d,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)
		self.leader_cache.store(
			character,
			graph,
			node,
			branch,
			turn,
			tick,
			is_unit,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)

	def set_keyframe(
		self,
		character: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict[CharName, dict[NodeName, bool]],
	):
		self.dict_cache.set_keyframe(character, branch, turn, tick, keyframe)
		for graph, kf in keyframe.items():
			self._set_keyframe((character, graph), branch, turn, tick, kf)

	def get_keyframe(
		self,
		character: CharName,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		copy: bool = True,
	) -> dict[NodeName, bool]:
		ret = self._get_keyframe((character, graph), branch, turn, tick)
		if copy:
			ret = ret.copy()
		return ret

	def contains_graph(
		self,
		char: CharName,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> bool:
		try:
			return bool(
				self._retrieve(char, graph, branch, turn, tick, search=search)
			)
		except KeyError:
			return False

	def contains_unit(
		self,
		char: CharName,
		graph: CharName,
		unit: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> bool:
		try:
			return bool(
				self._retrieve(
					char, graph, unit, branch, turn, tick, search=search
				)
			)
		except KeyError:
			return False

	retrieve = contains_unit

	def get_char_graph_units(
		self,
		char: CharName,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> set[NodeName]:
		return set(
			self._iter_entities_or_keys(
				char, graph, branch, turn, tick, forward=forward
			)
		)

	def count_graphs(
		self,
		char: CharName,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> int:
		return self._count_entities_or_keys(
			char, graph, branch, turn, tick, forward=forward
		)

	def iter_entities(
		self,
		char: CharName,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> Iterator[NodeName]:
		return self._iter_entities_or_keys(
			char, graph, branch, turn, tick, forward=forward
		)

	def count_entities(
		self,
		char: CharName,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> int:
		return self._count_entities_or_keys(
			char, graph, branch, turn, tick, forward=forward
		)

	def get_char_only_unit(
		self, char: CharName, branch: Branch, turn: Turn, tick: Tick
	):
		if self.dict_cache.count_entities(char, branch, turn, tick) != 1:
			raise ValueError("No unit, or more than one unit")
		for graph in self.dict_cache.iter_entities(char, branch, turn, tick):
			if self.count_entities(char, graph, branch, turn, tick) != 1:
				raise ValueError("No unit, or more than one unit")
			return graph, next(
				self.iter_entities(char, graph, branch, turn, tick)
			)

	def get_char_only_graph(
		self, char: CharName, branch: Branch, turn: Turn, tick: Tick
	):
		if self.count_entities(char, branch, turn, tick) != 1:
			raise ValueError("No unit, or more than one unit")
		return next(self.iter_entities(char, branch, turn, tick))

	def iter_char_graphs(
		self, char: CharName, branch: Branch, turn: Turn, tick: Tick
	):
		return self.dict_cache.iter_entities(char, branch, turn, tick)


def oner(f: Callable[..., Iterator]) -> Callable[..., Iterator]:
	@wraps(f)
	def oned(*args, **kwargs):
		already = set()
		for one in f(*args, **kwargs):
			if one in already:
				raise RuntimeError("Already yielded", one)
			already.add(one)
			yield one

	return oned


class RulesHandledCache(ABC):
	handled: dict[
		tuple[CharName, RulebookName, Branch, Turn]
		| tuple[CharName, NodeName, RulebookName, Branch, Turn]
		| tuple[CharName, NodeName, NodeName, RulebookName, Branch, Turn]
		| tuple[CharName, CharName, NodeName, RulebookName, Branch, Turn],
		set[RuleName],
	]
	handled_deep: dict[
		Branch,
		SettingsTurnDict[
			Turn,
			dict[
				Tick,
				tuple[
					EntityKey,
					RulebookName,
					RuleName,
				],
			],
		],
	]

	def __init__(self, engine: "engine.Engine", name: str):
		self.lock = RLock()
		self.engine = engine
		self.name = name
		self.handled = {}
		self.handled_deep = PickyDefaultDict(SettingsTurnDict)

	def __init_subclass__(cls, **kwargs):
		iter_unhandled_rules = getattr(cls, "iter_unhandled_rules")
		setattr(cls, "iter_unhandled_rules", oner(iter_unhandled_rules))

	@abstractmethod
	def get_rulebook(self, *args): ...

	@abstractmethod
	def iter_unhandled_rules(self, branch: Branch, turn: Turn, tick: Tick): ...

	@abstractmethod
	def was_handled(
		self,
		branch: Branch,
		turn: Turn,
		rulebook: RulebookName,
		rule: RuleName,
		entity: tuple[CharName]
		| tuple[CharName, NodeName]
		| tuple[CharName, NodeName, NodeName]
		| tuple[CharName, CharName, NodeName],
	) -> bool:
		key = (*entity, rulebook, branch, turn)
		if key not in self.handled:
			return False
		return rule in self.handled[key]

	def store(self, *args, loading: bool = False):
		entity: EntityKey = args[:-5]
		rulebook: RulebookName
		rule: RuleName
		branch: Branch
		turn: Turn
		tick: Tick
		rulebook, rule, branch, turn, tick = args[-5:]
		if turn in self.handled_deep[branch]:
			if tick in self.handled_deep[branch][turn]:
				self.handled_deep[branch][turn][tick].add(
					(entity, rulebook, rule)
				)
			else:
				self.handled_deep[branch].store_at(
					turn, tick, {(entity, rulebook, rule)}
				)
		else:
			self.handled_deep[branch] = {
				turn: {tick: {(entity, rulebook, rule)}}
			}
		key = (*entity, rulebook, branch, turn)
		if key in self.handled:
			rules_handled_this_turn = self.handled[key]
			if rule in rules_handled_this_turn:
				raise RuntimeError(
					"Rule already run for same entity and rulebook",
					entity,
					rulebook,
					rule,
					branch,
					turn,
					tick,
				)
			rules_handled_this_turn.add(rule)
		else:
			self.handled[key] = {rule}

	def remove_branch(self, branch: Branch):
		if branch in self.handled_deep:
			for turn, ticks in self.handled_deep[branch].items():
				for tick, rbset in ticks.items():
					for entity, rulebook, rule in rbset:
						if (entity, rulebook, branch, turn) in self.handled:
							del self.handled[entity, rulebook, branch, turn]
			del self.handled_deep[branch]

	def total_size(
		self, handlers: Optional[dict] = None, verbose: bool = False
	):
		"""Returns the approximate memory footprint an object and all of its contents.

		Automatically finds the contents of the following builtin containers and
		their subclasses:  tuple, list, deque, dict, set and frozenset.
		To search other containers, add handlers to iterate over their contents:

		    handlers = {SomeContainerClass: iter,
		                OtherContainerClass: OtherContainerClass.get_elements}

		From https://code.activestate.com/recipes/577504-compute-memory-footprint-of-an-object-and-its-cont/download/1/

		"""
		all_handlers = {
			tuple: iter,
			list: iter,
			RulesHandledCache: lambda d: [d.mark_handled, d.handled_deep],
			dict: lambda d: chain.from_iterable(d.items()),
			set: iter,
			frozenset: iter,
			Cache: lambda o: [
				o.branches,
				o.settings,
				o.presettings,
				o.keycache,
			],
		}
		if handlers:
			all_handlers.update(handlers)
		seen = set()  # track which object id's have already been seen
		default_size = getsizeof(
			0
		)  # estimate sizeof object without __sizeof__

		def sizeof(o):
			if id(o) in seen:  # do not double count the same object
				return 0
			seen.add(id(o))
			s = getsizeof(o, default_size)

			if verbose:
				print(s, type(o), repr(o), file=stderr)

			for typ, handler in all_handlers.items():
				if isinstance(o, typ):
					s += sum(map(sizeof, handler(o)))
					break
			return s

		return sizeof(self)

	def truncate(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		direction: Direction = Direction.FORWARD,
	):
		if isinstance(direction, str):
			direction = Direction(direction)
		if branch not in self.handled_deep:
			return

		with self.lock:
			turn_d = self.handled_deep[branch]
			if turn in turn_d and (
				view := turn_d[turn].future(tick)
				if direction == "forward"
				else turn_d[turn].past(tick)
			):
				for ruled in view.values():
					for entity, rulebook, rule in ruled:
						key = (*entity, rulebook, branch, turn)
						if key in self.handled:
							rule_set = self.handled[key]
							rule_set.discard(rule)
							if not rule_set:
								del self.handled[key]
				turn_d[turn].truncate(turn, direction)
			to_del = (
				turn_d.future(turn)
				if direction == "forward"
				else turn_d.past(turn)
			)
			for r in to_del.keys():
				for t, ruled in turn_d[r].items():
					for entity, rulebook, rule in ruled:
						if (entity, rulebook, branch, r) in self.handled:
							rule_set = self.handled[
								entity, rulebook, branch, r
							]
							rule_set.discard(t)
							if not rule_set:
								del self.handled[entity, rulebook, branch, r]
			turn_d.truncate(turn, Direction(direction))

	def retrieve(self, *args):
		return self.handled[args]

	def get_handled_rules(
		self,
		entity: tuple[CharName]
		| tuple[CharName, NodeName]
		| tuple[CharName, NodeName, NodeName]
		| tuple[CharName, CharName, NodeName],
		rulebook: RulebookName,
		branch: Branch,
		turn: Turn,
	):
		key = (*entity, rulebook, branch, turn)
		if key not in self.handled:
			if (
				branch in self.handled_deep
				and turn in self.handled_deep[branch]
			):
				raise RuntimeError(f"Incoherent {self.__class__.__name__}")
			self.handled[key] = set()
		return self.handled[key]


class CharacterRulesHandledCache(RulesHandledCache):
	handled: dict[tuple[CharName, RulebookName, Branch, Turn], set[RuleName]]

	def get_rulebook(
		self, character: CharName, branch: Branch, turn: Turn, tick: Tick
	):
		try:
			return self.engine._characters_rulebooks_cache.retrieve(
				character, branch, turn, tick
			)
		except KeyError:
			return ("character_rulebook", character)

	def iter_unhandled_rules(self, branch: Branch, turn: Turn, tick: Tick):
		for character in self.engine.character.keys():
			rb = self.get_rulebook(character, branch, turn, tick)
			try:
				rules, prio = self.engine._rulebooks_cache.retrieve(
					rb, branch, turn, tick
				)
			except KeyError:
				continue
			if not rules:
				continue
			handled = self.get_handled_rules((character,), rb, branch, turn)
			for rule in rules:
				if rule not in handled:
					yield prio, character, rb, rule

	def was_handled(
		self,
		branch: Branch,
		turn: Turn,
		rulebook: RulebookName,
		rule: RuleName,
		character: CharName,
	) -> bool:
		return super().was_handled(branch, turn, rulebook, rule, (character,))


class UnitRulesHandledCache(RulesHandledCache):
	handled: dict[
		tuple[CharName, CharName, NodeName, RulebookName, Branch, Turn],
		set[RuleName],
	]

	def get_rulebook(
		self, character: CharName, branch: Branch, turn: Turn, tick: Tick
	):
		try:
			return self.engine._units_rulebooks_cache.retrieve(
				character, branch, turn, tick
			)
		except KeyError:
			return "unit_rulebook", character

	def iter_unhandled_rules(self, branch: Branch, turn: Turn, tick: Tick):
		for charname in self.engine._graph_cache.iter_keys(branch, turn, tick):
			rb = self.get_rulebook(charname, branch, turn, tick)
			try:
				rules, prio = self.engine._rulebooks_cache.retrieve(
					rb, branch, turn, tick
				)
			except KeyError:
				continue
			if not rules:
				continue
			for graphname in self.engine._unitness_cache.dict_cache.iter_keys(
				charname, branch, turn, tick
			):
				# Seems bad that I have to check twice like this.
				try:
					existences = (
						self.engine._unitness_cache.dict_cache.retrieve(
							charname, graphname, branch, turn, tick
						)
					)
				except KeyError:
					continue
				for node, ex in existences.items():
					if not ex:
						continue
					handled = self.get_handled_rules(
						(charname, graphname, node), rb, branch, turn
					)
					for rule in rules:
						if rule not in handled:
							yield prio, charname, graphname, node, rb, rule

	def was_handled(
		self,
		branch: Branch,
		turn: Turn,
		rulebook: RulebookName,
		rule: RuleName,
		character_graph: CharName,
		unit_graph: CharName,
		unit: NodeName,
	) -> bool:
		return super().was_handled(
			branch, turn, rulebook, rule, (character_graph, unit_graph, unit)
		)


class CharacterThingRulesHandledCache(RulesHandledCache):
	handled: dict[
		tuple[CharName, NodeName, RulebookName, Branch, Turn], set[RuleName]
	]

	def get_rulebook(
		self, character: CharName, branch: Branch, turn: Turn, tick: Tick
	):
		try:
			return self.engine._characters_things_rulebooks_cache.retrieve(
				character, branch, turn, tick
			)
		except KeyError:
			return "character_thing_rulebook", character

	def iter_unhandled_rules(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> Iterator[
		tuple[RulebookPriority, CharName, NodeName, RulebookName, RuleName]
	]:
		charm = self.engine.character
		for character in sort_set(charm.keys()):
			rulebook = self.get_rulebook(character, branch, turn, tick)
			try:
				rules, prio = self.engine._rulebooks_cache.retrieve(
					rulebook, branch, turn, tick
				)
			except KeyError:
				continue
			if not rules:
				continue
			for thing in sort_set(charm[character].thing.keys()):
				handled = self.get_handled_rules(
					(character, thing), rulebook, branch, turn
				)
				for rule in rules:
					if rule not in handled:
						yield prio, character, thing, rulebook, rule

	def was_handled(
		self,
		branch: Branch,
		turn: Turn,
		rulebook: RulebookName,
		rule: RuleName,
		character: CharName,
		thing: NodeName,
	) -> bool:
		return super().was_handled(
			branch, turn, rulebook, rule, (character, thing)
		)


class CharacterPlaceRulesHandledCache(RulesHandledCache):
	handled: dict[
		tuple[CharName, NodeName, RulebookName, Branch, Turn], set[RuleName]
	]

	def get_rulebook(
		self, character: CharName, branch: Branch, turn: Turn, tick: Tick
	):
		try:
			return self.engine._characters_places_rulebooks_cache.retrieve(
				character, branch, turn, tick
			)
		except KeyError:
			return "character_place_rulebook", character

	def iter_unhandled_rules(self, branch: Branch, turn: Turn, tick: Tick):
		charm = self.engine.character
		for character in sort_set(charm.keys()):
			rulebook = self.get_rulebook(character, branch, turn, tick)
			try:
				rules, prio = self.engine._rulebooks_cache.retrieve(
					rulebook, branch, turn, tick
				)
			except KeyError:
				continue
			if not rules:
				continue
			for place in sort_set(charm[character].place.keys()):
				handled = self.get_handled_rules(
					(character, place), rulebook, branch, turn
				)
				for rule in rules:
					if rule not in handled:
						yield prio, character, place, rulebook, rule

	def was_handled(
		self,
		branch: Branch,
		turn: Turn,
		rulebook: RulebookName,
		rule: RuleName,
		character: CharName,
		place: NodeName,
	) -> bool:
		return super().was_handled(
			branch, turn, rulebook, rule, (character, place)
		)


class CharacterPortalRulesHandledCache(RulesHandledCache):
	handled: dict[
		tuple[CharName, NodeName, NodeName, RulebookName, Branch, Turn],
		set[RuleName],
	]

	def get_rulebook(
		self, character: CharName, branch: Branch, turn: Turn, tick: Tick
	):
		try:
			return self.engine._characters_portals_rulebooks_cache.retrieve(
				character, branch, turn, tick
			)
		except KeyError:
			return "character_portal_rulebook", character

	def iter_unhandled_rules(self, branch: Branch, turn: Turn, tick: Tick):
		charm = self.engine.character
		for character in sort_set(charm.keys()):
			rulebook = self.get_rulebook(character, branch, turn, tick)
			try:
				rules, prio = self.engine._rulebooks_cache.retrieve(
					rulebook, branch, turn, tick
				)
			except KeyError:
				continue
			if not rules:
				continue
			char = charm[character]
			charn = char.node
			charp = char.portal
			for orig in sort_set(charp.keys()):
				if orig not in charn:
					continue
				for dest in sort_set(charp[orig].keys()):
					if dest not in charn:
						continue
					handled = self.get_handled_rules(
						(character, orig, dest), rulebook, branch, turn
					)
					for rule in rules:
						if rule not in handled:
							yield prio, character, orig, dest, rulebook, rule

	def was_handled(
		self,
		branch: Branch,
		turn: Turn,
		rulebook: RulebookName,
		rule: RuleName,
		character: CharName,
		origin: NodeName,
		destination: NodeName,
	) -> bool:
		return super().was_handled(
			branch, turn, rulebook, rule, (character, origin, destination)
		)


class NodeRulesHandledCache(RulesHandledCache):
	handled: dict[
		tuple[CharName, NodeName, RulebookName, Branch, Turn], set[RuleName]
	]

	def get_rulebook(
		self,
		character: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	):
		try:
			return self.engine._nodes_rulebooks_cache.retrieve(
				character, node, branch, turn, tick
			)
		except KeyError:
			return character, node

	def iter_unhandled_rules(self, branch: Branch, turn: Turn, tick: Tick):
		charm = self.engine.character
		for character_name, character in sorted(
			charm.items(), key=itemgetter(0)
		):
			for node_name in character.node:
				rulebook = self.get_rulebook(
					character_name, node_name, branch, turn, tick
				)
				try:
					rules, prio = self.engine._rulebooks_cache.retrieve(
						rulebook, branch, turn, tick
					)
				except KeyError:
					continue
				handled = self.get_handled_rules(
					(character_name, node_name), rulebook, branch, turn
				)
				for rule in rules:
					if rule not in handled:
						yield prio, character_name, node_name, rulebook, rule

	def was_handled(
		self,
		branch: Branch,
		turn: Turn,
		rulebook: RulebookName,
		rule: RuleName,
		character: CharName,
		node: NodeName,
	) -> bool:
		return super().was_handled(
			branch, turn, rulebook, rule, (character, node)
		)


class PortalRulesHandledCache(RulesHandledCache):
	handled: dict[
		tuple[CharName, NodeName, NodeName, RulebookName, Branch, Turn],
		set[RuleName],
	]

	def get_rulebook(
		self,
		character: CharName,
		orig: NodeName,
		dest: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	):
		try:
			return self.engine._portals_rulebooks_cache.retrieve(
				character, orig, dest, branch, turn, tick
			)
		except KeyError:
			return character, orig, dest

	def iter_unhandled_rules(self, branch: Branch, turn: Turn, tick: Tick):
		for character_name, character in sorted(
			self.engine.character.items(), key=itemgetter(0)
		):
			for orig_name in sort_set(
				frozenset(
					self.engine._portals_rulebooks_cache.iter_keys(
						character_name, branch, turn, tick
					)
				)
			):
				try:
					destrbs = self.engine._portals_rulebooks_cache.retrieve_successors(
						character_name, orig_name, branch, turn, tick
					)
				except KeyError:
					# shouldn't happen, but apparently does??
					# Seems to be a case of too many keys showing up in the
					# iteration. Currently demonstrated by remake_college24.py
					# 2025-02-07
					continue
				for dest_name in sort_set(destrbs.keys()):
					rulebook = destrbs[dest_name]
					try:
						rules, prio = self.engine._rulebooks_cache.retrieve(
							rulebook, branch, turn, tick
						)
					except KeyError:
						continue
					handled = self.get_handled_rules(
						(character_name, orig_name, dest_name),
						rulebook,
						branch,
						turn,
					)
					for rule in rules:
						if rule not in handled:
							yield (
								prio,
								character_name,
								orig_name,
								dest_name,
								rulebook,
								rule,
							)

	def was_handled(
		self,
		branch: Branch,
		turn: Turn,
		rulebook: RulebookName,
		rule: RuleName,
		character: CharName,
		origin: NodeName,
		destination: NodeName,
	) -> bool:
		return super().was_handled(
			branch, turn, rulebook, rule, (character, origin, destination)
		)


class ThingsCache(Cache):
	def __init__(
		self,
		db: "engine.Engine",
		name: str,
		keyframe_dict: Optional[dict] = None,
	):
		super().__init__(db, name, keyframe_dict)
		self._make_node = db.thing_cls

	def _slow_iter_contents(
		self,
		character: CharName,
		place: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	):
		for thing in self._iter_entities_or_keys(
			character, branch, turn, tick
		):
			if self.retrieve(character, thing, branch, turn, tick) == place:
				yield thing

	def _retrieve_or_generate_contents(
		self,
		character: CharName,
		location: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
	):
		try:
			return self.db._node_contents_cache.retrieve(
				character,
				location,
				branch,
				turn,
				tick,
				search=True,
			)
		except KeyError:
			return frozenset(
				self._slow_iter_contents(
					character, location, branch, turn, tick
				)
			)

	def store(
		self,
		character: CharName,
		thing: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		location: NodeName,
		planning: Optional[bool] = None,
		forward: bool | None = None,
		loading: bool = False,
		contra: Optional[bool] = None,
	) -> None:
		with self._lock:
			oldloc: NodeName | ...
			try:
				oldloc = self.retrieve(character, thing, branch, turn, tick)
			except KeyError:
				oldloc = ...
			self._store(
				character,
				thing,
				branch,
				turn,
				tick,
				location,
				planning=planning,
				forward=forward,
				loading=loading,
				contra=contra,
			)
			if loading:
				return
			node_contents_cache = self.db._node_contents_cache
			this = frozenset((thing,))
			# Cache the contents of nodes
			todo = defaultdict(list)
			if oldloc is not ...:
				try:
					oldconts_orig = node_contents_cache.retrieve(
						character, oldloc, branch, turn, tick
					)
				except KeyError:
					oldconts_orig = frozenset()
				todo[turn, tick].append(
					(oldloc, oldconts_orig.difference(this))
				)
				# update any future contents caches pertaining to the old location
				if (character, oldloc) in node_contents_cache.loc_settings:
					locset = node_contents_cache.loc_settings[
						character, oldloc
					][branch]
					if turn in locset:
						for future_tick in locset[turn].future(tick):
							todo[turn, future_tick].append(
								(
									oldloc,
									self._retrieve_or_generate_contents(
										character,
										oldloc,
										branch,
										turn,
										future_tick,
									).difference(this),
								)
							)
					for future_turn, future_ticks in locset.future(
						turn
					).items():
						for future_tick in future_ticks:
							todo[future_turn, future_tick].append(
								(
									oldloc,
									self._retrieve_or_generate_contents(
										character,
										oldloc,
										branch,
										future_turn,
										future_tick,
									).difference(this),
								)
							)
			if location is not ...:
				todo[turn, tick].append(
					(
						location,
						self._retrieve_or_generate_contents(
							character, location, branch, turn, tick
						).union(this),
					)
				)
				# and the new location
				if (character, location) in node_contents_cache.loc_settings:
					locset = node_contents_cache.loc_settings[
						character, location
					][branch]
					if turn in locset:
						for future_tick in locset[turn].future(tick):
							todo[turn, future_tick].append(
								(
									location,
									self._retrieve_or_generate_contents(
										character,
										location,
										branch,
										turn,
										future_tick,
									).union(this),
								)
							)
					for future_turn, future_ticks in locset.future(
						turn
					).items():
						for future_tick in future_ticks:
							todo[future_turn, future_tick].append(
								(
									location,
									self._retrieve_or_generate_contents(
										character,
										location,
										branch,
										future_turn,
										future_tick,
									).union(this),
								)
							)
		for trn, tck in sorted(todo.keys()):
			for loc, conts in todo[trn, tck]:
				node_contents_cache.store(
					character, loc, branch, trn, tck, conts
				)

	def retrieve(
		self,
		character: CharName,
		thing: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> NodeName:
		return self._retrieve(
			character, thing, branch, turn, tick, search=search
		)

	def turn_before(
		self, character: CharName, thing: NodeName, branch: Branch, turn: Turn
	) -> Turn:
		with self._lock:
			try:
				self.retrieve(character, thing, branch, turn, 0)
			except KeyError:
				pass
			return self.keys[(character,)][thing][branch].rev_before(turn)

	def turn_after(
		self, character: CharName, thing: NodeName, branch: Branch, turn: Turn
	) -> Turn:
		with self._lock:
			try:
				self.retrieve(character, thing, branch, turn, 0)
			except KeyError:
				pass
			return self.keys[(character,)][thing][branch].rev_after(turn)

	def get_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		copy: bool = True,
	) -> dict[NodeName, NodeName]:
		ret = self._get_keyframe((graph,), branch, turn, tick)
		if copy:
			ret = ret.copy()
		return ret

	def set_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict[NodeName, NodeName],
	) -> None:
		self._set_keyframe((graph,), branch, turn, tick, keyframe)

	def thing_exists(
		self,
		graph: CharName,
		thing: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		search: bool = False,
	) -> bool:
		return self._contains_entity_or_key(
			graph, thing, branch, turn, tick, search=search
		)

	contains_entity = thing_exists

	def count_things(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> int:
		return self._count_entities_or_keys(
			graph, branch, turn, tick, forward=forward
		)

	count_keys = count_things

	def iter_things(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		*,
		forward: bool | None = None,
	) -> Iterator[NodeName]:
		return self._iter_entities_or_keys(
			graph, branch, turn, tick, forward=forward
		)


class NodeContentsCache(Cache):
	overwrite_journal = True

	def __init__(
		self,
		db: "engine.Engine",
		name: str,
		keyframe_dict: Optional[dict] = None,
	):
		super().__init__(db, name, keyframe_dict)
		self.loc_settings = StructuredDefaultDict(1, SettingsTurnDict)

	def delete_plan(self, plan: Plan) -> None:
		plan_ticks = self.db._plan_ticks[plan]
		with self.db.world_lock:
			for branch, trns in plan_ticks.items():
				times = trns.iter_times()
				for start_turn, start_tick in times:
					if self.db._branch_end(branch) < (start_turn, start_tick):
						break
				else:
					continue
				for trn, tck in times:
					self.remove(branch, trn, tck)

	def store(
		self,
		character: CharName,
		place: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		contents: frozenset[NodeName],
		planning: bool = True,
		forward: Optional[bool] = None,
		loading: bool = False,
		contra: Optional[bool] = None,
	):
		self.loc_settings[character, place][branch].store_at(
			turn, tick, contents
		)

		return self._store(
			character,
			place,
			branch,
			turn,
			tick,
			contents,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)

	def retrieve(
		self,
		character: CharName,
		node: NodeName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		search: bool = False,
	) -> frozenset[NodeName]:
		return self._retrieve(
			character, node, branch, turn, tick, search=search
		)

	def _iter_future_contradictions(
		self,
		entity: EntityKey,
		key: Key,
		turns: WindowDict,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		value: Value,
	):
		return self.db._things_cache._iter_future_contradictions(
			entity, key, turns, branch, turn, tick, value
		)

	def remove(self, branch: Branch, turn: Turn, tick: Tick):
		"""Delete data on or after this tick

		On the assumption that the future has been invalidated.

		"""
		with self._lock:
			assert not self.parents  # not how stuff is stored in this cache
			for branchkey, branches in list(self.branches.items()):
				if branch in branches:
					branhc = branches[branch]
					if turn in branhc:
						trun = branhc[turn]
						if tick in trun:
							del trun[tick]
						trun.truncate(tick)
						if not trun:
							del branhc[turn]
					branhc.truncate(turn)
					if not branhc:
						del branches[branch]
				if not branches:
					del self.branches[branchkey]
			for keykey, keys in list(self.keys.items()):
				for key, branchs in list(keys.items()):
					if branch in branchs:
						branhc = branchs[branch]
						if turn in branhc:
							trun = branhc[turn]
							if tick in trun:
								del trun[tick]
							trun.truncate(tick)
							if not trun:
								del branhc[turn]
						branhc.truncate(turn)
						if not branhc:
							del branchs[branch]
					if not branchs:
						del keys[key]
				if not keys:
					del self.keys[keykey]
			sets = self.settings[branch]
			if turn in sets:
				setsturn = sets[turn]
				if tick in setsturn:
					del setsturn[tick]
				setsturn.truncate(tick)
				if not setsturn:
					del sets[turn]
			sets.truncate(turn)
			if not sets:
				del self.settings[branch]
			presets = self.presettings[branch]
			if turn in presets:
				presetsturn = presets[turn]
				if tick in presetsturn:
					del presetsturn[tick]
				presetsturn.truncate(tick)
				if not presetsturn:
					del presets[turn]
			presets.truncate(turn)
			if not presets:
				del self.presettings[branch]
			for entity, brnch in list(self.keycache):
				if brnch == branch:
					kc = self.keycache[entity, brnch]
					if turn in kc:
						kcturn = kc[turn]
						if tick in kcturn:
							del kcturn[tick]
						kcturn.truncate(tick)
						if not kcturn:
							del kc[turn]
					kc.truncate(turn)
					if not kc:
						del self.keycache[entity, brnch]
			self.shallowest = OrderedDict()

	def get_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		copy: bool = True,
	) -> dict[NodeName, frozenset[NodeName]]:
		ret = self._get_keyframe((graph,), branch, turn, tick)
		if copy:
			ret = ret.copy()
		return ret

	def set_keyframe(
		self,
		graph: CharName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		keyframe: dict[NodeName, frozenset[NodeName]],
	) -> None:
		self._set_keyframe((graph,), branch, turn, tick, keyframe)
