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

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from functools import cached_property
from itertools import chain, pairwise
from operator import itemgetter
from sys import getsizeof, stderr
from threading import RLock
from typing import Hashable, Iterator

from .exc import (
	HistoricKeyError,
	KeyframeError,
	NotInKeyframeError,
	TotalKeyError,
)
from .typing import Branch, Key, Tick, Turn
from .util import sort_set
from .window import (
	Direction,
	EntikeySettingsTurnDict,
	FuturistWindowDict,
	SettingsTurnDict,
	TurnDict,
	WindowDict,
)


class SizedDict(OrderedDict):
	"""A dictionary that discards old entries when it gets too big."""

	def __init__(self, max_entries=1000):
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
		gettest=lambda k: None,
		settest=lambda k, v: None,
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


@dataclass
class TurnEndDict(dict):
	"""Tick on which a (branch, turn) ends, not including any plans"""

	engine: "lisien.Engine"

	@cached_property
	def other_d(self) -> dict[tuple[Branch, Turn], Tick]:
		return self.engine._turn_end_plan

	def __getitem__(self, item: tuple[Branch, Turn]) -> Tick:
		if item not in self:
			if item not in self.other_d:
				dict.__setitem__(self.other_d, item, 0)
				super().__setitem__(item, 0)
				return 0
			else:
				ret = dict.__getitem__(self.other_d, item)
				super().__setitem__(item, ret)
				return ret
		return super().__getitem__(item)

	def __setitem__(self, key: tuple[Branch, Turn], value: Tick):
		dict.__setitem__(self, key, value)
		if (
			not dict.__contains__(self.other_d, key)
			or dict.__getitem__(self.other_d, key) < value
		):
			dict.__setitem__(self.other_d, key, value)


@dataclass
class TurnEndPlanDict(TurnEndDict):
	"""Tick on which a (branch, turn) ends, including plans"""

	engine: "lisien.Engine"

	@cached_property
	def other_d(self) -> dict[tuple[Branch, Turn], Tick]:
		return self.engine._turn_end

	def __setitem__(self, key: tuple[Branch, Turn], value: Tick):
		if dict.__contains__(self.other_d, key):
			assert value >= dict.__getitem__(self.other_d, key)
		else:
			dict.__setitem__(self.other_d, key, value)
		dict.__setitem__(self, key, value)


class Cache:
	"""A data store that's useful for tracking graph revisions."""

	name: str

	def __init__(self, db, name: str, kfkvs=None):
		self.name = name
		self.db = db
		self.parents = StructuredDefaultDict(3, SettingsTurnDict)
		"""Entity data keyed by the entities' parents.

		An entity's parent is what it's contained in. When speaking of a node,
		this is its graph. When speaking of an edge, the parent is usually the
		graph and the origin in a pair, though for multigraphs the destination
		might be part of the parent as well.

		Deeper layers of this cache are keyed by branch and revision.

		"""
		self.keys = StructuredDefaultDict(2, SettingsTurnDict)
		"""Cache of entity data keyed by the entities themselves.

		That means the whole tuple identifying the entity is the
		top-level key in this cache here. The second-to-top level
		is the key within the entity.

		Deeper layers of this cache are keyed by branch, turn, and tick.

		"""
		self.keycache = PickyDefaultDict(SettingsTurnDict)
		"""Keys an entity has at a given turn and tick."""
		self.branches = StructuredDefaultDict(1, SettingsTurnDict)
		"""A less structured alternative to ``keys``.

		For when you already know the entity and the key within it,
		but still need to iterate through history to find the value.

		"""
		self.keyframe = StructuredDefaultDict(
			1, SettingsTurnDict, **(kfkvs or {})
		)
		"""Key-value dictionaries representing my state at a given time"""
		self.shallowest = OrderedDict()
		"""A dictionary for plain, unstructured hinting."""
		self.settings = PickyDefaultDict(EntikeySettingsTurnDict)
		"""All the ``entity[key] = value`` settings on some turn"""
		self.presettings = PickyDefaultDict(EntikeySettingsTurnDict)
		"""The values prior to ``entity[key] = value`` settings on some turn"""
		self.time_entity = {}
		self._kc_lru = OrderedDict()
		self._lock = RLock()
		self._store_stuff = (
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
			db._where_cached,
			self.keycache,
			db,
			self._update_keycache,
		)
		self._remove_stuff = (
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
		self._truncate_stuff = (
			self._lock,
			self.parents,
			self.branches,
			self.keys,
			self.settings,
			self.presettings,
			self.keycache,
		)
		self._store_journal_stuff: tuple[
			PickyDefaultDict, PickyDefaultDict, callable
		] = (self.settings, self.presettings, self._base_retrieve)

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
		self, graph_ent: tuple, branch: str, turn: int, tick: int
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

	def get_keyframe(
		self, graph_ent: tuple, branch: str, turn: int, tick: int, copy=True
	) -> dict:
		ret = self._get_keyframe(graph_ent, branch, turn, tick)
		if copy:
			ret = ret.copy()
		return ret

	def _set_keyframe(
		self, graph_ent: tuple, branch: str, turn: int, tick: int, keyframe
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

	def set_keyframe(
		self,
		graph_ent: tuple,
		branch: str,
		turn: int,
		tick: int,
		keyframe: dict,
	):
		self._set_keyframe(graph_ent, branch, turn, tick, keyframe)

	def alias_keyframe(self, branch_from, branch_to, turn, tick):
		for graph_ent in self.keyframe:
			try:
				kf = self._get_keyframe(graph_ent, branch_from, turn, tick)
			except KeyframeError:
				continue
			self._set_keyframe(graph_ent, branch_to, turn, tick, kf)

	def load(self, data):
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

		branches = defaultdict(list)
		for row in data:
			branches[row[-4]].append(row)
		db = self.db
		# Make keycaches and valcaches. Must be done chronologically
		# to make forwarding work.
		childbranch = db._childbranch
		branch2do = deque(["trunk"])

		store = self.store
		while branch2do:
			branch = branch2do.popleft()
			for row in sorted(branches[branch], key=sort_key):
				try:
					store(*row, planning=False, loading=True)
				except HistoricKeyError:
					continue
			if branch in childbranch:
				branch2do.extend(childbranch[branch])

	def _valcache_lookup(self, cache: dict, branch: str, turn: int, tick: int):
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
		parentity: tuple,
		branch: str,
		turn: int,
		tick: int,
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
							turn - 1,
							self.db.turn_end_plan(branch, turn - 1),
						)
					else:
						stoptime, _ = self.db._build_keyframe_window(
							branch, turn, tick - 1
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
		parentity: tuple,
		branch: str,
		turn: int,
		tick: int,
		*,
		forward: bool,
	):
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

	def _update_keycache(self, *args, forward: bool):
		"""Add or remove a key in the set describing the keys that exist."""
		entity: Hashable
		key: Hashable
		branch: str
		turn: int
		tick: int
		entity, key, branch, turn, tick, value = args[-6:]
		parent = args[:-6]
		kc = self._get_keycache(
			parent + (entity,), branch, turn, tick, forward=forward
		)
		if value is None:
			kc = kc.difference((key,))
		else:
			kc = kc.union((key,))
		if parent + (entity, branch) not in self.keycache:
			self.keycache[parent + (entity, branch)] = SettingsTurnDict(
				{turn: {tick: kc}}
			)
		else:
			self.keycache[parent + (entity, branch)][turn][tick] = kc

	def _get_adds_dels(
		self,
		entity: Hashable,
		branch: str,
		turn: int,
		tick: int,
		*,
		stoptime: tuple[str, int, int] = None,
		cache: dict = None,
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
						if turnd[trn][tck] is None:
							deleted.add(key)
						else:
							added.add(key)
						break
					else:
						trn -= 1
				if not turnd.rev_gettable(trn):
					break
				tickd = turnd[trn]
				if tickd.final() is None:
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

	def store(
		self,
		*args,
		planning: bool = None,
		forward: bool = None,
		loading=False,
		contra: bool = None,
		truncate: bool = False,
	):
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

		``truncate=True`` throws away stored data after the turn and tick
		provided. Be careful!

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
			db_where_cached,
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
		entity: Hashable
		key: Hashable
		branch: str
		turn: int
		tick: int
		entity, key, branch, turn, tick, value = args[-6:]
		if loading:
			self.db._updload(branch, turn, tick)
		parent = args[:-6]
		entikey = (entity, key)
		parentikey = parent + (entity, key)
		contras = []
		with lock:
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
				plan_ticks[plan][turn].add(tick)
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
			branches[branch] = turns
			if not loading and not planning:
				db_extend_branch(branch, turn, tick)
			self_store_journal(*args)
			self.shallowest[parent + (entity, key, branch, turn, tick)] = value
			if turn in turns:
				the_turn = turns[turn]
				if truncate:
					turns.truncate(turn)
					the_turn.truncate(tick)
				the_turn[tick] = value
			else:
				new = FuturistWindowDict()
				new[tick] = value
				turns[turn] = new
			self_time_entity[branch, turn, tick] = parent, entity, key
			where_cached = db_where_cached[args[-4:-1]]
			if self not in where_cached:
				where_cached.append(self)
			# if we're editing the past, have to invalidate the keycache
			keycache_key = parent + (entity, branch)
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
			if not db._no_kc:
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

	def _remove_btt_parentikey(self, branch, turn, tick, parent, entity, key):
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

	def remove(self, branch: str, turn: int, tick: int):
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
			branhc = settings[branch]
			pbranhc = presettings[branch]
			trn = branhc[turn]
			ptrn = pbranhc[turn]
			if tick in trn:
				del trn[tick]
			if tick in ptrn:
				del ptrn[tick]
			if not ptrn:
				del pbranhc[turn]
				del branhc[turn]
			if not pbranhc:
				del settings[branch]
				del presettings[branch]
			self.shallowest = OrderedDict()
			remove_keycache(parent + (entity, branch), turn, tick)

	def _remove_keycache(self, entity_branch: tuple, turn: int, tick: int):
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

	def truncate(self, branch: str, turn: int, tick: int, direction="forward"):
		if direction not in {"forward", "backward"}:
			raise ValueError("Illegal direction")
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
		entity: Hashable,
		key: Hashable,
		turns: WindowDict,
		branch: str,
		turn: int,
		tick: int,
		value,
	):
		"""Iterate over contradicted ``(turn, tick)`` if applicable"""
		# assumes that all future entries are in the plan
		if not turns:
			return
		if turn in turns:
			future_ticks = turns[turn].future(tick)
			for tck, newval in future_ticks.items():
				if newval != value:
					yield turn, tck
			future_turns = turns.future(turn)
		elif turns.rev_gettable(turn):
			future_turns = turns.future(turn)
		else:
			future_turns = turns
		if not future_turns:
			return
		for trn, ticks in future_turns.items():
			for tick, newval in ticks.items():
				if newval != value:
					yield trn, tick

	def _store_journal(self, *args):
		# overridden in lisien.cache.InitializedCache
		(settings, presettings, base_retrieve) = self._store_journal_stuff
		entity: Hashable
		key: Hashable
		branch: str
		turn: int
		tick: int
		entity, key, branch, turn, tick, value = args[-6:]
		parent = args[:-6]
		settings_turns = settings[branch]
		presettings_turns = presettings[branch]
		prev = base_retrieve(args[:-1], store_hint=False)
		if isinstance(prev, KeyError):
			prev = None
		if turn in settings_turns:
			# These assertions hold for most caches but not for the contents
			# caches, and are therefore commented out.
			# assert turn in presettings_turns \
			# or turn in presettings_turns.future()
			setticks = settings_turns[turn]
			# assert tick not in setticks
			presetticks = presettings_turns[turn]
			# assert tick not in presetticks
			presetticks[tick] = parent + (entity, key, prev)
			setticks[tick] = parent + (entity, key, value)
		else:
			presettings_turns[turn] = {tick: parent + (entity, key, prev)}
			settings_turns[turn] = {tick: parent + (entity, key, value)}

	def _base_retrieve(
		self, args, store_hint=True, retrieve_hint=True, search=False
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
		entity: tuple = args[:-4]
		key: Hashable
		branch: str
		turn: int
		tick: int
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
				if b0 in branchentk and r0 in branchentk[b0]:
					# No keyframe *right* now; we might have one earlier this turn
					# ... but let's check branches first
					if t0 in branchentk[b0][r0]:
						# Yeah, branches has a value at this very moment!
						return hint(branchentk[b0][r0][t0])
					elif (
						branches_tick := branchentk[b0][r0].rev_before(
							t0, search=search
						)
					) is not None:
						# branches has a value this turn.
						# Is there a loaded keyframe this turn, as well?
						if b0 == b1 and r0 == r1:
							# There is; is it more recent than branches' value?
							if t1 > branches_tick:
								# It is, so use the keyframe.
								# If that keyframe includes a value stored here,
								# return it; otherwise return an error
								if (
									b1 in keyframes
									and r1 in keyframes[b1]
									and t1 in keyframes[b1][r1]
								):
									kf = keyframes[b1][r1][t1]
									if key in kf:
										return hint(kf[key])
									else:
										return hint(
											NotInKeyframeError(
												"No value", entikey, b1, r1, t1
											)
										)
								else:
									return hint(
										NotInKeyframeError(
											"No value", entikey, b1, r1, t1
										)
									)
						# No keyframe this turn, so use the value from branches
						return hint(get(branchentk[b0][r0], t0))
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

	def retrieve(self, *args, search=False):
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
		if ret is None:
			raise HistoricKeyError("Set, then deleted", deleted=True)
		elif isinstance(ret, Exception):
			raise ret
		return ret

	def iter_entities_or_keys(self, *args, forward: bool = None) -> Iterator:
		"""Iterate over the keys an entity has, if you specify an entity.

		Otherwise iterate over the entities themselves, or at any rate the
		tuple specifying which entity.

		"""
		if forward is None:
			forward = self.db._forward
		entity: tuple = args[:-3]
		branch: str
		turn: int
		tick: int
		branch, turn, tick = args[-3:]
		if self.db._no_kc:
			kc = self._get_adds_dels(entity, branch, turn, tick)[0]
		else:
			try:
				kc = self._get_keycache(
					entity, branch, turn, tick, forward=forward
				)
			except KeyframeError:
				return iter(())
		return iter(kc)

	iter_entities = iter_keys = iter_entity_keys = iter_entities_or_keys

	def count_entities_or_keys(self, *args, forward: bool = None):
		"""Return the number of keys an entity has, if you specify an entity.

		Otherwise return the number of entities.

		"""
		if forward is None:
			forward = self.db._forward
		entity: tuple = args[:-3]
		branch: str
		turn: int
		tick: int
		branch, turn, tick = args[-3:]
		if self.db._no_kc:
			return len(self._get_adds_dels(entity, branch, turn, tick)[0])
		return len(
			self._get_keycache(entity, branch, turn, tick, forward=forward)
		)

	count_entities = count_keys = count_entity_keys = count_entities_or_keys

	def contains_entity_or_key(self, *args):
		"""Check if an entity has a key at the given time, if entity specified.

		Otherwise check if the entity exists.

		"""
		retr = self._base_retrieve(args)
		return retr is not None and not isinstance(retr, Exception)

	contains_entity = contains_key = contains_entity_key = (
		contains_entity_or_key
	)


class GraphValCache(Cache):
	def get_keyframe(
		self, graph: Key, branch: str, turn: int, tick: int, copy=True
	) -> dict:
		return super().get_keyframe((graph,), branch, turn, tick, copy)

	def set_keyframe(
		self, graph: Key, branch: str, turn: int, tick: int, keyframe: dict
	) -> None:
		super().set_keyframe((graph,), branch, turn, tick, keyframe)


class NodesCache(Cache):
	"""A cache for remembering whether nodes exist at a given time."""

	__slots__ = ()

	def store(
		self,
		graph: Hashable,
		node: Hashable,
		branch: str,
		turn: int,
		tick: int,
		ex: bool,
		*,
		planning: bool = None,
		forward: bool = None,
		loading=False,
		contra: bool = None,
	):
		if not ex:
			ex = None
		if contra is None:
			contra = not loading
		return super().store(
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

	def _update_keycache(self, *args, forward):
		graph: Hashable
		node: Hashable
		branch: str
		turn: int
		tick: int
		ex: bool | None
		graph, node, branch, turn, tick, ex = args
		if not ex:
			ex = None
		super()._update_keycache(
			graph, node, branch, turn, tick, ex, forward=forward
		)

	def _iter_future_contradictions(
		self,
		entity: Hashable,
		key: Hashable,
		turns: dict,
		branch: str,
		turn: int,
		tick: int,
		value,
	):
		yield from super()._iter_future_contradictions(
			entity, key, turns, branch, turn, tick, value
		)
		yield from self.db._edges_cache._slow_iter_node_contradicted_times(
			branch, turn, tick, entity, key
		)

	def get_keyframe(
		self, graph: Key, branch: str, turn: int, tick: int, copy=True
	):
		return super().get_keyframe((graph,), branch, turn, tick, copy)

	def set_keyframe(
		self, graph: Key, branch: str, turn: int, tick: int, keyframe: dict
	) -> None:
		super().set_keyframe((graph,), branch, turn, tick, keyframe)


class NodeValCache(Cache):
	def get_keyframe(
		self, graph: Key, branch: str, turn: int, tick: int, copy=True
	):
		ret = super().get_keyframe((graph,), branch, turn, tick, copy=copy)
		if copy:
			for k, v in ret.items():
				ret[k] = v.copy()
		return ret

	def set_keyframe(
		self,
		graph: Key,
		branch: str,
		turn: int,
		tick: int,
		keyframe: dict,
	):
		super().set_keyframe((graph,), branch, turn, tick, keyframe)
		for node, vals in keyframe.items():
			super().set_keyframe((graph, node), branch, turn, tick, vals)


class EdgesCache(Cache):
	"""A cache for remembering whether edges exist at a given time."""

	__slots__ = (
		"destcache",
		"origcache",
		"predecessors",
		"_origcache_lru",
		"_destcache_lru",
		"_get_destcache_stuff",
		"_get_origcache_stuff",
		"_additional_store_stuff",
	)

	@property
	def successors(self):
		return self.parents

	def __init__(self, db, name, kfkvs=None):
		super().__init__(db, name, kfkvs)
		self.destcache = PickyDefaultDict(SettingsTurnDict)
		self.origcache = PickyDefaultDict(SettingsTurnDict)
		self.predecessors = StructuredDefaultDict(3, TurnDict)
		self._origcache_lru = OrderedDict()
		self._destcache_lru = OrderedDict()
		self._get_destcache_stuff: tuple[
			PickyDefaultDict,
			OrderedDict,
			callable,
			StructuredDefaultDict,
			callable,
		] = (
			self.destcache,
			self._destcache_lru,
			self._get_keycachelike,
			self.successors,
			self._adds_dels_successors,
		)
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

	def total_size(self, handlers=(), verbose=False):
		all_handlers = {
			EdgesCache: lambda e: [
				e.predecessors,
				e.successors,
				e.destcache,
				e.origcache,
			]
		}
		all_handlers.update(handlers)
		return super().total_size(all_handlers, verbose)

	def get_keyframe(
		self, graph: Key, branch: str, turn: int, tick: int, copy=True
	) -> dict:
		ret = super().get_keyframe((graph,), branch, turn, tick, copy)
		if copy:
			for orig, dests in list(ret.items()):
				ret[orig] = dests.copy()
		return ret

	def set_keyframe(
		self, graph: Key, branch: str, turn: int, tick: int, keyframe: dict
	) -> None:
		super().set_keyframe((graph,), branch, turn, tick, keyframe)
		for orig, dests in keyframe.items():
			super().set_keyframe((graph, orig), branch, turn, tick, dests)
			for dest, ex in dests.items():
				super().set_keyframe(
					(graph, orig, dest), branch, turn, tick, {0: ex}
				)

	def _update_keycache(self, *args, forward: bool):
		super()._update_keycache(*args, forward=forward)
		dest: Hashable
		key: Hashable
		branch: str
		turn: int
		tick: int
		dest, key, branch, turn, tick, value = args[-6:]
		graph, orig = args[:-6]
		# it's possible either of these might cause unnecessary iteration
		dests = self._get_destcache(
			graph, orig, branch, turn, tick, forward=forward
		)
		origs = self._get_origcache(
			graph, dest, branch, turn, tick, forward=forward
		)
		if value is None:
			dests = dests.difference((dest,))
			origs = origs.difference((orig,))
		else:
			dests = dests.union((dest,))
			origs = origs.union((orig,))
		self.destcache[graph, orig, branch][turn][tick] = dests
		self.origcache[graph, dest, branch][turn][tick] = origs

	def _slow_iter_node_contradicted_times(
		self,
		branch: str,
		turn: int,
		tick: int,
		graph: Hashable,
		node: Hashable,
	):
		# slow and bad.
		retrieve = self._base_retrieve
		for items in (
			self.successors[graph, node].items(),
			self.predecessors[graph, node].items(),
		):
			for dest, idxs in items:  # dest might really be orig
				for idx, branches in idxs.items():
					brnch = branches[branch]
					if turn in brnch:
						ticks = brnch[turn]
						for tck, present in ticks.future(tick).items():
							if tck > tick and present is not retrieve(
								(graph, node, dest, idx, branch, turn, tick)
							):
								yield turn, tck
					for trn, ticks in brnch.future(turn).items():
						for tck, present in ticks.items():
							if present is not retrieve(
								(graph, node, dest, idx, branch, turn, tick)
							):
								yield trn, tck

	def _adds_dels_successors(
		self,
		parentity: tuple,
		branch: str,
		turn: int,
		tick: int,
		*,
		stoptime: tuple[str, int, int] = None,
		cache: dict = None,
	):
		graph, orig = parentity
		added = set()
		deleted = set()
		cache = cache or self.successors
		if (graph, orig) in cache and cache[graph, orig]:
			for dest in cache[graph, orig]:
				addidx, delidx = self._get_adds_dels(
					(graph, orig, dest), branch, turn, tick, stoptime=stoptime
				)
				if addidx and not delidx:
					added.add(dest)
				elif delidx and not addidx:
					deleted.add(dest)
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
						if kfgbr[tck][0] and dest not in deleted:
							added.add(dest)
						continue
				if kfgb.rev_gettable(trn):
					if kfgb[trn].final()[0] and dest not in deleted:
						added.add(dest)
		return added, deleted

	def _adds_dels_predecessors(
		self,
		parentity: tuple,
		branch: str,
		turn: int,
		tick: int,
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
							if kfgbr[tck][0]:
								added.add(orig)
							continue
					if kfgb.rev_gettable(trn):
						if kfgb[trn].final()[0]:
							added.add(orig)
		return added, deleted

	def _get_destcache(
		self,
		graph: Hashable,
		orig: Hashable,
		branch: str,
		turn: int,
		tick: int,
		*,
		forward: bool,
	):
		"""Return a set of destination nodes succeeding ``orig``"""
		(
			destcache,
			destcache_lru,
			get_keycachelike,
			successors,
			adds_dels_sucpred,
		) = self._get_destcache_stuff
		return get_keycachelike(
			destcache,
			successors,
			adds_dels_sucpred,
			(graph, orig),
			branch,
			turn,
			tick,
			forward=forward,
		)

	def _get_origcache(
		self,
		graph: Hashable,
		dest: Hashable,
		branch: str,
		turn: int,
		tick: int,
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
		self, graph, orig, branch, turn, tick, *, forward=None
	):
		"""Iterate over successors of a given origin node at a given time."""
		if self.db._no_kc:
			yield from self._adds_dels_successors(
				(graph, orig), branch, turn, tick
			)[0]
			return
		if forward is None:
			forward = self.db._forward
		yield from self._get_destcache(
			graph, orig, branch, turn, tick, forward=forward
		)

	def iter_predecessors(
		self,
		graph: Hashable,
		dest: Hashable,
		branch: str,
		turn: int,
		tick: int,
		*,
		forward: bool = None,
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

	def count_successors(
		self,
		graph: Hashable,
		orig: Hashable,
		branch: str,
		turn: int,
		tick: int,
		*,
		forward: str = None,
	):
		"""Return the number of successors to an origin node at a given time."""
		if self.db._no_kc:
			return len(
				self._adds_dels_successors((graph, orig), branch, turn, tick)[
					0
				]
			)
		if forward is None:
			forward = self.db._forward
		return len(
			self._get_destcache(
				graph, orig, branch, turn, tick, forward=forward
			)
		)

	def count_predecessors(
		self,
		graph: Hashable,
		dest: Hashable,
		branch: str,
		turn: int,
		tick: int,
		*,
		forward: bool = None,
	):
		"""Return the number of predecessors from a destination node at a time."""
		if self.db._no_kc:
			return len(
				self._adds_dels_predecessors(
					(graph, dest), branch, turn, tick
				)[0]
			)
		if forward is None:
			forward = self.db._forward
		return len(
			self._get_origcache(
				graph, dest, branch, turn, tick, forward=forward
			)
		)

	def has_successor(
		self,
		graph: Hashable,
		orig: Hashable,
		dest: Hashable,
		branch: str,
		turn: int,
		tick: int,
		*,
		forward: bool = None,
	):
		"""Return whether an edge connects the origin to the destination now"""
		# Use a keycache if we have it.
		# If we don't, only generate one if we're forwarding, and only
		# if it's no more than a turn ago.
		keycache_key = (graph, orig, dest, branch)
		if keycache_key in self.keycache:
			return dest in self._get_destcache(
				graph, orig, branch, turn, tick, forward=forward
			)
		got = self._base_retrieve((graph, orig, dest, 0, branch, turn, tick))
		return got is not None and not isinstance(got, Exception)

	def has_predecessor(
		self,
		graph: Hashable,
		dest: Hashable,
		orig: Hashable,
		branch: str,
		turn: int,
		tick: int,
		*,
		forward: bool = None,
	):
		"""Return whether an edge connects the destination to the origin now"""
		got = self._base_retrieve((graph, orig, dest, 0, branch, turn, tick))
		return got is not None and not isinstance(got, Exception)

	def store(
		self,
		graph,
		orig,
		dest,
		idx,
		branch,
		turn,
		tick,
		ex,
		*,
		planning: bool = None,
		forward: bool = None,
		loading=False,
		contra: bool = None,
	):
		if contra is None:
			contra = not loading
		db, predecessors, successors = self._additional_store_stuff
		if not ex:
			ex = None
		if planning is None:
			planning = db._planning
		Cache.store(
			self,
			graph,
			orig,
			dest,
			idx,
			branch,
			turn,
			tick,
			ex,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)
		try:
			predecessors[graph, dest][orig][idx][branch][turn] = successors[
				graph, orig
			][dest][idx][branch][turn]
		except HistoricKeyError:
			pass

	# if ex:
	# assert self.retrieve(graph, orig, dest, idx, branch, turn, tick)
	# assert self.has_successor(graph, orig, dest, branch, turn, tick)
	# assert self.has_predecessor(g
	# raph, dest, orig, branch, turn, tick)
	# else:
	# assert self._base_retrieve(
	# (graph, orig, dest, idx, branch, turn, tick)) in (None, KeyError)
	# assert not self.has_successor(
	# graph, orig, dest, branch, turn, tick)
	# assert not self.has_predecessor(
	# graph, dest, orig, branch, turn, tick)


class EdgeValCache(Cache):
	def get_keyframe(
		self, graph: Key, branch: str, turn: int, tick: int, copy=True
	):
		ret = super().get_keyframe((graph,), branch, turn, tick, copy)
		if copy:
			for orig, dests in list(ret.items()):
				redests = {}
				for dest, val in dests.items():
					redests[dest] = val.copy()
				ret[orig] = redests
		return ret

	def set_keyframe(
		self, graph: Key, branch: str, turn: int, tick: int, keyframe: dict
	):
		super().set_keyframe((graph,), branch, turn, tick, keyframe)
		for orig, dests in keyframe.items():
			for dest, val in dests.items():
				super().set_keyframe(
					(graph, orig, dest, 0), branch, turn, tick, val
				)


class EntitylessCache(Cache):
	__slots__ = ()

	def store(
		self,
		key,
		branch,
		turn,
		tick,
		value,
		*,
		planning: bool = None,
		forward: bool = None,
		loading=False,
		contra: bool = None,
	):
		super().store(
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

	def get_keyframe(self, branch, turn, tick, copy=True):
		return super().get_keyframe((None,), branch, turn, tick, copy=copy)

	def set_keyframe(self, branch, turn, tick, keyframe):
		super().set_keyframe((None,), branch, turn, tick, keyframe)

	def iter_entities_or_keys(self, branch, turn, tick, *, forward=None):
		return super().iter_entities_or_keys(
			None, branch, turn, tick, forward=forward
		)

	iter_entities = iter_keys = iter_entities_or_keys

	def contains_entity_or_key(self, ke, branch, turn, tick):
		return super().contains_entity_or_key(None, ke, branch, turn, tick)

	contains_entity = contains_key = contains_entity_or_key

	def retrieve(self, *args):
		return super().retrieve(*(None,) + args)


class InitializedCache(Cache):
	__slots__ = ()

	def _store_journal(self, *args):
		entity, key, branch, turn, tick, value = args[-6:]
		parent = args[:-6]
		settings_turns = self.settings[branch]
		presettings_turns = self.presettings[branch]
		try:
			prev = self.retrieve(*args[:-1])
		except KeyError:
			prev = None
		if prev == value:
			return  # not much point reporting on a non-change in a diff
		if turn in settings_turns or turn in settings_turns.future():
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


class NodesRulebooksCache(InitializedCache):
	def get_keyframe(
		self, graph: Key, branch: str, turn: int, tick: int, copy=True
	) -> dict:
		return super().get_keyframe((graph,), branch, turn, tick, copy)

	def set_keyframe(
		self,
		graph: Key,
		branch: str,
		turn: int,
		tick: int,
		keyframe: dict,
	) -> None:
		super().set_keyframe((graph,), branch, turn, tick, keyframe)


class InitializedEntitylessCache(EntitylessCache, InitializedCache):
	__slots__ = ()


class CharactersRulebooksCache(InitializedEntitylessCache):
	def set_keyframe(self, branch, turn, tick, keyframe):
		super().set_keyframe(branch, turn, tick, keyframe)
		for char, kf in keyframe.items():
			super(EntitylessCache, self).set_keyframe(
				(char,), branch, turn, tick, kf
			)


class PortalsRulebooksCache(InitializedCache):
	def store(
		self,
		*args,
		planning: bool = None,
		forward: bool = None,
		loading=False,
		contra: bool = None,
	):
		char, orig, dest, branch, turn, tick, rb = args
		try:
			destrbs = self.retrieve(char, orig, branch, turn, tick)
			destrbs[dest] = rb
		except KeyError:
			destrbs = {dest: rb}
		super().store(
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
		super().store(
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

	def get_keyframe(
		self, graph: Key, branch: str, turn: int, tick: int, copy=True
	) -> dict:
		ret = super().get_keyframe((graph,), branch, turn, tick, copy)
		if copy:
			for orig, dests in list(ret.items()):
				ret[orig] = dests.copy()
		return ret

	def get_orig_keyframe(
		self,
		graph: Key,
		orig: Key,
		branch: str,
		turn: int,
		tick: int,
		copy=True,
	):
		return super().get_keyframe((graph, orig), branch, turn, tick, copy)

	def set_keyframe(
		self,
		graph: Key,
		branch: str,
		turn: int,
		tick: int,
		keyframe,
	):
		super().set_keyframe((graph,), branch, turn, tick, keyframe)
		for orig, dests in keyframe.items():
			for dest, rulebook in dests.items():
				try:
					subkf = super().get_keyframe(
						(graph, orig), branch, turn, tick, copy=True
					)
					subkf[dest] = rulebook
				except KeyError:
					subkf = {dest: rulebook}
				super().set_keyframe((graph, orig), branch, turn, tick, subkf)


class UnitnessCache(Cache):
	"""A cache for remembering when a node is a unit of a character."""

	def __init__(self, db, name, kfkvs=None):
		super().__init__(db, name, kfkvs)
		self.user_cache = Cache(db, "user_cache")

	def store(
		self,
		character,
		graph,
		node,
		branch,
		turn,
		tick,
		is_unit,
		*,
		planning: bool = None,
		forward: bool = None,
		loading=False,
		contra: bool = None,
	):
		is_unit = True if is_unit else None
		super().store(
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
			noded = self.retrieve(character, graph, branch, turn, tick).copy()
			noded[node] = is_unit
		except KeyError:
			noded = {node: is_unit}
		super().store(
			character,
			graph,
			branch,
			turn,
			tick,
			noded,
			planning=planning,
			forward=forward,
			loading=loading,
			contra=contra,
		)
		try:
			users = self.user_cache.retrieve(graph, node, branch, turn, tick)
			users[character] = frozenset(users[character] | {node})
		except KeyError:
			users = {character: frozenset([node] if is_unit else [])}
		self.user_cache.store(graph, node, branch, turn, tick, users)
		self.user_cache.store(
			graph,
			node,
			character,
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
		character: Key,
		branch: str,
		turn: int,
		tick: int,
		keyframe,
	):
		super().set_keyframe((character,), branch, turn, tick, keyframe)
		for graph, subkf in keyframe.items():
			super().set_keyframe((character, graph), branch, turn, tick, subkf)
			if isinstance(subkf, dict):
				for unit, is_unit in subkf.items():
					try:
						kf = self.user_cache.get_keyframe(
							(graph, unit), branch, turn, tick
						)
						kf[character] = is_unit
					except KeyframeError:
						self.user_cache.set_keyframe(
							(graph, unit),
							branch,
							turn,
							tick,
							{character: is_unit},
						)

	def get_keyframe(
		self, characters, branch: str, turn: int, tick: int, copy=True
	) -> dict:
		if isinstance(characters, tuple) and len(characters) > 1:
			return super().get_keyframe(characters, branch, turn, tick, copy)
		# only one character
		return super().get_keyframe((characters,), branch, turn, tick, copy)

	def get_char_graph_units(self, char, graph, branch, turn, tick):
		return set(self.iter_entities(char, graph, branch, turn, tick))

	def get_char_only_unit(self, char, branch, turn, tick):
		if self.count_entities(char, branch, turn, tick) != 1:
			raise ValueError("No unit, or more than one unit")
		for graph in self.iter_entities(char, branch, turn, tick):
			if self.count_entities(char, graph, branch, turn, tick) != 1:
				raise ValueError("No unit, or more than one unit")
			return graph, next(
				self.iter_entities(char, graph, branch, turn, tick)
			)

	def get_char_only_graph(self, char, branch, turn, tick):
		if self.count_entities(char, branch, turn, tick) != 1:
			raise ValueError("No unit, or more than one unit")
		return next(self.iter_entities(char, branch, turn, tick))

	def iter_char_graphs(self, char, branch, turn, tick):
		return self.iter_entities(char, branch, turn, tick)


class RulesHandledCache:
	def __init__(self, engine, name: str):
		self.lock = RLock()
		self.engine = engine
		self.name = name
		self.handled = {}
		self.handled_deep = PickyDefaultDict(SettingsTurnDict)

	def get_rulebook(self, *args):
		raise NotImplementedError

	def iter_unhandled_rules(self, branch, turn, tick):
		raise NotImplementedError

	def store(self, *args, loading=False):
		entity = args[:-5]
		rulebook, rule, branch, turn, tick = args[-5:]
		self.handled.setdefault(entity + (rulebook, branch, turn), set()).add(
			rule
		)
		if turn in self.handled_deep[branch]:
			self.handled_deep[branch][turn][tick] = (entity, rulebook, rule)
		else:
			self.handled_deep[branch][turn] = {tick: (entity, rulebook, rule)}

	def remove_branch(self, branch: str):
		if branch in self.handled_deep:
			for turn, ticks in self.handled_deep[branch].items():
				for tick, (entity, rulebook, rule) in ticks.items():
					if (entity, rulebook, branch, turn) in self.handled:
						del self.handled[entity, rulebook, branch, turn]
			del self.handled_deep[branch]

	def truncate(self, branch: str, turn: int, tick: int, direction="forward"):
		if direction not in {"forward", "backward"}:
			raise ValueError("Illegal direction")
		if branch not in self.handled_deep:
			return

		with self.lock:
			turn_d = self.handled_deep[branch]
			if turn in turn_d:
				for t, (entity, rulebook, rule) in (
					turn_d[turn].future(tick)
					if direction == "forward"
					else turn_d[turn].past(tick)
				).items():
					if (entity, rulebook, branch, turn) in self.handled:
						tick_set = self.handled[entity, rulebook, branch, turn]
						tick_set.discard(t)
						if not tick_set:
							del self.handled[entity, rulebook, branch, turn]
				turn_d[turn].truncate(turn, direction)
			to_del = (
				turn_d.future(turn)
				if direction == "forward"
				else turn_d.past(turn)
			)
			for r in to_del.keys():
				for t, (entity, rulebook, rule) in turn_d[r].items():
					if (entity, rulebook, branch, r) in self.handled:
						tick_set = self.handled[entity, rulebook, branch, r]
						tick_set.discard(t)
						if not tick_set:
							del self.handled[entity, rulebook, branch, r]
			turn_d.truncate(turn, direction)
		if turn in self.handled_deep[branch]:
			self.handled_deep[branch][turn][tick] = (entity, rulebook, rule)
		else:
			self.handled_deep[branch][turn] = {tick: (entity, rulebook, rule)}

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
			RulesHandledCache: lambda d: [d.handled, d.handled_deep],
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

	def truncate(
		self,
		branch: str,
		turn: int,
		tick: int,
		direction: Direction = Direction.FORWARD,
	):
		if isinstance(direction, str):
			direction = Direction(direction)
		if branch not in self.handled_deep:
			return

		with self.lock:
			turn_d = self.handled_deep[branch]
			if turn in turn_d:
				for t, (entity, rulebook, rule) in (
					turn_d[turn].future(tick)
					if direction == "forward"
					else turn_d[turn].past(tick)
				).items():
					if (entity, rulebook, branch, turn) in self.handled:
						tick_set = self.handled[entity, rulebook, branch, turn]
						tick_set.discard(t)
						if not tick_set:
							del self.handled[entity, rulebook, branch, turn]
				turn_d[turn].truncate(turn, direction)
			to_del = (
				turn_d.future(turn)
				if direction == "forward"
				else turn_d.past(turn)
			)
			for r in to_del.keys():
				for t, (entity, rulebook, rule) in turn_d[r].items():
					if (entity, rulebook, branch, r) in self.handled:
						tick_set = self.handled[entity, rulebook, branch, r]
						tick_set.discard(t)
						if not tick_set:
							del self.handled[entity, rulebook, branch, r]
			turn_d.truncate(turn, direction)

	def retrieve(self, *args):
		return self.handled[args]

	def get_handled_rules(self, entity, rulebook, branch, turn):
		return self.handled.setdefault(
			entity + (rulebook, branch, turn), set()
		)


class CharacterRulesHandledCache(RulesHandledCache):
	def get_rulebook(self, character, branch, turn, tick):
		try:
			return self.engine._characters_rulebooks_cache.retrieve(
				character, branch, turn, tick
			)
		except KeyError:
			return ("character_rulebook", character)

	def iter_unhandled_rules(self, branch, turn, tick):
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


class UnitRulesHandledCache(RulesHandledCache):
	def get_rulebook(self, character, branch, turn, tick):
		try:
			return self.engine._units_rulebooks_cache.retrieve(
				character, branch, turn, tick
			)
		except KeyError:
			return "unit_rulebook", character

	def iter_unhandled_rules(self, branch, turn, tick):
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
			for graphname in self.engine._unitness_cache.iter_keys(
				charname, branch, turn, tick
			):
				# Seems bad that I have to check twice like this.
				try:
					existences = self.engine._unitness_cache.retrieve(
						charname, graphname, branch, turn, tick
					)
				except KeyError:
					continue
				for node, ex in existences.items():
					if not ex:
						continue
					handled = self.get_handled_rules(
						(charname, graphname), rb, branch, turn
					)
					for rule in rules:
						if rule not in handled:
							yield prio, charname, graphname, node, rb, rule


class CharacterThingRulesHandledCache(RulesHandledCache):
	def get_rulebook(self, character, branch, turn, tick):
		try:
			return self.engine._characters_things_rulebooks_cache.retrieve(
				character, branch, turn, tick
			)
		except KeyError:
			return "character_thing_rulebook", character

	def iter_unhandled_rules(self, branch, turn, tick):
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


class CharacterPlaceRulesHandledCache(RulesHandledCache):
	def get_rulebook(self, character, branch, turn, tick):
		try:
			return self.engine._characters_places_rulebooks_cache.retrieve(
				character, branch, turn, tick
			)
		except KeyError:
			return "character_place_rulebook", character

	def iter_unhandled_rules(self, branch, turn, tick):
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


class CharacterPortalRulesHandledCache(RulesHandledCache):
	def get_rulebook(self, character, branch, turn, tick):
		try:
			return self.engine._characters_portals_rulebooks_cache.retrieve(
				character, branch, turn, tick
			)
		except KeyError:
			return "character_portal_rulebook", character

	def iter_unhandled_rules(self, branch, turn, tick):
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


class NodeRulesHandledCache(RulesHandledCache):
	def get_rulebook(self, character, node, branch, turn, tick):
		try:
			return self.engine._nodes_rulebooks_cache.retrieve(
				character, node, branch, turn, tick
			)
		except KeyError:
			return character, node

	def iter_unhandled_rules(self, branch, turn, tick):
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


class PortalRulesHandledCache(RulesHandledCache):
	def get_rulebook(self, character, orig, dest, branch, turn, tick):
		try:
			return self.engine._portals_rulebooks_cache.retrieve(
				character, orig, dest, branch, turn, tick
			)
		except KeyError:
			return character, orig, dest

	def iter_unhandled_rules(self, branch, turn, tick):
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
					destrbs = self.engine._portals_rulebooks_cache.retrieve(
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


class ThingsCache(Cache):
	def __init__(self, db, name, kfkvs=None):
		super().__init__(db, name, kfkvs)
		self._make_node = db.thing_cls

	def _slow_iter_contents(self, character, place, branch, turn, tick):
		for thing in self.iter_keys(character, branch, turn, tick):
			try:
				loc = self.retrieve(character, thing, branch, turn, tick)
			except KeyError:
				continue
			if loc == place:
				yield thing

	def _retrieve_or_generate_contents(
		self, character, location, branch, turn, tick
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

	def store(self, *args, planning=None, loading=False, contra=None):
		character, thing, branch, turn, tick, location = args
		with self._lock:
			try:
				oldloc = self.retrieve(character, thing, branch, turn, tick)
			except KeyError:
				oldloc = None
			super().store(
				*args, planning=planning, loading=loading, contra=contra
			)
			node_contents_cache = self.db._node_contents_cache
			this = frozenset((thing,))
			# Cache the contents of nodes
			todo = defaultdict(list)
			if oldloc is not None:
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
			if location is not None:
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

	def turn_before(self, character, thing, branch, turn):
		with self._lock:
			try:
				self.retrieve(character, thing, branch, turn, 0)
			except KeyError:
				pass
			return self.keys[(character,)][thing][branch].rev_before(turn)

	def turn_after(self, character, thing, branch, turn):
		with self._lock:
			try:
				self.retrieve(character, thing, branch, turn, 0)
			except KeyError:
				pass
			return self.keys[(character,)][thing][branch].rev_after(turn)

	def get_keyframe(
		self, graph: Key, branch: str, turn: int, tick: int, copy=True
	):
		return super().get_keyframe((graph,), branch, turn, tick, copy)

	def set_keyframe(
		self, graph: Key, branch: str, turn: int, tick: int, keyframe: dict
	) -> None:
		super().set_keyframe((graph,), branch, turn, tick, keyframe)


class NodeContentsCache(Cache):
	def __init__(self, db, name, kfkvs=None):
		super().__init__(db, name, kfkvs)
		self.loc_settings = StructuredDefaultDict(1, SettingsTurnDict)

	def delete_plan(self, plan: int) -> None:
		branch, turn, tick = self.db._btt()
		plan_ticks = self.db._plan_ticks[plan]
		with self.db.world_lock:
			for trn, tcks in plan_ticks.items():
				if trn == turn:
					for tck in tcks:
						if (
							tck >= tick
							and self in self.db._where_cached[branch, trn, tck]
						):
							self.remove(branch, trn, tck)
							self.db._where_cached[branch, trn, tck].remove(
								self
							)
				elif trn > turn:
					for tck in tcks:
						if self not in self.db._where_cached[branch, trn, tck]:
							continue
						self.remove(branch, trn, tck)
						self.db._where_cached[branch, trn, tck].remove(self)

	def store(
		self,
		character: Key,
		place: Key,
		branch: str,
		turn: int,
		tick: int,
		contents: frozenset,
		planning: bool = True,
		forward: bool = None,
		loading=False,
		contra: bool = None,
		truncate: bool = True,
	):
		self.loc_settings[character, place][branch].store_at(
			turn, tick, contents
		)

		return super().store(
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
			truncate=truncate,
		)

	def _iter_future_contradictions(
		self, entity, key, turns, branch, turn, tick, value
	):
		return self.db._things_cache._iter_future_contradictions(
			entity, key, turns, branch, turn, tick, value
		)

	def remove(self, branch, turn, tick):
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
		self, graph: Key, branch: str, turn: int, tick: int, copy=True
	) -> dict:
		return super().get_keyframe((graph,), branch, turn, tick, copy)

	def set_keyframe(
		self, graph: Key, branch: str, turn: int, tick: int, keyframe: dict
	) -> None:
		super().set_keyframe((graph,), branch, turn, tick, keyframe)
