# This file is part of allegedb, a database abstraction for versioned graphs
# Copyright (c) Zachary Spector. public@zacharyspector.com
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

from typing import Tuple, Hashable, Optional

from .window import (
	WindowDict,
	HistoricKeyError,
	FuturistWindowDict,
	TurnDict,
	SettingsTurnDict,
	EntikeySettingsTurnDict,
)
from collections import OrderedDict, defaultdict, deque
from threading import RLock


class NotInKeyframeError(KeyError):
	pass


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

	__slots__ = ["type", "args_munger", "kwargs_munger", "parent", "key"]

	def __init__(
		self,
		type: type,
		args_munger: callable = _default_args_munger,
		kwargs_munger: callable = _default_kwargs_munger,
	):
		self.type = type
		self.args_munger = args_munger
		self.kwargs_munger = kwargs_munger

	def __getitem__(self, k):
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
		if type(v) is not self.type:
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
		self.layer = layers
		self.type = type
		self.args_munger = args_munger
		self.kwargs_munger = kwargs_munger
		self._stuff = (layers, type, args_munger, kwargs_munger)
		self.gettest = gettest
		self.settest = settest

	def __getitem__(self, k):
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


class Cache:
	"""A data store that's useful for tracking graph revisions."""

	def __init__(self, db, kfkvs=None):
		super().__init__()
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
			db.delete_plan,
			db._time_plan,
			self._iter_future_contradictions,
			db._branches,
			db._turn_end,
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
		self._store_journal_stuff: Tuple[
			PickyDefaultDict, PickyDefaultDict, callable
		] = (self.settings, self.presettings, self._base_retrieve)

	def _get_keyframe(
		self, graph_ent: tuple, branch: str, turn: int, tick: int, copy=True
	):
		if graph_ent not in self.keyframe:
			raise KeyError("Unknown graph-entity", graph_ent)
		g = self.keyframe[graph_ent]
		if branch not in g:
			raise KeyError("Unknown branch", branch)
		b = g[branch]
		if turn not in b:
			raise KeyError("Unknown turn", turn)
		r = b[turn]
		if tick not in r:
			raise KeyError("Unknown tick", tick)
		ret = r[tick]
		if copy:
			ret = ret.copy()
		return ret

	def get_keyframe(
		self, graph_ent: tuple, branch: str, turn: int, tick: int, copy=True
	):
		return self._get_keyframe(graph_ent, branch, turn, tick, copy=copy)

	def set_keyframe(
		self, graph_ent: tuple, branch: str, turn: int, tick: int, keyframe
	):
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

	def load(self, data):
		"""Add a bunch of data. Must be in chronological order.

		But it doesn't need to all be from the same branch, as long as
		each branch is chronological of itself.

		"""
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
			for row in branches[branch]:
				store(*row, planning=False, loading=True)
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
				# allegedb proper, meaning they'll all get cached. In LiSE this
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
							tick_before = keycache2[turn_before].end
							keys_before = keycache2[turn_before][tick_before]
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
			stoptime, _ = self.db._build_keyframe_window_new(
				branch, turn, tick
			)
			if stoptime is None:
				ret = None
				if branch in self.keyframe:
					kfb = self.keyframe[branch]
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
				except KeyError:
					adds, _ = get_adds_dels(
						parentity, branch, turn, tick, stoptime=stoptime
					)
					ret = frozenset(adds)
			else:
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
		self.keycache[parent + (entity, branch)][turn][tick] = kc

	def _get_adds_dels(
		self,
		entity: Hashable,
		branch: str,
		turn: int,
		tick: int,
		*,
		stoptime: Tuple[str, int, int] = None,
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
			elif kfb.rev_gettable(trn) and kfb[trn]:
				added.update(set(kfb[trn].final()).difference(deleted))
			break
		return added, deleted

	def store(
		self,
		*args,
		planning: bool = None,
		forward: bool = None,
		loading=False,
		contra=None,
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

		"""
		(
			lock,
			self_parents,
			self_branches,
			self_keys,
			delete_plan,
			time_plan,
			self_iter_future_contradictions,
			db_branches,
			db_turn_end,
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
				if turn in turns and tick < turns[turn].end:
					raise HistoricKeyError(
						"Already have some ticks after {} in turn {} of branch {}".format(
							tick, turn, branch
						)
					)
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
				parbranch, turn_start, tick_start, turn_end, tick_end = (
					db_branches[branch]
				)
				if (turn, tick) > (turn_end, tick_end):
					db_branches[branch] = (
						parbranch,
						turn_start,
						tick_start,
						turn,
						tick,
					)
				if tick > db_turn_end[branch, turn]:
					db_turn_end[branch, turn] = tick
			self_store_journal(*args)
			self.shallowest[parent + (entity, key, branch, turn, tick)] = value
			if turn in turns:
				the_turn = turns[turn]
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
		# overridden in LiSE.cache.InitializedCache
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
		prev = base_retrieve(args[:-1])
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
		if entikey in branches:
			branchentk = branches[entikey]
			for b, r, t in self.db._iter_parent_btt(branch, turn, tick):
				brancs = branchentk.get(b)
				if brancs is not None and brancs.rev_gettable(r):
					if r in brancs and brancs[r].rev_gettable(t):
						# if there's a keyframe *later* than in brancs,
						# but *earlier* than (b, r, t), use the
						# keyframe instead
						if b in keyframes and r in keyframes[b]:
							if search:
								kfbr = keyframes[b].search(r)
							else:
								kfbr = keyframes[b][r]
							if search:
								brancs.search(r)
							if kfbr.rev_before(
								t, search=search
							) is not None and (
								brancs[r].rev_before(t, search=search)
								< kfbr.rev_before(t, search=search)
								< t
							):
								kf = kfbr[t]
								if key in kf:
									ret = kf[key]
									if store_hint:
										shallowest[args] = ret
									return ret
								else:
									return NotInKeyframeError(
										f"No value for {entikey} at {b, r, t}"
									)
						if search:
							ret = brancs.search(r).search(t)
						else:
							ret = brancs[r][t]
						if store_hint:
							shallowest[args] = ret
						return ret
					elif brancs.rev_gettable(r - 1):
						if b in keyframes and keyframes[b].rev_gettable(r - 1):
							kfb = keyframes[b]
							if kfb.rev_before(
								r - 1, search=search
							) is not None and (
								brancs.rev_before(r - 1, search=search)
								< kfb.rev_before(r - 1, search=search)
							):
								kfbr = kfb[r - 1]
								kf = kfbr.final()
								if key in kf:
									ret = kf[key]
									if store_hint:
										shallowest[args] = ret
									return ret
								else:
									return NotInKeyframeError(
										f"No value for {entikey} at {b, r, t}"
									)
							elif brancs.rev_before(
								r - 1, search=search
							) == kfb.rev_before(r - 1, search=search):
								if search:
									kfbr = kfb.search(r - 1)
									trns = brancs.search(r - 1)
								else:
									kfbr = kfb[r - 1]
									trns = brancs[r - 1]
								if trns.end < kfbr.end:
									kf = kfbr.final()
									if key in kf:
										ret = kf[key]
										if store_hint:
											shallowest[args] = ret
										return ret
									else:
										return NotInKeyframeError(
											f"No value for {entikey} at {b, r, t}"
										)
						if search:
							ret = brancs.search(r - 1).final()
						else:
							ret = brancs[r - 1].final()
						if store_hint:
							shallowest[args] = ret
						return ret
					elif (
						b in keyframes
						and r in keyframes[b]
						and (
							(search and keyframes[b].search(r).rev_gettable(t))
							or (not search and keyframes[b][r].rev_gettable(t))
						)
					):
						brtk = keyframes[b][r][t]
						if key in brtk:
							ret = brtk[key]
							if store_hint:
								shallowest[args] = ret
							return ret
						else:
							return NotInKeyframeError(
								f"No value for {entikey} at {b, r, t}"
							)
					elif b in keyframes and keyframes[b].rev_gettable(r - 1):
						if search:
							finl = keyframes[b].search(r - 1)
						else:
							finl = keyframes[b][r - 1].final()
						if key in finl:
							ret = finl[key]
							if store_hint:
								shallowest[args] = ret
							return ret
						else:
							return NotInKeyframeError(
								f"No value for {entikey} at {b, r, t}"
							)
				elif b in keyframes:
					kfb = keyframes[b]
					if r in kfb:
						kfbr = kfb[r]
						if kfbr.rev_gettable(t):
							kf = kfbr[t]
							if key in kf:
								ret = kf[key]
								if store_hint:
									shallowest[args] = ret
								return ret
							else:
								return NotInKeyframeError(
									f"No value for {entikey} at {b, r, t}"
								)
					if kfb.rev_gettable(r - 1):
						kfbr = kfb[r]
						kf = kfbr.final()
						if key in kf:
							ret = kf[key]
							if store_hint:
								shallowest[args] = ret
							return ret
						else:
							return NotInKeyframeError(
								f"No value for {entikey} at {b, r, t}"
							)
		else:
			for b, r, t in self.db._iter_parent_btt(branch, turn, tick):
				if b in keyframes:
					kfb = keyframes[b]
					if r in kfb:
						if search:
							kfbr = kfb.search(r)
						else:
							kfbr = kfb[r]
						if kfbr.rev_gettable(t):
							kf = kfbr[t]
							if key in kf:
								ret = kf[key]
								if store_hint:
									shallowest[args] = ret
								return ret
							else:
								return NotInKeyframeError(
									f"No value for {entikey} at {b, r, t}"
								)
					if kfb.rev_gettable(r - 1):
						kfbr = kfb[r]
						kf = kfbr.final()
						if key in kf:
							ret = kf[key]
							if store_hint:
								shallowest[args] = ret
							return ret
						else:
							return NotInKeyframeError(
								f"No value for {entikey} at {b, r, t}"
							)
		return KeyError("No value, ever")

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

	def iter_entities_or_keys(self, *args, forward: bool = None):
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
			yield from self._get_adds_dels(entity, branch, turn, tick)[0]
			return
		yield from self._get_keycache(
			entity, branch, turn, tick, forward=forward
		)

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
		contra=True,
	):
		if not ex:
			ex = None
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
		ex: Optional[bool]
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

	def __init__(self, db):
		def gettest(k):
			assert len(k) == 3, "Bad key: " + repr(k)

		def settest(k, v):
			assert len(k) == 3, "Bad key: {}, to be set to {}".format(k, v)

		Cache.__init__(
			self, db, kfkvs={"gettest": gettest, "settest": settest}
		)
		self.destcache = PickyDefaultDict(SettingsTurnDict)
		self.origcache = PickyDefaultDict(SettingsTurnDict)
		self.predecessors = StructuredDefaultDict(3, TurnDict)
		self._origcache_lru = OrderedDict()
		self._destcache_lru = OrderedDict()
		self._get_destcache_stuff: Tuple[
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
		self._get_origcache_stuff: Tuple[
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

	def _get_keyframe(
		self, graph_ent: tuple, branch: str, turn: int, tick: int, copy=True
	):
		if len(graph_ent) == 3:
			return super()._get_keyframe(graph_ent, branch, turn, tick, copy)
		ret = {}
		for graph, orig, dest in self.keyframe:
			if (graph, orig) == graph_ent:
				ret[dest] = super()._get_keyframe(
					(graph, orig, dest), branch, turn, tick, copy
				)
		return ret

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
		stoptime: Tuple[str, int, int] = None,
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
		its = list(kf.items())
		for ks, v in its:
			assert len(ks) == 3, "Bad key in keyframe: " + repr(ks)
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
		for ks in kf.keys():
			assert len(ks) == 3, "BBadd key in keyframe: " + repr(ks)
		return added, deleted

	def _adds_dels_predecessors(
		self,
		parentity: tuple,
		branch: str,
		turn: int,
		tick: int,
		*,
		stoptime: Tuple[str, int, int] = None,
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
			for (grap, orig, dst), kfg in kf.items():  # too much iteration!
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
		planning=None,
		forward=None,
		loading=False,
		contra=True,
	):
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
		planning=None,
		forward=None,
		loading=False,
		contra=True,
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

	def get_keyframe(self, branch, turn, tick):
		return super().get_keyframe(None, branch, turn, tick)

	def set_keyframe(self, branch, turn, tick, keyframe):
		super().set_keyframe(None, branch, turn, tick, keyframe)

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
