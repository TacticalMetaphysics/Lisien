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
"""Wrap a LiSE engine so you can access and control it using only
ordinary method calls.

"""
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL
from operator import itemgetter
from re import match
from collections import defaultdict
from functools import partial, wraps
from importlib import import_module
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from multiprocessing import cpu_count
from typing import Dict, Tuple, Set, Callable, Union, Any, Hashable, List, Iterable, Optional

import msgpack

from .allegedb import OutOfTimelineError
from .engine import Engine
from .node import Node
from .portal import Portal
from .util import MsgpackExtensionType, AbstractCharacter

EMPTY_DELTA = ({}, {})

SlightlyPackedDeltaType = Dict[bytes, Dict[bytes, Union[bytes, Dict[
	bytes, Union[bytes, Dict[bytes, Union[bytes, Dict[bytes, bytes]]]]]]]]
FormerAndCurrentType = Tuple[Dict[bytes, bytes], Dict[bytes, bytes]]

TRUE: bytes = msgpack.packb(True)
FALSE: bytes = msgpack.packb(False)
NONE: bytes = msgpack.packb(None)
NODES: bytes = msgpack.packb('nodes')
EDGES: bytes = msgpack.packb('edges')
UNITS: bytes = msgpack.packb('units')
RULEBOOK: bytes = msgpack.packb('rulebook')
RULEBOOKS: bytes = msgpack.packb('rulebooks')
NODE_VAL: bytes = msgpack.packb('node_val')
EDGE_VAL: bytes = msgpack.packb('edge_val')
ETERNAL: bytes = msgpack.packb('eternal')
UNIVERSAL: bytes = msgpack.packb('universal')
STRINGS: bytes = msgpack.packb('strings')
RULES: bytes = msgpack.packb('rules')
LOCATION: bytes = msgpack.packb('location')
BRANCH: bytes = msgpack.packb('branch')


def _dict_delta_added(old: Dict[bytes, bytes], new: Dict[bytes, bytes],
						former: Dict[bytes, bytes], current: Dict[bytes,
																	bytes]):
	for k in new.keys() - old.keys():
		former[k] = NONE
		current[k] = new[k]


def _dict_delta_removed(old: Dict[bytes, bytes], new: Dict[bytes, bytes],
						former: Dict[bytes, bytes], current: Dict[bytes,
																	bytes]):
	for k in old.keys() - new.keys():
		former[k] = old[k]
		current[k] = NONE


def _packed_dict_delta_fallback(
		old: Dict[bytes, bytes], new: Dict[bytes,
											bytes]) -> FormerAndCurrentType:
	pre = {}
	post = {}
	added_thread = Thread(target=_dict_delta_added, args=(old, new, pre, post))
	removed_thread = Thread(target=_dict_delta_removed,
							args=(old, new, pre, post))
	added_thread.start()
	removed_thread.start()
	ks = old.keys() & new.keys()
	oldv_l = []
	newv_l = []
	k_l = []
	for k in ks:
		oldv_l.append(old[k])
		newv_l.append(new[k])
		k_l.append(k)
	for k, oldv, newv in zip(k_l, oldv_l, newv_l):
		if oldv != newv:
			pre[k] = oldv
			post[k] = newv
	added_thread.join()
	removed_thread.join()
	return pre, post


def _packed_dict_delta(old: Dict[bytes, bytes],
						new: Dict[bytes, bytes]) -> FormerAndCurrentType:
	"""Describe changes from one shallow dictionary of msgpack data to another

	"""
	post = {}
	pre = {}
	added_thread = Thread(target=_dict_delta_added, args=(old, new, pre, post))
	removed_thread = Thread(target=_dict_delta_removed,
							args=(old, new, pre, post))
	added_thread.start()
	removed_thread.start()
	ks = old.keys() & new.keys()
	oldv_l = []
	newv_l = []
	k_l = []
	for k in ks:
		# \xc1 is unused by msgpack.
		# I append it here so that the last byte will never be \x00,
		# which numpy uses as a terminator, but msgpack uses for
		# the integer zero.
		oldv_l.append(old[k] + b'\xc1')
		newv_l.append(new[k] + b'\xc1')
		k_l.append(k + b'\xc1')
	oldvs = np.array(oldv_l)
	newvs = np.array(newv_l)
	ks = np.array(k_l)
	changes = oldvs != newvs
	added_thread.join()
	removed_thread.join()
	if changes.any():
		changed_keys = ks[changes]
		changed_values = newvs[changes]
		former_values = oldvs[changes]
		for k, former, current in zip(changed_keys, former_values,
										changed_values):
			k = k[:-1]
			pre[k] = former[:-1]
			post[k] = current[:-1]
	return pre, post


try:
	import numpy as np
	packed_dict_delta = _packed_dict_delta
except ImportError:
	packed_dict_delta = _packed_dict_delta_fallback
	packed_dict_delta.__doc__ = _packed_dict_delta.__doc__


def _set_delta_added(old: Set[bytes], new: Set[bytes],
						former: Dict[bytes, bytes], current: Dict[bytes,
																	bytes]):
	for item in new - old:
		former[item] = FALSE
		current[item] = TRUE


def set_delta(old: Set[bytes], new: Set[bytes]) -> FormerAndCurrentType:
	"""Describes changes from one set of msgpack-packed values to another"""
	former = {}
	current = {}
	added_thread = Thread(target=_set_delta_added,
							args=(old, new, former, current))
	added_thread.start()
	for item in old - new:
		former[item] = TRUE
		current[item] = FALSE
	added_thread.join()
	return former, current


def concat_d(r: Dict[bytes, bytes]) -> bytes:
	"""Pack a dictionary of msgpack-encoded keys and values into msgpack bytes"""
	resp = msgpack.Packer().pack_map_header(len(r))
	for k, v in r.items():
		resp += k + v
	return resp


SET_CODE = MsgpackExtensionType.set.value.to_bytes(1, "big", signed=False)


def concat_s(s: Set[bytes]) -> bytes:
	"""Pack a set of msgpack-encoded values into a msgpack array with ext code"""
	resp = msgpack.Packer().pack_array_header(len(s))
	for v in s:
		resp += v
	data_len = len(resp)
	if data_len == 1:
		return b"\xd4" + SET_CODE + resp
	elif data_len == 2:
		return b"\xd5" + SET_CODE + resp
	elif data_len == 4:
		return b"\xd6" + SET_CODE + resp
	elif data_len == 8:
		return b"\xd7" + SET_CODE + resp
	elif data_len == 16:
		return b"\xd8" + SET_CODE + resp
	elif data_len < 2**8:
		return b"\xc7" + data_len.to_bytes(1, "big",
											signed=False) + SET_CODE + resp
	elif data_len < 2**16:
		return b"\xc8" + data_len.to_bytes(2, "big",
											signed=False) + SET_CODE + resp
	elif data_len < 2**32:
		return b"\xc9" + data_len.to_bytes(4, "big",
											signed=False) + SET_CODE + resp
	else:
		raise ValueError("Too long")


def prepacked(fun: Callable) -> Callable:
	fun.prepacked = True
	return fun


class EngineHandle(object):
	"""A wrapper for a :class:`LiSE.Engine` object that runs in the same
	process, but with an API built to be used in a command-processing
	loop that takes commands from another process.

	It's probably a bad idea to use this class unless you're
	developing your own API.

	"""
	_after_ret: Callable

	def __init__(self, args=(), kwargs=None, logq=None, loglevel=None):
		"""Instantiate an engine with the positional arguments ``args`` and
		the keyword arguments ``kwargs``.

		``logq`` is a :class:`Queue` into which I'll put tuples of
		``(loglevel, message)``. ``loglevel`` is one of
		`'debug'`, `'info'`, `'warning'`, `'error'`, or `'critical'`
		(or the constants from the `logging` module, or an integer)
		and controls what messages will be logged.

		"""
		if kwargs is None:
			kwargs = {}
		kwargs.setdefault('logfun', self.log)
		self._logq = logq
		self._loglevel = loglevel
		self._real = Engine(*args, cache_arranger=False, **kwargs)
		self.pack = pack = self._real.pack

		def pack_pair(pair):
			k, v = pair
			return pack(k), pack(v)

		self.pack_pair = pack_pair
		self.unpack = unpack = self._real.unpack

		def unpack_pair(pair):
			k, v = pair
			return unpack(k), unpack(v)

		def unpack_dict(d: dict):
			return dict(map(unpack_pair, d.items()))

		self.unpack_dict = unpack_dict
		self._muted_chars = set()
		# It's possible that the memoization I've done could break if this
		# handle observes the world while it's in a planning context. This
		# shouldn't happen, I think
		self._node_stat_copy_memo: Dict[tuple, Dict[bytes, bytes]] = {}
		self._portal_stat_copy_memo: Dict[tuple, Dict[bytes, bytes]] = {}
		self._char_stat_copy_memo: Dict[tuple, Dict[bytes, bytes]] = {}
		self._char_units_copy_memo: Dict[tuple, Dict[bytes, Set[bytes]]] = {}
		self._char_rulebooks_copy_memo: Dict[tuple, Dict[bytes, bytes]] = {}
		self._char_nodes_rulebooks_copy_memo: Dict[tuple, Dict[bytes,
																bytes]] = {}
		self._char_portals_rulebooks_copy_memo: Dict[Hashable, Dict[Tuple[
			str, int, int], Dict[bytes, Dict[bytes, bytes]]]] = {}
		self._char_nodes_copy_memo: Dict[tuple, Set[bytes]] = {}
		self._char_portals_copy_memo: Dict[tuple, Set[bytes]] = {}
		self._node_successors_copy_memo: Dict[tuple, Set[bytes]] = {}
		self._strings_cache = {}
		self._strings_copy_memo = {}
		self._eternal_cache: Dict[bytes, bytes] = {}
		self._eternal_copy_memo = {}
		self._universal_copy_memo = {}
		self._universal_delta_memo: Dict[Tuple[str, int, int], Dict[
			Tuple[str, int, int],
			Dict[bytes, bytes]]] = defaultdict(lambda: defaultdict(dict))
		self._rule_cache = defaultdict(dict)
		self._rule_copy_memo = {}
		self._rulebook_copy_memo = {}
		self._character_delta_memo: Dict[Hashable, Dict[
			Tuple[str, int, int],
			Dict[Tuple[str, int, int],
					SlightlyPackedDeltaType]]] = defaultdict(
						lambda: defaultdict(lambda: defaultdict(dict)))
		self._real.arrange_cache_signal.disconnect(
			self._real._arrange_caches_at_time)
		self._real.arrange_cache_signal.connect(self._precopy_at_time)
		self.threadpool = ThreadPoolExecutor(cpu_count())
		self._cache_arranger_started = False

	@property
	def branch(self):
		return self._real.branch

	@property
	def turn(self):
		return self._real.turn

	@property
	def tick(self):
		return self._real.tick

	def _precopy_at_time(self, _, *, branch: str, turn: int,
							tick: int) -> None:
		lock = self._real.world_lock
		btt = (branch, turn, tick)
		with lock:
			chars = list(self._real.character)
		for char in chars:
			with lock:
				self.character_stat_copy(char, btt=btt)
			with lock:
				self.character_nodes(char, btt=btt)
			with lock:
				self._character_nodes_stat_copy(char, btt=btt)
			with lock:
				self.character_portals(char, btt=btt)
			with lock:
				self._character_portals_stat_copy(char, btt=btt)
			with lock:
				self._character_units_copy(char, btt=btt)

	def log(self, level: Union[str, int], message: str) -> None:
		if isinstance(level, str):
			level = {
				'debug': 10,
				'info': 20,
				'warning': 30,
				'error': 40,
				'critical': 50
			}[level.lower()]
		if self._logq and level >= self._loglevel:
			self._logq.put((level, message))

	def debug(self, message: str) -> None:
		self.log(DEBUG, message)

	def info(self, message: str) -> None:
		self.log(INFO, message)

	def warning(self, message: str) -> None:
		self.log(WARNING, message)

	def error(self, message: str) -> None:
		self.log(ERROR, message)

	def critical(self, message: str) -> None:
		self.log(CRITICAL, message)

	def time_locked(self) -> bool:
		"""Return whether the sim-time has been prevented from advancing"""
		return hasattr(self._real, 'locktime')

	def advance(self) -> None:
		"""Run one rule"""
		self._real._advance()

	def _get_char_deltas(
			self,
			chars: Union[str, Iterable[Hashable]],
			*,
			btt_from: Tuple[str, int, int] = None,
			btt_to: Tuple[str, int, int] = None) -> SlightlyPackedDeltaType:
		"""Return a dict describing changes to characters since last call"""
		pack = self._real.pack
		ret: Dict[bytes, Dict[bytes, bytes]] = {}
		if chars == 'all':
			it = iter(self._real.character.keys())
		else:
			it = iter(chars)
		for char in it:
			delt = self._character_delta(char,
											btt_from=btt_from,
											btt_to=btt_to)
			if delt:
				ret[pack(char)] = delt
		return ret

	@prepacked
	def copy_character(self,
						char: Hashable,
						*,
						btt: Tuple[str, int,
									int] = None) -> Dict[bytes, bytes]:
		"""Return a mapping describing a character

		It has the keys 'nodes', 'edges', 'units', 'rulebooks', 'node_val',
		'edge_val', and whatever stats the character has.

		"""
		units = self._character_units_copy(char, btt=btt)
		ports = self._character_portals_stat_copy(char, btt=btt)
		ported = {}
		for orig, dests in ports.items():
			dest_stats = {}
			for dest, stats in dests.items():
				dest_stats[dest] = concat_d(stats)
			ported[orig] = concat_d(dest_stats)
		return {
			NODES:
			concat_d(
				{node: TRUE
					for node in self.character_nodes(char, btt=btt)}),
			EDGES:
			concat_d({
				origdest: TRUE
				for origdest in self.character_portals(char, btt=btt)
			}),
			UNITS:
			concat_d({k: concat_s(v)
						for (k, v) in units.items()}),
			RULEBOOKS:
			concat_d(self.character_rulebooks_copy(char, btt=btt)),
			NODE_VAL:
			concat_d(self._character_nodes_stat_copy(char, btt=btt)),
			EDGE_VAL:
			concat_d(ported),
			**self.character_stat_copy(char)
		}

	def get_keyframe(self, branch, turn, tick):
		now = self._real._btt()
		self._real._set_btt(branch, turn, tick)
		self._real.snap_keyframe()
		self._real._set_btt(*now)
		return self._real._get_kf(branch, turn, tick)

	def copy_chars(self, chars: Union[str, Iterable[Hashable]]):
		"""Return a mapping describing several characters

		See the `copy_character` method for details on the format of
		submappings.

		Special value 'all' gets mappings for every character that exists.

		"""
		if chars == 'all':
			it = iter(self._real.character.keys())
		else:
			it = iter(chars)
		self._real.snap_keyframe()
		kf = self._real._get_kf(*self._get_btt())
		ret = {}
		for char in it:
			chara = self._real.character[char]
			char_d = ret[char] = {
				'units': self._character_units_copy(char),
				'rulebooks': {
					'character': chara.rulebook.name,
					'unit': chara.unit.rulebook.name,
					'thing': chara.thing.rulebook.name,
					'place': chara.place.rulebook.name,
					'portal': chara.portal.rulebook.name
				},
			}
			if (char, ) in kf['graph_val']:
				char_d.update(kf['graph_val'][char, ])
			if (char, ) in kf['nodes']:
				char_d['nodes'] = {
					node: ex
					for (node, ex) in kf['nodes'][char, ].items() if ex
				}
		for (char, node), val in kf['node_val'].items():
			ret.setdefault(char, {}).setdefault('node_val', {})[node] = val
		for (char, orig, dest), ex in kf['edges'].items():
			ret.setdefault(char, {}).setdefault('edges', {})[orig,
																dest] = ex[0]
		for (char, orig, dest, idx), mapp in kf['edge_val'].items():
			for key, value in mapp.items():
				ret.setdefault(char, {}).setdefault('edge_val', {}).setdefault(
					orig, {}).setdefault(dest, {})[key] = value
		return ret

	@prepacked
	def get_char_deltas(
			self,
			chars: Union[str, Iterable[Hashable]],
			*,
			btt_from: Tuple[str, int, int] = None,
			btt_to: Tuple[str, int, int] = None) -> Dict[bytes, bytes]:
		"""Return mappings describing changes to characters

		By default, this will get the difference between the state previously
		time travelled to with the `time_travel` method and that at the current
		moment. Supply `btt_from` and `btt_to` to specify other times, in the
		format `(branch, turn, tick)`."""
		delt = self._get_char_deltas(chars, btt_from=btt_from, btt_to=btt_to)
		ret = {}
		for char, delta in delt.items():
			charret = ret[char] = {}
			if NODES in delta:
				charret[NODES] = concat_d(delta[NODES])
			if EDGES in delta:
				charret[EDGES] = concat_d(delta[EDGES])
			if UNITS in delta:
				graph_units = {}
				for graph, unitss in delta[UNITS].items():
					graph_units[graph] = concat_d(unitss)
				charret[UNITS] = concat_d(graph_units)
			if RULEBOOKS in delta:
				charret[RULEBOOKS] = concat_d(delta[RULEBOOKS])
			if NODE_VAL in delta:
				charret[NODE_VAL] = concat_d({
					node: concat_d(vals)
					for node, vals in delta[NODE_VAL].items()
				})
		if not self._cache_arranger_started:
			self._real._start_cache_arranger()
			self._cache_arranger_started = True
		return {char: concat_d(charret) for char, charret in ret.items()}

	def _pack_delta(self, delta):
		pack = self.pack
		slightly_packed_delta = {}
		mostly_packed_delta = {}
		for char, chardelta in delta.items():
			chardelta = chardelta.copy()
			pchar = pack(char)
			chard = slightly_packed_delta[pchar] = {}
			packd = mostly_packed_delta[pchar] = {}
			if 'nodes' in chardelta:
				nd = chard[NODES] = {
					pack(node): pack(ex)
					for node, ex in chardelta.pop('nodes').items()
				}
				packd[NODES] = concat_d(nd)
			if 'node_val' in chardelta:
				slightnoded = chard[NODE_VAL] = {}
				packnodevd = {}
				for node, vals in chardelta.pop('node_val').items():
					pnode = pack(node)
					pvals = dict(map(self.pack_pair, vals.items()))
					slightnoded[pnode] = pvals
					packnodevd[pnode] = concat_d(pvals)
				packd[NODE_VAL] = concat_d(packnodevd)
			if 'edges' in chardelta:
				ed = chard[EDGES] = {
					pack(origdest): pack(ex)
					for origdest, ex in chardelta.pop('edges')
				}
				packd[EDGES] = concat_d(ed)
			if 'edge_val' in chardelta:
				slightorigd = chard[EDGE_VAL] = {}
				packorigd = {}
				for orig, dests in chardelta.pop('edge_val').items():
					porig = pack(orig)
					slightdestd = slightorigd[porig] = {}
					packdestd = {}
					for dest, port in dests.items():
						pdest = pack(dest)
						slightportd = slightdestd[pdest] = dict(
							map(self.pack_pair, port.items()))
						packdestd[pdest] = concat_d(slightportd)
					packorigd[porig] = concat_d(packdestd)
				packd[EDGE_VAL] = concat_d(packorigd)
			if 'units' in chardelta:
				slightgraphd = chard[UNITS] = {}
				packunitd = {}
				for graph, unitss in chardelta.pop('units').items():
					pgraph = pack(graph)
					slightunitd = slightgraphd[pgraph] = dict(
						map(self.pack_pair, unitss.items()))
					packunitd[pgraph] = concat_d(slightunitd)
				packd[UNITS] = concat_d(packunitd)
			if 'rulebooks' in chardelta:
				chard[RULEBOOKS] = slightrbd = dict(
					map(self.pack_pair,
						chardelta.pop('rulebooks').items()))
				packd[RULEBOOKS] = concat_d(slightrbd)
			todo = dict(map(self.pack_pair, chardelta.items()))
			chard.update(todo)
			packd.update(todo)
		return slightly_packed_delta, concat_d({
			charn: concat_d(stuff)
			for charn, stuff in mostly_packed_delta.items()
		})

	@staticmethod
	def _concat_char_delta(delta: SlightlyPackedDeltaType) -> bytes:
		delta = delta.copy()
		mostly_packed_delta = {}
		eternal = delta.pop(ETERNAL, None)
		if eternal:
			mostly_packed_delta[ETERNAL] = eternal
		universal = delta.pop(UNIVERSAL, None)
		if universal:
			mostly_packed_delta[UNIVERSAL] = universal
		for char, chardelta in delta.items():
			chardelta = chardelta.copy()
			packd = mostly_packed_delta[char] = {}
			if NODES in chardelta:
				charnodes = chardelta.pop(NODES)
				packd[NODES] = concat_d(charnodes)
			if NODE_VAL in chardelta:
				slightnoded = {}
				packnodevd = {}
				for node, vals in chardelta.pop(NODE_VAL).items():
					slightnoded[node] = vals
					packnodevd[node] = concat_d(vals)
				packd[NODE_VAL] = concat_d(packnodevd)
			if EDGES in chardelta:
				es = chardelta.pop(EDGES)
				packd[EDGES] = concat_d(es)
			if EDGE_VAL in chardelta:
				packorigd = {}
				for orig, dests in chardelta.pop(EDGE_VAL).items():
					slightdestd = {}
					packdestd = {}
					for dest, port in dests.items():
						slightdestd[dest] = port
						packdestd[dest] = concat_d(port)
					packorigd[orig] = concat_d(packdestd)
				packd[EDGE_VAL] = concat_d(packorigd)
			if UNITS in chardelta:
				packunitd = {}
				for graph, unitss in chardelta[UNITS].items():
					packunitd[graph] = concat_d(unitss)
				packd[UNITS] = concat_d(packunitd)
			if RULEBOOKS in chardelta:
				packd[RULEBOOKS] = concat_d(chardelta[RULEBOOKS])
			packd.update(chardelta)
		almost_entirely_packed_delta = {
			charn: concat_d(stuff)
			for charn, stuff in mostly_packed_delta.items()
		}
		rulebooks = delta.pop(RULEBOOKS, None)
		if rulebooks:
			almost_entirely_packed_delta[RULEBOOKS] = rulebooks
		rules = delta.pop(RULES, None)
		if rules:
			almost_entirely_packed_delta[RULEBOOKS] = rules
		return concat_d(almost_entirely_packed_delta)

	@prepacked
	def next_turn(self) -> Tuple[bytes, bytes]:
		"""Simulate a turn. Return whatever result, as well as a delta"""
		pack = self.pack
		self.debug(
			'calling next_turn at {}, {}, {}'.format(*self._real._btt()))
		ret, delta = self._real.next_turn()
		slightly_packed_delta, packed_delta = self._pack_delta(delta)
		self.debug(
			'got results from next_turn at {}, {}, {}. Packing...'.format(
				*self._real._btt()))
		return pack(ret), packed_delta

	def _get_slow_delta(
			self,
			btt_from: Tuple[str, int, int] = None,
			btt_to: Tuple[str, int, int] = None) -> SlightlyPackedDeltaType:
		pack = self._real.pack
		delta: Dict[bytes, Any] = {}
		btt_from = self._get_btt(btt_from)
		btt_to = self._get_btt(btt_to)
		if btt_from == btt_to:
			return delta
		now = self._real._btt()
		self._real._set_btt(*btt_from)
		self._real.snap_keyframe()
		kf_from = self._real._get_kf(*btt_from)
		self._real._set_btt(*btt_to)
		self._real.snap_keyframe()
		kf_to = self._real._get_kf(*btt_to)
		self._real._set_btt(*now)
		keys = []
		values_from = []
		values_to = []
		# assemble the whole world state into three big arrays,
		# use numpy to compare keys and values,
		# then turn it all back into a dictionary of the right form
		for graph in kf_from['graph_val'].keys() | kf_to['graph_val'].keys():
			a = kf_from['graph_val'].get(graph, {})
			b = kf_to['graph_val'].get(graph, {})
			for k in a.keys() | b.keys():
				keys.append(('graph', graph[0], k))
				values_from.append(pack(a.get(k)) + b'\xc0')
				values_to.append(pack(b.get(k)) + b'\xc0')
		for graph, node in kf_from['node_val'].keys() | kf_to['node_val'].keys(
		):
			a = kf_from['node_val'].get((graph, node), {})
			b = kf_to['node_val'].get((graph, node), {})
			for k in a.keys() | b.keys():
				keys.append(('node', graph, node, k))
				values_from.append(pack(a.get(k)) + b'\xc0')
				values_to.append(pack(b.get(k)) + b'\xc0')
		for graph, orig, dest, i in kf_from['edge_val'].keys(
		) | kf_to['edge_val'].keys():
			a = kf_from['edge_val'].get((graph, orig, dest, i), {})
			b = kf_to['edge_val'].get((graph, orig, dest, i), {})
			for k in a.keys() | b.keys():
				keys.append(('edge', graph, orig, dest, k))
				values_from.append(pack(a.get(k)) + b'\xc0')
				values_to.append(pack(b.get(k)) + b'\xc0')
		values_changed = np.array(values_from) != np.array(values_to)
		delta = {}
		for k, v, _ in filter(itemgetter(2),
								zip(keys, values_to, values_changed)):
			v = v[:-1]
			if k[0] == 'node':
				_, graph, node, key = k
				graph, node, key = map(pack, (graph, node, key))
				if graph not in delta:
					delta[graph] = {NODE_VAL: {node: {key: v}}}
				elif node not in delta[graph][NODE_VAL]:
					delta[graph][NODE_VAL][node] = {key: v}
				else:
					delta[graph][NODE_VAL][node][key] = v
			elif k[0] == 'edge':
				_, graph, orig, dest, key = k
				graph, orig, dest, key = map(pack, (graph, orig, dest, key))
				if graph not in delta:
					delta[graph] = {EDGE_VAL: {orig: {dest: {key: v}}}}
				elif orig not in delta[graph][EDGE_VAL]:
					delta[graph][EDGE_VAL][orig] = {dest: {key: v}}
				elif dest not in delta[graph][EDGE_VAL][orig]:
					delta[graph][EDGE_VAL][orig][dest] = {key: v}
				else:
					delta[graph][EDGE_VAL][orig][dest][key] = v
			else:
				assert k[0] == 'graph'
				_, graph, key = k
				graph, key = map(pack, (graph, key))
				if graph in delta:
					delta[graph][key] = v
				else:
					delta[graph] = {key: v}
		for graph in kf_from['nodes'].keys() & kf_to['nodes'].keys():
			for node in kf_from['nodes'][graph].keys(
			) - kf_to['nodes'][graph].keys():
				grap, node = map(pack, (graph[0], node))
				if grap not in delta:
					delta[grap] = {NODES: {node: FALSE}}
				elif NODES not in delta[grap]:
					delta[grap][NODES] = {node: FALSE}
				else:
					delta[grap][NODES][node] = FALSE
			for node in kf_to['nodes'][graph].keys(
			) - kf_from['nodes'][graph].keys():
				grap, node = map(pack, (graph[0], node))
				if grap not in delta:
					delta[grap] = {NODES: {node: TRUE}}
				elif NODE_VAL not in delta[grap]:
					delta[grap][NODES] = {node: TRUE}
				else:
					delta[grap][NODES][node] = TRUE
		for graph, orig, dest in kf_from['edges'].keys() - kf_to['edges'].keys(
		):
			graph = pack(graph)
			if graph not in delta:
				delta[graph] = {EDGES: {pack((orig, dest)): FALSE}}
			elif EDGES not in delta[graph]:
				delta[graph][EDGES] = {pack((orig, dest)): FALSE}
			else:
				delta[graph][EDGES][pack((orig, dest))] = FALSE
		for graph, orig, dest in kf_to['edges'].keys() - kf_from['edges'].keys(
		):
			graph = pack(graph)
			if graph not in delta:
				delta[graph] = {EDGES: {pack((orig, dest)): TRUE}}
			elif EDGES not in delta[graph]:
				delta[graph][EDGES] = {pack((orig, dest)): TRUE}
			else:
				delta[graph][EDGES][pack((orig, dest))] = TRUE
		unid = self.universal_delta(btt_from=btt_from, btt_to=btt_to)
		if unid:
			delta[UNIVERSAL] = unid
		rud = self.all_rules_delta(btt_from=btt_from, btt_to=btt_to)
		if rud:
			delta[RULES] = {rule: pack(stuff) for rule, stuff in rud.items()}
		rbd = self.all_rulebooks_delta(btt_from=btt_from, btt_to=btt_to)
		if rbd:
			delta[RULEBOOKS] = dict(map(self.pack_pair, rbd.items()))
		return delta

	@prepacked
	def time_travel(
		self,
		branch,
		turn,
		tick=None,
	) -> Tuple[bytes, bytes]:
		"""Go to a different `(branch, turn, tick)` and return a delta

		For compatibility with `next_turn` this actually returns a tuple,
		the 0th item of which is `None`.

		"""
		if branch in self._real._branches:
			parent, turn_from, tick_from, turn_to, tick_to = self._real._branches[
				branch]
			if tick is None:
				if turn < turn_from or (self._real._enforce_end_of_time
										and turn > turn_to):
					raise OutOfTimelineError("Out of bounds",
												*self._real._btt(), branch,
												turn, tick)
			else:
				if (turn, tick) < (turn_from, tick_from) or (
					self._real._enforce_end_of_time and (turn, tick) >
					(turn_to, tick_to)):
					raise OutOfTimelineError("Out of bounds",
												*self._real._btt(), branch,
												turn, tick)
		branch_from, turn_from, tick_from = self._real._btt()
		slow_delta = branch != branch_from
		self._real.time = (branch, turn)
		if tick is None:
			tick = self._real.tick
		else:
			self._real.tick = tick
		if slow_delta:
			delta = self._get_slow_delta(btt_from=(branch_from, turn_from,
													tick_from),
											btt_to=(branch, turn, tick))
			packed_delta = self._concat_char_delta(delta)
		else:
			delta = self._real.get_delta(branch, turn_from, tick_from, turn,
											tick)
			slightly_packed_delta, packed_delta = self._pack_delta(delta)
		return NONE, packed_delta

	@prepacked
	def increment_branch(self) -> bytes:
		"""Generate a new branch name and switch to it

		Returns the name of the new branch.

		"""
		branch = self._real.branch
		m = match(r'(.*)(\d+)', branch)
		if m:
			stem, n = m.groups()
			branch = stem + str(int(n) + 1)
		else:
			stem = branch
			n = 1
			branch = stem + str(n)
		if branch in self._real._branches:
			if m:
				n = int(n)
			else:
				stem = branch[:-1]
				n = 1
			while stem + str(n) in self._real._branches:
				n += 1
			branch = stem + str(n)
		self._real.branch = branch
		return self.pack(branch)

	def add_character(self, char: Hashable, data: dict, attr: dict):
		"""Make a new character, initialized with whatever data"""
		# Probably not great that I am unpacking and then repacking the stats
		character = self._real.new_character(char, **attr)
		branch, turn, tick = self._get_btt()
		self._char_stat_copy_memo[char, branch, turn, tick] = dict(
			map(self.pack_pair, attr.items()))
		placedata = data.get('place', data.get('node', {}))
		node_stat_memo = self._node_stat_copy_memo
		for place, stats in placedata.items():
			character.add_place(place, **stats)
			branch, turn, tick = self._get_btt()
			node_stat_memo[place, branch, turn,
							tick] = dict(map(self.pack_pair, stats.items()))
		thingdata = data.get('thing', {})
		for thing, stats in thingdata.items():
			character.add_thing(thing, **stats)
			node_stat_memo[thing, branch, turn,
							tick] = dict(map(self.pack_pair, stats.items()))
		portdata = data.get('edge', data.get('portal', data.get('adj', {})))
		port_stat_memo = self._portal_stat_copy_memo
		for orig, dests in portdata.items():
			for dest, stats in dests.items():
				character.add_portal(orig, dest, **stats)
				branch, turn, tick = self._get_btt()
				port_stat_memo[char, orig, dest, branch, turn,
								tick] = dict(map(self.pack_pair,
													stats.items()))

	def commit(self):
		self._real.commit()

	def close(self):
		self._real.close()
		self.threadpool.shutdown()

	def get_time(self):
		return self._real.time

	def get_btt(self):
		return self._real._btt()

	def get_language(self):
		return str(self._real.string.language)

	def set_language(self, lang):
		self._real.string.language = lang
		return self.strings_copy(lang)

	def get_string_ids(self):
		return list(self._real.string)

	def get_string_lang_items(self, lang):
		return list(self._real.string.lang_items(lang))

	def strings_copy(self, lang=None):
		if lang is None:
			lang = self._real.string.language
		return dict(self._real.string.lang_items(lang))

	def get_string(self, k):
		return self._real.string[k]

	def have_string(self, k):
		return k in self._real.string

	def set_string(self, k, v):
		self._real.string[k] = v

	def del_string(self, k):
		del self._real.string[k]

	@prepacked
	def get_eternal(self, k):
		return self.pack(self._real.eternal[k])

	def set_eternal(self, k, v):
		self._real.eternal[k] = v
		self._eternal_cache[self.pack(k)] = self.pack(v)

	def del_eternal(self, k):
		del self._real.eternal[k]

	def have_eternal(self, k):
		return k in self._real.eternal

	@prepacked
	def eternal_copy(self):
		return dict(map(self.pack_pair, self._real.eternal.items()))

	def get_universal(self, k):
		return self._real.universal[k]

	def set_universal(self, k, v):
		self._real.universal[k] = v

	def del_universal(self, k):
		del self._real.universal[k]

	@prepacked
	def universal_copy(self, btt: Tuple[str, int, int] = None):
		btt = self._get_btt(btt)
		if btt in self._universal_copy_memo:
			return self._universal_copy_memo[btt]
		ret = self._universal_copy_memo[btt] = dict(
			map(self.pack_pair, self._real.universal.items()))
		return ret

	@prepacked
	def universal_delta(
			self,
			*,
			btt_from: Tuple[str, int, int] = None,
			btt_to: Tuple[str, int, int] = None) -> Dict[bytes, bytes]:
		memo = self._universal_delta_memo
		if btt_from is None:
			btt_from = self.get_btt()
		btt_to = self._get_btt(btt_to)
		if btt_from in memo and btt_to in memo[btt_from]:
			return memo[btt_from][btt_to]
		old = self.universal_copy(btt=btt_from)
		new = self._universal_copy_memo[btt_to] = self.universal_copy(
			btt=btt_to)
		former, current = packed_dict_delta(old, new)
		memo[btt_from][btt_to] = current
		memo[btt_to][btt_from] = former
		return current

	def init_character(self, char, statdict: dict = None):
		if char in self._real.character:
			raise KeyError("Already have character {}".format(char))
		if statdict is None:
			statdict = {}
		self._real.character[char] = {}
		self._real.character[char].stat.update(statdict)
		self.character_stat_copy(char)

	def del_character(self, char):
		del self._real.character[char]

	@prepacked
	def character_stat_copy(self, char, *, btt: Tuple[str, int, int] = None):
		if char not in self._real.character:
			raise KeyError("no such character")
		pack = self.pack
		key = self._get_btt(btt) + (char, )
		if key in self._char_stat_copy_memo:
			return self._char_stat_copy_memo[key]
		ret = self._char_stat_copy_memo[key] = {
			pack(k): pack(v.unwrap()) if hasattr(v, 'unwrap')
			and not hasattr(v, 'no_unwrap') else pack(v)
			for (k, v) in self._real.character[char].stat.items()
		}
		return ret

	@prepacked
	def _character_something_delta(self,
									char,
									copier,
									*args,
									btt_from: Tuple[str, int, int] = None,
									btt_to: Tuple[str, int, int] = None):
		btt_from = self._get_btt(btt_from)
		btt_to = self._get_btt(btt_to)
		if btt_from == btt_to:
			return {}, {}
		old = copier(char, *args, btt=btt_from)
		new = copier(char, *args, btt=btt_to)
		return packed_dict_delta(old, new)

	@prepacked
	def character_stat_delta(self,
								char,
								*,
								btt_from: Tuple[str, int, int] = None,
								btt_to: Tuple[str, int, int] = None):
		return self._character_something_delta(char,
												self.character_stat_copy,
												btt_from=btt_from,
												btt_to=btt_to)

	def _character_units_copy(
			self,
			char,
			*,
			btt: Tuple[str, int, int] = None) -> Dict[bytes, Set[bytes]]:
		pack = self._real.pack
		key = self._get_btt(btt) + (char, )
		memo = self._char_units_copy_memo
		if key in memo:
			return memo[key]
		ret = memo[key] = {
			pack(graph): set(map(pack, nodes.keys()))
			for (graph, nodes) in self._real.character[char].unit.items()
		}
		return ret

	def _character_units_delta(
		self,
		char,
		*,
		btt_from: Tuple[str, int, int] = None,
		btt_to: Tuple[str, int, int] = None
	) -> Tuple[Dict[bytes, Dict[bytes, bytes]], Dict[bytes, Dict[bytes,
																	bytes]]]:
		btt_from = self._get_btt(btt_from)
		btt_to = self._get_btt(btt_to)
		if btt_from == btt_to:
			return {}, {}
		old = self._character_units_copy(char, btt=btt_from)
		new = defaultdict(set)
		new.update(self._character_units_copy(char, btt=btt_to))
		former: Dict[bytes, Dict[bytes, bytes]] = {}
		current: Dict[bytes, Dict[bytes, bytes]] = {}
		for graph in old.keys() - new.keys():
			current[graph] = {node: FALSE for node in old[graph]}
			former[graph] = {node: TRUE for node in old[graph]}
		for graph in new.keys() - old.keys():
			current[graph] = {node: TRUE for node in new[graph]}
			former[graph] = {node: FALSE for node in new[graph]}
		for graph in old.keys() & new.keys():
			graph_nodes_former = {}
			graph_nodes_current = {}
			for node in old[graph].difference(new[graph]):
				graph_nodes_former[node] = TRUE
				graph_nodes_current[node] = FALSE
			for node in new[graph].difference(old[graph]):
				graph_nodes_former[node] = FALSE
				graph_nodes_current[node] = TRUE
			if graph_nodes_current:
				current[graph] = graph_nodes_current
			if graph_nodes_former:
				former[graph] = graph_nodes_former
		return former, current

	@prepacked
	def character_rulebooks_copy(
			self,
			char,
			btt: Tuple[str, int, int] = None) -> Dict[bytes, bytes]:
		chara = self._real.character[char]
		key = self._get_btt(btt) + (char, )
		if key in self._char_rulebooks_copy_memo:
			return self._char_rulebooks_copy_memo[key]
		ret = self._char_rulebooks_copy_memo[key] = dict(
			map(self.pack_pair, [('character', chara.rulebook.name),
									('unit', chara.unit.rulebook.name),
									('thing', chara.thing.rulebook.name),
									('place', chara.place.rulebook.name),
									('portal', chara.portal.rulebook.name)]))
		return ret

	@prepacked
	def character_rulebooks_delta(
			self,
			char,
			*,
			btt_from: Tuple[str, int, int] = None,
			btt_to: Tuple[str, int, int] = None) -> FormerAndCurrentType:
		return self._character_something_delta(char,
												self.character_rulebooks_copy,
												btt_from=btt_from,
												btt_to=btt_to)

	@prepacked
	def character_nodes_rulebooks_copy(
			self,
			char,
			nodes='all',
			btt: Tuple[str, int, int] = None) -> Dict[bytes, bytes]:
		key = self._get_btt(btt) + (char, nodes)
		if key in self._char_nodes_rulebooks_copy_memo:
			return self._char_nodes_rulebooks_copy_memo[key]
		chara = self._real.character[char]
		if nodes == 'all':
			nodeiter = iter(chara.node.values())
		else:
			nodeiter = (chara.node[k] for k in nodes)
		pack = self.pack
		ret = self._char_nodes_rulebooks_copy_memo[key] = {
			pack(node.name): pack(node.rulebook.name)
			for node in nodeiter
		}
		return ret

	@prepacked
	def character_nodes_rulebooks_delta(
			self,
			char,
			nodes='all',
			*,
			btt_from: Tuple[str, int, int] = None,
			btt_to: Tuple[str, int, int] = None) -> FormerAndCurrentType:
		return self._character_something_delta(
			char,
			self.character_nodes_rulebooks_copy,
			nodes,
			btt_from=btt_from,
			btt_to=btt_to)

	def character_portals_rulebooks_copy(self,
											char,
											portals='all',
											btt: Tuple[str, int, int] = None
											) -> Dict[bytes, Dict[bytes, bytes]]:
		memo = self._char_portals_rulebooks_copy_memo
		if portals == 'all' and char in memo and btt in memo[char]:
			return memo[char][btt]
		pack = self.pack
		chara = self._real.character[char]
		result: Dict[bytes, Dict[bytes, bytes]] = defaultdict(dict)
		branch, turn, tick = self._get_btt(btt)
		origtime = self._get_btt()
		if (branch, turn, tick) != origtime:
			self._real._set_btt(branch, turn, tick)
		if portals == 'all':
			portiter = chara.portals()
		else:
			portiter = (chara.portal[orig][dest] for (orig, dest) in portals)
		for portal in portiter:
			result[pack(portal['origin'])][pack(portal['destination'])] = pack(
				portal.rulebook.name)
		if (branch, turn, tick) != origtime:
			self._real._set_btt(*origtime)
		if portals == 'all':
			if char in memo:
				memo[char][btt] = result
			else:
				memo[char] = {btt: result}
		return result

	def _character_portals_rulebooks_delta(
		self,
		char,
		portals='all',
		*,
		btt_from: Tuple[str, int, int] = None,
		btt_to: Tuple[str, int, int] = None
	) -> Tuple[Dict[bytes, Dict[bytes, bytes]], Dict[bytes, Dict[bytes,
																	bytes]]]:
		btt_from = self._get_btt(btt_from)
		btt_to = self._get_btt(btt_to)
		if btt_from == btt_to:
			return {}, {}
		old = self.character_portals_rulebooks_copy(char, portals, btt_from)
		new: Dict[bytes,
					Dict[bytes,
							bytes]] = self.character_portals_rulebooks_copy(
								char, portals, btt=btt_to)
		former: Dict[bytes, Dict[bytes, bytes]] = {}
		current: Dict[bytes, Dict[bytes, bytes]] = {}
		futs = []
		for origin in old:
			if origin in new:
				fut = self.threadpool.submit(packed_dict_delta, old[origin],
												new[origin])
				fut.origin = origin
				futs.append(fut)
		for fut in as_completed(futs):
			former[fut.origin], current[fut.origin] = fut.result()
		return former, current

	def _character_delta(
		self,
		char,
		*,
		btt_from: Tuple[str, int, int] = None,
		btt_to: Tuple[str, int, int] = None
	) -> Dict[bytes, Union[bytes, Dict[bytes, bytes]]]:
		observed_btt = self._get_btt(btt_from)
		actual_btt = self._get_btt(btt_to)
		if observed_btt == actual_btt:
			return {}
		memo = self._character_delta_memo[char]
		if observed_btt in memo and actual_btt in memo[observed_btt]:
			return memo[observed_btt][actual_btt]
		nodes_fut: Future[Any] = self.threadpool.submit(
			self.character_nodes_delta,
			char,
			btt_from=observed_btt,
			btt_to=actual_btt)
		edges_fut = self.threadpool.submit(self.character_portals_delta,
											char,
											btt_from=observed_btt,
											btt_to=actual_btt)
		units_fut = self.threadpool.submit(self._character_units_delta,
											char,
											btt_from=observed_btt,
											btt_to=actual_btt)
		rbs_fut = self.threadpool.submit(self.character_rulebooks_delta,
											char,
											btt_from=observed_btt,
											btt_to=actual_btt)
		nrbs_fut = self.threadpool.submit(self.character_nodes_rulebooks_delta,
											char,
											btt_from=observed_btt,
											btt_to=actual_btt)
		porbs_fut = self.threadpool.submit(
			self._character_portals_rulebooks_delta,
			char,
			btt_from=observed_btt,
			btt_to=actual_btt)
		nv_fut = self.threadpool.submit(self._character_nodes_stat_delta,
										char,
										btt_from=observed_btt,
										btt_to=actual_btt)
		ev_fut = self.threadpool.submit(self._character_portals_stat_delta,
										char,
										btt_from=observed_btt,
										btt_to=actual_btt)
		chara = self._real.character[char]
		former, current = self.character_stat_delta(char,
													btt_from=observed_btt,
													btt_to=actual_btt)
		nodes_res = nodes_fut.result()
		if nodes_res != EMPTY_DELTA:
			former[NODES], current[NODES] = nodes_res
		edges_res = edges_fut.result()
		if edges_res != EMPTY_DELTA:
			former[EDGES], current[EDGES] = edges_res
		units_res = units_fut.result()
		if units_res != EMPTY_DELTA:
			former[UNITS], current[UNITS] = units_res
		rbs = rbs_fut.result()
		if rbs != EMPTY_DELTA:
			former[RULEBOOKS], current[RULEBOOKS] = rbs
		nv = nv_fut.result()
		nrbs = nrbs_fut.result()
		for (nw, nrbt) in zip(nv, nrbs):
			if not nrbt:
				continue
			for node, rb in nrbt.items():
				if node not in chara.node:
					continue
				if node in nw:
					nw[node][RULEBOOK] = rb
				else:
					nw[node] = {RULEBOOK: rb}
		if nv != EMPTY_DELTA:
			former[NODE_VAL], current[NODE_VAL] = nv
		ev = ev_fut.result()
		porbs = porbs_fut.result()
		for (ew, porbt) in zip(ev, porbs):
			if not porbt:
				continue
			for orig, dests in porbt.items():
				if orig not in chara.portal:
					continue
				portals = chara.portal[orig]
				for dest, rb in dests.items():
					if dest not in portals:
						continue
					ew.setdefault(orig, {}).setdefault(dest, {})[RULEBOOK] = rb
		if ev != EMPTY_DELTA:
			former[EDGE_VAL], current[EDGE_VAL] = ev
		self._character_delta_memo[char][observed_btt][
			actual_btt] = current  # canon
		self._character_delta_memo[char][actual_btt][observed_btt] = former
		return current

	def set_character_stat(self, char: Hashable, k: Hashable, v) -> None:
		self._real.character[char].stat[k] = v

	def del_character_stat(self, char: Hashable, k: Hashable) -> None:
		del self._real.character[char].stat[k]

	def update_character_stats(self, char: Hashable, patch: Dict) -> None:
		self._real.character[char].stat.update(patch)

	def update_character(self, char: Hashable, patch: Dict):
		self.update_character_stats(char, patch['character'])
		self.update_nodes(char, patch['node'])
		self.update_portals(char, patch['portal'])

	def characters(self) -> List[Hashable]:
		return list(self._real.character.keys())

	def set_node_stat(self, char: Hashable, node: Hashable, k: Hashable,
						v) -> None:
		self._real.character[char].node[node][k] = v

	def del_node_stat(self, char: Hashable, node: Hashable,
						k: Hashable) -> None:
		del self._real.character[char].node[node][k]

	def _get_btt(self,
					btt: Tuple[str, int, int] = None) -> Tuple[str, int, int]:
		if btt is None:
			return self._real._btt()
		return btt

	@prepacked
	def node_stat_copy(self,
						char: Hashable,
						node: Hashable,
						btt: Tuple[str, int,
									int] = None) -> Dict[bytes, bytes]:
		branch, turn, tick = self._get_btt(btt)
		memo = self._node_stat_copy_memo
		if (char, node, branch, turn, tick) in memo:
			return memo[char, node, branch, turn, tick]
		pack = self._real.pack
		origtime = self._real._btt()
		if (branch, turn, tick) != origtime:
			self._real._set_btt(branch, turn, tick)
		noden = node
		node = self._real.character[char].node[noden]
		ret = memo[char, noden, branch, turn, tick] = {
			pack(k): pack(v.unwrap()) if hasattr(v, 'unwrap')
			and not hasattr(v, 'no_unwrap') else pack(v)
			for (k, v) in node.items() if k not in
			{'character', 'name', 'arrival_time', 'next_arrival_time'}
		}
		if (branch, turn, tick) != origtime:
			self._real._set_btt(*origtime)
		return ret

	@prepacked
	def node_stat_delta(
			self,
			char: Hashable,
			node: Hashable,
			*,
			btt_from: Tuple[str, int, int] = None,
			btt_to: Tuple[str, int, int] = None) -> FormerAndCurrentType:
		"""Return a dictionary describing changes to a node's stats since the
		last time you looked at it.

		"""
		old = self.node_stat_copy(char, node, btt=btt_from)
		new = self.node_stat_copy(char, node, btt=btt_to)
		return packed_dict_delta(old, new)

	def _character_nodes_stat_delta(
		self,
		char: Hashable,
		*,
		btt_from: Tuple[str, int, int] = None,
		btt_to: Tuple[str, int, int] = None
	) -> Tuple[Dict[bytes, Dict[bytes, bytes]], Dict[bytes, Dict[bytes,
																	bytes]]]:
		"""Return a dictionary of ``node_stat_delta`` output for each node in a
		character.

		"""
		pack = self._real.pack
		former = {}
		current = {}
		origtime = self._real._btt()
		self._real._set_btt(*btt_from)
		nodes_from = set(self._real.character[char].node.keys())
		self._real._set_btt(*btt_to)
		nodes_to = set(self._real.character[char].node.keys())
		self._real._set_btt(*origtime)
		nodes = nodes_from & nodes_to
		futs = []
		for node in nodes:
			fut = self.threadpool.submit(self.node_stat_delta,
											char,
											node,
											btt_from=btt_from,
											btt_to=btt_to)
			fut.node = pack(node)
			futs.append(fut)
		for fut in as_completed(futs):
			d_former, d_current = fut.result()
			if d_current:
				assert d_former
				current[fut.node] = d_current
				former[fut.node] = d_former
		return former, current

	def _character_nodes_stat_copy(
			self,
			char: Hashable,
			btt: Tuple[str, int, int] = None) -> Dict[bytes, bytes]:
		pack = self._real.pack
		return {
			pack(node): concat_d(self.node_stat_copy(char, node, btt=btt))
			for node in self._real.character[char].node
		}

	def update_node(self, char: Hashable, node: Hashable, patch: Dict) -> None:
		"""Change a node's stats according to a dictionary.

		The ``patch`` dictionary should hold the new values of stats,
		keyed by the stats' names; a value of ``None`` deletes the
		stat.

		"""
		character = self._real.character[char]
		if patch is None:
			del character.node[node]
		elif node not in character.node:
			character.node[node] = patch
			return
		else:
			character.node[node].update(patch)

	def update_nodes(self, char: Hashable, patch: Dict, backdate=False):
		"""Change the stats of nodes in a character according to a
		dictionary.

		"""
		# Performance could be improved by preserving the packed values
		tick_now = self._real.tick
		if backdate:
			parbranch, parrev = self._real._parentbranch_rev.get(
				self._real.branch, ('trunk', 0))
			self._real.tick = parrev
		for i, (n, npatch) in enumerate(patch.items(), 1):
			self.update_node(char, n, npatch)
		if backdate:
			self._real.tick = tick_now

	def del_node(self, char, node):
		"""Remove a node from a character."""
		del self._real.character[char].node[node]

	@prepacked
	def character_nodes(self,
						char: Hashable,
						btt: Tuple[str, int, int] = None) -> Set[bytes]:
		pack = self.pack
		memo = self._char_nodes_copy_memo
		branch, turn, tick = self._get_btt(btt)
		if (char, branch, turn, tick) in memo:
			return memo[char, branch, turn, tick]
		origtime = self._real._btt()
		if (branch, turn, tick) != origtime:
			self._real.time = branch, turn
			self._real.tick = tick
		ret = memo[char, branch, turn,
					tick] = set(map(pack, self._real.character[char].node))
		if (branch, turn, tick) != origtime:
			self._real.time = origtime[:2]
			self._real.tick = origtime[2]
		return ret

	@prepacked
	def character_nodes_delta(
			self,
			char: Hashable,
			*,
			btt_from: Tuple[str, int, int] = None,
			btt_to: Tuple[str, int, int] = None) -> FormerAndCurrentType:
		old = self.character_nodes(char, btt=btt_from)
		new = self.character_nodes(char, btt=btt_to)
		return set_delta(old, new)

	def node_predecessors(self, char: Hashable,
							node: Hashable) -> List[Hashable]:
		return list(self._real.character[char].pred[node].keys())

	def character_set_node_predecessors(self, char: Hashable, node: Hashable,
										preds: Iterable) -> None:
		self._real.character[char].pred[node] = preds

	def character_del_node_predecessors(self, char: Hashable,
										node: Hashable) -> None:
		del self._real.character[char].pred[node]

	@prepacked
	def node_successors(self,
						char: Hashable,
						node: Hashable,
						btt: Tuple[str, int, int] = None) -> Set[bytes]:
		memo = self._node_successors_copy_memo
		branch, turn, tick = self._get_btt(btt)
		if (char, node, branch, turn, tick) in memo:
			return memo[char, node, branch, turn, tick]
		origtime = self._real._btt()
		if (branch, turn, tick) != origtime:
			self._real._set_btt(branch, turn, tick)
		ret = memo[char, node, branch, turn, tick] = set(
			map(self.pack, self._real.character[char].portal[node].keys()))
		if (branch, turn, tick) != origtime:
			self._real._set_btt(*origtime)
		return ret

	def nodes_connected(self, char: Hashable, orig: Hashable,
						dest: Hashable) -> bool:
		return dest in self._real.character[char].portal[orig]

	def init_thing(self,
					char: Hashable,
					thing: Hashable,
					statdict: dict = None) -> None:
		if thing in self._real.character[char].thing:
			raise KeyError('Already have thing in character {}: {}'.format(
				char, thing))
		if statdict is None:
			statdict = {}
		return self.set_thing(char, thing, statdict)

	def set_thing(self, char: Hashable, thing: Hashable,
					statdict: Dict) -> None:
		self._real.character[char].thing[thing] = statdict

	def add_thing(self, char: Hashable, thing: Hashable, loc: Hashable,
					statdict: Dict) -> None:
		self._real.character[char].add_thing(thing, loc, **statdict)

	def place2thing(self, char: Hashable, node: Hashable,
					loc: Hashable) -> None:
		self._real.character[char].place2thing(node, loc)

	def thing2place(self, char: Hashable, node: Hashable) -> None:
		self._real.character[char].thing2place(node)

	def add_things_from(self, char: Hashable, seq: Iterable) -> None:
		for thing in seq:
			self.add_thing(char, *thing)

	def get_thing_location(self, char: Hashable,
							thing: Hashable) -> Optional[Hashable]:
		try:
			return self._real.character[char].thing[thing]['location']
		except KeyError:
			return None

	def set_thing_location(self, char: Hashable, thing: Hashable,
							loc: Hashable) -> None:
		self._real.character[char].thing[thing]['location'] = loc

	def thing_follow_path(self, char: Hashable, thing: Hashable,
							path: List[Hashable], weight: Hashable) -> int:
		return self._real.character[char].thing[thing].follow_path(
			path, weight)

	def thing_go_to_place(self, char: Hashable, thing: Hashable,
							place: Hashable, weight: Hashable) -> int:
		return self._real.character[char].thing[thing].go_to_place(
			place, weight)

	def thing_travel_to(self,
						char: Hashable,
						thing: Hashable,
						dest: Hashable,
						weight: Hashable = None,
						graph=None) -> int:
		"""Make something find a path to ``dest`` and follow it.

		Optional argument ``weight`` is the portal stat to use to schedule movement times.

		Optional argument ``graph`` is an alternative graph to use for pathfinding.
		Should resemble a networkx DiGraph.

		"""
		return self._real.character[char].thing[thing].travel_to(
			dest, weight, graph)

	def init_place(self,
					char: Hashable,
					place: Hashable,
					statdict: Dict = None) -> None:
		if place in self._real.character[char].place:
			raise KeyError('Already have place in character {}: {}'.format(
				char, place))
		if statdict is None:
			statdict = {}
		return self.set_place(char, place, statdict)

	def set_place(self, char: Hashable, place: Hashable,
					statdict: Dict) -> None:
		self._real.character[char].place[place] = statdict
		self._after_ret = partial(self.node_stat_copy, char, place)

	def add_places_from(self, char: Hashable, seq: Iterable) -> None:
		self._real.character[char].add_places_from(seq)

	@prepacked
	def character_portals(self,
							char: Hashable,
							btt: Tuple[str, int, int] = None) -> Set[bytes]:
		pack = self._real.pack
		branch, turn, tick = self._get_btt(btt)
		r = set()
		portal = self._real.character[char].portal
		origtime = self._real._btt()
		if (branch, turn, tick) != origtime:
			self._real._set_btt(branch, turn, tick)
		for o in portal:
			for d in portal[o]:
				r.add(pack((o, d)))
		if (branch, turn, tick) != origtime:
			self._real._set_btt(*origtime)
		return r

	@prepacked
	def character_portals_delta(
			self,
			char: Hashable,
			*,
			btt_from: Tuple[str, int, int] = None,
			btt_to: Tuple[str, int, int] = None) -> FormerAndCurrentType:
		if btt_from == btt_to:
			return {}, {}
		old = self.character_portals(char, btt=btt_from)
		new = self.character_portals(char, btt=btt_to)
		return set_delta(old, new)

	def add_portal(self, char: Hashable, orig: Hashable, dest: Hashable,
					symmetrical: bool, statdict: Dict) -> None:
		self._real.character[char].add_portal(orig, dest, symmetrical,
												**statdict)

	def add_portals_from(self, char: Hashable, seq: Iterable,
							symmetrical: bool) -> None:
		self._real.character[char].add_portals_from(seq, symmetrical)

	def del_portal(self, char: Hashable, orig: Hashable,
					dest: Hashable) -> None:
		self._real.character[char].remove_edge(orig, dest)

	def set_portal_stat(self, char: Hashable, orig: Hashable, dest: Hashable,
						k: Hashable, v) -> None:
		self._real.character[char].portal[orig][dest][k] = v

	def del_portal_stat(self, char: Hashable, orig: Hashable, dest: Hashable,
						k: Hashable) -> None:
		del self._real.character[char][orig][dest][k]

	@prepacked
	def portal_stat_copy(
			self,
			char: Hashable,
			orig: Hashable,
			dest: Hashable,
			btt: Tuple[str, int, int] = None) -> Dict[bytes, bytes]:
		pack = self._real.pack
		branch, turn, tick = self._get_btt(btt)
		memo = self._portal_stat_copy_memo
		if (char, orig, dest, branch, turn, tick) in memo:
			return memo[char, orig, dest, branch, turn, tick]
		origtime = self._real._btt()
		if (branch, turn, tick) != origtime:
			self._real._set_btt(branch, turn, tick)
		ret = memo[char, orig, dest, branch, turn, tick] = {
			pack(k): pack(v.unwrap())
			if hasattr(v, 'unwrap') and not hasattr(v, 'no_unwrap') else v
			for (
				k, v) in self._real.character[char].portal[orig][dest].items()
		}
		if (branch, turn, tick) != origtime:
			self._real._set_btt(*origtime)
		return ret

	@prepacked
	def portal_stat_delta(
			self,
			char: Hashable,
			orig: Hashable,
			dest: Hashable,
			*,
			btt_from: Tuple[str, int, int] = None,
			btt_to: Tuple[str, int, int] = None) -> FormerAndCurrentType:
		btt_from = self._get_btt(btt_from)
		btt_to = self._get_btt(btt_to)
		if btt_from == btt_to:
			return {}, {}
		old = self.portal_stat_copy(char, orig, dest, btt=btt_from)
		new = self.portal_stat_copy(char, orig, dest, btt=btt_to)
		return packed_dict_delta(old, new)

	def _character_portals_stat_copy(
		self,
		char: Hashable,
		btt: Tuple[str, int, int] = None
	) -> Dict[bytes, Dict[bytes, Dict[bytes, bytes]]]:
		pack = self.pack
		r = {}
		btt = self._get_btt(btt)
		chara = self._real.character[char]
		origtime = self._real._btt()
		self._real._set_btt(*btt)
		portals = set(chara.portals())
		self._real._set_btt(*origtime)
		for orig, dest in portals:
			porig = pack(orig)
			pdest = pack(dest)
			if porig not in r:
				r[porig] = {}
			r[porig][pdest] = self.portal_stat_copy(char, orig, dest, btt=btt)
		return r

	def _character_portals_stat_delta(
		self,
		char: Hashable,
		*,
		btt_from: Tuple[str, int, int] = None,
		btt_to: Tuple[str, int, int] = None
	) -> Tuple[Dict[bytes, Dict[bytes, bytes]], Dict[bytes, Dict[bytes,
																	bytes]]]:
		btt_from = self._get_btt(btt_from)
		btt_to = self._get_btt(btt_to)
		if btt_from == btt_to:
			return {}, {}
		former = {}
		current = {}
		futs = []
		origtime = self._real._btt()
		self._real._set_btt(*btt_from)
		ports_from = set(self._real.character[char].portals())
		self._real._set_btt(*btt_to)
		ports_to = set(self._real.character[char].portals())
		self._real._set_btt(*origtime)
		for orig, dest in ports_from & ports_to:
			fut = self.threadpool.submit(self.portal_stat_delta,
											char,
											orig,
											dest,
											btt_from=btt_from,
											btt_to=btt_to)
			fut.orig = orig
			fut.dest = dest
			futs.append(fut)
		for fut in as_completed(futs):
			fmr, cur = fut.result()
			orig = fut.orig
			dest = fut.dest
			if cur:
				assert fmr
				if orig not in current:
					assert orig not in former
					current[orig] = {}
					former[orig] = {}
				current[orig][dest] = cur
				former[orig][dest] = fmr
		return former, current

	def update_portal(self, char: Hashable, orig: Hashable, dest: Hashable,
						patch: Dict) -> None:
		character = self._real.character[char]
		if patch is None:
			del character.portal[orig][dest]
		elif orig not in character.portal or dest not in character.portal[orig]:
			character.portal[orig][dest] = patch
		else:
			character.portal[orig][dest].update(patch)

	def update_portals(self, char: Hashable,
						patch: Dict[Tuple[Hashable, Hashable], Dict]) -> None:
		for ((orig, dest), ppatch) in patch.items():
			self.update_portal(char, orig, dest, ppatch)

	def add_unit(self, char: Hashable, graph: Hashable,
					node: Hashable) -> None:
		self._real.character[char].add_unit(graph, node)

	def remove_unit(self, char: Hashable, graph: Hashable,
					node: Hashable) -> None:
		self._real.character[char].remove_unit(graph, node)

	def new_empty_rule(self, rule: str) -> None:
		self._real.rule.new_empty(rule)

	def new_empty_rulebook(self, rulebook: Hashable, btt=None) -> list:
		branch, turn, tick = self._get_btt(btt)
		self._rulebook_copy_memo[rulebook, branch, turn, tick] = []
		self._real.rulebook.__getitem__(rulebook)
		return []

	def rulebook_copy(self,
						rulebook: Hashable,
						btt: Tuple[str, int, int] = None) -> List[str]:
		branch, turn, tick = self._get_btt(btt)
		memo = self._rulebook_copy_memo
		if (rulebook, branch, turn, tick) in memo:
			return memo[rulebook, branch, turn, tick]
		ret = memo[rulebook, branch, turn,
					tick] = list(self._real.rulebook[rulebook]._get_cache(
						branch, turn, tick))
		return ret

	def rulebook_delta(
			self,
			rulebook: Hashable,
			*,
			btt_from: Tuple[str, int, int] = None,
			btt_to: Tuple[str, int, int] = None) -> Optional[List[str]]:
		old = self.rulebook_copy(rulebook, btt=btt_from)
		new = self.rulebook_copy(rulebook, btt=btt_to)
		if old == new:
			return
		return new

	def all_rulebooks_delta(
			self,
			*,
			btt_from: Tuple[str, int, int] = None,
			btt_to: Tuple[str, int, int] = None) -> Dict[Hashable, List[str]]:
		btt_from = self._get_btt(btt_from)
		btt_to = self._get_btt(btt_to)
		if btt_from == btt_to:
			return {}
		ret = {}
		for rulebook in self._real.rulebook.keys():
			delta = self.rulebook_delta(rulebook,
										btt_from=btt_from,
										btt_to=btt_to)
			if delta:
				ret[rulebook] = delta
		return ret

	def all_rulebooks_copy(
			self,
			btt: Tuple[str, int, int] = None) -> Dict[Hashable, List[str]]:
		btt = self._get_btt(btt)
		origtime = self._real._btt()
		self._real._set_btt(*btt)
		ret = {
			rulebook: self.rulebook_copy(rulebook)
			for rulebook in self._real.rulebook.keys()
		}
		self._real._set_btt(*origtime)
		return ret

	def set_rulebook_rule(self, rulebook: Hashable, i: int, rule: str) -> None:
		self._real.rulebook[rulebook][i] = rule
		branch, turn, tick = self._real._btt()
		memo = self._rulebook_copy_memo
		if (rulebook, branch, turn, tick) in memo:
			memo[rulebook, branch, turn, tick][i] = rule

	def ins_rulebook_rule(self, rulebook: Hashable, i: int, rule: str) -> None:
		self._real.rulebook[rulebook].insert(i, rule)
		branch, turn, tick = self._real._btt()
		memo = self._rulebook_copy_memo
		if (rulebook, branch, turn, tick) in memo:
			memo[rulebook, branch, turn, tick].insert(i, rule)

	def del_rulebook_rule(self, rulebook: Hashable, i: int) -> None:
		del self._real.rulebook[rulebook][i]
		branch, turn, tick = self._real._btt()
		memo = self._rulebook_copy_memo
		if (rulebook, branch, turn, tick) in memo:
			del memo[rulebook, branch, turn, tick][i]

	def set_rule_triggers(self, rule: str, triggers: List[str]) -> None:
		self._real.rule[rule].triggers = triggers
		branch, turn, tick = self._real._btt()
		memo = self._rule_copy_memo
		if (rule, branch, turn, tick) in memo:
			memo[rule, branch, turn, tick]['triggers'] = triggers

	def set_rule_prereqs(self, rule: str, prereqs: List[str]) -> None:
		self._real.rule[rule].prereqs = prereqs
		branch, turn, tick = self._real._btt()
		memo = self._rule_copy_memo
		if (rule, branch, turn, tick) in memo:
			memo[rule, branch, turn, tick]['prereqs'] = prereqs

	def set_rule_actions(self, rule: str, actions: List[str]) -> None:
		self._real.rule[rule].actions = actions
		branch, turn, tick = self._real._btt()
		memo = self._rule_copy_memo
		if (rule, branch, turn, tick) in memo:
			memo[rule, branch, turn, tick]['actions'] = actions

	def set_character_rulebook(self, char: Hashable,
								rulebook: Hashable) -> None:
		self._real.character[char].rulebook = rulebook

	def set_unit_rulebook(self, char: Hashable, rulebook: Hashable) -> None:
		self._real.character[char].unit.rulebook = rulebook

	def set_character_thing_rulebook(self, char: Hashable,
										rulebook: Hashable) -> None:
		self._real.character[char].thing.rulebook = rulebook

	def set_character_place_rulebook(self, char: Hashable,
										rulebook: Hashable) -> None:
		self._real.character[char].place.rulebook = rulebook

	def set_character_node_rulebook(self, char: Hashable,
									rulebook: Hashable) -> None:
		self._real.character[char].node.rulebook = rulebook

	def set_character_portal_rulebook(self, char: Hashable,
										rulebook: Hashable) -> None:
		self._real.character[char].portal.rulebook = rulebook

	def set_node_rulebook(self, char: Hashable, node: Hashable,
							rulebook: Hashable) -> None:
		self._real.character[char].node[node].rulebook = rulebook

	def set_portal_rulebook(self, char: Hashable, orig: Hashable,
							dest: Hashable, rulebook: Hashable) -> None:
		self._real.character[char].portal[orig][dest].rulebook = rulebook

	def rule_copy(self,
					rule: str,
					btt: Tuple[str, int, int] = None) -> Dict[str, List[str]]:
		branch, turn, tick = self._get_btt(btt)
		memo = self._rule_copy_memo
		if (rule, branch, turn, tick) in memo:
			return memo[rule, branch, turn, tick]
		ret = memo[rule, branch, turn, tick] = {
			'triggers':
			list(self._real._triggers_cache.retrieve(rule, branch, turn,
														tick)),
			'prereqs':
			list(self._real._prereqs_cache.retrieve(rule, branch, turn, tick)),
			'actions':
			list(self._real._actions_cache.retrieve(rule, branch, turn, tick))
		}
		return ret

	def rule_delta(
			self,
			rule: str,
			*,
			btt_from: Tuple[str, int, int] = None,
			btt_to: Tuple[str, int, int] = None) -> Dict[str, List[str]]:
		btt_from = self._get_btt(btt_from)
		btt_to = self._get_btt(btt_to)
		if btt_from == btt_to:
			return {}
		old = self.rule_copy(rule, btt=btt_from)
		new = self.rule_copy(rule, btt=btt_to)
		ret = {}
		if new['triggers'] != old['triggers']:
			ret['triggers'] = new['triggers']
		if new['prereqs'] != old['prereqs']:
			ret['prereqs'] = new['prereqs']
		if new['actions'] != old['actions']:
			ret['actions'] = new['actions']
		return ret

	def all_rules_delta(
		self,
		*,
		btt_from: Tuple[str, int, int] = None,
		btt_to: Tuple[str, int,
						int] = None) -> Dict[str, Dict[str, List[str]]]:
		ret = {}
		btt_from = self._get_btt(btt_from)
		btt_to = self._get_btt(btt_to)
		if btt_from == btt_to:
			return ret
		for rule in self._real.rule.keys():
			delta = self.rule_delta(rule, btt_from=btt_from, btt_to=btt_to)
			if delta:
				ret[rule] = delta
		return ret

	def all_rules_copy(
			self,
			*,
			btt: Tuple[str, int,
						int] = None) -> Dict[str, Dict[str, List[str]]]:
		btt = self._get_btt(btt)
		origtime = self._real._btt()
		if btt != origtime:
			self._real._set_btt(*btt)
		ret = {
			rule: self.rule_copy(rule, btt)
			for rule in self._real.rule.keys()
		}
		if btt != origtime:
			self._real._set_btt(*origtime)
		return ret

	@prepacked
	def source_copy(self, store: str) -> Dict[bytes, bytes]:
		return dict(map(self.pack_pair,
						getattr(self._real, store).iterplain()))

	def get_source(self, store: str, name: str) -> str:
		return getattr(self._real, store).get_source(name)

	def store_source(self, store: str, v: str, name: str = None) -> None:
		getattr(self._real, store).store_source(v, name)

	def del_source(self, store: str, k: str) -> None:
		delattr(getattr(self._real, store), k)

	def call_stored_function(self, store: str, func: str, args: Tuple,
								kwargs: Dict) -> Any:
		if store == 'method':
			args = (self._real, ) + tuple(args)
		store = getattr(self._real, store)
		if store not in self._real.stores:
			raise ValueError("{} is not a function store".format(store))
		callme = getattr(store, func)
		return callme(*args, **kwargs)

	def call_randomizer(self, method: str, *args, **kwargs) -> Any:
		return getattr(self._real._rando, method)(*args, **kwargs)

	def install_module(self, module: str) -> None:
		import_module(module).install(self._real)

	def do_game_start(self) -> None:
		self._real.game_start()

	def is_parent_of(self, parent: str, child: str) -> bool:
		return self._real.is_parent_of(parent, child)

	def apply_choices(
		self,
		choices: List[dict],
		dry_run=False,
		perfectionist=False
	) -> Tuple[List[Tuple[Any, Any]], List[Tuple[Any, Any]]]:
		return self._real.apply_choices(choices, dry_run, perfectionist)

	@staticmethod
	def get_schedule(entity: Union[AbstractCharacter, Node,
									Portal], stats: Iterable[Hashable],
						beginning: int, end: int) -> Dict[Hashable, List]:
		ret = {}
		for stat in stats:
			ret[stat] = list(
				entity.historical(stat).iter_history(beginning, end))
		return ret

	@prepacked
	def grid_2d_8graph(self, character: Hashable, m: int, n: int) -> bytes:
		self._real.character[character].grid_2d_8graph(m, n)
		return self._concat_char_delta(
			self._get_char_deltas([character])[self.pack(character)])

	@prepacked
	def grid_2d_graph(self, character: Hashable, m: int, n: int,
						periodic: bool) -> bytes:
		self._real.character[character].grid_2d_graph(m, n, periodic)
		return self._concat_char_delta(
			self._get_char_deltas([character])[self.pack(character)])

	def rules_handled_turn(self,
							branch: str = None,
							turn: str = None) -> Dict[str, List[str]]:
		if branch is None:
			branch = self.branch
		if turn is None:
			turn = self.turn
		eng = self._real
		# assume the caches are all sync'd
		return {
			'character':
			eng._character_rules_handled_cache.handled_deep[branch][turn],
			'unit':
			eng._unit_rules_handled_cache.handled_deep[branch][turn],
			'character_thing':
			eng._character_thing_rules_handled_cache.handled_deep[branch]
			[turn],
			'character_place':
			eng._character_place_rules_handled_cache.handled_deep[branch]
			[turn],
			'character_portal':
			eng._character_portal_rules_handled_cache.handled_deep[branch]
			[turn],
			'node':
			eng._node_rules_handled_cache.handled_deep[branch][turn],
			'portal':
			eng._portal_rules_handled_cache.handled_deep[branch][turn]
		}

	def branches(self) -> Dict[str, Tuple[str, int, int, int, int]]:
		return self._real._branches
