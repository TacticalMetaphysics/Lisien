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
"""Wrap a lisien engine so you can access and control it using only
ordinary method calls.

"""

from importlib import import_module
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING
from re import match
from typing import Any, Callable, Iterable

import msgpack
import networkx as nx

from ..engine import (
	EDGE_VAL,
	EDGES,
	ETERNAL,
	NODE_VAL,
	NODES,
	NONE,
	RULEBOOK,
	RULEBOOKS,
	RULES,
	UNITS,
	UNIVERSAL,
	Engine,
)
from ..exc import OutOfTimelineError
from ..node import Node
from ..portal import Portal
from ..typing import Key
from ..util import AbstractCharacter, BadTimeException, timer

SlightlyPackedDeltaType = dict[
	bytes,
	dict[
		bytes,
		bytes | dict[bytes, bytes | dict[bytes, bytes | dict[bytes, bytes]]],
	],
]
FormerAndCurrentType = tuple[dict[bytes, bytes], dict[bytes, bytes]]


EMPTY_MAPPING = msgpack.packb({})


def concat_d(r: dict[bytes, bytes]) -> bytes:
	"""Pack a dictionary of msgpack-encoded keys and values into msgpack bytes"""
	resp = msgpack.Packer().pack_map_header(len(r))
	for k, v in r.items():
		resp += k + v
	return resp


def prepacked(fun: Callable) -> Callable:
	fun.prepacked = True
	return fun


class EngineHandle:
	"""A wrapper for a :class:`lisien.Engine` object that runs in the same
	process, but with an API built to be used in a command-processing
	loop that takes commands from another process.

	It's probably a bad idea to use this class unless you're
	developing your own API.

	"""

	_after_ret: Callable

	def __init__(self, *args, logq=None, loglevel=None, **kwargs):
		"""Instantiate an engine with the given arguments

		``logq`` is a :class:`Queue` into which I'll put tuples of
		``(loglevel, message)``. ``loglevel`` is one of
		`'debug'`, `'info'`, `'warning'`, `'error'`, or `'critical'`
		(or the constants from the `logging` module, or an integer)
		and controls what messages will be logged.

		"""
		kwargs.setdefault("logfun", self.log)
		do_game_start = kwargs.pop("do_game_start", False)
		self._logq = logq
		self._loglevel = loglevel
		self._real = Engine(*args, **kwargs)
		self.pack = pack = self._real.pack

		def pack_pair(pair):
			k, v = pair
			return pack(k), pack(v)

		self.pack_pair = pack_pair
		self.unpack = self._real.unpack

		self._cache_arranger_started = False
		if do_game_start:
			self.do_game_start()

	def log(self, level: str | int, message: str) -> None:
		if isinstance(level, str):
			level = {
				"debug": 10,
				"info": 20,
				"warning": 30,
				"error": 40,
				"critical": 50,
			}[level.lower()]
		if self._logq and level >= self._loglevel:
			self._logq.put((level, message))
		elif not self._logq:
			print(level, message)

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
		return hasattr(self._real, "locktime")

	def snap_keyframe(self, silent=False):
		return self._real.snap_keyframe(silent=silent)

	def _pack_delta(self, delta) -> tuple[SlightlyPackedDeltaType, bytes]:
		pack = self.pack
		slightly_packed_delta = {}
		mostly_packed_delta = {}
		for char, chardelta in delta.items():
			if chardelta is None or chardelta == {"name": None}:
				pchar = pack(char)
				slightly_packed_delta[pchar] = mostly_packed_delta[pchar] = (
					None
				)
				continue
			chardelta = chardelta.copy()
			pchar = pack(char)
			chard = slightly_packed_delta[pchar] = {}
			packd = mostly_packed_delta[pchar] = {}
			if "nodes" in chardelta:
				nd = chard[NODES] = {
					pack(node): pack(ex)
					for node, ex in chardelta.pop("nodes").items()
				}
				packd[NODES] = concat_d(nd)
			if "node_val" in chardelta:
				slightnoded = chard[NODE_VAL] = {}
				packnodevd = {}
				for node, vals in chardelta.pop("node_val").items():
					pnode = pack(node)
					pvals = dict(map(self.pack_pair, vals.items()))
					slightnoded[pnode] = pvals
					packnodevd[pnode] = concat_d(pvals)
				packd[NODE_VAL] = concat_d(packnodevd)
			if "edges" in chardelta:
				ed = chard[EDGES] = {
					pack(origdest): pack(ex)
					for origdest, ex in chardelta.pop("edges").items()
				}
				packd[EDGES] = concat_d(ed)
			if "edge_val" in chardelta:
				slightorigd = chard[EDGE_VAL] = {}
				packorigd = {}
				for orig, dests in chardelta.pop("edge_val").items():
					porig = pack(orig)
					slightdestd = slightorigd[porig] = {}
					packdestd = {}
					for dest, port in dests.items():
						pdest = pack(dest)
						slightportd = slightdestd[pdest] = dict(
							map(self.pack_pair, port.items())
						)
						packdestd[pdest] = concat_d(slightportd)
					packorigd[porig] = concat_d(packdestd)
				packd[EDGE_VAL] = concat_d(packorigd)
			if "units" in chardelta:
				slightgraphd = chard[UNITS] = {}
				packunitd = {}
				for graph, unitss in chardelta.pop("units").items():
					if unitss is None:
						continue
					pgraph = pack(graph)
					slightunitd = slightgraphd[pgraph] = dict(
						map(self.pack_pair, unitss.items())
					)
					packunitd[pgraph] = concat_d(slightunitd)
				packd[UNITS] = concat_d(packunitd)
			if "rulebooks" in chardelta:
				chard[RULEBOOKS] = slightrbd = dict(
					map(self.pack_pair, chardelta.pop("rulebooks").items())
				)
				packd[RULEBOOKS] = concat_d(slightrbd)
			todo = dict(map(self.pack_pair, chardelta.items()))
			chard.update(todo)
			packd.update(todo)
		return slightly_packed_delta, concat_d(
			{
				charn: (concat_d(stuff) if stuff is not None else NONE)
				for charn, stuff in mostly_packed_delta.items()
			}
		)

	@staticmethod
	def _concat_char_delta(delta: SlightlyPackedDeltaType) -> bytes:
		delta = delta.copy()
		mostly_packed_delta = packd = {}
		eternal = delta.pop(ETERNAL, None)
		if eternal:
			mostly_packed_delta[ETERNAL] = eternal
		universal = delta.pop(UNIVERSAL, None)
		if universal:
			mostly_packed_delta[UNIVERSAL] = universal
		if RULEBOOK in delta:
			mostly_packed_delta[RULEBOOK] = delta.pop(RULEBOOK)
		if RULES in delta:
			rules = delta.pop(RULES)
			mostly_packed_delta[RULES] = concat_d(
				{rule: concat_d(funcls) for (rule, funcls) in rules.items()}
			)
		if NODES in delta:
			charnodes = delta.pop(NODES)
			packd[NODES] = concat_d(charnodes)
		if NODE_VAL in delta:
			slightnoded = {}
			packnodevd = {}
			for node, vals in delta.pop(NODE_VAL).items():
				slightnoded[node] = vals
				packnodevd[node] = concat_d(vals)
			packd[NODE_VAL] = concat_d(packnodevd)
		if EDGES in delta:
			es = delta.pop(EDGES)
			packd[EDGES] = concat_d(es)
		if EDGE_VAL in delta:
			packorigd = {}
			for orig, dests in delta.pop(EDGE_VAL).items():
				slightdestd = {}
				packdestd = {}
				for dest, port in dests.items():
					slightdestd[dest] = port
					packdestd[dest] = concat_d(port)
				packorigd[orig] = concat_d(packdestd)
			packd[EDGE_VAL] = concat_d(packorigd)
		if UNITS in delta:
			if delta[UNITS] == NONE:
				packd[UNITS] = concat_d({})
				del delta[UNITS]
			else:
				packd[UNITS] = delta.pop(UNITS)
		mostly_packed_delta.update(delta)
		return concat_d(mostly_packed_delta)

	@prepacked
	def next_turn(self) -> tuple[bytes, bytes]:
		"""Simulate a turn. Return whatever result, as well as a delta"""
		pack = self.pack
		self.debug(
			"calling next_turn at {}, {}, {}".format(*self._real._btt())
		)
		ret, delta = self._real.next_turn()
		slightly_packed_delta, packed_delta = self._pack_delta(delta)
		return pack(ret), packed_delta

	def _get_slow_delta(
		self,
		btt_from: tuple[str, int, int] = None,
		btt_to: tuple[str, int, int] = None,
	) -> SlightlyPackedDeltaType:
		return self._real._get_slow_delta(btt_from, btt_to)

	def start_branch(self, parent: str, branch: str, turn: int, tick: int):
		self._real._start_branch(parent, branch, turn, tick)

	def extend_branch(self, branch: str, turn: int, tick: int):
		self._real._extend_branch(branch, turn, tick)

	def load_at(self, branch: str, turn: int, tick: int):
		self._real.load_at(branch, turn, tick)

	def turn_end(self, branch: str = None, turn: int = None) -> int:
		return self._real.turn_end(branch, turn)

	def turn_end_plan(self, branch: str = None, turn: int = None) -> int:
		return self._real.turn_end_plan(branch, turn)

	@prepacked
	def time_travel(
		self,
		branch,
		turn,
		tick=None,
	) -> tuple[bytes, bytes]:
		"""Go to a different `(branch, turn, tick)` and return a delta

		For compatibility with `next_turn` this actually returns a tuple,
		the 0th item of which is `None`.

		"""
		if branch in self._real.branches():
			if self._real._enforce_end_of_time:
				turn_end, tick_end = self._real._branch_end(branch)
				if (tick is None and turn > turn_end) or (
					tick is not None and (turn, tick) > (turn_end, tick_end)
				):
					raise OutOfTimelineError(
						"Not traveling past the end of the branch",
						branch,
						turn,
						tick,
						turn_end,
						tick_end,
					)
			self._real.load_at(branch, turn, tick)
		branch_from, turn_from, tick_from = self._real._btt()
		if tick is None:
			if (
				branch,
				turn,
				self._real.turn_end(branch, turn),
			) == (
				branch_from,
				turn_from,
				tick_from,
			):
				return NONE, EMPTY_MAPPING
			self._real.time = (branch, turn, self._real.turn_end(branch, turn))
		else:
			if (branch, turn, tick) == (
				branch_from,
				turn_from,
				tick_from,
			):
				return NONE, EMPTY_MAPPING
			self._real.time = (branch, turn, tick)
		if turn_from != turn and (
			branch_from != branch
			or None in (turn_from, turn)
			or self._real._is_timespan_too_big(branch, turn_from, turn)
		):
			# This branch avoids unpacking and re-packing the delta
			slightly: SlightlyPackedDeltaType = self._real._get_slow_delta(
				(branch_from, turn_from, tick_from), self._real._btt()
			)
			mostly = {}
			if UNIVERSAL in slightly:
				mostly[UNIVERSAL] = concat_d(slightly.pop(UNIVERSAL))
			if RULES in slightly:
				mostly[RULES] = concat_d(
					{
						rule: concat_d(rule_d)
						for (rule, rule_d) in slightly.pop(RULES).items()
					}
				)
			if RULEBOOK in slightly:
				mostly[RULEBOOK] = concat_d(slightly.pop(RULEBOOK))
			for char, chardeltapacked in slightly.items():
				if chardeltapacked == b"\xc0":
					mostly[char] = b"\xc0"
					continue
				mostly[char] = self._concat_char_delta(chardeltapacked)
			return NONE, concat_d(mostly)
		return NONE, self._pack_delta(
			self._real.get_delta(
				(branch_from, turn_from, tick_from), self._real._btt()
			)
		)[1]

	@prepacked
	def increment_branch(self) -> bytes:
		"""Generate a new branch name and switch to it

		Returns the name of the new branch.

		"""
		branch = self._real.branch
		m = match(r"(.*)(\d+)", branch)
		if m:
			stem, n = m.groups()
			branch = stem + str(int(n) + 1)
		else:
			stem = branch
			n = 1
			branch = stem + str(n)
		if branch in self._real.branches():
			if m:
				n = int(n)
			else:
				stem = branch[:-1]
				n = 1
			while stem + str(n) in self._real.branches():
				n += 1
			branch = stem + str(n)
		self._real.branch = branch
		return self.pack(branch)

	def add_character(
		self,
		char: Key,
		data: nx.Graph | nx.DiGraph = None,
		node: dict = None,
		edge: dict = None,
		**attr,
	):
		"""Make a new character, initialized with whatever data"""
		# Probably not great that I am unpacking and then repacking the stats
		self._real.add_character(char, data=data, node=node, edge=edge, **attr)

	def commit(self):
		self._real.commit()

	def close(self):
		self._real.close()

	def get_btt(self):
		return self._real._btt()

	def get_language(self):
		return str(self._real.string.language)

	def set_language(self, lang):
		self._real.string.language = lang
		return self.strings_copy(lang)

	def get_string_lang_items(self, lang=None):
		return list(self._real.string.lang_items(lang))

	def strings_copy(self, lang=None):
		return dict(self._real.string.lang_items(lang))

	def set_string(self, k, v):
		self._real.string[k] = v

	def del_string(self, k):
		del self._real.string[k]

	@prepacked
	def get_eternal(self, k):
		return self.pack(self._real.eternal[k])

	def set_eternal(self, k, v):
		self._real.eternal[k] = v

	def del_eternal(self, k):
		del self._real.eternal[k]

	@prepacked
	def eternal_copy(self):
		return dict(map(self.pack_pair, self._real.eternal.items()))

	def set_universal(self, k, v):
		self._real.universal[k] = v

	def del_universal(self, k):
		del self._real.universal[k]

	def del_character(self, char):
		del self._real.character[char]

	def set_character_stat(self, char: Key, k: Key, v) -> None:
		self._real.character[char].stat[k] = v

	def del_character_stat(self, char: Key, k: Key) -> None:
		del self._real.character[char].stat[k]

	def set_node_stat(self, char: Key, node: Key, k: Key, v) -> None:
		self._real.character[char].node[node][k] = v

	def del_node_stat(self, char: Key, node: Key, k: Key) -> None:
		del self._real.character[char].node[node][k]

	def _get_btt(
		self, btt: tuple[str, int, int] = None
	) -> tuple[str, int, int]:
		if btt is None:
			return self._real._btt()
		return btt

	def node_exists(self, char: Key, node: Key) -> bool:
		return self._real._node_exists(char, node)

	def update_nodes(self, char: Key, patch: dict):
		"""Change the stats of nodes in a character according to a
		dictionary.

		"""
		node = self._real.character[char].node
		with (
			self._real.batch(),
			timer("EngineHandle.update_nodes", self.debug),
		):
			for n, npatch in patch.items():
				if npatch is None:
					del node[n]
				elif n not in node:
					node[n] = npatch
				else:
					node[n].update(npatch)

	def del_node(self, char, node):
		"""Remove a node from a character."""
		del self._real.character[char].node[node]

	def character_set_node_predecessors(
		self, char: Key, node: Key, preds: Iterable
	) -> None:
		self._real.character[char].pred[node] = preds

	def set_thing(self, char: Key, thing: Key, statdict: dict) -> None:
		self._real.character[char].thing[thing] = statdict

	def add_thing(
		self, char: Key, thing: Key, loc: Key, statdict: dict
	) -> None:
		self._real.character[char].add_thing(thing, loc, **statdict)

	def place2thing(self, char: Key, place: Key, loc: Key):
		self._real.character[char].place2thing(place, loc)

	def thing2place(self, char: Key, thing: Key):
		self._real.character[char].thing2place(thing)

	def set_thing_location(self, char: Key, thing: Key, loc: Key) -> None:
		self._real.character[char].thing[thing]["location"] = loc

	def thing_follow_path(
		self, char: Key, thing: Key, path: list[Key], weight: Key
	) -> int:
		return (
			self._real.character[char].thing[thing].follow_path(path, weight)
		)

	def thing_go_to_place(
		self, char: Key, thing: Key, place: Key, weight: Key
	) -> int:
		return (
			self._real.character[char].thing[thing].go_to_place(place, weight)
		)

	def thing_travel_to(
		self, char: Key, thing: Key, dest: Key, weight: Key = None, graph=None
	) -> int:
		"""Make something find a path to ``dest`` and follow it.

		Optional argument ``weight`` is the portal stat to use to schedule
		movement times.

		Optional argument ``graph`` is an alternative graph to use for
		pathfinding. Should resemble a networkx DiGraph.

		"""
		return (
			self._real.character[char]
			.thing[thing]
			.travel_to(dest, weight, graph)
		)

	def set_place(self, char: Key, place: Key, statdict: dict) -> None:
		self._real.character[char].place[place] = statdict

	def add_places_from(self, char: Key, seq: Iterable) -> None:
		self._real.character[char].add_places_from(seq)

	def add_portal(
		self,
		char: Key,
		orig: Key,
		dest: Key,
		statdict: dict,
		symmetrical: bool = False,
	) -> None:
		self._real.character[char].add_portal(orig, dest, **statdict)
		if symmetrical:
			self._real.character[char].add_portal(dest, orig, **statdict)

	def add_portals_from(self, char: Key, seq: Iterable) -> None:
		self._real.character[char].add_portals_from(seq)

	def del_portal(self, char: Key, orig: Key, dest: Key) -> None:
		ch = self._real.character[char]
		ch.remove_edge(orig, dest)
		assert orig in ch.node
		assert dest in ch.node

	def set_portal_stat(
		self, char: Key, orig: Key, dest: Key, k: Key, v
	) -> None:
		self._real.character[char].portal[orig][dest][k] = v

	def del_portal_stat(self, char: Key, orig: Key, dest: Key, k: Key) -> None:
		del self._real.character[char][orig][dest][k]

	def add_unit(self, char: Key, graph: Key, node: Key) -> None:
		self._real.character[char].add_unit(graph, node)

	def remove_unit(self, char: Key, graph: Key, node: Key) -> None:
		self._real.character[char].remove_unit(graph, node)

	def new_empty_rule(self, rule: str) -> None:
		self._real.rule.new_empty(rule)

	def new_empty_rulebook(self, rulebook: Key) -> list:
		self._real.rulebook.__getitem__(rulebook)
		return []

	def set_rulebook_rules(self, rulebook: Key, rules: list[str]) -> None:
		self._real.rulebook[rulebook] = [rules]

	def set_rulebook_rule(self, rulebook: Key, i: int, rule: str) -> None:
		self._real.rulebook[rulebook][i] = rule

	def ins_rulebook_rule(self, rulebook: Key, i: int, rule: str) -> None:
		self._real.rulebook[rulebook].insert(i, rule)

	def del_rulebook_rule(self, rulebook: Key, i: int) -> None:
		del self._real.rulebook[rulebook][i]

	def del_rulebook(self, rulebook: Key) -> None:
		del self._real.rulebook[rulebook]

	def del_rule(self, rule: Key) -> None:
		del self._real.rule[rule]

	def set_rule_triggers(self, rule: str, triggers: list[str]) -> None:
		self._real.rule[rule].triggers = triggers

	def set_rule_prereqs(self, rule: str, prereqs: list[str]) -> None:
		self._real.rule[rule].prereqs = prereqs

	def set_rule_actions(self, rule: str, actions: list[str]) -> None:
		self._real.rule[rule].actions = actions

	def set_rule_neighborhood(
		self, rule: str, neighborhood: int | None
	) -> None:
		self._real.rule[rule].neighborhood = neighborhood

	def get_rule_neighborhood(self, rule: str) -> int | None:
		return self._real.rule[rule].neighborhood

	def set_character_rulebook(self, char: Key, rulebook: Key) -> None:
		self._real.character[char].rulebook = rulebook

	def set_unit_rulebook(self, char: Key, rulebook: Key) -> None:
		self._real.character[char].unit.rulebook = rulebook

	def set_character_thing_rulebook(self, char: Key, rulebook: Key) -> None:
		self._real.character[char].thing.rulebook = rulebook

	def set_character_place_rulebook(self, char: Key, rulebook: Key) -> None:
		self._real.character[char].place.rulebook = rulebook

	def set_character_node_rulebook(self, char: Key, rulebook: Key) -> None:
		self._real.character[char].node.rulebook = rulebook

	def set_character_portal_rulebook(self, char: Key, rulebook: Key) -> None:
		self._real.character[char].portal.rulebook = rulebook

	def set_node_rulebook(self, char: Key, node: Key, rulebook: Key) -> None:
		self._real.character[char].node[node].rulebook = rulebook

	def set_portal_rulebook(
		self, char: Key, orig: Key, dest: Key, rulebook: Key
	) -> None:
		self._real.character[char].portal[orig][dest].rulebook = rulebook

	@prepacked
	def source_copy(self, store: str) -> dict[bytes, bytes]:
		return dict(
			map(self.pack_pair, getattr(self._real, store).iterplain())
		)

	def get_source(self, store: str, name: str) -> str:
		return getattr(self._real, store).get_source(name)

	def store_source(self, store: str, v: str, name: str = None) -> None:
		getattr(self._real, store).store_source(v, name)

	def del_source(self, store: str, k: str) -> None:
		delattr(getattr(self._real, store), k)

	def call_stored_function(
		self, store: str, func: str, args: tuple, kwargs: dict
	) -> Any:
		branch, turn, tick = self._real._btt()
		if store == "method":
			args = (self._real,) + tuple(args)
		store = getattr(self._real, store)
		if store not in self._real.stores:
			raise ValueError("{} is not a function store".format(store))
		callme = getattr(store, func)
		res = callme(*args, **kwargs)
		_, turn_now, tick_now = self._real._btt()
		delta = self._real._get_branch_delta(
			branch, turn, tick, turn_now, tick_now
		)
		return res, delta

	def call_randomizer(self, method: str, *args, **kwargs) -> Any:
		return getattr(self._real._rando, method)(*args, **kwargs)

	def install_module(self, module: str) -> None:
		import_module(module).install(self._real)

	def do_game_start(self):
		time_from = self._real._btt()
		self._real.game_start()
		return [], self._real._get_branch_delta(
			*time_from, self._real.turn, self._real.tick
		)

	def is_ancestor_of(self, parent: str, child: str) -> bool:
		return self._real.is_ancestor_of(parent, child)

	def branch_start(self, branch: str) -> tuple[int, int]:
		return self._real._branch_start(branch)

	def branch_end(self, branch: str) -> tuple[int, int]:
		return self._real._branch_end(branch)

	def branch_parent(self, branch: str) -> str | None:
		return self._real.branch_parent(branch)

	def apply_choices(
		self, choices: list[dict], dry_run=False, perfectionist=False
	) -> tuple[list[tuple[Any, Any]], list[tuple[Any, Any]]]:
		return self._real.apply_choices(choices, dry_run, perfectionist)

	@staticmethod
	def get_schedule(
		entity: AbstractCharacter | Node | Portal,
		stats: Iterable[Key],
		beginning: int,
		end: int,
	) -> dict[Key, list]:
		ret = {}
		for stat in stats:
			ret[stat] = list(
				entity.historical(stat).iter_history(beginning, end)
			)
		return ret

	def rules_handled_turn(
		self, branch: str = None, turn: str = None
	) -> dict[str, list[str]]:
		if branch is None:
			branch = self._real.branch
		if turn is None:
			turn = self._real.turn
		eng = self._real
		# assume the caches are all sync'd
		return {
			"character": eng._character_rules_handled_cache.handled_deep[
				branch
			][turn],
			"unit": eng._unit_rules_handled_cache.handled_deep[branch][turn],
			"character_thing": eng._character_thing_rules_handled_cache.handled_deep[
				branch
			][turn],
			"character_place": eng._character_place_rules_handled_cache.handled_deep[
				branch
			][turn],
			"character_portal": eng._character_portal_rules_handled_cache.handled_deep[
				branch
			][turn],
			"node": eng._node_rules_handled_cache.handled_deep[branch][turn],
			"portal": eng._portal_rules_handled_cache.handled_deep[branch][
				turn
			],
		}

	def branches(self) -> dict[str, tuple[str, int, int, int, int]]:
		return self._real._branches_d

	def main_branch(self) -> str:
		return self._real.main_branch

	def switch_main_branch(self, branch: str) -> dict:
		self._real.switch_main_branch(branch)
		return self.snap_keyframe()

	def game_init(self) -> None:
		branch, turn, tick = self._real._btt()
		if (turn, tick) != (0, 0):
			raise BadTimeException(
				"You tried to start a game when it wasn't the start of time"
			)
		self.do_game_start()
		kf = self.snap_keyframe()
		functions = dict(self._real.function.iterplain())
		methods = dict(self._real.method.iterplain())
		triggers = dict(self._real.trigger.iterplain())
		prereqs = dict(self._real.prereq.iterplain())
		actions = dict(self._real.action.iterplain())
		return (
			kf,
			self._real.eternal,
			functions,
			methods,
			triggers,
			prereqs,
			actions,
		)


def serial_handle(prefix, **kwargs):
	kwargs["workers"] = 0
	return EngineHandle(prefix, **kwargs)
