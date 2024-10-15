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
from functools import partial
from operator import sub, or_

from .allegedb import Key
from .allegedb.cache import (
	Cache,
	PickyDefaultDict,
	StructuredDefaultDict,
	TurnDict,
	WindowDict,
	HistoricKeyError,
	EntitylessCache,
)
from .allegedb.window import EntikeySettingsTurnDict, SettingsTurnDict
from .util import singleton_get, sort_set
from collections import OrderedDict


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


class InitializedEntitylessCache(EntitylessCache, InitializedCache):
	__slots__ = ()


class UnitnessCache(Cache):
	"""A cache for remembering when a node is a unit of a character."""

	__slots__ = (
		"user_order",
		"user_shallow",
		"graphs",
		"graph_units",
		"char_units",
		"solo_unit",
		"unique_unit",
		"unique_graph",
		"users",
	)

	def __init__(self, engine):
		Cache.__init__(self, engine)
		self.user_order = StructuredDefaultDict(3, TurnDict)
		self.user_shallow = PickyDefaultDict(TurnDict)
		self.graphs = StructuredDefaultDict(1, TurnDict)
		self.graph_units = StructuredDefaultDict(1, TurnDict)
		self.char_units = StructuredDefaultDict(1, TurnDict)
		self.solo_unit = StructuredDefaultDict(1, TurnDict)
		self.unique_unit = StructuredDefaultDict(1, TurnDict)
		self.unique_graph = StructuredDefaultDict(1, TurnDict)
		self.users = StructuredDefaultDict(1, TurnDict)

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
		planning=None,
		forward=None,
		loading=False,
		contra=True,
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
		userturns = self.user_order[graph][node][character][branch]
		if turn in userturns:
			userturns[turn][tick] = is_unit
		else:
			userturns[turn] = {tick: is_unit}
		usershal = self.user_shallow[(graph, node, character, branch)]
		if turn in usershal:
			usershal[turn][tick] = is_unit
		else:
			usershal[turn] = {tick: is_unit}
		charavs = self.char_units[character]
		graphavs = self.graph_units[(character, graph)]
		graphs = self.graphs[character]
		uniqgraph = self.unique_graph[character][branch]
		soloav = self.solo_unit[(character, graph)][branch]
		uniqav = self.unique_unit[character][branch]
		users = self.users[graph, node]

		def add_or_discard_something(op, cache, what):
			for b, r, t in self.db._iter_parent_btt(branch, turn, tick):
				if b in cache:
					cb = cache[b]
					if r in cb and cb[r].rev_gettable(t):
						old_cache = cb[r][t]
						break
					elif cb.rev_gettable(r - 1):
						old_cache = cb[r - 1].final()
						break
			else:
				old_cache = frozenset()
			new_cache = op(old_cache, frozenset((what,)))
			if turn in cache[branch]:
				cache[branch][turn][tick] = new_cache
			elif branch in cache:
				cache[branch][turn] = {tick: new_cache}
			else:
				cache[branch] = {turn: {tick: new_cache}}

		add_something = partial(add_or_discard_something, or_)
		discard_something = partial(add_or_discard_something, sub)

		if is_unit:
			add_something(graphavs, node)
			add_something(charavs, (graph, node))
			add_something(graphs, graph)
			add_something(users, character)
		else:
			discard_something(graphavs, node)
			discard_something(charavs, (graph, node))
			try:
				if not graphavs[turn][tick]:
					discard_something(graphs, graph)
			except HistoricKeyError:
				pass
			try:
				if not charavs[turn][tick]:
					discard_something(users, character)
			except HistoricKeyError:
				pass
		try:
			graphav = singleton_get(graphavs[turn][tick])
			if turn in soloav:
				soloav[turn][tick] = graphav
			else:
				soloav[turn] = {tick: graphav}
		except HistoricKeyError:
			pass
		try:
			charav = singleton_get(charavs[turn][tick])
			if turn in uniqav:
				uniqav[turn][tick] = charav
			else:
				uniqav[turn] = {tick: charav}
		except HistoricKeyError:
			pass
		try:
			if not graphavs[turn][tick]:
				if graph in graphs[turn][tick]:
					graphs[turn][tick].remove(graph)
					if len(graphs[turn][tick]) == 1:
						uniqgraph[turn][tick] = next(iter(graphs[turn][tick]))
					else:
						uniqgraph[turn][tick] = None
		except HistoricKeyError:
			pass
		if (
			turn in graphavs
			and tick in graphavs[turn]
			and len(graphavs[turn][tick]) != 1
		):
			if turn in soloav:
				soloav[turn][tick] = None
			else:
				soloav[turn] = {tick: None}
		else:
			if turn in soloav:
				soloav[turn][tick] = node
			else:
				soloav[turn] = {tick: None}
		if (
			turn in charavs
			and charavs[turn].rev_gettable(tick)
			and len(charavs[turn][tick]) != 1
		):
			if turn in uniqav:
				uniqav[turn][tick] = None
			else:
				uniqav[turn] = {tick: None}
		elif turn in uniqav:
			uniqav[turn][tick] = (graph, node)
		else:
			uniqav[turn] = {tick: (graph, node)}
		if (
			turn in graphs
			and graphs[turn].rev_gettable(tick)
			and len(graphs[turn][tick]) != 1
		):
			if turn in uniqgraph:
				uniqgraph[turn][tick] = None
			else:
				uniqgraph[turn] = {tick: None}
		elif turn in uniqgraph:
			uniqgraph[turn][tick] = graph
		else:
			uniqgraph[turn] = {tick: graph}

	def get_char_graph_avs(self, char, graph, branch, turn, tick):
		return (
			self._valcache_lookup(
				self.graph_units[(char, graph)], branch, turn, tick
			)
			or set()
		)

	def get_char_graph_solo_av(self, char, graph, branch, turn, tick):
		return self._valcache_lookup(
			self.solo_unit[(char, graph)], branch, turn, tick
		)

	def get_char_only_av(self, char, branch, turn, tick):
		return self._valcache_lookup(
			self.unique_unit[char], branch, turn, tick
		)

	def get_char_only_graph(self, char, branch, turn, tick):
		return self._valcache_lookup(
			self.unique_graph[char], branch, turn, tick
		)

	def get_char_graphs(self, char, branch, turn, tick):
		return (
			self._valcache_lookup(self.graphs[char], branch, turn, tick)
			or set()
		)

	def _slow_iter_character_avatars(
		self, character, branch, turn, tick, *, forward
	):
		for graph in self.iter_entities(
			character, branch, turn, tick, forward=forward
		):
			for node in self.iter_entities(
				character, graph, branch, turn, tick, forward=forward
			):
				yield graph, node

	def _slow_iter_users(self, graph, node, branch, turn, tick):
		if graph not in self.user_order:
			return
		for character in self.user_order[graph][node]:
			if (graph, node, character, branch) not in self.user_shallow:
				for b, t, tc in self.db._iter_parent_btt(branch, turn, tick):
					if b in self.user_order[graph][node][character]:
						isav = self.user_order[graph][node][character][b][t]
						# side effect!! bad!
						self.store(
							character,
							graph,
							node,
							branch,
							turn,
							tick,
							isav[tc],
						)
						break
				else:
					self.store(
						character, graph, node, branch, turn, tick, None
					)
			try:
				if self.user_shallow[(graph, node, character, branch)][turn][
					tick
				]:
					yield character
			except HistoricKeyError:
				continue


class RulesHandledCache(object):
	def __init__(self, engine):
		self.engine = engine
		self.handled = {}
		self.handled_deep = StructuredDefaultDict(1, type=WindowDict)
		self.unhandled = {}

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
		self.handled_deep[branch][turn][tick] = (entity, rulebook, rule)
		unhandl = (
			self.unhandled.setdefault(entity, {})
			.setdefault(rulebook, {})
			.setdefault(branch, {})
		)
		if turn not in unhandl:
			unhandl[turn] = list(
				self.unhandled_rulebook_rules(
					entity, rulebook, branch, turn, tick
				)
			)
		try:
			unhandl[turn].remove(rule)
		except ValueError:
			pass

	def retrieve(self, *args):
		return self.handled[args]

	def unhandled_rulebook_rules(self, *args):
		entity = args[:-4]
		rulebook, branch, turn, tick = args[-4:]
		unh = self.unhandled
		if entity in unh:
			ue = unh[entity]
			if rulebook in ue:
				uer = ue[rulebook]
				if branch in uer:
					uerb = uer[branch]
					if turn in uerb:
						return uerb[turn]
		rbc = self.engine._rulebooks_cache
		if not rbc.contains_key(rulebook, branch, turn, tick):
			return []
		rulebook_rules = rbc.retrieve(rulebook, branch, turn, tick)[0]
		handled_rules = self.handled.setdefault(
			entity + (rulebook, branch, turn), set()
		)
		return [rule for rule in rulebook_rules if rule not in handled_rules]


class CharacterRulesHandledCache(RulesHandledCache):
	def get_rulebook(self, character, branch, turn, tick):
		try:
			return self.engine._characters_rulebooks_cache.retrieve(
				character, branch, turn, tick
			)
		except KeyError:
			return character, "character"

	def iter_unhandled_rules(self, branch, turn, tick):
		for character in self.engine.character.keys():
			rb = self.get_rulebook(character, branch, turn, tick)
			try:
				prio = self.engine._rulebooks_cache.retrieve(
					rb, branch, turn, tick
				)[1]
			except KeyError:
				continue
			for rule in self.unhandled_rulebook_rules(
				character, rb, branch, turn, tick
			):
				yield prio, character, rb, rule


class UnitRulesHandledCache(RulesHandledCache):
	def get_rulebook(self, character, branch, turn, tick):
		try:
			return self.engine._units_rulebooks_cache.retrieve(
				character, branch, turn, tick
			)
		except KeyError:
			return character, "unit"

	def iter_unhandled_rules(self, branch, turn, tick):
		charm = self.engine.character
		for character in sort_set(charm.keys()):
			rulebook = self.get_rulebook(character, branch, turn, tick)
			try:
				prio = self.engine._rulebooks_cache.retrieve(
					rulebook, branch, turn, tick
				)[1]
			except KeyError:
				continue
			charavm = charm[character].unit
			for graph in sort_set(charavm.keys()):
				for avatar in sort_set(charavm[graph].keys()):
					try:
						rules = self.unhandled_rulebook_rules(
							character,
							graph,
							avatar,
							rulebook,
							branch,
							turn,
							tick,
						)
					except KeyError:
						continue
					for rule in rules:
						yield prio, character, graph, avatar, rulebook, rule


class CharacterThingRulesHandledCache(RulesHandledCache):
	def get_rulebook(self, character, branch, turn, tick):
		try:
			return self.engine._characters_things_rulebooks_cache.retrieve(
				character, branch, turn, tick
			)
		except KeyError:
			return character, "character_thing"

	def iter_unhandled_rules(self, branch, turn, tick):
		charm = self.engine.character
		for character in charm.keys():
			rulebook = self.get_rulebook(character, branch, turn, tick)
			try:
				prio = self.engine._rulebooks_cache.retrieve(
					rulebook, branch, turn, tick
				)[1]
			except KeyError:
				continue
			for thing in charm[character].thing.keys():
				try:
					rules = self.unhandled_rulebook_rules(
						character, thing, rulebook, branch, turn, tick
					)
				except KeyError:
					continue
				for rule in rules:
					yield prio, character, thing, rulebook, rule


class CharacterPlaceRulesHandledCache(RulesHandledCache):
	def get_rulebook(self, character, branch, turn, tick):
		try:
			return self.engine._characters_places_rulebooks_cache.retrieve(
				character, branch, turn, tick
			)
		except KeyError:
			return character, "character_place"

	def iter_unhandled_rules(self, branch, turn, tick):
		charm = self.engine.character
		for character in charm.keys():
			rulebook = self.get_rulebook(character, branch, turn, tick)
			try:
				prio = self.engine._rulebooks_cache.retrieve(
					rulebook, branch, turn, tick
				)[1]
			except KeyError:
				continue
			for place in charm[character].place.keys():
				try:
					rules = self.unhandled_rulebook_rules(
						character, place, rulebook, branch, turn, tick
					)
				except KeyError:
					continue
				for rule in rules:
					yield prio, character, place, rulebook, rule


class CharacterPortalRulesHandledCache(RulesHandledCache):
	def get_rulebook(self, character, branch, turn, tick):
		try:
			return self.engine._characters_portals_rulebooks_cache.retrieve(
				character, branch, turn, tick
			)
		except KeyError:
			return character, "character_portal"

	def iter_unhandled_rules(self, branch, turn, tick):
		charm = self.engine.character
		for character in sort_set(charm.keys()):
			try:
				rulebook = self.get_rulebook(character, branch, turn, tick)
			except KeyError:
				continue
			try:
				rulebook_rules, prio = self.engine._rulebooks_cache.retrieve(
					rulebook, branch, turn, tick
				)
			except KeyError:
				continue
			if not rulebook_rules:
				continue
			char = charm[character]
			charn = char.node
			charp = char.portal
			for orig in charp.keys():
				if orig not in charn:
					continue
				for dest in charp[orig].keys():
					if dest not in charn:
						continue
					try:
						rules = self.unhandled_rulebook_rules(
							character, orig, dest, rulebook, branch, turn, tick
						)
					except KeyError:
						continue
					for rule in rules:
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
		nodes_with_rulebook_changed = {
			(character, node)
			for (
				character,
				node,
				_,
				_,
				_,
			) in self.engine._nodes_rulebooks_cache.shallowest
		}
		nodes_with_filled_default_rulebooks = {
			(character, node)
			for (_, (character, node)) in (
				{
					(None, (character, node))
					for character in charm
					for node in charm[character].node
				}
				& self.engine._rulebooks_cache.branches.keys()
			)
		}
		for character, node in (
			nodes_with_rulebook_changed | nodes_with_filled_default_rulebooks
		):
			try:
				rulebook = self.get_rulebook(
					character, node, branch, turn, tick
				)
				prio = self.engine._rulebooks_cache.retrieve(
					rulebook, branch, turn, tick
				)[1]
				rules = self.unhandled_rulebook_rules(
					character, node, rulebook, branch, turn, tick
				)
			except KeyError:
				continue
			for rule in rules:
				yield prio, character, node, rulebook, rule


class PortalRulesHandledCache(RulesHandledCache):
	def get_rulebook(self, character, orig, dest, branch, turn, tick):
		try:
			return self.engine._portals_rulebooks_cache.retrieve(
				character, orig, dest, branch, turn, tick
			)
		except KeyError:
			return character, orig, dest

	def iter_unhandled_rules(self, branch, turn, tick):
		portals_with_rulebook_changed = {
			(character, orig, dest)
			for (
				character,
				orig,
				dest,
				_,
				_,
				_,
			) in self.engine._portals_rulebooks_cache.shallowest
		}
		portals_default_rulebooks = {
			(None, port) for port in self.engine._edges_cache.keys
		}
		portals_with_filled_default_rulebooks = {
			(character, orig, dest)
			for (_, (character, orig, dest)) in (
				portals_default_rulebooks
				& self.engine._rulebooks_cache.branches.keys()
			)
		}
		for character, orig, dest in (
			portals_with_rulebook_changed
			| portals_with_filled_default_rulebooks
		):
			try:
				rulebook = self.get_rulebook(
					character, orig, dest, branch, turn, tick
				)
				prio = self.engine._rulebooks_cache.retrieve(
					rulebook, branch, turn, tick
				)[1]
				rules = self.unhandled_rulebook_rules(
					character, orig, dest, rulebook, branch, turn, tick
				)
			except KeyError:
				continue
			for rule in rules:
				yield prio, character, orig, dest, rulebook, rule


class ThingsCache(Cache):
	name = "things_cache"

	def __init__(self, db):
		Cache.__init__(self, db)
		self._make_node = db.thing_cls

	def store(self, *args, planning=None, loading=False, contra=True):
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
			if oldloc is not None:
				oldconts_orig = node_contents_cache.retrieve(
					character, oldloc, branch, turn, tick
				)
				newconts_orig = oldconts_orig.difference(this)
				node_contents_cache.store(
					character,
					oldloc,
					branch,
					turn,
					tick,
					newconts_orig,
					contra=False,
					loading=True,
				)
				todo = []
				# update any future contents caches pertaining to the old location
				if (character, oldloc) in node_contents_cache.loc_settings:
					locset = node_contents_cache.loc_settings[
						character, oldloc
					][branch]
					if turn in locset:
						for future_tick in locset[turn].future(tick):
							todo.append((turn, future_tick))
					for future_turn, future_ticks in locset.future(
						turn
					).items():
						for future_tick in future_ticks:
							todo.append((future_turn, future_tick))
				for trn, tck in todo:
					node_contents_cache.store(
						character,
						oldloc,
						branch,
						trn,
						tck,
						node_contents_cache.retrieve(
							character, oldloc, branch, trn, tck, search=True
						).difference(this),
						planning=False,
						contra=False,
						loading=True,
					)
			if location is not None:
				todo = []
				try:
					oldconts_dest = node_contents_cache.retrieve(
						character, location, branch, turn, tick
					)
				except KeyError:
					oldconts_dest = frozenset()
				newconts_dest = oldconts_dest.union(this)
				node_contents_cache.store(
					character,
					location,
					branch,
					turn,
					tick,
					newconts_dest,
					contra=False,
					loading=True,
				)
				# and the new location
				if (character, location) in node_contents_cache.loc_settings:
					locset = node_contents_cache.loc_settings[
						character, location
					][branch]
					if turn in locset:
						for future_tick in locset[turn].future(tick):
							todo.append((turn, future_tick))
					for future_turn, future_ticks in locset.future(
						turn
					).items():
						for future_tick in future_ticks:
							todo.append((future_turn, future_tick))
				for trn, tck in todo:
					node_contents_cache.store(
						character,
						location,
						branch,
						trn,
						tck,
						node_contents_cache.retrieve(
							character, location, branch, trn, tck, search=True
						).union(this),
						planning=False,
						contra=False,
						loading=True,
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


class NodeContentsCache(Cache):
	name = "node_contents_cache"

	def __init__(self, db, kfkvs=None):
		super().__init__(db, kfkvs)
		self.loc_settings = StructuredDefaultDict(1, SettingsTurnDict)

	def store(
		self,
		character: Key,
		place: Key,
		branch: str,
		turn: int,
		tick: int,
		contents: frozenset,
		planning: bool = None,
		forward: bool = None,
		loading=False,
		contra=None,
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
