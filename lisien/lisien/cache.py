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
from functools import partial
from operator import itemgetter
from typing import Tuple

from .util import Key
from .allegedb.cache import (
	Cache,
	StructuredDefaultDict,
	WindowDict,
	EntitylessCache,
	KeyframeError,
)
from .allegedb.window import SettingsTurnDict
from .util import sort_set
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
		super().store(char, orig, dest, branch, turn, tick, rb)
		super().store(char, orig, branch, turn, tick, destrbs)

	def set_keyframe(
		self,
		graph_ent: Tuple[Key],
		branch: str,
		turn: int,
		tick: int,
		keyframe,
	):
		super().set_keyframe(graph_ent, branch, turn, tick, keyframe)
		for orig, dests in keyframe.items():
			for dest, rulebook in dests.items():
				try:
					subkf = self.get_keyframe(
						(*graph_ent, orig), branch, turn, tick, copy=True
					)
					subkf[dest] = rulebook
				except KeyError:
					subkf = {dest: rulebook}
				super().set_keyframe(
					(*graph_ent, orig), branch, turn, tick, subkf
				)


class UnitnessCache(Cache):
	"""A cache for remembering when a node is a unit of a character."""

	def __init__(self, db):
		super().__init__(db, "unitness_cache")
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
		characters: Key,
		branch: str,
		turn: int,
		tick: int,
		keyframe,
	):
		super().set_keyframe(characters, branch, turn, tick, keyframe)
		for graph, subkf in keyframe.items():
			super().set_keyframe(
				(*characters, graph), branch, turn, tick, subkf
			)
			if isinstance(subkf, dict):
				if isinstance(characters, tuple) and len(characters) == 1:
					characters = characters[0]
				for unit, is_unit in subkf.items():
					try:
						kf = self.user_cache.get_keyframe(
							(graph, unit), branch, turn, tick
						)
						kf[characters] = is_unit
					except KeyframeError:
						self.user_cache.set_keyframe(
							(graph, unit),
							branch,
							turn,
							tick,
							{characters: is_unit},
						)

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
		self.engine = engine
		self.name = name
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
			unhandl[turn] = list(self.iter_unhandled_rules(branch, turn, tick))
		try:
			unhandl[turn].remove(rule)
		except ValueError:
			pass

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
	def __init__(self, engine):
		super().__init__(engine, "node_rules_handled_cache")

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
	def __init__(self, engine):
		super().__init__(engine, "portal_rules_handled_cache")

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
	def __init__(self, db):
		Cache.__init__(self, db, name="things_cache")
		self._make_node = db.thing_cls

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
			if oldloc is not None:
				try:
					oldconts_orig = node_contents_cache.retrieve(
						character, oldloc, branch, turn, tick
					)
				except KeyError:
					oldconts_orig = frozenset()
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
		contra: bool = None,
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
