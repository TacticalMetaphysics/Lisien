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
from .allegedb.cache import (Cache, PickyDefaultDict, StructuredDefaultDict,
								TurnDict, WindowDict, HistoricKeyError)
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
			return  # because you can't rewind past this
		if prev == value:
			return  # not much point reporting on a non-change in a diff
		if turn in settings_turns or turn in settings_turns.future():
			assert turn in presettings_turns or turn in presettings_turns.future(
			)
			setticks = settings_turns[turn]
			presetticks = presettings_turns[turn]
			presetticks[tick] = parent + (entity, key, prev)
			setticks[tick] = parent + (entity, key, value)
		else:
			presettings_turns[turn] = {tick: parent + (entity, key, prev)}
			settings_turns[turn] = {tick: parent + (entity, key, value)}


class EntitylessCache(Cache):
	__slots__ = ()

	def store(self,
				key,
				branch,
				turn,
				tick,
				value,
				*,
				planning=None,
				forward=None,
				loading=False,
				contra=True):
		super().store(None,
						key,
						branch,
						turn,
						tick,
						value,
						planning=planning,
						forward=forward,
						loading=loading,
						contra=contra)

	def iter_entities_or_keys(self, branch, turn, tick, *, forward=None):
		return super().iter_entities_or_keys(None,
												branch,
												turn,
												tick,
												forward=forward)

	iter_entities = iter_keys = iter_entities_or_keys

	def contains_entity_or_key(self, ke, branch, turn, tick):
		return super().contains_entity_or_key(None, ke, branch, turn, tick)

	contains_entity = contains_key = contains_entity_or_key

	def retrieve(self, *args):
		return super().retrieve(*(None, ) + args)


class InitializedEntitylessCache(EntitylessCache, InitializedCache):
	__slots__ = ()


class UnitnessCache(Cache):
	"""A cache for remembering when a node is a unit of a character."""
	__slots__ = ('user_order', 'user_shallow', 'graphs', 'graph_units',
					'char_units', 'solo_unit', 'unique_unit', 'unique_graph',
					'users')

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

	def store(self,
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
				contra=True):
		is_unit = True if is_unit else None
		super().store(character,
						graph,
						node,
						branch,
						turn,
						tick,
						is_unit,
						planning=planning,
						forward=forward,
						loading=loading,
						contra=contra)
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
		charavs = self.char_units[character][branch]
		graphavs = self.graph_units[(character, graph)][branch]
		graphs = self.graphs[character][branch]
		uniqgraph = self.unique_graph[character][branch]
		soloav = self.solo_unit[(character, graph)][branch]
		uniqav = self.unique_unit[character][branch]
		users = self.users[graph, node][branch]

		def add_something(cache, what):
			if turn in cache:
				nucache = cache[turn][tick].copy()
				nucache.add(what)
				cache[turn][tick] = nucache
			elif cache.rev_gettable(turn):
				cacheturn = cache[turn]
				nucache = cacheturn[cacheturn.end].copy()
				nucache.add(what)
				cache[turn] = {tick: nucache}
			else:
				cache[turn] = {tick: {what}}

		def remove_something(cache, what):
			if turn in cache:
				nucache = cache[turn][tick].copy()
				nucache.remove(what)
				cache[turn][tick] = nucache
			elif cache.rev_gettable(turn):
				cacheturn = cache[turn]
				nucache = cacheturn[cacheturn.end].copy()
				nucache.remove(what)
				cache[turn] = {tick: nucache}
			else:
				raise ValueError

		if is_unit:
			add_something(graphavs, node)
			add_something(charavs, (graph, node))
			add_something(graphs, graph)
			add_something(users, character)
		else:
			remove_something(graphavs, node)
			remove_something(charavs, (graph, node))
			if not graphavs[turn][tick]:
				remove_something(graphs, graph)
			if not charavs[turn][tick]:
				remove_something(users, character)
		graphav = singleton_get(graphavs[turn][tick])
		if turn in soloav:
			soloav[turn][tick] = graphav
		else:
			soloav[turn] = {tick: graphav}
		charav = singleton_get(charavs[turn][tick])
		if turn in uniqav:
			uniqav[turn][tick] = charav
		else:
			uniqav[turn] = {tick: charav}
		if not graphavs[turn][tick]:
			if graph in graphs[turn][tick]:
				graphs[turn][tick].remove(graph)
				if len(graphs[turn][tick]) == 1:
					uniqgraph[turn][tick] = next(iter(graphs[turn][tick]))
				else:
					uniqgraph[turn][tick] = None
		if turn in graphavs and tick in graphavs[turn] and len(
			graphavs[turn][tick]) != 1:
			if turn in soloav:
				soloav[turn][tick] = None
			else:
				soloav[turn] = {tick: None}
		else:
			if turn in soloav:
				soloav[turn][tick] = node
			else:
				soloav[turn] = {tick: None}
		if turn in charavs and charavs[turn].rev_gettable(tick) and len(
			charavs[turn][tick]) != 1:
			if turn in uniqav:
				uniqav[turn][tick] = None
			else:
				uniqav[turn] = {tick: None}
		elif turn in uniqav:
			uniqav[turn][tick] = (graph, node)
		else:
			uniqav[turn] = {tick: (graph, node)}
		if turn in graphs and graphs[turn].rev_gettable(tick) and len(
			graphs[turn][tick]) != 1:
			if turn in uniqgraph:
				uniqgraph[turn][tick] = None
			else:
				uniqgraph[turn] = {tick: None}
		elif turn in uniqgraph:
			uniqgraph[turn][tick] = graph
		else:
			uniqgraph[turn] = {tick: graph}

	def get_char_graph_avs(self, char, graph, branch, turn, tick):
		return self._valcache_lookup(self.graph_units[(char, graph)], branch,
										turn, tick) or set()

	def get_char_graph_solo_av(self, char, graph, branch, turn, tick):
		return self._valcache_lookup(self.solo_unit[(char, graph)], branch,
										turn, tick)

	def get_char_only_av(self, char, branch, turn, tick):
		return self._valcache_lookup(self.unique_unit[char], branch, turn,
										tick)

	def get_char_only_graph(self, char, branch, turn, tick):
		return self._valcache_lookup(self.unique_graph[char], branch, turn,
										tick)

	def get_char_graphs(self, char, branch, turn, tick):
		return self._valcache_lookup(self.graphs[char], branch, turn,
										tick) or set()

	def _slow_iter_character_avatars(self, character, branch, turn, tick, *,
										forward):
		for graph in self.iter_entities(character,
										branch,
										turn,
										tick,
										forward=forward):
			for node in self.iter_entities(character,
											graph,
											branch,
											turn,
											tick,
											forward=forward):
				yield graph, node

	def _slow_iter_users(self, graph, node, branch, turn, tick):
		if graph not in self.user_order:
			return
		for character in self.user_order[graph][node]:
			if (graph, node, character, branch) not in self.user_shallow:
				for (b, t, tc) in self.db._iter_parent_btt(branch, turn, tick):
					if b in self.user_order[graph][node][character]:
						isav = self.user_order[graph][node][character][b][t]
						# side effect!! bad!
						self.store(character, graph, node, branch, turn, tick,
									isav[tc])
						break
				else:
					self.store(character, graph, node, branch, turn, tick,
								None)
			try:
				if self.user_shallow[(graph, node, character,
										branch)][turn][tick]:
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
		self.handled.setdefault(entity + (rulebook, branch, turn),
								set()).add(rule)
		self.handled_deep[branch][turn][tick] = (entity, rulebook, rule)
		unhandl = self.unhandled.setdefault(entity, {}).setdefault(
			rulebook, {}).setdefault(branch, {})
		if turn not in unhandl:
			unhandl[turn] = list(
				self.unhandled_rulebook_rules(entity, rulebook, branch, turn,
												tick))
		try:
			unhandl[turn].remove(rule)
		except ValueError:
			if not loading:
				raise

	def fork(self, branch, turn, tick):
		parent_branch, parent_turn, parent_tick, end_turn, end_tick = self.engine._branches[
			branch]
		unhandl = self.unhandled
		handl = self.handled
		rbcache = self.engine._rulebooks_cache
		unhandrr = self.unhandled_rulebook_rules
		getrb = self.get_rulebook
		for entity in unhandl:
			rulebook = getrb(*entity + (branch, turn, tick))
			if entity + (rulebook, branch, turn) in handl:
				raise HistoricKeyError(
					"Tried to fork history in a RulesHandledCache, "
					"but it seems like rules have already been run where we're "
					"forking to")
			rules = set(rbcache.retrieve(rulebook, branch, turn, tick))
			unhandled_rules = unhandrr(entity, rulebook, parent_branch,
										parent_turn, tick)
			unhandled_rules_set = set(unhandled_rules)
			handl[entity + (rulebook, branch,
							turn)] = rules.difference(unhandled_rules_set)
			unhandl[entity][rulebook][branch][turn] = unhandled_rules

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
		rulebook_rules = rbc.retrieve(rulebook, branch, turn, tick)
		handled_rules = self.handled.setdefault(
			entity + (rulebook, branch, turn), set())
		return [rule for rule in rulebook_rules if rule not in handled_rules]


class CharacterRulesHandledCache(RulesHandledCache):

	def get_rulebook(self, character, branch, turn, tick):
		try:
			return self.engine._characters_rulebooks_cache.retrieve(
				character, branch, turn, tick)
		except KeyError:
			return character, 'character'

	def iter_unhandled_rules(self, branch, turn, tick):
		for character in sort_set(self.engine.character.keys()):
			rb = self.get_rulebook(character, branch, turn, tick)
			for rule in self.unhandled_rulebook_rules(character, rb, branch,
														turn, tick):
				yield character, rb, rule


class UnitRulesHandledCache(RulesHandledCache):

	def get_rulebook(self, character, branch, turn, tick):
		try:
			return self.engine._units_rulebooks_cache.retrieve(
				character, branch, turn, tick)
		except KeyError:
			return character, 'unit'

	def iter_unhandled_rules(self, branch, turn, tick):
		charm = self.engine.character
		for character in sort_set(charm.keys()):
			rulebook = self.get_rulebook(character, branch, turn, tick)
			charavm = charm[character].unit
			for graph in sort_set(charavm.keys()):
				for avatar in sort_set(charavm[graph].keys()):
					try:
						rules = self.unhandled_rulebook_rules(
							character, graph, avatar, rulebook, branch, turn,
							tick)
					except KeyError:
						continue
					for rule in rules:
						yield character, graph, avatar, rulebook, rule


class CharacterThingRulesHandledCache(RulesHandledCache):

	def get_rulebook(self, character, branch, turn, tick):
		try:
			return self.engine._characters_things_rulebooks_cache.retrieve(
				character, branch, turn, tick)
		except KeyError:
			return character, 'character_thing'

	def iter_unhandled_rules(self, branch, turn, tick):
		charm = self.engine.character
		for character in sort_set(charm.keys()):
			rulebook = self.get_rulebook(character, branch, turn, tick)
			things = sort_set(charm[character].thing.keys())
			pass
			for thing in things:
				try:
					rules = self.unhandled_rulebook_rules(
						character, thing, rulebook, branch, turn, tick)
				except KeyError:
					continue
				for rule in rules:
					yield character, thing, rulebook, rule


class CharacterPlaceRulesHandledCache(RulesHandledCache):

	def get_rulebook(self, character, branch, turn, tick):
		try:
			return self.engine._characters_places_rulebooks_cache.retrieve(
				character, branch, turn, tick)
		except KeyError:
			return character, 'character_place'

	def iter_unhandled_rules(self, branch, turn, tick):
		charm = self.engine.character
		for character in sort_set(charm.keys()):
			rulebook = self.get_rulebook(character, branch, turn, tick)
			for place in sort_set(charm[character].place.keys()):
				try:
					rules = self.unhandled_rulebook_rules(
						character, place, rulebook, branch, turn, tick)
				except KeyError:
					continue
				for rule in rules:
					yield character, place, rulebook, rule


class CharacterPortalRulesHandledCache(RulesHandledCache):

	def get_rulebook(self, character, branch, turn, tick):
		try:
			return self.engine._characters_portals_rulebooks_cache.retrieve(
				character, branch, turn, tick)
		except KeyError:
			return character, 'character_portal'

	def iter_unhandled_rules(self, branch, turn, tick):
		charm = self.engine.character
		for character in sort_set(charm.keys()):
			try:
				rulebook = self.get_rulebook(character, branch, turn, tick)
			except KeyError:
				continue
			try:
				rulebook_rules = self.engine._rulebooks_cache.retrieve(
					rulebook, branch, turn, tick)
			except KeyError:
				continue
			if not rulebook_rules:
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
					try:
						rules = self.unhandled_rulebook_rules(
							character, orig, dest, rulebook, branch, turn,
							tick)
					except KeyError:
						continue
					for rule in rules:
						yield character, orig, dest, rulebook, rule


class NodeRulesHandledCache(RulesHandledCache):

	def get_rulebook(self, character, node, branch, turn, tick):
		try:
			return self.engine._nodes_rulebooks_cache.retrieve(
				character, node, branch, turn, tick)
		except KeyError:
			return character, node

	def iter_unhandled_rules(self, branch, turn, tick):
		charm = self.engine.character
		nodes_with_rulebook_changed = {
			(character, node)
			for (character, node, _, _,
					_) in self.engine._nodes_rulebooks_cache.shallowest
		}
		nodes_with_filled_default_rulebooks = {
			(character, node)
			for (_, (character, node)) in (
				{(None, (character, node))
					for character in charm for node in charm[character].node}
				& self.engine._rulebooks_cache.branches.keys())
		}
		for (character,
				node) in sort_set(nodes_with_rulebook_changed
									| nodes_with_filled_default_rulebooks):
			try:
				rulebook = self.get_rulebook(character, node, branch, turn,
												tick)
				rules = self.unhandled_rulebook_rules(character, node,
														rulebook, branch, turn,
														tick)
			except KeyError:
				continue
			for rule in rules:
				yield character, node, rulebook, rule


class PortalRulesHandledCache(RulesHandledCache):

	def get_rulebook(self, character, orig, dest, branch, turn, tick):
		try:
			return self.engine._portals_rulebooks_cache.retrieve(
				character, orig, dest, branch, turn, tick)
		except KeyError:
			return character, orig, dest

	def iter_unhandled_rules(self, branch, turn, tick):
		portals_with_rulebook_changed = {
			(character, orig, dest)
			for (character, orig, dest, _, _,
					_) in self.engine._portals_rulebooks_cache.shallowest
		}
		portals_default_rulebooks = {
			(None, port)
			for port in self.engine._edges_cache.keys
		}
		portals_with_filled_default_rulebooks = {
			(character, orig, dest)
			for (_, (character, orig, dest)) in (
				portals_default_rulebooks
				& self.engine._rulebooks_cache.branches.keys())
		}
		for (character, orig,
				dest) in (portals_with_rulebook_changed
							| portals_with_filled_default_rulebooks):
			try:
				rulebook = self.get_rulebook(character, orig, dest, branch,
												turn, tick)
				rules = self.unhandled_rulebook_rules(character, orig, dest,
														rulebook, branch, turn,
														tick)
			except KeyError:
				continue
			for rule in rules:
				yield character, orig, dest, rulebook, rule


class ThingsCache(Cache):

	def __init__(self, db):
		Cache.__init__(self, db)
		self._make_node = db.thing_cls

	def store(self, *args, planning=None, loading=False, contra=True):
		character, thing, branch, turn, tick, location = args
		try:
			oldloc = self.retrieve(character, thing, branch, turn, tick)
		except KeyError:
			oldloc = None
		super().store(*args, planning=planning, loading=loading, contra=contra)
		node_contents_cache = self.db._node_contents_cache
		# Cache the contents of nodes
		if oldloc is not None:
			oldconts_orig = node_contents_cache.retrieve(
				character, oldloc, branch, turn, tick)
			newconts_orig = oldconts_orig.difference({thing})
			node_contents_cache.store(character,
										oldloc,
										branch,
										turn,
										tick,
										newconts_orig,
										contra=False,
										loading=True)
			future_location_data = node_contents_cache.settings[branch].future(
				turn)
			todo = []
			for trn in future_location_data:
				for tck in future_location_data[trn]:
					char, loca, contents = future_location_data[trn][tck]
					if char == character and loca == oldloc:
						todo.append((trn, tck))
			for trn, tck in todo:
				node_contents_cache.store(character,
											oldloc,
											branch,
											trn,
											tck,
											node_contents_cache.retrieve(
												character, oldloc, branch, trn,
												tck).difference({thing}),
											planning=False,
											contra=False,
											loading=True)
		if location is not None:
			try:
				oldconts_dest = node_contents_cache.retrieve(
					character, location, branch, turn, tick)
			except KeyError:
				oldconts_dest = frozenset()
			newconts_dest = oldconts_dest.union({thing})
			node_contents_cache.store(character,
										location,
										branch,
										turn,
										tick,
										newconts_dest,
										contra=False,
										loading=True)
			future_location_data = node_contents_cache.settings[branch].future(
				turn)
			todo = []
			for trn in future_location_data:
				for tck in sorted(future_location_data[trn]):
					char, loca, contents = future_location_data[trn][tck]
					if char == character and loca == location:
						todo.append((trn, tck))
			for trn, tck in todo:
				node_contents_cache.store(character,
											location,
											branch,
											trn,
											tck,
											node_contents_cache.retrieve(
												character, location, branch,
												trn, tck).union({thing}),
											planning=False,
											contra=False,
											loading=True)

	def turn_before(self, character, thing, branch, turn):
		try:
			self.retrieve(character, thing, branch, turn, 0)
		except KeyError:
			pass
		return self.keys[(character, )][thing][branch].rev_before(turn)

	def turn_after(self, character, thing, branch, turn):
		try:
			self.retrieve(character, thing, branch, turn, 0)
		except KeyError:
			pass
		return self.keys[(character, )][thing][branch].rev_after(turn)


class NodeContentsCache(Cache):

	def _iter_future_contradictions(self, entity, key, turns, branch, turn,
									tick, value):
		return self.db._things_cache._iter_future_contradictions(
			entity, key, turns, branch, turn, tick, value)

	def slow_iter_contents(self, character, place, branch, turn, tick):
		branch_now, turn_now, tick_now = self.db._btt()
		self.db.time = branch, turn
		self.db.tick = tick
		for thing in self.db.character[character].thing.values():
			if thing['location'] == place:
				yield thing.name
		self.db.time = branch_now, turn_now
		self.db.tick = tick_now

	def remove(self, branch, turn, tick):
		"""Delete data on or after this tick

		On the assumption that the future has been invalidated.

		"""
		for parent, entitys in list(self.parents.items()):
			for entity, keys in list(entitys.items()):
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
					del entitys[entity]
			if not entitys:
				del self.parents[parent]
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
						del branches[branch]
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

	def truncate_loc(self, character, location, branch, turn, tick):
		"""Remove future data about a particular location

		Return True if I deleted anything, False otherwise.

		"""
		r = False
		branches_turns = self.branches[character, location][branch]
		branches_turns.truncate(turn)
		if turn in branches_turns:
			bttrn = branches_turns[turn]
			if bttrn.future(tick):
				bttrn.truncate(tick)
				r = True
		keyses = self.keys[character, location]
		for keysbranches in keyses.values():
			if branch not in keysbranches:
				continue
			keysbranch = keysbranches[branch]
			if keysbranch.future(turn):
				keysbranch.truncate(turn)
				r = True
			if turn in keysbranch:
				keysbranchturn = keysbranch[turn]
				if keysbranchturn.future(tick):
					keysbranchturn.truncate(tick)
					r = True
		if branch in self.settings:
			for sets in (self.settings, self.presettings):
				sets_branch = sets[branch]
				if turn in sets_branch:
					sets_turn = sets_branch[turn]
					for tic, setting in list(sets_turn.future(tick).items()):
						if setting[:2] == (character, location):
							del sets_turn[tic]
							r = True
					if not sets_turn:
						del sets_branch[turn]
						assert r, "Found an empty cache when I didn't delete anything"
				for trn, tics in list(sets_branch.future(turn).items()):
					for tic, setting in list(tics.future(tick).items()):
						if setting[:2] == (character, location):
							del tics[tic]
							r = True
					if not tics:
						del sets_branch[trn]
						assert r, "Found an empty cache when I didn't delete anything"
				if not sets_branch:
					del sets[branch]
					assert r, "Found an empty cache when I didn't delete anything"
		self.shallowest = OrderedDict()
		return r
