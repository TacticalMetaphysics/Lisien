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
import tempfile
from shutil import rmtree

import pytest

import lisien.tests.test_all
from lisien.engine import Engine
from lisien.tests.data import CHARACTER_UPDATES


class CharacterTest(lisien.tests.test_all.AllegedTest):
	def setUp(self):
		self.tempdir = tempfile.mkdtemp()
		self.engine = Engine(
			self.tempdir, enforce_end_of_time=False, workers=0
		)
		self.graphmakers = (self.engine.new_character,)

	def tearDown(self):
		self.engine.close()
		rmtree(self.tempdir)


def set_in_mapping(mapp, stat, v):
	"""Sync a value in ``mapp``, having key ``stat``, with ``v``."""
	# Mutate the stuff in-place instead of simply replacing it,
	# because this could trigger side effects
	if stat == "name":
		return
	if v is None:
		del mapp[stat]
		return
	if stat not in mapp:
		mapp[stat] = v
		return
	if isinstance(v, (dict, set)):
		mapp[stat].update(v)
		for item in list(mapp[stat]):
			if item not in v:
				try:
					del mapp[stat][item]
				except TypeError:
					mapp[stat].remove(item)
	elif isinstance(v, list):
		for item in list(mapp[stat]):
			if item not in v:
				mapp[stat].remove(item)
		for i, item in enumerate(v):
			if mapp[stat][i] != item:
				mapp[stat].insert(i, item)
	else:
		mapp[stat] = v


def update_char(char, *, stat=(), nodes=(), portals=()):
	"""Make a bunch of changes to a character-like object"""

	def update(d, dd):
		for k, v in dd.items():
			if v is None and k in d:
				del d[k]
			else:
				d[k] = v

	end_stats = dict(char.stat)
	for stat, v in stat:
		set_in_mapping(char.stat, stat, v)
		if v is None and stat in end_stats:
			del end_stats[stat]
		else:
			end_stats[stat] = v
	end_places = dict(char.place)
	end_things = dict(char.thing)
	for node, v in nodes:
		if v is None:
			del char.node[node]
			if node in end_places:
				del end_places[node]
			if node in end_things:
				del end_things[node]
		elif node in char.place:
			if "location" in v:
				del end_places[node]
				char.place2thing(node, v.pop("location"))
				if node in end_places:
					me = end_things[node] = end_places.pop(node)
				else:
					me = end_things[node] = dict(char.thing[node])
				update(me, v)
				for k, vv in v.items():
					set_in_mapping(char.thing[node], k, vv)
			else:
				if node in end_places:
					me = end_places[node]
				else:
					me = end_places[node] = dict(char.place[node])
				update(me, v)
				for k, vv in v.items():
					set_in_mapping(char.place[node], k, vv)
		elif node in char.thing:
			if "location" in v and v["location"] is None:
				if node in end_things:
					me = end_places[node] = end_things.pop(node)
				else:
					me = end_places[node] = dict(char.thing[node])
				del me["location"]
				del v["location"]
				char.thing2place(node)
				del v["location"]
				for k, vv in v.items():
					set_in_mapping(char.place[node], k, vv)
					set_in_mapping(end_places[node], k, vv)
			else:
				for k, vv in v.items():
					set_in_mapping(char.thing[node], k, vv)
					set_in_mapping(end_things[node], k, vv)
		elif "location" in v:
			end_things[node] = v
			me = char.new_thing(node, v.pop("location"))
			for k, vv in v.items():
				set_in_mapping(me, k, vv)
		else:
			end_places[node] = v
			me = char.new_node(node)
			for k, vv in v.items():
				set_in_mapping(me, k, vv)
	end_edges = {}
	for orig, dests in char.portal.items():
		end_edges[orig] = here = {}
		for dest, port in dests.items():
			here[dest] = dict(port)
	for o, d, v in portals:
		if v is None:
			del char.edge[o][d]
			del end_edges[o][d]
		else:
			me = end_edges.setdefault(o, {}).setdefault(d, {})
			update(me, v)
			e = char.new_portal(o, d)
			for k, vv in v.items():
				set_in_mapping(e, k, vv)
	return {
		"stat": end_stats,
		"place": end_places,
		"thing": end_things,
		"portal": end_edges,
	}


@pytest.mark.parametrize(
	["name", "data", "stat", "nodestat", "statup", "nodeup", "edgeup"],
	CHARACTER_UPDATES,
)
def test_char_creation(
	engy, name, data, stat, nodestat, statup, nodeup, edgeup
):
	char = engy.new_character(name, data, **stat)
	assert set(char.node) == set(data)
	es = set()
	for k, v in data.items():
		for vv in v:
			es.add((k, vv))
	assert set(char.edges) == es
	assert char.stat == stat


@pytest.fixture(params=CHARACTER_UPDATES)
def char_data(request):
	return request.param


def test_facade_creation(tmp_path, char_data):
	name, data, stat, nodestat, statup, nodeup, edgeup = char_data
	with Engine(tmp_path, workers=0) as eng:
		char = eng.new_character(name, data, **stat)
		fac = char.facade()
		assert dict(fac.node) == dict(char.node)
		assert fac.node == char.node
		assert fac.edges == char.edges
		assert set(fac.edges) == set(char.edges)
		assert fac.stat == char.stat
		assert dict(fac.stat) == dict(char.stat)


# TODO parametrize bunch of characters
@pytest.fixture(scope="function", params=CHARACTER_UPDATES)
def character_updates(request, sqleng):
	name, data, stat, nodestat, statup, nodeup, edgeup = request.param
	char = sqleng.new_character(name, data, **stat)
	update_char(char, nodes=nodestat)
	yield char, statup, nodeup, edgeup


def test_facade(character_updates):
	"""Make sure you can alter a facade independent of the character it's from"""
	character, statup, nodeup, edgeup = character_updates
	start_stat = dict(character.stat)
	start_place = dict(character.place)
	start_thing = dict(character.thing)
	start_edge = {}
	for o in character.edge:
		for d in character.edge[o]:
			start_edge.setdefault(o, {})[d] = dict(
				character.edge[o][d].items()
			)
	facade = character.facade()
	updated = update_char(facade, stat=statup, nodes=nodeup, portals=edgeup)
	assert facade.stat == updated["stat"]
	assert facade.place == updated["place"]
	assert facade.thing == updated["thing"]
	assert facade.portal == updated["portal"]
	# changes to a facade should not impact the underlying character
	assert start_stat == dict(character.stat)
	assert start_place == dict(character.place)
	assert start_thing == dict(character.thing)
	end_edge = {}
	for o in character.edge:
		for d in character.edge[o]:
			end_edge.setdefault(o, {})[d] = dict(character.edge[o][d])
	assert start_edge == end_edge


def test_set_rulebook(proxyless_engine):
	eng = proxyless_engine
	eng.universal["list"] = []
	ch = eng.new_character("physical")

	@ch.rule(always=True)
	def rule0(cha):
		cha.engine.universal["list"].append(0)

	@eng.rule(always=True)
	def rule1(who):
		who.engine.universal["list"].append(1)

	eng.rulebook["rb1"] = [rule1]
	eng.next_turn()
	assert eng.universal["list"] == [0]
	default_rulebook_name = ch.rulebook.name
	ch.rulebook = "rb1"
	eng.next_turn()
	assert eng.universal["list"] == [0, 1]
	ch.rulebook = eng.rulebook[default_rulebook_name]
	eng.next_turn()
	assert eng.universal["list"] == [0, 1, 0]


def test_iter_portals(sqleng):
	from lisien.character import grid_2d_8graph

	ch = sqleng.new_character("physical", grid_2d_8graph(4, 4))
	portal_abs = {
		(portal.origin.name, portal.destination.name)
		for portal in ch.portals()
	}
	for a, ayes in ch.adj.items():
		for b in ayes:
			assert (a, b) in portal_abs
	for a, b in portal_abs:
		assert a in ch.edge
		assert b in ch.edge[a]
