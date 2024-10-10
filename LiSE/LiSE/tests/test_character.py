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
from shutil import rmtree
import tempfile
import pytest
import LiSE.allegedb.tests.test_all
from LiSE.engine import Engine


class CharacterTest(LiSE.allegedb.tests.test_all.AllegedTest):
	def setUp(self):
		self.tempdir = tempfile.mkdtemp()
		self.engine = Engine(self.tempdir, enforce_end_of_time=False)
		self.graphmakers = (self.engine.new_character,)

	def tearDown(self):
		self.engine.close()
		rmtree(self.tempdir)


class CharacterBranchLineageTest(
	CharacterTest, LiSE.allegedb.tests.test_all.AbstractBranchLineageTest
):
	pass


class CharacterDictStorageTest(
	CharacterTest, LiSE.allegedb.tests.test_all.DictStorageTest
):
	pass


class CharacterListStorageTest(
	CharacterTest, LiSE.allegedb.tests.test_all.ListStorageTest
):
	pass


class CharacterSetStorageTest(
	CharacterTest, LiSE.allegedb.tests.test_all.SetStorageTest
):
	pass


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


def update_char(char, *, stat=(), node=(), portal=()):
	"""Make a bunch of changes to a character-like object"""

	def update(d, dd):
		for k, v in dd.items():
			if v is None and k in d:
				del d[k]
			else:
				d[k] = v

	end_stats = char.stat.unwrap()
	for stat, v in stat:
		set_in_mapping(char.stat, stat, v)
		if v is None and stat in end_stats:
			del end_stats[stat]
		else:
			end_stats[stat] = v
	end_places = char.place.unwrap()
	end_things = char.thing.unwrap()
	for node, v in node:
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
				update(me, v)
				for k, vv in v.items():
					set_in_mapping(char.place[node], k, vv)
			else:
				me = end_things[node] = dict(char.thing[node])
				update(me, v)
				for k, vv in v.items():
					set_in_mapping(char.thing[node], k, vv)
		elif "location" in v:
			end_things[node] = v
			me = char.new_thing(node, v.pop("location"))
			for k, vv in v.items():
				set_in_mapping(me, k, vv)
		else:
			v["name"] = node
			end_places[node] = v
			me = char.new_node(node)
			for k, vv in v.items():
				set_in_mapping(me, k, vv)
	end_edges = char.portal.unwrap()
	for o, d, v in portal:
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


CHAR_DATA = [
	("empty", {}, {}, [], [], [], []),
	(
		"small",
		{0: [1], 1: [0], "kobold": []},
		{
			"spam": "eggs",
			"ham": {"baked beans": "delicious"},
			"qux": ["quux", "quuux"],
			"clothes": {"hats", "shirts", "pants"},
		},
		[
			("kobold", {"location": 0, "evil": True}),
			(0, {"evil": False}),
			(1, {"evil": False}),
		],
		[("spam", None), ("qux", ["quux"]), ("clothes", "no")],
		[(2, {"evil": False}), ("kobold", {"evil": False})],
		[(0, 1, None), (0, 2, {"hi": "hello"})],
	),
]


@pytest.mark.parametrize(
	["name", "data", "stat", "nodestat", "statup", "nodeup", "edgeup"],
	CHAR_DATA,
)
def test_char_creation(
	tmpdir, name, data, stat, nodestat, statup, nodeup, edgeup
):
	with Engine(tmpdir) as eng:
		char = eng.new_character(name, data, **stat)
		assert set(char.node) == set(data)
		es = set()
		for k, v in data.items():
			for vv in v:
				es.add((k, vv))
		assert set(char.edges) == es
		assert char.stat == stat


@pytest.mark.parametrize(
	["name", "data", "stat", "nodestat", "statup", "nodeup", "edgeup"],
	CHAR_DATA,
)
def test_facade_creation(
	tmpdir, name, data, stat, nodestat, statup, nodeup, edgeup
):
	with Engine(tmpdir) as eng:
		char = eng.new_character(name, data, **stat)
		fac = char.facade()
		assert dict(fac.node) == dict(char.node)
		assert fac.node == char.node
		assert fac.edges == char.edges
		assert set(fac.edges) == set(char.edges)
		assert fac.stat == char.stat
		assert dict(fac.stat) == dict(char.stat)


# TODO parametrize bunch of characters
@pytest.fixture(scope="function", params=CHAR_DATA)
def character_updates(request, engy):
	name, data, stat, nodestat, statup, nodeup, edgeup = request.param
	char = engy.new_character(name, data, **stat)
	update_char(char, node=nodestat)
	yield char, statup, nodeup, edgeup


def test_facade(character_updates):
	"""Make sure you can alter a facade independent of the character it's from"""
	character, statup, nodeup, edgeup = character_updates
	start_stat = character.stat.unwrap()
	start_place = character.place.unwrap()
	start_thing = character.thing.unwrap()
	start_edge = {}
	for o in character.edge:
		for d in character.edge[o]:
			start_edge.setdefault(o, {})[d] = character.edge[o][d].unwrap()
	facade = character.facade()
	updated = update_char(facade, stat=statup, node=nodeup, portal=edgeup)
	assert facade.stat == updated["stat"]
	assert facade.place == updated["place"]
	assert facade.thing == updated["thing"]
	assert facade.portal == updated["portal"]
	# changes to a facade should not impact the underlying character
	assert start_stat == character.stat.unwrap()
	assert start_place == character.place.unwrap()
	assert start_thing == character.thing.unwrap()
	end_edge = {}
	for o in character.edge:
		for d in character.edge[o]:
			end_edge.setdefault(o, {})[d] = dict(character.edge[o][d])
	assert start_edge == end_edge


def test_set_rulebook(engy):
	engy.universal["list"] = []
	ch = engy.new_character("physical")

	@ch.rule(always=True)
	def rule0(cha):
		cha.engine.universal["list"].append(0)

	@engy.rule(always=True)
	def rule1(who):
		who.engine.universal["list"].append(1)

	engy.rulebook["rb1"] = [rule1]
	engy.next_turn()
	assert engy.universal["list"] == [0]
	ch.rulebook = "rb1"
	engy.next_turn()
	assert engy.universal["list"] == [0, 1]
	ch.rulebook = engy.rulebook["physical", "character"]
	engy.next_turn()
	assert engy.universal["list"] == [0, 1, 0]


def test_iter_portals(engy):
	from LiSE.character import grid_2d_8graph

	ch = engy.new_character("physical", grid_2d_8graph(4, 4))
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
