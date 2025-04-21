# This file is part of Elide, frontend to Lisien, a framework for life simulation games.
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
from unittest.mock import call, patch

import networkx as nx
import pytest

from lisien.engine import Engine
from lisien.examples.kobold import inittest
from lisien.tests.util import make_test_engine

testgraphs = [nx.chvatal_graph()]
# have to name it after creation because it clears the create_using
path_graph_9 = nx.path_graph(9)
path_graph_9.name = "path_graph_9"
testgraphs.append(path_graph_9)


@pytest.fixture
def db(tmp_path, execution, database):
	with make_test_engine(tmp_path, execution, database) as orm:
		for graph in testgraphs:
			orm.new_character(graph.name, graph)
			if not graph.is_directed():
				graph = nx.to_directed(graph)
			assert set(graph.nodes.keys()) == set(
				orm.character[graph.name].nodes.keys()
			), "{}'s nodes changed during instantiation".format(graph.name)
			assert set(graph.edges) == set(
				orm.character[graph.name].edges.keys()
			), "{}'s edges changed during instantiation".format(graph.name)
		yield orm


def test_basic_load(db):
	for graph in testgraphs:
		if not graph.is_directed():
			graph = nx.to_directed(graph)
		alleged = db.character[graph.name]
		assert set(graph.nodes.keys()) == set(alleged.nodes.keys()), (
			"{}'s nodes are not the same after load".format(graph.name)
		)
		assert set(graph.edges) == set(alleged.edges), (
			"{}'s edges are not the same after load".format(graph.name)
		)


def test_keyframe_load(db):
	for graph in testgraphs:
		nodes_kf = db._nodes_cache.keyframe
		assert (graph.name,) in nodes_kf, "{} not in nodes cache".format(
			graph.name
		)
		assert "trunk" in nodes_kf[graph.name,], (
			"trunk branch not in nodes cache for {}".format(graph.name)
		)
		assert nodes_kf[graph.name,]["trunk"].rev_gettable(0), (
			"turn 0 not in nodes cache for {}".format(graph.name)
		)
		assert nodes_kf[graph.name,]["trunk"][0].rev_gettable(0), (
			"tick 0 not in nodes cache for {}".format(graph.name)
		)
		assert db._nodes_cache.keyframe[graph.name,]["trunk"][0][0] == {
			node: True for node in graph.nodes.keys()
		}, "{} not loaded correctly, got {}".format(
			graph.name, nodes_kf["trunk"][0][0]
		)
		edges_kf = db._edges_cache.keyframe
		if graph.is_multigraph():
			for orig in graph.adj:
				for dest in graph.adj[orig]:
					assert (
						graph.name,
						orig,
						dest,
					) in edges_kf, "{} not in edges cache".format(
						(graph.name, orig, dest)
					)
					this_edge = edges_kf[graph.name, orig, dest]
					assert "trunk" in this_edge, (
						"trunk branch not in edges cache for {}".format(
							(graph.name, orig, dest)
						)
					)
					assert this_edge["trunk"].rev_gettable(0), (
						"turn 0 not in trunk branch of edges cache for {}".format(
							(graph.name, orig, dest)
						)
					)
					assert this_edge["trunk"][0].rev_gettable(0), (
						"tick 0 not in turn 0 of trunk branch of edges cache for {}".format(
							(graph.name, orig, dest)
						)
					)
					assert db._edges_cache.keyframe[graph.name, orig, dest][
						"trunk"
					][0][0] == {idx: True for idx in graph.adj[orig][dest]}, (
						"{} not loaded".format((graph.name, orig, dest))
					)
		else:
			for orig in graph.adj:
				for dest in graph.adj[orig]:
					assert (
						graph.name,
						orig,
						dest,
					) in edges_kf, "{} not in edges cache".format(
						(graph.name, orig, dest)
					)
					this_edge = edges_kf[graph.name, orig, dest]
					assert "trunk" in this_edge, (
						"trunk branch not in edges cache for {}".format(
							(graph.name, orig, dest)
						)
					)
					assert this_edge["trunk"].rev_gettable(0), (
						"turn 0 not in trunk branch of edges cache for {}".format(
							(graph.name, orig, dest)
						)
					)
					assert this_edge["trunk"][0].rev_gettable(0), (
						"tick 0 not in turn 0 of trunk branch of edges cache for {}".format(
							(graph.name, orig, dest)
						)
					)
					assert db._edges_cache.keyframe[graph.name, orig, dest][
						"trunk"
					][0][0] == {0: True}, "{} not loaded".format(
						(graph.name, orig, dest)
					)
		for node, vals in graph.nodes.items():
			assert (
				db._node_val_cache.keyframe[graph.name, node]["trunk"][0][0]
				== vals
			)
		for edge in graph.edges:
			if graph.is_multigraph():
				assert (
					db._edge_val_cache.keyframe[(graph.name,) + edge]["trunk"][
						0
					][0]
					== graph.edges[edge]
				)
			else:
				assert (
					db._edge_val_cache.keyframe[(graph.name,) + edge + (0,)][
						"trunk"
					][0][0]
					== graph.edges[edge]
				)


def test_keyframe_unload(tmp_path, execution, non_null_database):
	# TODO: test edge cases involving tick-precise unloads
	eng = partial(make_test_engine, tmp_path, execution, non_null_database)
	with eng() as orm:
		g = orm.new_character("g", nx.grid_2d_graph(3, 3))
		orm.next_turn()
		assert orm.turn == 1
		assert (
			"g",
			(0, 0),
			(0, 1),
		) in orm._edges_cache.keyframe and 0 in orm._edges_cache.keyframe[
			"g", (0, 0), (0, 1)
		]["trunk"]
		del g.node[1, 1]
		g.add_node("a")
		g.add_edge((0, 0), "a")
		orm.next_turn()
		assert orm.turn == 2
		orm.snap_keyframe()
		g.add_node((4, 4))
		g.add_edge((3, 3), (4, 4))
		assert (
			"g",
			(0, 0),
			(0, 1),
		) in orm._edges_cache.keyframe and 0 in orm._edges_cache.keyframe[
			"g", (0, 0), (0, 1)
		]["trunk"]
		assert (
			("g",) in orm._nodes_cache.keyframe
			and "trunk" in orm._nodes_cache.keyframe["g",]
			and 0 in orm._nodes_cache.keyframe["g",]["trunk"]
		)
		orm.unload()
		assert not orm._time_is_loaded("trunk", 1)
		if "trunk" in orm._nodes_cache.keyframe["g",]:
			assert 0 not in orm._nodes_cache.keyframe["g",]["trunk"]
		assert ("g", (0, 0), (0, 1)) in orm._edges_cache.keyframe
		assert "trunk" in orm._edges_cache.keyframe["g", (0, 0), (0, 1)]
		assert 2 in orm._edges_cache.keyframe["g", (0, 0), (0, 1)]["trunk"]
		assert 0 not in orm._edges_cache.keyframe["g", (0, 0), (0, 1)]["trunk"]
		endtick = orm.tick
	with eng() as orm:
		assert not orm._time_is_loaded("trunk", 1)
		assert orm._time_is_loaded("trunk", 2, endtick)
		assert ("g", (0, 0), (0, 1)) in orm._edges_cache.keyframe
		assert 2 in orm._edges_cache.keyframe["g", (0, 0), (0, 1)]["trunk"]
		assert 0 not in orm._edges_cache.keyframe["g", (0, 0), (0, 1)]["trunk"]
		g = orm.character["g"]
		if "trunk" in orm._nodes_cache.keyframe["g",]:
			assert 0 not in orm._nodes_cache.keyframe["g",]["trunk"]
		if (
			("g", (0, 0), (0, 1)) in orm._edges_cache.keyframe
			and "trunk" in orm._edges_cache.keyframe["g", (0, 0), (0, 1)]
		):
			assert (
				0
				not in orm._edges_cache.keyframe["g", (0, 0), (0, 1)]["trunk"]
			)
		assert not orm._time_is_loaded("trunk", 1)
		orm.turn = 0
		assert orm._time_is_loaded("trunk", 1)
		assert 0 in orm._edges_cache.keyframe["g", (0, 0), (0, 1)]["trunk"]
		orm.branch = "u"
		del g.node[1, 2]
		orm.unload()
	with eng() as orm:
		assert orm.branch == "u"
		assert (
			("g", (1, 1), (1, 2)) not in orm._edges_cache.keyframe
			or "trunk" not in orm._edges_cache.keyframe["g", (1, 1), (1, 2)]
		)
		g = orm.character["g"]
		assert (1, 2) not in g.nodes
		orm.branch = "trunk"
		assert (1, 2) in g.nodes
		assert (
			("g", (1, 1), (1, 2)) in orm._edges_cache.keyframe
			and "trunk" in orm._edges_cache.keyframe["g", (1, 1), (1, 2)]
		)


def test_keyframe_load_init(tmp_path):
	"""Can load a keyframe at start of branch, including locations"""
	eng = Engine(tmp_path, workers=0)
	inittest(eng)
	eng.branch = "new"
	# eng.snap_keyframe()
	eng.close()
	eng = Engine(tmp_path, workers=0)
	# the graphs keyframe is coming up empty
	assert "kobold" in eng.character["physical"].thing
	assert (0, 0) in eng.character["physical"].place
	assert (0, 1) in eng.character["physical"].portal[0, 0]
	eng.close()


def test_multi_keyframe(tmp_path):
	eng = Engine(
		tmp_path, enforce_end_of_time=False, keyframe_on_close=False, workers=0
	)
	inittest(eng)
	eng.snap_keyframe()
	tick0 = eng.tick
	eng.turn = 1
	del eng.character["physical"].place[3, 3]
	eng.snap_keyframe()
	tick1 = eng.tick
	assert ("physical",) in eng._nodes_cache.keyframe
	assert "trunk" in eng._nodes_cache.keyframe["physical",]
	assert 1 in eng._nodes_cache.keyframe["physical",]["trunk"]
	assert tick1 in eng._nodes_cache.keyframe["physical",]["trunk"][1]
	assert (1, 1) in eng._nodes_cache.keyframe["physical",]["trunk"][1][tick1]
	assert (3, 3) not in eng._nodes_cache.keyframe["physical",]["trunk"][1][
		tick1
	]
	eng.close()
	eng = Engine(tmp_path, keyframe_on_close=False, workers=0)
	eng.load_at("trunk", 0, tick0)
	assert eng._time_is_loaded("trunk", 0, tick0)
	assert eng._time_is_loaded("trunk", 0, tick0 + 1)
	assert eng._time_is_loaded("trunk", 1, tick1 - 1)
	assert eng._time_is_loaded("trunk", 1, tick1)
	eng.close()


def test_keyframe_load_unload(tmp_path, non_null_database):
	"""All caches can load and unload before and after kfs"""
	if non_null_database == "sqlite":
		connect_str = f"sqlite:///{tmp_path}/world.sqlite3"
	else:
		connect_str = None
	with Engine(
		tmp_path,
		enforce_end_of_time=False,
		keyframe_on_close=False,
		workers=0,
		connect_string=connect_str,
	) as eng:
		eng.snap_keyframe()
		eng.turn = 1
		inittest(eng)
		eng.snap_keyframe()
		eng.turn = 2
		eng.universal["hi"] = "hello"
		now = eng._btt()
	with Engine(
		tmp_path,
		enforce_end_of_time=False,
		keyframe_on_close=False,
		workers=0,
		connect_string=connect_str,
	) as eng:
		assert eng._time_is_loaded(*now)
		assert not eng._time_is_loaded("trunk", 0)
		eng.turn = 1
		eng.tick = 0
		assert eng._time_is_loaded("trunk", 1)
		assert eng._time_is_loaded("trunk", 1, 0)
		assert eng._time_is_loaded(*now)
		eng.unload()
		assert eng._time_is_loaded("trunk", 1, 0)
		assert not eng._time_is_loaded(*now)
		eng.turn = 2
		eng.branch = "haha"
		eng.snap_keyframe()
		eng.unload()
		assert not eng._time_is_loaded("trunk")


@pytest.fixture
def some_state(tmp_path):
	with Engine(tmp_path, workers=0, random_seed=0) as eng:
		initial_state = nx.DiGraph(
			{
				0: {1: {"omg": "lol"}},
				1: {0: {"omg": "blasphemy"}},
				2: {},
				3: {},
				"it": {},
			}
		)
		initial_state.nodes()[2]["hi"] = "hello"
		initial_state.nodes()["it"]["location"] = 0
		initial_state.graph["wat"] = "nope"
		phys = eng.new_character("physical", initial_state)
		eng.add_character("pointless")
		kf0 = eng.snap_keyframe()
		del kf0["universal"]["rando_state"]
		eng.branch = "b"
		kf1 = eng.snap_keyframe()
		del kf1["universal"]["rando_state"]
		assert kf0 == kf1
		del phys.portal[1][0]
		port = phys.new_portal(0, 2)
		port["hi"] = "bye"
		phys.place[1]["wtf"] = "bbq"
		phys.thing["it"].location = phys.place[1]
		del phys.place[3]
		eng.add_character("pointed")
		del eng.character["pointless"]
		assert "pointless" not in eng.character, "Failed to delete character"
		phys.portal[0][1]["meaning"] = 42
		del phys.portal[0][1]["omg"]
		eng.branch = "trunk"
	return tmp_path


def test_load_branch_to_end(some_state):
	with Engine(some_state, workers=0, random_seed=0) as eng:
		assert eng.turn == 0
		phys = eng.character["physical"]
		assert 3 in phys.place
		assert phys.portal[1][0]["omg"] == "blasphemy"
		eng.branch = "b"
		assert 3 not in phys.place
		assert 0 not in phys.portal[1]
		assert 2 in phys.portal[0]
		assert phys.portal[0][2]["hi"] == "bye"
		assert phys.place[1]["wtf"] == "bbq"
		assert phys.thing["it"].location == phys.place[1]
		assert "pointless" not in eng.character, "Loaded deleted character"
		assert "pointed" in eng.character
		assert phys.portal[0][1]["meaning"] == 42
		assert "omg" not in phys.portal[0][1]
