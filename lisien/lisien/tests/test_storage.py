# This file is part of allegedb, an object-relational mapper for versioned graphs.
# Copyright (c) Zachary Spector. public@zacharyspector.com
from copy import deepcopy

import pytest

from lisien.engine import Engine

testkvs = [
	0,
	1,
	10,
	10**10,
	"spam",
	"eggs",
	"ham",
	"💧",
	"🔑",
	"𐦖",
	("spam", "eggs", "ham"),
]
testvs = [["spam", "eggs", "ham"], {"foo": "bar", 0: 1, "💧": "🔑"}]
testdata = []
for k in testkvs:
	for v in testkvs:
		testdata.append((k, v))
	for v in testvs:
		testdata.append((k, v))
testdata.append(("lol", deepcopy(testdata)))


def graph_objects_create_delete(engine: Engine):
	g = engine.new_character("physical")
	assert not engine._node_exists("physical", 0)
	g.add_node(0)
	assert engine._node_exists("physical", 0)
	assert 0 in g
	g.add_node(1)
	assert 1 in g
	g.add_edge(0, 1)
	assert 1 in g.adj[0]
	assert 1 in list(g.adj[0])
	g.add_edge(2, 3)
	assert 2 in g.node
	assert 3 in g.node
	assert 2 in g.adj
	assert 3 in g.adj[2]
	assert 3 in list(g.adj[2])
	assert 2 in g.pred[3]
	g.add_edge(2, 4)
	assert 2 in g.pred[4]
	assert 2 in list(g.pred[4])
	assert 4 in g.adj[2]
	assert 4 in list(g.adj[2])
	del g.pred[4]
	assert len(g.pred[4]) == 0
	assert 4 not in g.adj[2]
	assert 4 not in list(g.adj[2])
	assert 4 in g.node
	assert 0 not in g.adj[1]
	assert 0 not in list(g.adj[1])
	engine.next_turn()
	assert 0 in g
	assert 1 in g
	engine.branch = "physical_no_edge"
	assert 3 in g.node
	assert 0 in g
	assert engine._node_exists("physical", 0)
	assert 1 in g
	assert 1 in g.adj[0]
	assert 1 in list(g.adj[0])
	assert 0 not in g.adj[1]
	assert 0 not in list(g.adj[1])
	g.remove_edge(0, 1)
	assert 0 in g
	assert 1 in g
	assert 1 not in g.adj[0]
	assert 1 not in list(g.adj[0])
	assert 0 in g.adj
	assert 1 not in g.adj[0]
	engine.branch = "physical_triangle"
	assert 3 in g.node
	assert 2 in g
	g.add_edge(0, 1)
	assert 1 in g.adj[0]
	assert 1 in list(g.adj[0])
	g.add_edge(1, 0)
	assert 0 in g.adj[1]
	assert 0 in list(g.adj[1])
	g.add_edge(1, 2)
	g.add_edge(2, 1)
	g.add_edge(2, 0)
	g.add_edge(0, 2)
	assert 2 in g.adj[0]
	assert 2 in list(g.adj[0])
	engine.branch = "physical_square"
	assert engine._node_exists("physical", 0)
	assert 3 in g.node
	assert 2 in list(g.adj[0])
	assert 2 in g.adj[0]
	engine.next_turn()
	assert 2 in g
	assert 2 in list(g.node.keys())
	assert 2 in list(g.adj[0])
	assert 2 in g.adj[0]
	assert 2 in list(g.adj[0])
	assert engine._node_exists("physical", 0)
	g.remove_edge(2, 0)
	assert 0 not in g.adj[2]
	assert 0 not in list(g.adj[2])
	assert 0 in g.node
	assert engine._node_exists("physical", 0)
	assert 0 not in g.adj[3]
	g.add_edge(3, 0)
	assert 0 in g.adj[3]
	assert engine.turn == 2
	assert 0 in g.adj[3]
	assert 0 in list(g.adj[3])
	assert 0 in g.node
	assert engine._node_exists("physical", 0)
	assert 2 in g.pred[3]
	assert 3 in g.pred[0]
	engine.branch = "physical_de_edge"
	assert 3 in g.node
	assert 0 in g.node
	assert engine._node_exists("physical", 0)
	assert 3 in g.adj
	assert 0 in g.adj[3]
	g.remove_node(3)
	assert 3 not in g.node
	assert 3 not in g.adj
	assert 3 not in g.adj[2]
	assert 3 not in g.pred
	assert 3 not in g.pred[0]
	engine.branch = "physical_square"
	assert engine.turn == 2
	assert 0 not in g.adj[2]
	assert 0 not in list(g.adj[2])
	assert 0 in g.adj[3]
	assert 0 in list(g.adj[3])
	assert 3 in g.node
	engine.branch = "physical_nothing"
	assert 0 not in g.adj[2]
	assert 0 not in list(g.adj[2])
	assert 0 in g.adj[3]
	assert 0 in list(g.adj[3])
	assert 3 in g.node
	g.remove_nodes_from((0, 1, 2, 3))
	for n in (0, 1, 2, 3):
		assert n not in g.node
		assert n not in g.adj


def test_branch_lineage(engine):
	# I want an analogue of this test for when you're looking up keyframes
	# in parent branches
	graph_objects_create_delete(engine)
	assert engine.is_ancestor_of("trunk", "physical_no_edge")
	assert engine.is_ancestor_of("trunk", "physical_triangle")
	assert engine.is_ancestor_of("trunk", "physical_nothing")
	assert engine.is_ancestor_of("physical_no_edge", "physical_triangle")
	assert engine.is_ancestor_of("physical_square", "physical_nothing")
	assert not engine.is_ancestor_of("physical_nothing", "trunk")
	assert not engine.is_ancestor_of("physical_triangle", "physical_no_edge")
	engine.time = (
		"trunk",
		engine.branch_start_turn("trunk"),
		engine.branch_start_tick("trunk"),
	)
	g = engine.character["physical"]
	assert 0 not in g.node
	assert 1 not in g.node
	assert 0 not in g.edge
	engine.tick = engine.turn_end()
	assert 0 in g.node
	assert 1 in g.node
	assert 0 in g.edge
	assert 1 in g.edge[0]
	engine.turn = 0
	with pytest.raises(ValueError):
		engine.branch = "physical_no_edge"
	engine.turn = engine.branch_start_turn("physical_no_edge")
	engine.branch = "physical_no_edge"
	engine.next_turn()
	assert 0 in g
	assert 0 in list(g.node.keys())
	assert 1 not in g.edge[0]
	assert 0 not in g.edge[1]
	with pytest.raises(KeyError):
		g.edge[0][1]
	engine.branch = "physical_triangle"
	assert 2 in g.node
	for orig in (0, 1, 2):
		for dest in (0, 1, 2):
			if orig == dest:
				continue
			assert orig in g.edge
			assert dest in g.edge[orig]
	engine.branch = "physical_square"
	assert 0 not in g.edge[2]
	with pytest.raises(KeyError):
		g.edge[2][0]
	engine.turn = 2
	assert 3 in g.node
	assert 1 in g.edge[0]
	assert 2 in g.edge[1]
	assert 3 in g.edge[2]
	engine.branch = "physical_nothing"
	for node in (0, 1, 2):
		assert node not in g.node
		assert node not in g.edge
	engine.branch = "trunk"
	engine.turn = 0
	assert 0 in g.node
	assert 1 in g.node
	assert 0 in g.edge
	assert 1 in g.edge[0]


def test_store_value(engine):
	g = engine.new_character("testgraph")
	g.add_node(0)
	g.add_node(1)
	g.add_edge(0, 1)
	n = g.node[0]
	e = g.edge[0][1]
	for k, v in testdata:
		g.graph[k] = v
		assert k in g.graph
		assert g.graph[k] == v
		del g.graph[k]
		assert k not in g.graph
		n[k] = v
		assert k in n
		assert n[k] == v
		del n[k]
		assert k not in n
		e[k] = v
		assert k in e
		assert e[k] == v
		del e[k]
		assert k not in e


def test_store_dict(engine):
	g = engine.new_character("testgraph")
	g.add_node(0)
	g.add_node(1)
	g.add_edge(0, 1)
	n = g.node[0]
	e = g.edge[0][1]
	for entity in (g.graph, n, e):
		entity[0] = {
			"spam": "eggs",
			"ham": {"baked beans": "delicious"},
			"qux": ["quux", "quuux"],
			"clothes": {"hats", "shirts", "pants"},
			"dicts": {"foo": {"bar": "bas"}, "qux": {"quux": "quuux"}},
		}
	engine.next_turn()
	for entity in (g.graph, n, e):
		assert entity[0]["spam"] == "eggs"
		entity[0]["spam"] = "ham"
		assert entity[0]["spam"] == "ham"
		assert entity[0]["ham"] == {"baked beans": "delicious"}
		entity[0]["ham"]["baked beans"] = "disgusting"
		assert entity[0]["ham"] == {"baked beans": "disgusting"}
		assert entity[0]["qux"] == ["quux", "quuux"]
		entity[0]["qux"] = ["quuux", "quux"]
		assert entity[0]["qux"] == ["quuux", "quux"]
		assert entity[0]["clothes"] == {"hats", "shirts", "pants"}
		entity[0]["clothes"].remove("hats")
		assert entity[0]["clothes"] == {"shirts", "pants"}
		assert entity[0]["dicts"] == {
			"foo": {"bar": "bas"},
			"qux": {"quux": "quuux"},
		}
		del entity[0]["dicts"]["foo"]
		entity[0]["dicts"]["qux"]["foo"] = {"bar": "bas"}
		assert entity[0]["dicts"] == {
			"qux": {"foo": {"bar": "bas"}, "quux": "quuux"}
		}
	engine.turn = 0
	for entity in g.graph, n, e:
		assert entity[0]["spam"] == "eggs"
		assert entity[0]["ham"] == {"baked beans": "delicious"}
		assert entity[0]["qux"] == ["quux", "quuux"]
		assert entity[0]["clothes"] == {"hats", "shirts", "pants"}
		assert entity[0]["dicts"] == {
			"foo": {"bar": "bas"},
			"qux": {"quux": "quuux"},
		}


def test_store_list(engine):
	g = engine.new_character("testgraph")
	g.add_node(0)
	g.add_node(1)
	g.add_edge(0, 1)
	n = g.node[0]
	e = g.edge[0][1]
	for entity in g.graph, n, e:
		entity[0] = [
			"spam",
			("eggs", "ham"),
			{"baked beans": "delicious"},
			["qux", "quux", "quuux"],
			{"hats", "shirts", "pants"},
		]
	engine.next_turn()
	for entity in g.graph, n, e:
		assert entity[0][0] == "spam"
		entity[0][0] = "eggplant"
		assert entity[0][0] == "eggplant"
		assert entity[0][1] == ("eggs", "ham")
		entity[0][1] = ("ham", "eggs")
		assert entity[0][1] == ("ham", "eggs")
		assert entity[0][2] == {"baked beans": "delicious"}
		entity[0][2]["refried beans"] = "deliciouser"
		assert entity[0][2] == {
			"baked beans": "delicious",
			"refried beans": "deliciouser",
		}
		assert entity[0][3] == ["qux", "quux", "quuux"]
		assert entity[0][3].pop() == "quuux"
		assert entity[0][3] == ["qux", "quux"]
		assert entity[0][4] == {"hats", "shirts", "pants"}
		entity[0][4].discard("shame")
		entity[0][4].remove("pants")
		entity[0][4].add("sun")
		assert entity[0][4] == {"hats", "shirts", "sun"}
	engine.turn -= 1
	for entity in g.graph, n, e:
		assert entity[0][0] == "spam"
		assert entity[0][1] == ("eggs", "ham")
		assert entity[0][2] == {"baked beans": "delicious"}
		assert entity[0][3] == ["qux", "quux", "quuux"]
		assert entity[0][4] == {"hats", "shirts", "pants"}


def test_store_set(engine):
	g = engine.new_character("testgraph")
	g.add_node(0)
	g.add_node(1)
	g.add_edge(0, 1)
	n = g.node[0]
	e = g.edge[0][1]
	for entity in g.graph, n, e:
		entity[0] = set(range(10))
	engine.next_turn()
	for entity in g.graph, n, e:
		assert entity[0] == set(range(10))
		for j in range(0, 12, 2):
			entity[0].discard(j)
	engine.turn = 0
	for entity in g.graph, n, e:
		assert entity[0] == set(range(10))
