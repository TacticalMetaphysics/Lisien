from types import SimpleNamespace

import pytest

from lisien import Engine


def test_single_plan(engy):
	assert engy.turn == 0
	g = engy.new_digraph("graph")
	g.add_node(0)
	engy.next_turn()
	assert engy.turn == 1
	g.add_node(1)
	with engy.plan():
		engy.turn = 2
		g.add_node(2)
		g.node[2]["clever"] = False
		engy.turn = 3
		g.node[2]["funny"] = True
		g.add_node(3)
		engy.turn = 4
		g.node[2]["successful"] = True
	engy.turn = 1
	assert 2 not in g.node
	engy.branch = "b"
	assert 2 not in g.node
	assert 1 in g
	engy.next_turn()
	assert engy.turn == 2
	assert 2 in g.node
	assert set(g.node[2].keys()) == {"clever"}
	engy.next_turn()
	assert engy.turn == 3
	assert g.node[2]["funny"]
	engy.tick = engy.turn_end_plan()
	assert 3 in g
	assert set(g.node[2].keys()) == {"funny", "clever"}
	engy.next_turn()
	assert engy.turn == 4
	assert g.node[2].keys() == {"funny", "clever", "successful"}
	engy.turn = 2
	engy.tick = engy.turn_end_plan()
	engy.branch = "d"
	assert g.node[2].keys() == {"clever"}
	g.node[2]["funny"] = False
	assert g.node[2].keys() == {"funny", "clever"}
	engy.turn = 3
	assert not g.node[2]["funny"]
	assert 3 not in g.node
	engy.turn = 4
	assert g.node[2].keys() == {"funny", "clever"}
	engy.turn = 1
	engy.branch = "trunk"
	engy.turn = 0
	assert 1 not in g.node
	engy.branch = "c"
	engy.turn = 2
	assert 1 not in g.node
	assert 2 not in g.node
	engy.turn = 0
	engy.branch = "trunk"
	engy.turn = 2
	engy.tick = engy.turn_end_plan()
	assert 2 in g.node


def test_multi_plan(engy):
	g1 = engy.new_digraph(1)
	g2 = engy.new_digraph(2)
	with engy.plan():
		g1.add_node(1)
		g1.add_node(2)
		engy.turn = 1
		g1.add_edge(1, 2)
	engy.turn = 0
	with engy.plan():
		g2.add_node(1)
		g2.add_node(2)
		engy.turn = 1
		g2.add_edge(1, 2)
	engy.turn = 0
	# contradict the plan
	engy.tick = engy.turn_end_plan()
	del g1.node[2]
	assert 1 in g2.node
	assert 2 in g2.node
	engy.turn = 1
	engy.tick = engy.turn_end_plan()
	assert 2 not in g1.node
	with pytest.raises(KeyError):
		list(g1.edge[1])
	assert 2 in g2.edge[1]


def test_plan_vs_plan(engy):
	g1 = engy.new_digraph(1)
	with engy.plan():
		g1.add_node(1)
		g1.add_node(2)
		engy.turn = 1
		g1.add_edge(1, 2)
		g1.add_node(3)
		g1.add_edge(3, 1)
	engy.turn = 0
	with engy.plan():
		g1.add_node(0)  # not a contradiction, just two plans
		g1.add_edge(0, 1)
	engy.turn = 1
	engy.tick = engy.turn_end_plan()
	assert 0 in g1.node
	assert 1 in g1.node
	assert 2 in g1.node
	assert 3 in g1.node
	assert 1 in g1.edge[0]
	assert 2 in g1.edge[1]
	engy.turn = 0
	engy.tick = engy.turn_end_plan()
	with engy.plan():
		del g1.node[2]
	engy.turn = 2
	engy.tick = engy.turn_end_plan()
	assert 3 not in g1.node
	assert 3 not in g1.adj
	assert 0 in g1.node
	assert 1 in g1.adj[0]


def test_save_load_plan(tmp_path):
	with Engine(
		tmp_path,
		workers=0,
		function=SimpleNamespace(),
		method=SimpleNamespace(),
		trigger=SimpleNamespace(),
		prereq=SimpleNamespace(),
		action=SimpleNamespace(),
	) as orm:
		g1 = orm.new_digraph(1)
		g2 = orm.new_digraph(2)
		with orm.plan():
			g1.add_node(1)
			g1.add_node(2)
			orm.turn = 1
			g1.add_edge(1, 2)
		orm.turn = 0
		with orm.plan():
			g2.add_node(1)
			g2.add_node(2)
			tick2 = orm.tick
			orm.turn = 1
			g2.add_edge(1, 2)
			tick3 = orm.tick
		orm.turn = 0
	with Engine(
		tmp_path,
		workers=0,
		function=SimpleNamespace(),
		method=SimpleNamespace(),
		trigger=SimpleNamespace(),
		prereq=SimpleNamespace(),
		action=SimpleNamespace(),
	) as orm:
		g1 = orm.graph[1]
		g2 = orm.graph[2]
		assert 2 not in g1.node  # because we're before the plan
		# but if we go to after the plan...
		orm.tick = orm.turn_end_plan()
		assert 1 in g1.node
		assert 2 in g1.node
		# contradict the plan
		del g1.node[2]
		assert 1 in g2.node
		assert 2 in g2.node
		orm.next_turn()
		assert orm.turn == 1
		assert 2 not in g1.node
		assert 2 not in g1.edge[1]
		# but, since the stuff that happened in g2 was in a different plan,
		# it still happens
		orm.next_turn()
		assert 1 in g2.node
		assert 2 in g2.node
		assert 2 in g2.edge[1]
	with Engine(
		tmp_path,
		workers=0,
		function=SimpleNamespace(),
		method=SimpleNamespace(),
		trigger=SimpleNamespace(),
		prereq=SimpleNamespace(),
		action=SimpleNamespace(),
	) as orm:
		orm.turn = 0
		g1 = orm.graph[1]
		g2 = orm.graph[2]
		assert 1 in g2.node
		assert 2 in g2.node
		assert 2 not in g1.edge[1]
		assert 2 not in g2.edge[1]
		orm.turn = 1
		assert 2 not in g1.node
		orm.turn = 2
		assert 2 in g2.edge[1]
