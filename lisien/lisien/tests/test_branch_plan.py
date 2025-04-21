from lisien import Engine
from lisien.xcollections import FunctionStore, StringStore


def test_single_plan(serial_engine):
	eng = serial_engine
	assert eng.turn == 0
	g = eng.new_character("graph")
	g.add_node(0)
	eng.next_turn()
	assert eng.turn == 1
	g.add_node(1)
	with eng.plan():
		eng.turn = 2
		g.add_node(2)
		g.node[2]["clever"] = False
		eng.turn = 3
		g.node[2]["funny"] = True
		g.add_node(3)
		eng.turn = 4
		g.node[2]["successful"] = True
	eng.turn = 1
	assert 2 not in g.node
	eng.branch = "b"
	assert 2 not in g.node
	assert 1 in g
	eng.next_turn()
	assert eng.turn == 2
	assert 2 in g.node
	assert set(g.node[2].keys()) == {"clever"}
	eng.next_turn()
	assert eng.turn == 3
	assert g.node[2]["funny"]
	eng.tick = eng.turn_end_plan()
	assert 3 in g
	assert set(g.node[2].keys()) == {"funny", "clever"}
	eng.next_turn()
	assert eng.turn == 4
	assert g.node[2].keys() == {"funny", "clever", "successful"}
	eng.turn = 2
	eng.tick = eng.turn_end_plan()
	eng.branch = "d"
	assert g.node[2].keys() == {"clever"}
	g.node[2]["funny"] = False
	assert g.node[2].keys() == {"funny", "clever"}
	eng.turn = 3
	assert not g.node[2]["funny"]
	assert 3 not in g.node
	eng.turn = 4
	assert g.node[2].keys() == {"funny", "clever"}
	eng.turn = 1
	eng.branch = "trunk"
	eng.turn = 0
	assert 1 not in g.node
	eng.branch = "c"
	eng.turn = 2
	assert 1 not in g.node
	assert 2 not in g.node
	eng.turn = 0
	eng.branch = "trunk"
	eng.turn = 2
	eng.tick = eng.turn_end_plan()
	assert 2 in g.node


def test_multi_plan(serial_engine):
	eng = serial_engine
	g1 = eng.new_character(1)
	g2 = eng.new_character(2)
	with eng.plan():
		g1.add_node(1)
		g1.add_node(2)
		eng.turn = 1
		g1.add_edge(1, 2)
	eng.turn = 0
	with eng.plan():
		g2.add_node(1)
		g2.add_node(2)
		eng.turn = 1
		g2.add_edge(1, 2)
	eng.turn = 0
	# contradict the plan
	eng.tick = eng.turn_end_plan()
	del g1.node[2]
	assert 1 in g2.node
	assert 2 in g2.node
	eng.turn = 1
	eng.tick = eng.turn_end_plan()
	assert 2 not in g1.node
	assert 2 not in g1.edge[1]
	assert 2 in g2.edge[1]


def test_plan_vs_plan(serial_engine):
	eng = serial_engine
	g1 = eng.new_character(1)
	with eng.plan():
		g1.add_node(1)
		g1.add_node(2)
		eng.turn = 1
		g1.add_edge(1, 2)
		g1.add_node(3)
		g1.add_edge(3, 1)
	eng.turn = 0
	with eng.plan():
		g1.add_node(0)  # not a contradiction, just two plans
		g1.add_edge(0, 1)
	eng.turn = 1
	eng.tick = eng.turn_end_plan()
	assert 0 in g1.node
	assert 1 in g1.node
	assert 2 in g1.node
	assert 3 in g1.node
	assert 1 in g1.edge[0]
	assert 2 in g1.edge[1]
	eng.turn = 0
	eng.tick = eng.turn_end_plan()
	with eng.plan():
		del g1.node[2]
	eng.turn = 2
	eng.tick = eng.turn_end_plan()
	assert 3 not in g1.node
	assert 3 not in g1.adj
	assert 0 in g1.node
	assert 1 in g1.adj[0]


def test_save_load_plan(tmp_path, non_null_database):
	if non_null_database == "sqlite":
		connect_str = f"sqlite:///{tmp_path}/world.db"
		path = None
	else:
		connect_str = None
		path = tmp_path
	with Engine(
		path,
		function=FunctionStore(None),
		method=FunctionStore(None),
		trigger=FunctionStore(None),
		prereq=FunctionStore(None),
		action=FunctionStore(None),
		string={},
		workers=0,
		connect_string=connect_str,
	) as orm:
		g1 = orm.new_character(1)
		g2 = orm.new_character(2)
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
		path,
		workers=0,
		function=FunctionStore(None),
		method=FunctionStore(None),
		trigger=FunctionStore(None),
		prereq=FunctionStore(None),
		action=FunctionStore(None),
		string=StringStore({"language": "eng"}, None),
		connect_string=connect_str,
	) as orm:
		g1 = orm.character[1]
		g2 = orm.character[2]
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
		path,
		workers=0,
		function=FunctionStore(None),
		method=FunctionStore(None),
		trigger=FunctionStore(None),
		prereq=FunctionStore(None),
		action=FunctionStore(None),
		string=StringStore({"language": "eng"}, None),
		connect_string=connect_str,
	) as orm:
		orm.turn = 0
		g1 = orm.character[1]
		g2 = orm.character[2]
		assert 1 in g2.node
		assert 2 in g2.node
		assert 2 not in g1.edge[1]
		assert 2 not in g2.edge[1]
		orm.turn = 1
		assert 2 not in g1.node
		orm.turn = 2
		assert 2 in g2.edge[1]
