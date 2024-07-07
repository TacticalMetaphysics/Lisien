import pytest
import os
from LiSE.allegedb import ORM
import networkx as nx

testgraphs = [nx.chvatal_graph()]
# have to name it after creation because it clears the create_using
path_graph_9 = nx.path_graph(9)
path_graph_9.name = "path_graph_9"
testgraphs.append(path_graph_9)


@pytest.fixture
def db(tmpdbfile):
	with ORM("sqlite:///" + tmpdbfile) as orm:
		for graph in testgraphs:
			orm.new_digraph(graph.name, graph)
			if not graph.is_directed():
				graph = nx.to_directed(graph)
			assert set(graph.nodes.keys()) == set(
				orm.graph[graph.name].nodes.keys()
			), "{}'s nodes changed during instantiation".format(graph.name)
			assert set(graph.edges) == set(
				orm.graph[graph.name].edges.keys()
			), "{}'s edges changed during instantiation".format(graph.name)
	with ORM("sqlite:///" + tmpdbfile) as orm:
		yield orm


def test_basic_load(db):
	for graph in testgraphs:
		if not graph.is_directed():
			graph = nx.to_directed(graph)
		alleged = db.graph[graph.name]
		assert set(graph.nodes.keys()) == set(
			alleged.nodes.keys()
		), "{}'s nodes are not the same after load".format(graph.name)
		assert set(graph.edges) == set(
			alleged.edges
		), "{}'s edges are not the same after load".format(graph.name)


def test_keyframe_load(db):
	for graph in testgraphs:
		nodes_kf = db._nodes_cache.keyframe
		assert (graph.name,) in nodes_kf, "{} not in nodes cache".format(
			graph.name
		)
		assert (
			"trunk" in nodes_kf[graph.name,]
		), "trunk branch not in nodes cache for {}".format(graph.name)
		assert nodes_kf[graph.name,]["trunk"].rev_gettable(
			0
		), "turn 0 not in nodes cache for {}".format(graph.name)
		assert nodes_kf[graph.name,]["trunk"][0].rev_gettable(
			0
		), "tick 0 not in nodes cache for {}".format(graph.name)
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
					assert (
						"trunk" in this_edge
					), "trunk branch not in edges cache for {}".format(
						(graph.name, orig, dest)
					)
					assert this_edge[
						"trunk"
					].rev_gettable(
						0
					), "turn 0 not in trunk branch of edges cache for {}".format(
						(graph.name, orig, dest)
					)
					assert this_edge[
						"trunk"
					][
						0
					].rev_gettable(
						0
					), "tick 0 not in turn 0 of trunk branch of edges cache for {}".format(
						(graph.name, orig, dest)
					)
					assert db._edges_cache.keyframe[graph.name, orig, dest][
						"trunk"
					][0][0] == {
						idx: True for idx in graph.adj[orig][dest]
					}, "{} not loaded".format((graph.name, orig, dest))
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
					assert (
						"trunk" in this_edge
					), "trunk branch not in edges cache for {}".format(
						(graph.name, orig, dest)
					)
					assert this_edge[
						"trunk"
					].rev_gettable(
						0
					), "turn 0 not in trunk branch of edges cache for {}".format(
						(graph.name, orig, dest)
					)
					assert this_edge[
						"trunk"
					][
						0
					].rev_gettable(
						0
					), "tick 0 not in turn 0 of trunk branch of edges cache for {}".format(
						(graph.name, orig, dest)
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


def test_keyframe_unload(tmpdbfile):
	# TODO: test edge cases involving tick-precise unloads
	with ORM("sqlite:///" + tmpdbfile) as orm:
		g = orm.new_digraph("g", nx.grid_2d_graph(3, 3))
		orm.turn = 1
		assert orm._time_is_loaded(*orm._btt())
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
		orm.turn = 2
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
	with ORM("sqlite:///" + tmpdbfile) as orm:
		assert not orm._time_is_loaded("trunk", 1)
		assert orm._time_is_loaded("trunk", 2, 3)
		assert ("g", (0, 0), (0, 1)) in orm._edges_cache.keyframe
		assert 2 in orm._edges_cache.keyframe["g", (0, 0), (0, 1)]["trunk"]
		assert 0 not in orm._edges_cache.keyframe["g", (0, 0), (0, 1)]["trunk"]
		g = orm.graph["g"]
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
	with ORM("sqlite:///" + tmpdbfile) as orm:
		assert orm.branch == "u"
		assert (
			("g", (1, 1), (1, 2)) not in orm._edges_cache.keyframe
			or "trunk" not in orm._edges_cache.keyframe["g", (1, 1), (1, 2)]
		)
		g = orm.graph["g"]
		assert (1, 2) not in g.nodes
		orm.branch = "trunk"
		assert (1, 2) in g.nodes
		assert (
			("g", (1, 1), (1, 2)) in orm._edges_cache.keyframe
			and "trunk" in orm._edges_cache.keyframe["g", (1, 1), (1, 2)]
		)
