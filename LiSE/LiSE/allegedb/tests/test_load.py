import pytest
import os
from LiSE.allegedb import ORM
import networkx as nx


testgraphs = [
    nx.chvatal_graph()
]
# have to name it after creation because it clears the create_using
path_graph_9 = nx.path_graph(9)
path_graph_9.name = 'path_graph_9'
testgraphs.append(path_graph_9)


@pytest.fixture
def db(tmpdbfile):
    with ORM('sqlite:///' + tmpdbfile) as orm:
        for graph in testgraphs:
            orm.new_digraph(graph.name, graph)
            if not graph.is_directed():
                graph = nx.to_directed(graph)
            assert set(graph.nodes.keys()) == set(orm.graph[graph.name].nodes.keys()), \
                "{}'s nodes changed during instantiation".format(graph.name)
            assert set(graph.edges) == set(orm.graph[graph.name].edges.keys()), \
                "{}'s edges changed during instantiation".format(graph.name)
    with ORM('sqlite:///' + tmpdbfile) as orm:
        yield orm


def test_basic_load(db):
    for graph in testgraphs:
        if not graph.is_directed():
            graph = nx.to_directed(graph)
        alleged = db.graph[graph.name]
        assert set(graph.nodes.keys()) == set(alleged.nodes.keys()), "{}'s nodes are not the same after load".format(
            graph.name
        )
        assert set(graph.edges) == set(alleged.edges), "{}'s edges are not the same after load".format(graph.name)


def test_keyframe_load(db):
    for graph in testgraphs:
        nodes_kf = db._nodes_cache.keyframe
        assert (graph.name,) in nodes_kf, "{} not in nodes cache".format(
            graph.name)
        assert 'trunk' in nodes_kf[graph.name,], \
            "trunk branch not in nodes cache for {}".format(graph.name)
        assert nodes_kf[graph.name,]['trunk'].rev_gettable(0), \
            "turn 0 not in nodes cache for {}".format(graph.name)
        assert nodes_kf[graph.name,]['trunk'][0].rev_gettable(0), \
            "tick 0 not in nodes cache for {}".format(graph.name)
        assert db._nodes_cache.keyframe[graph.name,]['trunk'][0][0] == {
            node: True for node in graph.nodes.keys()
        }, "{} not loaded correctly, got {}".format(
            graph.name, nodes_kf['trunk'][0][0])
        edges_kf = db._edges_cache.keyframe
        if graph.is_multigraph():
            for orig in graph.adj:
                for dest in graph.adj[orig]:
                    assert (graph.name, orig, dest) in edges_kf, \
                        "{} not in edges cache".format(
                            (graph.name, orig, dest))
                    this_edge = edges_kf[graph.name, orig, dest]
                    assert 'trunk' in this_edge, \
                        "trunk branch not in edges cache for {}".format(
                            (graph.name, orig, dest)
                        )
                    assert this_edge['trunk'].rev_gettable(0), \
                        "turn 0 not in trunk branch of edges cache for {}".format(
                            (graph.name, orig, dest)
                        )
                    assert this_edge['trunk'][0].rev_gettable(0), \
                        "tick 0 not in turn 0 of trunk branch of edges cache for {}".format(
                            (graph.name, orig, dest)
                        )
                    assert db._edges_cache.keyframe[graph.name, orig, dest][
                        'trunk'][0][0] == {
                        idx: True for idx in graph.adj[orig][dest]}, \
                    "{} not loaded".format((graph.name, orig, dest))
        else:
            for orig in graph.adj:
                for dest in graph.adj[orig]:
                    assert (graph.name, orig, dest) in edges_kf, \
                        "{} not in edges cache".format(
                            (graph.name, orig, dest)
                        )
                    this_edge = edges_kf[graph.name, orig, dest]
                    assert 'trunk' in this_edge, \
                        "trunk branch not in edges cache for {}".format(
                            (graph.name, orig, dest)
                        )
                    assert this_edge['trunk'].rev_gettable(0), \
                        "turn 0 not in trunk branch of edges cache for {}".format(
                            (graph.name, orig, dest)
                        )
                    assert this_edge['trunk'][0].rev_gettable(0), \
                        "tick 0 not in turn 0 of trunk branch of edges cache for {}".format(
                            (graph.name, orig, dest)
                        )
                    assert db._edges_cache.keyframe[graph.name, orig, dest][
                        'trunk'][0][0] == {0: True}, "{} not loaded".format(
                        (graph.name, orig, dest)
                    )
        for node, vals in graph.nodes.items():
            assert db._node_val_cache.keyframe[graph.name, node]['trunk'][0][0] == vals
        for edge in graph.edges:
            if graph.is_multigraph():
                assert db._edge_val_cache.keyframe[(graph.name,) + edge]['trunk'][0][0] == graph.edges[edge]
            else:
                assert db._edge_val_cache.keyframe[(graph.name,) + edge + (0,)]['trunk'][0][0] == graph.edges[edge]