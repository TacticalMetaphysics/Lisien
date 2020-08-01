import pytest
import os
from allegedb import ORM
import networkx as nx


scalefreestart = nx.MultiDiGraph(name='scale_free_graph_5')
scalefreestart.add_edges_from([(0, 1), (1, 2), (2, 0)])


testgraphs = [
    nx.chvatal_graph(),
    nx.scale_free_graph(5, create_using=scalefreestart),
    # nx.chordal_cycle_graph(5, create_using=nx.MultiGraph(name='chordal_cycle_graph_5')),
    # The standard networkx edges iterator decides to flip some edges about in arbitrary-seeming
    # ways that I haven't been able to replicate and it doesn't seem worth it.
]
# have to name it after creation because it clears the create_using
path_graph_9 = nx.path_graph(9)
path_graph_9.name = 'path_graph_9'
testgraphs.append(path_graph_9)


@pytest.fixture
def db():
    name = 'allegedb_load_test.db'
    if os.path.exists(name):
        os.remove(name)
    with ORM('sqlite:///' + name) as orm:
        for graph in testgraphs:
            {
                nx.Graph: orm.new_graph,
                nx.DiGraph: orm.new_digraph,
                nx.MultiGraph: orm.new_multigraph,
                nx.MultiDiGraph: orm.new_multidigraph
            }[type(graph)](graph.name, graph)
            assert set(graph.nodes.keys()) == set(orm.graph[graph.name].nodes.keys()), \
                "{}'s nodes changed during instantiation".format(graph.name)
            assert set(graph.edges) == set(orm.graph[graph.name].edges), \
                "{}'s edges changed during instantiation".format(graph.name)
    with ORM('sqlite:///' + name) as orm:
        yield orm
    os.remove(name)


def test_basic_load(db):
    for graph in testgraphs:
        alleged = db.graph[graph.name]
        assert set(graph.nodes.keys()) == set(alleged.nodes.keys()), "{}'s nodes are not the same after load".format(
            graph.name
        )
        assert set(graph.edges) == set(alleged.edges), "{}'s edges are not the same after load".format(graph.name)


def test_keyframe_load(db):
    for graph in testgraphs:
        assert db._nodes_cache.keyframe[graph.name,]['trunk'][0][0] == {
            node: True for node in graph.nodes.keys()
        }
        if graph.is_multigraph():
            for orig in graph.adj:
                for dest in graph.adj[orig]:
                    assert db._edges_cache.keyframe[graph.name, orig, dest][
                        'trunk'][0][0] == {
                        idx: True for idx in graph.adj[orig][dest]}
        else:
            for orig in graph.adj:
                for dest in graph.adj[orig]:
                    assert db._edges_cache.keyframe[graph.name, orig, dest][
                        'trunk'][0][0] == {0: True}
        for node, vals in graph.nodes.items():
            assert db._node_val_cache.keyframe[graph.name, node]['trunk'][0][0] == vals
        for edge in graph.edges:
            if graph.is_multigraph():
                assert db._edge_val_cache.keyframe[(graph.name,) + edge]['trunk'][0][0] == graph.edges[edge]
            else:
                assert db._edge_val_cache.keyframe[(graph.name,) + edge + (0,)]['trunk'][0][0] == graph.edges[edge]