import pytest
import os
from allegedb import ORM
import networkx as nx


scalefreestart = nx.MultiDiGraph(name='scale_free_graph_5')
scalefreestart.add_edges_from([(0, 1), (1, 2), (2, 0)])


testgraphs = [
    nx.chvatal_graph(),
    nx.scale_free_graph(5, create_using=scalefreestart),
    nx.chordal_cycle_graph(5, create_using=nx.MultiGraph(name='chordal_cycle_graph_5')),
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
            assert set(graph.node.keys()) == set(orm.graph[graph.name].node.keys()), \
                "{}'s nodes changed during instantiation".format(graph.name)
            assert set(graph.edges) == set(orm.graph[graph.name].edges), \
                "{}'s edges changed during instantiation".format(graph.name)
    with ORM('sqlite:///' + name) as orm:
        yield orm
    os.remove(name)


def test_basic_load(db):
    for graph in testgraphs:
        alleged = db.graph[graph.name]
        assert set(graph.node.keys()) == set(alleged.node.keys()), "{}'s nodes are not the same after load".format(
            graph.name
        )
        assert set(graph.edges) == set(alleged.edges), "{}'s edges are not the same after load".format(graph.name)