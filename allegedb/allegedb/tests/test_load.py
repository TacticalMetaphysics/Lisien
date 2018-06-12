import pytest
import os
from allegedb import ORM
import networkx as nx


testgraphs = [
    nx.chvatal_graph(),
    nx.scale_free_graph(22),
    nx.chordal_cycle_graph(22),
    nx.path_graph(9, create_using=nx.MultiDiGraph())
]


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
    with ORM('sqlite:///' + name) as orm:
        yield orm
    os.remove(name)


def test_basic_load(db):
    for graph in testgraphs:
        alleged = db.graph[graph.name]
        assert graph.node.keys() == alleged.node.keys()
        assert graph.edges == alleged.edges