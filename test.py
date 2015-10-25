import unittest
import gorm
import networkx as nx
from networkx.generators.atlas import graph_atlas_g


class GormTest(unittest.TestCase):
    def setUp(self):
        self.engine = gorm.ORM('sqlite:///:memory:')
        self.engine.initdb()

    def tearDown(self):
        self.engine.close()

    def test_branch_lineage(self):
        """Create some branches of history and check that gorm remembers where
        each came from.

        """
        pass

    def test_global_storage(self):
        """Test that we can store arbitrary key-value pairs in the ``global``
        mapping.

        """
        pass

    def test_graph_storage(self):
        """Test that all the graph types can store and retrieve key-value pairs
        for the graph as a whole.

        """
        pass

    def test_node_storage(self):
        """Test that all the graph types can store and retrieve key-value
        pairs for particular nodes."""
        pass

    def test_edge_storage(self):
        """Test that all the graph types can store and retrieve key-value
        pairs for particular edges.

        """
        pass

    def test_compiled_queries(self):
        """Make sure that the queries generated in SQLAlchemy are the same as
        those precompiled into SQLite.

        """
        from gorm.alchemy import Alchemist
        self.assertTrue(hasattr(self.engine.db, 'alchemist'))
        self.assertTrue(isinstance(self.engine.db.alchemist, Alchemist))
        from json import loads
        precompiled = loads(
            open(self.engine.db.json_path + '/sqlite.json', 'r').read()
        )
        self.assertEqual(
            precompiled.keys(), self.engine.db.alchemist.sql.keys()
        )
        for (k, query) in precompiled.items():
            self.assertEqual(
                query,
                str(
                    self.engine.db.alchemist.sql[k]
                )
            )

    def test_graph_atlas(self):
        """Test saving and loading all the graphs in the networkx graph
        atlas.

        """
        for g in graph_atlas_g():
            print(g.name)
            gormg = self.engine.new_graph(g.name, g)
            for n in g.node:
                self.assertIn(n, gormg.node)
                self.assertEqual(g.node[n], gormg.node[n])
            for u in g.edge:
                for v in g.edge[u]:
                    self.assertIn(u, gormg.edge)
                    self.assertIn(v, gormg.edge[u])
                    self.assertEqual(g.edge[u][v], gormg.edge[u][v])


if __name__ == '__main__':
    unittest.main()
