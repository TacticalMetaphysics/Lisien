from .proxy import EngineProcessManager
import unittest
import allegedb.test


class ProxyTest(unittest.TestCase):
    def setUp(self):
        self.manager = EngineProcessManager()
        self.engine = self.manager.start('sqlite:///:memory:')
        self.graphmakers = (self.engine.new_character,)

    def tearDown(self):
        self.manager.shutdown()


class BranchLineageTest(ProxyTest, allegedb.test.BranchLineageTest):
    pass


class DictStorageTest(ProxyTest, allegedb.test.DictStorageTest):
    pass


class ListStorageTest(ProxyTest, allegedb.test.ListStorageTest):
    pass


class SetStorageTest(ProxyTest, allegedb.test.SetStorageTest):
    pass
