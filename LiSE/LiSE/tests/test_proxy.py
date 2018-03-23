# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  public@zacharyspector.com
from LiSE.proxy import EngineProcessManager
import allegedb.tests.test_all


class ProxyTest(allegedb.tests.test_all.AllegedTest):
    def setUp(self):
        self.manager = EngineProcessManager()
        self.engine = self.manager.start('sqlite:///:memory:')
        self.graphmakers = (self.engine.new_character,)

    def tearDown(self):
        self.manager.shutdown()


class ProxyGraphTest(allegedb.tests.test_all.AbstractGraphTest, ProxyTest):
    pass


class DictStorageTest(ProxyTest, allegedb.tests.test_all.DictStorageTest):
    pass


class ListStorageTest(ProxyTest, allegedb.tests.test_all.ListStorageTest):
    pass


class SetStorageTest(ProxyTest, allegedb.tests.test_all.SetStorageTest):
    pass
