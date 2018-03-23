# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  public@zacharyspector.com
from LiSE.proxy import EngineProcessManager
import allegedb.test


class ProxyTest(allegedb.test.AllegedTest):
    def setUp(self):
        self.manager = EngineProcessManager()
        self.engine = self.manager.start('sqlite:///:memory:')
        self.graphmakers = (self.engine.new_character,)

    def tearDown(self):
        self.manager.shutdown()


class ProxyGraphTest(allegedb.test.AbstractGraphTest, ProxyTest):
    pass


class DictStorageTest(ProxyTest, allegedb.test.DictStorageTest):
    pass


class ListStorageTest(ProxyTest, allegedb.test.ListStorageTest):
    pass


class SetStorageTest(ProxyTest, allegedb.test.SetStorageTest):
    pass
