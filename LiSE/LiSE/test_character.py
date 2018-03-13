import allegedb.test
from .engine import Engine


class CharacterTest(allegedb.test.AllegedTest):
    def setUp(self):
        self.engine = Engine("sqlite:///:memory:")
        self.graphmakers = (self.engine.new_character,)

    def tearDown(self):
        self.engine.close()


class CharacterDictStorageTest(CharacterTest, allegedb.test.DictStorageTest):
    pass


class CharacterListStorageTest(CharacterTest, allegedb.test.ListStorageTest):
    pass


class CharacterSetStorageTest(CharacterTest, allegedb.test.SetStorageTest):
    pass
