from LiSE import Engine
import pytest


@pytest.fixture
def someplace():
    with Engine("sqlite:///:memory:") as eng:
        yield eng.new_character('physical').new_place('someplace')


def test_contents(someplace):
    stuff = [someplace.new_thing(i) for i in range(10)]
    assert len(someplace.content) == 10
    for i in range(10):
        assert i in someplace.content
        assert someplace.content[i] == stuff[i]
    for that in stuff:
        assert that in someplace.contents()