from LiSE import Engine
import pytest


@pytest.fixture
def something():
    with Engine("sqlite:///:memory:") as eng:
        yield eng.new_character('physical').new_place('somewhere').new_thing('something')


def test_container(something):
    assert something.container == something.character.place['somewhere']
    something.character.add_place('somewhere2')
    something.character.add_portal('somewhere', 'somewhere2')
    something.next_location = 'somewhere2'
    assert something.container == something.character.portal['somewhere']['somewhere2']
    assert something.location == something.character.node['somewhere']