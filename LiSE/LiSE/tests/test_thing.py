from LiSE import Engine
import pytest


@pytest.fixture(scope='function')
def something():
    with Engine("sqlite:///:memory:") as eng:
        yield eng.new_character('physical').new_place('somewhere').new_thing('something')


def test_container(something):
    pl1 = something.character.place['somewhere']
    assert something.container == pl1
    pl2 = something.character.new_place('somewhere2')
    po = something.character.new_portal('somewhere', 'somewhere2')
    something.next_location = 'somewhere2'
    assert something.container == something.character.portal['somewhere']['somewhere2']
    assert something.location == something.character.node['somewhere']
    assert something.name in pl1.content
    assert not something.name in pl2.content
    assert [something] == list(pl1.contents())
    assert [] == list(pl2.contents())
    assert something.name in po.content
    assert [something] == list(po.contents())