import pytest
import allegedb


@pytest.fixture(scope='function')
def orm():
    with allegedb.ORM("sqlite:///:memory:") as it:
        yield it


def test_single_plan(orm):
    g = orm.new_graph('graph')
    g.add_node(0)
    orm.turn = 1
    g.add_node(1)
    with orm.plan():
        orm.turn = 2
        g.add_node(2)
    assert orm.turn == 1
    assert 2 not in g
    orm.branch = 'b'
    assert 2 not in g
    assert 1 in g
    orm.turn = 2
    assert 2 in g
    orm.turn = 1
    orm.branch = 'trunk'
    orm.turn = 0
    assert 1 not in g
    orm.branch = 'c'
    orm.turn = 2
    assert 1 not in g
    assert 2 not in g