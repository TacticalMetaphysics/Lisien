from allegedb.cache import WindowDict
from allegedb.window import within_history
from itertools import cycle
import pytest

testvs = ['a', 99, ['spam', 'eggs', 'ham'], {'foo': 'bar', 0: 1, '💧': '🔑'}]
testdata = []
for k, v in zip(range(100), cycle(testvs)):
    testdata.append((k, v))


@pytest.fixture
def windd():
    return WindowDict(testdata)


def test_keys(windd):
    keys = windd.keys()
    windd.seek(50)
    assert list(range(100)) == list(keys)
    for i in range(100):
        assert i in windd
        assert i in keys


def test_items(windd):
    for item in testdata:
        assert item in windd.items()
    assert list(windd.items()) == testdata
    assert (63, 36) not in windd.items()


def test_past(windd):
    assert list(reversed(range(100))) == list(windd.past())
    windd.seek(50)
    assert list(reversed(range(51))) == list(windd.past())
    unseen = []
    seen = []
    for item in testdata:
        if item not in windd.past().items():
            unseen.append(item)
        else:
            seen.append(item)
    seen.reverse()
    assert seen == list(windd.past().items())
    for item in seen:
        assert item in windd.past().items()


def test_future(windd):
    assert [] == list(windd.future())
    windd.seek(-1)
    assert list(range(100)) == list(windd.future())
    for item in testdata:
        assert item in windd.future().items()
    windd.seek(50)
    assert list(range(51, 100)) == list(windd.future())
    unseen = []
    seen = []
    for item in testdata:
        if item not in windd.past().items():
            unseen.append(item)
        else:
            seen.append(item)
    assert unseen == list(windd.future().items())
    for item in unseen:
        assert item in windd.future().items()
        assert item in windd.items()
    for item in windd.future().items():
        assert item in unseen
    for item in seen:
        assert item in windd.past().items()
        assert item in windd.items()


def test_empty():
    empty = WindowDict()
    items = empty.items()
    past = empty.past()
    future = empty.future()
    assert not items
    assert list(items) == []
    assert list(past) == []
    assert list(past.items()) == []
    assert list(future) == []
    assert list(future.items()) == []
    for data in testdata:
        assert data not in items
        assert data not in past
        assert data not in future


def test_slice(windd):
    assert list(windd[:50]) == [windd[i] for i in range(51)]
    assert list(reversed(windd[:50])) == [windd[i] for i in reversed(range(51))]
    assert list(windd[50:]) == [windd[i] for i in range(50, 100)]
    assert list(reversed(windd[50:])) == [windd[i] for i in reversed(range(50, 100))]
    assert list(windd[25:50]) == [windd[i] for i in range(25, 50)]
    assert list(reversed(windd[25:50])) == [windd[i] for i in reversed(range(25, 50))]
    assert list(windd[50:25]) == [windd[i] for i in range(50, 25, -1)]
    assert list(reversed(windd[50:25])) == [windd[i] for i in reversed(range(50, 25, -1))]
