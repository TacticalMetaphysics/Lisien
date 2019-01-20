from allegedb.cache import WindowDict
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
    assert list(range(100)) == list(windd.keys())


def test_items(windd):
    for item in testdata:
        assert item in windd.items()


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
    for item in seen:
        assert item in windd.past().items()
    for item in windd.past().items():
        assert item in seen


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
    for item in unseen:
        assert item in windd.future().items()
    for item in windd.future().items():
        assert item in unseen
    for item in seen:
        assert item in windd.past().items()
