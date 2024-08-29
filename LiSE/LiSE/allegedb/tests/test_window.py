from ..window import WindowDict
from .. import HistoricKeyError, ORM
from itertools import cycle
import pytest

testvs = ["a", 99, ["spam", "eggs", "ham"], {"foo": "bar", 0: 1, "💧": "🔑"}]
testdata = []
for k, v in zip(range(100), cycle(testvs)):
	if type(v) is str:
		testdata.append((k, v + str(k)))
	elif type(v) is list:
		testdata.append((k, v + [k]))
	elif type(v) is int:
		testdata.append((k, v + 100 * k))
	else:
		assert type(v) is dict
		vv = dict(v)
		vv["k"] = k
		testdata.append((k, vv))


@pytest.fixture
def windd():
	return WindowDict(testdata)


def test_keys(windd):
	keys = windd.keys()
	windd._seek(50)
	assert list(range(100)) == list(keys)
	for i in range(100):
		assert i in windd
		assert i in keys


def test_items(windd):
	for item in testdata:
		assert item in windd.items()
	assert list(windd.items()) == testdata
	assert (63, 36) not in windd.items()


def test_empty():
	empty = WindowDict()
	items = empty.items()
	assert not items
	assert list(items) == []
	for data in testdata:
		assert data not in items


def test_slice(windd):
	assert list(windd[:50]) == [windd[i] for i in range(50)]
	assert list(reversed(windd[:50])) == [
		windd[i] for i in reversed(range(50))
	]
	assert list(windd[:50:2]) == [windd[i] for i in range(0, 50, 2)]
	assert list(windd[50:]) == [windd[i] for i in range(50, 100)]
	assert list(reversed(windd[50:])) == [
		windd[i] for i in reversed(range(50, 100))
	]
	assert list(windd[50::2]) == [windd[i] for i in range(50, 100, 2)]
	assert list(windd[25:50]) == [windd[i] for i in range(25, 50)]
	assert list(reversed(windd[25:50])) == [
		windd[i] for i in reversed(range(25, 50))
	]
	assert list(windd[50:25]) == [windd[i] for i in range(50, 25, -1)]
	assert list(reversed(windd[50:25])) == [
		windd[i] for i in reversed(range(50, 25, -1))
	]
	assert list(windd[25:50:2]) == [windd[i] for i in range(25, 50, 2)]
	assert list(windd[50:25:-2]) == [windd[i] for i in range(50, 25, -2)]


def test_del(windd):
	for k, v in testdata:
		assert k in windd
		assert windd[k] == v
		del windd[k]
		assert k not in windd
		with pytest.raises(KeyError):
			windd[k]
	with pytest.raises(HistoricKeyError):
		windd[1]


def test_set():
	wd = WindowDict()
	assert 0 not in wd
	wd[0] = "foo"
	assert 0 in wd
	assert wd[0] == "foo"
	assert 1 not in wd
	wd[1] = "oof"
	assert 1 in wd
	assert wd[1] == "oof"
	assert 3 not in wd
	wd[3] = "ofo"
	assert 3 in wd
	assert wd[3] == "ofo"
	assert 2 not in wd
	wd[2] = "owo"
	assert 2 in wd
	assert wd[2] == "owo"
	wd[3] = "fof"
	assert wd[3] == "fof"
	wd[0] = "off"
	assert wd[0] == "off"
	assert 4 not in wd
	wd[4] = WindowDict({"spam": "eggs"})
	assert 4 in wd
	assert wd[4] == {"spam": "eggs"}
	assert 5 not in wd
	with ORM("sqlite:///:memory:") as orm:
		g = orm.new_digraph("g")
		g.node[5] = {"ham": {"spam": "beans"}}
		wd[5] = g.node[5]["ham"]
		assert wd[5] == {"spam": "beans"}
		assert wd[5] == g.node[5]["ham"]
