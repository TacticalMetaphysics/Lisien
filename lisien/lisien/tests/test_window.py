from itertools import cycle

import pytest

from ..exc import HistoricKeyError
from ..window import WindowDict
from .util import make_test_engine

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


def test_past(windd):
	assert list(reversed(range(100))) == list(windd.past(101))
	assert list(reversed(range(51))) == list(windd.past(50))
	unseen = testdata[51:]
	seen = testdata[:51]
	seen.reverse()
	assert seen == list(windd.past(50).items())
	for item in seen:
		assert item[0] in windd.past(50)
		assert item[0] in windd.past(50).keys()
		assert item in windd.past(50).items()
		assert item[1] in windd.past(50).values()
	for item in windd.past(50).items():
		assert item in seen
	for item in unseen:
		assert item[0] not in windd.past(50)
		assert item[0] not in windd.past(50).keys()
		assert item not in windd.past(50).items()
		assert item[1] not in windd.past(50).values()


def test_future(windd):
	assert [] == list(windd.future(101))
	assert list(range(100)) == list(windd.future(-1))
	for item in testdata:
		assert item in windd.future(-1).items()
	assert list(range(51, 100)) == list(windd.future(50))
	unseen = testdata[51:]
	seen = testdata[:51]
	assert unseen == list(windd.future(50).items())
	for item in unseen:
		assert item[0] in windd.future(50)
		assert item[0] in windd.future(50).keys()
		assert item in windd.future(50).items()
		assert item[1] in windd.future(50).values()
	for item in windd.future(50).items():
		assert item in unseen
	for item in seen:
		assert item[0] not in windd.future(50)
		assert item[0] not in windd.future(50).keys()
		assert item not in windd.future(50).items()
		assert item[1] not in windd.future(50).values()


def test_empty():
	empty = WindowDict()
	items = empty.items()
	past = empty.past(0)
	future = empty.future(0)
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
	assert list(windd[-1:1]) == [
		windd[i] for i in range(max(windd) - 1, 1, -1)
	]


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


def test_set(tmp_path):
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
	with make_test_engine(tmp_path, "serial", "sqlite") as orm:
		g = orm.new_character("g")
		g.node[5] = {"ham": {"spam": "beans"}}
		wd[5] = g.node[5]["ham"]
		assert wd[5] == {"spam": "beans"}
		assert wd[5] == g.node[5]["ham"]
