from allegedb.cache import WindowDict
from itertools import cycle

testvs = ['a', 99, ['spam', 'eggs', 'ham'], {'foo': 'bar', 0: 1, '💧': '🔑'}]
testdata = []
for k, v in zip(range(100), cycle(testvs)):
    testdata.append((k, v))


windd = WindowDict(testdata)


assert list(range(100)) == list(windd.keys())
for item in testdata:
    assert item in windd.items()
assert list(reversed(range(100))) == list(windd.past())
assert [] == list(windd.future())
windd.seek(-1)
assert list(range(100)) == list(windd.future())
for item in testdata:
    assert item in windd.future().items()
windd.seek(50)
unseen = []
seen = []
assert list(reversed(range(51))) == list(windd.past())
for item in testdata:
    if item not in windd.past().items():
        unseen.append(item)
    else:
        seen.append(item)
assert list(range(51, 100)) == list(windd.future())
for item in unseen:
    assert item in windd.future().items()
for item in windd.future().items():
    assert item in unseen
for item in seen:
    assert item in windd.past().items()
for item in windd.past().items():
    assert item in seen