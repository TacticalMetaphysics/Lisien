import tempfile
from time import monotonic

from LiSE.handle import EngineHandle
from LiSE.examples.polygons import install

with tempfile.TemporaryDirectory() as dir:
	hand = EngineHandle((dir, ), )
	install(hand._real)
	print(-1)
	hand._real.cache_arrange_queue.join()
	ts = monotonic()
	print(0)
	hand.time_travel('trunk', 2)
	print(1, monotonic() - ts)
	ts = monotonic()
	hand.time_travel('truck', 1)
	print(2, monotonic() - ts)
	ts = monotonic()
	hand.time_travel('trunk', 0)
	print(3, monotonic() - ts)
	ts = monotonic()
	hand.time_travel('truck', 1)
	print(4, monotonic() - ts)
	hand.close()
