import pytest

from lisien import Engine
from lisien.examples.polygons import install


# TODO: use a test sim that does everything in every cache
@pytest.mark.big
def test_resume(tmp_path):
	with Engine(tmp_path, keyframe_on_close=False, workers=0) as eng:
		install(eng)
		eng.next_turn()
		last_branch, last_turn, last_tick = eng._btt()
	with Engine(tmp_path, keyframe_on_close=False, workers=0) as eng:
		assert eng._btt() == (last_branch, last_turn, last_tick)
		curturn = eng.turn
		eng.next_turn()
		assert eng.turn == curturn + 1
