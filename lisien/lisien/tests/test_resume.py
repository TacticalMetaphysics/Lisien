import pytest

from lisien import Engine
from lisien.examples.polygons import install

from .util import make_test_engine_kwargs


# TODO: use a test sim that does everything in every cache
@pytest.mark.big
def test_resume(tmp_path, non_null_database):
	ekwargs = make_test_engine_kwargs(
		tmp_path, "serial", non_null_database, 69105
	)
	ekwargs["keyframe_on_close"] = False
	with Engine(**ekwargs) as eng:
		install(eng)
		eng.next_turn()
		last_branch, last_turn, last_tick = eng._btt()
	with Engine(**ekwargs) as eng:
		assert eng._btt() == (last_branch, last_turn, last_tick)
		curturn = eng.turn
		eng.next_turn()
		assert eng.turn == curturn + 1
