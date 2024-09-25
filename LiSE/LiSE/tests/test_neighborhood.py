import pytest

from LiSE.character import grid_2d_8graph


@pytest.fixture
def three3(engy):
	return engy.new_character("3x3", grid_2d_8graph(3, 3))


def test_sim_start(three3):
	@three3.place.rule(neighborhood=1, always=True)
	def did_it_run(place):
		if "it_ran" in place:
			place["it_ran"] += 1
		else:
			place["it_ran"] = 1

	eng = three3.engine
	eng.next_turn()
	assert three3.place[1, 1]["it_ran"] == 1
	eng.next_turn()
	assert three3.place[1, 1]["it_ran"] == 1
