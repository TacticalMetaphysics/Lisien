import pytest

from LiSE.character import grid_2d_8graph


@pytest.fixture
def three3(engy):
	return engy.new_character("3x3", grid_2d_8graph(3, 3))


def test_sim_start(three3):
	@three3.place.rule(neighborhood=1, always=True)
	def did_it_run(place):
		if "it_ran" in place:
			place.character.stat["it_ran"] = (
				place.character.stat.get("it_ran", 0) + 1
			)
		else:
			place["it_ran"] = True

	eng = three3.engine
	eng.next_turn()
	assert three3.place[1, 1]["it_ran"]
	assert "it_ran" not in three3.stat
	eng.next_turn()
	assert three3.place[1, 1]["it_ran"]
	assert three3.stat["it_ran"] == 9
	eng.next_turn()
	assert three3.stat["it_ran"] == 9  # still!
