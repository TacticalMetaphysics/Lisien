import pytest


@pytest.mark.parametrize("slow", [0, 1, 2])
def test_character_existence_delta(serial_engine, slow):
	eng = serial_engine
	eng.add_character(1)
	eng.add_character(2)
	eng.next_turn()
	eng.add_character(3)
	if slow == 2:
		eng.branch = "branch"
	elif slow == 1:
		eng.next_turn()
		eng.next_turn()
	else:
		eng.next_turn()
	del eng.character[2]
	eng.add_character(4)
	delta0 = eng.get_delta(("trunk", 0), ("trunk", 1))
	assert 3 in delta0
	assert 2 not in delta0
	delta1 = eng.get_delta(
		("trunk", 1), ("branch", 1) if slow else ("trunk", eng.turn)
	)
	assert 2 in delta1
	assert delta1[2] is None


def test_unit_delta():
	pass


def test_character_stat_delta():
	pass


def test_node_existence_delta():
	pass


def test_node_stat_delta():
	pass


def test_portal_existence_delta():
	pass


def test_thing_location_delta():
	pass


def test_character_rulebook_delta():
	pass


def test_unit_rulebook_delta():
	pass


def test_character_thing_rulebook_delta():
	pass


def test_character_place_rulebook_delta():
	pass


def test_character_portal_rulebook_delta():
	pass


def test_node_rulebook_delta():
	pass


def test_portal_rulebook_delta():
	pass
