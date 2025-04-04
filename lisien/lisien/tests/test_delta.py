import pytest


@pytest.fixture(params=["turn-delta", "branch-delta", "slow-delta"])
def codepath(request):
	return request.param


def test_character_existence_delta(serial_engine, codepath):
	eng = serial_engine
	eng.add_character(1)
	eng.add_character(2)
	eng.next_turn()
	eng.add_character(3)
	if codepath == "slow-delta":
		eng.branch = "branch"
	elif codepath == "branch-delta":
		eng.next_turn()
	else:
		assert codepath == "turn-delta"
	del eng.character[2]
	eng.add_character(4)
	delta0 = eng.get_delta(("trunk", 0), ("trunk", 1))
	assert 3 in delta0
	if codepath == "turn-delta":
		assert delta0[2] is None
	else:
		assert 2 not in delta0
	delta1 = eng.get_delta(
		("trunk", 1, 1),
		("branch", 1) if codepath == "slow-delta" else ("trunk", eng.turn),
	)
	assert 2 in delta1
	assert delta1[2] is None


def test_unit_delta(serial_engine, codepath):
	pass


def test_character_stat_delta(serial_engine, codepath):
	pass


def test_node_existence_delta(serial_engine, codepath):
	pass


def test_node_stat_delta(serial_engine, codepath):
	pass


def test_portal_existence_delta(serial_engine, codepath):
	pass


def test_thing_location_delta(serial_engine, codepath):
	pass


def test_character_rulebook_delta(serial_engine, codepath):
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


def test_character_created_delta():
	"""Test whether a delta includes the initial keyframe for characters created"""
	pass
