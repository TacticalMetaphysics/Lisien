import pytest


@pytest.fixture(params=["branch-delta", "slow-delta"])
def codepath(request):
	return request.param


def test_character_existence_delta(null_engine, codepath):
	eng = null_engine
	eng.add_character(1)
	eng.add_character(2)
	eng.next_turn()
	eng.add_character(3)
	if codepath == "slow-delta":
		eng.branch = "branch"
	else:
		eng.next_turn()
	del eng.character[2]
	eng.add_character(4)
	delta0 = eng.get_delta(("trunk", 0, 0), tuple(eng.time))
	assert 3 in delta0 and delta0[3] == {
		"character_place_rulebook": ("character_place_rulebook", 3),
		"character_portal_rulebook": ("character_portal_rulebook", 3),
		"character_rulebook": ("character_rulebook", 3),
		"character_thing_rulebook": ("character_thing_rulebook", 3),
		"unit_rulebook": ("unit_rulebook", 3),
	}
	assert 2 in delta0 and delta0[2] is ...
	delta1 = eng.get_delta(
		("trunk", 1, 1),
		("branch", 1) if codepath == "slow-delta" else ("trunk", eng.turn),
	)
	assert 2 in delta1
	assert delta1[2] is ...


def test_unit_delta(null_engine, codepath):
	eng = null_engine
	one = eng.new_character(1)
	six = one.new_place(6)
	seven = six.new_thing(7)
	one.new_place(8)
	two = eng.new_character(2)
	three = two.new_place(3)
	four = three.new_thing(4)
	two.add_place(5)
	one.add_unit(three)
	one.add_unit(four)
	time_a = tuple(eng.time)
	if codepath == "slow-delta":
		eng.branch = "branch"
	else:
		eng.next_turn()
	two.add_unit(six)
	two.add_unit(seven)
	one.remove_unit(three)
	one.remove_unit(four)
	time_b = tuple(eng.time)
	delta0 = eng.get_delta(time_a, time_b)
	assert delta0[1]["units"] == {2: {3: False, 4: False}}
	assert delta0[2]["units"] == {1: {6: True, 7: True}}
	delta1 = eng.get_delta(time_b, time_a)
	assert delta1[1]["units"] == {2: {3: True, 4: True}}
	assert delta1[2]["units"] == {1: {6: False, 7: False}}


def test_character_stat_delta(null_engine, codepath):
	eng = null_engine
	one = eng.new_character(1)
	two = eng.new_character(2)
	one.stat[3] = 4
	one.stat[5] = 6
	one.stat[7] = 8
	two.stat[11] = 12
	time_a = tuple(eng.time)
	if codepath == "branch-delta":
		eng.next_turn()
	else:
		eng.branch = "branch"
	del one.stat[3]
	del one.stat[5]
	two.stat[9] = 10
	time_b = tuple(eng.time)
	delta0 = eng.get_delta(time_a, time_b)
	assert delta0[1][3] is ...
	assert delta0[1][5] is ...
	assert 7 not in delta0[1]
	assert delta0[2][9] == 10
	assert 11 not in delta0[2]
	delta1 = eng.get_delta(time_b, time_a)
	assert delta1[1][3] == 4
	assert delta1[1][5] == 6
	assert 7 not in delta1[1]
	assert delta1[2][9] is ...
	assert 11 not in delta1[2]


def test_node_existence_delta(null_engine, codepath):
	eng = null_engine
	one = eng.new_character(1)
	two = one.new_place(2)
	two.add_thing(3)
	one.add_place(4)
	one.add_thing(5, 4)
	six = eng.new_character(6)
	six.add_place(7)
	six.add_thing(8, 7)
	time_a = tuple(eng.time)
	if codepath == "slow-delta":
		eng.branch = "branch"
	else:
		eng.next_turn()
	del one.place[2]
	six.add_place(9)
	six.add_thing(10, 9)
	time_b = tuple(eng.time)
	delta0 = eng.get_delta(time_a, time_b)
	assert delta0[1]["nodes"] == {3: False, 2: False}
	assert delta0[6]["nodes"] == {9: True, 10: True}
	delta1 = eng.get_delta(time_b, time_a)
	assert delta1[1]["nodes"] == {3: True, 2: True}
	assert delta1[6]["nodes"] == {9: False, 10: False}


def test_node_stat_delta(null_engine, codepath):
	eng = null_engine
	ch = eng.new_character("me")
	one = ch.new_place(1)
	two = ch.new_place(2)
	one[3] = 4
	two[5] = 6
	two[11] = 0
	one[7] = 8
	one[9] = 10
	time_a = tuple(eng.time)
	if codepath == "slow-delta":
		eng.branch = "branch"
	else:
		eng.next_turn()
	del one[7]
	del one[9]
	two[11] = 12
	two[13] = 14
	time_b = tuple(eng.time)
	delta0 = eng.get_delta(time_a, time_b)
	assert delta0["me"]["node_val"] == {
		1: {7: ..., 9: ...},
		2: {11: 12, 13: 14},
	}
	delta1 = eng.get_delta(time_b, time_a)
	assert delta1["me"]["node_val"] == {1: {7: 8, 9: 10}, 2: {11: 0, 13: ...}}


def test_portal_existence_delta(null_engine, codepath):
	pass


def test_thing_location_delta(null_engine, codepath):
	pass


def test_character_rulebook_delta(null_engine, codepath):
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
