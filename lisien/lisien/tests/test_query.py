# This file is part of Lisien, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import re
from collections import defaultdict
from functools import reduce

import pytest

from ..query import windows_intersection

pytestmark = [pytest.mark.slow, pytest.mark.big]


def roommate_collisions(college24_premade):
	"""Test queries' ability to tell that all of the students that share
	rooms have been in the same place.

	"""
	engine = college24_premade
	done = set()
	for chara in engine.character.values():
		if chara.name in done:
			continue
		match = re.match(r"dorm(\d)room(\d)student(\d)", chara.name)
		if not match:
			continue
		dorm, room, student = match.groups()
		other_student = "1" if student == "0" else "0"
		student = chara
		other_student = engine.character[
			"dorm{}room{}student{}".format(dorm, room, other_student)
		]
		cond = student.unit.only.historical(
			"location"
		) == other_student.unit.only.historical("location")
		same_loc_turns = {turn for (branch, turn) in cond._iter_times()}
		assert same_loc_turns, "{} and {} don't seem to share a room".format(
			student.name, other_student.name
		)
		assert len(same_loc_turns) >= 6, (
			"{} and {} did not share their room for at least 6 turns".format(
				student.name, other_student.name
			)
		)

		# *BOTH* _iter_times *AND* turns_when are inconsistent about whether
		# to include the present turn. Fix later...
		assert same_loc_turns - {24} == set(engine.turns_when(cond)) - {24}

		done.add(student.name)
		done.add(other_student.name)


def test_roomie_collisions_premade(college24_premade):
	roommate_collisions(college24_premade)


def sober_collisions(college24_premade):
	"""Students that are neither lazy nor drunkards should all have been
	in class together at least once.

	"""
	engine = college24_premade
	students = [
		stu
		for stu in engine.character["student_body"].stat["characters"]
		if not (stu.stat["drunkard"] or stu.stat["lazy"])
	]

	assert students

	def sameClasstime(stu0, stu1):
		assert list(
			engine.turns_when(
				(
					stu0.unit.only.historical("location")
					== stu1.unit.only.historical("location")
				)
				& (stu1.unit.only.historical("location") == "classroom")
			)
		), """{stu0} seems not to have been in the classroom 
				at the same time as {stu1}.
				{stu0} was there at turns {turns0}
				{stu1} was there at turns {turns1}""".format(
			stu0=stu0.name,
			stu1=stu1.name,
			turns0=list(
				engine.turns_when(
					stu0.unit.only.historical("location") == "classroom"
				)
			),
			turns1=list(
				engine.turns_when(
					stu1.unit.only.historical("location") == "classroom"
				)
			),
		)
		return stu1

	reduce(sameClasstime, students)


@pytest.mark.skip(
	"I think the underlying sim is not doing what I expect, "
	"so this test doesn't tell me anything"
)
def test_sober_collisions_premade(college24_premade):
	sober_collisions(college24_premade)


def noncollision(college24_premade):
	"""Make sure students *not* from the same room never go there together"""
	engine = college24_premade
	dorm = defaultdict(lambda: defaultdict(dict))
	for character in engine.character.values():
		match = re.match(r"dorm(\d)room(\d)student(\d)", character.name)
		if not match:
			continue
		d, r, s = match.groups()
		dorm[d][r][s] = character
	for d in dorm:
		other_dorms = [dd for dd in dorm if dd != d]
		for r in dorm[d]:
			other_rooms = [rr for rr in dorm[d] if rr != r]
			for stu0 in dorm[d][r].values():
				for rr in other_rooms:
					for stu1 in dorm[d][rr].values():
						assert not list(
							engine.turns_when(
								stu0.unit.only.historical("location")
								== stu1.unit.only.historical("location")
								== "dorm{}room{}".format(d, r)
							)
						), "{} seems to share a room with {}".format(
							stu0.name, stu1.name
						)
				common = "common{}".format(d)
				for dd in other_dorms:
					for rr in dorm[dd]:
						for stu1 in dorm[dd][rr].values():
							assert not list(
								engine.turns_when(
									stu0.unit.only.historical("location")
									== stu1.unit.only.historical("location")
									== common
								)
							), (
								"{} seems to have been in the same common room  as {}".format(
									stu0.name, stu1.name
								)
							)


def test_noncollision_premade(college24_premade):
	noncollision(college24_premade)


def test_windows_intersection():
	assert windows_intersection([(2, None), (0, 1)]) == []
	assert windows_intersection([(1, 2), (0, 1)]) == [(1, 1)]


def test_graph_val_select_eq(sqleng):
	assert sqleng.turn == 0
	me = sqleng.new_character("me")
	me.stat["foo"] = "bar"
	me.stat["qux"] = "bas"
	sqleng.next_turn()
	assert sqleng.turn == 1
	me.stat["foo"] = ""
	me.stat["foo"] = "bas"
	me.stat["qux"] = "bar"
	sqleng.next_turn()
	assert sqleng.turn == 2
	me.stat["qux"] = "bas"
	sqleng.next_turn()
	assert sqleng.turn == 3
	me.stat["qux"] = "bar"
	sqleng.next_turn()
	assert sqleng.turn == 4
	sqleng.branch = "leaf"
	assert sqleng.turn == 4
	sqleng.next_turn()
	assert sqleng.turn == 5
	me.stat["foo"] = "bar"
	sqleng.next_turn()
	assert sqleng.turn == 6
	me.stat["foo"] = "bas"
	me.stat["qux"] = "bas"
	sqleng.next_turn()
	assert sqleng.turn == 7
	foo_alias = me.historical("foo")
	qux_alias = me.historical("qux")
	qry = foo_alias == qux_alias
	turn_end_result = sqleng.turns_when(qry)
	assert 5 in turn_end_result
	assert 3 not in turn_end_result
	assert turn_end_result == set(turn_end_result) == {2, 5, 6, 7}
	assert sqleng.turns_when(qry)[-1] == 7
	assert sqleng.turns_when(qry)[0] == 2
	mid_turn_result = sqleng.turns_when(qry, mid_turn=True)
	assert 3 in mid_turn_result
	assert 4 not in mid_turn_result
	assert mid_turn_result == set(mid_turn_result) == {1, 2, 3, 5, 6, 7}
	assert (
		list(sqleng.turns_when(qry)) == list(turn_end_result) == [2, 5, 6, 7]
	)
	assert (
		list(reversed(sqleng.turns_when(qry)))
		== list(reversed(turn_end_result))
		== [7, 6, 5, 2]
	)
	assert sqleng.turns_when(qry, mid_turn=True)[-1] == 7
	assert sqleng.turns_when(qry, mid_turn=True)[0] == 1


def test_graph_nodeval_select_eq(sqleng):
	assert sqleng.turn == 0
	me = sqleng.new_character("me")
	me.stat["foo"] = "bar"
	qux = me.new_place("qux")
	qux["quux"] = "bas"
	sqleng.next_turn()
	assert sqleng.turn == 1
	me.stat["foo"] = ""
	me.stat["foo"] = "bas"
	qux["quux"] = "bar"
	sqleng.next_turn()
	assert sqleng.turn == 2
	qux["quux"] = "bas"
	sqleng.next_turn()
	assert sqleng.turn == 3
	qux["quux"] = "bar"
	sqleng.next_turn()
	assert sqleng.turn == 4
	sqleng.branch = "leaf"
	assert sqleng.turn == 4
	sqleng.next_turn()
	assert sqleng.turn == 5
	me.stat["foo"] = "bar"
	sqleng.next_turn()
	assert sqleng.turn == 6
	me.stat["foo"] = "bas"
	qux["quux"] = "bas"
	sqleng.next_turn()
	assert sqleng.turn == 7
	foo_alias = me.historical("foo")
	qux_alias = qux.historical("quux")
	qry = foo_alias == qux_alias
	turn_end_result = sqleng.turns_when(qry)
	assert 5 in turn_end_result
	assert 3 not in turn_end_result
	assert turn_end_result == set(turn_end_result) == {2, 5, 6, 7}
	assert sqleng.turns_when(qry)[-1] == 7
	assert sqleng.turns_when(qry)[0] == 2
	mid_turn_result = sqleng.turns_when(qry, mid_turn=True)
	assert 3 in mid_turn_result
	assert 4 not in mid_turn_result
	assert mid_turn_result == set(mid_turn_result) == {1, 2, 3, 5, 6, 7}
	assert (
		list(sqleng.turns_when(qry)) == list(turn_end_result) == [2, 5, 6, 7]
	)
	assert (
		list(reversed(sqleng.turns_when(qry)))
		== list(reversed(turn_end_result))
		== [7, 6, 5, 2]
	)
	assert sqleng.turns_when(qry, mid_turn=True)[-1] == 7
	assert sqleng.turns_when(qry, mid_turn=True)[0] == 1


def test_location_qry(sqleng):
	phys = sqleng.new_character("physical")
	place1 = phys.new_place(1)
	place2 = phys.new_place(2)
	place3 = phys.new_place(3)
	thing1 = place3.new_thing("t1")
	thing2 = place1.new_thing("t2")
	sqleng.next_turn()
	thing1.location = place2
	thing2.location = place2
	thing2.location = place1
	sqleng.next_turn()
	thing2.location = place2
	qry = thing1.historical("location") == thing2.historical("location")
	res0 = sqleng.turns_when(qry)
	assert 2 in res0
	assert 1 not in res0
	assert list(res0) == [2]
	res1 = sqleng.turns_when(qry, mid_turn=True)
	assert 0 not in res1
	assert 1 in res1
	assert 2 in res1
	assert list(res1) == [1, 2]


def test_place_val_qry(sqleng):
	phys = sqleng.new_character("physical")
	place1 = phys.new_place(1)
	place2 = phys.new_place(2)
	assert sqleng.turn == 0
	place1["flavor"] = "delicious"
	place2["flavor"] = "disgusting"
	sqleng.next_turn()
	assert sqleng.turn == 1
	place2["flavor"] = "delicious"
	sqleng.next_turn()
	assert sqleng.turn == 2
	place1["flavor"] = "disgusting"
	sqleng.next_turn()
	assert sqleng.turn == 3
	place2["flavor"] = "disgusting"
	qry = place1.historical("flavor") == place2.historical("flavor")
	res = sqleng.turns_when(qry)
	assert 1 in res
	assert 3 in res
	assert 0 not in res
	assert 2 not in res
	assert set(res) == {1, 3}


@pytest.mark.skip("I'll optimize later")
@pytest.mark.slow
def test_stress_graph_val_select_eq(sqleng):
	import random
	from time import monotonic

	me = sqleng.new_character("me")
	me.stat["qux"] = random.choice(["foo", "bar", "bas"])
	me.stat["quux"] = random.choice(["foo", "bar", "bas"])
	for i in range(10000):
		sqleng.next_turn()
		me.stat["qux"] = random.choice(["foo", "bar", "bas"])
		me.stat["quux"] = random.choice(["foo", "bar", "bas"])
	qry = me.historical("qux") == me.historical("quux")
	start_ts = monotonic()
	res = sqleng.turns_when(qry)
	assert monotonic() - start_ts < 1
	start_ts = monotonic()
	rez = list(res)
	print(len(rez))
	assert monotonic() - start_ts < 1


def test_graph_val_select_lt_gt(sqleng):
	me = sqleng.new_character("me")
	me.stat["foo"] = 10
	me.stat["bar"] = 1
	sqleng.next_turn()
	me.stat["foo"] = 2
	me.stat["bar"] = 8
	sqleng.next_turn()
	me.stat["foo"] = 3
	sqleng.next_turn()
	me.stat["foo"] = 9
	sqleng.next_turn()
	sqleng.branch = "leaf"
	me.stat["bar"] = 5
	sqleng.next_turn()
	me.stat["bar"] = 2
	sqleng.next_turn()
	me.stat["bar"] = 10
	sqleng.next_turn()
	me.stat["bar"] = 1
	sqleng.next_turn()
	me.stat["bar"] = 10
	foo_hist = me.historical("foo")
	bar_hist = me.historical("bar")
	res = sqleng.turns_when(foo_hist < bar_hist)
	assert set(res) == {1, 2, 6, 8}
	assert str([1, 2, 6, 8]) in str(sqleng.turns_when(foo_hist < bar_hist))
	res = sqleng.turns_when(foo_hist > bar_hist)
	assert set(res) == {0, 3, 4, 5, 7}


@pytest.mark.skip("I'll optimize later")
@pytest.mark.slow
def test_stress_graph_val_select_lt(sqleng):
	import random
	from time import monotonic

	me = sqleng.new_character("me")
	me.stat["foo"] = random.randrange(0, 10)
	me.stat["bar"] = random.randrange(0, 10)
	for i in range(10000):
		sqleng.next_turn()
		me.stat["foo"] = random.randrange(0, 10)
		me.stat["bar"] = random.randrange(0, 10)
	qry = me.historical("foo") < me.historical("bar")
	sqleng.commit()
	start_ts = monotonic()
	res = sqleng.turns_when(qry)
	assert monotonic() - start_ts < 1
	start_ts = monotonic()
	rez = list(res)
	print(len(rez))
	assert monotonic() - start_ts < 1


def test_graph_val_compound(sqleng):
	you = sqleng.new_character("you")
	assert sqleng.turn == 0
	me = sqleng.new_character("me")
	me.stat["foo"] = "bar"
	me.stat["qux"] = "bas"
	you.stat["foo"] = 10
	you.stat["bar"] = 1
	sqleng.next_turn()
	assert sqleng.turn == 1
	me.stat["foo"] = ""
	me.stat["foo"] = "bas"
	me.stat["qux"] = "bar"
	you.stat["foo"] = 2
	you.stat["bar"] = 8
	sqleng.next_turn()
	assert sqleng.turn == 2
	me.stat["qux"] = "bas"
	you.stat["foo"] = 3
	sqleng.next_turn()
	assert sqleng.turn == 3
	me.stat["qux"] = "bar"
	you.stat["foo"] = 9
	sqleng.next_turn()
	assert sqleng.turn == 4
	sqleng.branch = "leaf"
	assert sqleng.turn == 4
	you.stat["bar"] = 5
	sqleng.next_turn()
	assert sqleng.turn == 5
	me.stat["foo"] = "bar"
	you.stat["bar"] = 2
	sqleng.next_turn()
	assert sqleng.turn == 6
	me.stat["foo"] = "bas"
	me.stat["qux"] = "bas"
	you.stat["bar"] = 10
	sqleng.next_turn()
	you.stat["bar"] = 1
	assert sqleng.turn == 7
	sqleng.next_turn()
	assert sqleng.turn == 8
	you.stat["bar"] = 10
	eq_qry = me.historical("foo") == me.historical("qux")
	correct_eq = {2, 5, 6, 7, 8}
	assert set(sqleng.turns_when(eq_qry)) == correct_eq
	lt_qry = you.historical("foo") < you.historical("bar")
	correct_lt = {1, 2, 6, 8}
	assert set(sqleng.turns_when(lt_qry)) == correct_lt
	assert sqleng.turns_when(lt_qry & eq_qry) == correct_eq & correct_lt
	assert sqleng.turns_when(lt_qry | eq_qry) == correct_eq | correct_lt
	assert sqleng.turns_when(lt_qry - eq_qry) == correct_lt - correct_eq
	assert sqleng.turns_when(eq_qry - lt_qry) == correct_eq - correct_lt
