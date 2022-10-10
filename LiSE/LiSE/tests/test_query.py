# This file is part of LiSE, a framework for life simulation games.
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
from functools import reduce
from collections import defaultdict
from ..engine import Engine
from ..query import windows_intersection, combine_chronological_data_end_turn
import pytest
import os
import shutil
import tempfile

pytestmark = [pytest.mark.slow]


@pytest.fixture(scope='module')
def college24_premade():
	directory = tempfile.mkdtemp('.')
	shutil.unpack_archive(
		os.path.join(os.path.abspath(os.path.dirname(__file__)),
						'college24_premade.tar.xz'), directory)
	with Engine(directory) as eng:
		yield eng
	shutil.rmtree(directory)


def roommate_collisions(college24_premade):
	"""Test queries' ability to tell that all of the students that share
	rooms have been in the same place.

	"""
	engine = college24_premade
	done = set()
	for chara in engine.character.values():
		if chara.name in done:
			continue
		match = re.match(r'dorm(\d)room(\d)student(\d)', chara.name)
		if not match:
			continue
		dorm, room, student = match.groups()
		other_student = '1' if student == '0' else '0'
		student = chara
		other_student = engine.character['dorm{}room{}student{}'.format(
			dorm, room, other_student)]
		cond = student.unit.only.historical(
			'location') == other_student.unit.only.historical('location')
		same_loc_turns = {turn for (branch, turn) in cond.iter_times()}
		assert same_loc_turns, "{} and {} don't seem to share a room".format(
			student.name, other_student.name)
		assert len(
			same_loc_turns
		) >= 6, "{} and {} did not share their room for at least 6 turns".format(
			student.name, other_student.name)

		assert same_loc_turns == engine.turns_when(cond)

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
		stu for stu in engine.character['student_body'].stat['characters']
		if not (stu.stat['drunkard'] or stu.stat['lazy'])
	]

	assert students

	def sameClasstime(stu0, stu1):
		assert list(
			engine.turns_when(
				stu0.unit.only.historical('location') == stu1.unit.only.
				historical('location') == engine.alias('classroom'))
		), """{stu0} seems not to have been in the classroom 
				at the same time as {stu1}.
				{stu0} was there at turns {turns0}
				{stu1} was there at turns {turns1}""".format(
			stu0=stu0.name,
			stu1=stu1.name,
			turns0=list(
				engine.turns_when(
					stu0.unit.only.historical('location') == engine.alias(
						'classroom'))),
			turns1=list(
				engine.turns_when(
					stu1.unit.only.historical('location') == engine.alias(
						'classroom'))))
		return stu1

	reduce(sameClasstime, students)


def test_sober_collisions_premade(college24_premade):
	sober_collisions(college24_premade)


def noncollision(college24_premade):
	"""Make sure students *not* from the same room never go there together"""
	engine = college24_premade
	dorm = defaultdict(lambda: defaultdict(dict))
	for character in engine.character.values():
		match = re.match(r'dorm(\d)room(\d)student(\d)', character.name)
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
								stu0.unit.only.historical('location') ==
								stu1.unit.only.historical('location') ==
								engine.alias('dorm{}room{}'.format(d, r)))
						), "{} seems to share a room with {}".format(
							stu0.name, stu1.name)
				common = 'common{}'.format(d)
				for dd in other_dorms:
					for rr in dorm[dd]:
						for stu1 in dorm[dd][rr].values():
							assert not list(
								engine.turns_when(
									stu0.unit.only.historical('location') ==
									stu1.unit.only.historical(
										'location') == engine.alias(common))
							), "{} seems to have been in the same common room  as {}".format(
								stu0.name, stu1.name)


def test_noncollision_premade(college24_premade):
	noncollision(college24_premade)


def test_windows_intersection():
	assert windows_intersection([(2, None), (0, 1)]) == []
	assert windows_intersection([(1, 2), (0, 1)]) == [(1, 1)]


def test_graph_val_select_eq(engy):
	me = engy.new_character('me')
	me.stat['foo'] = 'bar'
	me.stat['qux'] = 'bas'
	engy.next_turn()
	me.stat['foo'] = 'bas'
	me.stat['qux'] = 'bar'
	engy.next_turn()
	engy.branch = 'leaf'
	me.stat['foo'] = 'bar'
	engy.next_turn()
	me.stat['foo'] = 'bas'
	me.stat['qux'] = 'bas'
	foo_alias = me.historical('foo')
	qux_alias = me.historical('qux')
	qry = foo_alias == qux_alias
	assert engy.turns_when(qry) == {1}
	assert engy.turns_when(qry, mid_turn=True) == {1, 2}


def test_graph_val_select_lt_gt(engy):
	me = engy.new_character('me')
	me.stat['foo'] = 10
	me.stat['bar'] = 1
	engy.next_turn()
	me.stat['foo'] = 2
	me.stat['bar'] = 8
	engy.next_turn()
	me.stat['foo'] = 3
	engy.next_turn()
	me.stat['foo'] = 9
	engy.next_turn()
	engy.branch = 'leaf'
	me.stat['bar'] = 5
	engy.next_turn()
	me.stat['bar'] = 2
	engy.next_turn()
	me.stat['bar'] = 10
	engy.next_turn()
	me.stat['bar'] = 1
	engy.next_turn()
	me.stat['bar'] = 10
	foo_hist = me.historical('foo')
	bar_hist = me.historical('bar')
	assert engy.turns_when(foo_hist < bar_hist) == {1, 2, 5, 7}
	assert engy.turns_when(foo_hist > bar_hist) == {0, 3, 4, 6}


def test_combine_chronological_data_end_turn():
	left = [(0, 1, 'foo'), (1, 5, 'boo'), (5, 9, 'gru'), (9, 10, None),
			(10, None, 'foo')]
	right = [(0, 1, 'bar'), (1, 3, 'baz'), (3, None, 'bau')]
	correct = [(0, 1, 'foo', 'bar'), (1, 3, 'boo', 'baz'),
				(3, 5, 'boo', 'bau'), (5, 9, 'gru', 'bau'),
				(9, 10, None, 'bau'), (10, None, 'foo', 'bau')]
	assert combine_chronological_data_end_turn(left, right) == correct
	left = [(1, 2, 'foo')]
	right = [(0, 5, 'bar'), (5, 7, 'bas')]
	correct = [(0, 1, None, 'bar'), (1, 2, 'foo', 'bar'), (2, 5, None, 'bar'),
				(5, 7, None, 'bas')]
	assert combine_chronological_data_end_turn(left, right) == correct
	(left, right) = (right, left)
	correct = [(t[0], t[1], t[3], t[2]) for t in correct]
	assert combine_chronological_data_end_turn(left, right) == correct
	left = [(0, 2, 'foo'), (2, 5, 'bar')]
	right = [(1, 3, 'bas'), (3, 4, 'qux')]
	correct = [(0, 1, 'foo', None), (1, 2, 'foo', 'bas'), (2, 3, 'bar', 'bas'),
				(3, 4, 'bar', 'qux'), (4, 5, 'bar', None)]
	assert combine_chronological_data_end_turn(left, right) == correct
	(left, right) = (right, left)
	correct = [(t[0], t[1], t[3], t[2]) for t in correct]
	assert combine_chronological_data_end_turn(left, right) == correct
	with pytest.raises(ValueError):
		combine_chronological_data_end_turn([], [])
	assert combine_chronological_data_end_turn([(0, 1, 'foo')],
												[]) == [(0, 1, 'foo', None)]
	assert combine_chronological_data_end_turn([], [(0, 1, 'foo')]) == [
		(0, 1, None, 'foo')
	]
	left = [(0, 2, 'foo'), (2, 4, 'bas')]
	right = [(1, 3, 'bar')]
	correct = [(0, 1, 'foo', None), (1, 2, 'foo', 'bar'), (2, 3, 'bas', 'bar'),
				(3, 4, 'bas', None)]
	assert combine_chronological_data_end_turn(left, right) == correct
	(left, right) = (right, left)
	correct = [(t[0], t[1], t[3], t[2]) for t in correct]
	assert combine_chronological_data_end_turn(left, right) == correct
