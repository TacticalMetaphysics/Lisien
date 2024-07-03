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
import pytest

from LiSE.allegedb import OutOfTimelineError


@pytest.fixture(scope="function")
def chara(engy):
	yield engy.new_character("chara")


def test_many_things_in_place(chara):
	place = chara.new_place(0)
	things = [place.new_thing(i) for i in range(1, 10)]
	for thing in things:
		assert thing in place.contents()
	for that in place.content:
		assert place.content[that].location == place
	things.sort(key=lambda th: th.name)
	contents = sorted(place.contents(), key=lambda th: th.name)
	assert things == contents


def test_contents_over_time(chara):
	place = chara.new_place(0)
	correct_contents = set()
	for i in range(10):
		chara.engine.next_turn()
		with chara.engine.advancing():
			place.new_thing(chara.engine.turn)
			del chara.thing[chara.engine.turn]
			assert set(place.content.keys()) == correct_contents
			place.new_thing(chara.engine.turn)
			correct_contents.add(chara.engine.turn)
		assert set(place.content.keys()) == correct_contents
	del chara.thing[9]
	correct_contents.remove(9)
	assert set(place.content.keys()) == correct_contents
	del chara.thing[8]
	correct_contents.remove(8)
	assert set(place.content.keys()) == correct_contents
	chara.engine.turn = 5
	with pytest.raises(OutOfTimelineError):
		del chara.thing[5]
	chara.engine.branch = "b"
	del chara.thing[5]
	assert set(place.content.keys()) == {1, 2, 3, 4}


def test_contents_in_plan(chara):
	place = chara.new_place(0)
	correct_contents = {1, 2, 3, 4, 5}
	for th in correct_contents:
		place.new_thing(th)
	with chara.engine.plan():
		for i in range(6, 15):
			chara.engine.turn += 1
			assert set(place.content) == correct_contents
			place.new_thing(i)
			del chara.thing[i]
			assert set(place.content) == correct_contents
			place.new_thing(i)
			correct_contents.add(i)
			assert set(place.content) == correct_contents
	chara.engine.turn = 4
	assert set(place.content) == {1, 2, 3, 4, 5, 6, 7, 8, 9}
	chara.engine.turn = 2
	assert set(place.content) == {1, 2, 3, 4, 5, 6, 7}
	# this does not contradict the plan
	place.new_thing(15)
	assert set(place.content) == {1, 2, 3, 4, 5, 6, 7, 15}
	chara.engine.turn = 4
	assert set(place.content) == {1, 2, 3, 4, 5, 6, 7, 8, 9, 15}
	# this neither
	there = chara.new_place("there")
	chara.thing[9].location = there
	assert set(place.content) == {1, 2, 3, 4, 5, 6, 7, 8, 15}
	chara.engine.turn = 5
	assert set(place.content) == {1, 2, 3, 4, 5, 6, 7, 8, 10, 15}
	chara.engine.turn = 10
	assert set(place.content) == correct_contents.union({15}).difference({9})
	# but this does
	chara.engine.turn = 5
	place.new_thing(11)
	assert set(place.content) == {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 15}
	chara.engine.turn = 10
	assert set(place.content) == {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 15}
