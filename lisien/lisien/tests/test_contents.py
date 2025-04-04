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
import pytest


@pytest.fixture(scope="function")
def char(engy):
	yield engy.new_character("chara")


@pytest.fixture
def chara_chron(proxyless_engine):
	yield proxyless_engine.new_character("chara")


def test_many_things_in_place(char):
	place = char.new_place(0)
	things = [place.new_thing(i) for i in range(1, 10)]
	for thing in things:
		assert thing in place.contents()
	for that in place.content:
		assert place.content[that].location == place
	things.sort(key=lambda th: th.name)
	contents = sorted(place.contents(), key=lambda th: th.name)
	assert things == contents


def test_contents_over_time(chara_chron):
	chara = chara_chron
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
	chara.engine.branch = "bb"
	del chara.thing[5]
	assert set(place.content.keys()) == {1, 2, 3, 4}


def test_contents_in_plan(chara_chron):
	chara = chara_chron
	place = chara.new_place(0)
	correct_contents = {1, 2, 3, 4, 5}
	for th in correct_contents:
		place.new_thing(th)
	engine = chara.engine
	with engine.plan():
		for i in range(6, 15):
			engine.turn += 1
			assert set(place.content) == correct_contents
			place.new_thing(i)
			del chara.thing[i]
			assert set(place.content) == correct_contents
			place.new_thing(i)
			correct_contents.add(i)
			assert set(place.content) == correct_contents
		engine.turn = 4
		assert set(place.content) == {1, 2, 3, 4, 5, 6, 7, 8, 9}
	assert engine.turn == 0
	assert set(place.content) == {1, 2, 3, 4, 5}
	engine.next_turn()
	engine.next_turn()
	engine.tick = engine.turn_end_plan()
	assert set(place.content) == {1, 2, 3, 4, 5, 6, 7}
	# this does not contradict the plan
	place.new_thing(15)
	assert set(place.content) == {1, 2, 3, 4, 5, 6, 7, 15}
	engine.next_turn()
	engine.tick = engine.turn_end_plan()
	assert set(place.content) == {1, 2, 3, 4, 5, 6, 7, 8, 15}
	engine.next_turn()
	engine.next_turn()
	assert engine.turn == 5
	engine.tick = engine.turn_end_plan()
	assert set(place.content) == {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15}
	# this neither
	there = chara.new_place("there")
	# but this does
	chara.thing[9].location = there
	assert set(place.content) == {1, 2, 3, 4, 5, 6, 7, 8, 10, 15}
	engine.turn = 10
	assert set(place.content) == (correct_contents - {9}) | {15}
