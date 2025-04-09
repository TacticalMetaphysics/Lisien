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
import networkx as nx
import pytest


@pytest.fixture(scope="function")
def something(proxyless_engine):
	yield (
		proxyless_engine.new_character("physical")
		.new_place("somewhere")
		.new_thing("something")
	)


def test_contents(something):
	pl1 = something.character.place["somewhere"]
	pl2 = something.character.new_place("somewhere2")
	assert something.location == something.character.node["somewhere"]
	assert something.name in pl1.content
	assert something.name not in pl2.content
	assert [something] == list(pl1.contents())
	assert [] == list(pl2.contents())


def test_future_contents(something):
	engine = something.engine
	somewhere = something.location
	with engine.plan():
		engine.turn = 1
		something.location = None
		engine.turn = 2
		somebody = somewhere.new_thing("somebody")
		engine.turn = 0
		someone = somewhere.new_thing("someone")
		sometick = engine.tick
	engine.turn = 0
	engine.tick = sometick
	assert len(somewhere.contents()) == 2
	assert something in somewhere.contents()
	assert somebody not in somewhere.contents()
	assert someone in somewhere.contents()
	engine.turn = 1
	engine.tick = engine.turn_end_plan()
	assert len(somewhere.contents()) == 1
	assert something not in somewhere.contents()
	assert someone in somewhere.contents()
	assert somebody not in somewhere.contents()
	engine.turn = 2
	engine.tick = engine.turn_end_plan()
	assert len(somewhere.contents()) == 2
	assert somebody in somewhere.contents()
	assert someone in somewhere.contents()
	assert something not in somewhere.contents()


def test_travel(proxyless_engine):
	phys = proxyless_engine.new_character(
		"physical", data=nx.grid_2d_graph(8, 8)
	)
	del phys.place[1, 1]
	del phys.place[6, 1]
	thing1 = phys.place[0, 0].new_thing(1)
	thing2 = phys.place[7, 0].new_thing(2)
	thing1.travel_to(phys.place[7, 7])
	thing2.travel_to(phys.place[0, 7])
	for _ in range(15):
		proxyless_engine.next_turn()
	assert thing1.location == phys.place[7, 7]
	assert thing2.location == phys.place[0, 7]
	thing1.go_to_place(phys.place[6, 7])
	thing2.go_to_place(phys.place[1, 7])
	proxyless_engine.next_turn()
	proxyless_engine.next_turn()
	assert thing1.location == phys.place[6, 7]
	assert thing2.location == phys.place[1, 7]


def test_neighbors_no_successors(sqleng):
	phys = sqleng.new_character("physical")
	here = phys.new_place("here")
	phys.new_place("there")
	assert list(here.neighbors()) == []
