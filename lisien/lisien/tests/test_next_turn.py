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
"""Tests for the rules engine's basic polling functionality

Make sure that every type of rule gets followed, and that the fact
it was followed got recorded correctly.

"""


def test_character_dot_rule(engine):
	"""Test that a rule on a character is polled correctly"""
	eng = engine
	char = eng.new_character("who")

	@char.rule(always=True)
	def yes(char):
		char.stat["run"] = True

	eng.next_turn()
	btt = eng._btt()
	assert char.stat["run"]
	eng.time = "trunk", 0, 0
	assert "run" not in char.stat
	eng.next_turn()
	assert btt == eng._btt()
	assert char.stat["run"]


def test_unit_dot_rule(engine):
	"""Test that a rule applied to a character's avatars is polled correctly"""
	char = engine.new_character("char")
	graph = engine.new_character("graph")
	av = graph.new_place("av")
	char.add_unit(av)
	starttick = engine.tick

	@char.unit.rule(always=True)
	def yes(av):
		av["run"] = True

	engine.next_turn()
	btt = engine._btt()
	assert av["run"]
	engine.time = "trunk", 0, starttick
	assert "run" not in av
	engine.next_turn()
	assert btt == engine._btt()
	assert av["run"]


def test_thing_dot_rule(engine):
	"""Test that a rule applied to a thing mapping is polled correctly"""
	char = engine.new_character("char")
	place = char.new_place("place")
	thing = place.new_thing("thing")
	starttick = engine.tick

	@char.thing.rule(always=True)
	def yes(thing):
		thing["run"] = True

	engine.next_turn()
	btt = engine._btt()
	assert thing["run"]
	engine.time = "trunk", 0, starttick
	assert "run" not in thing
	engine.next_turn()
	assert btt == engine._btt()
	assert thing["run"]


def test_place_dot_rule(engine):
	"""Test that a rule applied to a place mapping is polled correctly"""
	char = engine.new_character("char")
	place = char.new_place("place")
	starttick = engine.tick

	@char.place.rule(always=True)
	def yes(plac):
		plac["run"] = True

	engine.next_turn()
	btt = engine._btt()
	assert place["run"]
	engine.time = "trunk", 0, starttick
	assert "run" not in place
	engine.next_turn()
	assert btt == engine._btt()
	assert place["run"]


def test_portal_dot_rule(engine):
	"""Test that a rule applied to a portal mapping is polled correctly"""
	char = engine.new_character("char")
	orig = char.new_place("orig")
	dest = char.new_place("dest")
	port = orig.new_portal(dest)
	starttick = engine.tick

	@char.portal.rule(always=True)
	def yes(portl):
		portl["run"] = True

	engine.next_turn()
	btt = engine._btt()
	assert port["run"]
	engine.time = "trunk", 0, starttick
	assert "run" not in port
	engine.next_turn()
	assert btt == engine._btt()
	assert port["run"]


def test_node_rule(engine):
	"""Test that a rule applied to one node is polled correctly"""
	char = engine.new_character("char")
	place = char.new_place("place")
	thing = place.new_thing("thing")
	starttick = engine.tick

	@place.rule(always=True)
	def yes(plac):
		plac["run"] = True

	@thing.rule(always=True)
	def definitely(thig):
		thig["run"] = True

	engine.next_turn()
	btt = engine._btt()
	assert place["run"]
	assert thing["run"]
	engine.time = "trunk", 0, starttick
	assert "run" not in place
	assert "run" not in thing
	engine.next_turn()
	assert btt == engine._btt()
	assert place["run"]
	assert thing["run"]


def test_portal_rule(engine):
	"""Test that a rule applied to one portal is polled correctly"""
	char = engine.new_character("char")
	orig = char.new_place("orig")
	dest = char.new_place("dest")
	port = orig.new_portal(dest)
	starttick = engine.tick

	@port.rule(always=True)
	def yes(portl):
		portl["run"] = True

	engine.next_turn()
	btt = engine._btt()
	assert port["run"]
	engine.time = "trunk", 0, starttick
	assert "run" not in port
	engine.next_turn()
	assert btt == engine._btt()
	assert port["run"]


def test_post_time_travel_increment(engine):
	"""Test that, when the rules are run after time travel resulting in
	a tick greater than zero, we advance to the next turn before running rules

	"""
	char = engine.new_character("char")
	char.stat["something"] = 0
	place = char.new_place("there")
	place["otherthing"] = 0

	@char.rule(always=True)
	def incr(chara):
		chara.stat["something"] += 1

	@place.rule(always=True)
	def decr(plac):
		plac["otherthing"] -= 1

	engine.next_turn()
	engine.next_turn()
	assert engine.tick == 2
	engine.branch = "branch1"
	assert engine.tick == 2
	engine.next_turn()
	assert engine.tick == 2
	engine.turn = 2
	engine.branch = "trunk"
	assert engine.tick == 2
	engine.next_turn()
	assert engine.tick == 2
