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


def test_character_dot_rule(proxyless_engine):
	"""Test that a rule on a character is polled correctly"""
	eng = proxyless_engine
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


def test_unit_dot_rule(proxyless_engine):
	"""Test that a rule applied to a character's avatars is polled correctly"""
	char = proxyless_engine.new_character("char")
	graph = proxyless_engine.new_character("graph")
	av = graph.new_place("av")
	char.add_unit(av)
	starttick = proxyless_engine.tick

	@char.unit.rule(always=True)
	def yes(av):
		av["run"] = True

	proxyless_engine.next_turn()
	btt = proxyless_engine._btt()
	assert av["run"]
	proxyless_engine.time = "trunk", 0, starttick
	assert "run" not in av
	proxyless_engine.next_turn()
	assert btt == proxyless_engine._btt()
	assert av["run"]


def test_thing_dot_rule(proxyless_engine):
	"""Test that a rule applied to a thing mapping is polled correctly"""
	char = proxyless_engine.new_character("char")
	place = char.new_place("place")
	thing = place.new_thing("thing")
	starttick = proxyless_engine.tick

	@char.thing.rule(always=True)
	def yes(thing):
		thing["run"] = True

	proxyless_engine.next_turn()
	btt = proxyless_engine._btt()
	assert thing["run"]
	proxyless_engine.time = "trunk", 0, starttick
	assert "run" not in thing
	proxyless_engine.next_turn()
	assert btt == proxyless_engine._btt()
	assert thing["run"]


def test_place_dot_rule(proxyless_engine):
	"""Test that a rule applied to a place mapping is polled correctly"""
	char = proxyless_engine.new_character("char")
	place = char.new_place("place")
	starttick = proxyless_engine.tick

	@char.place.rule(always=True)
	def yes(plac):
		plac["run"] = True

	proxyless_engine.next_turn()
	btt = proxyless_engine._btt()
	assert place["run"]
	proxyless_engine.time = "trunk", 0, starttick
	assert "run" not in place
	proxyless_engine.next_turn()
	assert btt == proxyless_engine._btt()
	assert place["run"]


def test_portal_dot_rule(proxyless_engine):
	"""Test that a rule applied to a portal mapping is polled correctly"""
	char = proxyless_engine.new_character("char")
	orig = char.new_place("orig")
	dest = char.new_place("dest")
	port = orig.new_portal(dest)
	starttick = proxyless_engine.tick

	@char.portal.rule(always=True)
	def yes(portl):
		portl["run"] = True

	proxyless_engine.next_turn()
	btt = proxyless_engine._btt()
	assert port["run"]
	proxyless_engine.time = "trunk", 0, starttick
	assert "run" not in port
	proxyless_engine.next_turn()
	assert btt == proxyless_engine._btt()
	assert port["run"]


def test_node_rule(proxyless_engine):
	"""Test that a rule applied to one node is polled correctly"""
	char = proxyless_engine.new_character("char")
	place = char.new_place("place")
	thing = place.new_thing("thing")
	starttick = proxyless_engine.tick

	@place.rule(always=True)
	def yes(plac):
		plac["run"] = True

	@thing.rule(always=True)
	def definitely(thig):
		thig["run"] = True

	proxyless_engine.next_turn()
	btt = proxyless_engine._btt()
	assert place["run"]
	assert thing["run"]
	proxyless_engine.time = "trunk", 0, starttick
	assert "run" not in place
	assert "run" not in thing
	proxyless_engine.next_turn()
	assert btt == proxyless_engine._btt()
	assert place["run"]
	assert thing["run"]


def test_portal_rule(proxyless_engine):
	"""Test that a rule applied to one portal is polled correctly"""
	char = proxyless_engine.new_character("char")
	orig = char.new_place("orig")
	dest = char.new_place("dest")
	port = orig.new_portal(dest)
	starttick = proxyless_engine.tick

	@port.rule(always=True)
	def yes(portl):
		portl["run"] = True

	proxyless_engine.next_turn()
	btt = proxyless_engine._btt()
	assert port["run"]
	proxyless_engine.time = "trunk", 0, starttick
	assert "run" not in port
	proxyless_engine.next_turn()
	assert btt == proxyless_engine._btt()
	assert port["run"]


def test_post_time_travel_increment(proxyless_engine):
	"""Test that, when the rules are run after time travel resulting in
	a tick greater than zero, we advance to the next turn before running rules

	"""
	char = proxyless_engine.new_character("char")
	char.stat["something"] = 0
	place = char.new_place("there")
	place["otherthing"] = 0

	@char.rule(always=True)
	def incr(chara):
		chara.stat["something"] += 1

	@place.rule(always=True)
	def decr(plac):
		plac["otherthing"] -= 1

	proxyless_engine.next_turn()
	proxyless_engine.next_turn()
	assert proxyless_engine.tick == 2
	proxyless_engine.branch = "branch1"
	assert proxyless_engine.tick == 2
	proxyless_engine.next_turn()
	assert proxyless_engine.tick == 2
	proxyless_engine.turn = 2
	proxyless_engine.branch = "trunk"
	assert proxyless_engine.tick == 2
	proxyless_engine.next_turn()
	assert proxyless_engine.tick == 2
