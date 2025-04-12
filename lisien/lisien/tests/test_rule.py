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
"""Test the API of the Rule objects and mappings"""

import networkx as nx
import pytest


def something_dot_rule_test(something, engy):
	"""Utility function to test some rule-follower"""
	eng = engy

	@something.rule
	def somerule():
		pass

	@somerule.trigger
	def otherthing():
		pass

	@somerule.prereq
	def anotherthing():
		pass

	assert "somerule" in eng.rule
	assert somerule.triggers
	assert eng.trigger.otherthing in somerule.triggers
	assert somerule.prereqs
	assert eng.prereq.anotherthing in somerule.prereqs

	@somerule.trigger
	def thirdthing():
		pass

	assert somerule.triggers.index(eng.trigger.otherthing) == 0
	assert somerule.triggers.index(eng.trigger.thirdthing) == 1

	@somerule.prereq
	def fourththing():
		pass

	assert somerule.prereqs.index(eng.prereq.anotherthing) == 0
	assert somerule.prereqs.index(eng.prereq.fourththing) == 1

	@somerule.action
	def fifththing():
		pass

	assert somerule.actions.index(eng.action.somerule) == 0
	assert somerule.actions.index(eng.action.fifththing) == 1

	del somerule.triggers[0]
	del somerule.prereqs[0]
	del somerule.actions[0]
	assert somerule.triggers[0] == eng.trigger.thirdthing
	assert somerule.prereqs[0] == eng.prereq.fourththing
	assert somerule.actions[0] == eng.action.fifththing

	somerule.triggers.append("otherthing")
	somerule.prereqs.append("anotherthing")
	somerule.actions.append("somerule")
	assert len(somerule.triggers) == 2
	assert len(somerule.prereqs) == 2
	assert len(somerule.actions) == 2
	assert somerule.triggers[1] == eng.trigger.otherthing
	assert somerule.prereqs[1] == eng.prereq.anotherthing
	assert somerule.actions[1] == eng.action.somerule

	eng.turn = 1
	assert len(somerule.triggers) == 2
	assert len(somerule.prereqs) == 2
	somerule.triggers.remove("otherthing")
	del somerule.prereqs[1]
	somerule.actions[somerule.actions.index("somerule")] = "fifththing"
	assert len(somerule.triggers) == 1
	assert len(somerule.prereqs) == 1
	assert somerule.actions[1] == eng.action.fifththing

	eng.turn = 0

	assert len(somerule.triggers) == 2
	assert len(somerule.prereqs) == 2
	assert somerule.triggers[1] == eng.trigger.otherthing
	assert somerule.prereqs[1] == eng.prereq.anotherthing
	assert somerule.actions[1] == eng.action.somerule

	return somerule


def test_engine_dot_rule(proxyless_engine):
	"""The global rule mapping can be used to make and change rules"""
	something_dot_rule_test(proxyless_engine, proxyless_engine)


def test_character_dot_rule(proxyless_engine):
	"""You can make and change rules on characters"""
	character = proxyless_engine.new_character("physical")
	rule = something_dot_rule_test(character, proxyless_engine)
	assert character.rulebook[0] == rule


def test_character_dot_thing_dot_rule(proxyless_engine):
	"""You can make and change rules on the thing mapping of a character"""
	character = proxyless_engine.new_character("physical")
	rule = something_dot_rule_test(character.thing, proxyless_engine)
	assert character.thing.rulebook[0] == rule


def test_character_dot_place_dot_rule(proxyless_engine):
	"""You can make and change rules on the place mapping of a character"""
	character = proxyless_engine.new_character("physical")
	rule = something_dot_rule_test(character.place, proxyless_engine)
	assert character.place.rulebook[0] == rule


def test_character_dot_portal_dot_rule(proxyless_engine):
	"""You can make and change rules on the portal mapping of a character"""
	character = proxyless_engine.new_character("physical")
	rule = something_dot_rule_test(character.portal, proxyless_engine)
	assert character.portal.rulebook[0] == rule


def test_node_dot_rule(proxyless_engine):
	"""You can make and change rules on a node"""
	here = proxyless_engine.new_character("physical").new_place(1)
	rule = something_dot_rule_test(here, proxyless_engine)
	assert here.rulebook[0] == rule


def test_portal_dot_rule(proxyless_engine):
	"""You can make and change rules on a portal"""
	character = proxyless_engine.new_character("physical")
	character.new_place(0)
	character.new_place(1)
	port = character.new_portal(0, 1)
	rule = something_dot_rule_test(port, proxyless_engine)
	assert port.rulebook[0] == rule


def test_rule_priority(proxyless_engine):
	"""Rules run in the order given in their priorities"""
	firstchar = proxyless_engine.new_character("first")
	secondchar = proxyless_engine.new_character("second")
	proxyless_engine.universal["list"] = []

	@firstchar.rule(always=True)
	def first(ch):
		ch.engine.universal["list"].append("first")

	@secondchar.rule(always=True)
	def second(ch):
		ch.engine.universal["list"].append("second")

	firstchar.rule.priority = 1
	secondchar.rule.priority = 2

	proxyless_engine.next_turn()

	assert proxyless_engine.universal["list"] == ["first", "second"]

	firstchar.rule.priority = 3

	proxyless_engine.next_turn()

	assert proxyless_engine.universal["list"] == [
		"first",
		"second",
		"second",
		"first",
	]
