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

from lisien.exc import AmbiguousUserError


@pytest.fixture(scope="function")
def someplace(engy):
	yield engy.new_character("physical").new_place("someplace")


def test_contents(someplace):
	stuff = [someplace.new_thing(i) for i in range(10)]
	assert len(someplace.content) == len(stuff)
	assert len(someplace.contents()) == len(stuff)
	for i in range(10):
		assert i in someplace.content
		assert someplace.content[i] == stuff[i]
	for that in stuff:
		assert that in someplace.contents()
	for that in someplace.contents():
		assert that in stuff
	elsewhere = someplace.character.new_place("elsewhere")
	fust = elsewhere.new_thing(11)
	assert fust not in someplace.contents()
	assert 11 not in someplace.content
	with pytest.raises(KeyError):
		someplace.content[11]


def test_portal(someplace):
	assert not someplace.portal
	assert "there" not in someplace.portal
	there = someplace.character.new_place("there")
	assert not there.preportal
	assert "someplace" not in there.preportal
	someplace.character.new_portal("someplace", "there")
	assert someplace.portal
	assert "there" in someplace.portal
	assert "there" not in someplace.preportal
	assert there.preportal
	assert "someplace" in there.preportal
	assert "someplace" not in there.portal
	someplace.character.remove_edge("someplace", "there")
	assert "there" not in someplace.portal
	assert "someplace" not in there.preportal


def test_user(someplace):
	with pytest.raises(AmbiguousUserError):
		someplace.user.only
	someone = someplace.engine.new_character("someone")
	someone.add_unit(someplace)
	assert someplace.user.only is someone
	assert "someone" in someplace.user
	assert someplace.user["someone"] is someone
	noone = someplace.engine.new_character("noone")
	assert "noone" not in someplace.user
	noone.add_unit(someplace)
	with pytest.raises(AmbiguousUserError):
		someplace.user.only
	assert "noone" in someplace.user
	assert someplace.user["noone"] is noone


def test_rulebook(someplace):
	assert someplace.rulebook.name == ("physical", "someplace")
	someplace.rulebook = "imadeitup"
	assert someplace.rulebook.name == "imadeitup"


def test_deletion_after_keyframe(proxyless_engine):
	eng = proxyless_engine
	phys = eng.new_character("physical", data=nx.grid_2d_graph(8, 8))
	del phys.place[5, 5]
	assert (5, 5) not in phys.place
	assert (5, 5) not in list(phys.place)
	while eng.turn < 20:
		eng.next_turn()
	assert (5, 5) not in phys.place
	assert (5, 5) not in list(phys.place)


def test_clear(engy):
	phys = engy.new_character("physical")
	place = phys.new_place("here")
	place["a"] = 1
	place["b"] = 2
	assert place["a"] == 1
	assert place["b"] == 2
	place.clear()
	assert "a" not in place
	with pytest.raises(KeyError):
		print(place["b"])
