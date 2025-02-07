# This file is part of Elide, frontend to Lisien, a framework for life simulation games.
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
from unittest.mock import patch, call

import networkx as nx
import pytest

from lisien.engine import Engine
from lisien.examples.kobold import inittest


def test_keyframe_load_init(tmp_path):
	"""Can load a keyframe at start of branch, including locations"""
	eng = Engine(tmp_path, workers=0)
	inittest(eng)
	eng.branch = "new"
	eng.snap_keyframe()
	eng.close()
	eng = Engine(tmp_path)
	assert "kobold" in eng.character["physical"].thing
	assert (0, 0) in eng.character["physical"].place
	assert (0, 1) in eng.character["physical"].portal[0, 0]
	eng.close()


def test_multi_keyframe(tmp_path):
	eng = Engine(
		tmp_path, enforce_end_of_time=False, keyframe_on_close=False, workers=0
	)
	inittest(eng)
	eng.snap_keyframe()
	tick0 = eng.tick
	eng.turn = 1
	del eng.character["physical"].place[3, 3]
	eng.snap_keyframe()
	tick1 = eng.tick
	assert ("physical",) in eng._nodes_cache.keyframe
	assert "trunk" in eng._nodes_cache.keyframe["physical",]
	assert 1 in eng._nodes_cache.keyframe["physical",]["trunk"]
	assert tick1 in eng._nodes_cache.keyframe["physical",]["trunk"][1]
	assert (1, 1) in eng._nodes_cache.keyframe["physical",]["trunk"][1][tick1]
	assert (3, 3) not in eng._nodes_cache.keyframe["physical",]["trunk"][1][
		tick1
	]
	eng.close()
	eng = Engine(tmp_path, keyframe_on_close=False, workers=0)
	eng._load_at("trunk", 0, tick0)
	assert eng._time_is_loaded("trunk", 0, tick0)
	assert eng._time_is_loaded("trunk", 0, tick0 + 1)
	assert eng._time_is_loaded("trunk", 1, tick1 - 1)
	assert eng._time_is_loaded("trunk", 1, tick1)
	eng.close()


def test_keyframe_load_unload(tmp_path):
	"""Make sure all of the caches can load and unload before and after kfs"""
	with Engine(
		tmp_path, enforce_end_of_time=False, keyframe_on_close=False, workers=0
	) as eng:
		eng.snap_keyframe()
		eng.turn = 1
		inittest(eng)
		eng.snap_keyframe()
		eng.turn = 2
		eng.universal["hi"] = "hello"
		now = eng._btt()
	with Engine(
		tmp_path, enforce_end_of_time=False, keyframe_on_close=False, workers=0
	) as eng:
		assert eng._time_is_loaded(*now)
		assert not eng._time_is_loaded("trunk", 0)
		eng.turn = 1
		eng.tick = 0
		assert eng._time_is_loaded("trunk", 1)
		assert eng._time_is_loaded("trunk", 1, 0)
		assert not eng._time_is_loaded("trunk", 0)
		assert eng._time_is_loaded(*now)
		eng.unload()
		assert eng._time_is_loaded("trunk", 1, 0)
		assert not eng._time_is_loaded(*now)
		eng.turn = 2
		eng.branch = "haha"
		eng.snap_keyframe()
		eng.unload()
		assert not eng._time_is_loaded("trunk")


@pytest.fixture
def some_state(tmp_path):
	with Engine(tmp_path, workers=0, random_seed=0) as eng:
		initial_state = nx.DiGraph(
			{
				0: {1: {"omg": "lol"}},
				1: {0: {"omg": "blasphemy"}},
				2: {},
				3: {},
				"it": {},
			}
		)
		initial_state.nodes()[2]["hi"] = "hello"
		initial_state.nodes()["it"]["location"] = 0
		initial_state.graph["wat"] = "nope"
		phys = eng.new_character("physical", initial_state)
		eng.add_character("pointless")
		kf0 = eng.snap_keyframe()
		del kf0["universal"]["rando_state"]
		eng.branch = "b"
		kf1 = eng.snap_keyframe()
		del kf1["universal"]["rando_state"]
		assert kf0 == kf1
		del phys.portal[1][0]
		port = phys.new_portal(0, 2)
		port["hi"] = "bye"
		phys.place[1]["wtf"] = "bbq"
		phys.thing["it"].location = phys.place[1]
		del phys.place[3]
		eng.add_character("pointed")
		del eng.character["pointless"]
		assert "pointless" not in eng.character, "Failed to delete character"
		phys.portal[0][1]["meaning"] = 42
		del phys.portal[0][1]["omg"]
		eng.branch = "trunk"
	return tmp_path


def test_load_branch_to_end(some_state):
	with Engine(some_state, workers=0, random_seed=0) as eng:
		assert eng.turn == 0
		phys = eng.character["physical"]
		assert 3 in phys.place
		assert phys.portal[1][0]["omg"] == "blasphemy"
		eng.branch = "b"
		assert 3 not in phys.place
		assert 0 not in phys.portal[1]
		assert 2 in phys.portal[0]
		assert phys.portal[0][2]["hi"] == "bye"
		assert phys.place[1]["wtf"] == "bbq"
		assert phys.thing["it"].location == phys.place[1]
		assert "pointless" not in eng.character, "Loaded deleted character"
		assert "pointed" in eng.character
		assert phys.portal[0][1]["meaning"] == 42
		assert "omg" not in phys.portal[0][1]
