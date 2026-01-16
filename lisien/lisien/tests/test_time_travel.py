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
from collections import defaultdict

import pytest

from lisien.exc import OutOfTimelineError


def test_build_keyframe_window(null_engine):
	null_engine._branch_parents = defaultdict(
		set, {"lol": {"trunk"}, "omg": {"trunk", "lol"}, "trunk": {None}}
	)
	null_engine._keyframes_loaded = {
		("lol", 5, 241),
		("lol", 9, 37),
		("lol", 10, 163),
		("lol", 10, 2187),
		("trunk", 0, 0),
		("trunk", 0, 2),
		("trunk", 0, 3),
		("trunk", 5, 240),
		("trunk", 8, 566),
		("trunk", 10, 578),
		("trunk", 10, 3139),
	}
	for b, r, t in null_engine._keyframes_loaded:
		if b in null_engine._keyframes_dict:
			if r in null_engine._keyframes_dict[b]:
				null_engine._keyframes_dict[b][r].add(t)
			else:
				null_engine._keyframes_dict[b][r] = {t}
		else:
			null_engine._keyframes_dict[b] = {r: {t}}
	null_engine._branches_d.update(
		{
			"lol": ("trunk", 5, 240, 10, 3877),
			"omg": ("lol", 5, 241, 5, 241),
			"trunk": (None, 0, 0, 10, 3284),
		}
	)
	assert null_engine._build_keyframe_window("lol", 5, 241) == (
		("lol", 5, 241),
		("lol", 9, 37),
	)
	assert null_engine._build_keyframe_window("omg", 5, 241) == (
		("lol", 5, 241),
		None,
	)
	assert null_engine._build_keyframe_window("lol", 5, 242) == (
		("lol", 5, 241),
		("lol", 9, 37),
	)
	assert null_engine._build_keyframe_window("omg", 5, 241) == (
		("lol", 5, 241),
		None,
	)
	assert null_engine._build_keyframe_window("omg", 6, 0) == (
		("lol", 5, 241),
		None,
	)
	assert null_engine._build_keyframe_window("lol", 5, 240) == (
		("trunk", 5, 240),
		("lol", 5, 241),
	)
	assert null_engine._build_keyframe_window("trunk", 0, 2) == (
		("trunk", 0, 2),
		("trunk", 0, 3),
	)


def test_plan_block(engine):
	char = engine.new_character("somebody")
	assert engine.turn == engine.tick == 0
	char.stat["evil"] = True
	assert engine.tick == 1
	with engine.plan():
		char.stat["cool"] = False
		assert engine.tick == 2
		engine.turn = 1
		char.stat["evil"] = False
		assert engine.turn == 1
		assert engine.tick == 1
	assert engine.turn == 0
	assert engine.tick == 1
	with pytest.raises(OutOfTimelineError):
		engine.turn = 1
	with pytest.raises(OutOfTimelineError):
		engine.tick = 2
	with engine.plan():
		assert engine.tick == 1
		assert "cool" not in char.stat
		engine.tick = 2
		assert char.stat["cool"] is False


def test_bookmark(engine):
	engine.bookmark("a")
	engine.next_turn()
	engine.bookmark("b")
	engine.branch = "branch"
	engine.bookmark("c")
	assert engine.time == ("branch", 1, 0)
	engine.bookmark("b")
	assert engine.time == ("trunk", 1, 0)
	engine.bookmark("a")
	assert engine.time == ("trunk", 0, 0)
