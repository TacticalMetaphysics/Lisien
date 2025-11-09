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
from unittest.mock import patch

import networkx as nx
import pytest

from lisien import Engine
from lisien.db import NullDatabaseConnector
from lisien.examples import (
	college,
	kobold,
	pathfind,
	polygons,
	sickle,
	wolfsheep,
)
from lisien.proxy.handle import EngineHandle
from lisien.proxy.manager import Sub
from lisien.types import GraphNodeValKeyframe, GraphValKeyframe, Keyframe

pytestmark = [pytest.mark.big]


@pytest.mark.slow
def test_college_nodb(serial_or_parallel):
	kwargs = {}
	if serial_or_parallel == "serial":
		kwargs["workers"] = 0
	else:
		kwargs["workers"] = 2
		kwargs["sub_mode"] = Sub(serial_or_parallel)
	with Engine(None, **kwargs, database=NullDatabaseConnector()) as eng:
		college.install(eng)
		for i in range(10):
			eng.next_turn()


@pytest.mark.slow
def test_college_premade(
	tmp_path, database_connector_part, serial_or_parallel
):
	"""The college example still works when loaded from disk"""
	# Caught a nasty loader bug once. Worth keeping.

	def validate_final_keyframe(kf: Keyframe):
		node_val: GraphNodeValKeyframe = kf["node_val"]
		phys_node_val = node_val["physical"]
		graph_val: GraphValKeyframe = kf["graph_val"]
		assert "student_body" in graph_val
		assert "units" in graph_val["student_body"]
		assert "physical" in graph_val["student_body"]["units"]
		for unit in graph_val["student_body"]["units"]["physical"]:
			assert unit in phys_node_val
			assert "location" in phys_node_val[unit]

	kwargs = {}
	if serial_or_parallel == "serial":
		kwargs["workers"] = 0
	else:
		kwargs["workers"] = 2
		kwargs["sub_mode"] = Sub(serial_or_parallel)

	with Engine(tmp_path, **kwargs, database=database_connector_part()) as eng:
		eng._validate_final_keyframe = validate_final_keyframe
		college.install(eng)
		for i in range(10):
			eng.next_turn()
	with (
		patch(
			"lisien.Engine._validate_initial_keyframe_load",
			staticmethod(validate_final_keyframe),
			create=True,
		),
		Engine(tmp_path, **kwargs, database=database_connector_part()) as eng,
	):
		for i in range(10):
			eng.next_turn()


def test_kobold(engy):
	kobold.inittest(engy, shrubberies=20, kobold_sprint_chance=0.9)
	for i in range(10):
		engy.next_turn()


def test_polygons(engy):
	polygons.install(engy)
	for i in range(10):
		engy.next_turn()


def test_char_stat_startup(tmp_path, database_connector_part):
	with Engine(
		tmp_path, workers=0, database=database_connector_part()
	) as eng:
		eng.new_character("physical", nx.hexagonal_lattice_graph(20, 20))
		tri = eng.new_character("triangle")
		sq = eng.new_character("square")

		sq.stat["min_sameness"] = 0.1
		assert "min_sameness" in sq.stat
		sq.stat["max_sameness"] = 0.9
		assert "max_sameness" in sq.stat
		tri.stat["min_sameness"] = 0.2
		assert "min_sameness" in tri.stat
		tri.stat["max_sameness"] = 0.8
		assert "max_sameness" in tri.stat

	with Engine(
		tmp_path, workers=0, database=database_connector_part()
	) as eng:
		assert "min_sameness" in eng.character["square"].stat
		assert "max_sameness" in eng.character["square"].stat
		assert "min_sameness" in eng.character["triangle"].stat
		assert "max_sameness" in eng.character["triangle"].stat


def test_sickle(engy):
	sickle.install(engy)
	for i in range(50):
		engy.next_turn()


@pytest.mark.slow
def test_wolfsheep(tmp_path, database_connector_part, serial_or_parallel):
	workers = 0 if serial_or_parallel == "serial" else 2
	with Engine(
		tmp_path,
		random_seed=69105,
		workers=workers,
		database=database_connector_part(),
	) as engy:
		wolfsheep.install(engy, seed=69105)
		for i in range(10):
			engy.next_turn()
		engy.turn = 5
		engy.branch = "lol"
		engy.universal["haha"] = "lol"
		for i in range(5):
			print(i + 5)
			engy.next_turn()
		engy.turn = 5
		engy.branch = "omg"
		sheep = engy.character["sheep"]
		initial_locations = [unit.location.name for unit in sheep.units()]
		assert initial_locations
		initial_bare_places = list(
			engy.character["physical"].stat["bare_places"]
		)
	hand = EngineHandle(
		tmp_path,
		random_seed=69105,
		workers=workers,
		database=database_connector_part(),
	)
	try:
		hand.next_turn()
		assert [
			unit.location.name
			for unit in hand._real.character["sheep"].units()
		] != initial_locations
		assert (
			hand._real.character["physical"].stat["bare_places"]
			!= initial_bare_places
		)
	finally:
		hand.close()


@pytest.mark.slow
@pytest.mark.parallel
def test_pathfind():
	with Engine(None, flush_interval=None, commit_interval=None) as eng:
		pathfind.install(eng, 69105)
		locs = [
			thing.location.name
			for thing in sorted(
				eng.character["physical"].thing.values(), key=lambda t: t.name
			)
		]
		for i in range(10):
			eng.next_turn()
		assert locs != [
			thing.location.name
			for thing in sorted(
				eng.character["physical"].thing.values(), key=lambda t: t.name
			)
		]
