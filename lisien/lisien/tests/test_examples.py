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
)
from lisien.proxy.handle import EngineHandle
from lisien.proxy.manager import Sub
from lisien.tests.data import DATA_DIR
from lisien.types import GraphNodeValKeyframe, GraphValKeyframe, Keyframe

pytestmark = [pytest.mark.big]


@pytest.mark.slow
def test_college_nodb(serial_or_executor):
	kwargs = {
		"executor": serial_or_executor,
		"workers": 0 if serial_or_executor is None else 2,
	}
	with Engine(None, **kwargs, database=NullDatabaseConnector()) as eng:
		college.install(eng)
		for i in range(3):
			eng.next_turn()


@pytest.mark.slow
def test_college_premade(tmp_path, college10):
	"""The college example still works when loaded from disk"""
	# Caught a nasty loader bug once. Worth keeping.
	for i in range(3):
		college10.next_turn()


def test_kobold(engy):
	kobold.inittest(engy, shrubberies=20, kobold_sprint_chance=0.9)
	for i in range(10):
		engy.next_turn()


def test_polygons(engy):
	polygons.install(engy)
	for i in range(10):
		engy.next_turn()


def test_char_stat_startup(tmp_path, database_connector_part):
	with Engine(tmp_path, workers=0, database=database_connector_part) as eng:
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
		tmp_path, workers=0, database=reusing_database_connector_part()
	) as eng:
		assert "min_sameness" in eng.character["square"].stat
		assert "max_sameness" in eng.character["square"].stat
		assert "min_sameness" in eng.character["triangle"].stat
		assert "max_sameness" in eng.character["triangle"].stat


def test_sickle(sickle):
	for i in range(50):
		sickle.next_turn()


@pytest.mark.slow
def test_wolfsheep(
	tmp_path,
	serial_or_executor,
	database_connector_part,
):
	workers = 0 if serial_or_executor is None else 2
	with Engine.from_archive(
		DATA_DIR.joinpath("wolfsheep.lisien"),
		tmp_path,
		workers=workers,
		database=database_connector_part,
		executor=serial_or_executor,
	) as engine:
		sheep = engine.character["sheep"]
		physical = engine.character["physical"]
		initial_locations = [unit.location.name for unit in sheep.units()]
		initial_bare_places = list(physical.stat["bare_places"])
		assert initial_locations
		for _ in range(10):
			engine.next_turn()
		assert [
			unit.location.name for unit in sheep.units()
		] != initial_locations
		assert physical.stat["bare_places"] != initial_bare_places
		engine.turn = 5
		engine.branch = "lol"
		engine.universal["haha"] = "lol"
		for i in range(5):
			print(i + 5)
			engine.next_turn()
		engine.turn = 5
		engine.branch = "omg"
		final_locations = [unit.location.name for unit in sheep.units()]
		final_bare_places = list(physical.stat["bare_places"])
		assert initial_bare_places
	hand = EngineHandle(
		tmp_path,
		workers=workers,
		executor=serial_or_executor,
		database=database_connector_part,
	)
	try:
		hand.next_turn()
		assert [
			unit.location.name
			for unit in hand._real.character["sheep"].units()
		] != final_locations
		assert (
			hand._real.character["physical"].stat["bare_places"]
			!= final_bare_places
		)
	finally:
		hand.close()


@pytest.mark.slow
@pytest.mark.parallel
def test_pathfind(pathfind):
	locs = [
		thing.location.name
		for thing in sorted(
			pathfind.character["physical"].thing.values(), key=lambda t: t.name
		)
	]
	for i in range(10):
		pathfind.next_turn()
	assert locs != [
		thing.location.name
		for thing in sorted(
			pathfind.character["physical"].thing.values(), key=lambda t: t.name
		)
	]
