from unittest.mock import patch

import networkx as nx
import pytest

from lisien import Engine
from lisien.examples import (
	college,
	kobold,
	pathfind,
	polygons,
	sickle,
	wolfsheep,
)
from lisien.proxy.handle import EngineHandle
from lisien.types import GraphNodeValKeyframe, GraphValKeyframe, Keyframe

pytestmark = [pytest.mark.big]


@pytest.mark.slow
def test_college_nodb(serial_or_parallel):
	with Engine(
		None, workers=0 if serial_or_parallel == "serial" else 2
	) as eng:
		college.install(eng)
		for i in range(10):
			eng.next_turn()


@pytest.mark.slow
def test_college_premade(tmp_path, non_null_database, serial_or_parallel):
	"""The college example still works when loaded from disk"""
	# Caught a nasty loader bug once. Worth keeping.
	connect_string = None
	if non_null_database == "sqlite":
		connect_string = f"sqlite:///{tmp_path}/world.db"

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

	with Engine(
		tmp_path,
		connect_string=connect_string,
		workers=0 if serial_or_parallel == "serial" else 2,
	) as eng:
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
		Engine(
			tmp_path,
			connect_string=connect_string,
			workers=0 if serial_or_parallel == "serial" else 2,
		) as eng,
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


def test_char_stat_startup(tmp_path, non_null_database):
	connect_string = None
	if non_null_database == "sqlite":
		connect_string = f"sqlite:///{tmp_path}/world.sqlite3"
	with Engine(tmp_path, workers=0, connect_string=connect_string) as eng:
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

	with Engine(tmp_path, workers=0, connect_string=connect_string) as eng:
		assert "min_sameness" in eng.character["square"].stat
		assert "max_sameness" in eng.character["square"].stat
		assert "min_sameness" in eng.character["triangle"].stat
		assert "max_sameness" in eng.character["triangle"].stat


def test_sickle(engy):
	sickle.install(engy)
	for i in range(50):
		engy.next_turn()


@pytest.mark.slow
def test_wolfsheep(tmp_path, non_null_database, serial_or_parallel):
	connect_string = None
	if non_null_database == "sqlite":
		connect_string = f"sqlite:///{tmp_path}/world.sqlite3"
	workers = 0 if serial_or_parallel == "serial" else 2
	with Engine(
		tmp_path,
		random_seed=69105,
		workers=workers,
		connect_string=connect_string,
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
		connect_string=connect_string,
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
