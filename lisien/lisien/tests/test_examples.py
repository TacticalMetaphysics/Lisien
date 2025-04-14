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

pytestmark = [pytest.mark.big]


def test_college(engy):
	college.install(engy)
	for i in range(10):
		engy.next_turn()


def test_kobold(engy):
	kobold.inittest(engy, shrubberies=20, kobold_sprint_chance=0.9)
	for i in range(10):
		engy.next_turn()


def test_polygons(engy):
	polygons.install(engy)
	for i in range(10):
		engy.next_turn()


def test_char_stat_startup(tmp_path):
	with Engine(tmp_path) as eng:
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

	with Engine(tmp_path) as eng:
		assert "min_sameness" in eng.character["square"].stat
		assert "max_sameness" in eng.character["square"].stat
		assert "min_sameness" in eng.character["triangle"].stat
		assert "max_sameness" in eng.character["triangle"].stat


def test_sickle(engy):
	sickle.install(engy)
	for i in range(50):
		engy.next_turn()


def test_wolfsheep(tmp_path):
	with Engine(tmp_path, random_seed=69105, workers=0) as engy:
		wolfsheep.install(engy, seed=69105)
		for i in range(10):
			engy.next_turn()
		engy.turn = 5
		engy.branch = "lol"
		engy.universal["haha"] = "lol"
		for i in range(5):
			engy.next_turn()
		engy.turn = 5
		engy.branch = "omg"
		sheep = engy.character["sheep"]
		sheep.rule(engy.action.breed, always=True)
		initial_locations = [unit.location.name for unit in sheep.units()]
		initial_bare_places = list(
			engy.character["physical"].stat["bare_places"]
		)
	hand = EngineHandle(tmp_path, random_seed=69105, workers=0)
	hand.next_turn()
	assert [
		unit.location.name for unit in hand._real.character["sheep"].units()
	] != initial_locations
	assert (
		hand._real.character["physical"].stat["bare_places"]
		!= initial_bare_places
	)
	hand.close()


def test_pathfind(tmp_path):
	with Engine(tmp_path) as eng:
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
