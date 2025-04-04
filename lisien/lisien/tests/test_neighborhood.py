from itertools import product

import networkx as nx
import pytest

from lisien.character import grid_2d_8graph


@pytest.fixture
def three3(proxyless_engine):
	return proxyless_engine.new_character("3x3", grid_2d_8graph(3, 3))


def test_sim_start(three3):
	@three3.place.rule(neighborhood=1, always=True)
	def did_it_run(place):
		if "it_ran" in place:
			place.character.stat["it_ran"] = (
				place.character.stat.get("it_ran", 0) + 1
			)
		else:
			place["it_ran"] = True

	eng = three3.engine
	eng.next_turn()
	assert three3.place[1, 1]["it_ran"]
	assert "it_ran" not in three3.stat
	eng.next_turn()
	assert three3.place[1, 1]["it_ran"]
	assert three3.stat["it_ran"] == 9
	eng.next_turn()
	assert three3.stat["it_ran"] == 9  # still!


@pytest.mark.parametrize(
	("branched", "rulebook"), product(*[(True, False)] * 2)
)
def test_rule_neighborhood(serial_engine, branched, rulebook):
	"""Test a rule applied to all nodes of a character with a neighborhood"""
	char = serial_engine.new_character("char", nx.grid_2d_graph(5, 5))

	@serial_engine.rule
	def it_ran(node):
		node["it_ran"] = True

	@it_ran.trigger
	def should_it_run(node):
		node["trigger_evaluated"] = True
		for neighbor in node.neighbors():
			if neighbor.get("should_run"):
				return True
		return False

	it_ran.neighborhood = 1
	assert it_ran.neighborhood == 1
	if rulebook:
		serial_engine.rulebook["it_ran"] = [it_ran]

	serial_engine.next_turn()
	if rulebook:
		char.place.rulebook = "it_ran"
	else:
		char.place.rule(it_ran)
	char.place[3, 3]["should_run"] = True
	assert it_ran.neighborhood == 1
	if branched:
		serial_engine.branch = "eeeee"
	serial_engine.next_turn()

	for nabor in [(2, 3), (4, 3), (3, 4), (3, 2)]:
		assert char.place[nabor]["it_ran"]

	for outer in [(1, 1), (4, 4)]:
		assert "trigger_evaluated" not in char.place[outer]
