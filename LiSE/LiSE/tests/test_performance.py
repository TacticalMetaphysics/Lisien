from time import monotonic

import networkx as nx

from LiSE import Engine
from LiSE.proxy import EngineProcessManager


def test_follow_path(tmp_path):
	big_grid = nx.grid_2d_graph(100, 100)
	big_grid.add_node("them", location=(0, 0))
	straightly = nx.shortest_path(big_grid, (0, 0), (99, 99))
	with Engine(tmp_path) as eng:
		eng.add_character("grid", big_grid)
	with EngineProcessManager(tmp_path) as prox:
		them = prox.character["grid"].thing["them"]
		start = monotonic()
		them.follow_path(straightly)
		elapsed = monotonic() - start
		assert (
			elapsed < 0.15
		), f"Took too long to follow a path of length {len(straightly)}: {elapsed:.2} seconds"
