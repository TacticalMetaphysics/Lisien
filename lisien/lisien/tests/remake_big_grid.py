import os

import networkx as nx

from lisien.engine import Engine
from lisien.tests.data import DATA_DIR


def main():
	big_grid = nx.grid_2d_graph(100, 100)
	big_grid.add_node("them", location=(0, 0))
	big_grid.graph["straightly"] = nx.shortest_path(big_grid, (0, 0), (99, 99))
	with Engine(None, workers=0) as eng:

		@eng.method
		def find_path(eng, node_a, node_b):
			from networkx import shortest_path

			return shortest_path(eng.character["grid"], node_a, node_b)

		eng.add_character("grid", big_grid)
		eng.export("grid", os.path.join(DATA_DIR, "big_grid.lisien"))


if __name__ == "__main__":
	main()
