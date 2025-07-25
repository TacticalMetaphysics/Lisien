import random

import networkx as nx

import lisien.db


def install(eng, seed=None):
	if seed is not None:
		random.seed(seed)
	grid: nx.Graph = nx.grid_2d_graph(100, 100)

	for node in list(grid):
		if random.random() < 0.1:
			grid.remove_node(node)
		elif random.random() < 0.01:
			grid.add_node(f"{node}_inhabitant", location=node)

	phys = eng.new_character("physical", grid)

	@eng.function
	def find_path_somewhere(node):
		from math import sqrt

		from networkx.algorithms import astar_path

		x, y = node.location.name
		destx = 100 - int(x)
		desty = 100 - int(y)
		while (destx, desty) not in node.character.place:
			if destx < 99:
				destx += 1
			elif desty < 99:
				destx = 0
				desty += 1
			else:
				destx = desty = 0
		ret = astar_path(
			node.character,
			node.location.name,
			(destx, desty),
			lambda a, b: sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2),
		)
		node.engine.debug(
			f"{node.name}'s shortest path to {destx, desty} is {ret}"
		)
		return ret

	@phys.rule(big=True)
	def go_places(char):
		from time import monotonic

		from networkx.exception import NetworkXNoPath

		def log_as_completed(fut):
			try:
				char.engine.debug(
					f"Got path for {fut.thing.name}: {fut.result()}"
				)
			except NetworkXNoPath:
				char.engine.debug(f"No path for {fut.thing.name}")

		futs = []
		start_all = monotonic()
		for thing in char.thing.values():
			fut = char.engine.submit(
				char.engine.function.find_path_somewhere, thing
			)
			fut.thing = thing
			fut.add_done_callback(log_as_completed)
			futs.append(fut)
		for fut in futs:
			try:
				result = fut.result()
				thing = fut.thing
				start = monotonic()
				thing.follow_path(result, check=False)
				char.engine.debug(
					f"followed path for thing {thing.name} in {monotonic() - start:.2} seconds"
				)
			except NetworkXNoPath:
				char.engine.debug(f"got no path for thing {fut.thing.name}")
				continue
		char.engine.debug(
			f"followed all paths in {monotonic() - start_all:.2} seconds"
		)

	@go_places.trigger
	def turn_one_only(char):
		return char.engine.turn == 1


if __name__ == "__main__":
	from tempfile import mkdtemp

	from lisien import Engine

	td = mkdtemp()
	with Engine(td) as eng:
		install(eng)
		for _ in range(10):
			eng.next_turn()
	print("View the sim with:")
	print("python -m elide " + td)
