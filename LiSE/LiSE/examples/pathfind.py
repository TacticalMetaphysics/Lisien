import random

import networkx as nx


def install(eng, seed=None):
	if seed is not None:
		random.seed(seed)
	grid: nx.Graph = nx.grid_2d_graph(100, 100)

	for node in list(grid):
		if random.choice((True, False)):
			grid.remove_node(node)
		elif random.random() < 0.1:
			grid.add_node(f"{node}_inhabitant", location=node)

	phys = eng.new_character("physical", grid)

	@eng.function
	def find_path_somewhere(node):
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
		ret = node.shortest_path(node.character.place[destx, desty])
		print(f"{node.name}'s shortest path to {destx, desty} is {ret}")
		return ret

	@phys.rule
	def go_places(char):
		from networkx.exception import NetworkXNoPath

		futs = []
		with char.engine.pool as pool:
			for thing in char.thing.values():
				fut = pool.submit(
					char.engine.function.find_path_somewhere, thing
				)
				fut.thing = thing
				futs.append(fut)
		for fut in futs:
			try:
				result = fut.result()
				thing = fut.thing
				thing.follow_path(result)
			except NetworkXNoPath:
				continue

	@go_places.trigger
	def turn_one_only(char):
		return char.engine.turn == 1
