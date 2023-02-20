import random
import networkx as nx
import numpy as np

from LiSE import Engine


def install(eng: Engine, map_size=(100, 100), wolves=10, sheep=10):
	proto: nx.Graph = nx.grid_2d_graph(*map_size)
	for node in proto.nodes.values():
		node['soil'] = random.choice([True, False])
		node['_image_paths'] = [
			'atlas://rltiles/floor/' +
			('floor-normal' if node['soil'] else 'floor-moss')
		]
	phys = eng.new_character('physical', proto)
	wolfs = eng.new_character('wolf')
	sheeps = eng.new_character('sheep')
	unoccupied = [(x, y) for x in range(map_size[0])
					for y in range(map_size[0])]
	random.shuffle(unoccupied)
	for i in range(wolves):
		loc = phys.place[unoccupied.pop()]
		wolf = loc.new_thing(
			f'wolf{i}', _image_paths=['atlas://rltiles/dc-mon/war_dog'])
		wolfs.add_unit(wolf)
		print('wolf', i)
	for i in range(sheep):
		loc = phys.place[unoccupied.pop()]
		shep = loc.new_thing(f'sheep{i}',
								_image_paths=['atlas://rltiles/dc-mon/sheep'])
		sheeps.add_unit(shep)
		print('sheep', i)

	@wolfs.unit.rule(always=True)
	def pursue_sheep(wolff):
		# find the sheep that's nearest
		sheep_locs = np.array([
			sheep['location']
			for sheep in wolff.engine.character['sheep'].units()
		])
		my_loc = wolff['location']
		dists = np.linalg.norm(sheep_locs - my_loc, axis=1)
		nearest = tuple(sheep_locs[dists.argmin()])
		if my_loc == nearest:  # om nom nom
			sheepch = wolff.engine.character['sheep']
			for the_sheep in wolff.contents():
				if the_sheep.user is sheepch:
					the_sheep.delete()
			return
		# take a step closer
		if nearest[0] > my_loc[0]:
			wolff['location'] = (my_loc[0] + 1, my_loc[1])
		elif nearest[0] < my_loc[0]:
			wolff['location'] = (my_loc[0] - 1, my_loc[1])
		elif nearest[1] > my_loc[1]:
			wolff['location'] = (my_loc[0], my_loc[1] + 1)
		else:
			assert nearest[1] < my_loc[1]
			wolff['location'] = (my_loc[0], my_loc[1] - 1)


if __name__ == '__main__':
	import sys

	args = []
	kwargs = {}
	if len(sys.argv) == 2:
		args.append(sys.argv[-1])
	else:
		kwargs['connect_string'] = 'sqlite:///:memory:'
	with Engine(*args, **kwargs) as engn:
		install(engn)
