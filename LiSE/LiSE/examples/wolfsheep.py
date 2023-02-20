import random
import networkx as nx

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
