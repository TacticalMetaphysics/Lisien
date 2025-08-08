import random

import networkx as nx

from lisien import Engine


def install(
	eng: Engine,
	map_size=(25, 25),
	wolves=10,
	sheep=25,
	n_sickles_wolf=3,
	n_sickles_sheep=5,
	seed=None,
):
	if seed is not None:
		random.seed(seed)
	bare_places = []
	proto: nx.Graph = nx.grid_2d_graph(*map_size)
	for node_name, node in proto.nodes.items():
		node["bare"] = random.choice([True, False])
		node["_image_paths"] = [
			"atlas://rltiles/floor/"
			+ ("floor-normal" if node["bare"] else "floor-moss")
		]
		if node["bare"]:
			bare_places.append(node_name)
	phys = eng.new_character("physical", proto)
	phys.stat["bare_places"] = bare_places
	wolfs = eng.new_character(
		"wolf",
		child_name="pup",
		image_paths=["atlas://rltiles/dc-mon/war_dog"],
		n_creatures=wolves,
		mate_chance=1.0,
		malaria_chance=0.01,
	)
	sheeps = eng.new_character(
		"sheep",
		child_name="lamb",
		image_paths=["atlas://rltiles/dc-mon/sheep"],
		n_creatures=sheep,
		mate_chance=1.0,
		malaria_chance=0.01,
	)
	unoccupied = [
		(x, y) for x in range(map_size[0]) for y in range(map_size[0])
	]
	random.shuffle(unoccupied)
	for i in range(wolves):
		loc = phys.place[unoccupied.pop()]
		wolf = loc.new_thing(
			f"wolf{i}",
			_image_paths=["atlas://rltiles/dc-mon/war_dog"],
			sickle_a=i < n_sickles_wolf,
			sickle_b=False,
			male=eng.coin_flip(),
			last_mate_turn=-1,
		)
		wolfs.add_unit(wolf)
		print("wolf", i)
	for i in range(sheep):
		loc = phys.place[unoccupied.pop()]
		shep = loc.new_thing(
			f"sheep{i}",
			_image_paths=["atlas://rltiles/dc-mon/sheep"],
			sickle_a=i < n_sickles_sheep,
			sickle_b=False,
			male=eng.coin_flip(),
			last_mate_turn=-1,
		)
		sheeps.add_unit(shep)
		print("sheep", i)

	@eng.rule(always=True)
	def die_of_malaria_or_anemia(critter):
		ret = (
			"malaria"
			if not (critter["sickle_a"] or critter["sickle_b"])
			else "anemia"
		)
		critter.delete()
		return ret

	@die_of_malaria_or_anemia.prereq
	def malaria(critter):
		return (
			not (critter["sickle_a"] or critter["sickle_b"])
			and critter.engine.random()
			< critter.leader.only.stat["malaria_chance"]
		)

	@phys.rule(always=True)
	def grow(chara):
		bare_places = chara.stat["bare_places"]
		i = chara.engine.randrange(0, len(bare_places))
		there = chara.place[bare_places.pop(i)]
		there["bare"] = False
		there["_image_paths"] = ["atlas://rltiles/floor/floor-moss"]

	@sheeps.unit.rule
	def graze(shep):
		shep.location["bare"] = True
		shep.engine.character["physical"].stat["bare_places"].append(
			shep["location"]
		)
		shep.location["_image_paths"] = ["atlas://rltiles/floor/floor-normal"]

	@graze.trigger
	def grass_here(shep):
		return not shep.location["bare"]

	sheeps.unit.rulebook.append(die_of_malaria_or_anemia)

	@sheeps.unit.rule
	def mate(critter):
		"""If I share my location with another critter, attempt to mate"""
		engine = critter.engine
		species = critter.leader.only
		suitors = list(
			oc
			for oc in critter.location.contents()
			if oc["male"] != critter["male"]
		)
		assert len(suitors) > 0
		other_critter = critter.engine.choice(suitors)
		sickles = [
			critter["sickle_a"],
			critter["sickle_b"],
			other_critter["sickle_a"],
			other_critter["sickle_b"],
		]
		engine.shuffle(sickles)
		species.stat["n_creatures"] += 1
		name = str(critter.leader.only.name) + str(species.stat["n_creatures"])
		engine.character["physical"].add_thing(
			name,
			critter["location"],
			sickle_a=sickles.pop(),
			sickle_b=sickles.pop(),
			male=engine.coin_flip(),
			last_mate_turn=engine.turn,
		)
		species.add_unit("physical", name)
		critter["last_mate_turn"] = other_critter["last_mate_turn"] = (
			engine.turn
		)

	@mate.trigger
	def once_per_turn(critter):
		return critter["last_mate_turn"] < critter.engine.turn

	@mate.prereq
	def mate_present(critter):
		for oc in critter.location.contents():
			if oc["male"] != critter["male"]:
				return True
		return False

	@mate.prereq
	def in_the_mood(critter):
		return (
			critter.engine.random() < critter.leader.only.stat["mate_chance"]
		)

	@sheeps.unit.rule(always=True)
	def wander(shep):
		here = shep.location
		x, y = here.name
		physical = shep.engine.character["physical"]
		neighbors = list(
			filter(
				lambda b: b in physical.place,
				[(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)],
			)
		)
		for neighbor in neighbors:
			neighbor = physical.place[neighbor]
			if not neighbor["bare"] and not neighbor.contents():
				shep.location = neighbor
				return
		shep.location = shep.engine.choice(neighbors)

	wolfs.unit.rulebook.append(die_of_malaria_or_anemia)
	wolfs.unit.rulebook.append(mate)

	@wolfs.unit.rule(always=True)
	def pursue_sheep(wolff):
		import numpy as np

		# find the sheep that's nearest
		sheep_locs = [
			sheep["location"]
			for sheep in wolff.engine.character["sheep"].units()
		]
		my_loc = wolff["location"]
		assert my_loc is not None
		if len(sheep_locs) <= 2:
			nearest = sheep_locs[0]
		else:
			sheep_locs = np.array(sheep_locs)
			dists = np.linalg.norm((sheep_locs - my_loc), axis=1)
			nearest = tuple(sheep_locs[dists.argmin()])
		if my_loc == nearest:  # om nom nom
			sheepch = wolff.engine.character["sheep"]
			n_del = 0
			for the_sheep in wolff.location.contents():
				if the_sheep.leader.only is sheepch:
					the_sheep.delete()
					n_del += 1
			return
		# take a step closer
		if nearest[0] > my_loc[0]:
			wolff["location"] = (my_loc[0] + 1, my_loc[1])
		elif nearest[0] < my_loc[0]:
			wolff["location"] = (my_loc[0] - 1, my_loc[1])
		elif nearest[1] > my_loc[1]:
			wolff["location"] = (my_loc[0], my_loc[1] + 1)
		else:
			assert nearest[1] < my_loc[1]
			wolff["location"] = (my_loc[0], my_loc[1] - 1)

	@pursue_sheep.prereq
	def sheep_remains(wolff):
		for _ in wolff.engine.character["sheep"].units():
			return True
		return False


if __name__ == "__main__":
	import sys

	args = []
	kwargs = {"random_seed": 0}
	if len(sys.argv) == 4:
		kwargs["random_seed"] = sys.argv[-1]
		kwargs["connect_string"] = sys.argv[-2]
		args.append(sys.argv[-3])
	elif len(sys.argv) == 3:
		kwargs["random_seed"] = sys.argv[-1]
		args.append(sys.argv[-2])
	elif len(sys.argv) == 2:
		args.append(sys.argv[-1])
	else:
		kwargs["connect_string"] = "sqlite:///:memory:"
	with Engine(*args, **kwargs) as engn:
		install(engn, seed=0)
