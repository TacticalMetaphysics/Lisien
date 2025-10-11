import os
from functools import partial
from itertools import product
from tempfile import TemporaryDirectory

from lisien.engine import Engine
from lisien.exporter import game_path_to_xml
from lisien.tests.data import DATA_DIR

RANDOM_SEED = 69105

for turns, sim in product([0, 1], ["kobold", "polygons", "wolfsheep"]):
	if sim == "kobold":
		from lisien.examples.kobold import inittest as install
	elif sim == "polygons":
		from lisien.examples.polygons import install
	elif sim == "wolfsheep":
		from lisien.examples.wolfsheep import install

		install = partial(install, seed=RANDOM_SEED)
	else:
		raise RuntimeError("Unknown sim", sim)
	with TemporaryDirectory() as tmp_path:
		prefix = os.path.join(tmp_path, "game")
		with Engine(
			prefix,
			workers=0,
			random_seed=RANDOM_SEED,
			connect_string=f"sqlite:///{prefix}/world.sqlite3",
			keyframe_on_close=False,
		) as eng:
			install(eng)
			for _ in range(turns):
				eng.next_turn()
		game_path_to_xml(
			prefix,
			os.path.join(DATA_DIR, f"{sim}_{turns}.xml"),
			name="test_export",
		)
