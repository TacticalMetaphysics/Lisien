import filecmp
import os
from functools import partial

import pytest

from lisien import Engine
from lisien.export import game_path_to_xml
from lisien.tests.data import DATA_DIR


@pytest.fixture(params=["kobold", "polygons", "wolfsheep"])
def exported(tmp_path, random_seed, request):
	if request.param == "kobold":
		from lisien.examples.kobold import inittest as install

		path = os.path.join(DATA_DIR, "kobold.xml")
	elif request.param == "polygons":
		from lisien.examples.polygons import install

		path = os.path.join(DATA_DIR, "polygons.xml")
	elif request.param == "wolfsheep":
		from lisien.examples.wolfsheep import install

		install = partial(install, seed=random_seed)

		path = os.path.join(DATA_DIR, "wolfsheep.xml")
	else:
		raise ValueError("Unknown sim", request.param)
	with Engine(
		tmp_path,
		workers=0,
		random_seed=random_seed,
		connect_string=f"sqlite:///{tmp_path}/world.sqlite3",
	) as eng:
		install(eng)
		for _ in range(1):
			eng.next_turn()
	yield path


def test_export(tmp_path, exported):
	test_xml = os.path.join(tmp_path, "test.xml")
	game_path_to_xml(tmp_path, test_xml)

	assert filecmp.cmp(test_xml, exported)
