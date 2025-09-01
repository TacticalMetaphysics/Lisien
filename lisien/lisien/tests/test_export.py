import os

import pytest

from lisien import Engine
from lisien.examples import kobold, polygons, wolfsheep
from lisien.export import game_path_to_etree


@pytest.fixture(params=["kobold", "polygons", "wolfsheep"], autouse=True)
def sim(tmp_path, random_seed, request):
	if request.param == "kobold":
		install = kobold.inittest
	elif request.param == "polygons":
		install = polygons.install
	elif request.param == "wolfsheep":
		install = wolfsheep.install
	else:
		raise ValueError("Unknown sim", request.param)
	with Engine(
		tmp_path,
		workers=0,
		random_seed=random_seed,
		connect_string=f"sqlite:///{tmp_path}/world.sqlite3",
	) as eng:
		install(eng)


def test_export(tmp_path):
	from io import BytesIO

	f = BytesIO()
	game_path_to_etree(tmp_path).write(f)
	print(f.getvalue().decode())
