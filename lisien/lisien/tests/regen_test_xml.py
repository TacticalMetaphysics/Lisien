# This file is part of Lisien, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import os
from functools import partial
from itertools import product
from tempfile import TemporaryDirectory

from lisien.db import PythonDatabaseConnector
from lisien.engine import Engine
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
			database=PythonDatabaseConnector(),
			keyframe_on_close=False,
		) as eng:
			install(eng)
			for _ in range(turns):
				eng.next_turn()
			eng.to_xml(
				os.path.join(DATA_DIR, f"{sim}_{turns}.xml"),
				name="test_export",
			)
