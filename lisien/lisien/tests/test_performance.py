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
import sys
from concurrent.futures import wait
from time import monotonic

import pytest

from lisien import Engine
from lisien.tests.data import DATA_DIR
from lisien.util import timer


@pytest.mark.parquetdb
def test_follow_path(tmp_path):
	with (
		Engine.from_archive(
			DATA_DIR.joinpath("big_grid.lisien"), tmp_path, workers=0
		) as eng,
	):
		grid = eng.character["grid"]
		them = grid.thing["them"]
		straightly = grid.stat["straightly"]
		start = monotonic()
		them.follow_path(straightly)
		elapsed = monotonic() - start
		assert elapsed < 20, (
			f"Took too long to follow a path of length {len(straightly)}: {elapsed:.2} seconds"
		)


@pytest.mark.parametrize(
	"kind",
	[
		pytest.param("process", marks=pytest.mark.subprocess),
		pytest.param(
			"interpreter",
			marks=[
				pytest.mark.subinterpreter,
				pytest.mark.skipif(
					lambda: sys.version_info.minor < 14,
					reason="Interpreters are only available in Python 3.14+",
				),
			],
		),
	],
)
def test_pathfind(tmp_path, kind):
	with Engine.from_archive(
		os.path.join(DATA_DIR, "big_grid.lisien"), tmp_path, sub_mode=kind
	) as eng:
		grid = eng.character["grid"]
		them = grid.thing["them"]
		places = list(grid.place.keys())
		places.remove(them.location.name)
		eng.shuffle(places)
		for i in range(20):
			place_n = places.pop()
			place = grid.place[place_n]
			while place.content:
				print(
					f"Not removing {place_n} because there's {', '.join(map(str, place.content.keys()))} in it"
				)
				place_n = places.pop()
				place = grid.place[place_n]
			print(f"Removing {place_n}")
			del grid.place[place_n]
		cpus = os.cpu_count()
		with timer(f"Found {cpus} paths") as timed:
			futs = []
			for _ in range(cpus):
				futs.append(
					eng.submit(eng.find_path, places.pop(), places.pop())
				)
			wait(futs)
		assert timed.get() < 10, f"Took too long to find {cpus} paths"


if __name__ == "__main__":
	from tempfile import TemporaryDirectory

	with TemporaryDirectory() as tmp_path:
		test_follow_path(tmp_path)
