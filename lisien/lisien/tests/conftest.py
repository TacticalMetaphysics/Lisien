# This file is part of Elide, frontend to Lisien, a framework for life simulation games.
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
import shutil
from tempfile import TemporaryDirectory
from itertools import product

import pytest

from lisien import Engine

from ..examples import kobold


@pytest.fixture(scope="function")
def handle(tmp_path):
	from lisien.handle import EngineHandle

	hand = EngineHandle(
		tmp_path,
		connect_string="sqlite:///:memory:",
		random_seed=69105,
		workers=0,
	)
	yield hand
	hand.close()


@pytest.fixture(
	scope="function",
	params=[
		lambda eng: kobold.inittest(
			eng, shrubberies=20, kobold_sprint_chance=0.9
		),
		# college.install,
		# sickle.install
	],
)
def handle_initialized(request, handle):
	with handle._real.advancing():
		request.param(handle._real)
	yield handle


def pytest_addoption(parser):
	parser.addoption("--serial", action="store_true", default=False)


@pytest.fixture(
	scope="function",
	params=list(product(["parallel", "serial"], ["sqlite", "parquetdb"])),
)
def engy(tmp_path, request):
	execution, database = request.param
	if request.config.getoption("serial") and execution == "parallel":
		raise pytest.skip("Skipping parallel execution.")
	kwargs = {"random_seed": 69105, "enforce_end_of_time": False}
	if execution == "parallel":
		kwargs["threaded_triggers"] = True
		kwargs["workers"] = 2
	else:
		kwargs["threaded_triggers"] = False
		kwargs["workers"] = 0
	if database == "sqlite":
		kwargs["connect_string"] = f"sqlite:///{tmp_path}/world.sqlite3"
	with Engine(tmp_path, **kwargs) as eng:
		yield eng


@pytest.fixture(scope="function")
def sqleng(tmp_path):
	with Engine(
		tmp_path,
		random_seed=69105,
		enforce_end_of_time=False,
		connect_string="sqlite:///world.sqlite3",
	) as eng:
		yield eng


@pytest.fixture(scope="function")
def serial_engine(tmp_path):
	with Engine(
		tmp_path,
		random_seed=69105,
		enforce_end_of_time=False,
		threaded_triggers=False,
		workers=0,
	) as eng:
		yield eng


@pytest.fixture(scope="module")
def college24_premade():
	with TemporaryDirectory() as tmp_path:
		shutil.unpack_archive(
			os.path.join(
				os.path.abspath(os.path.dirname(__file__)),
				"college24_premade.tar.xz",
			),
			tmp_path,
		)
		with Engine(tmp_path, workers=0) as eng:
			yield eng
