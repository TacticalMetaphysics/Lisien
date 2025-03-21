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

import pytest

from lisien import Engine
from lisien.proxy.handle import EngineHandle

from ..examples import kobold
from .util import make_test_engine_kwargs


@pytest.fixture(scope="function")
def handle(tmp_path):
	hand = EngineHandle(
		tmp_path,
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
def handle_initialized(request, tmp_path, database):
	with Engine(
		tmp_path,
		workers=0,
		random_seed=69105,
		connect_string=f"sqlite:///{tmp_path}/world.sqlite3"
		if database == "sqlite"
		else None,
	) as eng:
		request.param(eng)
	return EngineHandle(
		tmp_path,
		workers=0,
		connect_string=f"sqlite:///{tmp_path}/world.sqlite3"
		if database == "sqlite"
		else None,
	)


def pytest_addoption(parser):
	parser.addoption("--serial", action="store_true", default=False)


@pytest.fixture(params=["parallel", "serial"])
def execution(request):
	if request.config.getoption("serial") and request.param == "parallel":
		raise pytest.skip("Skipping parallel execution.")
	return request.param


@pytest.fixture(params=["sqlite", "parquetdb"])
def database(request):
	return request.param


@pytest.fixture(
	scope="function",
)
def engy(tmp_path, execution, database):
	with Engine(
		**make_test_engine_kwargs(tmp_path, execution, database)
	) as eng:
		yield eng


@pytest.fixture(scope="function", params=["serial", "parallel"])
def sqleng(tmp_path, request):
	execution = request.param
	if request.config.getoption("serial") and execution == "parallel":
		raise pytest.skip("Skipping parallel execution.")
	with Engine(
		tmp_path,
		random_seed=69105,
		enforce_end_of_time=False,
		workers=0 if execution == "serial" else 2,
		connect_string=f"sqlite:///{tmp_path}/world.sqlite3",
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


@pytest.fixture(scope="function")
def college24_premade(tmp_path):
	shutil.unpack_archive(
		os.path.join(
			os.path.abspath(os.path.dirname(__file__)),
			"college24_premade.tar.xz",
		),
		tmp_path,
	)
	with Engine(
		tmp_path,
		workers=0,
		connect_string=f"sqlite:///{tmp_path}/world.sqlite3",
	) as eng:
		yield eng
