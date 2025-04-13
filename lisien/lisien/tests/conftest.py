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
from functools import partial
from queue import SimpleQueue

import pytest

from lisien import Engine
from lisien.proxy.handle import EngineHandle

from ..examples import college, kobold, sickle
from ..proxy import EngineProxy, WorkerLogger
from . import data
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
	params=["kobold", "college", "sickle"],
)
def handle_initialized(request, tmp_path, database):
	if request.param == "kobold":
		install = partial(
			kobold.inittest, shrubberies=20, kobold_sprint_chance=0.9
		)
		keyframe = {0: data.KOBOLD_KEYFRAME_0, 1: data.KOBOLD_KEYFRAME_1}
	elif request.param == "college":
		install = college.install
		keyframe = {0: data.COLLEGE_KEYFRAME_0, 1: data.COLLEGE_KEYFRAME_1}
	else:
		assert request.param == "sickle"
		install = sickle.install
		keyframe = {0: data.SICKLE_KEYFRAME_0, 1: data.SICKLE_KEYFRAME_1}
	with Engine(
		tmp_path,
		workers=0,
		random_seed=69105,
		connect_string=f"sqlite:///{tmp_path}/world.sqlite3"
		if database == "sqlite"
		else None,
	) as eng:
		install(eng)
	ret = EngineHandle(
		tmp_path,
		workers=0,
		connect_string=f"sqlite:///{tmp_path}/world.sqlite3"
		if database == "sqlite"
		else None,
	)
	ret.keyframe = keyframe
	yield ret
	ret.close()


def pytest_addoption(parser):
	parser.addoption("--no-parallel", action="store_true", default=False)
	parser.addoption("--no-sqlite", action="store_true", default=False)


@pytest.fixture(params=["proxy", "parallel", "serial"])
def execution(request):
	if request.config.getoption("no_parallel") and request.param == "parallel":
		raise pytest.skip("Skipping parallel execution.")
	return request.param


@pytest.fixture(params=["sqlite", "parquetdb"])
def database(request):
	if request.config.getoption("no_sqlite") and request.param == "sqlite":
		raise pytest.skip("Skipping SQLite.")
	return request.param


@pytest.fixture(
	scope="function",
)
def engy(tmp_path, execution, database):
	if execution == "proxy":
		logq = SimpleQueue()
		logger = WorkerLogger(logq, 0)
		eng = EngineProxy(
			None,
			None,
			logger,
			prefix=tmp_path,
			i=0,
			eternal={"language": "eng"},
			branches={},
		)
		(eng._branch, eng._turn, eng._tick, eng._initialized) = (
			"trunk",
			0,
			0,
			True,
		)
		eng._mutable_worker = True
		yield eng
		return
	with Engine(
		**make_test_engine_kwargs(tmp_path, execution, database)
	) as eng:
		yield eng


@pytest.fixture(params=["serial", "parallel"])
def proxyless_engine(tmp_path, request, database):
	if request.config.getoption("no_parallel") and request.param == "parallel":
		raise pytest.skip("Skipping parallel execution.")
	with Engine(
		tmp_path,
		random_seed=69105,
		enforce_end_of_time=False,
		workers=0 if request.param == "serial" else 2,
		connect_string=f"sqlite:///{tmp_path}/world.sqlite3"
		if database == "sqlite"
		else None,
	) as eng:
		yield eng


@pytest.fixture
def sqleng(tmp_path, request, execution):
	if request.config.getoption("no_parallel") and execution == "parallel":
		raise pytest.skip("Skipping parallel execution.")
	if execution == "proxy":
		logq = SimpleQueue()
		logger = WorkerLogger(logq, 0)
		eng = EngineProxy(
			None,
			None,
			logger,
			prefix=tmp_path,
			i=0,
			eternal={"language": "eng"},
			branches={},
		)
		(eng._branch, eng._turn, eng._tick, eng._initialized) = (
			"trunk",
			0,
			0,
			True,
		)
		eng._mutable_worker = True
		yield eng
	else:
		with Engine(
			tmp_path,
			random_seed=69105,
			enforce_end_of_time=False,
			workers=0 if execution == "serial" else 2,
			connect_string=f"sqlite:///{tmp_path}/world.sqlite3",
		) as eng:
			yield eng


@pytest.fixture(scope="function")
def serial_engine(tmp_path, database):
	with Engine(
		tmp_path,
		random_seed=69105,
		enforce_end_of_time=False,
		threaded_triggers=False,
		workers=0,
		connect_string=f"sqlite:///{tmp_path}/world.sqlite3"
		if database == "sqlite"
		else None,
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
