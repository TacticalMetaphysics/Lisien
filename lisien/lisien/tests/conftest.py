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
from logging import getLogger

import pytest

from lisien import Engine
from lisien.proxy.handle import EngineHandle

from ..examples import college, kobold, sickle
from ..proxy import EngineProcessManager, EngineProxy
from . import data
from .util import make_test_engine_facade, make_test_engine_kwargs


@pytest.fixture(scope="function")
def handle(tmp_path):
	hand = EngineHandle(
		tmp_path,
		random_seed=69105,
		workers=0,
	)
	yield hand
	hand.close()


@pytest.fixture
def engine_facade():
	return make_test_engine_facade()


@pytest.fixture(scope="session")
def random_seed():
	yield 69105


@pytest.fixture(
	scope="function",
	params=[
		"kobold",
		pytest.param("college", marks=pytest.mark.slow),
		"sickle",
	],
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
	if database == "nodb":
		ret = EngineHandle(None, workers=0, random_seed=69105)
		install(ret._real)
		ret.keyframe = keyframe
		yield ret
		ret.close()
		return
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


@pytest.fixture(
	params=[
		"proxy",
		"serial",
		pytest.param("parallel", marks=pytest.mark.parallel),
	]
)
def execution(request):
	return request.param


@pytest.fixture(
	params=["serial", pytest.param("parallel", marks=pytest.mark.parallel)]
)
def serial_or_parallel(request):
	return request.param


@pytest.fixture(
	params=[
		"nodb",
		pytest.param("parquetdb", marks=pytest.mark.parquetdb),
		pytest.param("sqlite", marks=pytest.mark.sqlite),
	]
)
def database(request):
	return request.param


@pytest.fixture(
	params=[
		pytest.param("parquetdb", marks=pytest.mark.parquetdb),
		pytest.param("sqlite", marks=pytest.mark.sqlite),
	]
)
def non_null_database(request):
	return request.param


@pytest.fixture(
	scope="function",
)
def engy(tmp_path, execution, database):
	"""Engine or EngineProxy, but not connected to a subprocess"""
	if execution == "proxy":
		eng = EngineProxy(
			None,
			None,
			getLogger("lisien proxy"),
			prefix=None,
			worker_index=0,
			eternal={"language": "eng"},
			function={},
			method={},
			trigger={},
			prereq={},
			action={},
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


@pytest.fixture(params=["local", "remote"])
def local_or_remote(request):
	return request.param


@pytest.fixture
def engine(tmp_path, serial_or_parallel, local_or_remote, database):
	"""Engine or EngineProxy with a subprocess"""
	if local_or_remote == "remote":
		procman = EngineProcessManager()
		with procman.start(
			**make_test_engine_kwargs(tmp_path, serial_or_parallel, database)
		) as proxy:
			yield proxy
		procman.shutdown()
	else:
		with Engine(
			**make_test_engine_kwargs(tmp_path, serial_or_parallel, database)
		) as eng:
			yield eng


def proxyless_engine(tmp_path, serial_or_parallel, database):
	with Engine(
		tmp_path if database != "null" else None,
		random_seed=69105,
		enforce_end_of_time=False,
		workers=0 if serial_or_parallel == "serial" else 2,
		connect_string=f"sqlite:///{tmp_path}/world.sqlite3"
		if database == "sqlite"
		else None,
	) as eng:
		yield eng


@pytest.fixture(params=[pytest.param("sqlite", marks=[pytest.mark.sqlite])])
def sqleng(tmp_path, request, execution):
	if execution == "proxy":
		eng = EngineProxy(
			None,
			None,
			getLogger("sqleng"),
			prefix=tmp_path,
			worker_index=0,
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
		tmp_path if database == "parquetdb" else None,
		random_seed=69105,
		enforce_end_of_time=False,
		workers=0,
		connect_string=f"sqlite:///:memory:" if database == "sqlite" else None,
	) as eng:
		yield eng


@pytest.fixture(scope="function")
def null_engine():
	with Engine(
		None, random_seed=69105, enforce_end_of_time=False, workers=0
	) as eng:
		yield eng


@pytest.fixture(
	scope="function", params=[pytest.param("sqlite", marks=pytest.mark.sqlite)]
)
def college24_premade(tmp_path, request):
	shutil.unpack_archive(
		os.path.join(
			os.path.abspath(os.path.dirname(__file__)),
			"data",
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


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_sessionfinish(session, exitstatus):
	# Remove handlers from all loggers to prevent logging errors on exit
	# From https://github.com/blacklanternsecurity/bbot/pull/1555
	# Works around a bug in Python 3.10 I think?
	import logging

	loggers = list(logging.Logger.manager.loggerDict.values())
	for logger in loggers:
		handlers = getattr(logger, "handlers", [])
		for handler in handlers:
			logger.removeHandler(handler)

	yield
