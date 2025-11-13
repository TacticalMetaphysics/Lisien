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
import resource
import shutil
import sys
from functools import partial
from logging import getLogger

import pytest

from lisien import Engine
from lisien.db import NullDatabaseConnector, PythonDatabaseConnector
from lisien.proxy.handle import EngineHandle
from lisien.proxy.manager import Sub

from ..examples import college, kobold, sickle
from ..pqdb import ParquetDatabaseConnector
from ..proxy.engine import EngineProxy
from ..proxy.manager import EngineProxyManager
from ..sql import SQLAlchemyDatabaseConnector
from . import data
from .util import (
	make_test_engine,
	make_test_engine_facade,
	make_test_engine_kwargs,
)


@pytest.fixture(scope="session", autouse=True)
def lots_of_open_files():
	"""Allow ParquetDB to make all the files it wants"""
	resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 69105))


@pytest.fixture(
	params=[
		"thread",
		"process",
		pytest.param(
			"interpreter",
			marks=pytest.mark.skipif(
				sys.version_info.minor < 14,
				reason="Subinterpreters are unavailable before Python 3.14",
			),
		),
	]
)
def sub_mode(request):
	"""Modes that workers and the Lisien core can run parallel in

	Originally just 'process', this has expanded to include 'thread' and
	'interpreter', of which the latter only exists on Python 3.14 and later.

	"""
	if sys.version_info.minor < 14 and request.param == "interpreter":
		raise pytest.skip("Subinterpreters are unavailable before Python 3.14")
	yield Sub(request.param)


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
	if database in {"nodb", "python"}:
		if database == "nodb":
			connector = NullDatabaseConnector()
		else:
			assert database == "python"
			connector = PythonDatabaseConnector()
		ret = EngineHandle(
			None, workers=0, random_seed=69105, database=connector
		)
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


KINDS_OF_PARALLEL = [
	pytest.param(
		"process", marks=[pytest.mark.parallel, pytest.mark.subprocess]
	),
	pytest.param(
		"interpreter",
		marks=[
			pytest.mark.parallel,
			pytest.mark.subinterpreter,
			pytest.mark.skipif(
				sys.version_info.minor < 14,
				reason="Subinterpreters are unavailable before Python 3.14",
			),
		],
	),
	pytest.param(
		"thread", marks=[pytest.mark.parallel, pytest.mark.subthread]
	),
]


@pytest.fixture(
	params=[
		pytest.param("proxy", marks=pytest.mark.proxy),
		"serial",
		*KINDS_OF_PARALLEL,
	]
)
def execution(request):
	return request.param


@pytest.fixture(params=["serial", *KINDS_OF_PARALLEL])
def serial_or_parallel(request):
	return request.param


@pytest.fixture(
	params=[
		"nodb",
		"python",
		pytest.param("parquetdb", marks=pytest.mark.parquetdb),
		pytest.param("sqlite", marks=pytest.mark.sqlite),
	]
)
def database(request):
	return request.param


@pytest.fixture(
	params=[
		pytest.param("python"),
		pytest.param("parquetdb", marks=pytest.mark.parquetdb),
		pytest.param("sqlite", marks=pytest.mark.sqlite),
	]
)
def non_null_database(request):
	return request.param


@pytest.fixture(
	params=[
		pytest.param("parquetdb", marks=pytest.mark.parquetdb),
		pytest.param("sqlite", marks=pytest.mark.sqlite),
	]
)
def persistent_database(request):
	return request.param


@pytest.fixture
def database_connector_part(tmp_path, non_null_database):
	match non_null_database:
		case "python":
			real_connector = PythonDatabaseConnector()
			return lambda: real_connector
		case "sqlite":
			return partial(
				SQLAlchemyDatabaseConnector,
				f"sqlite:///{tmp_path}/world.sqlite3",
			)
		case "parquetdb":
			return partial(
				ParquetDatabaseConnector, os.path.join(tmp_path, "world")
			)
	raise RuntimeError("Unknown database", non_null_database)


@pytest.fixture
def persistent_database_connector_part(tmp_path, persistent_database):
	match persistent_database:
		case "sqlite":
			return partial(
				SQLAlchemyDatabaseConnector,
				f"sqlite:///{tmp_path}/world.sqlite3",
			)
		case "parquetdb":
			return partial(
				ParquetDatabaseConnector, os.path.join(tmp_path, "world")
			)
	raise RuntimeError("Unknown database", persistent_database)


@pytest.fixture(scope="function")
def database_connector(database_connector_part):
	return database_connector_part()


@pytest.fixture(
	scope="function",
)
def engy(tmp_path, execution, database, random_seed):
	"""Engine or EngineProxy, but, if EngineProxy, it's not connected to a core"""
	with make_test_engine(tmp_path, execution, database, random_seed) as eng:
		yield eng
	if hasattr(eng, "_worker_log_threads"):
		for t in eng._worker_log_threads:
			assert not t.is_alive()
		assert not eng._fut_manager_thread.is_alive()


@pytest.fixture(params=["local", "remote"])
def local_or_remote(request):
	return request.param


@pytest.fixture
def engine(
	tmp_path,
	serial_or_parallel,
	local_or_remote,
	non_null_database,
	random_seed,
):
	"""Engine or EngineProxy with a subprocess"""
	if local_or_remote == "remote":
		procman = EngineProxyManager()
		with procman.start(
			**make_test_engine_kwargs(
				tmp_path, serial_or_parallel, non_null_database, random_seed
			)
		) as proxy:
			yield proxy
		procman.shutdown()
	else:
		with Engine(
			**make_test_engine_kwargs(
				tmp_path, serial_or_parallel, non_null_database, random_seed
			)
		) as eng:
			yield eng
		if hasattr(eng, "_worker_log_threads"):
			for t in eng._worker_log_threads:
				assert not t.is_alive()
			assert not eng._fut_manager_thread.is_alive()


def proxyless_engine(tmp_path, serial_or_parallel, database_connector):
	with Engine(
		tmp_path,
		random_seed=69105,
		enforce_end_of_time=False,
		workers=0 if serial_or_parallel == "serial" else 2,
		database=database_connector,
	) as eng:
		yield eng
	if hasattr(eng, "_worker_log_threads"):
		for t in eng._worker_log_threads:
			assert not t.is_alive()
		assert not eng._fut_manager_thread.is_alive()


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
			sub_mode=Sub(execution) if execution != "serial" else None,
			connect_string=f"sqlite:///{tmp_path}/world.sqlite3",
		) as eng:
			yield eng
	if hasattr(eng, "_worker_log_threads"):
		for t in eng._worker_log_threads:
			assert not t.is_alive()
		assert not eng._fut_manager_thread.is_alive()


@pytest.fixture(scope="function")
def serial_engine(tmp_path, persistent_database):
	with Engine(
		tmp_path,
		random_seed=69105,
		enforce_end_of_time=False,
		workers=0,
		connect_string=f"sqlite:///{tmp_path}/world.sqlite3"
		if persistent_database == "sqlite"
		else None,
	) as eng:
		yield eng
	if hasattr(eng, "_worker_log_threads"):
		for t in eng._worker_log_threads:
			assert not t.is_alive()
		assert not eng._fut_manager_thread.is_alive()


@pytest.fixture(scope="function")
def null_engine():
	with Engine(
		None,
		random_seed=69105,
		enforce_end_of_time=False,
		workers=0,
		database=NullDatabaseConnector(),
	) as eng:
		yield eng
	if hasattr(eng, "_worker_log_threads"):
		for t in eng._worker_log_threads:
			assert not t.is_alive()
		assert not eng._fut_manager_thread.is_alive()


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
	import threading

	loggers = list(logging.Logger.manager.loggerDict.values())
	for logger in loggers:
		handlers = getattr(logger, "handlers", [])
		for handler in handlers:
			logger.removeHandler(handler)

	print("Remaining threads:", threading.enumerate())

	yield
