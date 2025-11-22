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
import sys
from functools import partial
from logging import getLogger

import pytest

from lisien.engine import (
	Engine,
)
from ..futures import (
	LisienThreadExecutorProxy,
	LisienProcessExecutorProxy,
	LisienInterpreterExecutorProxy,
)
from lisien.db import (
	NullDatabaseConnector,
	PythonDatabaseConnector,
)
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
	get_database_connector_part,
	college_engine,
	restart_executor,
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
def handle_initialized(
	request, tmp_path, database, random_seed, serial_or_executor
):
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
			None,
			workers=0 if serial_or_executor is None else 2,
			executor=serial_or_executor,
			random_seed=random_seed,
			database=connector,
		)
		install(ret._real)
		ret.keyframe = keyframe
		yield ret
		ret.close()
		return
	with Engine(
		tmp_path,
		workers=0,
		random_seed=random_seed,
		connect_string=f"sqlite:///{tmp_path}/world.sqlite3"
		if database == "sqlite"
		else None,
	) as eng:
		install(eng)
	ret = EngineHandle(
		tmp_path,
		workers=0 if serial_or_executor is None else 2,
		executor=serial_or_executor,
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
	scope="session",
	params=[
		pytest.param("proxy", marks=pytest.mark.proxy),
		"serial",
		*KINDS_OF_PARALLEL,
	],
)
def execution(request):
	return request.param


@pytest.fixture(scope="session", params=["serial", *KINDS_OF_PARALLEL])
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
	return get_database_connector_part(tmp_path, non_null_database)


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
def database_connector(tmp_path, non_null_database):
	match non_null_database:
		case "python":
			connector = PythonDatabaseConnector()
		case "sqlite":
			connector = SQLAlchemyDatabaseConnector(
				f"sqlite:///{tmp_path}/world.sqlite3"
			)
		case "parquetdb":
			connector = ParquetDatabaseConnector(
				os.path.join(tmp_path, "world")
			)
		case _:
			raise ValueError("Unknown database", non_null_database)
	if hasattr(connector, "is_empty"):
		assert connector.is_empty()
	return connector


@pytest.fixture(scope="session")
def process_executor():
	with LisienProcessExecutorProxy(
		None,
		getLogger("lisien"),
		("trunk", 0, 0),
		{"branch": "trunk", "turn": 0, "tick": 0, "trunk": "trunk"},
		{"trunk": (None, 0, 0, 0, 0)},
		2,
	) as x:
		yield x


@pytest.fixture(scope="session")
def thread_executor():
	with LisienThreadExecutorProxy(
		None,
		getLogger("lisien"),
		("trunk", 0, 0),
		{"branch": "trunk", "turn": 0, "tick": 0, "trunk": "trunk"},
		{"trunk": (None, 0, 0, 0, 0)},
		2,
	) as x:
		yield x


@pytest.fixture(scope="session")
def interpreter_executor():
	if sys.version_info.minor < 14:
		yield None
		return
	with LisienInterpreterExecutorProxy(
		None,
		getLogger("lisien"),
		("trunk", 0, 0),
		{"branch": "trunk", "turn": 0, "tick": 0, "trunk": "trunk"},
		{"trunk": (None, 0, 0, 0, 0)},
		2,
	) as x:
		yield x


@pytest.fixture(scope="session")
def executor(
	execution, process_executor, thread_executor, interpreter_executor
):
	ex = None
	match execution:
		case "process":
			ex = process_executor
		case "thread":
			ex = thread_executor
		case "interpreter":
			ex = interpreter_executor
		case _:
			ex = None
	yield ex
	if ex is None:
		return
	if hasattr(ex, "_worker_log_threads"):
		for t in ex._worker_log_threads:
			assert not t.is_alive()
		assert not ex._fut_manager_thread.is_alive()


@pytest.fixture(scope="session")
def serial_or_executor(
	serial_or_parallel, process_executor, thread_executor, interpreter_executor
):
	ex = None
	match serial_or_parallel:
		case "process":
			ex = process_executor
		case "thread":
			ex = thread_executor
		case "interpreter":
			ex = interpreter_executor
		case _:
			ex = None
	yield ex
	if ex is None:
		return
	if hasattr(ex, "_worker_log_threads"):
		for t in ex._worker_log_threads:
			assert not t.is_alive()
		assert not ex._fut_manager_thread.is_alive()


@pytest.fixture(
	scope="function",
)
def engy(tmp_path, execution, database, random_seed, executor):
	"""Engine or EngineProxy, but, if EngineProxy, it's not connected to a core"""
	with make_test_engine(
		tmp_path, execution, database, random_seed, executor=executor
	) as eng:
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
	database_connector_part,
	random_seed,
	serial_or_executor,
):
	"""Engine or EngineProxy with a subprocess"""
	if local_or_remote == "remote":
		procman = EngineProxyManager()
		with procman.start(
			**make_test_engine_kwargs(
				tmp_path,
				serial_or_parallel,
				database_connector_part,
				random_seed,
				executor=serial_or_executor,
			)
		) as proxy:
			yield proxy
		procman.shutdown()
	else:
		with Engine(
			**make_test_engine_kwargs(
				tmp_path,
				serial_or_parallel,
				database_connector_part,
				random_seed,
				executor=serial_or_executor,
			)
		) as eng:
			yield eng


@pytest.fixture
def no_proxy_executor(
	tmp_path,
	serial_or_parallel,
	random_seed,
	thread_executor,
	process_executor,
	interpreter_executor,
):
	match serial_or_parallel:
		case "serial":
			yield None
		case "thread":
			restart_executor(thread_executor, tmp_path, random_seed)
			yield thread_executor
		case "process":
			restart_executor(process_executor, tmp_path, random_seed)
			yield process_executor
		case "interpreter":
			restart_executor(interpreter_executor, tmp_path, random_seed)
			yield interpreter_executor


def proxyless_engine(
	tmp_path, serial_or_parallel, database_connector, no_proxy_executor
):
	executor = None
	with Engine(
		tmp_path,
		random_seed=69105,
		enforce_end_of_time=False,
		workers=0 if serial_or_parallel == "serial" else 2,
		database=database_connector,
		executor=no_proxy_executor,
	) as eng:
		yield eng
	if hasattr(eng, "_worker_log_threads"):
		for t in eng._worker_log_threads:
			assert not t.is_alive()
		assert not eng._fut_manager_thread.is_alive()


@pytest.fixture(params=[pytest.param("sqlite", marks=[pytest.mark.sqlite])])
def sqleng(tmp_path, request, execution, executor):
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
			executor=executor,
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


@pytest.fixture
def college10(
	tmp_path,
	serial_or_parallel,
	database_connector,
	no_proxy_executor,
):
	with college_engine(
		"college10.lisien",
		tmp_path,
		serial_or_parallel,
		database_connector,
		executor=no_proxy_executor,
	) as eng:
		yield eng


@pytest.fixture
def college24(
	tmp_path,
	serial_or_parallel,
	database_connector,
	no_proxy_executor,
):
	with college_engine(
		"college24.lisien",
		tmp_path,
		serial_or_parallel,
		database_connector,
		executor=no_proxy_executor,
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
