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
import pickle
import resource
import sys
from functools import partial
from logging import getLogger

import pytest

from lisien.engine import Engine
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

from ..pqdb import ParquetDatabaseConnector
from ..proxy.engine import EngineProxy
from ..proxy.manager import EngineProxyManager
from ..sql import SQLAlchemyDatabaseConnector
from . import data
from .util import (
	make_test_engine,
	make_test_engine_facade,
	make_test_engine_kwargs,
	college_engine,
	tar_cache,
	untar_cache,
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
		prefix=tmp_path,
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
def handle_initialized(request, tmp_path, database, random_seed):
	if request.param == "kobold":
		from lisien.examples.kobold import inittest

		install = partial(inittest, shrubberies=20, kobold_sprint_chance=0.9)
		keyframe = {0: data.KOBOLD_KEYFRAME_0, 1: data.KOBOLD_KEYFRAME_1}
	elif request.param == "college":
		from lisien.examples.college import install

		keyframe = {0: data.COLLEGE_KEYFRAME_0, 1: data.COLLEGE_KEYFRAME_1}
	else:
		assert request.param == "sickle"
		from lisien.examples.sickle import install

		keyframe = {0: data.SICKLE_KEYFRAME_0, 1: data.SICKLE_KEYFRAME_1}
	if database in {"nodb", "python"}:
		if database == "nodb":
			connector = NullDatabaseConnector()
		else:
			assert database == "python"
			connector = PythonDatabaseConnector()
		ret = EngineHandle(
			None,
			workers=0,
			random_seed=random_seed,
			database=connector,
		)
		install(ret._real)
		ret.keyframe = keyframe
		yield ret
		ret.close()
		return
	with Engine(
		prefix=tmp_path,
		workers=0,
		random_seed=random_seed,
		connect_string=f"sqlite:///{tmp_path}/world.sqlite3"
		if database == "sqlite"
		else None,
	) as eng:
		install(eng)
	ret = EngineHandle(
		prefix=tmp_path,
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
	scope="session",
	params=[
		pytest.param("python"),
		pytest.param("parquetdb", marks=pytest.mark.parquetdb),
		pytest.param("sqlite", marks=pytest.mark.sqlite),
	],
)
def non_null_database(request):
	return request.param


@pytest.fixture(
	scope="session",
	params=[
		pytest.param("parquetdb", marks=pytest.mark.parquetdb),
		pytest.param("sqlite", marks=pytest.mark.sqlite),
	],
)
def persistent_database(request):
	return request.param


def database_connector_partial(tmp_path, database):
	match database:
		case "python":
			try:
				with open(tmp_path.joinpath("database.pkl"), "rb") as f:
					pydb = pickle.load(f)
				return pydb
			except FileNotFoundError:
				db = PythonDatabaseConnector()

				def part(*_):
					return db

				part.is_python = True
				return part
		case "sqlite":
			return partial(
				SQLAlchemyDatabaseConnector,
				connect_string=f"sqlite:///{tmp_path}/world.sqlite3",
			)
		case "parquetdb":
			return partial(
				ParquetDatabaseConnector, path=tmp_path.joinpath("world")
			)


@pytest.fixture
def database_connector_part(tmp_path, non_null_database):
	return database_connector_partial(tmp_path, non_null_database)


@pytest.fixture
def database_connector_part2(tmp_path, non_null_database):
	return database_connector_partial(tmp_path, non_null_database)


@pytest.fixture
def reusing_database_connector_part2(
	tmp_path,
	non_null_database,
	reusing_python_database_connector_part,
	reusing_sqlalchemy_database_connector_part,
	reusing_parquetdb_database_connector_part,
):
	match non_null_database:
		case "python":
			yield reusing_python_database_connector_part
		case "sqlite":
			yield reusing_sqlalchemy_database_connector_part
		case "parquetdb":
			yield reusing_parquetdb_database_connector_part


@pytest.fixture(scope="session")
def session_process_executor():
	with LisienProcessExecutorProxy(None) as x:
		yield x


@pytest.fixture(scope="session")
def session_thread_executor():
	with LisienThreadExecutorProxy(None) as x:
		yield x


@pytest.fixture(scope="session")
def session_interpreter_executor():
	if sys.version_info.minor < 14:
		yield None
		return
	with LisienInterpreterExecutorProxy(None) as x:
		yield x


@pytest.fixture(scope="session")
def session_executor(
	execution,
	session_process_executor,
	session_thread_executor,
	session_interpreter_executor,
):
	ex = None
	match execution:
		case "process":
			ex = session_process_executor
		case "thread":
			ex = session_thread_executor
		case "interpreter":
			ex = session_interpreter_executor
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
	serial_or_parallel,
	session_process_executor,
	session_thread_executor,
	session_interpreter_executor,
):
	ex = None
	match serial_or_parallel:
		case "process":
			ex = session_process_executor
		case "thread":
			ex = session_thread_executor
		case "interpreter":
			ex = session_interpreter_executor
		case _:
			ex = None
	yield ex
	if ex is None:
		return
	if hasattr(ex, "_worker_log_threads"):
		for t in ex._worker_log_threads:
			assert not t.is_alive()
		assert not ex._fut_manager_thread.is_alive()


@pytest.fixture(scope="function", params=KINDS_OF_PARALLEL)
def parallel_executor(
	request,
	session_process_executor,
	session_thread_executor,
	session_interpreter_executor,
):
	match request.param:
		case "thread":
			with session_thread_executor:
				yield session_thread_executor
		case "process":
			with session_process_executor:
				yield session_process_executor
		case "interpreter":
			with session_interpreter_executor:
				yield session_interpreter_executor


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
			tmp_path,
			**make_test_engine_kwargs(
				tmp_path,
				serial_or_parallel,
				non_null_database,
				random_seed,
			),
		) as proxy:
			yield proxy
		procman.shutdown()
	else:
		with Engine(
			tmp_path,
			**make_test_engine_kwargs(
				tmp_path,
				serial_or_parallel,
				non_null_database,
				random_seed,
			),
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
			branches_d={"trunk": (None, 0, 0, 0, 0)},
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


@pytest.fixture(scope="session")
def college10_tar(non_null_database):
	with tar_cache("college10", non_null_database, college_engine) as path:
		yield path


@pytest.fixture
def college10(
	college10_tar,
	tmp_path,
	database_connector_part,
	serial_or_parallel,
):
	with untar_cache(
		college10_tar, tmp_path, database_connector_part, serial_or_parallel
	) as eng:
		yield eng


@pytest.fixture(scope="session")
def college24_sql_tar():
	with tar_cache("college24", "sqlite", college_engine) as path:
		yield path


@pytest.fixture(scope="session")
def college24_pqdb_tar():
	with tar_cache("college24", "parquetdb", college_engine) as path:
		yield path


@pytest.fixture(scope="session")
def college24_python_tar():
	with tar_cache("college24", "python", college_engine) as path:
		yield path


@pytest.fixture(scope="session")
def college24_tar(
	non_null_database,
	college24_python_tar,
	college24_sql_tar,
	college24_pqdb_tar,
):
	match non_null_database:
		case "python":
			yield college24_python_tar
		case "parquetdb":
			yield college24_pqdb_tar
		case "sqlite":
			yield college24_sql_tar
		case _:
			raise RuntimeError("Database not supported", non_null_database)


@pytest.fixture
def college24(
	college24_tar,
	tmp_path,
	database_connector_part,
	serial_or_parallel,
):
	with untar_cache(
		college24_tar, tmp_path, database_connector_part, serial_or_parallel
	) as eng:
		yield eng


@pytest.fixture(scope="session")
def sickle_tar(non_null_database):
	with tar_cache("sickle", non_null_database) as path:
		yield path


@pytest.fixture
def sickle(sickle_tar, tmp_path, database_connector_part, serial_or_parallel):
	with untar_cache(
		sickle_tar,
		tmp_path,
		database_connector_part,
		serial_or_parallel,
	) as eng:
		yield eng


@pytest.fixture(scope="session")
def wolfsheep_tar(non_null_database):
	with tar_cache("wolfsheep", non_null_database) as path:
		yield path


@pytest.fixture
def wolfsheep(
	wolfsheep_tar,
	tmp_path,
	database_connector_part,
	serial_or_parallel,
):
	with untar_cache(
		wolfsheep_tar,
		tmp_path,
		database_connector_part,
		serial_or_parallel,
	) as eng:
		yield eng


@pytest.fixture(scope="session")
def pathfind_tar(non_null_database):
	with tar_cache("pathfind", non_null_database) as path:
		yield path


@pytest.fixture(params=KINDS_OF_PARALLEL)
def pathfind(pathfind_tar, tmp_path, database_connector_part, request):
	with untar_cache(
		pathfind_tar,
		tmp_path,
		database_connector_part,
		request.param,
	) as eng:
		yield eng


def pytest_collection_modifyitems(items):
	for item in items:
		fixturenames = getattr(item, "fixturenames", ())
		if "college10" in fixturenames or "college24" in fixturenames:
			item.add_marker("college")


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
