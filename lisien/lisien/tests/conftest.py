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
from contextlib import contextmanager
from functools import partial

import pytest

from lisien.db import NullDatabaseConnector, PythonDatabaseConnector
from lisien.db.pqdb import ParquetDatabaseConnector
from lisien.db.sql import SQLAlchemyDatabaseConnector
from lisien.engine import Engine
from lisien.proxy.handle import EngineHandle

from ..enum import Sub
from ..futures import InterpreterExecutor, ProcessExecutor, ThreadExecutor
from ..proxy.manager import EngineProxyManager
from . import data
from .util import make_test_engine_facade


@pytest.fixture(scope="session", autouse=True)
def lots_of_open_files():
	"""Allow ParquetDB to make all the files it wants"""
	resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 691050))


at_least_314 = pytest.mark.skipif(
	sys.version_info < (3, 14),
	reason="Subinterpreters are only available in Python 3.14 or newer",
)


@pytest.fixture(
	params=[
		pytest.param(
			"thread_workers",
			marks=[pytest.mark.parallel, pytest.mark.subthread],
		),
		pytest.param(
			"process_workers",
			marks=[pytest.mark.parallel, pytest.mark.subprocess],
		),
		pytest.param(
			"interpreter_workers",
			marks=[
				pytest.mark.parallel,
				pytest.mark.subinterpreter,
				at_least_314,
			],
		),
	]
)
def worker_sub_mode(request):
	"""Modes that workers and the Lisien core can run parallel in

	Originally just 'process', this has expanded to include 'thread' and
	'interpreter', of which the latter only exists on Python 3.14 and later.

	"""
	yield Sub(request.param.removesuffix("_workers"))


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
	if database in {"nodb", "pythondb"}:
		if database == "nodb":
			connector = NullDatabaseConnector
		else:
			assert database == "pythondb"
			connector = PythonDatabaseConnector
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
		"process_workers", marks=[pytest.mark.parallel, pytest.mark.subprocess]
	),
	pytest.param(
		"interpreter_workers",
		marks=[
			pytest.mark.parallel,
			pytest.mark.subinterpreter,
			at_least_314,
		],
	),
	pytest.param(
		"thread_workers", marks=[pytest.mark.parallel, pytest.mark.subthread]
	),
]


@pytest.fixture(scope="session", params=["serial", *KINDS_OF_PARALLEL])
def serial_or_parallel(request):
	return request.param.removesuffix("_workers")


@pytest.fixture(
	params=[
		"nodb",
		"pythondb",
		pytest.param("parquetdb", marks=pytest.mark.parquetdb),
		pytest.param("sqlite", marks=pytest.mark.sqlite),
	]
)
def database(request):
	return request.param


@pytest.fixture(
	scope="session",
	params=[
		pytest.param("pythondb"),
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
		case "pythondb":
			try:
				with open(tmp_path.joinpath("database.pkl"), "rb") as f:
					pydb = pickle.load(f)
				return pydb
			except FileNotFoundError:
				return PythonDatabaseConnector
		case "sqlite":
			return partial(
				SQLAlchemyDatabaseConnector,
				connect_string=f"sqlite:///{tmp_path}/world.sqlite3",
			)
		case "parquetdb":
			worldpath = tmp_path.joinpath("world")
			worldpath.mkdir(parents=True)
			return partial(ParquetDatabaseConnector, path=worldpath)


@pytest.fixture
def database_connector_part(tmp_path, non_null_database):
	return database_connector_partial(tmp_path, non_null_database)


@pytest.fixture
def persistent_database_connector_part(tmp_path, persistent_database):
	return database_connector_partial(tmp_path, persistent_database)


@pytest.fixture(
	scope="session",
	params=[
		"no_manager",
		pytest.param("thread_manager", marks=pytest.mark.subthread),
		pytest.param("process_manager", marks=pytest.mark.subprocess),
		pytest.param(
			"interpreter_manager",
			marks=[pytest.mark.subinterpreter, at_least_314],
		),
	],
)
def kind_of_proxy_manager(request):
	yield request.param


@pytest.fixture(scope="session")
def proxy_manager(
	serial_or_parallel,
	kind_of_proxy_manager,
):
	# The initial instantiation of the proxy manager doesn't reflect
	# what type of parallelism the core uses. You need to indicate that in the
	# ``sub_mode`` keyword argument to the ``start`` method of the proxy manager
	# -- not the proxy manager's own initializer.
	# We make a new, session-wide proxy manager for each possible mode
	# of parallelism that the core *and* the manager might be running in.
	if kind_of_proxy_manager == "no_manager":
		yield None
		return
	with EngineProxyManager(
		sub_mode=kind_of_proxy_manager.removesuffix("_manager"), reuse=True
	) as proxman:
		yield proxman


@pytest.fixture(scope="session")
def serial_or_executor(
	serial_or_parallel,
):
	match serial_or_parallel:
		case "serial":
			yield None
		case "process":
			with ProcessExecutor(None) as ex:
				yield ex
		case "thread":
			with ThreadExecutor(None) as ex:
				yield ex
		case "interpreter":
			with InterpreterExecutor(None) as ex:
				yield ex
		case _:
			raise ValueError("Unknown form of parallelism", serial_or_parallel)


@pytest.fixture(params=["local", "remote"])
def local_or_remote(request):
	return request.param


@pytest.fixture
def engine(
	tmp_path,
	serial_or_parallel,
	random_seed,
	proxy_manager,
	serial_or_executor,
	non_null_database,
	database_connector_part,
):
	"""Engine or EngineProxy with a subprocess"""
	if proxy_manager is None:
		with Engine(
			tmp_path,
			random_seed=random_seed,
			executor=serial_or_executor,
			workers=0 if serial_or_parallel == "serial" else 2,
			database=database_connector_part,
		) as eng:
			yield eng
	else:
		if (
			proxy_manager.sub_mode == Sub.interpreter
			and non_null_database == "parquetdb"
		):
			raise pytest.skip("PyArrow does not yet run in subinterpreters")
		with proxy_manager.start(
			tmp_path,
			random_seed=random_seed,
			workers=0 if serial_or_parallel == "serial" else 2,
			sub_mode=Sub(serial_or_parallel),
			database=database_connector_part,
		) as proxy:
			yield proxy


@pytest.fixture(params=[pytest.param("sqlite", marks=[pytest.mark.sqlite])])
def sqleng(tmp_path, request, serial_or_parallel):
	with Engine(
		tmp_path,
		random_seed=69105,
		enforce_end_of_time=False,
		workers=0 if serial_or_parallel == "serial" else 2,
		sub_mode=Sub(serial_or_parallel),
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
		database=NullDatabaseConnector,
	) as eng:
		yield eng
	if hasattr(eng, "_worker_log_threads"):
		for t in eng._worker_log_threads:
			assert not t.is_alive()
		assert not eng._fut_manager_thread.is_alive()


@contextmanager
def engine_from_archive(
	archive_name: str,
	tmp_path,
	serial_or_parallel,
	non_null_database,
	database_connector_part,
	serial_or_executor,
	proxy_manager,
):
	if (
		serial_or_parallel == "interpreter"
		and non_null_database == "parquetdb"
	):
		raise pytest.skip("PyArrow does not yet run in subinterpreters")
	if proxy_manager is None:
		with Engine.from_archive(
			data.DATA_DIR.joinpath(archive_name),
			tmp_path,
			executor=serial_or_executor,
			database=database_connector_part,
		) as eng:
			yield eng
	else:
		with proxy_manager.load_archive(
			data.DATA_DIR.joinpath(archive_name),
			tmp_path,
			workers=0 if serial_or_parallel == "serial" else 2,
			sub_mode=Sub(serial_or_parallel),
			database=database_connector_part,
		) as eng:
			yield eng


@pytest.fixture(scope="function")
def college10(
	tmp_path,
	serial_or_parallel,
	non_null_database,
	database_connector_part,
	serial_or_executor,
	proxy_manager,
):
	if (
		proxy_manager
		and proxy_manager.sub_mode == Sub.interpreter
		and non_null_database == "parquetdb"
	):
		raise pytest.skip("Can't run PyArrow in subinterpreters yet")
	with engine_from_archive(
		"college10.lisien",
		tmp_path,
		serial_or_parallel,
		non_null_database,
		database_connector_part,
		serial_or_executor,
		proxy_manager,
	) as eng:
		yield eng


@pytest.fixture(scope="function")
def college24(
	tmp_path,
	serial_or_parallel,
	non_null_database,
	database_connector_part,
	serial_or_executor,
	proxy_manager,
):
	if (
		proxy_manager
		and proxy_manager.sub_mode == Sub.interpreter
		and non_null_database == "parquetdb"
	):
		raise pytest.skip("Can't run PyArrow in subinterpreters yet")
	with engine_from_archive(
		"college24.lisien",
		tmp_path,
		serial_or_parallel,
		non_null_database,
		database_connector_part,
		serial_or_executor,
		proxy_manager,
	) as eng:
		yield eng


@pytest.fixture(scope="function")
def sickle(
	tmp_path,
	serial_or_parallel,
	non_null_database,
	database_connector_part,
	serial_or_executor,
	proxy_manager,
):
	if (
		proxy_manager
		and proxy_manager.sub_mode == Sub.interpreter
		and non_null_database == "parquetdb"
	):
		raise pytest.skip("Can't run PyArrow in subinterpreters yet")
	with engine_from_archive(
		"sickle.lisien",
		tmp_path,
		serial_or_parallel,
		non_null_database,
		database_connector_part,
		serial_or_executor,
		proxy_manager,
	) as eng:
		yield eng


@pytest.fixture(scope="function")
def wolfsheep(
	tmp_path,
	serial_or_parallel,
	non_null_database,
	database_connector_part,
	serial_or_executor,
	proxy_manager,
):
	if (
		proxy_manager
		and proxy_manager.sub_mode == Sub.interpreter
		and non_null_database == "parquetdb"
	):
		raise pytest.skip("Can't run PyArrow in subinterpreters yet")
	with engine_from_archive(
		"wolfsheep.lisien",
		tmp_path,
		serial_or_parallel,
		non_null_database,
		database_connector_part,
		serial_or_executor,
		proxy_manager,
	) as eng:
		yield eng


@pytest.fixture(scope="function")
def pathfind(
	tmp_path,
	serial_or_parallel,
	non_null_database,
	database_connector_part,
	serial_or_executor,
	proxy_manager,
):
	if (
		proxy_manager
		and proxy_manager.sub_mode == Sub.interpreter
		and non_null_database == "parquetdb"
	):
		raise pytest.skip("Can't run PyArrow in subinterpreters yet")
	with engine_from_archive(
		"pathfind.lisien",
		tmp_path,
		serial_or_parallel,
		non_null_database,
		database_connector_part,
		serial_or_executor,
		proxy_manager,
	) as eng:
		yield eng
