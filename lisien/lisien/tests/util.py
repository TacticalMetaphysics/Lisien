from __future__ import annotations

import os
from contextlib import contextmanager
from functools import wraps, partial
from itertools import product
from logging import getLogger
from os import PathLike
from queue import Empty, SimpleQueue
from threading import Thread
from typing import Any, Callable, TypeVar, Literal
from unittest.mock import MagicMock, patch

from lisien import Engine
from lisien.db import (
	NullDatabaseConnector,
	PythonDatabaseConnector,
	AbstractDatabaseConnector,
)
from lisien.facade import EngineFacade
from lisien.pqdb import ParquetDatabaseConnector
from lisien.proxy.engine import EngineProxy
from lisien.proxy.manager import Sub
from lisien.sql import SQLAlchemyDatabaseConnector
from lisien.tests.data import DATA_DIR
from lisien.types import (
	LoadedDict,
	Keyframe,
	GraphNodeValKeyframe,
	GraphValKeyframe,
)

_RETURNS = TypeVar("_RETURNS")


def fail_after(
	timeout: float, now: bool = True
) -> Callable[[Callable[[], _RETURNS]], _RETURNS]:
	"""If the wrapped function doesn't finish before the timeout, fail the test.

	This executes the wrapped function in a daemonic subthread. If the function
	doesn't exit, neither will the subthread, which may cause pytest headaches
	if that thread keeps running after all the tests are done.

	By default, this immediately starts running the wrapped function. If you
	don't want this, set ``now=False``.

	"""

	def call_and_put_return_value(
		func: Callable[[], Any], q: SimpleQueue
	) -> None:
		q.put(func())

	def fail_after_decorator(
		func: Callable[[], _RETURNS],
	) -> Callable[[], _RETURNS]:
		@wraps(func)
		def fails():
			q: SimpleQueue = SimpleQueue()
			call_thread = Thread(
				target=call_and_put_return_value, args=(func, q), daemon=True
			)
			call_thread.start()
			try:
				ret = q.get(timeout=timeout)
			except Empty as ex:
				raise TimeoutError(f"{func!r} timed out") from ex
			return ret

		if now:
			fails()
		return fails

	return fail_after_decorator


def make_test_engine_kwargs(
	path,
	execution,
	database,
	random_seed,
	enforce_end_of_time=False,
):
	kwargs = {
		"random_seed": random_seed,
		"enforce_end_of_time": enforce_end_of_time,
		"prefix": None if database in {"nodb", "python"} else path,
	}
	if database == "sqlite":
		kwargs["connect_string"] = f"sqlite:///{path}/world.sqlite3"
	if execution == "serial":
		kwargs["workers"] = 0
	elif execution != "proxy":
		kwargs["workers"] = 2
		kwargs["sub_mode"] = Sub(execution)
	return kwargs


def make_test_engine(path, execution, database, random_seed):
	kwargs = {"random_seed": random_seed}
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
		return eng
	if execution == "serial":
		kwargs["workers"] = 0
	else:
		kwargs["workers"] = 2
		kwargs["sub_mode"] = Sub(execution)
	match database:
		case "python":
			kwargs["database"] = PythonDatabaseConnector()
		case "sqlite":
			kwargs["database"] = SQLAlchemyDatabaseConnector(
				f"sqlite:///{path}/world.sqlite3"
			)
		case "parquetdb":
			kwargs["database"] = ParquetDatabaseConnector(
				os.path.join(path, "world")
			)
		case "nodb":
			kwargs["database"] = NullDatabaseConnector()
		case _:
			raise RuntimeError("Unknown database", database)
	return Engine(path, **kwargs)


def make_test_engine_facade() -> EngineFacade:
	fac = EngineFacade(None)
	fac.function = MagicMock()
	fac.method = MagicMock()
	fac.trigger = MagicMock()
	fac.prereq = MagicMock()
	fac.action = MagicMock()
	return fac


def hash_loaded_dict(data: LoadedDict) -> dict[str, dict[str, int] | int]:
	def unlist(o):
		if isinstance(o, (list, tuple)):
			return tuple(map(unlist, o))
		return o

	hashes = {}
	for k, v in data.items():
		if isinstance(v, list):
			the_hash = 0
			for elem in v:
				the_hash |= hash(tuple(map(unlist, elem)))
			hashes[k] = the_hash
		elif isinstance(v, dict):
			hasheses = hashes[k] = {}
			for k, v in v.items():
				the_hash = 0
				for elem in v:
					the_hash |= hash(tuple(map(unlist, elem)))
				hasheses[k] = the_hash
		else:
			raise TypeError("Invalid loaded dictionary")
	return hashes


def get_database_connector_part(
	tmp_path: str | PathLike, s: Literal["python", "sqlite", "parquetdb"]
):
	def get_python_connector():
		if not hasattr(get_database_connector_part, "_pyconnector"):
			get_database_connector_part._pyconnector = (
				PythonDatabaseConnector()
			)
		pyconnector = get_database_connector_part._pyconnector
		pyconnector._plan_times = {}
		return pyconnector

	match s:
		case "python":
			return get_python_connector
		case "sqlite":
			return partial(
				SQLAlchemyDatabaseConnector,
				f"sqlite:///{tmp_path}/world.sqlite3",
			)
		case "parquetdb":
			return partial(
				ParquetDatabaseConnector, os.path.join(tmp_path, "world")
			)


@contextmanager
def college_engine(
	archive_fn: str,
	tmp_path: str | PathLike,
	serial_or_parallel: str,
	database_connector_part: Callable[[], AbstractDatabaseConnector],
):
	def validate_final_keyframe(kf: Keyframe):
		node_val: GraphNodeValKeyframe = kf["node_val"]
		phys_node_val = node_val["physical"]
		graph_val: GraphValKeyframe = kf["graph_val"]
		assert "student_body" in graph_val
		assert "units" in graph_val["student_body"]
		assert "physical" in graph_val["student_body"]["units"]
		for unit in graph_val["student_body"]["units"]["physical"]:
			assert unit in phys_node_val
			assert "location" in phys_node_val[unit]
		for key, dorm, room, student in product(
			["graph_val", "node_val", "edge_val"],
			[0, 1],
			[0, 1],
			[0, 1],
		):
			assert f"dorm{dorm}room{room}student{student}" in kf[key]

	with (
		patch(
			"lisien.Engine._validate_initial_keyframe_load",
			staticmethod(validate_final_keyframe),
			create=True,
		),
		Engine.from_archive(
			os.path.join(DATA_DIR, archive_fn),
			tmp_path,
			workers=0 if serial_or_parallel == "serial" else 2,
			sub_mode=None
			if serial_or_parallel == "serial"
			else Sub(serial_or_parallel),
			database=database_connector_part(),
		) as eng,
	):
		yield eng
