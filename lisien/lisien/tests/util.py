import os
from functools import wraps
from queue import SimpleQueue, Empty
from threading import Thread
from typing import Any, Callable, TypeVar
from unittest.mock import MagicMock

from lisien import Engine
from lisien.db import (
	PythonDatabaseConnector,
	SQLAlchemyDatabaseConnector,
	ParquetDatabaseConnector,
)
from lisien.facade import EngineFacade
from lisien.proxy.manager import Sub


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
	random_seed=69105,
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


def make_test_engine(path, execution, database):
	kwargs = {
		"workers": 0 if execution == "serial" else 2,
	}
	if execution != "serial":
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
