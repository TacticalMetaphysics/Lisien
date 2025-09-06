from unittest.mock import MagicMock

from lisien import Engine
from lisien.facade import EngineFacade


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
		"prefix": None if database == "nodb" else path,
	}
	if database == "sqlite":
		kwargs["connect_string"] = f"sqlite:///{path}/world.sqlite3"
	kwargs["workers"] = 2 if execution == "parallel" else 0
	return kwargs


def make_test_engine(path, execution, database):
	return Engine(**make_test_engine_kwargs(path, execution, database))


def make_test_engine_facade() -> EngineFacade:
	fac = EngineFacade(None)
	fac.function = MagicMock()
	fac.method = MagicMock()
	fac.trigger = MagicMock()
	fac.prereq = MagicMock()
	fac.action = MagicMock()
	return fac
