from types import SimpleNamespace

from lisien import Engine


def make_test_engine_kwargs(
	path,
	execution,
	database,
	funcstores=True,
	random_seed=69105,
	enforce_end_of_time=False,
):
	kwargs = {
		"random_seed": random_seed,
		"enforce_end_of_time": enforce_end_of_time,
		"prefix": path,
	}
	if database == "sqlite":
		kwargs["connect_string"] = f"sqlite:///{path}/world.sqlite3"
	if not funcstores:
		for funcstore in ("function", "method", "trigger", "prereq", "action"):
			kwargs[funcstore] = SimpleNamespace()
	kwargs["workers"] = 2 if execution == "parallel" else 0
	return kwargs


def make_test_engine(path, execution, database, funcstores=True):
	return Engine(
		**make_test_engine_kwargs(path, execution, database, funcstores)
	)
