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
import difflib
import filecmp
import json
import os
from ast import parse, unparse
from functools import partial

import pytest

from ..db import AbstractDatabaseConnector, PythonDatabaseConnector
from ..engine import Engine
from ..pqdb import ParquetDatabaseConnector
from ..sql import SQLAlchemyDatabaseConnector
from .data import DATA_DIR


def get_install_func(sim, random_seed):
	if sim == "kobold":
		from lisien.examples.kobold import inittest as install

		return install
	elif sim == "polygons":
		from lisien.examples.polygons import install

		return install
	elif sim == "wolfsheep":
		from lisien.examples.wolfsheep import install

		return partial(install, seed=random_seed)
	else:
		raise ValueError("Unknown sim", sim)


@pytest.fixture(params=["zero", "one"])
def turns(request):
	yield {"zero": 0, "one": 1}[request.param]


@pytest.fixture(params=["kobold", "polygons", "wolfsheep"])
def engine_and_exported_xml(
	tmp_path, random_seed, persistent_database, request, turns
):
	install = get_install_func(request.param, random_seed)
	prefix = os.path.join(tmp_path, "game")
	with Engine(
		prefix,
		workers=0,
		random_seed=random_seed,
		connect_string=f"sqlite:///{prefix}/world.sqlite3"
		if persistent_database == "sqlite"
		else None,
		keyframe_on_close=False,
	) as eng:
		install(eng)
		for _ in range(turns):
			eng.next_turn()
		yield eng, str(os.path.join(DATA_DIR, request.param + f"_{turns}.xml"))


def test_export_db(tmp_path, engine_and_exported_xml):
	test_xml = os.path.join(tmp_path, "test.xml")
	eng, outpath = engine_and_exported_xml
	eng.to_xml(test_xml, name="test_export")

	if not filecmp.cmp(outpath, test_xml):
		with (
			open(test_xml, "rt") as testfile,
			open(outpath, "rt") as goodfile,
		):
			differences = list(
				difflib.unified_diff(
					goodfile.readlines(),
					testfile.readlines(),
					test_xml,
					outpath,
				)
			)
		assert filecmp.cmp(outpath, test_xml), "".join(differences)


@pytest.fixture(params=["kobold", "polygons", "wolfsheep"])
def exported(
	tmp_path, random_seed, persistent_database_connector_part, request, turns
):
	install = get_install_func(request.param, random_seed)
	prefix = os.path.join(tmp_path, "game")
	with Engine(
		prefix,
		workers=0,
		random_seed=random_seed,
		keyframe_on_close=False,
		database=persistent_database_connector_part(),
	) as eng:
		install(eng)
		for _ in range(turns):
			eng.next_turn()
		archive_name = eng.export(request.param)
	yield archive_name


def test_round_trip(tmp_path, exported, non_null_database, random_seed, turns):
	prefix1 = os.path.join(tmp_path, "game")
	os.makedirs(prefix1, exist_ok=True)
	prefix2 = os.path.join(tmp_path, "game2")
	os.makedirs(prefix2, exist_ok=True)
	match non_null_database:
		case "python":
			db1 = PythonDatabaseConnector()
			db2 = PythonDatabaseConnector()
		case "sqlite":
			db1 = SQLAlchemyDatabaseConnector(
				f"sqlite:///{prefix1}/world.sqlite3"
			)
			db2 = SQLAlchemyDatabaseConnector(
				f"sqlite:///{prefix2}/world.sqlite3"
			)
		case "parquetdb":
			db1 = ParquetDatabaseConnector(os.path.join(prefix1, "world"))
			db2 = ParquetDatabaseConnector(os.path.join(prefix2, "world"))
		case _:
			raise RuntimeError("Unknown database", non_null_database)
	if exported.endswith("kobold.lisien"):
		from lisien.examples.kobold import inittest as install
	elif exported.endswith("wolfsheep.lisien"):
		from lisien.examples.wolfsheep import install

		install = partial(install, seed=random_seed)
	elif exported.endswith("polygons.lisien"):
		from lisien.examples.polygons import install
	else:
		raise pytest.fail(f"Unknown export: {exported}")
	with (
		Engine.from_archive(
			exported,
			prefix1,
			workers=0,
			database=db1,
			keyframe_on_close=False,
		) as eng1,
		Engine(
			prefix2,
			workers=0,
			database=db2,
			keyframe_on_close=False,
			random_seed=random_seed,
		) as eng2,
	):
		install(eng2)
		for _ in range(turns):
			eng2.next_turn()
		compare_engines_world_state(eng1, eng2)
	db1.close()
	db2.close()

	compare_stored_strings(prefix2, prefix1)
	compare_stored_python_code(prefix2, prefix1)


def compare_engines_world_state(
	correct_engine: Engine | AbstractDatabaseConnector,
	test_engine: Engine | AbstractDatabaseConnector,
):
	test_engine.commit()
	correct_engine.commit()
	test_engine = getattr(test_engine, "db", test_engine)
	correct_engine = getattr(correct_engine, "db", test_engine)
	test_dump = test_engine.dump_everything()
	correct_dump = correct_engine.dump_everything()
	assert test_dump.keys() == correct_dump.keys()
	for k, test_data in test_dump.items():
		if k.endswith("rules_handled"):
			continue
		correct_data = correct_dump[k]
		print(k)
		assert correct_data == test_data, f"{k} tables differ"


def compare_stored_strings(
	correct_prefix: str | os.PathLike, test_prefix: str | os.PathLike
):
	langs = os.listdir(os.path.join(test_prefix, "strings"))
	assert langs == os.listdir(os.path.join(correct_prefix, "strings")), (
		"Different languages"
	)
	for lang in langs:
		with (
			open(os.path.join(test_prefix, lang), "rb") as test_file,
			open(os.path.join(correct_prefix, lang), "rb") as correct_file,
		):
			assert json.load(correct_file) == json.load(test_file), (
				f"Different strings for language: {lang[:-5]}"
			)


def compare_stored_python_code(
	correct_prefix: str | os.PathLike, test_prefix: str | os.PathLike
):
	test_ls = os.listdir(test_prefix)
	correct_ls = os.listdir(correct_prefix)
	for module in ("function", "method", "trigger", "prereq", "action"):
		pyfilename = module + ".py"
		if pyfilename in test_ls:
			assert pyfilename in correct_ls, (
				f"{pyfilename} is in test data, but shouldn't be"
			)
			with (
				open(os.path.join(test_prefix, pyfilename), "rt") as test_py,
				open(os.path.join(correct_prefix, pyfilename)) as good_py,
			):
				test_parsed = parse(test_py.read())
				correct_parsed = parse(good_py.read())
			assert unparse(correct_parsed) == unparse(test_parsed), (
				f"{pyfilename} has incorrect Python code"
			)
		else:
			assert pyfilename not in correct_ls, (
				f"{pyfilename} should be in test data, but isn't"
			)


@pytest.fixture
def pqdb_connector_under_test(tmp_path, engine_facade):
	test_world = os.path.join(tmp_path, "testworld")
	connector = ParquetDatabaseConnector(test_world)
	(connector.pack, connector.unpack) = (
		engine_facade.pack,
		engine_facade.unpack,
	)
	yield connector
	connector.close()


@pytest.fixture
def pqdb_connector_correct(tmp_path, engine_facade):
	correct_world = os.path.join(tmp_path, "world")
	connector = ParquetDatabaseConnector(correct_world)
	(connector.pack, connector.unpack) = (
		engine_facade.pack,
		engine_facade.unpack,
	)
	yield connector
	connector.close()


@pytest.mark.parquetdb
def test_import_parquetdb(
	tmp_path,
	engine_and_exported_xml,
	pqdb_connector_under_test,
	pqdb_connector_correct,
):
	_, xml = engine_and_exported_xml
	pqdb_connector_under_test.load_xml(xml)
	compare_engines_world_state(
		pqdb_connector_correct, pqdb_connector_under_test
	)


@pytest.fixture
def sql_connector_under_test(tmp_path, engine_facade):
	test_world = os.path.join(tmp_path, "testworld.sqlite3")
	connector = SQLAlchemyDatabaseConnector(
		"sqlite:///" + test_world,
	)
	(connector.pack, connector.unpack) = (
		engine_facade.pack,
		engine_facade.unpack,
	)
	yield connector
	connector.close()


@pytest.fixture
def sql_connector_correct(tmp_path, engine_facade):
	correct_world = os.path.join(tmp_path, "world.sqlite3")
	connector = SQLAlchemyDatabaseConnector(
		"sqlite:///" + correct_world,
	)
	(connector.pack, connector.unpack) = (
		engine_facade.pack,
		engine_facade.unpack,
	)
	yield connector
	connector.close()


@pytest.mark.sqlite
def test_import_sqlite(
	tmp_path,
	engine_and_exported_xml,
	sql_connector_correct,
	sql_connector_under_test,
):
	_, xml = engine_and_exported_xml
	sql_connector_under_test.load_xml(xml)
	compare_engines_world_state(
		sql_connector_correct, sql_connector_under_test
	)
