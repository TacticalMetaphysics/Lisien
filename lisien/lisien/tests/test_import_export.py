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
from ..facade import EngineFacade
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
	prefix = tmp_path.joinpath("game")
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
		yield eng, DATA_DIR.joinpath(request.param + f"_{turns}.xml")


def test_export_db(tmp_path, engine_and_exported_xml):
	test_xml = tmp_path.joinpath("test.xml")
	eng, outpath = engine_and_exported_xml
	eng.to_xml(test_xml, name="test_export")

	if not filecmp.cmp(outpath, test_xml):
		with (
			test_xml.open("rt") as testfile,
			outpath.open("rt") as goodfile,
		):
			differences = list(
				difflib.unified_diff(
					goodfile.readlines(),
					testfile.readlines(),
					test_xml.name,
					outpath.name,
				)
			)
		assert filecmp.cmp(outpath.absolute(), test_xml.absolute()), "".join(
			differences
		)


@pytest.fixture(params=["kobold", "polygons", "wolfsheep"])
def exported(tmp_path, random_seed, request, turns):
	install = get_install_func(request.param, random_seed)
	with Engine(
		tmp_path,
		workers=0,
		random_seed=random_seed,
		keyframe_on_close=False,
		database=PythonDatabaseConnector,
	) as eng:
		install(eng)
		for _ in range(turns):
			eng.next_turn()
		archive_name = eng.export(
			request.param, tmp_path.joinpath(request.param + ".lisien")
		)
		old_data = eng.database.dump_everything()
	yield archive_name, old_data


def test_round_trip(
	tmp_path,
	exported,
	database_connector_part,
	random_seed,
):
	prefix2 = tmp_path.joinpath("game2")
	prefix2.mkdir(parents=True, exist_ok=True)
	archive_name, old_data = exported
	db2 = database_connector_part
	with (
		Engine(
			tmp_path,
			workers=0,
			database=partial(PythonDatabaseConnector, old_data=old_data),
			keyframe_on_close=False,
		) as eng1,
		Engine.from_archive(
			archive_name,
			prefix2,
			workers=0,
			database=db2,
			keyframe_on_close=False,
		) as eng2,
	):
		compare_engines_world_state(eng1, eng2)

	compare_stored_strings(prefix2, tmp_path)
	compare_stored_python_code(prefix2, tmp_path)


def compare_engines_world_state(
	correct_engine: Engine | AbstractDatabaseConnector,
	test_engine: Engine | AbstractDatabaseConnector,
):
	test_engine.commit()
	correct_engine.commit()
	test_engine = getattr(test_engine, "database", test_engine)
	correct_engine = getattr(correct_engine, "database", test_engine)
	test_dump = test_engine.dump_everything()
	correct_dump = correct_engine.dump_everything()
	assert test_dump.keys() == correct_dump.keys()
	# PythonDatabaseConnector doesn't really serialize its data, meaning
	# it sorts differently. So, for testing, use our own serializer.
	fake = EngineFacade(None, mock=True)
	for k, test_data in test_dump.items():
		if k.endswith("rules_handled"):
			continue
		correct_data = correct_dump[k]
		print(k)
		test_data.sort(key=fake.pack)
		correct_data.sort(key=fake.pack)
		assert correct_data == test_data, f"{k} tables differ"


def compare_stored_strings(
	correct_prefix: os.PathLike[str], test_prefix: os.PathLike[str]
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
	correct_prefix: os.PathLike[str], test_prefix: os.PathLike[str]
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
	test_world = tmp_path.joinpath("test_world")
	test_world.mkdir(parents=True)
	connector = ParquetDatabaseConnector(engine_facade, path=test_world)
	yield connector
	connector.close()


@pytest.fixture
def pqdb_connector_correct(tmp_path, engine_facade):
	correct_world = tmp_path.joinpath("world")
	correct_world.mkdir(parents=True)
	connector = ParquetDatabaseConnector(engine_facade, path=correct_world)
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
		engine_facade,
		"sqlite:///" + test_world,
	)
	yield connector
	connector.close()


@pytest.fixture
def sql_connector_correct(tmp_path, engine_facade):
	correct_world = os.path.join(tmp_path, "world.sqlite3")
	connector = SQLAlchemyDatabaseConnector(
		engine_facade,
		"sqlite:///" + correct_world,
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
