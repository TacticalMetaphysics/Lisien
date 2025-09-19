import difflib
import filecmp
import json
import os
from ast import parse
from functools import partial

import pytest
from astunparse import unparse

from ..db import (
	AbstractDatabaseConnector,
	ParquetDatabaseConnector,
	SQLAlchemyDatabaseConnector,
)
from ..engine import Engine
from ..exporter import game_path_to_xml
from ..importer import xml_to_pqdb, xml_to_sqlite
from .data import DATA_DIR


@pytest.fixture(params=["kobold", "polygons", "wolfsheep"])
def exported(tmp_path, random_seed, non_null_database, request):
	if request.param == "kobold":
		from lisien.examples.kobold import inittest as install

		path = os.path.join(DATA_DIR, "kobold.xml")
	elif request.param == "polygons":
		from lisien.examples.polygons import install

		path = os.path.join(DATA_DIR, "polygons.xml")
	elif request.param == "wolfsheep":
		from lisien.examples.wolfsheep import install

		install = partial(install, seed=random_seed)

		path = os.path.join(DATA_DIR, "wolfsheep.xml")
	else:
		raise ValueError("Unknown sim", request.param)
	prefix = os.path.join(tmp_path, "game")
	with Engine(
		prefix,
		workers=0,
		random_seed=random_seed,
		connect_string=f"sqlite:///{prefix}/world.sqlite3"
		if non_null_database == "sqlite"
		else None,
		keyframe_on_close=False,
	) as eng:
		install(eng)
		for _ in range(1):
			eng.next_turn()
	yield path


def test_export_db(tmp_path, exported):
	test_xml = os.path.join(tmp_path, "test.xml")
	game_path_to_xml(
		os.path.join(tmp_path, "game"), test_xml, name="test_export"
	)

	if not filecmp.cmp(test_xml, exported):
		with (
			open(test_xml, "rt") as testfile,
			open(exported, "rt") as goodfile,
		):
			differences = list(
				difflib.unified_diff(
					testfile.readlines(),
					goodfile.readlines(),
					test_xml,
					exported,
				)
			)
			assert not differences, "".join(differences)


DUMP_METHOD_NAMES = (
	"global_dump",
	"turns_completed_dump",
	"universals_dump",
	"rulebooks_dump",
	"rules_dump",
	"rule_triggers_dump",
	"rule_prereqs_dump",
	"rule_actions_dump",
	"rule_neighborhood_dump",
	"rule_big_dump",
	"node_rulebook_dump",
	"portal_rulebook_dump",
	"nodes_dump",
	"edges_dump",
	"things_dump",
	"units_dump",
	"node_val_dump",
	"edge_val_dump",
	"graph_val_dump",
	"keyframes_graphs_dump",
	"keyframe_extensions_dump",
)


def compare_engines_world_state(
	test_engine: Engine | AbstractDatabaseConnector,
	correct_engine: Engine | AbstractDatabaseConnector,
):
	test_engine = getattr(test_engine, "query", test_engine)
	correct_engine = getattr(correct_engine, "query", test_engine)
	for dump_method in DUMP_METHOD_NAMES:
		test_data = list(getattr(test_engine, dump_method)())
		correct_data = list(getattr(correct_engine, dump_method)())
		print(dump_method)
		assert test_data == correct_data, (
			dump_method + " gave different results"
		)


def compare_stored_strings(
	test_prefix: str | os.PathLike, correct_prefix: str | os.PathLike
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
			assert json.load(test_file) == json.load(correct_file), (
				f"Different strings for language: {lang[:-5]}"
			)


def compare_stored_python_code(
	test_prefix: str | os.PathLike, correct_prefix: str | os.PathLike
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
			assert unparse(test_parsed) == unparse(correct_parsed), (
				f"{pyfilename} has incorrect Python code"
			)
		else:
			assert pyfilename not in correct_ls, (
				f"{pyfilename} should be in test data, but isn't"
			)


def test_export_import_engine(tmp_path, non_null_database, exported):
	prefix = os.path.join(tmp_path, "game")
	prefix2 = os.path.join(tmp_path, "game2")
	connect_string = (
		f"sqlite:///{prefix}/world.sqlite3"
		if non_null_database == "sqlite"
		else None
	)
	with Engine(
		prefix,
		workers=0,
		connect_string=connect_string,
		keyframe_on_close=False,
	) as correct_engine:
		correct_engine.export("test")
		os.mkdir(prefix2)
		with Engine.from_archive(
			"test.lisien",
			prefix2,
			workers=0,
			connect_string=connect_string,
			keyframe_on_close=False,
		) as test_engine:
			compare_engines_world_state(test_engine, correct_engine)
	compare_stored_strings(prefix2, prefix)
	compare_stored_python_code(prefix2, prefix)


def test_import_db(tmp_path, exported, non_null_database, engine_facade):
	if non_null_database == "parquetdb":
		test_world = os.path.join(tmp_path, "testworld")
		correct_world = os.path.join(tmp_path, "world")
		xml_to_pqdb(exported, test_world)
		test_engine = ParquetDatabaseConnector(
			test_world, engine_facade.pack, engine_facade.unpack
		)
		correct_engine = ParquetDatabaseConnector(
			correct_world, engine_facade.pack, engine_facade.unpack
		)
	else:
		test_world = os.path.join(tmp_path, "testworld.sqlite3")
		correct_world = os.path.join(tmp_path, "world.sqlite3")
		xml_to_sqlite(exported, test_world)
		test_engine = SQLAlchemyDatabaseConnector(
			"sqlite:///" + test_world,
			{},
			engine_facade.pack,
			engine_facade.unpack,
		)
		correct_engine = SQLAlchemyDatabaseConnector(
			"sqlite:///" + correct_world,
			{},
			engine_facade.pack,
			engine_facade.unpack,
		)

	compare_engines_world_state(test_engine, correct_engine)
