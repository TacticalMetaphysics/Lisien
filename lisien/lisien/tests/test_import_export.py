import difflib
import filecmp
import json
import os
from ast import parse, unparse
from functools import partial

import pytest

from ..db import (
	AbstractDatabaseConnector,
	ParquetDatabaseConnector,
	SQLAlchemyDatabaseConnector,
)
from ..engine import Engine
from ..exporter import game_path_to_xml
from ..importer import xml_to_pqdb, xml_to_sqlite
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
def export_to(tmp_path, random_seed, non_null_database, request, turns):
	install = get_install_func(request.param, random_seed)
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
		for _ in range(turns):
			eng.next_turn()
	yield str(os.path.join(DATA_DIR, request.param + f"_{turns}.xml"))


def test_export_db(tmp_path, export_to):
	test_xml = os.path.join(tmp_path, "test.xml")
	game_path_to_xml(
		os.path.join(tmp_path, "game"), test_xml, name="test_export"
	)

	if not filecmp.cmp(export_to, test_xml):
		with (
			open(test_xml, "rt") as testfile,
			open(export_to, "rt") as goodfile,
		):
			differences = list(
				difflib.unified_diff(
					goodfile.readlines(),
					testfile.readlines(),
					test_xml,
					export_to,
				)
			)
			assert not differences, "".join(differences)


@pytest.fixture(params=["kobold", "polygons", "wolfsheep"])
def exported(tmp_path, random_seed, non_null_database, request, turns):
	install = get_install_func(request.param, random_seed)
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
		for _ in range(turns):
			eng.next_turn()
		archive_name = eng.export(request.param)
	yield archive_name


def test_round_trip(tmp_path, exported, non_null_database, random_seed, turns):
	prefix1 = os.path.join(tmp_path, "game")
	prefix2 = os.path.join(tmp_path, "game2")
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
			connect_string=f"sqlite:///{prefix1}/world.sqlite3"
			if non_null_database == "sqlite"
			else None,
			keyframe_on_close=False,
		) as eng1,
		Engine(
			prefix2,
			workers=0,
			connect_string=f"sqlite:///{prefix2}/world.sqlite3"
			if non_null_database == "sqlite"
			else None,
			keyframe_on_close=False,
			random_seed=random_seed,
		) as eng2,
	):
		install(eng2)
		for _ in range(turns):
			eng2.next_turn()
		compare_engines_world_state(eng1, eng2)

	compare_stored_strings(prefix2, prefix1)
	compare_stored_python_code(prefix2, prefix1)


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
	correct_engine: Engine | AbstractDatabaseConnector,
	test_engine: Engine | AbstractDatabaseConnector,
):
	test_engine.commit()
	correct_engine.commit()
	test_engine = getattr(test_engine, "query", test_engine)
	correct_engine = getattr(correct_engine, "query", test_engine)
	for dump_method in DUMP_METHOD_NAMES:
		test_data = list(getattr(test_engine, dump_method)())
		correct_data = list(getattr(correct_engine, dump_method)())
		print(dump_method)
		assert correct_data == test_data, (
			dump_method + " gave different results"
		)


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


def test_import_db(tmp_path, export_to, non_null_database, engine_facade):
	if non_null_database == "parquetdb":
		test_world = os.path.join(tmp_path, "testworld")
		correct_world = os.path.join(tmp_path, "world")
		xml_to_pqdb(export_to, test_world)
		test_engine = ParquetDatabaseConnector(
			test_world, engine_facade.pack, engine_facade.unpack
		)
		correct_engine = ParquetDatabaseConnector(
			correct_world, engine_facade.pack, engine_facade.unpack
		)
	else:
		test_world = os.path.join(tmp_path, "testworld.sqlite3")
		correct_world = os.path.join(tmp_path, "world.sqlite3")
		xml_to_sqlite(export_to, test_world)
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

	compare_engines_world_state(correct_engine, test_engine)
