import filecmp
import os
from functools import partial

import pytest

from ..db import ParquetDatabaseConnector, SQLAlchemyDatabaseConnector
from ..engine import Engine
from ..exporter import game_path_to_xml
from ..importer import xml_to_sqlite, xml_to_pqdb
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
	with Engine(
		tmp_path,
		workers=0,
		random_seed=random_seed,
		connect_string=f"sqlite:///{tmp_path}/world.sqlite3"
		if non_null_database == "sqlite"
		else None,
		keyframe_on_close=False,
	) as eng:
		install(eng)
		for _ in range(1):
			eng.next_turn()
	yield path


def test_export(tmp_path, exported):
	test_xml = os.path.join(tmp_path, "test.xml")
	game_path_to_xml(tmp_path, test_xml, name="test_export")

	assert filecmp.cmp(test_xml, exported)


def test_import(tmp_path, exported, non_null_database, engine_facade):
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

	for dump_method in (
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
	):
		test_data = sorted(getattr(test_engine, dump_method)())
		correct_data = sorted(getattr(correct_engine, dump_method)())
		assert test_data == correct_data, (
			dump_method + " gave different results"
		)
