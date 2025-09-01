import os
from pathlib import Path
from xml.etree.ElementTree import ElementTree, Element

import msgpack

from lisien.db import AbstractQueryEngine


def sqlite_to_etree(
	sqlite_path: str | os.PathLike, tree: ElementTree | None = None
) -> ElementTree:
	from .db import SQLAlchemyQueryEngine

	if not isinstance(sqlite_path, os.PathLike):
		sqlite_path = Path(sqlite_path)

	query = SQLAlchemyQueryEngine(
		"sqlite:///" + str(os.path.abspath(sqlite_path)),
		{},
		pack=msgpack.packb,
		unpack=lambda x: x,
	)
	if tree is None:
		tree = ElementTree(Element("lisien"))
	return _query_engine_to_tree(
		str(os.path.basename(sqlite_path)).removesuffix(".sqlite3"),
		query,
		tree,
	)


def game_path_to_etree(game_path: str | os.PathLike) -> ElementTree:
	game_history = sqlite_to_etree(os.path.join(game_path, "world.sqlite3"))
	# add python files here?
	return game_history


def game_path_to_xml(
	game_path: str | os.PathLike, xml_file_path: str | os.PathLike
) -> None:
	if not isinstance(game_path, os.PathLike):
		game_path = Path(game_path)
	if not isinstance(xml_file_path, os.PathLike):
		xml_file_path = Path(xml_file_path)

	tree = game_path_to_etree(game_path)
	tree.write(xml_file_path)


def _query_engine_to_tree(
	name: str, query: AbstractQueryEngine, tree: ElementTree
) -> ElementTree:
	root = tree.getroot()
	for branch, turn, tick in query.main_branch_ends():
		playthru = Element(
			"playthru",
			game=name,
			branch=branch,
			turn_from="0",
			tick_from="0",
			turn_to=str(turn),
			tick_to=str(tick),
		)
		root.append(playthru)
	return tree


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("game_path", type=str)
	parser.add_argument("-o", "--output", type=str, required=False)
	parsed = parser.parse_args()
	game_name = os.path.basename(parsed.game_path)
	if parsed.output:
		output_path = parsed.output
	else:
		output_path = game_name.replace(" ", "_") + "_export.xml"
	game_path_to_xml(parsed.game_path, output_path)
