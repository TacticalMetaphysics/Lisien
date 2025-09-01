import os
from collections import deque
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
	trunks = set()
	branches_d = {}
	branch_elements = {}
	playtrees = {}
	branch2do = deque(query.all_branches())
	while branch2do:
		(
			branch,
			parent,
			parent_turn,
			parent_tick,
			end_turn,
			end_tick,
		) = branch2do.popleft()
		branches_d[branch] = (
			parent,
			parent_turn,
			parent_tick,
			end_turn,
			end_tick,
		)
		if parent is None:
			trunks.add(branch)
			playtree = Element("playtree", game=name, trunk=branch)
			playtrees[branch] = playtree
			branch_element = branch_elements[branch] = Element(
				"branch",
				start_turn="0",
				start_tick="0",
				end_turn=str(end_turn),
				end_tick=str(end_tick),
			)
			root.append(playtree)
			playtree.append(branch_element)
		elif parent in branch_elements:
			branch_el = Element(
				"branch",
				parent=parent,
				start_turn=str(parent_turn),
				start_tick=str(parent_tick),
				end_turn=str(end_turn),
				end_tick=str(end_tick),
			)
			branch_elements[parent].append(branch_el)
		else:
			branch2do.append(
				(
					branch,
					parent,
					parent_turn,
					parent_tick,
					end_turn,
					end_tick,
				)
			)
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
