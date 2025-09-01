import os
from collections import defaultdict, deque
from pathlib import Path
from types import FunctionType, MethodType
from xml.etree.ElementTree import ElementTree, Element

import networkx as nx
from tblib import Traceback

import lisien.graph
from lisien.db import AbstractQueryEngine
from lisien.facade import EngineFacade
from lisien.types import (
	Keyframe,
	Value,
	Key,
	RuleName,
	TriggerFuncName,
	PrereqFuncName,
	ActionFuncName,
	RuleNeighborhood,
	RuleBig,
	RulebookName,
	RulebookKeyframe,
	RulebookPriority,
	GraphNodeValKeyframe,
	CharName,
	GraphEdgeValKeyframe,
	GraphValKeyframe,
)
from lisien.util import AbstractThing


def sqlite_to_etree(
	sqlite_path: str | os.PathLike, tree: ElementTree | None = None
) -> ElementTree:
	from .db import SQLAlchemyQueryEngine

	if not isinstance(sqlite_path, os.PathLike):
		sqlite_path = Path(sqlite_path)

	eng = EngineFacade(None)
	query = SQLAlchemyQueryEngine(
		"sqlite:///" + str(os.path.abspath(sqlite_path)),
		{},
		pack=eng.pack,
		unpack=eng.unpack,
	)
	if tree is None:
		tree = ElementTree(Element("lisien"))
	return query_engine_to_tree(
		str(os.path.basename(os.path.dirname(sqlite_path))),
		query,
		tree,
	)


def game_path_to_etree(game_path: str | os.PathLike) -> ElementTree:
	game_history = sqlite_to_etree(os.path.join(game_path, "world.sqlite3"))
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


def value_to_xml(value: Value | dict[Key, Value]) -> Element:
	if value is ...:
		return Element("...")
	elif value is None:
		return Element("None")
	elif isinstance(value, int):
		return Element("int", value=str(value))
	elif isinstance(value, float):
		return Element("float", value=str(value))
	elif isinstance(value, str):
		return Element("str", value=value)
	elif isinstance(value, lisien.graph.DiGraph):
		# Since entity names are restricted to what we can use for dict
		# keys and also serialize to msgpack, I don't think there's any name
		# an entity can have that can't be repr'd
		return Element("character", name=repr(value.name))
	elif isinstance(value, AbstractThing):
		return Element(
			"thing",
			character=repr(value.character.name),
			name=repr(value.name),
			location=repr(value.location.name),
		)
	elif isinstance(value, lisien.graph.Node):
		return Element(
			"place", character=repr(value.graph.name), name=repr(value.name)
		)
	elif isinstance(value, lisien.graph.Edge):
		return Element(
			"portal",
			character=repr(value.graph.name),
			origin=repr(value.orig),
			destination=repr(value.dest),
		)
	elif isinstance(value, nx.Graph):
		return nx.readwrite.GraphMLWriter(value).myElement
	elif isinstance(value, FunctionType) or isinstance(value, MethodType):
		if value.__module__ not in (
			"trigger",
			"prereq",
			"action",
			"function",
			"method",
		):
			raise ValueError(
				"Callable is not stored in the Lisien engine", value
			)
		return Element(value.__module__, name=value.__name__)
	elif isinstance(value, Exception):
		# weird but ok
		el = Element("exception", pyclass=value.__class__.__name__)
		if hasattr(value, "__traceback__"):
			el.set("traceback", str(Traceback(value.__traceback__)))
		for arg in value.args:
			el.append(value_to_xml(arg))
		return el
	elif isinstance(value, list):
		el = Element("list")
		for v in value:
			el.append(value_to_xml(v))
		return el
	elif isinstance(value, tuple):
		el = Element("tuple")
		for v in value:
			el.append(value_to_xml(v))
		return el
	elif isinstance(value, set):
		el = Element("set")
		for v in value:
			el.append(value_to_xml(v))
		return el
	elif isinstance(value, frozenset):
		el = Element("frozenset")
		for v in value:
			el.append(value_to_xml(v))
		return el
	elif isinstance(value, dict):
		el = Element("dict")
		for k, v in value.items():
			dict_item = Element("dict_item", key=repr(k))
			dict_item.append(value_to_xml(v))
			el.append(dict_item)
		return el
	else:
		raise TypeError("Can't convert to XML", value)


def query_engine_to_tree(
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

	def make_keyframe_dict() -> Keyframe:
		return {
			"universal": {},
			"triggers": {},
			"prereqs": {},
			"actions": {},
			"rulebook": {},
			"big": {},
			"neighborhood": {},
			"node_val": {},
			"edge_val": {},
			"graph_val": {},
		}

	keyframes = defaultdict(make_keyframe_dict)
	for (
		branch,
		turn,
		tick,
		universal,
		rule,
		rulebook,
	) in query.get_all_keyframe_extensions_ever():
		kf = keyframes[branch, turn, tick]
		kf["universal"] = universal
		kf["rulebook"] = rulebook
		for rule, d in rule.items():
			kf["triggers"][rule] = d.get("triggers", [])
			kf["prereqs"][rule] = d.get("prereqs", [])
			kf["actions"][rule] = d.get("actions", [])
			kf["neighborhood"][rule] = d.get("neighborhood", None)
			kf["big"][rule] = d.get("big", False)
	for (
		char_name,
		branch,
		turn,
		tick,
		nodes,
		edges,
		graph_val,
	) in query.get_all_keyframe_graphs_ever():
		kf = keyframes[branch, turn, tick]
		kf["node_val"][char_name] = nodes
		kf["edge_val"][char_name] = edges
		kf["graph_val"][char_name] = graph_val
	for (branch, turn, tick), keyframe in keyframes.items():
		if not (branch in branches_d and branch in branch_elements):
			raise RuntimeError("Keyframe in invalid branch", branch)
		kfel = Element("keyframe", turn=str(turn), tick=str(tick))
		branch_elements[branch].append(kfel)
		universal_d: dict[Key, Value] = keyframe["universal"]
		univel = value_to_xml(universal_d)
		univel.tag = "universal"
		kfel.append(univel)
		triggers_kf: dict[RuleName, list[TriggerFuncName]] = keyframe[
			"triggers"
		]
		prereqs_kf: dict[RuleName, list[PrereqFuncName]] = keyframe["prereqs"]
		actions_kf: dict[RuleName, list[ActionFuncName]] = keyframe["actions"]
		neighborhoods_kf: dict[RuleName, RuleNeighborhood] = keyframe[
			"neighborhood"
		]
		bigs_kf: dict[RuleName, RuleBig] = keyframe["big"]
		for rule_name in (
			triggers_kf.keys() | prereqs_kf.keys() | actions_kf.keys()
		):
			rule_name: RuleName
			rule_el = Element(
				"rule",
				name=rule_name,
				big="T" if bigs_kf.get(rule_name) else "F",
			)
			kfel.append(rule_el)
			if neighborhood := neighborhoods_kf.get(rule_name) is not None:
				rule_el.set("neighborhood", neighborhood)
			if trigs := triggers_kf.get(rule_name):
				for trig in trigs:
					rule_el.append(Element("trigger", name=trig))
			if preqs := prereqs_kf.get(rule_name):
				for preq in preqs:
					rule_el.append(Element("prereq", name=preq))
			if acts := actions_kf.get(rule_name):
				for act in acts:
					rule_el.append(Element("action", name=act))
		rulebook_kf: dict[
			RulebookName, tuple[list[RuleName], RulebookPriority]
		] = keyframe["rulebook"]
		for rulebook_name, (rule_list, priority) in rulebook_kf.items():
			rulebook_el = Element(
				"rulebook", name=repr(rulebook_name), priority=repr(priority)
			)
			kfel.append(rulebook_el)
			for rule_name in rule_list:
				rulebook_el.append(Element("rule", name=rule_name))
		char_els: dict[CharName, Element] = {}
		graph_val_kf: GraphValKeyframe = keyframe["graph_val"]
		for char_name, vals in graph_val_kf.items():
			char_el = char_els[char_name] = Element(
				"character", name=repr(char_name)
			)

		node_val_kf: GraphNodeValKeyframe = keyframe["node_val"]
		for char_name, node_vals in node_val_kf.items():
			if char_name in char_els:
				char_el = char_els[char_name]
			else:
				char_el = char_els[char_name] = Element(
					"character", name=repr(char_name)
				)
				kfel.append(char_el)
			for node, val in node_vals.items():
				node_el = Element("node", name=repr(node))
				char_el.append(node_el)
				for k, v in val.items():
					item_el = Element("dict_item", key=repr(k))
					node_el.append(item_el)
					item_el.append(value_to_xml(v))
		edge_val_kf: GraphEdgeValKeyframe = keyframe["edge_val"]
		for char_name, edge_vals in edge_val_kf.items():
			if char_name in char_els:
				char_el = char_els[char_name]
			else:
				char_el = char_els[char_name] = Element(
					"character", name=repr(char_name)
				)
				kfel.append(char_el)
			for orig, dests in edge_vals.items():
				for dest, val in dests.items():
					edge_el = Element("edge", orig=repr(orig), dest=repr(dest))
					char_el.append(edge_el)
					for k, v in val.items():
						item_el = Element("dict_item", key=repr(k))
						edge_el.append(item_el)
						item_el.append(value_to_xml(v))
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
