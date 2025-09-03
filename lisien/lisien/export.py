import os
from collections import defaultdict, deque
from pathlib import Path
from types import FunctionType, MethodType
from typing import Literal, TypeAlias
from xml.etree.ElementTree import ElementTree, Element

import networkx as nx
from tblib import Traceback

import lisien.graph
from lisien.db import AbstractQueryEngine, LoadedCharWindow
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
	RulebookPriority,
	GraphNodeValKeyframe,
	CharName,
	GraphEdgeValKeyframe,
	GraphValKeyframe,
	Turn,
	Tick,
	Branch,
	Stat,
	UniversalRowType,
	RulebookRowType,
	Time,
	RuleRowType,
	TriggerRowType,
	PrereqRowType,
	ActionRowType,
	RuleNeighborhoodRowType,
	RuleBigRowType,
	NodeRowType,
	NodeValRowType,
	EdgeRowType,
	EdgeValRowType,
	GraphValRowType,
	ThingRowType,
	CharRulebookRowType,
	NodeRulebookRowType,
	PortalRulebookRowType,
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
		return Element("Ellipsis")
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


def add_keyframe_to_branch_el(
	branch_el: Element,
	branch: Branch,
	turn: Turn,
	tick: Tick,
	keyframe: Keyframe,
) -> None:
	kfel = Element("keyframe", branch=branch, turn=str(turn), tick=str(tick))
	branch_el.append(kfel)
	universal_d: dict[Key, Value] = keyframe["universal"]
	univel = value_to_xml(universal_d)
	univel.tag = "universal"
	kfel.append(univel)
	triggers_kf: dict[RuleName, list[TriggerFuncName]] = keyframe["triggers"]
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


def fill_branch_element(
	query: AbstractQueryEngine,
	turn_from: Turn,
	turn_to: Turn,
	turn_ends: dict[Turn, Tick],
	data: dict[
		Literal[
			"universals",
			"rulebooks",
			"rule_triggers",
			"rule_prereqs",
			"rule_actions",
			"rule_neighborhood",
			"rule_big",
		]
		| CharName,
		list[UniversalRowType]
		| list[RulebookRowType]
		| list[RuleRowType]
		| LoadedCharWindow,
	],
	branch_el: Element,
	keyframe_times: set[Time],
	branch: Branch,
):
	def append_univ_el(universal_rec: UniversalRowType):
		key, b, r, t, val = universal_rec
		univ_el = Element(
			"universal",
			key=repr(key),
			branch=b,
			turn=str(r),
			tick=str(t),
		)
		univ_el.append(value_to_xml(val))
		branch_el.append(univ_el)

	def append_rulebook_el(rulebook_rec: RulebookRowType):
		rb, b, r, t, (rules, prio) = rulebook_rec
		rb_el = Element("rulebook", name=repr(rulebook), priority=repr(prio))
		branch_el.append(rb_el)
		for i, rule in enumerate(rules):
			rb_el.append(Element("rule_placement", rule=rule, idx=str(i)))

	def append_rule_el(
		trig_rec: TriggerRowType,
		preq_rec: PrereqRowType,
		act_rec: ActionRowType,
		nbr_rec: RuleNeighborhoodRowType,
		big_rec: RuleBigRowType,
	):
		assert (
			trig_rec[0]
			== preq_rec[0]
			== act_rec[0]
			== nbr_rec[0]
			== big_rec[0]
		)
		assert (
			trig_rec[1:4]
			== preq_rec[1:4]
			== act_rec[1:4]
			== nbr_rec[1:4]
			== big_rec[1:4]
			== (branch, turn, tick)
		)
		trigs = trig_rec[-1]
		preqs = preq_rec[-1]
		acts = act_rec[-1]
		nbr = nbr_rec[-1]
		big = big_rec[-1]
		rule_el = Element(
			"rule",
			name=rule,
			branch=branch,
			turn=str(turn),
			tick=str(tick),
			big="T" if big else "F",
		)
		branch_el.append(rule_el)
		if nbr is not None:
			rule_el.set("neighborhood", str(nbr))
		if trigs:
			trigs_el = Element("triggers")
			rule_el.append(trigs_el)
			for trig in trigs:
				trigs_el.append(Element("trigger", name=trig))
		if preqs:
			preqs_el = Element("prereqs")
			rule_el.append(preqs_el)
			for preq in preqs:
				preqs_el.append(Element("prereq", name=preq))
		if acts:
			acts_el = Element("actions")
			rule_el.append(acts_el)
			for act in acts:
				acts_el.append(Element("action", name=act))

	def append_graph_val_el(graph_val: GraphValRowType):
		char, stat, b, r, t, val = graph_val
		graph_val_el = Element(
			"graph_val",
			character=repr(char),
			key=repr(stat),
			branch=b,
			turn=str(r),
			tick=str(t),
		)
		branch_el.append(graph_val_el)
		graph_val_el.append(value_to_xml(val))

	def append_nodes_el(nodes: NodeRowType):
		char, node, b, r, t, ex = nodes
		branch_el.append(
			Element(
				"node",
				character=repr(char),
				name=repr(node),
				branch=b,
				turn=str(r),
				tick=str(t),
				exists="T" if ex else "F",
			)
		)

	def append_node_val_el(node_val: NodeValRowType):
		char, node, stat, b, r, t, val = node_val
		node_val_el = Element(
			"node_val",
			character=repr(char),
			node=repr(node),
			key=repr(stat),
			branch=b,
			turn=str(r),
			tick=str(t),
		)
		branch_el.append(node_val_el)
		node_val_el.append(value_to_xml(val))

	def append_edges_el(edges: EdgeRowType):
		char, orig, dest, b, r, t, ex = edges
		branch_el.append(
			Element(
				"edge",
				character=repr(char),
				orig=repr(orig),
				dest=repr(dest),
				branch=b,
				turn=str(r),
				tick=str(t),
				exists="T" if ex else "F",
			)
		)

	def append_edge_val_el(edge_val: EdgeValRowType):
		char, orig, dest, stat, b, r, t, val = edge_val
		edge_val_el = Element(
			"edge_val",
			character=repr(char),
			orig=repr(orig),
			dest=repr(dest),
			key=repr(stat),
			branch=b,
			turn=str(r),
			tick=str(t),
		)
		branch_el.append(edge_val_el)
		edge_val_el.append(value_to_xml(val))

	def append_thing_el(thing: ThingRowType):
		char, thing, b, r, t, loc = thing
		branch_el.append(
			Element(
				"location",
				character=repr(char),
				thing=repr(thing),
				branch=b,
				turn=str(r),
				tick=str(t),
				location=repr(loc),
			)
		)

	def append_char_rb_el(rbtyp: str, rbrow: CharRulebookRowType):
		char, b, r, t, rb = rbrow
		branch_el.append(
			Element(
				rbtyp,
				character=repr(char),
				branch=b,
				turn=str(r),
				tick=str(t),
				rulebook=repr(rb),
			)
		)

	def append_node_rb_el(nrb_row: NodeRulebookRowType):
		char, node, b, r, t, rb = nrb_row
		branch_el.append(
			Element(
				"node_rulebok",
				character=repr(char),
				node=repr(node),
				branch=b,
				turn=str(r),
				tick=str(t),
				rulebook=repr(rb),
			)
		)

	def append_portal_rb_el(port_rb_row: PortalRulebookRowType):
		char, orig, dest, b, r, t, rb = port_rb_row
		branch_el.append(
			Element(
				"portal_rulebook",
				character=repr(char),
				node=repr(node),
				branch=b,
				turn=str(r),
				tick=str(t),
				rulebook=repr(rb),
			)
		)

	turn: Turn
	for turn in range(turn_from, turn_to + 1):
		tick: Tick
		for tick in range(turn_ends[turn]):
			if (branch, turn, tick) in keyframe_times:
				kf = query.get_keyframe(branch, turn, tick)
				add_keyframe_to_branch_el(branch_el, branch, turn, tick, kf)
			if data["universals"]:
				universal_rec: UniversalRowType = data["universals"][0]
				key, branch_now, turn_now, tick_now, val = universal_rec
				if (branch_now, turn_now, tick_now) != (branch, turn, tick):
					append_univ_el(universal_rec)
					del data["universals"][0]
			if data["rulebooks"]:
				rulebook_rec: RulebookRowType = data["rulebooks"][0]
				rulebook, branch_now, turn_now, tick_now, (rules, prio) = (
					rulebook_rec
				)
				if (branch_now, turn_now, tick_now) == (branch, turn, tick):
					append_rulebook_el(rulebook_rec)
					del data["rulebooks"][0]
			if data["rule_triggers"]:
				trig_rec: TriggerRowType = data["rule_triggers"][0]
				rule, branch_now, turn_now, tick_now, trigs = trig_rec
				if (branch_now, turn_now, tick_now) == (branch, turn, tick):
					preq_rec: PrereqRowType = data["rule_prereqs"][0]
					act_rec: ActionRowType = data["rule_actions"][0]
					nbr_rec: RuleNeighborhoodRowType = data[
						"rule_neighborhood"
					][0]
					big_rec: RuleBigRowType = data["rule_big"][0]
					append_rule_el(
						trig_rec, preq_rec, act_rec, nbr_rec, big_rec
					)
					del data["rule_triggers"][0]
					del data["rule_prereqs"][0]
					del data["rule_actions"][0]
					del data["rule_neighborhood"][0]
					del data["rule_big"][0]
			for char_name in data.keys() - {
				"universals",
				"rulebooks",
				"rule_triggers",
				"rule_prereqs",
				"rule_actions",
				"rule_neighborhood",
				"rule_big",
			}:
				char_data: LoadedCharWindow = data[char_name]
				if char_data["graph_val"]:
					graph_val_row: GraphValRowType = char_data["graph_val"][0]
					_, __, b, r, t, ___ = graph_val_row
					if (b, r, t) == (branch, turn, tick):
						append_graph_val_el(graph_val_row)
						del char_data["graph_val"][0]
				if char_data["nodes"]:
					nodes_row: NodeRowType = char_data["nodes"][0]
					char, node, branch_now, turn_now, tick_now, ex = nodes_row
					if (branch_now, turn_now, tick_now) == (
						branch,
						turn,
						tick,
					):
						append_nodes_el(nodes_row)
						del char_data["nodes"][0]
				if char_data["node_val"]:
					node_val_row: NodeValRowType = char_data["node_val"][0]
					char, node, stat, branch_now, turn_now, tick_now, val = (
						node_val_row
					)
					if (branch_now, turn_now, tick_now) == (
						branch,
						turn,
						tick,
					):
						append_node_val_el(node_val_row)
				if char_data["edges"]:
					edges_row: EdgeRowType = char_data["edges"][0]
					_, __, ___, b, r, t, ____ = edges_row
					if (b, r, t) == (branch, turn, tick):
						append_edges_el(edges_row)
						del char_data["edges"][0]
				if char_data["edge_val"]:
					edge_val_row: EdgeValRowType = char_data["edge_val"][0]
					_, __, ___, ____, b, r, t, _____ = edge_val_row
					if (b, r, t) == (branch, turn, tick):
						append_edge_val_el(edge_val_row)
						del char_data["edge_val"][0]
				if char_data["things"]:
					thing_row: ThingRowType = char_data["things"][0]
					_, __, b, r, t, ___ = thing_row
					if (b, r, t) == (branch, turn, tick):
						append_thing_el(thing_row)
						del char_data["things"][0]
				for char_rb_typ in (
					"character_rulebook",
					"unit_rulebook",
					"character_thing_rulebook",
					"character_place_rulebook",
					"character_portal_rulebook",
				):
					if char_data[char_rb_typ]:
						char_rb_row: CharRulebookRowType = char_data[
							char_rb_typ
						][0]
						_, b, r, t, __ = char_rb_row
						if (b, r, t) == (branch, turn, tick):
							append_char_rb_el(char_rb_typ, char_rb_row)
							del char_data[char_rb_typ][0]
				if "node_rulebook" in char_data:
					node_rb_row: NodeRulebookRowType = char_data[
						"node_rulebook"
					][0]
					_, __, b, r, t, ___ = node_rb_row
					if (b, r, t) == (branch, turn, tick):
						append_node_rb_el(node_rb_row)
						del char_data["node_rulebook"][0]
				if "portal_rulebook" in char_data:
					port_rb_row: PortalRulebookRowType = char_data[
						"portal_rulebook"
					][0]
					_, __, ___, b, r, t, ____ = port_rb_row
					if (b, r, t) == (branch, turn, tick):
						append_portal_rb_el(port_rb_row)
						del char_data["portal_rulebook"][0]


def query_engine_to_tree(
	name: str, query: AbstractQueryEngine, tree: ElementTree
) -> ElementTree:
	root = tree.getroot()
	trunks = set()
	branches_d = {}
	turn_end_plan_d = {}
	branch_elements = {}
	playtrees = {}
	for branch, turn, last_real_tick, last_planned_tick in query.turns_dump():
		if branch in turn_end_plan_d:
			turn_end_plan_d[branch][turn] = last_planned_tick
		else:
			turn_end_plan_d[branch] = {turn: last_planned_tick}
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
		add_keyframe_to_branch_el(
			branch_elements[branch], turn, tick, keyframe
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
