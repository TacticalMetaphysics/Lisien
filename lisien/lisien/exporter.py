import os
from io import IOBase
from collections import deque
from pathlib import Path
from types import FunctionType, MethodType
from typing import Literal
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
	GraphRowType,
)
from lisien.util import AbstractThing, AbstractEngine


def sqlite_to_etree(
	sqlite_path: str | os.PathLike,
	tree: ElementTree | None = None,
	engine: AbstractEngine | None = None,
) -> ElementTree:
	from .db import SQLAlchemyQueryEngine

	if not isinstance(sqlite_path, os.PathLike):
		sqlite_path = Path(sqlite_path)

	if engine is None:
		engine = EngineFacade(None)
	query = SQLAlchemyQueryEngine(
		"sqlite:///" + str(os.path.abspath(sqlite_path)),
		{},
		pack=engine.pack,
		unpack=engine.unpack,
	)
	if tree is None:
		tree = ElementTree(Element("lisien"))
	return query_engine_to_tree(
		str(os.path.basename(os.path.dirname(sqlite_path))),
		query,
		tree,
	)


def pqdb_to_etree(
	pqdb_path: str | os.PathLike,
	tree: ElementTree | None = None,
	engine: AbstractEngine | None = None,
) -> ElementTree:
	from .db import ParquetQueryEngine

	if not isinstance(pqdb_path, os.PathLike):
		pqdb_path = Path(pqdb_path)

	if engine is None:
		engine = EngineFacade(None)
	query = ParquetQueryEngine(
		pqdb_path, pack=engine.pack, unpack=engine.unpack
	)
	if tree is None:
		tree = ElementTree(Element("lisien"))
	return query_engine_to_tree(
		str(os.path.basename(os.path.dirname(pqdb_path))), query, tree
	)


def game_path_to_etree(game_path: str | os.PathLike) -> ElementTree:
	world = os.path.join(game_path, "world")
	if os.path.isdir(world):
		game_history = pqdb_to_etree(world)
	else:
		game_history = sqlite_to_etree(world + ".sqlite3")
	return game_history


def game_path_to_xml(
	game_path: str | os.PathLike,
	xml_file_path: str | os.PathLike | IOBase,
	indent: bool = True,
) -> None:
	if not isinstance(game_path, os.PathLike):
		game_path = Path(game_path)
	if not isinstance(xml_file_path, os.PathLike) and not isinstance(
		xml_file_path, IOBase
	):
		xml_file_path = Path(xml_file_path)

	tree = game_path_to_etree(game_path)
	if indent:
		from xml.etree.ElementTree import indent

		indent(tree)
	tree.write(xml_file_path, encoding="utf-8")


def value_to_xml(value: Value | dict[Key, Value]) -> Element:
	if value is ...:
		return Element("Ellipsis")
	elif value is None:
		return Element("None")
	elif isinstance(value, bool):
		return Element("bool", value="T" if value else "F")
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
	elif isinstance(value, lisien.graph.Node):
		return Element(
			"node", character=repr(value.graph.name), name=repr(value.name)
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
			dict_item = Element("dict-item", key=repr(k))
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
	universal_d: dict[Key, Value] = keyframe.get("universal", {})
	univel = value_to_xml(universal_d)
	univel.tag = "universal"
	kfel.append(univel)
	triggers_kf: dict[RuleName, list[TriggerFuncName]] = keyframe.get(
		"triggers", {}
	)
	prereqs_kf: dict[RuleName, list[PrereqFuncName]] = keyframe.get(
		"prereqs", {}
	)
	actions_kf: dict[RuleName, list[ActionFuncName]] = keyframe.get(
		"actions", {}
	)
	neighborhoods_kf: dict[RuleName, RuleNeighborhood] = keyframe.get(
		"neighborhood", {}
	)
	bigs_kf: dict[RuleName, RuleBig] = keyframe.get("big", {})
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
		if rule_name in neighborhoods_kf:
			neighborhood = neighborhoods_kf[rule_name]
			rule_el.set(
				"neighborhood",
				"" if neighborhood is None else str(neighborhood),
			)
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
	] = keyframe.get("rulebook", {})
	for rulebook_name, (rule_list, priority) in rulebook_kf.items():
		rulebook_el = Element(
			"rulebook", name=repr(rulebook_name), priority=repr(priority)
		)
		kfel.append(rulebook_el)
		for rule_name in rule_list:
			rulebook_el.append(Element("rule", name=rule_name))
	char_els: dict[CharName, Element] = {}
	graph_val_kf: GraphValKeyframe = keyframe.get("graph_val", {})
	for char_name, vals in graph_val_kf.items():
		graph_el = char_els[char_name] = Element(
			"character", name=repr(char_name)
		)
		for k, v in vals.items():
			item_el = Element("dict_item", key=repr(k))
			graph_el.append(item_el)
			item_el.append(value_to_xml(v))
	node_val_kf: GraphNodeValKeyframe = keyframe.get("node_val", {})
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
	edge_val_kf: GraphEdgeValKeyframe = keyframe.get("edge_val", {})
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
	turn_ends: dict[Turn, tuple[Tick, Tick]],
	data: dict[
		Literal[
			"universals",
			"rulebooks",
			"rule_triggers",
			"rule_prereqs",
			"rule_actions",
			"rule_neighborhood",
			"rule_big",
			"graphs",
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
		rb_el = Element(
			"rulebook",
			name=repr(rulebook),
			priority=repr(prio),
			branch=b,
			turn=str(r),
			tick=str(t),
		)
		branch_el.append(rb_el)
		for i, rule in enumerate(rules):
			rb_el.append(Element("rule_placement", rule=rule, idx=str(i)))

	def append_rule_el(
		trig_rec: TriggerRowType = ...,
		preq_rec: PrereqRowType = ...,
		act_rec: ActionRowType = ...,
		nbr_rec: RuleNeighborhoodRowType = ...,
		big_rec: RuleBigRowType = ...,
	):
		if trig_rec is not ...:
			trigs = trig_rec[-1]
		else:
			trigs = None
		if preq_rec is not ...:
			preqs = preq_rec[-1]
		else:
			preqs = None
		if act_rec is not ...:
			acts = act_rec[-1]
		else:
			acts = None
		if nbr_rec is not ...:
			nbr = nbr_rec[-1]
		else:
			nbr = ...
		if big_rec is not ...:
			big = big_rec[-1]
		else:
			big = ...
		for reck in (trig_rec, preq_rec, act_rec, nbr_rec, big_rec):
			if reck is not ...:
				rule = reck[0]
				break
		else:
			raise ValueError("Appending rule element without rule data?")
		rule_el = Element(
			"rule",
			name=rule,
			branch=branch,
			turn=str(turn),
			tick=str(tick),
		)
		if big is not ...:
			rule_el.set("big", "T" if big else "F")
		branch_el.append(rule_el)
		if nbr is not ...:
			rule_el.set("neighborhood", "" if nbr is None else str(nbr))
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

	def append_graph_el(graph: GraphRowType):
		char, b, r, t, typ_str = graph
		graph_el = Element(
			"graph",
			character=repr(char),
			branch=b,
			turn=str(r),
			tick=str(t),
			type=typ_str,
		)
		branch_el.append(graph_el)

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
				"node_rulebook",
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
				orig=repr(orig),
				dest=repr(dest),
				branch=b,
				turn=str(r),
				tick=str(t),
				rulebook=repr(rb),
			)
		)

	turn: Turn
	turns_el = Element("turns", branch=branch)
	branch_el.append(turns_el)
	for turn, (ending_tick, plan_ending_tick) in turn_ends.items():
		turns_el.append(
			Element(
				"turn",
				branch=branch,
				turn=str(turn),
				end_tick=str(ending_tick),
				plan_end_tick=str(plan_ending_tick),
			)
		)
	for turn in range(turn_from, turn_to + 1):
		tick: Tick
		for tick in range(turn_ends[turn][1] + 1):
			if (branch, turn, tick) in keyframe_times:
				kf = query.get_keyframe(branch, turn, tick)
				add_keyframe_to_branch_el(branch_el, branch, turn, tick, kf)
				keyframe_times.remove((branch, turn, tick))
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
			if data["graphs"]:
				graph_rec: GraphRowType = data["graphs"][0]
				graph, branch_now, turn_now, tick_now, typ_str = graph_rec
				if (branch_now, turn_now, tick_now) == (branch, turn, tick):
					append_graph_el(graph_rec)
					del data["graphs"][0]
			rule_kwargs = {}
			if (
				"rule_triggers" in data
				and data["rule_triggers"]
				and data["rule_triggers"][0][1:4]
				== (
					branch,
					turn,
					tick,
				)
			):
				trig_rec: TriggerRowType = data["rule_triggers"][0]
				rule_kwargs["trig_rec"] = trig_rec
			if (
				"rule_prereqs" in data
				and data["rule_prereqs"]
				and data["rule_prereqs"][0][1:4]
				== (
					branch,
					turn,
					tick,
				)
			):
				preq_rec: PrereqRowType = data["rule_prereqs"][0]
				rule_kwargs["preq_rec"] = preq_rec
			if (
				"rule_actions" in data
				and data["rule_actions"]
				and data["rule_actions"][0][1:4]
				== (
					branch,
					turn,
					tick,
				)
			):
				act_rec: ActionRowType = data["rule_actions"][0]
				rule_kwargs["act_rec"] = act_rec
			if (
				"rule_neighborhood" in data
				and data["rule_neighborhood"]
				and data["rule_neighborhood"][0][1:4] == (branch, turn, tick)
			):
				nbr_rec: RuleNeighborhoodRowType = data["rule_neighborhood"][0]
				rule_kwargs["nbr_rec"] = nbr_rec
			if (
				"rule_big" in data
				and data["rule_big"]
				and data["rule_big"][0][1:4]
				== (
					branch,
					turn,
					tick,
				)
			):
				big_rec: RuleBigRowType = data["rule_big"][0]
				rule_kwargs["big_rec"] = big_rec
			if rule_kwargs:
				append_rule_el(**rule_kwargs)
			if "trig_rec" in rule_kwargs:
				del data["rule_triggers"][0]
			if "preq_rec" in rule_kwargs:
				del data["rule_prereqs"][0]
			if "act_rec" in rule_kwargs:
				del data["rule_actions"][0]
			if "nbr_rec" in rule_kwargs:
				del data["rule_neighborhood"][0]
			if "big_rec" in rule_kwargs:
				del data["rule_big"][0]
			for char_name in data.keys() - {
				"graphs",
				"universals",
				"rulebooks",
				"rule_triggers",
				"rule_prereqs",
				"rule_actions",
				"rule_neighborhoods",
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
				if char_data["node_rulebook"]:
					node_rb_row: NodeRulebookRowType = char_data[
						"node_rulebook"
					][0]
					_, __, b, r, t, ___ = node_rb_row
					if (b, r, t) == (branch, turn, tick):
						append_node_rb_el(node_rb_row)
						del char_data["node_rulebook"][0]
				if char_data["portal_rulebook"]:
					port_rb_row: PortalRulebookRowType = char_data[
						"portal_rulebook"
					][0]
					_, __, ___, b, r, t, ____ = port_rb_row
					if (b, r, t) == (branch, turn, tick):
						append_portal_rb_el(port_rb_row)
						del char_data["portal_rulebook"][0]
	assert not keyframe_times, keyframe_times


def query_engine_to_tree(
	name: str, query: AbstractQueryEngine, tree: ElementTree
) -> ElementTree:
	root = tree.getroot()
	for k, v in query.eternal.items():
		el = Element("dict-item", key=repr(k))
		root.append(el)
		el.append(value_to_xml(v))
	trunks = set()
	branches_d = {}
	branch_descendants = {}
	turn_end_plan_d: dict[Branch, dict[Turn, tuple[Tick, Tick]]] = {}
	branch_elements = {}
	playtrees: dict[Branch, Element] = {}
	keyframe_times: set[Time] = set(query.keyframes_dump())
	for branch, turn, last_real_tick, last_planned_tick in query.turns_dump():
		if branch in turn_end_plan_d:
			turn_end_plan_d[branch][turn] = (last_real_tick, last_planned_tick)
		else:
			turn_end_plan_d[branch] = {
				turn: (last_real_tick, last_planned_tick)
			}
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
				name=branch,
				start_turn="0",
				start_tick="0",
				end_turn=str(end_turn),
				end_tick=str(end_tick),
			)
			root.append(playtree)
			playtree.append(branch_element)
		else:
			if parent in branch_descendants:
				branch_descendants[parent].add(branch)
			else:
				branch_descendants[parent] = {branch}
			if parent in branch_elements:
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

	def recurse_branch(b: Branch):
		parent, turn_from, tick_from, turn_to, tick_to = branches_d[b]
		data = query.load_windows(
			[(b, turn_from, tick_from, turn_to, tick_to)]
		)
		fill_branch_element(
			query,
			turn_from,
			turn_to,
			turn_end_plan_d[b],
			data,
			branch_elements[b],
			keyframe_times,
			b,
		)
		if b in branch_descendants:
			for desc in sorted(branch_descendants[b], key=branches_d.get):
				recurse_branch(desc)

	for trunk in trunks:
		recurse_branch(trunk)

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
