import os
import sys
from ast import literal_eval
from functools import partialmethod
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, parse

from lisien.types import (
	RuleName,
	PrereqFuncName,
	TriggerFuncName,
	ActionFuncName,
	RuleNeighborhood,
	Branch,
	Turn,
	Tick,
	RuleBig,
	FuncName,
	RulebookPriority,
	RulebookName,
	Key,
	GraphValKeyframe,
	GraphNodeValKeyframe,
	GraphEdgeValKeyframe,
	UniversalKey,
	Value,
	CharName,
	NodeName,
	RuleKeyframe,
	UniversalKeyframe,
	Time,
	Stat,
)
from .db import AbstractQueryEngine
from .facade import EngineFacade
from .util import AbstractEngine


class Importer:
	def __init__(
		self, query: AbstractQueryEngine, engine: AbstractEngine | None = None
	):
		if engine is None:
			engine = EngineFacade(None)
		self.query = query
		self.engine = engine
		self.known_rules: set[RuleName] = set()

	def element_to_value(self, el: Element) -> Value:
		eng = self.engine
		match el.tag:
			case "Ellipsis":
				return ...
			case "None":
				return Value(None)
			case "int":
				return Value(int(el.get("value")))
			case "float":
				return Value(float(el.get("value")))
			case "str":
				return Value(el.get("value"))
			case "bool":
				return Value(el.get("value") == "T")
			case "character":
				name = CharName(literal_eval(el.get("name")))
				return eng.character[name]
			case "node":
				char_name = CharName(literal_eval(el.get("character")))
				place_name = NodeName(literal_eval(el.get("name")))
				return eng.character[char_name].node[place_name]
			case "portal":
				char_name = CharName(literal_eval(el.get("character")))
				orig = NodeName(literal_eval(el.get("origin")))
				dest = NodeName(literal_eval(el.get("destination")))
				return eng.character[char_name].portal[orig][dest]
			case "list":
				return Value([self.element_to_value(listel) for listel in el])
			case "tuple":
				return Value(
					tuple(self.element_to_value(tupel) for tupel in el)
				)
			case "set":
				return Value({self.element_to_value(setel) for setel in el})
			case "frozenset":
				return Value(
					frozenset(self.element_to_value(setel) for setel in el)
				)
			case "dict":
				ret = {}
				for dict_item_el in el:
					ret[literal_eval(dict_item_el.get("key"))] = (
						self.element_to_value(dict_item_el[0])
					)
				return Value(ret)
			case "exception":
				raise NotImplementedError(
					"Deserializing exceptions from XML not implemented"
				)
			case s if s in {
				"trigger",
				"prereq",
				"action",
				"function",
				"method",
			}:
				return getattr(getattr(eng, s), el.get("name"))
			case default:
				raise ValueError("Can't deserialize the element", default)

	@staticmethod
	def _get_time(branch_el: Element, turn_el: Element, el: Element) -> Time:
		ret = (
			Branch(branch_el.get("name")),
			Turn(int(turn_el.get("number"))),
			Tick(int(el.get("tick"))),
		)
		if not isinstance(ret[0], str):
			raise TypeError("nonstring branch", ret[0])
		return ret

	def keyframe(self, branch_el: Element, turn_el: Element, kf_el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, kf_el)
		self.query.keyframe_insert(branch, turn, tick)
		universal_kf: UniversalKeyframe = {}
		triggers_kf: dict[RuleName, list[TriggerFuncName]] = {}
		prereqs_kf: dict[RuleName, list[PrereqFuncName]] = {}
		actions_kf: dict[RuleName, list[ActionFuncName]] = {}
		neighborhoods_kf: dict[RuleName, RuleNeighborhood] = {}
		bigs_kf: dict[RuleName, RuleBig] = {}
		rule_kf: RuleKeyframe = {
			"triggers": triggers_kf,
			"prereqs": prereqs_kf,
			"actions": actions_kf,
			"neighborhood": neighborhoods_kf,
			"big": bigs_kf,
		}
		rulebook_kf: dict[
			RulebookName, tuple[list[RuleName], RulebookPriority]
		] = {}
		graph_val_kf: GraphValKeyframe = {}
		node_val_kf: GraphNodeValKeyframe = {}
		edge_val_kf: GraphEdgeValKeyframe = {}
		for subel in kf_el:
			if subel.tag == "universal":
				for univel in subel:
					k = literal_eval(univel.get("key"))
					v = self.element_to_value(univel[0])
					universal_kf[k] = v
			elif subel.tag == "rule":
				rule = RuleName(subel.get("name"))
				if rule is None:
					raise TypeError("Rules need names")
				bigs_kf[rule] = RuleBig(kf_el.get("big") == "T")
				neighborhoods_kf[rule] = kf_el.get("neighborhood") or None
				for funcl_el in subel:
					name = FuncName(funcl_el.get("name"))
					if not isinstance(name, str):
						raise TypeError("Function name must be str", name)
					if funcl_el.tag == "trigger":
						if rule in triggers_kf:
							triggers_kf[rule].append(TriggerFuncName(name))
						else:
							triggers_kf[rule] = [TriggerFuncName(name)]
					elif funcl_el.tag == "prereq":
						if rule in prereqs_kf:
							prereqs_kf[rule].append(PrereqFuncName(name))
						else:
							prereqs_kf[rule] = [PrereqFuncName(name)]
					elif funcl_el.tag == "action":
						if rule in actions_kf:
							actions_kf[rule].append(ActionFuncName(name))
						else:
							actions_kf[rule] = [ActionFuncName(name)]
					else:
						raise ValueError("Unknown rule tag", funcl_el.tag)
			elif subel.tag == "rulebook":
				name = subel.get("name")
				if name is None:
					raise TypeError("rulebook tag missing name")
				name = literal_eval(name)
				if not isinstance(name, Key):
					raise TypeError("Rulebook name must be Key", name)
				name = RulebookName(name)
				prio = subel.get("priority")
				if prio is None:
					raise TypeError("rulebook tag missing priority")
				prio = RulebookPriority(float(prio))
				rules: list[RuleName] = []
				for rule_el in subel:
					if rule_el.tag != "rule":
						raise ValueError("Expected a rule tag", rule_el.tag)
					rules.append(RuleName(rule_el.get("name")))
				rulebook_kf[name] = (rules, prio)
			elif subel.tag == "character":
				name = subel.get("name")
				if name is None:
					raise TypeError("character tag missing name")
				name = literal_eval(name)
				if not isinstance(name, Key):
					raise TypeError("character names must be Key", name)
				char_name = CharName(name)
				graph_vals = graph_val_kf[char_name] = {}
				node_vals = node_val_kf[char_name] = {}
				edge_vals = edge_val_kf[char_name] = {}
				for key_el in subel:
					if key_el.tag == "dict_item":
						key = literal_eval(key_el.get("key"))
						graph_vals[key] = self.element_to_value(key_el[0])
					elif key_el.tag == "node":
						name = literal_eval(key_el.get("name"))
						if name in node_vals:
							val = node_vals[name]
						else:
							val = node_vals[name] = {}
						for item_el in key_el:
							val[literal_eval(item_el.get("key"))] = (
								self.element_to_value(item_el[0])
							)
					elif key_el.tag == "edge":
						orig = literal_eval(key_el.get("orig"))
						dest = literal_eval(key_el.get("dest"))
						if orig not in edge_vals:
							edge_vals[orig] = {dest: {}}
						if dest not in edge_vals[orig]:
							edge_vals[orig][dest] = {}
						val = edge_vals[orig][dest]
						for item_el in key_el:
							val[literal_eval(item_el.get("key"))] = (
								self.element_to_value(item_el[0])
							)
					else:
						raise ValueError(
							"Don't know how to deal with tag", key_el.tag
						)
			else:
				raise ValueError("Don't know how to deal with tag", subel.tag)
		self.query.keyframe_extension_insert(
			branch, turn, tick, universal_kf, rule_kf, rulebook_kf
		)
		for graph in (
			graph_val_kf.keys() | node_val_kf.keys() | edge_val_kf.keys()
		):
			self.query.keyframe_graph_insert(
				graph,
				branch,
				turn,
				tick,
				node_val_kf.get(graph, {}),
				edge_val_kf.get(graph, {}),
				graph_val_kf.get(graph, {}),
			)

	def universal(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		key = UniversalKey(literal_eval(el.get("key")))
		value = self.element_to_value(el[0])
		self.query.universal_set(key, branch, turn, tick, value)

	def rulebook(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		rulebook = RulebookName(literal_eval(el.get("name")))
		priority = RulebookPriority(float(el.get("priority")))
		rules: list[RuleName] = []
		for subel in el:
			if subel.tag != "rule":
				raise ValueError("Don't know what to do with tag", subel.tag)
			rules.append(RuleName(subel.get("name")))
		self.query.set_rulebook(rulebook, branch, turn, tick, rules, priority)

	def _rule_func_list(
		self, what: str, branch_el: Element, turn_el: Element, el: Element
	):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		rule = RuleName(el.get("rule"))
		funcs = [FuncName(func_el.get("name")) for func_el in el]
		if rule not in self.known_rules:
			self.query.create_rule(rule)
		self.known_rules.add(rule)
		mth = getattr(self.query, f"set_rule_{what}")
		mth(rule, branch, turn, tick, funcs)

	rule_triggers = partialmethod(_rule_func_list, "triggers")
	rule_prereqs = partialmethod(_rule_func_list, "prereqs")
	rule_actions = partialmethod(_rule_func_list, "actions")

	def rule_neighborhood(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		nbrs = el.get("neighbors")
		if nbrs is not None:
			nbrs = int(nbrs)
		rule = RuleName(el.get("rule"))
		if rule not in self.known_rules:
			self.query.create_rule(rule)
			self.known_rules.add(rule)
		self.query.set_rule_neighborhood(rule, branch, turn, tick, nbrs)

	def rule_big(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		big = RuleBig(el.get("big") == "T")
		rule = RuleName(el.get("rule"))
		if rule not in self.known_rules:
			self.query.create_rule(rule)
			self.known_rules.add(rule)
		self.query.set_rule_big(rule, branch, turn, tick, big)

	def graph(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		graph = CharName(literal_eval(el.get("character")))
		typ_str = el.get("type")
		self.query.graphs_insert(graph, branch, turn, tick, typ_str)

	def graph_val(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		graph = CharName(literal_eval(el.get("character")))
		key = Stat(literal_eval(el.get("key")))
		value = self.element_to_value(el[0])
		self.query.graph_val_set(graph, key, branch, turn, tick, value)

	def node(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		char = CharName(literal_eval(el.get("character")))
		node = NodeName(literal_eval(el.get("name")))
		ex = el.get("exists") == "T"
		self.query.exist_node(char, node, branch, turn, tick, ex)

	def node_val(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		char = CharName(literal_eval(el.get("character")))
		node = NodeName(literal_eval(el.get("node")))
		key = Stat(literal_eval(el.get("key")))
		val = self.element_to_value(el[0])
		self.query.node_val_set(char, node, key, branch, turn, tick, val)

	def edge(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		char = CharName(literal_eval(el.get("character")))
		orig = NodeName(literal_eval(el.get("orig")))
		dest = NodeName(literal_eval(el.get("dest")))
		ex = el.get("exists") == "T"
		self.query.exist_edge(char, orig, dest, branch, turn, tick, ex)

	def edge_val(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		char = CharName(literal_eval(el.get("character")))
		orig = NodeName(literal_eval(el.get("orig")))
		dest = NodeName(literal_eval(el.get("dest")))
		key = Stat(literal_eval(el.get("key")))
		val = self.element_to_value(el[0])
		self.query.edge_val_set(char, orig, dest, key, branch, turn, tick, val)

	def location(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		char = CharName(literal_eval(el.get("character")))
		thing = NodeName(literal_eval(el.get("thing")))
		location = NodeName(literal_eval(el.get("location")))
		self.query.set_thing_loc(char, thing, branch, turn, tick, location)

	def _some_character_rulebook(
		self, branch_el: Element, turn_el: Element, rbtyp: str, el: Element
	):
		assert el.tag == rbtyp
		meth = getattr(self.query, f"set_{rbtyp}")
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		char = CharName(literal_eval(el.get("character")))
		rb = RulebookName(literal_eval(el.get("rulebook")))
		meth(char, branch, turn, tick, rb)

	def character_rulebook(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		self._some_character_rulebook(
			branch_el, turn_el, "character_rulebook", el
		)

	def unit_rulebook(self, branch_el: Element, turn_el: Element, el: Element):
		self._some_character_rulebook(branch_el, turn_el, "unit_rulebook", el)

	def character_thing_rulebook(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		self._some_character_rulebook(
			branch_el, turn_el, "character_thing_rulebook", el
		)

	def character_place_rulebook(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		self._some_character_rulebook(
			branch_el, turn_el, "character_place_rulebook", el
		)

	def character_portal_rulebook(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		self._some_character_rulebook(
			branch_el, turn_el, "character_portal_rulebook", el
		)

	def node_rulebook(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		char = CharName(literal_eval(el.get("character")))
		node = NodeName(literal_eval(el.get("node")))
		rb = RulebookName(literal_eval(el.get("rulebook")))
		self.query.set_node_rulebook(char, node, branch, turn, tick, rb)

	def portal_rulebook(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		char = CharName(literal_eval(el.get("character")))
		orig = NodeName(literal_eval(el.get("orig")))
		dest = NodeName(literal_eval(el.get("dest")))
		rb = RulebookName(literal_eval(el.get("rulebook")))
		self.query.set_portal_rulebook(
			char, orig, dest, branch, turn, tick, rb
		)


def tree_to_db(
	tree: ElementTree,
	query: AbstractQueryEngine,
	engine: AbstractEngine | None = None,
):
	importer = Importer(query, engine)
	root = tree.getroot()
	for el in root:
		if el.tag == "playtree":
			for branch_el in el:
				parent: Branch | None = branch_el.get("parent")
				branch = Branch(branch_el.get("name"))
				start_turn = Turn(int(branch_el.get("start_turn")))
				start_tick = Tick(int(branch_el.get("start_tick")))
				end_turn = Turn(int(branch_el.get("end_turn")))
				end_tick = Tick(int(branch_el.get("end_tick")))
				query.set_branch(
					branch, parent, start_turn, start_tick, end_turn, end_tick
				)
				try:
					last_completed_turn = Turn(
						int(branch_el.get("last_turn_completed"))
					)
					query.complete_turn(branch, last_completed_turn, False)
				except KeyError:
					pass

				for turn_el in branch_el:
					turn = Turn(int(turn_el.get("number")))
					end_tick = Tick(int(turn_el.get("end_tick")))
					plan_end_tick = Tick(int(turn_el.get("plan_end_tick")))
					query.set_turn(branch, turn, end_tick, plan_end_tick)
					for elem in turn_el:
						getattr(importer, elem.tag.replace("-", "_"))(
							branch_el, turn_el, elem
						)
		else:
			k = literal_eval(el.get("key"))
			v = importer.element_to_value(el[0])
			query.eternal[k] = v
	query.commit()


def tree_to_sqlite(
	tree: ElementTree,
	sqlite_path: str | os.PathLike,
	engine: AbstractEngine | None = None,
):
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
	return tree_to_db(tree, query, engine)


def xml_to_sqlite(
	xml_path: str | os.PathLike,
	sqlite_path: str | os.PathLike,
	engine: AbstractEngine | None = None,
):
	if not isinstance(xml_path, os.PathLike):
		xml_path = Path(xml_path)

	tree = parse(xml_path)

	return tree_to_sqlite(tree, sqlite_path, engine)


def tree_to_pqdb(
	tree: ElementTree,
	pqdb_path: str | os.PathLike,
	engine: AbstractEngine | None = None,
):
	from .db import ParquetQueryEngine

	if not isinstance(pqdb_path, os.PathLike):
		pqdb_path = Path(pqdb_path)

	if engine is None:
		engine = EngineFacade(None)

	query = ParquetQueryEngine(
		pqdb_path, pack=engine.pack, unpack=engine.unpack
	)

	return tree_to_db(tree, query, engine)


def xml_to_pqdb(
	xml_path: str | os.PathLike,
	pqdb_path: str | os.PathLike,
	engine: AbstractEngine | None = None,
):
	if not isinstance(xml_path, os.PathLike):
		xml_path = Path(xml_path)

	tree = parse(xml_path)

	return tree_to_pqdb(tree, pqdb_path, engine)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("xml_path", type=str)
	parser.add_argument("-o", "--output", type=str, required=False)
	parser.add_argument("-f", "--format", type=str, default="parquet")
	parsed = parser.parse_args()
	if parsed.output:
		output_path = parsed.output
	else:
		output_path = (
			"world" if parsed.format == "parquet" else "world.sqlite3"
		)
	if parsed.format == "parquet":
		xml_to_pqdb(parsed.xml_path, output_path)
	elif parsed.format == "sqlite":
		xml_to_sqlite(parsed.xml_path, output_path)
	else:
		sys.exit(f"Unknown output format: {parsed.format}")
