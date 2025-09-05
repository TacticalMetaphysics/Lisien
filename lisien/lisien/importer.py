import os
import sys
from ast import literal_eval
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

	def xml_to_value(self, el: Element) -> Value:
		eng = self.engine
		match el.tag:
			case "Ellipsis":
				return ...
			case "None":
				return Value(None)
			case "int":
				return Value(int(el.get("value")))
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
				return Value([self.xml_to_value(listel) for listel in el])
			case "tuple":
				return Value(tuple(self.xml_to_value(tupel) for tupel in el))
			case "set":
				return Value({self.xml_to_value(setel) for setel in el})
			case "frozenset":
				return Value(
					frozenset(self.xml_to_value(setel) for setel in el)
				)
			case "dict":
				ret = {}
				for dict_item_el in el:
					ret[literal_eval(dict_item_el.get("key"))] = (
						self.xml_to_value(dict_item_el[0])
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

	def keyframe(self, el: Element):
		branch = Branch(el.get("branch"))
		turn = Turn(int(el.get("turn")))
		tick = Tick(int(el.get("tick")))
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
		for subel in el:
			if subel.tag == "universal":
				for univel in subel:
					k = literal_eval(univel.get("key"))
					v = self.xml_to_value(univel[0])
					universal_kf[k] = v
			elif subel.tag == "rule":
				rule = RuleName(el.get("name"))
				bigs_kf[rule] = RuleBig(el.get("big") == "T")
				neighborhoods_kf[rule] = RuleNeighborhood(
					el.get("neighborhood") or None
				)
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
				name = CharName(name)
				graph_vals = graph_val_kf[name] = {}
				node_vals = node_val_kf[name] = {}
				edge_vals = edge_val_kf[name] = {}
				for key_el in subel:
					if key_el.tag == "dict_item":
						key = literal_eval(key_el.get("key"))
						graph_vals[key] = self.xml_to_value(key_el[0])
					elif key_el.tag == "node":
						char = literal_eval(key_el.get("character"))
						name = literal_eval(key_el.get("name"))
						nv = node_vals.setdefault(char, {})
						if name in nv:
							val = nv[name]
						else:
							val = nv[name] = {}
						for item_el in key_el:
							val[literal_eval(item_el.get("key"))] = (
								self.xml_to_value(item_el[0])
							)
					elif key_el.tag == "edge":
						char = literal_eval(key_el.get("character"))
						orig = literal_eval(key_el.get("orig"))
						dest = literal_eval(key_el.get("dest"))
						ev = edge_vals.setdefault(char, {})
						if orig not in ev:
							ev[orig] = {dest: {}}
						if dest not in ev[orig]:
							ev[orig][dest] = {}
						val = ev[orig][dest]
						for item_el in key_el:
							val[literal_eval(item_el.get("key"))] = (
								self.xml_to_value(item_el[0])
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

	def universal(self, el: Element):
		branch = Branch(el.get("branch"))
		turn = Turn(int(el.get("turn")))
		tick = Tick(int(el.get("tick")))
		key = UniversalKey(literal_eval(el.get("key")))
		value = self.xml_to_value(el[0])
		self.query.universal_set(key, branch, turn, tick, value)

	def rulebook(self, el: Element):
		branch = Branch(el.get("branch"))
		turn = Turn(int(el.get("turn")))
		tick = Tick(int(el.get("tick")))
		rulebook = RulebookName(literal_eval(el.get("name")))
		priority = RulebookPriority(float(el.get("priority")))
		rules: list[RuleName] = []
		for subel in el:
			if subel.tag != "rule_placement":
				raise ValueError("Don't know what to do with tag", subel.tag)
			rules.append(RuleName(subel.get("rule")))
		self.query.set_rulebook(rulebook, branch, turn, tick, rules, priority)

	@staticmethod
	def _get_time(el: Element) -> Time:
		return (
			Branch(el.get("branch")),
			Turn(int(el.get("turn"))),
			Tick(int(el.get("tick"))),
		)

	def rule(self, el: Element):
		branch, turn, tick = self._get_time(el)
		rule = RuleName(el.get("name"))
		triggers = []
		prereqs = []
		actions = []
		neighborhood = el.get("neighborhood") or None
		big = RuleBig(el.get("big") == "T")
		for subel in el:
			if subel.tag == "triggers":
				for trigel in subel:
					triggers.append(trigel.get("name"))
			elif subel.tag == "prereqs":
				for preqel in subel:
					prereqs.append(preqel.get("name"))
			elif subel.tag == "actions":
				for actel in subel:
					actions.append(actel.get("name"))
			else:
				raise ValueError("Don't know what to do with tag", subel.tag)
		self.query.set_rule(
			rule,
			branch,
			turn,
			tick,
			triggers,
			prereqs,
			actions,
			neighborhood,
			big,
		)

	def graph_val(self, el: Element):
		branch, turn, tick = self._get_time(el)
		graph = CharName(literal_eval(el.get("character")))
		key = Stat(literal_eval(el.get("key")))
		value = self.xml_to_value(el[0])
		self.query.graph_val_set(graph, key, branch, turn, tick, value)

	def node(self, el: Element):
		branch, turn, tick = self._get_time(el)
		char = CharName(literal_eval(el.get("character")))
		node = NodeName(literal_eval(el.get("name")))
		ex = el.get("exists") == "T"
		self.query.exist_node(char, node, branch, turn, tick, ex)

	def node_val(self, el: Element):
		branch, turn, tick = self._get_time(el)
		char = CharName(literal_eval(el.get("character")))
		node = NodeName(literal_eval(el.get("node")))
		key = Stat(literal_eval(el.get("key")))
		val = self.xml_to_value(el[0])
		self.query.node_val_set(char, node, key, branch, turn, tick, val)

	def edge(self, el: Element):
		branch, turn, tick = self._get_time(el)
		char = CharName(literal_eval(el.get("character")))
		orig = NodeName(literal_eval(el.get("orig")))
		dest = NodeName(literal_eval(el.get("dest")))
		ex = el.get("exists") == "T"
		self.query.exist_edge(char, orig, dest, branch, turn, tick, ex)

	def edge_val(self, el: Element):
		branch, turn, tick = self._get_time(el)
		char = CharName(literal_eval(el.get("character")))
		orig = NodeName(literal_eval(el.get("orig")))
		dest = NodeName(literal_eval(el.get("dest")))
		key = Stat(literal_eval(el.get("key")))
		val = self.xml_to_value(el[0])
		self.query.edge_val_set(char, orig, dest, key, branch, turn, tick, val)

	def location(self, el: Element):
		branch, turn, tick = self._get_time(el)
		char = CharName(literal_eval(el.get("character")))
		thing = NodeName(literal_eval(el.get("thing")))
		location = NodeName(literal_eval(el.get("location")))
		self.query.set_thing_loc(char, thing, branch, turn, tick, location)

	def _some_character_rulebook(self, rbtyp: str, el: Element):
		assert el.tag == rbtyp
		meth = getattr(self.query, f"set_{rbtyp}")
		branch, turn, tick = self._get_time(el)
		char = CharName(literal_eval(el.get("character")))
		rb = RulebookName(literal_eval(el.get("rulebook")))
		meth(char, branch, turn, tick, rb)

	def character_rulebook(self, el: Element):
		self._some_character_rulebook("character_rulebook", el)

	def unit_rulebook(self, el: Element):
		self._some_character_rulebook("unit_rulebook", el)

	def character_thing_rulebook(self, el: Element):
		self._some_character_rulebook("character_thing_rulebook", el)

	def character_place_rulebook(self, el: Element):
		self._some_character_rulebook("character_place_rulebook", el)

	def character_portal_rulebook(self, el: Element):
		self._some_character_rulebook("character_portal_rulebook", el)

	def node_rulebook(self, el: Element):
		branch, turn, tick = self._get_time(el)
		char = CharName(literal_eval(el.get("character")))
		node = NodeName(literal_eval(el.get("node")))
		rb = RulebookName(literal_eval(el.get("rulebook")))
		self.query.set_node_rulebook(char, node, branch, turn, tick, rb)

	def portal_rulebook(self, el: Element):
		branch, turn, tick = self._get_time(el)
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
			for branch in el:
				parent: Branch | None = branch.get("parent")
				name = Branch(branch.get("name"))
				start_turn = Turn(int(branch.get("start_turn")))
				start_tick = Tick(int(branch.get("start_tick")))
				end_turn = Turn(int(branch.get("end_turn")))
				end_tick = Tick(int(branch.get("end_tick")))
				query.set_branch(
					name, parent, start_turn, start_tick, end_turn, end_tick
				)

				for elem in branch:
					getattr(importer, elem.tag)(elem)
		else:
			k = literal_eval(el.get("key"))
			v = importer.xml_to_value(el[0])
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
