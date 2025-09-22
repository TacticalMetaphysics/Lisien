import os
import sys
from ast import literal_eval
from functools import partialmethod
from itertools import chain
from operator import itemgetter
from pathlib import Path
from typing import Any, Literal, TypeAlias

try:
	from lxml.etree import Element, ElementTree, parse
except ModuleNotFoundError:
	from xml.etree.ElementTree import Element, ElementTree, parse

from lisien.types import (
	ActionFuncName,
	Branch,
	CharName,
	FuncName,
	GraphEdgeValKeyframe,
	GraphNodeValKeyframe,
	GraphValKeyframe,
	Key,
	NodeName,
	PrereqFuncName,
	RuleBig,
	RulebookName,
	RulebookPriority,
	RuleFuncName,
	RuleKeyframe,
	RuleName,
	RuleNeighborhood,
	Stat,
	Tick,
	Time,
	TriggerFuncName,
	Turn,
	UniversalKey,
	UniversalKeyframe,
	Value,
)
from lisien.window import SettingsTurnDict

from .db import AbstractDatabaseConnector
from .facade import EngineFacade
from .util import AbstractEngine


class Importer:
	def __init__(
		self,
		query: AbstractDatabaseConnector,
		engine: AbstractEngine | None = None,
	):
		if engine is None:
			engine = EngineFacade(None)
		self.query = query
		self.engine = engine
		self.known_triggers: dict[
			RuleName,
			dict[
				Branch,
				SettingsTurnDict[Turn, dict[Tick, list[TriggerFuncName]]],
			],
		] = {}
		self.known_prereqs: dict[
			RuleName,
			dict[
				Branch,
				SettingsTurnDict[Turn, dict[Tick, list[PrereqFuncName]]],
			],
		] = {}
		self.known_actions: dict[
			RuleName,
			dict[
				Branch,
				SettingsTurnDict[Turn, dict[Tick, list[ActionFuncName]]],
			],
		] = {}
		self.known_neighborhoods: dict[
			RuleName,
			dict[Branch, SettingsTurnDict[Turn, dict[Tick, RuleNeighborhood]]],
		] = {}
		self.known_big: dict[
			RuleName, dict[Branch, SettingsTurnDict[Turn, dict[Tick, RuleBig]]]
		] = {}

	def _element_to_value(
		self, el: Element
	) -> (
		Value
		| list[Value]
		| tuple[Value, ...]
		| set[Key]
		| frozenset[Key]
		| dict[Key, Value]
	):
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
				return Value(el.get("value") in {"T", "true"})
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
				return [self._element_to_value(listel) for listel in el]
			case "tuple":
				return tuple(self._element_to_value(tupel) for tupel in el)
			case "set":
				return {self._element_to_value(setel) for setel in el}
			case "frozenset":
				return frozenset(self._element_to_value(setel) for setel in el)
			case "dict":
				ret = {}
				for dict_item_el in el:
					ret[literal_eval(dict_item_el.get("key"))] = (
						self._element_to_value(dict_item_el[0])
					)
				return ret
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

	def _keyframe(self, branch_el: Element, turn_el: Element, kf_el: Element):
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
					v = self._element_to_value(univel[0])
					universal_kf[k] = v
			elif subel.tag == "rule":
				rule = RuleName(subel.get("name"))
				if rule is None:
					raise TypeError("Rules need names")
				if "big" in subel.keys():
					bigs_kf[rule] = RuleBig(subel.get("big") in {"T", "true"})
				if "neighborhood" in subel.keys():
					neighborhoods_kf[rule] = int(subel.get("neighborhood"))
				else:
					neighborhoods_kf[rule] = None
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
				for k in (
					"character-rulebook",
					"unit-rulebook",
					"character-thing-rulebook",
					"character-place-rulebook",
					"character-portal-rulebook",
				):
					if k in subel.keys():
						graph_vals[k.replace("-", "_")] = literal_eval(
							subel.get(k)
						)
				node_vals = node_val_kf[char_name] = {}
				edge_vals = edge_val_kf[char_name] = {}
				for key_el in subel:
					if key_el.tag == "dict-item":
						key = literal_eval(key_el.get("key"))
						graph_vals[key] = self._element_to_value(key_el[0])
					elif key_el.tag == "node":
						name = literal_eval(key_el.get("name"))
						if name in node_vals:
							val = node_vals[name]
						else:
							val = node_vals[name] = {}
						if "rulebook" in key_el.keys():
							val["rulebook"] = literal_eval(
								key_el.get("rulebook")
							)
						for item_el in key_el:
							val[literal_eval(item_el.get("key"))] = (
								self._element_to_value(item_el[0])
							)
					elif key_el.tag == "edge":
						orig = literal_eval(key_el.get("orig"))
						dest = literal_eval(key_el.get("dest"))
						if orig not in edge_vals:
							edge_vals[orig] = {dest: {}}
						if dest not in edge_vals[orig]:
							edge_vals[orig][dest] = {}
						val = edge_vals[orig][dest]
						if "rulebook" in key_el.keys():
							val["rulebook"] = literal_eval(
								key_el.get("rulebook")
							)
						for item_el in key_el:
							val[literal_eval(item_el.get("key"))] = (
								self._element_to_value(item_el[0])
							)
					elif key_el.tag == "units":
						graph_vals["units"] = {}
						for unit_graph_el in key_el:
							unit_graph_name = literal_eval(
								unit_graph_el.get("character")
							)
							unit_graph_nodes_d = graph_vals["units"][
								unit_graph_name
							] = {}
							for unit_node_el in unit_graph_el:
								unit_graph_nodes_d[
									literal_eval(unit_node_el.get("node"))
								] = True
					else:
						raise ValueError(
							"Don't know how to deal with tag", key_el.tag
						)
			else:
				raise ValueError("Don't know how to deal with tag", subel.tag)
		self.query.keyframe_insert(branch, turn, tick)
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

	def _universal(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		key = UniversalKey(literal_eval(el.get("key")))
		value = self._element_to_value(el[0])
		self.query.universal_set(key, branch, turn, tick, value)

	def _rulebook(self, branch_el: Element, turn_el: Element, el: Element):
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
		self,
		what: Literal["triggers", "prereqs", "actions"],
		branch_el: Element,
		turn_el: Element,
		el: Element,
	):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		rule = RuleName(el.get("rule"))
		funcs: list[RuleFuncName] = [
			FuncName(func_el.get("name")) for func_el in el
		]
		self._memorize_rule(what, rule, branch, turn, tick, funcs)

	def _memorize_rule(
		self,
		what: Literal["triggers", "prereqs", "actions", "neighborhood", "big"],
		rule: RuleName,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		datum: list[TriggerFuncName]
		| list[PrereqFuncName]
		| list[ActionFuncName]
		| RuleNeighborhood
		| RuleBig,
	):
		if what == "triggers":
			d = self.known_triggers
		elif what == "prereqs":
			d = self.known_prereqs
		elif what == "actions":
			d = self.known_actions
		elif what == "neighborhood":
			d = self.known_neighborhoods
		elif what == "big":
			d = self.known_big
		else:
			raise ValueError(what)
		if rule in d:
			if branch in d[rule]:
				if turn in d[rule][branch]:
					d[rule][branch][turn][tick] = datum
				else:
					d[rule][branch][turn] = {tick: datum}
			else:
				d[rule][branch] = SettingsTurnDict({turn: {tick: datum}})
		else:
			d[rule] = {branch: SettingsTurnDict({turn: {tick: datum}})}

	_rule_triggers = partialmethod(_rule_func_list, "triggers")
	_rule_prereqs = partialmethod(_rule_func_list, "prereqs")
	_rule_actions = partialmethod(_rule_func_list, "actions")

	def _rule_neighborhood(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		rule = RuleName(el.get("rule"))
		neighborhood = el.get("neighbors")
		if neighborhood is not None:
			neighborhood = int(neighborhood)
		self._memorize_rule(
			"neighborhood",
			rule,
			branch,
			turn,
			tick,
			neighborhood,
		)

	def _rule_big(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		big = RuleBig(el.get("big") in {"T", "true"})
		rule = RuleName(el.get("rule"))
		self._memorize_rule("big", rule, branch, turn, tick, big)

	def _graph(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		graph = CharName(literal_eval(el.get("character")))
		typ_str = el.get("type")
		self.query.graphs_insert(graph, branch, turn, tick, typ_str)

	def _graph_val(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		graph = CharName(literal_eval(el.get("character")))
		key = Stat(literal_eval(el.get("key")))
		value = self._element_to_value(el[0])
		self.query.graph_val_set(graph, key, branch, turn, tick, value)

	def _node(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		char = CharName(literal_eval(el.get("character")))
		node = NodeName(literal_eval(el.get("name")))
		ex = el.get("exists") in {"T", "true"}
		self.query.exist_node(char, node, branch, turn, tick, ex)

	def _node_val(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		char = CharName(literal_eval(el.get("character")))
		node = NodeName(literal_eval(el.get("node")))
		key = Stat(literal_eval(el.get("key")))
		val = self._element_to_value(el[0])
		self.query.node_val_set(char, node, key, branch, turn, tick, val)

	def _edge(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		char = CharName(literal_eval(el.get("character")))
		orig = NodeName(literal_eval(el.get("orig")))
		dest = NodeName(literal_eval(el.get("dest")))
		ex = el.get("exists") in {"T", "true"}
		self.query.exist_edge(char, orig, dest, branch, turn, tick, ex)

	def _edge_val(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		char = CharName(literal_eval(el.get("character")))
		orig = NodeName(literal_eval(el.get("orig")))
		dest = NodeName(literal_eval(el.get("dest")))
		key = Stat(literal_eval(el.get("key")))
		val = self._element_to_value(el[0])
		self.query.edge_val_set(char, orig, dest, key, branch, turn, tick, val)

	def _location(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		char = CharName(literal_eval(el.get("character")))
		thing = NodeName(literal_eval(el.get("thing")))
		location = NodeName(literal_eval(el.get("location")))
		self.query.set_thing_loc(char, thing, branch, turn, tick, location)

	def _unit(self, branch_el: Element, turn_el: Element, el: Element):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		char = CharName(literal_eval(el.get("character-graph")))
		graph = CharName(literal_eval(el.get("unit-graph")))
		node = NodeName(literal_eval(el.get("unit-node")))
		self.query.unit_set(
			char,
			graph,
			node,
			branch,
			turn,
			tick,
			el.get("is-unit", "false") in {"T", "true"},
		)

	def _some_character_rulebook(
		self, branch_el: Element, turn_el: Element, rbtyp: str, el: Element
	):
		meth = getattr(self.query, f"set_{rbtyp}")
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		char = CharName(literal_eval(el.get("character")))
		rb = RulebookName(literal_eval(el.get("rulebook")))
		meth(char, branch, turn, tick, rb)

	def _character_rulebook(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		self._some_character_rulebook(
			branch_el, turn_el, "character_rulebook", el
		)

	def _unit_rulebook(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		self._some_character_rulebook(branch_el, turn_el, "unit_rulebook", el)

	def _character_thing_rulebook(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		self._some_character_rulebook(
			branch_el, turn_el, "character_thing_rulebook", el
		)

	def _character_place_rulebook(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		self._some_character_rulebook(
			branch_el, turn_el, "character_place_rulebook", el
		)

	def _character_portal_rulebook(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		self._some_character_rulebook(
			branch_el, turn_el, "character_portal_rulebook", el
		)

	def _node_rulebook(
		self, branch_el: Element, turn_el: Element, el: Element
	):
		branch, turn, tick = self._get_time(branch_el, turn_el, el)
		char = CharName(literal_eval(el.get("character")))
		node = NodeName(literal_eval(el.get("node")))
		rb = RulebookName(literal_eval(el.get("rulebook")))
		self.query.set_node_rulebook(char, node, branch, turn, tick, rb)

	def _portal_rulebook(
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

	@staticmethod
	def _iter_descendants(
		branch_descendants: dict[Branch, list[Branch]],
		branch: Branch = "trunk",
		stop=lambda obj: False,
		key=None,
	):
		branch_descendants[branch].sort(key=key)
		for desc in branch_descendants[branch]:
			yield desc
			if stop(desc):
				continue
			if desc in branch_descendants:
				yield from Importer._iter_descendants(branch_descendants, desc)

	def _create_rule(
		self, rule: RuleName, branch: Branch, turn: Turn, tick: Tick
	):
		kwargs = {}
		for mapping, kwarg in [
			(self.known_triggers, "triggers"),
			(self.known_prereqs, "prereqs"),
			(self.known_actions, "actions"),
			(self.known_neighborhoods, "neighborhood"),
			(self.known_big, "big"),
		]:
			if (
				rule in mapping
				and branch in mapping[rule]
				and turn in mapping[rule][branch]
				and tick in mapping[rule][branch][turn]
			):
				kwargs[kwarg] = mapping[rule][branch][turn].pop(tick)
		self.query.create_rule(rule, branch, turn, tick, **kwargs)

	def etree_to_db(
		self,
		tree: ElementTree,
	) -> None:
		root = tree.getroot()
		branch_descendants: dict[Branch, list[Branch]] = {Branch("trunk"): []}
		branch_starts: dict[Branch, tuple[Turn, Tick]] = {}
		if "_lisien_schema_version" in self.query.eternal:
			if self.query.eternal["_lisien_schema_version"] != int(
				root.get("db-schema-version")
			):
				raise RuntimeError("Incompatible database versions")
		else:
			self.query.eternal["_lisien_schema_version"] = int(
				root.get("db-schema-version")
			)
		self.query.eternal["trunk"] = root.get("trunk")
		self.query.eternal["branch"] = root.get("branch")
		self.query.eternal["turn"] = int(root.get("turn"))
		self.query.eternal["tick"] = int(root.get("tick"))
		for el in root:
			if el.tag == "language":
				continue
			if el.tag == "playtree":
				for branch_el in el:
					parent: Branch | None = branch_el.get("parent")
					branch = Branch(branch_el.get("name"))
					if parent is not None:
						if parent in branch_descendants:
							branch_descendants[parent].append(branch)
						else:
							branch_descendants[parent] = [branch]
					start_turn = Turn(int(branch_el.get("start-turn")))
					start_tick = Tick(int(branch_el.get("start-tick")))
					branch_starts[branch] = (start_turn, start_tick)
					end_turn = Turn(int(branch_el.get("end-turn")))
					end_tick = Tick(int(branch_el.get("end-tick")))
					self.query.set_branch(
						branch,
						parent,
						start_turn,
						start_tick,
						end_turn,
						end_tick,
					)
					if "last-turn-completed" in branch_el.keys():
						last_completed_turn = Turn(
							int(branch_el.get("last-turn-completed"))
						)
						self.query.complete_turn(
							branch, last_completed_turn, False
						)

					for turn_el in branch_el:
						turn = Turn(int(turn_el.get("number")))
						end_tick = Tick(int(turn_el.get("end-tick")))
						plan_end_tick = Tick(int(turn_el.get("plan-end-tick")))
						self.query.set_turn(
							branch, turn, end_tick, plan_end_tick
						)
						for elem in turn_el:
							getattr(self, "_" + elem.tag.replace("-", "_"))(
								branch_el, turn_el, elem
							)
				known_rules = (
					self.known_triggers.keys()
					| self.known_prereqs.keys()
					| self.known_actions.keys()
					| self.known_neighborhoods.keys()
					| self.known_big.keys()
				)
				trunk = Branch(el.get("trunk"))
				rules_created = set(self.query.rules_dump())
				for rule in known_rules:
					for mapp in [
						self.known_triggers,
						self.known_prereqs,
						self.known_actions,
						self.known_neighborhoods,
						self.known_big,
					]:
						if rule not in mapp:
							continue
						if rule not in rules_created:
							# Iterate depth first down the timestream, but no
							# deeper than when the rule is first set.
							# The game may have a rule by the same name
							# created in many branches independently.
							for branch in (
								trunk,
								*self._iter_descendants(
									branch_descendants,
									trunk,
									mapp[rule].__contains__,
									branch_starts.get,
								),
							):
								turn, tick = mapp[rule][branch].start_time()
								self._create_rule(rule, branch, turn, tick)
								rules_created.add(rule)
				for mapp, setter in [
					(
						self.known_triggers,
						self.query.set_rule_triggers,
					),
					(self.known_prereqs, self.query.set_rule_prereqs),
					(self.known_actions, self.query.set_rule_actions),
					(
						self.known_neighborhoods,
						self.query.set_rule_neighborhood,
					),
					(self.known_big, self.query.set_rule_big),
				]:
					for rule in mapp:
						for branch in mapp[rule]:
							for turn in mapp[rule][branch]:
								# Turn and tick are guaranteed to be in
								# chronological order here, because that's what
								# a SettingsTurnDict does.
								for tick, datum in mapp[rule][branch][
									turn
								].items():
									setter(rule, branch, turn, tick, datum)
			else:
				k = literal_eval(el.get("key"))
				v = self._element_to_value(el[0])
				self.query.eternal[k] = v
		self.query.commit()


def etree_to_sqlite(
	tree: ElementTree,
	sqlite_path: str | os.PathLike,
	engine: AbstractEngine | None = None,
):
	from .db import SQLAlchemyDatabaseConnector

	if not isinstance(sqlite_path, os.PathLike):
		sqlite_path = Path(sqlite_path)

	if engine is None:
		engine = EngineFacade(None)
		engine._mockup = True

	query = SQLAlchemyDatabaseConnector(
		"sqlite:///" + str(os.path.abspath(sqlite_path)),
		{},
		pack=engine.pack,
		unpack=engine.unpack,
	)
	return Importer(query, engine).etree_to_db(tree)


def xml_to_sqlite(
	xml_path: str | os.PathLike,
	sqlite_path: str | os.PathLike,
	engine: AbstractEngine | None = None,
):
	if not isinstance(xml_path, os.PathLike):
		xml_path = Path(xml_path)

	tree = parse(xml_path)

	return etree_to_sqlite(tree, sqlite_path, engine)


def etree_to_pqdb(
	tree: ElementTree,
	pqdb_path: str | os.PathLike,
	engine: AbstractEngine | None = None,
):
	from .db import ParquetDatabaseConnector

	if not isinstance(pqdb_path, os.PathLike):
		pqdb_path = Path(pqdb_path)

	if engine is None:
		engine = EngineFacade(None)
		engine._mockup = True

	query = ParquetDatabaseConnector(
		pqdb_path, pack=engine.pack, unpack=engine.unpack
	)

	return Importer(query, engine).etree_to_db(tree)


def xml_to_pqdb(
	xml_path: str | os.PathLike,
	pqdb_path: str | os.PathLike,
	engine: AbstractEngine | None = None,
):
	if not isinstance(xml_path, os.PathLike):
		xml_path = Path(xml_path)

	tree = parse(xml_path)

	return etree_to_pqdb(tree, pqdb_path, engine)


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
