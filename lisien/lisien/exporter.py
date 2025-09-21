import os
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from io import IOBase
from pathlib import Path
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Literal, Mapping, MutableSet, Set

if TYPE_CHECKING:
	from xml.etree.ElementTree import Element, ElementTree
	from xml.etree.ElementTree import indent as indent_tree
else:
	try:
		from lxml.etree import Element, ElementTree
		from lxml.etree import indent as indent_tree
	except ModuleNotFoundError:
		from xml.etree.ElementTree import (
			ElementTree,
			Element,
			indent as indent_tree,
		)

import networkx as nx
from tblib import Traceback

import lisien.types
from lisien.db import AbstractDatabaseConnector, LoadedCharWindow
from lisien.facade import EngineFacade
from lisien.types import (
	ActionFuncName,
	ActionRowType,
	Branch,
	CharName,
	CharRulebookRowType,
	EdgeRowType,
	EdgeValRowType,
	GraphEdgeValKeyframe,
	GraphNodeValKeyframe,
	GraphRowType,
	GraphValKeyframe,
	GraphValRowType,
	Key,
	Keyframe,
	NodeRowType,
	NodeRulebookRowType,
	NodeValRowType,
	PortalRulebookRowType,
	PrereqFuncName,
	PrereqRowType,
	RuleBig,
	RuleBigRowType,
	RulebookName,
	RulebookPriority,
	RulebookRowType,
	RuleName,
	RuleNeighborhood,
	RuleNeighborhoodRowType,
	RuleRowType,
	ThingRowType,
	Tick,
	Time,
	TriggerFuncName,
	TriggerRowType,
	Turn,
	UnitRowType,
	UniversalRowType,
	Value,
)
from lisien.util import AbstractEngine, sort_set

from .collections import GROUP_SEP, REC_SEP, FunctionStore


@dataclass
class Exporter:
	db: AbstractDatabaseConnector
	engine: AbstractEngine = field(default_factory=partial(EngineFacade, None))
	tree: ElementTree = field(
		default_factory=lambda: ElementTree(Element("lisien"))
	)

	def db_to_etree(self, name: str) -> ElementTree:
		root = self.tree.getroot()
		query = self.db
		query.commit()
		eternals = dict(query.eternal.items())
		root.set(
			"db-schema-version", str(eternals.pop("_lisien_schema_version"))
		)
		root.set("trunk", str(eternals.pop("trunk")))
		root.set("branch", str(eternals.pop("branch")))
		root.set("turn", str(eternals.pop("turn")))
		root.set("tick", str(eternals.pop("tick")))
		root.set("language", str(eternals.pop("language")))
		for k in sort_set(eternals.keys()):
			el = Element("dict-item", key=repr(k))
			root.append(el)
			el.append(self._value_to_xml_el(eternals[k]))
		trunks = set()
		branches_d = {}
		branch_descendants = {}
		turn_end_plan_d: dict[Branch, dict[Turn, tuple[Tick, Tick]]] = {}
		branch_elements = {}
		playtrees: dict[Branch, Element] = {}
		turns_completed_d: dict[Branch, Turn] = dict(
			query.turns_completed_dump()
		)
		keyframe_times: set[Time] = set(query.keyframes_dump())
		for (
			branch,
			turn,
			last_real_tick,
			last_planned_tick,
		) in query.turns_dump():
			if branch in turn_end_plan_d:
				turn_end_plan_d[branch][turn] = (
					last_real_tick,
					last_planned_tick,
				)
			else:
				turn_end_plan_d[branch] = {
					turn: (last_real_tick, last_planned_tick)
				}
		branch2do = deque(sorted(query.branches_dump()))
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
					{
						"name": branch,
						"start-turn": "0",
						"start-tick": "0",
						"end-turn": str(end_turn),
						"end-tick": str(end_tick),
					},
				)
				if branch in turns_completed_d:
					branch_element.set(
						"last-turn-completed", str(turns_completed_d[branch])
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
						{
							"name": branch,
							"parent": parent,
							"start-turn": str(parent_turn),
							"start-tick": str(parent_tick),
							"end-turn": str(end_turn),
							"end-tick": str(end_tick),
						},
					)
					if branch in turns_completed_d:
						branch_el.set(
							"last-turn-completed",
							str(turns_completed_d[branch]),
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
			if b in turn_end_plan_d:
				turn_to, tick_to = max(
					[
						(turn_to, tick_to),
						*(
							(r, t)
							for r, (_, t) in turn_end_plan_d[branch].items()
						),
					]
				)
			data = query.load_windows(
				[(b, turn_from, tick_from, turn_to, tick_to)]
			)
			self._fill_branch_element(
				branch_elements[b],
				turn_end_plan_d[b],
				keyframe_times,
				data,
			)
			if b in branch_descendants:
				for desc in sorted(branch_descendants[b], key=branches_d.get):
					recurse_branch(desc)

		for trunk in trunks:
			recurse_branch(trunk)

		return self.tree

	@classmethod
	def _value_to_xml_el(cls, value: Value | dict[Key, Value]) -> Element:
		if value is ...:
			return Element("Ellipsis")
		elif value is None:
			return Element("None")
		elif isinstance(value, bool):
			return Element("bool", value="true" if value else "false")
		elif isinstance(value, int):
			return Element("int", value=str(value))
		elif isinstance(value, float):
			return Element("float", value=str(value))
		elif isinstance(value, str):
			return Element("str", value=value)
		elif isinstance(value, lisien.types.DiGraph):
			# Since entity names are restricted to what we can use for dict
			# keys and also serialize to msgpack, I don't think there's any name
			# an entity can have that can't be repr'd
			return Element("character", name=repr(value.name))
		elif isinstance(value, lisien.types.Node):
			return Element(
				"node", character=repr(value.graph.name), name=repr(value.name)
			)
		elif isinstance(value, lisien.types.Edge):
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
				el.append(cls._value_to_xml_el(arg))
			return el
		elif isinstance(value, list):
			el = Element("list")
			for v in value:
				el.append(cls._value_to_xml_el(v))
			return el
		elif isinstance(value, tuple):
			el = Element("tuple")
			for v in value:
				el.append(cls._value_to_xml_el(v))
			return el
		elif isinstance(value, Set):
			if isinstance(value, (set, MutableSet)):
				el = Element("set")
				for v in value:
					el.append(cls._value_to_xml_el(v))
				return el
			else:
				el = Element("frozenset")
				for v in value:
					el.append(cls._value_to_xml_el(v))
				return el
		elif isinstance(value, Mapping):
			el = Element("dict")
			for k, v in value.items():
				dict_item = Element("dict-item", key=repr(k))
				dict_item.append(cls._value_to_xml_el(v))
				el.append(dict_item)
			return el
		else:
			raise TypeError("Can't convert to XML", value)

	@classmethod
	def _add_keyframe_to_turn_el(
		cls,
		turn_el: Element,
		tick: Tick,
		keyframe: Keyframe,
	) -> None:
		kfel = Element("keyframe", tick=str(tick))
		turn_el.append(kfel)
		universal_d: dict[Key, Value] = keyframe.get("universal", {})
		univel = cls._value_to_xml_el(universal_d)
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
		for rule_name in sorted(
			triggers_kf.keys() | prereqs_kf.keys() | actions_kf.keys()
		):
			rule_name: RuleName
			rule_el = Element(
				"rule",
				name=rule_name,
			)
			kfel.append(rule_el)
			if rule_name in bigs_kf and bigs_kf[rule_name]:
				rule_el.set("big", "true")
			if (
				rule_name in neighborhoods_kf
				and neighborhoods_kf[rule_name] is not None
			):
				rule_el.set(
					"neighborhood",
					str(neighborhoods_kf[rule_name]),
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
		for char_name, vals in sorted(graph_val_kf.items()):
			graph_el = char_els[char_name] = Element(
				"character", name=repr(char_name)
			)
			kfel.append(graph_el)
			if units_kf := vals.pop("units", {}):
				units_el = Element("units")
				any_unit_graphs = False
				for graph, nodes in units_kf.items():
					unit_graphs_el = Element("graph", character=repr(graph))
					any_unit_nodes = False
					for node, is_unit in nodes.items():
						if is_unit:
							any_unit_nodes = True
							unit_graphs_el.append(
								Element("unit", node=repr(node))
							)
					if any_unit_nodes:
						units_el.append(unit_graphs_el)
						any_unit_graphs = True
				if any_unit_graphs:
					graph_el.append(units_el)
			if "character_rulebook" in vals:
				graph_el.set(
					"character-rulebook",
					repr(
						vals.pop(
							"character_rulebook",
							("character_rulebook", char_name),
						)
					),
				)
			if "unit_rulebook" in vals:
				graph_el.set(
					"unit-rulebook",
					repr(
						vals.pop("unit_rulebook", ("unit_rulebook", char_name))
					),
				)
			if "character_thing_rulebook" in vals:
				graph_el.set(
					"character-thing-rulebook",
					repr(
						vals.pop(
							"character_thing_rulebook",
							("character_thing_rulebook", char_name),
						)
					),
				)
			if "character_place_rulebook" in vals:
				graph_el.set(
					"character-place-rulebook",
					repr(
						vals.pop(
							"character_place_rulebook",
							("character_place_rulebook", char_name),
						)
					),
				)
			if "character_portal_rulebook" in vals:
				graph_el.set(
					"character-portal-rulebook",
					repr(
						vals.pop(
							"character_portal_rulebook",
							("character_portal_rulebook", char_name),
						)
					),
				)
			for k, v in vals.items():
				item_el = Element("dict-item", key=repr(k))
				graph_el.append(item_el)
				item_el.append(cls._value_to_xml_el(v))
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
				node_el = Element(
					"node",
					name=repr(node),
				)
				if "rulebook" in val:
					node_el.set("rulebook", repr(val.pop("rulebook")))
				char_el.append(node_el)
				for k, v in val.items():
					item_el = Element("dict-item", key=repr(k))
					node_el.append(item_el)
					item_el.append(cls._value_to_xml_el(v))
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
					edge_el = Element(
						"edge",
						orig=repr(orig),
						dest=repr(dest),
					)
					if "rulebook" in val:
						edge_el.set("rulebook", repr(val.pop("rulebook")))
					char_el.append(edge_el)
					for k, v in val.items():
						item_el = Element("dict-item", key=repr(k))
						edge_el.append(item_el)
						item_el.append(Exporter._value_to_xml_el(v))

	def write_xml(
		self,
		to: str | os.PathLike | IOBase,
		name: str = None,
		indent: bool = True,
	) -> None:
		if not isinstance(to, os.PathLike) and not isinstance(to, IOBase):
			if to is None:
				if name is None:
					raise ValueError("Need a name or a path")
				to = os.path.join(os.getcwd(), name + ".xml")
			to = Path(to)
		if isinstance(to, Path) and name is None:
			name = os.path.basename(to)
		if name[-4:] == ".xml":
			name = name[:-4]

		tree = self.db_to_etree(name)

		if indent:
			indent_tree(tree)
		tree.write(to, encoding="utf-8")

	@classmethod
	def _append_univ_el(
		cls, turn_el: Element, universal_rec: UniversalRowType
	):
		key, b, r, t, val = universal_rec
		univ_el = Element(
			"universal",
			key=repr(key),
			tick=str(t),
		)
		univ_el.append(cls._value_to_xml_el(val))
		turn_el.append(univ_el)

	@staticmethod
	def _append_rulebook_el(turn_el: Element, rulebook_rec: RulebookRowType):
		rb, b, r, t, (rules, prio) = rulebook_rec
		rb_el = Element(
			"rulebook",
			name=repr(rb),
			priority=repr(prio),
			tick=str(t),
		)
		turn_el.append(rb_el)
		for i, rule in enumerate(rules):
			rb_el.append(Element("rule", name=rule))

	@staticmethod
	def _append_rule_flist_el(
		typ: str,
		turn_el: Element,
		rec: TriggerRowType | PrereqRowType | ActionRowType,
	):
		rule, _, __, tick, funcs = rec
		func_el = Element(f"{typ}s", rule=rule, tick=str(tick))
		turn_el.append(func_el)
		for func in funcs:
			func_el.append(Element(typ[5:], name=func))

	append_rule_triggers_el = partial(_append_rule_flist_el, "rule-trigger")
	append_rule_prereqs_el = partial(_append_rule_flist_el, "rule-prereq")
	append_rule_actions_el = partial(_append_rule_flist_el, "rule-action")

	@staticmethod
	def _append_rule_neighborhood_el(
		turn_el: Element, nbr_rec: RuleNeighborhoodRowType
	):
		rule, _, __, tick, nbr = nbr_rec
		if nbr is not None:
			nbr_el = Element(
				"rule-neighborhood",
				rule=rule,
				tick=str(tick),
				neighbors=str(nbr),
			)
		else:
			nbr_el = Element("rule-neighborhood", rule=rule, tick=str(tick))
		turn_el.append(nbr_el)

	@staticmethod
	def _append_rule_big_el(turn_el: Element, big_rec: RuleBigRowType):
		rule, _, __, tick, big = big_rec
		turn_el.append(
			Element(
				"rule-big",
				rule=rule,
				tick=str(tick),
				big="true" if big else "false",
			)
		)

	@staticmethod
	def _append_graph_el(turn_el: Element, graph: GraphRowType):
		char, b, r, t, typ_str = graph
		graph_el = Element(
			"graph",
			character=repr(char),
			tick=str(t),
			type=typ_str,
		)
		turn_el.append(graph_el)

	@staticmethod
	def _append_graph_val_el(turn_el: Element, graph_val: GraphValRowType):
		char, stat, b, r, t, val = graph_val
		graph_val_el = Element(
			"graph-val",
			character=repr(char),
			key=repr(stat),
			tick=str(t),
		)
		turn_el.append(graph_val_el)
		graph_val_el.append(Exporter._value_to_xml_el(val))

	@staticmethod
	def _append_nodes_el(turn_el: Element, nodes: NodeRowType):
		char, node, b, r, t, ex = nodes
		turn_el.append(
			Element(
				"node",
				character=repr(char),
				name=repr(node),
				tick=str(t),
				exists="true" if ex else "false",
			)
		)

	@staticmethod
	def _append_node_val_el(turn_el: Element, node_val: NodeValRowType):
		char, node, stat, b, r, t, val = node_val
		node_val_el = Element(
			"node-val",
			character=repr(char),
			node=repr(node),
			key=repr(stat),
			tick=str(t),
		)
		turn_el.append(node_val_el)
		node_val_el.append(Exporter._value_to_xml_el(val))

	@staticmethod
	def _append_edges_el(turn_el: Element, edges: EdgeRowType):
		char, orig, dest, b, r, t, ex = edges
		turn_el.append(
			Element(
				"edge",
				character=repr(char),
				orig=repr(orig),
				dest=repr(dest),
				tick=str(t),
				exists="true" if ex else "false",
			)
		)

	@staticmethod
	def _append_edge_val_el(turn_el: Element, edge_val: EdgeValRowType):
		char, orig, dest, stat, b, r, t, val = edge_val
		edge_val_el = Element(
			"edge-val",
			character=repr(char),
			orig=repr(orig),
			dest=repr(dest),
			key=repr(stat),
			tick=str(t),
		)
		turn_el.append(edge_val_el)
		edge_val_el.append(Exporter._value_to_xml_el(val))

	@staticmethod
	def _append_thing_el(turn_el: Element, thing: ThingRowType):
		char, thing, b, r, t, loc = thing
		turn_el.append(
			Element(
				"location",
				character=repr(char),
				thing=repr(thing),
				tick=str(t),
				location=repr(loc),
			)
		)

	@staticmethod
	def _append_unit_el(turn_el: Element, unit: UnitRowType):
		char, graph, node, b, r, t, is_unit = unit
		unit_el = Element(
			"unit",
			{
				"character-graph": repr(char),
				"unit-graph": repr(graph),
				"unit-node": repr(node),
				"tick": str(t),
			},
		)
		unit_el.set("is-unit", "true" if is_unit else "false")
		turn_el.append(unit_el)

	@staticmethod
	def _append_char_rb_el(
		turn_el: Element, rbtyp: str, rbrow: CharRulebookRowType
	):
		char, b, r, t, rb = rbrow
		turn_el.append(
			Element(
				rbtyp,
				character=repr(char),
				tick=str(t),
				rulebook=repr(rb),
			)
		)

	@staticmethod
	def _append_node_rb_el(turn_el: Element, nrb_row: NodeRulebookRowType):
		char, node, b, r, t, rb = nrb_row
		turn_el.append(
			Element(
				"node-rulebook",
				character=repr(char),
				node=repr(node),
				tick=str(t),
				rulebook=repr(rb),
			)
		)

	@staticmethod
	def _append_portal_rb_el(
		turn_el: Element, port_rb_row: PortalRulebookRowType
	):
		char, orig, dest, b, r, t, rb = port_rb_row
		turn_el.append(
			Element(
				"portal-rulebook",
				character=repr(char),
				orig=repr(orig),
				dest=repr(dest),
				tick=str(t),
				rulebook=repr(rb),
			)
		)

	def _fill_branch_element(
		self,
		branch_el: Element,
		turn_ends: dict[Turn, tuple[Tick, Tick]],
		keyframe_times: set[Time],
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
	):
		query = self.db
		branch_ = branch_el.get("name")
		if branch_ is None:
			raise TypeError("branch missing")
		branch = Branch(branch_)

		uncharacterized = {
			"graphs",
			"universals",
			"rulebooks",
			"rule_triggers",
			"rule_prereqs",
			"rule_actions",
			"rule_neighborhood",
			"rule_big",
		}
		turn: Turn
		for turn, (ending_tick, plan_ending_tick) in sorted(turn_ends.items()):
			turn_el = Element(
				"turn",
				{
					"number": str(turn),
					"end-tick": str(ending_tick),
					"plan-end-tick": str(plan_ending_tick),
				},
			)
			branch_el.append(turn_el)
			tick: Tick
			for tick in range(plan_ending_tick + 1):
				if (branch, turn, tick) in keyframe_times:
					kf = query.get_keyframe(branch, turn, tick)
					Exporter._add_keyframe_to_turn_el(turn_el, tick, kf)
					keyframe_times.remove((branch, turn, tick))
				if data["universals"]:
					universal_rec: UniversalRowType = data["universals"][0]
					key, branch_now, turn_now, tick_now, _ = universal_rec
					if (branch_now, turn_now, tick_now) == (
						branch,
						turn,
						tick,
					):
						self._append_univ_el(turn_el, universal_rec)
						del data["universals"][0]
				if data["rulebooks"]:
					rulebook_rec: RulebookRowType = data["rulebooks"][0]
					rulebook, branch_now, turn_now, tick_now, _ = rulebook_rec
					if (branch_now, turn_now, tick_now) == (
						branch,
						turn,
						tick,
					):
						self._append_rulebook_el(turn_el, rulebook_rec)
						del data["rulebooks"][0]
				if data["graphs"]:
					graph_rec: GraphRowType = data["graphs"][0]
					_, branch_now, turn_now, tick_now, _ = graph_rec
					if (branch_now, turn_now, tick_now) == (
						branch,
						turn,
						tick,
					):
						self._append_graph_el(turn_el, graph_rec)
						del data["graphs"][0]
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
					self.append_rule_triggers_el(
						turn_el, data["rule_triggers"].pop(0)
					)
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
					self.append_rule_prereqs_el(
						turn_el, data["rule_prereqs"].pop(0)
					)
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
					self.append_rule_actions_el(
						turn_el, data["rule_actions"].pop(0)
					)
				if (
					"rule_neighborhood" in data
					and data["rule_neighborhood"]
					and data["rule_neighborhood"][0][1:4]
					== (branch, turn, tick)
				):
					self._append_rule_neighborhood_el(
						turn_el, data["rule_neighborhood"].pop(0)
					)
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
					self._append_rule_big_el(turn_el, data["rule_big"].pop(0))
				for char_name in data.keys() - uncharacterized:
					char_data: LoadedCharWindow = data[char_name]
					if char_data["graph_val"]:
						graph_val_row: GraphValRowType = char_data[
							"graph_val"
						][0]
						_, __, b, r, t, ___ = graph_val_row
						if (b, r, t) == (branch, turn, tick):
							self._append_graph_val_el(turn_el, graph_val_row)
							del char_data["graph_val"][0]
					if char_data["nodes"]:
						nodes_row: NodeRowType = char_data["nodes"][0]
						char, node, branch_now, turn_now, tick_now, ex = (
							nodes_row
						)
						if (branch_now, turn_now, tick_now) == (
							branch,
							turn,
							tick,
						):
							self._append_nodes_el(turn_el, nodes_row)
							del char_data["nodes"][0]
					if char_data["node_val"]:
						node_val_row: NodeValRowType = char_data["node_val"][0]
						(
							char,
							node,
							stat,
							branch_now,
							turn_now,
							tick_now,
							val,
						) = node_val_row
						if (branch_now, turn_now, tick_now) == (
							branch,
							turn,
							tick,
						):
							self._append_node_val_el(turn_el, node_val_row)
							del char_data["node_val"][0]
					if char_data["edges"]:
						edges_row: EdgeRowType = char_data["edges"][0]
						_, __, ___, b, r, t, ____ = edges_row
						if (b, r, t) == (branch, turn, tick):
							self._append_edges_el(edges_row)
							del char_data["edges"][0]
					if char_data["edge_val"]:
						edge_val_row: EdgeValRowType = char_data["edge_val"][0]
						_, __, ___, ____, b, r, t, _____ = edge_val_row
						if (b, r, t) == (branch, turn, tick):
							self._append_edge_val_el(turn_el, edge_val_row)
							del char_data["edge_val"][0]
					if char_data["things"]:
						thing_row: ThingRowType = char_data["things"][0]
						_, __, b, r, t, ___ = thing_row
						if (b, r, t) == (branch, turn, tick):
							self._append_thing_el(turn_el, thing_row)
							del char_data["things"][0]
					if char_data["units"]:
						units_row: UnitRowType = char_data["units"][0]
						_, __, ___, b, r, t, ____ = units_row
						if (b, r, t) == (branch, turn, tick):
							self._append_unit_el(turn_el, units_row)
							del char_data["units"][0]
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
								self._append_char_rb_el(
									turn_el,
									char_rb_typ.replace("_", "-"),
									char_rb_row,
								)
								del char_data[char_rb_typ][0]
					if char_data["node_rulebook"]:
						node_rb_row: NodeRulebookRowType = char_data[
							"node_rulebook"
						][0]
						_, __, b, r, t, ___ = node_rb_row
						if (b, r, t) == (branch, turn, tick):
							self._append_node_rb_el(turn_el, node_rb_row)
							del char_data["node_rulebook"][0]
					if char_data["portal_rulebook"]:
						port_rb_row: PortalRulebookRowType = char_data[
							"portal_rulebook"
						][0]
						_, __, ___, b, r, t, ____ = port_rb_row
						if (b, r, t) == (branch, turn, tick):
							self._append_portal_rb_el(turn_el, port_rb_row)
							del char_data["portal_rulebook"][0]
		for k in uncharacterized:
			if k in data:
				assert not data[k]
		for char_name in data.keys() - uncharacterized:
			for k, v in data[char_name].items():
				assert not v, f"Leftover data in {char_name}'s {k}: {v}"
		assert not keyframe_times, keyframe_times


def sqlite_to_etree(
	sqlite_path: str | os.PathLike,
	name: str | None = None,
	tree: ElementTree | None = None,
	engine: AbstractEngine | None = None,
) -> ElementTree:
	from .db import SQLAlchemyDatabaseConnector

	if not isinstance(sqlite_path, os.PathLike):
		sqlite_path = Path(sqlite_path)

	if engine is None:
		engine = EngineFacade(None)
	query = SQLAlchemyDatabaseConnector(
		"sqlite:///" + str(os.path.abspath(sqlite_path)),
		{},
		pack=engine.pack,
		unpack=engine.unpack,
	)
	if tree is None:
		tree = ElementTree(Element("lisien"))
	if name is None:
		name = str(os.path.basename(os.path.dirname(sqlite_path)))
	return Exporter(query, tree).db_to_etree(name)


def pqdb_to_etree(
	pqdb_path: str | os.PathLike,
	name: str | None = None,
	tree: ElementTree | None = None,
	engine: AbstractEngine | None = None,
) -> ElementTree:
	from .db import ParquetDatabaseConnector

	if not isinstance(pqdb_path, os.PathLike):
		pqdb_path = Path(pqdb_path)

	if engine is None:
		engine = EngineFacade(None)
	query = ParquetDatabaseConnector(
		pqdb_path, pack=engine.pack, unpack=engine.unpack
	)
	if tree is None:
		tree = ElementTree(Element("lisien"))
	if name is None:
		name = str(os.path.basename(os.path.dirname(pqdb_path)))
	return Exporter(query, tree).db_to_etree(name)


def game_path_to_etree(
	game_path: str | os.PathLike, name: str | None = None
) -> ElementTree:
	import json
	from base64 import b64encode
	from hashlib import blake2b

	from .collections import FunctionStore

	world = os.path.join(game_path, "world")
	if os.path.isdir(world):
		game_history = pqdb_to_etree(world, name)
	else:
		game_history = sqlite_to_etree(world + ".sqlite3", name)
	lisien_el: Element = game_history.getroot()

	ls = os.listdir(game_path)
	# Take the hash of code and strings--*not* the hash of their *files*--
	# so that if the file gets reformatted, as often happens as a side effect
	# of ast.parse and astunparse, this does not change its hash.
	for modname in ("function", "method", "trigger", "prereq", "action"):
		modpy = modname + ".py"
		if modpy in ls:
			srchash = FunctionStore(os.path.join(game_path, modpy)).blake2b()
			lisien_el.set(modname, b64encode(srchash).decode("utf-8"))
	strings_dir = os.path.join(game_path, "strings")
	if (
		"strings" in ls
		and os.path.isdir(strings_dir)
		and (langfiles := os.listdir(strings_dir))
	):
		for fn in langfiles:
			langhash = blake2b()
			with open(os.path.join(strings_dir, fn), "rb") as inf:
				langlines = json.load(inf)
			for k in sort_set(langlines.keys()):
				langhash.update(k.encode())
				langhash.update(GROUP_SEP)
				langhash.update(langlines[k].encode())
				langhash.update(REC_SEP)
			if len(fn) > 5 and fn[-5:] == ".json":
				fn = fn[:-5]
			lisien_el.append(
				Element(
					"language",
					code=fn,
					blake2b=b64encode(langhash.digest()).decode("utf-8"),
				)
			)
	return game_history


def game_path_to_xml(
	game_path: str | os.PathLike,
	xml_file_path: str | os.PathLike | IOBase,
	indent: bool = True,
	name: str | None = None,
) -> None:
	if not isinstance(game_path, os.PathLike):
		game_path = Path(game_path)
	if not isinstance(xml_file_path, os.PathLike) and not isinstance(
		xml_file_path, IOBase
	):
		xml_file_path = Path(xml_file_path)

	tree = game_path_to_etree(game_path, name)
	if indent:
		indent_tree(tree)
	tree.write(xml_file_path, encoding="utf-8")


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
