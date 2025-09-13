# This file is part of Lisien, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from collections import OrderedDict
from functools import partial
from json import dumps

from sqlalchemy import (
	BLOB,
	BOOLEAN,
	FLOAT,
	INT,
	TEXT,
	CheckConstraint,
	Column,
	ForeignKey,
	ForeignKeyConstraint,
	MetaData,
	Table,
	and_,
	bindparam,
	func,
	null,
	or_,
	select,
)
from sqlalchemy.sql.ddl import CreateIndex, CreateTable

BaseColumn = Column
Column = partial(BaseColumn, nullable=False)


def tables(meta: MetaData):
	"""Return a dictionary full of all the tables I need for lisien. Use the
	provided metadata object.

	"""
	Table(
		"global",
		meta,
		Column("key", BLOB, primary_key=True),
		Column("value", BLOB, nullable=True),
		sqlite_with_rowid=False,
	)
	Table(
		"branches",
		meta,
		Column(
			"branch",
			TEXT,
			primary_key=True,
			default="trunk",
		),
		Column("parent", TEXT, default="trunk", nullable=True),
		Column("parent_turn", INT, default=0),
		Column("parent_tick", INT, default=0),
		Column("end_turn", INT, default=0),
		Column("end_tick", INT, default=0),
		CheckConstraint("branch<>parent"),
		sqlite_with_rowid=False,
	)
	Table(
		"turns",
		meta,
		Column("branch", TEXT, primary_key=True),
		Column("turn", INT, primary_key=True),
		Column("end_tick", INT),
		Column("plan_end_tick", INT),
		sqlite_with_rowid=False,
	)
	Table(
		"bookmarks",
		meta,
		Column("key", TEXT, primary_key=True),
		Column("branch", TEXT, default="trunk"),
		Column("turn", INT),
		Column("tick", INT),
	)
	Table(
		"graphs",
		meta,
		Column("branch", TEXT, primary_key=True),
		Column("turn", INT, primary_key=True),
		Column("tick", INT, primary_key=True),
		Column("graph", BLOB, primary_key=True),
		Column("type", TEXT, default="Graph", nullable=True),
		CheckConstraint(
			"type IN "
			"('Graph', 'DiGraph', 'MultiGraph', 'MultiDiGraph', 'Deleted')"
		),
		sqlite_with_rowid=False,
	)
	kfs = Table(
		"keyframes",
		meta,
		Column(
			"branch",
			TEXT,
			ForeignKey("branches.branch"),
			primary_key=True,
			default="trunk",
		),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
	)
	Table(
		"keyframes_graphs",
		meta,
		Column(
			"branch",
			TEXT,
			primary_key=True,
			default="trunk",
		),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("graph", BLOB, primary_key=True),
		Column("nodes", BLOB),
		Column("edges", BLOB),
		Column("graph_val", BLOB),
		ForeignKeyConstraint(
			["branch", "turn", "tick"], [kfs.c.branch, kfs.c.turn, kfs.c.tick]
		),
		sqlite_with_rowid=False,
	)
	Table(
		"graph_val",
		meta,
		Column(
			"branch",
			TEXT,
			ForeignKey("branches.branch"),
			primary_key=True,
			default="trunk",
		),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("graph", BLOB, primary_key=True),
		Column("key", BLOB, primary_key=True),
		Column("value", BLOB),
		sqlite_with_rowid=False,
	)
	Table(
		"nodes",
		meta,
		Column(
			"branch",
			TEXT,
			ForeignKey("branches.branch"),
			primary_key=True,
			default="trunk",
		),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("graph", BLOB, primary_key=True),
		Column("node", BLOB, primary_key=True),
		Column("extant", BOOLEAN),
		sqlite_with_rowid=False,
	)
	Table(
		"node_val",
		meta,
		Column(
			"branch",
			TEXT,
			ForeignKey("branches.branch"),
			primary_key=True,
			default="trunk",
		),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("graph", BLOB, primary_key=True),
		Column("node", BLOB, primary_key=True),
		Column("key", BLOB, primary_key=True),
		Column("value", BLOB),
		sqlite_with_rowid=False,
	)
	Table(
		"edges",
		meta,
		Column(
			"branch",
			TEXT,
			ForeignKey("branches.branch"),
			primary_key=True,
			default="trunk",
		),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("graph", BLOB, primary_key=True),
		Column("orig", BLOB, primary_key=True),
		Column("dest", BLOB, primary_key=True),
		Column("extant", BOOLEAN),
		sqlite_with_rowid=False,
	)
	Table(
		"edge_val",
		meta,
		Column(
			"branch",
			TEXT,
			ForeignKey("branches.branch"),
			primary_key=True,
			default="trunk",
		),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("graph", BLOB, primary_key=True),
		Column("orig", BLOB, primary_key=True),
		Column("dest", BLOB, primary_key=True),
		Column("key", BLOB, primary_key=True),
		Column("value", BLOB),
		sqlite_with_rowid=False,
	)
	Table(
		"plans",
		meta,
		Column(
			"id",
			INT,
			primary_key=True,
			autoincrement=False,
		),
		Column("branch", TEXT),
		Column("turn", INT),
		Column("tick", INT),
	)
	Table(
		"plan_ticks",
		meta,
		Column("plan_id", INT, primary_key=True),
		Column("branch", TEXT, primary_key=True),
		Column("turn", INT, primary_key=True),
		Column("tick", INT, primary_key=True),
		ForeignKeyConstraint(("plan_id",), ("plans.id",)),
		sqlite_with_rowid=False,
	)

	# Table for global variables that are not sensitive to sim-time.
	Table(
		"universals",
		meta,
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("key", BLOB, primary_key=True),
		Column("value", BLOB),
		sqlite_with_rowid=False,
	)
	kfs = meta.tables["keyframes"]

	Table(
		"keyframe_extensions",
		meta,
		Column(
			"branch",
			TEXT,
			primary_key=True,
			default="trunk",
		),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("universal", BLOB),
		Column("rule", BLOB),
		Column("rulebook", BLOB),
		ForeignKeyConstraint(
			["branch", "turn", "tick"], [kfs.c.branch, kfs.c.turn, kfs.c.tick]
		),
	)

	Table(
		"rules",
		meta,
		Column("rule", TEXT, primary_key=True),
		sqlite_with_rowid=False,
	)

	# Table grouping rules into lists called rulebooks.
	Table(
		"rulebooks",
		meta,
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("rulebook", BLOB, primary_key=True),
		Column("rules", BLOB, default=b"\x90"),  # empty array
		Column("priority", FLOAT, default=0.0),
		sqlite_with_rowid=False,
	)

	# Table for rules' triggers, those functions that return True only
	# when their rule should run (or at least check its prereqs).
	Table(
		"rule_triggers",
		meta,
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("rule", TEXT, primary_key=True),
		Column("triggers", BLOB, default=b"\x90"),
		ForeignKeyConstraint(("rule",), ["rules.rule"]),
		sqlite_with_rowid=False,
	)

	# Table for rules' neighborhoods, which govern when triggers should be
	# checked -- when that makes sense. Basically just rules on character.place
	Table(
		"rule_neighborhood",
		meta,
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("rule", TEXT, primary_key=True),
		Column("neighborhood", INT, nullable=True, default=None),
		ForeignKeyConstraint(("rule",), ["rules.rule"]),
		sqlite_with_rowid=False,
	)

	# Table for rules' prereqs, functions with veto power over a rule
	# being followed
	Table(
		"rule_prereqs",
		meta,
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("rule", TEXT, primary_key=True),
		Column("prereqs", BLOB, default=b"\x90"),
		ForeignKeyConstraint(("rule",), ["rules.rule"]),
		sqlite_with_rowid=False,
	)

	# Table for rules' actions, the functions that do what the rule
	# does.
	Table(
		"rule_actions",
		meta,
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("rule", TEXT, primary_key=True),
		Column("actions", BLOB, default=b"\x90"),
		ForeignKeyConstraint(("rule",), ["rules.rule"]),
		sqlite_with_rowid=False,
	)

	# Table indicating which rules make big changes to the world.
	Table(
		"rule_big",
		meta,
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("rule", TEXT, primary_key=True),
		Column("big", BOOLEAN, default=False),
		ForeignKeyConstraint(("rule",), ["rules.rule"]),
		sqlite_with_rowid=False,
	)

	# The top level of the lisien world model, the character. Includes
	# rulebooks for the character itself, its units, and all the things,
	# places, and portals it contains--though those may have their own
	# rulebooks as well.

	for name in (
		"character_rulebook",
		"unit_rulebook",
		"character_thing_rulebook",
		"character_place_rulebook",
		"character_portal_rulebook",
	):
		Table(
			name,
			meta,
			Column("branch", TEXT, primary_key=True, default="trunk"),
			Column("turn", INT, primary_key=True, default=0),
			Column("tick", INT, primary_key=True, default=0),
			Column("character", BLOB, primary_key=True),
			Column("rulebook", BLOB),
			sqlite_with_rowid=False,
		)

	# Rules handled within the rulebook associated with one node in
	# particular.
	nrh = Table(
		"node_rules_handled",
		meta,
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("character", BLOB, primary_key=True),
		Column("node", BLOB, primary_key=True),
		Column("rulebook", BLOB, primary_key=True),
		Column("rule", TEXT, primary_key=True),
		Column("tick", INT),
		sqlite_with_rowid=False,
	)

	# Rules handled within the rulebook associated with one portal in
	# particular.
	porh = Table(
		"portal_rules_handled",
		meta,
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("character", BLOB, primary_key=True),
		Column("orig", BLOB, primary_key=True),
		Column("dest", BLOB, primary_key=True),
		Column("rulebook", BLOB, primary_key=True),
		Column("rule", TEXT, primary_key=True),
		Column("tick", INT),
		sqlite_with_rowid=False,
	)

	# Table for Things, being those nodes in a Character graph that have
	# locations.
	#
	# A Thing's location can be either a Place or another Thing, as long
	# as it's in the same Character.
	Table(
		"things",
		meta,
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("character", BLOB, primary_key=True),
		Column("thing", BLOB, primary_key=True),
		# when location is null, this node is not a thing, but a place
		Column("location", BLOB),
		sqlite_with_rowid=False,
	)

	# The rulebook followed by a given node.
	Table(
		"node_rulebook",
		meta,
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("character", BLOB, primary_key=True),
		Column("node", BLOB, primary_key=True),
		Column("rulebook", BLOB),
		sqlite_with_rowid=False,
	)

	# The rulebook followed by a given Portal.
	#
	# "Portal" is lisien's term for an edge in any of the directed
	# graphs it uses. The name is different to distinguish them from
	# Edge objects, which exist in an underlying object-relational
	# mapper called allegedb, and have a different API.
	Table(
		"portal_rulebook",
		meta,
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("character", BLOB, primary_key=True),
		Column("orig", BLOB, primary_key=True),
		Column("dest", BLOB, primary_key=True),
		Column("rulebook", BLOB),
		sqlite_with_rowid=False,
	)

	# The units representing one Character in another.
	#
	# In the common situation where a Character, let's say Alice has her
	# own stats and skill tree and social graph, and also has a location
	# in physical space, you can represent this by creating a Thing in
	# the Character that represents physical space, and then making that
	# Thing an unit of Alice. On its own this doesn't do anything,
	# it's just a convenient way of indicating the relation -- but if
	# you like, you can make rules that affect all units of some
	# Character, irrespective of what Character the unit is actually
	# *in*.
	Table(
		"units",
		meta,
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("character_graph", BLOB, primary_key=True),
		Column("unit_graph", BLOB, primary_key=True),
		Column("unit_node", BLOB, primary_key=True),
		Column("is_unit", BOOLEAN),
		sqlite_with_rowid=False,
	)

	Table(
		"character_rules_handled",
		meta,
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True),
		Column("character", BLOB, primary_key=True),
		Column("rulebook", BLOB, primary_key=True),
		Column("rule", TEXT, primary_key=True),
		Column("tick", INT),
		sqlite_with_rowid=True,
	)

	Table(
		"unit_rules_handled",
		meta,
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True),
		Column("character", BLOB, primary_key=True),
		Column("graph", BLOB, primary_key=True),
		Column(
			"unit",
			BLOB,
			primary_key=True,
		),
		Column("rulebook", BLOB, primary_key=True),
		Column("rule", TEXT, primary_key=True),
		Column("tick", INT),
		sqlite_with_rowid=True,
	)

	Table(
		"character_thing_rules_handled",
		meta,
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True),
		Column("character", BLOB, primary_key=True),
		Column("rulebook", BLOB, primary_key=True),
		Column("rule", TEXT, primary_key=True),
		Column("thing", BLOB, primary_key=True),
		Column("tick", INT),
	)

	Table(
		"character_place_rules_handled",
		meta,
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True),
		Column("character", BLOB, primary_key=True),
		Column("place", BLOB, primary_key=True),
		Column("rulebook", BLOB, primary_key=True),
		Column("rule", TEXT, primary_key=True),
		Column("tick", INT),
		sqlite_with_rowid=True,
	)

	Table(
		"character_portal_rules_handled",
		meta,
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True),
		Column("character", BLOB, primary_key=True),
		Column("orig", BLOB, primary_key=True),
		Column("dest", BLOB, primary_key=True),
		Column("rulebook", BLOB, primary_key=True),
		Column("rule", TEXT, primary_key=True),
		Column("tick", INT, primary_key=True),
		sqlite_with_rowid=True,
	)

	Table(
		"turns_completed",
		meta,
		Column("branch", TEXT, primary_key=True),
		Column("turn", INT),
		sqlite_with_rowid=False,
	)

	return meta.tables


def queries(meta: MetaData):
	def update_where(updcols, wherecols):
		"""Return an ``UPDATE`` statement that updates the columns ``updcols``
		when the ``wherecols`` match. Every column has a bound parameter of
		the same name.

		updcols are strings, wherecols are column objects

		"""
		vmap = OrderedDict()
		for col in updcols:
			vmap[col] = bindparam(col)
		wheres = [c == bindparam(c.name) for c in wherecols]
		tab = wherecols[0].table
		return tab.update().values(**vmap).where(and_(*wheres))

	def tick_to_end_clause(tab):
		return and_(
			tab.c.branch == bindparam("branch"),
			or_(
				tab.c.turn > bindparam("turn_from"),
				and_(
					tab.c.turn == bindparam("turn_from"),
					tab.c.tick >= bindparam("tick_from"),
				),
			),
		)

	def tick_to_tick_clause(tab):
		return and_(
			tick_to_end_clause(tab),
			or_(
				tab.c.turn < bindparam("turn_to"),
				and_(
					tab.c.turn == bindparam("turn_to"),
					tab.c.tick <= bindparam("tick_to"),
				),
			),
		)

	table = meta.tables

	graphs = table["graphs"]
	globtab = table["global"]
	edge_val = table["edge_val"]
	edges = table["edges"]
	nodes = table["nodes"]
	node_val = table["node_val"]
	graph_val = table["graph_val"]
	branches = table["branches"]
	turns = table["turns"]
	keyframes_graphs = table["keyframes_graphs"]
	keyframes = table["keyframes"]
	r = {
		"global_get": select(globtab.c.value).where(
			globtab.c.key == bindparam("key")
		),
		"global_update": globtab.update()
		.values(value=bindparam("value"))
		.where(globtab.c.key == bindparam("key")),
		"graph_type": select(graphs.c.type).where(
			graphs.c.graph == bindparam("graph")
		),
		"del_edge_val_after": edge_val.delete().where(
			and_(
				edge_val.c.graph == bindparam("graph"),
				edge_val.c.orig == bindparam("orig"),
				edge_val.c.dest == bindparam("dest"),
				edge_val.c.key == bindparam("key"),
				edge_val.c.branch == bindparam("branch"),
				or_(
					edge_val.c.turn > bindparam("turn"),
					and_(
						edge_val.c.turn == bindparam("turn"),
						edge_val.c.tick >= bindparam("tick"),
					),
				),
			)
		),
		"del_edges_graph": edges.delete().where(
			edges.c.graph == bindparam("graph")
		),
		"del_edges_after": edges.delete().where(
			and_(
				edges.c.graph == bindparam("graph"),
				edges.c.orig == bindparam("orig"),
				edges.c.dest == bindparam("dest"),
				edges.c.branch == bindparam("branch"),
				or_(
					edges.c.turn > bindparam("turn"),
					and_(
						edges.c.turn == bindparam("turn"),
						edges.c.tick >= bindparam("tick"),
					),
				),
			)
		),
		"del_nodes_after": nodes.delete().where(
			and_(
				nodes.c.graph == bindparam("graph"),
				nodes.c.node == bindparam("node"),
				nodes.c.branch == bindparam("branch"),
				or_(
					nodes.c.turn > bindparam("turn"),
					and_(
						nodes.c.turn == bindparam("turn"),
						nodes.c.tick >= bindparam("tick"),
					),
				),
			)
		),
		"del_node_val_after": node_val.delete().where(
			and_(
				node_val.c.graph == bindparam("graph"),
				node_val.c.node == bindparam("node"),
				node_val.c.key == bindparam("key"),
				node_val.c.branch == bindparam("branch"),
				or_(
					node_val.c.turn > bindparam("turn"),
					and_(
						node_val.c.turn == bindparam("turn"),
						node_val.c.tick >= bindparam("tick"),
					),
				),
			)
		),
		"del_graph_val_after": graph_val.delete().where(
			and_(
				graph_val.c.graph == bindparam("graph"),
				graph_val.c.key == bindparam("key"),
				graph_val.c.branch == bindparam("branch"),
				or_(
					graph_val.c.turn > bindparam("turn"),
					and_(
						graph_val.c.turn == bindparam("turn"),
						graph_val.c.tick >= bindparam("tick"),
					),
				),
			)
		),
		"global_delete": globtab.delete().where(
			globtab.c.key == bindparam("key")
		),
		"graphs_types": select(graphs.c.graph, graphs.c.type),
		"graphs_delete": graphs.delete().where(
			and_(
				graphs.c.graph == bindparam("graph"),
				graphs.c.branch == bindparam("branch"),
				graphs.c.turn == bindparam("turn"),
				graphs.c.tick == bindparam("tick"),
			)
		),
		"graphs_named": select(func.COUNT())
		.select_from(graphs)
		.where(graphs.c.graph == bindparam("graph")),
		"graphs_between": select(
			graphs.c.graph,
			graphs.c.turn,
			graphs.c.tick,
			graphs.c.type,
		).where(
			and_(
				graphs.c.branch == bindparam("branch"),
				or_(
					graphs.c.turn > bindparam("turn_from_a"),
					and_(
						graphs.c.turn == bindparam("turn_from_b"),
						graphs.c.tick >= bindparam("tick_from"),
					),
				),
				or_(
					graphs.c.turn < bindparam("turn_to_a"),
					and_(
						graphs.c.turn == bindparam("turn_to_b"),
						graphs.c.tick <= bindparam("tick_to"),
					),
				),
			)
		),
		"graphs_after": select(
			graphs.c.graph,
			graphs.c.turn,
			graphs.c.tick,
			graphs.c.type,
		).where(
			and_(
				graphs.c.branch == bindparam("branch"),
				or_(
					graphs.c.turn > bindparam("turn_from_a"),
					and_(
						graphs.c.turn == bindparam("turn_from_b"),
						graphs.c.tick >= bindparam("tick_from"),
					),
				),
			)
		),
		"main_branch_ends": select(
			branches.c.branch,
			branches.c.end_turn,
			branches.c.end_tick,
		).where(branches.c.parent == null()),
		"update_branches": branches.update()
		.values(
			parent=bindparam("parent"),
			parent_turn=bindparam("parent_turn"),
			parent_tick=bindparam("parent_tick"),
			end_turn=bindparam("end_turn"),
			end_tick=bindparam("end_tick"),
		)
		.where(branches.c.branch == bindparam("branch")),
		"update_turns": turns.update()
		.values(
			end_tick=bindparam("end_tick"),
			plan_end_tick=bindparam("plan_end_tick"),
		)
		.where(
			and_(
				turns.c.branch == bindparam("branch"),
				turns.c.turn == bindparam("turn"),
			)
		),
		"keyframes_graphs_list": select(
			keyframes_graphs.c.graph,
			keyframes_graphs.c.branch,
			keyframes_graphs.c.turn,
			keyframes_graphs.c.tick,
		),
		"all_graphs_in_keyframe": select(
			keyframes_graphs.c.graph,
			keyframes_graphs.c.nodes,
			keyframes_graphs.c.edges,
			keyframes_graphs.c.graph_val,
		)
		.where(
			and_(
				keyframes_graphs.c.branch == bindparam("branch"),
				keyframes_graphs.c.turn == bindparam("turn"),
				keyframes_graphs.c.tick == bindparam("tick"),
			)
		)
		.order_by(keyframes_graphs.c.graph),
		"get_keyframe_graph": select(
			keyframes_graphs.c.nodes,
			keyframes_graphs.c.edges,
			keyframes_graphs.c.graph_val,
		).where(
			and_(
				keyframes_graphs.c.graph == bindparam("graph"),
				keyframes_graphs.c.branch == bindparam("branch"),
				keyframes_graphs.c.turn == bindparam("turn"),
				keyframes_graphs.c.tick == bindparam("tick"),
			)
		),
		"delete_keyframe": keyframes.delete().where(
			and_(
				keyframes.c.branch == bindparam("branch"),
				keyframes.c.turn == bindparam("turn"),
				keyframes.c.tick == bindparam("tick"),
			)
		),
		"delete_keyframe_graph": keyframes_graphs.delete().where(
			and_(
				keyframes_graphs.c.graph == bindparam("graph"),
				keyframes_graphs.c.branch == bindparam("branch"),
				keyframes_graphs.c.turn == bindparam("turn"),
				keyframes_graphs.c.tick == bindparam("tick"),
			)
		),
		"load_graphs_tick_to_end": select(
			graphs.c.graph, graphs.c.turn, graphs.c.tick, graphs.c.type
		)
		.where(tick_to_end_clause(graphs))
		.order_by(graphs.c.turn, graphs.c.tick, graphs.c.graph),
		"load_graphs_tick_to_tick": select(
			graphs.c.graph, graphs.c.turn, graphs.c.tick, graphs.c.type
		)
		.where(tick_to_tick_clause(graphs))
		.order_by(graphs.c.turn, graphs.c.tick, graphs.c.graph),
		"load_nodes_tick_to_end": select(
			nodes.c.graph,
			nodes.c.node,
			nodes.c.turn,
			nodes.c.tick,
			nodes.c.extant,
		)
		.where(tick_to_end_clause(nodes))
		.order_by(
			nodes.c.turn,
			nodes.c.tick,
			nodes.c.graph,
			nodes.c.node,
		),
		"load_nodes_tick_to_tick": select(
			nodes.c.graph,
			nodes.c.node,
			nodes.c.turn,
			nodes.c.tick,
			nodes.c.extant,
		)
		.where(tick_to_tick_clause(nodes))
		.order_by(
			nodes.c.turn,
			nodes.c.tick,
			nodes.c.graph,
			nodes.c.node,
		),
		"load_edges_tick_to_end": select(
			edges.c.graph,
			edges.c.orig,
			edges.c.dest,
			edges.c.turn,
			edges.c.tick,
			edges.c.extant,
		)
		.where(tick_to_end_clause(edges))
		.order_by(
			edges.c.turn,
			edges.c.tick,
			edges.c.graph,
			edges.c.orig,
			edges.c.dest,
		),
		"load_edges_tick_to_tick": select(
			edges.c.graph,
			edges.c.orig,
			edges.c.dest,
			edges.c.turn,
			edges.c.tick,
			edges.c.extant,
		)
		.where(tick_to_tick_clause(edges))
		.order_by(
			edges.c.turn,
			edges.c.tick,
			edges.c.graph,
			edges.c.orig,
			edges.c.dest,
		),
		"load_node_val_tick_to_end": select(
			node_val.c.graph,
			node_val.c.node,
			node_val.c.key,
			node_val.c.turn,
			node_val.c.tick,
			node_val.c.value,
		)
		.where(tick_to_end_clause(node_val))
		.order_by(
			node_val.c.turn,
			node_val.c.tick,
			node_val.c.graph,
			node_val.c.node,
			node_val.c.key,
		),
		"load_node_val_tick_to_tick": select(
			node_val.c.graph,
			node_val.c.node,
			node_val.c.key,
			node_val.c.turn,
			node_val.c.tick,
			node_val.c.value,
		)
		.where(tick_to_tick_clause(node_val))
		.order_by(
			node_val.c.turn,
			node_val.c.tick,
			node_val.c.graph,
			node_val.c.node,
			node_val.c.key,
		),
		"load_edge_val_tick_to_end": select(
			edge_val.c.graph,
			edge_val.c.orig,
			edge_val.c.dest,
			edge_val.c.key,
			edge_val.c.turn,
			edge_val.c.tick,
			edge_val.c.value,
		)
		.where(tick_to_end_clause(edge_val))
		.order_by(
			edge_val.c.turn,
			edge_val.c.tick,
			edge_val.c.graph,
			edge_val.c.orig,
			edge_val.c.dest,
			edge_val.c.key,
		),
		"load_edge_val_tick_to_tick": select(
			edge_val.c.graph,
			edge_val.c.orig,
			edge_val.c.dest,
			edge_val.c.key,
			edge_val.c.turn,
			edge_val.c.tick,
			edge_val.c.value,
		)
		.where(tick_to_tick_clause(edge_val))
		.order_by(
			edge_val.c.turn,
			edge_val.c.tick,
			edge_val.c.graph,
			edge_val.c.orig,
			edge_val.c.dest,
			edge_val.c.key,
		),
		"load_graph_val_tick_to_end": select(
			graph_val.c.graph,
			graph_val.c.key,
			graph_val.c.turn,
			graph_val.c.tick,
			graph_val.c.value,
		)
		.where(tick_to_end_clause(graph_val))
		.order_by(
			graph_val.c.turn,
			graph_val.c.tick,
			graph_val.c.graph,
			graph_val.c.key,
		),
		"load_graph_val_tick_to_tick": select(
			graph_val.c.graph,
			graph_val.c.key,
			graph_val.c.turn,
			graph_val.c.tick,
			graph_val.c.value,
		)
		.where(tick_to_tick_clause(graph_val))
		.order_by(
			graph_val.c.turn,
			graph_val.c.tick,
			graph_val.c.graph,
			graph_val.c.key,
		),
	}

	for t in table.values():
		r["create_" + t.name] = CreateTable(t)
		r["truncate_" + t.name] = t.delete()
		key = list(t.primary_key)
		if (
			"branch" in t.columns
			and "turn" in t.columns
			and "tick" in t.columns
		):
			branch = t.columns["branch"]
			turn = t.columns["turn"]
			tick = t.columns["tick"]
			if branch in key and turn in key and tick in key:
				key = [branch, turn, tick]
				r[t.name + "_del_time"] = t.delete().where(
					and_(
						t.c.branch == bindparam("branch"),
						t.c.turn == bindparam("turn"),
						t.c.tick == bindparam("tick"),
					)
				)
		r[t.name + "_dump"] = select(*t.c.values()).order_by(*key)
		r[t.name + "_insert"] = t.insert().values(
			tuple(bindparam(cname) for cname in t.c.keys())
		)
		r[t.name + "_count"] = select(func.COUNT()).select_from(t)
		r[t.name + "_del"] = t.delete().where(
			and_(*[c == bindparam(c.name) for c in (t.primary_key or t.c)])
		)

	rulebooks = table["rulebooks"]
	r["rulebooks_update"] = update_where(
		["rules"],
		[
			rulebooks.c.rulebook,
			rulebooks.c.branch,
			rulebooks.c.turn,
			rulebooks.c.tick,
		],
	)

	for t in table.values():
		key = list(t.primary_key)
		if (
			"branch" in t.columns
			and "turn" in t.columns
			and "tick" in t.columns
		):
			branch = t.columns["branch"]
			turn = t.columns["turn"]
			tick = t.columns["tick"]
			if branch in key and turn in key and tick in key:
				key = [branch, turn, tick]
		r[t.name + "_dump"] = select(*t.c.values()).order_by(*key)
		r[t.name + "_insert"] = t.insert().values(
			tuple(bindparam(cname) for cname in t.c.keys())
		)
		r[t.name + "_count"] = select(func.COUNT("*")).select_from(t)
	things = table["things"]
	r["del_things_after"] = things.delete().where(
		and_(
			things.c.character == bindparam("character"),
			things.c.thing == bindparam("thing"),
			things.c.branch == bindparam("branch"),
			or_(
				things.c.turn > bindparam("turn"),
				and_(
					things.c.turn == bindparam("turn"),
					things.c.tick >= bindparam("tick"),
				),
			),
		)
	)
	units = table["units"]
	r["del_units_after"] = units.delete().where(
		and_(
			units.c.character_graph == bindparam("character"),
			units.c.unit_graph == bindparam("graph"),
			units.c.unit_node == bindparam("unit"),
			units.c.branch == bindparam("branch"),
			or_(
				units.c.turn > bindparam("turn"),
				and_(
					units.c.turn == bindparam("turn"),
					units.c.tick >= bindparam("tick"),
				),
			),
		)
	)
	bookmarks = table["bookmarks"]
	r["update_bookmark"] = (
		bookmarks.update()
		.where(bookmarks.c.key == bindparam("key"))
		.values(
			branch=bindparam("branch"),
			turn=bindparam("turn"),
			tick=bindparam("tick"),
		)
	)
	r["delete_bookmark"] = bookmarks.delete().where(
		bookmarks.c.key == bindparam("key")
	)

	def to_end_clause(tab: Table):
		return and_(
			tab.c.branch == bindparam("branch"),
			or_(
				tab.c.turn > bindparam("turn_from"),
				and_(
					tab.c.turn == bindparam("turn_from"),
					tab.c.tick >= bindparam("tick_from"),
				),
			),
		)

	def to_tick_clause(tab: Table):
		return and_(
			to_end_clause(tab),
			or_(
				tab.c.turn < bindparam("turn_to"),
				and_(
					tab.c.turn == bindparam("turn_to"),
					tab.c.tick <= bindparam("tick_to"),
				),
			),
		)

	r["load_things_tick_to_end"] = (
		select(
			things.c.character,
			things.c.thing,
			things.c.turn,
			things.c.tick,
			things.c.location,
		)
		.where(to_end_clause(things))
		.order_by(
			things.c.turn, things.c.tick, things.c.character, things.c.thing
		)
	)
	r["load_things_tick_to_tick"] = (
		select(
			things.c.character,
			things.c.thing,
			things.c.turn,
			things.c.tick,
			things.c.location,
		)
		.where(to_tick_clause(things))
		.order_by(
			things.c.turn, things.c.tick, things.c.character, things.c.thing
		)
	)

	units = table["units"]
	r["load_units_tick_to_end"] = (
		select(
			units.c.character_graph,
			units.c.unit_graph,
			units.c.unit_node,
			units.c.turn,
			units.c.tick,
			units.c.is_unit,
		)
		.where(to_end_clause(units))
		.order_by(
			units.c.turn,
			units.c.tick,
			units.c.character_graph,
			units.c.unit_graph,
			units.c.unit_node,
		)
	)
	r["load_units_tick_to_tick"] = (
		select(
			units.c.character_graph,
			units.c.unit_graph,
			units.c.unit_node,
			units.c.turn,
			units.c.tick,
			units.c.is_unit,
		)
		.where(to_tick_clause(units))
		.order_by(
			units.c.turn,
			units.c.tick,
			units.c.character_graph,
			units.c.unit_graph,
			units.c.unit_node,
		)
	)
	for name in (
		"character_rulebook",
		"unit_rulebook",
		"character_thing_rulebook",
		"character_place_rulebook",
		"character_portal_rulebook",
	):
		tab = table[name]
		sel = select(
			tab.c.character,
			tab.c.turn,
			tab.c.tick,
			tab.c.rulebook,
		)
		r[f"load_{name}_tick_to_end"] = sel.where(to_end_clause(tab)).order_by(
			tab.c.turn, tab.c.tick, tab.c.character
		)
		r[f"load_{name}_tick_to_tick"] = sel.where(
			to_tick_clause(tab)
		).order_by(tab.c.turn, tab.c.tick, tab.c.character)
		r[f"{name}_delete"] = tab.delete().where(
			and_(
				tab.c.character == bindparam("character"),
				tab.c.branch == bindparam("branch"),
				tab.c.turn == bindparam("turn"),
				tab.c.tick == bindparam("tick"),
			)
		)
	ntab = table["node_rulebook"]
	node_rb_select = select(
		ntab.c.character,
		ntab.c.node,
		ntab.c.turn,
		ntab.c.tick,
		ntab.c.rulebook,
	)
	r["load_node_rulebook_tick_to_end"] = node_rb_select.where(
		to_end_clause(ntab)
	).order_by(ntab.c.turn, ntab.c.tick, ntab.c.character, ntab.c.node)
	r["load_node_rulebook_tick_to_tick"] = node_rb_select.where(
		to_tick_clause(ntab)
	).order_by(ntab.c.turn, ntab.c.tick, ntab.c.character, ntab.c.node)
	ptab = table["portal_rulebook"]
	port_rb_select = select(
		ptab.c.character,
		ptab.c.orig,
		ptab.c.dest,
		ptab.c.turn,
		ptab.c.tick,
		ptab.c.rulebook,
	)
	r["load_portal_rulebook_tick_to_end"] = port_rb_select.where(
		to_end_clause(ptab)
	).order_by(
		ptab.c.turn, ptab.c.tick, ptab.c.character, ptab.c.orig, ptab.c.dest
	)
	r["load_portal_rulebook_tick_to_tick"] = port_rb_select.where(
		to_tick_clause(ptab)
	).order_by(
		ptab.c.turn, ptab.c.tick, ptab.c.character, ptab.c.orig, ptab.c.dest
	)

	univ = table["universals"]
	r["load_universals_tick_to_end"] = (
		select(univ.c.key, univ.c.turn, univ.c.tick, univ.c.value)
		.where(tick_to_end_clause(univ))
		.order_by(univ.c.turn, univ.c.tick)
	)
	r["load_universals_tick_to_tick"] = (
		select(univ.c.key, univ.c.turn, univ.c.tick, univ.c.value)
		.where(tick_to_tick_clause(univ))
		.order_by(univ.c.turn, univ.c.tick)
	)

	rbs = table["rulebooks"]
	rbsel = select(
		rbs.c.rulebook,
		rbs.c.turn,
		rbs.c.tick,
		rbs.c.rules,
		rbs.c.priority,
	)
	r["load_rulebooks_tick_to_end"] = rbsel.where(
		tick_to_end_clause(rbs)
	).order_by(rbs.c.turn, rbs.c.tick, rbs.c.rulebook)
	r["load_rulebooks_tick_to_tick"] = rbsel.where(
		tick_to_tick_clause(rbs)
	).order_by(rbs.c.turn, rbs.c.tick, rbs.c.rulebook)

	def rule_update_cond(t: Table) -> and_:
		return and_(
			t.c.rule == bindparam("rule"),
			t.c.branch == bindparam("branch"),
			t.c.turn == bindparam("turn"),
			t.c.tick == bindparam("tick"),
		)

	hood = table["rule_neighborhood"]
	r["rule_neighborhood_update"] = (
		hood.update()
		.where(rule_update_cond(hood))
		.values(neighborhood=bindparam("neighborhood"))
	)
	big = table["rule_big"]
	r["rule_big_update"] = (
		big.update().where(rule_update_cond(big)).values(big=bindparam("big"))
	)
	trig = table["rule_triggers"]
	r["rule_triggers_update"] = (
		trig.update()
		.where(rule_update_cond(trig))
		.values(triggers=bindparam("triggers"))
	)
	preq = table["rule_prereqs"]
	r["rule_prereqs_update"] = (
		preq.update()
		.where(rule_update_cond(preq))
		.values(prereqs=bindparam("prereqs"))
	)
	act = table["rule_actions"]
	r["rule_actions_update"] = (
		act.update()
		.where(rule_update_cond(act))
		.values(actions=bindparam("actions"))
	)
	hoodsel = select(
		hood.c.rule,
		hood.c.turn,
		hood.c.tick,
		hood.c.neighborhood,
	)
	r["load_rule_neighborhoods_tick_to_end"] = hoodsel.where(
		tick_to_end_clause(hood)
	).order_by(hood.c.turn, hood.c.tick, hood.c.rule)
	r["load_rule_neighborhoods_tick_to_tick"] = hoodsel.where(
		tick_to_tick_clause(hood)
	).order_by(hood.c.turn, hood.c.tick, hood.c.rule)
	bigsel = select(big.c.rule, big.c.turn, big.c.tick, big.c.big)
	r["load_rule_big_tick_to_end"] = bigsel.where(
		tick_to_end_clause(big)
	).order_by(big.c.turn, big.c.tick, big.c.rule)
	r["load_rule_big_tick_to_tick"] = bigsel.where(
		tick_to_tick_clause(big)
	).order_by(big.c.turn, big.c.tick, big.c.rule)
	trigsel = select(trig.c.rule, trig.c.turn, trig.c.tick, trig.c.triggers)
	r["load_rule_triggers_tick_to_end"] = trigsel.where(
		tick_to_end_clause(trig)
	).order_by(trig.c.turn, trig.c.tick, trig.c.rule)
	r["load_rule_triggers_tick_to_tick"] = trigsel.where(
		tick_to_tick_clause(trig)
	).order_by(trig.c.turn, trig.c.tick, trig.c.rule)
	preqsel = select(preq.c.rule, preq.c.turn, preq.c.tick, preq.c.prereqs)
	r["load_rule_prereqs_tick_to_end"] = preqsel.where(
		tick_to_end_clause(preq)
	).order_by(preq.c.turn, preq.c.tick, preq.c.rule)
	r["load_rule_prereqs_tick_to_tick"] = preqsel.where(
		tick_to_tick_clause(preq)
	).order_by(preq.c.turn, preq.c.tick, preq.c.rule)
	actsel = select(act.c.rule, act.c.turn, act.c.tick, act.c.actions)
	r["load_rule_actions_tick_to_end"] = actsel.where(
		tick_to_end_clause(act)
	).order_by(act.c.turn, act.c.tick, act.c.rule)
	r["load_rule_actions_tick_to_tick"] = actsel.where(
		tick_to_tick_clause(act)
	).order_by(act.c.turn, act.c.tick, act.c.rule)
	kf = keyframes

	def time_clause(tab):
		return and_(
			tab.c.branch == bindparam("branch"),
			tab.c.turn == bindparam("turn"),
			tab.c.tick == bindparam("tick"),
		)

	r["delete_from_keyframes"] = kf.delete().where(time_clause(kf))
	kfg = keyframes_graphs
	r["delete_from_keyframes_graphs"] = kfg.delete().where(time_clause(kfg))
	kfx = table["keyframe_extensions"]
	r["delete_from_keyframe_extensions"] = kfx.delete().where(time_clause(kfx))
	r["get_keyframe_extensions"] = select(
		kfx.c.universal,
		kfx.c.rule,
		kfx.c.rulebook,
	).where(time_clause(kfx))

	for handledtab in (
		"character_rules_handled",
		"unit_rules_handled",
		"character_thing_rules_handled",
		"character_place_rules_handled",
		"character_portal_rules_handled",
		"node_rules_handled",
		"portal_rules_handled",
	):
		ht = table[handledtab]
		r["del_{}_turn".format(handledtab)] = ht.delete().where(
			and_(
				ht.c.branch == bindparam("branch"),
				ht.c.turn == bindparam("turn"),
			)
		)

	branches = branches

	r["branch_children"] = select(branches.c.branch).where(
		branches.c.parent == bindparam("branch")
	)

	tc = table["turns_completed"]
	r["turns_completed_update"] = update_where(["turn"], [tc.c.branch])

	return r


meta = MetaData()
tables(meta)
