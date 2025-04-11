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
"""A script to generate the SQL needed for lisien's database backend,
and output it in JSON form.

This uses sqlalchemy to describe the queries. It extends the module of
the same name in the ``allegedb`` package. If you change anything here,
you won't be able to use your changes until you put the generated JSON
where lisien will look for it, as in:

``python3 sqlalchemy.py >sqlite.json``

"""

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
	or_,
	select,
)
from sqlalchemy.sql.ddl import CreateIndex, CreateTable

BaseColumn = Column
Column = partial(BaseColumn, nullable=False)


def tables_for_meta(meta):
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
		"graphs",
		meta,
		Column("graph", BLOB, primary_key=True),
		Column("branch", TEXT, primary_key=True),
		Column("turn", INT, primary_key=True),
		Column("tick", INT, primary_key=True),
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
		Column("graph", BLOB, primary_key=True),
		Column(
			"branch",
			TEXT,
			primary_key=True,
			default="trunk",
		),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
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
		Column("graph", BLOB, primary_key=True),
		Column("key", BLOB, primary_key=True),
		Column(
			"branch",
			TEXT,
			ForeignKey("branches.branch"),
			primary_key=True,
			default="trunk",
		),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("value", BLOB),
		sqlite_with_rowid=False,
	)
	Table(
		"nodes",
		meta,
		Column("graph", BLOB, primary_key=True),
		Column("node", BLOB, primary_key=True),
		Column(
			"branch",
			TEXT,
			ForeignKey("branches.branch"),
			primary_key=True,
			default="trunk",
		),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("extant", BOOLEAN),
		sqlite_with_rowid=False,
	)
	Table(
		"node_val",
		meta,
		Column("graph", BLOB, primary_key=True),
		Column("node", BLOB, primary_key=True),
		Column("key", BLOB, primary_key=True),
		Column(
			"branch",
			TEXT,
			ForeignKey("branches.branch"),
			primary_key=True,
			default="trunk",
		),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("value", BLOB),
		sqlite_with_rowid=False,
	)
	Table(
		"edges",
		meta,
		Column("graph", BLOB, primary_key=True),
		Column("orig", BLOB, primary_key=True),
		Column("dest", BLOB, primary_key=True),
		Column("idx", INT, primary_key=True),
		Column(
			"branch",
			TEXT,
			ForeignKey("branches.branch"),
			primary_key=True,
			default="trunk",
		),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("extant", BOOLEAN),
		sqlite_with_rowid=False,
	)
	Table(
		"edge_val",
		meta,
		Column("graph", BLOB, primary_key=True),
		Column("orig", BLOB, primary_key=True),
		Column("dest", BLOB, primary_key=True),
		Column("idx", INT, primary_key=True),
		Column("key", BLOB, primary_key=True),
		Column(
			"branch",
			TEXT,
			ForeignKey("branches.branch"),
			primary_key=True,
			default="trunk",
		),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
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
		Column("turn", INT, primary_key=True),
		Column("tick", INT, primary_key=True),
		ForeignKeyConstraint(("plan_id",), ("plans.id",)),
		sqlite_with_rowid=False,
	)

	# Table for global variables that are not sensitive to sim-time.
	Table(
		"universals",
		meta,
		Column("key", BLOB, primary_key=True),
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
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
		Column("rulebook", BLOB, primary_key=True),
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("rules", BLOB, default=b"\x90"),  # empty array
		Column("priority", FLOAT, default=0.0),
		sqlite_with_rowid=False,
	)

	# Table for rules' triggers, those functions that return True only
	# when their rule should run (or at least check its prereqs).
	Table(
		"rule_triggers",
		meta,
		Column("rule", TEXT, primary_key=True),
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("triggers", BLOB, default=b"\x90"),
		ForeignKeyConstraint(("rule",), ["rules.rule"]),
		sqlite_with_rowid=False,
	)

	# Table for rules' neighborhoods, which govern when triggers should be
	# checked -- when that makes sense. Basically just rules on character.place
	Table(
		"rule_neighborhood",
		meta,
		Column("rule", TEXT, primary_key=True),
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("neighborhood", BLOB, default=b"\xc0"),
		ForeignKeyConstraint(("rule",), ["rules.rule"]),
		sqlite_with_rowid=False,
	)

	# Table for rules' prereqs, functions with veto power over a rule
	# being followed
	Table(
		"rule_prereqs",
		meta,
		Column("rule", TEXT, primary_key=True),
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("prereqs", BLOB, default=b"\x90"),
		ForeignKeyConstraint(("rule",), ["rules.rule"]),
		sqlite_with_rowid=False,
	)

	# Table for rules' actions, the functions that do what the rule
	# does.
	Table(
		"rule_actions",
		meta,
		Column("rule", TEXT, primary_key=True),
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("actions", BLOB, default=b"\x90"),
		ForeignKeyConstraint(("rule",), ["rules.rule"]),
		sqlite_with_rowid=False,
	)

	# Table indicating which rules make big changes to the world.
	Table(
		"rule_big",
		meta,
		Column("rule", TEXT, primary_key=True),
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
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
			Column("character", BLOB, primary_key=True),
			Column("branch", TEXT, primary_key=True, default="trunk"),
			Column("turn", INT, primary_key=True, default=0),
			Column("tick", INT, primary_key=True, default=0),
			Column("rulebook", BLOB),
			sqlite_with_rowid=False,
		)

	# Rules handled within the rulebook associated with one node in
	# particular.
	nrh = Table(
		"node_rules_handled",
		meta,
		Column("character", BLOB, primary_key=True),
		Column("node", BLOB, primary_key=True),
		Column("rulebook", BLOB, primary_key=True),
		Column("rule", TEXT, primary_key=True),
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT),
		sqlite_with_rowid=False,
	)

	# Rules handled within the rulebook associated with one portal in
	# particular.
	porh = Table(
		"portal_rules_handled",
		meta,
		Column("character", BLOB, primary_key=True),
		Column("orig", BLOB, primary_key=True),
		Column("dest", BLOB, primary_key=True),
		Column("rulebook", BLOB, primary_key=True),
		Column("rule", TEXT, primary_key=True),
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
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
		Column("character", BLOB, primary_key=True),
		Column("thing", BLOB, primary_key=True),
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		# when location is null, this node is not a thing, but a place
		Column("location", BLOB),
		sqlite_with_rowid=False,
	)

	# The rulebook followed by a given node.
	Table(
		"node_rulebook",
		meta,
		Column("character", BLOB, primary_key=True),
		Column("node", BLOB, primary_key=True),
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
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
		Column("character", BLOB, primary_key=True),
		Column("orig", BLOB, primary_key=True),
		Column("dest", BLOB, primary_key=True),
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
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
		Column("character_graph", BLOB, primary_key=True),
		Column("unit_graph", BLOB, primary_key=True),
		Column("unit_node", BLOB, primary_key=True),
		Column("branch", TEXT, primary_key=True, default="trunk"),
		Column("turn", INT, primary_key=True, default=0),
		Column("tick", INT, primary_key=True, default=0),
		Column("is_unit", BOOLEAN),
		sqlite_with_rowid=False,
	)

	crh = Table(
		"character_rules_handled",
		meta,
		Column("character", BLOB),
		Column("rulebook", BLOB),
		Column("rule", TEXT),
		Column("branch", TEXT, default="trunk"),
		Column("turn", INT),
		Column("tick", INT),
		sqlite_with_rowid=True,
	)

	arh = Table(
		"unit_rules_handled",
		meta,
		Column("character", BLOB),
		Column("graph", BLOB),
		Column("unit", BLOB),
		Column("rulebook", BLOB),
		Column("rule", TEXT),
		Column("branch", TEXT, default="trunk"),
		Column("turn", INT),
		Column("tick", INT),
		sqlite_with_rowid=True,
	)

	ctrh = Table(
		"character_thing_rules_handled",
		meta,
		Column("character", BLOB),
		Column("thing", BLOB),
		Column("rulebook", BLOB),
		Column("rule", TEXT),
		Column("branch", TEXT, default="trunk"),
		Column("turn", INT),
		Column("tick", INT),
		sqlite_with_rowid=True,
	)

	cprh = Table(
		"character_place_rules_handled",
		meta,
		Column("character", BLOB),
		Column("place", BLOB),
		Column("rulebook", BLOB),
		Column("rule", TEXT),
		Column("branch", TEXT, default="trunk"),
		Column("turn", INT),
		Column("tick", INT),
		sqlite_with_rowid=True,
	)

	cporh = Table(
		"character_portal_rules_handled",
		meta,
		Column("character", BLOB),
		Column("orig", BLOB),
		Column("dest", BLOB),
		Column("rulebook", BLOB),
		Column("rule", TEXT),
		Column("branch", TEXT, default="trunk"),
		Column("turn", INT),
		Column("tick", INT),
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


def indices_for_table_dict(table):
	return {}


def queries(table):
	"""Given dictionaries of tables and view-queries, return a dictionary
	of all the rest of the queries I need.

	"""

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

	r = {
		"global_get": select(table["global"].c.value).where(
			table["global"].c.key == bindparam("key")
		),
		"global_update": table["global"]
		.update()
		.values(value=bindparam("value"))
		.where(table["global"].c.key == bindparam("key")),
		"graph_type": select(table["graphs"].c.type).where(
			table["graphs"].c.graph == bindparam("graph")
		),
		"del_edge_val_after": table["edge_val"]
		.delete()
		.where(
			and_(
				table["edge_val"].c.graph == bindparam("graph"),
				table["edge_val"].c.orig == bindparam("orig"),
				table["edge_val"].c.dest == bindparam("dest"),
				table["edge_val"].c.idx == bindparam("idx"),
				table["edge_val"].c.key == bindparam("key"),
				table["edge_val"].c.branch == bindparam("branch"),
				or_(
					table["edge_val"].c.turn > bindparam("turn"),
					and_(
						table["edge_val"].c.turn == bindparam("turn"),
						table["edge_val"].c.tick >= bindparam("tick"),
					),
				),
			)
		),
		"del_edges_graph": table["edges"]
		.delete()
		.where(table["edges"].c.graph == bindparam("graph")),
		"del_edges_after": table["edges"]
		.delete()
		.where(
			and_(
				table["edges"].c.graph == bindparam("graph"),
				table["edges"].c.orig == bindparam("orig"),
				table["edges"].c.dest == bindparam("dest"),
				table["edges"].c.idx == bindparam("idx"),
				table["edges"].c.branch == bindparam("branch"),
				or_(
					table["edges"].c.turn > bindparam("turn"),
					and_(
						table["edges"].c.turn == bindparam("turn"),
						table["edges"].c.tick >= bindparam("tick"),
					),
				),
			)
		),
		"del_nodes_after": table["nodes"]
		.delete()
		.where(
			and_(
				table["nodes"].c.graph == bindparam("graph"),
				table["nodes"].c.node == bindparam("node"),
				table["nodes"].c.branch == bindparam("branch"),
				or_(
					table["nodes"].c.turn > bindparam("turn"),
					and_(
						table["nodes"].c.turn == bindparam("turn"),
						table["nodes"].c.tick >= bindparam("tick"),
					),
				),
			)
		),
		"del_node_val_after": table["node_val"]
		.delete()
		.where(
			and_(
				table["node_val"].c.graph == bindparam("graph"),
				table["node_val"].c.node == bindparam("node"),
				table["node_val"].c.key == bindparam("key"),
				table["node_val"].c.branch == bindparam("branch"),
				or_(
					table["node_val"].c.turn > bindparam("turn"),
					and_(
						table["node_val"].c.turn == bindparam("turn"),
						table["node_val"].c.tick >= bindparam("tick"),
					),
				),
			)
		),
		"del_graph_val_after": table["graph_val"]
		.delete()
		.where(
			and_(
				table["graph_val"].c.graph == bindparam("graph"),
				table["graph_val"].c.key == bindparam("key"),
				table["graph_val"].c.branch == bindparam("branch"),
				or_(
					table["graph_val"].c.turn > bindparam("turn"),
					and_(
						table["graph_val"].c.turn == bindparam("turn"),
						table["graph_val"].c.tick >= bindparam("tick"),
					),
				),
			)
		),
		"global_delete": table["global"]
		.delete()
		.where(table["global"].c.key == bindparam("key")),
		"graphs_types": select(
			table["graphs"].c.graph, table["graphs"].c.type
		),
		"graphs_named": select(func.COUNT())
		.select_from(table["graphs"])
		.where(table["graphs"].c.graph == bindparam("graph")),
		"graphs_between": select(
			table["graphs"].c.graph,
			table["graphs"].c.turn,
			table["graphs"].c.tick,
			table["graphs"].c.type,
		).where(
			and_(
				table["graphs"].c.branch == bindparam("branch"),
				or_(
					table["graphs"].c.turn > bindparam("turn_from_a"),
					and_(
						table["graphs"].c.turn == bindparam("turn_from_b"),
						table["graphs"].c.tick >= bindparam("tick_from"),
					),
				),
				or_(
					table["graphs"].c.turn < bindparam("turn_to_a"),
					and_(
						table["graphs"].c.turn == bindparam("turn_to_b"),
						table["graphs"].c.tick <= bindparam("tick_to"),
					),
				),
			)
		),
		"graphs_after": select(
			table["graphs"].c.graph,
			table["graphs"].c.turn,
			table["graphs"].c.tick,
			table["graphs"].c.type,
		).where(
			and_(
				table["graphs"].c.branch == bindparam("branch"),
				or_(
					table["graphs"].c.turn > bindparam("turn_from_a"),
					and_(
						table["graphs"].c.turn == bindparam("turn_from_b"),
						table["graphs"].c.tick >= bindparam("tick_from"),
					),
				),
			)
		),
		"update_branches": table["branches"]
		.update()
		.values(
			parent=bindparam("parent"),
			parent_turn=bindparam("parent_turn"),
			parent_tick=bindparam("parent_tick"),
			end_turn=bindparam("end_turn"),
			end_tick=bindparam("end_tick"),
		)
		.where(table["branches"].c.branch == bindparam("branch")),
		"update_turns": table["turns"]
		.update()
		.values(
			end_tick=bindparam("end_tick"),
			plan_end_tick=bindparam("plan_end_tick"),
		)
		.where(
			and_(
				table["turns"].c.branch == bindparam("branch"),
				table["turns"].c.turn == bindparam("turn"),
			)
		),
		"keyframes_graphs_list": select(
			table["keyframes_graphs"].c.graph,
			table["keyframes_graphs"].c.branch,
			table["keyframes_graphs"].c.turn,
			table["keyframes_graphs"].c.tick,
		),
		"all_graphs_in_keyframe": select(
			table["keyframes_graphs"].c.graph,
			table["keyframes_graphs"].c.nodes,
			table["keyframes_graphs"].c.edges,
			table["keyframes_graphs"].c.graph_val,
		)
		.where(
			and_(
				table["keyframes_graphs"].c.branch == bindparam("branch"),
				table["keyframes_graphs"].c.turn == bindparam("turn"),
				table["keyframes_graphs"].c.tick == bindparam("tick"),
			)
		)
		.order_by(table["keyframes_graphs"].c.graph),
		"get_keyframe_graph": select(
			table["keyframes_graphs"].c.nodes,
			table["keyframes_graphs"].c.edges,
			table["keyframes_graphs"].c.graph_val,
		).where(
			and_(
				table["keyframes_graphs"].c.graph == bindparam("graph"),
				table["keyframes_graphs"].c.branch == bindparam("branch"),
				table["keyframes_graphs"].c.turn == bindparam("turn"),
				table["keyframes_graphs"].c.tick == bindparam("tick"),
			)
		),
		"delete_keyframe_graph": table["keyframes_graphs"]
		.delete()
		.where(
			and_(
				table["keyframes_graphs"].c.graph == bindparam("graph"),
				table["keyframes_graphs"].c.branch == bindparam("branch"),
				table["keyframes_graphs"].c.turn == bindparam("turn"),
				table["keyframes_graphs"].c.tick == bindparam("tick"),
			)
		),
		"load_nodes_tick_to_end": select(
			table["nodes"].c.graph,
			table["nodes"].c.node,
			table["nodes"].c.turn,
			table["nodes"].c.tick,
			table["nodes"].c.extant,
		).where(tick_to_end_clause(table["nodes"])),
		"load_nodes_tick_to_tick": select(
			table["nodes"].c.graph,
			table["nodes"].c.node,
			table["nodes"].c.turn,
			table["nodes"].c.tick,
			table["nodes"].c.extant,
		).where(tick_to_tick_clause(table["nodes"])),
		"load_edges_tick_to_end": select(
			table["edges"].c.graph,
			table["edges"].c.orig,
			table["edges"].c.dest,
			table["edges"].c.idx,
			table["edges"].c.turn,
			table["edges"].c.tick,
			table["edges"].c.extant,
		).where(tick_to_end_clause(table["edges"])),
		"load_edges_tick_to_tick": select(
			table["edges"].c.graph,
			table["edges"].c.orig,
			table["edges"].c.dest,
			table["edges"].c.idx,
			table["edges"].c.turn,
			table["edges"].c.tick,
			table["edges"].c.extant,
		).where(tick_to_tick_clause(table["edges"])),
		"load_node_val_tick_to_end": select(
			table["node_val"].c.graph,
			table["node_val"].c.node,
			table["node_val"].c.key,
			table["node_val"].c.turn,
			table["node_val"].c.tick,
			table["node_val"].c.value,
		).where(tick_to_end_clause(table["node_val"])),
		"load_node_val_tick_to_tick": select(
			table["node_val"].c.graph,
			table["node_val"].c.node,
			table["node_val"].c.key,
			table["node_val"].c.turn,
			table["node_val"].c.tick,
			table["node_val"].c.value,
		).where(tick_to_tick_clause(table["node_val"])),
		"load_edge_val_tick_to_end": select(
			table["edge_val"].c.graph,
			table["edge_val"].c.orig,
			table["edge_val"].c.dest,
			table["edge_val"].c.idx,
			table["edge_val"].c.key,
			table["edge_val"].c.turn,
			table["edge_val"].c.tick,
			table["edge_val"].c.value,
		).where(tick_to_end_clause(table["edge_val"])),
		"load_edge_val_tick_to_tick": select(
			table["edge_val"].c.graph,
			table["edge_val"].c.orig,
			table["edge_val"].c.dest,
			table["edge_val"].c.idx,
			table["edge_val"].c.key,
			table["edge_val"].c.turn,
			table["edge_val"].c.tick,
			table["edge_val"].c.value,
		).where(tick_to_tick_clause(table["edge_val"])),
		"load_graph_val_tick_to_end": select(
			table["graph_val"].c.graph,
			table["graph_val"].c.key,
			table["graph_val"].c.turn,
			table["graph_val"].c.tick,
			table["graph_val"].c.value,
		).where(tick_to_end_clause(table["graph_val"])),
		"load_graph_val_tick_to_tick": select(
			table["graph_val"].c.graph,
			table["graph_val"].c.key,
			table["graph_val"].c.turn,
			table["graph_val"].c.tick,
			table["graph_val"].c.value,
		).where(tick_to_tick_clause(table["graph_val"])),
	}
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

	r["load_things_tick_to_end"] = select(
		things.c.character,
		things.c.thing,
		things.c.turn,
		things.c.tick,
		things.c.location,
	).where(to_end_clause(things))
	r["load_things_tick_to_tick"] = select(
		things.c.character,
		things.c.thing,
		things.c.turn,
		things.c.tick,
		things.c.location,
	).where(to_tick_clause(things))
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
		r[f"load_{name}_tick_to_end"] = sel.where(to_end_clause(tab))
		r[f"load_{name}_tick_to_tick"] = sel.where(to_tick_clause(tab))
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
	)
	r["load_node_rulebook_tick_to_tick"] = node_rb_select.where(
		to_tick_clause(ntab)
	)
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
	)
	r["load_portal_rulebook_tick_to_tick"] = port_rb_select.where(
		to_tick_clause(ptab)
	)

	def generic_tick_to_end_clause(tab: Table):
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

	def generic_tick_to_tick_clause(tab: Table):
		return and_(
			generic_tick_to_end_clause(tab),
			or_(
				tab.c.turn < bindparam("turn_to"),
				and_(
					tab.c.turn == bindparam("turn_to"),
					tab.c.tick <= bindparam("tick_to"),
				),
			),
		)

	univ = table["universals"]
	r["load_universals_tick_to_end"] = select(
		univ.c.key, univ.c.turn, univ.c.tick, univ.c.value
	).where(generic_tick_to_end_clause(univ))
	r["load_universals_tick_to_tick"] = select(
		univ.c.key, univ.c.turn, univ.c.tick, univ.c.value
	).where(generic_tick_to_tick_clause(univ))

	rbs = table["rulebooks"]
	rbsel = select(
		rbs.c.rulebook,
		rbs.c.turn,
		rbs.c.tick,
		rbs.c.rules,
		rbs.c.priority,
	)
	r["load_rulebooks_tick_to_end"] = rbsel.where(
		generic_tick_to_end_clause(rbs)
	)
	r["load_rulebooks_tick_to_tick"] = rbsel.where(
		generic_tick_to_tick_clause(rbs)
	)

	hood = table["rule_neighborhood"]
	trig = table["rule_triggers"]
	preq = table["rule_prereqs"]
	act = table["rule_actions"]
	hoodsel = select(
		hood.c.rule,
		hood.c.turn,
		hood.c.tick,
		hood.c.neighborhood,
	)
	r["load_rule_neighborhoods_tick_to_end"] = hoodsel.where(
		generic_tick_to_end_clause(hood)
	)
	r["load_rule_neighborhoods_tick_to_tick"] = hoodsel.where(
		generic_tick_to_tick_clause(hood)
	)
	big = table["rule_big"]
	bigsel = select(big.c.rule, big.c.turn, big.c.tick, big.c.big)
	r["load_rule_big_tick_to_end"] = bigsel.where(
		generic_tick_to_end_clause(big)
	)
	r["load_rule_big_tick_to_tick"] = bigsel.where(
		generic_tick_to_tick_clause(big)
	)
	trigsel = select(trig.c.rule, trig.c.turn, trig.c.tick, trig.c.triggers)
	r["load_rule_triggers_tick_to_end"] = trigsel.where(
		generic_tick_to_end_clause(trig)
	)
	r["load_rule_triggers_tick_to_tick"] = trigsel.where(
		generic_tick_to_tick_clause(trig)
	)
	preqsel = select(preq.c.rule, preq.c.turn, preq.c.tick, preq.c.prereqs)
	r["load_rule_prereqs_tick_to_end"] = preqsel.where(
		generic_tick_to_end_clause(preq)
	)
	r["load_rule_prereqs_tick_to_tick"] = preqsel.where(
		generic_tick_to_tick_clause(preq)
	)
	actsel = select(act.c.rule, act.c.turn, act.c.tick, act.c.actions)
	r["load_rule_actions_tick_to_end"] = actsel.where(
		generic_tick_to_end_clause(act)
	)
	r["load_rule_actions_tick_to_tick"] = actsel.where(
		generic_tick_to_tick_clause(act)
	)
	kfx = table["keyframe_extensions"]
	r["get_keyframe_extensions"] = select(
		kfx.c.universal,
		kfx.c.rule,
		kfx.c.rulebook,
	).where(
		and_(
			kfx.c.branch == bindparam("branch"),
			kfx.c.turn == bindparam("turn"),
			kfx.c.tick == bindparam("tick"),
		)
	)

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

	branches = table["branches"]

	r["branch_children"] = select(branches.c.branch).where(
		branches.c.parent == bindparam("branch")
	)

	tc = table["turns_completed"]
	r["turns_completed_update"] = update_where(["turn"], [tc.c.branch])

	return r


def gather_sql(meta):
	r = {}
	table = tables_for_meta(meta)
	index = indices_for_table_dict(table)
	query = queries(table)

	for t in table.values():
		r["create_" + t.name] = CreateTable(t)
		r["truncate_" + t.name] = t.delete()
	for tab, idx in index.items():
		r["index_" + tab] = CreateIndex(idx)
	r.update(query)

	return r


meta = MetaData()
table = tables_for_meta(meta)

if __name__ == "__main__":
	from sqlalchemy.dialects.sqlite.pysqlite import SQLiteDialect_pysqlite

	r = {}
	dia = SQLiteDialect_pysqlite()
	for n, t in table.items():
		r["create_" + n] = str(CreateTable(t).compile(dialect=dia))
		r["truncate_" + n] = str(t.delete().compile(dialect=dia))
	index = indices_for_table_dict(table)
	for n, x in index.items():
		r["index_" + n] = str(CreateIndex(x).compile(dialect=dia))
	query = queries(table)
	for n, q in query.items():
		r[n] = str(q.compile(dialect=dia))
	print(dumps(r, sort_keys=True, indent=4))
