# This file is part of LiSE, a framework for life simulation games.
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
"""A script to generate the SQL needed for LiSE's database backend,
and output it in JSON form.

This uses sqlalchemy to describe the queries. It extends the module of
the same name in the ``allegedb`` package. If you change anything here,
you won't be able to use your changes until you put the generated JSON
where LiSE will look for it, as in:

``python3 sqlalchemy.py >sqlite.json``

"""

from collections import OrderedDict
from functools import partial
from json import dumps

from sqlalchemy import (
	Table,
	Column,
	ForeignKeyConstraint,
	select,
	bindparam,
	func,
	and_,
	or_,
	INT,
	TEXT,
	BOOLEAN,
	FLOAT,
	BLOB,
)
from sqlalchemy import MetaData
from sqlalchemy.sql.ddl import CreateTable, CreateIndex

from .allegedb import alchemy

BaseColumn = Column
Column = partial(BaseColumn, nullable=False)


def tables_for_meta(meta):
	"""Return a dictionary full of all the tables I need for LiSE. Use the
	provided metadata object.

	"""
	alchemy.tables_for_meta(meta)

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

	# The top level of the LiSE world model, the character. Includes
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
	# "Portal" is LiSE's term for an edge in any of the directed
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

	r = alchemy.queries_for_table_dict(table)

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
	things_to_end_clause = and_(
		things.c.character == bindparam("character"),
		things.c.branch == bindparam("branch"),
		or_(
			things.c.turn > bindparam("turn_from_a"),
			and_(
				things.c.turn == bindparam("turn_from_b"),
				things.c.tick >= bindparam("tick_from"),
			),
		),
	)
	r["load_things_tick_to_end"] = select(
		things.c.thing, things.c.turn, things.c.tick, things.c.location
	).where(things_to_end_clause)
	r["load_things_tick_to_tick"] = select(
		things.c.thing, things.c.turn, things.c.tick, things.c.location
	).where(
		and_(
			things_to_end_clause,
			or_(
				things.c.turn < bindparam("turn_to_a"),
				and_(
					things.c.turn == bindparam("turn_to_b"),
					things.c.tick <= bindparam("tick_to"),
				),
			),
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
	from sqlalchemy.sql.ddl import CreateTable, CreateIndex

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
