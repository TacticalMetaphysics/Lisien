# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
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
from sqlalchemy import Table, Column, ForeignKeyConstraint, select, bindparam, func, and_, or_, INT, TEXT, BOOLEAN
from sqlalchemy.sql.ddl import CreateTable, CreateIndex


BaseColumn = Column
Column = partial(BaseColumn, nullable=False)


from json import dumps

from .allegedb import alchemy


def tables_for_meta(meta):
    """Return a dictionary full of all the tables I need for LiSE. Use the
    provided metadata object.

    """
    alchemy.tables_for_meta(meta)

    # Table for global variables that are not sensitive to sim-time.
    Table(
        'universals', meta,
        Column('key', TEXT, primary_key=True),
        Column(
            'branch', TEXT, primary_key=True, default='trunk'
        ),
        Column('turn', INT, primary_key=True, default=0),
        Column('tick', INT, primary_key=True, default=0),
        Column('value', TEXT)
    )

    Table(
        'rules', meta,
        Column('rule', TEXT, primary_key=True)
    )

    # Table grouping rules into lists called rulebooks.
    Table(
        'rulebooks', meta,
        Column('rulebook', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True, default='trunk'),
        Column('turn', INT, primary_key=True, default=0),
        Column('tick', INT, primary_key=True, default=0),
        Column('rules', TEXT, default='[]')
    )

    # Table for rules' triggers, those functions that return True only
    # when their rule should run (or at least check its prereqs).
    Table(
        'rule_triggers', meta,
        Column('rule', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True, default='trunk'),
        Column('turn', INT, primary_key=True, default=0),
        Column('tick', INT, primary_key=True, default=0),
        Column('triggers', TEXT, default='[]'),
        ForeignKeyConstraint(
            ['rule'], ['rules.rule']
        )
    )

    # Table for rules' prereqs, functions with veto power over a rule
    # being followed
    Table(
        'rule_prereqs', meta,
        Column('rule', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True, default='trunk'),
        Column('turn', INT, primary_key=True, default=0),
        Column('tick', INT, primary_key=True, default=0),
        Column('prereqs', TEXT, default='[]'),
        ForeignKeyConstraint(
            ['rule'], ['rules.rule']
        )
    )

    # Table for rules' actions, the functions that do what the rule
    # does.
    Table(
        'rule_actions', meta,
        Column('rule', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True, default='trunk'),
        Column('turn', INT, primary_key=True, default=0),
        Column('tick', INT, primary_key=True, default=0),
        Column('actions', TEXT, default='[]'),
        ForeignKeyConstraint(
            ['rule'], ['rules.rule']
        )
    )

    # The top level of the LiSE world model, the character. Includes
    # rulebooks for the character itself, its avatars, and all the things,
    # places, and portals it contains--though those may have their own
    # rulebooks as well.

    for name in (
        'character_rulebook',
        'avatar_rulebook',
        'character_thing_rulebook',
        'character_place_rulebook',
        'character_portal_rulebook'
    ):
        Table(
            name, meta,
            Column('character', TEXT, primary_key=True),
            Column('branch', TEXT, primary_key=True, default='trunk'),
            Column('turn', INT, primary_key=True, default=0),
            Column('tick', INT, primary_key=True, default=0),
            Column('rulebook', TEXT),
            ForeignKeyConstraint(
                ['character'], ['graphs.graph']
            ),
            ForeignKeyConstraint(
                ['rulebook'], ['rulebooks.rulebook']
            )
        )

    # Rules handled within the rulebook associated with one node in
    # particular.
    nrh = Table(
        'node_rules_handled', meta,
        Column('character', TEXT, primary_key=True),
        Column('node', TEXT, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True, default='trunk'),
        Column('turn', INT, primary_key=True, default=0),
        Column('tick', INT),
        ForeignKeyConstraint(
            ['character', 'node'], ['nodes.graph', 'nodes.node']
        )
    )

    Table(
        'node_rules_changes', meta,
        Column('character', TEXT, primary_key=True),
        Column('node', TEXT, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True),
        Column('turn', INT, primary_key=True),
        Column('tick', INT, primary_key=True),
        Column('handled_branch', TEXT),
        Column('handled_turn', INT),
        ForeignKeyConstraint(
            ['character', 'node', 'rulebook', 'rule', 'handled_branch', 'handled_turn'],
            [nrh.c.character, nrh.c.node, nrh.c.rulebook, nrh.c.rule, nrh.c.branch, nrh.c.turn]
        )
    )

    # Rules handled within the rulebook associated with one portal in
    # particular.
    porh = Table(
        'portal_rules_handled', meta,
        Column('character', TEXT, primary_key=True),
        Column('orig', TEXT, primary_key=True),
        Column('dest', TEXT, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True, default='trunk'),
        Column('turn', INT, primary_key=True, default=0),
        Column('tick', INT),
        ForeignKeyConstraint(
            ['character', 'orig', 'dest'], ['edges.graph', 'edges.orig', 'edges.dest']
        )
    )

    Table(
        'portal_rules_changes', meta,
        Column('character', TEXT, primary_key=True),
        Column('orig', TEXT, primary_key=True),
        Column('dest', TEXT, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True),
        Column('turn', INT, primary_key=True),
        Column('tick', INT, primary_key=True),
        Column('handled_branch', TEXT),
        Column('handled_turn', INT),
        ForeignKeyConstraint(
            ['character', 'orig', 'dest', 'rulebook', 'rule', 'handled_branch', 'handled_turn'],
            [porh.c.character, porh.c.orig, porh.c.dest, porh.c.rulebook, porh.c.rule, porh.c.branch, porh.c.turn]
        )
    )

    # The function to use for a given sense.
    #
    # Characters use senses to look at other characters. To model this,
    # sense functions are called with a facade representing the
    # character under observation; the function munges this facade to
    # make it look as it does through the sense in question, and returns
    # that.
    Table(
        'senses', meta,
        # null character field means all characters have this sense
        Column(
            'character', TEXT, primary_key=True, nullable=True
        ),
        Column('sense', TEXT, primary_key=True),
        Column(
            'branch', TEXT, primary_key=True, default='trunk'
        ),
        Column('turn', INT, primary_key=True, default=0),
        Column('tick', INT, primary_key=True, default=0),
        Column('function', TEXT, nullable=True),
        ForeignKeyConstraint(['character'], ['graphs.graph'])
    )

    # Table for Things, being those nodes in a Character graph that have
    # locations.
    #
    # A Thing's location can be either a Place or another Thing, as long
    # as it's in the same Character.
    Table(
        'things', meta,
        Column('character', TEXT, primary_key=True),
        Column('thing', TEXT, primary_key=True),
        Column(
            'branch', TEXT, primary_key=True, default='trunk'
        ),
        Column('turn', INT, primary_key=True, default=0),
        Column('tick', INT, primary_key=True, default=0),
        # when location is null, this node is not a thing, but a place
        Column('location', TEXT),
        ForeignKeyConstraint(
            ['character', 'thing'], ['nodes.graph', 'nodes.node']
        ),
        ForeignKeyConstraint(
            ['character', 'location'], ['nodes.graph', 'nodes.node']
        )
    )

    # The rulebook followed by a given node.
    Table(
        'node_rulebook', meta,
        Column('character', TEXT, primary_key=True),
        Column('node', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True, default='trunk'),
        Column('turn', INT, primary_key=True, default=0),
        Column('tick', INT, primary_key=True, default=0),
        Column('rulebook', TEXT),
        ForeignKeyConstraint(
            ['character', 'node'], ['nodes.graph', 'nodes.node']
        )
    )

    # The rulebook followed by a given Portal.
    #
    # "Portal" is LiSE's term for an edge in any of the directed
    # graphs it uses. The name is different to distinguish them from
    # Edge objects, which exist in an underlying object-relational
    # mapper called allegedb, and have a different API.
    Table(
        'portal_rulebook', meta,
        Column('character', TEXT, primary_key=True),
        Column('orig', TEXT, primary_key=True),
        Column('dest', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True, default='trunk'),
        Column('turn', INT, primary_key=True, default=0),
        Column('tick', INT, primary_key=True, default=0),
        Column('rulebook', TEXT),
        ForeignKeyConstraint(
            ['character', 'orig', 'dest'],
            ['edges.graph', 'edges.orig', 'edges.dest']
        )
    )

    # The avatars representing one Character in another.
    #
    # In the common situation where a Character, let's say Alice has her
    # own stats and skill tree and social graph, and also has a location
    # in physical space, you can represent this by creating a Thing in
    # the Character that represents physical space, and then making that
    # Thing an avatar of Alice. On its own this doesn't do anything,
    # it's just a convenient way of indicating the relation -- but if
    # you like, you can make rules that affect all avatars of some
    # Character, irrespective of what Character the avatar is actually
    # *in*.
    Table(
        'avatars', meta,
        Column('character_graph', TEXT, primary_key=True),
        Column('avatar_graph', TEXT, primary_key=True),
        Column('avatar_node', TEXT, primary_key=True),
        Column(
            'branch', TEXT, primary_key=True, default='trunk'
        ),
        Column('turn', INT, primary_key=True, default=0),
        Column('tick', INT, primary_key=True, default=0),
        Column('is_avatar', BOOLEAN),
        ForeignKeyConstraint(['character_graph'], ['graphs.graph']),
        ForeignKeyConstraint(
            ['avatar_graph', 'avatar_node'],
            ['nodes.graph', 'nodes.node']
        )
    )

    crh = Table(
        'character_rules_handled', meta,
        Column('character', TEXT, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True, default='trunk'),
        Column('turn', INT, primary_key=True),
        Column('tick', INT),
        ForeignKeyConstraint(
            ['character', 'rulebook'], ['character_rulebook.character', 'character_rulebook.rulebook']
        )
    )

    Table(
        'character_rules_changes', meta,
        Column('character', TEXT, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True),
        Column('turn', INT, primary_key=True),
        Column('tick', INT, primary_key=True),
        Column('handled_branch', TEXT),
        Column('handled_turn', TEXT),
        ForeignKeyConstraint(
            ['character', 'rulebook', 'rule', 'handled_branch', 'handled_turn'],
            [crh.c.character, crh.c.rulebook, crh.c.rule, crh.c.branch, crh.c.turn]
        )
    )

    arh = Table(
        'avatar_rules_handled', meta,
        Column('character', TEXT, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('graph', TEXT, primary_key=True),
        Column('avatar', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True, default='trunk'),
        Column('turn', INT, primary_key=True),
        Column('tick', INT),
        ForeignKeyConstraint(
            ['character', 'rulebook'], ['avatar_rulebook.character', 'avatar_rulebook.rulebook']
        )
    )

    Table(
        'avatar_rules_changes', meta,
        Column('character', TEXT, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('graph', TEXT, primary_key=True),
        Column('avatar', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True),
        Column('turn', INT, primary_key=True),
        Column('tick', INT, primary_key=True),
        Column('handled_branch', TEXT),
        Column('handled_turn', TEXT),
        ForeignKeyConstraint(
            ['character', 'rulebook', 'rule', 'graph', 'avatar', 'handled_branch', 'handled_turn'],
            [arh.c.character, arh.c.rulebook, arh.c.rule, arh.c.graph, arh.c.avatar, arh.c.branch, arh.c.turn]
        )
    )

    ctrh = Table(
        'character_thing_rules_handled', meta,
        Column('character', TEXT, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('thing', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True, default='trunk'),
        Column('turn', INT, primary_key=True),
        Column('tick', INT),
        ForeignKeyConstraint(
            ['character', 'rulebook'], ['character_thing_rulebook.character', 'character_thing_rulebook.rulebook']
        ),
        ForeignKeyConstraint(
            ['character', 'thing'], ['things.character', 'things.thing']
        )
    )

    Table(
        'character_thing_rules_changes', meta,
        Column('character', TEXT, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('thing', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True),
        Column('turn', INT, primary_key=True),
        Column('tick', INT, primary_key=True),
        Column('handled_branch', TEXT),
        Column('handled_turn', INT),
        ForeignKeyConstraint(
            ['character', 'rulebook', 'rule', 'thing', 'handled_branch', 'handled_turn'],
            [ctrh.c.character, ctrh.c.rulebook, ctrh.c.rule, ctrh.c.thing, ctrh.c.branch, ctrh.c.turn]
        )
    )

    cprh = Table(
        'character_place_rules_handled', meta,
        Column('character', TEXT, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('place', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True, default='trunk'),
        Column('turn', INT, primary_key=True),
        Column('tick', INT),
        ForeignKeyConstraint(
            ['character', 'rulebook'], ['character_place_rulebook.character', 'character_place_rulebook.rulebook']
        ),
        ForeignKeyConstraint(
            ['character', 'place'], ['nodes.graph', 'nodes.node']
        )
    )

    Table(
        'character_place_rules_changes', meta,
        Column('character', TEXT, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('place', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True),
        Column('turn', INT, primary_key=True),
        Column('tick', INT, primary_key=True),
        Column('handled_branch', TEXT),
        Column('handled_turn', INT),
        ForeignKeyConstraint(
            ['character', 'rulebook', 'rule', 'place', 'handled_branch', 'handled_turn'],
            [cprh.c.character, cprh.c.rulebook, cprh.c.rule, cprh.c.place, cprh.c.branch, cprh.c.turn]
        )
    )

    cporh = Table(
        'character_portal_rules_handled', meta,
        Column('character', TEXT, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('orig', TEXT, primary_key=True),
        Column('dest', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True, default='trunk'),
        Column('turn', INT, primary_key=True),
        Column('tick', INT),
        ForeignKeyConstraint(
            ['character', 'rulebook'], ['character_portal_rulebook.character', 'character_portal_rulebook.rulebook']
        ),
        ForeignKeyConstraint(
            ['character', 'orig', 'dest'], ['edges.graph', 'edges.orig', 'edges.dest']
        )
    )

    Table(
        'character_portal_rules_changes', meta,
        Column('character', TEXT, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('orig', TEXT, primary_key=True),
        Column('dest', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True),
        Column('turn', INT, primary_key=True),
        Column('tick', INT, primary_key=True),
        Column('handled_branch', TEXT),
        Column('handled_turn', INT),
        ForeignKeyConstraint(
            ['character', 'rulebook', 'rule', 'orig', 'dest', 'handled_branch', 'handled_turn'],
            [cporh.c.character, cporh.c.rulebook, cporh.c.rule, cporh.c.orig, cporh.c.dest, cporh.c.branch, cporh.c.turn]
        )
    )

    Table(
        'turns_completed', meta,
        Column('branch', TEXT, primary_key=True),
        Column('turn', INT)
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
        wheres = [
            c == bindparam(c.name) for c in wherecols
        ]
        tab = wherecols[0].table
        return tab.update().values(**vmap).where(and_(*wheres))

    r = LiSE.allegedb.alchemy.queries_for_table_dict(table)

    rulebooks = table['rulebooks']
    r['rulebooks_update'] = update_where(['rules'], [rulebooks.c.rulebook, rulebooks.c.branch, rulebooks.c.turn, rulebooks.c.tick])

    for t in table.values():
        key = list(t.primary_key)
        if 'branch' in t.columns and 'turn' in t.columns and 'tick' in t.columns:
            branch = t.columns['branch']
            turn = t.columns['turn']
            tick = t.columns['tick']
            if branch in key and turn in key and tick in key:
                key = [branch, turn, tick]
        r[t.name + '_dump'] = select(list(t.c.values())).order_by(*key)
        r[t.name + '_insert'] = t.insert().values(tuple(bindparam(cname) for cname in t.c.keys()))
        r[t.name + '_count'] = select([func.COUNT('*')]).select_from(t)

    r['del_char_things'] = table['things'].delete().where(
        table['things'].c.character == bindparam('character')
    )

    r['del_char_avatars'] = table['avatars'].delete().where(
        table['avatars'].c.character_graph == bindparam('character')
    )
    things = table['things']
    r['del_things_after'] = things.delete().where(and_(
        things.c.character == bindparam('character'),
        things.c.thing == bindparam('thing'),
        things.c.branch == bindparam('branch'),
        or_(
            things.c.turn > bindparam('turn'),
            and_(
                things.c.turn == bindparam('turn'),
                things.c.tick >= bindparam('tick')
            )
        )
    ))
    avatars = table['avatars']
    r['del_avatars_after'] = avatars.delete().where(and_(
        avatars.c.character_graph == bindparam('character'),
        avatars.c.avatar_graph == bindparam('graph'),
        avatars.c.avatar_node == bindparam('avatar'),
        avatars.c.branch == bindparam('branch'),
        or_(
            avatars.c.turn > bindparam('turn'),
            and_(
                avatars.c.turn == bindparam('turn'),
                avatars.c.tick >= bindparam('tick')
            )
        )
    ))

    for handledtab in (
        'character_rules_handled',
        'avatar_rules_handled',
        'character_thing_rules_handled',
        'character_place_rules_handled',
        'character_portal_rules_handled',
        'node_rules_handled',
        'portal_rules_handled'
    ):
        ht = table[handledtab]
        r['del_{}_turn'.format(handledtab)] = ht.delete().where(and_(
            ht.c.branch == bindparam('branch'),
            ht.c.turn == bindparam('turn')
        ))


    branches = table['branches']

    r['branch_children'] = select(
        [branches.c.branch]
    ).where(
        branches.c.parent == bindparam('branch')
    )

    tc = table['turns_completed']
    r['turns_completed_update'] = update_where(['turn'], [tc.c.branch])

    return r


if __name__ == '__main__':
    from sqlalchemy import MetaData
    from sqlalchemy.dialects.sqlite.pysqlite import SQLiteDialect_pysqlite
    meta = MetaData()
    r = {}
    table = tables_for_meta(meta)
    dia = SQLiteDialect_pysqlite()
    for (n, t) in table.items():
        r["create_" + n] = str(
            CreateTable(t).compile(dialect=dia)
        )
    index = indices_for_table_dict(table)
    for (n, x) in index.items():
        r["index_" + n] = str(
            CreateIndex(x).compile(dialect=dia)
        )
    query = queries(table)
    for (n, q) in query.items():
        r[n] = str(q.compile(dialect=dia))
    print(dumps(r, sort_keys=True, indent=4))
