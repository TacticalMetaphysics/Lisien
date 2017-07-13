# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""A script to generate the SQL needed for LiSE's database backend,
and output it in JSON form.

This uses sqlalchemy to describe the queries. It extends the module of
the same name in the ``allegedb`` package. If you change anything here,
you won't be able to use your changes until you put the generated JSON
where LiSE will look for it, as in:

``python3 sqlalchemy.py >sqlite.json``

"""
from sqlalchemy import (
    Table,
    Index,
    Column,
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    Boolean,
    String,
    DateTime,
    MetaData,
    select,
    distinct,
    func,
    and_,
    or_,
    null
)
from sqlalchemy import create_engine
from json import dumps

from sqlalchemy.sql import bindparam, column
from sqlalchemy.sql.ddl import CreateTable, CreateIndex
from sqlalchemy.sql.expression import union
import allegedb.alchemy

# Constants

TEXT = String(50)

strtyps = (
    'strings',
)
"""Names of the different string tables, currently just 'strings'.

String tables store text to be used in the game, probably displayed to the
player. There's accommodation for more than one such table in case it
becomes useful to handle different kinds of text differently,
such as text meant to be used in format strings vs. text meant to be
displayed as written.

"""


def tables_for_meta(meta):
    """Return a dictionary full of all the tables I need for LiSE. Use the
    provided metadata object.

    """
    def handled_table(prefix):
        """Return a Table for recording the fact that a particular type of
        rule has been handled on a particular tick.

        """
        name = "{}_rules_handled".format(prefix)
        r = Table(
            name, meta,
            Column('character', TEXT, primary_key=True),
            Column('rulebook', TEXT, primary_key=True),
            Column('rule', TEXT, primary_key=True),
            Column('branch', TEXT, primary_key=True, default='trunk'),
            Column('tick', Integer, primary_key=True, default=0),
            ForeignKeyConstraint(
                ['character', 'rulebook'],
                [
                    'characters.character',
                    'characters.{}_rulebook'.format(prefix)
                ]
            )
        )
        return r

    def string_store_table(name):
        """Return a Table for storing strings, some of which may have
        different versions for different languages.

        """
        r = Table(
            name, meta,
            Column('id', TEXT, primary_key=True),
            Column('language', TEXT, primary_key=True, default='eng'),
            Column('date', DateTime, nullable=True),
            Column('creator', TEXT, nullable=True),
            Column('description', TEXT, nullable=True),
            Column('string', TEXT)
        )
        return r

    r = allegedb.alchemy.tables_for_meta(meta)

    for strtyp in strtyps:
        r[strtyp] = string_store_table(strtyp)

    # Table for global variables that are not sensitive to sim-time.
    r['universals'] = Table(
        'universals', meta,
        Column('key', TEXT, primary_key=True),
        Column(
            'branch', TEXT, primary_key=True, default='trunk'
        ),
        Column('tick', Integer, primary_key=True, default=0),
        Column('date', DateTime, nullable=True),
        Column('creator', TEXT, nullable=True),
        Column('description', TEXT, nullable=True),
        Column('value', TEXT, nullable=True)
    )

    # Header table for rules that exist.
    r['rules'] = Table(
        'rules', meta,
        Column('rule', TEXT, primary_key=True),
        Column('date', DateTime, nullable=True, default=None),
        Column('creator', TEXT, nullable=True, default=None),
        Column('description', TEXT, nullable=True, default=None)
    )

    # Table for rules' triggers, those functions that return True only
    # when their rule should run (or at least check its prereqs).
    r['rule_triggers'] = Table(
        'rule_triggers', meta,
        Column('rule', TEXT, primary_key=True),
        Column('idx', Integer, primary_key=True),
        Column('trigger', TEXT, nullable=False),
        ForeignKeyConstraint(['rule'], ['rules.rule']),
    )

    # Table for rules' prereqs, functions with veto power over a rule
    # being followed
    r['rule_prereqs'] = Table(
        'rule_prereqs', meta,
        Column('rule', TEXT, primary_key=True),
        Column('idx', Integer, primary_key=True),
        Column('prereq', TEXT, nullable=False),
        ForeignKeyConstraint(['rule'], ['rules.rule']),
    )

    # Table for rules' actions, the functions that do what the rule
    # does.
    r['rule_actions'] = Table(
        'rule_actions', meta,
        Column('rule', TEXT, primary_key=True),
        Column('idx', Integer, primary_key=True),
        Column('action', TEXT, nullable=False),
        ForeignKeyConstraint(['rule'], ['rules.rule']),
    )

    # Table grouping rules into lists called rulebooks.
    r['rulebooks'] = Table(
        'rulebooks', meta,
        Column('rulebook', TEXT, primary_key=True),
        Column('idx', Integer, primary_key=True),
        Column('date', DateTime, nullable=True),
        Column('contributor', TEXT, nullable=True),
        Column('description', TEXT, nullable=True),
        Column('rule', TEXT),
        ForeignKeyConstraint(['rule'], ['rules.rule'])
    )

    # Rules within a given rulebook that are active at a particular
    # (branch, tick).
    r['active_rules'] = Table(
        'active_rules', meta,
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column(
            'branch', TEXT, primary_key=True, default='trunk'
        ),
        Column('tick', Integer, primary_key=True, default=0),
        Column('date', DateTime, nullable=True),
        Column('contributor', TEXT, nullable=True),
        Column('description', TEXT, nullable=True),
        Column('active', Boolean, default=True),
        ForeignKeyConstraint(
            ['rulebook', 'rule'],
            ['rulebooks.rulebook', 'rulebooks.rule']
        )
    )

    # The top level of the LiSE world model, the character. Includes
    # rulebooks for the character itself, its avatars, and all the things,
    # places, and portals it contains--though those may have their own
    # rulebooks as well.
    r['characters'] = Table(
        'characters', meta,
        Column('character', TEXT, primary_key=True),
        Column('date', DateTime, nullable=True),
        Column('creator', TEXT, nullable=True),
        Column('description', TEXT, nullable=True),
        Column('character_rulebook', TEXT, nullable=False),
        Column('avatar_rulebook', TEXT, nullable=False),
        Column('character_thing_rulebook', TEXT, nullable=False),
        Column('character_place_rulebook', TEXT, nullable=False),
        Column('character_node_rulebook', TEXT, nullable=False),
        Column('character_portal_rulebook', TEXT, nullable=False),
        ForeignKeyConstraint(['character'], ['graphs.graph']),
        ForeignKeyConstraint(
            ['character_rulebook'], ['rulebooks.rulebook']
        ),
        ForeignKeyConstraint(['avatar_rulebook'], ['rulebooks.rulebook']),
        ForeignKeyConstraint(
            ['character_thing_rulebook'], ['rulebooks.rulebook']
        ),
        ForeignKeyConstraint(
            ['character_place_rulebook'], ['rulebooks.rulebook']
        ),
        ForeignKeyConstraint(
            ['character_portal_rulebook'], ['rulebooks.rulebook']
        )
    )

    # Rules handled within the rulebook associated with one thing in
    # particular.
    r['thing_rules_handled'] = Table(
        'thing_rules_handled', meta,
        Column('character', TEXT, primary_key=True),
        Column('thing', TEXT, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True),
        Column('tick', Integer, primary_key=True)
    )

    # Rules handled within the rulebook associated with one place in
    # particular.
    r['place_rules_handled'] = Table(
        'place_rules_handled', meta,
        Column('character', TEXT, primary_key=True),
        Column('place', TEXT, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True),
        Column('tick', Integer, primary_key=True)
    )

    # Rules handled within the rulebook associated with one portal in
    # particular.
    r['portal_rules_handled'] = Table(
        'portal_rules_handled', meta,
        Column('character', TEXT, primary_key=True),
        Column('nodeA', TEXT, primary_key=True),
        Column('nodeB', TEXT, primary_key=True),
        Column('idx', Integer, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True),
        Column('tick', Integer, primary_key=True)
    )

    # The function to use for a given sense.
    #
    # Characters use senses to look at other characters. To model this,
    # sense functions are called with a facade representing the
    # character under observation; the function munges this facade to
    # make it look as it does through the sense in question, and returns
    # that.
    #
    # Just which function to use for a given sense may change over time,
    # and a sense might not be usable all the time, in which case the
    # 'active' field will be ``False``.
    r['senses'] = Table(
        'senses', meta,
        # null character field means all characters have this sense
        Column(
            'character', TEXT, primary_key=True, nullable=True
        ),
        Column('sense', TEXT, primary_key=True),
        Column(
            'branch', TEXT, primary_key=True, default='trunk'
        ),
        Column('tick', Integer, primary_key=True, default=0),
        Column('date', DateTime, nullable=True),
        Column('contributor', TEXT, nullable=True),
        Column('description', TEXT, nullable=True),
        Column('function', TEXT),
        Column('active', Boolean, default=True),
        ForeignKeyConstraint(['character'], ['graphs.graph'])
    )

    # A list of tests that Things have to pass in order to move.
    #
    # Whenever a Thing tries to set its ``next_location``, its character
    # will pass the Thing itself and the Portal it wants to travel
    # into each of these functions, and will only allow it if all
    # return True (or if there are no functions in travel_reqs for the
    # character).
    #
    # Not used as of 2015-11-09
    r['travel_reqs'] = Table(
        'travel_reqs', meta,
        Column(
            'character', TEXT, primary_key=True, nullable=True
        ),
        Column('date', DateTime, nullable=True),
        Column('contributor', TEXT, nullable=True),
        Column('description', TEXT, nullable=True),
        Column('reqs', TEXT, default='[]'),
        ForeignKeyConstraint(['character'], ['graphs.graph'])
    )

    # Table for Things, being those nodes in a Character graph that have
    # locations.
    #
    # A Thing's location can be either a Place or another Thing, as long
    # as it's in the same Character. Things also have a
    # ``next_location``, defaulting to ``None``, which when set
    # indicates that the thing is in transit to that location.
    r['things'] = Table(
        'things', meta,
        Column('character', TEXT, primary_key=True),
        Column('thing', TEXT, primary_key=True),
        Column(
            'branch', TEXT, primary_key=True, default='trunk'
        ),
        Column('tick', Integer, primary_key=True, default=0),
        Column('date', DateTime, nullable=True),
        Column('contributor', TEXT, nullable=True),
        Column('description', TEXT, nullable=True),
        # when location is null, this node is not a thing, but a place
        Column('location', TEXT, nullable=True),
        # when next_location is not null, thing is en route between
        # location and next_location
        Column('next_location', TEXT, nullable=True),
        ForeignKeyConstraint(
            ['character', 'thing'], ['nodes.graph', 'nodes.node']
        ),
        ForeignKeyConstraint(
            ['character', 'location'], ['nodes.graph', 'nodes.node']
        ),
        ForeignKeyConstraint(
            ['character', 'next_location'], ['nodes.graph', 'nodes.node']
        )
    )

    # The rulebook followed by a given node.
    r['node_rulebook'] = Table(
        'node_rulebook', meta,
        Column('character', TEXT, primary_key=True),
        Column('node', TEXT, primary_key=True),
        Column('date', DateTime, nullable=True),
        Column('contributor', TEXT, nullable=True),
        Column('description', TEXT, nullable=True),
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
    r['portal_rulebook'] = Table(
        'portal_rulebook', meta,
        Column('character', TEXT, primary_key=True),
        Column('nodeA', TEXT, primary_key=True),
        Column('nodeB', TEXT, primary_key=True),
        Column('idx', Integer, primary_key=True, default=0),
        Column('date', DateTime, nullable=True),
        Column('contributor', TEXT, nullable=True),
        Column('description', TEXT, nullable=True),
        Column('rulebook', TEXT),
        ForeignKeyConstraint(
            ['character', 'nodeA', 'nodeB', 'idx'],
            ['edges.graph', 'edges.nodeA', 'edges.nodeB', 'edges.idx']
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
    r['avatars'] = Table(
        'avatars', meta,
        Column('character_graph', TEXT, primary_key=True),
        Column('avatar_graph', TEXT, primary_key=True),
        Column('avatar_node', TEXT, primary_key=True),
        Column(
            'branch', TEXT, primary_key=True, default='trunk'
        ),
        Column('tick', Integer, primary_key=True, default=0),
        Column('date', DateTime, nullable=True),
        Column('contributor', TEXT, nullable=True),
        Column('description', TEXT, nullable=True),
        Column('is_avatar', Boolean),
        ForeignKeyConstraint(['character_graph'], ['graphs.graph']),
        ForeignKeyConstraint(
            ['avatar_graph', 'avatar_node'],
            ['nodes.graph', 'nodes.node']
        )
    )

    for tab in (
        handled_table('character'),
        handled_table('avatar'),
        handled_table('character_thing'),
        handled_table('character_place'),
        handled_table('character_node'),
        handled_table('character_portal'),
    ):
        r[tab.name] = tab

    return r


def views_for_table_dict(table):
    """Create queries to use in views"""
    r = {}
    prh = table['place_rules_handled']
    trh = table['thing_rules_handled']
    r['node_rules_handled'] = union(
        select(
            [
                prh.c.character,
                prh.c.place.label('node'),
                prh.c.rulebook,
                prh.c.rule,
                prh.c.branch,
                prh.c.tick
            ]
        ),
        select(
            [
                trh.c.character,
                trh.c.thing.label('node'),
                trh.c.rulebook,
                trh.c.rule,
                trh.c.branch,
                trh.c.tick
            ]
        )
    )
    return r


def indices_for_table_dict(table):
    """Given the dictionary of tables returned by ``tables_for_meta``,
    return a dictionary of indices for the tables.

    """
    def handled_idx(prefix):
        """Return an index for the _rules_handled table with the given
        prefix.

        """
        t = table['{}_rules_handled'.format(prefix)]
        return Index(
            "{}_rules_handled_idx".format(prefix),
            t.c.character,
            t.c.rulebook,
            t.c.rule
        )

    r = allegedb.alchemy.indices_for_table_dict(table)

    for idx in (
            Index(
                'active_rules_idx',
                table['active_rules'].c.rulebook,
                table['active_rules'].c.rule
            ),
            Index(
                'senses_idx',
                table['senses'].c.character,
                table['senses'].c.sense
            ),
            Index(
                'travel_reqs_idx',
                table['travel_reqs'].c.character
            ),
            Index(
                'things_idx',
                table['things'].c.character,
                table['things'].c.thing
            ),
            Index(
                'avatars_idx',
                table['avatars'].c.character_graph,
                table['avatars'].c.avatar_graph,
                table['avatars'].c.avatar_node
            ),
            handled_idx('character'),
            handled_idx('avatar'),
            handled_idx('character_thing'),
            handled_idx('character_place'),
            handled_idx('character_node'),
            handled_idx('character_portal'),
            Index(
                'thing_rules_handled_idx',
                table['thing_rules_handled'].c.character,
                table['thing_rules_handled'].c.thing,
                table['thing_rules_handled'].c.rulebook,
                table['thing_rules_handled'].c.rule
            ),
            Index(
                'place_rules_handled_idx',
                table['place_rules_handled'].c.character,
                table['place_rules_handled'].c.place,
                table['place_rules_handled'].c.rulebook,
                table['place_rules_handled'].c.rule
            ),
            Index(
                'portal_rules_handled_idx',
                table['portal_rules_handled'].c.character,
                table['portal_rules_handled'].c.nodeA,
                table['portal_rules_handled'].c.nodeB,
                table['portal_rules_handled'].c.idx,
                table['portal_rules_handled'].c.rulebook,
                table['portal_rules_handled'].c.rule
            )
    ):
        r[idx.table.name] = idx

    return r


def queries(table, view):
    """Given dictionaries of tables and view-queries, return a dictionary
    of all the rest of the queries I need.

    """
    def insert_cols(t, *cols):
        """Return an ``INSERT`` statement into table ``t`` with
        bind-parameters for the columns ``cols``, which must be actual
        columns in ``t``.

        """
        vmap = {
            col: bindparam(col) for col in cols
        }
        return t.insert().values(**vmap)

    def select_where(t, selcols, wherecols):
        """Return a ``SELECT`` statement that selects the columns ``selcols``
        from the table ``t`` where the columns in ``wherecols`` equal
        the bound parameters.

        """
        wheres = [
            getattr(t.c, col) == bindparam(col)
            for col in wherecols
        ]
        return select(
            [getattr(t.c, col) for col in selcols]
        ).where(and_(*wheres))

    def update_where(t, updcols, wherecols):
        """Return an ``UPDATE`` statement that updates the columns ``updcols``
        (with bindparams for each) in the table ``t`` in which the
        columns ``wherecols`` equal the bound parameters.

        """
        vmap = {
            col: bindparam(col) for col in updcols
        }
        wheres = [
            getattr(t.c, col) == bindparam(col)
            for col in wherecols
        ]
        return t.update().values(**vmap).where(and_(*wheres))

    def func_table_iter(t):
        """Select the ``name`` column."""
        return select(
            [t.c.name]
        )

    def func_table_name_plaincode(t):
        """Select the ``name`` and ``plaincode`` columns."""
        return select(
            [t.c.name, t.c.plaincode]
        )

    def func_table_get(t):
        """Get all columns for a given function (except ``name``).

        * ``bytecode``
        * ``base``
        * ``keywords``
        * ``date``
        * ``creator``
        * ``contributor``
        * ``description``
        * ``plaincode``
        * ``version``

        """
        return select(
            [
                t.c.bytecode,
                t.c.base,
                t.c.keywords,
                t.c.date,
                t.c.creator,
                t.c.contributor,
                t.c.description,
                t.c.plaincode,
                t.c.version
            ]
        ).where(
            t.c.name == bindparam('name')
        )

    def func_table_ins(t):
        """Return an ``INSERT`` statement for a function table.

        Inserts the fields:

        * ``name``
        * ``bytecode``
        * ``plaincode``

        """
        return t.insert().values(
            name=bindparam('name'),
            keywords=bindparam('keywords'),
            bytecode=bindparam('bytecode'),
            plaincode=bindparam('plaincode')
        )

    def func_table_upd(t):
        """Return an ``UPDATE`` statement to change the ``bytecode`` and
        ``plaincode`` for a function of a given name.

        """
        return t.update().values(
            keywords=bindparam('keywords'),
            bytecode=bindparam('bytecode'),
            plaincode=bindparam('plaincode')
        ).where(
            t.c.name == bindparam('name')
        )

    def func_table_del(t):
        """Return a ``DELETE`` statement to delete the function by a given
        name.

        """
        return t.delete().where(
            t.c.name == bindparam('name')
        )

    def string_table_lang_items(t):
        """Return all the strings and their IDs for a given language."""
        return select(
            [t.c.id, t.c.string]
        ).where(
            t.c.language == bindparam('language')
        ).order_by(
            t.c.id
        )

    def string_table_get(t):
        """Return a ``SELECT`` statement to get a string based on its language
        and ID.

        """
        return select(
            [t.c.string]
        ).where(
            and_(
                t.c.language == bindparam('language'),
                t.c.id == bindparam('id')
            )
        )

    def string_table_ins(t):
        """Return an ``INSERT`` statement for a string's ID, its language, and
        the string itself.

        """
        return t.insert().values(
            id=bindparam('id'),
            language=bindparam('language'),
            string=bindparam('string')
        )

    def string_table_upd(t):
        """Return an ``UPDATE`` statement to change a string in a given
        language, with a given ID.

        """
        return t.update().values(
            string=bindparam('string')
        ).where(
            and_(
                t.c.language == bindparam('language'),
                t.c.id == bindparam('id')
            )
        )

    def string_table_del(t):
        """Return a ``DELETE`` statement to get rid of a string in a given
        language, with a given ID.

        """
        return t.delete().where(
            and_(
                t.c.language == bindparam('language'),
                t.c.id == bindparam('id')
            )
        )

    r = allegedb.alchemy.queries_for_table_dict(table)

    for t in table.values():
        r[t.name + '_dump'] = select().select_from(t)

    for strtyp in strtyps:
        r['{}_lang_items'.format(strtyp)] = string_table_lang_items(
            table[strtyp]
        )
        r['{}_get'.format(strtyp)] = string_table_get(table[strtyp])
        r['{}_ins'.format(strtyp)] = string_table_ins(table[strtyp])
        r['{}_upd'.format(strtyp)] = string_table_upd(table[strtyp])
        r['{}_del'.format(strtyp)] = string_table_del(table[strtyp])

    characters = table['characters']

    r['characters'] = select([table['characters'].c.character])

    r['characters_rulebooks'] = select([
        characters.c.character,
        characters.c.character_rulebook,
        characters.c.avatar_rulebook,
        characters.c.character_thing_rulebook,
        characters.c.character_place_rulebook,
        characters.c.character_node_rulebook,
        characters.c.character_portal_rulebook
    ])

    r['character_rulebooks'] = select([
        characters.c.character_rulebook,
        characters.c.avatar_rulebook,
        characters.c.character_thing_rulebook,
        characters.c.character_place_rulebook,
        characters.c.character_node_rulebook,
        characters.c.character_portal_rulebook
    ]).where(
        table['characters'].c.character == bindparam('character')
    )

    r['ct_characters'] = select([func.COUNT(table['characters'].c.character)])

    r['ct_character'] = select(
        [func.COUNT(table['characters'].c.character)]
    ).where(
        table['characters'].c.character == bindparam('character')
    )

    node_rulebook = table['node_rulebook']
    rulebooks = table['rulebooks']


    # Note that you have to pass in the branch and tick *twice*, and
    # prior to the character and node, if you're using sqlite

    portal_rulebook = table['portal_rulebook']

    r['portal_rulebook'] = select_where(
        portal_rulebook,
        ['rulebook'],
        ['character', 'nodeA', 'nodeB', 'idx']
    )
    r['portals_rulebooks'] = select([
        portal_rulebook.c.character,
        portal_rulebook.c.nodeA,
        portal_rulebook.c.nodeB,
        portal_rulebook.c.rulebook
    ])

    r['ins_portal_rulebook'] = insert_cols(
        portal_rulebook,
        'character',
        'nodeA',
        'nodeB',
        'idx',
        'rulebook'
    )

    r['upd_portal_rulebook'] = update_where(
        portal_rulebook,
        ['rulebook'],
        ['character', 'nodeA', 'nodeB', 'idx']
    )

    characters = table['characters']

    r['handled_thing_rule'] = insert_cols(
        table['thing_rules_handled'],
        'character',
        'thing',
        'rulebook',
        'rule',
        'branch',
        'tick'
    )

    r['handled_place_rule'] = insert_cols(
        table['place_rules_handled'],
        'character',
        'place',
        'rulebook',
        'rule',
        'branch',
        'tick'
    )

    r['handled_portal_rule'] = insert_cols(
        table['portal_rules_handled'],
        'character',
        'nodeA',
        'nodeB',
        'idx',
        'rulebook',
        'rule',
        'branch',
        'tick'
    )

    r['del_char_things'] = table['things'].delete().where(
        table['things'].c.character == bindparam('character')
    )

    r['del_char_avatars'] = table['avatars'].delete().where(
        table['avatars'].c.character_graph == bindparam('character')
    )

    things = table['things']

    avatars = table['avatars']

    r['arrival_time_get'] = select(
        [func.MAX(things.c.tick)]
    ).where(
        and_(
            things.c.character == bindparam('character'),
            things.c.thing == bindparam('thing'),
            things.c.location == bindparam('location'),
            things.c.branch == bindparam('branch'),
            things.c.tick <= bindparam('tick')
        )
    )

    r['next_arrival_time_get'] = select(
        [func.MIN(things.c.tick)]
    ).where(
        and_(
            things.c.character == bindparam('character'),
            things.c.thing == bindparam('thing'),
            things.c.location == bindparam('location'),
            things.c.branch == bindparam('branch'),
            things.c.tick > bindparam('tick')
        )
    )

    r['thing_loc_and_next_ins'] = insert_cols(
        things,
        'character',
        'thing',
        'branch',
        'tick',
        'location',
        'next_location'
    )

    r['thing_loc_and_next_upd'] = update_where(
        things,
        ['location', 'next_location'],
        ['character', 'thing', 'branch', 'tick']
    )

    r['node_val_data_branch'] = select_where(
        table['node_val'],
        ['key', 'rev', 'value'],
        ['graph', 'node', 'branch']
    )

    r['node_rulebook'] = select_where(
        node_rulebook,
        ['rulebook'],
        ['character', 'node']
    )
    r['nodes_rulebooks'] = select([
        node_rulebook.c.character,
        node_rulebook.c.node,
        node_rulebook.c.rulebook
    ])

    r['ins_node_rulebook'] = insert_cols(
        node_rulebook,
        'character',
        'node',
        'rulebook'
    )

    r['upd_node_rulebook'] = update_where(
        node_rulebook,
        ['rulebook'],
        ['character', 'node']
    )

    graph_val = table['graph_val']

    r['char_stat_branch_data'] = select(
        [
            graph_val.c.key,
            graph_val.c.rev,
            graph_val.c.value
        ]
    ).where(
        and_(
            graph_val.c.graph == bindparam('character'),
            graph_val.c.branch == bindparam('branch')
        )
    )

    node_val = table['node_val']

    r['node_stat_branch_data'] = select(
        [
            node_val.c.key,
            node_val.c.rev,
            node_val.c.value
        ]
    ).where(
        and_(
            node_val.c.graph == bindparam('character'),
            node_val.c.node == bindparam('node'),
            node_val.c.branch == bindparam('branch')
        )
    )

    edge_val = table['edge_val']

    r['edge_stat_branch_data'] = select(
        [
            edge_val.c.key,
            edge_val.c.rev,
            edge_val.c.value
        ]
    ).where(
        and_(
            edge_val.c.graph == bindparam('character'),
            edge_val.c.nodeA == bindparam('origin'),
            edge_val.c.nodeB == bindparam('destination'),
            edge_val.c.branch == bindparam('branch')
        )
    )

    r['thing_locs_branch_data'] = select(
        [things.c.tick, things.c.location, things.c.next_location]
    ).where(
        and_(
            things.c.character == bindparam('character'),
            things.c.thing == bindparam('thing'),
            things.c.branch == bindparam('branch')
        )
    )

    r['avatars_ever'] = select(
        [
            avatars.c.avatar_graph,
            avatars.c.avatar_node,
            avatars.c.branch,
            avatars.c.tick,
            avatars.c.is_avatar
        ]
    ).where(
        avatars.c.character_graph == bindparam('character')
    )

    senses = table['senses']

    r['sense_fun_ins'] = insert_cols(
        senses,
        'character',
        'sense',
        'branch',
        'tick',
        'function',
        'active'
    )

    r['sense_fun_upd'] = update_where(
        senses,
        ['function', 'active'],
        ['character', 'sense', 'branch', 'tick']
    )

    r['sense_ins'] = insert_cols(
        senses,
        'character',
        'sense',
        'branch',
        'tick',
        'active'
    )

    r['sense_upd'] = update_where(
        senses,
        ['active'],
        ['character', 'sense', 'branch', 'tick']
    )

    r['character_ins'] = insert_cols(
        characters,
        'character',
        'character_rulebook',
        'avatar_rulebook',
        'character_thing_rulebook',
        'character_place_rulebook',
        'character_node_rulebook',
        'character_portal_rulebook'
    )

    r['avatar_ins'] = insert_cols(
        avatars,
        'character_graph',
        'avatar_graph',
        'avatar_node',
        'branch',
        'tick',
        'is_avatar'
    )

    r['avatar_upd'] = update_where(
        avatars,
        ['is_avatar'],
        [
            'character_graph',
            'avatar_graph',
            'avatar_node',
            'branch',
            'tick'
        ]
    )

    rules = table['rules']
    rule_triggers = table['rule_triggers']
    rule_prereqs = table['rule_prereqs']
    rule_actions = table['rule_actions']
    r['rule_triggers'] = select([rule_triggers.c.trigger]).where(rule_triggers.c.rule == bindparam('rule'))
    r['rule_prereqs'] = select([rule_prereqs.c.prereq]).where(rule_prereqs.c.rule == bindparam('rule'))
    r['rule_actions'] = select([rule_actions.c.action]).where(rule_actions.c.rule == bindparam('rule'))

    def rule_something_count(tab):
        """Return query to count the number of functions in a rule table."""
        return select([func.count(tab.c.idx)]).where(
            tab.c.rule == bindparam('rule')
        )

    def rule_something_ins(tab, something):
        """Return query to insert a function into a rule table."""
        return tab.insert().values({
            'rule': bindparam('rule'),
            something: bindparam('something'),
            'idx': bindparam('idx')
        })

    def rule_something_inc(tab):
        """Return query to make room to insert a function into a rule table."""
        return tab.update().values(
            idx=tab.c.idx+column('1', is_literal=True)
        ).where(and_(
            tab.c.rule == bindparam('rule'),
            tab.c.idx >= bindparam('idx')
        ))

    def rule_something_dec(tab):
        """Return query to correct indices in a rule table after deletion.

        Looks at all records for some rule, and decrements the indices
        of those with idx greater than that given.

        """
        return tab.update().values(
            idx=tab.c.idx+column('1', is_literal=True)
        ).where(and_(
            tab.c.rule == bindparam('rule'),
            tab.c.idx > bindparam('idx')
        ))

    def rule_something_del(tab):
        """Return a query to delete a function from a rule table."""
        return tab.delete().where(and_(
            tab.c.rule == bindparam('rule'),
            tab.c.idx == bindparam('idx')
        ))

    def rule_something_del_all(tab):
        """Return a query to delete all functions in a rule."""
        return tab.delete().where(tab.c.rule == bindparam('rule'))

    r['ins_rule'] = rules.insert().values(
        rule=bindparam('rule'),
        date=bindparam('date'),
        creator=bindparam('creator'),
        description=bindparam('description')
    )
    r['upd_rule'] = rules.update().values(
        date=bindparam('date'),
        creator=bindparam('creator'),
        description=bindparam('description')
    ).where(rules.c.rule == bindparam('rule'))
    r['del_rule'] = rules.delete().where(rules.c.rule == bindparam('rule'))
    r['rule_triggers_count'] = rule_something_count(rule_triggers)
    r['rule_triggers_ins'] = rule_something_ins(rule_triggers, 'trigger')
    r['rule_triggers_inc'] = rule_something_inc(rule_triggers)
    r['rule_triggers_dec'] = rule_something_dec(rule_triggers)
    r['rule_triggers_del'] = rule_something_del(rule_triggers)
    r['rule_triggers_del_all'] = rule_something_del_all(rule_triggers)
    r['rule_prereqs_count'] = rule_something_count(rule_prereqs)
    r['rule_prereqs_ins'] = rule_something_ins(rule_prereqs, 'prereq')
    r['rule_prereqs_inc'] = rule_something_inc(rule_prereqs)
    r['rule_prereqs_dec'] = rule_something_dec(rule_prereqs)
    r['rule_prereqs_del'] = rule_something_del(rule_prereqs)
    r['rule_prereqs_del_all'] = rule_something_del_all(rule_prereqs)
    r['rule_actions_count'] = rule_something_count(rule_actions)
    r['rule_actions_ins'] = rule_something_ins(rule_actions, 'action')
    r['rule_actions_inc'] = rule_something_inc(rule_actions)
    r['rule_actions_dec'] = rule_something_dec(rule_actions)
    r['rule_actions_del'] = rule_something_del(rule_actions)
    r['rule_actions_del_all'] = rule_something_del_all(rule_actions)

    travreqs = table['travel_reqs']

    r['travel_reqs'] = select(
        [travreqs.c.reqs]
    ).where(
        travreqs.c.character == bindparam('character')
    )

    r['ins_travel_reqs'] = travreqs.insert().values(
        character=bindparam('character'),
        reqs=bindparam('reqs')
    )

    r['upd_travel_reqs'] = travreqs.update().values(
        reqs=bindparam('reqs')
    ).where(
        travreqs.c.character == bindparam('character')
    )

    r['rulebooks'] = select([rulebooks.c.rulebook])

    r['ct_rulebooks'] = select([func.COUNT(distinct(rulebooks.c.rulebook))])

    r['rulebook_rules'] = select(
        [rulebooks.c.rule]
    ).where(
        rulebooks.c.rulebook == bindparam('rulebook')
    ).order_by(rulebooks.c.idx)

    r['rulebooks_rules'] = select([
        rulebooks.c.rulebook,
        rulebooks.c.rule
    ]).order_by(rulebooks.c.idx)

    r['ct_rulebook_rules'] = select(
        [func.COUNT(rulebooks.c.rule)]
    ).where(
        rulebooks.c.rulebook == bindparam('rulebook')
    )

    r['rulebook_get'] = select(
        [rulebooks.c.rule]
    ).where(
        and_(
            rulebooks.c.rulebook == bindparam('rulebook'),
            rulebooks.c.idx == bindparam('idx')
        )
    )

    r['rulebook_ins'] = insert_cols(
        rulebooks,
        'rulebook',
        'idx',
        'rule'
    )

    r['rulebook_upd'] = update_where(
        rulebooks,
        ['rule'],
        ['rulebook', 'idx']
    )

    r['rulebook_inc'] = rulebooks.update().values(
        idx=rulebooks.c.idx+column('1', is_literal=True)
    ).where(
        and_(
            rulebooks.c.rulebook == bindparam('rulebook'),
            rulebooks.c.idx >= bindparam('idx')
        )
    )

    r['rulebook_dec'] = rulebooks.update().values(
        idx=rulebooks.c.idx-column('1', is_literal=True)
    ).where(
        and_(
            rulebooks.c.rulebook == bindparam('rulebook'),
            rulebooks.c.idx > bindparam('idx')
        )
    )

    r['rulebook_del'] = rulebooks.delete().where(
        and_(
            rulebooks.c.rulebook == bindparam('rulebook'),
            rulebooks.c.idx == bindparam('idx')
        )
    )

    r['rulebook_del_all'] = rulebooks.delete().where(rulebooks.c.rulebook == bindparam('rulebook'))

    rules = table['rules']

    r['allrules'] = select([rules.c.rule])

    r['haverule'] = select(
        [rules.c.rule]
    ).where(
        rules.c.rule == bindparam('rule')
    )

    r['ctrules'] = select([func.COUNT()]).select_from(rules)

    r['ruleins'] = rules.insert().values(rule=bindparam('rule'))

    r['ruledel'] = rules.delete().where(rules.c.rule == bindparam('rule'))

    branches = table['branches']

    r['branch_children'] = select(
        [branches.c.branch]
    ).where(
        branches.c.parent == bindparam('branch')
    )

    for (n, t) in table.items():
        r['count_all_{}'.format(n)] = select(
            [getattr(t.c, col) for col in t.c.keys()]
        ).count()

    return r


if __name__ == '__main__':
    e = create_engine('sqlite:///:memory:')
    meta = MetaData()
    r = {}
    table = tables_for_meta(meta)
    for (n, t) in table.items():
        r["create_" + n] = str(
            CreateTable(t).compile(
                dialect=e.dialect
            )
        )
        t.create(e)
    index = indices_for_table_dict(table)
    for (n, x) in index.items():
        r["index_" + n] = str(
            CreateIndex(x).compile(
                dialect=e.dialect
            )
        )
        x.create(e)
    viewquery = views_for_table_dict(table)
    for (n, v) in viewquery.items():
        r["view_" + n] = "CREATE VIEW {} AS ".format(n) + str(
            v.compile(dialect=e.dialect)
        )
        e.execute(r["view_" + n])
    query = queries(table, viewquery)
    for (n, q) in query.items():
        r[n] = str(q.compile(dialect=e.dialect))
    print(dumps(r, sort_keys=True, indent=4))
