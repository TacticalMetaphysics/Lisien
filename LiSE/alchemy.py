# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
"""A script to generate the SQL needed for LiSE's database backend,
and output it in JSON form.

This uses sqlalchemy to describe the queries. It extends the module of
the same name in the ``gorm`` package. If you change anything here,
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
import gorm.alchemy



### Constants
length = 50

TEXT = String(length)

functyps = (
    'actions',
    'prereqs',
    'triggers',
    'functions'
)

strtyps = (
    'strings',
)


def tables_for_meta(meta):
    """Return a dictionary full of all the tables I need for LiSE. Use the
    provided metadata object.

    """
    def handled_table(prefix):
        name = "{}_rules_handled".format(prefix)
        return Table(
            name, meta,
            Column('character', TEXT, primary_key=True),
            Column('rulebook', TEXT, primary_key=True),
            Column('rule', TEXT, primary_key=True),
            Column('branch', TEXT, primary_key=True, default='master'),
            Column('tick', Integer, primary_key=True, default=0),
            ForeignKeyConstraint(
                ['character', 'rulebook'],
                [
                    'characters.character',
                    'characters.{}_rulebook'.format(prefix)
                ]
            )
        )
        """Table to keep track of rules that have already been handled at some
        sim-time.

        """

    def string_store_table(name):
        return Table(
            name, meta,
            Column('id', TEXT, primary_key=True),
            Column('language', TEXT, primary_key=True, default='eng'),
            Column('date', DateTime, nullable=True),
            Column('creator', TEXT, nullable=True),
            Column('description', TEXT, nullable=True),
            Column('string', TEXT)
        )
        """Table to store strings, possibly for display to the player."""

    def func_store_table(name):
        r = Table(
            name, meta,
            Column('name', TEXT, primary_key=True),
            Column(
                'base', TEXT,
                ForeignKey('{}.name'.format(name)),
                nullable=True,
                default=None
            ),
            Column('keywords', TEXT, nullable=False, default='[]'),
            Column('bytecode', TEXT, nullable=True),
            Column('date', DateTime, nullable=True),
            Column('creator', TEXT, nullable=True),
            Column('contributor', TEXT, nullable=True),
            Column('description', TEXT, nullable=True),
            Column('plaincode', TEXT, nullable=True),
            Column('version', TEXT, nullable=True),
        )
        """Table to store functions, both source code and bytecode.

        If the 'base' field of a given record is filled in with the
        name of another function, the function in this record is a
        partial. These work a bit differently from
        ``functools.partial``: only keyword arguments may be
        prefilled, and these are kept in a JSON object in the
        'plaincode' field.

        """
        r.append_constraint(
            CheckConstraint(or_(
                r.c.bytecode != None,
                r.c.plaincode != None
            ))
        )
        return r

    r = gorm.alchemy.tables_for_meta(meta)

    for functyp in functyps:
        r[functyp] = func_store_table(functyp)

    for strtyp in strtyps:
        r[strtyp] = string_store_table(strtyp)

    r['lise_globals'] = Table(
        'lise_globals', meta,
        Column('key', TEXT, primary_key=True),
        Column(
            'branch', TEXT, primary_key=True, default='master'
        ),
        Column('tick', Integer, primary_key=True, default=0),
        Column('date', DateTime, nullable=True),
        Column('creator', TEXT, nullable=True),
        Column('description', TEXT, nullable=True),
        Column('value', TEXT, nullable=True)
    )
    """Table for global variables that are not sensitive to sim-time.

    """

    r['rules'] = Table(
        'rules', meta,
        Column('rule', TEXT, primary_key=True),
        Column('date', DateTime, nullable=True),
        Column('creator', TEXT, nullable=True),
        Column('description', TEXT, nullable=True),
        Column('actions', TEXT, default='[]'),
        Column('prereqs', TEXT, default='[]'),
        Column('triggers', TEXT, default='[]'),
    )
    """Table listing the actions, prereqs, and triggers that make up each
    rule.

    Lists are JSON encoded strings of function names. There's no
    constraint on what to use for function names because there are
    multiple possible methods for storing and retrieving functions.

    """

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
    """Table grouping rules into lists called rulebooks.

    """

    r['active_rules'] = Table(
        'active_rules', meta,
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column(
            'branch', TEXT, primary_key=True, default='master'
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
    """Rules within a given rulebook that are active at a particular
    ``(branch, tick)``.

    """

    r['characters'] = Table(
        'characters', meta,
        Column('character', TEXT, primary_key=True),
        Column('date', DateTime, nullable=True),
        Column('creator', TEXT, nullable=True),
        Column('description', TEXT, nullable=True),
        Column('character_rulebook', TEXT, nullable=True),
        Column('avatar_rulebook', TEXT, nullable=True),
        Column('character_thing_rulebook', TEXT, nullable=True),
        Column('character_place_rulebook', TEXT, nullable=True),
        Column('character_node_rulebook', TEXT, nullable=True),
        Column('character_portal_rulebook', TEXT, nullable=True),
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
    """The top level of the LiSE world model, the character. Includes
    rulebooks for the character itself, its avatars, and the things,
    places, and portals it contains.

    """

    r['thing_rules_handled'] = Table(
        'thing_rules_handled', meta,
        Column('character', TEXT, primary_key=True),
        Column('thing', TEXT, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True),
        Column('tick', Integer, primary_key=True)
    )
    """Rules handled within the rulebook associated with one thing in
    particular.

    """

    r['place_rules_handled'] = Table(
        'place_rules_handled', meta,
        Column('character', TEXT, primary_key=True),
        Column('place', TEXT, primary_key=True),
        Column('rulebook', TEXT, primary_key=True),
        Column('rule', TEXT, primary_key=True),
        Column('branch', TEXT, primary_key=True),
        Column('tick', Integer, primary_key=True)
    )
    """Rules handled within the rulebook associated with one place in
    particular.

    """

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
    """Rules handled within the rulebook associated with one portal in
    particular.

    """

    r['senses'] = Table(
        'senses', meta,
        Column(
            'character', TEXT, primary_key=True, nullable=True
        ),
        # blank character field means all characters have this sense
        Column('sense', TEXT, primary_key=True),
        Column(
            'branch', TEXT, primary_key=True, default='master'
        ),
        Column('tick', Integer, primary_key=True, default=0),
        Column('date', DateTime, nullable=True),
        Column('contributor', TEXT, nullable=True),
        Column('description', TEXT, nullable=True),
        Column('function', TEXT),
        Column('active', Boolean, default=True),
        ForeignKeyConstraint(['character'], ['graphs.graph'])
    )
    """The function to use for a given sense.

    Characters use senses to look at other characters. To model this,
    sense functions are called with a facade representing the
    character under observation; the function munges this facade to
    make it look as it does through the sense in question, and returns
    that.

    Just which function to use for a given sense may change over time,
    and a sense might not be usable all the time, in which case the
    'active' field will be ``False``.

    """

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
    """A list of tests that Things have to pass in order to move.

    Whenever a Thing tries to set its ``next_location``, its character
    will pass the Thing itself and the Portal it wants to travel
    into each of these functions, and will only allow it if all
    return True (or if there are no functions in travel_reqs for the
    charcater).

    """

    r['things'] = Table(
        'things', meta,
        Column('character', TEXT, primary_key=True),
        Column('thing', TEXT, primary_key=True),
        Column(
            'branch', TEXT, primary_key=True, default='master'
        ),
        Column('tick', Integer, primary_key=True, default=0),
        Column('date', DateTime, nullable=True),
        Column('contributor', TEXT, nullable=True),
        Column('description', TEXT, nullable=True),
        Column('location', TEXT, nullable=True),
        # when location is null, this node is not a thing, but a place
        Column('next_location', TEXT, nullable=True),
        # when next_location is not null, thing is en route between
        # location and next_location
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
    """Table for Things, being those nodes in a Character graph that have
    locations.

    A Thing's location can be either a Place or another Thing, as long
    as it's in the same Character. Things also have a
    ``next_location``, defaulting to ``None``, which when set
    indicates that the thing is in transit to that location.

    """

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
    """The rulebook followed by a given node."""

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
    """The rulebook followed by a given Portal."""

    r['avatars'] = Table(
        'avatars', meta,
        Column('character_graph', TEXT, primary_key=True),
        Column('avatar_graph', TEXT, primary_key=True),
        Column('avatar_node', TEXT, primary_key=True),
        Column(
            'branch', TEXT, primary_key=True, default='master'
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
    """The avatars representing one Character in another.

    In the common situation where a Character, let's say Alice has its
    own stats and skill tree and social graph, and also has a location
    in physical space, you can represent this by creating a Thing in
    the Character that represents physical space, and then making that
    Thing an avatar of Alice. On its own this doesn't do anything,
    it's just a convenient way of indicating the relation -- but if
    you like, you can make rules that affect all avatars of some
    Character, irrespective of what Character the avatar is actually
    *in*.

    """

    for tab in (
        handled_table('character'),
        handled_table('avatar'),
        handled_table('character_thing'),
        handled_table('character_place'),
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
    cprh = table['character_place_rules_handled']
    ctrh = table['character_thing_rules_handled']
    r['character_node_rules_handled'] = union(
        select([
            cprh.c.character,
            cprh.c.rulebook,
            cprh.c.rule,
            cprh.c.branch,
            cprh.c.tick
        ]),
        select([
            ctrh.c.character,
            ctrh.c.rulebook,
            ctrh.c.rule,
            ctrh.c.branch,
            ctrh.c.tick
        ])
    )
    return r


def indices_for_table_dict(table):
    """Given the dictionary of tables returned by ``tables_for_meta``,
    return a dictionary of indices for the tables.

    """
    def handled_idx(prefix):
        t = table['{}_rules_handled'.format(prefix)]
        return Index(
            "{}_rules_handled_idx".format(prefix),
            t.c.character,
            t.c.rulebook,
            t.c.rule
        )

    r = gorm.alchemy.indices_for_table_dict(table)

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

    r = gorm.alchemy.queries_for_table_dict(table)

    for functyp in functyps:
        r['func_{}_name_plaincode'.format(functyp)] \
            = func_table_name_plaincode(table[functyp])
        r['func_{}_iter'.format(functyp)] = func_table_iter(table[functyp])
        r['func_{}_get'.format(functyp)] = func_table_get(table[functyp])
        r['func_{}_ins'.format(functyp)] = func_table_ins(table[functyp])
        r['func_{}_upd'.format(functyp)] = func_table_upd(table[functyp])
        r['func_{}_del'.format(functyp)] = func_table_del(table[functyp])

    for strtyp in strtyps:
        r['{}_lang_items'.format(strtyp)] = string_table_lang_items(
            table[strtyp]
        )
        r['{}_get'.format(strtyp)] = string_table_get(table[strtyp])
        r['{}_ins'.format(strtyp)] = string_table_ins(table[strtyp])
        r['{}_upd'.format(strtyp)] = string_table_upd(table[strtyp])
        r['{}_del'.format(strtyp)] = string_table_del(table[strtyp])

    def universal_hitick(*columns):
        whereclause = [
            getattr(table['lise_globals'].c, col) == bindparam(col)
            for col in columns
        ]
        whereclause.append(
            table['lise_globals'].c.tick <= bindparam('tick')
        )
        return select(
            [
                table['lise_globals'].c.key,
                table['lise_globals'].c.branch,
                func.MAX(table['lise_globals'].c.tick).label('tick')
            ]
        ).where(
            and_(*whereclause)
        ).group_by(
            table['lise_globals'].c.key,
            table['lise_globals'].c.branch
        )

    r['universal_items'] = select(
        [
            table['lise_globals'].c.key,
            table['lise_globals'].c.value
        ]
    ).select_from(universal_hitick('branch'))

    r['universal_get'] = select(
        [table['lise_globals'].c.value]
    ).select_from(universal_hitick('key', 'branch'))

    r['universal_ins'] = insert_cols(
        table['lise_globals'],
        'key',
        'branch',
        'tick',
        'value'
    )

    r['universal_upd'] = update_where(
        table['lise_globals'],
        ['value'],
        ['key', 'branch', 'tick']
    )

    r['characters'] = select([table['characters'].c.character])

    r['ct_characters'] = select([func.COUNT(table['characters'].c.character)])

    r['ct_character'] = select(
        [func.COUNT(table['characters'].c.character)]
    ).where(
        table['characters'].c.character == bindparam('character')
    )

    def arhitick(*cols):
        wheres = [
            getattr(table['active_rules'].c, col) == bindparam(col)
            for col in cols
        ] + [table['active_rules'].c.tick <= bindparam('tick')]
        return select(
            [
                table['active_rules'].c.rulebook,
                table['active_rules'].c.rule,
                table['active_rules'].c.branch,
                func.MAX(table['active_rules'].c.tick).label('tick')
            ]
        ).group_by(
            table['active_rules'].c.rulebook,
            table['active_rules'].c.rule,
            table['active_rules'].c.branch
        ).where(and_(*wheres)).alias('hitick')

    active_rules_hitick = arhitick('branch')

    node_rules_handled = view['node_rules_handled']

    node_rulebook = table['node_rulebook']
    active_rules = table['active_rules']
    rulebooks = table['rulebooks']

    current_active_rules = select(
        [
            active_rules.c.rulebook,
            active_rules.c.rule,
            active_rules.c.branch,
            active_rules.c.tick,
            active_rules.c.active
        ]
    ).select_from(
        active_rules.join(
            active_rules_hitick,
            and_(
                active_rules.c.rulebook == active_rules_hitick.c.rulebook,
                active_rules.c.rule == active_rules_hitick.c.rule,
                active_rules.c.branch == active_rules_hitick.c.branch,
                active_rules.c.tick == active_rules_hitick.c.tick
            )
        )
    ).alias('curactrule')

    nrhandle = select(
        [
            node_rules_handled.c.character,
            node_rules_handled.c.node,
            node_rules_handled.c.rulebook,
            node_rules_handled.c.rule,
            column('1').label('handled')
        ]
    ).where(
        and_(
            node_rules_handled.c.branch == bindparam('branch'),
            node_rules_handled.c.tick == bindparam('tick'),
        )
    ).alias('nrhandle')

    def node_rules(*wheres):
        return select(
            [
                node_rulebook.c.character,
                node_rulebook.c.node,
                node_rulebook.c.rulebook,
                current_active_rules.c.rule,
                current_active_rules.c.active,
            ]
        ).select_from(
            node_rulebook.join(
                rulebooks,
                rulebooks.c.rulebook == node_rulebook.c.rulebook,
            ).join(
                current_active_rules,
                and_(
                    rulebooks.c.rulebook == current_active_rules.c.rulebook,
                    rulebooks.c.rule == current_active_rules.c.rule
                )
            ).join(
                nrhandle,
                and_(
                    node_rulebook.c.character == nrhandle.c.character,
                    node_rulebook.c.node == nrhandle.c.node,
                    node_rulebook.c.rulebook == nrhandle.c.rulebook,
                    current_active_rules.c.rule == nrhandle.c.rule
                ),
                isouter=True
            )
        ).where(
            and_(
                nrhandle.c.handled == null(),
                *wheres
            )
        ).order_by(
            node_rulebook.c.character,
            node_rulebook.c.node,
            rulebooks.c.rulebook,
            rulebooks.c.idx
        )

    r['poll_node_rules'] = node_rules()
    r['node_rules'] = node_rules(
        node_rulebook.c.character == bindparam('character'),
        node_rulebook.c.node == bindparam('node')
    )
    # Note that you have to pass in the branch and tick *twice*, and
    # prior to the character and node, if you're using sqlite

    portal_rulebook = table['portal_rulebook']

    r['portal_rulebook'] = select_where(
        portal_rulebook,
        ['rulebook'],
        ['character', 'nodeA', 'nodeB', 'idx']
    )

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

    portal_rules_handled = table['portal_rules_handled']

    prhandle = select(
        [
            portal_rules_handled.c.character,
            portal_rules_handled.c.nodeA,
            portal_rules_handled.c.nodeB,
            portal_rules_handled.c.idx,
            portal_rules_handled.c.rulebook,
            portal_rules_handled.c.rule,
            column('1').label('handled')
        ]
    ).where(
        and_(
            portal_rules_handled.c.branch == bindparam('branch'),
            portal_rules_handled.c.tick == bindparam('tick')
        )
    ).alias('handle')

    def portal_rules(*wheres):
        return select(
            [
                portal_rulebook.c.character,
                portal_rulebook.c.nodeA,
                portal_rulebook.c.nodeB,
                portal_rulebook.c.idx,
                current_active_rules.c.rule,
                current_active_rules.c.active,
                prhandle.c.handled
            ]
        ).select_from(
            portal_rulebook.join(
                current_active_rules,
                portal_rulebook.c.rulebook == current_active_rules.c.rulebook
            ).join(
                rulebooks,
                and_(
                    rulebooks.c.rulebook == portal_rulebook.c.rulebook,
                    rulebooks.c.rule == current_active_rules.c.rule
                ),
                isouter=True
            ).join(
                prhandle,
                and_(
                    prhandle.c.character == portal_rulebook.c.character,
                    prhandle.c.nodeA == portal_rulebook.c.nodeA,
                    prhandle.c.nodeB == portal_rulebook.c.nodeB,
                    prhandle.c.idx == portal_rulebook.c.idx,
                    prhandle.c.rulebook == portal_rulebook.c.rulebook,
                    prhandle.c.rule == current_active_rules.c.rule
                ),
                isouter=True
            )
        ).where(
            and_(
                prhandle.c.handled == null(),
                *wheres
            )
        ).order_by(
            portal_rulebook.c.character,
            portal_rulebook.c.nodeA,
            portal_rulebook.c.nodeB,
            portal_rulebook.c.idx,
            rulebooks.c.rulebook,
            rulebooks.c.idx
        )

    r['poll_portal_rules'] = portal_rules()
    r['portal_rules'] = portal_rules(
        portal_rulebook.c.character == bindparam('character'),
        portal_rulebook.c.nodeA == bindparam('nodeA'),
        portal_rulebook.c.nodeB == bindparam('nodeB'),
        portal_rulebook.c.idx == bindparam('idx')
    )

    characters = table['characters']

    def poll_char_rules(prefix):
        _rulebook = '{}_rulebook'.format(prefix)
        tabn = '{}_rules_handled'.format(prefix)
        try:
            rules_handled = table[tabn]
        except KeyError:
            rules_handled = view[tabn]
        crhandle = select(
            [
                rules_handled.c.character,
                rules_handled.c.rulebook,
                rules_handled.c.rule,
                column('1').label('handled')
            ]
        ).where(
            and_(
                rules_handled.c.branch == bindparam('branch'),
                rules_handled.c.tick == bindparam('tick')
            )
        ).alias('handle')
        return select(
            [
                characters.c.character,
                getattr(characters.c, _rulebook),
                current_active_rules.c.rule,
                current_active_rules.c.active,
                crhandle.c.handled
            ]
        ).select_from(
            characters.join(
                current_active_rules,
                getattr(characters.c, _rulebook)
                == current_active_rules.c.rulebook
            ).join(
                rulebooks,
                and_(
                    rulebooks.c.rulebook == getattr(characters.c, _rulebook),
                    rulebooks.c.rule == current_active_rules.c.rule
                ),
                isouter=True
            ).join(
                crhandle,
                and_(
                    crhandle.c.character == characters.c.character,
                    crhandle.c.rulebook == getattr(characters.c, _rulebook),
                    crhandle.c.rule == current_active_rules.c.rule
                ),
                isouter=True
            )
        ).where(
            crhandle.c.handled == null()
        ).order_by(
            characters.c.character,
            rulebooks.c.rulebook,
            rulebooks.c.idx
        )

    r['poll_character_rules'] = poll_char_rules('character')
    r['poll_avatar_rules'] = poll_char_rules('avatar')
    r['poll_character_node_rules'] = poll_char_rules('character_node')
    r['poll_character_thing_rules'] = poll_char_rules('character_thing')
    r['poll_character_place_rules'] = poll_char_rules('character_place')
    r['poll_character_portal_rules'] = poll_char_rules('character_portal')

    def handled_character_ruletyp(typ):
        tab = table['{}_rules_handled'.format(typ)]
        return insert_cols(
            tab,
            'character',
            'rulebook',
            'rule',
            'branch',
            'tick'
        )

    r['handled_character_rule'] = handled_character_ruletyp('character')
    r['handled_avatar_rule'] = handled_character_ruletyp('avatar')
    r['handled_character_thing_rule'] \
        = handled_character_ruletyp('character_thing')
    r['handled_character_place_rule'] \
        = handled_character_ruletyp('character_place')
    r['handled_character_portal_rule'] \
        = handled_character_ruletyp('character_portal')

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

    arsr_hitick = arhitick('rulebook', 'branch')

    r['active_rules_rulebook'] = select(
        [
            active_rules.c.rule,
            active_rules.c.active
        ]
    ).select_from(
        active_rules.join(
            arsr_hitick,
            and_(
                active_rules.c.rulebook == arsr_hitick.c.rulebook,
                active_rules.c.rule == arsr_hitick.c.rule,
                active_rules.c.branch == arsr_hitick.c.branch,
                active_rules.c.tick == arsr_hitick.c.tick
            )
        )
    )

    arr_hitick = arhitick('rulebook', 'rule', 'branch')

    r['active_rule_rulebook'] = select(
        [
            active_rules.c.active
        ]
    ).select_from(
        active_rules.join(
            arr_hitick,
            and_(
                active_rules.c.rulebook == arr_hitick.c.rulebook,
                active_rules.c.rule == arr_hitick.c.rule,
                active_rules.c.branch == arr_hitick.c.branch,
                active_rules.c.tick == arr_hitick.c.tick
            )
        )
    )

    # fetch all rules & whether they are active right now (their "activeness")
    # for a given:
    # character;
    # character's avatars;
    # character's things;
    # character's places;
    # character's portals;
    # node (thing or place);
    # portal

    def current_rules_activeness(tbl, col):
        """Query all rules and their activeness for rulebooks given in this
        column of this table.

        """
        hitick = arhitick('branch')
        return select(
            [col, active_rules.c.active]
        ).select_from(
            tbl.join(
                active_rules.join(
                    hitick,
                    and_(
                        active_rules.c.rulebook == hitick.c.rulebook,
                        active_rules.c.rule == hitick.c.rule,
                        active_rules.c.branch == hitick.c.branch,
                        active_rules.c.tick == hitick.c.tick
                    )
                ),
                col == active_rules.c.rulebook
            )
        )

    def character_rulebook_rules_activeness(prefix):
        coln = prefix + '_rulebook'
        return current_rules_activeness(
            characters, getattr(characters.c, coln)
        ).where(
            characters.c.character == bindparam('character')
        )

    r['current_rules_character'] \
        = character_rulebook_rules_activeness('character')
    r['current_rules_avatar'] \
        = character_rulebook_rules_activeness('avatar')
    r['current_rules_character_thing'] \
        = character_rulebook_rules_activeness('character_thing')
    r['current_rules_character_place'] \
        = character_rulebook_rules_activeness('character_place')
    r['current_rules_character_portal'] \
        = character_rulebook_rules_activeness('character_portal')

    r['current_rules_node'] \
        = current_rules_activeness(
            node_rulebook, node_rulebook.c.rulebook
        ).where(
            and_(
                node_rulebook.c.character == bindparam('character'),
                node_rulebook.c.node == bindparam('node')
            )
        )
    r['current_rules_portal'] \
        = current_rules_activeness(
            portal_rulebook, portal_rulebook.c.rulebook
        ).where(
            and_(
                portal_rulebook.c.character == bindparam('character'),
                portal_rulebook.c.nodeA == bindparam('nodeA'),
                portal_rulebook.c.nodeB == bindparam('nodeB')
            )
        )

    def rules_handled_hitick(prefix):
        tbl = table['{}_rules_handled'.format(prefix)]
        return select(
            [
                tbl.c.rulebook,
                tbl.c.rule,
                tbl.c.branch,
                func.MAX(tbl.c.tick).label('tick')
            ]
        ).where(
            and_(
                tbl.c.character == bindparam('character'),
                tbl.c.rulebook == bindparam('rulebook'),
                tbl.c.rule == bindparam('rule'),
                tbl.c.branch == bindparam('branch'),
                tbl.c.tick <= bindparam('tick')
            )
        ).group_by(
            tbl.c.rulebook,
            tbl.c.rule,
            tbl.c.branch
        )

    def active_rule_char(prefix):
        hitick = rules_handled_hitick(prefix)
        return select(
            [
                active_rules.c.active
            ]
        ).select_from(
            active_rules.join(
                hitick,
                and_(
                    active_rules.c.rulebook == hitick.c.rulebook,
                    active_rules.c.rule == hitick.c.rule,
                    active_rules.c.branch == hitick.c.branch,
                    active_rules.c.tick == hitick.c.tick
                )
            )
        )

    r['active_rule_character'] = active_rule_char('character')
    r['active_rule_avatar'] = active_rule_char('avatar')
    r['active_rule_character_thing'] = active_rule_char('character_thing')
    r['active_rule_character_place'] = active_rule_char('character_place')
    r['active_rule_character_portal'] = active_rule_char('character_portal')

    r['rule_ins'] = insert_cols(
        active_rules,
        'rulebook',
        'rule',
        'branch',
        'tick',
        'active'
    )

    r['rule_upd'] = update_where(
        active_rules,
        ['active'],
        ['rulebook', 'rule', 'branch', 'tick']
    )

    r['del_char_things'] = table['things'].delete().where(
        table['things'].c.character == bindparam('character')
    )

    r['del_char_avatars'] = table['avatars'].delete().where(
        table['avatars'].c.character_graph == bindparam('character')
    )

    things = table['things']

    def things_hitick(*cols):
        wheres = [
            getattr(things.c, col) == bindparam(col)
            for col in cols
        ] + [things.c.tick <= bindparam('tick')]
        return select(
            [
                things.c.character,
                things.c.thing,
                things.c.branch,
                func.MAX(things.c.tick).label('tick')
            ]
        ).where(and_(*wheres)).group_by(
            things.c.character,
            things.c.thing,
            things.c.branch
        ).alias('hitick')

    ctb_hitick = things_hitick('character', 'thing', 'branch')

    r['node_is_thing'] = select(
        [things.c.location]
    ).select_from(
        things.join(
            ctb_hitick,
            and_(
                things.c.character == ctb_hitick.c.character,
                things.c.thing == ctb_hitick.c.thing,
                things.c.branch == ctb_hitick.c.branch,
                things.c.tick == ctb_hitick.c.tick
            )
        )
    )

    def rulebook_get_char(rulemap):
        return select(
            [getattr(characters.c, '{}_rulebook'.format(rulemap))]
        ).where(
            characters.c.character == bindparam('character')
        )

    r['character_rulebook'] = select(
        [characters.c.character_rulebook]
    ).where(characters.c.character == bindparam('character'))
    r['rulebook_get_character'] = rulebook_get_char('character')
    r['rulebook_get_avatar'] = rulebook_get_char('avatar')
    r['rulebook_get_character_thing'] = rulebook_get_char('character_thing')
    r['rulebook_get_character_place'] = rulebook_get_char('character_place')
    r['rulebook_get_character_node'] = rulebook_get_char('character_node')
    r['rulebook_get_character_portal'] = rulebook_get_char('character_portal')

    def upd_rulebook_char(rulemap):
        kwvalues = {'{}_rulebook'.format(rulemap): bindparam('rulebook')}
        return characters.update().values(**kwvalues).where(
            characters.c.character == bindparam('character')
        )

    r['upd_rulebook_character'] = upd_rulebook_char('character')
    r['upd_rulebook_avatar'] = upd_rulebook_char('avatar')
    r['upd_rulebook_character_thing'] = upd_rulebook_char('character_thing')
    r['upd_rulebook_character_place'] = upd_rulebook_char('character_place')
    r['upd_rulebook_character_portal'] = upd_rulebook_char('character_portal')

    avatars = table['avatars']

    def hitick_avatars(*cols):
        wheres = [
            getattr(avatars.c, col) == bindparam(col)
            for col in cols
        ] + [avatars.c.tick <= bindparam('tick')]
        return select(
            [
                avatars.c.character_graph,
                avatars.c.avatar_graph,
                avatars.c.avatar_node,
                avatars.c.branch,
                func.MAX(avatars.c.tick).label('tick')
            ]
        ).where(and_(*wheres)).alias('hitick')

    au_hitick = hitick_avatars('avatar_graph', 'avatar_node', 'branch')

    r['avatar_users'] = select(
        [
            avatars.c.avatar_graph
        ]
    ).select_from(
        avatars.join(
            au_hitick,
            and_(
                avatars.c.character_graph == au_hitick.c.character_graph,
                avatars.c.avatar_graph == au_hitick.c.avatar_graph,
                avatars.c.avatar_node == au_hitick.c.avatar_node,
                avatars.c.branch == au_hitick.c.branch,
                avatars.c.tick == au_hitick.c.tick
            )
        )
    )

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

    r['thing_loc_and_next_get'] = select(
        [
            things.c.location,
            things.c.next_location
        ]
    ).select_from(
        things.join(
            ctb_hitick,
            and_(
                things.c.character == ctb_hitick.c.character,
                things.c.thing == ctb_hitick.c.thing,
                things.c.branch == ctb_hitick.c.branch,
                things.c.tick == ctb_hitick.c.tick
            )
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

    nodes = table['nodes']

    def hirev_nodes_extant_cols(*cols):
        wheres = [
            getattr(nodes.c, col) == bindparam(col)
            for col in cols
        ]
        return select(
            [
                nodes.c.graph,
                nodes.c.node,
                nodes.c.branch,
                func.MAX(nodes.c.rev).label('rev')
            ]
        ).where(
            and_(
                nodes.c.rev <= bindparam('rev'),
                *wheres
            )
        ).group_by(
            nodes.c.graph,
            nodes.c.node,
            nodes.c.branch
        ).alias('ext_hirev')

    def nodes_existence_cols(*cols):
        hirev = hirev_nodes_extant_cols(*cols)
        return select(
            [
                nodes.c.graph,
                nodes.c.node,
                nodes.c.branch,
                nodes.c.rev,
                nodes.c.extant
            ]
        ).select_from(
            nodes.join(
                hirev,
                and_(
                    nodes.c.graph == hirev.c.graph,
                    nodes.c.node == hirev.c.node,
                    nodes.c.branch == hirev.c.branch,
                    nodes.c.rev == hirev.c.rev
                )
            )
        ).alias('existence')

    nodes_existence = nodes_existence_cols('graph', 'branch')

    node_existence = nodes_existence_cols('graph', 'node', 'branch')

    cb_hitick = things_hitick('character', 'branch')

    r['thing_loc_items'] = select(
        [
            things.c.thing,
            things.c.location
        ]
    ).select_from(
        things.join(
            cb_hitick,
            and_(
                things.c.character == cb_hitick.c.character,
                things.c.thing == cb_hitick.c.thing,
                things.c.branch == cb_hitick.c.branch,
                things.c.tick == cb_hitick.c.tick
            )
        ).join(
            nodes_existence,
            and_(
                things.c.character == nodes_existence.c.graph,
                things.c.thing == nodes_existence.c.node
            ),
            isouter=True
        )
    ).where(nodes_existence.c.extant)

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

    r['thing_and_loc'] = select(
        [
            things.c.thing,
            things.c.location
        ]
    ).select_from(
        things.join(
            ctb_hitick,
            and_(
                things.c.character == ctb_hitick.c.character,
                things.c.thing == ctb_hitick.c.thing,
                things.c.branch == ctb_hitick.c.branch,
                things.c.tick == ctb_hitick.c.tick
            )
        ).join(
            node_existence,
            and_(
                things.c.character == node_existence.c.graph,
                things.c.thing == node_existence.c.node
            ),
            isouter=True
        )
    ).where(node_existence.c.extant)

    r['character_things_items'] = select(
        [
            things.c.thing,
            things.c.location
        ]
    ).select_from(
        things.join(
            cb_hitick,
            and_(
                things.c.character == cb_hitick.c.character,
                things.c.thing == cb_hitick.c.thing,
                things.c.branch == cb_hitick.c.branch,
                things.c.tick == cb_hitick.c.tick
            )
        )
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

    gb_hitick = hitick_avatars('character_graph', 'branch')

    avatars_recent = avatars.join(
        gb_hitick,
        and_(
            avatars.c.character_graph == gb_hitick.c.character_graph,
            avatars.c.avatar_graph == gb_hitick.c.avatar_graph,
            avatars.c.avatar_node == gb_hitick.c.avatar_node,
            avatars.c.branch == gb_hitick.c.branch,
            avatars.c.tick == gb_hitick.c.tick
        )
    )

    r['avatarness'] = select(
        [
            avatars.c.avatar_graph,
            avatars.c.avatar_node,
            avatars.c.is_avatar
        ]
    ).select_from(
        avatars_recent
    )

    r['avatars_now'] = select(
        [
            avatars.c.avatar_graph,
            avatars.c.avatar_node,
            avatars.c.is_avatar
        ]
    ).select_from(
        avatars_recent.join(
            nodes_existence,
            and_(
                avatars.c.avatar_graph == nodes_existence.c.graph,
                avatars.c.avatar_node == nodes_existence.c.node
            ),
            isouter=True
        )
    ).where(nodes_existence.c.extant)

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

    big_av_hitick = hitick_avatars(
        'character_graph', 'avatar_graph', 'avatar_node', 'branch'
    )

    r['is_avatar_of'] = select(
        [avatars.c.is_avatar]
    ).select_from(
        avatars.join(
            big_av_hitick,
            and_(
                avatars.c.character_graph == big_av_hitick.c.character_graph,
                avatars.c.avatar_graph == big_av_hitick.c.avatar_graph,
                avatars.c.avatar_node == big_av_hitick.c.avatar_node,
                avatars.c.branch == big_av_hitick.c.branch,
                avatars.c.tick == big_av_hitick.c.tick
            )
        )
    )

    senses = table['senses']

    def senses_hitick(*cols):
        wheres = [
            getattr(senses.c, col) == bindparam(col)
            for col in cols
        ] + [senses.c.tick <= bindparam('tick')]
        return select(
            [
                senses.c.character,
                senses.c.sense,
                senses.c.branch,
                func.MAX(senses.c.tick).label('tick')
            ]
        ).where(and_(*wheres)).group_by(
            senses.c.character,
            senses.c.sense,
            senses.c.branch
        ).alias('hitick')

    senses_hitick_csb = senses_hitick('character', 'sense', 'branch')

    r['sense_func_get'] = select(
        [senses.c.function]
    ).select_from(
        senses.join(
            senses_hitick_csb,
            and_(
                senses.c.character == senses_hitick_csb.c.character,
                senses.c.sense == senses_hitick_csb.c.sense,
                senses.c.branch == senses_hitick_csb.c.branch,
                senses.c.tick == senses_hitick_csb.c.tick
            )
        )
    )

    def sense_active_hitick(*cols):
        wheres = [
            getattr(senses.c, col) == bindparam(col)
            for col in cols
        ]
        return select(
            [
                senses.c.character,
                senses.c.sense,
                senses.c.branch,
                func.MAX(senses.c.tick).label('tick')
            ]
        ).where(
            and_(
                or_(
                    senses.c.character == null(),
                    senses.c.character == bindparam('character')
                ),
                senses.c.tick <= bindparam('tick'),
                *wheres
            )
        ).group_by(
            senses.c.character,
            senses.c.sense,
            senses.c.branch
        ).alias('hitick')

    saht_general = sense_active_hitick('branch')

    r['sense_active_items'] = select(
        [
            senses.c.sense,
            senses.c.active
        ]
    ).select_from(
        senses.join(
            saht_general,
            and_(
                senses.c.character == saht_general.c.character,
                senses.c.sense == saht_general.c.sense,
                senses.c.branch == saht_general.c.branch,
                senses.c.tick == saht_general.c.tick
            )
        )
    )

    saht_specific = sense_active_hitick('sense', 'branch')

    r['sense_is_active'] = select(
        [senses.c.active]
    ).select_from(
        senses.join(
            saht_specific,
            and_(
                senses.c.character == saht_specific.c.character,
                senses.c.sense == saht_specific.c.sense,
                senses.c.branch == saht_specific.c.branch,
                senses.c.tick == saht_specific.c.tick
            )
        )
    )

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

    r['rule_triggers'] = select(
        [rules.c.triggers]
    ).where(
        rules.c.rule == bindparam('rule')
    )

    r['upd_rule_triggers'] = rules.update().values(
        triggers=bindparam('triggers')
    ).where(
        rules.c.rule == bindparam('rule')
    )

    r['rule_prereqs'] = select(
        [rules.c.prereqs]
    ).where(
        rules.c.rule == bindparam('rule')
    )

    r['upd_rule_prereqs'] = rules.update().values(
        prereqs=bindparam('prereqs')
    ).where(
        rules.c.rule == bindparam('rule')
    )

    r['rule_actions'] = select(
        [rules.c.actions]
    ).where(
        rules.c.rule == bindparam('rule')
    )

    r['ins_rule'] = rules.insert().values(
        rule=bindparam('rule'),
        triggers=bindparam('triggers'),
        prereqs=bindparam('prereqs'),
        actions=bindparam('actions'),
    )

    r['upd_rule_actions'] = rules.update().values(
        actions=bindparam('actions')
    ).where(
        rules.c.rule == bindparam('rule')
    )

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
        idx=rulebooks.c.idx+1  # not sure if legal
    ).where(
        and_(
            rulebooks.c.rulebook == bindparam('rulebook'),
            rulebooks.c.idx >= bindparam('idx')
        )
    )

    r['rulebook_dec'] = rulebooks.update().values(
        idx=rulebooks.c.idx-column('1')
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

    cab_hitick = hitick_avatars('character_graph', 'avatar_graph', 'branch')

    r['avatar_branch_data'] = select(
        [avatars.c.avatar_node, avatars.c.is_avatar]
    ).select_from(
        avatars.join(
            cab_hitick,
            and_(
                avatars.c.character_graph == cab_hitick.c.character_graph,
                avatars.c.avatar_graph == cab_hitick.c.avatar_graph,
                avatars.c.avatar_node == cab_hitick.c.avatar_node,
                avatars.c.branch == cab_hitick.c.branch,
                avatars.c.tick == cab_hitick.c.tick
            )
        )
    )

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
    print(dumps(r))
