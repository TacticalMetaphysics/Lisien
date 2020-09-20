# This file is part of allegedb, an object relational mapper for versioned graphs.
# Copyright (C) Zachary Spector. public@zacharyspector.com
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
"""SQLAlchemy code to generate the SQL used by the allegedb ORM

If SQLAlchemy is installed at runtime, this will use it to generate SQL on the fly;
if it is not, allegedb can use a pregenerated file "sqlite.json" to store graphs in
a SQLite database. Generate sqlite.json by running this on the command line:

python3 alchemy.py >sqlite.json

"""
from functools import partial
from sqlalchemy import (
    Table,
    ForeignKey,
    Column,
    CheckConstraint,
    ForeignKeyConstraint,
    INT,
    TEXT,
    BOOLEAN,
    MetaData,
    ForeignKey,
    select,
    func,
)


BaseColumn = Column
Column = partial(BaseColumn, nullable=False)


from sqlalchemy.sql import bindparam, and_, or_
from json import dumps
from functools import partial


def tables_for_meta(meta):
    Table(
        'global', meta,
        Column('key', TEXT, primary_key=True),
        Column('value', TEXT, nullable=True)
    )
    Table(
        'branches', meta,
        Column(
            'branch', TEXT, ForeignKey('branches.parent'),
            primary_key=True, default='trunk'
        ),
        Column('parent', TEXT, default='trunk', nullable=True),
        Column('parent_turn', INT, default=0),
        Column('parent_tick', INT, default=0),
        Column('end_turn', INT, default=0),
        Column('end_tick', INT, default=0),
        CheckConstraint('branch<>parent')
    )
    Table(
        'turns', meta,
        Column('branch', TEXT, primary_key=True),
        Column('turn', INT, primary_key=True),
        Column('end_tick', INT),
        Column('plan_end_tick', INT)
    )
    Table(
        'graphs', meta,
        Column('graph', TEXT, primary_key=True),
        Column('type', TEXT, default='Graph'),
        CheckConstraint(
            "type IN ('Graph', 'DiGraph', 'MultiGraph', 'MultiDiGraph')"
        )
    )
    Table(
        'keyframes', meta,
        Column('graph', TEXT, ForeignKey('graphs.graph'), primary_key=True),
        Column('branch', TEXT, ForeignKey('branches.branch'), primary_key=True,
               default='trunk'),
        Column('turn', INT, primary_key=True, default=0),
        Column('tick', INT, primary_key=True, default=0),
        Column('nodes', TEXT),
        Column('edges', TEXT),
        Column('graph_val', TEXT)
    )
    Table(
        'graph_val', meta,
        Column('graph', TEXT, ForeignKey('graphs.graph'),
               primary_key=True),
        Column('key', TEXT, primary_key=True),
        Column('branch', TEXT, ForeignKey('branches.branch'),
               primary_key=True, default='trunk'),
        Column('turn', INT, primary_key=True, default=0),
        Column('tick', INT, primary_key=True, default=0),
        Column('value', TEXT, nullable=True)
    )
    Table(
        'nodes', meta,
        Column('graph', TEXT, ForeignKey('graphs.graph'),
               primary_key=True),
        Column('node', TEXT, primary_key=True),
        Column('branch', TEXT, ForeignKey('branches.branch'),
               primary_key=True, default='trunk'),
        Column('turn', INT, primary_key=True, default=0),
        Column('tick', INT, primary_key=True, default=0),
        Column('extant', BOOLEAN)
    )
    Table(
        'node_val', meta,
        Column('graph', TEXT, primary_key=True),
        Column('node', TEXT, primary_key=True),
        Column('key', TEXT, primary_key=True),
        Column('branch', TEXT, ForeignKey('branches.branch'),
               primary_key=True, default='trunk'),
        Column('turn', INT, primary_key=True, default=0),
        Column('tick', INT, primary_key=True, default=0),
        Column('value', TEXT, nullable=True),
        ForeignKeyConstraint(
            ['graph', 'node'], ['nodes.graph', 'nodes.node']
        )
    )
    Table(
        'edges', meta,
        Column('graph', TEXT, ForeignKey('graphs.graph'),
               primary_key=True),
        Column('orig', TEXT, primary_key=True),
        Column('dest', TEXT, primary_key=True),
        Column('idx', INT, primary_key=True),
        Column('branch', TEXT, ForeignKey('branches.branch'),
               primary_key=True, default='trunk'),
        Column('turn', INT, primary_key=True, default=0),
        Column('tick', INT, primary_key=True, default=0),
        Column('extant', BOOLEAN),
        ForeignKeyConstraint(
            ['graph', 'orig'], ['nodes.graph', 'nodes.node']
        ),
        ForeignKeyConstraint(
            ['graph', 'dest'], ['nodes.graph', 'nodes.node']
        )
    )
    Table(
        'edge_val', meta,
        Column('graph', TEXT, primary_key=True),
        Column('orig', TEXT, primary_key=True),
        Column('dest', TEXT, primary_key=True),
        Column('idx', INT, primary_key=True),
        Column('key', TEXT, primary_key=True),
        Column('branch', TEXT, ForeignKey('branches.branch'),
               primary_key=True, default='trunk'),
        Column('turn', INT, primary_key=True, default=0),
        Column('tick', INT, primary_key=True, default=0),
        Column('value', TEXT, nullable=True),
        ForeignKeyConstraint(
            ['graph', 'orig', 'dest', 'idx'],
            ['edges.graph', 'edges.orig', 'edges.dest', 'edges.idx']
        )
    )
    Table(
        'plans', meta,
        Column('id', INT, primary_key=True),
        Column('branch', TEXT),
        Column('turn', INT),
        Column('tick', INT)
    )
    Table(
        'plan_ticks', meta,
        Column('plan_id', INT, primary_key=True),
        Column('turn', INT, primary_key=True),
        Column('tick', INT, primary_key=True),
        ForeignKeyConstraint(
            ('plan_id',),
            ('plans.id',)
        )
    )
    return meta.tables


def indices_for_table_dict(table):
    return {}


def queries_for_table_dict(table):
    r = {
        'global_get': select(
            [table['global'].c.value]
        ).where(
            table['global'].c.key == bindparam('key')
        ),
        'global_update': table['global'].update().values(
            value=bindparam('value')
        ).where(table['global'].c.key == bindparam('key')),
        'graph_type': select(
            [table['graphs'].c.type]
        ).where(
            table['graphs'].c.graph == bindparam('graph')
        ),
        'del_edge_val_graph': table['edge_val'].delete().where(
            table['edge_val'].c.graph == bindparam('graph')
        ),
        'del_edge_val_after': table['edge_val'].delete().where(and_(
            table['edge_val'].c.graph == bindparam('graph'),
            table['edge_val'].c.orig == bindparam('orig'),
            table['edge_val'].c.dest == bindparam('dest'),
            table['edge_val'].c.idx == bindparam('idx'),
            table['edge_val'].c.key == bindparam('key'),
            table['edge_val'].c.branch == bindparam('branch'),
            or_(
                table['edge_val'].c.turn > bindparam('turn'),
                and_(
                    table['edge_val'].c.turn == bindparam('turn'),
                    table['edge_val'].c.tick >= bindparam('tick')
                )
            )
        )),
        'del_edges_graph': table['edges'].delete().where(
            table['edges'].c.graph == bindparam('graph')
        ),
        'del_edges_after': table['edges'].delete().where(and_(
            table['edges'].c.graph == bindparam('graph'),
            table['edges'].c.orig == bindparam('orig'),
            table['edges'].c.dest == bindparam('dest'),
            table['edges'].c.idx == bindparam('idx'),
            table['edges'].c.branch == bindparam('branch'),
            or_(
                table['edges'].c.turn > bindparam('turn'),
                and_(
                    table['edges'].c.turn == bindparam('turn'),
                    table['edges'].c.tick >= bindparam('tick')
                )
            )
        )),
        'del_nodes_graph': table['nodes'].delete().where(
            table['nodes'].c.graph == bindparam('graph')
        ),
        'del_nodes_after': table['nodes'].delete().where(and_(
            table['nodes'].c.graph == bindparam('graph'),
            table['nodes'].c.node == bindparam('node'),
            table['nodes'].c.branch == bindparam('branch'),
            or_(
                table['nodes'].c.turn > bindparam('turn'),
                and_(
                    table['nodes'].c.turn == bindparam('turn'),
                    table['nodes'].c.tick >= bindparam('tick')
                )
            )
        )),
        'del_node_val_graph': table['node_val'].delete().where(
            table['node_val'].c.graph == bindparam('graph')
        ),
        'del_node_val_after': table['node_val'].delete().where(and_(
            table['node_val'].c.graph == bindparam('graph'),
            table['node_val'].c.node == bindparam('node'),
            table['node_val'].c.key == bindparam('key'),
            table['node_val'].c.branch == bindparam('branch'),
            or_(
                table['node_val'].c.turn > bindparam('turn'),
                and_(
                    table['node_val'].c.turn == bindparam('turn'),
                    table['node_val'].c.tick >= bindparam('tick')
                )
            )
        )),
        'del_graph': table['graphs'].delete().where(
            table['graphs'].c.graph == bindparam('graph')
        ),
        'del_graph_val_after': table['graph_val'].delete().where(and_(
            table['graph_val'].c.graph == bindparam('graph'),
            table['graph_val'].c.key == bindparam('key'),
            table['graph_val'].c.branch == bindparam('branch'),
            or_(
                table['graph_val'].c.turn > bindparam('turn'),
                and_(
                    table['graph_val'].c.turn == bindparam('turn'),
                    table['graph_val'].c.tick >= bindparam('tick')
                )
            )
        )),
        'global_delete': table['global'].delete().where(
            table['global'].c.key == bindparam('key')
        ),
        'graphs_types': select([
            table['graphs'].c.graph,
            table['graphs'].c.type
        ]),
        'graphs_named': select([func.COUNT()]).select_from(table['graphs']).where(
            table['graphs'].c.graph == bindparam('graph')
        ),
        'update_branches': table['branches'].update().values(
            parent=bindparam('parent'),
            parent_turn=bindparam('parent_turn'),
            parent_tick=bindparam('parent_tick'),
            end_turn=bindparam('end_turn'),
            end_tick=bindparam('end_tick')
        ).where(table['branches'].c.branch == bindparam('branch')),
        'update_turns': table['turns'].update().values(
            end_tick=bindparam('end_tick'),
            plan_end_tick=bindparam('plan_end_tick')
        ).where(and_(
            table['turns'].c.branch == bindparam('branch'),
            table['turns'].c.turn == bindparam('turn')
        ))
    }
    for t in table.values():
        key = list(t.primary_key)
        if 'branch' in t.columns and 'turn' in t.columns and 'tick' in t.columns:
            branch = t.columns['branch']
            turn = t.columns['turn']
            tick = t.columns['tick']
            if branch in key and turn in key and tick in key:
                key = [branch, turn, tick]
                r[t.name + '_del_time'] = t.delete().where(and_(
                    t.c.branch == bindparam('branch'),
                    t.c.turn == bindparam('turn'),
                    t.c.tick == bindparam('tick')
                ))
        r[t.name + '_dump'] = select(list(t.c.values())).order_by(*key)
        r[t.name + '_insert'] = t.insert().values(tuple(bindparam(cname) for cname in t.c.keys()))
        r[t.name + '_count'] = select([func.COUNT()]).select_from(t)
        r[t.name + '_del'] = t.delete().where(and_(*[c == bindparam(c.name) for c in t.primary_key]))
    return r


def compile_sql(dialect, meta):
    from sqlalchemy.sql.ddl import CreateTable, CreateIndex
    r = {}
    table = tables_for_meta(meta)
    index = indices_for_table_dict(table)
    query = queries_for_table_dict(table)

    for t in table.values():
        r['create_' + t.name] = CreateTable(t).compile(dialect=dialect)
    for (tab, idx) in index.items():
        r['index_' + tab] = CreateIndex(idx).compile(dialect=dialect)
    for (name, q) in query.items():
        r[name] = q.compile(dialect=dialect)

    return r


class Alchemist(object):
    """Holds an engine and runs queries on it.

    """
    def __init__(self, engine):
        self.engine = engine
        self.conn = self.engine.connect()
        self.meta = MetaData()
        self.sql = compile_sql(self.engine.dialect, self.meta)
        def caller(k, *largs):
            statement = self.sql[k]
            if hasattr(statement, 'positiontup'):
                return self.conn.execute(statement, **dict(zip(statement.positiontup, largs)))
            elif largs:
                raise TypeError("{} is a DDL query, I think".format(k))
            return self.conn.execute(statement)
        def manycaller(k, *largs):
            statement = self.sql[k]
            return self.conn.execute(statement, *(dict(zip(statement.positiontup, larg)) for larg in largs))
        class Many(object):
            pass
        self.many = Many()
        for (key, query) in self.sql.items():
            setattr(self, key, partial(caller, key))
            setattr(self.many, key, partial(manycaller, key))


if __name__ == '__main__':
    from sqlalchemy.dialects.sqlite.pysqlite import SQLiteDialect_pysqlite
    out = dict(
        (k, str(v)) for (k, v) in
        compile_sql(SQLiteDialect_pysqlite(), MetaData()).items()
    )

    print(dumps(out, indent=4, sort_keys=True))
