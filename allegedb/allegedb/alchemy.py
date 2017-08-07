# This file is part of allegedb, an object relational mapper for versioned graphs.
# Copyright (C) Zachary Spector.
from functools import partial
from sqlalchemy import (
    Table,
    Index,
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


from sqlalchemy.sql import bindparam
from sqlalchemy.sql.ddl import CreateTable, CreateIndex
from sqlalchemy import create_engine
from json import dumps
from functools import partial

length = 50


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
        Column('parent', TEXT, default='trunk'),
        Column('parent_turn', INT, default=0),
        CheckConstraint('branch<>parent')
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
        'global_update': table['global'].insert().values(
            key=bindparam('key'),
            value=bindparam('value')
        ),
        'new_graph': table['graphs'].insert().values(
            graph=bindparam('graph'),
            type=bindparam('type')
        ),
        'graph_type': select(
            [table['graphs'].c.type]
        ).where(
            table['graphs'].c.graph == bindparam('graph')
        ),
        'del_edge_val_graph': table['edge_val'].delete().where(
            table['edge_val'].c.graph == bindparam('graph')
        ),
        'del_edge_graph': table['edges'].delete().where(
            table['edges'].c.graph == bindparam('graph')
        ),
        'del_node_val_graph': table['node_val'].delete().where(
            table['node_val'].c.graph == bindparam('graph')
        ),
        'del_node_graph': table['nodes'].delete().where(
            table['nodes'].c.graph == bindparam('graph')
        ),
        'del_graph': table['graphs'].delete().where(
            table['graphs'].c.graph == bindparam('graph')
        ),
        'global_update': table['global'].update().values(
            value=bindparam('value')
        ).where(
            table['global'].c.key == bindparam('key')
        ),
        'global_delete': table['global'].delete().where(
            table['global'].c.key == bindparam('key')
        ),
        'graphs_types': select([
            table['graphs'].c.graph,
            table['graphs'].c.type
        ])
    }
    for t in table.values():
        r[t.name + '_dump'] = select(list(t.c.values())).order_by(*t.primary_key)
        r[t.name + '_insert'] = t.insert().values(tuple(bindparam(cname) for cname in t.c.keys()))
        r[t.name + '_count'] = select([func.COUNT('*')]).select_from(t)
    return r


def compile_sql(dialect, meta):
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
    e = create_engine('sqlite:///:memory:')
    out = dict(
        (k, str(v)) for (k, v) in
        compile_sql(e.dialect, MetaData()).items()
    )

    print(dumps(out, indent=4, sort_keys=True))
