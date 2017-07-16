# This file is part of allegedb, an object relational mapper for versioned graphs.
# Copyright (C) Zachary Spector.
from functools import partial
from sqlalchemy import (
    Table,
    Index,
    Column,
    CheckConstraint,
    ForeignKeyConstraint,
    Integer,
    Boolean,
    String,
    DateTime,
    MetaData,
    ForeignKey,
    select,
    func,
    and_,
    null
)


BaseColumn = Column
Column = partial(BaseColumn, nullable=False)


from sqlalchemy.sql import bindparam
from sqlalchemy.sql.ddl import CreateTable, CreateIndex
from sqlalchemy import create_engine
from json import dumps
from functools import partial

length = 50

TEXT = String(length)


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
        Column('parent_rev', Integer, default=0),
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
        Column('rev', Integer, primary_key=True, default=0),
        Column('value', TEXT, nullable=True)
    )
    Table(
        'nodes', meta,
        Column('graph', TEXT, ForeignKey('graphs.graph'),
               primary_key=True),
        Column('node', TEXT, primary_key=True),
        Column('branch', TEXT, ForeignKey('branches.branch'),
               primary_key=True, default='trunk'),
        Column('rev', Integer, primary_key=True, default=0),
        Column('extant', Boolean)
    )
    Table(
        'node_val', meta,
        Column('graph', TEXT, primary_key=True),
        Column('node', TEXT, primary_key=True),
        Column('key', TEXT, primary_key=True),
        Column('branch', TEXT, ForeignKey('branches.branch'),
               primary_key=True, default='trunk'),
        Column('rev', Integer, primary_key=True, default=0),
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
        Column('idx', Integer, primary_key=True),
        Column('branch', TEXT, ForeignKey('branches.branch'),
               primary_key=True, default='trunk'),
        Column('rev', Integer, primary_key=True, default=0),
        Column('extant', Boolean),
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
        Column('idx', Integer, primary_key=True),
        Column('key', TEXT, primary_key=True),
        Column('branch', TEXT, ForeignKey('branches.branch'),
               primary_key=True, default='trunk'),
        Column('rev', Integer, primary_key=True, default=0),
        Column('value', TEXT, nullable=True),
        ForeignKeyConstraint(
            ['graph', 'orig', 'dest', 'idx'],
            ['edges.graph', 'edges.orig', 'edges.dest', 'edges.idx']
        )
    )
    return meta.tables


def indices_for_table_dict(table):
    return {
        'graph_val_time': Index(
            'graph_val_time_idx',
            table['graph_val'].c.graph,
            table['graph_val'].c.branch,
            table['graph_val'].c.rev
        ),
        'nodes_time': Index(
            'nodes_time_idx',
            table['nodes'].c.graph,
            table['nodes'].c.branch,
            table['nodes'].c.rev
        ),
        'node_val_time': Index(
            'node_val_time_idx',
            table['node_val'].c.graph,
            table['node_val'].c.node,
            table['node_val'].c.branch,
            table['node_val'].c.rev
        ),
        'edges_time': Index(
            'edges_time_idx',
            table['edges'].c.graph,
            table['edges'].c.branch,
            table['edges'].c.rev
        ),
        'edge_val_time': Index(
            'edge_val_time_idx',
            table['edge_val'].c.graph,
            table['edge_val'].c.orig,
            table['edge_val'].c.dest,
            table['edge_val'].c.idx,
            table['edge_val'].c.branch,
            table['edge_val'].c.rev
        )
    }


def queries_for_table_dict(table):
    return {
        'ctbranch': select(
            [func.COUNT(table['branches'].c.branch)]
        ).where(
            table['branches'].c.branch == bindparam('branch')
        ),
        'ctgraph': select(
            [func.COUNT(table['graphs'].c.graph)]
        ).where(
            table['graphs'].c.graph == bindparam('graph')
        ),
        'allbranch': select(
            [
                table['branches'].c.branch,
                table['branches'].c.parent,
                table['branches'].c.parent_rev
            ]
        ).order_by(table['branches'].c.branch),
        'global_get': select(
            [table['global'].c.value]
        ).where(
            table['global'].c.key == bindparam('key')
        ),
        'edge_val_ins': table['edge_val'].insert().prefix_with('OR REPLACE').values(
            graph=bindparam('graph'),
            orig=bindparam('orig'),
            dest=bindparam('dest'),
            idx=bindparam('idx'),
            key=bindparam('key'),
            branch=bindparam('branch'),
            rev=bindparam('rev'),
            value=bindparam('value')
        ),
        'edge_val_upd': table['edge_val'].update().values(
            value=bindparam('value')
        ).where(
            and_(
                table['edge_val'].c.graph == bindparam('graph'),
                table['edge_val'].c.orig == bindparam('orig'),
                table['edge_val'].c.dest == bindparam('dest'),
                table['edge_val'].c.idx == bindparam('idx'),
                table['edge_val'].c.key == bindparam('key'),
                table['edge_val'].c.branch == bindparam('branch'),
                table['edge_val'].c.rev == bindparam('rev')
            )
        ),
        'global_items': select(
            [
                table['global'].c.key,
                table['global'].c.value
            ]
        ),
        'ctglobal': select(
            [func.COUNT(table['global'].c.key)]
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
        'new_branch': table['branches'].insert().values(
            branch=bindparam('branch'),
            parent=bindparam('parent'),
            parent_rev=bindparam('parent_rev')
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
        'parrev': select(
            [table['branches'].c.parent_rev]
        ).where(
            table['branches'].c.branch == bindparam('branch')
        ),
        'parparrev': select(
            [table['branches'].c.parent, table['branches'].c.parent_rev]
        ).where(
            table['branches'].c.branch == bindparam('branch')
        ),
        'global_ins': table['global'].insert().values(
            key=bindparam('key'),
            value=bindparam('value')
        ),
        'global_upd': table['global'].update().values(
            value=bindparam('value')
        ).where(
            table['global'].c.key == bindparam('key')
        ),
        'global_del': table['global'].delete().where(
            table['global'].c.key == bindparam('key')
        ),
        'exist_node_ins': table['nodes'].insert().prefix_with('OR REPLACE').values(
            graph=bindparam('graph'),
            node=bindparam('node'),
            branch=bindparam('branch'),
            rev=bindparam('rev'),
            extant=bindparam('extant')
        ),
        'exist_node_upd': table['nodes'].update().values(
            extant=bindparam('extant')
        ).where(
            and_(
                table['nodes'].c.graph == bindparam('graph'),
                table['nodes'].c.node == bindparam('node'),
                table['nodes'].c.branch == bindparam('branch'),
                table['nodes'].c.rev == bindparam('rev')
            )
        ),
        'graphs_types': select([
            table['graphs'].c.graph,
            table['graphs'].c.type
        ]),
        'nodes_dump': select([
            table['nodes'].c.graph,
            table['nodes'].c.node,
            table['nodes'].c.branch,
            table['nodes'].c.rev,
            table['nodes'].c.extant
        ]).order_by(
            table['nodes'].c.graph,
            table['nodes'].c.branch,
            table['nodes'].c.rev,
            table['nodes'].c.node
        ),
        'graph_val_dump': select([
            table['graph_val'].c.graph,
            table['graph_val'].c.key,
            table['graph_val'].c.branch,
            table['graph_val'].c.rev,
            table['graph_val'].c.value
        ]).order_by(
            table['graph_val'].c.graph,
            table['graph_val'].c.branch,
            table['graph_val'].c.rev,
            table['graph_val'].c.key
        ),
        'graph_val_ins': table['graph_val'].insert().prefix_with('OR REPLACE').values(
            graph=bindparam('graph'),
            key=bindparam('key'),
            branch=bindparam('branch'),
            rev=bindparam('rev'),
            value=bindparam('value')
        ),
        'graph_val_upd': table['graph_val'].update().values(
            value=bindparam('value')
        ).where(
            and_(
                table['graph_val'].c.graph == bindparam('graph'),
                table['graph_val'].c.key == bindparam('key'),
                table['graph_val'].c.branch == bindparam('branch'),
                table['graph_val'].c.rev == bindparam('rev')
            )
        ),
        'node_val_dump': select([
            table['node_val'].c.graph,
            table['node_val'].c.node,
            table['node_val'].c.key,
            table['node_val'].c.branch,
            table['node_val'].c.rev,
            table['node_val'].c.value
        ]).order_by(
            table['node_val'].c.graph,
            table['node_val'].c.node,
            table['node_val'].c.branch,
            table['node_val'].c.rev,
            table['node_val'].c.key
        ),
        'node_val_ins': table['node_val'].insert().prefix_with('OR REPLACE').values(
            graph=bindparam('graph'),
            node=bindparam('node'),
            key=bindparam('key'),
            branch=bindparam('branch'),
            rev=bindparam('rev'),
            value=bindparam('value')
        ),
        'edges_dump': select([
            table['edges'].c.graph,
            table['edges'].c.orig,
            table['edges'].c.dest,
            table['edges'].c.idx,
            table['edges'].c.branch,
            table['edges'].c.rev,
            table['edges'].c.extant
        ]).order_by(
            table['edges'].c.graph,
            table['edges'].c.branch,
            table['edges'].c.rev,
            table['edges'].c.orig,
            table['edges'].c.dest,
            table['edges'].c.idx
        ),
        'edge_exist_ins': table['edges'].insert().prefix_with('OR REPLACE').values(
            graph=bindparam('graph'),
            orig=bindparam('orig'),
            dest=bindparam('dest'),
            idx=bindparam('idx'),
            branch=bindparam('branch'),
            rev=bindparam('rev'),
            extant=bindparam('extant')
        ),
        'edge_exist_upd': table['edges'].update().values(
            extant=bindparam('extant')
        ).where(
            and_(
                table['edges'].c.graph == bindparam('graph'),
                table['edges'].c.orig == bindparam('orig'),
                table['edges'].c.dest == bindparam('dest'),
                table['edges'].c.idx == bindparam('idx'),
                table['edges'].c.branch == bindparam('branch'),
                table['edges'].c.rev == bindparam('rev')
            )
        ),
        'edge_val_dump': select([
            table['edge_val'].c.graph,
            table['edge_val'].c.orig,
            table['edge_val'].c.dest,
            table['edge_val'].c.idx,
            table['edge_val'].c.key,
            table['edge_val'].c.branch,
            table['edge_val'].c.rev,
            table['edge_val'].c.value
        ]).order_by(
            table['edge_val'].c.graph,
            table['edge_val'].c.orig,
            table['edge_val'].c.dest,
            table['edge_val'].c.idx,
            table['edge_val'].c.branch,
            table['edge_val'].c.rev,
            table['edge_val'].c.key
        )
    }


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
