# This file is part of gorm, an object relational mapper for versioned graphs.
# Copyright (C) 2014 Zachary Spector.
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
from sqlalchemy.sql import bindparam
from sqlalchemy.sql.ddl import CreateTable, CreateIndex
from sqlalchemy import create_engine
from json import dumps
from functools import partial

length = 50

TEXT = String(length)


def tables_for_meta(meta):
    return {
        'global': Table(
            'global', meta,
            Column('key', TEXT, primary_key=True),
            Column('date', DateTime, nullable=True),
            Column('creator', TEXT, nullable=True),
            Column('description', TEXT, nullable=True),
            Column('value', TEXT, nullable=True)
        ),
        'branches': Table(
            'branches', meta,
            Column(
                'branch', TEXT, ForeignKey('branches.parent'),
                primary_key=True, default='master'
            ),
            Column('date', DateTime, nullable=True),
            Column('creator', TEXT, nullable=True),
            Column('description', TEXT, nullable=True),
            Column('parent', TEXT, default='master'),
            Column('parent_rev', Integer, default=0)
        ),
        'graphs': Table(
            'graphs', meta,
            Column('graph', TEXT, primary_key=True),
            Column('date', DateTime, nullable=True),
            Column('creator', TEXT, nullable=True),
            Column('description', TEXT, nullable=True),
            Column('type', TEXT, default='Graph'),
            CheckConstraint(
                "type IN ('Graph', 'DiGraph', 'MultiGraph', 'MultiDiGraph')"
            )
        ),
        'graph_val': Table(
            'graph_val', meta,
            Column('graph', TEXT, ForeignKey('graphs.graph'),
                   primary_key=True),
            Column('key', TEXT, primary_key=True),
            Column('branch', TEXT, ForeignKey('branches.branch'),
                   primary_key=True, default='master'),
            Column('rev', Integer, primary_key=True, default=0),
            Column('date', DateTime, nullable=True),
            Column('contributor', TEXT, nullable=True),
            Column('description', TEXT, nullable=True),
            Column('value', TEXT, nullable=True)
        ),
        'nodes': Table(
            'nodes', meta,
            Column('graph', TEXT, ForeignKey('graphs.graph'),
                   primary_key=True),
            Column('node', TEXT, primary_key=True),
            Column('branch', TEXT, ForeignKey('branches.branch'),
                   primary_key=True, default='master'),
            Column('rev', Integer, primary_key=True, default=0),
            Column('date', DateTime, nullable=True),
            Column('creator', TEXT, nullable=True),
            Column('description', TEXT, nullable=True),
            Column('extant', Boolean)
        ),
        'node_val': Table(
            'node_val', meta,
            Column('graph', TEXT, primary_key=True),
            Column('node', TEXT, primary_key=True),
            Column('key', TEXT, primary_key=True),
            Column('branch', TEXT, ForeignKey('branches.branch'),
                   primary_key=True, default='master'),
            Column('rev', Integer, primary_key=True, default=0),
            Column('date', DateTime, nullable=True),
            Column('contributor', TEXT, nullable=True),
            Column('description', TEXT, nullable=True),
            Column('value', TEXT, nullable=True),
            ForeignKeyConstraint(
                ['graph', 'node'], ['nodes.graph', 'nodes.node']
            )
        ),
        'edges': Table(
            'edges', meta,
            Column('graph', TEXT, ForeignKey('graphs.graph'),
                   primary_key=True),
            Column('nodeA', TEXT, primary_key=True),
            Column('nodeB', TEXT, primary_key=True),
            Column('idx', Integer, primary_key=True),
            Column('branch', TEXT, ForeignKey('branches.branch'),
                   primary_key=True, default='master'),
            Column('rev', Integer, primary_key=True, default=0),
            Column('date', DateTime, nullable=True),
            Column('creator', TEXT, nullable=True),
            Column('description', TEXT, nullable=True),
            Column('extant', Boolean),
            ForeignKeyConstraint(
                ['graph', 'nodeA'], ['nodes.graph', 'nodes.node']
            ),
            ForeignKeyConstraint(
                ['graph', 'nodeB'], ['nodes.graph', 'nodes.node']
            )
        ),
        'edge_val': Table(
            'edge_val', meta,
            Column('graph', TEXT, primary_key=True),
            Column('nodeA', TEXT, primary_key=True),
            Column('nodeB', TEXT, primary_key=True),
            Column('idx', Integer, primary_key=True),
            Column('key', TEXT, primary_key=True),
            Column('branch', TEXT, ForeignKey('branches.branch'),
                   primary_key=True, default='master'),
            Column('rev', Integer, primary_key=True, default=0),
            Column('date', DateTime, nullable=True),
            Column('contributor', TEXT, nullable=True),
            Column('description', TEXT, nullable=True),
            Column('value', TEXT, nullable=True),
            ForeignKeyConstraint(
                ['graph', 'nodeA', 'nodeB', 'idx'],
                ['edges.graph', 'edges.nodeA', 'edges.nodeB', 'edges.idx']
            )
        )
    }


def indices_for_table_dict(table):
    return {
        'graph_val': Index(
            "graph_val_idx",
            table['graph_val'].c.graph,
            table['graph_val'].c.key
        ),
        'nodes': Index(
            "nodes_idx",
            table['nodes'].c.graph,
            table['nodes'].c.node
        ),
        'node_val': Index(
            "node_val_idx",
            table['node_val'].c.graph,
            table['node_val'].c.node
        ),
        'edges': Index(
            "edges_idx",
            table['edges'].c.graph,
            table['edges'].c.nodeA,
            table['edges'].c.nodeB,
            table['edges'].c.idx
        ),
        'edge_val': Index(
            "edge_val_idx",
            table['edge_val'].c.graph,
            table['edge_val'].c.nodeA,
            table['edge_val'].c.nodeB,
            table['edge_val'].c.idx,
            table['edge_val'].c.key
        )
    }


def queries_for_table_dict(table):
    def hirev_nodes_join(wheres):
        hirev = select(
            [
                table['nodes'].c.graph,
                table['nodes'].c.node,
                table['nodes'].c.branch,
                func.MAX(table['nodes'].c.rev).label('rev')
            ]
        ).where(and_(*wheres)).group_by(
            table['nodes'].c.graph,
            table['nodes'].c.node,
            table['nodes'].c.branch
        ).alias('hirev')
        return table['nodes'].join(
            hirev,
            and_(
                table['nodes'].c.graph == hirev.c.graph,
                table['nodes'].c.node == hirev.c.node,
                table['nodes'].c.branch == hirev.c.branch,
                table['nodes'].c.rev == hirev.c.rev
            )
        )

    def hirev_graph_val_join(wheres):
        hirev = select(
            [
                table['graph_val'].c.graph,
                table['graph_val'].c.key,
                table['graph_val'].c.branch,
                func.MAX(table['graph_val'].c.rev).label('rev')
            ]
        ).where(and_(*wheres)).group_by(
            table['graph_val'].c.graph,
            table['graph_val'].c.key,
            table['graph_val'].c.branch
        ).alias('hirev')
        return table['graph_val'].join(
            hirev,
            and_(
                table['graph_val'].c.graph == hirev.c.graph,
                table['graph_val'].c.key == hirev.c.key,
                table['graph_val'].c.branch == hirev.c.branch,
                table['graph_val'].c.rev == hirev.c.rev
            )
        )

    def node_val_hirev_join(wheres):
        hirev = select(
            [
                table['node_val'].c.graph,
                table['node_val'].c.node,
                table['node_val'].c.branch,
                table['node_val'].c.key,
                func.MAX(table['node_val'].c.rev).label('rev')
            ]
        ).where(and_(*wheres)).group_by(
            table['node_val'].c.graph,
            table['node_val'].c.node,
            table['node_val'].c.branch,
            table['node_val'].c.key
        ).alias('hirev')

        return table['node_val'].join(
            hirev,
            and_(
                table['node_val'].c.graph == hirev.c.graph,
                table['node_val'].c.node == hirev.c.node,
                table['node_val'].c.key == hirev.c.key,
                table['node_val'].c.branch == hirev.c.branch,
                table['node_val'].c.rev == hirev.c.rev
            )
        )

    def edges_recent_join(wheres=None):
        hirev = select(
            [
                table['edges'].c.graph,
                table['edges'].c.nodeA,
                table['edges'].c.nodeB,
                table['edges'].c.idx,
                table['edges'].c.branch,
                func.MAX(table['edges'].c.rev).label('rev')
            ]
        )
        if wheres:
            hirev = hirev.where(and_(*wheres))
        hirev = hirev.group_by(
            table['edges'].c.graph,
            table['edges'].c.nodeA,
            table['edges'].c.nodeB,
            table['edges'].c.idx,
            table['edges'].c.branch
        ).alias('hirev')
        return table['edges'].join(
            hirev,
            and_(
                table['edges'].c.graph == hirev.c.graph,
                table['edges'].c.nodeA == hirev.c.nodeA,
                table['edges'].c.nodeB == hirev.c.nodeB,
                table['edges'].c.idx == hirev.c.idx,
                table['edges'].c.branch == hirev.c.branch,
                table['edges'].c.rev == hirev.c.rev
            )
        )

    def edge_val_recent_join(wheres=None):
        hirev = select(
            [
                table['edge_val'].c.graph,
                table['edge_val'].c.nodeA,
                table['edge_val'].c.nodeB,
                table['edge_val'].c.idx,
                table['edge_val'].c.key,
                table['edge_val'].c.branch,
                func.MAX(table['edge_val'].c.rev).label('rev')
            ]
        )
        if wheres:
            hirev = hirev.where(
                and_(*wheres)
            )
        hirev = hirev.group_by(
            table['edge_val'].c.graph,
            table['edge_val'].c.nodeA,
            table['edge_val'].c.nodeB,
            table['edge_val'].c.idx,
            table['edge_val'].c.key,
            table['edge_val'].c.branch
        ).alias('hirev')
        return table['edge_val'].join(
            hirev,
            and_(
                table['edge_val'].c.graph == hirev.c.graph,
                table['edge_val'].c.nodeA == hirev.c.nodeA,
                table['edge_val'].c.nodeB == hirev.c.nodeB,
                table['edge_val'].c.idx == hirev.c.idx,
                table['edge_val'].c.branch == hirev.c.branch,
                table['edge_val'].c.rev == hirev.c.rev
            )
        )

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
        ),
        'global_get': select(
            [table['global'].c.value]
        ).where(
            table['global'].c.key == bindparam('key')
        ),
        'edge_val_ins': table['edge_val'].insert().prefix_with('OR REPLACE').values(
            graph=bindparam('graph'),
            nodeA=bindparam('orig'),
            nodeB=bindparam('dest'),
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
                table['edge_val'].c.nodeA == bindparam('orig'),
                table['edge_val'].c.nodeB == bindparam('dest'),
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
        'nodes_extant': select(
            [table['nodes'].c.node]
        ).select_from(
            hirev_nodes_join(
                [
                    table['nodes'].c.graph == bindparam('graph'),
                    table['nodes'].c.branch == bindparam('branch'),
                    table['nodes'].c.rev <= bindparam('rev')
                ]
            )
        ).where(
            table['nodes'].c.extant
        ),
        'node_exists': select(
            [table['nodes'].c.extant]
        ).select_from(
            hirev_nodes_join(
                [
                    table['nodes'].c.graph == bindparam('graph'),
                    table['nodes'].c.node == bindparam('node'),
                    table['nodes'].c.branch == bindparam('branch'),
                    table['nodes'].c.rev <= bindparam('rev')
                ]
            )
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
        ]),
        'graph_val_items': select(
            [
                table['graph_val'].c.key,
                table['graph_val'].c.value
            ]
        ).select_from(
            hirev_graph_val_join(
                [
                    table['graph_val'].c.graph == bindparam('graph'),
                    table['graph_val'].c.branch == bindparam('branch'),
                    table['graph_val'].c.rev <= bindparam('rev')
                ]
            )
        ),
        'graph_val_dump': select([
            table['graph_val'].c.graph,
            table['graph_val'].c.key,
            table['graph_val'].c.branch,
            table['graph_val'].c.rev,
            table['graph_val'].c.value
        ]),
        'graph_val_get': select(
            [
                table['graph_val'].c.value
            ]
        ).select_from(
            hirev_graph_val_join(
                [
                    table['graph_val'].c.graph == bindparam('graph'),
                    table['graph_val'].c.key == bindparam('key'),
                    table['graph_val'].c.branch == bindparam('branch'),
                    table['graph_val'].c.rev <= bindparam('rev')
                ]
            )
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
        'node_val_items': select(
            [
                table['node_val'].c.key,
                table['node_val'].c.value
            ]
        ).select_from(
            node_val_hirev_join(
                [
                    table['node_val'].c.graph == bindparam('graph'),
                    table['node_val'].c.node == bindparam('node'),
                    table['node_val'].c.branch == bindparam('branch'),
                    table['node_val'].c.rev <= bindparam('rev')
                ]
            )
        ),
        'node_val_dump': select([
            table['node_val'].c.graph,
            table['node_val'].c.node,
            table['node_val'].c.key,
            table['node_val'].c.branch,
            table['node_val'].c.rev,
            table['node_val'].c.value
        ]),
        'node_val_get': select(
            [
                table['node_val'].c.value
            ]
        ).select_from(
            node_val_hirev_join(
                [
                    table['node_val'].c.graph == bindparam('graph'),
                    table['node_val'].c.node == bindparam('node'),
                    table['node_val'].c.key == bindparam('key'),
                    table['node_val'].c.branch == bindparam('branch'),
                    table['node_val'].c.rev <= bindparam('rev')
                ]
            )
        ).where(
            table['node_val'].c.value != null()
        ),
        'node_val_ins': table['node_val'].insert().prefix_with('OR REPLACE').values(
            graph=bindparam('graph'),
            node=bindparam('node'),
            key=bindparam('key'),
            branch=bindparam('branch'),
            rev=bindparam('rev'),
            value=bindparam('value')
        ),
        'edge_exists': select(
            [table['edges'].c.extant]
        ).select_from(
            edges_recent_join(
                [
                    table['edges'].c.graph == bindparam('graph'),
                    table['edges'].c.nodeA == bindparam('nodeA'),
                    table['edges'].c.nodeB == bindparam('nodeB'),
                    table['edges'].c.idx == bindparam('idx'),
                    table['edges'].c.branch == bindparam('branch'),
                    table['edges'].c.rev <= bindparam('rev')
                ]
            )
        ),
        'edges_extant': select(
            [
                table['edges'].c.nodeA,
                table['edges'].c.extant
            ]
        ).select_from(
            edges_recent_join(
                [
                    table['edges'].c.graph == bindparam('graph'),
                    table['edges'].c.branch == bindparam('branch'),
                    table['edges'].c.rev <= bindparam('rev')
                ]
            )
        ),
        'nodeAs': select(
            [
                table['edges'].c.nodeA,
                table['edges'].c.extant
            ]
        ).select_from(
            edges_recent_join(
                [
                    table['edges'].c.graph == bindparam('graph'),
                    table['edges'].c.nodeB == bindparam('dest'),
                    table['edges'].c.branch == bindparam('branch'),
                    table['edges'].c.rev <= bindparam('rev')
                ]
            )
        ),
        'nodeBs': select(
            [
                table['edges'].c.nodeB,
                table['edges'].c.extant
            ]
        ).select_from(
            edges_recent_join(
                [
                    table['edges'].c.graph == bindparam('graph'),
                    table['edges'].c.nodeA == bindparam('orig'),
                    table['edges'].c.branch == bindparam('branch'),
                    table['edges'].c.rev <= bindparam('rev')
                ]
            )
        ),
        'multi_edges': select(
            [
                table['edges'].c.idx,
                table['edges'].c.extant
            ]
        ).select_from(
            edges_recent_join(
                [
                    table['edges'].c.graph == bindparam('graph'),
                    table['edges'].c.nodeA == bindparam('orig'),
                    table['edges'].c.nodeB == bindparam('dest'),
                    table['edges'].c.branch == bindparam('branch'),
                    table['edges'].c.rev <= bindparam('rev')
                ]
            )
        ),
        'edges_dump': select([
            table['edges'].c.graph,
            table['edges'].c.nodeA,
            table['edges'].c.nodeB,
            table['edges'].c.idx,
            table['edges'].c.branch,
            table['edges'].c.rev,
            table['edges'].c.extant
        ]),
        'edge_exist_ins': table['edges'].insert().prefix_with('OR REPLACE').values(
            graph=bindparam('graph'),
            nodeA=bindparam('orig'),
            nodeB=bindparam('dest'),
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
                table['edges'].c.nodeA == bindparam('orig'),
                table['edges'].c.nodeB == bindparam('dest'),
                table['edges'].c.idx == bindparam('idx'),
                table['edges'].c.branch == bindparam('branch'),
                table['edges'].c.rev == bindparam('rev')
            )
        ),
        'edge_val_dump': select([
            table['edge_val'].c.graph,
            table['edge_val'].c.nodeA,
            table['edge_val'].c.nodeB,
            table['edge_val'].c.idx,
            table['edge_val'].c.key,
            table['edge_val'].c.branch,
            table['edge_val'].c.rev,
            table['edge_val'].c.value
        ]),
        'edge_val_items': select(
            [
                table['edge_val'].c.key,
                table['edge_val'].c.value
            ]
        ).select_from(
            edge_val_recent_join(
                [
                    table['edge_val'].c.graph == bindparam('graph'),
                    table['edge_val'].c.nodeA == bindparam('orig'),
                    table['edge_val'].c.nodeB == bindparam('dest'),
                    table['edge_val'].c.idx == bindparam('idx'),
                    table['edge_val'].c.branch == bindparam('branch'),
                    table['edge_val'].c.rev <= bindparam('rev')
                ]
            )
        ),
        'edge_val_get': select(
            [
                table['edge_val'].c.value
            ]
        ).select_from(
            edge_val_recent_join(
                [
                    table['edge_val'].c.graph == bindparam('graph'),
                    table['edge_val'].c.nodeA == bindparam('orig'),
                    table['edge_val'].c.nodeB == bindparam('dest'),
                    table['edge_val'].c.idx == bindparam('idx'),
                    table['edge_val'].c.key == bindparam('key'),
                    table['edge_val'].c.branch == bindparam('branch'),
                    table['edge_val'].c.rev <= bindparam('rev')
                ]
            )
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
