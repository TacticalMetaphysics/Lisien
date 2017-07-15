# This file is part of allegedb, an object relational mapper for graphs.
# Copyright (c) Zachary Spector
"""Wrapper to run SQL queries in a lightly abstracted way, such that
code that's more to do with the queries than with the data per se
doesn't pollute the other files so much.

"""
from collections import MutableMapping
from sqlite3 import IntegrityError as sqliteIntegError
try:
    # python 2
    import xjson
except ImportError:
    # python 3
    from allegedb import xjson
import os
xjpath = os.path.dirname(xjson.__file__)
alchemyIntegError = None
try:
    from sqlalchemy.exc import IntegrityError as alchemyIntegError
except ImportError:
    pass


IntegrityError = (
    alchemyIntegError, sqliteIntegError
) if alchemyIntegError is not None else sqliteIntegError


class GlobalKeyValueStore(MutableMapping):
    """A dict-like object that keeps its contents in a table.

    Mostly this is for holding the current branch and revision.

    """
    def __init__(self, qe):
        self.qe = qe

    def __iter__(self):
        if hasattr(self.qe, '_global_cache'):
            yield from self.qe._global_cache
            return
        for (k, v) in self.qe.global_items():
            yield k

    def __len__(self):
        if hasattr(self.qe, '_global_cache'):
            return len(self.qe._global_cache)
        return self.qe.ctglobal()

    def __getitem__(self, k):
        if hasattr(self.qe, '_global_cache'):
            return self.qe._global_cache[k]
        return self.qe.global_get(k)

    def __setitem__(self, k, v):
        self.qe.global_set(k, v)
        if hasattr(self.qe, '_global_cache'):
            self.qe._global_cache[k] = v

    def __delitem__(self, k):
        self.qe.global_del(k)
        if hasattr(self.qe, '_global_cache'):
            del self.qe._global_cache[k]


class QueryEngine(object):
    """Wrapper around either a DBAPI2.0 connection or an
    Alchemist. Provides methods to run queries using either.

    """
    json_path = xjpath

    def __init__(
            self, dbstring, connect_args, alchemy,
            json_dump=None, json_load=None
    ):
        """If ``alchemy`` is True and ``dbstring`` is a legit database URI,
        instantiate an Alchemist and start a transaction with
        it. Otherwise use sqlite3.

        You may pass an already created sqlalchemy :class:`Engine`
        object in place of ``dbstring`` if you wish. I'll still create
        my own transaction though.

        """
        dbstring = dbstring or 'sqlite:///:memory:'

        def alchem_init(dbstring, connect_args):
            from sqlalchemy import create_engine
            from sqlalchemy.engine.base import Engine
            from allegedb.alchemy import Alchemist
            if isinstance(dbstring, Engine):
                self.engine = dbstring
            else:
                self.engine = create_engine(
                    dbstring,
                    connect_args=connect_args
                )
            self.alchemist = Alchemist(self.engine)
            self.transaction = self.alchemist.conn.begin()

        def lite_init(dbstring, connect_args):
            from sqlite3 import connect, Connection
            from json import loads
            self.strings = loads(
                open(self.json_path + '/sqlite.json', 'r').read()
            )
            if isinstance(dbstring, Connection):
                self.connection = dbstring
            else:
                if dbstring.startswith('sqlite:'):
                    slashidx = dbstring.rindex('/')
                    dbstring = dbstring[slashidx+1:]
                self.connection = connect(dbstring)

        if alchemy:
            try:
                alchem_init(dbstring, connect_args)
            except ImportError:
                lite_init(dbstring, connect_args)
        else:
            lite_init(dbstring, connect_args)

        self.globl = GlobalKeyValueStore(self)
        self._branches = {}
        self._nodevals2set = []
        self._edgevals2set = []
        self._graphvals2set = []
        self._nodes2set = []
        self._edges2set = []
        self.json_dump = json_dump or xjson.json_dump
        self.json_load = json_load or xjson.json_load

    def sql(self, stringname, *args, **kwargs):
        """Wrapper for the various prewritten or compiled SQL calls.

        First argument is the name of the query, either a key in
        ``sqlite.json`` or a method name in
        ``allegedb.alchemy.Alchemist``. The rest of the arguments are
        parameters to the query.

        """
        if hasattr(self, 'alchemist'):
            return getattr(self.alchemist, stringname)(*args, **kwargs)
        else:
            s = self.strings[stringname]
            return self.connection.cursor().execute(
                s.format(**kwargs) if kwargs else s, args
            )

    def sqlmany(self, stringname, *args):
        """Wrapper for executing many SQL calls on my connection.

        First arg is the name of a query, either a key in the
        precompiled JSON or a method name in
        ``allegedb.alchemy.Alchemist``. Remaining arguments should be
        tuples of argument sequences to be passed to the query.

        """
        if hasattr(self, 'alchemist'):
            return getattr(self.alchemist.many, stringname)(*args)
        s = self.strings[stringname]
        return self.connection.cursor().executemany(s, args)

    def active_branches(self, branch, rev):
        """Yield a series of ``(branch, rev)`` pairs, starting with the
        ``branch`` and ``rev`` provided; proceeding to the parent
        branch and the revision therein when the provided branch
        began; and recursing through the entire genealogy of branches
        until we reach the branch 'trunk'.

        Though not private, this is unlikely to be useful unless
        you're adding functionality to allegedb.

        """
        yield (branch, rev)
        while branch != 'trunk':
            if branch not in self._branches:
                (b, r) = self.parparrev(branch)
                self._branches[branch] = (b, self.json_load(r))
            (branch, rev) = self._branches[branch]
            yield (branch, rev)

    def have_graph(self, graph):
        """Return whether I have a graph by this name."""
        graph = self.json_dump(graph)
        return bool(self.sql('ctgraph', graph).fetchone()[0])

    def new_graph(self, graph, typ):
        """Declare a new graph by this name of this type."""
        graph = self.json_dump(graph)
        return self.sql('new_graph', graph, typ)

    def del_graph(self, graph):
        """Delete all records to do with the graph"""
        g = self.json_dump(graph)
        self.sql('del_edge_val_graph', g)
        self.sql('del_edge_graph', g)
        self.sql('del_node_val_graph', g)
        self.sql('del_edge_val_graph', g)
        self.sql('del_graph', g)

    def graph_type(self, graph):
        """What type of graph is this?"""
        graph = self.json_dump(graph)
        return self.sql('graph_type', graph).fetchone()[0]

    def have_branch(self, branch):
        """Return whether the branch thus named exists in the database."""
        return bool(self.sql('ctbranch', branch).fetchone()[0])

    def all_branches(self):
        """Return all the branch data in tuples of (branch, parent,
        parent_rev).

        """
        return self.sql('allbranch').fetchall()

    def global_get(self, key):
        """Return the value for the given key in the ``globals`` table."""
        key = self.json_dump(key)
        r = self.sql('global_get', key).fetchone()
        if r is None:
            raise KeyError("Not set")
        return self.json_load(r[0])

    def global_items(self):
        """Iterate over (key, value) pairs in the ``globals`` table."""
        for (k, v) in self.sql('global_items'):
            yield (self.json_load(k), self.json_load(v))

    def global_set(self, key, value):
        """Set ``key`` to ``value`` globally (not at any particular branch or
        revision)

        """
        (key, value) = map(self.json_dump, (key, value))
        try:
            return self.sql('global_ins', key, value)
        except IntegrityError:
            return self.sql('global_upd', value, key)

    def global_del(self, key):
        """Delete the global record for the key."""
        key = self.json_dump(key)
        return self.sql('global_del', key)

    def parrev(self, branch):
        """Return the parent of the branch."""
        return self.sql('parrev', branch).fetchone()[0]

    def parparrev(self, branch):
        """Return the parent and start revision of the branch."""
        return self.sql('parparrev', branch)

    def new_branch(self, branch, parent, parent_rev):
        """Declare that the ``branch`` is descended from ``parent`` at
        ``parent_rev``

        """
        return self.sql('new_branch', branch, parent, parent_rev)

    def graph_val_dump(self):
        """Yield the entire contents of the graph_val table."""
        self._flush_graph_val()
        for (graph, key, branch, rev, value) in self.sql('graph_val_dump'):
            yield (
                self.json_load(graph),
                self.json_load(key),
                branch,
                rev,
                self.json_load(value)
            )

    def graph_val_keys(self, graph, branch, rev):
        """Return an iterable of keys that are set on the graph at the given
        revision.

        """
        self._flush_graph_val()
        graph = self.json_dump(graph)
        seen = set()
        for (b, r) in self.active_branches(branch, rev):
            data = self.sql(
                'graph_val_items', graph, b, r
            )
            for (k, v) in data:
                if k not in seen:
                    yield self.json_load(k)
                seen.add(k)

    def graph_val_get(self, graph, key, branch, rev):
        """Return the value of a key that a graph has, as of the given
        revision.

        """
        self._flush_graph_val()
        (graph, key) = map(self.json_dump, (graph, key))
        for (b, r) in self.active_branches(branch, rev):
            for row in self.sql(
                'graph_val_get',
                graph,
                key,
                branch,
                rev
            ):
                if row is None:
                    raise KeyError("Key not set")
                return self.json_load(row[0])
        raise KeyError("Key never set")

    def _flush_graph_val(self):
        """Send all new and changed graph values to the database."""
        def convert_arg(arg):
            if isinstance(arg, dict):
                return (
                    self.json_dump(arg['graph']),
                    self.json_dump(arg['key']),
                    arg['branch'], arg['rev'],
                    self.json_dump(arg['value'])
                )
            elif isinstance(arg, tuple) or isinstance(arg, list):
                graph, key, branch, rev, value = arg
                return (
                    self.json_dump(graph),
                    self.json_dump(key),
                    branch, rev,
                    self.json_dump(value)
                )
            else:
                raise TypeError(
                    "Expected dict, list, or tuple, got {}".format(type(arg))
                )

        if not self._graphvals2set:
            return
        self.sqlmany('graph_val_ins', *map(convert_arg, self._graphvals2set))
        self._graphvals2set = []

    def graph_val_set(self, graph, key, branch, rev, value):
        """Set a key to a value on a graph at a particular revision."""
        self._graphvals2set.append((graph, key, branch, rev, value))

    def graph_val_del(self, graph, key, branch, rev):
        """Indicate that the key is unset."""
        self.graph_val_set(graph, key, branch, rev, None)

    def graphs_types(self):
        for (graph, typ) in self.sql('graphs_types'):
            yield (self.json_load(graph), typ)

    def nodes_extant(self, graph, branch, rev):
        """Return an iterable of nodes that exist in this graph at this
        revision.

        """
        self._flush_nodes()
        graph = self.json_dump(graph)
        seen = set()
        for (b, r) in self.active_branches(branch, rev):
            data = self.sql(
                'nodes_extant',
                graph,
                branch,
                rev
            )
            for (n,) in data:
                if n is not None and n not in seen:
                    yield self.json_load(n)
                seen.add(n)

    def node_exists(self, graph, node, branch, rev):
        """Return whether there's a node by this name in this graph at this
        revision.

        """
        self._flush_nodes()
        (graph, node) = map(self.json_dump, (graph, node))
        for (b, r) in self.active_branches(branch, rev):
            for x in self.sql(
                'node_exists', graph, node, b, r
            ):
                return bool(x[0])
        return False

    def _flush_nodes(self):
        def convert_arg(arg):
            if isinstance(arg, dict):
                return (
                    self.json_dump(arg['graph']),
                    self.json_dump(arg['node']),
                    arg['branch'], arg['rev'], arg['extant']
                )
            elif isinstance(arg, tuple) or isinstance(arg, list):
                graph, node, branch, rev, extant = arg
                return (
                    self.json_dump(graph),
                    self.json_dump(node),
                    branch, rev, extant
                )
            else:
                raise TypeError(
                    "Expected dict, list, or tuple, got {}".format(type(arg))
                )

        if not self._nodes2set:
            return
        self.sqlmany('exist_node_ins', *map(convert_arg, self._nodes2set))
        self._nodes2set = []

    def exist_node(self, graph, node, branch, rev, extant):
        """Declare that the node exists or doesn't.

        Inserts a new record or updates an old one, as needed.

        """
        self._nodes2set.append((graph, node, branch, rev, extant))

    def nodes_dump(self):
        """Dump the entire contents of the nodes table."""
        self._flush_nodes()
        for (graph, node, branch, tick, extant) in self.sql('nodes_dump'):
            yield (
                self.json_load(graph),
                self.json_load(node),
                branch,
                tick,
                bool(extant)
            )

    def node_val_dump(self):
        """Yield the entire contents of the node_val table."""
        self._flush_node_val()
        for (
                graph, node, key, branch, rev, value
        ) in self.sql('node_val_dump'):
            yield (
                self.json_load(graph),
                self.json_load(node),
                self.json_load(key),
                branch,
                rev,
                self.json_load(value)
            )

    def node_val_keys(self, graph, node, branch, rev):
        """Return an iterable of keys that are set on the node at the given
        revision.

        """
        self._flush_node_val()
        (graph, node) = map(self.json_dump, (graph, node))
        seen = set()
        for (b, r) in self.active_branches(branch, rev):
            for (k, v) in self.sql(
                    'node_val_items',
                    graph,
                    node,
                    branch,
                    rev
            ):
                if k not in seen and v is not None:
                    yield self.json_load(k)
                seen.add(k)

    def node_vals_ever(self, graph, node):
        """Iterate over all values set on a node through time."""
        self._flush_node_val()
        (graph, node) = map(self.json_dump, (graph, node))
        for (key, branch, tick, value) in self.sql(
                'node_vals_ever', graph, node
        ):
            yield (self.json_load(key), branch, tick, self.json_load(value))

    def node_val_get(self, graph, node, key, branch, rev):
        """Get the value of the node's key as it was at the given revision."""
        self._flush_node_val()
        (graph, node, key) = map(self.json_dump, (graph, node, key))
        for (b, r) in self.active_branches(branch, rev):
            for row in self.sql(
                'node_val_get',
                graph,
                node,
                key,
                branch,
                rev
            ):
                if row[0] is None:
                    raise KeyError("Key not set")
                return self.json_load(row[0])
        raise KeyError("Key {} never set".format(key))

    def _flush_node_val(self):
        def convert_arg(arg):
            if isinstance(arg, dict):
                return (
                    self.json_dump(arg['graph']),
                    self.json_dump(arg['node']),
                    self.json_dump(arg['key']),
                    arg['branch'],
                    arg['rev'],
                    self.json_dump(arg['value'])
                )
            elif isinstance(arg, tuple) or isinstance(arg, list):
                graph, node, key, branch, rev, value = arg
                return (
                    self.json_dump(graph),
                    self.json_dump(node),
                    self.json_dump(key),
                    branch,
                    rev,
                    self.json_dump(value)
                )
            else:
                raise TypeError(
                    "Need dict, list, or tuple, not {}".format(type(arg))
                )

        if not self._nodevals2set:
            return
        self.sqlmany('node_val_ins', *map(convert_arg, self._nodevals2set))
        self._nodevals2set = []

    def node_val_set(self, graph, node, key, branch, rev, value):
        """Set a key-value pair on a node at a specific branch and revision"""
        self._nodevals2set.append((graph, node, key, branch, rev, value))

    def node_val_del(self, graph, node, key, branch, rev):
        """Delete a key from a node at a specific branch and revision"""
        self.node_val_set(graph, node, key, branch, rev, None)

    def edges_dump(self):
        """Dump the entire contents of the edges table."""
        self._flush_edges()
        for (
                graph, orig, dest, idx, branch, rev, extant
        ) in self.sql('edges_dump'):
            yield (
                self.json_load(graph),
                self.json_load(orig),
                self.json_load(dest),
                idx,
                branch,
                rev,
                bool(extant)
            )

    def edges_extant(self, graph, branch, rev):
        """Return an iterable of nodes that have edges from them, in this
        graph, at this revision.

        """
        self._flush_edges()
        graph = self.json_dump(graph)
        seen = set()
        for (b, r) in self.active_branches(branch, rev):
            for row in self.sql(
                'edges_extant', graph, branch, rev
            ):
                if row[0] not in seen and row[1]:
                    yield self.json_load(row[0])
                seen.add(row[0])

    def edge_exists(self, graph, orig, dest, idx, branch, rev):
        """Return whether the edge exists now, or None if there's no data
        about it in this branch.

        """
        self._flush_edges()
        (graph, orig, dest) = map(self.json_dump, (graph, orig, dest))
        for (b, r) in self.active_branches(branch, rev):
            for row in self.sql(
                'edge_exists',
                graph,
                orig,
                dest,
                idx,
                b,
                r
            ):
                return bool(row[1])
        return False

    def origs(self, graph, dest, branch, rev):
        """Return an iterable of nodes that have an edge leading to the given
        node.

        """
        self._flush_nodes()
        self._flush_edges()
        (graph, dest) = map(self.json_dump, (graph, dest))
        seen = set()
        for (b, r) in self.active_branches(branch, rev):
            for row in self.sql(
                'origs',
                graph,
                dest,
                b,
                r
            ):
                if row[0] not in seen and row[1]:
                    yield self.json_load(row[0])
                seen.add(row[0])

    def dests(self, graph, orig, branch, rev):
        """Return an iterable of nodes you can get to from the given one."""
        self._flush_nodes()
        self._flush_edges()
        (graph, orig) = map(self.json_dump, (graph, orig))
        seen = set()
        for (b, r) in self.active_branches(branch, rev):
            for row in self.sql(
                'dests', graph, orig, b, r
            ):
                if row[0] not in seen and row[1]:
                    yield self.json_load(row[0])
                seen.add(row[0])

    def multi_edges(self, graph, orig, dest, branch, rev):
        """Return an iterable of edge indices for all edges between these two
        nodes.

        """
        self._flush_nodes()
        self._flush_edges()
        (graph, orig, dest) = map(self.json_dump, (graph, orig, dest))
        seen = set()
        for (b, r) in self.active_branches(branch, rev):
            for row in self.sql(
                'multi_edges', graph, orig, dest, branch, rev
            ):
                if row[0] not in seen and row[1]:
                    yield row[0]
                seen.add(row[0])

    def _flush_edges(self):
        def convert_arg(arg):
            if isinstance(arg, dict):
                return (
                    self.json_dump(arg['graph']),
                    self.json_dump(arg['orig']),
                    self.json_dump(arg['dest']),
                    arg['idx'], arg['branch'], arg['rev'], arg['extant']
                )
            elif isinstance(arg, list) or isinstance(arg, tuple):
                graph, orig, dest, idx, branch, rev, extant = arg
                return (
                    self.json_dump(graph),
                    self.json_dump(orig),
                    self.json_dump(dest),
                    idx, branch, rev, extant
                )
            else:
                raise TypeError(
                    "Expected dict, list, or tuple, got {}".format(type(arg))
                )

        if not self._edges2set:
            return
        self.sqlmany('edge_exist_ins', *map(convert_arg, self._edges2set))
        self._edges2set = []

    def exist_edge(self, graph, orig, dest, idx, branch, rev, extant):
        """Declare whether or not this edge exists."""
        self._edges2set.append((graph, orig, dest, idx, branch, rev, extant))

    def edge_val_dump(self):
        """Yield the entire contents of the edge_val table."""
        self._flush_edge_val()
        for (
                graph, orig, dest, idx, key, branch, rev, value
        ) in self.sql('edge_val_dump'):
            yield (
                self.json_load(graph),
                self.json_load(orig),
                self.json_load(dest),
                idx,
                self.json_load(key),
                branch,
                rev,
                self.json_load(value)
            )

    def edge_val_keys(self, graph, orig, dest, idx, branch, rev):
        """Return an iterable of keys this edge has."""
        self._flush_edge_val()
        (graph, orig, dest) = map(self.json_dump, (graph, orig, dest))
        seen = set()
        for (b, r) in self.active_branches(branch, rev):
            for row in self.sql(
                'edge_val_items', graph, orig, dest, idx, b, r
            ):
                if row[0] not in seen:
                    yield self.json_load(row[0])
                seen.add(row[0])

    def edge_val_get(self, graph, orig, dest, idx, key, branch, rev):
        """Return the value of this key of this edge."""
        self._flush_edge_val()
        (graph, orig, dest, key) = map(
            self.json_dump, (graph, orig, dest, key)
        )
        for (b, r) in self.active_branches(branch, rev):
            for row in self.sql(
                'edge_val_get', graph, orig, dest, idx, key, b, r
            ):
                if row[0] is None:
                    raise KeyError("Key not set")
                return self.json_load(row[0])
        raise KeyError("Key never set")

    def _flush_edge_val(self):
        def convert_arg(arg):
            if isinstance(arg, dict):
                return (
                    self.json_dump(arg['graph']),
                    self.json_dump(arg['orig']),
                    self.json_dump(arg['dest']),
                    arg['idx'],
                    self.json_dump(arg['key']),
                    arg['branch'], arg['rev'],
                    self.json_dump(arg['value'])
                )
            elif isinstance(arg, tuple) or isinstance(arg, list):
                graph, orig, dest, idx, key, branch, rev, value = arg
                return (
                    self.json_dump(graph),
                    self.json_dump(orig),
                    self.json_dump(dest),
                    idx,
                    self.json_dump(key),
                    branch, rev,
                    self.json_dump(value)
                )
            else:
                raise TypeError(
                    "Expected dict, list, or tuple, got {}".format(type(arg))
                )

        if not self._edgevals2set:
            return
        self.sqlmany('edge_val_ins', *map(convert_arg, self._edgevals2set))
        self._edgevals2set = []

    def edge_val_set(self, graph, orig, dest, idx, key, branch, rev, value):
        """Set this key of this edge to this value."""
        self._edgevals2set.append(
            (graph, orig, dest, idx, key, branch, rev, value)
        )

    def edge_val_del(self, graph, orig, dest, idx, key, branch, rev):
        """Declare that the key no longer applies to this edge, as of this
        branch and revision.

        """
        self.edge_val_set(graph, orig, dest, idx, key, branch, rev, None)

    def initdb(self):
        """Create tables and indices as needed."""
        if hasattr(self, 'alchemist'):
            self.alchemist.meta.create_all(self.engine)
            if 'branch' not in self.globl:
                self.globl['branch'] = 'trunk'
            if 'rev' not in self.globl:
                self.globl['rev'] = 0
            return
        from sqlite3 import OperationalError
        cursor = self.connection.cursor()
        try:
            cursor.execute('SELECT * FROM global;')
        except OperationalError:
            cursor.execute(self.strings['create_global'])
        if 'branch' not in self.globl:
            self.globl['branch'] = 'trunk'
        if 'rev' not in self.globl:
            self.globl['rev'] = 0
        try:
            cursor.execute('SELECT * FROM branches;')
        except OperationalError:
            cursor.execute(self.strings['create_branches'])
        try:
            cursor.execute('SELECT * FROM graphs;')
        except OperationalError:
            cursor.execute(self.strings['create_graphs'])
        try:
            cursor.execute('SELECT * FROM graph_val;')
        except OperationalError:
            cursor.execute(self.strings['create_graph_val'])
            cursor.execute(self.strings['index_graph_val_time'])
        try:
            cursor.execute('SELECT * FROM nodes;')
        except OperationalError:
            cursor.execute(self.strings['create_nodes'])
            cursor.execute(self.strings['index_nodes_time'])

        try:
            cursor.execute('SELECT * FROM node_val;')
        except OperationalError:
            cursor.execute(self.strings['create_node_val'])
            cursor.execute(self.strings['index_node_val_time'])
        try:
            cursor.execute('SELECT * FROM edges;')
        except OperationalError:
            cursor.execute(self.strings['create_edges'])
            cursor.execute(self.strings['index_edges_time'])
        try:
            cursor.execute('SELECT * FROM edge_val;')
        except OperationalError:
            cursor.execute(self.strings['create_edge_val'])
            cursor.execute(self.strings['index_edge_val_time'])

    def flush(self):
        """Put all pending changes into the SQL transaction."""
        self._flush_nodes()
        self._flush_edges()
        self._flush_graph_val()
        self._flush_node_val()
        self._flush_edge_val()

    def commit(self):
        """Commit the transaction"""
        self.flush()
        if hasattr(self, 'transaction'):
            self.transaction.commit()
        else:
            self.connection.commit()

    def close(self):
        """Commit the transaction, then close the connection"""
        self.commit()
        if hasattr(self, 'connection'):
            self.connection.close()
