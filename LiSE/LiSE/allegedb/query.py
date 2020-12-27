# This file is part of allegedb, an object relational mapper for graphs.
# Copyright (c) Zachary Spector. public@zacharyspector.com
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
"""Wrapper to run SQL queries in a lightly abstracted way, such that
code that's more to do with the queries than with the data per se
doesn't pollute the other files so much.

"""
import os
from collections.abc import MutableMapping
from sqlite3 import IntegrityError as sqliteIntegError
try:
    # python 2
    import wrap
except ImportError:
    # python 3
    from LiSE.allegedb import wrap
wrappath = os.path.dirname(wrap.__file__)
alchemyIntegError = None
try:
    from sqlalchemy.exc import IntegrityError as alchemyIntegError
except ImportError:
    pass
from time import monotonic

IntegrityError = (
    alchemyIntegError, sqliteIntegError
) if alchemyIntegError is not None else sqliteIntegError


class TimeError(ValueError):
    """Exception class for problems with the time model"""


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
    path = os.path.dirname(__file__)
    flush_edges_t = 0
    def __init__(
            self, dbstring, connect_args, alchemy,
            pack=None, unpack=None
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
            from LiSE.allegedb.alchemy import Alchemist
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
            from json import load
            with open(os.path.join(self.path, 'sqlite.json')) as strf:
                self.strings = load(strf)
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
        self._btts = set()
        if unpack is None:
            from ast import literal_eval as unpack
        self.pack = pack or repr
        self.unpack = unpack
        self._exist_edge_stuff = (self._btts, self._edges2set)
        self._edge_val_set_stuff = (self._btts, self._edgevals2set)

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

    def have_graph(self, graph):
        """Return whether I have a graph by this name."""
        graph = self.pack(graph)
        return bool(self.sql('graphs_named', graph).fetchone()[0])

    def new_graph(self, graph, typ):
        """Declare a new graph by this name of this type."""
        graph = self.pack(graph)
        return self.sql('graphs_insert', graph, typ)

    def keyframes_insert(self, graph, branch, turn, tick, nodes, edges, graph_val):
        graph, nodes, edges, graph_val = map(self.pack, (graph, nodes, edges, graph_val))
        return self.sql('keyframes_insert', graph, branch, turn, tick, nodes, edges, graph_val)

    def keyframes_dump(self):
        unpack = self.unpack
        for (graph, branch, turn, tick, nodes, edges, graph_val) in self.sql('keyframes_dump'):
            yield unpack(graph), branch, turn, tick, unpack(nodes), unpack(edges), unpack(graph_val)

    def del_graph(self, graph):
        """Delete all records to do with the graph"""
        g = self.pack(graph)
        self.sql('del_edge_val_graph', g)
        self.sql('del_node_val_graph', g)
        self.sql('del_edge_val_graph', g)
        self.sql('del_edges_graph', g)
        self.sql('del_nodes_graph', g)
        self.sql('del_graph', g)

    def graph_type(self, graph):
        """What type of graph is this?"""
        graph = self.pack(graph)
        return self.sql('graph_type', graph).fetchone()[0]

    def have_branch(self, branch):
        """Return whether the branch thus named exists in the database."""
        return bool(self.sql('ctbranch', branch).fetchone()[0])

    def all_branches(self):
        """Return all the branch data in tuples of (branch, parent,
        parent_turn).

        """
        return self.sql('branches_dump').fetchall()

    def global_get(self, key):
        """Return the value for the given key in the ``globals`` table."""
        key = self.pack(key)
        r = self.sql('global_get', key).fetchone()
        if r is None:
            raise KeyError("Not set")
        return self.unpack(r[0])

    def global_items(self):
        """Iterate over (key, value) pairs in the ``globals`` table."""
        unpack = self.unpack
        for (k, v) in self.sql('global_dump'):
            yield (unpack(k), unpack(v))

    def get_branch(self):
        v = self.sql('global_get', self.pack('branch')).fetchone()
        if v is None:
            return 'trunk'
        return self.unpack(v[0])

    def get_turn(self):
        v = self.sql('global_get', self.pack('turn')).fetchone()
        if v is None:
            return 0
        return self.unpack(v[0])

    def get_tick(self):
        v = self.sql('global_get', self.pack('tick')).fetchone()
        if v is None:
            return 0
        return self.unpack(v[0])

    def global_set(self, key, value):
        """Set ``key`` to ``value`` globally (not at any particular branch or
        revision)

        """
        (key, value) = map(self.pack, (key, value))
        try:
            return self.sql('global_insert', key, value)
        except IntegrityError:
            return self.sql('global_update', value, key)

    def global_del(self, key):
        """Delete the global record for the key."""
        key = self.pack(key)
        return self.sql('global_del', key)

    def new_branch(self, branch, parent, parent_turn, parent_tick):
        """Declare that the ``branch`` is descended from ``parent`` at
        ``parent_turn``, ``parent_tick``

        """
        return self.sql('branches_insert', branch, parent, parent_turn, parent_tick, parent_turn, parent_tick)

    def update_branch(self, branch, parent, parent_turn, parent_tick, end_turn, end_tick):
        return self.sql('update_branches', parent, parent_turn, parent_tick, end_turn, end_tick, branch)

    def set_branch(self, branch, parent, parent_turn, parent_tick, end_turn, end_tick):
        try:
            self.sql('branches_insert', branch, parent, parent_turn, parent_tick, end_turn, end_tick)
        except IntegrityError:
            self.update_branch(branch, parent, parent_turn, parent_tick, end_turn, end_tick)

    def new_turn(self, branch, turn, end_tick=0, plan_end_tick=0):
        return self.sql('turns_insert', branch, turn, end_tick, plan_end_tick)

    def update_turn(self, branch, turn, end_tick, plan_end_tick):
        return self.sql('update_turns', end_tick, plan_end_tick, branch, turn)

    def set_turn(self, branch, turn, end_tick, plan_end_tick):
        try:
            return self.sql('turns_insert', branch, turn, end_tick, plan_end_tick)
        except IntegrityError:
            return self.sql('update_turns', end_tick, plan_end_tick, branch, turn)

    def turns_dump(self):
        return self.sql('turns_dump')

    def graph_val_dump(self):
        """Yield the entire contents of the graph_val table."""
        self._flush_graph_val()
        unpack = self.unpack
        for (graph, key, branch, turn, tick, value) in self.sql('graph_val_dump'):
            yield (
                unpack(graph),
                unpack(key),
                branch,
                turn,
                tick,
                unpack(value)
            )

    def _flush_graph_val(self):
        """Send all new and changed graph values to the database."""
        if not self._graphvals2set:
            return
        delafter = {}
        for graph, key, branch, turn, tick, value in self._graphvals2set:
            if (graph, key, branch) in delafter:
                delafter[graph, key, branch] = min((
                    (turn, tick),
                    delafter[graph, key, branch]
                ))
            else:
                delafter[graph, key, branch] = (turn, tick)
        self.sqlmany(
            'del_graph_val_after',
            *((graph, key, branch, turn, turn, tick)
              for ((graph, key, branch), (turn, tick)) in delafter.items())
        )
        self.sqlmany('graph_val_insert', *self._graphvals2set)
        self._graphvals2set = []

    def graph_val_set(self, graph, key, branch, turn, tick, value):
        if (branch, turn, tick) in self._btts:
            raise TimeError
        self._btts.add((branch, turn, tick))
        graph, key, value = map(self.pack, (graph, key, value))
        self._graphvals2set.append((graph, key, branch, turn, tick, value))

    def graph_val_del_time(self, branch, turn, tick):
        self._flush_graph_val()
        self.sql('graph_val_del_time', branch, turn, tick)
        self._btts.discard((branch, turn, tick))

    def graphs_types(self):
        for (graph, typ) in self.sql('graphs_types'):
            yield (self.unpack(graph), typ)

    def _flush_nodes(self):
        if not self._nodes2set:
            return
        self.sqlmany('nodes_insert', *self._nodes2set)
        self._nodes2set = []

    def exist_node(self, graph, node, branch, turn, tick, extant):
        """Declare that the node exists or doesn't.

        Inserts a new record or updates an old one, as needed.

        """
        if (branch, turn, tick) in self._btts:
            raise TimeError
        self._btts.add((branch, turn, tick))
        self._nodes2set.append((self.pack(graph), self.pack(node), branch, turn, tick, extant))

    def nodes_del_time(self, branch, turn, tick):
        self._flush_nodes()
        self.sql('nodes_del_time', branch, turn, tick)
        self._btts.discard((branch, turn, tick))

    def nodes_dump(self):
        """Dump the entire contents of the nodes table."""
        self._flush_nodes()
        unpack = self.unpack
        for (graph, node, branch, turn,tick, extant) in self.sql('nodes_dump'):
            yield (
                unpack(graph),
                unpack(node),
                branch,
                turn,
                tick,
                bool(extant)
            )

    def node_val_dump(self):
        """Yield the entire contents of the node_val table."""
        self._flush_node_val()
        unpack = self.unpack
        for (
                graph, node, key, branch, turn, tick, value
        ) in self.sql('node_val_dump'):
            yield (
                unpack(graph),
                unpack(node),
                unpack(key),
                branch,
                turn,
                tick,
                unpack(value)
            )

    def _flush_node_val(self):
        if not self._nodevals2set:
            return
        self.sqlmany('node_val_insert', *self._nodevals2set)
        self._nodevals2set = []

    def node_val_set(self, graph, node, key, branch, turn, tick, value):
        """Set a key-value pair on a node at a specific branch and revision"""
        if (branch, turn, tick) in self._btts:
            raise TimeError
        self._btts.add((branch, turn, tick))
        graph, node, key, value = map(self.pack, (graph, node, key, value))
        self._nodevals2set.append((graph, node, key, branch, turn, tick, value))

    def node_val_del_time(self, branch, turn, tick):
        self._flush_node_val()
        self.sql('node_val_del_time', branch, turn, tick)
        self._btts.discard((branch, turn, tick))

    def edges_dump(self):
        """Dump the entire contents of the edges table."""
        self._flush_edges()
        unpack = self.unpack
        for (
                graph, orig, dest, idx, branch, turn, tick, extant
        ) in self.sql('edges_dump'):
            yield (
                unpack(graph),
                unpack(orig),
                unpack(dest),
                idx,
                branch,
                turn,
                tick,
                bool(extant)
            )

    def _pack_edge2set(self, tup):
        graph, orig, dest, idx, branch, turn, tick, extant = tup
        pack = self.pack
        return pack(graph), pack(orig), pack(dest), idx, branch, turn, tick, extant

    def _flush_edges(self):
        start = monotonic()
        if not self._edges2set:
            return
        self.sqlmany('edges_insert', *map(self._pack_edge2set, self._edges2set))
        self._edges2set = []
        QueryEngine.flush_edges_t += monotonic() - start

    def exist_edge(self, graph, orig, dest, idx, branch, turn, tick, extant):
        """Declare whether or not this edge exists."""
        btts, edges2set = self._exist_edge_stuff
        if (branch, turn, tick) in btts:
            raise TimeError
        btts.add((branch, turn, tick))
        edges2set.append((graph, orig, dest, idx, branch, turn, tick, extant))

    def edges_del_time(self, branch, turn, tick):
        self._flush_edges()
        self.sql('edges_del_time', branch, turn, tick)
        self._btts.discard((branch, turn, tick))

    def edge_val_dump(self):
        """Yield the entire contents of the edge_val table."""
        self._flush_edge_val()
        unpack = self.unpack
        for (
                graph, orig, dest, idx, key, branch, turn, tick, value
        ) in self.sql('edge_val_dump'):
            yield (
                unpack(graph),
                unpack(orig),
                unpack(dest),
                idx,
                unpack(key),
                branch,
                turn,
                tick,
                unpack(value)
            )

    def _pack_edgeval2set(self, tup):
        graph, orig, dest, idx, key, branch, turn, tick, value = tup
        pack = self.pack
        return pack(graph), pack(orig), pack(dest), idx, pack(key), branch, turn, tick, pack(value)

    def _flush_edge_val(self):
        if not self._edgevals2set:
            return
        self.sqlmany('edge_val_insert', *map(self._pack_edgeval2set, self._edgevals2set))
        self._edgevals2set = []

    def edge_val_set(self, graph, orig, dest, idx, key, branch, turn, tick, value):
        """Set this key of this edge to this value."""
        if (branch, turn, tick) in self._btts:
            raise TimeError
        self._btts.add((branch, turn, tick))
        self._edgevals2set.append(
            (graph, orig, dest, idx, key, branch, turn, tick, value)
        )

    def edge_val_del_time(self, branch, turn, tick):
        self._flush_edge_val()
        self.sql('edge_val_del_time', branch, turn, tick)
        self._btts.discard((branch, turn, tick))

    def plans_dump(self):
        return self.sql('plans_dump')

    def plans_insert(self, plan_id, branch, turn, tick):
        return self.sql('plans_insert', plan_id, branch, turn, tick)

    def plans_insert_many(self, many):
        return self.sqlmany('plans_insert', *many)

    def plan_ticks_insert(self, plan_id, turn, tick):
        return self.sql('plan_ticks_insert', plan_id, turn, tick)

    def plan_ticks_insert_many(self, many):
        return self.sqlmany('plan_ticks_insert', *many)

    def plan_ticks_dump(self):
        return self.sql('plan_ticks_dump')

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
        if 'turn' not in self.globl:
            self.globl['turn'] = 0
        if 'tick' not in self.globl:
            self.globl['tick'] = 0
        strings = self.strings
        for table in (
            'branches',
            'turns',
            'graphs',
            'graph_val',
            'nodes',
            'node_val',
            'edges',
            'edge_val',
            'plans',
            'plan_ticks',
            'keyframes'
        ):
            try:
                cursor.execute('SELECT * FROM ' + table + ';')
            except OperationalError:
                cursor.execute(strings['create_' + table])

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
        if hasattr(self, 'transaction') and self.transaction.is_active:
            self.transaction.commit()
        elif hasattr(self, 'connection'):
            self.connection.commit()

    def close(self):
        """Commit the transaction, then close the connection"""
        self.commit()
        if hasattr(self, 'connection'):
            self.connection.close()
