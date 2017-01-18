# This file is part of allegedb, an object relational mapper for versioned graphs.
# Copyright (C) Zachary Spector.
from collections import defaultdict, deque
from .graph import (
    Graph,
    DiGraph,
    MultiGraph,
    MultiDiGraph,
    Node,
    Edge
)
from .query import QueryEngine
from .cache import Cache, NodesCache, EdgesCache


class GraphNameError(KeyError):
    pass


class ORM(object):
    """Instantiate this with the same string argument you'd use for a
    SQLAlchemy ``create_engine`` call. This will be your interface to
    allegedb.

    """
    node_cls = _make_node = Node
    edge_cls = _make_edge = Edge

    def __init__(
            self,
            dbstring,
            alchemy=True,
            connect_args={},
            query_engine_class=QueryEngine,
            json_dump=None,
            json_load=None,
            caching=True
    ):
        """Make a SQLAlchemy engine if possible, else a sqlite3 connection. In
        either case, begin a transaction.

        """
        self.query = query_engine_class(
            dbstring, connect_args, alchemy, json_dump, json_load
        )
        self._obranch = None
        self._orev = None
        self.query.initdb()
        # I will be recursing a lot so just cache all the branch info
        if caching:
            self.caching = True
            self._global_cache = self.query._global_cache = {}
            self._node_objs = {}
            self._edge_objs = {}
            for k, v in self.query.global_items():
                if k == 'branch':
                    self._obranch = v
                elif k == 'rev':
                    self._orev = v
                else:
                    self._global_cache[k] = v
            self._childbranch = defaultdict(set)
            self._parentbranch_rev = {}
            for (branch, parent, parent_rev) in self.query.all_branches():
                if branch != 'master':
                    self._parentbranch_rev[branch] = (parent, parent_rev)
                self._childbranch[parent].add(branch)
            self._graph_val_cache = Cache(self)
            self._nodes_cache = NodesCache(self)
            self._edges_cache = EdgesCache(self)
            self._node_val_cache = Cache(self)
            self._edge_val_cache = Cache(self)
            self._obranch = self.branch
            self._orev = self.rev
            self._active_branches_cache = []
            self.query.active_branches = self._active_branches
            # Make sure to load in the correct order, so that
            # caches for child branches get built after their parents
            graphval = defaultdict(list)
            nodeval = defaultdict(list)
            edgeval = defaultdict(list)
            nodes = defaultdict(list)
            edges = defaultdict(list)
            for (graph, key, branch, rev, val) in self.query.graph_val_dump():
                graphval[branch].append((graph, key, branch, rev, val))
            for (graph, node, branch, rev, ex) in self.query.nodes_dump():
                nodes[branch].append((graph, node, branch, rev, ex))
            for (graph, u, v, i, branch, rev, ex) in self.query.edges_dump():
                edges[branch].append((graph, u, v, i, branch, rev, ex))
            for (
                    graph, node, key, branch, rev, val
            ) in self.query.node_val_dump():
                nodeval[branch].append((graph, node, key, branch, rev, val))
            for (
                    graph, u, v, i, key, branch, rev, val
            ) in self.query.edge_val_dump():
                edgeval[branch].append((graph, u, v, i, key, branch, rev, val))
            branch2do = deque(['master'])
            while branch2do:
                branch = branch2do.pop()
                for row in graphval[branch]:
                    self._graph_val_cache.store(*row)
                for row in nodes[branch]:
                    self._nodes_cache.store(*row)
                for row in edges[branch]:
                    self._edges_cache.store(*row)
                for row in nodeval[branch]:
                    self._node_val_cache.store(*row)
                for row in edgeval[branch]:
                    self._edge_val_cache.store(*row)
                if branch in self._childbranch:
                    branch2do.extend(self._childbranch[branch])
            if not hasattr(self, 'graph'):
                self.graph = {}
            for (graph, typ) in self.query.graphs_types():
                self.graph[graph] = {
                    'Graph': Graph,
                    'DiGraph': DiGraph,
                    'MultiGraph': MultiGraph,
                    'MultiDiGraph': MultiDiGraph
                }[typ](self, graph)

    def __enter__(self):
        """Enable the use of the ``with`` keyword"""
        return self

    def __exit__(self, *args):
        """Alias for ``close``"""
        self.close()

    def _havebranch(self, b):
        """Private use. Checks that the branch is known about."""
        if self.caching and b in self._parentbranch_rev:
            return True
        return self.query.have_branch(b)

    def is_parent_of(self, parent, child):
        """Return whether ``child`` is a branch descended from ``parent`` at
        any remove.

        """
        if parent == 'master':
            return True
        if child == 'master':
            return False
        if child not in self._parentbranch_rev:
            raise ValueError(
                "The branch {} seems not to have ever been created".format(
                    child
                )
            )
        if self._parentbranch_rev[child][0] == parent:
            return True
        return self.is_parent_of(parent, self._parentbranch_rev[child][0])

    @property
    def branch(self):
        """Return the global value ``branch``, or ``self._obranch`` if it's
        set

        """
        if self._obranch is not None:
            return self._obranch
        return self.db.globl['branch']

    @branch.setter
    def branch(self, v):
        """Set the global value ``branch`` and note that the branch's (parent,
        parent_rev) are the (branch, tick) set previously

        """
        curbranch = self.branch
        currev = self.rev
        if not self._havebranch(v):
            # assumes the present revision in the parent branch has
            # been finalized.
            self.query.new_branch(v, curbranch, currev)
        # make sure I'll end up within the revision range of the
        # destination branch
        if v != 'master':
            if self.caching:
                if v not in self._parentbranch_rev:
                    self._parentbranch_rev[v] = (curbranch, currev)
                parrev = self._parentbranch_rev[v][1]
            else:
                parrev = self.query.parrev(v)
            if currev < parrev:
                raise ValueError(
                    "Tried to jump to branch {br}, "
                    "which starts at revision {rv}. "
                    "Go to rev {rv} or later to use this branch.".format(
                        br=v,
                        rv=parrev
                    )
                )
        if self.caching:
            self._obranch = v
        else:
            self.query.globl['branch'] = v

    @property
    def rev(self):
        """Return the global value ``rev``, or ``self._orev`` if that's set"""
        if self._orev is not None:
            return self._orev
        return self.query.globl['rev']

    @rev.setter
    def rev(self, v):
        """Set the global value ``rev``, first checking that it's not before
        the start of this branch. If it is, also go to the parent
        branch.

        """
        # first make sure the cursor is not before the start of this branch
        branch = self.branch
        if branch != 'master':
            if self.caching:
                (parent, parent_rev) = self._parentbranch_rev[branch]
            else:
                (parent, parent_rev) = self.query.parparrev(branch)
            if v < int(parent_rev):
                raise ValueError(
                    "The revision number {revn} "
                    "occurs before the start of "
                    "the branch {brnch}".format(revn=v, brnch=branch)
                )
        if self.caching:
            self._orev = v
        else:
            self.query.globl['rev'] = v

    def commit(self):
        """Alias of ``self.query.commit``"""
        if self.caching:
            self.query.globl['branch'] = self._obranch
            self.query.globl['rev'] = self._orev
        self.query.commit()

    def close(self):
        """Alias of ``self.query.close``"""
        if self.caching:
            self.query.globl['branch'] = self._obranch
            self.query.globl['rev'] = self._orev
        self.query.close()

    def initdb(self):
        """Alias of ``self.query.initdb``"""
        self.query.initdb()

    def _init_graph(self, name, type_s='Graph'):
        if self.query.have_graph(name):
            raise GraphNameError("Already have a graph by that name")
        self.query.new_graph(name, type_s)

    def new_graph(self, name, data=None, **attr):
        """Return a new instance of type Graph, initialized with the given
        data if provided.

        """
        self._init_graph(name, 'Graph')
        g = Graph(self, name, data, **attr)
        if self.caching:
            self.graph[name] = g
        return g

    def new_digraph(self, name, data=None, **attr):
        """Return a new instance of type DiGraph, initialized with the given
        data if provided.

        """
        self._init_graph(name, 'DiGraph')
        dg = DiGraph(self, name, data, **attr)
        if self.caching:
            self.graph[name] = dg
        return dg

    def new_multigraph(self, name, data=None, **attr):
        """Return a new instance of type MultiGraph, initialized with the given
        data if provided.

        """
        self._init_graph(name, 'MultiGraph')
        mg = MultiGraph(self, name, data, **attr)
        if self.caching:
            self.graph[name] = mg
        return mg

    def new_multidigraph(self, name, data=None, **attr):
        """Return a new instance of type MultiDiGraph, initialized with the given
        data if provided.

        """
        self._init_graph(name, 'MultiDiGraph')
        mdg = MultiDiGraph(self, name, data, **attr)
        if self.caching:
            self.graph[name] = mdg
        return mdg

    def get_graph(self, name):
        """Return a graph previously created with ``new_graph``,
        ``new_digraph``, ``new_multigraph``, or
        ``new_multidigraph``

        """
        if self.caching and name in self.graph:
            return self.graph[name]
        graphtypes = {
            'Graph': Graph,
            'DiGraph': DiGraph,
            'MultiGraph': MultiGraph,
            'MultiDiGraph': MultiDiGraph
        }
        type_s = self.query.graph_type(name)
        if type_s not in graphtypes:
            raise GraphNameError(
                "I don't know of a graph named {}".format(name)
            )
        g = graphtypes[type_s](self, name)
        if self.caching:
            self.graph[name] = g
        return g

    def del_graph(self, name):
        """Remove all traces of a graph's existence from the database"""
        # make sure the graph exists before deleting anything
        self.get_graph(name)
        self.query.del_graph(name)
        if self.caching and name in self.graph:
            del self.graph[name]

    def _active_branches(self, branch=None, rev=None):
        """Private use. Iterate over (branch, rev) pairs, where the branch is
        a descendant of the previous (starting with whatever branch is
        presently active and ending at 'master'), and the rev is the
        latest revision in the branch that matters.

        """
        b = self.branch if branch is None else branch
        r = self.rev if rev is None else rev
        if self.caching:
            yield b, r
            while b in self._parentbranch_rev:
                (b, r) = self._parentbranch_rev[b]
                yield b, r
            return

        for pair in self.query.active_branches(b, r):
            yield pair

    def _branch_descendants(self, branch=None):
        """Iterate over all branches immediately descended from the current
        one (or the given one, if available).

        """
        branch = branch or self.branch
        if not self.caching:
            for desc in self.query.branch_descendants(branch):
                yield desc
            return
        for (parent, (child, rev)) in self._parentbranch_rev.items():
            if parent == branch:
                yield child


__all__ = [ORM, 'alchemy', 'graph', 'query', 'window', 'xjson']
