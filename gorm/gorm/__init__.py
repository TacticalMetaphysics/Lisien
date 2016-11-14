# This file is part of gorm, an object relational mapper for versioned graphs.
# Copyright (C) 2014 Zachary Spector.
from collections import defaultdict, deque
from .graph import (
    Graph,
    DiGraph,
    MultiGraph,
    MultiDiGraph,
)
from .query import QueryEngine
from .cache import Cache, NodesCache, EdgesCache


class GraphNameError(KeyError):
    pass


class ORM(object):
    """Instantiate this with the same string argument you'd use for a
    SQLAlchemy ``create_engine`` call. This will be your interface to
    gorm.

    """
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
        self.db = query_engine_class(dbstring, connect_args, alchemy, json_dump, json_load)
        self._obranch = None
        self._orev = None
        self.db.initdb()
        # I will be recursing a lot so just cache all the branch info
        if caching:
            self.caching = True
            self._global_cache = self.db._global_cache = {}
            for k, v in self.db.global_items():
                if k == 'branch':
                    self._obranch = v
                elif k == 'rev':
                    self._orev = v
                else:
                    self._global_cache[k] = v
            self._childbranch = defaultdict(set)
            self._parentbranch_rev = {}
            for (branch, parent, parent_rev) in self.db.all_branches():
                if branch != 'master':
                    self._parentbranch_rev[branch] = (parent, parent_rev)
                self._childbranch[parent].add(branch)
            self.graph = {}
            for (graph, typ) in self.db.graphs_types():
                self.graph[graph] = {
                    'Graph': Graph,
                    'DiGraph': DiGraph,
                    'MultiGraph': MultiGraph,
                    'MultiDiGraph': MultiDiGraph
                }[typ](self, graph)
            self._obranch = self.branch
            self._orev = self.rev
            self._active_branches_cache = []
            self.db.active_branches = self._active_branches
            todo = deque(self.db.timestream_data())
            while todo:
                (branch, parent, parent_tick) = working = todo.popleft()
                if branch == 'master':
                    continue
                if parent in self._branches:
                    self._parentbranch_rev[branch] = (parent, parent_tick)
                    self._childbranch[parent].add(branch)
                else:
                    todo.append(working)
            self._graph_val_cache = Cache(self)
            for row in self.db.graph_val_dump():
                self._graph_val_cache.store(*row)
            self._node_val_cache = Cache(self)
            for row in self.db.node_val_dump():
                self._node_val_cache.store(*row)
            self._nodes_cache = NodesCache(self)
            for row in self.db.nodes_dump():
                self._nodes_cache.store(*row)
            self._edge_val_cache = Cache(self)
            for row in self.db.edge_val_dump():
                self._edge_val_cache.store(*row)
            self._edges_cache = EdgesCache(self)
            for row in self.db.edges_dump():
                self._edges_cache.store(*row)

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
        return self.db.have_branch(b)

    def is_parent_of(self, parent, child):
        """Return whether ``child`` is a branch descended from ``parent`` at
        any remove.

        """
        if parent == 'master':
            return True
        if child == 'master':
            return False
        if child not in self._parentbranch_rev:
            raise ValueError("The branch {} seems not to have ever been created".format(child))
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
            self.db.new_branch(v, curbranch, currev)
        # make sure I'll end up within the revision range of the
        # destination branch
        if v != 'master':
            if self.caching:
                if v not in self._parentbranch_rev:
                    self._parentbranch_rev[v] = (curbranch, currev)
                parrev = self._parentbranch_rev[v][1]
            else:
                parrev = self.db.parrev(v)
            if currev < parrev:
                raise ValueError(
                    "Tried to jump to branch {br}, which starts at revision {rv}. "
                    "Go to rev {rv} or later to use this branch.".format(
                        br=v,
                        rv=parrev
                    )
                )
        if self.caching:
            self._obranch = v
        else:
            self.db.globl['branch'] = v

    @property
    def rev(self):
        """Return the global value ``rev``, or ``self._orev`` if that's set"""
        if self._orev is not None:
            return self._orev
        return self.db.globl['rev']

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
                (parent, parent_rev) = self.db.parparrev(branch)
            if v < int(parent_rev):
                raise ValueError(
                    "The revision number {revn} "
                    "occurs before the start of "
                    "the branch {brnch}".format(revn=v, brnch=branch)
                )
        if self.caching:
            self._orev = v
        else:
            self.db.globl['rev'] = v

    def commit(self):
        """Alias of ``self.db.commit``"""
        if self.caching:
            self.db.globl['branch'] = self._obranch
            self.db.globl['rev'] = self._orev
        self.db.commit()

    def close(self):
        """Alias of ``self.db.close``"""
        if self.caching:
            self.db.globl['branch'] = self._obranch
            self.db.globl['rev'] = self._orev
        self.db.close()

    def initdb(self):
        """Alias of ``self.db.initdb``"""
        self.db.initdb()

    def _init_graph(self, name, type_s='Graph'):
        if self.db.have_graph(name):
            raise GraphNameError("Already have a graph by that name")
        self.db.new_graph(name, type_s)

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
        type_s = self.db.graph_type(name)
        if type_s not in graphtypes:
            raise GraphNameError("I don't know of a graph named {}".format(name))
        g = graphtypes[type_s](self, name)
        if self.caching:
            self.graph[name] = g
        return g

    def del_graph(self, name):
        """Remove all traces of a graph's existence from the database"""
        # make sure the graph exists before deleting anything
        self.get_graph(name)
        self.db.del_graph(name)
        if self.caching and name in self.graph:
            del self.graph[name]

    def _active_branches(self, branch=None, rev=None):
        """Private use. Iterate over (branch, rev) pairs, where the branch is
        a descendant of the previous (starting with whatever branch is
        presently active and ending at 'master'), and the rev is the
        latest revision in the branch that matters.

        """
        b = branch or self.branch
        r = rev or self.rev
        if self.caching:
            yield b, r
            while b in self._parentbranch_rev:
                (b, r) = self._parentbranch_rev[b]
                yield b, r
            return

        for pair in self.db.active_branches(b, r):
            yield pair

    def _branch_descendants(self, branch=None):
        """Iterate over all branches immediately descended from the current
        one (or the given one, if available).

        """
        branch = branch or self.branch
        if not self.caching:
            for desc in self.db.branch_descendants(branch):
                yield desc
            return
        for (parent, (child, rev)) in self._parentbranch_rev.items():
            if parent == branch:
                yield child


__all__ = [ORM, 'alchemy', 'graph', 'query', 'window', 'xjson']
