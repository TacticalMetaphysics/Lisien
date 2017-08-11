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

    def _init_caches(self):
        self._global_cache = self.query._global_cache = {}
        self._node_objs = {}
        self._edge_objs = {}
        for k, v in self.query.global_items():
            if k == 'branch':
                self._obranch = v
            elif k == 'turn':
                self._oturn = v
            elif k == 'tick':
                self._otick = v
            else:
                self._global_cache[k] = v
        self._childbranch = defaultdict(set)
        self._parentbranch_turn = {}
        self._branch_end = defaultdict(lambda: 0)
        self._turn_end = defaultdict(lambda: 0)
        self._graph_val_cache = Cache(self)
        self._nodes_cache = NodesCache(self)
        self._edges_cache = EdgesCache(self)
        self._node_val_cache = Cache(self)
        self._edge_val_cache = Cache(self)
        self._active_branches_cache = []
        self.query.active_branches = self._active_branches
        self._graph_objs = {}

    def _load_graphs(self):
        for (graph, typ) in self.query.graphs_types():
            self._graph_objs[graph] = {
                'Graph': Graph,
                'DiGraph': DiGraph,
                'MultiGraph': MultiGraph,
                'MultiDiGraph': MultiDiGraph
            }[typ](self, graph)

    def _init_load(self):
        graphval = defaultdict(lambda: defaultdict(list))
        nodeval = defaultdict(lambda: defaultdict(list))
        edgeval = defaultdict(lambda: defaultdict(list))
        nodes = defaultdict(lambda: defaultdict(list))
        edges = defaultdict(lambda: defaultdict(list))
        for (graph, key, branch, turn, tick, val) in self.query.graph_val_dump():
            graphval[branch][turn].append((graph, key, branch, turn, tick, val))
        for (graph, node, branch, turn, tick, ex) in self.query.nodes_dump():
            nodes[branch][turn].append((graph, node, branch, turn, tick, ex))
        for (graph, u, v, i, branch, turn, tick, ex) in self.query.edges_dump():
            edges[branch][turn].append((graph, u, v, i, branch, turn, tick, ex))
        for (
                graph, node, key, branch, turn, tick, val
        ) in self.query.node_val_dump():
            nodeval[branch][turn].append((graph, node, key, branch, turn, tick, val))
        for (
                graph, u, v, i, key, branch, turn, tick, val
        ) in self.query.edge_val_dump():
            edgeval[branch][turn].append((graph, u, v, i, key, branch, turn, tick, val))
        return [
            (self._nodes_cache, nodes),
            (self._edges_cache, edges),
            (self._graph_val_cache, graphval),
            (self._node_val_cache, nodeval),
            (self._edge_val_cache, edgeval)
        ]

    def _cache_histories(self, histories):
        # Make sure to load in the correct order, so that
        # caches for child branches get built after their parents.
        # Do the graphs first, so that everything else will have
        # someplace to be.
        if not hasattr(self, 'graph'):
            self.graph = self._graph_objs
        branch2do = deque(['trunk'])
        while branch2do:
            branch = branch2do.popleft()
            # find the last turn in the branch
            last_turn = 0
            for cache, history in histories:
                if len(history[branch]) - 1 > last_turn:
                    last_turn = len(history[branch]) - 1
            # store all the histories in their corresponding caches
            for rev in range(last_turn+1):
                for cache, history in histories:
                    for row in history[branch][rev]:
                        cache.store(*row)
            self._branch_end[branch] = last_turn
            if branch in self._childbranch:
                branch2do.extend(self._childbranch[branch])

    def __init__(
            self,
            dbstring,
            alchemy=True,
            connect_args={},
            query_engine_class=QueryEngine,
            json_dump=None,
            json_load=None
    ):
        """Make a SQLAlchemy engine if possible, else a sqlite3 connection. In
        either case, begin a transaction.

        """
        self.query = query_engine_class(
            dbstring, connect_args, alchemy, json_dump, json_load
        )
        self.query.initdb()
        self._obranch = self.query.globl['branch']
        self._oturn = self.query.globl['turn']
        self._otick = self.query.globl['tick']
        self._init_caches()
        for (branch, parent, parent_turn) in self.query.all_branches():
            if branch != 'trunk':
                self._parentbranch_turn[branch] = (parent, parent_turn)
            self._childbranch[parent].add(branch)
        self._load_graphs()
        self._cache_histories(self._init_load())

    def __enter__(self):
        """Enable the use of the ``with`` keyword"""
        return self

    def __exit__(self, *args):
        """Alias for ``close``"""
        self.close()

    def _havebranch(self, b):
        """Private use. Checks that the branch is known about."""
        return b in self._parentbranch_turn

    def is_parent_of(self, parent, child):
        """Return whether ``child`` is a branch descended from ``parent`` at
        any remove.

        """
        if parent == 'trunk':
            return True
        if child == 'trunk':
            return False
        if child not in self._parentbranch_turn:
            raise ValueError(
                "The branch {} seems not to have ever been created".format(
                    child
                )
            )
        if self._parentbranch_turn[child][0] == parent:
            return True
        return self.is_parent_of(parent, self._parentbranch_turn[child][0])

    @property
    def branch(self):
        return self._obranch

    @branch.setter
    def branch(self, v):
        curbranch = self.branch
        if curbranch == v:
            return
        curturn = self.turn
        if not self._havebranch(v):
            # assumes the present turn in the parent branch has
            # been finalized.
            self.query.new_branch(v, curbranch, curturn)
        # make sure I'll end up within the revision range of the
        # destination branch
        if v != 'trunk':
            if v not in self._parentbranch_turn:
                self._parentbranch_turn[v] = (curbranch, curturn)
            parturn = self._parentbranch_turn[v][1]
            if curturn < parturn:
                raise ValueError(
                    "Tried to jump to branch {br}, "
                    "which starts at turn {rv}. "
                    "Go to turn {rv} or later to use this branch.".format(
                        br=v,
                        rv=parturn
                    )
                )
        self._obranch = v

    @property
    def turn(self):
        return self._oturn

    @turn.setter
    def turn(self, v):
        if v == self.turn:
            return
        # first make sure the cursor is not before the start of this branch
        branch = self.branch
        if branch != 'trunk':
            (parent, parent_rev) = self._parentbranch_turn[branch]
            if v < int(parent_rev):
                raise ValueError(
                    "The turn number {} "
                    "occurs before the start of "
                    "the branch {}".format(v, branch)
                )
        self._oturn = v

    @property
    def tick(self):
        return self._otick

    @tick.setter
    def tick(self, v):
        time = self._obranch, self._otick
        if v > self._turn_end[time]:
            self._turn_end[time] = v
        self._otick = v

    def btt(self):
        return self._obranch, self._oturn, self._otick

    def commit(self):
        self.query.globl['branch'] = self._obranch
        self.query.globl['turn'] = self._oturn
        self.query.globl['tick'] = self._otick
        self.query.commit()

    def close(self):
        self.query.globl['branch'] = self._obranch
        self.query.globl['turn'] = self._oturn
        self.query.globl['tick'] = self._otick
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
        self._graph_objs[name] = g
        return g

    def new_digraph(self, name, data=None, **attr):
        """Return a new instance of type DiGraph, initialized with the given
        data if provided.

        """
        self._init_graph(name, 'DiGraph')
        dg = DiGraph(self, name, data, **attr)
        self._graph_objs[name] = dg
        return dg

    def new_multigraph(self, name, data=None, **attr):
        """Return a new instance of type MultiGraph, initialized with the given
        data if provided.

        """
        self._init_graph(name, 'MultiGraph')
        mg = MultiGraph(self, name, data, **attr)
        self._graph_objs[name] = mg
        return mg

    def new_multidigraph(self, name, data=None, **attr):
        """Return a new instance of type MultiDiGraph, initialized with the given
        data if provided.

        """
        self._init_graph(name, 'MultiDiGraph')
        mdg = MultiDiGraph(self, name, data, **attr)
        self._graph_objs[name] = mdg
        return mdg

    def get_graph(self, name):
        """Return a graph previously created with ``new_graph``,
        ``new_digraph``, ``new_multigraph``, or
        ``new_multidigraph``

        """
        if name in self._graph_objs:
            return self._graph_objs[name]
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
        self._graph_objs[name] = g
        return g

    def del_graph(self, name):
        """Remove all traces of a graph's existence from the database"""
        # make sure the graph exists before deleting anything
        self.get_graph(name)
        self.query.del_graph(name)
        if name in self._graph_objs:
            del self._graph_objs[name]

    def _active_branches(self, branch=None, turn=None):
        """Private use. Iterate over (branch, turn) pairs, where the branch is
        a descendant of the previous (starting with whatever branch is
        presently active and ending at 'trunk'), and the turn is the
        latest revision in the branch that matters.

        """
        b = self.branch if branch is None else branch
        t = self.turn if turn is None else turn
        yield b, t
        while b in self._parentbranch_turn:
            (b, t) = self._parentbranch_turn[b]
            yield b, t

    def _branch_descendants(self, branch=None):
        """Iterate over all branches immediately descended from the current
        one (or the given one, if available).

        """
        branch = branch or self.branch
        for (parent, (child, rev)) in self._parentbranch_turn.items():
            if parent == branch:
                yield child


__all__ = [ORM, 'alchemy', 'graph', 'query', 'window', 'xjson']
