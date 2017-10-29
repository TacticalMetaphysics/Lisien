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
from .cache import Cache, NodesCache, EdgesCache, HistoryError


class GraphNameError(KeyError):
    pass


class PlanningContext(object):
    """A context manager for 'hypothetical' edits.

    Start a block of code like:

    with orm.plan:
        ...

    and any changes you make to the world state within that block will be
    'plans,' meaning that they are used as defaults. The world will
    obey your plan unless you make changes to the same entities outside
    of the plan, in which case the world will obey those.

    New branches cannot be started within plans.

    """
    __slots__ = ['orm', 'time']

    def __init__(self, orm):
        self.orm = orm

    def __enter__(self):
        self.orm.planning = True
        self.time = self.orm.btt()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.orm._obranch, self.orm._oturn, self.orm._otick = self.time
        self.orm.planning = False


class AdvancingContext(object):
    """A context manager for when time is moving forward.

    When used in LiSE, this means that the game is being simulated.
    It changes how the caching works, making it more efficient.

    """
    __slots__ = ['orm']

    def __init__(self, orm):
        self.orm = orm

    def __enter__(self):
        self.orm.forward = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.orm.forward = False


class ORM(object):
    """Instantiate this with the same string argument you'd use for a
    SQLAlchemy ``create_engine`` call. This will be your interface to
    allegedb.

    """
    node_cls = _make_node = Node
    edge_cls = _make_edge = Edge
    query_engine_cls = QueryEngine

    @property
    def plan(self):
        return PlanningContext(self)
    plan.__doc__ = PlanningContext.__doc__

    @property
    def advancing(self):
        return AdvancingContext(self)
    advancing.__doc__ = AdvancingContext.__doc__

    def _init_caches(self):
        self._global_cache = self.query._global_cache = {}
        self._node_objs = {}
        self._edge_objs = {}
        for k, v in self.query.global_items():
            if k == 'branch':
                self._obranch = v
            elif k == 'turn':
                self._oturn = int(v)
            elif k == 'tick':
                self._otick = int(v)
            else:
                self._global_cache[k] = v
        self._childbranch = defaultdict(set)
        self._branches = {}
        self._turn_end = defaultdict(lambda: 0)
        self._graph_val_cache = Cache(self)
        self._nodes_cache = NodesCache(self)
        self._edges_cache = EdgesCache(self)
        self._node_val_cache = Cache(self)
        self._edge_val_cache = Cache(self)
        self._graph_objs = {}

    def load_graphs(self):
        for (graph, typ) in self.query.graphs_types():
            self._graph_objs[graph] = {
                'Graph': Graph,
                'DiGraph': DiGraph,
                'MultiGraph': MultiGraph,
                'MultiDiGraph': MultiDiGraph
            }[typ](self, graph)

    def __init__(
            self,
            dbstring,
            alchemy=True,
            connect_args={},
            validate=False
    ):
        """Make a SQLAlchemy engine if possible, else a sqlite3 connection. In
        either case, begin a transaction.

        """
        self.planning = False
        self.forward = False
        if not hasattr(self, 'query'):
            self.query = self.query_engine_cls(
                dbstring, connect_args, alchemy,
                getattr(self, 'json_dump', None), getattr(self, 'json_load', None)
            )
        self.query.initdb()
        self._init_caches()
        for (branch, parent, parent_turn, parent_tick, end_turn, end_tick) in self.query.all_branches():
            self._branches[branch] = (parent, parent_turn, parent_tick, end_turn, end_tick)
            self._childbranch[parent].add(branch)
        if 'trunk' not in self._branches:
            self._branches['trunk'] = None, 0, 0, 0, 0
        self.load_graphs()
        self._init_load(validate=validate)

    def _init_load(self, validate=False):
        noderows = list(self.query.nodes_dump())
        self._nodes_cache.load(noderows, validate=validate)
        edgerows = list(self.query.edges_dump())
        self._edges_cache.load(edgerows, validate=validate)
        self._graph_val_cache.load(self.query.graph_val_dump(), validate=validate)
        self._node_val_cache.load(self.query.node_val_dump(), validate=validate)
        self._edge_val_cache.load(self.query.edge_val_dump(), validate=validate)
        if not hasattr(self, 'graph'):
            self.graph = self._graph_objs
        for graph, node, branch, turn, tick, ex in noderows:
            self._node_objs[(graph, node)] = self._make_node(self.graph[graph], node)
        for graph, orig, dest, idx, branch, turn, tick, ex in edgerows:
            self._edge_objs[(graph, orig, dest, idx)] = self._make_edge(self.graph[graph], orig, dest, idx)

    def __enter__(self):
        """Enable the use of the ``with`` keyword"""
        return self

    def __exit__(self, *args):
        """Alias for ``close``"""
        self.close()

    def is_parent_of(self, parent, child):
        """Return whether ``child`` is a branch descended from ``parent`` at
        any remove.

        """
        if parent == 'trunk':
            return True
        if child == 'trunk':
            return False
        if child not in self._branches:
            raise ValueError(
                "The branch {} seems not to have ever been created".format(
                    child
                )
            )
        if self._branches[child][0] == parent:
            return True
        return self.is_parent_of(parent, self._branches[child][0])

    def _set_branch(self, v):
        curbranch, curturn, curtick = self.btt()
        if curbranch == v:
            return
        if v not in self._branches:
            # assumes the present turn in the parent branch has
            # been finalized.
            self.query.new_branch(v, curbranch, curturn, curtick)
        # make sure I'll end up within the revision range of the
        # destination branch
        if v != 'trunk':
            parturn = self._branches[v][1]
            if curturn < parturn:
                raise ValueError(
                    "Tried to jump to branch {br}, "
                    "which starts at turn {rv}. "
                    "Go to turn {rv} or later to use this branch.".format(
                        br=v,
                        rv=parturn
                    )
                )
            if not self.planning and v not in self._branches:
                self._branches[v] = (curbranch, curturn, curtick, curturn, curtick)
        self._obranch = v
    branch = property(lambda self: self._obranch, _set_branch)  # easier to override this way

    def _set_turn(self, v):
        if v == self.turn:
            return
        if not isinstance(v, int):
            raise TypeError("turn must be an integer")
        # enforce the arrow of time, if it's in effect
        if self.forward and v < self._oturn:
            raise ValueError("Can't time travel backward in a forward context")
        # first make sure the cursor is not before the start of this branch
        branch = self.branch
        tick = self._turn_end.setdefault((branch, v), 0)
        parent, turn_start, tick_start, turn_end, tick_end = self._branches[branch]
        if branch != 'trunk':
            if v < turn_start:
                raise ValueError(
                    "The turn number {} "
                    "occurs before the start of "
                    "the branch {}".format(v, branch)
                )
        if not self.planning and v > turn_end:
            self._branches[branch] = parent, turn_start, tick_start, v, tick
        self._otick = tick
        self._oturn = v
    turn = property(lambda self: self._oturn, _set_turn)  # easier to override this way

    def _set_tick(self, v):
        if not isinstance(v, int):
            raise TypeError("tick must be an integer")
        time = branch, turn = self._obranch, self._oturn
        # enforce the arrow of time, if it's in effect
        if self.forward and v < self._otick:
            raise ValueError("Can't time travel backward in a forward context")
        if not self.planning:
            if v > self._turn_end[time]:
                self._turn_end[time] = v
            parent, turn_start, tick_start, turn_end, tick_end = self._branches[branch]
            if turn == turn_end and v > tick_end:
                self._branches[branch] = parent, turn_start, tick_start, turn, v
        self._otick = v
    tick = property(lambda self: self._otick, _set_tick)  # easier to override this way

    def btt(self):
        return self._obranch, self._oturn, self._otick

    def nbtt(self):
        """Increment the tick and return branch, turn, tick

        Unless we're viewing the past, in which case raise HistoryError.

        Idea is you use this when you want to advance time, which you
        can only do once per branch, turn, tick.

        """
        branch, turn, tick = self.btt()
        tick += 1
        if self._turn_end[branch, turn] > tick:
            raise HistoryError(
                "You're not at the end of turn {}. Go to tick {} to change things".format(
                    turn, self._turn_end[branch, turn]
                )
            )
        parent, turn_start, tick_start, turn_end, tick_end = self._branches[branch]
        if turn_end > turn:
            raise HistoryError(
                "You're in the past. Go to turn {} to change things".format(turn_end)
            )
        if not self.planning:
            if turn_end != turn:
                raise HistoryError(
                    "When advancing time outside of a plan, you can't skip turns. Go to turn {}".format(turn_end)
                )
            self._branches[branch] = parent, turn_start, tick_start, turn_end, tick
            self._turn_end[branch, turn] = self._otick = tick
        return branch, turn, tick

    def commit(self):
        self.query.globl['branch'] = self._obranch
        self.query.globl['turn'] = self._oturn
        self.query.globl['tick'] = self._otick
        for branch, (parent, turn_start, tick_start, turn_end, tick_end) in self._branches.items():
            self.query.update_branch(branch, parent, turn_start, tick_start, turn_end, tick_end)
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

    def _iter_parent_btt(self, branch=None, turn=None, tick=None):
        """Private use. Iterate over (branch, turn, tick), where the branch is
        a descendant of the previous (starting with whatever branch is
        presently active and ending at 'trunk'), and the turn is the
        latest revision in the branch that matters.

        """
        b = branch or self.branch
        trn = self.turn if turn is None else turn
        tck = self.tick if tick is None else tick
        yield b, trn, tck
        while b in self._branches:
            (b, trn, tck, _, _) = self._branches[b]
            yield b, trn, self._turn_end[b, trn]

    def _branch_descendants(self, branch=None):
        """Iterate over all branches immediately descended from the current
        one (or the given one, if available).

        """
        branch = branch or self.branch
        for (parent, (child, _, _, _, _)) in self._branches.items():
            if parent == branch:
                yield child


__all__ = [ORM, 'alchemy', 'graph', 'query', 'window', 'xjson']
