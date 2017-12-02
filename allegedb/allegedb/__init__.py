# This file is part of allegedb, an object relational mapper for versioned graphs.
# Copyright (C) Zachary Spector.
from collections import defaultdict
from functools import partial
from blinker import Signal
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
    """A context manager for when time is moving forward one turn at a time.

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


class TimeSignal(Signal):
    """Acts like a tuple of the time in (branch, turn) for the most part.

    This is a Signal, so pass a function to the connect(...) method and
    it will be called whenever the time changes. Not when the tick
    changes, though. If you really need something done whenever the
    tick changes, override the _set_tick method of
    :class:`allegedb.ORM`.

    """
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def __iter__(self):
        yield self.engine.branch
        yield self.engine.turn

    def __len__(self):
        return 2

    def __getitem__(self, i):
        if i in ('branch', 0):
            return self.engine.branch
        if i in ('turn', 1):
            return self.engine.turn

    def __setitem__(self, i, v):
        branch_then, turn_then, tick_then = self.engine.btt()
        if i in ('branch', 0):
            self.engine.branch = v
        if i in ('turn', 1):
            self.engine.turn = v
        branch_now, turn_now, tick_now = self.engine.btt()
        self.send(
            self, branch_then=branch_then, turn_then=turn_then, tick_then=tick_then,
            branch_now=branch_now, turn_now=turn_now, tick_now=tick_now
        )

    def __str__(self):
        return str((self.engine.branch, self.engine.turn))


class TimeSignalDescriptor:
    __doc__ = TimeSignal.__doc__
    signals = {}

    def __get__(self, inst, cls):
        if id(inst) not in self.signals:
            self.signals[id(inst)] = TimeSignal(inst)
        return self.signals[id(inst)]

    def __set__(self, inst, val):
        if id(inst) not in self.signals:
            self.signals[id(inst)] = TimeSignal(inst)
        real = self.signals[id(inst)]
        branch_then, turn_then, tick_then = real.engine.btt()
        branch_now, turn_now = val
        if (branch_then, turn_then) == (branch_now, turn_now):
            return
        e = real.engine
        # enforce the arrow of time, if it's in effect
        if e.forward:
            if branch_now != branch_then:
                raise ValueError("Can't change branches in a forward context")
            if turn_now < turn_then:
                raise ValueError("Can't time travel backward in a forward context")
            if turn_now > turn_then + 1:
                raise ValueError("Can't skip turns in a forward context")
        # make sure I'll end up within the revision range of the
        # destination branch
        branches = e._branches
        tick_now = e._turn_end_plan.setdefault((branch_now, turn_now), 0)
        if branch_now in branches:
            parent, turn_start, tick_start, turn_end, tick_end = branches[branch_now]
            if turn_now < turn_start:
                raise ValueError(
                    "The turn number {} "
                    "occurs before the start of "
                    "the branch {}".format(turn_now, branch_now)
                )
            if not e.planning and (turn_now > turn_end or tick_now > tick_end):
                branches[branch_now] = parent, turn_start, tick_start, turn_now, tick_now
        else:
            branches[branch_now] = (
                branch_then, turn_now, tick_now, turn_now, tick_now
            )
            e.query.new_branch(branch_now, branch_then, turn_now, tick_now)
        e._obranch, e._oturn = branch, turn = val

        if turn > e._turn_end_plan[val]:
            e._turn_end_plan[val] = turn
        if not e.planning:
            if tick_now > e._turn_end[val]:
                e._turn_end[val] = tick_now
        e._otick = tick_now
        real.send(
            e,
            branch_then=branch_then,
            turn_then=turn_then,
            tick_then=tick_then,
            branch_now=branch_now,
            turn_now=turn_now,
            tick_now=tick_now
        )


def setgraphval(delta, graph, key, val):
    delta.setdefault(graph, {})[key] = val


def setnode(delta, graph, node, exists):
    delta.setdefault(graph, {}).setdefault('nodes', {})[node] = bool(exists)


def setnodeval(delta, graph, node, key, value):
    if (
        graph in delta and 'nodes' in delta[graph] and
        node in delta[graph]['nodes'] and not delta[graph]['nodes'][node]
    ):
        return
    delta.setdefault(graph, {}).setdefault('node_val', {}).setdefault(node, {})[key] = value


def setedge(delta, is_multigraph, graph, orig, dest, idx, exists):
    if is_multigraph(graph):
        delta.setdefault(graph, {}).setdefault('edges', {})\
            .setdefault(orig, {}).setdefault(dest, {})[idx] = bool(exists)
    else:
        delta.setdefault(graph, {}).setdefault('edges', {})\
            .setdefault(orig, {})[dest] = bool(exists)


def setedgeval(delta, is_multigraph, graph, orig, dest, idx, key, value):
    if is_multigraph(graph):
        if (
            graph in delta and 'edges' in delta[graph] and
            orig in delta[graph]['edges'] and dest in delta[graph]['edges'][orig]
            and idx in delta[graph]['edges'][orig][dest]
            and not delta[graph]['edges'][orig][dest][idx]
        ):
            return
        delta.setdefault(graph, {}).setdefault('edge_val', {})\
            .setdefault(orig, {}).setdefault(dest, {})\
            .setdefault(idx, {})[key] = value
    else:
        if (
                                    graph in delta and 'edges' in delta[graph] and
                                orig in delta[graph]['edges'] and dest in delta[graph]['edges'][orig]
                and not delta[graph]['edges'][orig][dest]
        ):
            return
        delta.setdefault(graph, {}).setdefault('edge_val', {})\
            .setdefault(orig, {}).setdefault(dest, {})[key] = value


def update_window(turn_from, tick_from, turn_to, tick_to, updfun, branchd):
    if branchd.has_exact_rev(turn_from):
        # Not including the exact tick you started from because deltas are *changes*
        for past_state in branchd[turn_from][tick_from+1:]:
            updfun(*past_state)
    for midturn in range(turn_from+1, turn_to):
        if branchd.has_exact_rev(midturn):
            for past_state in branchd[midturn][:]:
                updfun(*past_state)
    if branchd.has_exact_rev(turn_to):
        for past_state in branchd[turn_to][:tick_to]:
            updfun(*past_state)


def update_backward_window(turn_from, tick_from, turn_to, tick_to, updfun, branchd):
    if branchd.has_exact_rev(turn_from):
        for future_state in reversed(branchd[turn_from][:tick_from]):
            updfun(*future_state)
    for midturn in range(turn_from-1, turn_to, -1):
        if branchd.has_exact_rev(midturn):
            for future_state in reversed(branchd[midturn][:]):
                updfun(*future_state)
    if branchd.has_exact_rev(turn_to):
        for future_state in reversed(branchd[turn_to][tick_to:]):
            updfun(*future_state)


class ORM(object):
    """Instantiate this with the same string argument you'd use for a
    SQLAlchemy ``create_engine`` call. This will be your interface to
    allegedb.

    """
    node_cls = _make_node = Node
    edge_cls = _make_edge = Edge
    query_engine_cls = QueryEngine
    illegal_graph_names = ['global']
    illegal_node_names = ['nodes', 'node_val', 'edges', 'edge_val']
    time = TimeSignalDescriptor()

    @property
    def plan(self):
        return PlanningContext(self)
    plan.__doc__ = PlanningContext.__doc__

    @property
    def advancing(self):
        return AdvancingContext(self)
    advancing.__doc__ = AdvancingContext.__doc__

    def get_delta(self, branch, turn_from, tick_from, turn_to, tick_to):
        """Get a dictionary describing changes to all graphs.

        The keys are graph names. Their values are dictionaries of the graphs'
        attributes' new values, with ``None`` for deleted keys. Also in those graph
        dictionaries are special keys 'node_val' and 'edge_val' describing changes
        to node and edge attributes, and 'nodes' and 'edges' full of booleans
        indicating whether a node or edge exists.

        """
        if turn_from == turn_to:
            return self.get_turn_delta(branch, turn_from, tick_from, tick_to)
        delta = {}
        graph_objs = self._graph_objs
        if turn_to < turn_from:
            updater = partial(update_backward_window, turn_from, tick_from, turn_to, tick_to)
            gvbranches = self._graph_val_cache.presettings
            nbranches = self._nodes_cache.presettings
            nvbranches = self._node_val_cache.presettings
            ebranches = self._edges_cache.presettings
            evbranches = self._edge_val_cache.presettings
        else:
            updater = partial(update_window, turn_from, tick_from, turn_to, tick_to)
            gvbranches = self._graph_val_cache.settings
            nbranches = self._nodes_cache.settings
            nvbranches = self._node_val_cache.settings
            ebranches = self._edges_cache.settings
            evbranches = self._edges_cache.settings

        if branch in gvbranches:
            updater(partial(setgraphval, delta), gvbranches[branch])

        if branch in nbranches:
            updater(partial(setnode, delta), nbranches[branch])

        if branch in nvbranches:
            updater(partial(setnodeval, delta), nvbranches[branch])

        if branch in ebranches:
            updater(partial(setedge, delta, lambda g: graph_objs[g].is_multigraph()), ebranches[branch])

        if branch in evbranches:
            updater(partial(setedgeval, delta, lambda g: graph_objs[g].is_multigraph()), evbranches[branch])

        return delta

    def get_turn_delta(self, branch=None, turn=None, tick_from=0, tick_to=None):
        """Get a dictionary describing changes made on a given turn.

        If ``tick_to`` is not supplied, report all changes after ``tick_from``
        (default 0).

        The keys are graph names. Their values are dictionaries of the graphs'
        attributes' new values, with ``None`` for deleted keys. Also in those graph
        dictionaries are special keys 'node_val' and 'edge_val' describing changes
        to node and edge attributes, and 'nodes' and 'edges' full of booleans
        indicating whether a node or edge exists.

        """
        branch = branch or self.branch
        turn = turn or self.turn
        tick_to = tick_to or self.tick
        delta = {}
        if tick_from < tick_to:
            gvbranches = self._graph_val_cache.settings
            nbranches = self._nodes_cache.settings
            nvbranches = self._node_val_cache.settings
            ebranches = self._edges_cache.settings
            evbranches = self._edge_val_cache.settings
        else:
            gvbranches = self._graph_val_cache.presettings
            nbranches = self._nodes_cache.presettings
            nvbranches = self._node_val_cache.presettings
            ebranches = self._edges_cache.presettings
            evbranches = self._edge_val_cache.presettings

        if branch in gvbranches and gvbranches[branch].has_exact_rev(turn):
            for graph, key, value in gvbranches[branch][turn][tick_from:tick_to]:
                if graph in delta:
                    delta[graph][key] = value
                else:
                    delta[graph] = {key: value}

        if branch in nbranches and nbranches[branch].has_exact_rev(turn):
            for graph, node, exists in nbranches[branch][turn][tick_from:tick_to]:
                delta.setdefault(graph, {}).setdefault('nodes', {})[node] = bool(exists)

        if branch in nvbranches and nvbranches[branch].has_exact_rev(turn):
            for graph, node, key, value in nvbranches[branch][turn][tick_from:tick_to]:
                if (
                    graph in delta and 'nodes' in delta[graph] and
                    node in delta[graph]['nodes'] and not delta[graph]['nodes'][node]
                ):
                    continue
                nodevd = delta.setdefault(graph, {}).setdefault('node_val', {})
                if node in nodevd:
                    nodevd[node][key] = value
                else:
                    nodevd[node] = {key: value}

        graph_objs = self._graph_objs
        if branch in ebranches and ebranches[branch].has_exact_rev(turn):
            for graph, orig, dest, idx, exists in ebranches[branch][turn][tick_from:tick_to]:
                if graph_objs[graph].is_multigraph():
                    if (
                        graph in delta and 'edges' in delta[graph] and
                        orig in delta[graph]['edges'] and dest in delta[graph]['edges'][orig]
                        and idx in delta[graph]['edges'][orig][dest]
                        and not delta[graph]['edges'][orig][dest][idx]
                    ):
                        continue
                    delta.setdefault(graph, {}).setdefault('edges', {})\
                        .setdefault(orig, {}).setdefault(dest, {})[idx] = bool(exists)
                else:
                    if (
                        graph in delta and 'edges' in delta[graph] and
                        orig in delta[graph]['edges'] and dest in delta[graph]['edges'][orig]
                        and not delta[graph]['edges'][orig][dest]
                    ):
                        continue
                    delta.setdefault(graph, {}).setdefault('edges', {})\
                        .setdefault(orig, {})[dest] = bool(exists)

        if branch in evbranches and evbranches[branch].has_exact_rev(turn):
            for graph, orig, dest, idx, key, value in evbranches[branch][turn][tick_from:tick_to]:
                edgevd = delta.setdefault(graph, {}).setdefault('edge_val', {})\
                    .setdefault(orig, {}).setdefault(dest, {})
                if graph_objs[graph].is_multigraph():
                    if idx in edgevd:
                        edgevd[idx][key] = value
                    else:
                        edgevd[idx] = {key: value}
                else:
                    edgevd[key] = value

        return delta

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
        self._turn_end_plan = defaultdict(lambda: 0)
        self._graph_val_cache = Cache(self)
        self._nodes_cache = NodesCache(self)
        self._edges_cache = EdgesCache(self)
        self._node_val_cache = Cache(self)
        self._edge_val_cache = Cache(self)
        self._graph_objs = {}

    def _load_graphs(self):
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
        # in case this is the first startup
        self._otick = self._oturn = 0
        self._init_caches()
        for (branch, parent, parent_turn, parent_tick, end_turn, end_tick) in self.query.all_branches():
            self._branches[branch] = (parent, parent_turn, parent_tick, end_turn, end_tick)
            self._childbranch[parent].add(branch)
        for (branch, turn, end_tick, plan_end_tick) in self.query.turns_dump():
            self._turn_end[branch, turn] = end_tick
            self._turn_end_plan[branch, turn] = plan_end_tick
        if 'trunk' not in self._branches:
            self._branches['trunk'] = None, 0, 0, 0, 0
        self._load_graphs()
        self._init_load(validate=validate)

    def _init_load(self, validate=False):
        noderows = [
            (graph, node, branch, turn, tick, ex if ex else None)
            for (graph, node, branch, turn, tick, ex)
            in self.query.nodes_dump()
        ]
        self._nodes_cache.load(noderows, validate=validate)
        edgerows = [
            (graph, orig, dest, idx, branch, turn, tick, ex if ex else None)
            for (graph, orig, dest, idx, branch, turn, tick, ex)
            in self.query.edges_dump()
        ]
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
            if not self.planning:
                self._branches[v] = curbranch, curturn, curtick, curturn, curtick
        # make sure I'll end up within the revision range of the
        # destination branch
        if v != 'trunk' and not self.planning:
            if v in self._branches:
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
            else:
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
        tick = self._turn_end_plan.setdefault((branch, v), 0)
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
        if v > self._turn_end_plan[time]:
            self._turn_end_plan[time] = v
        if not self.planning:
            if v > self._turn_end[time]:
                self._turn_end[time] = v
            parent, turn_start, tick_start, turn_end, tick_end = self._branches[branch]
            if turn == turn_end and v > tick_end:
                self._branches[branch] = parent, turn_start, tick_start, turn, v
        self._otick = v
    tick = property(lambda self: self._otick, _set_tick)  # easier to override this way

    def btt(self):
        """Return the branch, turn, and tick."""
        return self._obranch, self._oturn, self._otick

    def nbtt(self):
        """Increment the tick and return branch, turn, tick

        Unless we're viewing the past, in which case raise HistoryError.

        Idea is you use this when you want to advance time, which you
        can only do once per branch, turn, tick.

        """
        branch, turn, tick = self.btt()
        tick += 1
        if (branch, turn) in self._turn_end_plan:
            if tick > self._turn_end_plan[branch, turn]:
                self._turn_end_plan[branch, turn] = tick
            else:
                tick = self._turn_end_plan[branch, turn] + 1
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
            self._turn_end[branch, turn] = tick
        self._otick = tick
        return branch, turn, tick

    def commit(self):
        """Write the state of all graphs to the database and commit the transaction.

        Also saves the current branch, turn, and tick.

        """
        self.query.globl['branch'] = self._obranch
        self.query.globl['turn'] = self._oturn
        self.query.globl['tick'] = self._otick
        set_branch = self.query.set_branch
        for branch, (parent, turn_start, tick_start, turn_end, tick_end) in self._branches.items():
            set_branch(branch, parent, turn_start, tick_start, turn_end, tick_end)
        turn_end = self._turn_end
        set_turn = self.query.set_turn
        for (branch, turn), plan_end_tick in self._turn_end_plan.items():
            set_turn(branch, turn, turn_end[branch], plan_end_tick)
        self.query.commit()

    def close(self):
        """Write changes to database and close the connection"""
        self.commit()
        self.query.close()

    def initdb(self):
        """Alias of ``self.query.initdb``"""
        self.query.initdb()

    def _init_graph(self, name, type_s='Graph'):
        if self.query.have_graph(name):
            raise GraphNameError("Already have a graph by that name")
        if name in self.illegal_graph_names:
            raise GraphNameError("Illegal name")
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


__all__ = [ORM, 'graph', 'query', 'xjson']
