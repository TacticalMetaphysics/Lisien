# This file is part of allegedb, an object relational mapper for versioned graphs.
# Copyright (C) Zachary Spector. public@zacharyspector.com
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
"""The main interface to the allegedb ORM, and some supporting functions and classes"""
from contextlib import ContextDecorator
from weakref import WeakValueDictionary

from blinker import Signal

from allegedb.window import update_window, update_backward_window
from .graph import (
    Graph,
    DiGraph,
    MultiGraph,
    MultiDiGraph,
    Node,
    Edge
)
from .query import QueryEngine, TimeError
from .window import HistoryError


class GraphNameError(KeyError):
    """For errors involving graphs' names"""


class PlanningContext(ContextDecorator):
    """A context manager for 'hypothetical' edits.

    Start a block of code like:

    ```
    with orm.plan():
        ...
    ```

    and any changes you make to the world state within that block will be
    'plans,' meaning that they are used as defaults. The world will
    obey your plan unless you make changes to the same entities outside
    of the plan, in which case the world will obey those, and cancel any
    future plan.

    New branches cannot be started within plans.

    """
    __slots__ = ['orm', 'time']

    def __init__(self, orm):
        self.orm = orm

    def __enter__(self):
        if self.orm._planning:
            raise ValueError("Already planning")
        self.orm._planning = True
        self.time = self.orm.btt()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.orm._obranch, self.orm._oturn, self.orm._otick = self.time
        self.orm._planning = False


class TimeSignal(Signal):
    """Acts like a list of ``[branch, turn]`` for the most part.

    You can set these to new values, or even replace them with a whole new
    ``[branch, turn]`` if you wish. It's even possible to use the strings
    ``'branch'`` or ``'turn'`` in the place of indices, but at that point
    you might prefer to set ``engine.branch`` or ``engine.turn`` directly.

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

    def __eq__(self, other):
        return tuple(self) == other

    def __ne__(self, other):
        return tuple(self) != other

    def __gt__(self, other):
        return tuple(self) > other

    def __ge__(self, other):
        return tuple(self) >= other

    def __lt__(self, other):
        return tuple(self) < other

    def __le__(self, other):
        return tuple(self) <= other


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
        if e._forward:
            if branch_now != branch_then:
                raise TimeError("Can't change branches in a forward context")
            if turn_now < turn_then:
                raise TimeError("Can't time travel backward in a forward context")
            if turn_now > turn_then + 1:
                raise TimeError("Can't skip turns in a forward context")
        # make sure I'll end up within the revision range of the
        # destination branch
        branches = e._branches

        if branch_now in branches:
            tick_now = e._turn_end_plan.setdefault(
                (branch_now, turn_now),
                tick_then
            )
            parent, turn_start, tick_start, turn_end, tick_end = branches[branch_now]
            if turn_now < turn_start:
                raise ValueError(
                    "The turn number {} "
                    "occurs before the start of "
                    "the branch {}".format(turn_now, branch_now)
                )
            if turn_now == turn_start and tick_now < tick_start:
                raise ValueError(
                    "The tick number {}"
                    "on turn {} "
                    "occurs before the start of "
                    "the branch {}".format(
                        tick_now, turn_now, branch_now
                    )
                )
            if not e._planning and (
                turn_now > turn_end or (
                    turn_now == turn_end and tick_now > tick_end
                )
            ):
                branches[branch_now] = parent, turn_start, tick_start, turn_now, tick_now
        else:
            tick_now = tick_then
            branches[branch_now] = (
                branch_then, turn_now, tick_now, turn_now, tick_now
            )
            e.query.new_branch(branch_now, branch_then, turn_now, tick_now)
        e._obranch, e._oturn = val

        if not e._planning:
            if tick_now > e._turn_end[val]:
                e._turn_end[val] = tick_now
        e._otick = e._turn_end_plan[val] = tick_now
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
    """Change a delta to say that a graph stat was set to a certain value"""
    delta.setdefault(graph, {})[key] = val


def setnode(delta, graph, node, exists):
    """Change a delta to say that a node was created or deleted"""
    delta.setdefault(graph, {}).setdefault('nodes', {})[node] = bool(exists)


def setnodeval(delta, graph, node, key, value):
    """Change a delta to say that a node stat was set to a certain value"""
    if (
        graph in delta and 'nodes' in delta[graph] and
        node in delta[graph]['nodes'] and not delta[graph]['nodes'][node]
    ):
        return
    delta.setdefault(graph, {}).setdefault('node_val', {}).setdefault(node, {})[key] = value


def setedge(delta, is_multigraph, graph, orig, dest, idx, exists):
    """Change a delta to say that an edge was created or deleted"""
    if is_multigraph(graph):
        delta.setdefault(graph, {}).setdefault('edges', {})\
            .setdefault(orig, {}).setdefault(dest, {})[idx] = bool(exists)
    else:
        delta.setdefault(graph, {}).setdefault('edges', {})\
            .setdefault(orig, {})[dest] = bool(exists)


def setedgeval(delta, is_multigraph, graph, orig, dest, idx, key, value):
    """Change a delta to say that an edge stat was set to a certain value"""
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


class ORM(object):
    """Instantiate this with the same string argument you'd use for a
    SQLAlchemy ``create_engine`` call. This will be your interface to
    allegedb.

    """
    node_cls = Node
    edge_cls = Edge
    query_engine_cls = QueryEngine
    illegal_graph_names = ['global']
    illegal_node_names = ['nodes', 'node_val', 'edges', 'edge_val']
    time = TimeSignalDescriptor()

    def _make_node(self, graph, node):
        return self.node_cls(graph, node)

    def _get_node(self, graph, node):
        key = (graph.name, node)
        if key in self._node_objs:
            return self._node_objs[key]
        if not self._node_exists(graph.name, node):
            self._exist_node(graph.name, node)
        ret = self._make_node(graph, node)
        self._node_objs[key] = ret
        return ret

    def _make_edge(self, graph, orig, dest, idx):
        return self.edge_cls(graph, orig, dest, idx)

    def _get_edge(self, graph, orig, dest, idx=0):
        key = (graph.name, orig, dest, idx)
        if key in self._edge_objs:
            return self._edge_objs[key]
        if not self._edge_exists(graph.name, orig, dest, idx):
            self._exist_edge(graph.name, orig, dest, idx)
        ret = self._make_edge(graph, orig, dest, idx)
        self._edge_objs[key] = ret
        return ret

    def plan(self):
        return PlanningContext(self)
    plan.__doc__ = PlanningContext.__doc__

    from contextlib import contextmanager

    @contextmanager
    def advancing(self):
        """A context manager for when time is moving forward one turn at a time.

        When used in LiSE, this means that the game is being simulated.
        It changes how the caching works, making it more efficient.

        """
        if self._forward:
            raise ValueError("Already advancing")
        self._forward = True
        yield
        self._forward = False

    @contextmanager
    def batch(self):
        """A context manager for when you're creating lots of state.

        Reads will be much slower in a batch, but writes will be faster.

        You *can* combine this with ``advancing`` but it isn't any faster.

        """
        if self._no_kc:
            raise ValueError("Already in a batch")
        self._no_kc = True
        yield
        self._no_kc = False

    def get_delta(self, branch, turn_from, tick_from, turn_to, tick_to):
        """Get a dictionary describing changes to all graphs.

        The keys are graph names. Their values are dictionaries of the graphs'
        attributes' new values, with ``None`` for deleted keys. Also in those graph
        dictionaries are special keys 'node_val' and 'edge_val' describing changes
        to node and edge attributes, and 'nodes' and 'edges' full of booleans
        indicating whether a node or edge exists.

        """
        from functools import partial
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
            evbranches = self._edge_val_cache.settings

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

        if branch in gvbranches and turn in gvbranches[branch]:
            for graph, key, value in gvbranches[branch][turn][tick_from:tick_to]:
                if graph in delta:
                    delta[graph][key] = value
                else:
                    delta[graph] = {key: value}

        if branch in nbranches and turn in nbranches[branch]:
            for graph, node, exists in nbranches[branch][turn][tick_from:tick_to]:
                delta.setdefault(graph, {}).setdefault('nodes', {})[node] = bool(exists)

        if branch in nvbranches and turn in nvbranches[branch]:
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
        if branch in ebranches and turn in ebranches[branch]:
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

        if branch in evbranches and turn in evbranches[branch]:
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
        from collections import defaultdict
        from .cache import Cache, NodesCache, EdgesCache
        self._global_cache = self.query._global_cache = {}
        self._node_objs = WeakValueDictionary()
        self._edge_objs = WeakValueDictionary()
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
        """Immediate children of a branch"""
        self._branches = {}
        """Start time, end time, and parent of each branch"""
        self._branch_parents = defaultdict(set)
        """Parents of a branch at any remove"""
        self._turn_end = defaultdict(lambda: 0)
        """Tick on which a (branch, turn) ends"""
        self._turn_end_plan = defaultdict(lambda: 0)
        """Tick on which a (branch, turn) ends, even if it hasn't been simulated"""
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
        self._planning = False
        self._forward = False
        self._no_kc = False
        if not hasattr(self, 'query'):
            self.query = self.query_engine_cls(
                dbstring, connect_args, alchemy,
                getattr(self, 'pack', None), getattr(self, 'unpack', None)
            )
        self.query.initdb()
        # in case this is the first startup
        self._otick = self._oturn = 0
        self._init_caches()
        for (branch, parent, parent_turn, parent_tick, end_turn, end_tick) in self.query.all_branches():
            self._branches[branch] = (parent, parent_turn, parent_tick, end_turn, end_tick)
            self._upd_branch_parentage(parent, branch)
        for (branch, turn, end_tick, plan_end_tick) in self.query.turns_dump():
            self._turn_end[branch, turn] = end_tick
            self._turn_end_plan[branch, turn] = plan_end_tick
        if 'trunk' not in self._branches:
            self._branches['trunk'] = None, 0, 0, 0, 0
        self._load_graphs()
        self._init_load(validate=validate)

    def _upd_branch_parentage(self, parent, child):
        self._childbranch[parent].add(child)
        self._branch_parents[child].add(parent)
        while parent in self._branches:
            parent, _, _, _, _ = self._branches[parent]
            self._branch_parents[child].add(parent)

    def _init_load(self, validate=False):
        if not hasattr(self, 'graph'):
            self.graph = self._graph_objs
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

    def _get_branch(self):
        return self._obranch

    def _set_branch(self, v):
        if self._planning:
            raise ValueError("Don't change branches while planning")
        curbranch, curturn, curtick = self.btt()
        if curbranch == v:
            self._otick = self._turn_end_plan[curbranch, curturn]
            return
        # make sure I'll end up within the revision range of the
        # destination branch
        if v != 'trunk' and v in self._branches:
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
        if v not in self._branches:
            # assumes the present turn in the parent branch has
            # been finalized.
            self.query.new_branch(v, curbranch, curturn, curtick)
            self._branches[v] = curbranch, curturn, curtick, curturn, curtick
            self._upd_branch_parentage(v, curbranch)
            self._turn_end_plan[v, curturn] = self._turn_end[v, curturn] = curtick
        self._obranch = v
        self._otick = self._turn_end_plan[v, curturn]

    # easier to override things this way
    @property
    def branch(self):
        return self._get_branch()

    @branch.setter
    def branch(self, v):
        self._set_branch(v)

    def _get_turn(self):
        return self._oturn

    def _set_turn(self, v):
        if v == self.turn:
            self._otick = self._turn_end_plan[tuple(self.time)]
            return
        if not isinstance(v, int):
            raise TypeError("turn must be an integer")
        # enforce the arrow of time, if it's in effect
        if self._forward and v < self._oturn:
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
        if not self._planning and v > turn_end:
            self._branches[branch] = parent, turn_start, tick_start, v, tick
        self._otick = tick
        self._oturn = v

    # easier to override things this way
    @property
    def turn(self):
        return self._get_turn()

    @turn.setter
    def turn(self, v):
        self._set_turn(v)

    def _get_tick(self):
        return self._otick

    def _set_tick(self, v):
        if not isinstance(v, int):
            raise TypeError("tick must be an integer")
        time = branch, turn = self._obranch, self._oturn
        # enforce the arrow of time, if it's in effect
        if self._forward and v < self._otick:
            raise ValueError("Can't time travel backward in a forward context")
        if v > self._turn_end_plan[time]:
            self._turn_end_plan[time] = v
        if not self._planning:
            if v > self._turn_end[time]:
                self._turn_end[time] = v
            parent, turn_start, tick_start, turn_end, tick_end = self._branches[branch]
            if turn == turn_end and v > tick_end:
                self._branches[branch] = parent, turn_start, tick_start, turn, v
        self._otick = v

    # easier to override things this way
    @property
    def tick(self):
        return self._get_tick()

    @tick.setter
    def tick(self, v):
        self._set_tick(v)

    def btt(self):
        """Return the branch, turn, and tick."""
        return self._obranch, self._oturn, self._otick

    def nbtt(self):
        """Increment the tick and return branch, turn, tick

        Unless we're viewing the past, in which case raise HistoryError.

        Idea is you use this when you want to advance time, which you
        can only do once per branch, turn, tick.

        """
        from .cache import HistoryError
        branch, turn, tick = self.btt()
        tick += 1
        if (branch, turn) in self._turn_end_plan:
            if tick > self._turn_end_plan[branch, turn]:
                self._turn_end_plan[branch, turn] = tick
            else:
                tick = self._turn_end_plan[branch, turn] + 1
        self._turn_end_plan[branch, turn] = tick
        if self._turn_end[branch, turn] > tick:
            raise HistoryError(
                "You're not at the end of turn {}. Go to tick {} to change things".format(
                    turn, self._turn_end[branch, turn]
                )
            )
        parent, turn_start, tick_start, turn_end, tick_end = self._branches[branch]
        if turn < turn_end or (
            turn == turn_end and tick < tick_end
        ):
            raise HistoryError(
                "You're in the past. Go to turn {}, tick {} to change things".format(turn_end, tick_end)
            )
        if not self._planning:
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

    def _iter_parent_btt(self, branch=None, turn=None, tick=None, *, stoptime=None):
        """Private use. Iterate over (branch, turn, tick), where the branch is
        a descendant of the previous (starting with whatever branch is
        presently active and ending at 'trunk'), and the turn is the
        latest revision in the branch that matters.

        Keyword ``stoptime`` may be a branch, in which case iteration will stop
        instead of proceeding into that branch's parent; or it may be a triple,
        ``(branch, turn, tick)``, in which case iteration will stop instead of
        yielding any time before that. The tick may be ``None``, in which case
        iteration will stop instead of yielding the turn.

        """
        branch = branch or self.branch
        trn = self.turn if turn is None else turn
        tck = self.tick if tick is None else tick
        yield branch, trn, tck
        stopbranches = set()
        if stoptime:
            if type(stoptime) is tuple:
                stopbranch = stoptime[0]
                stopbranches.add(stopbranch)
                stopbranches.update(self._branch_parents[stopbranch])
            else:
                stopbranch = stoptime
                stopbranches = self._branch_parents[stopbranch]
        _branches = self._branches
        while branch in _branches:
            # ``par`` is the parent branch;
            # ``(trn, tck)`` is when ``branch`` forked off from ``par``
            (branch, trn, tck, _, _) = _branches[branch]
            if branch in stopbranches and (
                trn < stoptime[1] or (
                    trn == stoptime[1] and (
                        stoptime[2] is None or tck <= stoptime[2]
                    )
                )
            ):
                return
            yield branch, trn, tck

    def _branch_descendants(self, branch=None):
        """Iterate over all branches immediately descended from the current
        one (or the given one, if available).

        """
        branch = branch or self.branch
        for (parent, (child, _, _, _, _)) in self._branches.items():
            if parent == branch:
                yield child

    def _node_exists(self, character, node):
        return self._nodes_cache.contains_entity(character, node, *self.btt())

    def _exist_node(self, character, node, exist=True):
        branch, turn, tick = self.nbtt()
        self.query.exist_node(
            character,
            node,
            branch,
            turn,
            tick,
            True
        )
        self._nodes_cache.store(character, node, branch, turn, tick, exist)

    def _edge_exists(self, character, orig, dest, idx=0):
        return self._edges_cache.contains_entity(
            character, orig, dest, idx, *self.btt()
        )

    def _exist_edge(
            self, character, orig, dest, idx=0, exist=True
    ):
        branch, turn, tick = self.nbtt()
        self.query.exist_edge(
            character,
            orig,
            dest,
            idx,
            branch,
            turn,
            tick,
            exist
        )
        self._edges_cache.store(
            character, orig, dest, idx, branch, turn, tick, exist
        )
