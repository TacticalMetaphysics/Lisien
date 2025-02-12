# This file is part of allegedb, an object relational mapper for versioned graphs.
# Copyright (C) Zachary Spector. public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""The main interface to the allegedb ORM"""

from contextlib import ContextDecorator, contextmanager
from functools import wraps
import gc
from itertools import pairwise, chain
from threading import RLock
from typing import (
	Callable,
	Dict,
	Any,
	Union,
	Tuple,
	Optional,
	List,
	Iterator,
	Set,
)

from blinker import Signal
import networkx as nx

from .cache import (
	KeyframeError,
	PickyDefaultDict,
	TurnEndDict,
	TurnEndPlanDict,
)
from .window import update_window, update_backward_window, WindowDict
from .graph import DiGraph, Node, Edge, GraphsMapping
from .query import (
	QueryEngine,
	TimeError,
)
from ..util import sort_set, Key, HistoricKeyError

"""Type hint for things lisien can use as keys

They have to be serializable using lisien's particular msgpack schema,
as well as hashable.

"""

Graph = DiGraph  # until I implement other graph types...

StatDict = Dict[Key, Any]
GraphValDict = Dict[Key, StatDict]
NodeValDict = Dict[Key, StatDict]
GraphNodeValDict = Dict[Key, NodeValDict]
EdgeValDict = Dict[Key, Dict[Key, StatDict]]
GraphEdgeValDict = Dict[Key, EdgeValDict]
DeltaDict = Dict[
	Key,
	Union[GraphValDict, GraphNodeValDict, GraphEdgeValDict, StatDict, None],
]
KeyframeTuple = Tuple[
	Key,
	str,
	int,
	int,
	GraphNodeValDict,
	GraphEdgeValDict,
	GraphValDict,
]
NodesDict = Dict[Key, bool]
GraphNodesDict = Dict[Key, NodesDict]
EdgesDict = Dict[Key, Dict[Key, bool]]
GraphEdgesDict = Dict[Key, EdgesDict]


def world_locked(fn: Callable) -> Callable:
	"""Decorator for functions that alter the world state

	They will hold a reentrant lock, preventing more than one function
	from mutating the world at a time.

	"""

	@wraps(fn)
	def lockedy(*args, **kwargs):
		with args[0].world_lock:
			return fn(*args, **kwargs)

	return lockedy


class GraphNameError(KeyError):
	"""For errors involving graphs' names"""


class OutOfTimelineError(ValueError):
	"""You tried to access a point in time that didn't happen"""

	@property
	def branch_from(self):
		return self.args[1]

	@property
	def turn_from(self):
		return self.args[2]

	@property
	def tick_from(self):
		return self.args[3]

	@property
	def branch_to(self):
		return self.args[4]

	@property
	def turn_to(self):
		return self.args[5]

	@property
	def tick_to(self):
		return self.args[6]


class PlanningContext(ContextDecorator):
	"""A context manager for 'hypothetical' edits.

	Start a block of code like::

		with orm.plan():
			...


	and any changes you make to the world state within that block will be
	'plans,' meaning that they are used as defaults. The world will
	obey your plan unless you make changes to the same entities outside
	the plan, in which case the world will obey those, and cancel any
	future plan.

	Plans are *not* canceled when concerned entities are deleted, although
	they are unlikely to be followed.

	New branches cannot be started within plans. The ``with orm.forward():``
	optimization is disabled within a ``with orm.plan():`` block, so
	consider another approach instead of making a very large plan.

	With ``reset=True`` (the default), when the plan block closes,
	the time will reset to when it began.

	"""

	__slots__ = ["orm", "id", "forward", "reset"]

	def __init__(self, orm: "ORM", reset=True):
		self.orm = orm
		if reset:
			self.reset = orm._btt()
		else:
			self.reset = None

	def __enter__(self):
		orm = self.orm
		if orm._planning:
			raise ValueError("Already planning")
		orm._planning = True
		branch, turn, tick = orm._btt()
		self.id = myid = orm._last_plan = orm._last_plan + 1
		self.forward = orm._forward
		if orm._forward:
			orm._forward = False
		orm._plans[myid] = branch, turn, tick
		orm._plans_uncommitted.append((myid, branch, turn, tick))
		orm._branches_plans[branch].add(myid)
		return myid

	def __exit__(self, exc_type, exc_val, exc_tb):
		self.orm._planning = False
		if self.reset is not None:
			self.orm._set_btt(*self.reset)
		if self.forward:
			self.orm._forward = True


class TimeSignal(Signal):
	"""Acts like a tuple of ``(branch, turn)`` for the most part.

	This is a ``Signal``. To set a function to be called whenever the
	branch or turn changes, pass it to my ``connect`` method.

	"""

	def __init__(self, engine: "ORM"):
		super().__init__()
		self.engine = engine
		self.branch = self.engine.branch
		self.turn = self.engine.turn

	def __iter__(self):
		yield self.branch
		yield self.turn

	def __len__(self):
		return 2

	def __getitem__(self, i: Union[str, int]) -> Union[str, int]:
		if i in ("branch", 0):
			return self.branch
		if i in ("turn", 1):
			return self.turn
		raise IndexError(i)

	def __setitem__(self, i: Union[str, int], v: Union[str, int]) -> None:
		if i in ("branch", 0):
			self.engine.branch = v
		elif i in ("turn", 1):
			self.engine.turn = v
		else:
			raise KeyError(
				"Can only set branch or turn. Set `Engine.tick` directly if you really want that."
			)

	def __str__(self):
		return str(tuple(self))

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

	def __get__(self, inst, cls):
		if not hasattr(inst, "_time_signal"):
			inst._time_signal = TimeSignal(inst)
		return inst._time_signal

	def __set__(self, inst: "ORM", val: Tuple[str, int]):
		if not hasattr(inst, "_time_signal"):
			inst._time_signal = TimeSignal(inst)
		sig = inst._time_signal
		branch_then, turn_then, tick_then = inst._btt()
		branch_now, turn_now = val
		if (branch_then, turn_then) == (branch_now, turn_now):
			return
		e = inst
		# enforce the arrow of time, if it's in effect
		if e._forward and not e._planning:
			if branch_now != branch_then:
				raise TimeError("Can't change branches in a forward context")
			if turn_now < turn_then:
				raise TimeError(
					"Can't time travel backward in a forward context"
				)
			if turn_now > turn_then + 1:
				raise TimeError("Can't skip turns in a forward context")
		# make sure I'll end up within the revision range of the
		# destination branch
		branches = e._branches

		if branch_now in branches:
			tick_now = e._turn_end_plan.setdefault(
				(branch_now, turn_now), tick_then
			)
			parent, turn_start, tick_start, turn_end, tick_end = branches[
				branch_now
			]
			if turn_now < turn_start:
				raise OutOfTimelineError(
					"The turn number {} "
					"occurs before the start of "
					"the branch {}".format(turn_now, branch_now),
					branch_then,
					turn_then,
					tick_then,
					branch_now,
					turn_now,
					tick_now,
				)
			if turn_now == turn_start and tick_now < tick_start:
				raise OutOfTimelineError(
					"The tick number {}"
					"on turn {} "
					"occurs before the start of "
					"the branch {}".format(tick_now, turn_now, branch_now),
					branch_then,
					turn_then,
					tick_then,
					branch_now,
					turn_now,
					tick_now,
				)
			if not e._planning and (
				turn_now > turn_end
				or (turn_now == turn_end and tick_now > tick_end)
			):
				branches[branch_now] = (
					parent,
					turn_start,
					tick_start,
					turn_now,
					tick_now,
				)
		else:
			tick_now = tick_then
			branches[branch_now] = (
				branch_then,
				turn_now,
				tick_now,
				turn_now,
				tick_now,
			)
			inst._turn_end_plan[branch_now, turn_now] = max(
				(inst._turn_end_plan[branch_now, turn_now], tick_now)
			)
			if not inst._planning:
				inst._branch_end[branch_now] = max(
					(inst._branch_end[branch_now], turn_now)
				)
				inst._turn_end[branch_now, turn_now] = max(
					(inst._turn_end[branch_now, turn_now], tick_now)
				)
			e.query.new_branch(branch_now, branch_then, turn_now, tick_now)
		e._obranch, e._oturn = val
		if not e._time_is_loaded(*val, tick_now):
			e._load_at(*val, tick_now)

		if not e._planning:
			if tick_now > e._turn_end[val]:
				e._turn_end[val] = tick_now
		e._otick = e._turn_end_plan[val] = tick_now
		sig.send(
			e,
			branch_then=branch_then,
			turn_then=turn_then,
			tick_then=tick_then,
			branch_now=branch_now,
			turn_now=turn_now,
			tick_now=tick_now,
		)


class ORM:
	"""Instantiate this with the same string argument you'd use for a
	SQLAlchemy ``create_engine`` call. This will be your interface to
	allegedb.

	"""

	node_cls = Node
	edge_cls = Edge
	query_engine_cls = QueryEngine
	illegal_graph_names = {"global"}
	illegal_node_names = {"nodes", "node_val", "edges", "edge_val"}
	time = TimeSignalDescriptor()

	def _graph_state_hash(
		self, nodes: NodeValDict, edges: EdgeValDict, vals: StatDict
	) -> bytes:
		from hashlib import blake2b

		qpac = self.query.pack

		if isinstance(qpac(" "), str):

			def pack(x):
				return qpac(x).encode()
		else:
			pack = qpac
		nodes_hash = 0
		for name, val in nodes.items():
			hash = blake2b(pack(name))
			hash.update(pack(val))
			nodes_hash ^= int.from_bytes(hash.digest(), "little")
		edges_hash = 0
		for orig, dests in edges.items():
			for dest, idxs in dests.items():
				for idx, val in idxs.items():
					hash = blake2b(pack(orig))
					hash.update(pack(dest))
					hash.update(pack(idx))
					hash.update(pack(val))
					edges_hash ^= int.from_bytes(hash.digest(), "little")
		val_hash = 0
		for key, val in vals.items():
			hash = blake2b(pack(key))
			hash.update(pack(val))
			val_hash ^= int.from_bytes(hash.digest(), "little")
		total_hash = blake2b(nodes_hash.to_bytes(64, "little"))
		total_hash.update(edges_hash.to_bytes(64, "little"))
		total_hash.update(val_hash.to_bytes(64, "little"))
		return total_hash.digest()

	def _kfhash(
		self,
		graphn: Key,
		branch: str,
		turn: int,
		tick: int,
		nodes: NodeValDict,
		edges: EdgeValDict,
		vals: StatDict,
	) -> bytes:
		"""Return a hash digest of a keyframe"""
		from hashlib import blake2b

		qpac = self.query.pack

		if isinstance(qpac(" "), str):

			def pack(x):
				return qpac(x).encode()
		else:
			pack = qpac
		total_hash = blake2b(pack(graphn))
		total_hash.update(pack(branch))
		total_hash.update(pack(turn))
		total_hash.update(pack(tick))
		total_hash.update(self._graph_state_hash(nodes, edges, vals))
		return total_hash.digest()

	def _make_node(self, graph: Key, node: Key):
		return self.node_cls(graph, node)

	def _get_node(self, graph: Union[Key, Graph], node: Key):
		node_objs, node_exists, make_node = self._get_node_stuff
		if type(graph) is str:
			graphn = graph
			graph = self.graph[graphn]
		else:
			graphn = graph.name
		key = (graphn, node)
		if key in node_objs:
			ret = node_objs[key]
			if ret._validate_node_type():
				return ret
			else:
				del node_objs[key]
		if not node_exists(graphn, node):
			raise KeyError("No such node: {} in {}".format(node, graphn))
		ret = make_node(graph, node)
		node_objs[key] = ret
		return ret

	def _make_edge(self, graph, orig, dest, idx):
		return self.edge_cls(graph, orig, dest, idx)

	def _get_edge(self, graph, orig, dest, idx=0):
		edge_objs, edge_exists, make_edge = self._get_edge_stuff
		if type(graph) is str:
			graphn = graph
			graph = self.graph[graphn]
		else:
			graphn = graph.name
		key = (graphn, orig, dest, idx)
		if key in edge_objs:
			return edge_objs[key]
		if not edge_exists(graphn, orig, dest, idx):
			raise KeyError(
				"No such edge: {}->{}[{}] in {}".format(
					orig, dest, idx, graphn
				)
			)
		ret = make_edge(graph, orig, dest, idx)
		edge_objs[key] = ret
		return ret

	def plan(self, reset=True) -> PlanningContext:
		return PlanningContext(self, reset)

	plan.__doc__ = PlanningContext.__doc__

	@contextmanager
	def advancing(self):
		"""A context manager for when time is moving forward one turn at a time.

		When used in lisien, this means that the game is being simulated.
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
			yield
			return
		self._no_kc = True
		gc_was_active = gc.isenabled()
		if gc_was_active:
			gc.disable()
		yield
		if gc_was_active:
			gc.enable()
			gc.collect()
		self._no_kc = False

	def _arrange_caches_at_time(
		self, _, *, branch: str, turn: int, tick: int
	) -> None:
		with self.world_lock:
			graphs = list(self.graph)
			for graph in graphs:
				graph_stats = self._graph_val_cache._get_keycache(
					(graph,), branch, turn, tick, forward=False
				)
				for stat in graph_stats:
					self._graph_val_cache._base_retrieve(
						(graph, stat, branch, turn, tick)
					)
				nodes = self._nodes_cache._get_keycache(
					(graph,), branch, turn, tick, forward=False
				)
				for node in nodes:
					self._nodes_cache._base_retrieve(
						(graph, node, branch, turn, tick)
					)
					node_stats = self._node_val_cache._get_keycache(
						(graph, node), branch, turn, tick, forward=False
					)
					for stat in node_stats:
						self._node_val_cache._base_retrieve(
							(graph, node, stat, branch, turn, tick)
						)
					dests = self._edges_cache._get_destcache(
						graph, node, branch, turn, tick, forward=False
					)
					for dest in dests:
						self._edges_cache._base_retrieve(
							(graph, node, dest, branch, turn, tick)
						)
						edge_stats = self._edge_val_cache._get_keycache(
							(graph, node, dest),
							branch,
							turn,
							tick,
							forward=False,
						)
						for stat in edge_stats:
							self._edge_val_cache._base_retrieve(
								(graph, node, dest, stat, branch, turn, tick)
							)

	def _get_branch_delta(
		self,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> DeltaDict:
		"""Get a dictionary describing changes to all graphs.

		The keys are graph names. Their values are dictionaries of the
		graphs' attributes' new values, with ``None`` for deleted keys. Also
		in those graph dictionaries are special keys 'node_val' and
		'edge_val' describing changes to node and edge attributes,
		and 'nodes' and 'edges' full of booleans indicating whether a node
		or edge exists.

		"""

		def setgraph(delta: DeltaDict, _: None, graph: Key, val: Any) -> None:
			"""Change a delta to say that a graph was deleted or not"""
			if val in (None, "Deleted"):
				delta[graph] = None
			elif graph in delta and delta[graph] is None:
				delta[graph] = {}

		def setgraphval(
			delta: DeltaDict, graph: Key, key: Key, val: Any
		) -> None:
			"""Change a delta to say that a graph stat was set to a certain value"""
			if (graphstat := delta.setdefault(graph, {})) is not None:
				graphstat[key] = val

		def setnode(
			delta: DeltaDict, graph: Key, node: Key, exists: Optional[bool]
		) -> None:
			"""Change a delta to say that a node was created or deleted"""
			if (graphstat := delta.setdefault(graph, {})) is not None:
				graphstat.setdefault("nodes", {})[node] = bool(exists)

		def setnodeval(
			delta: DeltaDict, graph: Key, node: Key, key: Key, value: Any
		) -> None:
			"""Change a delta to say that a node stat was set to a certain value"""
			if (graphstat := delta.setdefault(graph, {})) is not None:
				if (
					graph in delta
					and "nodes" in delta[graph]
					and node in delta[graph]["nodes"]
					and not delta[graph]["nodes"][node]
				):
					return
				graphstat.setdefault("node_val", {}).setdefault(node, {})[
					key
				] = value

		def setedge(
			delta: DeltaDict,
			is_multigraph: Callable,
			graph: Key,
			orig: Key,
			dest: Key,
			idx: int,
			exists: Optional[bool],
		) -> None:
			"""Change a delta to say that an edge was created or deleted"""
			if (graphstat := delta.setdefault(graph, {})) is not None:
				if is_multigraph(graph):
					raise NotImplementedError("Only digraphs for now")
				if "edges" in graphstat:
					es = graphstat["edges"]
					if orig in es:
						es[orig][dest] = exists
					else:
						es[orig] = {dest: exists}
				else:
					graphstat["edges"] = {orig: {dest: exists}}

		def setedgeval(
			delta: DeltaDict,
			is_multigraph: Callable,
			graph: Key,
			orig: Key,
			dest: Key,
			idx: int,
			key: Key,
			value: Any,
		) -> None:
			"""Change a delta to say that an edge stat was set to a certain value"""
			if (graphstat := delta.setdefault(graph, {})) is not None:
				if is_multigraph(graph):
					raise NotImplementedError("Only digraphs for now")
				if (
					"edges" in graphstat
					and orig in graphstat["edges"]
					and dest in graphstat["edges"][orig]
					and not graphstat["edges"][orig][dest]
				):
					return
				graphstat.setdefault("edge_val", {}).setdefault(
					orig, {}
				).setdefault(dest, {})[key] = value

		from functools import partial

		if turn_from == turn_to:
			return self._get_turn_delta(branch, turn_from, tick_from, tick_to)
		delta = {}
		graph_objs = self._graph_objs
		if turn_to < turn_from:
			updater = partial(
				update_backward_window, turn_from, tick_from, turn_to, tick_to
			)
			gbranches = self._graph_cache.presettings
			gvbranches = self._graph_val_cache.presettings
			nbranches = self._nodes_cache.presettings
			nvbranches = self._node_val_cache.presettings
			ebranches = self._edges_cache.presettings
			evbranches = self._edge_val_cache.presettings
			tick_to += 1
		else:
			updater = partial(
				update_window, turn_from, tick_from, turn_to, tick_to
			)
			gbranches = self._graph_cache.settings
			gvbranches = self._graph_val_cache.settings
			nbranches = self._nodes_cache.settings
			nvbranches = self._node_val_cache.settings
			ebranches = self._edges_cache.settings
			evbranches = self._edge_val_cache.settings

		if branch in gbranches:
			updater(partial(setgraph, delta), gbranches[branch])

		if branch in gvbranches:
			updater(partial(setgraphval, delta), gvbranches[branch])

		if branch in nbranches:
			updater(partial(setnode, delta), nbranches[branch])

		if branch in nvbranches:
			updater(partial(setnodeval, delta), nvbranches[branch])

		if branch in ebranches:
			updater(
				partial(
					setedge, delta, lambda g: graph_objs[g].is_multigraph()
				),
				ebranches[branch],
			)

		if branch in evbranches:
			updater(
				partial(
					setedgeval, delta, lambda g: graph_objs[g].is_multigraph()
				),
				evbranches[branch],
			)

		return delta

	def _get_turn_delta(
		self,
		branch: str = None,
		turn: int = None,
		tick_from=0,
		tick_to: int = None,
	) -> DeltaDict:
		"""Get a dictionary describing changes made on a given turn.

		If ``tick_to`` is not supplied, report all changes after ``tick_from``
		(default 0).

		The keys are graph names. Their values are dictionaries of the
		graphs' attributes' new values, with ``None`` for deleted keys. Also
		in those graph dictionaries are special keys 'node_val' and
		'edge_val' describing changes to node and edge attributes,
		and 'nodes' and 'edges' full of booleans indicating whether a node
		or edge exists.

		:arg branch: A branch of history; defaults to the current branch
		:arg turn: The turn in the branch; defaults to the current turn
		:arg tick_from: Starting tick; defaults to 0

		"""
		branch = branch or self.branch
		turn = turn or self.turn
		tick_to = tick_to or self.tick
		delta = {}
		if tick_from < tick_to:
			gbranches = self._graph_cache.settings
			gvbranches = self._graph_val_cache.settings
			nbranches = self._nodes_cache.settings
			nvbranches = self._node_val_cache.settings
			ebranches = self._edges_cache.settings
			evbranches = self._edge_val_cache.settings
			tick_to += 1
		else:
			gbranches = self._graph_cache.presettings
			gvbranches = self._graph_val_cache.presettings
			nbranches = self._nodes_cache.presettings
			nvbranches = self._node_val_cache.presettings
			ebranches = self._edges_cache.presettings
			evbranches = self._edge_val_cache.presettings

		if branch in gbranches and turn in gbranches[branch]:
			for _, graph, typ in gbranches[branch][turn][tick_from:tick_to]:
				# typ may be None if the graph was never deleted, but we're
				# traveling back to before it was created
				if typ in ("Deleted", None):
					delta[graph] = None
				else:
					delta[graph] = {}

		if branch in gvbranches and turn in gvbranches[branch]:
			for graph, key, value in gvbranches[branch][turn][
				tick_from:tick_to
			]:
				if graph in delta:
					if delta[graph] is None:
						continue
					delta[graph][key] = value
				else:
					delta[graph] = {key: value}

		if branch in nbranches and turn in nbranches[branch]:
			for graph, node, exists in nbranches[branch][turn][
				tick_from:tick_to
			]:
				if graph in delta and delta[graph] is None:
					continue
				delta.setdefault(graph, {}).setdefault("nodes", {})[node] = (
					bool(exists)
				)

		if branch in nvbranches and turn in nvbranches[branch]:
			for graph, node, key, value in nvbranches[branch][turn][
				tick_from:tick_to
			]:
				if graph in delta and (
					delta[graph] is None
					or (
						"nodes" in delta[graph]
						and node in delta[graph]["nodes"]
						and not delta[graph]["nodes"][node]
					)
				):
					continue
				nodevd = delta.setdefault(graph, {}).setdefault("node_val", {})
				if node in nodevd:
					nodevd[node][key] = value
				else:
					nodevd[node] = {key: value}

		graph_objs = self._graph_objs
		if branch in ebranches and turn in ebranches[branch]:
			for graph, orig, dest, idx, exists in ebranches[branch][turn][
				tick_from:tick_to
			]:
				if graph_objs[graph].is_multigraph():
					if graph in delta and (
						delta[graph] is None
						or (
							"edges" in delta[graph]
							and orig in delta[graph]["edges"]
							and dest in delta[graph]["edges"][orig]
							and idx in delta[graph]["edges"][orig][dest]
							and not delta[graph]["edges"][orig][dest][idx]
						)
					):
						continue
					delta.setdefault(graph, {}).setdefault(
						"edges", {}
					).setdefault(orig, {})[dest] = bool(exists)
				else:
					if graph in delta and (
						delta[graph] is None
						or (
							"edges" in delta[graph]
							and orig in delta[graph]["edges"]
							and dest in delta[graph]["edges"][orig]
							and not delta[graph]["edges"][orig][dest]
						)
					):
						continue
					delta.setdefault(graph, {}).setdefault(
						"edges", {}
					).setdefault(orig, {})[dest] = bool(exists)

		if branch in evbranches and turn in evbranches[branch]:
			for graph, orig, dest, idx, key, value in evbranches[branch][turn][
				tick_from:tick_to
			]:
				if graph in delta and delta[graph] is None:
					continue
				edgevd = (
					delta.setdefault(graph, {})
					.setdefault("edge_val", {})
					.setdefault(orig, {})
					.setdefault(dest, {})
				)
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
		from .cache import EntitylessCache, Cache, NodesCache, EdgesCache

		node_cls = self.node_cls
		edge_cls = self.edge_cls
		self._where_cached = defaultdict(list)
		self._node_objs = node_objs = {}
		self._get_node_stuff: Tuple[
			dict, Callable[[Key, Key], bool], Callable[[Key, Key], node_cls]
		] = (node_objs, self._node_exists, self._make_node)
		self._edge_objs = edge_objs = {}
		self._get_edge_stuff: Tuple[
			dict,
			Callable[[Key, Key, Key, int], bool],
			Callable[[Key, Key, Key, int], edge_cls],
		] = (edge_objs, self._edge_exists, self._make_edge)
		self._childbranch: Dict[str, Set[str]] = defaultdict(set)
		"""Immediate children of a branch"""
		self._branches: Dict[
			str, Tuple[Optional[str], int, int, int, int]
		] = {}
		"""Parent, start time, and end time of each branch. Includes plans."""
		self._branch_parents: Dict[str, Set[str]] = defaultdict(set)
		"""Parents of a branch at any remove"""
		self._turn_end: Dict[Tuple[str, int], int] = TurnEndDict()
		self._turn_end_plan: Dict[Tuple[str, int], int] = TurnEndPlanDict()
		self._turn_end_plan.other_d = self._turn_end
		self._turn_end.other_d = self._turn_end_plan
		self._branch_end: Dict[str, int] = defaultdict(lambda: 0)
		"""Turn on which a branch ends, not including plans"""
		self._graph_objs = {}
		self._plans: Dict[int, Tuple[str, int, int]] = {}
		self._branches_plans: Dict[str, Set[int]] = defaultdict(set)
		self._plan_ticks: Dict[int, Dict[int, List[int]]] = defaultdict(
			lambda: defaultdict(list)
		)
		self._time_plan: Dict[int, Tuple[str, int, int]] = {}
		self._plans_uncommitted: List[Tuple[int, str, int, int]] = []
		self._plan_ticks_uncommitted: List[Tuple[int, int, int]] = []
		self._graph_cache = EntitylessCache(self, name="graph_cache")
		self._graph_val_cache = Cache(self, name="graph_val_cache")
		self._nodes_cache = NodesCache(self)
		self._edges_cache = EdgesCache(self)
		self._node_val_cache = Cache(self, name="node_val_cache")
		self._edge_val_cache = Cache(self, name="edge_val_cache")
		self._caches = [
			self._graph_val_cache,
			self._nodes_cache,
			self._edges_cache,
			self._node_val_cache,
			self._edge_val_cache,
		]

	def _get_keyframe(
		self, branch: str, turn: int, tick: int, copy=True, silent=False
	):
		"""Load the keyframe if it's not loaded, and return it"""
		if (branch, turn, tick) in self._keyframes_loaded:
			return self._get_kf(branch, turn, tick, copy=copy)
		with (
			self.batch()
		):  # so that iter_keys doesn't try fetching the kf we're about to make
			keyframe_graphs = list(
				self.query.get_all_keyframe_graphs(branch, turn, tick)
			)
			self._graph_cache.set_keyframe(
				branch,
				turn,
				tick,
				{graph: "DiGraph" for (graph, _, _, _) in keyframe_graphs},
			)
			for (
				graph,
				nodes,
				edges,
				graph_val,
			) in keyframe_graphs:
				self._snap_keyframe_de_novo_graph(
					graph, branch, turn, tick, nodes, edges, graph_val
				)
		self._updload(branch, turn, tick)
		if branch in self._keyframes_dict:
			if turn in self._keyframes_dict[branch]:
				self._keyframes_dict[branch][turn].add(tick)
			else:
				self._keyframes_dict[branch][turn] = {tick}
		else:
			self._keyframes_dict[branch] = {turn: {tick}}
		self._keyframes_times.add((branch, turn, tick))
		self._keyframes_loaded.add((branch, turn, tick))
		if not silent:
			return self._get_kf(branch, turn, tick, copy=copy)

	def _load_graphs(self):
		self.graph = GraphsMapping(self)
		for graph, branch, turn, tick, typ in self.query.graphs_dump():
			self._graph_cache.store(
				graph, branch, turn, tick, (typ if typ != "Deleted" else None)
			)
			if typ not in {"DiGraph", "Deleted"}:
				raise NotImplementedError("Only DiGraph for now")
			# still create object for deleted graphs, in case you time travel
			# to when they existed
			self._graph_objs[graph] = DiGraph(self, graph)

	def _has_graph(self, graph, branch=None, turn=None, tick=None):
		if branch is None:
			branch = self.branch
		if turn is None:
			turn = self.turn
		if tick is None:
			tick = self.tick
		try:
			return (
				self._graph_cache.retrieve(graph, branch, turn, tick)
				!= "Deleted"
			)
		except KeyError:
			return False

	def __init__(
		self,
		dbstring,
		clear=False,
		connect_args: dict = None,
		main_branch=None,
		enforce_end_of_time=False,
	):
		"""Make a SQLAlchemy engine and begin a transaction

		:arg dbstring: rfc1738 URL for a database connection.

		:arg connect_args: Dictionary of
		keyword arguments to be used for the database connection.

		"""
		self.world_lock = RLock()
		connect_args = connect_args or {}
		self._planning = False
		self._forward = False
		self._no_kc = False
		self._enforce_end_of_time = enforce_end_of_time
		# in case this is the first startup
		self._obranch = main_branch or "trunk"
		self._otick = self._oturn = 0
		self._init_caches()
		if hasattr(self, "_post_init_cache_hook"):
			self._post_init_cache_hook()
		if not hasattr(self, "query"):
			self.query = self.query_engine_cls(
				dbstring,
				connect_args,
				getattr(self, "pack", None),
				getattr(self, "unpack", None),
			)
		if clear:
			self.query.truncate_all()
		self._edge_val_cache.setdb = self.query.edge_val_set
		self._edge_val_cache.deldb = self.query.edge_val_del_time
		self._node_val_cache.setdb = self.query.node_val_set
		self._node_val_cache.deldb = self.query.node_val_del_time
		self._edges_cache.setdb = self.query.exist_edge
		self._edges_cache.deldb = self.query.edges_del_time
		self._nodes_cache.setdb = self.query.exist_node
		self._nodes_cache.deldb = self.query.nodes_del_time
		self._graph_val_cache.setdb = self.query.graph_val_set
		self._graph_val_cache.deldb = self.query.graph_val_del_time
		self._keyframes_list = []
		self._keyframes_dict = PickyDefaultDict(WindowDict)
		self._keyframes_times = set()
		self._keyframes_loaded = set()
		self.query.initdb()
		if main_branch is not None:
			self.query.globl["main_branch"] = main_branch
		elif "main_branch" not in self.query.globl:
			main_branch = self.query.globl["main_branch"] = "trunk"
		else:
			main_branch = self.query.globl["main_branch"]
		self._obranch = self.query.get_branch()
		self._oturn = self.query.get_turn()
		self._otick = self.query.get_tick()
		for (
			branch,
			parent,
			parent_turn,
			parent_tick,
			end_turn,
			end_tick,
		) in self.query.all_branches():
			self._branches[branch] = (
				parent,
				parent_turn,
				parent_tick,
				end_turn,
				end_tick,
			)
			self._upd_branch_parentage(parent, branch)
		for branch, turn, end_tick, plan_end_tick in self.query.turns_dump():
			self._turn_end_plan[branch, turn] = max(
				(self._turn_end_plan[branch, turn], plan_end_tick)
			)
		if main_branch not in self._branches:
			self._branches[main_branch] = None, 0, 0, 0, 0
		self._nbtt_stuff = (
			self._btt,
			self._branch_end,
			self._turn_end_plan,
			self._turn_end,
			self._plan_ticks,
			self._plan_ticks_uncommitted,
			self._time_plan,
			self._branches,
		)
		self._node_exists_stuff: Tuple[
			Callable[[Tuple[Key, Key, str, int, int]], Any],
			Callable[[], Tuple[str, int, int]],
		] = (self._nodes_cache._base_retrieve, self._btt)
		self._exist_node_stuff: Tuple[
			Callable[[], Tuple[str, int, int]],
			Callable[[Key, Key, str, int, int, bool], None],
			Callable[[Key, Key, str, int, int, Any], None],
		] = (self._nbtt, self.query.exist_node, self._nodes_cache.store)
		self._edge_exists_stuff: Tuple[
			Callable[[Tuple[Key, Key, Key, int, str, int, int]], bool],
			Callable[[], Tuple[str, int, int]],
		] = (self._edges_cache._base_retrieve, self._btt)
		self._exist_edge_stuff: Tuple[
			Callable[[], Tuple[str, int, int]],
			Callable[[Key, Key, Key, int, str, int, int, bool], None],
			Callable[[Key, Key, Key, int, str, int, int, Any], None],
		] = (self._nbtt, self.query.exist_edge, self._edges_cache.store)
		self._load_graphs()
		assert hasattr(self, "graph")
		self._loaded: Dict[
			str, Tuple[int, int, int, int]
		] = {}  # branch: (turn_from, tick_from, turn_to, tick_to)
		self._load_plans()
		self._load_at(*self._btt())

	def _get_kf(
		self, branch: str, turn: int, tick: int, copy=True
	) -> Dict[
		Key,
		Union[
			GraphNodesDict,
			GraphNodeValDict,
			GraphEdgesDict,
			GraphEdgeValDict,
			GraphValDict,
		],
	]:
		"""Get a keyframe that's already in memory"""
		assert (branch, turn, tick) in self._keyframes_loaded
		graph_val: GraphValDict = {}
		nodes: GraphNodesDict = {}
		node_val: GraphNodeValDict = {}
		edges: GraphEdgesDict = {}
		edge_val: GraphEdgeValDict = {}
		ret = {
			"graph_val": graph_val,
			"nodes": nodes,
			"node_val": node_val,
			"edges": edges,
			"edge_val": edge_val,
		}
		for graph in self._graph_cache.iter_keys(branch, turn, tick):
			try:
				self._graph_cache.retrieve(graph, branch, turn, tick)
			except KeyError:
				continue
			graph_val[graph] = {}
		for k in graph_val:
			try:
				graph_val[k] = self._graph_val_cache.get_keyframe(
					(k,), branch, turn, tick, copy
				)
			except KeyframeError:
				pass
			try:
				nodes[k] = self._nodes_cache.get_keyframe(
					(k,), branch, turn, tick, copy
				)
			except KeyframeError:
				pass
		for graph, node in self._node_val_cache.keyframe:
			try:
				nvv: StatDict = self._node_val_cache.get_keyframe(
					(graph, node), branch, turn, tick, copy
				)
			except KeyframeError:  # node not present in this keyframe
				continue
			if graph in node_val:
				node_val[graph][node] = nvv
			else:
				node_val[graph] = {node: nvv}
		for graph, orig, dest in self._edges_cache.keyframe:
			try:
				idx_ex = self._edges_cache.get_keyframe(
					(graph, orig, dest), branch, turn, tick, copy
				)
			except KeyframeError:  # edge not present in this keyframe
				continue
			assert idx_ex.keys() == {0}, (
				"Not doing edge indexes until multigraphs come back"
			)
			assert idx_ex[0], "Stored a keyframe for a nonexistent edge"
			if graph in edges:
				if orig in edges[graph]:
					edges[graph][orig][dest] = True
				else:
					edges[graph][orig] = {dest: True}
			else:
				edges[graph] = {orig: {dest: True}}
		for graph, orig, dest, idx in self._edge_val_cache.keyframe:
			assert idx == 0, (
				"Not doing idx other than 0 until multigraphs come back"
			)
			try:
				evv = self._edge_val_cache.get_keyframe(
					(graph, orig, dest, idx), branch, turn, tick, copy
				)
			except KeyframeError:  # edge not present in this keyframe
				continue
			if graph in edge_val:
				if orig in edge_val[graph]:
					edge_val[graph][orig][dest] = evv
				else:
					edge_val[graph][orig] = {dest: evv}
			else:
				edge_val[graph] = {orig: {dest: evv}}
		return ret

	def _load_plans(self) -> None:
		keyframes_list = self._keyframes_list
		keyframes_dict = self._keyframes_dict
		keyframes_times = self._keyframes_times
		for branch, turn, tick in self.query.keyframes_dump():
			if branch not in keyframes_dict:
				keyframes_dict[branch] = {turn: {tick}}
			else:
				keyframes_dict_branch = keyframes_dict[branch]
				if turn not in keyframes_dict_branch:
					keyframes_dict_branch[turn] = {tick}
				else:
					keyframes_dict_branch[turn].add(tick)
			keyframes_times.add((branch, turn, tick))

		keyframes_list.extend(self.query.keyframes_graphs())

		last_plan = -1
		plans = self._plans
		branches_plans = self._branches_plans
		for plan, branch, turn, tick in self.query.plans_dump():
			plans[plan] = branch, turn, tick
			branches_plans[branch].add(plan)
			if plan > last_plan:
				last_plan = plan
		self._last_plan = last_plan
		plan_ticks = self._plan_ticks
		time_plan = self._time_plan
		turn_end_plan = self._turn_end_plan
		branches = self._branches
		for plan, turn, tick in self.query.plan_ticks_dump():
			plan_ticks[plan][turn].append(tick)
			branch = plans[plan][0]
			parent, turn_start, tick_start, turn_end, tick_end = branches[
				branch
			]
			if (turn, tick) > (turn_end, tick_end):
				branches[branch] = parent, turn_start, tick_start, turn, tick
			turn_end_plan[branch, turn] = max(
				(turn_end_plan[branch, turn], tick)
			)
			time_plan[branch, turn, tick] = plan

	def _upd_branch_parentage(self, parent: str, child: str) -> None:
		self._childbranch[parent].add(child)
		self._branch_parents[child].add(parent)
		while parent in self._branches:
			parent, _, _, _, _ = self._branches[parent]
			self._branch_parents[child].add(parent)

	def _snap_keyframe_de_novo(
		self, branch: str, turn: int, tick: int
	) -> None:
		kfl = self._keyframes_list
		kfd = self._keyframes_dict
		kfs = self._keyframes_times
		kfsl = self._keyframes_loaded
		inskf = self.query.keyframe_graph_insert
		was = self._btt()
		self._set_btt(branch, turn, tick)
		self.query.keyframe_insert(branch, turn, tick)
		for graphn in self._graph_cache.iter_keys(branch, turn, tick):
			graph = self.graph[graphn]
			nodes = graph._nodes_state()
			edges = graph._edges_state()
			val = graph._val_state()
			self._snap_keyframe_de_novo_graph(
				graphn, branch, turn, tick, nodes, edges, val
			)
			inskf(graphn, branch, turn, tick, nodes, edges, val)
			kfl.append((graphn, branch, turn, tick))
		if branch not in kfd:
			kfd[branch] = {
				turn: {
					tick,
				}
			}
		elif turn not in kfd[branch]:
			kfd[branch][turn] = {
				tick,
			}
		else:
			kfd[branch][turn].add(tick)
		kfs.add((branch, turn, tick))
		kfsl.add((branch, turn, tick))
		self._set_btt(*was)

	def _snap_keyframe_de_novo_graph(
		self,
		graph: Key,
		branch: str,
		turn: int,
		tick: int,
		nodes: NodeValDict,
		edges: EdgeValDict,
		graph_val: StatDict,
	) -> None:
		try:
			graphs_keyframe = self._graph_cache.get_keyframe(
				branch, turn, tick
			)
		except KeyframeError:
			graphs_keyframe = {
				g: "DiGraph"
				for g in self._graph_cache.iter_keys(branch, turn, tick)
			}
		graphs_keyframe[graph] = "DiGraph"
		self._graph_cache.set_keyframe(branch, turn, tick, graphs_keyframe)
		self._graph_cache.keycache.clear()
		self._nodes_cache.set_keyframe(
			(graph,), branch, turn, tick, {node: True for node in nodes}
		)
		nvc = self._node_val_cache
		for node, vals in nodes.items():
			nvc.set_keyframe((graph, node), branch, turn, tick, vals)
		ec = self._edges_cache
		evc = self._edge_val_cache
		for orig, dests in edges.items():
			for dest, vals in dests.items():
				ec.set_keyframe(
					(graph, orig, dest), branch, turn, tick, {0: True}
				)
				evc.set_keyframe(
					(graph, orig, dest, 0), branch, turn, tick, vals
				)
		self._graph_val_cache.set_keyframe(
			(graph,), branch, turn, tick, graph_val
		)
		if (branch, turn, tick) not in self._keyframes_times:
			self._keyframes_times.add((branch, turn, tick))
			self._keyframes_loaded.add((branch, turn, tick))
			if branch in self._keyframes_dict:
				turns = self._keyframes_dict[branch]
				if turn in turns:
					turns[turn].add(tick)
				else:
					turns[turn] = {tick}
			else:
				self._keyframes_dict[branch] = {turn: {tick}}
			self._keyframes_list.append((graph, branch, turn, tick))

	def _copy_kf(self, branch_from, branch_to, turn, tick):
		"""Copy a keyframe from one branch to another

		This aliases the data, rather than really copying. Keyframes don't
		change, so it should be fine.

		"""
		try:
			graph_keyframe = self._graph_cache.get_keyframe(
				branch_from, turn, tick
			)
		except KeyframeError:
			graph_keyframe = {
				graph: self._graph_cache.retrieve(
					graph, branch_from, turn, tick
				)
				for graph in self._graph_cache.iter_entities(
					branch_from, turn, tick
				)
			}
		self._graph_cache.set_keyframe(
			branch_to,
			turn,
			tick,
			graph_keyframe,
		)
		for graph in self._graph_cache.iter_keys(branch_to, turn, tick):
			try:
				graph_vals = self._graph_val_cache.get_keyframe(
					(graph,), branch_from, turn, tick, copy=False
				)
			except KeyframeError:
				graph_vals = {}
			self._graph_val_cache.set_keyframe(
				(graph,), branch_to, turn, tick, graph_vals
			)
			try:
				nodes = self._nodes_cache.get_keyframe(
					(graph,), branch_from, turn, tick, copy=False
				)
			except KeyframeError:
				nodes = {}
			self._nodes_cache.set_keyframe(
				(graph,), branch_to, turn, tick, nodes
			)
			node_vals = {}
			edge_vals = {}
			for node in nodes:
				node_val = self._node_val_cache.get_keyframe(
					(graph, node), branch_from, turn, tick, copy=False
				)
				self._node_val_cache.set_keyframe(
					(graph, node), branch_to, turn, tick, node_val
				)
				node_vals[node] = node_val
				for dest in self._edges_cache.iter_successors(
					graph, node, branch_from, turn, tick
				):
					self._edges_cache.set_keyframe(
						(graph, node, dest),
						branch_to,
						turn,
						tick,
						self._edges_cache.get_keyframe(
							(graph, node, dest),
							branch_from,
							turn,
							tick,
							copy=False,
						),
					)
					evkf = self._edge_val_cache.get_keyframe(
						(graph, node, dest, 0),
						branch_from,
						turn,
						tick,
						copy=False,
					)
					self._edge_val_cache.set_keyframe(
						(graph, node, dest, 0), branch_to, turn, tick, evkf
					)
					if node in edge_vals:
						edge_vals[node][dest] = evkf
					else:
						edge_vals[node] = {dest: evkf}
			self.query.keyframe_graph_insert(
				graph,
				branch_to,
				turn,
				tick,
				node_vals,
				edge_vals,
				graph_vals,
			)
		self._keyframes_list.append((branch_to, turn, tick))
		self._keyframes_times.add((branch_to, turn, tick))
		self._keyframes_loaded.add((branch_to, turn, tick))
		if branch_to in self._keyframes_dict:
			kdb = self._keyframes_dict[branch_to]
			if turn in kdb:
				kdb[turn].add(tick)
			else:
				kdb[turn] = {tick}
		else:
			self._keyframes_dict[branch_to] = {turn: {tick}}
		self._nudge_loaded(branch_to, turn, tick)

	def _snap_keyframe_from_delta(
		self,
		then: Tuple[str, int, int],
		now: Tuple[str, int, int],
		delta: DeltaDict,
	) -> None:
		# may mutate delta
		assert then[0] == now[0]
		whens = [now]
		kfl = self._keyframes_list
		kfd = self._keyframes_dict
		kfs = self._keyframes_times
		kfsl = self._keyframes_loaded
		kfs.add(now)
		kfsl.add(now)
		self.query.keyframe_insert(*now)
		branch, turn, tick = now
		if branch not in kfd:
			kfd[branch] = {
				turn: {
					tick,
				}
			}
		elif turn not in kfd[branch]:
			kfd[branch][turn] = {
				tick,
			}
		else:
			kfd[branch][turn].add(tick)
		inskf = self.query.keyframe_graph_insert
		keyframe = self._get_keyframe(*then)
		graph_val_keyframe: GraphValDict = keyframe["graph_val"]
		nodes_keyframe: GraphNodesDict = keyframe["nodes"]
		node_val_keyframe: GraphNodeValDict = keyframe["node_val"]
		edges_keyframe: GraphEdgesDict = keyframe["edges"]
		edge_val_keyframe: GraphEdgeValDict = keyframe["edge_val"]
		graphs_keyframe = {g: "DiGraph" for g in graph_val_keyframe}
		for graph in (
			graph_val_keyframe.keys() | delta.keys()
		) - self.illegal_graph_names:
			# apply the delta to the keyframes, then save the keyframes back
			# into the caches, and possibly copy them to another branch as well
			deltg = delta.get(graph, {})
			if deltg is None:
				del graphs_keyframe[graph]
				continue
			elif graph not in graphs_keyframe:
				graphs_keyframe[graph] = "DiGraph"
			nkg: NodesDict = nodes_keyframe.setdefault(graph, {})
			nvkg: NodeValDict = node_val_keyframe.setdefault(graph, {})
			ekg: EdgesDict = edges_keyframe.setdefault(graph, {})
			evkg: EdgeValDict = edge_val_keyframe.setdefault(graph, {})
			if deltg is not None and "nodes" in deltg:
				dn = deltg.pop("nodes")
				for node, exists in dn.items():
					if node in nkg:
						if not exists:
							del nkg[node]
							if node in nvkg:
								del nvkg[node]
					elif exists:
						nkg[node] = True
			self._nodes_cache.set_keyframe((graph,), *now, nkg)
			for node, ex in nkg.items():
				if ex and node not in nvkg:
					nvkg[node] = {}
			if deltg is not None and "node_val" in deltg:
				dnv = deltg.pop("node_val")
				for node, value in dnv.items():
					node: Key
					value: StatDict
					if node in nvkg:
						nvgn = nvkg[node]
						for k, v in value.items():
							if v is None:
								if k in nvgn:
									del nvgn[k]
							else:
								nvgn[k] = v
					else:
						nvkg[node] = value
			for node, val in keyframe["node_val"][graph].items():
				val: StatDict
				self._node_val_cache.set_keyframe((graph, node), *now, val)
			if deltg is not None and "edges" in deltg:
				dge = deltg.pop("edges")
				for orig, dests in dge.items():
					for dest, exists in dests.items():
						if orig in ekg:
							if exists:
								ekg[orig][dest] = exists
							else:
								if dest in ekg[orig]:
									del ekg[orig][dest]
								if orig in evkg and dest in evkg[orig]:
									del evkg[orig][dest]
						elif exists:
							ekg[orig] = {dest: exists}
			if graph in edges_keyframe:
				if graph not in edge_val_keyframe:
					edge_val_keyframe[graph] = {}
				for orig, dests in edges_keyframe[graph].items():
					if orig not in edge_val_keyframe[graph]:
						edge_val_keyframe[graph][orig] = {}
					for dest, ex in dests.items():
						self._edges_cache.set_keyframe(
							(graph, orig, dest), *now, {0: ex}
						)
						if ex and dest not in edge_val_keyframe[graph][orig]:
							edge_val_keyframe[graph][orig][dest] = {}
			if deltg is not None and "edge_val" in deltg:
				dgev = deltg.pop("edge_val")
				if graph in edge_val_keyframe:
					for orig, dests in dgev.items():
						if orig in evkg:
							evkgo = evkg[orig]
							for dest, vals in dests.items():
								if dest in evkgo:
									evkgo[dest].update(vals)
				else:
					edge_val_keyframe[graph] = dgev
			if graph in edge_val_keyframe:
				for orig, dests in edge_val_keyframe[graph].items():
					for dest, val in dests.items():
						self._edge_val_cache.set_keyframe(
							(graph, orig, dest, 0), *now, val
						)
			if deltg:
				if graph in graph_val_keyframe:
					if (
						"units" in graph_val_keyframe[graph]
						and "units" in deltg
					):
						units_kf = graph_val_keyframe[graph]["units"]
						units_update = deltg.pop("units")
						if not units_update:
							continue
						for newgraf in units_update.keys() - units_kf.keys():
							units_kf[newgraf] = units_update[newgraf]
						for oldgraf, unitz in units_kf.items():
							unitz.update(units_update[oldgraf])
					graph_val_keyframe[graph].update(deltg)
				else:
					graph_val_keyframe[graph] = deltg
			self._graph_val_cache.set_keyframe(
				(graph,), *now, graph_val_keyframe.get(graph, {})
			)
			for when in whens:
				inskf(
					graph,
					*when,
					node_val_keyframe.get(graph, {}),
					edge_val_keyframe.get(graph, {}),
					graph_val_keyframe.get(graph, {}),
				)
				kfl.append((graph, *when))
		self._graph_cache.set_keyframe(*now, graphs_keyframe)
		self._nudge_loaded(*now)

	def _recurse_delta_keyframes(self, time_from):
		"""Make keyframes until we have one in the current branch"""
		kfd = self._keyframes_dict
		if time_from[0] in kfd:
			# could probably avoid these sorts by restructuring kfd
			for turn in sorted(kfd[time_from[0]].keys(), reverse=True):
				if turn < time_from[1]:
					return time_from[0], turn, max(kfd[time_from[0]][turn])
				elif turn == time_from[1]:
					for tick in sorted(kfd[time_from[0]][turn], reverse=True):
						if time_from[2] <= tick:
							return time_from[0], turn, tick
		parent, branched_turn_from, branched_tick_from, turn_to, tick_to = (
			self._branches[time_from[0]]
		)
		if parent is None:
			self._snap_keyframe_de_novo(*time_from)
			return time_from
		else:
			(parent, turn_from, tick_from) = self._recurse_delta_keyframes(
				(parent, branched_turn_from, branched_tick_from)
			)
			if (
				parent,
				branched_turn_from,
				branched_tick_from,
			) not in self._keyframes_times:
				self._get_keyframe(parent, turn_from, tick_from)
				self._snap_keyframe_from_delta(
					(parent, turn_from, tick_from),
					(parent, branched_turn_from, branched_tick_from),
					self._get_branch_delta(
						parent,
						turn_from,
						tick_from,
						branched_turn_from,
						branched_tick_from,
					),
				)
			if (
				time_from[0],
				branched_turn_from,
				branched_tick_from,
			) not in self._keyframes_times:
				assert (
					parent,
					branched_turn_from,
					branched_tick_from,
				) in self._keyframes_times
				self._copy_kf(
					parent,
					time_from[0],
					branched_turn_from,
					branched_tick_from,
				)
		return time_from[0], branched_turn_from, branched_tick_from

	@world_locked
	def snap_keyframe(self, silent=False) -> Optional[dict]:
		"""Make a copy of the complete state of the world.

		You need to do this occasionally in order to keep time travel
		performant.

		The keyframe will be saved to the database at the next call to
		``flush``.

		Return the keyframe by default. With ``silent=True``,
		return ``None``. This is a little faster, and uses a little less
		memory.

		"""
		branch, turn, tick = self._btt()
		if (branch, turn, tick) in self._keyframes_times:
			if silent:
				return
			return self._get_keyframe(branch, turn, tick)
		kfd = self._keyframes_dict
		the_kf: Optional[Tuple[str, int, int]] = None
		if branch in kfd:
			# I could probably avoid sorting these by using windowdicts
			for trn in sorted(kfd[branch].keys(), reverse=True):
				if trn < turn:
					the_kf = (branch, trn, max(kfd[branch][trn]))
					break
				elif trn == turn:
					for tck in sorted(kfd[branch][trn], reverse=True):
						if tck <= tick:
							the_kf = (branch, trn, tck)
							break
				if the_kf is not None:
					break
		if the_kf is None:
			parent, _, _, turn_to, tick_to = self._branches[branch]
			if parent is None:
				self._snap_keyframe_de_novo(branch, turn, tick)
				if silent:
					return
				else:
					return self._get_kf(branch, turn, tick)
			the_kf = self._recurse_delta_keyframes((branch, turn, tick))
		if the_kf not in self._keyframes_loaded:
			self._get_keyframe(*the_kf, silent=True)
		if the_kf != (branch, turn, tick):
			self._snap_keyframe_from_delta(
				the_kf,
				(branch, turn, tick),
				self._get_branch_delta(*the_kf, turn, tick),
			)
			if the_kf[0] != branch:
				self._copy_kf(the_kf[0], branch, turn, tick)
		if not silent:
			return self._get_kf(branch, turn, tick)

	def _build_loading_windows(
		self,
		branch_from: str,
		turn_from: int,
		tick_from: int,
		branch_to: str,
		turn_to: Optional[int],
		tick_to: Optional[int],
	) -> List[Tuple[str, int, int, int, int]]:
		"""Return windows of time I've got to load

		In order to have a complete timeline between these points.

		Returned windows are in reverse chronological order.

		"""
		if branch_from == branch_to:
			return [(branch_from, turn_from, tick_from, turn_to, tick_to)]
		windows = []
		if turn_to is None:
			(
				branch1,
				turn1,
				tick1,
				_,
				_,
			) = self._branches[branch_to]
			windows.append(
				(
					branch_to,
					turn1,
					tick1,
					None,
					None,
				)
			)
			parentage_iter = self._iter_parent_btt(branch1, turn1, tick1)
		else:
			parentage_iter = self._iter_parent_btt(branch_to, turn_to, tick_to)
			branch1, turn1, tick1 = next(parentage_iter)
		for branch0, turn0, tick0 in parentage_iter:
			windows.append((branch1, turn0, tick0, turn1, tick1))
			(branch1, turn1, tick1) = (branch0, turn0, tick0)
			if branch0 == branch_from:
				windows.append((branch0, turn_from, tick_from, turn0, tick0))
				break
		else:
			raise HistoricKeyError("Couldn't build sensible loading windows")
		return windows

	def _updload(self, branch, turn, tick):
		loaded = self._loaded
		if branch not in loaded:
			loaded[branch] = (turn, tick, turn, tick)
			return
		(early_turn, early_tick, late_turn, late_tick) = loaded[branch]
		if turn < early_turn or (turn == early_turn and tick < early_tick):
			(early_turn, early_tick) = (turn, tick)
		if turn > late_turn or (turn == late_turn and tick > late_tick):
			(late_turn, late_tick) = (turn, tick)
		loaded[branch] = (early_turn, early_tick, late_turn, late_tick)

	def _build_keyframe_window(
		self, branch: str, turn: int, tick: int, loading=False
	) -> Tuple[Optional[Tuple[str, int, int]], Optional[Tuple[str, int, int]]]:
		"""Return a pair of keyframes that contain the given moment

		They give the smallest contiguous span of time I can reasonably load.

		"""
		branch_now = branch
		turn_now = turn
		tick_now = tick
		latest_past_keyframe: Optional[Tuple[str, int, int]] = None
		earliest_future_keyframe: Optional[Tuple[str, int, int]] = None
		branch_parents = self._branch_parents
		cache = self._keyframes_times if loading else self._keyframes_loaded
		for branch, turn, tick in cache:
			# Figure out the latest keyframe that is earlier than the present
			# moment, and the earliest keyframe that is later than the
			# present moment, for each graph. Can I avoid iterating over the
			# entire keyframes table, somehow?
			if branch == branch_now:
				if turn < turn_now:
					if latest_past_keyframe:
						(late_branch, late_turn, late_tick) = (
							latest_past_keyframe
						)
						if (
							late_branch != branch
							or late_turn < turn
							or (late_turn == turn and late_tick < tick)
						):
							latest_past_keyframe = (branch, turn, tick)
					else:
						latest_past_keyframe = (branch, turn, tick)
				elif turn > turn_now:
					if earliest_future_keyframe:
						(early_branch, early_turn, early_tick) = (
							earliest_future_keyframe
						)
						if (
							early_branch != branch
							or early_turn > turn
							or (early_turn == turn and early_tick > tick)
						):
							earliest_future_keyframe = (branch, turn, tick)
					else:
						earliest_future_keyframe = (branch, turn, tick)
				elif tick < tick_now:
					if latest_past_keyframe:
						(late_branch, late_turn, late_tick) = (
							latest_past_keyframe
						)
						if (
							late_branch != branch
							or late_turn < turn
							or (late_turn == turn and late_tick < tick)
						):
							latest_past_keyframe = (branch, turn, tick)
					else:
						latest_past_keyframe = (branch, turn, tick)
				elif tick > tick_now:
					if earliest_future_keyframe:
						(early_branch, early_turn, early_tick) = (
							earliest_future_keyframe
						)
						if (
							early_branch != branch
							or early_turn > turn
							or (early_turn == turn and early_tick > tick)
						):
							earliest_future_keyframe = (branch, turn, tick)
					else:
						earliest_future_keyframe = (branch, turn, tick)
				else:
					latest_past_keyframe = (branch, turn, tick)
			elif branch in branch_parents[branch_now]:
				if latest_past_keyframe:
					(late_branch, late_turn, late_tick) = latest_past_keyframe
					if branch == late_branch:
						if turn > late_turn or (
							turn == late_turn and tick > late_tick
						):
							latest_past_keyframe = (branch, turn, tick)
					elif late_branch in branch_parents[branch]:
						latest_past_keyframe = (branch, turn, tick)
				else:
					latest_past_keyframe = (branch, turn, tick)
		(branch, turn, tick) = (branch_now, turn_now, tick_now)
		if not loading or branch not in self._loaded:
			return latest_past_keyframe, earliest_future_keyframe
		if (
			earliest_future_keyframe
			and earliest_future_keyframe[1:] < self._loaded[branch][:2]
		):
			earliest_future_keyframe = (branch, *self._loaded[branch][2:])
		if (
			latest_past_keyframe
			and self._loaded[branch][2:] < latest_past_keyframe[1:]
		):
			latest_past_keyframe = (branch, *self._loaded[branch][:2])
		return latest_past_keyframe, earliest_future_keyframe

	@world_locked
	def _load_between(
		self,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	):
		self._get_keyframe(branch, turn_from, tick_from, silent=True)
		noderows = []
		nodevalrows = []
		edgerows = []
		edgevalrows = []
		graphvalrows = []
		graphsrows = list(
			self.query.graphs_types(
				branch, turn_from, tick_from, turn_to, tick_to
			)
		)
		self._graph_cache.load(graphsrows)
		loaded_graphs = self.query.load_windows(
			[(branch, turn_from, tick_from, turn_to, tick_to)]
		)
		for graph, loaded in loaded_graphs.items():
			noderows.extend(loaded["nodes"])
			edgerows.extend(loaded["edges"])
			nodevalrows.extend(loaded["node_val"])
			edgevalrows.extend(loaded["edge_val"])
			graphvalrows.extend(loaded["graph_val"])
			loaded_graphs[graph] = loaded
		self._nodes_cache.load(noderows)
		self._node_val_cache.load(nodevalrows)
		self._edges_cache.load(edgerows)
		self._edge_val_cache.load(edgevalrows)
		self._graph_val_cache.load(graphvalrows)
		return loaded_graphs

	def load_between(
		self,
		branch: str,
		turn_from: int,
		tick_from: int,
		turn_to: int,
		tick_to: int,
	) -> None:
		self._load_between(branch, turn_from, tick_from, turn_to, tick_to)

	@world_locked
	def _read_at(
		self, branch: str, turn: int, tick: int
	) -> Tuple[
		Optional[Tuple[str, int, int]],
		Optional[Tuple[str, int, int]],
		list,
		dict,
	]:
		latest_past_keyframe: Optional[Tuple[str, int, int]]
		earliest_future_keyframe: Optional[Tuple[str, int, int]]
		branch_now, turn_now, tick_now = branch, turn, tick
		(latest_past_keyframe, earliest_future_keyframe) = (
			self._build_keyframe_window(
				branch_now,
				turn_now,
				tick_now,
				loading=True,
			)
		)
		# If branch is a descendant of branch_now, don't load the keyframe
		# there, because then we'd potentially be loading keyframes from any
		# number of possible futures, and we're trying to be conservative
		# about what we load. If neither branch is an ancestor of the other,
		# we can't use the keyframe for this load

		if latest_past_keyframe is None:
			if earliest_future_keyframe is None:
				return (
					None,
					None,
					list(
						self.query.graphs_types(
							self.query.globl["main_branch"], 0, 0
						)
					),
					self.query.load_windows(
						[(self.query.globl["main_branch"], 0, 0, None, None)]
					),
				)
			else:
				windows = self._build_loading_windows(
					self.query.globl["main_branch"], 0, 0, branch, turn, tick
				)
		else:
			past_branch, past_turn, past_tick = latest_past_keyframe
			if earliest_future_keyframe is None:
				# Load data from the keyframe to now
				windows = self._build_loading_windows(
					past_branch,
					past_turn,
					past_tick,
					branch,
					None,
					None,
				)
			else:
				# Load data between the two keyframes
				(future_branch, future_turn, future_tick) = (
					earliest_future_keyframe
				)
				windows = self._build_loading_windows(
					past_branch,
					past_turn,
					past_tick,
					future_branch,
					future_turn,
					future_tick,
				)
		graphs_types = []
		for window in windows:
			graphs_types.extend(self.query.graphs_types(*window))
		return (
			latest_past_keyframe,
			earliest_future_keyframe,
			graphs_types,
			self.query.load_windows(windows),
		)

	@world_locked
	def _load_at(self, branch: str, turn: int, tick: int) -> None:
		if self._time_is_loaded(branch, turn, tick):
			return
		self._load(*self._read_at(branch, turn, tick))

	def _load(
		self,
		latest_past_keyframe: Optional[Tuple[str, int, int]],
		earliest_future_keyframe: Optional[Tuple[str, int, int]],
		graphs_rows: list,
		loaded: dict,
	):
		if latest_past_keyframe:
			self._get_keyframe(*latest_past_keyframe)

		self._graph_cache.load(graphs_rows)
		noderows = []
		edgerows = []
		nodevalrows = []
		edgevalrows = []
		graphvalrows = []
		for graph, graph_loaded in loaded.items():
			noderows.extend(graph_loaded["nodes"])
			edgerows.extend(graph_loaded["edges"])
			nodevalrows.extend(graph_loaded["node_val"])
			edgevalrows.extend(graph_loaded["edge_val"])
			graphvalrows.extend(graph_loaded["graph_val"])

		self._graph_cache.load(graphs_rows)
		self._nodes_cache.load(noderows)
		self._edges_cache.load(edgerows)
		self._graph_val_cache.load(graphvalrows)
		self._node_val_cache.load(nodevalrows)
		self._edge_val_cache.load(edgevalrows)

	def load_at(self, branch: str, turn: int, tick: int) -> None:
		self._load_at(branch, turn, tick)

	@world_locked
	def unload(self) -> None:
		"""Remove everything from memory that can be removed."""
		# find the slices of time that need to stay loaded
		branch, turn, tick = self._btt()
		iter_parent_btt = self._iter_parent_btt
		kfd = self._keyframes_dict
		if not kfd:
			return
		loaded = self._loaded
		to_keep = {}
		# Find a path to the latest past keyframe we can use. Keep things
		# loaded from there to here.
		for past_branch, past_turn, past_tick in iter_parent_btt(
			branch, turn, tick
		):
			if past_branch not in loaded:
				continue  # nothing happened in this branch i guess
			early_turn, early_tick, late_turn, late_tick = loaded[past_branch]
			if past_branch in kfd:
				for kfturn, kfticks in kfd[past_branch].items():
					# this can't possibly perform very well.
					# Maybe I need another loadedness dict that gives the two
					# keyframes I am between and gets upkept upon time travel
					for kftick in kfticks:
						if (
							(early_turn, early_tick)
							<= (kfturn, kftick)
							<= (late_turn, late_tick)
						):
							if (
								kfturn < turn
								or (kfturn == turn and kftick < tick)
							) and (
								kfturn > early_turn
								or (
									kfturn == early_turn
									and kftick > early_tick
								)
							):
								early_turn, early_tick = kfturn, kftick
							elif (
								kfturn > turn
								or (kfturn == turn and kftick >= tick)
							) and (
								kfturn < late_turn
								or (kfturn == late_turn and kftick < late_tick)
							):
								late_turn, late_tick = kfturn, kftick
				to_keep[past_branch] = (
					early_turn,
					early_tick,
					*max(((past_turn, past_tick), (late_turn, late_tick))),
				)
				break
			else:
				to_keep[past_branch] = (
					early_turn,
					early_tick,
					late_turn,
					late_tick,
				)
		if not to_keep:
			# unloading literally everything would make the game unplayable,
			# so don't
			if hasattr(self, "warning"):
				self.warning("Not unloading, due to lack of keyframes")
			return
		caches = self._caches
		kf_to_keep = set()
		for past_branch, (
			early_turn,
			early_tick,
			late_turn,
			late_tick,
		) in to_keep.items():
			# I could optimize this with windowdicts
			if early_turn == late_turn:
				if (
					past_branch in self._keyframes_dict
					and early_turn in self._keyframes_dict[past_branch]
				):
					for tick in self._keyframes_dict[past_branch][early_turn]:
						if early_tick <= tick <= late_tick:
							kf_to_keep.add((past_branch, early_turn, tick))
			else:
				if past_branch in self._keyframes_dict:
					for turn, ticks in self._keyframes_dict[
						past_branch
					].items():
						if early_turn == turn:
							for tick in ticks:
								if early_tick <= tick:
									kf_to_keep.add((past_branch, turn, tick))
						elif turn == late_turn:
							for tick in ticks:
								if tick <= late_tick:
									kf_to_keep.add((past_branch, turn, tick))
						else:
							kf_to_keep.update(
								(past_branch, turn, tick) for tick in ticks
							)
			for cache in caches:
				cache.truncate(past_branch, early_turn, early_tick, "backward")
				cache.truncate(past_branch, late_turn, late_tick, "forward")
				for graph, branches in cache.keyframe.items():
					turns = branches[past_branch]
					turns.truncate(late_turn, "forward")
					try:
						late = turns[late_turn]
					except HistoricKeyError:
						pass
					else:
						late.truncate(late_tick, "forward")
					turns.truncate(early_turn, "backward")
					try:
						early = turns[early_turn]
					except HistoricKeyError:
						pass
					else:
						early.truncate(early_tick, "backward")
		self._keyframes_loaded = kf_to_keep
		loaded.update(to_keep)
		for branch in set(loaded).difference(to_keep):
			for cache in caches:
				cache.remove_branch(branch)
			del loaded[branch]

	def _time_is_loaded(
		self, branch: str, turn: int = None, tick: int = None
	) -> bool:
		loaded = self._loaded
		if branch not in loaded:
			return False
		if turn is None:
			if tick is not None:
				raise ValueError("Need both or neither of turn and tick")
			return True
		if tick is None:
			(past_turn, _, future_turn, _) = loaded[branch]
			return past_turn <= turn <= future_turn
		else:
			early_turn, early_tick, late_turn, late_tick = loaded[branch]
			return (
				(early_turn, early_tick)
				<= (turn, tick)
				<= (late_turn, late_tick)
			)

	def __enter__(self):
		"""Enable the use of the ``with`` keyword"""
		return self

	def __exit__(self, *args):
		"""Alias for ``close``"""
		self.close()

	def is_ancestor_of(self, parent: str, child: str) -> bool:
		"""Return whether ``child`` is a branch descended from ``parent`` at
		any remove.

		"""
		if parent == self.main_branch:
			return True
		if child == self.main_branch:
			return False
		if child not in self._branches:
			raise ValueError(
				"The branch {} seems not to have ever been created".format(
					child
				)
			)
		if self._branches[child][0] == parent:
			return True
		return self.is_ancestor_of(parent, self._branches[child][0])

	def branches(self) -> set:
		return set(self._branches)

	def branch_parent(self, branch: str) -> Optional[str]:
		return self._branches[branch][0]

	def branch_start(self, branch: str) -> Tuple[int, int]:
		return self._branches[branch][1:3]

	def branch_end(self, branch: str) -> Tuple[int, int]:
		return self._branches[branch][3:5]

	def turn_end(self, branch: str = None, turn: int = None) -> int:
		branch = branch or self._obranch
		turn = turn or self._oturn
		return self._turn_end[branch, turn]

	def turn_end_plan(self, branch: str = None, turn: int = None):
		branch = branch or self._obranch
		turn = turn or self._oturn
		return self._turn_end_plan[branch, turn]

	def _get_branch(self) -> str:
		return self._obranch

	@world_locked
	def _set_branch(self, v: str):
		if self._planning:
			raise ValueError("Don't change branches while planning")
		curbranch, curturn, curtick = self._btt()
		if curbranch == v:
			self._otick = self._turn_end_plan[curbranch, curturn]
			return
		# make sure I'll end up within the revision range of the
		# destination branch
		if v != self.main_branch and v in self._branches:
			parturn = self._branches[v][1]
			if curturn < parturn:
				raise OutOfTimelineError(
					"Tried to jump to branch {br} at turn {tr}, "
					"but {br} starts at turn {rv}. "
					"Go to turn {rv} or later to use this branch.".format(
						br=v, tr=self.turn, rv=parturn
					),
					self.branch,
					self.turn,
					self.tick,
					v,
					self.turn,
					self.tick,
				)
		branch_is_new = v not in self._branches
		if branch_is_new:
			# assumes the present turn in the parent branch has
			# been finalized.
			self.query.new_branch(v, curbranch, curturn, curtick)
			self._branches[v] = curbranch, curturn, curtick, curturn, curtick
			self._upd_branch_parentage(v, curbranch)
			self._turn_end_plan[v, curturn] = self._turn_end[v, curturn] = (
				curtick
			)
		self._obranch = v
		self._otick = tick = self._turn_end_plan[v, curturn]
		loaded = self._loaded
		if branch_is_new:
			self._copy_plans(curbranch, curturn, curtick)
			self.snap_keyframe(silent=True)
			loaded[v] = (curturn, tick, curturn, tick)
			return
		elif v not in loaded:
			self._load_at(v, curturn, tick)
			return
		(start_turn, start_tick, end_turn, end_tick) = loaded[v]
		if (
			curturn > end_turn or (curturn == end_turn and tick > end_tick)
		) or (
			curturn < start_turn
			or (curturn == start_turn and tick < start_tick)
		):
			self._load_at(v, curturn, tick)

	@world_locked
	def _copy_plans(
		self, branch_from: str, turn_from: int, tick_from: int
	) -> None:
		"""Copy all plans active at the given time to the current branch"""
		plan_ticks = self._plan_ticks
		plan_ticks_uncommitted = self._plan_ticks_uncommitted
		time_plan = self._time_plan
		plans = self._plans
		branch = self.branch
		where_cached = self._where_cached
		turn_end_plan = self._turn_end_plan
		for plan_id in self._branches_plans[branch_from]:
			_, start_turn, start_tick = plans[plan_id]
			if start_turn > turn_from or (
				start_turn == turn_from and start_tick > tick_from
			):
				continue
			incremented = False
			for turn, ticks in list(plan_ticks[plan_id].items()):
				if turn < turn_from:
					continue
				for tick in ticks:
					if turn == turn_from and tick < tick_from:
						continue
					if not incremented:
						self._last_plan += 1
						incremented = True
						plans[self._last_plan] = branch, turn, tick
					plan_ticks[self._last_plan][turn].append(tick)
					plan_ticks_uncommitted.append(
						(self._last_plan, turn, tick)
					)
					for cache in where_cached[branch_from, turn, tick]:
						data = cache.settings[branch_from][turn][tick]
						value = data[-1]
						key = data[:-1]
						args = key + (branch, turn, tick, value)
						if hasattr(cache, "setdb"):
							cache.setdb(*args)
						cache.store(*args, planning=True)
						time_plan[branch, turn, tick] = self._last_plan
						turn_end_plan[branch, turn] = tick

	@world_locked
	def delete_plan(self, plan: int) -> None:
		"""Delete the portion of a plan that has yet to occur.

		:arg plan: integer ID of a plan, as given by
				   ``with self.plan() as plan:``

		"""
		branch, turn, tick = self._btt()
		to_delete = []
		plan_ticks = self._plan_ticks[plan]
		for (
			trn,
			tcks,
		) in (
			plan_ticks.items()
		):  # might improve performance to use a WindowDict for plan_ticks
			if turn == trn:
				for tck in tcks:
					if tck >= tick:
						to_delete.append((trn, tck))
			elif trn > turn:
				to_delete.extend((trn, tck) for tck in tcks)
		# Delete stuff that happened at contradicted times,
		# and then delete the times from the plan
		where_cached = self._where_cached
		time_plan = self._time_plan
		for trn, tck in to_delete:
			for cache in where_cached[branch, trn, tck]:
				cache.remove(branch, trn, tck)
				if hasattr(cache, "deldb"):
					cache.deldb(branch, trn, tck)
			del where_cached[branch, trn, tck]
			plan_ticks[trn].remove(tck)
			if not plan_ticks[trn]:
				del plan_ticks[trn]
			del time_plan[branch, trn, tck]

	# easier to override things this way
	@property
	def branch(self) -> str:
		"""The fork of the timestream that we're on."""
		return self._get_branch()

	@branch.setter
	def branch(self, v: str):
		self._set_branch(v)

	def _get_turn(self) -> int:
		return self._oturn

	@world_locked
	def _set_turn(self, v: int):
		branch = self.branch
		loaded = self._loaded
		if v == self.turn:
			self._otick = tick = self._turn_end_plan[tuple(self.time)]
			if branch not in loaded:
				self._load_at(branch, v, tick)
				return
			(start_turn, start_tick, end_turn, end_tick) = loaded[branch]
			if v > end_turn or (v == end_turn and tick > end_tick):
				self._load_at(branch, v, tick)
			return
		if not isinstance(v, int):
			raise TypeError("turn must be an integer")
		# enforce the arrow of time, if it's in effect
		if self._forward and v < self._oturn:
			raise ValueError("Can't time travel backward in a forward context")

		parent, turn_start, tick_start, turn_end, tick_end = self._branches[
			branch
		]
		if v < turn_start:
			raise OutOfTimelineError(
				"The turn number {} "
				"occurs before the start of "
				"the branch {}".format(v, branch),
				self.branch,
				self.turn,
				self.tick,
				self.branch,
				v,
				self.tick,
			)
		elif self._enforce_end_of_time and v > turn_end and not self._planning:
			raise OutOfTimelineError(
				f"The turn number {v} occurs after the end "
				f"of the branch {branch}",
				self.branch,
				self.turn,
				self.tick,
				self.branch,
				v,
				self.tick,
			)
		tick = self._turn_end_plan[branch, v]
		if (turn_start, tick_start) <= (v, tick) <= (turn_end, tick_end):
			if branch in loaded:
				(start_turn, start_tick, end_turn, end_tick) = loaded[branch]
				if v > end_turn or (v == end_turn and tick > end_tick):
					self._load_at(branch, v, tick)
				elif v < start_turn or (v == start_turn and tick < start_tick):
					self._load_at(branch, v, tick)
			else:
				self._load_at(branch, v, tick)
		if v > turn_end and not self._planning:
			self._branches[branch] = parent, turn_start, tick_start, v, tick
		self._otick = tick
		self._oturn = v

	# easier to override things this way
	@property
	def turn(self) -> int:
		"""Units of time that have passed since the sim started."""
		return self._get_turn()

	@turn.setter
	def turn(self, v: int):
		self._set_turn(v)

	def _get_tick(self) -> int:
		return self._otick

	@world_locked
	def _set_tick(self, v: int) -> None:
		if not isinstance(v, int):
			raise TypeError("tick must be an integer")
		time = (branch, turn) = (self._obranch, self._oturn)
		# enforce the arrow of time, if it's in effect
		if self._forward and v < self._otick:
			raise ValueError("Can't time travel backward in a forward context")
		if v > self._turn_end_plan[time]:  # TODO: only mutate after load
			self._turn_end_plan[time] = v
		if not self._planning and v > self._turn_end[time]:
			self._turn_end[time] = v
		(parent, turn_start, tick_start, turn_end, tick_end) = self._branches[
			branch
		]
		if turn >= turn_end and v > tick_end:
			self._branches[branch] = (
				parent,
				turn_start,
				tick_start,
				turn,
				v,
			)
		self._otick = v
		loaded = self._loaded
		if branch not in loaded:
			self._load_at(branch, turn, v)
			return
		(start_turn, start_tick, end_turn, end_tick) = loaded[branch]
		if turn > end_turn or (turn == end_turn and v > end_tick):
			if (branch, end_turn, end_tick) in self._keyframes_times:
				self._load_at(branch, turn, v)
				return
			loaded[branch] = (start_turn, start_tick, turn, v)
		elif turn < start_turn or (turn == start_turn and v < start_tick):
			self._load_at(branch, turn, v)

	# easier to override things this way
	@property
	def tick(self) -> int:
		"""A counter of how many changes have occurred this turn.

		Can be set manually, but is more often set to the last tick in a turn
		as a side effect of setting ``turn``.

		"""
		return self._get_tick()

	@tick.setter
	def tick(self, v):
		self._set_tick(v)

	def _btt(self) -> Tuple[str, int, int]:
		"""Return the branch, turn, and tick."""
		return self._obranch, self._oturn, self._otick

	def _set_btt(self, branch: str, turn: int, tick: int):
		(self._obranch, self._oturn, self._otick) = (branch, turn, tick)

	@world_locked
	def _nbtt(self) -> Tuple[str, int, int]:
		"""Increment the tick and return branch, turn, tick

		Unless we're viewing the past, in which case raise HistoryError.

		Idea is you use this when you want to advance time, which you
		can only do once per branch, turn, tick.

		"""
		(
			btt,
			branch_end,
			turn_end_plan,
			turn_end,
			plan_ticks,
			plan_ticks_uncommitted,
			time_plan,
			branches,
		) = self._nbtt_stuff
		branch, turn, tick = btt()
		branch_turn = (branch, turn)
		tick += 1
		if branch_turn in turn_end_plan and tick <= turn_end_plan[branch_turn]:
			tick = turn_end_plan[branch_turn] + 1
		if turn_end[branch_turn] > tick:
			raise HistoricKeyError(
				"You're not at the end of turn {}. "
				"Go to tick {} to change things".format(
					turn, turn_end[branch_turn]
				)
			)
		parent, start_turn, start_tick, end_turn, end_tick = branches[branch]
		if self._planning:
			last_plan = self._last_plan
			if (turn, tick) in plan_ticks[last_plan]:
				raise OutOfTimelineError(
					"Trying to make a plan at {}, "
					"but that time already happened".format(
						(branch, turn, tick)
					),
					self.branch,
					self.turn,
					self.tick,
					self.branch,
					self.turn,
					tick,
				)
			plan_ticks[last_plan][turn].append(tick)
			plan_ticks_uncommitted.append((last_plan, turn, tick))
			time_plan[branch, turn, tick] = last_plan
		else:
			if turn < branch_end[branch]:
				raise OutOfTimelineError(
					"You're in the past. Go to turn {} to change things"
					" -- or start a new branch".format(branch_end[branch]),
					*btt(),
					branch,
					turn,
					tick,
				)
			elif turn == branch_end[branch]:
				# Accept any plans made for this turn
				tick = turn_end_plan[branch, turn] + 1
			if tick > turn_end[branch_turn]:
				turn_end[branch_turn] = tick
		if turn > end_turn or (turn == end_turn and tick > end_tick):
			branches[branch] = parent, start_turn, start_tick, turn, tick
		if tick > turn_end_plan[branch_turn]:
			turn_end_plan[branch_turn] = tick
		loaded = self._loaded
		if branch in loaded:
			(early_turn, early_tick, late_turn, late_tick) = loaded[branch]
			if turn > late_turn:
				(late_turn, late_tick) = (turn, tick)
			elif turn == late_turn and tick > late_tick:
				late_tick = tick
			loaded[branch] = (early_turn, early_tick, late_turn, late_tick)
		else:
			loaded[branch] = (turn, tick, turn, tick)
		self._otick = tick
		return branch, turn, tick

	def flush(self) -> None:
		"""Write pending changes to disk.

		You can set a ``flush_interval`` when you instantiate ``Engine``
		to call this every so many turns. However, this may cause your game to
		hitch up sometimes, so it's better to call ``flush`` when you know the
		player won't be running the simulation for a while.

		"""
		turn_end = self._turn_end
		set_turn = self.query.set_turn
		for (branch, turn), plan_end_tick in self._turn_end_plan.items():
			set_turn(branch, turn, turn_end[branch, turn], plan_end_tick)
		if self._plans_uncommitted:
			self.query.plans_insert_many(self._plans_uncommitted)
		if self._plan_ticks_uncommitted:
			self.query.plan_ticks_insert_many(self._plan_ticks_uncommitted)
		self.query.flush()
		self._plans_uncommitted = []
		self._plan_ticks_uncommitted = []

	@property
	def main_branch(self):
		return self.query.globl["main_branch"]

	def switch_main_branch(self, branch: str) -> None:
		if self.branch != self.main_branch or self.turn != 0 or self.tick != 0:
			raise ValueError("Go to the start of time first")
		if branch in self._branches and self._branches[branch][0] is not None:
			raise ValueError("Not a main branch")
		self.query.globl["main_branch"] = self.branch = branch

	@world_locked
	def commit(self, unload=True) -> None:
		"""Write the state of all graphs and commit the transaction.

		Also saves the current branch, turn, and tick.

		Call with ``unload=False`` if you want to keep the written state in memory.

		"""
		self.query.globl["branch"] = self._obranch
		self.query.globl["turn"] = self._oturn
		self.query.globl["tick"] = self._otick
		set_branch = self.query.set_branch
		for branch, (
			parent,
			turn_start,
			tick_start,
			turn_end,
			tick_end,
		) in self._branches.items():
			set_branch(
				branch, parent, turn_start, tick_start, turn_end, tick_end
			)
		self.flush()
		self.query.commit()
		if unload:
			self.unload()

	def close(self) -> None:
		"""Write changes to database and close the connection"""
		self.commit()
		self.query.close()

	def _nudge_loaded(self, branch: str, turn: int, tick: int) -> None:
		loaded = self._loaded
		if branch in loaded:
			past_turn, past_tick, future_turn, future_tick = loaded[branch]
			if turn < past_turn or (turn == past_turn and tick < past_tick):
				loaded[branch] = turn, tick, future_turn, future_tick
			elif turn > future_turn or (
				turn == future_turn and tick > future_tick
			):
				loaded[branch] = past_turn, past_tick, turn, tick
		else:
			loaded[branch] = turn, tick, turn, tick

	@world_locked
	def _init_graph(
		self,
		name: Key,
		type_s="DiGraph",
		data: Union[Graph, nx.Graph, dict, KeyframeTuple] = None,
	) -> None:
		if name in self.illegal_graph_names:
			raise GraphNameError("Illegal name")
		branch, turn, tick = self._btt()
		try:
			self._graph_cache.retrieve(name, branch, turn, tick)
			branch, turn, tick = self._nbtt()
		except KeyError as ex:
			if getattr(ex, "deleted", False):
				branch, turn, tick = self._nbtt()
		self._graph_cache.store(name, branch, turn, tick, type_s)
		self.query.new_graph(name, branch, turn, tick, type_s)
		self._nudge_loaded(branch, turn, tick)
		if data is None:
			data = ({}, {}, {})
		if isinstance(data, DiGraph):
			nodes = data._nodes_state()
			edges = data._edges_state()
			val = data._val_state()
			self._snap_keyframe_de_novo_graph(
				name, branch, turn, tick, nodes, edges, val
			)
			self.query.keyframe_graph_insert(
				name, branch, turn, tick, nodes, edges, val
			)
		elif isinstance(data, nx.Graph):
			self._snap_keyframe_de_novo_graph(
				name, branch, turn, tick, data._node, data._adj, data.graph
			)
			self.query.keyframe_graph_insert(
				name,
				branch,
				turn,
				tick,
				data._node,
				data._adj,
				data.graph,
			)
		elif isinstance(data, dict):
			try:
				data = nx.from_dict_of_dicts(data)
			except AttributeError:
				data = nx.from_dict_of_lists(data)
			self._snap_keyframe_de_novo_graph(
				name, branch, turn, tick, data._node, data._adj, data.graph
			)
			self.query.keyframe_graph_insert(
				name,
				branch,
				turn,
				tick,
				data._node,
				data._adj,
				data.graph,
			)
		else:
			if len(data) != 3 or not all(isinstance(d, dict) for d in data):
				raise ValueError("Invalid graph data")
			self._snap_keyframe_de_novo_graph(name, branch, turn, tick, *data)
			self.query.keyframe_graph_insert(name, branch, turn, tick, *data)

	def new_digraph(self, name: Key, data: dict = None, **attr) -> DiGraph:
		"""Return a new instance of type DiGraph, initialized with the given
		data if provided.

		:arg name: a name for the graph
		:arg data: dictionary or NetworkX graph object providing initial state

		"""
		if data and isinstance(data, nx.Graph):
			if not data.is_directed():
				data = nx.to_directed(data)
			self._init_graph(
				name, "DiGraph", [data._node, data._succ, data.graph]
			)
		else:
			self._init_graph(name, "DiGraph", data)
		ret = self._graph_objs[name] = DiGraph(self, name)
		return ret

	@world_locked
	def del_graph(self, name: Key) -> None:
		"""Mark a graph as deleted

		:arg name: name of an existing graph

		"""
		# make sure the graph exists before deleting anything
		graph = self.graph[name]
		for orig in list(graph.adj):
			for dest in list(graph.adj[orig]):
				del graph.adj[orig][dest]
		for node in list(graph.node):
			del graph.node[node]
		for stat in set(graph.graph) - {"name"}:
			del graph.graph[stat]
		branch, turn, tick = self._nbtt()
		self.query.graphs_insert(name, branch, turn, tick, "Deleted")
		self._graph_cache.store(name, branch, turn, tick, None)
		self._graph_cache.keycache.clear()

	def _kf_loaded(self, branch: str, turn: int, tick: int = None) -> bool:
		"""Is this keyframe currently loaded into the program?

		Absent a specific ``tick``, return whether there is at least one keyframe
		on the given turn.

		"""
		if tick is None:
			if branch not in self._keyframes_dict:
				return False
			if turn not in self._keyframes_dict[branch]:
				return False
			return any(
				(branch, turn, t) in self._keyframes_loaded
				for t in self._keyframes_dict[branch][turn]
			)
		return (branch, turn, tick) in self._keyframes_loaded

	def _iter_keyframes(
		self,
		branch: str,
		turn: int,
		tick: int,
		*,
		loaded=False,
		with_fork_points=False,
		stoptime: Tuple[str, int, int] = None,
	):
		"""Iterate back over (branch, turn, tick) at which there is a keyframe

		Follows the timestream, like :method:`_iter_parent_btt`, but yields more times.
		We may have any number of keyframes in the same branch, and will yield
		them all.

		With ``loaded=True``, only yield keyframes that are in memory now.

		Use ``with_fork_points=True`` to also include all the times that the
		timeline branched.

		``stoptime`` is as in :method:`_iter_parent_btt`.

		"""
		kfd = self._keyframes_dict
		kfs = self._keyframes_times
		kfl = self._keyframes_loaded
		it = pairwise(
			self._iter_parent_btt(branch, turn, tick, stoptime=stoptime)
		)
		try:
			a, b = next(it)
		except StopIteration:
			assert branch in self._branches and self._branches[branch][
				1:3
			] == (0, 0)
			a = (branch, turn, tick)
			b = (branch, 0, 0)
			if a == b:
				if (loaded and a in kfl) or (not loaded and a in kfs):
					yield a
				return
		for (b0, r0, t0), (b1, r1, t1) in chain([(a, b)], it):
			# we're going up the timestream, meaning that b1, r1, t1
			# is *before* b0, r0, t0
			if loaded:
				if (b0, r0, t0) in kfl:
					yield b0, r0, t0
			elif (b0, r0, t0) in kfs:
				yield b0, r0, t0
			if b0 not in kfd:
				continue
			assert b0 in self._branches
			kfdb = kfd[b0]
			if r0 in kfdb:
				tcks = sorted(kfdb[r0])
				while tcks and tcks[-1] > t0:
					tcks.pop()
				if loaded:
					for tck in reversed(tcks):
						if r0 == r1 and tck <= t1:
							break
						if (b0, r0, tck) != (b0, r0, t0) and (
							b0,
							r0,
							tck,
						) in kfl:
							yield b0, r0, tck
				else:
					for tck in reversed(tcks):
						if tck < t0:
							break
						yield b0, r0, tck
			for r_between in range(r0 - 1, r1, -1):  # too much iteration?
				if r_between in kfdb:
					tcks = sorted(kfdb[r_between], reverse=True)
					if loaded:
						for tck in tcks:
							if (b0, r_between, tck) in kfl:
								yield b0, r_between, tck
					else:
						for tck in tcks:
							yield b0, r_between, tck
			if r1 in kfdb:
				tcks = sorted(kfdb[r1], reverse=True)
				while tcks[-1] > t1:
					tcks.pop()
				if not tcks:
					if with_fork_points:
						yield b1, r1, t1
					continue
				if loaded:
					for tck in tcks:
						if (b1, r1, tck) in kfl:
							yield b1, r1, tck
				else:
					for tck in tcks:
						yield b1, r1, tck
				if with_fork_points and tcks[-1] == t1:
					continue
			if with_fork_points:
				yield b1, r1, t1

	def _iter_parent_btt(
		self,
		branch: str = None,
		turn: int = None,
		tick: int = None,
		*,
		stoptime: Tuple[str, int, int] = None,
	) -> Iterator[Tuple[str, int, int]]:
		"""Private use. Iterate over (branch, turn, tick), where the branch is
		a descendant of the previous (starting with whatever branch is
		presently active and ending at the main branch), and the turn is the
		latest revision in the branch that matters.

		:arg stoptime: a triple, ``(branch, turn, tick)``. Iteration will
		stop instead of yielding that time or any before it. The tick may be
		``None``, in which case, iteration will stop instead of yielding the
		turn.

		"""
		branch = branch or self.branch
		trn = self.turn if turn is None else turn
		tck = self.tick if tick is None else tick
		yield branch, trn, tck
		_branches = self._branches
		if stoptime:
			stopbranch, stopturn, stoptick = stoptime
			stopping = stopbranch == branch
			while branch in _branches and not stopping:
				(branch, trn, tck, _, _) = _branches[branch]
				if branch is None:
					return
				if branch == stopbranch:
					stopping = True
					if trn < stopturn or (
						trn == stopturn
						and (stoptick is None or tck <= stoptick)
					):
						return
				yield branch, trn, tck
		else:
			while branch in _branches:
				(branch, trn, tck, _, _) = _branches[branch]
				if branch is None:
					return
				yield branch, trn, tck

	def _branch_descendants(self, branch=None) -> Iterator[str]:
		"""Iterate over all branches immediately descended from the current
		one (or the given one, if available).

		"""
		branch = branch or self.branch
		for parent, (child, _, _, _, _) in self._branches.items():
			if parent == branch:
				yield child

	def _node_exists(self, character: Key, node: Key) -> bool:
		retrieve, btt = self._node_exists_stuff
		args = (character, node) + btt()
		retrieved = retrieve(args)
		return retrieved is not None and not isinstance(retrieved, Exception)

	@world_locked
	def _exist_node(self, character: Key, node: Key, exist=True) -> None:
		nbtt, exist_node, store = self._exist_node_stuff
		branch, turn, tick = nbtt()
		exist_node(character, node, branch, turn, tick, exist)
		store(character, node, branch, turn, tick, exist)

	def _edge_exists(
		self, character: Key, orig: Key, dest: Key, idx=0
	) -> bool:
		retrieve, btt = self._edge_exists_stuff
		args = (character, orig, dest, idx) + btt()
		retrieved = retrieve(args)
		return retrieved is not None and not isinstance(retrieved, Exception)

	@world_locked
	def _exist_edge(
		self, character: Key, orig: Key, dest: Key, idx=0, exist=True
	) -> None:
		nbtt, exist_edge, store = self._exist_edge_stuff
		branch, turn, tick = nbtt()
		exist_edge(
			character, orig, dest, idx, branch, turn, tick, exist or False
		)
		store(character, orig, dest, idx, branch, turn, tick, exist)
		if (character, orig, dest) in self._edge_objs:
			del self._edge_objs[character, orig, dest]
