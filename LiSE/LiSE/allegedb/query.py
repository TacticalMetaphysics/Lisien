# This file is part of allegedb, an object relational mapper for graphs.
# Copyright (c) Zachary Spector. public@zacharyspector.com
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
"""Wrapper to run SQL queries in a lightly abstracted way, such that
code that's more to do with the queries than with the data per se
doesn't pollute the other files so much.

"""

from threading import Thread, Lock
from time import monotonic
from typing import Tuple, Any, Iterator, Hashable
from queue import Queue
import os
from collections.abc import MutableMapping

from sqlalchemy.sql import Select
from sqlalchemy import create_engine, MetaData
from sqlalchemy.engine.base import Engine
from sqlalchemy.exc import ArgumentError, IntegrityError, OperationalError
from sqlalchemy.pool import NullPool

from . import wrap
from .wrap import DictWrapper, SetWrapper, ListWrapper

wrappath = os.path.dirname(wrap.__file__)

NodeRowType = Tuple[Hashable, Hashable, str, int, int, bool]
EdgeRowType = Tuple[Hashable, Hashable, Hashable, int, str, int, int, bool]
GraphValRowType = Tuple[Hashable, Hashable, str, int, int, Any]
NodeValRowType = Tuple[Hashable, Hashable, Hashable, str, int, int, Any]
EdgeValRowType = Tuple[Hashable, Hashable, Hashable, int, str, int, int, Any]


class TimeError(ValueError):
	"""Exception class for problems with the time model"""


class GlobalKeyValueStore(MutableMapping):
	"""A dict-like object that keeps its contents in a table.

	Mostly this is for holding the current branch and revision.

	"""

	def __init__(self, qe):
		self.qe = qe
		self._cache = dict(qe.global_items())

	def __iter__(self):
		yield from self._cache

	def __len__(self):
		return len(self._cache)

	def __getitem__(self, k):
		ret = self._cache[k]
		if isinstance(ret, dict):
			return DictWrapper(
				lambda: self._cache[k],
				lambda v: self.__setitem__(k, v),
				self,
				k,
			)
		elif isinstance(ret, list):
			return ListWrapper(
				lambda: self._cache[k],
				lambda v: self.__setitem__(k, v),
				self,
				k,
			)
		elif isinstance(ret, set):
			return SetWrapper(
				lambda: self._cache[k],
				lambda v: self.__setitem__(k, v),
				self,
				k,
			)
		return ret

	def __setitem__(self, k, v):
		if hasattr(v, "unwrap"):
			v = v.unwrap()
		self.qe.global_set(k, v)
		self._cache[k] = v

	def __delitem__(self, k):
		del self._cache[k]
		self.qe.global_del(k)


class ConnectionHolder:
	strings: dict

	def __init__(
		self, dbstring, connect_args, inq, outq, fn, tables, gather=None
	):
		self.lock = Lock()
		self.existence_lock = Lock()
		self.existence_lock.acquire()
		self._dbstring = dbstring
		self._connect_args = connect_args
		self._fn = fn
		self.inq = inq
		self.outq = outq
		self.tables = tables
		if gather is not None:
			self.gather = gather

	def commit(self):
		self.transaction.commit()
		self.transaction = self.connection.begin()

	def init_table(self, tbl):
		return self.call_one("create_{}".format(tbl))

	def run(self):
		dbstring = self._dbstring
		connect_args = self._connect_args
		if hasattr(self, "gather"):
			gather_sql = self.gather
		else:
			from .alchemy import gather_sql
		if isinstance(dbstring, Engine):
			self.engine = dbstring
		else:
			try:
				self.engine = create_engine(
					dbstring, connect_args=connect_args, poolclass=NullPool
				)
			except ArgumentError:
				self.engine = create_engine(
					"sqlite:///" + dbstring,
					connect_args=connect_args,
					poolclass=NullPool,
				)
		self.meta = MetaData()
		self.sql = gather_sql(self.meta)
		self.connection = self.engine.connect()
		self.transaction = self.connection.begin()
		while True:
			inst = self.inq.get()
			if inst == "shutdown":
				self.commit()
				self.transaction.close()
				self.connection.close()
				self.engine.dispose()
				self.existence_lock.release()
				return
			if inst == "commit":
				self.commit()
				continue
			if inst == "initdb":
				self.outq.put(self.initdb())
				continue
			if isinstance(inst, Select):
				res = self.connection.execute(inst).fetchall()
				self.outq.put(res)
				continue
			silent = False
			if inst[0] == "silent":
				inst = inst[1:]
				silent = True
			if inst[0] == "echo":
				self.outq.put(inst[1])
			elif inst[0] == "one":
				try:
					res = self.call_one(inst[1], *inst[2])
					if not silent:
						if hasattr(res, "returns_rows"):
							if res.returns_rows:
								o = list(res)
								self.outq.put(o)
							else:
								self.outq.put(None)
						else:
							o = list(res)
							self.outq.put(o)
				except Exception as ex:
					if not silent:
						self.outq.put(ex)
			elif inst[0] != "many":
				raise ValueError(f"Invalid instruction: {inst[0]}")
			else:
				try:
					res = self.call_many(inst[1], inst[2])
					if not silent:
						if hasattr(res, "returns_rows"):
							if res.returns_rows:
								self.outq.put(list(res))
							else:
								self.outq.put(None)
						else:
							rez = list(res.fetchall())
							self.outq.put(rez or None)
				except Exception as ex:
					if not silent:
						self.outq.put(ex)

	def call_one(self, k, *largs):
		statement = self.sql[k].compile(dialect=self.engine.dialect)
		if hasattr(statement, "positiontup"):
			return self.connection.execute(
				statement, dict(zip(statement.positiontup, largs))
			)
		elif largs:
			raise TypeError("{} is a DDL query, I think".format(k))
		return self.connection.execute(self.sql[k])

	def call_many(self, k, largs):
		statement = self.sql[k].compile(dialect=self.engine.dialect)
		return self.connection.execute(
			statement,
			[dict(zip(statement.positiontup, larg)) for larg in largs],
		)

	def initdb(self):
		"""Create tables and indices as needed."""
		for table in (
			"branches",
			"turns",
			"graphs",
			"graph_val",
			"nodes",
			"node_val",
			"edges",
			"edge_val",
			"plans",
			"plan_ticks",
			"keyframes",
			"global",
		):
			try:
				ret = self.init_table(table)
			except OperationalError:
				pass
			except Exception as ex:
				return ex
		self.commit()


class QueryEngine(object):
	flush_edges_t = 0
	holder_cls = ConnectionHolder
	tables = (
		"global",
		"branches",
		"turns",
		"graphs",
		"keyframes",
		"graph_val",
		"nodes",
		"node_val",
		"edges",
		"edge_val",
		"plans",
		"plan_ticks",
		"universals",
	)

	def __init__(
		self, dbstring, connect_args, pack=None, unpack=None, gather=None
	):
		dbstring = dbstring or "sqlite:///:memory:"
		self._inq = Queue()
		self._outq = Queue()
		self._holder = self.holder_cls(
			dbstring, connect_args, self._inq, self._outq, self.tables, gather
		)

		if pack is None:

			def pack(o: Any) -> bytes:
				return repr(o).encode()

		if unpack is None:
			from ast import literal_eval

			def unpack(b: bytes) -> Any:
				return literal_eval(b.decode())

		self.pack = pack
		self.unpack = unpack
		self._branches = {}
		self._nodevals2set = []
		self._edgevals2set = []
		self._graphvals2set = []
		self._nodes2set = []
		self._edges2set = []
		self._btts = set()
		self._t = Thread(target=self._holder.run, daemon=True)
		self._t.start()

	def echo(self, string):
		self._inq.put(("echo", string))
		return self._outq.get()

	def call_one(self, string, *args, **kwargs):
		__doc__ = ConnectionHolder.call_one.__doc__
		with self._holder.lock:
			self._inq.put(("one", string, args, kwargs))
			ret = self._outq.get()
		if isinstance(ret, Exception):
			raise ret
		return ret

	def call_many(self, string, args):
		__doc__ = ConnectionHolder.call_many.__doc__
		with self._holder.lock:
			self._inq.put(("many", string, args))
			ret = self._outq.get()
		if isinstance(ret, Exception):
			raise ret
		return ret

	def execute(self, stmt):
		if not isinstance(stmt, Select):
			raise TypeError("Only select statements should be executed")
		with self._holder.lock:
			self.flush()
			self._inq.put(stmt)
			return self._outq.get()

	def new_graph(self, graph, branch, turn, tick, typ):
		"""Declare a new graph by this name of this type."""
		graph = self.pack(graph)
		return self.call_one("graphs_insert", graph, branch, turn, tick, typ)

	def keyframes_insert(
		self, graph, branch, turn, tick, nodes, edges, graph_val
	):
		graph, nodes, edges, graph_val = map(
			self.pack, (graph, nodes, edges, graph_val)
		)
		return self.call_one(
			"keyframes_insert",
			graph,
			branch,
			turn,
			tick,
			nodes,
			edges,
			graph_val,
		)

	def keyframes_insert_many(self, many):
		pack = self.pack
		return self.call_many(
			"keyframes_insert",
			[
				(
					pack(graph),
					branch,
					turn,
					tick,
					pack(nodes),
					pack(edges),
					pack(graph_val),
				)
				for (
					graph,
					branch,
					turn,
					tick,
					nodes,
					edges,
					graph_val,
				) in many
			],
		)

	def keyframes_dump(self):
		unpack = self.unpack
		for (
			graph,
			branch,
			turn,
			tick,
			nodes,
			edges,
			graph_val,
		) in self.call_one("keyframes_dump"):
			yield (
				unpack(graph),
				branch,
				turn,
				tick,
				unpack(nodes),
				unpack(edges),
				unpack(graph_val),
			)

	def keyframes_list(self):
		unpack = self.unpack
		for graph, branch, turn, tick in self.call_one("keyframes_list"):
			yield unpack(graph), branch, turn, tick

	def get_keyframe(self, graph, branch, turn, tick):
		unpack = self.unpack
		stuff = self.call_one(
			"get_keyframe", self.pack(graph), branch, turn, tick
		)
		if not stuff:
			return
		nodes, edges, graph_val = stuff[0]
		return unpack(nodes), unpack(edges), unpack(graph_val)

	def graph_type(self, graph):
		"""What type of graph is this?"""
		graph = self.pack(graph)
		return self.call_one("graph_type", graph)[0][0]

	def have_branch(self, branch):
		"""Return whether the branch thus named exists in the database."""
		return bool(self.call_one("ctbranch", branch)[0][0])

	def all_branches(self):
		"""Return all the branch data in tuples of (branch, parent,
		parent_turn).

		"""
		return self.call_one("branches_dump")

	def global_get(self, key):
		"""Return the value for the given key in the ``globals`` table."""
		key = self.pack(key)
		r = self.call_one("global_get", key)[0]
		if r is None:
			raise KeyError("Not set")
		return self.unpack(r[0])

	def global_items(self):
		"""Iterate over (key, value) pairs in the ``globals`` table."""
		unpack = self.unpack
		dumped = self.call_one("global_dump")
		for k, v in dumped:
			yield (unpack(k), unpack(v))

	def get_branch(self):
		v = self.call_one("global_get", self.pack("branch"))[0]
		if v is None:
			return self.globl["main_branch"]
		return self.unpack(v[0])

	def get_turn(self):
		v = self.call_one("global_get", self.pack("turn"))[0]
		if v is None:
			return 0
		return self.unpack(v[0])

	def get_tick(self):
		v = self.call_one("global_get", self.pack("tick"))[0]
		if v is None:
			return 0
		return self.unpack(v[0])

	def global_set(self, key, value):
		"""Set ``key`` to ``value`` globally (not at any particular branch or
		revision)

		"""
		(key, value) = map(self.pack, (key, value))
		try:
			return self.call_one("global_insert", key, value)
		except IntegrityError:
			return self.call_one("global_update", value, key)

	def global_del(self, key):
		"""Delete the global record for the key."""
		key = self.pack(key)
		return self.call_one("global_del", key)

	def new_branch(self, branch, parent, parent_turn, parent_tick):
		"""Declare that the ``branch`` is descended from ``parent`` at
		``parent_turn``, ``parent_tick``

		"""
		return self.call_one(
			"branches_insert",
			branch,
			parent,
			parent_turn,
			parent_tick,
			parent_turn,
			parent_tick,
		)

	def update_branch(
		self, branch, parent, parent_turn, parent_tick, end_turn, end_tick
	):
		return self.call_one(
			"update_branches",
			parent,
			parent_turn,
			parent_tick,
			end_turn,
			end_tick,
			branch,
		)

	def set_branch(
		self, branch, parent, parent_turn, parent_tick, end_turn, end_tick
	):
		try:
			self.call_one(
				"branches_insert",
				branch,
				parent,
				parent_turn,
				parent_tick,
				end_turn,
				end_tick,
			)
		except IntegrityError:
			self.update_branch(
				branch, parent, parent_turn, parent_tick, end_turn, end_tick
			)

	def new_turn(self, branch, turn, end_tick=0, plan_end_tick=0):
		return self.call_one(
			"turns_insert", branch, turn, end_tick, plan_end_tick
		)

	def update_turn(self, branch, turn, end_tick, plan_end_tick):
		return self.call_one(
			"update_turns", end_tick, plan_end_tick, branch, turn
		)

	def set_turn(self, branch, turn, end_tick, plan_end_tick):
		try:
			return self.call_one(
				"turns_insert", branch, turn, end_tick, plan_end_tick
			)
		except IntegrityError:
			return self.call_one(
				"update_turns", end_tick, plan_end_tick, branch, turn
			)

	def set_turn_completed(self, branch, turn):
		try:
			return self.call_one("turns_completed_insert", branch, turn)
		except IntegrityError:
			return self.call_one("turns_completed_update", turn, branch)

	def turns_dump(self):
		return self.call_one("turns_dump")

	def graph_val_dump(self) -> Iterator[GraphValRowType]:
		"""Yield the entire contents of the graph_val table."""
		self._flush_graph_val()
		unpack = self.unpack
		for graph, key, branch, turn, tick, value in self.call_one(
			"graph_val_dump"
		):
			yield (
				unpack(graph),
				unpack(key),
				branch,
				turn,
				tick,
				unpack(value),
			)

	def load_graph_val(
		self, graph, branch, turn_from, tick_from, turn_to=None, tick_to=None
	) -> Iterator[GraphValRowType]:
		if (turn_to is None) ^ (tick_to is None):
			raise TypeError("I need both or neither of turn_to and tick_to")
		self._flush_graph_val()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.call_one(
				"load_graph_val_tick_to_end",
				pack(graph),
				branch,
				turn_from,
				turn_from,
				tick_from,
			)
		else:
			it = self.call_one(
				"load_graph_val_tick_to_tick",
				pack(graph),
				branch,
				turn_from,
				turn_from,
				tick_from,
				turn_to,
				turn_to,
				tick_to,
			)
		for key, turn, tick, value in it:
			yield graph, unpack(key), branch, turn, tick, unpack(value)

	def _flush_graph_val(self):
		"""Send all new and changed graph values to the database."""
		if not self._graphvals2set:
			return
		pack = self.pack
		self.call_many(
			"graph_val_insert",
			(
				(pack(graph), pack(key), branch, turn, tick, pack(value))
				for (
					graph,
					key,
					branch,
					turn,
					tick,
					value,
				) in self._graphvals2set
			),
		)
		self._graphvals2set = []

	def graph_val_set(self, graph, key, branch, turn, tick, value):
		if (branch, turn, tick) in self._btts:
			raise TimeError
		self._btts.add((branch, turn, tick))
		self._graphvals2set.append((graph, key, branch, turn, tick, value))

	def graph_val_del_time(self, branch, turn, tick):
		self._flush_graph_val()
		self.call_one("graph_val_del_time", branch, turn, tick)
		self._btts.discard((branch, turn, tick))

	def graphs_types(self):
		for graph, typ in self.call_one("graphs_types"):
			yield (self.unpack(graph), typ)

	def graphs_dump(self):
		unpack = self.unpack
		for graph, branch, turn, tick, typ in self.call_one("graphs_dump"):
			yield unpack(graph), branch, turn, tick, typ

	def graphs_insert(self, graph, branch, turn, tick, typ):
		self.call_one(
			"graphs_insert", self.pack(graph), branch, turn, tick, typ
		)

	def _flush_nodes(self):
		if not self._nodes2set:
			return
		pack = self.pack
		self.call_many(
			"nodes_insert",
			(
				(pack(graph), pack(node), branch, turn, tick, bool(extant))
				for (
					graph,
					node,
					branch,
					turn,
					tick,
					extant,
				) in self._nodes2set
			),
		)
		self._nodes2set = []

	def exist_node(self, graph, node, branch, turn, tick, extant):
		"""Declare that the node exists or doesn't.

		Inserts a new record or updates an old one, as needed.

		"""
		if (branch, turn, tick) in self._btts:
			raise TimeError
		self._btts.add((branch, turn, tick))
		self._nodes2set.append((graph, node, branch, turn, tick, extant))

	def nodes_del_time(self, branch, turn, tick):
		self._flush_nodes()
		self.call_one("nodes_del_time", branch, turn, tick)
		self._btts.discard((branch, turn, tick))

	def nodes_dump(self) -> Iterator[NodeRowType]:
		"""Dump the entire contents of the nodes table."""
		self._flush_nodes()
		unpack = self.unpack
		for graph, node, branch, turn, tick, extant in self.call_one(
			"nodes_dump"
		):
			yield (
				unpack(graph),
				unpack(node),
				branch,
				turn,
				tick,
				bool(extant),
			)

	def load_nodes(
		self, graph, branch, turn_from, tick_from, turn_to=None, tick_to=None
	) -> Iterator[NodeRowType]:
		if (turn_to is None) ^ (tick_to is None):
			raise TypeError("I need both or neither of turn_to and tick_to")
		self._flush_nodes()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.call_one(
				"load_nodes_tick_to_end",
				pack(graph),
				branch,
				turn_from,
				turn_from,
				tick_from,
			)
		else:
			it = self.call_one(
				"load_nodes_tick_to_tick",
				pack(graph),
				branch,
				turn_from,
				turn_from,
				tick_from,
				turn_to,
				turn_to,
				tick_to,
			)
		for node, turn, tick, extant in it:
			yield graph, unpack(node), branch, turn, tick, extant

	def node_val_dump(self) -> Iterator[NodeValRowType]:
		"""Yield the entire contents of the node_val table."""
		self._flush_node_val()
		unpack = self.unpack
		for graph, node, key, branch, turn, tick, value in self.call_one(
			"node_val_dump"
		):
			yield (
				unpack(graph),
				unpack(node),
				unpack(key),
				branch,
				turn,
				tick,
				unpack(value),
			)

	def load_node_val(
		self, graph, branch, turn_from, tick_from, turn_to=None, tick_to=None
	) -> Iterator[NodeValRowType]:
		if (turn_to is None) ^ (tick_to is None):
			raise TypeError("I need both or neither of turn_to and tick_to")
		self._flush_node_val()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.call_one(
				"load_node_val_tick_to_end",
				pack(graph),
				branch,
				turn_from,
				turn_from,
				tick_from,
			)
		else:
			it = self.call_one(
				"load_node_val_tick_to_tick",
				pack(graph),
				branch,
				turn_from,
				turn_from,
				tick_from,
				turn_to,
				turn_to,
				tick_to,
			)
		for node, key, turn, tick, value in it:
			yield (
				graph,
				unpack(node),
				unpack(key),
				branch,
				turn,
				tick,
				unpack(value),
			)

	def _flush_node_val(self):
		if not self._nodevals2set:
			return
		pack = self.pack
		self.call_many(
			"node_val_insert",
			(
				(
					pack(graph),
					pack(node),
					pack(key),
					branch,
					turn,
					tick,
					pack(value),
				)
				for (
					graph,
					node,
					key,
					branch,
					turn,
					tick,
					value,
				) in self._nodevals2set
			),
		)
		self._nodevals2set = []

	def node_val_set(self, graph, node, key, branch, turn, tick, value):
		"""Set a key-value pair on a node at a specific branch and revision"""
		if (branch, turn, tick) in self._btts:
			raise TimeError
		self._btts.add((branch, turn, tick))
		self._nodevals2set.append(
			(graph, node, key, branch, turn, tick, value)
		)

	def node_val_del_time(self, branch, turn, tick):
		self._flush_node_val()
		self.call_one("node_val_del_time", branch, turn, tick)
		self._btts.discard((branch, turn, tick))

	def edges_dump(self) -> Iterator[EdgeRowType]:
		"""Dump the entire contents of the edges table."""
		self._flush_edges()
		unpack = self.unpack
		for (
			graph,
			orig,
			dest,
			idx,
			branch,
			turn,
			tick,
			extant,
		) in self.call_one("edges_dump"):
			yield (
				unpack(graph),
				unpack(orig),
				unpack(dest),
				idx,
				branch,
				turn,
				tick,
				bool(extant),
			)

	def load_edges(
		self, graph, branch, turn_from, tick_from, turn_to=None, tick_to=None
	) -> Iterator[EdgeRowType]:
		if (turn_to is None) ^ (tick_to is None):
			raise TypeError("I need both or neither of turn_to and tick_to")
		self._flush_edge_val()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.call_one(
				"load_edges_tick_to_end",
				pack(graph),
				branch,
				turn_from,
				turn_from,
				tick_from,
			)
		else:
			it = self.call_one(
				"load_edges_tick_to_tick",
				pack(graph),
				branch,
				turn_from,
				turn_from,
				tick_from,
				turn_to,
				turn_to,
				tick_to,
			)
		for orig, dest, idx, turn, tick, extant in it:
			yield (
				graph,
				unpack(orig),
				unpack(dest),
				idx,
				branch,
				turn,
				tick,
				extant,
			)

	def _pack_edge2set(self, tup):
		graph, orig, dest, idx, branch, turn, tick, extant = tup
		pack = self.pack
		return (
			pack(graph),
			pack(orig),
			pack(dest),
			idx,
			branch,
			turn,
			tick,
			extant,
		)

	def _flush_edges(self):
		start = monotonic()
		if not self._edges2set:
			return
		self.call_many(
			"edges_insert", map(self._pack_edge2set, self._edges2set)
		)
		self._edges2set = []
		QueryEngine.flush_edges_t += monotonic() - start

	def exist_edge(self, graph, orig, dest, idx, branch, turn, tick, extant):
		"""Declare whether or not this edge exists."""
		if (branch, turn, tick) in self._btts:
			raise TimeError
		self._btts.add((branch, turn, tick))
		self._edges2set.append(
			(graph, orig, dest, idx, branch, turn, tick, extant)
		)

	def edges_del_time(self, branch, turn, tick):
		self._flush_edges()
		self.call_one("edges_del_time", branch, turn, tick)
		self._btts.discard((branch, turn, tick))

	def edge_val_dump(self) -> Iterator[EdgeValRowType]:
		"""Yield the entire contents of the edge_val table."""
		self._flush_edge_val()
		unpack = self.unpack
		for (
			graph,
			orig,
			dest,
			idx,
			key,
			branch,
			turn,
			tick,
			value,
		) in self.call_one("edge_val_dump"):
			yield (
				unpack(graph),
				unpack(orig),
				unpack(dest),
				idx,
				unpack(key),
				branch,
				turn,
				tick,
				unpack(value),
			)

	def load_edge_val(
		self, graph, branch, turn_from, tick_from, turn_to=None, tick_to=None
	) -> Iterator[EdgeValRowType]:
		if (turn_to is None) ^ (tick_to is None):
			raise TypeError("I need both or neither of turn_to and tick_to")
		self._flush_edge_val()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.call_one(
				"load_edge_val_tick_to_end",
				pack(graph),
				branch,
				turn_from,
				turn_from,
				tick_from,
			)
		else:
			it = self.call_one(
				"load_edge_val_tick_to_tick",
				pack(graph),
				branch,
				turn_from,
				turn_from,
				tick_from,
				turn_to,
				turn_to,
				tick_to,
			)
		for orig, dest, idx, key, turn, tick, value in it:
			yield (
				graph,
				unpack(orig),
				unpack(dest),
				idx,
				unpack(key),
				branch,
				turn,
				tick,
				unpack(value),
			)

	def _pack_edgeval2set(self, tup):
		graph, orig, dest, idx, key, branch, turn, tick, value = tup
		pack = self.pack
		return (
			pack(graph),
			pack(orig),
			pack(dest),
			idx,
			pack(key),
			branch,
			turn,
			tick,
			pack(value),
		)

	def _flush_edge_val(self):
		if not self._edgevals2set:
			return
		self.call_many(
			"edge_val_insert", map(self._pack_edgeval2set, self._edgevals2set)
		)
		self._edgevals2set = []

	def edge_val_set(
		self, graph, orig, dest, idx, key, branch, turn, tick, value
	):
		"""Set this key of this edge to this value."""
		if (branch, turn, tick) in self._btts:
			raise TimeError
		self._btts.add((branch, turn, tick))
		self._edgevals2set.append(
			(graph, orig, dest, idx, key, branch, turn, tick, value)
		)

	def edge_val_del_time(self, branch, turn, tick):
		self._flush_edge_val()
		self.call_one("edge_val_del_time", branch, turn, tick)
		self._btts.discard((branch, turn, tick))

	def plans_dump(self):
		return self.call_one("plans_dump")

	def plans_insert(self, plan_id, branch, turn, tick):
		return self.call_one("plans_insert", plan_id, branch, turn, tick)

	def plans_insert_many(self, many):
		return self.call_many("plans_insert", many)

	def plan_ticks_insert(self, plan_id, turn, tick):
		return self.call_one("plan_ticks_insert", plan_id, turn, tick)

	def plan_ticks_insert_many(self, many):
		return self.call_many("plan_ticks_insert", many)

	def plan_ticks_dump(self):
		return self.call_one("plan_ticks_dump")

	def flush(self):
		"""Put all pending changes into the SQL transaction."""
		pack = self.pack
		put = self._inq.put
		if self._nodes2set:
			put(
				(
					"silent",
					"many",
					"nodes_insert",
					[
						(
							pack(graph),
							pack(node),
							branch,
							turn,
							tick,
							bool(extant),
						)
						for (
							graph,
							node,
							branch,
							turn,
							tick,
							extant,
						) in self._nodes2set
					],
				)
			)
			self._nodes2set = []
		if self._edges2set:
			put(
				(
					"silent",
					"many",
					"edges_insert",
					list(map(self._pack_edge2set, self._edges2set)),
				)
			)
			self._edges2set = []
		if self._graphvals2set:
			put(
				(
					"silent",
					"many",
					"graph_val_insert",
					[
						(
							pack(graph),
							pack(key),
							branch,
							turn,
							tick,
							pack(value),
						)
						for (
							graph,
							key,
							branch,
							turn,
							tick,
							value,
						) in self._graphvals2set
					],
				)
			)
			self._graphvals2set = []
		if self._nodevals2set:
			put(
				(
					"silent",
					"many",
					"node_val_insert",
					[
						(
							pack(graph),
							pack(node),
							pack(key),
							branch,
							turn,
							tick,
							pack(value),
						)
						for (
							graph,
							node,
							key,
							branch,
							turn,
							tick,
							value,
						) in self._nodevals2set
					],
				)
			)
			self._nodevals2set = []
		if self._edgevals2set:
			put(
				(
					"silent",
					"many",
					"edge_val_insert",
					list(map(self._pack_edgeval2set, self._edgevals2set)),
				)
			)
			self._edgevals2set = []

	def commit(self):
		"""Commit the transaction"""
		self.flush()
		self._inq.put("commit")

	def close(self):
		"""Commit the transaction, then close the connection"""
		self.flush()
		assert self.echo("flushed") == "flushed"
		self._inq.put("shutdown")
		self._holder.existence_lock.acquire()
		self._holder.existence_lock.release()
		self._t.join()

	def initdb(self):
		with self._holder.lock:
			self._inq.put("initdb")
			ret = self._outq.get()
			if isinstance(ret, Exception):
				raise ret
		self.globl = GlobalKeyValueStore(self)
		if "main_branch" not in self.globl:
			self.globl["main_branch"] = "trunk"
		if "branch" not in self.globl:
			self.globl["branch"] = self.globl["main_branch"]
		if "turn" not in self.globl:
			self.globl["turn"] = 0
		if "tick" not in self.globl:
			self.globl["tick"] = 0

	def truncate_all(self):
		"""Delete all data from every table"""
		for table in self.tables:
			try:
				self.call_one("truncate_" + table)
			except OperationalError:
				pass  # table wasn't created yet
		self.commit()
