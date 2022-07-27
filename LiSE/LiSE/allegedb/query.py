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
import sqlite3

from . import wrap

wrappath = os.path.dirname(wrap.__file__)
alchemyIntegError = None
alchemyOperationalError = None
try:
	import sqlalchemy.exc

	IntegrityError = (sqlalchemy.exc.IntegrityError, sqlite3.IntegrityError)
	OperationalError = (sqlalchemy.exc.OperationalError,
						sqlite3.OperationalError)
except ImportError:
	IntegrityError = sqlite3.IntegrityError
	OperationalError = sqlite3.OperationalError

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
		return self._cache[k]

	def __setitem__(self, k, v):
		self.qe.global_set(k, v)
		self._cache[k] = v

	def __delitem__(self, k):
		del self._cache[k]
		self.qe.global_del(k)


class ConnectionHolder:

	def __init__(self, dbstring, connect_args, alchemy, inq, outq, fn, tables):
		self.lock = Lock()
		self.existence_lock = Lock()
		self.existence_lock.acquire()
		self._dbstring = dbstring
		self._connect_args = connect_args
		self._alchemy = alchemy
		self._fn = fn
		self.inq = inq
		self.outq = outq
		self.tables = tables

	def commit(self):
		if hasattr(self, 'transaction') and self.transaction.is_active:
			self.transaction.commit()
		elif hasattr(self, 'connection'):
			self.connection.commit()

	def init_table(self, tbl):
		return self.sql('create_{}'.format(tbl))

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
				s.format(**kwargs) if kwargs else s, args)

	def sqlmany(self, stringname, args):
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

	def run(self):
		alchemy = self._alchemy
		dbstring = self._dbstring
		connect_args = self._connect_args

		def lite_init(dbstring):
			from json import load
			with open(self._fn, "rb") as strf:
				self.strings = load(strf)
			assert not isinstance(dbstring, sqlite3.Connection)
			if dbstring.startswith('sqlite:///'):
				dbstring = dbstring[10:]
			self.connection = sqlite3.connect(dbstring)

		if alchemy:
			try:
				from sqlalchemy import create_engine
				from sqlalchemy.engine.base import Engine
				from sqlalchemy.exc import ArgumentError
				from sqlalchemy.pool import NullPool
				from .alchemy import Alchemist
				if isinstance(dbstring, Engine):
					self.engine = dbstring
				else:
					try:
						self.engine = create_engine(dbstring,
													connect_args=connect_args,
													poolclass=NullPool)
					except ArgumentError:
						self.engine = create_engine('sqlite:///' + dbstring,
													connect_args=connect_args,
													poolclass=NullPool)
				self.alchemist = Alchemist(self.engine)
				self.transaction = self.alchemist.conn.begin()
			except ImportError:
				self._alchemy = False
				lite_init(dbstring)
		else:
			lite_init(dbstring)
		while True:
			inst = self.inq.get()
			if inst == 'shutdown':
				self.commit()
				if hasattr(self, 'alchemist'):
					self.alchemist.conn.close()
					self.engine.dispose()
				else:
					self.connection.close()
				self.existence_lock.release()
				return
			if inst == 'commit':
				self.commit()
				continue
			if inst == 'initdb':
				self.outq.put(self.initdb())
				continue
			silent = False
			if inst[0] == 'silent':
				inst = inst[1:]
				silent = True
			if inst[0] == 'one':
				try:
					res = self.sql(inst[1], *inst[2], **inst[3])
					if not silent:
						if hasattr(res, 'returns_rows'):
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
			elif inst[0] != 'many':
				raise ValueError(f"Invalid instruction: {inst[0]}")
			else:
				try:
					res = self.sqlmany(inst[1], inst[2])
					if not silent:
						if hasattr(res, 'returns_rows'):
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

	def initdb(self):
		"""Create tables and indices as needed."""
		for table in ('branches', 'turns', 'graphs', 'graph_val', 'nodes',
						'node_val', 'edges', 'edge_val', 'plans', 'plan_ticks',
						'keyframes', 'global'):
			try:
				ret = self.init_table(table)
			except OperationalError:
				pass
			except Exception as ex:
				return ex
		self.commit()


class QueryEngine(object):
	"""Wrapper around either a DBAPI2.0 connection or an
    Alchemist. Provides methods to run queries using either.

    """
	flush_edges_t = 0
	holder_cls = ConnectionHolder
	tables = ('global', 'branches', 'turns', 'graphs', 'keyframes',
				'graph_val', 'nodes', 'node_val', 'edges', 'edge_val', 'plans',
				'plan_ticks', 'universals')

	def __init__(self,
					dbstring,
					connect_args,
					alchemy,
					strings_filename: str = None,
					pack=None,
					unpack=None):
		"""If ``alchemy`` is True and ``dbstring`` is a legit database URI,
        instantiate an Alchemist and start a transaction with
        it. Otherwise use sqlite3.

        You may pass an already created sqlalchemy :class:`Engine`
        object in place of ``dbstring`` if you wish. I'll still create
        my own transaction though.

        """
		dbstring = dbstring or 'sqlite:///:memory:'
		self._inq = Queue()
		self._outq = Queue()
		if strings_filename is None:
			strings_filename = os.path.join(
				os.path.dirname(os.path.abspath(__file__)), "sqlite.json")
		if not os.path.exists(strings_filename) or os.path.isdir(
			strings_filename):
			raise FileNotFoundError("No SQL in JSON found at " +
									strings_filename)
		self._holder = self.holder_cls(dbstring, connect_args, alchemy,
										self._inq, self._outq,
										strings_filename, self.tables)

		if unpack is None:
			from ast import literal_eval as unpack
		self.pack = pack or repr
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

	def sql(self, string, *args, **kwargs):
		__doc__ = ConnectionHolder.sql.__doc__
		with self._holder.lock:
			self._inq.put(('one', string, args, kwargs))
			ret = self._outq.get()
		if isinstance(ret, Exception):
			raise ret
		return ret

	def sqlmany(self, string, args):
		__doc__ = ConnectionHolder.sqlmany.__doc__
		with self._holder.lock:
			self._inq.put(('many', string, args))
			ret = self._outq.get()
		if isinstance(ret, Exception):
			raise ret
		return ret

	def have_graph(self, graph):
		"""Return whether I have a graph by this name."""
		graph = self.pack(graph)
		return bool(self.sql('graphs_named', graph)[0][0])

	def new_graph(self, graph, typ):
		"""Declare a new graph by this name of this type."""
		graph = self.pack(graph)
		return self.sql('graphs_insert', graph, typ)

	def keyframes_insert(self, graph, branch, turn, tick, nodes, edges,
							graph_val):
		graph, nodes, edges, graph_val = map(
			self.pack, (graph, nodes, edges, graph_val))
		return self.sql('keyframes_insert', graph, branch, turn, tick, nodes,
						edges, graph_val)

	def keyframes_insert_many(self, many):
		pack = self.pack
		return self.sqlmany('keyframes_insert', [
			(pack(graph), branch, turn, tick, pack(nodes), pack(edges),
				pack(graph_val))
			for (graph, branch, turn, tick, nodes, edges, graph_val) in many
		])

	def keyframes_dump(self):
		unpack = self.unpack
		for (graph, branch, turn, tick, nodes, edges,
				graph_val) in self.sql('keyframes_dump'):
			yield unpack(graph), branch, turn, tick, unpack(nodes), unpack(
				edges), unpack(graph_val)

	def keyframes_list(self):
		unpack = self.unpack
		for (graph, branch, turn, tick) in self.sql('keyframes_list'):
			yield unpack(graph), branch, turn, tick

	def get_keyframe(self, graph, branch, turn, tick):
		unpack = self.unpack
		stuff = self.sql('get_keyframe', self.pack(graph), branch, turn, tick)
		if not stuff:
			return
		nodes, edges, graph_val = stuff[0]
		return unpack(nodes), unpack(edges), unpack(graph_val)

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
		return self.sql('graph_type', graph)[0][0]

	def have_branch(self, branch):
		"""Return whether the branch thus named exists in the database."""
		return bool(self.sql('ctbranch', branch)[0][0])

	def all_branches(self):
		"""Return all the branch data in tuples of (branch, parent,
        parent_turn).

        """
		return self.sql('branches_dump')

	def global_get(self, key):
		"""Return the value for the given key in the ``globals`` table."""
		key = self.pack(key)
		r = self.sql('global_get', key)[0]
		if r is None:
			raise KeyError("Not set")
		return self.unpack(r[0])

	def global_items(self):
		"""Iterate over (key, value) pairs in the ``globals`` table."""
		unpack = self.unpack
		dumped = self.sql('global_dump')
		for (k, v) in dumped:
			yield (unpack(k), unpack(v))

	def get_branch(self):
		v = self.sql('global_get', self.pack('branch'))[0]
		if v is None:
			return 'trunk'
		return self.unpack(v[0])

	def get_turn(self):
		v = self.sql('global_get', self.pack('turn'))[0]
		if v is None:
			return 0
		return self.unpack(v[0])

	def get_tick(self):
		v = self.sql('global_get', self.pack('tick'))[0]
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
		return self.sql('branches_insert', branch, parent, parent_turn,
						parent_tick, parent_turn, parent_tick)

	def update_branch(self, branch, parent, parent_turn, parent_tick, end_turn,
						end_tick):
		return self.sql('update_branches', parent, parent_turn, parent_tick,
						end_turn, end_tick, branch)

	def set_branch(self, branch, parent, parent_turn, parent_tick, end_turn,
					end_tick):
		try:
			self.sql('branches_insert', branch, parent, parent_turn,
						parent_tick, end_turn, end_tick)
		except IntegrityError:
			self.update_branch(branch, parent, parent_turn, parent_tick,
								end_turn, end_tick)

	def new_turn(self, branch, turn, end_tick=0, plan_end_tick=0):
		return self.sql('turns_insert', branch, turn, end_tick, plan_end_tick)

	def update_turn(self, branch, turn, end_tick, plan_end_tick):
		return self.sql('update_turns', end_tick, plan_end_tick, branch, turn)

	def set_turn(self, branch, turn, end_tick, plan_end_tick):
		try:
			return self.sql('turns_insert', branch, turn, end_tick,
							plan_end_tick)
		except IntegrityError:
			return self.sql('update_turns', end_tick, plan_end_tick, branch,
							turn)

	def set_turn_completed(self, branch, turn):
		try:
			return self.sql('turns_completed_insert', branch, turn)
		except IntegrityError:
			return self.sql('turns_completed_update', turn, branch)

	def turns_dump(self):
		return self.sql('turns_dump')

	def graph_val_dump(self) -> Iterator[GraphValRowType]:
		"""Yield the entire contents of the graph_val table."""
		self._flush_graph_val()
		unpack = self.unpack
		for (graph, key, branch, turn, tick,
				value) in self.sql('graph_val_dump'):
			yield (unpack(graph), unpack(key), branch, turn, tick,
					unpack(value))

	def load_graph_val(self,
						graph,
						branch,
						turn_from,
						tick_from,
						turn_to=None,
						tick_to=None) -> Iterator[GraphValRowType]:
		if (turn_to is None) ^ (tick_to is None):
			raise TypeError("I need both or neither of turn_to and tick_to")
		self._flush_graph_val()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.sql('load_graph_val_tick_to_end', pack(graph), branch,
							turn_from, turn_from, tick_from)
		else:
			it = self.sql('load_graph_val_tick_to_tick', pack(graph), branch,
							turn_from, turn_from, tick_from, turn_to, turn_to,
							tick_to)
		for (key, turn, tick, value) in it:
			yield graph, unpack(key), branch, turn, tick, unpack(value)

	def _flush_graph_val(self):
		"""Send all new and changed graph values to the database."""
		if not self._graphvals2set:
			return
		pack = self.pack
		self.sqlmany(
			'graph_val_insert',
			((pack(graph), pack(key), branch, turn, tick, pack(value))
				for (graph, key, branch, turn, tick,
						value) in self._graphvals2set))
		self._graphvals2set = []

	def graph_val_set(self, graph, key, branch, turn, tick, value):
		if (branch, turn, tick) in self._btts:
			raise TimeError
		self._btts.add((branch, turn, tick))
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
		pack = self.pack
		self.sqlmany(
			'nodes_insert',
			((pack(graph), pack(node), branch, turn, tick, bool(extant))
				for (graph, node, branch, turn, tick,
						extant) in self._nodes2set))
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
		self.sql('nodes_del_time', branch, turn, tick)
		self._btts.discard((branch, turn, tick))

	def nodes_dump(self) -> Iterator[NodeRowType]:
		"""Dump the entire contents of the nodes table."""
		self._flush_nodes()
		unpack = self.unpack
		for (graph, node, branch, turn, tick,
				extant) in self.sql('nodes_dump'):
			yield (unpack(graph), unpack(node), branch, turn, tick,
					bool(extant))

	def load_nodes(self,
					graph,
					branch,
					turn_from,
					tick_from,
					turn_to=None,
					tick_to=None) -> Iterator[NodeRowType]:
		if (turn_to is None) ^ (tick_to is None):
			raise TypeError("I need both or neither of turn_to and tick_to")
		self._flush_nodes()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.sql('load_nodes_tick_to_end', pack(graph), branch,
							turn_from, turn_from, tick_from)
		else:
			it = self.sql('load_nodes_tick_to_tick', pack(graph), branch,
							turn_from, turn_from, tick_from, turn_to, turn_to,
							tick_to)
		for (node, turn, tick, extant) in it:
			yield graph, unpack(node), branch, turn, tick, extant

	def node_val_dump(self) -> Iterator[NodeValRowType]:
		"""Yield the entire contents of the node_val table."""
		self._flush_node_val()
		unpack = self.unpack
		for (graph, node, key, branch, turn, tick,
				value) in self.sql('node_val_dump'):
			yield (unpack(graph), unpack(node), unpack(key), branch, turn,
					tick, unpack(value))

	def load_node_val(self,
						graph,
						branch,
						turn_from,
						tick_from,
						turn_to=None,
						tick_to=None) -> Iterator[NodeValRowType]:
		if (turn_to is None) ^ (tick_to is None):
			raise TypeError("I need both or neither of turn_to and tick_to")
		self._flush_node_val()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.sql('load_node_val_tick_to_end', pack(graph), branch,
							turn_from, turn_from, tick_from)
		else:
			it = self.sql('load_node_val_tick_to_tick', pack(graph), branch,
							turn_from, turn_from, tick_from, turn_to, turn_to,
							tick_to)
		for (node, key, turn, tick, value) in it:
			yield graph, unpack(node), unpack(key), branch, turn, tick, unpack(
				value)

	def _flush_node_val(self):
		if not self._nodevals2set:
			return
		pack = self.pack
		self.sqlmany('node_val_insert',
						((pack(graph), pack(node), pack(key), branch, turn,
							tick, pack(value))
							for (graph, node, key, branch, turn, tick,
									value) in self._nodevals2set))
		self._nodevals2set = []

	def node_val_set(self, graph, node, key, branch, turn, tick, value):
		"""Set a key-value pair on a node at a specific branch and revision"""
		if (branch, turn, tick) in self._btts:
			raise TimeError
		self._btts.add((branch, turn, tick))
		self._nodevals2set.append(
			(graph, node, key, branch, turn, tick, value))

	def node_val_del_time(self, branch, turn, tick):
		self._flush_node_val()
		self.sql('node_val_del_time', branch, turn, tick)
		self._btts.discard((branch, turn, tick))

	def edges_dump(self) -> Iterator[EdgeRowType]:
		"""Dump the entire contents of the edges table."""
		self._flush_edges()
		unpack = self.unpack
		for (graph, orig, dest, idx, branch, turn, tick,
				extant) in self.sql('edges_dump'):
			yield (unpack(graph), unpack(orig), unpack(dest), idx, branch,
					turn, tick, bool(extant))

	def load_edges(self,
					graph,
					branch,
					turn_from,
					tick_from,
					turn_to=None,
					tick_to=None) -> Iterator[EdgeRowType]:
		if (turn_to is None) ^ (tick_to is None):
			raise TypeError("I need both or neither of turn_to and tick_to")
		self._flush_edge_val()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.sql('load_edges_tick_to_end', pack(graph), branch,
							turn_from, turn_from, tick_from)
		else:
			it = self.sql('load_edges_tick_to_tick', pack(graph), branch,
							turn_from, turn_from, tick_from, turn_to, turn_to,
							tick_to)
		for (orig, dest, idx, turn, tick, extant) in it:
			yield graph, unpack(orig), unpack(
				dest), idx, branch, turn, tick, extant

	def _pack_edge2set(self, tup):
		graph, orig, dest, idx, branch, turn, tick, extant = tup
		pack = self.pack
		return pack(graph), pack(orig), pack(
			dest), idx, branch, turn, tick, extant

	def _flush_edges(self):
		start = monotonic()
		if not self._edges2set:
			return
		self.sqlmany('edges_insert', map(self._pack_edge2set, self._edges2set))
		self._edges2set = []
		QueryEngine.flush_edges_t += monotonic() - start

	def exist_edge(self, graph, orig, dest, idx, branch, turn, tick, extant):
		"""Declare whether or not this edge exists."""
		if (branch, turn, tick) in self._btts:
			raise TimeError
		self._btts.add((branch, turn, tick))
		self._edges2set.append(
			(graph, orig, dest, idx, branch, turn, tick, extant))

	def edges_del_time(self, branch, turn, tick):
		self._flush_edges()
		self.sql('edges_del_time', branch, turn, tick)
		self._btts.discard((branch, turn, tick))

	def edge_val_dump(self) -> Iterator[EdgeValRowType]:
		"""Yield the entire contents of the edge_val table."""
		self._flush_edge_val()
		unpack = self.unpack
		for (graph, orig, dest, idx, key, branch, turn, tick,
				value) in self.sql('edge_val_dump'):
			yield (unpack(graph), unpack(orig), unpack(dest), idx, unpack(key),
					branch, turn, tick, unpack(value))

	def load_edge_val(self,
						graph,
						branch,
						turn_from,
						tick_from,
						turn_to=None,
						tick_to=None) -> Iterator[EdgeValRowType]:
		if (turn_to is None) ^ (tick_to is None):
			raise TypeError("I need both or neither of turn_to and tick_to")
		self._flush_edge_val()
		pack = self.pack
		unpack = self.unpack
		if turn_to is None:
			it = self.sql('load_edge_val_tick_to_end', pack(graph), branch,
							turn_from, turn_from, tick_from)
		else:
			it = self.sql('load_edge_val_tick_to_tick', pack(graph), branch,
							turn_from, turn_from, tick_from, turn_to, turn_to,
							tick_to)
		for (orig, dest, idx, key, turn, tick, value) in it:
			yield graph, unpack(orig), unpack(dest), idx, unpack(
				key), branch, turn, tick, unpack(value)

	def _pack_edgeval2set(self, tup):
		graph, orig, dest, idx, key, branch, turn, tick, value = tup
		pack = self.pack
		return pack(graph), pack(orig), pack(dest), idx, pack(
			key), branch, turn, tick, pack(value)

	def _flush_edge_val(self):
		if not self._edgevals2set:
			return
		self.sqlmany('edge_val_insert',
						map(self._pack_edgeval2set, self._edgevals2set))
		self._edgevals2set = []

	def edge_val_set(self, graph, orig, dest, idx, key, branch, turn, tick,
						value):
		"""Set this key of this edge to this value."""
		if (branch, turn, tick) in self._btts:
			raise TimeError
		self._btts.add((branch, turn, tick))
		self._edgevals2set.append(
			(graph, orig, dest, idx, key, branch, turn, tick, value))

	def edge_val_del_time(self, branch, turn, tick):
		self._flush_edge_val()
		self.sql('edge_val_del_time', branch, turn, tick)
		self._btts.discard((branch, turn, tick))

	def plans_dump(self):
		return self.sql('plans_dump')

	def plans_insert(self, plan_id, branch, turn, tick):
		return self.sql('plans_insert', plan_id, branch, turn, tick)

	def plans_insert_many(self, many):
		return self.sqlmany('plans_insert', many)

	def plan_ticks_insert(self, plan_id, turn, tick):
		return self.sql('plan_ticks_insert', plan_id, turn, tick)

	def plan_ticks_insert_many(self, many):
		return self.sqlmany('plan_ticks_insert', many)

	def plan_ticks_dump(self):
		return self.sql('plan_ticks_dump')

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
		self._inq.put('commit')

	def close(self):
		"""Commit the transaction, then close the connection"""
		self._inq.put('shutdown')
		self._holder.existence_lock.acquire()
		self._holder.existence_lock.release()
		self._t.join()

	def initdb(self):
		with self._holder.lock:
			self._inq.put('initdb')
			ret = self._outq.get()
			if isinstance(ret, Exception):
				raise ret
		self.globl = GlobalKeyValueStore(self)
		if 'branch' not in self.globl:
			self.globl['branch'] = 'trunk'
		if 'turn' not in self.globl:
			self.globl['turn'] = 0
		if 'tick' not in self.globl:
			self.globl['tick'] = 0

	def truncate_all(self):
		"""Delete all data from every table"""
		for table in self.tables:
			try:
				self.sql('truncate_' + table)
			except OperationalError:
				pass  # table wasn't created yet
		self.commit()
