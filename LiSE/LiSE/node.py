# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
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
"""The nodes of LiSE's character graphs.

Every node that actually exists is either a Place or a Thing, but they
have a lot in common.

"""

from __future__ import annotations
from collections.abc import Mapping, ValuesView, Set
from typing import Optional, Union, Iterator, List

import networkx as nx
from networkx import shortest_path, shortest_path_length

from .allegedb import graph, Key, HistoricKeyError

from .util import getatt, AbstractCharacter
from .query import StatusAlias
from . import rule
from .exc import AmbiguousUserError, TravelException


class UserMapping(Mapping):
	"""A mapping of the characters that have a particular node as a unit.

	Getting characters from here isn't any better than getting them from
	the engine direct, but with this you can do things like use the
	.get() method to get a character if it's a user and otherwise
	get something else; or test whether the character's name is in
	the keys; and so on.

	"""

	__slots__ = ["node"]

	def __init__(self, node) -> None:
		"""Store the node"""
		self.node = node

	engine = getatt("node.engine")

	def _user_names(self) -> Iterator[Key]:
		node = self.node
		engine = self.engine
		charn = node.character.name
		nn = node.name
		cache = engine._unitness_cache.user_order
		if charn not in cache or nn not in cache[charn]:
			return
		cache = cache[charn][nn]
		seen = set()
		for user in cache:
			if user in seen:
				continue
			for branch, turn, tick in engine._iter_parent_btt():
				if branch in cache[user]:
					branchd = cache[user][branch]
					if turn in branchd:
						if branchd[turn].rev_gettable(tick):
							if branchd[turn][tick]:
								yield user
							seen.add(user)
							break
					elif branchd.rev_gettable(turn):
						turnd = branchd[turn]
						if turnd.final():
							yield user
						seen.add(user)
						break

	@property
	def only(self) -> Node:
		"""If there's only one unit, return it.

		Otherwise, raise ``AmbiguousUserError``, a type of ``AttributeError``.

		"""
		if len(self) != 1:
			raise AmbiguousUserError("No users, or more than one")
		return next(iter(self.values()))

	def __iter__(self) -> Iterator[Key]:
		yield from self._user_names()

	def __len__(self) -> int:
		n = 0
		for user in self._user_names():
			n += 1
		return n

	def __bool__(self) -> bool:
		for user in self._user_names():
			return True
		return False

	def __contains__(self, item) -> bool:
		chara = self.engine.character[item]
		nn = self.node.name
		charn = self.node.character.name
		return charn in chara.unit and nn in chara.unit[charn]

	def __getitem__(self, k) -> AbstractCharacter:
		ret = self.engine.character[k]
		node = self.node
		charn = node.character.name
		nn = node.name
		avatar = ret.unit
		if charn not in avatar or nn not in avatar[charn]:
			raise KeyError("{} not used by {}".format(self.node.name, k))
		return ret


class NodeContentValues(ValuesView):
	_mapping: "NodeContent"

	def __iter__(self) -> Iterator[Thing]:
		node = self._mapping.node
		nodem = node.character.node
		try:
			conts = node.engine._node_contents(node.character.name, node.name)
		except KeyError:
			return
		for name in conts:
			if name not in nodem:
				return
			yield nodem[name]

	def __contains__(self, item) -> bool:
		try:
			return item.location == self._mapping.node
		except AttributeError:
			return False


class NodeContent(Mapping):
	__slots__ = ("node",)

	def __init__(self, node) -> None:
		self.node = node

	def __iter__(self) -> Iterator:
		try:
			it = self.node.engine._node_contents_cache.retrieve(
				self.node.character.name,
				self.node.name,
				*self.node.engine._btt(),
			)
		except KeyError:
			return
		yield from it

	def __len__(self) -> int:
		try:
			return len(
				self.node.engine._node_contents_cache.retrieve(
					self.node.character.name,
					self.node.name,
					*self.node.engine._btt(),
				)
			)
		except KeyError:
			return 0

	def __contains__(self, item) -> bool:
		try:
			return self.node.character.thing[item].location == self.node
		except KeyError:
			return False

	def __getitem__(self, item) -> Thing:
		if item not in self:
			raise KeyError
		return self.node.character.thing[item]

	def values(self) -> NodeContentValues:
		return NodeContentValues(self)


class DestsValues(ValuesView):
	_mapping: "Dests"

	def __contains__(self, item) -> bool:
		_, name = self._mapping._pn
		return item.origin.name == name


class Dests(Mapping):
	__slots__ = ("_ecnb", "_pn")

	def __init__(self, node) -> None:
		name = node.name
		character = node.character
		engine = node.engine
		self._pn = (character.portal, name)
		self._ecnb = (engine._edges_cache, character.name, name, engine._btt)

	def __iter__(self) -> Iterator:
		edges_cache, charname, name, btt = self._ecnb
		return edges_cache.iter_successors(charname, name, *btt())

	def __len__(self) -> int:
		edges_cache, charname, name, btt = self._ecnb
		return edges_cache.count_successors(charname, name, *btt())

	def __contains__(self, item) -> bool:
		edges_cache, charname, name, btt = self._ecnb
		return edges_cache.has_successor(charname, name, item, *btt())

	def __getitem__(self, item) -> "LiSE.portal.Portal":
		portal, name = self._pn
		return portal[name][item]

	def values(self) -> DestsValues:
		return DestsValues(self)


class OrigsValues(ValuesView):
	_mapping: "Origs"

	def __contains__(self, item) -> bool:
		_, name = self._mapping._pn
		return item.destination.name == name


class Origs(Mapping):
	__slots__ = ("_pn", "_ecnb")

	def __init__(self, node) -> None:
		name = node.name
		character = node.character
		engine = node.engine
		self._pn = (character.portal, name)
		self._ecnb = (engine._edges_cache, character.name, name, engine._btt)

	def __iter__(self) -> Iterator[Node]:
		edges_cache, charname, name, btt = self._ecnb
		return edges_cache.iter_predecessors(charname, name, *btt())

	def __contains__(self, item) -> bool:
		edges_cache, charname, name, btt = self._ecnb
		return edges_cache.has_predecessor(charname, name, item, *btt())

	def __len__(self) -> int:
		edges_cache, charname, name, btt = self._ecnb
		return edges_cache.count_predecessors(charname, name, *btt())

	def __getitem__(self, item) -> Node:
		if item not in self:
			raise KeyError
		portal, name = self._pn
		return portal[item][name]

	def values(self) -> OrigsValues:
		return OrigsValues(self)


class Portals(Set):
	__slots__ = ("_pn", "_pecnb")

	def __init__(self, node) -> None:
		name = node.name
		character = node.character
		engine = node.engine
		self._pn = (character.portal, name)
		self._pecnb = (
			engine._get_edge,
			engine._edges_cache,
			character,
			character.name,
			name,
			engine._btt,
		)

	def __contains__(self, x) -> bool:
		_, edges_cache, _, charname, name, btt_f = self._pecnb
		btt = btt_f()
		return edges_cache.has_predecessor(
			charname, name, x, *btt
		) or edges_cache.has_successor(charname, name, x, *btt)

	def __len__(self) -> int:
		_, edges_cache, _, charname, name, btt_f = self._pecnb
		btt = btt_f()
		return edges_cache.count_predecessors(
			charname, name, *btt
		) + edges_cache.count_successors(charname, name, *btt)

	def __iter__(self) -> Iterator["LiSE.portal.Portal"]:
		get_edge, edges_cache, character, charname, name, btt_f = self._pecnb
		btt = btt_f()
		for dest in edges_cache.iter_successors(charname, name, *btt):
			yield get_edge(character, name, dest, 0)
		for orig in edges_cache.iter_predecessors(charname, name, *btt):
			yield get_edge(character, orig, name, 0)


class NeighborValues(ValuesView):
	_mapping: "NeighborMapping"

	def __contains__(self, item) -> bool:
		return item.name in self._mapping


class NeighborMapping(Mapping):
	__slots__ = ("_nn", "_ecnb")

	def __init__(self, node: Node) -> None:
		name = node.name
		character = node.character
		engine = node.engine
		self._nn = (character.node, name)
		self._ecnb = (engine._edges_cache, character.name, name, engine._btt)

	def __iter__(self) -> Iterator[Node]:
		edges_cache, charname, name, btt = self._ecnb
		seen = set()
		for succ in edges_cache.iter_successors(charname, name, *btt()):
			yield succ
			seen.add(succ)
		for pred in edges_cache.iter_predecessors(charname, name, *btt()):
			if pred in seen:
				continue
			yield pred
			seen.add(pred)

	def __contains__(self, item) -> bool:
		edges_cache, charname, name, btt = self._ecnb
		return edges_cache.has_predecessor(
			charname, name, item, *btt()
		) or edges_cache.has_successor(charname, name, item, *btt())

	def __len__(self) -> int:
		return len(set(iter(self)))

	def __getitem__(self, item) -> Node:
		node, name = self._nn
		if item not in self:
			raise KeyError(f"{item} is not a neighbor of {name}")
		return node[item]

	def values(self) -> NeighborValues:
		return NeighborValues(self)


class Node(graph.Node, rule.RuleFollower):
	"""The fundamental graph component, which portals go between.

	Every LiSE node is either a thing or a place. They share in common
	the abilities to follow rules; to be connected by portals; and to
	contain things.

	This is truthy if it exists, falsy if it's been deleted.

	"""

	__slots__ = ("_real_rule_mapping",)
	character = getatt("graph")
	name = getatt("node")
	no_unwrap = True
	_extra_keys = {
		"name",
	}

	def _get_rule_mapping(self):
		return rule.RuleMapping(self.db, self.rulebook)

	def _get_rulebook_name(self):
		try:
			return self.engine._nodes_rulebooks_cache.retrieve(
				self.character.name, self.name, *self.engine._btt()
			)
		except KeyError:
			return self.character.name, self.name

	def _get_rulebook(self):
		return rule.RuleBook(self.engine, self._get_rulebook_name())

	def _set_rulebook_name(self, rulebook):
		character = self.character.name
		node = self.name
		cache = self.engine._nodes_rulebooks_cache
		try:
			if rulebook == cache.retrieve(
				character, node, *self.engine._btt()
			):
				return
		except KeyError:
			pass
		branch, turn, tick = self.engine._nbtt()
		cache.store(character, node, branch, turn, tick, rulebook)
		self.engine.query.set_node_rulebook(
			character, node, branch, turn, tick, rulebook
		)

	successor = succ = adj = edge = getatt("portal")
	predecessor = pred = getatt("preportal")
	engine = getatt("db")

	@property
	def user(self) -> UserMapping:
		__doc__ = UserMapping.__doc__
		return UserMapping(self)

	def __init__(self, character, name):
		super().__init__(character, name)
		self.db = character.engine

	@property
	def neighbor(self) -> NeighborMapping:
		return NeighborMapping(self)

	def neighbors(self):
		return self.neighbor.values()

	@property
	def portal(self) -> Dests:
		"""A mapping of portals leading out from this node.

		Aliases ``portal``, ``adj``, ``edge``, ``successor``, and ``succ``
		are available.

		"""
		return Dests(self)

	@property
	def preportal(self) -> Origs:
		"""A mapping of portals leading to this node.

		Aliases ``preportal``, ``predecessor`` and ``pred`` are available.

		"""
		return Origs(self)

	def portals(self) -> Portals:
		"""A set-like object of portals connected to this node."""
		return Portals(self)

	@property
	def content(self) -> NodeContent:
		"""A mapping of ``Thing`` objects that are here"""
		return NodeContent(self)

	def contents(self) -> NodeContentValues:
		"""A set-like object containing ``Thing`` objects that are here"""
		return self.content.values()

	def __iter__(self):
		yield from super().__iter__()
		yield from self._extra_keys
		return

	def clear(self) -> None:
		"""Delete all my keys"""
		for key in super().__iter__():
			del self[key]

	def __contains__(self, k):
		"""Handle extra keys, then delegate."""
		return k in self._extra_keys or super().__contains__(k)

	def __setitem__(self, k, v):
		super().__setitem__(k, v)

	def __delitem__(self, k):
		super().__delitem__(k)

	def successors(self) -> Iterator["Place"]:
		"""Iterate over nodes with edges leading from here to there."""
		for port in self.portal.values():
			yield port.destination

	def predecessors(self) -> Iterator["Place"]:
		"""Iterate over nodes with edges leading here from there."""
		for port in self.preportal.values():
			yield port.origin

	def _plain_dest_name(self, dest):
		if isinstance(dest, Node):
			if dest.character != self.character:
				raise ValueError(
					"{} not in {}".format(dest.name, self.character.name)
				)
			return dest.name
		else:
			if dest in self.character.node:
				return dest
			raise ValueError("{} not in {}".format(dest, self.character.name))

	def shortest_path_length(
		self, dest: Union["Key", "Node"], weight: "Key" = None
	) -> int:
		"""Return the length of the path from me to ``dest``.

		Raise ``ValueError`` if ``dest`` is not a node in my character
		or the name of one.

		"""

		return shortest_path_length(
			self.character, self.name, self._plain_dest_name(dest), weight
		)

	def shortest_path(
		self, dest: Union[Key, "Node"], weight: Key = None
	) -> List[Key]:
		"""Return a list of node names leading from me to ``dest``.

		Raise ``ValueError`` if ``dest`` is not a node in my character
		or the name of one.

		"""
		return shortest_path(
			self.character, self.name, self._plain_dest_name(dest), weight
		)

	def path_exists(
		self, dest: Union[Key, "Node"], weight: Key = None
	) -> bool:
		"""Return whether there is a path leading from me to ``dest``.

		With ``weight``, only consider edges that have a stat by the
		given name.

		Raise ``ValueError`` if ``dest`` is not a node in my character
		or the name of one.

		"""
		try:
			return bool(self.shortest_path_length(dest, weight))
		except KeyError:
			return False

	def delete(self) -> None:
		"""Get rid of this, starting now.

		Apart from deleting the node, this also informs all its users
		that it doesn't exist and therefore can't be their unit
		anymore.

		"""
		self.clear()
		for contained in list(self.contents()):
			contained.delete()
		if self.name in self.character.portal:
			del self.character.portal[self.name]
		if self.name in self.character.preportal:
			del self.character.preportal[self.name]
		for user in list(self.user.values()):
			user.remove_unit(self.character.name, self.name)
		branch, turn, tick = self.engine._nbtt()
		self.engine._nodes_cache.store(
			self.character.name, self.name, branch, turn, tick, False
		)
		self.engine.query.exist_node(
			self.character.name, self.name, branch, turn, tick, False
		)
		self.character.node.send(self.character.node, key=self.name, val=None)

	def new_portal(
		self, other: Union[Key, "Node"], **stats
	) -> "LiSE.portal.Portal":
		"""Connect a portal from here to another node, and return it."""
		return self.character.new_portal(
			self.name, getattr(other, "name", other), **stats
		)

	def new_thing(self, name: Key, **stats) -> "Thing":
		"""Create a new thing, located here, and return it."""
		return self.character.new_thing(name, self.name, **stats)

	def historical(self, stat: Key) -> StatusAlias:
		"""Return a reference to the values that a stat has had in the past.

		You can use the reference in comparisons to make a history
		query, and execute the query by calling it, or passing it to
		``self.engine.ticks_when``.

		"""
		return StatusAlias(entity=self, stat=stat)

	def __bool__(self):
		return self.name in self.character.node


class Place(Node):
	"""The kind of node where a thing might ultimately be located.

	LiSE entities are truthy so long as they exist, falsy if they've
	been deleted.

	"""

	__slots__ = (
		"graph",
		"db",
		"node",
		"_rulebook",
		"_rulebooks",
		"_real_rule_mapping",
	)

	extrakeys = {
		"name",
	}

	def __getitem__(self, key):
		if key == "name":
			return self.name
		return super().__getitem__(key)

	def __repr__(self):
		return "<{}.character[{}].place[{}]>".format(
			repr(self.engine), self.character.name, self.name
		)

	def _validate_node_type(self):
		try:
			self.engine._things_cache.retrieve(
				self.character.name, self.name, *self.engine._btt()
			)
			return False
		except:
			return True

	def delete(self) -> None:
		"""Remove myself from the world model immediately."""
		super().delete()
		self.character.place.send(
			self.character.place, key=self.name, val=None
		)


def roerror(*args):
	raise RuntimeError("Read-only")


class Thing(Node):
	"""The sort of item that has a particular location at any given time.

	Things are always in Places or other Things, and may additionally be
	travelling through a Portal.

	LiSE entities are truthy so long as they exist, falsy if they've
	been deleted.

	"""

	__slots__ = (
		"graph",
		"db",
		"node",
		"_rulebook",
		"_rulebooks",
		"_real_rule_mapping",
	)

	_extra_keys = {"name", "location"}

	def _getname(self):
		return self.name

	def _getloc(self):
		ret = self.engine._things_cache._base_retrieve(
			(self.character.name, self.name, *self.engine._btt())
		)
		if ret is None or isinstance(ret, Exception):
			return None
		return ret

	def _validate_node_type(self):
		return self._getloc() is not None

	def _get_arrival_time(self):
		charn = self.character.name
		n = self.name
		thingcache = self.engine._things_cache
		for b, trn, tck in self.engine._iter_parent_btt():
			try:
				v = thingcache.turn_before(charn, n, b, trn)
			except KeyError:
				v = thingcache.turn_after(charn, n, b, trn)
			if v is not None:
				return v
		else:
			raise ValueError("Couldn't find arrival time")

	def _set_loc(self, loc: Optional[Key]):
		self.engine._set_thing_loc(self.character.name, self.name, loc)

	_getitem_dispatch = {"name": _getname, "location": _getloc}

	_setitem_dispatch = {"name": roerror, "location": _set_loc}

	def __getitem__(self, key: Key):
		"""Return one of my stats stored in the database, or special cases:

		``name``: return the name that uniquely identifies me within
		my Character

		``location``: return the name of my location

		"""
		disp = self._getitem_dispatch
		if key in disp:
			return disp[key](self)
		else:
			return super().__getitem__(key)

	def __setitem__(self, key, value):
		"""Set ``key``=``value`` for the present game-time."""
		try:
			self._setitem_dispatch[key](self, value)
		except HistoricKeyError as ex:
			raise ex
		except KeyError:
			super().__setitem__(key, value)

	def __delitem__(self, key):
		"""As of now, this key isn't mine."""
		if key in self._extra_keys:
			raise ValueError("Can't delete {}".format(key))
		super().__delitem__(key)

	def __repr__(self):
		return "<{}.character['{}'].thing['{}']>".format(
			self.engine, self.character.name, self.name
		)

	def delete(self) -> None:
		super().delete()
		self._set_loc(None)
		self.character.thing.send(
			self.character.thing, key=self.name, val=None
		)

	def clear(self) -> None:
		"""Unset everything."""
		for k in list(self.keys()):
			if k not in self._extra_keys:
				del self[k]

	@property
	def location(self) -> Node:
		"""The ``Thing`` or ``Place`` I'm in."""
		locn = self["location"]
		if locn is None:
			raise AttributeError("Not really a Thing")
		return self.engine._get_node(self.character, locn)

	@location.setter
	def location(self, v: Union[Node, Key]):
		if hasattr(v, "name"):
			v = v.name
		self["location"] = v

	def go_to_place(self, place: Union[Node, Key], weight: Key = None) -> int:
		"""Assuming I'm in a node that has a :class:`Portal` direct
		to the given node, schedule myself to travel to the
		given :class:`Place`, taking an amount of time indicated by
		the ``weight`` stat on the :class:`Portal`, if given; else 1
		turn.

		Return the number of turns the travel will take.

		"""
		if hasattr(place, "name"):
			placen = place.name
		else:
			placen = place
		curloc = self["location"]
		orm = self.character.engine
		turns = (
			1
			if weight is None
			else self.engine._portal_objs[
				(self.character.name, curloc, place)
			].get(weight, 1)
		)
		with self.engine.plan():
			orm.turn += turns
			self["location"] = placen
		return turns

	def follow_path(
		self, path: list, weight: Key = None, check: bool = True
	) -> int:
		"""Go to several nodes in succession, deciding how long to
		spend in each by consulting the ``weight`` stat of the
		:class:`Portal` connecting the one node to the next,
		default 1 turn.

		Return the total number of turns the travel will take. Raise
		:class:`TravelException` if I can't follow the whole path,
		either because some of its nodes don't exist, or because I'm
		scheduled to be somewhere else. Set ``check=False`` if
		you're really sure the path is correct, and this function
		will be faster.

		"""
		if len(path) < 2:
			raise ValueError("Paths need at least 2 nodes")
		eng = self.character.engine
		with eng.world_lock:
			if check:
				prevplace = path.pop(0)
				if prevplace != self["location"]:
					raise ValueError(
						"Path does not start at my present location"
					)
				subpath = [prevplace]
				for place in path:
					if (
						prevplace not in self.character.portal
						or place not in self.character.portal[prevplace]
					):
						raise TravelException(
							"Couldn't follow portal from {} to {}".format(
								prevplace, place
							),
							path=subpath,
							traveller=self,
						)
					subpath.append(place)
					prevplace = place
			else:
				subpath = path.copy()
			turns_total = 0
			prevsubplace = subpath.pop(0)
			turn_incs = []
			myname = self.name
			charn = self.character.name
			branch, turn, tick = eng._btt()
			for subplace in subpath:
				if weight is not None:
					turn_incs.append(
						self.engine._edge_val_cache.retrieve(
							charn,
							prevsubplace,
							subplace,
							0,
							branch,
							turn,
							tick,
						)
					)
				else:
					turn_incs.append(1)
				turns_total += turn_incs[-1]
				turn += turn_incs[-1]
				tick = eng._turn_end_plan.get(turn, 0)
				if check:
					eng._nodes_cache.retrieve(
						charn, subplace, branch, turn, tick
					)
				_, start_turn, start_tick, end_turn, end_tick = eng._branches[
					branch
				]
				if (
					(start_turn < turn < end_turn)
					or (
						start_turn == end_turn == turn
						and start_tick <= tick < end_tick
					)
					or (start_turn == turn and start_tick <= tick)
					or (end_turn == turn and tick < end_tick)
				):
					eng.load_at(branch, turn, tick)
			with eng.plan():
				for subplace, turn_inc in zip(subpath, turn_incs):
					eng.turn += turn_inc
					branch, turn, tick = eng._nbtt()
					eng._things_cache.store(
						charn, myname, branch, turn, tick, subplace
					)
					eng.query.set_thing_loc(
						charn, myname, branch, turn, tick, subplace
					)
			return turns_total

	def travel_to(
		self,
		dest: Union[Node, Key],
		weight: Key = None,
		graph: nx.DiGraph = None,
	) -> int:
		"""Find the shortest path to the given node from where I am
		now, and follow it.

		If supplied, the ``weight`` stat of each :class:`Portal` along
		the path will be used in pathfinding, and for deciding how
		long to stay in each Place along the way.

		The ``graph`` argument may be any NetworkX-style graph. It
		will be used for pathfinding if supplied, otherwise I'll use
		my :class:`Character`. In either case, however, I will attempt
		to actually follow the path using my :class:`Character`, which
		might not be possible if the supplied ``graph`` and my
		:class:`Character` are too different. If it's not possible,
		I'll raise a :class:`TravelException`, whose ``subpath``
		attribute holds the part of the path that I *can* follow. To
		make me follow it, pass it to my ``follow_path`` method.

		Return value is the number of turns the travel will take.

		"""
		destn = dest.name if hasattr(dest, "name") else dest
		if destn == self.location.name:
			raise ValueError("I'm already at {}".format(destn))
		graph = self.character if graph is None else graph
		path = nx.shortest_path(graph, self["location"], destn, weight)
		return self.follow_path(path, weight)
