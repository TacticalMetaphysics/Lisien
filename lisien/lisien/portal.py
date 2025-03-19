# This file is part of Lisien, a framework for life simulation games.
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
"""Directed edges, as used by lisien."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .exc import HistoricKeyError
from .facade import FacadePortal
from .graph import Edge
from .query import StatusAlias
from .rule import RuleFollower
from .rule import RuleMapping as BaseRuleMapping
from .typing import Key
from .util import AbstractCharacter, getatt


class RuleMapping(BaseRuleMapping):
	"""Mapping to get rules followed by a portal."""

	def __init__(self, portal):
		"""Store portal, engine, and rulebook."""
		super().__init__(portal.engine, portal.rulebook)
		self.portal = portal


class Portal(Edge, RuleFollower):
	"""Connection between two nodes that :class:`lisien.node.Thing` travel along

	lisien entities are truthy so long as they exist, falsy if they've
	been deleted.

	"""

	__slots__ = (
		"graph",
		"orig",
		"dest",
		"idx",
		"origin",
		"destination",
		"_rulebook",
		"_real_rule_mapping",
	)
	character = getatt("graph")
	engine = getatt("db")
	no_unwrap = True

	def __init__(self, graph: AbstractCharacter, orig: Key, dest: Key):
		super().__init__(graph, orig, dest, 0)
		self.origin = graph.node[orig]
		self.destination = graph.node[dest]

	@property
	def _cache(self):
		return self.db._edge_val_cache[self.character.name][self.orig][
			self.dest
		][0]

	def _rule_name_activeness(self):
		rulebook_name = self._get_rulebook_name()
		cache = self.engine._active_rules_cache
		if rulebook_name not in cache:
			return
		cache = cache[rulebook_name]
		for rule in cache:
			for branch, turn, tick in self.engine._iter_parent_btt():
				if branch not in cache[rule]:
					continue
				try:
					yield (rule, cache[rule][branch][turn][tick])
					break
				except ValueError:
					continue
				except HistoricKeyError as ex:
					if ex.deleted:
						break
		raise KeyError("{}->{} has no rulebook?".format(self.orig, self.dest))

	def _get_rulebook_name(self):
		btt = self.engine._btt()
		try:
			return self.engine._portals_rulebooks_cache.retrieve(
				self.character.name, self.orig, self.dest, *btt
			)
		except KeyError:
			ret = (self.character.name, self.orig, self.dest)
			self.engine._portals_rulebooks_cache.store(*ret, *btt, ret)
			self.engine.query.set_portal_rulebook(*ret, *btt, ret)
			return ret

	def _set_rulebook_name(self, rulebook):
		character = self.character.name
		orig = self.orig
		dest = self.dest
		cache = self.engine._portals_rulebooks_cache
		try:
			if rulebook == cache.retrieve(
				character, orig, dest, *self.engine._btt()
			):
				return
		except KeyError:
			pass
		branch, turn, tick = self.engine._nbtt()
		cache.store(character, orig, dest, branch, turn, tick, rulebook)
		self.engine.query.set_portal_rulebook(
			character, orig, dest, branch, turn, tick, rulebook
		)

	def _get_rule_mapping(self):
		return RuleMapping(self)

	def __getitem__(self, key):
		if key == "origin":
			return self.orig
		elif key == "destination":
			return self.dest
		elif key == "character":
			return self.character.name
		else:
			return super().__getitem__(key)

	def __setitem__(self, key, value):
		if key in ("origin", "destination", "character"):
			raise KeyError("Can't change " + key)
		super().__setitem__(key, value)

	def __repr__(self):
		"""Describe character, origin, and destination"""
		return "<{}.character[{}].portal[{}][{}]>".format(
			repr(self.engine),
			repr(self["character"]),
			repr(self["origin"]),
			repr(self["destination"]),
		)

	def __bool__(self):
		"""It means something that I exist, even if I have no data."""
		return (
			self.orig in self.character.portal
			and self.dest in self.character.portal[self.orig]
		)

	@property
	def reciprocal(self) -> "Portal":
		"""If there's another Portal connecting the same origin and
		destination that I do, but going the opposite way, return
		it. Else raise KeyError.

		"""
		try:
			return self.character.portal[self.dest][self.orig]
		except KeyError:
			raise AttributeError("This portal has no reciprocal")

	def facade(self):
		face = self.character.facade()
		ret = FacadePortal(face.portal[self.orig], self.dest)
		face.portal._patch = {self.orig: {self.dest: ret}}
		return ret

	def historical(self, stat: Key) -> StatusAlias:
		"""Return a reference to the values that a stat has had in the past.

		You can use the reference in comparisons to make a history
		query, and execute the query by calling it, or passing it to
		``self.engine.ticks_when``.

		"""
		return StatusAlias(entity=self, stat=stat)

	def update(self, e: Mapping | list[tuple[Any, Any]] = None, **f) -> None:
		"""Works like regular update, but less

		Only actually updates when the new value and the old value differ.
		This is necessary to prevent certain infinite loops.

		"""
		if e is not None:
			if hasattr(e, "keys") and callable(e.keys):
				for k in e.keys():
					if k not in self:
						self[k] = e[k]
					else:
						v = e[k]
						if self[k] != v:
							self[k] = v
			else:
				for k, v in e:
					if k not in self or self[k] != v:
						self[k] = v
		for k, v in f.items():
			if k not in self or self[k] != v:
				self[k] = v

	def delete(self) -> None:
		"""Remove myself from my :class:`Character`.

		For symmetry with :class:`Thing` and :class:`Place`.

		"""
		self.clear()
		self.engine._exist_edge(
			self.character.name, self.orig, self.dest, exist=None
		)

	def unwrap(self) -> dict:
		"""Return a dictionary representation of this entity"""
		return {
			k: v.unwrap()
			if hasattr(v, "unwrap") and not hasattr(v, "no_unwrap")
			else v
			for (k, v) in self.items()
		}
