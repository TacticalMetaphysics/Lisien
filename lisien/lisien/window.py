# This file is part of allegedb, an object-relational mapper for versioned graphs.
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
"""WindowDict, the core data structure used by allegedb's caching system.

It resembles a dictionary, more specifically a defaultdict-like where retrieving
a key that isn't set will get the highest set key that is lower than the key
you asked for (and thus, keys must be orderable). It is optimized for retrieval
of the same key and neighboring ones repeatedly and in sequence.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import (
	ItemsView,
	KeysView,
	Mapping,
	MutableMapping,
	ValuesView,
)
from enum import Enum
from functools import partial
from itertools import chain
from operator import itemgetter, le, lt, ge, gt
from threading import RLock
from typing import Any, Iterable, Union, Iterator, Callable

from .exc import HistoricKeyError
from .typing import Tick, Turn, Value

get0 = itemgetter(0)
get1 = itemgetter(1)


class Direction(Enum):
	FORWARD = "forward"
	BACKWARD = "backward"


def update_window(
	turn_from: Turn,
	tick_from: Tick,
	turn_to: Turn,
	tick_to: Tick,
	updfun: Callable[[Turn, Tick, ...], None],
	branchd: SettingsTurnDict,
):
	"""Iterate over some time in ``branchd``, call ``updfun`` on the values"""
	if turn_from == turn_to:
		if turn_from not in branchd:
			return
		for tick, state in branchd[turn_from].future(tick_from).items():
			if tick > tick_to:
				return
			updfun(turn_from, tick, *state)
		return
	if turn_from in branchd:
		for tick, state in branchd[turn_from].future(tick_from).items():
			updfun(turn_from, tick, *state)
	midturn: Turn
	for midturn in range(turn_from + 1, turn_to):
		if midturn in branchd:
			for tick, state in branchd[midturn].items():
				updfun(midturn, tick, *state)
	if turn_to in branchd:
		for tick, state in reversed(
			branchd[turn_to].past(tick_to + 1).items()
		):
			updfun(turn_to, tick, *state)


def update_backward_window(
	turn_from: Turn,
	tick_from: Tick,
	turn_to: Turn,
	tick_to: Tick,
	updfun: Callable[[Turn, Tick, ...], None],
	branchd: SettingsTurnDict,
):
	"""Iterate backward over time in ``branchd``, call ``updfun`` on the values"""
	if turn_from == turn_to:
		if turn_from not in branchd:
			return
		for tick, state in branchd[turn_from].past(tick_to).items():
			if tick < tick_to:
				return
			updfun(turn_from, tick, *state)
		return
	if turn_from in branchd:
		for tick, state in branchd[turn_from].past(tick_from).items():
			updfun(turn_from, tick, *state)
	midturn: Turn
	for midturn in range(turn_from - 1, turn_to, -1):
		if midturn in branchd:
			for tick, state in reversed(branchd[midturn].items()):
				updfun(midturn, tick, *state)
	if turn_to in branchd:
		for tick, state in branchd[turn_to].future(tick_to - 1).items():
			updfun(turn_to, tick, *state)


class WindowDictKeysView(KeysView):
	"""Look through all the keys a WindowDict contains."""

	_mapping: "WindowDict"

	def __contains__(self, rev: int):
		with self._mapping._lock:
			return rev in self._mapping._keys

	def __iter__(self):
		with self._mapping._lock:
			past = self._mapping._past
			future = self._mapping._future
			if past:
				yield from map(get0, past)
			if future:
				yield from map(get0, reversed(future))

	def __reversed__(self):
		with self._mapping._lock:
			past = self._mapping._past
			future = self._mapping._future
			if future:
				yield from map(get0, future)
			if past:
				yield from map(get0, reversed(past))

	def __repr__(self):
		return f"<WindowDictKeysView containing {list(self)}>"


class WindowDictItemsView(ItemsView):
	"""Look through everything a WindowDict contains."""

	_mapping: "WindowDict"

	def __contains__(self, item: tuple[int, Any]):
		with self._mapping._lock:
			return item in self._mapping._past or item in self._mapping._future

	def __iter__(self):
		with self._mapping._lock:
			past = self._mapping._past
			future = self._mapping._future
			if past:
				yield from past
			if future:
				yield from reversed(future)

	def __reversed__(self):
		with self._mapping._lock:
			past = self._mapping._past
			future = self._mapping._future
			if future:
				yield from future
			if past:
				yield from reversed(past)


class WindowDictPastKeysView(KeysView):
	"""View on a WindowDict's keys relative to last lookup"""

	_mapping: WindowDictPastView

	def __iter__(self):
		with self._mapping.lock:
			yield from map(get0, reversed(self._mapping.stack))

	def __reversed__(self):
		with self._mapping.lock:
			yield from map(get0, self._mapping.stack)

	def __contains__(self, item: int):
		return item in self._mapping


class WindowDictFutureKeysView(KeysView):
	_mapping: WindowDictFutureView

	def __iter__(self):
		with self._mapping.lock:
			yield from map(get0, self._mapping.stack)

	def __reversed__(self):
		with self._mapping.lock:
			yield from map(get0, reversed(self._mapping.stack))

	def __contains__(self, item):
		return item in self._mapping


class WindowDictPastFutureItemsView(ItemsView):
	_mapping: Union["WindowDictPastView", "WindowDictFutureView"]

	@staticmethod
	@abstractmethod
	def _out_of_range(item: tuple, stack: list):
		pass

	def __contains__(self, item: tuple[int, Any]):
		with self._mapping.lock:
			if self._out_of_range(item, self._mapping.stack):
				return False
			k, v = item
			return self._mapping[k] == v


class WindowDictPastItemsView(WindowDictPastFutureItemsView):
	def __iter__(self):
		with self._mapping.lock:
			yield from reversed(self._mapping.stack)

	def __reversed__(self):
		with self._mapping.lock:
			yield from self._mapping.stack

	@staticmethod
	def _out_of_range(item: tuple[int, Any], stack: list[tuple[int, Any]]):
		return item[0] < stack[0][0] or item[0] > stack[-1][0]


class WindowDictFutureItemsView(WindowDictPastFutureItemsView):
	"""View on a WindowDict's future items relative to last lookup"""

	def __iter__(self):
		with self._mapping.lock:
			yield from self._mapping.stack

	def __reversed__(self):
		with self._mapping.lock:
			yield from reversed(self._mapping.stack)

	@staticmethod
	def _out_of_range(item: tuple[int, Any], stack: list[tuple[int, Any]]):
		return item[0] < stack[-1][0] or item[0] > stack[0][0]


class WindowDictPastFutureValuesView(ValuesView):
	"""Abstract class for views on the past or future values of a WindowDict"""

	_mapping: Union["WindowDictPastView", "WindowDictFutureView"]

	def __iter__(self):
		with self._mapping.lock:
			yield from map(get1, reversed(self._mapping.stack))

	def __contains__(self, item: Any):
		with self._mapping.lock:
			return item in map(get1, self._mapping.stack)


class WindowDictValuesView(ValuesView):
	"""Look through all the values that a WindowDict contains."""

	_mapping: "WindowDict"

	def __contains__(self, value: Any):
		with self._mapping._lock:
			return value in map(get1, self._mapping._past) or value in map(
				get1, self._mapping._future
			)

	def __iter__(self):
		with self._mapping._lock:
			past = self._mapping._past
			future = self._mapping._future
			if past:
				yield from map(get1, past)
			if future:
				yield from map(get1, reversed(future))

	def __reversed__(self):
		with self._mapping._lock:
			past = self._mapping._past
			future = self._mapping._future
			if future:
				yield from map(get1, future)
			if past:
				yield from map(get1, reversed(past))


class WindowDictPastFutureView(ABC, Mapping):
	"""Abstract class for historical views on WindowDict"""

	__slots__ = ("stack", "lock")
	stack: list[tuple[int, Value]]

	def __init__(self, stack: list[tuple[int, Value]], lock: RLock) -> None:
		self.stack = stack
		self.lock = lock

	def __len__(self) -> int:
		with self.lock:
			stack = self.stack
			if not stack:
				return 0
			return len(stack)


class WindowDictPastView(WindowDictPastFutureView):
	"""Read-only mapping of just the past of a WindowDict

	Iterates in descending order

	"""

	def __iter__(self) -> Iterator[int]:
		with self.lock:
			yield from map(get0, reversed(self.stack))

	def __reversed__(self) -> Iterator[int]:
		with self.lock:
			yield from map(get0, self.stack)

	def __getitem__(self, key: int) -> Any:
		with self.lock:
			stack = self.stack
			if not stack or key < stack[0][0] or key > stack[-1][0]:
				raise KeyError("Out of range", key)
			return _recurse(key, stack)

	def keys(self) -> WindowDictPastKeysView:
		return WindowDictPastKeysView(self)

	def items(self) -> WindowDictPastItemsView:
		return WindowDictPastItemsView(self)

	def values(self) -> WindowDictPastFutureValuesView:
		return WindowDictPastFutureValuesView(self)


class WindowDictFutureView(WindowDictPastFutureView):
	"""Read-only mapping of just the future of a WindowDict

	Iterates in ascending order

	"""

	def __iter__(self) -> Iterator[int]:
		with self.lock:
			yield from map(get0, reversed(self.stack))

	def __reversed__(self) -> Iterator[int]:
		with self.lock:
			yield from map(get0, self.stack)

	def __getitem__(self, key: int):
		with self.lock:
			stack = self.stack
			if not stack:
				raise KeyError("No data")
			if key < stack[-1][0] or key > stack[0][0]:
				raise KeyError("No such revision", key)
			return _recurse(key, stack)

	def keys(self) -> WindowDictFutureKeysView:
		return WindowDictFutureKeysView(self)

	def items(self) -> WindowDictFutureItemsView:
		return WindowDictFutureItemsView(self)

	def values(self) -> WindowDictPastFutureValuesView:
		return WindowDictPastFutureValuesView(self)


class WindowDictSlice:
	__slots__ = ["dic", "slic"]
	dic: "WindowDict"
	slic: slice

	def __init__(self, dic: "WindowDict", slic: slice):
		self.dic = dic
		self.slic = slic

	def _get_reversed_iterator(self) -> Iterator[Value]:
		step_iter = self._step_iter
		iter_items_until = self._iter_items_until
		dic = self.dic
		if not dic:
			return iter(())
		slic = self.slic
		start = slic.start
		if start is not None and start < 0:
			start = len(dic) - start
			if start < 0:
				raise IndexError("WindowDict index out of range", start)
		stop = slic.stop
		if stop is not None and stop < 0:
			stop = len(dic) - stop
			if stop < 0:
				raise IndexError("WindowDict index out of range", stop)
		step = slic.step
		seek = dic._seek
		past = dic._past
		future = dic._future
		if start is None and stop is None:
			if step is None:
				return chain(map(get1, future), map(get1, reversed(past)))
			else:
				return step_iter(chain(future, reversed(past)), step)
		if start is not None and stop is not None:
			if stop == start:
				seek(start)
				if past and past[-1][0] == start:
					return iter((past[-1][1],))
				else:
					return iter(())
			if start < stop:
				left, right = start, stop
				cmp = ge
			elif step is None or step > 0:
				return iter(())
			else:
				left, right = stop, start
				cmp = gt
			seek(right)
			if not past:
				return iter(())
			if past[-1][0] == right:
				future.append(past.pop())
			if not past:
				return iter(())
			if step is None or step > 0:
				inner_inner_it = reversed(past)
			else:
				inner_inner_it = iter(past)
			inner_it = iter_items_until(inner_inner_it, partial(cmp, left))
			if step is None:
				return map(get1, inner_it)
			else:
				return step_iter(inner_it, abs(step))
		elif start is None:
			seek(stop)
			if not past:
				return iter(())
			if step is None:
				return map(get1, reversed(past))
			elif step < 0:
				return step_iter(iter(past), abs(step))
			else:
				return step_iter(reversed(past), step)
		else:
			assert stop is None
			seek(start)
			if past and past[-1][0] == start:
				future.append(past.pop())
			if step is None:
				return map(get1, future)
			elif step < 0:
				return step_iter(reversed(future), abs(step))
			else:
				return step_iter(iter(future), step)

	def __reversed__(self) -> Iterator[Value]:
		with self.dic._lock:
			yield from self._get_reversed_iterator()

	@staticmethod
	def _step_iter(
		it: Iterator[tuple[int, Value]], step: int
	) -> Iterator[Value]:
		for rev, val in it:
			if rev % step == 0:
				yield val

	@staticmethod
	def _iter_items_until(
		it: Iterator[tuple[int, Value]], until: Callable[[int], bool]
	) -> Iterator[tuple[int, Value]]:
		for rev, val in it:
			if until(rev):
				return
			yield rev, val

	def _get_iterator(self) -> Iterator[Value]:
		step_iter = self._step_iter
		iter_items_until = self._iter_items_until
		dic = self.dic
		if not dic:
			return iter(())
		slic = self.slic
		start = slic.start
		if start is not None and start < 0:
			start = len(dic) - start
			if start < 0:
				raise IndexError("WindowDict index out of range", start)
		stop = slic.stop
		if stop is not None and stop < 0:
			stop = len(dic) - stop
			if stop < 0:
				raise IndexError("WindowDict index out of range", stop)
		step = slic.step
		seek = dic._seek
		past = dic._past
		future = dic._future
		if not past and not future:
			return iter(())
		if start is None and stop is None:
			if step is None:
				return chain(map(get1, past), map(get1, reversed(future)))
			elif step < 0:
				return step_iter(chain(future, reversed(past)), abs(step))
			else:
				return step_iter(chain(past, reversed(future)), step)
		if start is not None and stop is not None:
			if stop == start:
				seek(start)
				if past and past[-1][0] == start:
					return iter((past[-1][1],))
				else:
					return iter(())
			if start < stop:
				left, right = start, stop
				cmp = lt
			elif step is None or step > 0:
				return iter(())
			else:
				left, right = stop, start
				cmp = le
			seek(right)
			if not past:
				return iter(())
			if past[-1][0] == right:
				future.append(past.pop())
			if not past:
				return iter(())
			seek(left)
			if past and past[-1][0] == start:
				future.append(past.pop())
			if step is None or step > 0:
				inner_inner_it = reversed(future)
			else:
				inner_inner_it = iter(future)
			inner_it = iter_items_until(inner_inner_it, partial(cmp, right))
			if step is None:
				return map(get1, inner_it)
			else:
				return step_iter(inner_it, abs(step))
		elif start is None:
			seek(stop)
			if not past:
				return iter(())
			if step is None:
				return map(get1, past)
			elif step < 0:
				return step_iter(reversed(past), abs(step))
			else:
				return step_iter(iter(past), step)
		else:
			assert stop is None
			seek(start)
			if past and past[-1][0] == start:
				future.append(past.pop())
			if step is None:
				return map(get1, reversed(future))
			elif step < 0:
				return step_iter(iter(future), abs(step))
			else:
				return step_iter(reversed(future), step)

	def __iter__(self) -> Iterator[Value]:
		with self.dic._lock:
			yield from self._get_iterator()


def _recurse(rev: int, revs: list[tuple[int, Any]]) -> tuple[int, Any]:
	if len(revs) < 1:
		raise HistoricKeyError("No data ever for revision", rev, deleted=False)
	elif len(revs) == 1:
		if revs[0][0] <= rev:
			return revs[0]
		raise HistoricKeyError("Can't retrieve revision", rev, deleted=False)
	pivot = len(revs) // 2
	before = revs[:pivot]
	after = revs[pivot:]
	assert before and after
	if rev < after[0][0]:
		if rev > before[-1][0]:
			return before[-1]
		return _recurse(rev, before)
	elif rev == after[0][0]:
		return after[0]
	else:
		return _recurse(rev, after)


class WindowDict(MutableMapping):
	"""A dict that keeps every value that a variable has had over time.

	Look up a revision number in this dict, and it will give you the
	effective value as of that revision. Keys should always be
	revision numbers.

	Optimized for the cases where you look up the same revision
	repeatedly, or its neighbors.

	This supports slice notation to get all values in a given
	time-frame. If you do not supply a step, you'll just get the
	values, with no indication of when they're from exactly --
	so explicitly supply a step of 1 to get the value at each point in
	the slice, or use the ``future`` and ``past`` methods to get read-only
	mappings of data relative to a particular revision.

	Unlike slices of eg. lists, you can slice with a start greater than the stop
	even if you don't supply a step. That will get you values in reverse order.

	"""

	__slots__ = ("_future", "_past", "_keys", "_last", "_lock")

	_past: list[tuple[int, Any]]
	_future: list[tuple[int, Any]]
	_keys: set[int]
	_last: int | None

	@property
	def beginning(self) -> int | None:
		with self._lock:
			if not self._past:
				if not self._future:
					return None
				return self._future[-1][0]
			return self._past[0][0]

	@property
	def end(self) -> int | None:
		with self._lock:
			if not self._future:
				if not self._past:
					return None
				return self._past[-1][0]
			return self._future[0][0]

	def future(self, rev: int = None) -> WindowDictFutureView:
		"""Return a Mapping of items after the given revision.

		Default revision is the last one looked up.

		"""
		if rev is not None:
			with self._lock:
				self._seek(rev)
		return WindowDictFutureView(self._future, self._lock)

	def past(self, rev: int = None) -> WindowDictPastView:
		"""Return a Mapping of items at or before the given revision.

		Default revision is the last one looked up.

		"""
		if rev is not None:
			with self._lock:
				self._seek(rev)
		return WindowDictPastView(self._past, self._lock)

	def search(self, rev: int) -> Any:
		"""Alternative access for far-away revisions

		This uses a binary search, which is faster in the case of random
		access, but not in the case of fast-forward and rewind, which are
		more common in time travel.

		This doesn't change the state of the cache.

		"""

		with self._lock:
			revs = self._past + list(reversed(self._future))
			if len(revs) == 1:
				result_rev, result = revs[0]
				if rev < result_rev:
					raise HistoricKeyError(
						"No data ever for revision", rev, deleted=False
					)
			else:
				result_rev, result = _recurse(rev, revs)
			return result

	def _seek(self, rev: int) -> None:
		"""Arrange the caches to help look up the given revision."""
		if rev == self._last:
			return
		past = self._past
		future = self._future
		if future:
			appender = past.append
			popper = future.pop
			future_start = future[-1][0]
			while future_start <= rev:
				appender(popper())
				if future:
					future_start = future[-1][0]
				else:
					break
		if past:
			popper = past.pop
			appender = future.append
			past_end = past[-1][0]
			while past_end > rev:
				appender(popper())
				if past:
					past_end = past[-1][0]
				else:
					break
		self._last = rev

	def rev_gettable(self, rev: int) -> bool:
		beg = self.beginning
		if beg is None:
			return False
		return rev >= beg

	def rev_before(self, rev: int, search=False) -> int | None:
		"""Return the latest past rev on which the value changed.

		If it changed on this exact rev, return the rev.

		"""
		with self._lock:
			if search:
				rev, _ = _recurse(
					rev, self._past + list(reversed(self._future))
				)
				return rev
			else:
				self._seek(rev)
				if self._past:
					return self._past[-1][0]
				else:
					return None

	def rev_after(self, rev: int) -> int | None:
		"""Return the earliest future rev on which the value will change."""
		with self._lock:
			self._seek(rev)
			if self._future:
				return self._future[-1][0]
			else:
				return None

	def initial(self) -> Any:
		"""Return the earliest value we have"""
		with self._lock:
			if self._past:
				return self._past[0][1]
			if self._future:
				return self._future[-1][1]
			raise KeyError("No data")

	def final(self) -> Any:
		"""Return the latest value we have"""
		with self._lock:
			if self._future:
				return self._future[0][1]
			if self._past:
				return self._past[-1][1]
			raise KeyError("No data")

	def truncate(
		self, rev: int, direction: Direction = Direction.FORWARD
	) -> set[int]:
		"""Delete everything after the given revision, exclusive.

		With direction='backward', delete everything before the revision,
		exclusive, instead.

		Return a set of keys deleted.

		"""
		if not isinstance(direction, Direction):
			direction = Direction(direction)
		deleted = set()
		with self._lock:
			self._seek(rev)
			if direction == Direction.FORWARD:
				to_delete = set(map(get0, self._future))
				deleted.update(to_delete)
				self._keys.difference_update(to_delete)
				self._future = []
			elif direction == Direction.BACKWARD:
				if not self._past:
					return deleted
				if self._past[-1][0] == rev:
					to_delete = set(map(get0, self._past[:-1]))
					deleted.update(to_delete)
					self._keys.difference_update(to_delete)
					self._past = [self._past[-1]]
				else:
					to_delete = set(map(get0, self._past))
					deleted.update(to_delete)
					self._keys.difference_update(to_delete)
					self._past = []
			else:
				raise ValueError("Need direction 'forward' or 'backward'")
		return deleted

	def keys(self) -> WindowDictKeysView:
		return WindowDictKeysView(self)

	def items(self) -> WindowDictItemsView:
		return WindowDictItemsView(self)

	def values(self) -> WindowDictValuesView:
		return WindowDictValuesView(self)

	def __bool__(self) -> bool:
		return bool(self._keys)

	def copy(self):
		with self._lock:
			empty = WindowDict.__new__(WindowDict)
			empty._past = self._past.copy()
			empty._future = self._future.copy()
			empty._keys = self._keys.copy()
			empty._last = self._last
			return empty

	def __init__(
		self, data: Union[list[tuple[int, Any]], dict[int, Any]] = None
	) -> None:
		self._lock = RLock()
		with self._lock:
			if not data:
				self._past = []
			elif isinstance(data, Mapping):
				self._past = list(data.items())
			else:
				# assume it's an orderable sequence of pairs
				self._past = list(data)
			self._past.sort()
			self._future = []
			self._keys = set(map(get0, self._past))
			self._last = None

	def __iter__(self) -> Iterable[Any]:
		if not self:
			return
		with self._lock:
			if self._past:
				yield from map(get0, self._past)
			if self._future:
				yield from map(get0, self._future)

	def __contains__(self, item: int) -> bool:
		with self._lock:
			return item in self._keys

	def __len__(self) -> int:
		with self._lock:
			return len(self._keys)

	def __getitem__(self, rev: int) -> Any:
		if isinstance(rev, slice):
			return WindowDictSlice(self, rev)
		with self._lock:
			self._seek(rev)
			past = self._past
			if not past:
				raise HistoricKeyError(
					"Revision {} is before the start of history".format(rev)
				)
			return past[-1][1]

	def __setitem__(self, rev: int, v: Any) -> None:
		past = self._past
		with self._lock:
			if past or self._future:
				self._seek(rev)
				if past:
					if past[-1][0] == rev:
						past[-1] = (rev, v)
					else:
						past.append((rev, v))
				else:
					past.append((rev, v))
			else:
				past.append((rev, v))
			self._keys.add(rev)

	def __delitem__(self, rev: int) -> None:
		self.del_item(rev)

	def del_item(self, rev: int, search=False) -> None:
		# Not checking for rev's presence at the beginning because
		# to do so would likely require iterating thru history,
		# which I have to do anyway in deleting.
		# But handle degenerate case.
		if not self:
			raise HistoricKeyError("Tried to delete from an empty WindowDict")
		if self.beginning is None:
			if self.end is not None and rev > self.end:
				raise HistoricKeyError(
					"Rev outside of history: {}".format(rev)
				)
		elif self.end is None:
			if self.beginning is not None and rev < self.beginning:
				raise HistoricKeyError(
					"Rev outside of history: {}".format(rev)
				)
		elif not self.beginning <= rev <= self.end:
			raise HistoricKeyError("Rev outside of history: {}".format(rev))
		with self._lock:
			if search:
				self.search(rev)
			else:
				self._seek(rev)
			past = self._past
			if not past or past[-1][0] != rev:
				raise HistoricKeyError("Rev not present: {}".format(rev))
			del past[-1]
			self._keys.remove(rev)

	def __repr__(self) -> str:
		me = {}
		if self._past:
			me.update(self._past)
		if self._future:
			me.update(self._future)
		return "{}({})".format(self.__class__.__name__, me)


class FuturistWindowDict(WindowDict):
	"""A WindowDict that does not let you rewrite the past."""

	__slots__ = (
		"_future",
		"_past",
	)
	_future: list[tuple[int, Any]]
	_past: list[tuple[int, Any]]

	def __setitem__(self, rev: int, v: Any) -> None:
		if hasattr(v, "unwrap") and not hasattr(v, "no_unwrap"):
			v = v.unwrap()
		with self._lock:
			self._seek(rev)
			past = self._past
			future = self._future
			if future:
				raise HistoricKeyError(
					"Already have some history after {}".format(rev)
				)
			if not past:
				past.append((rev, v))
			elif rev > past[-1][0]:
				past.append((rev, v))
			elif rev == past[-1][0]:
				past[-1] = (rev, v)
			else:
				raise HistoricKeyError(
					"Already have some history after {} "
					"(and my seek function is broken?)".format(rev)
				)
			self._keys.add(rev)


class TurnDict(FuturistWindowDict):
	__slots__ = ("_future", "_past")
	_future: list[tuple[Turn, Any]]
	_past: list[tuple[Turn, Any]]
	cls = FuturistWindowDict

	def __setitem__(self, turn: Turn, value: Any) -> None:
		if type(value) is not FuturistWindowDict:
			value = FuturistWindowDict(value)
		FuturistWindowDict.__setitem__(self, turn, value)


class EntikeyWindowDict(WindowDict):
	__slots__ = ("_past", "_future", "entikeys")

	def __init__(
		self, data: Union[list[tuple[int, Any]], dict[int, Any]] = None
	) -> None:
		if data:
			if hasattr(data, "values") and callable(data.values):
				self.entikeys = {value[:-2] for value in data.values()}
			else:
				self.entikeys = {value[:-2] for value in data}
		else:
			self.entikeys = set()
		super().__init__(data)

	def __setitem__(self, rev: int, v: tuple) -> None:
		self.entikeys.add(v[:-2])
		super().__setitem__(rev, v)

	def __delitem__(self, rev: int) -> None:
		entikey = self[rev][:-2]
		super().__delitem__(rev)
		for tup in self.values():
			if tup[:-2] == entikey:
				return
		self.entikeys.remove(entikey)


class SettingsTurnDict(WindowDict):
	"""A WindowDict that contains a span of time, indexed as turns and ticks

	Each turn is a series of ticks. Once a value is set at some turn and tick,
	it's in effect at every tick in the turn after that one, and every
	further turn.

	"""

	__slots__ = ("_future", "_past")
	_future: list[tuple[Turn, Any]]
	_past: list[tuple[Turn, Any]]
	cls = WindowDict

	def __setitem__(self, turn: Turn, value: cls | dict) -> None:
		if not isinstance(value, self.cls):
			value = self.cls(value)
		WindowDict.__setitem__(self, turn, value)

	def retrieve(self, turn: Turn, tick: Tick) -> Any:
		"""Retrieve the value that was in effect at this turn and tick

		Whether or not it was *set* at this turn and tick

		"""
		if turn in self and self[turn].rev_gettable(tick):
			return self[turn][tick]
		elif self.rev_gettable(turn - 1):
			return self[turn - 1].final()
		raise KeyError(f"Can't retrieve turn {turn}, tick {tick}")

	def retrieve_exact(self, turn: Turn, tick: Tick) -> Any:
		"""Retrieve the value only if it was set at this exact turn and tick"""
		if turn not in self:
			raise KeyError(f"No data in turn {turn}")
		if tick not in self[turn]:
			raise KeyError(f"No data for tick {tick} in turn {turn}")
		return self[turn][tick]

	def store_at(self, turn: Turn, tick: Tick, value: Any) -> None:
		"""Set a value at a time, creating the turn if needed"""
		if turn in self:
			self[turn][tick] = value
		else:
			self[turn] = {tick: value}


class EntikeySettingsTurnDict(SettingsTurnDict):
	cls = EntikeyWindowDict
