# This file is part of allegedb, an object-relational mapper for versioned graphs.
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
"""WindowDict, the core data structure used by allegedb's caching system.

It resembles a dictionary, more specifically a defaultdict-like where retrieving
a key that isn't set will get the highest set key that is lower than the key
you asked for (and thus, keys must be orderable). It is optimized for retrieval
of the same key and neighboring ones repeatedly and in sequence.

"""
from collections import deque, Mapping, MutableMapping, KeysView, ItemsView, ValuesView
from operator import itemgetter

# TODO: cancel changes that would put something back to where it was at the start
# This will complicate the update_window functions though, and I don't think it'll
# improve much apart from a bit of efficiency in that the deltas are smaller
# sometimes.
def update_window(turn_from, tick_from, turn_to, tick_to, updfun, branchd):
    """Iterate over a window of time in ``branchd`` and call ``updfun`` on the values"""
    if turn_from in branchd:
        # Not including the exact tick you started from because deltas are *changes*
        for past_state in branchd[turn_from][tick_from+1:]:
            updfun(*past_state)
    for midturn in range(turn_from+1, turn_to):
        if midturn in branchd:
            for past_state in branchd[midturn][:]:
                updfun(*past_state)
    if turn_to in branchd:
        for past_state in branchd[turn_to][:tick_to]:
            updfun(*past_state)


def update_backward_window(turn_from, tick_from, turn_to, tick_to, updfun, branchd):
    """Iterate backward over a window of time in ``branchd`` and call ``updfun`` on the values"""
    if turn_from in branchd:
        for future_state in reversed(branchd[turn_from][:tick_from]):
            updfun(*future_state)
    for midturn in range(turn_from-1, turn_to, -1):
        if midturn in branchd:
            for future_state in reversed(branchd[midturn][:]):
                updfun(*future_state)
    if turn_to in branchd:
        for future_state in reversed(branchd[turn_to][tick_to+1:]):
            updfun(*future_state)


class HistoryError(KeyError):
    """You tried to access the past in a bad way."""

    def __init__(self, *args, deleted=False):
        super().__init__(*args)
        self.deleted = deleted


def within_history(rev, windowdict):
    """Return whether the windowdict has history at the revision."""
    if not windowdict:
        return False
    begin = windowdict._past[0][0] if windowdict._past else \
            windowdict._future[0][0]
    end = windowdict._future[-1][0] if windowdict._future else \
          windowdict._past[-1][0]
    return begin <= rev <= end


class WindowDictKeysView(KeysView):
    """Look through all the keys a WindowDict contains."""
    def __contains__(self, rev):
        if not within_history(rev, self._mapping):
            return False
        for mrev, mv in self._mapping._past:
            if mrev == rev:
                return True
        for mrev, mv in self._mapping._future:
            if mrev == rev:
                return True
        return False

    def __iter__(self):
        if self._mapping._past:
            for rev, v in self._mapping._past:
                yield rev
        if self._mapping._future:
            for rev, v in self._mapping._future:
                yield rev


class WindowDictItemsView(ItemsView):
    """Look through everything a WindowDict contains."""
    def __contains__(self, item):
        (rev, v) = item
        if not within_history(rev, self._mapping):
            return False
        for mrev, mv in self._mapping._past:
            if mrev == rev:
                return mv == v
        for mrev, mv in self._mapping._future:
            if mrev == rev:
                return mv == v
        return False

    def __iter__(self):
        if self._mapping._past:
            yield from self._mapping._past
        if self._mapping._future:
            yield from self._mapping._future


class WindowDictPastFutureKeysView(KeysView):
    """Abstract class for views on the past or future keys of a WindowDict"""
    def __contains__(self, item):
        if not self._mapping.deq:
            return
        deq = self._mapping.deq
        if not deq or item < deq[0][0] or item > deq[-1][0]:
            return False
        for rev in map(itemgetter(0), deq):
            if rev == item:
                return True
        return False


class WindowDictPastKeysView(WindowDictPastFutureKeysView):
    """View on a WindowDict's past keys relative to last lookup"""
    def __iter__(self):
        if not self._mapping.deq:
            return
        yield from map(itemgetter(0), reversed(self._mapping.deq))


class WindowDictFutureKeysView(WindowDictPastFutureKeysView):
    """View on a WindowDict's future keys relative to last lookup"""
    def __iter__(self):
        if not self._mapping.deq:
            return
        yield from map(itemgetter(0), self._mapping.deq)


class WindowDictPastFutureItemsView(ItemsView):
    """Abstract class for views on the past or future items of a WindowDict"""
    def __contains__(self, item):
        deq = self._mapping.deq
        if not deq or item[0] < deq[0][0] or item[0] > deq[-1][0]:
            return False
        rev, v = item
        for mrev, mv in deq:
            if mrev == rev:
                return mv == v
        return False


class WindowDictPastItemsView(WindowDictPastFutureItemsView):
    """View on a WindowDict's past items relative to last lookup"""
    def __iter__(self):
        if not self._mapping.deq:
            return
        yield from reversed(self._mapping.deq)


class WindowDictFutureItemsView(WindowDictPastFutureItemsView):
    """View on a WindowDict's future items relative to last lookup"""
    def __iter__(self):
        if not self._mapping.deq:
            return
        yield from self._mapping.deq


class WindowDictPastFutureValuesView(ValuesView):
    """Abstract class for views on the past or future values of a WindowDict"""
    def __contains__(self, item):
        if not self._mapping.deq:
            return False
        for v in map(itemgetter(1), self._mapping.deq):
            if v == item:
                return True
        return False


class WindowDictPastValuesView(WindowDictPastFutureValuesView):
    """View on a WindowDict's past values relative to last lookup"""
    def __iter__(self):
        if not self._mapping.deq:
            return
        yield from map(itemgetter(1), reversed(self._mapping.deq))


class WindowDictFutureValuesView(WindowDictPastFutureValuesView):
    """View on a WindowDict's future values relative to last lookup"""
    def __iter__(self):
        if not self._mapping.deq:
            return
        yield from map(itemgetter(1), self._mapping.deq)


class WindowDictValuesView(ValuesView):
    """Look through all the values that a WindowDict contains."""
    def __contains__(self, value):
        if self._mapping._past:
            for rev, v in self._mapping._past:
                if v == value:
                    return True
        if self._mapping._future:
            for rev, v in self._mapping._future:
                if v == value:
                    return True
        return False

    def __iter__(self):
        if self._mapping._past:
            for rev, v in self._mapping._past:
                yield v
        if self._mapping._future:
            for rev, v in self._mapping._future:
                yield v


class WindowDictPastFutureView(Mapping):
    """Abstract class for historical views on WindowDict"""
    __slots__ = ('deq',)

    def __init__(self, deq):
        self.deq = deq

    def __len__(self):
        if not self.deq:
            return 0
        return len(self.deq)

    def __getitem__(self, key):
        if not self.deq or key < self.deq[0][0] or key > self.deq[-1][0]:
            raise KeyError
        for rev, value in self.deq:
            if rev == key:
                return value
        raise KeyError


class WindowDictPastView(WindowDictPastFutureView):
    """Read-only mapping of just the past of a WindowDict"""
    def __iter__(self):
        if not self.deq:
            return
        yield from map(itemgetter(0), reversed(self.deq))

    def keys(self):
        return WindowDictPastKeysView(self)

    def items(self):
        return WindowDictPastItemsView(self)

    def values(self):
        return WindowDictPastValuesView(self)


class WindowDictFutureView(WindowDictPastFutureView):
    """Read-only mapping of just the future of a WindowDict"""
    def __iter__(self):
        if not self.deq:
            return
        yield from map(itemgetter(0), self.deq)

    def __getitem__(self, key):
        if not self.deq or key < self.deq[0][0] or key > self.deq[-1][0]:
            raise KeyError
        for rev, value in self.deq:
            if rev == key:
                return value
        raise KeyError

    def keys(self):
        return WindowDictFutureKeysView(self)

    def items(self):
        return WindowDictFutureItemsView(self)

    def values(self):
        return WindowDictFutureValuesView(self)


class WindowDictSlice:
    """A slice of history in which the start is earlier than the stop"""
    __slots__ = ['dict', 'slice']

    def __init__(self, dict, slice):
        self.dict = dict
        self.slice = slice

    def __reversed__(self):
        return iter(WindowDictReverseSlice(self.dict, self.slice))

    def __iter__(self):
        dic = self.dict
        if not dic:
            return
        slic = self.slice
        if slic.step is not None:
            for i in range(slic.start or dic.beginning, slic.stop or dic.end, slic.step):
                yield dic[i]
        if slic.start is None and slic.stop is None:
            if dic._past:
                yield from map(itemgetter(1), dic._past)
            if dic._future:
                yield from map(itemgetter(1), dic._future)
        elif None not in (slic.start, slic.stop):
            if slic.stop == slic.start:
                yield dic[slic.stop]
                return
            left, right = (slic.start, slic.stop) if slic.start < slic.stop else (slic.stop, slic.start)
            dic.seek(right)
            if not dic._past:
                return
            past = dic._past.copy()
            popper = getattr(past, 'popleft', lambda: past.pop(0))
            while past and past[0][0] < left:
                popper()
            yield from map(itemgetter(1), past)
        elif slic.start is None:
            stac = []
            if dic._past:
                stac.extend(dic._past)
            if dic._future:
                stac.extend(dic._future)
            while stac and stac[-1][0] > slic.stop:
                stac.pop()
            yield from map(itemgetter(1), stac)
            return
        else:  # slic.stop is None
            stac = deque()
            if dic._past:
                stac.extend(dic._past)
            if dic._future:
                stac.extend(dic._future)
            while stac and stac[0][0] < slic.start:
                stac.popleft()
            yield from map(itemgetter(1), stac)


class WindowDictReverseSlice:
    """A slice of history in which the start is later than the stop"""
    __slots__ = ['dict', 'slice']

    def __init__(self, dict, slice):
        self.dict = dict
        self.slice = slice

    def __reversed__(self):
        return iter(WindowDictSlice(self.dict, self.slice))

    def __iter__(self):
        dic = self.dict
        if not dic:
            return
        slic = self.slice
        if slic.step is not None:
            for i in range(slic.start or dic.end, slic.stop or dic.beginning, slic.step):
                yield dic[i]
        if slic.start is None and slic.stop is None:
            if dic._future:
                yield from map(itemgetter(1), reversed(dic._future))
            if dic._past:
                yield from map(itemgetter(1), reversed(dic._past))
        elif None not in (slic.start, slic.stop):
            if slic.start == slic.stop:
                yield dic[slic.stop]
                return
            left, right = (slic.start, slic.stop) if slic.start < slic.stop else (slic.stop, slic.start)
            dic.seek(right)
            for frev, fv in reversed(dic._past):
                if frev <= left:
                    return
                yield fv
        elif slic.start is None:
            stac = deque()
            if dic._past:
                stac.extend(dic._past)
            if dic._future:
                stac.extend(dic._future)
            while stac and stac[-1][0] > slic.stop:
                stac.pop()
            yield from map(itemgetter(1), reversed(stac))
        else:  # slic.stop is None
            stac = deque()
            if dic._past:
                stac.extend(dic._past)
            if dic._future:
                stac.extend(dic._future)
            while stac and stac[0][0] < slic.start:
                stac.popleft()
            yield from map(itemgetter(1), reversed(stac))


class WindowDict(MutableMapping):
    """A dict that keeps every value that a variable has had over time.

    Look up a revision number in this dict and it will give you the
    effective value as of that revision. Keys should always be
    revision numbers. Once a key is set, all greater keys are
    considered to be in this dict unless the value is ``None``. Keys
    after that one aren't "set" until one's value is non-``None``
    again.

    Optimized for the cases where you look up the same revision
    repeatedly, or its neighbors.

    This supports slice notation to get all values in a given
    time-frame. If you do not supply a step, you'll just get the
    values, with no indication of when they're from exactly --
    so explicitly supply a step of 1 to get the value at each point in
    the slice, or use the ``future`` and ``past`` methods to get read-only
    mappings of data relative to when you last got an item from this.

    Unlike slices of eg. lists, you can slice with a start greater than the stop
    even if you don't supply a step. That will get you values in reverse order,
    still without retaining the revision they're from.

    """
    __slots__ = ('_future', '_past')

    DEQUE_THRESHOLD = 50
    """How long my past or future has to get before I'll turn it into a deque"""

    def future(self, rev=None):
        """Return a Mapping of items after the given revision."""
        if rev is not None:
            self.seek(rev)
        return WindowDictFutureView(self._future)

    def past(self, rev=None):
        """Return a Mapping of items at or before the given revision."""
        if rev is not None:
            self.seek(rev)
        return WindowDictPastView(self._past)

    def seek(self, rev):
        """Arrange the caches to help look up the given revision."""
        # TODO: binary search? Perhaps only when one or the other
        # deque is very large?
        if not self:
            return
        if self._past and self._past[-1][0] <= rev and (
                not self._future or self._future[0][0] > rev
        ):
            return
        if self._past is None:
            self._past = []
        if self._future:
            popper = getattr(self._future, 'popleft', lambda: self._future.pop(0))
            while self._future and self._future[0][0] <= rev:
                self._past.append(popper())
        if self._past and self._future is None:
            self._future = []
        if self._past:
            prepender = getattr(self._future, 'appendleft', lambda x: self._future.insert(0, x))
            while self._past and self._past[-1][0] > rev:
                prepender(self._past.pop())

    def rev_gettable(self, rev):
        if self._past:
            return rev >= self._past[0][0]
        elif self._future:
            return rev >= self._future[0][0]
        else:
            return False

    def rev_before(self, rev):
        """Return the latest past rev on which the value changed."""
        self.seek(rev)
        if self._past:
            return self._past[-1][0]

    def rev_after(self, rev):
        """Return the earliest future rev on which the value will change."""
        self.seek(rev)
        if self._future:
            return self._future[0][0]

    def truncate(self, rev):
        """Delete everything after the given revision."""
        self.seek(rev)
        self._future = None

    @property
    def beginning(self):
        if self._past:
            return self._past[0][0]
        elif self._future:
            return self._future[0][0]
        else:
            raise HistoryError("No history yet")

    @property
    def end(self):
        if self._future:
            return self._future[-1][0]
        elif self._past:
            return self._past[-1][0]
        else:
            raise HistoryError("No history yet")

    def keys(self):
        return WindowDictKeysView(self)

    def items(self):
        return WindowDictItemsView(self)

    def values(self):
        return WindowDictValuesView(self)

    def __bool__(self):
        return bool(self._past) or bool(self._future)

    def __init__(self, data=None):
        if hasattr(data, 'items'):
            if len(data) > self.DEQUE_THRESHOLD:
                self._past = deque(sorted(data.items()))
            else:
                self._past = list(sorted(data.items()))
        elif data:
            # assume it's an orderable sequence of pairs
            if len(data) > self.DEQUE_THRESHOLD:
                self._past = deque(sorted(data))
            else:
                self._past = list(sorted(data))
        else:
            self._past = None
        self._future = None

    def __iter__(self):
        if not self:
            return
        if self._past:
            for (rev, v) in self._past:
                yield rev
        if self._future:
            for (rev, v) in self._future:
                yield rev

    def __contains__(self, item):
        if not within_history(item, self):
            return False
        self.seek(item)
        return self._past and self._past[-1][0] == item

    def __len__(self):
        return len(self._past or ()) + len(self._future or ())

    def __getitem__(self, rev):
        if not self:
            raise HistoryError("No history yet")
        if isinstance(rev, slice):
            if None not in (rev.start, rev.stop) and rev.start > rev.stop:
                return WindowDictReverseSlice(self, rev)
            return WindowDictSlice(self, rev)
        self.seek(rev)
        if not self._past:
            raise HistoryError(
                "Revision {} is before the start of history".format(rev)
            )
        ret = self._past[-1][1]
        if ret is None:
            raise HistoryError("Set, then deleted", deleted=True)
        return ret

    def __setitem__(self, rev, v):
        if hasattr(v, 'unwrap') and not hasattr(v, 'no_unwrap'):
            v = v.unwrap()
        if self._past is None:
            self._past = []
        if not self._past and not self._future:
            self._past.append((rev, v))
        elif self._past and rev < self._past[0][0]:
            self._past.insert(0, (rev, v))
        elif self._past and rev == self._past[0][0]:
            self._past[0] = (rev, v)
        else:
            self.seek(rev)
            if not self._past:
                self._past.append((rev, v))
            elif self._past[-1][0] == rev:
                self._past[-1] = (rev, v)
            else:
                assert self._past[-1][0] < rev
                self._past.append((rev, v))
        if type(self._past) is list and len(self._past) > self.DEQUE_THRESHOLD:
            self._past = deque(self._past)
        if type(self._future) is list and len(self._future) > self.DEQUE_THRESHOLD:
            self._future = deque(self._future)

    def __delitem__(self, rev):
        # Not checking for rev's presence at the beginning because
        # to do so would likely require iterating thru history,
        # which I have to do anyway in deleting.
        # But handle degenerate case.
        if not self:
            raise HistoryError("Tried to delete from an empty WindowDict")
        if not self.beginning <= rev <= self.end:
            raise HistoryError("Rev outside of history: {}".format(rev))
        self.seek(rev)
        if self._past[-1][0] != rev:
            raise HistoryError("Rev not present: {}".format(rev))
        del self._past[-1]

    def __repr__(self):
        me = dict(self._past)
        me.update(self._future)
        return "{}({})".format(self.__class__.__name__, me)
