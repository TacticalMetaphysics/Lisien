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
from collections import deque
from collections.abc import Mapping, MutableMapping, KeysView, ItemsView, ValuesView
from operator import itemgetter, lt, le
from itertools import chain
try:
    import cython
except ImportError:
    class cython:
        def locals(**kwargs):
            def passthru(fun):
                return fun
            return passthru
        cfunc = locals
        int = None
        bint = None

get0 = itemgetter(0)
get1 = itemgetter(1)

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
        for past_state in branchd[turn_to][:tick_to+1]:
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
            windowdict._future[-1][0]
    end = windowdict._future[0][0] if windowdict._future else \
          windowdict._past[-1][0]
    return begin <= rev <= end


class WindowDictKeysView(KeysView):
    """Look through all the keys a WindowDict contains."""
    def __contains__(self, rev):
        return rev in self._mapping._keys

    def __iter__(self):
        past = self._mapping._past
        future = self._mapping._future
        if past:
            yield from map(get0, past)
        if future:
            yield from map(get0, reversed(future))


class WindowDictItemsView(ItemsView):
    """Look through everything a WindowDict contains."""
    def __contains__(self, item):
        (rev, v) = item
        mapp = self._mapping
        if not within_history(rev, mapp):
            return False
        for mrev, mv in mapp._past:
            if mrev == rev:
                return mv == v
        for mrev, mv in mapp._future:
            if mrev == rev:
                return mv == v
        return False

    def __iter__(self):
        past = self._mapping._past
        future = self._mapping._future
        if past:
            yield from past
        if future:
            yield from future


class WindowDictPastFutureKeysView(KeysView):
    """View on a WindowDict's keys relative to last lookup"""
    def __iter__(self):
        if not self._mapping.stack:
            return
        yield from map(get0, reversed(self._mapping.stack))

    def __contains__(self, item):
        return item in self._mapping and item in map(get0, self._mapping.stack)


class WindowDictPastFutureItemsView(ItemsView):
    def __iter__(self):
        if not self._mapping.stack:
            return
        yield from reversed(self._mapping.stack)

    def __contains__(self, item):
        if self._out_of_range(item, self._mapping.stack):
            return False
        i0, i1 = item
        if i0 not in self._mapping:
            return False
        for j0, j1 in self._mapping.stack:
            if i0 == j0:
                return i1 == j1
        return False


class WindowDictPastItemsView(WindowDictPastFutureItemsView):
    @staticmethod
    def _out_of_range(item, stack):
        return item[0] < stack[0][0] or item[0] > stack[-1][0]


class WindowDictFutureItemsView(WindowDictPastFutureItemsView):
    """View on a WindowDict's future items relative to last lookup"""
    @staticmethod
    def _out_of_range(item, stack):
        return item[0] < stack[-1][0] or item[0] > stack[0][0]


class WindowDictPastFutureValuesView(ValuesView):
    """Abstract class for views on the past or future values of a WindowDict"""
    def __iter__(self):
        stack = self._mapping.stack
        if not stack:
            return
        yield from map(get1, reversed(stack))

    def __contains__(self, item):
        stack = self._mapping.stack
        if not stack:
            return False
        return item in map(get1, stack)


class WindowDictValuesView(ValuesView):
    """Look through all the values that a WindowDict contains."""
    def __contains__(self, value):
        past = self._mapping._past
        future = self._mapping._future
        if past:
            for rev, v in past:
                if v == value:
                    return True
        if future:
            for rev, v in future:
                if v == value:
                    return True
        return False

    def __iter__(self):
        past = self._mapping._past
        future = self._mapping._future
        if past:
            yield from map(get1, past)
        if future:
            yield from map(get1, future)


class WindowDictPastFutureView(Mapping):
    """Abstract class for historical views on WindowDict"""
    __slots__ = ('stack',)

    def __init__(self, stack):
        self.stack = stack

    def __len__(self):
        stack = self.stack
        if not stack:
            return 0
        return len(stack)


class WindowDictPastView(WindowDictPastFutureView):
    """Read-only mapping of just the past of a WindowDict"""
    def __iter__(self):
        stack = self.stack
        if not stack:
            return
        yield from map(get0, reversed(stack))

    def __getitem__(self, key):
        stack = self.stack
        if not stack or key < stack[0][0] or key > stack[-1][0]:
            raise KeyError
        for rev, value in stack:
            if rev == key:
                return value
        raise KeyError

    def keys(self):
        return WindowDictPastFutureKeysView(self)

    def items(self):
        return WindowDictPastItemsView(self)

    def values(self):
        return WindowDictPastFutureValuesView(self)


class WindowDictFutureView(WindowDictPastFutureView):
    """Read-only mapping of just the future of a WindowDict"""
    def __iter__(self):
        stack = self.stack
        if not stack:
            return
        yield from map(get0, reversed(stack))

    def __getitem__(self, key):
        stack = self.stack
        if not stack or key < stack[-1][0] or key > stack[0][0]:
            raise KeyError
        for rev, value in stack:
            if rev == key:
                return value
        raise KeyError

    def keys(self):
        return WindowDictPastFutureKeysView(self)

    def items(self):
        return WindowDictFutureItemsView(self)

    def values(self):
        return WindowDictPastFutureValuesView(self)


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
            for i in range(slic.start or dic.beginning, slic.stop or dic.end+1, slic.step):
                yield dic[i]
            return
        if slic.start is None and slic.stop is None:
            yield from map(get1, dic._past)
            yield from map(get1, reversed(dic._future))
        elif None not in (slic.start, slic.stop):
            if slic.stop == slic.start:
                yield dic[slic.stop]
                return
            past = dic._past
            future = dic._future
            if slic.start < slic.stop:
                left, right = slic.start, slic.stop
                dic.seek(right)
                if not past:
                    return
                if past[-1][0] == right:
                    future.append(past.pop())
                cmp = lt
            else:
                left, right = slic.stop, slic.start
                dic.seek(right)
                if not past:
                    return
                cmp = le
            it = iter(past)
            p0, p1 = next(it)
            while cmp(p0, left):
                p0, p1 = next(it)
            else:
                yield p1
            yield from map(get1, it)
        elif slic.start is None:
            stac = dic._past + list(reversed(dic._future))
            while stac and stac[-1][0] >= slic.stop:
                stac.pop()
            yield from map(get1, stac)
            return
        else:  # slic.stop is None
            if not dic._past and not dic._future:
                return
            chan = chain(dic._past, reversed(dic._future))
            nxt = next(chan)
            while nxt[0] < slic.start:
                try:
                    nxt = next(chan)
                except StopIteration:
                    return
            yield get1(nxt)
            yield from map(get1, chan)


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
            return
        if slic.start is None and slic.stop is None:
            yield from map(get1, dic._future)
            yield from map(get1, reversed(dic._past))
        elif None not in (slic.start, slic.stop):
            if slic.start == slic.stop:
                yield dic[slic.stop]
                return
            if slic.start < slic.stop:
                left, right = slic.start, slic.stop
                dic.seek(right)
                it = reversed(dic._past)
                next(it)
                cmp = lt
            else:
                left, right = slic.stop, slic.start
                dic.seek(right)
                it = reversed(dic._past)
                cmp = le
            for frev, fv in it:
                if cmp(frev, left):
                    return
                yield fv
        elif slic.start is None:
            stac = dic._past + list(reversed(dic._future))
            while stac and stac[-1][0] >= slic.stop:
                stac.pop()
            yield from map(get1, reversed(stac))
        else:  # slic.stop is None
            stac = deque(dic._past)
            stac.extend(reversed(dic._future))
            while stac and stac[0][0] < slic.start:
                stac.popleft()
            yield from map(get1, reversed(stac))


class WindowDict(MutableMapping):
    """A dict that keeps every value that a variable has had over time.

    Look up a revision number in this dict and it will give you the
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
    __slots__ = ('_future', '_past', '_keys', 'beginning', 'end', '_last')

    def future(self, rev=None):
        """Return a Mapping of items after the given revision.

        Default revision is the last one looked up.

        """
        if rev is not None:
            self.seek(rev)
        return WindowDictFutureView(self._future)

    def past(self, rev=None):
        """Return a Mapping of items at or before the given revision.

        Default revision is the last one looked up.

        """
        if rev is not None:
            self.seek(rev)
        return WindowDictPastView(self._past)

    @cython.locals(rev=cython.int, past_end=cython.int, future_start=cython.int)
    def seek(self, rev):
        """Arrange the caches to help look up the given revision."""
        # TODO: binary search? Perhaps only when one or the other
        # stack is very large?
        if rev == self._last:
            return
        if type(rev) is not int:
            raise TypeError("rev must be int")
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

    def rev_before(self, rev: int) -> int:
        """Return the latest past rev on which the value changed."""
        self.seek(rev)
        if self._past:
            return self._past[-1][0]

    def rev_after(self, rev: int) -> int:
        """Return the earliest future rev on which the value will change."""
        self.seek(rev)
        if self._future:
            return self._future[-1][0]

    def initial(self):
        """Return the earliest value we have"""
        if self._past:
            return self._past[0][1]
        if self._future:
            return self._future[-1][1]
        raise KeyError("No data")

    def final(self):
        """Return the latest value we have"""
        if self._future:
            return self._future[0][1]
        if self._past:
            return self._past[-1][1]
        raise KeyError("No data")

    def truncate(self, rev: int) -> None:
        """Delete everything after the given revision."""
        self.seek(rev)
        self._keys.difference_update(map(get0, self._future))
        self._future = []
        if not self._past:
            self.beginning = self.end = None

    def keys(self):
        return WindowDictKeysView(self)

    def items(self):
        return WindowDictItemsView(self)

    def values(self):
        return WindowDictValuesView(self)

    def __bool__(self):
        return bool(self._past) or bool(self._future)

    def __init__(self, data=None):
        if not data:
            self._past = []
        elif hasattr(data, 'items'):
            self._past = list(data.items())
        else:
            # assume it's an orderable sequence of pairs
            self._past = list(data)
        self._past.sort()
        self._future = []
        self._keys = set(map(get0, self._past))
        self.beginning = None if not self._past else self._past[0][0]
        self.end = None if not self._past else self._past[-1][0]
        self._last = None

    def __iter__(self):
        if not self:
            return
        if self._past:
            yield from map(get0, self._past)
        if self._future:
            yield from map(get0, self._future)

    def __contains__(self, item):
        return item in self._keys

    def __len__(self):
        return len(self._past) + len(self._future)

    def __getitem__(self, rev):
        if isinstance(rev, slice):
            if None not in (rev.start, rev.stop) and rev.start > rev.stop:
                return WindowDictReverseSlice(self, rev)
            return WindowDictSlice(self, rev)
        self.seek(rev)
        past = self._past
        if not past:
            raise HistoryError(
                "Revision {} is before the start of history".format(rev)
            )
        return past[-1][1]

    @cython.locals(rev=cython.int)
    def __setitem__(self, rev, v):
        past = self._past
        if past or self._future:
            self.seek(rev)
            if past:
                if past[-1][0] == rev:
                    past[-1] = (rev, v)
                else:
                    past.append((rev, v))
            else:
                past.append((rev, v))
                self.beginning = rev
            end = self.end
            if end is None or rev > end:
                self.end = rev
        else:
            past.append((rev, v))
            self.beginning = self.end = self._last = rev
        self._keys.add(rev)

    @cython.locals(rev=cython.int, past_end=cython.int)
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
        past = self._past
        future = self._future
        if not past or past[-1][0] != rev:
            raise HistoryError("Rev not present: {}".format(rev))
        del past[-1]
        if not past:
            if future:
                self.beginning = future[-1][0]
            else:
                self.beginning = self.end = None
        elif not future:
            self.end = past[-1][0]
        self._keys.remove(rev)

    def __repr__(self):
        me = dict(self._past)
        me.update(self._future)
        return "{}({})".format(self.__class__.__name__, me)


class FuturistWindowDict(WindowDict):
    """A WindowDict that does not let you rewrite the past."""
    __slots__ = ('_future', '_past', 'beginning')

    def __setitem__(self, rev, v):
        if hasattr(v, 'unwrap') and not hasattr(v, 'no_unwrap'):
            v = v.unwrap()
        self.seek(rev)
        past = self._past
        future = self._future
        if future:
            raise HistoryError(
                "Already have some history after {}".format(rev)
            )
        if not past:
            self.beginning = rev
            past.append((rev, v))
        elif rev > past[-1][0]:
            past.append((rev, v))
        elif rev == past[-1][0]:
            past[-1] = (rev, v)
        else:
            raise HistoryError(
                "Already have some history after {} "
                "(and my seek function is broken?)".format(rev)
            )
        if self.end is None or rev > self.end:
            self.end = rev
        self._keys.add(rev)


class TurnDict(FuturistWindowDict):
    __slots__ = ('_future', '_past')
    cls = FuturistWindowDict

    def __setitem__(self, turn, value):
        if type(value) is not FuturistWindowDict:
            value = FuturistWindowDict(value)
        FuturistWindowDict.__setitem__(self, turn, value)


class SettingsTurnDict(WindowDict):
    __slots__ = ('_future', '_past')
    cls = WindowDict

    def __setitem__(self, turn, value):
        if type(value) is not WindowDict:
            value = WindowDict(value)
        WindowDict.__setitem__(self, turn, value)
