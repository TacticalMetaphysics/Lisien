# This file is part of allegedb, a database abstraction for versioned graphs
# Copyright (c) Zachary Spector. ZacharySpector@gmail.com
from collections import deque, MutableMapping, KeysView, ItemsView, ValuesView


class HistoryError(KeyError):
    """You tried to access the past in a bad way."""


def within_history(rev, windowdict):
    """Return whether the windowdict has history at the revision."""
    if not (windowdict._past or windowdict._future):
        return False
    begin = windowdict._past[0][0] if windowdict._past else \
            windowdict._future[0][0]
    end = windowdict._future[-1][0] if windowdict._future else \
          windowdict._past[-1][0]
    if not begin <= rev <= end:
        return False
    return True


class WindowDictKeysView(KeysView):
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
        for rev, v in self._mapping._past:
            yield rev
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
        yield from self._mapping._past
        yield from self._mapping._future


class WindowDictValuesView(ValuesView):
    """Look through all the values that a WindowDict contains."""
    def __contains__(self, value):
        for rev, v in self._mapping._past:
            if v == value:
                return True
        for rev, v in self._mapping._future:
            if v == value:
                return True
        return False

    def __iter__(self):
        for rev, v in self._mapping._past:
            yield v
        for rev, v in self._mapping._future:
            yield v


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

    """
    __slots__ = ['_past', '_future']

    def seek(self, rev):
        """Arrange the caches to help look up the given revision."""
        # TODO: binary search? Perhaps only when one or the other
        # deque is very large?
        if self._past and self._past[-1][0] <= rev and (
                not self._future or self._future[0][0] > rev
        ):
            return
        while self._future and self._future[0][0] <= rev:
            self._past.append(self._future.popleft())
        while self._past and self._past[-1][0] > rev:
            self._future.appendleft(self._past.pop())

    def has_exact_rev(self, rev):
        """Return whether I have a value at this exact revision."""
        if not within_history(rev, self):
            return False
        self.seek(rev)
        return self._past and self._past[-1][0] == rev

    def rev_before(self, rev):
        """Return the last rev prior to the given one on which the value changed."""
        self.seek(rev)
        return self._past[-1][0]

    def rev_after(self, rev):
        """Return the next rev after the given one on which the value will change, or None if it never will."""
        self.seek(rev)
        if self._future:
            return self._future[0][0]

    def keys(self):
        return WindowDictKeysView(self)

    def items(self):
        return WindowDictItemsView(self)

    def values(self):
        return WindowDictValuesView(self)

    def __init__(self, data={}):
        self._past = deque(sorted(data.items()))
        self._future = deque()

    def __iter__(self):
        for (rev, v) in self._past:
            yield rev
        for (rev, v) in self._future:
            yield rev

    def __len__(self):
        return len(self._past) + len(self._future)

    def __getitem__(self, rev):
        self.seek(rev)
        if not self._past:
            raise HistoryError(
                "Revision {} is before the start of history".format(rev)
            )
        ret = self._past[-1][1]
        if ret is None:
            raise HistoryError("Set, then deleted")
        return ret

    def __setitem__(self, rev, v):
        if not self._past and not self._future:
            self._past.append((rev, v))
        elif rev < self._past[0][0]:
            self._past.appendleft((rev, v))
        elif rev == self._past[0][0]:
            self._past[0] = (rev, v)
        else:
            self.seek(rev)
            if not self._past:
                self._past.append((rev, v))
            elif self._past[-1][0] == rev:
                self._past[-1][1] = v
            else:
                assert self._past[-1][0] < rev
                self._past.append((rev, v))

    def __delitem__(self, rev):
        # Not checking for rev's presence at the beginning because
        # to do so would likely require iterating thru history,
        # which I have to do anyway in deleting.
        # But handle degenerate case.
        if not within_history(rev, self):
            raise HistoryError("Rev outside of history: {}".format(rev))
        name = '_past' if rev <= self._rev else '_future'
        stack = getattr(self, name)
        waste = deque()
        setattr(self, name, waste)
        deleted = False
        while stack:
            (r, v) = stack.popleft()
            if r != rev:
                waste.append((r, v))
            else:
                assert not deleted
                deleted = True
        if not deleted:
            raise HistoryError("Rev not present: {}".format(rev))

    def __repr__(self):
        return "WindowDict({})".format(repr(dict(self)))


class WindowDefaultDict(WindowDict):
    """A WindowDict that has a default type.

    See PickyDefaultDict for details of how to use the default class.

    """
    __slots__ = ['_future', '_past', 'type', 'args_munger', 'kwargs_munger']

    def __init__(
            self, type,
            args_munger=lambda k: tuple(),
            kwargs_munger=lambda k: {},
            data={}
    ):
        super(WindowDefaultDict, self).__init__(data)
        self.type = type
        self.args_munger = args_munger
        self.kwargs_munger = kwargs_munger

    def __getitem__(self, k):
        if k in self:
            return super(WindowDefaultDict, self).__getitem__(k)
        ret = self[k] = self.type(
            *self.args_munger(k), **self.kwargs_munger(k)
        )
        return ret


class FuturistWindowDict(WindowDict):
    """A WindowDict that does not let you rewrite the past."""

    def __setitem__(self, rev, v):
        if not self._past and not self._future:
            self._past.append((rev, v))
            return
        self.seek(rev)
        if self._future:
            raise HistoryError(
                "Already have some history after {}".format(rev)
            )
        if not self._past or rev > self._past[-1][0]:
            self._past.append((rev, v))
        elif rev == self._past[-1][0]:
            self._past[-1] = (rev, v)
        else:
            raise HistoryError(
                "Already have some history after {} "
                "(and my seek function is broken?)".format(rev)
            )


class PickyDefaultDict(dict):
    """A ``defaultdict`` alternative that requires values of a specific type.

    Pass some type object (such as a class) to the constructor to
    specify what type to use by default, which is the only type I will
    accept.

    Default values are constructed with no arguments by default;
    supply ``args_munger`` and/or ``kwargs_munger`` to override this.
    They take arguments ``self`` and the unused key being looked up.

    """
    __slots__ = ['type', 'args_munger', 'kwargs_munger']

    def __init__(
            self, type=object,
            args_munger=lambda self, k: tuple(),
            kwargs_munger=lambda self, k: dict()
    ):
        self.type = type
        self.args_munger = args_munger
        self.kwargs_munger = kwargs_munger

    def __getitem__(self, k):
        if k in self:
            return super(PickyDefaultDict, self).__getitem__(k)
        ret = self[k] = self.type(
            *self.args_munger(self, k),
            **self.kwargs_munger(self, k)
        )
        return ret

    def __setitem__(self, k, v):
        if not isinstance(v, self.type):
            raise TypeError("Expected {}, got {}".format(self.type, type(v)))
        super(PickyDefaultDict, self).__setitem__(k, v)


class StructuredDefaultDict(dict):
    """A ``defaultdict``-like class that expects values stored at a specific depth.

    Requires an integer to tell it how many layers deep to go.
    The innermost layer will be ``PickyDefaultDict``, which will take the
    ``type``, ``args_munger``, and ``kwargs_munger`` arguments supplied
    to my constructor.

    This will never accept manual assignments at any layer but the deepest.

    """
    __slots__ = ['layer', 'type', 'args_munger', 'kwargs_munger']

    def __init__(
            self, layers, type=object,
            args_munger=lambda self, k: tuple(),
            kwargs_munger=lambda self, k: dict()
    ):
        if layers < 1:
            raise ValueError("Not enough layers")
        self.layer = layers
        self.type = type
        self.args_munger = args_munger
        self.kwargs_munger = kwargs_munger

    def __getitem__(self, k):
        if k in self:
            return super(StructuredDefaultDict, self).__getitem__(k)
        if self.layer < 2:
            ret = PickyDefaultDict(
                self.type, self.args_munger, self.kwargs_munger
            )
        else:
            ret = StructuredDefaultDict(
                self.layer-1, self.type,
                self.args_munger, self.kwargs_munger
            )
        super(StructuredDefaultDict, self).__setitem__(k, ret)
        return ret

    def __setitem__(self, k, v):
        raise TypeError("Can't set layer {}".format(self.layer))


class Cache(object):
    """A data store that's useful for tracking graph revisions."""
    def __init__(self, db):
        self.db = db
        self.parents = StructuredDefaultDict(3, FuturistWindowDict)
        self.keys = StructuredDefaultDict(2, FuturistWindowDict)
        self.keycache = {}
        self.branches = StructuredDefaultDict(1, FuturistWindowDict)
        self.shallow = PickyDefaultDict(FuturistWindowDict)
        self.shallower = {}

    def _forward_branch(self, mapp, key, branch, rev, newval=None, copy=True):
        if branch not in mapp[key]:
            for b, t in self.db._active_branches():
                if b in mapp[key] and (
                        newval is None or
                        mapp[key][b][t] != newval
                ):
                    mapp[key][branch][rev] \
                        = mapp[key][b][t].copy() if copy else mapp[key][b][t]
            raise ValueError("No data to forward")

    def _forward_keycache(self, parentity, branch, rev):
        keycache_key = parentity + (branch,)
        if keycache_key in self.keycache:
            return
        kc = FuturistWindowDict()
        for (b, r) in self.db._active_branches():
            other_branch_key = parentity + (b,)
            if other_branch_key in self.keycache and \
               r in self.keycache[other_branch_key]:
                kc[rev] = self.keycache[other_branch_key][r].copy()
                break
        self.keycache[keycache_key] = kc

    def store(self, *args):
        """Put a value in various dictionaries for later .retrieve(...).

        Needs at least five arguments, of which the -1th is the value
        to store, the -2th is the revision to store it at, the -3th
        is the branch to store it in, the -4th is the key to store it
        under, and the rest specify the entity (eg. node, edge, graph)
        that it's about.

        """
        entity, key, branch, rev, value = args[-5:]
        parent = args[:-5]
        if parent:
            try:
                self._forward_branch(
                    self.parents[parent][entity], key, branch, rev, value
                )
            except ValueError:
                pass
            self.parents[parent][entity][key][branch][rev] = value
        try:
            self._forward_branch(
                self.keys[parent+(entity,)], key, branch, rev, value
            )
        except ValueError:
            pass
        self.keys[parent+(entity,)][key][branch][rev] = value
        self.branches[parent+(entity,key)][branch][rev] = value
        self.shallow[parent+(entity,key,branch)][rev] = value
        self.shallower[parent+(entity,key,branch,rev)] = value
        self._forward_keycache(parent+(entity,), branch, rev)
        self._forward_keycache((entity,), branch, rev)
        keycached = None
        for kc in (
                self.keycache[parent+(entity,branch)],
                self.keycache[(entity,branch)]
        ):
            if kc is keycached:
                return
            keycached = kc
            if rev in kc:
                if not kc.has_exact_rev(rev):
                    kc[rev] = kc[rev].copy()
                if value is None:
                    kc[rev].discard(key)
                else:
                    kc[rev].add(key)
            elif value is None:
                kc[rev] = set()
            else:
                kc[rev] = set([key])

    def retrieve(self, *args):
        """Get a value previously .store(...)'d.

        Needs at least four arguments. The -1th is the revision
        that you want, the -2th is the branch, the -3th is the key,
        and all the rest specify the entity (eg. graph, node, edge).

        """
        try:
            ret = self.shallower[args]
            if ret is None:
                raise HistoryError("Set, then deleted")
            return ret
        except KeyError:
            pass
        entity = args[:-3]
        key, branch, rev = args[-3:]
        if rev not in self.shallow[entity+(key, branch)]:
            for (b, r) in self.db._active_branches(branch, rev):
                if (
                        b in self.branches[entity+(key,)]
                        and r in self.branches[entity+(key,)][b]
                ):
                    v = self.branches[entity+(key,)][b][r]
                    self.store(*entity+(key, branch, rev, v))
                    self.store(*entity+(key, b, r, v))
                    break
            else:
                self.store(*entity+(key, branch, rev, None))
        ret = self.shallower[args] = self.shallow[entity+(key,branch)][rev]
        return ret

    def iter_entities_or_keys(self, *args):
        """Iterate over the keys an entity has, if you specify an entity.

        Otherwise iterate over the entities themselves, or at any rate the
        tuple specifying which entity.

        Needs at least two arguments, the branch and the revision. Any
        that come before that will be taken to identify the entity.

        """
        entity = args[:-2]
        branch, rev = args[-2:]
        self._forward_keycache(entity, branch, rev)
        try:
            keys = self.keycache[entity+(branch,)][rev]
        except KeyError:
            return
        yield from keys
    iter_entities = iter_keys = iter_entity_keys = iter_entities_or_keys

    def count_entities_or_keys(self, *args):
        """Return the number of keys an entity has, if you specify an entity.

        Otherwise return the number of entities.

        Needs at least two arguments, the branch and the revision. Any
        that come before that will be taken to identify the entity.

        """
        entity = args[:-2]
        branch, rev = args[-2:]
        self._forward_keycache(entity, branch, rev)
        try:
            return len(self.keycache[entity+(branch,)][rev])
        except KeyError:
            return 0
    count_entities = count_keys = count_entity_keys = count_entities_or_keys

    def contains_entity_or_key(self, *args):
        """Check if an entity has a key at the given time, if entity specified.

        Otherwise check if the entity exists.

        Needs at least three arguments, the key, the branch, and the revision.
        Any that come before that will be taken to identify the entity.

        """
        try:
            return self.shallower[args] is not None
        except KeyError:
            pass
        entity = args[:-3]
        key, branch, rev = args[-3:]
        if key not in self.keys[entity]:
            return False
        self._forward_keycache(entity, branch, rev)
        try:
            keys = self.keycache[entity+(branch,)][rev]
        except KeyError:
            return False
        return key in keys
    contains_entity = contains_key = contains_entity_key \
                      = contains_entity_or_key


class NodesCache(Cache):
    def store(self, graph, node, branch, rev, ex):
        """Store whether a node exists, and create an object for it"""
        if not ex:
            ex = None
        if (graph, node) not in self.db._node_objs:
            self.db._node_objs[(graph, node)] \
                = self.db._make_node(self.db.graph[graph], node)
        Cache.store(self, graph, node, branch, rev, ex)


class EdgesCache(Cache):
    def __init__(self, db):
        Cache.__init__(self, db)
        self.predecessors = StructuredDefaultDict(3, FuturistWindowDict)

    def store(self, graph, nodeA, nodeB, idx, branch, rev, ex):
        """Store whether an edge exists, and create an object for it

        Also stores predecessors for every edge.

        """
        if not ex:
            ex = None
        if (graph, nodeA, nodeB, idx) not in self.db._edge_objs:
            self.db._edge_objs[(graph, nodeA, nodeB, idx)] \
                = self.db._make_edge(self.db.graph[graph], nodeA, nodeB, idx)
        Cache.store(self, graph, nodeA, nodeB, idx, branch, rev, ex)
        self.predecessors[(graph, nodeB)][nodeA][idx][branch][rev] = ex
