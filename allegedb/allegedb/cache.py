# This file is part of allegedb, a database abstraction for versioned graphs
# Copyright (c) Zachary Spector. ZacharySpector@gmail.com
"""Classes for in-memory storage and retrieval of historical graph data.

The main one to use is ``Cache``, mostly for its ``store`` and ``retrieve``
methods. But if you need to store historical data some other way,
you might want to store it in a ``WindowDict``.

"""
from copy import copy as copier
from collections import deque, MutableMapping, KeysView, ItemsView, ValuesView


TESTING = True
"""Change this to True to validate kecaches whenever they change.

It will make things very slow.

"""


class HistoryError(KeyError):
    """You tried to access the past in a bad way."""

    def __init__(self, *args, deleted=False):
        super().__init__(*args)
        self.deleted = deleted


def within_history(rev, windowdict):
    """Return whether the windowdict has history at the revision."""
    if not (windowdict._past or windowdict._future):
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
        """Return the latest past rev on which the value changed."""
        self.seek(rev)
        if self._past:
            return self._past[-1][0]

    def rev_after(self, rev):
        """Return the earliest future rev on which the value will change."""
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
            raise HistoryError("Set, then deleted", deleted=True)
        return ret

    def __setitem__(self, rev, v):
        if not self._past and not self._future:
            self._past.append((rev, v))
        elif self._past and rev < self._past[0][0]:
            self._past.appendleft((rev, v))
        elif self._past and rev == self._past[0][0]:
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
        ret = {
            rev: v for (rev, v) in self._past
        }
        ret.update({
            rev: v for (rev, v) in self._future
        })
        return "WindowDict({})".format(ret)


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
    __slots__ = ['db', 'parents', 'keys', 'keycache', 'branches', 'shallow', 'shallower']

    def __init__(self, db):
        self.db = db
        self.parents = StructuredDefaultDict(3, FuturistWindowDict)
        """Entity data keyed by the entities' parents.

        An entity's parent is what it's contained in. When speaking of a node,
        this is its graph. When speaking of an edge, the parent is usually the
        graph and the origin in a pair, though for multigraphs the destination
        might be part of the parent as well.

        Deeper layers of this cache are keyed by branch and revision.

        """
        self.keys = StructuredDefaultDict(2, FuturistWindowDict)
        """Cache of entity data keyed by the entities themselves.

        That means the whole tuple identifying the entity is the
        top-level key in this cache here. The second-to-top level
        is the key within the entity.

        Deeper layers of this cache are keyed by branch and revision.

        """
        self.keycache = {}
        """A simple dict holding sets of keys an entity has now."""
        self.branches = StructuredDefaultDict(1, FuturistWindowDict)
        """A less structured alternative to ``keys``.

        For when you already know the entity and the key within it,
        but still need to iterate through history to find the value.

        """
        self.shallow = PickyDefaultDict(FuturistWindowDict)
        """Less structured alternative to ``branches`` ."""
        self.shallower = {}
        """Even less structured alternative to ``shallow``."""

    def _forward_valcache(self, cache, branch, rev, copy=True):
        if branch in cache:
            return cache[branch].get(rev, None)
        for b, r in self.db._active_branches(branch, rev):
            if b in cache:
                cache[branch][r] = copier(cache[b][r]) if copy else cache[b].get(r, None)
                return cache[branch].get(r, None)

    def _forward_keycachelike(self, keycache, keys, slow_iter_keys, parentity, branch, rev):
        keycache_key = parentity + (branch,)
        if keycache_key in keycache:
            kc = keycache[keycache_key]
            try:
                if not kc.has_exact_rev(rev):
                    if kc.rev_before(rev) == rev - 1:
                        kc[rev] = kc[rev].copy()
                    else:
                        kc[rev] = set(slow_iter_keys(keys[parentity], branch, rev))
                return kc[rev]
            except HistoryError:
                pass
        kc = keycache[keycache_key] = FuturistWindowDict()
        for (b, r) in self.db._active_branches(branch, rev):
            other_branch_key = parentity + (b,)
            if other_branch_key in keycache and \
               r in keycache[other_branch_key]:
                kc[rev] = keycache[other_branch_key][r].copy()
                break
        else:
            kc[rev] = set(slow_iter_keys(keys[parentity], branch, rev))
        return kc[rev]

    def _forward_keycache(self, parentity, branch, rev):
        return self._forward_keycachelike(self.keycache, self.keys, self._slow_iter_keys, parentity, branch, rev)

    def _update_keycache(self, entpar, branch, rev, key, value):
        kc = self._forward_keycache(entpar, branch, rev)
        if value is None:
            kc.discard(key)
        else:
            kc.add(key)
        return kc

    def _slow_iter_keys(self, cache, branch, rev):
        for key in cache:
            for (branch, rev) in self.db._active_branches(branch, rev):
                try:
                    if cache[key][branch][rev] is not None:
                        yield key
                except HistoryError as err:
                    if err.deleted:
                        break

    def _validate_keycache(self, cache, keycache, branch, rev, entpar):
        if not TESTING:
            return
        kc = keycache
        correct = set(self._slow_iter_keys(cache, branch, rev))
        assert kc == correct, """
        Invalid keycache for {} at branch {}, rev {}
        """.format(entpar, branch, rev)

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
            if branch not in self.parents[parent][entity][key]:
                self._forward_valcache(
                    self.parents[parent][entity][key], branch, rev
                )
            self.parents[parent][entity][key][branch][rev] = value
        if branch not in self.keys[parent+(entity,)][key]:
            self._forward_valcache(
                self.keys[parent+(entity,)][key], branch, rev
            )
        self.keys[parent+(entity,)][key][branch][rev] = value
        if branch not in self.branches[parent+(entity, key)]:
            self._forward_valcache(
                self.branches[parent+(entity, key)], branch, rev
            )
        self.branches[parent+(entity, key)][branch][rev] = value
        self.shallow[parent+(entity, key, branch)][rev] = value
        self.shallower[parent+(entity, key, branch, rev)] = value
        if parent:
            kc = self._update_keycache(
                parent+(entity,), branch, rev, key, value
            )
            self._validate_keycache(
                self.parents[parent][entity],
                kc,
                branch, rev, parent+(entity,)
            )
            self._validate_keycache(
                self.keys[parent+(entity,)],
                kc,
                branch, rev, parent+(entity,)
            )
        else:
            self._validate_keycache(
                self.keys[(entity,)],
                self._update_keycache((entity,), branch, rev, key, value),
                branch, rev, (entity,)
            )

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
                    self.store(*entity+(key, branch, r, v))
                    if rev > r:
                        self.store(*entity+(key, branch, rev, v))
                    break
            else:
                raise KeyError
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
        self._forward_keycache(entity, branch, rev)
        try:
            keys = self.keycache[entity+(branch,)][rev]
        except KeyError:
            return False
        return key in keys
    contains_entity = contains_key = contains_entity_key \
                      = contains_entity_or_key


class NodesCache(Cache):
    def __init__(self, db):
        super().__init__(db)
        self._make_node = db._make_node

    def store(self, graph, node, branch, rev, ex):
        """Store whether a node exists, and create an object for it"""
        if not ex:
            ex = None
        if (graph, node) not in self.db._node_objs:
            self.db._node_objs[(graph, node)] \
                = self._make_node(self.db.graph[graph], node)
        Cache.store(self, graph, node, branch, rev, ex)
        # does this work??
        self._validate_keycache(
            self.keys[(graph,)],
            self._update_keycache((graph,), branch, rev, node, ex),
            branch, rev, (graph,)
        )


class EdgesCache(Cache):
    __slots__ = ['db', 'parents', 'keys', 'keycache', 'branches', 'shallow', 'shallower',
                 'origcache', 'destcache', 'predecessors']

    @property
    def successors(self):
        return self.parents

    def __init__(self, db):
        Cache.__init__(self, db)
        self.destcache = {}
        self.origcache = {}
        self.predecessors = StructuredDefaultDict(3, FuturistWindowDict)

    def _slow_iter_successors(self, cache, branch, rev):
        for nodeB, nodeBs in cache.items():
            for idx in self._slow_iter_keys(nodeBs, branch, rev):
                yield nodeB
                break

    def _slow_iter_predecessors(self, cache, branch, rev):
        for nodeA, nodeAs in cache.items():
            for idx in self._slow_iter_keys(nodeAs, branch, rev):
                yield nodeA
                break

    def _forward_destcache(self, graph, orig, branch, rev):
        return self._forward_keycachelike(self.destcache, self.successors, self._slow_iter_successors, (graph, orig), branch, rev)

    def _update_destcache(self, graph, orig, branch, rev, dest, value):
        kc = self._forward_destcache(graph, orig, branch, rev)
        if value is None:
            kc.discard(dest)
        else:
            kc.add(dest)
        return kc

    def _forward_origcache(self, graph, dest, branch, rev):
        return self._forward_keycachelike(self.origcache, self.predecessors, self._slow_iter_predecessors, (graph, dest), branch, rev)

    def _update_origcache(self, graph, dest, branch, rev, orig, value):
        kc = self._forward_origcache(graph, dest, branch, rev)
        if value is None:
            kc.discard(orig)
        else:
            kc.add(orig)
        return kc

    def iter_successors(self, graph, orig, branch, rev):
        self._forward_destcache(graph, orig, branch, rev)
        try:
            succs = self.destcache[(graph, orig, branch)][rev]
        except KeyError:
            return
        yield from succs

    def iter_predecessors(self, graph, dest, branch, rev):
        self._forward_origcache(graph, dest, branch, rev)
        try:
            preds = self.origcache[(graph, dest, branch)][rev]
        except KeyError:
            return
        yield from preds

    def count_successors(self, graph, orig, branch, rev):
        self._forward_destcache(graph, orig, branch, rev)
        try:
            return len(self.destcache[(graph, orig, branch)])[rev]
        except KeyError:
            return 0

    def count_predecessors(self, graph, dest, branch, rev):
        self._forward_origcache(graph, dest, branch, rev)
        try:
            return len(self.origcache[(graph, dest, branch)][rev])
        except KeyError:
            return 0

    def has_successor(self, graph, orig, dest, branch, rev):
        self._forward_keycachelike(self.destcache, self.successors, self._slow_iter_successors, (graph, orig), branch, rev)
        return dest in self.destcache[(graph, orig, branch)][rev]
    
    def has_predecessor(self, graph, dest, orig, branch, rev):
        self._forward_keycachelike(self.origcache, self.predecessors, self._slow_iter_predecessors, (graph, dest), branch, rev)
        return orig in self.origcache[(graph, orig, branch)][rev]

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
        oc = self._update_origcache(graph, nodeB, branch, rev, nodeA, ex)
        dc = self._update_destcache(graph, nodeA, branch, rev, nodeB, ex)
        if TESTING:
            correct_oc = set(self._slow_iter_predecessors(self.predecessors[(graph, nodeB)], branch, rev))
            assert correct_oc == oc
            correct_dc = set(self._slow_iter_successors(self.successors[(graph, nodeA)], branch, rev))
            assert correct_dc == dc
