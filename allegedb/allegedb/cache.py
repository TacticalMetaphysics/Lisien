# This file is part of allegedb, a database abstraction for versioned graphs
# Copyright (c) Zachary Spector. ZacharySpector@gmail.com
"""Classes for in-memory storage and retrieval of historical graph data.

The main one to use is ``Cache``, mostly for its ``store`` and ``retrieve``
methods. But if you need to store historical data some other way,
you might want to store it in a ``WindowDict``.

"""
from copy import copy as copier
from collections import defaultdict, deque, MutableMapping, KeysView, ItemsView, ValuesView


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

    def __init__(self, data={}):
        self._past = deque(sorted(data.items()))
        self._future = deque()

    def __iter__(self):
        for (rev, v) in self._past:
            yield rev
        for (rev, v) in self._future:
            yield rev

    def __contains__(self, item):
        if self._past:
            return item >= self._past[0][0]
        elif self._future:
            return item >= self._future[0][0]
        else:
            return False

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
                self._past[-1] = (rev, v)
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
        name = '_past' if self._past and rev <= self._past[-1][0] else '_future'
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
        return "{}({})".format(self.__class__.__name__, ret)


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


class TurnDict(FuturistWindowDict):
    cls = FuturistWindowDict

    def __getitem__(self, rev):
        try:
            return super().__getitem__(rev)
        except KeyError:
            ret = self[rev] = self.cls()
            return ret

    def __setitem__(self, turn, value):
        if not isinstance(value, self.cls):
            value = self.cls(value)
        super().__setitem__(turn, value)


class PickyDefaultDict(dict):
    """A ``defaultdict`` alternative that requires values of a specific type.

    Pass some type object (such as a class) to the constructor to
    specify what type to use by default, which is the only type I will
    accept.

    Default values are constructed with no arguments by default;
    supply ``args_munger`` and/or ``kwargs_munger`` to override this.
    They take arguments ``self`` and the unused key being looked up.

    """
    __slots__ = ['type', 'args_munger', 'kwargs_munger', 'parent', 'key']

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
            v = self.type(v)
        super(PickyDefaultDict, self).__setitem__(k, v)


class StructuredDefaultDict(dict):
    """A ``defaultdict``-like class that expects values stored at a specific depth.

    Requires an integer to tell it how many layers deep to go.
    The innermost layer will be ``PickyDefaultDict``, which will take the
    ``type``, ``args_munger``, and ``kwargs_munger`` arguments supplied
    to my constructor.

    This will never accept manual assignments at any layer but the deepest.

    """
    __slots__ = ['layer', 'type', 'args_munger', 'kwargs_munger', 'parent', 'key']

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
        ret.parent = self
        ret.key = k
        super(StructuredDefaultDict, self).__setitem__(k, ret)
        return ret

    def __setitem__(self, k, v):
        raise TypeError("Can't set layer {}".format(self.layer))


class Cache(object):
    """A data store that's useful for tracking graph revisions."""
    __slots__ = ['db', 'parents', 'keys', 'keycache', 'branches', 'shallow', 'shallower', 'shallowest']

    def __init__(self, db):
        self.db = db
        self.parents = StructuredDefaultDict(3, TurnDict)
        """Entity data keyed by the entities' parents.

        An entity's parent is what it's contained in. When speaking of a node,
        this is its graph. When speaking of an edge, the parent is usually the
        graph and the origin in a pair, though for multigraphs the destination
        might be part of the parent as well.

        Deeper layers of this cache are keyed by branch and revision.

        """
        self.keys = StructuredDefaultDict(2, TurnDict)
        """Cache of entity data keyed by the entities themselves.

        That means the whole tuple identifying the entity is the
        top-level key in this cache here. The second-to-top level
        is the key within the entity.

        Deeper layers of this cache are keyed by branch, turn, and tick.

        """
        self.keycache = PickyDefaultDict(TurnDict)
        """Keys an entity has at a given turn and tick."""
        self.branches = StructuredDefaultDict(1, TurnDict)
        """A less structured alternative to ``keys``.

        For when you already know the entity and the key within it,
        but still need to iterate through history to find the value.

        """
        self.shallow = PickyDefaultDict(TurnDict)
        """Less structured alternative to ``branches`` ."""
        self.shallower = PickyDefaultDict(WindowDict)
        """Even less structured alternative to ``shallow``."""
        self.shallowest = {}

    def load(self, data, validate=False):
        """Add a bunch of data. It doesn't need to be in chronological order."""
        def fw_upd(*args):
            self._forward_valcaches(*args, validate=validate)
            self._update_keycache(*args, validate=validate)
        store = self._store
        dd3 = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        branch_end = defaultdict(lambda: 0)
        turn_end = defaultdict(lambda: 0)
        for row in data:
            entity, key, branch, turn, tick, value = row[-6:]
            branch_end[branch] = max((turn, branch_end[branch]))
            turn_end[branch, turn] = max((tick, turn_end[branch, turn]))
            dd3[branch][turn][tick].append(row)
            store(*row)
        real_branches = self.db._branches
        real_turn_end = self.db._turn_end
        for branch, end in branch_end.items():
            if branch not in real_branches:
                assert branch == 'trunk'

            parent, start_turn, start_tick, end_turn, end_tick = real_branches[branch]
            if end > end_turn:
                if (branch, end) in turn_end:
                    tick = turn_end[branch, end]
                elif (branch, end) in real_turn_end:
                    tick = real_turn_end[branch, end]
                else:
                    tick = 0
                real_branches[branch] = parent, start_turn, start_tick, end, tick
        for (branch, turn), end in turn_end.items():
            real_turn_end[branch, turn] = max((real_turn_end[branch, turn], end))
        # Make keycaches and valcaches. Must be done chronologically
        # to make forwarding work.
        childbranch = self.db._childbranch
        branch2do = deque(['trunk'])
        while branch2do:
            branch = branch2do.popleft()
            turns = branch_end[branch] + 1
            for turn in range(turns):
                ticks = turn_end[branch, turn] + 1
                for tick in range(ticks):
                    for row in dd3[branch][turn][tick]:
                        fw_upd(*row)
            if branch in childbranch:
                branch2do.extend(childbranch[branch])

    def _forward_valcache(self, cache, branch, turn, tick, copy=True):
        if branch in cache:
            try:
                turnd = cache[branch][turn]
                return turnd.get(tick, None)
            except HistoryError:
                return
        for b, r, t in self.db._iter_parent_btt(branch, turn, tick):
            if b in cache and r in cache[b] and t in cache[b][r]:
                try:
                    turnd = cache[b][r]
                    v = turnd[t]
                    if copy:
                        v = copier(v)
                    cache[branch][turn][0] = v
                except HistoryError as ex:
                    if ex.deleted:
                        cturnd = cache[branch][turn]
                        try:
                            cturnd[0] = None
                        except HistoryError:
                            cturnd[tick] = None
                        return
                    cache[branch][r][t] = v
                    return
        b, r, t, _, _ = self.db._branches[branch] if branch != 'trunk' else 'trunk', 0, 0, None, None
        cache[branch][r][t] = None

    def _forward_keycachelike(self, keycache, keys, slow_iter_keys, parentity, branch, turn, tick):
        # Take valid values from the past of a keycache and copy them forward, into the present.
        keycache_key = parentity + (branch,)
        if keycache_key in keycache:
            kc = keycache[keycache_key]
            try:
                if not kc.has_exact_rev(turn):
                    if tick == 0 and kc.rev_before(turn) == turn - 1:
                        # We had valid keys a turn ago. Reuse those.
                        old_turn_kc = kc[turn]
                        new_turn_kc = FuturistWindowDict()
                        keys = old_turn_kc[old_turn_kc.end]
                        new_turn_kc[0] = keys.copy()
                        kc[turn] = new_turn_kc
                    else:
                        kc[turn][tick] = set(slow_iter_keys(keys[parentity], branch, turn, tick))
                kcturn = kc[turn]
                if not kcturn.has_exact_rev(tick):
                    if kcturn.rev_before(tick) == tick - 1:
                        # We have keys from the previous tick. Use those.
                        kcturn[tick] = kcturn[tick - 1].copy()
                    else:
                        kcturn[tick] = set(slow_iter_keys(keys[parentity], branch, turn, tick))
                return kcturn[tick]
            except HistoryError:
                pass
        # this may throw out some valid cache if there's a gap; that's acceptable
        kc = keycache[keycache_key] = TurnDict()
        kc[turn][tick] = ret = set(slow_iter_keys(keys[parentity], branch, turn, tick))
        return ret

    def _forward_keycache(self, parentity, branch, turn, tick):
        return self._forward_keycachelike(self.keycache, self.keys, self._slow_iter_keys, parentity, branch, turn, tick)

    def _slow_iter_keys(self, cache, branch, turn, tick):
        for key, branches in cache.items():
            for (branc, trn, tck) in self.db._iter_parent_btt(branch, turn, tick):
                if branc not in branches or trn not in branches[branc]:
                    continue
                turnd = branches[branc]
                if turnd.has_exact_rev(trn):
                    if tck in turnd[trn]:
                        try:
                            if turnd[trn][tck] is not None:
                                yield key
                                break
                        except HistoryError as ex:
                            if ex.deleted:
                                break
                    else:
                        trn -= 1
                        if trn not in turnd:
                            break
                tickd = turnd[trn]
                try:
                    if tickd[tickd.end] is not None:
                        yield key
                        break
                except HistoryError as ex:
                    if ex.deleted:
                        break

    def store(self, *args, validate=False, planning=False):
        """Put a value in various dictionaries for later .retrieve(...).

        Needs at least five arguments, of which the -1th is the value
        to store, the -2th is the tick to store it at, the -3th
        is the turn to store it in, the -4th is the branch the
        revision is in, the -5th is the key the value is for,
        and the remaining arguments identify the entity that has
        the key, eg. a graph, node, or edge.

        """
        self._store(*args, planning=planning)
        self._forward_valcaches(*args, validate=validate)
        self._update_keycache(*args, validate=validate)

    def _update_keycache(self, *args, validate=False):
        entity, key, branch, turn, tick, value = args[-6:]
        parent = args[:-6]
        kc = self._forward_keycache(parent + (entity,), branch, turn, tick)
        if value is None:
            kc.discard(key)
        else:
            kc.add(key)
        if validate:
            if parent:
                correct = set(self._slow_iter_keys(self.parents[parent][entity], branch, turn, tick))
                if kc != correct:
                    raise ValueError("Invalid parents cache")
            correct = set(self._slow_iter_keys(self.keys[parent+(entity,)], branch, turn, tick))
            if kc != correct:
                raise ValueError("Invalid keys cache")

    def _store(self, *args, planning=False):
        entity, key, branch, turn, tick, value = args[-6:]
        parent = args[:-6]
        if parent:
            parents = self.parents[parent][entity][key][branch]
            if parents.has_exact_rev(turn):
                parents[turn][tick] = value
            else:
                newp = FuturistWindowDict()
                newp[tick] = value
                parents[turn] = newp
        branches = self.branches[parent+(entity, key)][branch]
        if planning and turn < branches.end:
            raise HistoryError(
                "Can't plan for the past. "
                "Already have some turns after {} in branch {}".format(
                    turn, branch
                )
            )
        if branches.has_exact_rev(turn):
            if turn < branches.end:
                # deal with the paradox by erasing history after this turn
                branches.seek(turn)
                branches._future = deque()

            branchesturn = branches[turn]
            if tick <= branchesturn.end:
                raise HistoryError(
                    "Already have some ticks after {} in turn {} of branch {}".format(
                        tick, turn, branch
                    )
                )
            branchesturn[tick] = value
        else:
            newb = FuturistWindowDict()
            newb[tick] = value
            branches[turn] = newb
        keys = self.keys[parent+(entity,)][key][branch]
        if keys.has_exact_rev(turn):
            keys[turn][tick] = value
        else:
            newt = FuturistWindowDict()
            newt[tick] = value
            keys[turn] = newt
        shallow = self.shallow[parent+(entity, key, branch)]
        if shallow.has_exact_rev(turn):
            shallow[turn][tick] = value
        else:
            news = FuturistWindowDict()
            news[tick] = value
            shallow[turn] = news
        self.shallower[parent+(entity, key, branch, turn)][tick] = value
        try:
            hash(parent+(entity, key, branch, turn, tick))
        except TypeError:
            pass
        self.shallowest[parent+(entity, key, branch, turn, tick)] = value

    def _forward_valcaches(self, *args, validate=False):
        entity, key, branch, turn, tick, value = args[-6:]
        parent = args[:-6]
        if parent and branch not in self.parents[parent][entity][key]:
            self._forward_valcache(
                self.parents[parent][entity][key], branch, turn, tick
            )
        if branch not in self.keys[parent+(entity,)][key]:
            self._forward_valcache(
                self.keys[parent+(entity,)][key], branch, turn, tick
            )
        if branch not in self.branches[parent+(entity, key)]:
            self._forward_valcache(
                self.branches[parent+(entity, key)], branch, turn, tick
            )

    def retrieve(self, *args):
        """Get a value previously .store(...)'d.

        Needs at least five arguments. The -1th is the tick
        within the turn you want,
        the -2th is that turn, the -3th is the branch,
        and the -4th is the key. All other arguments identify
        the entity that the key is in.

        """
        try:
            ret = self.shallowest[args]
            if ret is None:
                raise HistoryError("Set, then deleted", deleted=True)
            return ret
        except KeyError:
            pass
        entity = args[:-4]
        key, branch, turn, tick = args[-4:]
        if entity+(key, branch, turn) in self.shallower and \
                self.shallower[entity+(key, branch, turn)].has_exact_rev(tick):
            ret = self.shallowest[args] \
                = self.shallower[entity+(key, branch, turn)][tick]
            return ret
        if entity+(key, branch) in self.shallow and \
                self.shallow[entity+(key, branch)].has_exact_rev(turn) and \
                tick in self.shallow[entity+(key, branch)][turn]:
            ret = self.shallowest[args] \
                = self.shallower[entity+(key, branch, turn)][tick] \
                = self.shallow[entity + (key, branch)][turn].get(tick)
            return ret
        for (b, r, t) in self.db._iter_parent_btt(branch):
            if (
                    b in self.branches[entity+(key,)]
                    and r in self.branches[entity+(key,)][b]
            ):
                brancs = self.branches[entity+(key,)][b]
                if brancs.has_exact_rev(r) and t in brancs[r]:
                    ret = brancs[r][t]
                else:
                    ret = brancs[r]
                    ret = ret[ret.end]
                if args not in self.shallowest:
                    self.shallowest[args] = ret
                try:
                    self.shallower[entity+(key, branch, turn)][tick] = ret
                except HistoryError:
                    pass
                try:
                    self.shallow[entity+(key, branch)][turn][tick] = ret
                except HistoryError:
                    pass
                try:
                    self.keys[entity][key][branch][turn][tick] = ret
                except HistoryError:
                    pass
                return ret
        else:
            raise KeyError

    def iter_entities_or_keys(self, *args):
        """Iterate over the keys an entity has, if you specify an entity.

        Otherwise iterate over the entities themselves, or at any rate the
        tuple specifying which entity.

        Needs at least two arguments, the branch and the revision. Any
        that come before that will be taken to identify the entity.

        """
        entity = args[:-3]
        branch, turn, tick = args[-3:]
        self._forward_keycache(entity, branch, turn, tick)
        try:
            keys = self.keycache[entity+(branch,)][turn][tick]
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
        entity = args[:-3]
        branch, turn, tick = args[-3:]
        self._forward_keycache(entity, branch, turn, tick)
        try:
            return len(self.keycache[entity+(branch,)][turn][tick])
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
            return self.shallowest[args] is not None
        except KeyError:
            pass
        entity = args[:-4]
        key, branch, turn, tick = args[-4:]
        self._forward_keycache(entity, branch, turn, tick)
        try:
            keys = self.keycache[entity+(branch,)][turn][tick]
        except KeyError:
            return False
        return key in keys
    contains_entity = contains_key = contains_entity_key \
                    = contains_entity_or_key


class NodesCache(Cache):
    def __init__(self, db):
        super().__init__(db)
        self._make_node = db._make_node

    def store(self, graph, node, branch, turn, tick, ex, *, planning=False, validate=False):
        """Store whether a node exists, and create an object for it"""
        if ex and (graph, node) not in self.db._node_objs:
            self.db._node_objs[(graph, node)] \
                = self._make_node(self.db.graph[graph], node)
        Cache.store(self, graph, node, branch, turn, tick, ex, planning=planning, validate=validate)
        if validate:
            kc = self.keycache[graph, node, branch]
            if (
                    node not in kc or
                    not self.contains_entity_or_key(graph, node, branch, turn, tick) or
                    node not in set(self.iter_entities(graph, branch, turn, tick)) or
                    kc != set(self._slow_iter_keys(self.keys[(graph,)], branch, turn, tick))
            ):
                raise ValueError("Invalid keycache")

    def _forward_valcaches(self, graph, node, branch, turn, tick, ex, *, validate=False):
        if not ex:
            ex = None
        super()._forward_valcaches(graph, node, branch, turn, tick, ex, validate=validate)

    def _store(self, graph, node, branch, turn, tick, ex, *, planning=False):
        if not ex:
            ex = None
        return super()._store(graph, node, branch, turn, tick, ex, planning=planning)

    def _update_keycache(self, graph, node, branch, turn, tick, ex, *, validate=False):
        if not ex:
            ex = None
        return super()._update_keycache(graph, node, branch, turn, tick, ex, validate=validate)


class EdgesCache(Cache):
    __slots__ = ['db', 'parents', 'keys', 'keycache', 'branches', 'shallow', 'shallower',
                 'origcache', 'destcache', 'predecessors']

    @property
    def successors(self):
        return self.parents

    def __init__(self, db):
        Cache.__init__(self, db)
        self.destcache = PickyDefaultDict(TurnDict)
        self.origcache = PickyDefaultDict(TurnDict)
        self.predecessors = StructuredDefaultDict(3, TurnDict)

    def _slow_iter_successors(self, cache, branch, turn, tick):
        for dest, dests in cache.items():
            for idx in self._slow_iter_keys(dests, branch, turn, tick):
                yield dest
                break

    def _slow_iter_predecessors(self, cache, branch, turn, tick):
        for orig, origs in cache.items():
            for idx in self._slow_iter_keys(origs, branch, turn, tick):
                yield orig
                break

    def _forward_destcache(self, graph, orig, branch, turn, tick):
        return self._forward_keycachelike(
            self.destcache, self.successors, self._slow_iter_successors, (graph, orig), branch, turn, tick
        )

    def _update_destcache(self, graph, orig, branch, turn, tick, dest, value):
        kc = self._forward_destcache(graph, orig, branch, turn, tick)
        if value is None:
            kc.discard(dest)
        else:
            kc.add(dest)
        return kc

    def _forward_origcache(self, graph, dest, branch, turn, tick):
        return self._forward_keycachelike(
            self.origcache, self.predecessors, self._slow_iter_predecessors, (graph, dest), branch, turn, tick
        )

    def _update_origcache(self, graph, dest, branch, turn, tick, orig, value):
        kc = self._forward_origcache(graph, dest, branch, turn, tick)
        if value is None:
            kc.discard(orig)
        else:
            kc.add(orig)
        return kc

    def iter_successors(self, graph, orig, branch, turn, tick):
        self._forward_destcache(graph, orig, branch, turn, tick)
        try:
            succs = self.destcache[(graph, orig, branch)][turn][tick]
        except KeyError:
            return
        yield from succs

    def iter_predecessors(self, graph, dest, branch, turn, tick):
        self._forward_origcache(graph, dest, branch, turn, tick)
        try:
            preds = self.origcache[(graph, dest, branch)][turn][tick]
        except KeyError:
            return
        yield from preds

    def count_successors(self, graph, orig, branch, turn, tick):
        self._forward_destcache(graph, orig, branch, turn, tick)
        try:
            return len(self.destcache[(graph, orig, branch)][turn][tick])
        except KeyError:
            return 0

    def count_predecessors(self, graph, dest, branch, turn, tick):
        self._forward_origcache(graph, dest, branch, turn, tick)
        try:
            return len(self.origcache[(graph, dest, branch)][turn][tick])
        except KeyError:
            return 0

    def has_successor(self, graph, orig, dest, branch, turn, tick):
        self._forward_keycachelike(
            self.destcache, self.successors, self._slow_iter_successors, (graph, orig), branch, turn, tick
        )
        return dest in self.destcache[(graph, orig, branch)][turn][tick]
    
    def has_predecessor(self, graph, dest, orig, branch, turn, tick):
        self._forward_keycachelike(
            self.origcache, self.predecessors, self._slow_iter_predecessors, (graph, dest), branch, turn, tick
        )
        return orig in self.origcache[(graph, orig, branch)][turn][tick]

    def _store(self, graph, orig, dest, idx, branch, turn, tick, ex, *, planning=False):
        if not ex:
            ex = None
        Cache._store(self, graph, orig, dest, idx, branch, turn, tick, ex, planning=planning)
        if (graph, orig, dest, idx) not in self.db._edge_objs:
            self.db._edge_objs[(graph, orig, dest, idx)] \
                = self.db._make_edge(self.db.graph[graph], orig, dest, idx)
        preds = self.predecessors[(graph, dest)][orig][idx][branch]
        if preds.has_exact_rev(turn):
            preds[turn][tick] = ex
        else:
            newp = FuturistWindowDict()
            newp[tick] = ex
            preds[turn] = newp

    def _forward_valcaches(self, graph, orig, dest, key, branch, turn, tick, ex, *, validate=False):
        super()._forward_valcaches(graph, orig, dest, key, branch, turn, tick, ex, validate=validate)
        oc = self._update_origcache(graph, dest, branch, turn, tick, orig, ex)
        dc = self._update_destcache(graph, orig, branch, turn, tick, dest, ex)
        if validate:
            if oc != set(self._slow_iter_predecessors(self.predecessors[(graph, dest)], branch, turn, tick)):
                raise ValueError("Invalid origcache")
            if dc != set(self._slow_iter_successors(self.successors[(graph, orig)], branch, turn, tick)):
                raise ValueError("Invalid destcache")
