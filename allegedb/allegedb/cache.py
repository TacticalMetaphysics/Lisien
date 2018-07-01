# This file is part of allegedb, a database abstraction for versioned graphs
# Copyright (c) Zachary Spector. public@zacharyspector.com
"""Classes for in-memory storage and retrieval of historical graph data.
"""
from .window import WindowDict, HistoryError


class FuturistWindowDict(WindowDict):
    """A WindowDict that does not let you rewrite the past."""

    def __setitem__(self, rev, v):
        if hasattr(v, 'unwrap') and not hasattr(v, 'no_unwrap'):
            v = v.unwrap()
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


class AbstractTurnDict(WindowDict):
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


class TurnDict(AbstractTurnDict, FuturistWindowDict):
    cls = FuturistWindowDict


class SettingsTurnDict(AbstractTurnDict):
    cls = WindowDict


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
        try:
            ret = self[k] = self.type(
                *self.args_munger(self, k),
                **self.kwargs_munger(self, k)
            )
        except TypeError:
            raise KeyError
        return ret

    def _create(self, v):
        return self.type(v)

    def __setitem__(self, k, v):
        if not isinstance(v, self.type):
            v = self._create(v)
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
        self.shallow = PickyDefaultDict(SettingsTurnDict)
        """Less structured alternative to ``branches`` ."""
        self.shallower = PickyDefaultDict(WindowDict)
        """Even less structured alternative to ``shallow``."""
        self.shallowest = {}
        """A dictionary for plain, unstructured hinting."""
        self.settings = PickyDefaultDict(SettingsTurnDict)
        """All the ``entity[key] = value`` operations that were performed on some turn"""
        self.presettings = PickyDefaultDict(SettingsTurnDict)
        """The values prior to ``entity[key] = value`` operations performed on some turn"""

    def load(self, data, validate=False, cb=None):
        """Add a bunch of data. It doesn't need to be in chronological order.

        With ``validate=True``, raise ValueError if this results in an
        incoherent cache.

        If a callable ``cb`` is provided, it will be called with each row.
        It will also be passed my ``validate`` argument.

        """
        from collections import defaultdict, deque
        dd2 = defaultdict(lambda: defaultdict(list))
        for row in data:
            entity, key, branch, turn, tick, value = row[-6:]
            dd2[branch][turn, tick].append(row)
        # Make keycaches and valcaches. Must be done chronologically
        # to make forwarding work.
        childbranch = self.db._childbranch
        branch2do = deque(['trunk'])
        update_keycache = self._update_keycache

        def store(*args):
            self._store(*args, planning=False)
        while branch2do:
            branch = branch2do.popleft()
            dd2b = dd2[branch]
            for turn, tick in sorted(dd2b.keys()):
                rows = dd2b[turn, tick]
                for row in rows:
                    store(*row)
                    update_keycache(*row, validate=validate, forward=True)
                    if cb:
                        cb(row, validate=validate)
            if branch in childbranch:
                branch2do.extend(childbranch[branch])

    def _valcache_lookup(self, cache, branch, turn, tick):
        if branch in cache:
            branc = cache[branch]
            if turn in branc:
                return branc[turn].get(tick, None)
            try:
                turnd = branc[turn]
                return turnd[turnd.end]
            except HistoryError:
                return
        for b, r, t in self.db._iter_parent_btt(branch, turn, tick):
            if b in cache and r in cache[b] and t in cache[b][r]:
                try:
                    turnd = cache[b][r]
                    return turnd[t]
                except HistoryError as ex:
                    if ex.deleted:
                        return

    @staticmethod
    def _get_keycachelike(keycache, keys, slow_iter_keys, parentity, branch, turn, tick, *, forward):
        keycache_key = parentity + (branch,)
        if keycache_key in keycache and turn in keycache[keycache_key] and tick in keycache[keycache_key][turn]:
            return keycache[keycache_key][turn][tick]
        if forward and keycache_key in keycache:
            # Take valid values from the past of a keycache and copy them forward, into the present.
            # Assumes that time is only moving forward, never backward, never skipping any turns or ticks,
            # and any changes to the world state are happening through allegedb proper, meaning they'll all get cached.
            # In LiSE this means every change to the world state should happen inside of a call to
            # ``Engine.next_turn`` in a rule.
            kc = keycache[keycache_key]
            try:
                if turn not in kc:
                    if kc.rev_gettable(turn):
                        old_turn_kc = kc[turn]
                        new_turn_kc = FuturistWindowDict()
                        new_turn_kc[0] = old_turn_kc[old_turn_kc.end]
                        kc[turn] = new_turn_kc
                    else:
                        kc[turn][tick] = frozenset(slow_iter_keys(keys[parentity], branch, turn, tick))
                kcturn = kc[turn]
                if tick not in kcturn:
                    if kcturn.rev_gettable(tick):
                        kcturn[tick] = kcturn[tick]
                    else:
                        kcturn[tick] = frozenset(slow_iter_keys(keys[parentity], branch, turn, tick))
                return kcturn[tick]
            except HistoryError:
                pass
        kc = keycache[keycache_key] = TurnDict()
        kc[turn][tick] = ret = frozenset(slow_iter_keys(keys[parentity], branch, turn, tick))
        return ret

    def _get_keycache(self, parentity, branch, turn, tick, *, forward):
        return self._get_keycachelike(
            self.keycache, self.keys, self._slow_iter_keys,
            parentity, branch, turn, tick, forward=forward
        )

    def _slow_iter_keys(self, cache, branch, turn, tick):
        for key, branches in cache.items():
            for (branc, trn, tck) in self.db._iter_parent_btt(branch, turn, tick):
                if branc not in branches or not branches[branc].rev_gettable(trn):
                    continue
                turnd = branches[branc]
                if trn in turnd:
                    if turnd[trn].rev_gettable(tck):
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

    def store(self, *args, validate=False, planning=None, forward=None):
        """Put a value in various dictionaries for later .retrieve(...).

        Needs at least five arguments, of which the -1th is the value
        to store, the -2th is the tick to store it at, the -3th
        is the turn to store it in, the -4th is the branch the
        revision is in, the -5th is the key the value is for,
        and the remaining arguments identify the entity that has
        the key, eg. a graph, node, or edge.

        With ``validate=True``, check the cache after storing,
        and raise ``ValueError`` if it's not coherent.

        With ``planning=True``, raise HistoryError instead of overwriting.
        Otherwise, any information about this key of this entity in the
        future will be deleted. Separate branches of history will be
        unaffected.

        With ``forward=True``, enable an optimization that assumes time
        will never go backward.

        """
        if planning is None:
            planning = self.db._planning
        if forward is None:
            forward = self.db._forward
        self._store(*args, planning=planning)
        self._update_keycache(*args, validate=validate, forward=forward)

    def _update_keycache(self, *args, validate, forward):
        entity, key, branch, turn, tick, value = args[-6:]
        parent = args[:-6]
        kc = self._get_keycache(parent + (entity,), branch, turn, tick, forward=forward)
        if value is None:
            kc = kc.difference({key})
        else:
            kc = kc.union({key})
        self.keycache[parent+(entity, branch)][turn][tick] = kc
        if validate:
            if parent:
                correct = set(self._slow_iter_keys(self.parents[parent][entity], branch, turn, tick))
                if kc != correct:
                    raise ValueError("Invalid parents cache")
            correct = set(self._slow_iter_keys(self.keys[parent+(entity,)], branch, turn, tick))
            if kc != correct:
                raise ValueError("Invalid keys cache")

    def _store(self, *args, planning):
        entity, key, branch, turn, tick, value = args[-6:]
        parent = args[:-6]
        settings_turns = self.settings[branch]
        branches = self.branches[parent+(entity, key)][branch]
        keys = self.keys[parent+(entity,)][key][branch]
        shallow = self.shallow[parent+(entity, key, branch)]
        if planning:
            if shallow:
                if turn in shallow and tick < shallow[turn].end:
                    raise HistoryError(
                        "Already have some ticks after {} in turn {} of branch {}".format(
                            tick, turn, branch
                        )
                    )
            if keys:
                if turn in keys and tick <= keys[turn].end:
                    raise HistoryError(
                        "Already have some ticks after {} in turn {} of branch {}".format(
                            tick, turn, branch
                        )
                    )
            if branches:
                if turn in branches and tick <= branches[turn].end:
                    raise HistoryError(
                        "Already have some ticks after {} in turn {} of branch {}".format(
                            tick, turn, branch
                        )
                    )
        else:
            # truncate settings
            prefix = parent + (entity, key)
            for settings in self.settings, self.presettings:
                settings_turns = settings[branch]
                settings_turns.seek(turn)
                deletable = {}
                for trn, tics in settings_turns.future().items():
                    deletable[trn] = [
                        tic for tic in tics
                        if tic >= tick and tics[tic][:len(prefix)] == prefix
                    ]
                for trn, tics in deletable.items():
                    for tic in tics:
                        del settings_turns[trn][tic]
        self._store_journal(*args)
        new = None
        if parent:
            parents = self.parents[parent][entity][key][branch]
            if turn in parents:
                parentsturn = parents[turn]
                parentsturn.truncate(tick)
                parentsturn[tick] = value
            else:
                new = FuturistWindowDict()
                new[tick] = value
                parents[turn] = new
        if branches and turn < branches.end:
            # deal with the paradox by erasing history after this tick and turn
            if turn in branches:
                mapp_turn = branches[turn]
                settings_turn = settings_turns[turn]
                if tick in mapp_turn:
                    del settings_turn[tick]
                for tic in mapp_turn.future():
                    del settings_turn[tic]
            branches.truncate(turn)
            keys.truncate(turn)
            shallow.truncate(turn)
        if turn in branches:
            assert turn in keys
            assert turn in shallow
            branchesturn = branches[turn]
            assert branchesturn is keys[turn]
            branchesturn.truncate(tick)
            branchesturn[tick] = value
            shallowturn = shallow[turn]
            shallowturn.truncate(tick)
            shallowturn[tick] = value
        else:
            if new is None:
                new = FuturistWindowDict()
                new[tick] = value
            branches[turn] = keys[turn] = new
            shallow[turn] = {tick: value}
        self.shallower[parent+(entity, key, branch, turn)][tick] = value
        self.shallowest[parent+(entity, key, branch, turn, tick)] = value

    def _store_journal(self, *args):
        # overridden in LiSE.cache.InitializedCache
        entity, key, branch, turn, tick, value = args[-6:]
        parent = args[:-6]
        settings_turns = self.settings[branch]
        presettings_turns = self.presettings[branch]
        try:
            prev = self.retrieve(*args[:-1])
        except KeyError:
            prev = None
        if turn in settings_turns or turn in settings_turns.future():
            assert turn in presettings_turns or turn in presettings_turns.future()
            setticks = settings_turns[turn]
            assert tick not in setticks
            presetticks = presettings_turns[turn]
            assert tick not in presetticks
            presetticks[tick] = parent + (entity, key, prev)
            setticks[tick] = parent + (entity, key, value)
        else:
            presettings_turns[turn] = {tick: parent + (entity, key, prev)}
            settings_turns[turn] = {tick: parent + (entity, key, value)}

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
                tick in self.shallower[entity+(key, branch, turn)]:
            ret = self.shallowest[args] \
                = self.shallower[entity+(key, branch, turn)][tick]
            return ret
        if entity+(key, branch) in self.shallow and \
                turn in self.shallow[entity+(key, branch)] and \
                self.shallow[entity+(key, branch)][turn].rev_gettable(tick):
            ret = self.shallowest[args] \
                = self.shallower[entity+(key, branch, turn)][tick] \
                = self.shallow[entity + (key, branch)][turn].get(tick)
            return ret
        for (b, r, t) in self.db._iter_parent_btt(branch):
            if (
                    b in self.branches[entity+(key,)]
                    and self.branches[entity+(key,)][b].rev_gettable(r)
            ):
                brancs = self.branches[entity+(key,)][b]
                if r in brancs and t in brancs[r]:
                    ret = brancs[r][t]
                else:
                    ret = brancs[r]
                    ret = ret[ret.end]
                    self.shallow[entity+(key, branch)][turn][tick] = ret
                    self.shallower[entity+(key, branch, turn)][tick] = ret
                    self.shallowest[args] = ret
                return ret
        else:
            raise KeyError

    def iter_entities_or_keys(self, *args, forward=None):
        """Iterate over the keys an entity has, if you specify an entity.

        Otherwise iterate over the entities themselves, or at any rate the
        tuple specifying which entity.

        Needs at least two arguments, the branch and the revision. Any
        that come before that will be taken to identify the entity.

        """
        if forward is None:
            forward = self.db._forward
        entity = args[:-3]
        branch, turn, tick = args[-3:]
        yield from self._get_keycache(entity, branch, turn, tick, forward=forward)
    iter_entities = iter_keys = iter_entity_keys = iter_entities_or_keys

    def count_entities_or_keys(self, *args, forward=None):
        """Return the number of keys an entity has, if you specify an entity.

        Otherwise return the number of entities.

        Needs at least two arguments, the branch and the revision. Any
        that come before that will be taken to identify the entity.

        """
        if forward is None:
            forward = self.db._forward
        entity = args[:-3]
        branch, turn, tick = args[-3:]
        return len(self._get_keycache(entity, branch, turn, tick, forward=forward))
    count_entities = count_keys = count_entity_keys = count_entities_or_keys

    def contains_entity_or_key(self, *args, forward=None):
        """Check if an entity has a key at the given time, if entity specified.

        Otherwise check if the entity exists.

        Needs at least three arguments, the key, the branch, and the revision.
        Any that come before that will be taken to identify the entity.

        """
        try:
            return self.shallowest[args] is not None
        except KeyError:
            pass
        if forward is None:
            forward = self.db._forward
        entity = args[:-4]
        key, branch, turn, tick = args[-4:]
        return key in self._get_keycache(entity, branch, turn, tick, forward=forward)
    contains_entity = contains_key = contains_entity_key \
                    = contains_entity_or_key


class NodesCache(Cache):
    """A cache for remembering whether nodes exist at a given time."""
    def __init__(self, db):
        super().__init__(db)
        self._make_node = db._make_node

    def store(self, graph, node, branch, turn, tick, ex, *, planning=None, forward=None, validate=False):
        """Store whether a node exists, and create an object for it"""
        if ex and (graph, node) not in self.db._node_objs:
            self.db._node_objs[(graph, node)] \
                = self._make_node(self.db.graph[graph], node)
        Cache.store(self, graph, node, branch, turn, tick, ex, planning=planning, forward=forward, validate=validate)
        if validate:
            kc = self._get_keycache((graph,), branch, turn, tick, forward=forward)
            correct_kc = set(self._slow_iter_keys(self.keys[(graph,)], branch, turn, tick))
            if (
                    node not in kc or
                    not self.contains_entity_or_key(graph, node, branch, turn, tick) or
                    node not in set(self.iter_entities(graph, branch, turn, tick, forward=forward)) or
                    kc != correct_kc
            ):
                raise ValueError("Invalid keycache")

    def _store(self, graph, node, branch, turn, tick, ex, *, planning):
        if not ex:
            ex = None
        return super()._store(graph, node, branch, turn, tick, ex, planning=planning)

    def _update_keycache(self, graph, node, branch, turn, tick, ex, *, validate, forward):
        if not ex:
            ex = None
        return super()._update_keycache(graph, node, branch, turn, tick, ex, validate=validate, forward=forward)


class EdgesCache(Cache):
    """A cache for remembering whether edges exist at a given time."""
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

    def _get_destcache(self, graph, orig, branch, turn, tick, *, forward):
        return self._get_keycachelike(
            self.destcache, self.successors, self._slow_iter_successors, (graph, orig),
            branch, turn, tick, forward=forward
        )

    def _update_destcache(self, graph, orig, branch, turn, tick, dest, value, *, forward):
        kc = self._get_destcache(graph, orig, branch, turn, tick, forward=forward)
        if value is None:
            kc = kc.difference({dest})
        else:
            kc = kc.union({dest})
        self.destcache[graph, orig, branch][turn][tick] = kc
        return kc

    def _get_origcache(self, graph, dest, branch, turn, tick, *, forward):
        return self._get_keycachelike(
            self.origcache, self.predecessors, self._slow_iter_predecessors, (graph, dest),
            branch, turn, tick, forward=forward
        )

    def _update_origcache(self, graph, dest, branch, turn, tick, orig, value, *, forward):
        kc = self._get_origcache(graph, dest, branch, turn, tick, forward=forward)
        if value is None:
            kc = kc.difference({orig})
        else:
            kc = kc.union({orig})
        self.origcache[graph, dest, branch][turn][tick] = kc
        return kc

    def _update_keycache(self, *args, validate, forward):
        super()._update_keycache(*args, validate=validate, forward=forward)
        graph, orig, dest, idx, branch, turn, tick, value = args
        self._update_origcache(graph, dest, branch, turn, tick, orig, value, forward=forward)
        self._update_destcache(graph, orig, branch, turn, tick, dest, value, forward=forward)

    def iter_successors(self, graph, orig, branch, turn, tick, *, forward=None):
        """Iterate over successors of a given origin node at a given time."""
        if forward is None:
            forward = self.db._forward
        yield from self._get_destcache(graph, orig, branch, turn, tick, forward=forward)

    def iter_predecessors(self, graph, dest, branch, turn, tick, *, forward=None):
        """Iterate over predecessors to a given destination node at a given time."""
        if forward is None:
            forward = self.db._forward
        yield from self._get_origcache(graph, dest, branch, turn, tick, forward=forward)

    def count_successors(self, graph, orig, branch, turn, tick, *, forward=None):
        """Return the number of successors to a given origin node at a given time."""
        if forward is None:
            forward = self.db._forward
        return len(self._get_destcache(graph, orig, branch, turn, tick, forward=forward))

    def count_predecessors(self, graph, dest, branch, turn, tick, *, forward=None):
        """Return the number of predecessors from a given destination node at a given time."""
        if forward is None:
            forward = self.db._forward
        return len(self._get_origcache(graph, dest, branch, turn, tick, forward=forward))

    def has_successor(self, graph, orig, dest, branch, turn, tick, *, forward=None):
        """Return whether an edge connects the origin to the destination at the given time."""
        if forward is None:
            forward = self.db._forward
        return dest in self._get_keycachelike(
            self.destcache, self.successors, self._slow_iter_successors, (graph, orig),
            branch, turn, tick, forward=forward
        )
    
    def has_predecessor(self, graph, dest, orig, branch, turn, tick, forward=None):
        """Return whether an edge connects the destination to the origin at the given time."""
        if forward is None:
            forward = self.db._forward
        return orig in self._get_keycachelike(
            self.origcache, self.predecessors, self._slow_iter_predecessors, (graph, dest),
            branch, turn, tick, forward=forward
        )

    def _store(self, graph, orig, dest, idx, branch, turn, tick, ex, *, planning=None):
        if not ex:
            ex = None
        if planning is None:
            planning = self.db.planning
        Cache._store(self, graph, orig, dest, idx, branch, turn, tick, ex, planning=planning)
        if (graph, orig, dest, idx) not in self.db._edge_objs:
            self.db._edge_objs[(graph, orig, dest, idx)] \
                = self.db._make_edge(self.db.graph[graph], orig, dest, idx)
        preds = self.predecessors[(graph, dest)][orig][idx][branch]
        if turn in preds:
            preds[turn][tick] = ex
        else:
            newp = FuturistWindowDict()
            newp[tick] = ex
            preds[turn] = newp

    def _forward_valcaches(self, graph, orig, dest, key, branch, turn, tick, ex, *, validate=False):
        if not ex:
            ex = None
        oc = self._update_origcache(graph, dest, branch, turn, tick, orig, ex)
        dc = self._update_destcache(graph, orig, branch, turn, tick, dest, ex)
        if validate:
            if oc != set(self._slow_iter_predecessors(self.predecessors[(graph, dest)], branch, turn, tick)):
                raise ValueError("Invalid origcache")
            if dc != set(self._slow_iter_successors(self.successors[(graph, orig)], branch, turn, tick)):
                raise ValueError("Invalid destcache")
