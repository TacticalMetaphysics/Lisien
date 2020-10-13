# This file is part of allegedb, a database abstraction for versioned graphs
# Copyright (c) Zachary Spector. public@zacharyspector.com
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
"""Classes for in-memory storage and retrieval of historical graph data.
"""
from .window import WindowDict, HistoryError, FuturistWindowDict, TurnDict, SettingsTurnDict
from collections import OrderedDict, defaultdict, deque
from time import monotonic


def _default_args_munger(self, k):
    """By default, :class:`PickyDefaultDict`'s ``type`` is instantiated with no positional arguments."""
    return tuple()


def _default_kwargs_munger(self, k):
    """By default, :class:`PickyDefaultDict`'s ``type`` is instantiated with no keyword arguments."""
    return {}


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
            args_munger=_default_args_munger,
            kwargs_munger=_default_kwargs_munger
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
        if type(v) is not self.type:
            v = self._create(v)
        super(PickyDefaultDict, self).__setitem__(k, v)


class StructuredDefaultDict(dict):
    """A ``defaultdict``-like class that expects values stored at a specific depth.

    Requires an integer to tell it how many layers deep to go.
    The innermost layer will be ``PickyDefaultDict``, which will take the
    ``type``, ``args_munger``, and ``kwargs_munger`` arguments supplied
    to my constructor.

    """
    __slots__ = ('layer', 'type', 'args_munger', 'kwargs_munger', 'parent', 'key', '_stuff', 'gettest', 'settest')

    def __init__(
            self, layers, type=object,
            args_munger=_default_args_munger,
            kwargs_munger=_default_kwargs_munger,
            gettest=lambda k: None,
            settest=lambda k, v: None
    ):
        if layers < 1:
            raise ValueError("Not enough layers")
        self.layer = layers
        self.type = type
        self.args_munger = args_munger
        self.kwargs_munger = kwargs_munger
        self._stuff = (layers, type, args_munger, kwargs_munger)
        self.gettest = gettest
        self.settest = settest

    def __getitem__(self, k):
        self.gettest(k)
        if k in self:
            return dict.__getitem__(self, k)
        layer, typ, args_munger, kwargs_munger = self._stuff
        if layer == 1:
            ret = PickyDefaultDict(
                typ, args_munger, kwargs_munger
            )
        elif layer < 1:
            raise ValueError("Invalid layer")
        else:
            ret = StructuredDefaultDict(
                layer-1, typ,
                args_munger, kwargs_munger
            )
        ret.parent = self
        ret.key = k
        dict.__setitem__(self, k, ret)
        return ret

    def __setitem__(self, k, v):
        self.settest(k, v)
        if type(v) is StructuredDefaultDict:
            layer, typ, args_munger, kwargs_munger = self._stuff
            if (
                    v.layer == layer - 1
                    and v.type is typ
                    and v.args_munger is args_munger
                    and v.kwargs_munger is kwargs_munger
            ):
                super().__setitem__(k, v)
                return
        elif type(v) is PickyDefaultDict:
            layer, typ, args_munger, kwargs_munger = self._stuff
            if (
                layer == 1
                and v.type is typ
                and v.args_munger is args_munger
                and v.kwargs_munger is kwargs_munger
            ):
                super().__setitem__(k, v)
                return
        raise TypeError("Can't set layer {}".format(self.layer))


KEYCACHE_MAXSIZE = 1024


def lru_append(kc, lru, kckey, maxsize):
    """Delete old data from ``kc``, then add new ``kckey`` to ``lru``

    :param kc: a three-layer keycache
    :param lru: an :class:`OrderedDict` with a key for each triple that should fill out ``kc``'s three layers
    :param kckey: a triple that indexes into ``kc``, which will be added to ``lru`` if needed
    :param maxsize: maximum number of entries in ``lru`` and, therefore, ``kc``

    """
    if kckey in lru:
        return
    while len(lru) >= maxsize:
        (peb, turn, tick), _ = lru.popitem(False)
        if peb not in kc:
            continue
        kcpeb = kc[peb]
        if turn not in kcpeb:
            continue
        kcpebturn = kcpeb[turn]
        if tick not in kcpebturn:
            continue
        del kcpebturn[tick]
        if not kcpebturn:
            del kcpeb[turn]
        if not kcpeb:
            del kc[peb]
    lru[kckey] = True


class Cache:
    """A data store that's useful for tracking graph revisions."""
    __slots__ = (
        'db', 'parents', 'keys', 'keycache', 'branches', 'shallowest',
        'settings', 'presettings', 'time_entity', '_kc_lru',
        '_store_stuff', '_remove_stuff', '_truncate_stuff',
        'setdb', 'deldb', 'keyframe', 'name'
    )

    def __init__(self, db, kfkvs=None):
        super().__init__()
        self.db = db
        self.parents = StructuredDefaultDict(3, SettingsTurnDict)
        """Entity data keyed by the entities' parents.

        An entity's parent is what it's contained in. When speaking of a node,
        this is its graph. When speaking of an edge, the parent is usually the
        graph and the origin in a pair, though for multigraphs the destination
        might be part of the parent as well.

        Deeper layers of this cache are keyed by branch and revision.

        """
        self.keys = StructuredDefaultDict(2, SettingsTurnDict)
        """Cache of entity data keyed by the entities themselves.

        That means the whole tuple identifying the entity is the
        top-level key in this cache here. The second-to-top level
        is the key within the entity.

        Deeper layers of this cache are keyed by branch, turn, and tick.

        """
        self.keycache = PickyDefaultDict(SettingsTurnDict)
        """Keys an entity has at a given turn and tick."""
        self.branches = StructuredDefaultDict(1, SettingsTurnDict)
        """A less structured alternative to ``keys``.

        For when you already know the entity and the key within it,
        but still need to iterate through history to find the value.

        """
        self.keyframe = StructuredDefaultDict(1, SettingsTurnDict, **(kfkvs or {}))
        """Key-value dictionaries representing my state at a given time"""
        self.shallowest = OrderedDict()
        """A dictionary for plain, unstructured hinting."""
        self.settings = PickyDefaultDict(SettingsTurnDict)
        """All the ``entity[key] = value`` operations that were performed on some turn"""
        self.presettings = PickyDefaultDict(SettingsTurnDict)
        """The values prior to ``entity[key] = value`` operations performed on some turn"""
        self.time_entity = {}
        self._kc_lru = OrderedDict()
        self._store_stuff = (
            self.parents, self.branches, self.keys, db.delete_plan,
            db._time_plan, self._iter_future_contradictions,
            db._branches, db._turn_end, self._store_journal,
            self.time_entity, db._where_cached, self.keycache,
            db, self._update_keycache
        )
        self._remove_stuff = (
            self.time_entity, self.parents, self.branches, self.keys,
            self.settings, self.presettings, self._remove_keycache
        )
        self._truncate_stuff = (
            self.parents, self.branches, self.keys, self.settings, self.presettings,
            self.keycache
        )

    def load(self, data):
        """Add a bunch of data. Must be in chronological order.

        But it doesn't need to all be from the same branch, as long as
        each branch is chronological of itself.

        """
        branches = defaultdict(list)
        for row in data:
            branches[row[-4]].append(row)
        db = self.db
        # Make keycaches and valcaches. Must be done chronologically
        # to make forwarding work.
        childbranch = db._childbranch
        branch2do = deque(['trunk'])

        store = self.store
        with db.batch():
            while branch2do:
                branch = branch2do.popleft()
                for row in branches[branch]:
                    store(*row, planning=False, loading=True)
                if branch in childbranch:
                    branch2do.extend(childbranch[branch])

    def _valcache_lookup(self, cache, branch, turn, tick):
        """Return the value at the given time in ``cache``"""
        if branch in cache:
            branc = cache[branch]
            try:
                if turn in branc and branc[turn].rev_gettable(tick):
                    return branc[turn][tick]
                elif branc.rev_gettable(turn-1):
                    turnd = branc[turn-1]
                    return turnd.final()
            except HistoryError as ex:
                # probably shouldn't ever happen, empty branches shouldn't be kept in the cache at all...
                # but it's easy to handle
                if ex.deleted:
                    raise
        for b, r, t in self.db._iter_parent_btt(branch, turn, tick):
            if b in cache:
                if r in cache[b] and cache[b][r].rev_gettable(t):
                    try:
                        return cache[b][r][t]
                    except HistoryError as ex:
                        if ex.deleted:
                            raise
                elif cache[b].rev_gettable(r-1):
                    cbr = cache[b][r-1]
                    try:
                        return cbr.final()
                    except HistoryError as ex:
                        if ex.deleted:
                            raise

    def _get_keycachelike(self, keycache, keys, get_adds_dels, parentity, branch, turn, tick, *, forward):
        """Try to retrieve a frozenset representing extant keys.

        If I can't, generate one, store it, and return it.

        """
        keycache_key = parentity + (branch,)
        keycache2 = keycache3 = None
        if keycache_key in keycache:
            keycache2 = keycache[keycache_key]
            if turn in keycache2:
                keycache3 = keycache2[turn]
                if tick in keycache3:
                    return keycache3[tick]
        if forward:
            # Take valid values from the past of a keycache and copy them forward, into the present.
            # Assumes that time is only moving forward, never backward, never skipping any turns or ticks,
            # and any changes to the world state are happening through allegedb proper, meaning they'll all get cached.
            # In LiSE this means every change to the world state should happen inside of a call to
            # ``Engine.next_turn`` in a rule.
            if keycache2 and keycache2.rev_gettable(turn):
                # there's a keycache from a prior turn in this branch. Get it
                if turn not in keycache2:
                    # since it's not this *exact* turn there might be changes...
                    old_turn = keycache2.rev_before(turn)
                    old_turn_kc = keycache2[turn]
                    added, deleted = get_adds_dels(
                        parentity, branch, turn, tick, stoptime=(
                            branch, old_turn, old_turn_kc.end
                        ), cache=keys
                    )
                    ret = old_turn_kc.final().union(added).difference(deleted)
                    # assert ret == get_adds_dels(keys[parentity], branch, turn, tick)[0]  # slow
                    new_turn_kc = WindowDict()
                    new_turn_kc[tick] = ret
                    keycache2[turn] = new_turn_kc
                    return ret
                if not keycache3:
                    keycache3 = keycache2[turn]
                if tick not in keycache3:
                    if keycache3.rev_gettable(tick):
                        added, deleted = get_adds_dels(
                            parentity, branch, turn, tick, stoptime=(
                                branch, turn, keycache3.rev_before(tick)
                            ), cache=keys
                        )
                        ret = keycache3[tick].union(added).difference(deleted)
                        # assert ret == get_adds_dels(keys[parentity], branch, turn, tick)[0]  # slow
                        keycache3[tick] = ret
                        return ret
                    else:
                        turn_before = keycache2.rev_before(turn)
                        tick_before = keycache2[turn_before].end
                        keys_before = keycache2[turn_before][tick_before]
                        added, deleted = get_adds_dels(
                            parentity, branch, turn, tick, stoptime=(
                                branch, turn_before, tick_before
                            ), cache=keys
                        )
                        ret = keycache3[tick] = keys_before.union(added).difference(deleted)
                        # assert ret == get_adds_dels(keys[parentity], branch, turn, tick)[0]  # slow
                        return ret
                # assert kcturn[tick] == get_adds_dels(keys[parentity], branch, turn, tick)[0]  # slow
                return keycache3[tick]
        ret = frozenset(get_adds_dels(parentity, branch, turn, tick)[0])
        if keycache2:
            if keycache3:
                keycache3[tick] = ret
            else:
                keycache2[turn] = {tick: ret}
        else:
            kcc = SettingsTurnDict()
            kcc[turn] = {tick: ret}
            keycache[keycache_key] = kcc
        return ret

    def _get_keycache(self, parentity, branch, turn, tick, *, forward):
        """Get a frozenset of keys that exist in the entity at the moment.

        With ``forward=True``, enable an optimization that copies old key sets
        forward and updates them.

        """
        lru_append(self.keycache, self._kc_lru, (parentity+(branch,), turn, tick), KEYCACHE_MAXSIZE)
        return self._get_keycachelike(
            self.keycache, self.keys, self._get_adds_dels,
            parentity, branch, turn, tick, forward=forward
        )

    def _update_keycache(self, *args, forward):
        """Add or remove a key in the set describing the keys that exist."""
        entity, key, branch, turn, tick, value = args[-6:]
        parent = args[:-6]
        kc = self._get_keycache(parent + (entity,), branch, turn, tick, forward=forward)
        if value is None:
            kc = kc.difference((key,))
        else:
            kc = kc.union((key,))
        self.keycache[parent+(entity, branch)][turn][tick] = kc

    def _get_adds_dels(self, entity, branch, turn, tick, *, stoptime=None, cache=None):
        """Return a pair of sets describing changes to the entity's keys

        Returns a pair of sets: ``(added, deleted)``. These are the changes
        to the key set that occurred since ``stoptime``, which, if present,
        should be a triple ``(branch, turn, tick)``.

        With ``stoptime=None`` (the default), ``added`` will in fact be all
        keys, and ``deleted`` will be empty.

        """
        # Not using the journal because that doesn't distinguish entities.
        # I think I might not want to use ``stoptime`` at all, now that
        # there is such a thing as keyframes...
        cache = cache or self.keys[entity]
        added = set()
        deleted = set()
        kf = self.keyframe.get(entity, None)
        for key, branches in cache.items():
            for (branc, trn, tck) in self.db._iter_parent_btt(branch, turn, tick, stoptime=stoptime):
                if branc not in branches \
                        or not branches[branc].rev_gettable(trn):
                    continue
                turnd = branches[branc]
                if trn in turnd:
                    if turnd[trn].rev_gettable(tck):
                        if turnd[trn][tck] is None:
                            deleted.add(key)
                        else:
                            added.add(key)
                        break
                    else:
                        trn -= 1
                if not turnd.rev_gettable(trn):
                    break
                tickd = turnd[trn]
                if tickd.final() is None:
                    deleted.add(key)
                else:
                    added.add(key)
                break
        else:
            if stoptime or not kf:
                return added, deleted
            for (branc, trn, tck) in self.db._iter_parent_btt(branch, turn,
                                                              tick):
                if branc not in kf or not kf[branc].rev_gettable(trn):
                    continue
                kfb = kf[branc]
                if trn in kfb:
                    kfbr = kfb[trn]
                    if kfbr.rev_gettable(tck):
                        added.update(set(kfbr[tck]).difference(deleted))
                elif kfb.rev_gettable(trn):
                    added.update(set(kfb[trn].final()).difference(deleted))
        return added, deleted

    def store(self, *args, planning=None, forward=None, loading=False, contra=True):
        """Put a value in various dictionaries for later .retrieve(...).

        Needs at least five arguments, of which the -1th is the value
        to store, the -2th is the tick to store it at, the -3th
        is the turn to store it in, the -4th is the branch the
        revision is in, the -5th is the key the value is for,
        and the remaining arguments identify the entity that has
        the key, eg. a graph, node, or edge.

        With ``planning=True``, you will be permitted to alter
        "history" that takes place after the last non-planning
        moment of time, without much regard to consistency.
        Otherwise, contradictions will be handled by deleting
        everything in the contradicted plan after the present moment,
        unless you set ``contra=False``.

        ``loading=True`` prevents me from updating the ORM's records
        of the ends of branches and turns.

        """
        (
            self_parents, self_branches, self_keys, delete_plan,
            time_plan, self_iter_future_contradictions,
            db_branches, db_turn_end, self_store_journal,
            self_time_entity, db_where_cached, keycache, db,
            update_keycache
        ) = self._store_stuff
        if planning is None:
            planning = db._planning
        if forward is None:
            forward = db._forward
        entity, key, branch, turn, tick, value = args[-6:]
        parent = args[:-6]
        entikey = (entity, key)
        parentikey = parent + (entity, key)
        if parent:
            parentity = self_parents[parent][entity]
            if key in parentity:
                branches = parentity[key]
                turns = branches[branch]
            else:
                branches = self_branches[parentikey] \
                    = self_keys[parent + (entity,)][key] \
                    = parentity[key]
                turns = branches[branch]
        else:
            if entikey in self_branches:
                branches = self_branches[entikey]
                turns = branches[branch]
            else:
                branches = self_branches[entikey]
                self_keys[entity,][key] = branches
                turns = branches[branch]
        if planning:
            if turn in turns and tick < turns[turn].end:
                raise HistoryError(
                    "Already have some ticks after {} in turn {} of branch {}".format(
                        tick, turn, branch
                    )
                )
        if contra:
            contras = list(
                self_iter_future_contradictions(entity, key, turns, branch,
                                                turn, tick, value))
            if contras:
                self.shallowest = OrderedDict()
            for contra_turn, contra_tick in contras:
                if (branch, contra_turn,
                    contra_tick) in time_plan:  # could've been deleted in this very loop
                    delete_plan(time_plan[branch, contra_turn, contra_tick])
            if not turns:  # turns may be mutated in delete_plan
                branches[branch] = turns
            if parentikey not in self_branches:
                self_branches[parentikey] = branches
        if not loading and not planning:
            parbranch, turn_start, tick_start, turn_end, tick_end = \
            db_branches[branch]
            db_branches[branch] = parbranch, turn_start, tick_start, turn, tick
            db_turn_end[branch, turn] = tick
        self_store_journal(*args)
        self.shallowest[parent + (entity, key, branch, turn, tick)] = value
        if turn in turns:
            the_turn = turns[turn]
            the_turn.truncate(tick)
            the_turn[tick] = value
        else:
            new = FuturistWindowDict()
            new[tick] = value
            turns[turn] = new
        self_time_entity[branch, turn, tick] = parent, entity, key
        where_cached = db_where_cached[args[-4:-1]]
        if self not in where_cached:
            where_cached.append(self)
        # if we're editing the past, have to invalidate the keycache
        keycache_key = parent + (entity, branch)
        if keycache_key in keycache:
            thiskeycache = keycache[keycache_key]
            if turn in thiskeycache:
                del thiskeycache[turn]
            thiskeycache.truncate(turn)
            if not thiskeycache:
                del keycache[keycache_key]
        if not db._no_kc:
            update_keycache(*args, forward=forward)

    def remove(self, branch, turn, tick):
        """Delete all data from a specific tick"""
        time_entity, parents, branches, keys, settings, presettings, remove_keycache = self._remove_stuff
        parent, entity, key = time_entity[branch, turn, tick]
        branchkey = parent + (entity, key)
        keykey = parent + (entity,)
        if parent in parents:
            parentt = parents[parent]
            if entity in parentt:
                entty = parentt[entity]
                if key in entty:
                    kee = entty[key]
                    if branch in kee:
                        branhc = kee[branch]
                        if turn in branhc:
                            trn = branhc[turn]
                            del trn[tick]
                            if not trn:
                                del branhc[turn]
                            if not branhc:
                                del kee[branch]
                    if not kee:
                        del entty[key]
                if not entty:
                    del parentt[entity]
            if not parentt:
                del parents[parent]
        if branchkey in branches:
            entty = branches[branchkey]
            if branch in entty:
                branhc = entty[branch]
                if turn in branhc:
                    trn = branhc[turn]
                    if tick in trn:
                        del trn[tick]
                    if not trn:
                        del branhc[turn]
                if not branhc:
                    del entty[branch]
            if not entty:
                del branches[branchkey]
        if keykey in keys:
            entty = keys[keykey]
            if key in entty:
                kee = entty[key]
                if branch in kee:
                    branhc = kee[branch]
                    if turn in branhc:
                        trn = entty[turn]
                        if tick in trn:
                            del trn[tick]
                        if not trn:
                            del branhc[turn]
                    if not branhc:
                        del kee[branch]
                if not kee:
                    del entty[key]
            if not entty:
                del keys[keykey]
        branhc = settings[branch]
        pbranhc = presettings[branch]
        trn = branhc[turn]
        ptrn = pbranhc[turn]
        if tick in trn:
            del trn[tick]
        if tick in ptrn:
            del ptrn[tick]
        if not ptrn:
            del pbranhc[turn]
            del branhc[turn]
        if not pbranhc:
            del settings[branch]
            del presettings[branch]
        self.shallowest = OrderedDict()
        remove_keycache(parent + (entity, branch), turn, tick)

    def _remove_keycache(self, entity_branch, turn, tick):
        """Remove the future of a given entity from a branch in the keycache"""
        keycache = self.keycache
        if entity_branch in keycache:
            kc = keycache[entity_branch]
            if turn in kc:
                kcturn = kc[turn]
                if tick in kcturn:
                    del kcturn[tick]
                kcturn.truncate(tick)
                if not kcturn:
                    del kc[turn]
            kc.truncate(turn)
            if not kc:
                del keycache[entity_branch]

    def truncate(self, branch, turn, tick):
        """Delete all data after (not on) a specific tick"""
        parents, branches, keys, settings, presettings, keycache = self._truncate_stuff
        def truncate_branhc(branhc):
            if turn in branhc:
                trn = branhc[turn]
                trn.truncate(tick)
                branhc.truncate(turn)
                if not trn:
                    del branhc[turn]
            else:
                branhc.truncate(turn)
        for entities in parents.values():
            for keys in entities.values():
                for branches in keys.values():
                    if branch not in branches:
                        continue
                    truncate_branhc(branches[branch])
        for branches in branches.values():
            if branch not in branches:
                continue
            truncate_branhc(branches[branch])
        for keys in keys.values():
            for branches in keys.values():
                if branch not in branches:
                    continue
                truncate_branhc(branches[branch])
        truncate_branhc(settings[branch])
        truncate_branhc(presettings[branch])
        self.shallowest = OrderedDict()
        for entity_branch in keycache:
            if entity_branch[-1] == branch:
                truncate_branhc(keycache[entity_branch])

    @staticmethod
    def _iter_future_contradictions(entity, key, turns, branch, turn, tick, value):
        """If setting ``key=value`` would result in a contradiction, iterate over contradicted ``(turn, tick)``s."""
        # assumes that all future entries are in the plan
        if not turns:
            return
        if turn in turns:
            future_ticks = turns[turn].future(tick)
            for tck, newval in future_ticks.items():
                if newval != value:
                    yield turn, tck
            future_turns = turns.future(turn)
        elif turns.rev_gettable(turn):
            future_turns = turns.future(turn)
        else:
            future_turns = turns
        if not future_turns:
            return
        for trn, ticks in future_turns.items():
            for tick, newval in ticks.items():
                if newval != value:
                    yield trn, tick

    def _store_journal(self, *args):
        # overridden in LiSE.cache.InitializedCache
        entity, key, branch, turn, tick, value = args[-6:]
        parent = args[:-6]
        settings_turns = self.settings[branch]
        presettings_turns = self.presettings[branch]
        prev = self._base_retrieve(args[:-1])
        if prev is KeyError:
            prev = None
        if turn in settings_turns or turn in settings_turns.future():
            # These assertions hold for most caches but not for the contents
            # caches, and are therefore commented out.
            # assert turn in presettings_turns or turn in presettings_turns.future()
            setticks = settings_turns[turn]
            # assert tick not in setticks
            presetticks = presettings_turns[turn]
            # assert tick not in presetticks
            presetticks[tick] = parent + (entity, key, prev)
            setticks[tick] = parent + (entity, key, value)
        else:
            presettings_turns[turn] = {tick: parent + (entity, key, prev)}
            settings_turns[turn] = {tick: parent + (entity, key, value)}

    def _base_retrieve(self, args):
        shallowest = self.shallowest
        if args in shallowest:
            return shallowest[args]
        entity = args[:-4]
        key, branch, turn, tick = args[-4:]
        keyframes = self.keyframe.get(entity, {})
        branches = self.branches
        entikey = entity + (key,)
        if entikey in branches:
            branchentk = branches[entikey]
            brancs = branchentk.get(branch)
            if brancs is not None and brancs.rev_gettable(turn):
                if turn in brancs:
                    if brancs[turn].rev_gettable(tick):
                        ret = brancs[turn][tick]
                        shallowest[args] = ret
                        return ret
                    elif brancs.rev_gettable(turn-1):
                        b1 = brancs[turn-1]
                        ret = b1.final()
                        shallowest[args] = ret
                        return ret
                else:
                    ret = brancs[turn].final()
                    shallowest[args] = ret
                    return ret
            for (b, r, t) in self.db._iter_parent_btt(branch):
                brancs = branchentk.get(b)
                if brancs is not None and brancs.rev_gettable(r):
                    if r in brancs and brancs[r].rev_gettable(t):
                        # if there's a keyframe *later* than in brancs,
                        # but *earlier* than (b, r, t), use the
                        # keyframe instead
                        if b in keyframes and r in keyframes[b]:
                            kfbr = keyframes[b][r]
                            if brancs[r].rev_before(t) < kfbr.rev_before(t) < t:
                                kf = kfbr[t]
                                if key in kf:
                                    ret = kf[key]
                                    shallowest[args] = ret
                                    return ret
                        ret = brancs[r][t]
                        shallowest[args] = ret
                        return ret
                    elif brancs.rev_gettable(r - 1):
                        if b in keyframes and keyframes[b].rev_gettable(r - 1):
                            kfb = keyframes[b]
                            if brancs.rev_before(r - 1) < kfb.rev_before(r - 1):
                                kfbr = kfb[r - 1]
                                kf = kfbr.final()
                                if key in kf:
                                    ret = kf[key]
                                    shallowest[args] = ret
                                    return ret
                            elif brancs.rev_before(r - 1) == kfb.rev_before(r - 1):
                                kfbr = kfb[r - 1]
                                trns = brancs[r - 1]
                                if trns.end < kfbr.end:
                                    kf = kfbr.final()
                                    if key in kf:
                                        ret = kf[key]
                                        shallowest[args] = ret
                                        return ret
                        ret = brancs[r - 1].final()
                        shallowest[args] = ret
                        return ret
                    elif b in keyframes and r in keyframes[b] \
                            and keyframes[b][r].rev_gettable(t) \
                            and key in keyframes[b][r][t]:
                        ret = keyframes[b][r][t][key]
                        shallowest[args] = ret
                        return ret
                    elif b in keyframes and keyframes[b].rev_gettable(r - 1) \
                            and key in keyframes[b][r - 1].final():
                        ret = keyframes[b][r - 1].final()[key]
                        shallowest[args] = ret
                        return ret
                elif b in keyframes:
                    kfb = keyframes[branch]
                    if r in kfb:
                        kfbr = kfb[turn]
                        if kfbr.rev_gettable(tick):
                            kf = kfbr[tick]
                            if key in kf:
                                ret = kf[key]
                                shallowest[args] = ret
                                return ret
                    if kfb.rev_gettable(r - 1):
                        kfbr = kfb[r]
                        kf = kfbr.final()
                        if key in kf:
                            ret = kf[key]
                            shallowest[args] = ret
                            return ret
        else:
            if branch in keyframes:
                kfb = keyframes[branch]
                if turn in kfb:
                    kfbr = kfb[turn]
                    if kfbr.rev_gettable(tick):
                        kf = kfbr[tick]
                        if key in kf:
                            ret = kf[key]
                            shallowest[args] = ret
                            return ret
                if kfb.rev_gettable(turn-1):
                    kfbr = kfb[turn]
                    kf = kfbr.final()
                    if key in kf:
                        ret = kf[key]
                        shallowest[args] = ret
                        return ret
            for (b, r, t) in self.db._iter_parent_btt(branch):
                if b in keyframes:
                    kfb = keyframes[b]
                    if r in kfb:
                        kfbr = kfb[turn]
                        if kfbr.rev_gettable(tick):
                            kf = kfbr[tick]
                            if key in kf:
                                ret = kf[key]
                                shallowest[args] = ret
                                return ret
                    if kfb.rev_gettable(r-1):
                        kfbr = kfb[r]
                        kf = kfbr.final()
                        if key in kf:
                            ret = kf[key]
                            shallowest[args] = ret
                            return ret
        return KeyError

    def retrieve(self, *args):
        """Get a value previously .store(...)'d.

        Needs at least five arguments. The -1th is the tick
        within the turn you want,
        the -2th is that turn, the -3th is the branch,
        and the -4th is the key. All other arguments identify
        the entity that the key is in.

        """
        ret = self._base_retrieve(args)
        if ret is None:
            raise HistoryError("Set, then deleted", deleted=True)
        elif ret is KeyError:
            raise ret
        return ret

    def iter_entities_or_keys(self, *args, forward=None):
        """Iterate over the keys an entity has, if you specify an entity.

        Otherwise iterate over the entities themselves, or at any rate the
        tuple specifying which entity.

        """
        if forward is None:
            forward = self.db._forward
        entity = args[:-3]
        branch, turn, tick = args[-3:]
        if self.db._no_kc:
            yield from self._get_adds_dels(entity, branch, turn, tick)[0]
            return
        yield from self._get_keycache(entity, branch, turn, tick, forward=forward)
    iter_entities = iter_keys = iter_entity_keys = iter_entities_or_keys

    def count_entities_or_keys(self, *args, forward=None):
        """Return the number of keys an entity has, if you specify an entity.

        Otherwise return the number of entities.

        """
        if forward is None:
            forward = self.db._forward
        entity = args[:-3]
        branch, turn, tick = args[-3:]
        if self.db._no_kc:
            return len(self._get_adds_dels(entity, branch, turn, tick)[0])
        return len(self._get_keycache(entity, branch, turn, tick, forward=forward))
    count_entities = count_keys = count_entity_keys = count_entities_or_keys

    def contains_entity_or_key(self, *args):
        """Check if an entity has a key at the given time, if entity specified.

        Otherwise check if the entity exists.

        """
        try:
            return self.retrieve(*args) is not None
        except KeyError:
            return False
    contains_entity = contains_key = contains_entity_key \
                    = contains_entity_or_key


class NodesCache(Cache):
    """A cache for remembering whether nodes exist at a given time."""
    __slots__ = ()

    def store(self, graph, node, branch, turn, tick, ex, *, planning=None, forward=None, loading=False, contra=True):
        if not ex:
            ex = None
        return super().store(graph, node, branch, turn, tick, ex, planning=planning, forward=forward, loading=loading, contra=contra)

    def _update_keycache(self, *args, forward):
        graph, node, branch, turn, tick, ex = args
        if not ex:
            ex = None
        super()._update_keycache(graph, node, branch, turn, tick, ex, forward=forward)

    def _iter_future_contradictions(self, entity, key, turns, branch, turn, tick, value):
        yield from super()._iter_future_contradictions(entity, key, turns, branch, turn, tick, value)
        yield from self.db._edges_cache._slow_iter_node_contradicted_times(
            branch, turn, tick, entity, key
        )


class EdgesCache(Cache):
    """A cache for remembering whether edges exist at a given time."""
    __slots__ = (
        'destcache', 'origcache', 'predecessors',
        '_origcache_lru', '_destcache_lru', '_get_destcache_stuff',
        '_get_origcache_stuff', '_additional_store_stuff'
    )
    @property
    def successors(self):
        return self.parents

    def __init__(self, db):
        def gettest(k):
            assert len(k) == 3, "Bad key: " + repr(k)

        def settest(k, v):
            assert len(k) == 3, "Bad key: {}, to be set to {}".format(k, v)
        Cache.__init__(self, db, kfkvs={'gettest': gettest, 'settest': settest})
        self.destcache = PickyDefaultDict(SettingsTurnDict)
        self.origcache = PickyDefaultDict(SettingsTurnDict)
        self.predecessors = StructuredDefaultDict(3, TurnDict)
        self._origcache_lru = OrderedDict()
        self._destcache_lru = OrderedDict()
        self._get_destcache_stuff = (
            self.destcache, self._destcache_lru, self._get_keycachelike,
            self.successors, self._adds_dels_successors
        )
        self._get_origcache_stuff = (
            self.origcache, self._origcache_lru, self._get_keycachelike,
            self.predecessors, self._adds_dels_predecessors
        )
        self._additional_store_stuff = (
            self.db, self.predecessors, self.successors
        )

    def _update_keycache(self, *args, forward):
        super()._update_keycache(*args, forward=forward)
        dest, key, branch, turn, tick, value = args[-6:]
        graph, orig = args[:-6]
        # it's possible either of these might cause unnecessary iteration
        dests = self._get_destcache(graph, orig, branch, turn, tick, forward=forward)
        origs = self._get_origcache(graph, dest, branch, turn, tick, forward=forward)
        if value is None:
            dests = dests.difference((dest,))
            origs = origs.difference((orig,))
        else:
            dests = dests.union((dest,))
            origs = origs.union((orig,))
        self.destcache[graph, orig, branch][turn][tick] = dests
        self.origcache[graph, dest, branch][turn][tick] = origs

    def _slow_iter_node_contradicted_times(self, branch, turn, tick, graph, node):
        # slow and bad.
        retrieve = self._base_retrieve
        for items in (self.successors[graph, node].items(), self.predecessors[graph, node].items()):
            for dest, idxs in items:  # dest might really be orig
                for idx, branches in idxs.items():
                    brnch = branches[branch]
                    if turn in brnch:
                        ticks = brnch[turn]
                        for tck, present in ticks.future(tick).items():
                            if tck > tick and present is not retrieve((graph, node, dest, idx, branch, turn, tick)):
                                yield turn, tck
                    for trn, ticks in brnch.future(turn).items():
                        for tck, present in ticks.items():
                            if present is not retrieve((graph, node, dest, idx, branch, turn, tick)):
                                yield trn, tck

    def _adds_dels_successors(self, parentity, branch, turn, tick, *,
                              stoptime=None, cache=None):
        graph, orig = parentity
        added = set()
        deleted = set()
        cache = cache or self.successors
        if (graph, orig) in cache and cache[graph, orig]:
            for dest in cache[graph, orig]:
                addidx, delidx = self._get_adds_dels(
                    (graph, orig, dest), branch, turn, tick,
                    stoptime=stoptime)
                if addidx and not delidx:
                    added.add(dest)
                elif delidx and not addidx:
                    deleted.add(dest)
        if stoptime:
            return added, deleted
        kf = self.keyframe
        itparbtt = self.db._iter_parent_btt
        its = list(kf.items())
        for ks, v in its:
            assert len(ks) == 3, "Bad key in keyframe: " + repr(ks)
        for (grap, org, dest), kfg in its:  # too much iteration!
            if (grap, org) != (graph, orig):
                continue
            for branc, trn, tck in itparbtt(branch, turn, tick):
                if branc not in kfg:
                    continue
                kfgb = kfg[branc]
                if trn in kfgb:
                    kfgbr = kfgb[trn]
                    if kfgbr.rev_gettable(tck):
                        if kfgbr[tck][0] and dest not in deleted:
                            added.add(dest)
                        continue
                if kfgb.rev_gettable(trn):
                    if kfgb[trn].final()[0] and dest not in deleted:
                        added.add(dest)
        for ks in kf.keys():
            assert len(ks) == 3, "BBadd key in keyframe: " + repr(ks)
        return added, deleted

    def _adds_dels_predecessors(self, parentity, branch, turn, tick, *,
                                stoptime=None, cache=None):
        graph, dest = parentity
        added = set()
        deleted = set()
        cache = cache or self.predecessors
        if cache[graph, dest]:
            for orig in cache[graph, dest]:
                addidx, delidx = self._get_adds_dels(
                    (graph, orig, dest), branch, turn, tick,
                    stoptime=stoptime)
                if addidx and not delidx:
                    added.add(orig)
                elif delidx and not addidx:
                    deleted.add(orig)
        else:
            if stoptime:
                return added, deleted
            kf = self.keyframe
            itparbtt = self.db._iter_parent_btt
            for (grap, orig, dst), kfg in kf.items():  # too much iteration!
                if (grap, dst) != (graph, dest):
                    continue
                for branc, trn, tck in itparbtt(branch, turn, tick):
                    if branc not in kfg:
                        continue
                    kfgb = kfg[branc]
                    if trn in kfgb:
                        kfgbr = kfgb[trn]
                        if kfgbr.rev_gettable(tck):
                            if kfgbr[tick][0]:
                                added.add(orig)
                            continue
                    if kfgb.rev_gettable(trn):
                        if kfgb[trn].final()[0]:
                            added.add(orig)
        return added, deleted

    def _get_destcache(self, graph, orig, branch, turn, tick, *, forward):
        """Return a set of destination nodes succeeding ``orig``"""
        destcache, destcache_lru, get_keycachelike, successors, adds_dels_sucpred = self._get_destcache_stuff
        lru_append(destcache, destcache_lru, ((graph, orig, branch), turn, tick), KEYCACHE_MAXSIZE)
        return get_keycachelike(
            destcache, successors, adds_dels_sucpred, (graph, orig),
            branch, turn, tick, forward=forward
        )

    def _get_origcache(self, graph, dest, branch, turn, tick, *, forward):
        """Return a set of origin nodes leading to ``dest``"""
        origcache, origcache_lru, get_keycachelike, predecessors, adds_dels_sucpred = self._get_origcache_stuff
        lru_append(origcache, origcache_lru, ((graph, dest, branch), turn, tick), KEYCACHE_MAXSIZE)
        return get_keycachelike(
            origcache, predecessors, adds_dels_sucpred, (graph, dest),
            branch, turn, tick, forward=forward
        )

    def iter_successors(self, graph, orig, branch, turn, tick, *, forward=None):
        """Iterate over successors of a given origin node at a given time."""
        if self.db._no_kc:
            yield from self._adds_dels_successors(
                (graph, orig), branch, turn, tick)[0]
            return
        if forward is None:
            forward = self.db._forward
        yield from self._get_destcache(graph, orig, branch, turn, tick, forward=forward)

    def iter_predecessors(self, graph, dest, branch, turn, tick, *, forward=None):
        """Iterate over predecessors to a given destination node at a given time."""
        if self.db._no_kc:
            yield from self._adds_dels_predecessors(
                (graph, dest), branch, turn, tick)[0]
            return
        if forward is None:
            forward = self.db._forward
        yield from self._get_origcache(graph, dest, branch, turn, tick, forward=forward)

    def count_successors(self, graph, orig, branch, turn, tick, *, forward=None):
        """Return the number of successors to a given origin node at a given time."""
        if self.db._no_kc:
            return len(self._adds_dels_successors(
                (graph, orig), branch, turn, tick)[0])
        if forward is None:
            forward = self.db._forward
        return len(self._get_destcache(graph, orig, branch, turn, tick, forward=forward))

    def count_predecessors(self, graph, dest, branch, turn, tick, *, forward=None):
        """Return the number of predecessors from a given destination node at a given time."""
        if self.db._no_kc:
            return len(self._adds_dels_predecessors(graph, dest, branch, turn, tick)[0])
        if forward is None:
            forward = self.db._forward
        return len(self._get_origcache(graph, dest, branch, turn, tick, forward=forward))

    def has_successor(self, graph, orig, dest, branch, turn, tick, *, forward=None):
        """Return whether an edge connects the origin to the destination at the given time.

        Doesn't require the edge's index, which makes it slower than retrieving a
        particular edge.

        """
        if forward is None:
            forward = self.db._forward
        return dest in self._get_destcache(graph, orig, branch, turn, tick, forward=forward)

    def has_predecessor(self, graph, dest, orig, branch, turn, tick, *, forward=None):
        """Return whether an edge connects the destination to the origin at the given time.

        Doesn't require the edge's index, which makes it slower than retrieving a
        particular edge.

        """
        if forward is None:
            forward = self.db._forward
        return orig in self._get_origcache(graph, dest, branch, turn, tick, forward=forward)

    def store(self, graph, orig, dest, idx, branch, turn, tick, ex, *, planning=None, forward=None, loading=False, contra=True):
        db, predecessors, successors = self._additional_store_stuff
        if not ex:
            ex = None
        if planning is None:
            planning = db._planning
        Cache.store(self, graph, orig, dest, idx, branch, turn, tick, ex, planning=planning, forward=forward, loading=loading, contra=contra)
        predecessors[graph, dest][orig][idx][branch][turn] \
            = successors[graph, orig][dest][idx][branch][turn]
        # if ex:
        #     assert self.retrieve(graph, orig, dest, idx, branch, turn, tick)
        #     assert self.has_successor(graph, orig, dest, branch, turn, tick)
        #     assert self.has_predecessor(graph, dest, orig, branch, turn, tick)
        # else:
        #     assert self._base_retrieve((graph, orig, dest, idx, branch, turn, tick)) in (None, KeyError)
        #     assert not self.has_successor(graph, orig, dest, branch, turn, tick)
        #     assert not self.has_predecessor(graph, dest, orig, branch, turn, tick)