# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""The query engine provides Pythonic methods to access the database."""
from operator import gt, lt, eq, ne, le, ge
from functools import partialmethod

import allegedb.query

from .exc import (
    IntegrityError,
    OperationalError,
    RedundantRuleError
)
from .util import EntityStatAccessor
import LiSE


def windows_union(windows):
    def fix_overlap(left, right):
        if left == right:
            return [left]
        assert left[0] < right[0]
        if left[1] >= right[0]:
            if right[1] > left[1]:
                return [(left[0], right[1])]
            else:
                return [left]
        return [left, right]

    if len(windows) == 1:
        yield windows[0]
        return
    none_left = []
    otherwise = []
    for window in windows:
        if window[0] is None:
            none_left.append(window)
        else:
            otherwise.append(window)

    res = []
    otherwise.sort()
    for window in none_left:
        if not res:
            res.append(window)
            continue
        res.extend(fix_overlap(res.pop(), window))
    while otherwise:
        window = otherwise.pop(0)
        if not res:
            res.append(window)
            continue
        res.extend(fix_overlap(res.pop(), window))
    return res


def windows_intersection(windows):
    """

    :rtype: list
    """

    def intersect2(left, right):
        if left == right:
            return left
        elif left is (None, None):
            return right
        elif right is (None, None):
            return left
        elif left[0] is None:
            if right[0] is None:
                return None, min((left[1], right[1]))
            elif right[1] is None:
                if left[1] <= right[0]:
                    return left[1], right[0]
                else:
                    return None
            elif right[0] <= left[1]:
                return right[0], left[1]
            else:
                return None
        elif left[1] is None:
            if right[0] is None:
                return left[0], right[1]
            else:
                return right  # assumes left[0] <= right[0]
        # None not in left
        elif right[0] is None:
            return left[0], min((left[1], right[1]))
        elif right[1] is None:
            if left[1] >= right[0]:
                return right[0], left[1]
            else:
                return None
        assert None not in left and None not in right and left[0] < right[1]
        if left[1] >= right[0]:
            if right[1] > left[1]:
                return right[0], left[1]
            else:
                return right
        return None

    if len(windows) == 1:
        return windows
    left_none = []
    otherwise = []
    for window in windows:
        if window[0] is None:
            left_none.append(window)
        else:
            otherwise.append(window)

    done = []
    todo = left_none + sorted(otherwise)
    for window in todo:
        if not done:
            done.append(window)
            continue
        res = intersect2(done.pop(), window)
        if res:
            done.append(res)
    return done


class Query(object):
    def __new__(cls, engine, leftside, rightside=None, **kwargs):
        if rightside is None:
            if not isinstance(leftside, cls):
                raise TypeError("You can't make a query with only one side")
            me = leftside
        else:
            me = super().__new__(cls)
            me.leftside = leftside
            me.rightside = rightside
        me.engine = engine
        me.windows = kwargs.get('windows', [])
        return me

    def __call__(self):
        raise NotImplementedError("Query is abstract")

    def __eq__(self, other):
        return EqQuery(self.engine, self, self.engine.entityfy(other))

    def __gt__(self, other):
        return GtQuery(self.engine, self, self.engine.entityfy(other))

    def __ge__(self, other):
        return GeQuery(self.engine, self, self.engine.entityfy(other))

    def __lt__(self, other):
        return LtQuery(self.engine, self, self.engine.entityfy(other))

    def __le__(self, other):
        return LeQuery(self.engine, self, self.engine.entityfy(other))

    def __ne__(self, other):
        return NeQuery(self.engine, self, self.engine.entityfy(other))

    def and_before(self, end):
        if self.windows:
            new_windows = windows_intersection(
                sorted(self.windows + [(None, end)])
            )
        else:
            new_windows = [(0, end)]
        return type(self)(self.leftside, self.rightside, windows=new_windows)
    before = and_before

    def or_before(self, end):
        if self.windows:
            new_windows = windows_union(self.windows + [(None, end)])
        else:
            new_windows = [(None, end)]
        return type(self)(self.leftside, self.rightside, windows=new_windows)

    def and_after(self, start):
        if self.windows:
            new_windows = windows_intersection(self.windows + [(start, None)])
        else:
            new_windows = [(start, None)]
        return type(self)(self.leftside, self.rightside, windows=new_windows)
    after = and_after

    def or_between(self, start, end):
        if self.windows:
            new_windows = windows_union(self.windows + [(start, end)])
        else:
            new_windows = [(start, end)]
        return type(self)(self.leftside, self.rightside, windows=new_windows)

    def and_between(self, start, end):
        if self.windows:
            new_windows = windows_intersection(self.windows + [(start, end)])
        else:
            new_windows = [(start, end)]
        return type(self)(self.leftside, self.rightside, windows=new_windows)
    between = and_between

    def or_during(self, tick):
        return self.or_between(tick, tick)

    def and_during(self, tick):
        return self.and_between(tick, tick)
    during = and_during


class Union(Query):
    pass


class ComparisonQuery(Query):
    oper = lambda x, y: NotImplemented

    def __call__(self):
        return QueryResults(iter_eval_cmp(self, self.oper, engine=self.engine))


class EqQuery(ComparisonQuery):
    oper = eq


class NeQuery(ComparisonQuery):
    oper = ne


class GtQuery(ComparisonQuery):
    oper = gt


class LtQuery(ComparisonQuery):
    oper = lt


class GeQuery(ComparisonQuery):
    oper = ge


class LeQuery(ComparisonQuery):
    oper = le


comparisons = {
    'eq': EqQuery,
    'ne': NeQuery,
    'gt': GtQuery,
    'lt': LtQuery,
    'ge': GeQuery,
    'le': LeQuery
}


class StatusAlias(EntityStatAccessor):
    def __eq__(self, other):
        return EqQuery(self.engine, self, other)

    def __ne__(self, other):
        return NeQuery(self.engine, self, other)

    def __gt__(self, other):
        return GtQuery(self.engine, self, other)

    def __lt__(self, other):
        return LtQuery(self.engine, self, other)

    def __ge__(self, other):
        return GeQuery(self.engine, self, other)

    def __le__(self, other):
        return LeQuery(self.engine, self, other)


def intersect_qry(qry):
    windows = []
    windowses = 0
    if hasattr(qry.leftside, 'windows'):
        windows.extend(qry.leftside.windows)
        windowses += 1
    if hasattr(qry.rightside, 'windows'):
        windows.extend(qry.rightside.windows)
        windowses += 1
    if windowses > 1:
        windows = windows_intersection(windowses)
    return windows


def iter_intersection_ticks2check(ticks, windows):
    windows = windows_intersection(windows)
    if not windows:
        yield from ticks
        return
    for tick in sorted(ticks):
        (left, right) = windows.pop(0)
        if left is None:
            if tick <= right:
                yield tick
                windows.insert(0, (left, right))
        elif right is None:
            if tick >= left:
                yield from ticks
                return
            windows.insert(0, (left, right))
        elif left <= tick <= right:
            yield tick
            windows.insert(0, (left, right))
        elif tick < left:
            windows.insert(0, (left, right))


class QueryResults(object):
    def __init__(self, iter):
        self.iter = iter
        try:
            self.next = next(self.iter)
        except StopIteration:
            return

    def __iter__(self):
        return self

    def __next__(self):
        try:
            r = self.next
        except AttributeError:
            raise StopIteration
        try:
            self.next = next(self.iter)
        except StopIteration:
            del self.next
        return r

    def __bool__(self):
        return hasattr(self, 'next')


def iter_eval_cmp(qry, oper, start_branch=None, engine=None):
    def mungeside(side):
        if isinstance(side, Query):
            return side()
        elif isinstance(side, StatusAlias):
            return EntityStatAccessor(
                side.entity, side.stat, side.engine,
                side.branch, side.tick, side.current, side.mungers
            )
        elif isinstance(side, EntityStatAccessor):
            return side
        else:
            return lambda b, t: side

    def getcache(side):
        if hasattr(side, 'cache'):
            return side.cache
        if hasattr(side, 'entity'):
            if side.stat in (
                    'location', 'next_location', 'locations',
                    'arrival_time', 'next_arrival_time'
            ):
                return engine._things_cache.branches[
                    (side.entity.character.name, side.entity.name)]
            if side.stat in side.entity._cache:
                return side.entity._cache[side.stat]

    leftside = mungeside(qry.leftside)
    rightside = mungeside(qry.rightside)
    windows = qry.windows or [(0, None)]
    engine = engine or leftside.engine or rightside.engine
    for (branch, _) in engine._active_branches(start_branch):
        try:
            lkeys = frozenset(getcache(leftside)[branch].keys())
        except AttributeError:
            lkeys = frozenset()
        try:
            rkeys = getcache(rightside)[branch].keys()
        except AttributeError:
            rkeys = frozenset()
        ticks = lkeys.union(rkeys)
        if ticks:
            yield from (
                (branch, tick) for tick in
                iter_intersection_ticks2check(ticks, windows)
                if oper(leftside(branch, tick), rightside(branch, tick))
            )
        else:
            yield from (
                (branch, tick) for tick in
                range(engine._branch_start.get(branch, 0), engine.tick+1)
                if oper(leftside(branch, tick), rightside(branch, tick))
            )


class QueryEngine(allegedb.query.QueryEngine):
    json_path = LiSE.__path__[0]
    IntegrityError = IntegrityError
    OperationalError = OperationalError

    def universals_dump(self):
        for key, branch, tick, value in self.sql('universals_dump'):
            yield self.json_load(key), branch, tick, self.json_load(value)

    def rulebooks_dump(self):
        for rulebook, branch, tick, rules in self.sql('rulebooks_dump'):
            yield rulebook, branch, tick, self.json_load(rules)

    def _rule_dump(self, typ):
        for rule, branch, tick, lst in self.sql('rule_{}_dump'.format(typ)):
            yield rule, branch, tick, self.json_load(lst)

    def rule_triggers_dump(self):
        return self._rule_dump('triggers')

    def rule_prereqs_dump(self):
        return self._rule_dump('prereqs')

    def rule_actions_dump(self):
        return self._rule_dump('actions')

    def characters_dump(self):
        for graph, typ in self.sql('graphs_dump'):
            if typ == 'DiGraph':
                yield self.json_load(graph)
    characters = characters_dump

    def node_rulebook_dump(self):
        for character, node, branch, tick, rulebook in self.sql('node_rulebook_dump'):
            yield self.json_load(character), self.json_load(node), branch, tick, rulebook

    def portal_rulebook_dump(self):
        for character, orig, dest, branch, tick, rulebook in self.sql('portal_rulebook_dump'):
            yield (
                self.json_load(character), self.json_load(orig), self.json_load(dest),
                branch, tick, rulebook
            )

    def _charactery_rulebook_dump(self, qry):
        for character, branch, tick, rulebook in self.sql(qry+'_rulebook_dump'):
            yield self.json_load(character), branch, tick, rulebook

    character_rulebook_dump = partialmethod(_charactery_rulebook_dump, 'character')
    avatar_rulebook_dump = partialmethod(_charactery_rulebook_dump, 'avatar')
    character_thing_rulebook_dump = partialmethod(_charactery_rulebook_dump, 'character_thing')
    character_place_rulebook_dump = partialmethod(_charactery_rulebook_dump, 'character_place')
    character_portal_rulebook_dump = partialmethod(_charactery_rulebook_dump, 'character_portal')

    def character_rules_handled_dump(self):
        for character, rulebook, rule, branch, tick in self.sql('character_rules_handled_dump'):
            yield self.json_load(character), rulebook, rule, branch, tick

    def avatar_rules_handled_dump(self):
        for character, rulebook, rule, graph, avatar, branch, tick in self.sql('avatar_rules_handled_dump'):
            yield self.json_load(character), rulebook, rule, self.json_load(graph), self.json_load(avatar), branch, tick

    def character_thing_rules_handled_dump(self):
        for character, rulebook, rule, thing, branch, tick in self.sql('character_thing_rules_handled_dump'):
            yield self.json_load(character), rulebook, rule, self.json_load(thing), branch, tick

    def character_place_rules_handled_dump(self):
        for character, rulebook, rule, place, branch, tick in self.sql('character_place_rules_handled_dump'):
            yield self.json_load(character), rulebook, rule, self.json_load(place), branch, tick

    def character_portal_rules_handled_dump(self):
        for character, rulebook, rule, orig, dest, branch, tick in self.sql('character_portal_rules_handled_dump'):
            yield (
                self.json_load(character), rulebook, rule, self.json_load(orig), self.json_load(dest),
                branch, tick
            )

    def node_rules_handled_dump(self):
        for character, node, rulebook, rule, branch, tick in self.sql('node_rules_handled_dump'):
            yield self.json_load(character), self.json_load(node), rulebook, rule, branch, tick

    def portal_rules_handled_dump(self):
        for character, orig, dest, rulebook, rule, branch, tick in self.sql('portal_rules_handled_dump'):
            yield (
                self.json_load(character), self.json_load(orig), self.json_load(dest),
                rulebook, rule, branch, tick
            )

    def senses_dump(self):
        for character, sense, branch, tick, function in self.sql('senses_dump'):
            yield self.json_load(character), sense, branch, tick, function

    def things_dump(self):
        for character, thing, branch, tick, location, next_location in self.sql('things_dump'):
            yield (
                self.json_load(character), self.json_load(thing), branch, tick,
                self.json_load(location), self.json_load(next_location) if next_location else None
            )

    def avatars_dump(self):
        for character_graph, avatar_graph, avatar_node, branch, tick, is_av in self.sql('avatars_dump'):
            yield (
                self.json_load(character_graph), self.json_load(avatar_graph),
                self.json_load(avatar_node), branch, tick, is_av
            )

    def universal_set(self, key, branch, tick, val):
        key, val = map(self.json_dump, (key, val))
        try:
            self.sql('universals_insert', key, branch, tick, val)
        except IntegrityError:
            self.sql('universals_update', val, key, branch, tick)

    def universal_del(self, key, branch, tick):
        key = self.json_dump(key)
        try:
            self.sql('universals_insert', key, branch, tick, None)
        except IntegrityError:
            self.sql('universals_update', None, key, branch, tick)

    def comparison(
            self, entity0, stat0, entity1,
            stat1=None, oper='eq', windows=[]
    ):
        stat1 = stat1 or stat0
        return comparisons[oper](
            leftside=entity0.status(stat0),
            rightside=entity1.status(stat1),
            windows=windows
        )

    def count_all_table(self, tbl):
        return self.sql('{}_count'.format(tbl)).fetchone()[0]

    def init_table(self, tbl):
        try:
            return self.sql('create_{}'.format(tbl))
        except OperationalError:
            pass

    def index_table(self, tbl):
        try:
            return self.sql('index_{}'.format(tbl))
        except OperationalError:
            pass

    def rules_dump(self):
        return self.sql('rules_dump')

    def rule_triggers(self, rule):
        rule = self.json_dump(rule)
        for row in self.sql('rule_triggers', rule):
            yield row[0]

    def rule_prereqs(self, rule):
        rule = self.json_dump(rule)
        for row in self.sql('rule_prereqs', rule):
            yield row[0]

    def rule_actions(self, rule):
        rule = self.json_dump(rule)
        for row in self.sql('rule_actions', rule):
            yield row[0]

    def set_rulebook(self, name, branch='trunk', tick=0, rules=None):
        name, rules = map(self.json_dump, (name, rules or []))
        try:
            self.sql('rulebooks_insert', name, branch, tick, rules)
        except IntegrityError:
            self.sql('rulebooks_update', rules, name, branch, tick)

    def init_character(self, name, branch='trunk', tick=0, **stats):
        self.sql('new_graph', name, 'DiGraph')
        for rbtyp in (
            'character',
            'avatar',
            'character_thing',
            'character_place',
            'character_portal'
        ):
            self.set_rulebook((name, rbtyp), branch, tick)
            self.sql(rbtyp + '_rulebook_insert', name, branch, tick, self.json_dump((name, rbtyp)))
        for k, v in stats.items():
            self.graph_val_set(name, k, branch, tick, v)

    def _set_rulebook(self, rbtyp, char, rb, branch, tick):
        char, rb = map(self.json_dump, (char, rb))
        try:
            self.sql(rbtyp + '_rulebook_insert', char, rb, branch, tick)
        except IntegrityError:
            self.sql(rbtyp + '_rulebook_update', rb, char, branch, tick)

    set_character_rulebook = partialmethod(_set_rulebook, 'character')
    set_avatar_rulebook = partialmethod(_set_rulebook, 'avatar')
    set_character_thing_rulebook = partialmethod(_set_rulebook, 'character_thing')
    set_character_place_rulebook = partialmethod(_set_rulebook, 'character_place')
    set_character_portal_rulebook = partialmethod(_set_rulebook, 'character_portal')

    def rulebooks(self):
        for book in self.sql('rulebooks'):
            yield self.json_load(book)

    def set_node_rulebook(self, character, node, rulebook):
        (character, node, rulebook) = map(
            self.json_dump, (character, node, rulebook)
        )
        try:
            return self.sql('ins_node_rulebook', character, node, rulebook)
        except IntegrityError:
            return self.sql('upd_node_rulebook', rulebook, character, node)

    def portal_rulebook(self, character, orig, dest):
        (character, orig, dest) = map(
            self.json_dump, (character, orig, dest)
        )
        r = self.sql(
            'portal_rulebook',
            character,
            orig,
            dest,
            0
        ).fetchone()
        if r is None:
            raise KeyError(
                "No rulebook for portal {}->{} in character {}".format(
                    orig, dest, character
                )
            )
        return self.json_load(r[0])

    def set_portal_rulebook(self, character, orig, dest, branch, tick, rulebook):
        (character, orig, dest, rulebook) = map(
            self.json_dump, (character, orig, dest, rulebook)
        )
        try:
            return self.sql(
                'portal_rulebook_insert',
                character,
                orig,
                dest,
                branch,
                tick,
                rulebook
            )
        except IntegrityError:
            return self.sql(
                'portal_rulebook_update',
                rulebook,
                character,
                orig,
                dest,
                branch,
                tick
            )

    def handled_character_rule(
            self, character, rulebook, rule, branch, tick
    ):
        (character, rulebook, rule) = map(
            self.json_dump, (character, rulebook, rule)
        )
        try:
            return self.sql(
                'handled_character_rule',
                character,
                rulebook,
                rule,
                branch,
                tick,
            )
        except IntegrityError:
            raise RedundantRuleError(
                "Already handled rule {rule} in rulebook {book} "
                "for character {ch} at tick {t} of branch {b}".format(
                    ch=character,
                    book=rulebook,
                    rule=rule,
                    b=branch,
                    t=tick
                )
            )

    def handled_thing_rule(
            self, character, thing, rulebook, rule, branch, tick
    ):
        (character, thing, rulebook, rule) = map(
            self.json_dump, (character, thing, rulebook, rule)
        )
        try:
            return self.sql(
                'handled_thing_rule',
                character,
                thing,
                rulebook,
                rule,
                branch,
                tick
            )
        except IntegrityError:
            raise RedundantRuleError(
                "Already handled rule {r} in rulebook {book} "
                "for thing {th} "
                "at tick {t} of branch {b}".format(
                    r=rule,
                    book=rulebook,
                    th=thing,
                    b=branch,
                    t=tick
                )
            )

    def handled_place_rule(
            self, character, place, rulebook, rule, branch, tick
    ):
        (character, place, rulebook, rule) = map(
            self.json_dump, (character, place, rulebook, rule)
        )
        try:
            return self.sql(
                'handled_place_rule',
                character,
                place,
                rulebook,
                rule,
                branch,
                tick
            )
        except IntegrityError:
            raise RedundantRuleError(
                "Already handled rule {rule} in rulebook {book} "
                "for place {place} at tick {tick} of branch {branch}".format(
                    place=place,
                    rulebook=rulebook,
                    rule=rule,
                    branch=branch,
                    tick=tick
                )
            )

    def handled_portal_rule(
            self, character, orig, dest, rulebook, rule, branch, tick
    ):
        (character, orig, dest, rulebook, rule) = map(
            self.json_dump, (character, orig, dest, rulebook, rule)
        )
        try:
            return self.sql(
                'handled_portal_rule',
                character,
                orig,
                dest,
                0,
                rulebook,
                rule,
                branch,
                tick
            )
        except IntegrityError:
            raise RedundantRuleError(
                "Already handled rule {rule} in rulebook {book} "
                "for portal from {orig} to {dest} "
                "at tick {tick} of branch {branch}".format(
                    orig=orig,
                    dest=dest,
                    book=rulebook,
                    rule=rule,
                    branch=branch,
                    tick=tick
                )
            )

    def get_rulebook_char(self, rulemap, character):
        character = self.json_dump(character)
        for (book,) in self.sql(
                'rulebook_get_{}'.format(rulemap), character
        ):
            return self.json_load(book)
        raise KeyError("No rulebook")

    def upd_rulebook_char(self, rulemap, character):
        return self.sql('upd_rulebook_char_fmt', character, rulemap=rulemap)

    def thing_loc_and_next_set(
            self, character, thing, branch, tick, loc, nextloc
    ):
        (character, thing) = map(
            self.json_dump,
            (character, thing)
        )
        loc = self.json_dump(loc) if loc else None
        nextloc = self.json_dump(nextloc) if nextloc else None
        try:
            return self.sql(
                'things_insert',
                character,
                thing,
                branch,
                tick,
                loc,
                nextloc
            )
        except IntegrityError:
            return self.sql(
                'things_update',
                loc,
                nextloc,
                character,
                thing,
                branch,
                tick
            )

    # TODO character_init, sense_fun stuff

    def avatar_set(self, character, graph, node, branch, tick, isav):
        (character, graph, node) = map(
            self.json_dump, (character, graph, node)
        )
        try:
            return self.sql(
                'avatar_ins', character, graph, node, branch, tick, isav
            )
        except IntegrityError:
            return self.sql(
                'avatar_upd', isav, character, graph, node, branch, tick
            )

    def rulebooks_rules(self):
        for (rulebook, rule) in self.sql('rulebooks_rules'):
            yield map(self.json_load, (rulebook, rule))

    def rulebook_get(self, rulebook, idx):
        return self.json_load(
            self.sql(
                'rulebook_get', self.json_dump(rulebook), idx
            ).fetchone()[0]
        )

    def branch_descendants(self, branch):
        for child in self.sql('branch_children', branch):
            yield child
            yield from self.branch_descendants(child)

    def initdb(self):
        """Set up the database schema, both for allegedb and the special
        extensions for LiSE

        """
        super().initdb()
        for table in (
            'universals',
            'rules',
            'rulebooks',
            'senses',
            'things',
            'character_rulebook',
            'avatar_rulebook',
            'character_thing_rulebook',
            'character_place_rulebook',
            'character_portal_rulebook',
            'node_rulebook',
            'portal_rulebook',
            'avatars',
            'character_rules_handled',
            'avatar_rules_handled',
            'character_thing_rules_handled',
            'character_place_rules_handled',
            'character_portal_rules_handled',
            'node_rules_handled',
            'portal_rules_handled',
            'rule_triggers',
            'rule_prereqs',
            'rule_actions'
        ):
            self.init_table(table)
        for idx in (
            'senses',
            'things',
            'avatars',
            'character_rules_handled',
            'avatar_rules_handled',
            'character_thing_rules_handled',
            'character_place_rules_handled',
            'character_portal_rules_handled',
            'node_rules_handled',
            'portal_rules_handled'
        ):
            self.index_table(idx)
