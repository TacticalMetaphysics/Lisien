# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""The query engine provides Pythonic methods to access the database."""
from inspect import getsource
from types import FunctionType
from marshal import loads as unmarshalled
from marshal import dumps as marshalled
from operator import gt, lt, eq, ne, le, ge

import allegedb.query

from .exc import (
    IntegrityError,
    OperationalError,
    RedundantRuleError,
    UserFunctionError
)
from .util import EntityStatAccessor
import LiSE

string_defaults = {
    'strings': {'eng': [('README', 'Write release notes for your game here.')]}
}


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
        return self.sql('count_all_{}'.format(tbl)).fetchone()[0]

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

    def set_rule(self, rule, date=None, creator=None, description=None):
        rule = self.json_dump(rule)
        try:
            self.sql('ins_rule', rule, date, creator, description)
        except IntegrityError:
            self.sql('upd_rule', date, creator, description, rule)

    def rule_triggers(self, rule):
        rule = self.json_dump(rule)
        for row in self.sql('rule_triggers', rule):
            yield row[0]

    def insert_rule_trigger(self, rule, i, trigger):
        rule = self.json_dump(rule)
        self.sql('rule_triggers_inc', rule, i)
        self.sql('rule_triggers_ins', rule, i, trigger)

    def append_rule_trigger(self, rule, trigger):
        rule = self.json_dump(rule)
        ct = self.sql('rule_triggers_count', rule).fetchone()[0]
        self.sql('rule_triggers_ins', rule, ct, trigger)

    def delete_rule_trigger(self, rule, i):
        rule = self.json_dump(rule)
        self.sql('rule_triggers_del', rule, i)
        self.sql('rule_triggers_dec', rule, i)

    def replace_rule_trigger(self, rule, i, trigger):
        rule = self.json_dump(rule)
        self.sql('rule_triggers_del', rule, i)
        self.sql('rule_triggers_ins', rule, i, trigger)

    def clear_rule_triggers(self, rule):
        self.sql('rule_triggers_del_all', self.json_dump(rule))

    def replace_all_rule_triggers(self, rule, triggers):
        rule = self.json_dump(rule)
        self.sql('rule_triggers_del_all', rule)
        for i in range(0, len(triggers)):
            self.sql('rule_triggers_ins', rule, i, triggers[i])

    def rule_prereqs(self, rule):
        rule = self.json_dump(rule)
        for row in self.sql('rule_prereqs', rule):
            yield row[0]

    def insert_rule_prereq(self, rule, i, prereq):
        rule = self.json_dump(rule)
        self.sql('rule_prereqs_inc', rule, i)
        self.sql('rule_prereqs_ins', rule, i, prereq)

    def append_rule_prereq(self, rule, prereq):
        rule = self.json_dump(rule)
        ct = self.sql('rule_prereqs_count', rule).fetchone()[0]
        self.sql('rule_prereqs_ins', rule, ct, prereq)

    def delete_rule_prereq(self, rule, i):
        rule = self.json_dump(rule)
        self.sql('rule_prereqs_del', rule, i)
        self.sql('rule_prereqs_dec', rule, i)

    def replace_rule_prereq(self, rule, i, prereq):
        rule = self.json_dump(rule)
        self.sql('rule_prereqs_del', rule, i)
        self.sql('rule_prereqs_ins', rule, i, prereq)

    def clear_rule_prereqs(self, rule):
        self.sql('rule_prereqs_del_all', self.json_dump(rule))

    def replace_all_rule_prereqs(self, rule, prereqs):
        rule = self.json_dump(rule)
        self.sql('rule_prereqs_del_all', rule)
        for i in range(0, len(prereqs)):
            self.sql('rule_prereqs_ins', rule, i, prereqs[i])

    def rule_actions(self, rule):
        rule = self.json_dump(rule)
        for row in self.sql('rule_actions', rule):
            yield row[0]

    def insert_rule_action(self, rule, i, action):
        rule = self.json_dump(rule)
        self.sql('rule_actions_inc', rule, i)
        self.sql('rule_actions_ins', rule, i, action)

    def delete_rule_action(self, rule, i):
        rule = self.json_dump(rule)
        self.sql('rule_actions_del', rule, i)
        self.sql('rule_actions_dec', rule, i)

    def append_rule_action(self, rule, action):
        rule = self.json_dump(rule)
        ct = self.sql('rule_actions_count', rule).fetchone()[0]
        self.sql('rule_actions_ins', rule, ct, action)

    def replace_rule_action(self, rule, i, action):
        rule = self.json_dump(rule)
        self.sql('rule_actions_del', rule, i)
        self.sql('rule_actions_ins', rule, i, action)

    def clear_rule_actions(self, rule):
        self.sql('rule_actions_del_all', self.json_dump(rule))

    def replace_all_rule_actions(self, rule, actions):
        rule = self.json_dump(rule)
        self.sql('rule_actions_del_all', rule)
        for i in range(0, len(actions)):
            self.sql('rule_actions_ins', rule, i, actions[i])

    def universal_items(self, branch, tick):
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (k, v) in self.sql('universal_items', branch, tick):
                if k not in seen and v is not None:
                    yield (self.json_load(k), self.json_load(v))
                seen.add(k)

    def dump_universal(self):
        for key, branch, tick, value in self.sql('dump_universal'):
            yield self.json_load(key), branch, tick, self.json_load(value)

    def universal_get(self, key, branch, tick):
        key = self.json_dump(key)
        for (b, t) in self.active_branches(branch, tick):
            for (v,) in self.sql('universal_get', key, b, t):
                if v is None:
                    raise KeyError("Key not set")
                return self.json_load(v)
        raise KeyError("Key never set")

    def universal_set(self, key, branch, tick, value):
        (key, value) = map(self.json_dump, (key, value))
        try:
            return self.sql('universal_ins', key, branch, tick, value)
        except IntegrityError:
            return self.sql('universal_upd', value, key, branch, tick)

    def universal_del(self, key, branch, tick):
        key = self.json_dump(key)
        try:
            return self.sql('universal_ins', key, branch, tick, None)
        except IntegrityError:
            return self.sql('universal_upd', None, key, branch, tick)

    def characters(self):
        for (ch,) in self.sql('characters'):
            yield self.json_load(ch)

    def characters_rulebooks(self):
        for row in self.sql('characters_rulebooks'):
            yield map(self.json_load, row)

    def ct_characters(self):
        return self.sql('ct_characters').fetchone()[0]

    def have_character(self, name):
        name = self.json_dump(name)
        return self.sql('ct_character', name).fetchone()[0] > 0

    def del_character(self, name):
        name = self.json_dump(name)
        self.sql('del_char_things', name)
        self.sql('del_char_avatars', name)
        for tbl in (
                "node_val",
                "edge_val",
                "edges",
                "nodes",
                "graph_val",
                "characters",
                "graph"
        ):
            self.sql('char_del_fmt', name, tbl=tbl)

    def rulebooks(self):
        for book in self.sql('rulebooks'):
            yield self.json_load(book)

    def ct_rulebooks(self):
        return self.sql('ct_rulebooks').fetchone()[0]

    def node_rulebook(self, character, node):
        (character, node) = map(self.json_dump, (character, node))
        r = self.sql('node_rulebook', character, node).fetchone()
        if r is None:
            raise KeyError(
                'No rulebook for node {} in character {}'.format(
                    node, character
                )
            )
        return self.json_load(r[0])

    def nodes_rulebooks(self):
        for row in self.sql('nodes_rulebooks'):
            yield map(self.json_load, row)

    def set_node_rulebook(self, character, node, rulebook):
        (character, node, rulebook) = map(
            self.json_dump, (character, node, rulebook)
        )
        try:
            return self.sql('ins_node_rulebook', character, node, rulebook)
        except IntegrityError:
            return self.sql('upd_node_rulebook', rulebook, character, node)

    def portal_rulebook(self, character, nodeA, nodeB):
        (character, nodeA, nodeB) = map(
            self.json_dump, (character, nodeA, nodeB)
        )
        r = self.sql(
            'portal_rulebook',
            character,
            nodeA,
            nodeB,
            0
        ).fetchone()
        if r is None:
            raise KeyError(
                "No rulebook for portal {}->{} in character {}".format(
                    nodeA, nodeB, character
                )
            )
        return self.json_load(r[0])

    def portals_rulebooks(self):
        for row in self.sql('portals_rulebooks'):
            yield map(self.json_load, row)

    def set_portal_rulebook(self, character, nodeA, nodeB, rulebook):
        (character, nodeA, nodeB, rulebook) = map(
            self.json_dump, (character, nodeA, nodeB, rulebook)
        )
        try:
            return self.sql(
                'ins_portal_rulebook',
                character,
                nodeA,
                nodeB,
                0,
                rulebook
            )
        except IntegrityError:
            return self.sql(
                'upd_portal_rulebook',
                rulebook,
                character,
                nodeA,
                nodeB,
                0
            )

    def dump_active_rules(self):
        for (
                rulebook,
                rule,
                branch,
                tick,
                active
        ) in self.sql('dump_active_rules'):
            yield (
                self.json_load(rulebook),
                self.json_load(rule),
                branch,
                tick,
                bool(active)
            )

    def character_rulebook(self, character):
        character = self.json_dump(character)
        for (rb,) in self.sql('character_rulebook', character):
            return self.json_load(rb)

    def set_rule_activeness(self, rulebook, rule, branch, tick, active):
        (rulebook, rule) = map(self.json_dump, (rulebook, rule))
        try:
            self.sql(
                'active_rules_ins', rulebook, rule, branch, tick, active
            )
        except IntegrityError:
            self.sql(
                'active_rules_upd', active, rulebook, rule, branch, tick
            )

    def dump_character_rules_handled(self):
        for (
            character,
            rulebook,
            rule,
            branch,
            tick
        ) in self.sql('dump_character_rules_handled'):
            yield (
                self.json_load(character),
                self.json_load(rulebook),
                self.json_load(rule),
                branch,
                tick
            )

    def dump_avatar_rules_handled(self):
        for (
            character,
            rulebook,
            rule,
            graph,
            avatar,
            branch,
            tick
        ) in self.sql('dump_avatar_rules_handled'):
            yield (
                self.json_load(character),
                self.json_load(rulebook),
                self.json_load(rule),
                self.json_load(graph),
                self.json_load(avatar),
                branch,
                tick
            )

    def dump_character_thing_rules_handled(self):
        for (
            character,
            rulebook,
            rule,
            thing,
            branch,
            tick
        ) in self.sql('dump_character_thing_rules_handled'):
            yield (
                self.json_load(character),
                self.json_load(rulebook),
                self.json_load(rule),
                self.json_load(thing),
                branch,
                tick
            )

    def dump_character_place_rules_handled(self):
        for (
            character,
            rulebook,
            rule,
            place,
            branch,
            tick
        ) in self.sql('dump_character_place_rules_handled'):
            yield (
                self.json_load(character),
                self.json_load(rulebook),
                self.json_load(rule),
                self.json_load(place),
                branch,
                tick
            )

    def dump_character_node_rules_handled(self):
        for (
            character,
            rulebook,
            rule,
            node,
            branch,
            tick
        ) in self.sql('dump_character_node_rules_handled'):
            yield (
                self.json_load(character),
                self.json_load(rulebook),
                self.json_load(rule),
                self.json_load(node),
                branch,
                tick
            )

    def dump_character_portal_rules_handled(self):
        for (
            character,
            rulebook,
            rule,
            nodeA,
            nodeB,
            idx,
            branch,
            tick
        ) in self.sql('dump_character_portal_rules_handled'):
            yield (
                self.json_load(character),
                self.json_load(rulebook),
                self.json_load(rule),
                self.json_load(nodeA),
                self.json_load(nodeB),
                idx,
                branch,
                tick
            )

    def dump_node_rules_handled(self):
        for (
                character,
                node,
                rulebook,
                rule,
                branch,
                tick
        ) in self.sql("dump_node_rules_handled"):
            yield (
                self.json_load(character),
                self.json_load(node),
                self.json_load(rulebook),
                self.json_load(rule),
                branch,
                tick
            )

    def dump_portal_rules_handled(self):
        for (
                character,
                nodeA,
                nodeB,
                idx,
                rulebook,
                rule,
                branch,
                tick
        ) in self.sql('dump_portal_rules_handled'):
            yield (
                self.json_load(character),
                self.json_load(nodeA),
                self.json_load(nodeB),
                idx,
                self.json_load(rulebook),
                self.json_load(rule),
                branch,
                tick
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
            self, character, nodeA, nodeB, rulebook, rule, branch, tick
    ):
        (character, nodeA, nodeB, rulebook, rule) = map(
            self.json_dump, (character, nodeA, nodeB, rulebook, rule)
        )
        try:
            return self.sql(
                'handled_portal_rule',
                character,
                nodeA,
                nodeB,
                0,
                rulebook,
                rule,
                branch,
                tick
            )
        except IntegrityError:
            raise RedundantRuleError(
                "Already handled rule {rule} in rulebook {book} "
                "for portal from {nodeA} to {nodeB} "
                "at tick {tick} of branch {branch}".format(
                    nodeA=nodeA,
                    nodeB=nodeB,
                    book=rulebook,
                    rule=rule,
                    branch=branch,
                    tick=tick
                )
            )

    def dump_things(self):
        for (
                character, thing, branch, tick, loc, nextloc
        ) in self.sql('dump_things'):
            yield (
                self.json_load(character),
                self.json_load(thing),
                branch,
                tick,
                self.json_load(loc),
                self.json_load(nextloc) if nextloc else None
            )

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
                'thing_loc_and_next_ins',
                character,
                thing,
                branch,
                tick,
                loc,
                nextloc
            )
        except IntegrityError:
            return self.sql(
                'thing_loc_and_next_upd',
                loc,
                nextloc,
                character,
                thing,
                branch,
                tick
            )

    def node_stats_branch(self, character, node, branch):
        (character, node) = map(self.json_dump, (character, node))
        for (key, tick, value) in self.sql(
                'node_var_data_branch', character, node, branch
        ):
            yield (
                self.json_load(key),
                tick,
                self.json_load(value)
            )

    def sense_fun_set(self, character, sense, branch, tick, funn, active):
        character = self.json_dump(character)
        try:
            self.sql(
                'sense_fun_ins', character, sense, branch, tick, funn, active
            )
        except IntegrityError:
            self.sql(
                'sense_fun_upd', funn, active, character, sense, branch, tick
            )

    def sense_set(self, character, sense, branch, tick, active):
        character = self.json_dump(character)
        try:
            self.sql('sense_ins', character, sense, branch, tick, active)
        except IntegrityError:
            self.sql('sense_upd', active, character, sense, branch, tick)

    def init_character(
            self, character, character_rulebook=None, avatar_rulebook=None,
            thing_rulebook=None, place_rulebook=None, node_rulebook=None,
            portal_rulebook=None
    ):
        character_rulebook = character_rulebook or (character, 'character')
        avatar_rulebook = avatar_rulebook or (character, 'avatar')
        thing_rulebook = thing_rulebook or (character, 'character_thing')
        place_rulebook = place_rulebook or (character, 'character_place')
        portal_rulebook = portal_rulebook or (character, 'character_portal')
        (character, character_rulebook, avatar_rulebook, thing_rulebook,
         place_rulebook, node_rulebook, portal_rulebook) = map(
            self.json_dump,
            (character, character_rulebook, avatar_rulebook, thing_rulebook,
             place_rulebook, node_rulebook, portal_rulebook)
        )
        try:
            return self.sql(
                'character_ins',
                character,
                character_rulebook,
                avatar_rulebook,
                thing_rulebook,
                place_rulebook,
                portal_rulebook
            )
        except IntegrityError:
            pass

    def avatars_ever(self, character):
        character = self.json_dump(character)
        for (g, n, b, t, a) in self.sql('avatars_ever', character):
            yield (self.json_load(g), self.json_load(n), b, t, a)

    def dump_avatars(self):
        for (
                character,
                graph,
                node,
                branch,
                tick,
                is_avatar
        ) in self.sql('dump_avatars'):
            yield (
                self.json_load(character),
                self.json_load(graph),
                self.json_load(node),
                branch,
                tick,
                bool(is_avatar)
            )

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

    def rulebook_ins(self, rulebook, idx, rule):
        (rulebook, rule) = map(self.json_dump, (rulebook, rule))
        self.sql('rulebook_inc', rulebook, idx)
        try:
            return self.sql('rulebook_ins', rulebook, idx, rule)
        except IntegrityError:
            return self.sql('rulebook_upd', rule, rulebook, idx)

    def rulebook_set(self, rulebook, idx, rule):
        (rulebook, rule) = map(self.json_dump, (rulebook, rule))
        try:
            return self.sql('rulebook_ins', rulebook, idx, rule)
        except IntegrityError:
            return self.sql('rulebook_upd', rule, rulebook, idx)

    def rulebook_decr(self, rulebook, idx):
        self.sql('rulebook_dec', self.json_dump(rulebook), idx)

    def rulebook_del(self, rulebook, idx):
        rulebook = self.json_dump(rulebook)
        self.sql('rulebook_del', rulebook, idx)
        self.sql('rulebook_dec', rulebook, idx)

    def rulebook_rules(self, rulebook):
        for (rule,) in self.sql('rulebook_rules', self.json_dump(rulebook)):
            yield self.json_load(rule)

    def rulebooks_rules(self):
        for (rulebook, rule) in self.sql('rulebooks_rules'):
            yield map(self.json_load, (rulebook, rule))

    def ct_rulebook_rules(self, rulebook):
        return self.sql(
            'ct_rulebook_rules', self.json_dump(rulebook)
        ).fetchone()[0]

    def rulebook_get(self, rulebook, idx):
        return self.json_load(
            self.sql(
                'rulebook_get', self.json_dump(rulebook), idx
            ).fetchone()[0]
        )

    def allrules(self):
        for (rule,) in self.sql('allrules'):
            yield self.json_load(rule)

    def ctrules(self):
        return self.sql('ctrules').fetchone()[0]

    def ruledel(self, rule):
        self.sql('ruledel', self.json_dump(rule))

    def haverule(self, rule):
        for r in self.sql('haverule', self.json_dump(rule)):
            return True
        return False

    def ruleins(self, rule):
        self.sql('ruleins', self.json_dump(rule), '[]', '[]', '[]')

    def thing_locs_branch_data(self, character, thing, branch):
        (character, thing) = map(self.json_dump, (character, thing))
        for (tick, loc, nextloc) in self.sql(
                'thing_locs_branch_data', character, thing, branch
        ):
            yield (tick, self.json_load(loc), self.json_load(nextloc))

    def char_stat_branch_data(self, character, branch):
        character = self.json_dump(character)
        for (key, tick, value) in self.sql(
                'char_stat_branch_data', character, branch
        ):
            yield (self.json_load(key), tick, self.json_load(value))

    def node_stat_branch_data(self, character, node, branch):
        (character, node) = map(self.json_dump, (character, node))
        for (key, tick, value) in self.sql(
                'node_stat_branch_data', character, node, branch
        ):
            yield (self.json_load(key), tick, self.json_load(value))

    def edge_stat_branch_data(self, character, orig, dest, branch):
        (character, orig, dest) = map(self.json_dump, (character, orig, dest))
        for (key, tick, value) in self.sql(
                'edge_stat_branch_data', character, orig, dest, branch
        ):
            yield (self.json_load(key), tick, self.json_load(value))

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
            'universal',
            'rules',
            'rulebooks',
            'active_rules',
            'characters',
            'senses',
            'things',
            'node_rulebook',
            'portal_rulebook',
            'avatars',
            'character_rules_handled',
            'avatar_rules_handled',
            'character_thing_rules_handled',
            'character_place_rules_handled',
            'character_portal_rules_handled',
            'thing_rules_handled',
            'place_rules_handled',
            'portal_rules_handled',
            'rule_triggers',
            'rule_prereqs',
            'rule_actions'
        ):
            self.init_table(table)
        try:
            self.sql('view_node_rules_handled')
        except OperationalError:
            pass
        for idx in (
            'active_rules',
            'senses',
            'things',
            'avatars',
            'character_rules_handled',
            'avatar_rules_handled',
            'character_thing_rules_handled',
            'character_place_rules_handled',
            'character_portal_rules_handled',
            'thing_rules_handled',
            'place_rules_handled',
            'portal_rules_handled'
        ):
            self.index_table(idx)
