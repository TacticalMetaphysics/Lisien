# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""The query engine provides Pythonic methods to access the database."""
from inspect import getsource
from types import FunctionType
from marshal import loads as unmarshalled
from marshal import dumps as marshalled
from operator import gt, lt, eq, ne, le, ge

import gorm.query

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
                side.entity, side.stat, side.engine, side.branch, side.tick, side.current, side.mungers
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
                    'location', 'next_location', 'locations', 'arrival_time', 'next_arrival_time'
            ):
                return engine._things_cache.branches[(side.entity.character.name, side.entity.name)]
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


class QueryEngine(gorm.query.QueryEngine):
    json_path = LiSE.__path__[0]
    IntegrityError = IntegrityError
    OperationalError = OperationalError

    def comparison(self, entity0, stat0, entity1, stat1=None, oper='eq', windows=[]):
        engine = entity0.engine
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

    def init_string_table(self, tbl):
        self.init_table(tbl)
        if tbl in string_defaults:
            for (lang, defaults) in string_defaults[tbl].items():
                for (k, v) in defaults:
                    self.string_table_set(tbl, lang, k, v)

    def init_func_table(self, tbl):
        self.init_table(tbl)

    def func_table_iter(self, tbl):
        return self.sql('func_{}_iter'.format(tbl))

    def func_table_name_plaincode(self, tbl):
        return self.sql('func_{}_name_plaincode'.format(tbl))

    def func_table_contains(self, tbl, key):
        for row in self.sql('func_{}_get'.format(tbl), key):
            return True

    def func_table_get(self, tbl, key, use_globals=True):
        bytecode = self.sql('func_{}_get'.format(tbl), key).fetchone()
        if bytecode is None:
            raise KeyError("No such function")
        globd = (
            globals() if use_globals is True else
            use_globals if isinstance(use_globals, dict) else
            {}
        )
        return FunctionType(
            unmarshalled(bytecode[0]),
            globd
        )

    def func_table_get_plain(self, tbl, key):
        return self.func_table_get_all(tbl, key)['plaincode']

    def func_table_get_all(self, tbl, key):
        data = self.sql(
            'func_{}_get'.format(tbl), key
        ).fetchone()
        if not data:
            raise KeyError("No such function: " + key)
        (
            bytecode,
            base,
            keywords,
            date,
            creator,
            contributor,
            description,
            plaincode,
            version
        ) = data
        return {
            'bytecode': bytecode,
            'base': base,
            'keywords': self.json_load(keywords),
            'date': date,
            'creator': creator,
            'contributor': contributor,
            'description': description,
            'plaincode': plaincode,
            'version': version
        }

    def func_table_set(self, tbl, key, fun, keywords=[]):
        try:
            s = getsource(fun)
        except OSError:
            s = ''
        m = marshalled(fun.__code__)
        kws = self.json_dump(keywords)
        try:
            return self.sql('func_{}_ins'.format(tbl), key, kws, m, s)
        except IntegrityError:
            return self.sql('func_{}_upd'.format(tbl), kws, m, s, key)

    def func_table_set_source(
            self, tbl, key, source, keywords=[], use_globals=True
    ):
        locd = {}
        globd = (
            globals() if use_globals is True else
            use_globals if isinstance(use_globals, dict) else
            {}
        )
        try:
            exec(source, globd, locd)
        except SyntaxError:  # hack to allow 'empty' functions
            source += '\n    pass'
            exec(source, globd, locd)
        if len(locd) != 1:
            raise UserFunctionError(
                "Input code contains more than the one function definition."
            )
        if key not in locd:
            raise UserFunctionError(
                "Function in input code has different name ({}) "
                "than expected ({}).".format(
                    next(locd.keys()),
                    self.name
                )
            )
        fun = locd[key]
        m = marshalled(fun.__code__)
        kws = self.json_dump(keywords)
        try:
            return self.sql('func_{}_ins'.format(tbl), key, kws, m, source)
        except IntegrityError:
            return self.sql('func_{}_upd'.format(tbl), kws, m, source, key)

    def func_table_del(self, tbl, key):
        return self.sql('func_{}_del'.format(tbl), key)

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
        return []

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
        return []

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
        return []

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

    def travel_reqs(self, character):
        character = self.json_dump(character)
        for row in self.sql('travel_reqs', character):
            return self.json_load(row[0])
        return []

    def set_travel_reqs(self, character, reqs):
        (char, reqs) = map(self.json_dump, (character, reqs))
        try:
            return self.sql('ins_travel_reqs', char, reqs)
        except IntegrityError:
            return self.sql('upd_travel_reqs', reqs, char)

    def string_table_lang_items(self, tbl, lang):
        return self.sql('{}_lang_items'.format(tbl), lang)

    def string_table_get(self, tbl, lang, key):
        for row in self.sql('{}_get'.format(tbl), lang, key):
            return row[0]

    def string_table_set(self, tbl, lang, key, value):
        try:
            self.sql('{}_ins'.format(tbl), key, lang, value)
        except IntegrityError:
            self.sql('{}_upd'.format(tbl), value, lang, key)

    def string_table_del(self, tbl, lang, key):
        self.sql('{}_del'.format(tbl), lang, key)

    def universal_items(self, branch, tick):
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (k, v) in self.sql('universal_items', branch, tick):
                if k not in seen and v is not None:
                    yield (self.json_load(k), self.json_load(v))
                seen.add(k)

    def universal_dump(self):
        for key, branch, tick, value in self.sql('universal_dump'):
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

    def active_rules_rulebook(self, rulebook, branch, tick):
        rulebook = self.json_dump(rulebook)
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (rule, active) in self.sql(
                    'active_rules_rulebook', rulebook, branch, tick
            ):
                if active and rule not in seen:
                    yield self.json_load(rule)
                seen.add(rule)

    def active_rule_rulebook(self, rulebook, rule, branch, tick):
        (rulebook, rule) = map(self.json_dump, (rulebook, rule))
        for (b, t) in self.active_branches(branch, tick):
            for (active,) in self.sql(
                    'active_rule_rulebook', rulebook, rule, branch, tick
            ):
                return bool(active)
        return False

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
        (character, nodeA, nodeB) = map(self.json_dump, (character, nodeA, nodeB))
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

    def poll_char_rules(self, branch, tick):
        """Poll character-wide rules for all the entity types."""
        for rulemap in (
                'character',
                'avatar',
                'character_thing',
                'character_place',
                'character_node',
                'character_portal'
        ):
            seen = set()
            for (b, t) in self.active_branches(branch, tick):
                for (c, rulebook, rule, active, handled) in self.sql(
                        'poll_{}_rules'.format(rulemap), b, t, b, t
                ):
                    if (c, rulebook, rule) in seen:
                        continue
                    seen.add((c, rulebook, rule))
                    if active:
                        yield (rulemap,) + tuple(map(
                            self.json_load, (c, rulebook, rule)
                        ))

    def handled_rules_on_characters(self, typ):
        for (
                character,
                rulebook,
                rule,
                branch,
                tick
        ) in self.sql('handled_{}_rules'.format(typ)):
            yield (
                self.json_load(character),
                self.json_load(rulebook),
                self.json_load(rule),
                branch,
                tick
            )

    def handled_character_rules(self):
        return self.handled_rules_on_characters('character')

    def handled_avatar_rules(self):
        return self.handled_rules_on_characters('avatar')

    def handled_character_thing_rules(self):
        return self.handled_rules_on_characters('character_thing')

    def handled_character_place_rules(self):
        return self.handled_rules_on_characters('character_place')

    def handled_character_node_rules(self):
        return self.handled_rules_on_characters('character_node')

    def handled_character_portal_rules(self):
        return self.handled_rules_on_characters('character_portal')

    def poll_node_rules(self, branch, tick):
        """Poll rules assigned to particular Places or Things."""
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (char, n, rulebook, rule, active) in self.sql(
                    'poll_node_rules', b, t, b, t
            ):
                if (char, n, rulebook, rule) in seen:
                    continue
                seen.add((char, n, rulebook, rule))
                if active:
                    yield tuple(map(
                        self.json_load,
                        (char, n, rulebook, rule)
                    ))

    def node_rules(self, character, node, branch, tick):
        (character, node) = map(self.json_dump, (character, node))
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (char, n, rulebook, rule, active) in self.sql(
                'node_rules', b, t, b, t, character, node
            ):
                if (char, n, rulebook, rule) in seen:
                    continue
                seen.add((char, n, rulebook, rule))
                if active:
                    yield tuple(
                        map(self.json_load, (char, n, rulebook, rule))
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

    def poll_portal_rules(self, branch, tick):
        """Poll rules assigned to particular portals."""
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (char, a, b, i, rulebook, rule, active, handled) in self.sql(
                'poll_portal_rules', b, t, b, t
            ):
                if (char, a, b, i, rulebook, rule) in seen:
                    continue
                seen.add((char, a, b, i, rulebook, rule))
                if active:
                    yield (
                        self.json_load(char),
                        self.json_load(a),
                        self.json_load(b),
                        i,
                        self.json_load(rulebook),
                        self.json_load(rule)
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

    def portal_rules(self, character, nodeA, nodeB, branch, tick):
        (character, nodeA, nodeB) = map(self.json_dump, (character, nodeA, nodeB))
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (char, a, b, i, rulebook, rule, active, handled) in self.sql(
                'portal_rules', b, t, b, t, character, nodeA, nodeB, 0
            ):
                if (char, a, b, i, rulebook, rule) in seen:
                    continue
                seen.add((char, a, b, i, rulebook, rule))
                if active:
                    yield (
                        self.json_load(char),
                        self.json_load(a),
                        self.json_load(b),
                        i,
                        self.json_load(rulebook),
                        self.json_load(rule)
                    )

    def handled_character_rule(
            self, ruletyp, character, rulebook, rule, branch, tick
    ):
        (character, rulebook, rule) = map(self.json_dump, (character, rulebook, rule))
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

    def character_rule_handled(
            self, ruletyp, character, rule, branch, tick
    ):
        (character, rule) = map(self.json_dump, (character, rule))
        return bool(self.sql(
            {
                'character': 'character_rule_handled',
                'avatar': 'avatar_rule_handled',
                'node': 'character_node_rule_handled',
                'character_node': 'character_node_rule_handled',
                'thing': 'character_thing_rule_handled',
                'character_thing': 'character_thing_rule_handled',
                'place': 'character_place_rule_handled',
                'character_place': 'character_place_rule_handled',
                'portal': 'character_portal_rule_handled',
                'character_portal': 'character_portal_rule_handled'
            }[ruletyp],
            character,
            rule,
            branch,
            tick
        ))

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

    def node_is_thing(self, character, node, branch, tick):
        (character, node) = map(self.json_dump, (character, node))
        for (b, t) in self.active_branches(branch, tick):
            for (loc,) in self.sql(
                    'node_is_thing', character, node, branch, tick
            ):
                return bool(loc)
        return False

    def get_rulebook_char(self, rulemap, character):
        character = self.json_dump(character)
        for (book,) in self.sql(
                'rulebook_get_{}'.format(rulemap), character
        ):
            return self.json_load(book)
        raise KeyError("No rulebook")

    def upd_rulebook_char(self, rulemap, character):
        return self.sql('upd_rulebook_char_fmt', character, rulemap=rulemap)

    def avatar_users(self, graph, node, branch, tick):
        (graph, node) = map(self.json_dump, (graph, node))
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (av_g,) in self.sql('avatar_users', graph, node, b, t):
                if av_g not in seen:
                    yield self.json_load(av_g)

    def arrival_time_get(self, character, thing, location, branch, tick):
        (character, thing, location) = map(
            self.json_dump, (character, thing, location)
        )
        for (b, t) in self.active_branches(branch, tick):
            for hitick in self.sql(
                    'arrival_time_get',
                    character,
                    thing,
                    location,
                    b,
                    t
            ):
                return hitick
        raise ValueError("No arrival time recorded")

    def next_arrival_time_get(self, character, thing, location, branch, tick):
        (character, thing, location) = map(
            self.json_dump, (character, thing, location)
        )
        for (b, t) in self.active_branches(branch, tick):
            for (hitick,) in self.sql(
                    'next_arrival_time_get',
                    character,
                    thing,
                    location,
                    branch,
                    tick
            ):
                return hitick
        return None

    def thing_loc_and_next_get(self, character, thing, branch, tick):
        (character, thing) = map(self.json_dump, (character, thing))
        for (b, t) in self.active_branches(branch, tick):
            for (loc, nextloc) in self.sql(
                    'thing_loc_and_next_get', character, thing, b, t
            ):
                return (self.json_load(loc), self.json_load(nextloc))

    def things_dump(self):
        for (
                character, thing, branch, tick, loc, nextloc
        ) in self.sql('things_dump'):
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

    def thing_loc_items(self, character, branch, tick):
        character = self.json_dump(character)
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (n, l) in self.sql(
                    'thing_loc_items',
                    character,
                    b,
                    t,
                    t,
                    character,
                    b
            ):
                if l is not None and n not in seen:
                    yield (self.json_load(n), self.json_load(l))
                seen.add(n)

    def thing_and_loc(self, character, thing, branch, tick):
        (character, thing) = map(self.json_dump, (character, thing))
        for (b, t) in self.active_branches(branch, tick):
            for (th, l) in self.sql(
                    'thing_and_loc',
                    character,
                    thing,
                    b,
                    t,
                    t,
                    character,
                    thing,
                    b
            ):
                if l is None:
                    raise KeyError("Thing does not exist")
                return (self.json_load(th), self.json_load(l))
        raise KeyError("Thing never existed")

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

    def character_things_items(self, character, branch, tick):
        character = self.json_dump(character)
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (th, l) in self.sql(
                    'character_things_items', character, b, t
            ):
                if l is not None and th not in seen:
                    yield (self.json_load(th), self.json_load(l))
                seen.add(th)

    def avatarness(self, character, branch, tick):
        character = self.json_dump(character)
        d = {}
        for (b, t) in self.active_branches(branch, tick):
            for (graph, node, avatar) in self.sql(
                    'avatarness', character, b, t
            ):
                g = self.json_load(graph)
                n = self.json_load(node)
                is_av = bool(avatar)
                if g not in d:
                    d[g] = {}
                d[g][n] = is_av
        return d

    def is_avatar_of(self, character, graph, node, branch, tick):
        (character, graph, node) = map(self.json_dump, (character, graph, node))
        for (avatarness,) in self.sql(
                'is_avatar_of', character, graph, node, branch, tick
        ):
            return avatarness

    def sense_func_get(self, character, sense, branch, tick):
        character = self.json_dump(character)
        for (b, t) in self.active_branches(branch, tick):
            for (func,) in self.sql(
                    'sense_func_get', character, sense, branch, tick
            ):
                return func

    def sense_active_items(self, character, branch, tick):
        character = self.json_dump(character)
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (sense, active) in self.sql(
                    'sense_active_items', character, b, t
            ):
                if sense not in seen and active:
                    yield sense
                seen.add(sense)

    def sense_is_active(self, character, sense, branch, tick):
        character = self.json_dump(character)
        for (b, t) in self.active_branches(branch, tick):
            for (act,) in self.sql(
                    'sense_is_active', character, sense, branch, tick
            ):
                return bool(act)
        return False

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
        node_rulebook = node_rulebook or (character, 'character_node')
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
                node_rulebook,
                portal_rulebook
            )
        except IntegrityError:
            pass

    def avatars_now(self, character, branch, tick):
        character = self.json_dump(character)
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (g, n, a) in self.sql('avatars_now', character, b, t, b, t):
                if (g, n) not in seen:
                    yield (self.json_load(g), self.json_load(n), a)
                seen.add((g, n))

    def avatars_ever(self, character):
        character = self.json_dump(character)
        for (g, n, b, t, a) in self.sql('avatars_ever', character):
            yield (self.json_load(g), self.json_load(n), b, t, a)

    def avatarness_dump(self):
        for (
                character,
                graph,
                node,
                branch,
                tick,
                is_avatar
        ) in self.sql('avatarness_dump'):
            yield (
                self.json_load(character),
                self.json_load(graph),
                self.json_load(node),
                branch,
                tick,
                bool(is_avatar)
            )

    def avatar_set(self, character, graph, node, branch, tick, isav):
        (character, graph, node) = map(self.json_dump, (character, graph, node))
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

    def current_rules_character(self, character, branch, tick):
        for rule in self.sql(
                'current_rules_character',
                self.json_dump(character),
                branch,
                tick
        ):
            yield self.json_load(rule)

    def current_rules_avatar(self, character, branch, tick):
        for rule in self.sql(
            'current_rules_avatar',
            self.json_dump(character),
            branch,
            tick
        ):
            yield self.json_load(rule)

    def current_rules_character_thing(self, character, branch, tick):
        for rule in self.sql(
            'current_rules_character_thing',
            self.json_dump(character),
            branch,
            tick
        ):
            yield self.json_load(rule)

    def current_rules_character_place(self, character, branch, tick):
        for rule in self.sql(
            'current_rules_character_place',
            self.json_dump(character),
            branch,
            tick
        ):
            yield self.json_load(rule)

    def current_rules_character_node(self, character, branch, tick):
        for rule in self.sql(
            'current_rules_character_node',
            self.json_dump(character),
            branch,
            tick
        ):
            yield self.json_load(rule)

    def current_rules_character_portal(self, character, branch, tick):
        for rule in self.sql(
            'current_rules_character_portal',
            self.json_dump(character),
            branch,
            tick
        ):
            yield self.json_load(rule)

    def current_rules_node(self, character, node, branch, tick):
        (character, node) = map(self.json_dump, (character, node))
        for rule in self.sql(
            'current_rules_node',
            character,
            node,
            branch,
            tick
        ):
            yield self.json_load(rule)

    def current_rules_portal(self, character, nodeA, nodeB, branch, tick):
        (character, nodeA, nodeB) = map(self.json_dump, (character, nodeA, nodeB))
        for rule in self.sql(
            'current_rules_portal',
            character,
            nodeA,
            nodeB,
            branch,
            tick
        ):
            yield self.json_load(rule)

    def ct_rulebook_rules(self, rulebook):
        return self.sql('ct_rulebook_rules', self.json_dump(rulebook)).fetchone()[0]

    def rulebook_get(self, rulebook, idx):
        return self.json_load(
            self.sql('rulebook_get', self.json_dump(rulebook), idx).fetchone()[0]
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

    def avatar_branch_data(self, character, graph, branch, tick):
        (character, graph) = map(self.json_dump, (character, graph))
        for (node, isav) in self.sql(
                'avatar_branch_data', character, graph, branch, tick
        ):
            yield (self.json_load(node), bool(isav))

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
        """Set up the database schema, both for gorm and the special
        extensions for LiSE

        """
        super().initdb()
        for table in (
            'lise_globals',
            'rules',
            'rulebooks',
            'active_rules',
            'characters',
            'senses',
            'travel_reqs',
            'things',
            'node_rulebook',
            'portal_rulebook',
            'avatars',
            'character_rules_handled',
            'avatar_rules_handled',
            'character_thing_rules_handled',
            'character_place_rules_handled',
            'character_node_rules_handled',
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
            'travel_reqs',
            'things',
            'avatars',
            'character_rules_handled',
            'avatar_rules_handled',
            'character_thing_rules_handled',
            'character_place_rules_handled',
            'character_node_rules_handled',
            'character_portal_rules_handled',
            'thing_rules_handled',
            'place_rules_handled',
            'portal_rules_handled'
        ):
            self.index_table(idx)
