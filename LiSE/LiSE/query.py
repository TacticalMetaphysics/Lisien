# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
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
"""The query engine provides Pythonic methods to access the database.

This module also contains a notably unfinished implementation of a query
language specific to LiSE. Access some stats using entities' method
``historical``, and do comparisons on those, and instead of a boolean
result you'll get a callable object that will return an iterator over
turn numbers in which the comparison evaluated to ``True``.

"""
from operator import gt, lt, eq, ne, le, ge
from functools import partialmethod
from time import monotonic

from .allegedb import query

from .exc import (
    IntegrityError,
    OperationalError
)
from .util import EntityStatAccessor
import LiSE


def windows_union(windows):
    """Given a list of (beginning, ending), return a minimal version that contains the same ranges.

    :rtype: list

    """
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
        return windows
    none_left = []
    none_right = []
    otherwise = []
    for window in windows:
        if window[0] is None:
            none_left.append(window)
        elif window[1] is None:
            none_right.append(window)
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
    for window in none_right:
        if not res:
            res.append(window)
            continue
        res.extend(fix_overlap(res.pop(), window))
    return res


def windows_intersection(windows):
    """Given a list of (beginning, ending), return another describing where they overlap.

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
    right_none = []
    otherwise = []
    for window in windows:
        assert window is not None, None
        if window[0] is None:
            left_none.append(window)
        elif window[1] is None:
            right_none.append(window)
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

    def iter_turns(self):
        raise NotImplementedError

    def iter_ticks(self, turn):
        raise NotImplementedError

    def __eq__(self, other):
        return EqQuery(self.engine, self, self.engine._entityfy(other))

    def __gt__(self, other):
        return GtQuery(self.engine, self, self.engine._entityfy(other))

    def __ge__(self, other):
        return GeQuery(self.engine, self, self.engine._entityfy(other))

    def __lt__(self, other):
        return LtQuery(self.engine, self, self.engine._entityfy(other))

    def __le__(self, other):
        return LeQuery(self.engine, self, self.engine._entityfy(other))

    def __ne__(self, other):
        return NeQuery(self.engine, self, self.engine._entityfy(other))

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

    def iter_turns(self):
        return slow_iter_turns_eval_cmp(self, self.oper, engine=self.engine)


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


def slow_iter_turns_eval_cmp(qry, oper, start_branch=None, engine=None):
    """Iterate over all turns on which a comparison holds.

    This is expensive. It evaluates the query for every turn in history.

    """
    def mungeside(side):
        if isinstance(side, Query):
            return side.iter_turns
        elif isinstance(side, StatusAlias):
            return EntityStatAccessor(
                side.entity, side.stat, side.engine,
                side.branch, side.turn, side.tick, side.current, side.mungers
            )
        elif isinstance(side, EntityStatAccessor):
            return side
        else:
            return lambda: side
    leftside = mungeside(qry.leftside)
    rightside = mungeside(qry.rightside)
    engine = engine or leftside.engine or rightside.engine

    for (branch, _, _) in engine._iter_parent_btt(start_branch or engine.branch):
        if branch is None:
            return
        parent, turn_start, tick_start, turn_end, tick_end = engine._branches[branch]
        for turn in range(turn_start, engine.turn + 1):
            if oper(leftside(branch, turn), rightside(branch, turn)):
                yield branch, turn


class QueryEngine(query.QueryEngine):
    exist_edge_t = 0
    path = LiSE.__path__[0]
    IntegrityError = IntegrityError
    OperationalError = OperationalError

    def universals_dump(self):
        unpack = self.unpack
        for key, branch, turn, tick, value in self.sql('universals_dump'):
            yield unpack(key), branch, turn, tick, unpack(value)

    def rulebooks_dump(self):
        unpack = self.unpack
        for rulebook, branch, turn, tick, rules in self.sql('rulebooks_dump'):
            yield unpack(rulebook), branch, turn, tick, unpack(rules)

    def _rule_dump(self, typ):
        unpack = self.unpack
        for rule, branch, turn, tick, lst in self.sql('rule_{}_dump'.format(typ)):
            yield rule, branch, turn, tick, unpack(lst)

    def rule_triggers_dump(self):
        return self._rule_dump('triggers')

    def rule_prereqs_dump(self):
        return self._rule_dump('prereqs')

    def rule_actions_dump(self):
        return self._rule_dump('actions')

    def characters_dump(self):
        unpack = self.unpack
        for graph, typ in self.sql('graphs_dump'):
            if typ == 'DiGraph':
                yield unpack(graph)
    characters = characters_dump

    def node_rulebook_dump(self):
        unpack = self.unpack
        for character, node, branch, turn, tick, rulebook in self.sql('node_rulebook_dump'):
            yield unpack(character), unpack(node), branch, turn, tick, unpack(rulebook)

    def portal_rulebook_dump(self):
        unpack = self.unpack
        for character, orig, dest, branch, turn, tick, rulebook in self.sql('portal_rulebook_dump'):
            yield (
                unpack(character), unpack(orig), unpack(dest),
                branch, turn, tick, unpack(rulebook)
            )

    def _charactery_rulebook_dump(self, qry):
        unpack = self.unpack
        for character, branch, turn, tick, rulebook in self.sql(qry+'_rulebook_dump'):
            yield unpack(character), branch, turn, tick, unpack(rulebook)

    character_rulebook_dump = partialmethod(_charactery_rulebook_dump, 'character')
    avatar_rulebook_dump = partialmethod(_charactery_rulebook_dump, 'avatar')
    character_thing_rulebook_dump = partialmethod(_charactery_rulebook_dump, 'character_thing')
    character_place_rulebook_dump = partialmethod(_charactery_rulebook_dump, 'character_place')
    character_portal_rulebook_dump = partialmethod(_charactery_rulebook_dump, 'character_portal')

    def character_rules_handled_dump(self):
        unpack = self.unpack
        for character, rulebook, rule, branch, turn, tick in self.sql('character_rules_handled_dump'):
            yield unpack(character), unpack(rulebook), rule, branch, turn, tick

    def character_rules_changes_dump(self):
        unpack = self.unpack
        for (
                character, rulebook, rule, branch, turn, tick, handled_branch, handled_turn
        ) in self.sql('character_rules_changes_dump'):
            yield (
                unpack(character), unpack(rulebook),
                rule, branch, turn, tick, handled_branch, handled_turn
            )

    def avatar_rules_handled_dump(self):
        unpack = self.unpack
        for character, rulebook, rule, graph, avatar, branch, turn, tick in self.sql('avatar_rules_handled_dump'):
            yield (
                unpack(character), unpack(rulebook), rule,
                unpack(graph), unpack(avatar), branch, turn, tick
            )

    def avatar_rules_changes_dump(self):
        jl = self.unpack
        for (
            character, rulebook, rule, graph, avatar, branch, turn, tick, handled_branch, handled_turn
        ) in self.sql('avatar_rules_changes_dump'):
            yield (
                jl(character), jl(rulebook), rule, jl(graph), jl(avatar),
                branch, turn, tick, handled_branch, handled_turn
            )

    def character_thing_rules_handled_dump(self):
        unpack = self.unpack
        for character, rulebook, rule, thing, branch, turn, tick in self.sql('character_thing_rules_handled_dump'):
            yield unpack(character), unpack(rulebook), rule, unpack(thing), branch, turn, tick

    def character_thing_rules_changes_dump(self):
        jl = self.unpack
        for (
            character, rulebook, rule, thing, branch, turn, tick, handled_branch, handled_turn
        ) in self.sql('character_thing_rules_changes_dump'):
            yield (
                jl(character), jl(rulebook), rule, jl(thing),
                branch, turn, tick, handled_branch, handled_turn
            )

    def character_place_rules_handled_dump(self):
        unpack = self.unpack
        for character, rulebook, rule, place, branch, turn, tick in self.sql('character_place_rules_handled_dump'):
            yield unpack(character), unpack(rulebook), rule, unpack(place), branch, turn, tick

    def character_place_rules_changes_dump(self):
        jl = self.unpack
        for (
            character, rulebook, rule, place, branch, turn, tick, handled_branch, handled_turn
        ) in self.sql('character_place_rules_changes_dump'):
            yield (
                jl(character), jl(rulebook), rule, jl(place),
                branch, turn, tick, handled_branch, handled_turn
            )

    def character_portal_rules_handled_dump(self):
        unpack = self.unpack
        for character, rulebook, rule, orig, dest, branch, turn, tick in self.sql('character_portal_rules_handled_dump'):
            yield (
                unpack(character), unpack(rulebook), rule, unpack(orig), unpack(dest),
                branch, turn, tick
            )

    def character_portal_rules_changes_dump(self):
        jl = self.unpack
        for (
            character, rulebook, rule, orig, dest, branch, turn, tick, handled_branch, handled_turn
        ) in self.sql('character_portal_rules_changes_dump'):
            yield (
                jl(character), jl(rulebook), rule, jl(orig), jl(dest),
                branch, turn, tick, handled_branch, handled_turn
            )

    def node_rules_handled_dump(self):
        for character, node, rulebook, rule, branch, turn, tick in self.sql('node_rules_handled_dump'):
            yield self.unpack(character), self.unpack(node), self.unpack(rulebook), rule, branch, turn, tick

    def node_rules_changes_dump(self):
        jl = self.unpack
        for (
                character, node, rulebook, rule, branch, turn, tick, handled_branch, handled_turn
        ) in self.sql('node_rules_changes_dump'):
            yield (
                jl(character), jl(node), jl(rulebook), rule,
                branch, turn, tick, handled_branch, handled_turn
            )

    def portal_rules_handled_dump(self):
        unpack = self.unpack
        for character, orig, dest, rulebook, rule, branch, turn, tick in self.sql('portal_rules_handled_dump'):
            yield (
                unpack(character), unpack(orig), unpack(dest),
                unpack(rulebook), rule, branch, turn, tick
            )

    def portal_rules_changes_dump(self):
        jl = self.unpack
        for (
            character, orig, dest, rulebook, rule, branch, turn, tick, handled_branch, handled_turn
        ) in self.sql('portal_rules_changes_dump'):
            yield (
                jl(character), jl(orig), jl(dest), jl(rulebook), rule,
                branch, turn, tick, handled_branch, handled_turn
            )

    def senses_dump(self):
        unpack = self.unpack
        for character, sense, branch, turn, tick, function in self.sql('senses_dump'):
            yield unpack(character), sense, branch, turn, tick, function

    def things_dump(self):
        unpack = self.unpack
        for character, thing, branch, turn, tick, location in self.sql('things_dump'):
            yield (
                unpack(character), unpack(thing), branch, turn, tick,
                unpack(location)
            )

    def avatars_dump(self):
        unpack = self.unpack
        for character_graph, avatar_graph, avatar_node, branch, turn, tick, is_av in self.sql('avatars_dump'):
            yield (
                unpack(character_graph), unpack(avatar_graph),
                unpack(avatar_node), branch, turn, tick, is_av
            )

    def universal_set(self, key, branch, turn, tick, val):
        key, val = map(self.pack, (key, val))
        self.sql('universals_insert', key, branch, turn, tick, val)

    def universal_del(self, key, branch, turn, tick):
        key = self.pack(key)
        self.sql('universals_insert', key, branch, turn, tick, None)

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

    def rules_dump(self):
        for (name,) in self.sql('rules_dump'):
            yield name

    def _set_rule_something(self, what, rule, branch, turn, tick, flist):
        flist = self.pack(flist)
        return self.sql('rule_{}_insert'.format(what), rule, branch, turn, tick, flist)

    set_rule_triggers = partialmethod(_set_rule_something, 'triggers')
    set_rule_prereqs = partialmethod(_set_rule_something, 'prereqs')
    set_rule_actions = partialmethod(_set_rule_something, 'actions')

    def set_rule(self, rule, branch, turn, tick, triggers=None, prereqs=None, actions=None):
        self.sql('rules_insert', rule)
        self.set_rule_triggers(rule, branch, turn, tick, triggers or [])
        self.set_rule_prereqs(rule, branch, turn, tick, prereqs or [])
        self.set_rule_actions(rule, branch, turn, tick, actions or [])

    def set_rulebook(self, name, branch, turn, tick, rules=None):
        name, rules = map(self.pack, (name, rules or []))
        self.sql('rulebooks_insert', name, branch, turn, tick, rules)

    def _set_rulebook_on_character(self, rbtyp, char, branch, turn, tick, rb):
        char, rb = map(self.pack, (char, rb))
        self.sql(rbtyp + '_rulebook_insert', char, branch, turn, tick, rb)

    set_character_rulebook = partialmethod(_set_rulebook_on_character, 'character')
    set_avatar_rulebook = partialmethod(_set_rulebook_on_character, 'avatar')
    set_character_thing_rulebook = partialmethod(_set_rulebook_on_character, 'character_thing')
    set_character_place_rulebook = partialmethod(_set_rulebook_on_character, 'character_place')
    set_character_portal_rulebook = partialmethod(_set_rulebook_on_character, 'character_portal')

    def rulebooks(self):
        for book in self.sql('rulebooks'):
            yield self.unpack(book)

    def exist_node(self, character, node, branch, turn, tick, extant):
        super().exist_node(character, node, branch, turn, tick, extant)

    def exist_edge(self, character, orig, dest, idx, branch, turn, tick, extant=None):
        start = monotonic()
        if extant is None:
            branch, turn, tick, extant = idx, branch, turn, tick
            idx = 0
        super().exist_edge(character, orig, dest, idx, branch, turn, tick, extant)
        QueryEngine.exist_edge_t += monotonic() - start

    def set_node_rulebook(self, character, node, branch, turn, tick, rulebook):
        (character, node, rulebook) = map(
            self.pack, (character, node, rulebook)
        )
        return self.sql('node_rulebook_insert', character, node, branch, turn, tick, rulebook)

    def set_portal_rulebook(self, character, orig, dest, branch, turn, tick, rulebook):
        (character, orig, dest, rulebook) = map(
            self.pack, (character, orig, dest, rulebook)
        )
        return self.sql(
            'portal_rulebook_insert',
            character,
            orig,
            dest,
            branch,
            turn,
            tick,
            rulebook
        )

    def handled_character_rule(
            self, character, rulebook, rule, branch, turn, tick
    ):
        (character, rulebook) = map(
            self.pack, (character, rulebook)
        )
        return self.sql(
            'character_rules_handled_insert',
            character,
            rulebook,
            rule,
            branch,
            turn,
            tick,
        )

    def handled_avatar_rule(self, character,  rulebook, rule, graph, av, branch, turn, tick):
        character, graph, av, rulebook = map(
            self.pack, (character, graph, av, rulebook)
        )
        return self.sql(
            'avatar_rules_handled_insert',
            character,
            rulebook,
            rule,
            graph,
            av,
            branch,
            turn,
            tick
        )

    def handled_character_thing_rule(self, character, rulebook, rule, thing, branch, turn, tick):
        character, thing, rulebook = map(
            self.pack, (character, thing, rulebook)
        )
        return self.sql(
            'character_thing_rules_handled_insert',
            character,
            rulebook,
            rule,
            thing,
            branch,
            turn,
            tick
        )

    def handled_character_place_rule(self, character, rulebook, rule, place, branch, turn, tick):
        character, rulebook, place = map(
            self.pack, (character, rulebook, place)
        )
        return self.sql(
            'character_place_rules_handled_insert',
            character,
            rulebook,
            rule,
            place,
            branch,
            turn,
            tick
        )

    def handled_character_portal_rule(self, character, rulebook, rule, orig, dest, branch, turn, tick):
        character, rulebook, orig, dest = map(
            self.pack, (character, rulebook, orig, dest)
        )
        return self.sql(
            'character_portal_rules_handled_insert',
            character,
            rulebook,
            rule,
            orig,
            dest,
            branch,
            turn,
            tick
        )

    def handled_node_rule(
            self, character, node, rulebook, rule, branch, turn, tick
    ):
        (character, node, rulebook) = map(
            self.pack, (character, node, rulebook)
        )
        return self.sql(
            'node_rules_handled_insert',
            character,
            node,
            rulebook,
            rule,
            branch,
            turn,
            tick
        )

    def handled_portal_rule(
            self, character, orig, dest, rulebook, rule, branch, turn, tick
    ):
        (character, orig, dest, rulebook) = map(
            self.pack, (character, orig, dest, rulebook)
        )
        return self.sql(
            'portal_rules_handled_insert',
            character,
            orig,
            dest,
            rulebook,
            rule,
            branch,
            turn,
            tick
        )

    def get_rulebook_char(self, rulemap, character):
        character = self.pack(character)
        for (book,) in self.sql(
                'rulebook_get_{}'.format(rulemap), character
        ):
            return self.unpack(book)
        raise KeyError("No rulebook")

    def set_thing_loc(
            self, character, thing, branch, turn, tick, loc
    ):
        (character, thing) = map(
            self.pack,
            (character, thing)
        )
        loc = self.pack(loc)
        self.sql('del_things_after', character, thing, branch, turn, turn, tick)
        self.sql(
            'things_insert',
            character,
            thing,
            branch,
            turn,
            tick,
            loc
        )

    def avatar_set(self, character, graph, node, branch, turn, tick, isav):
        (character, graph, node) = map(
            self.pack, (character, graph, node)
        )
        self.sql(
            'del_avatars_after',
            character, graph, node, branch, turn, turn, tick
        )
        self.sql(
            'avatars_insert', character, graph, node, branch, turn, tick, isav
        )

    def rulebooks_rules(self):
        for (rulebook, rule) in self.sql('rulebooks_rules'):
            yield map(self.unpack, (rulebook, rule))

    def rulebook_get(self, rulebook, idx):
        return self.unpack(
            self.sql(
                'rulebook_get', self.pack(rulebook), idx
            ).fetchone()[0]
        )

    def rulebook_set(self, rulebook, branch, turn, tick, rules):
        # what if the rulebook has other values set afterward? wipe them out, right?
        # should that happen in the query engine or elsewhere?
        rulebook, rules = map(self.pack, (rulebook, rules))
        try:
            self.sql('rulebooks_insert', rulebook, branch, turn, tick, rules)
        except IntegrityError:
            self.sql('rulebooks_update', rules, rulebook, branch, turn, tick)

    def rulebook_del_time(self, branch, turn, tick):
        self.sql('rulebooks_del_time', branch, turn, tick)

    def branch_descendants(self, branch):
        for child in self.sql('branch_children', branch):
            yield child
            yield from self.branch_descendants(child)

    def turns_completed_dump(self):
        return self.sql('turns_completed_dump')

    def complete_turn(self, branch, turn):
        try:
            self.sql('turns_completed_insert', branch, turn)
        except IntegrityError:
            self.sql('turns_completed_update', turn, branch)
        self.sql('del_character_rules_handled_turn', branch, turn)
        self.sql('del_avatar_rules_handled_turn', branch, turn)
        self.sql('del_character_thing_rules_handled_turn', branch, turn)
        self.sql('del_character_place_rules_handled_turn', branch, turn)
        self.sql('del_character_portal_rules_handled_turn', branch, turn)

    def initdb(self):
        """Set up the database schema, both for allegedb and the special
        extensions for LiSE

        """
        super().initdb()
        init_table = self.init_table
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
            'rule_actions',
            'turns_completed'
        ):
            init_table(table)
