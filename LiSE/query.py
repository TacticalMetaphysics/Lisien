# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from inspect import getsource
from types import FunctionType
from marshal import loads as unmarshalled
from marshal import dumps as marshalled
from gorm.xjson import json_dump, json_load
from .util import IntegrityError, OperationalError, RedundantRuleError

import gorm.query
import LiSE


class QueryEngine(gorm.query.QueryEngine):
    json_path = LiSE.__path__[0]
    IntegrityError = IntegrityError
    OperationalError = OperationalError

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

    def func_table_set(self, tbl, key, fun):
        try:
            s = getsource(fun)
        except OSError:
            s = ''
        m = marshalled(fun.__code__)
        try:
            return self.sql('func_{}_ins'.format(tbl), key, m, s)
        except IntegrityError:
            return self.sql('func_{}_upd'.format(tbl), m, s, key)

    def func_table_set_source(self, tbl, key, source, use_globals=True):
        locd = {}
        globd = (
            globals() if incl_globals is True else
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
        try:
            return self.sql('func_{}_ins'.format(tbl), key, m, source)
        except IntegrityError:
            return self.sql('func_{}_upd'.format(tbl), m, source, key)

    def func_table_del(self, tbl, key):
        return self.sql('func_{}_del'.format(tbl), key)

    def rule_triggers(self, rule):
        for row in self.sql('rule_triggers', rule):
            return json_load(row[0])
        return []

    def set_rule_triggers(self, rule, triggers):
        triggers = json_dump(triggers)
        try:
            return self.sql('ins_rule', rule, triggers, '["list"]', '["list"]')
        except IntegrityError:
            return self.sql('upd_rule_triggers', triggers, rule)

    def rule_prereqs(self, rule):
        for row in self.sql('rule_prereqs', rule):
            return json_load(row[0])
        return []

    def set_rule_prereqs(self, rule, prereqs):
        prereqs = json_dump(prereqs)
        try:
            return self.sql('ins_rule', rule, '["list"]', prereqs, '["list"]')
        except IntegrityError:
            return self.sql('upd_rule_prereqs', prereqs, rule)

    def rule_actions(self, rule):
        for row in self.sql('rule_actions', rule):
            return json_load(row[0])
        return []

    def set_rule_actions(self, rule, actions):
        actions = json_dump(actions)
        try:
            return self.sql('ins_rule', rule, '["list"]', '["list"]', actions)
        except IntegrityError:
            return self.sql('upd_rule_actions', actions, rule)

    def travel_reqs(self, character):
        chararcter = json_dump(character)
        for row in self.sql('travel_reqs', character):
            return json_load(row[0])
        return []

    def set_travel_reqs(self, character, reqs):
        (char, reqs) = map(json_dump, (character, reqs))
        try:
            return self.sql('ins_travel_reqs', char, reqs)
        except IntegrityError:
            return self.sql('upd_travel_reqs', reqs, char)

    def init_string_table(self, tbl):
        try:
            return self.sql('create_{}'.format(tbl))
        except OperationalError:
            pass

    def string_table_lang_items(self, tbl, lang):
        return self.sql('string_{}_lang_items'.format(tbl), lang)

    def string_table_get(self, tbl, lang, key):
        for row in self.sql('string_get', lang, key, tbl=tbl):
            return row[0]

    def string_table_set(self, tbl, lang, key, value):
        try:
            self.sql('string_{}_ins'.format(tbl), key, lang, value)
        except IntegrityError:
            self.sql('string_{}_upd'.format(tbl), value, lang, key)

    def string_table_del(self, tbl, lang, key):
        self.sql('string_{}_del'.format(tbl), lang, key)

    def universal_items(self, branch, tick):
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (k, v) in self.sql('universal_items', branch, tick):
                if k not in seen and v is not None:
                    yield (json_load(k), json_load(v))
                seen.add(k)

    def universal_get(self, key, branch, tick):
        key = json_dump(key)
        for (b, t) in self.active_branches(branch, tick):
            for (v,) in self.sql('universal_get', key, b, t):
                if v is None:
                    raise KeyError("Key not set")
                return json_load(v)
        raise KeyError("Key never set")

    def universal_set(self, key, branch, tick, value):
        (key, value) = map(json_dump, (key, value))
        try:
            return self.sql('universal_ins', key, branch, tick, value)
        except IntegrityError:
            return self.sql('universal_upd', value, key, branch, tick)

    def universal_del(self, key, branch, tick):
        key = json_dump(key)
        try:
            return self.sql('universal_ins', key, branch, tick, None)
        except IntegrityError:
            return self.sql('universal_upd', None, key, branch, tick)

    def characters(self):
        for (ch,) in self.sql('characters'):
            yield json_load(ch)

    def ct_characters(self):
        return self.sql('ct_characters').fetchone()[0]

    def have_character(self, name):
        return bool(self.sql('ct_character', name))

    def del_character(self, name):
        name = json_dump(name)
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

    def active_rules_rulebook(self, rulebook, branch, tick):
        rulebook = json_dump(rulebook)
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (rule, active) in self.sql(
                    'active_rules_rulebook', rulebook, branch, tick
            ):
                if active and rule not in seen:
                    yield rule
                seen.add(rule)

    def active_rule_rulebook(self, rulebook, rule, branch, tick):
        rulebook = json_dump(rulebook)
        for (b, t) in self.active_branches(branch, tick):
            for (active,) in self.sql(
                    'active_rule_rulebook', rulebook, rule, branch, tick
            ):
                return bool(active)
        return False

    def node_rulebook(self, character, node):
        (character, node) = map(json_dump, (character, node))
        r = self.sql('node_rulebook', character, node).fetchone()
        if r is None:
            raise KeyError(
                'No rulebook for node {} in character {}'.format(
                    node, character
                )
            )
        return json_load(r[0])

    def set_node_rulebook(self, character, node, rulebook):
        (character, node, rulebook) = map(
            json_dump, (character, node, rulebook)
        )
        try:
            return self.sql('ins_node_rulebook', character, node, rulebook)
        except IntegrityError:
            return self.sql('upd_node_rulebook', rulebook, character, node)

    def portal_rulebook(self, character, nodeA, nodeB):
        (character, nodeA, nodeB) = map(json_dump, (character, nodeA, nodeB))
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
        return json_load(r[0])

    def set_portal_rulebook(self, character, nodeA, nodeB, rulebook):
        (character, nodeA, nodeB, rulebook) = map(
            json_dump, (character, nodeA, nodeB, rulebook)
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

    def active_rules_char(self, tbl, character, rulebook, branch, tick):
        character = json_dump(character)
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (rule, active) in self.sql(
                    'active_rules_char_fmt', character, rulebook, b, t, tbl=tbl
            ):
                if active and rule not in seen:
                    yield rule
                seen.add(rule)

    def active_rule_char(self, tbl, character, rulebook, rule, branch, tick):
        character = json_dump(character)
        for (b, t) in self.active_branches(branch, tick):
            for (active,) in self.sql(
                    'active_rule_char_fmt',
                    character,
                    rulebook,
                    rule,
                    b,
                    t,
                    tbl=tbl
            ):
                return bool(active)
        return False

    def character_rulebook(self, character):
        character = json_dump(character)
        for (rb,) in self.sql('character_rulebook', character):
            return rb

    def rule_set(self, rulebook, rule, branch, tick, active):
        rulebook = json_dump(rulebook)
        try:
            self.sql(
                'rule_ins', rulebook, rule, branch, tick, active
            )
        except IntegrityError:
            self.sql(
                'rule_upd', active, rulebook, rule, branch, tick
            )

    def poll_char_rules(self, branch, tick):
        """Poll character-wide rules for all the entity types."""
        for rulemap in (
                'character',
                'avatar',
                'character_thing',
                'character_place',
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
                        yield (rulemap, json_load(c), rulebook, rule)

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
                    yield (
                        json_load(char),
                        json_load(n),
                        rulebook,
                        rule
                    )

    def node_rules(self, character, node, branch, tick):
        (character, node) = map(json_dump, (character, node))
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (char, n, rulebook, rule, active) in self.sql(
                'node_rules', b, t, b, t, character, node
            ):
                if (char, n, rulebook, rule) in seen:
                    continue
                seen.add((char, n, rulebook, rule))
                if active:
                    yield(
                        json_load(char),
                        json_load(n),
                        rulebook,
                        rule
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
                        json_load(char),
                        json_load(a),
                        json_load(b),
                        i,
                        rulebook,
                        rule
                    )

    def portal_rules(self, character, nodeA, nodeB, branch, tick):
        (character, nodeA, nodeB) = map(json_dump, (character, nodeA, nodeB))
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
                        json_load(char),
                        json_load(a),
                        json_load(b),
                        i,
                        rulebook,
                        rule
                    )

    def handled_character_rule(
            self, ruletyp, character, rulebook, rule, branch, tick
    ):
        (character, rulebook) = map(json_dump, (character, rulebook))
        try:
            return self.sql(
                'handled_character_rule_fmt',
                character,
                rulebook,
                rule,
                branch,
                tick,
                ruletyp=ruletyp
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
        (character, thing, rulebook) = map(
            json_dump, (character, thing, rulebook)
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
        (character, place, rulebook) = map(json_dump, (character, rulebook))
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
        (character, nodeA, nodeB, rulebook) = map(
            json_dump, (character, nodeA, nodeB, rulebook)
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
        (character, node) = map(json_dump, (character, node))
        for (b, t) in self.active_branches(branch, tick):
            for (loc,) in self.sql(
                    'node_is_thing', character, node, branch, tick
            ):
                return bool(loc)
        return False

    def get_rulebook_char(self, rulemap, character):
        character = json_dump(character)
        for (book,) in self.sql(
                'rulebook_get_char_fmt', character, rulemap=rulemap
        ):
            return book
        raise KeyError("No rulebook")

    def upd_rulebook_char(self, rulemap, character):
        return self.sql('upd_rulebook_char_fmt', character, rulemap=rulemap)

    def avatar_users(self, graph, node, branch, tick):
        (graph, node) = map(json_dump, (graph, node))
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (av_g,) in self.sql('avatar_users', graph, node, b, t):
                if av_g not in seen:
                    yield json_load(av_g)

    def arrival_time_get(self, character, thing, location, branch, tick):
        (character, thing, location) = map(
            json_dump, (character, thing, location)
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
            json_dump, (character, thing, location)
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
        (character, thing) = map(json_dump, (character, thing))
        for (b, t) in self.active_branches(branch, tick):
            for (loc, nextloc) in self.sql(
                    'thing_loc_and_next_get', character, thing, b, t
            ):
                return (json_load(loc), json_load(nextloc))

    def thing_loc_and_next_set(
            self, character, thing, branch, tick, loc, nextloc
    ):
        (character, thing, loc) = map(
            json_dump,
            (character, thing, loc)
        )
        nextloc = json_dump(nextloc) if nextloc else None
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

    def thing_loc_and_next_del(self, character, thing, branch, tick):
        (character, thing) = map(json_dump, (character, thing))
        try:
            self.sql(
                'thing_loc_and_next_ins',
                character,
                thing,
                branch,
                tick,
                None,
                None
            )
        except IntegrityError:
            self.sql(
                'thing_loc_and_next_upd',
                None,
                None,
                character,
                thing,
                branch,
                tick
            )

    def thing_loc_items(self, character, branch, tick):
        character = json_dump(character)
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
                    yield (json_load(n), json_load(l))
                seen.add(n)

    def thing_and_loc(self, character, thing, branch, tick):
        (character, thing) = map(json_dump, (character, thing))
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
                return (json_load(th), json_load(l))
        raise KeyError("Thing never existed")

    def node_stats_branch(self, character, node, branch):
        (character, node) = map(json_dump, (character, node))
        for (key, tick, value) in self.sql(
                'node_var_data_branch', character, node, branch
        ):
            yield (
                json_load(key),
                tick,
                json_load(value)
            )

    def character_things_items(self, character, branch, tick):
        character = json_dump(character)
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (th, l) in self.sql(
                    'character_things_items', character, b, t
            ):
                if l is not None and th not in seen:
                    yield (json_load(th), json_load(l))
                seen.add(th)

    def avatarness(self, character, branch, tick):
        character = json_dump(character)
        d = {}
        for (b, t) in self.active_branches(branch, tick):
            for (graph, node, avatar) in self.sql(
                    'avatarness', character, b, t
            ):
                g = json_load(graph)
                n = json_load(node)
                is_av = bool(avatar)
                if g not in d:
                    d[g] = {}
                d[g][n] = is_av
        return d

    def is_avatar_of(self, character, graph, node, branch, tick):
        (character, graph, node) = map(json_dump, (character, graph, node))
        for (avatarness,) in self.sql(
                'is_avatar_of', character, graph, node, branch, tick
        ):
            return avatarness and self.node_exists(
                graph, node, branch, tick
            )

    def sense_func_get(self, character, sense, branch, tick):
        character = json_dump(character)
        for (b, t) in self.active_branches(branch, tick):
            for (func,) in self.sql(
                    'sense_func_get', character, sense, branch, tick
            ):
                return func

    def sense_active_items(self, character, branch, tick):
        character = json_dump(character)
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (sense, active) in self.sql(
                    'sense_active_items', character, b, t
            ):
                if sense not in seen and active:
                    yield sense
                seen.add(sense)

    def sense_is_active(self, character, sense, branch, tick):
        character = json_dump(character)
        for (b, t) in self.active_branches(branch, tick):
            for (act,) in self.sql(
                    'sense_is_active', character, sense, branch, tick
            ):
                return bool(act)
        return False

    def sense_fun_set(self, character, sense, branch, tick, funn, active):
        character = json_dump(character)
        try:
            self.sql(
                'sense_fun_ins', character, sense, branch, tick, funn, active
            )
        except IntegrityError:
            self.sql(
                'sense_fun_upd', funn, active, character, sense, branch, tick
            )

    def sense_set(self, character, sense, branch, tick, active):
        character = json_dump(character)
        try:
            self.sql('sense_ins', character, sense, branch, tick, active)
        except IntegrityError:
            self.sql('sense_upd', active, character, sense, branch, tick)

    def init_character(
            self, character, charrule, avrule, thingrule, placerule, portrule
    ):
        character = json_dump(character)
        try:
            return self.sql(
                'character_ins',
                character,
                charrule,
                avrule,
                thingrule,
                placerule,
                portrule
            )
        except IntegrityError:
            pass

    def avatars_now(self, character, branch, tick):
        character = json_dump(character)
        seen = set()
        for (b, t) in self.active_branches(branch, tick):
            for (g, n, a) in self.sql('avatars_now', character, b, t, b, t):
                if (g, n) not in seen:
                    yield (json_load(g), json_load(n), a)
                seen.add((g, n))

    def avatars_ever(self, character):
        character = json_dump(character)
        for (g, n, b, t, a) in self.sql('avatars_ever', character):
            yield (json_load(g), json_load(n), b, t, a)

    def avatar_set(self, character, graph, node, branch, tick, isav):
        (character, graph, node) = map(json_dump, (character, graph, node))
        try:
            return self.sql(
                'avatar_ins', character, graph, node, branch, tick, isav
            )
        except IntegrityError:
            return self.sql(
                'avatar_upd', isav, character, graph, node, branch, tick
            )

    def rulebook_set(self, rulebook, idx, rule):
        rulebook = json_dump(rulebook)
        try:
            return self.sql('rulebook_ins', rulebook, idx, rule)
        except IntegrityError:
            return self.sql('rulebook_upd', rule, rulebook, idx)

    def rulebook_decr(self, rulebook, idx):
        self.sql('rulebook_dec', json_dump(rulebook), idx)

    def rulebook_del(self, rulebook, idx):
        rulebook = json_dump(rulebook)
        self.sql('rulebook_del', rulebook, idx)
        self.sql('rulebook_dec', rulebook, idx)

    def rulebook_rules(self, rulebook):
        for (rule,) in self.sql('rulebook_rules', json_dump(rulebook)):
            yield rule

    def current_rules_character(self, character, branch, tick):
        return self.sql(
                'current_rules_character',
                json_dump(character),
                branch,
                tick
        )

    def current_rules_avatar(self, character, branch, tick):
        return self.sql(
            'current_rules_avatar',
            json_dump(character),
            branch,
            tick
        )

    def current_rules_character_thing(self, character, branch, tick):
        return self.sql(
            'current_rules_character_thing',
            json_dump(character),
            branch,
            tick
        )

    def current_rules_character_place(self, character, branch, tick):
        return self.sql(
            'current_rules_character_place',
            json_dump(character),
            branch,
            tick
        )

    def current_rules_character_portal(self, character, branch, tick):
        return self.sql(
            'current_rules_character_portal',
            json_dump(character),
            branch,
            tick
        )

    def current_rules_node(self, character, node, branch, tick):
        (character, node) = map(json_dump, (character, node))
        return self.sql(
            'current_rules_node',
            character,
            node,
            branch,
            tick
        )

    def current_rules_portal(self, character, nodeA, nodeB, branch, tick):
        (character, nodeA, nodeB) = map(json_dump, (character, nodeA, nodeB))
        return self.sql(
            'current_rules_portal',
            character,
            nodeA,
            nodeB,
            branch,
            tick
        )

    def ct_rulebook_rules(self, rulebook):
        return self.sql('ct_rulebook_rules', json_dump(rulebook)).fetchone()[0]

    def rulebook_get(self, rulebook, idx):
        return self.sql('rulebook_get', json_dump(rulebook), idx).fetchone()[0]

    def allrules(self):
        for (rule,) in self.sql('allrules'):
            yield rule

    def ctrules(self):
        return self.sql('ctrules').fetchone()[0]

    def ruledel(self, rule):
        self.sql('ruledel', rule)

    def haverule(self, rule):
        for r in self.sql('haverule', rule):
            return True
        return False

    def ruleins(self, rule):
        self.sql('ruleins', rule, '[]', '[]', '[]')

    def avatar_branch_data(self, character, graph, branch, tick):
        (character, graph) = map(json_dump, (character, graph))
        for (node, isav) in self.sql(
                'avatar_branch_data', character, graph, branch, tick
        ):
            yield (json_load(node), bool(isav))

    def thing_locs_branch_data(self, character, thing, branch):
        (character, thing) = map(json_dump, (character, thing))
        for (tick, loc, nextloc) in self.sql(
                'thing_locs_branch_data', character, thing, branch
        ):
            yield (tick, json_load(loc), json_load(nextloc))

    def char_stat_branch_data(self, character, branch):
        character = json_dump(character)
        for (key, tick, value) in self.sql(
                'char_stat_branch_data', character, branch
        ):
            yield (json_load(key), tick, json_load(value))

    def node_stat_branch_data(self, character, node, branch):
        (character, node) = map(json_dump, (character, node))
        for (key, tick, value) in self.sql(
                'node_stat_branch_data', character, node, branch
        ):
            yield (json_load(key), tick, json_load(value))

    def edge_stat_branch_data(self, character, orig, dest, branch):
        (character, orig, dest) = map(json_dump, (character, orig, dest))
        for (key, tick, value) in self.sql(
                'edge_stat_branch_data', character, orig, dest, branch
        ):
            yield (json_load(key), tick, json_load(value))

    def timestream_data(self):
        yield from self.sql('allbranch')

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
                'character_portal_rules_handled',
                'thing_rules_handled',
                'place_rules_handled',
                'portal_rules_handled'
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
                'character_portal_rules_handled',
                'character_thing_rules_handled',
                'character_place_rules_handled',
                'character_portal_rules_handled',
                'thing_rules_handled',
                'place_rules_handled',
                'portal_rules_handled'
        ):
            self.index_table(idx)
