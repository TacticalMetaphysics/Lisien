# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from collections import MutableMapping, MutableSequence, Callable
from sqlite3 import IntegrityError
from gorm.graph import json_dump, json_load
from .funlist import FunList


class Rule(object):
    """A collection of actions, being functions that enact some change on
    the world, which will be called each tick if and only if all of
    the prereqs return True, they being boolean functions that do not
    change the world.

    """
    def __init__(self, engine, name):
        """Store the ENGINE and my name, make myself a record in the database if
        needed, and instantiate FunList once for my actions and again
        for my prereqs.

        """
        self.engine = engine
        self.name = name
        self.engine.cursor.execute(
            "SELECT COUNT(*) FROM rules WHERE rule=?;",
            (name,)
        )
        (ct,) = self.engine.cursor.fetchone()
        if ct == 0:
            self.engine.cursor.execute(
                "INSERT INTO rules (rule) VALUES (?);",
                (name,)
            )
        self.actions = FunList(self.engine, 'rules', ['rule'], [self.name], 'actions')
        self.prereqs = FunList(self.engine, 'rules', ['rule'], [self.name], 'prereqs')

    def __call__(self, engine, *args):
        """First check the prereqs. If they all pass, execute the actions and
        return a list of all their results.

        After each call to a prereq or action, the sim-time is reset
        to what it was before the rule was called.

        """
        curtime = engine.time
        r = None
        for prereq in self.prereqs:
            # in case one of them moves the time
            if not prereq(engine, *args):
                r = []
            engine.time = curtime
            if r is not None:
                break
        if r is not None:
            return r
        r = []
        for action in self.actions:
            r.append(action(engine, *args))
            engine.time = curtime
        return r

    def prereq(self, fun):
        """Decorator to append the function to my prereqs list."""
        self.prereqs.append(fun)

    def action(self, fun):
        """Decorator to append the function to my actions list."""
        self.actions.append(fun)


class RuleBook(MutableSequence):
    def __init__(self, engine, name):
        self.engine = engine
        self.name = name

    def __iter__(self):
        data = self.engine.cursor.execute(
            "SELECT rule FROM rulebooks WHERE rulebook=? ORDER BY idx ASC;",
            (self.name,)
        )
        for (rulen,) in data:
            yield self.engine.rule[rulen]

    def __len__(self):
        return self.engine.cursor.execute(
            "SELECT COUNT(*) FROM rulebooks WHERE rulebook=?;",
            (self.name,)
        ).fetchone()[0]

    def __getitem__(self, i):
        rulen = self.engine.cursor.execute(
            "SELECT rule FROM rulebooks WHERE rulebook=? AND idx=?;",
            (self.name, i)
        ).fetchone()[0]
        return self.engine.rule[rulen]

    def __setitem__(self, i, v):
        if isinstance(v, Rule):
            rule = v
        elif isinstance(v, str):
            rule = self.engine.rule[v]
        else:
            rule = Rule(self.engine, v)
        try:
            self.engine.cursor.execute(
                "INSERT INTO rulebooks (rulebook, idx, rule) VALUES (?, ?, ?);",
                (
                    self.name,
                    i,
                    rule.name
                )
            )
        except IntegrityError:
            self.engine.cursor.execute(
                "UPDATE rulebooks SET rule=? WHERE rulebook=? AND idx=?;",
                (
                    rule.name,
                    self.name,
                    i
                )
            )

    def insert(self, i, v):
        self.engine.cursor.execute(
            "UPDATE rulebooks SET idx=idx+1 WHERE "
            "rulebook=? AND "
            "idx>=?;",
            (self.name, i)
        )
        self[i] = v

    def __delitem__(self, i):
        self.engine.cursor.execute(
            "DELETE FROM rulebooks WHERE rulebook=? AND idx=?;",
            (self.name, i)
        )
        self.engine.cursor.execute(
            "UPDATE rulebooks SET idx=idx-1 WHERE rulebook=? AND idx>?;",
            (self.name, i)
        )


class RuleMapping(MutableMapping):
    def __init__(self, character, rulebook, booktyp):
        self.character = character
        self.rulebook = rulebook
        self.engine = rulebook.engine
        self.table = booktyp + "_rules"

    def _activate_rule(self, rule):
        (branch, tick) = self.engine.time
        if rule not in self.rulebook:
            self.rulebook.append(rule)
        try:
            self.engine.cursor.execute(
                "INSERT INTO active_rules "
                "(rulebook, rule, branch, tick, active) "
                "VALUES (?, ?, ?, ?, ?);",
                (
                    self.rulebook.name,
                    rule.name,
                    branch,
                    tick,
                    True
                )
            )
        except IntegrityError:
            self.engine.cursor.execute(
                "UPDATE active_rules SET active=? WHERE "
                "rulebook=? AND "
                "rule=? AND "
                "branch=? AND "
                "tick=?;",
                (
                    self.rulebook.name,
                    rule.name,
                    branch,
                    tick
                )
            )

    def __iter__(self):
        seen = set()
        for (branch, tick) in self.engine._active_branches():
            data = self.engine.cursor.execute(
                "SELECT active_rules.rule, active_rules.active "
                "FROM active_rules JOIN "
                "(SELECT rulebook, rule, branch, MAX(tick) AS tick "
                "FROM {tab} WHERE "
                "character=? AND "
                "rulebook=? AND "
                "branch=? AND "
                "tick<=? GROUP BY rulebook, rule, branch) AS hitick "
                "ON active_rules.rulebook=hitick.rulebook "
                "AND active_rules.branch=hitick.branch "
                "AND active_rules.tick=hitick.tick;",
                (
                    self.character._name,
                    self.rulebook.name,
                    branch,
                    tick
                )
            ).fetchall()
            for (r, active) in data:
                rulen = json_load(r)
                if active and rulen not in seen:
                    yield rulen
                seen.add(rulen)

    def __len__(self):
        """Count the rules presently in effect"""
        n = 0
        for rule in self:
            n += 1
        return n

    def __contains__(self, k):
        for (branch, tick) in self.engine._active_branches():
            data = self.engine.cursor.execute(
                "SELECT active_rules.active FROM active_rules JOIN ("
                "SELECT rulebook, rule, branch, MAX(tick) AS tick "
                "FROM {tab} WHERE "
                "character=? AND "
                "rulebook=? AND "
                "rule=? AND "
                "branch=? AND "
                "tick<=? GROUP BY rulebook, rule, branch) AS hitick "
                "ON active_rules.rulebook=hitick.rulebook "
                "AND active_rules.rule=hitick.rule "
                "AND active_rules.branch=hitick.branch "
                "AND active_rules.tick=hitick.tick;".format(tab=self.table),
                (
                    self.character._name,
                    self.rulebook.name,
                    k,
                    branch,
                    tick
                )
            ).fetchall()
            if len(data) == 0:
                continue
            elif len(data) > 1:
                raise ValueError("Silly data in {tab} table".format(tab=self.table))
            else:
                return bool(data[0])
        return False

    def __getitem__(self, k):
        """Get the rule by the given name, if it is in effect"""
        if k not in self:
            raise KeyError("Rule is not active at the moment, if it ever existed.")
        return Rule(self.engine, k)

    def __getattr__(self, k):
        """Alias for ``__getitem__`` for the convenience of decorators."""
        try:
            return self[k]
        except KeyError:
            raise AttributeError

    def __setitem__(self, k, v):
        if isinstance(v, Rule):
            if v._name != json_dump(k):
                raise KeyError("That rule doesn't go by that name")
            self._activate_rule(v)
        elif isinstance(v, Callable):
            # create a new rule, named k, performing action v
            if k in self.engine.rule:
                raise KeyError(
                    "Already have a rule named {k}. "
                    "Set engine.rule[{k}] to a new value if you really mean to "
                    "replace the old rule.".format(
                        k=k
                    )
                )
            funn = k
            if funn in self.engine.function:
                funn += "0"
            i = 1
            while funn in self.engine.function:
                funn = funn[:-1] + str(i)
                i += 1
            self.engine.function[funn] = v
            rule = Rule(self.engine, k)
            rule.actions.append(funn)
            self._activate_rule(rule)
        else:
            # v is the name of a rule. Maybe it's been created
            # previously or maybe it'll get initialized in Rule's
            # __init__.
            self._activate_rule(Rule(self.engine, v))

    def __call__(self, v):
        self.__setitem__(v.__name__, v)
        return self[v.__name__]

    def __delitem__(self, k):
        """Deactivate the rule"""
        (branch, tick) = self.engine.time
        try:
            self.engine.cursor.execute(
                "INSERT INTO {tab} "
                "(rulebook, rule, branch, tick, active) "
                "VALUES (?, ?, ?, ?, ?);".format(tab=self.table),
                (
                    self.rulebook.name,
                    k,
                    branch,
                    tick,
                    False
                )
            )
        except IntegrityError:
            self.engine.cursor.execute(
                "UPDATE {tab} SET active=? WHERE "
                "rulebook=? AND "
                "rule=? AND "
                "branch=? AND "
                "tick=?;".format(tab=self.table),
                (
                    False,
                    self.rulebook.name,
                    k,
                    branch,
                    tick
                )
            )


class AllRules(MutableMapping):
    def __init__(self, engine):
        self.engine = engine

    def __iter__(self):
        for (rule,) in self.engine.cursor.execute(
            "SELECT rule FROM rules;"
        ).fetchall():
            yield rule

    def __len__(self):
        return self.engine.cursor.execute(
            "SELECT COUNT(*) FROM rules;"
        ).fetchone()[0]

    def __contains__(self, k):
        n = self.engine.cursor.execute(
            "SELECT COUNT(*) FROM rules WHERE rule=?;",
            (k,)
        ).fetchone()[0]
        return n > 0

    def __getitem__(self, k):
        if k in self:
            return Rule(self.engine, k)

    def __setitem__(self, k, v):
        new = Rule(self.engine, k)
        new.actions = v.actions
        new.prereqs = v.prereqs

    def __delitem__(self, k):
        if k not in self:
            raise KeyError("No such rule")
        self.engine.cursor.execute(
            "DELETE FROM rules WHERE rule=?;",
            (k,)
        )

    def __call__(self, v):
        new = Rule(self.engine, v if isinstance(v, str) else v.__name__)
        new.action(v)
