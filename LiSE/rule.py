# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from collections import (
    MutableMapping,
    MutableSequence,
    Callable,
    defaultdict
)
from .funlist import FunList
from .util import (
    dispatch,
    listener
)


class Rule(object):
    """A collection of actions, being functions that enact some change on
    the world, which will be called each tick if and only if all of
    the prereqs return True, they being boolean functions that do not
    change the world.

    """
    def __init__(
            self,
            engine,
            name,
            triggers=None,
            prereqs=None,
            actions=None
    ):
        """Store the engine and my name, make myself a record in the database
        if needed, and instantiate one FunList each for my triggers,
        actions, and prereqs.

        """
        self.engine = engine
        self.name = name
        # if I don't yet have a database record, make one
        if not self.engine.db.haverule(name):
            self.engine.db.ruleins(name)

        funl = lambda store, field: FunList(
            self.engine, store, 'rules', ['rule'], [self.name], field
        )
        self.actions = funl(self.engine.action, 'actions')
        self.prereqs = funl(self.engine.prereq, 'prereqs')
        self.triggers = funl(self.engine.trigger, 'triggers')
        if triggers:
            self.triggers.extend(triggers)
        if prereqs:
            self.prereqs.extend(prereqs)
        if actions:
            self.actions.extend(actions)

    def __call__(self, engine, *args):
        """If at least one trigger fires, check the prereqs. If all the
        prereqs pass, perform the actions.

        After each call to a trigger, prereq, or action, the sim-time
        is reset to what it was before the rule was called.

        """
        if not self.check_triggers(engine, *args):
            return []
        if not self.check_prereqs(engine, *args):
            return []
            # maybe a result object that informs you as to why I
            # didn't run?
        return self.run_actions(engine, *args)

    def __repr__(self):
        return 'Rule({})'.format(self.name)

    def trigger(self, fun):
        """Decorator to append the function to my triggers list."""
        self.triggers.append(fun)

    def prereq(self, fun):
        """Decorator to append the function to my prereqs list."""
        self.prereqs.append(fun)

    def action(self, fun):
        """Decorator to append the function to my actions list."""
        self.actions.append(fun)

    def duplicate(self, newname):
        """Return a new rule that's just like this one, but under a new
        name.

        """
        if self.engine.db.haverule(newname):
            raise KeyError("Already have a rule called {}".format(newname))
        return Rule(
            self.engine,
            newname,
            list(self.triggers),
            list(self.prereqs),
            list(self.actions)
        )

    def always_run(self):
        """Arrange to be triggered every tick, regardless of circumstance."""
        def truth(*args):
            return True
        self.triggers = [truth]

    def check_triggers(self, engine, *args):
        """Run each trigger in turn. If one returns True, return True
        myself. If none do, return False.

        """
        curtime = engine.time
        for trigger in self.triggers:
            result = trigger(engine, *args)
            engine.time = curtime
            if result:
                return True
        return False

    def check_prereqs(self, engine, *args):
        """Run each prereq in turn. If all return True, return True myself. If
        one doesn't, return False.

        """
        curtime = engine.time
        for prereq in self.prereqs:
            result = prereq(engine, *args)
            engine.time = curtime
            if not result:
                return False
        return True

    def run_actions(self, engine, *args):
        """Run all my actions and return a list of their results.

        """
        curtime = engine.time
        r = []
        for action in self.actions:
            r.append(action(engine, *args))
            engine.time = curtime
        return r


class RuleBook(MutableSequence):
    """A list of rules to be followed for some Character, or a part of it
    anyway.

    """
    def __init__(self, engine, name):
        self.engine = engine
        assert(isinstance(name, str))
        self.name = name

    def __iter__(self):
        for rule in self.engine.db.rulebook_rules(self.name):
            yield self.engine.rule[rule]

    def __len__(self):
        return self.engine.db.ct_rulebook_rules(self.name)

    def __getitem__(self, i):
        return self.engine.rule[self.engine.db.rulebook_get(self.name, i)]

    def __setitem__(self, i, v):
        if isinstance(v, Rule):
            rule = v
        elif isinstance(v, str):
            rule = self.engine.rule[v]
        else:
            rule = Rule(self.engine, v)
        self.engine.db.rulebook_set(self.name, i, rule.name)

    def insert(self, i, v):
        self.engine.db.rulebook_decr(self.name, i)
        self[i] = v

    def __delitem__(self, i):
        self.engine.db.rulebook_del(self.name, i)


class RuleMapping(MutableMapping):
    def __init__(self, engine, rulebook):
        self.engine = engine
        self.rulebook = rulebook
        self._listeners = defaultdict(list)
        self._rule_cache = {}

    def listener(self, f=None, rule=None):
        return listener(self._listeners, f, rule)

    def _dispatch(self, rule, active):
        dispatch(self._listeners, rule.name, self, rule, active)

    def _activate_rule(self, rule):
        (branch, tick) = self.engine.time
        if rule not in self.rulebook:
            self.rulebook.append(rule)
        self.engine.db.rule_set(
            self.rulebook.name,
            rule.name,
            branch,
            tick,
            True
        )
        self._dispatch(rule, True)

    def __repr__(self):
        return 'RuleMapping({})'.format([k for k in self])

    def __iter__(self):
        return self.engine.db.active_rules_rulebook(
            self.rulebook.name,
            *self.engine.time
        )

    def __len__(self):
        n = 0
        for rule in self:
            n += 1
        return n

    def __contains__(self, k):
        return self.engine.db.active_rule_rulebook(
            self.rulebook.name,
            k,
            *self.engine.time
        )

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("Rule '{}' is not in effect".format(k))
        if k not in self._rule_cache:
            self._rule_cache[k] = Rule(self.engine, k)
        return self._rule_cache[k]

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError

    def __setitem__(self, k, v):
        if isinstance(v, Rule):
            if v.name != k:
                raise ValueError("That rule doesn't go by that name")
            self._activate_rule(v)
        elif isinstance(v, Callable):
            # create a new rule, named k, performing ation v
            if k in self.engine.rule:
                raise KeyError(
                    "Already have a rule named {k}. "
                    "Set engine.rule[{k}] to a new value "
                    "if you really mean to replace "
                    "the old rule.".format(
                        k=k
                    )
                )
            funn = k
            if funn in self.engine.action:
                funn += "0"
            i = 1
            while funn in self.engine.action:
                funn = funn[:-1] + str(i)
                i += 1
            self.engine.action[funn] = v
            rule = Rule(self.engine, k)
            self._rule_cache[k] = rule
            rule.actions.append(funn)
            self._activate_rule(rule)
        else:
            # v is the name of a rule. Maybe it's been created
            # previously or maybe it'll get initialized in Rule's
            # __init__.
            self._rule_cache[k] = Rule(self.engine, v)
            self._activate_rule(self._rule_cache[k])

    def __call__(self, v, name=None):
        name = name if name is not None else v.__name__
        self[name] = v
        return self[name]

    def __delitem__(self, k):
        """Deactivate the rule"""
        (branch, tick) = self.engine.time
        rule = self[k]
        self.engine.db.rule_set(
            self.rulebook.name,
            k,
            branch,
            tick,
            False
        )
        self._dispatch(rule, False)


class CharRuleMapping(RuleMapping):
    def __init__(self, character, rulebook, booktyp):
        super().__init__(rulebook.engine, rulebook)
        self.character = character
        self._table = booktyp + "_rules"

    def __iter__(self):
        return self.engine.db.active_rules_char(
            self._table,
            self.character.name,
            self.rulebook.name,
            *self.engine.time
        )

    def __contains__(self, k):
        return self.engine.db.active_rule_char(
            self._table,
            self.character.name,
            self.rulebook.name,
            k,
            *self.engine.time
        )


class AllRules(MutableMapping):
    def __init__(self, engine):
        self.engine = engine
        self._cache = {}
        self._listeners = defaultdict(list)

    def listener(self, f=None, rule=None):
        return listener(self._listeners, f, rule)

    def _dispatch(self, rule, active):
        dispatch(self._listeners, rule.name, self, rule, active)

    def __iter__(self):
        yield from self.engine.db.allrules()

    def __len__(self):
        return self.engine.db.ctrules()

    def __contains__(self, k):
        return self.engine.db.haverule(k)

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("No such rule: {}".format(k))
        if k not in self._cache:
            self._cache[k] = Rule(self.engine, k)
        return self._cache[k]

    def __setitem__(self, k, v):
        if k not in self._cache:
            self._cache[k] = Rule(self.engine, k)
        new = self._cache[k]
        new.actions = [v]
        self._dispatch(new, True)

    def __delitem__(self, k):
        if k not in self:
            raise KeyError("No such rule")
        old = self[k]
        self.engine.db.ruledel(k)
        self._dispatch(old, False)

    def __call__(self, v=None, name=None):
        if v is None and name is not None:
            def r(f):
                self[name] = f
                return self[name]
            return r
        k = name if name is not None else v.__name__
        self[k] = v
        return self[k]
