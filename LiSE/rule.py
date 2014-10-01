# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from collections import MutableMapping, MutableSequence, Callable
from .funlist import FunList


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

        def funl(store, field):
            """Create a list of functions stored in ``store`` listed in field ``field``

            That's a field in the world database, not the function store one.

            """
            return FunList(
                self.engine,
                store,
                'rules',
                ['rule'],
                [self.name],
                field
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
        """First check the prereqs. If they all pass, execute the actions and
        return a list of all their results.

        After each call to a prereq or action, the sim-time is reset
        to what it was before the rule was called.

        """
        curtime = engine.time
        triggered = False
        allowed = True
        for trigger in self.triggers:
            if trigger(engine, *args):
                triggered = True
            engine.time = curtime
            if triggered:
                break
        if not triggered:
            return []
        for prereq in self.prereqs:
            # in case one of them moves the time
            if not prereq(engine, *args):
                allowed = False
            engine.time = curtime
            if not allowed:
                break
        if not allowed:
            return []  # maybe return something more informative here?
        r = []
        for action in self.actions:
            r.append(action(engine, *args))
            engine.time = curtime
        return r

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
    def __init__(self, character, rulebook, booktyp):
        self.character = character
        self.rulebook = rulebook
        self.engine = rulebook.engine
        self._table = booktyp + "_rules"

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

    def __iter__(self):
        yield from self.engine.db.active_rules(
            self._table,
            self.character.name,
            self.rulebook.name,
            *self.engine.time
        )

    def __len__(self):
        """Count the rules presently in effect"""
        n = 0
        for rule in self:
            n += 1
        return n

    def __contains__(self, k):
        return self.engine.db.active_rule(
            self._table,
            self.character.name,
            self.rulebook.name,
            k,
            *self.engine.time
        )

    def __getitem__(self, k):
        """Get the rule by the given name, if it is in effect"""
        if k not in self:
            raise KeyError(
                "Rule is not active at the moment, if it ever was."
            )
        return Rule(self.engine, k)

    def __getattr__(self, k):
        """Alias for ``__getitem__`` for the convenience of decorators."""
        try:
            return self[k]
        except KeyError:
            raise AttributeError

    def __setitem__(self, k, v):
        if isinstance(v, Rule):
            if v.name != k:
                raise KeyError("That rule doesn't go by that name")
            self._activate_rule(v)
        elif isinstance(v, Callable):
            # create a new rule, named k, performing action v
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
            rule.actions.append(funn)
            self._activate_rule(rule)
        else:
            # v is the name of a rule. Maybe it's been created
            # previously or maybe it'll get initialized in Rule's
            # __init__.
            self._activate_rule(Rule(self.engine, v))

    def __call__(self, v, name=None):
        name = name if name is not None else v.__name__
        self.__setitem__(name, v)
        return self[name]

    def __delitem__(self, k):
        """Deactivate the rule"""
        (branch, tick) = self.engine.time
        self.engine.db.rule_set(
            self.rulebook.name,
            k,
            branch,
            tick,
            False
        )


class AllRules(MutableMapping):
    def __init__(self, engine):
        self.engine = engine

    def __iter__(self):
        yield from self.engine.db.allrules()

    def __len__(self):
        return self.engine.db.ctrules()

    def __contains__(self, k):
        return self.engine.db.haverule(k)

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
        self.engine.db.ruledel(k)

    def __call__(self, v):
        new = Rule(self.engine, v if isinstance(v, str) else v.__name__)
        new.action(v)
