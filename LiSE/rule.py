# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from collections import (
    Mapping,
    MutableMapping,
    MutableSequence,
    Callable,
    defaultdict
)
from .funlist import FunList
from .util import (
    dispatch,
    listener,
    listen
)


class RuleFunList(FunList):
    def __init__(self, rule):
        self.rule = rule
        super().__init__(rule.engine, rule.engine.rule.db)


class TriggerList(RuleFunList):
    @property
    def funcstore(self):
        return self.engine.trigger

    def _savelist(self, l):
        self.db.set_rule_triggers(self.rule.name, l)

    def _loadlist(self):
        return self.db.rule_triggers(self.rule.name)


class PrereqList(RuleFunList):
    @property
    def funcstore(self):
        return self.engine.prereq

    def _savelist(self, l):
        self.db.set_rule_prereqs(self.rule.name, l)

    def _loadlist(self):
        return self.db.rule_prereqs(self.rule.name)


class ActionList(RuleFunList):
    @property
    def funcstore(self):
        return self.engine.action

    def _savelist(self, l):
        self.db.set_rule_actions(self.rule.name, l)

    def _loadlist(self):
        return self.db.rule_actions(self.rule.name)


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
        self.name = self.__name__ = name
        if name not in self.engine.rule:
            # normally I'd use rule.new_empty but that causes a recursion error in this case
            self.engine.rule.db.create_blank_rule(name)
        self._actions = ActionList(self)
        self._prereqs = PrereqList(self)
        self._triggers = TriggerList(self)
        if triggers:
            self.triggers.extend(triggers)
        if prereqs:
            self.prereqs.extend(prereqs)
        if actions:
            self.actions.extend(actions)

    def __eq__(self, other):
        return (
            hasattr(other, 'name') and
            self.name == other.name
        )

    def __getattr__(self, attrn):
        if attrn == 'triggers':
            return self._triggers
        elif attrn == 'prereqs':
            return self._prereqs
        elif attrn == 'actions':
            return self._actions
        else:
            raise AttributeError("No attribute: {}".format(attrn))

    def __setattr__(self, attrn, val):
        if attrn == 'triggers':
            self._triggers._setlist([])
            self._triggers.extend(val)
        elif attrn == 'prereqs':
            self._prereqs._setlist([])
            self._prereqs.extend(val)
        elif attrn == 'actions':
            self._actions._setlist([])
            self._actions.extend(val)
        else:
            super().__setattr__(attrn, val)

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
        self._triggers.append(fun)

    def prereq(self, fun):
        """Decorator to append the function to my prereqs list."""
        self._prereqs.append(fun)

    def action(self, fun):
        """Decorator to append the function to my actions list."""
        self._actions.append(fun)

    def duplicate(self, newname):
        """Return a new rule that's just like this one, but under a new
        name.

        """
        if self.engine.rule.db.haverule(newname):
            raise KeyError("Already have a rule called {}".format(newname))
        return Rule(
            self.engine,
            newname,
            list(self.triggers),
            list(self.prereqs),
            list(self.actions)
        )

    def always(self):
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
        print('running actions {} for rule {}'.format(
            [action.__name__ for action in self.actions],
            self.name
        ))
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
        self.name = name
        self._listeners = []
        if self.engine.caching:
            self._cache = list(self.engine.db.rulebook_rules(self.name))

    def __contains__(self, v):
        if self.engine.caching:
            cache = self._cache
        else:
            cache = list(self.engine.db.rulebook_rules(self.name))
        if isinstance(v, Rule):
            v = v.name
        return v in cache

    def __iter__(self):
        if self.engine.caching:
            for rulen in self._cache:
                yield self.engine.rule[rulen]
            return
        for rule in self.engine.db.rulebook_rules(self.name):
            yield self.engine.rule[rule]

    def __len__(self):
        if self.engine.caching:
            return len(self._cache)
        return self.engine.db.ct_rulebook_rules(self.name)

    def __getitem__(self, i):
        if self.engine.caching:
            return self.engine.rule[self._cache[i]]
        return self.engine.rule[
            self.engine.db.rulebook_get(
                self.name,
                i
            )
        ]

    def _dispatch(self):
        self.engine.rulebook.dispatch(self)

    def _activate_rule(self, rule, active=True):
        (branch, tick) = self.engine.time
        self.engine.db.rule_set(
            self.name,
            rule.name,
            branch,
            tick,
            active
        )

    def __setitem__(self, i, v):
        if isinstance(v, Rule):
            rule = v
        elif isinstance(v, str):
            rule = self.engine.rule[v]
        else:
            rule = Rule(self.engine, v)
        self.engine.db.rulebook_set(self.name, i, rule.name)
        self._activate_rule(rule)
        if self.engine.caching:
            while len(self._cache) <= i:
                self._cache.append(None)
            self._cache[i] = rule.name
        self._dispatch()

    def insert(self, i, v):
        self.engine.db.rulebook_decr(self.name, i)
        self[i] = v

    def index(self, v):
        if isinstance(v, str):
            i = 0
            for rule in self:
                if rule.name == v:
                    return i
                i += 1
            else:
                raise ValueError(
                    "No rule named {} in rulebook {}".format(
                        v, self.name
                    )
                )
        return super().index(v)

    def __delitem__(self, i):
        self.engine.db.rulebook_del(self.name, i)
        if self.engine.caching:
            del self._cache[i]
        self._dispatch()

    def listener(self, fun):
        self.engine.rulebook.listener(rulebook=self.name)(fun)


class RuleMapping(MutableMapping):
    """A wrapper around a :class:`RuleBook` that lets you get at its rules
    by name.

    You can access the rules in this either dictionary-style or as
    attributes; this is for convenience if you want to get at a rule's
    decorators, eg. to add an Action to the rule.

    Using this as a decorator will create a new rule, named for the
    decorated function, and using the decorated function as the
    initial Action.

    Using this like a dictionary will let you create new rules,
    appending them onto the underlying :class:`RuleBook`; replace one
    rule with another, where the new one will have the same index in
    the :class:`RuleBook` as the old one; and activate or deactivate
    rules. The name of a rule may be used in place of the actual rule,
    so long as the rule already exists.

    You can also set a rule active or inactive by setting it to
    ``True`` or ``False``, respectively. Inactive rules are still in
    the rulebook but won't be followed until activated again.

    """
    def __init__(self, engine, rulebook):
        self.engine = engine
        if isinstance(rulebook, RuleBook):
            self.rulebook = rulebook
        elif isinstance(rulebook, str):
            self.rulebook = RuleBook(engine, rulebook)
        else:
            raise TypeError(
                "Need a rulebook or the name of one, not {}".format(
                    type(rulebook)
                )
            )
        self._listeners = defaultdict(list)
        self._rule_cache = {}

    def listener(self, f=None, rule=None):
        return listener(self._listeners, f, rule)

    def _dispatch(self, rule, active):
        dispatch(self._listeners, rule.name, self, rule, active)

    def _activate_rule(self, rule, active=True):
        if rule not in self.rulebook:
            self.rulebook.append(rule)
        else:
            self.rulebook._activate_rule(rule, active)
        self._dispatch(rule, active)

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
            self._rule_cache[k].active = True
        return self._rule_cache[k]

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError

    def __setitem__(self, k, v):
        if isinstance(v, bool):
            if k not in self:
                raise KeyError(
                    "Can't activate or deactivate {}, "
                    "because it is not in my rulebook ({}).".format(
                        k, self.rulebook.name
                    )
                )
            self._activate_rule(k, v)
            return
        if isinstance(v, str):
            v = self.engine.rule[k]
        if isinstance(v, Rule):
            # may raise ValueError
            i = self.rulebook.index(k)
            if self.rulebook[i] != v:
                self.rulebook[i] = v
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
            if k not in self.engine.rule:
                self.engine.rule[k] = v
                rule = self.engine.rule[k]
            else:
                rule = self.engine.rule[k]
                rule.actions.append(funn)
            self._activate_rule(rule)
        else:
            raise TypeError(
                "{} is not a rule or the name of one".format(
                    type(v)
                )
            )

    def __call__(self, v, name=None):
        name = name if name is not None else v.__name__
        self[name] = v
        return self[name]

    def __delitem__(self, k):
        i = self.rulebook.index(k)
        del self.rulebook[i]
        self._dispatch(rule, None)


class RuleFollower(object):
    """Interface for that which has a rulebook associated, which you can
    get a :class:`RuleMapping` into

    """
    @property
    def rule(self, v=None, name=None):
        if not hasattr(self, '_rule_mapping'):
            self._rule_mapping = self._get_rule_mapping()
        if v is not None:
            return self._rule_mapping(v, name)
        return self._rule_mapping

    @property
    def _rulebook_listeners(self):
        if not hasattr(self, '_rbl'):
            self._rbl = []
        return self._rbl

    @_rulebook_listeners.setter
    def _rulebook_listeners(self, v):
        self._rbl = v

    @property
    def rulebook(self):
        if not hasattr(self, '_rulebook'):
            self._upd_rulebook()
        return self._rulebook

    @rulebook.setter
    def rulebook(self, v):
        if not (isinstance(v, str) or isinstance(v, RuleBook)):
            raise TypeError("Use a :class:`RuleBook` or the name of one")
        n = v.name if isinstance(v, RuleBook) else v
        self._set_rulebook_name(n)
        self._dispatch_rulebook(v)
        self._upd_rulebook()

    def _upd_rulebook(self):
        """Set my ``_rulebook`` property to my rulebook as of this moment, and
        call all of my ``_rulebook_listeners``.

        """
        self._rulebook = self._get_rulebook()
        for f in self._rulebook_listeners:
            f(self, self._rulebook)

    def _get_rulebook(self):
        """Return my :class:`RuleBook` as of this moment."""
        raise NotImplementedError

    def rules(self):
        if not hasattr(self, 'engine'):
            raise AttributeError("Need an engine before I can get rules")
        for (rulen, active) in self._rule_names():
            if (
                hasattr(self.rule, '_rule_cache') and
                rulen in self.rulebook._rule_cache
            ):
                rule = self.rule._rule_cache[rulen]
            else:
                rule = Rule(self.engine, rulen)
            rule.active = active
            yield rule

    def rulebook_listener(self, f):
        listen(self._rulebook_listeners, f)

    def _rule_names_activeness(self):
        """Iterate over pairs of rule names and their activeness for each rule
        in my rulebook.

        """
        raise NotImplementedError

    def _get_rule_mapping(self):
        """Get the :class:`RuleMapping` for my rulebook."""
        raise NotImplementedError

    def _get_rulebook_name(self):
        """Get the name of my rulebook."""
        raise NotImplementedError

    def _set_rulebook_name(self, n):
        """Tell the database that this is the name of the rulebook to use for
        me.

        """
        raise NotImplementedError


class AllRuleBooks(Mapping):
    def __init__(self, engine, db):
        self.engine = engine
        self.db = db
        self.db.init_table('rulebooks')
        self._cache = {}
        self._listeners = defaultdict(list)

    def __iter__(self):
        yield from self.db.rulebooks()

    def __len__(self):
        return self.db.ct_rulebooks()

    def __contains__(self, k):
        if k in self._cache:
            return self._cache[k]
        return self.db.ct_rulebook_rules(k) > 0

    def __getitem__(self, k):
        if k not in self._cache:
            self._cache[k] = RuleBook(self.engine, k)
        return self._cache[k]

    def listener(self, f=None, rulebook=None):
        return listener(self._listeners, f, rulebook)

    def dispatch(self, rulebook):
        for fun in self._listeners[rulebook.name]:
            fun(rulebook)


class AllRules(MutableMapping):
    def __init__(self, engine, db):
        self.engine = engine
        self.db = db
        self.db.init_table('rules')
        self.db.init_table('rulebooks')
        self._cache = {}
        self._listeners = defaultdict(list)

    def listener(self, f=None, rule=None):
        return listener(self._listeners, f, rule)

    def _dispatch(self, rule, active):
        dispatch(self._listeners, rule.name, self, rule, active)

    def __iter__(self):
        yield from self.db.allrules()

    def __len__(self):
        return self.db.ctrules()

    def __contains__(self, k):
        return self.db.haverule(k)

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("No such rule: {}".format(k))
        if k not in self._cache:
            self._cache[k] = Rule(self.engine, k)
        return self._cache[k]

    def __setitem__(self, k, v):
        if isinstance(v, str):
            v = self.action[v]
        if callable(v):
            if k not in self._cache:
                self._cache[k] = Rule(self.engine, k)
            new = self._cache[k]
            new.actions = [v]
        elif isinstance(v, Rule):
            self._cache[k] = v
            new = v
        else:
            raise TypeError(
                "Don't know how to store {} as a rule.".format(type(v))
            )
        self._dispatch(new, True)

    def __delitem__(self, k):
        if k not in self:
            raise KeyError("No such rule")
        old = self[k]
        self.db.ruledel(k)
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

    def new_empty(self, name):
        if name in self:
            raise KeyError("Already have rule {}".format(name))
        new = Rule(self.engine, name)
        self._cache[name] = new
        self._dispatch(new, True)
        return new
