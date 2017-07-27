# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
""" The fundamental unit of game logic, the Rule, and structures to
store and organize them in.

A Rule is three lists of functions: triggers, prereqs, and actions.
The actions do something, anything that you need your game to do, but
probably making a specific change to the world model. The triggers and
prereqs between them specify when the action should occur: any of its
triggers can tell it to happen, but then any of its prereqs may stop it
from happening.

Rules are assembled into RuleBooks, essentially just lists of Rules
that can then be assigned to be followed by any game entity --
but each game entity has its own RuleBook by default, and you never really
need to change that.

"""
from collections import (
    Mapping,
    MutableMapping,
    MutableSequence,
    defaultdict
)
from functools import partial
from inspect import getsource, getsourcelines
from ast import parse
from astunparse import unparse
from blinker import Signal

from .util import reify


class RuleFuncList(MutableSequence, Signal):
    __slots__ = ['rule']

    def __init__(self, rule):
        super().__init__()
        self.rule = rule

    def _nominate(self, v):
        if callable(v):
            if hasattr(self._funcstore, v.__name__):
                if unparse(parse(getsource(getattr(self._funcstore, v.__name__)))) \
                        != unparse(parse(self._funcstore._dedent_sourcelines(getsourcelines(v)[0]))):
                    raise KeyError(
                        "Already have a {typ} function named {n}. "
                        "If you really mean to replace it, set "
                        "engine.{typ}[{n}]".format(
                            typ=self._funcstore._filename.rstrip('.py'),
                            n=v.__name__
                        )
                    )
            else:
                self._funcstore(v)
            v = v.__name__
        if not hasattr(self._funcstore, v):
            raise KeyError("No {typ} function named {n}".format(
                typ=self._funcstore._filename.rstrip('.py'), n=v
            ))
        return v

    def _get(self):
        return self._cache.retrieve(self.rule.name, *self.rule.engine.time)

    def _set(self, v):
        branch, tick = self.rule.engine.time
        self._cache.store(self.rule.name, branch, tick, v)
        self._setter(self.rule.name, branch, tick, v)

    def __iter__(self):
        for funcname in self._get():
            yield getattr(self._funcstore, funcname)

    def __len__(self):
        return len()

    def __getitem__(self, i):
        return self._get()[i]

    def __setitem__(self, i, v):
        v = self._nominate(v)
        l = self._get()
        l[i] = v
        self._set(l)
        self.send(self)

    def __delitem__(self, i):
        l = self._get()
        del l[i]
        self._set(l)
        self.send(self)

    def insert(self, i, v):
        l = self._get()
        l.insert(i, self._nominate(v))
        self._set(l)
        self.send(self)

    def append(self, v):
        l = self._get()
        l.append(self._nominate(v))
        self._set(l)
        self.send(self)


class TriggerList(RuleFuncList):
    @reify
    def _funcstore(self):
        return self.rule.engine.trigger

    @reify
    def _cache(self):
        return self.rule.engine._triggers_cache

    @reify
    def _setter(self):
        return self.rule.engine.query.set_rule_triggers


class PrereqList(RuleFuncList):
    @reify
    def _funcstore(self):
        return self.rule.engine.prereq

    @reify
    def _cache(self):
        return self.rule.engine._prereqs_cache

    @reify
    def _setter(self):
        return self.rule.engine.query.set_rule_prereqs


class ActionList(RuleFuncList):
    @reify
    def _funcstore(self):
        return self.rule.engine.action

    @reify
    def _cache(self):
        return self.rule.engine._actions_cache

    @reify
    def _setter(self):
        return self.rule.engine.query.set_rule_actions


class RuleFuncListDescriptor(object):
    __slots__ = ['cls']

    def __init__(self, cls):
        self.cls = cls

    @property
    def flid(self):
        return '_funclist' + str(id(self))

    def __get__(self, obj, type=None):
        if not hasattr(obj, self.flid):
            setattr(obj, self.flid, self.cls(obj))
        return getattr(obj, self.flid)

    def __set__(self, obj, value):
        if not hasattr(obj, self.flid):
            setattr(obj, self.flid, self.cls(obj))
        flist = getattr(obj, self.flid)
        namey_value = [flist._nominate(v) for v in value]
        flist._set(namey_value)
        branch, tick = obj.engine.time
        flist._cache.store(obj.name, branch, tick, namey_value)
        flist.send(flist)

    def __delete__(self, obj):
        raise TypeError("Rules must have their function lists")


class Rule(object):
    """A collection of actions, being functions that enact some change on
    the world, which will be called each tick if and only if all of
    the prereqs return True, they being boolean functions that do not
    change the world.

    """

    triggers = RuleFuncListDescriptor(TriggerList)
    prereqs = RuleFuncListDescriptor(PrereqList)
    actions = RuleFuncListDescriptor(ActionList)

    def __init__(
            self,
            engine,
            name,
            typ='character',
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
        self.type = typ
        branch, tick = engine.time
        triggers = triggers or []
        prereqs = prereqs or []
        actions = actions or []
        self.engine.query.set_rule(name, typ, triggers, prereqs, actions, branch, tick)
        self.engine._triggers_cache.store(name, branch, tick, triggers)
        self.engine._prereqs_cache.store(name, branch, tick, prereqs)
        self.engine._actions_cache.store(name, branch, tick, actions)

    def __eq__(self, other):
        return (
            hasattr(other, 'name') and
            self.name == other.name
        )

    def _fun_names_iter(self, functyp, val):
        """Iterate over the names of the functions in ``val``,
        adding them to ``funcstore`` if they are missing;
        or if the items in ``val`` are already the names of functions
        in ``funcstore``, iterate over those.

        """
        funcstore = getattr(self.engine, functyp)
        for v in val:
            if callable(v):
                if v.__name__ in funcstore:
                    if funcstore[v.__name__] != v:
                        raise KeyError(
                            "Already have a {typ} function named "
                            "{k}. If you really mean to replace it, assign "
                            "it to engine.{typ}[{k}].".format(
                                typ=functyp,
                                k=v.__name__
                            )
                        )
                    else:
                        funcstore[v.__name__] = v
                else:
                    funcstore[v.__name__] = v
                yield v.__name__
            elif v not in funcstore:
                raise KeyError("Function {} not present in {}".format(
                    v, funcstore._tab
                ))
            else:
                yield v

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
        return fun

    def prereq(self, fun):
        """Decorator to append the function to my prereqs list."""
        self.prereqs.append(fun)
        return fun

    def action(self, fun):
        """Decorator to append the function to my actions list."""
        self.actions.append(fun)
        return fun

    def duplicate(self, newname):
        """Return a new rule that's just like this one, but under a new
        name.

        """
        if self.engine.rule.query.haverule(newname):
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
        if 'truth' in self.engine.trigger:
            truth = self.engine.trigger.truth
        else:
            def truth(*args):
                return True
        self.triggers = [truth]

    def check_triggers(self, engine, *args):
        """Run each trigger in turn. If one returns True, return True
        myself. If none do, return False.

        """
        curtime = (branch, tick) = engine.time
        for trigger in self.triggers:
            result = trigger(engine, *args)
            if engine.time != curtime:
                engine.time = curtime
            if result:
                return True
        return False

    def check_prereqs(self, engine, *args):
        """Run each prereq in turn. If all return True, return True myself. If
        one doesn't, return False.

        """
        curtime = (branch, tick) = engine.time
        for prereq in self.prereqs:
            result = prereq(self.engine, *args)
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


class RuleBook(MutableSequence, Signal):
    """A list of rules to be followed for some Character, or a part of it
    anyway.

    """

    @property
    def _cache(self):
        return self.engine._rulebooks_cache.retrieve(self.name, *self.engine.time)
    @_cache.setter
    def _cache(self, v):
        branch, tick = self.engine.time
        self.engine._rulebooks_cache.store(self.name, branch, tick, v)

    def __init__(self, engine, name):
        super().__init__()
        self.engine = engine
        self.name = name

    def __contains__(self, v):
        return getattr(v, 'name', v) in self._cache

    def __iter__(self):
        return iter(self._cache)

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, i):
        return self.engine.rule[self._cache[i]]

    def _coerce_rule(self, v):
        if isinstance(v, Rule):
            return v
        elif isinstance(v, str):
            return self.engine.rule[v]
        else:
            return Rule(self.engine, v)

    def __setitem__(self, i, v):
        v = getattr(v, 'name', v)
        cache = self._cache
        cache[i] = v
        branch, tick = self.engine.time
        self.engine.query.set_rulebook(self.name, branch, tick, cache)
        self.engine._rulebooks_cache.store(self.name, branch, tick, cache)
        self.engine.rulebook.send(self, i=i, v=v)
        self.send(self, i=i, v=v)

    def insert(self, i, v):
        v = getattr(v, 'name', v)
        cache = self._cache
        cache.insert(i, v)
        branch, tick = self.engine.time
        self.engine.query.set_rulebook(self.name, branch, tick, cache)
        self.engine._rulebooks_cache.store(self.name, branch, tick, cache)
        self.engine.rulebook.send(self, i=i, v=v)
        self.send(self, i=i, v=v)

    def index(self, v):
        if isinstance(v, str):
            return self._cache.index(v)
        return super().index(v)

    def __delitem__(self, i):
        cache = self._cache
        del cache[i]
        branch, tick = self.engine.time
        self.engine.query.set_rulebook(self.name, branch, tick, cache)
        self.engine._rulebooks_cache.store(self.name, branch, tick, cache)
        self.engine.rulebook.send(self, i=i, v=None)
        self.send(self, i=i, v=None)


class RuleMapping(MutableMapping, Signal):
    """Wraps a :class:`RuleBook` so you can get its rules by name.

    You can access the rules in this either dictionary-style or as
    attributes. This is for convenience if you want to get at a rule's
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

    """

    def __init__(self, engine, rulebook):
        super().__init__()
        self.engine = engine
        self._rule_cache = self.engine.rule._cache
        if isinstance(rulebook, RuleBook):
            self.rulebook = rulebook
        else:
            self.rulebook = self.engine.rulebook[rulebook]

    def __repr__(self):
        return 'RuleMapping({})'.format([k for k in self])

    def __iter__(self):
        return iter(self.rulebook)

    def __len__(self):
        return len(self.rulebook)

    def __contains__(self, k):
        return k in self.rulebook

    def __getitem__(self, k):
        if k not in self:
            raise KeyError("Rule '{}' is not in effect".format(k))
        return self._rule_cache[k]

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError

    def __setitem__(self, k, v):
        if v in self.engine.rule:
            v = self.engine.rule[v]
        elif isinstance(v, str) and hasattr(self.engine.function, v):
            v = getattr(self.engine.function, v)
        if not isinstance(v, Rule) and callable(v):
            if k in self.engine.rule:
                raise KeyError(
                    "Already have a rule named {name}. "
                    "If you really mean to replace it, set "
                    "self.rule[{name}] to a new Rule object.".format(name=k)
                )
            # create a new rule, named k, performing action v
            self.engine.rule[k] = v
            v = self.engine.rule[k]
        assert isinstance(v, Rule)
        if len(self.rulebook) == 0:
            self.rulebook.append(v)
        elif isinstance(k, int):
            self.rulebook[k] = v
        else:
            self.rulebook[0] = v

    def __call__(self, v=None, name=None, always=False):
        def wrap(name, always, v):
            name = name if name is not None else v.__name__
            self[name] = v
            r = self[name]
            if always:
                r.always()
            return r
        if v is None:
            return partial(wrap, name, always)
        return wrap(name, always, v)

    def __delitem__(self, k):
        i = self.rulebook.index(k)
        del self.rulebook[i]
        self.send(self, key=k, val=None)


rule_mappings = {}
rulebooks = {}


class RuleFollower(object):
    """Interface for that which has a rulebook associated, which you can
    get a :class:`RuleMapping` into

    """
    @property
    def _rule_mapping(self):
        if id(self) not in rule_mappings:
            rule_mappings[id(self)] = self._get_rule_mapping()
        return rule_mappings[id(self)]

    # keeping _rulebooks out of the instance lets subclasses
    # use __slots__ without having _rulebooks in the slots
    @property
    def _rulebooks(self):
        return rulebooks[id(self)]

    @_rulebooks.setter
    def _rulebooks(self, v):
        rulebooks[id(self)] = v

    @property
    def rule(self, v=None, name=None):
        if v is not None:
            return self._rule_mapping(v, name)
        return self._rule_mapping

    @property
    def rulebook(self):
        if not hasattr(self, '_rulebook'):
            self._upd_rulebook()
        return self._rulebook

    @rulebook.setter
    def rulebook(self, v):
        n = v.name if isinstance(v, RuleBook) else v
        self._set_rulebook_name(n)
        self._upd_rulebook()

    def _upd_rulebook(self):
        self._rulebook = self._get_rulebook()

    def _get_rulebook(self):
        return self.engine.rulebook[self._get_rulebook_name()]

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


class AllRuleBooks(Mapping, Signal):
    __slots__ = ['engine', '_cache']

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self._cache = {}

    def __iter__(self):
        return self.engine._rulebooks_cache.iter_entities(*self.engine.time)

    def __len__(self):
        return len(list(self))

    def __contains__(self, k):
        return self.engine._rulebooks_cache.contains_entity(k, *self.engine.time)

    def __getitem__(self, k):
        if k not in self._cache:
            self._cache[k] = RuleBook(self.engine, k)
        return self._cache[k]


class AllRules(MutableMapping, Signal):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def _init_load(self):
        self._cache = {name: Rule(self.engine, name, typ) for name, typ in self.engine.query.rules_dump()}

    def __iter__(self):
        yield from self._cache

    def __len__(self):
        return len(self._cache)

    def __contains__(self, k):
        return k in self._cache

    def __getitem__(self, k):
        return self._cache[k]

    def __setitem__(self, k, v):
        # you can use the name of a stored function or rule
        if isinstance(v, str):
            if hasattr(self.engine.action, v):
                v = getattr(self.engine.action, v)
            elif hasattr(self.engine.function, v):
                v = getattr(self.engine.function, v)
            elif hasattr(self.engine.rule, v):
                v = getattr(self.engine.rule, v)
            else:
                raise ValueError("Unknown function: " + v)
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
        self.send(self, key=new, rule=v)

    def __delitem__(self, k):
        if k not in self:
            raise KeyError("No such rule")
        for rulebook in self.engine.rulebooks.values():
            try:
                del rulebook[rulebook.index(k)]
            except IndexError:
                pass
        del self._cache[k]
        self.send(self, key=k, rule=None)

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
        self.send(self, rule=new, active=True)
        return new
