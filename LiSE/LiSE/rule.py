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
but each game entity has its own RuleBook by default, and you never
need to change that.

To add a new rule to a LiSE entity, the easiest thing is to use the
decorator syntax:

```
@entity.rule
def do_something(entity):
    ...

@do_something.trigger
def whenever(entity):
    ...

@do_something.trigger
def forever(entity):
    ...

@do_something.action
def do_something_else(entity):
    ...

```

When run, this code will:
* copy the `do_something` function to `action.py`, where LiSE knows
  to run it when a rule triggers it
* create a new rule named `'do_something'`
* set the function `do_something` as the first (and, so far, only) entry
  in the actions list of the rule by that name
* copy the `whenever` function to `trigger.py`, where LiSE knows to
  call it when a rule has it as a trigger
* set the function `whenever` as the first entry in the triggers list
  of the rule `'do_something'`
* append the function `forever` to the same triggers list
* copy `do_something_else` to `action.py`
* append `do_something_else` to the actions list of the rule

The `trigger`, `prereq`, and `action` attributes of Rule objects
may also be used like lists. You can put functions in them yourself,
provided they are already present in the correct module. If it's
inconvenient to get the actual function object, use a string of
the function's name.

"""
from collections.abc import (
    MutableMapping,
    MutableSequence,
    Hashable
)
from abc import ABC, abstractmethod
from functools import partial
from inspect import getsource
from ast import parse

from astunparse import unparse
from blinker import Signal

from .reify import reify
from .util import dedent_source


def roundtrip_dedent(source):
    """Reformat some lines of code into what unparse makes."""
    return unparse(parse(dedent_source(source)))


class RuleFuncList(MutableSequence, Signal):
    """Abstract class for lists of functions like trigger, prereq, action"""
    __slots__ = ['rule']

    def __init__(self, rule):
        super().__init__()
        self.rule = rule

    def _nominate(self, v):
        if callable(v):
            if hasattr(self._funcstore, v.__name__):
                if v == getattr(self._funcstore, v.__name__):
                    return v.__name__
                elif hasattr(self._funcstore, 'get_source'):
                    stored_source = self._funcstore.get_source(v.__name__)
                else:
                    stored_source = getsource(getattr(self._funcstore, v.__name__))
                new_source = getsource(v)
                if roundtrip_dedent(stored_source) != roundtrip_dedent(new_source):
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
        return self._cache.retrieve(self.rule.name, *self.rule.engine._btt())

    def _set(self, v):
        branch, turn, tick = self.rule.engine._nbtt()
        self._cache.store(self.rule.name, branch, turn, tick, v)
        self._setter(self.rule.name, branch, turn, tick, v)

    def __iter__(self):
        for funcname in self._get():
            yield getattr(self._funcstore, funcname)

    def __len__(self):
        return len(self._get())

    def __getitem__(self, i):
        return getattr(self._funcstore, self._get()[i])

    def __setitem__(self, i, v):
        v = self._nominate(v)
        l = list(self._get())
        l[i] = v
        self._set(tuple(l))
        self.send(self)

    def __delitem__(self, i):
        l = list(self._get())
        del l[i]
        self._set(tuple(l))
        self.send(self)

    def insert(self, i, v):
        l = list(self._get())
        l.insert(i, self._nominate(v))
        self._set(tuple(l))
        self.send(self)

    def append(self, v):
        self._set(self._get() + (self._nominate(v),))
        self.send(self)

    def index(self, x, start=0, end=None):
        if not callable(x):
            x = getattr(self._funcstore, x)
        return super().index(x, start, end)


class TriggerList(RuleFuncList):
    """A list of trigger functions for rules"""
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
    """A list of prereq functions for rules"""
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
    """A list of action functions for rules"""
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
    """Descriptor that lets you get and set a whole RuleFuncList at once"""
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
        namey_value = tuple(flist._nominate(v) for v in value)
        flist._set(namey_value)
        branch, turn, tick = obj.engine._nbtt()
        flist._cache.store(obj.name, branch, turn, tick, namey_value)
        flist.send(flist)

    def __delete__(self, obj):
        raise TypeError("Rules must have their function lists")


class Rule(object):
    """Stuff that might happen in the simulation under some conditions

    Rules are comprised of three lists of functions:

    * actions, which mutate the world state
    * triggers, which make the actions happen
    * prereqs, which prevent the actions from happening when triggered

    Each kind of function should be stored in the appropriate module
    supplied to the LiSE core at startup. This makes it possible to
    load the functions on later startups. You may instead use the string
    name of a function already stored in the module, or use the
    `trigger`, `prereq`, or `action` decorator on a new function to
    add it to both the module and the rule.

    """

    triggers = RuleFuncListDescriptor(TriggerList)
    prereqs = RuleFuncListDescriptor(PrereqList)
    actions = RuleFuncListDescriptor(ActionList)

    def __init__(
            self,
            engine,
            name,
            triggers=None,
            prereqs=None,
            actions=None,
            create=True
    ):
        """Store the engine and my name, make myself a record in the database
        if needed, and instantiate one FunList each for my triggers,
        actions, and prereqs.

        """
        self.engine = engine
        self.name = self.__name__ = name
        branch, turn, tick = engine._btt()
        if create and not self.engine._triggers_cache.contains_key(name, branch, turn, tick):
            tick += 1
            self.engine.tick = tick
            triggers = tuple(self._fun_names_iter('trigger', triggers or []))
            prereqs = tuple(self._fun_names_iter('prereq', prereqs or []))
            actions = tuple(self._fun_names_iter('action', actions or []))
            self.engine.query.set_rule(
                name, branch, turn, tick, triggers, prereqs, actions
            )
            self.engine._triggers_cache.store(name, branch, turn, tick, triggers)
            self.engine._prereqs_cache.store(name, branch, turn, tick, prereqs)
            self.engine._actions_cache.store(name, branch, turn, tick, actions)

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
                # Overwrites anything already on the funcstore, is that bad?
                setattr(funcstore, v.__name__, v)
                yield v.__name__
            elif v not in funcstore:
                raise KeyError("Function {} not present in {}".format(
                    v, funcstore._tab
                ))
            else:
                yield v

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
        self.triggers = [self.engine.trigger.truth]


class RuleBook(MutableSequence, Signal):
    """A list of rules to be followed for some Character, or a part of it
    anyway.

    """
    def _get_cache(self, branch, turn, tick):
        try:
            return self.engine._rulebooks_cache.retrieve(
                self.name, branch, turn, tick
            )
        except KeyError:
            return []

    def _set_cache(self, branch, turn, tick, v):
        self.engine._rulebooks_cache.store(self.name, branch, turn, tick, v)

    def __init__(self, engine, name):
        super().__init__()
        self.engine = engine
        self.name = name

    def __contains__(self, v):
        return getattr(v, 'name', v) in self._get_cache(*self.engine._btt())

    def __iter__(self):
        return iter(self._get_cache(*self.engine._btt()))

    def __len__(self):
        try:
            return len(self._get_cache(*self.engine._btt()))
        except KeyError:
            return 0

    def __getitem__(self, i):
        return self.engine.rule[self._get_cache(*self.engine._btt())[i]]

    def _coerce_rule(self, v):
        if isinstance(v, Rule):
            return v
        elif isinstance(v, str):
            return self.engine.rule[v]
        else:
            return Rule(self.engine, v)

    def __setitem__(self, i, v):
        v = getattr(v, 'name', v)
        branch, turn, tick = self.engine._nbtt()
        try:
            cache = self._get_cache(branch, turn, tick)
            cache[i] = v
        except KeyError:
            if i != 0:
                raise IndexError
            cache = [v]
            self._set_cache(branch, turn, tick, cache)
        self.engine.query.set_rulebook(self.name, branch, turn, tick, cache)
        self.engine._rulebooks_cache.store(self.name, branch, turn, tick, cache)
        self.engine.rulebook.send(self, i=i, v=v)
        self.send(self, i=i, v=v)

    def insert(self, i, v):
        v = getattr(v, 'name', v)
        branch, turn, tick = self.engine._nbtt()
        try:
            cache = self._get_cache(branch, turn, tick)
            cache.insert(i, v)
        except KeyError:
            if i != 0:
                raise IndexError
            cache = [v]
        self._set_cache(branch, turn, tick, cache)
        self.engine.query.set_rulebook(self.name, branch, turn, tick, cache)
        self.engine.rulebook.send(self, i=i, v=v)
        self.send(self, i=i, v=v)

    def index(self, v, start=0, stop=None):
        if isinstance(v, str):
            try:
                return self._get_cache(*self.engine._btt()).index(v, start, stop)
            except KeyError:
                raise ValueError
        return super().index(v)

    def __delitem__(self, i):
        branch, turn, tick = self.engine._btt()
        try:
            cache = self._get_cache(branch, turn, tick)
        except KeyError:
            raise IndexError
        del cache[i]
        self.engine.query.set_rulebook(self.name, branch, turn, tick, cache)
        self.engine._rulebooks_cache.store(self.name, branch, turn, tick, cache)
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
        if k in self:
            return self[k]
        raise AttributeError

    def __setitem__(self, k, v):
        if isinstance(v, Hashable) and v in self.engine.rule:
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
        if isinstance(k, int):
            self.rulebook[k] = v
        else:
            self.rulebook.append(v)

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


class RuleFollower(ABC):
    """Interface for that which has a rulebook associated, which you can
    get a :class:`RuleMapping` into

    """
    __slots__ = ()

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
        try:
            if n == self._get_rulebook_name():
                return
        except KeyError:
            pass
        self._set_rulebook_name(n)
        self._upd_rulebook()

    def _upd_rulebook(self):
        self._rulebook = self._get_rulebook()

    def _get_rulebook(self):
        return self.engine.rulebook[self._get_rulebook_name()]

    def rules(self):
        if not hasattr(self, 'engine'):
            raise AttributeError("Need an engine before I can get rules")
        return self._rule_mapping.values()

    @abstractmethod
    def _get_rule_mapping(self):
        """Get the :class:`RuleMapping` for my rulebook."""
        raise NotImplementedError

    @abstractmethod
    def _get_rulebook_name(self):
        """Get the name of my rulebook."""
        raise NotImplementedError

    @abstractmethod
    def _set_rulebook_name(self, n):
        """Tell the database that this is the name of the rulebook to use for
        me.

        """
        raise NotImplementedError


class AllRuleBooks(MutableMapping, Signal):
    __slots__ = ['engine', '_cache']

    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self._cache = {}

    def __iter__(self):
        return self.engine._rulebooks_cache.iter_entities(*self.engine._btt())

    def __len__(self):
        return len(list(self))

    def __contains__(self, k):
        return self.engine._rulebooks_cache.contains_entity(k, *self.engine._btt())

    def __getitem__(self, k):
        if k not in self._cache:
            self._cache[k] = RuleBook(self.engine, k)
        return self._cache[k]

    def __setitem__(self, key, value):
        if key not in self._cache:
            self._cache[key] = RuleBook(self.engine, key)
        rb = self._cache[key]
        while len(rb) > 0:
            del rb[0]
        rb.extend(value)

    def __delitem__(self, key):
        self.engine._del_rulebook(key)


class AllRules(MutableMapping, Signal):
    """A mapping of every rule in the game.

    You can use this as a decorator to make a rule and not assign it
    to anything.

    """
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    @reify
    def _cache(self):
        return self.engine._rules_cache

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
                self._cache[k] = Rule(self.engine, k, actions=[v])
                new = self._cache[k]
            else:
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
        """Make a new rule with no actions or anything, and return it."""
        if name in self:
            raise KeyError("Already have rule {}".format(name))
        new = Rule(self.engine, name)
        self._cache[name] = new
        self.send(self, rule=new, active=True)
        return new
