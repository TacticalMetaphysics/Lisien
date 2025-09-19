# This file is part of Lisien, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3.
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

To add a new rule to a lisien entity, the easiest thing is to use the
decorator syntax::

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



When run, this code will:

	* copy the ``do_something`` function to ``action.py``, where lisien knows\
			to run it when a rule triggers it

	* create a new rule named ``'do_something'``

	* set the function ``do_something`` as the first (and, so far, only) entry\
			in the actions list of the rule by that name

	* copy the ``whenever`` function to ``trigger.py``, where lisien knows to\
			call it when a rule has it as a trigger

	* set the function ``whenever`` as the first entry in the triggers list\
			of the rule ``'do_something'``

	* append the function ``forever`` to the same triggers list

	* copy ``do_something_else`` to ``action.py``

	* append ``do_something_else`` to the actions list of the rule

The ``trigger``, ``prereq``, and ``action`` attributes of Rule objects
may also be used like lists. You can put functions in them yourself,
provided they are already present in the correct module. If it's
inconvenient to get the actual function object, use a string of
the function's name.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from ast import parse
from collections.abc import Hashable, Iterable, MutableMapping, MutableSequence
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal, Optional

from astunparse import unparse
from blinker import Signal

from .cache import FuncListCache
from .collections import FunctionStore
from .types import (
	ActionFuncName,
	Branch,
	Key,
	PrereqFuncName,
	RuleBig,
	RulebookName,
	RulebookPriority,
	RuleFunc,
	RuleFuncName,
	RuleName,
	RuleNeighborhood,
	Tick,
	TriggerFuncName,
	Turn,
	Value,
)
from .util import AbstractEngine, dedent_source

if TYPE_CHECKING:
	from .engine import Engine


def roundtrip_dedent(source):
	"""Reformat some lines of code into what unparse makes."""
	return unparse(parse(dedent_source(source)))


class RuleFuncList(MutableSequence, Signal, ABC):
	"""Abstract class for lists of functions like trigger, prereq, action"""

	__slots__ = ["rule"]
	_funcstore: FunctionStore
	_cache: FuncListCache
	_setter: Callable[[RuleName, Branch, Turn, Tick, list[RuleFuncName]], None]

	def __init__(self, rule: Rule):
		super().__init__()
		self.rule = rule

	def __repr__(self):
		return f"<class 'lisien.rule.{self.__class__.__name__}' [{', '.join(self._get())}]>"

	def _nominate(self, v: RuleFunc | str | RuleFuncName) -> RuleFuncName:
		if callable(v):
			self._funcstore(v)
			return v.__name__
		if not hasattr(self._funcstore, v):
			raise KeyError(
				"No function by that name in this store", v, self._funcstore
			)
		return v

	def _get(self) -> list[RuleFuncName]:
		try:
			return self._cache.retrieve(
				self.rule.name, *self.rule.engine._btt()
			)
		except KeyError:
			return []

	def _set(self, v: list[RuleFuncName]) -> None:
		branch, turn, tick = self.rule.engine._nbtt()
		self._cache.store(Key(self.rule.name), branch, turn, tick, v)
		self._setter(self.rule.name, branch, turn, tick, v)

	def __iter__(self):
		for funcname in self._get():
			yield getattr(self._funcstore, funcname)

	def __contains__(self, item: RuleFunc | str | RuleFuncName):
		if hasattr(item, "__name__"):
			item = item.__name__
		return item in self._get()

	def __len__(self):
		return len(self._get())

	def __getitem__(self, i: int):
		return getattr(self._funcstore, self._get()[i])

	def __setitem__(self, i: int, v: RuleFunc | str | RuleFuncName):
		v = self._nominate(v)
		l = list(self._get())
		l[i] = v
		self._set(list(l))
		self.send(self)

	def __delitem__(self, i: int):
		l = list(self._get())
		del l[i]
		self._set(list(l))
		self.send(self)

	def insert(self, i: int, v: RuleFunc | str | RuleFuncName) -> None:
		l = list(self._get())
		l.insert(i, self._nominate(v))
		self._set(list(l))
		self.send(self)

	def append(self, v: RuleFunc | str | RuleFuncName) -> None:
		try:
			old = self._get()
		except KeyError:
			old = []
		self._set(old + [self._nominate(v)])
		self.send(self)

	def index(
		self,
		x: RuleFunc | str | RuleFuncName,
		start: int = 0,
		end: int | None = None,
	):
		if not callable(x):
			x = getattr(self._funcstore, x)
		return super().index(x, start, end)


class TriggerList(RuleFuncList):
	"""A list of trigger functions for rules"""

	@cached_property
	def _funcstore(self) -> FunctionStore:
		return self.rule.engine.trigger

	@cached_property
	def _cache(self) -> FuncListCache:
		return self.rule.engine._triggers_cache

	@cached_property
	def _setter(
		self,
	) -> Callable[[RuleName, Branch, Turn, Tick, list[TriggerFuncName]], None]:
		return self.rule.engine.query.set_rule_triggers


class PrereqList(RuleFuncList):
	"""A list of prereq functions for rules"""

	@cached_property
	def _funcstore(self) -> FunctionStore:
		return self.rule.engine.prereq

	@cached_property
	def _cache(self) -> FuncListCache:
		return self.rule.engine._prereqs_cache

	@cached_property
	def _setter(
		self,
	) -> Callable[[RuleName, Branch, Turn, Tick, list[PrereqFuncName]], None]:
		return self.rule.engine.query.set_rule_prereqs


class ActionList(RuleFuncList):
	"""A list of action functions for rules"""

	@cached_property
	def _funcstore(self) -> FunctionStore:
		return self.rule.engine.action

	@cached_property
	def _cache(self) -> FuncListCache:
		return self.rule.engine._actions_cache

	@cached_property
	def _setter(
		self,
	) -> Callable[[RuleName, Branch, Turn, Tick, list[ActionFuncName]], None]:
		return self.rule.engine.query.set_rule_actions


class RuleFuncListDescriptor:
	"""Descriptor that lets you get and set a whole RuleFuncList at once"""

	__slots__ = ("cls",)

	def __init__(self, cls):
		self.cls = cls

	@property
	def flid(self):
		return "_funclist" + str(id(self))

	def __get__(self, obj: Rule, type=None) -> RuleFuncList:
		if not hasattr(obj, self.flid):
			setattr(obj, self.flid, self.cls(obj))
		return getattr(obj, self.flid)

	def __set__(self, obj: Rule, value: list[RuleFunc | str | RuleFuncName]):
		if not hasattr(obj, self.flid):
			setattr(obj, self.flid, self.cls(obj))
		flist = getattr(obj, self.flid)
		namey_value = [flist._nominate(v) for v in value]
		flist._set(namey_value)
		flist.send(flist)

	def __delete__(self, obj):
		raise TypeError("Rules must have their function lists")


class Rule:
	"""Stuff that might happen in the simulation under some conditions

	Rules are comprised of three lists of functions:

	* actions, which mutate the world state
	* triggers, which make the actions happen
	* prereqs, which prevent the actions from happening when triggered

	Each kind of function should be stored in the appropriate module
	supplied to the lisien core at startup. This makes it possible to
	load the functions on later startups. You may instead use the string
	name of a function already stored in the module, or use the
	``trigger``, ``prereq``, or ``action`` decorator on a new function to
	add it to both the module and the rule.

	"""

	triggers: TriggerList = RuleFuncListDescriptor(TriggerList)
	prereqs: PrereqList = RuleFuncListDescriptor(PrereqList)
	actions: ActionList = RuleFuncListDescriptor(ActionList)
	name: RuleName

	@property
	def neighborhood(self) -> RuleNeighborhood:
		try:
			ret = self.engine._neighborhoods_cache.retrieve(
				self.name, *self.engine._btt()
			)
		except KeyError:
			return None
		return ret

	@neighborhood.setter
	def neighborhood(self, neighbors: RuleNeighborhood):
		if neighbors is not None and not isinstance(neighbors, int):
			raise TypeError("Not an int", neighbors)
		if neighbors < 0:
			raise ValueError("Can't be negative", neighbors)
		btt = self.engine._nbtt()
		self.engine._neighborhoods_cache.store(
			Key(self.name), *btt, Value(neighbors)
		)
		self.engine.query.set_rule_neighborhood(self.name, *btt, neighbors)

	@property
	def big(self) -> RuleBig:
		try:
			return self.engine._rule_bigness_cache.retrieve(
				self.name, *self.engine._btt()
			)
		except KeyError:
			return RuleBig(False)

	@big.setter
	def big(self, big: bool | RuleBig):
		btt = self.engine._nbtt()
		self.engine._rule_bigness_cache.store(Key(self.name), *btt, Value(big))
		self.engine.query.set_rule_big(self.name, *btt, big)

	def __init__(
		self,
		engine: "Engine",
		name: RuleName | str,
		triggers: list[TriggerFuncName | Callable[[Any], bool] | str]
		| None = None,
		prereqs: list[PrereqFuncName | Callable[[Any], bool] | str]
		| None = None,
		actions: list[ActionFuncName | Callable[[Any], bool] | str]
		| None = None,
		neighborhood: RuleNeighborhood = None,
		big: RuleBig = RuleBig(False),
		create: bool = True,
	):
		"""Store the engine and my name, make myself a record in the database
		if needed, and instantiate one FunList each for my triggers,
		actions, and prereqs.

		"""
		self.engine = engine
		self.name = self.__name__ = RuleName(name)
		# It'd be more correct to subclass InitializedEntitylessCache and
		# give all its methods more specific signatures that take RuleName.
		name = Key(name)
		if create:
			branch, turn, tick = engine._nbtt()
			if (
				self.engine._triggers_cache.contains_key(
					name, branch, turn, tick
				)
				or self.engine._prereqs_cache.contains_key(
					name, branch, turn, tick
				)
				or self.engine._actions_cache.contains_key(
					name, branch, turn, tick
				)
			):
				(branch, turn, tick) = self.engine._nbtt()
			triggers = list(self._fun_names_iter("trigger", triggers or []))
			prereqs = list(self._fun_names_iter("prereq", prereqs or []))
			actions = list(self._fun_names_iter("action", actions or []))
			self.engine.query.create_rule(
				name,
				branch,
				turn,
				tick,
				triggers=triggers,
				prereqs=prereqs,
				actions=actions,
				neighborhood=neighborhood,
				big=big,
			)
			self.engine._triggers_cache.store(
				name, branch, turn, tick, triggers
			)
			self.engine._prereqs_cache.store(name, branch, turn, tick, prereqs)
			self.engine._actions_cache.store(name, branch, turn, tick, actions)
			self.engine._neighborhoods_cache.store(
				name, branch, turn, tick, neighborhood
			)
			self.engine._rule_bigness_cache.store(
				name, branch, turn, tick, big
			)
			# Don't *make* a keyframe -- but if there happens to already *be*
			# a keyframe at this very moment, add the new rule to it
			if (branch, turn, tick) in self.engine._keyframes_times:
				# ensure it's loaded
				self.engine._get_keyframe(branch, turn, tick, silent=True)
				# Just because there's a keyframe doesn't mean it's in every cache.
				# I should probably change that.
				try:
					trigkf = self.engine._triggers_cache.get_keyframe(
						branch, turn, tick
					)
				except KeyError:
					trigkf = {
						aname: self.engine._triggers_cache.retrieve(
							aname, branch, turn, tick
						)
						for aname in self.engine._triggers_cache.iter_keys(
							branch, turn, tick
						)
					}
				try:
					preqkf = self.engine._prereqs_cache.get_keyframe(
						branch, turn, tick
					)
				except KeyError:
					preqkf = {
						aname: self.engine._prereqs_cache.retrieve(
							aname, branch, turn, tick
						)
						for aname in self.engine._prereqs_cache.iter_keys(
							branch, turn, tick
						)
					}
				try:
					actkf = self.engine._actions_cache.get_keyframe(
						branch, turn, tick
					)
				except KeyError:
					actkf = {
						aname: self.engine._actions_cache.retrieve(
							aname, branch, turn, tick
						)
						for aname in self.engine._actions_cache.iter_keys(
							branch, turn, tick
						)
					}
				try:
					nbrkf = self.engine._neighborhoods_cache.get_keyframe(
						branch, turn, tick
					)
				except KeyError:
					nbrkf = {
						aname: self.engine._neighborhoods_cache.retrieve(
							aname, branch, turn, tick
						)
						for aname in self.engine._neighborhoods_cache.iter_keys(
							branch, turn, tick
						)
					}
				try:
					bigkf = self.engine._rule_bigness_cache.get_keyframe(
						branch, turn, tick
					)

				except KeyError:
					bigkf = {
						aname: self.engine._rule_bigness_cache.retrieve(
							aname, branch, turn, tick
						)
						for aname in self.engine._rule_bigness_cache.iter_keys(
							branch, turn, tick
						)
					}
				trigkf[name] = triggers
				preqkf[name] = prereqs
				actkf[name] = actions
				nbrkf[name] = neighborhood
				bigkf[name] = big
				self.engine._triggers_cache.set_keyframe(
					branch, turn, tick, trigkf
				)
				self.engine._prereqs_cache.set_keyframe(
					branch, turn, tick, preqkf
				)
				self.engine._actions_cache.set_keyframe(
					branch, turn, tick, actkf
				)
				self.engine._neighborhoods_cache.set_keyframe(
					branch, turn, tick, nbrkf
				)
				self.engine._rule_bigness_cache.set_keyframe(
					branch, turn, tick, bigkf
				)

	def __eq__(self, other: Rule):
		return isinstance(other, Rule) and self.name == other.name

	def _fun_names_iter(
		self,
		functyp: Literal["trigger", "prereq", "action"],
		val: Iterable[RuleFunc | str | RuleFuncName],
	):
		"""Iterate over the names of the functions in ``val``,
		adding them to ``funcstore`` if they are missing;
		or if the items in ``val`` are already the names of functions
		in ``funcstore``, iterate over those.

		"""
		funcstore = getattr(self.engine, functyp)
		for v in val:
			if callable(v):
				if v != getattr(funcstore, v.__name__, None):
					setattr(funcstore, v.__name__, v)
				yield v.__name__
			elif v not in funcstore:
				raise KeyError(
					"Function {} not present in {}".format(v, funcstore._tab)
				)
			else:
				yield v

	def __repr__(self):
		return f"<Rule({repr(self.name)})>"

	def trigger(
		self, fun: RuleFunc | str | TriggerFuncName
	) -> RuleFunc | str | TriggerFuncName:
		"""Decorator to append the function to my triggers list."""
		self.triggers.append(fun)
		return fun

	def prereq(
		self, fun: RuleFunc | str | PrereqFuncName
	) -> RuleFunc | str | PrereqFuncName:
		"""Decorator to append the function to my prereqs list."""
		self.prereqs.append(fun)
		return fun

	def action(
		self, fun: RuleFunc | str | ActionFuncName
	) -> RuleFunc | str | ActionFuncName:
		"""Decorator to append the function to my actions list."""
		self.actions.append(fun)
		return fun

	def always(self) -> None:
		"""Arrange to be triggered every turn"""
		self.triggers = [self.engine.trigger.truth]


class RuleBook(MutableSequence, Signal):
	"""A list of rules to be followed for some Character, or a part of it"""

	def _get_cache(
		self, branch: Branch, turn: Turn, tick: Tick
	) -> tuple[list[RuleName], RulebookPriority]:
		rules: list[RuleName] = []
		prio = RulebookPriority(0.0)
		try:
			rules, prio = self.engine._rulebooks_cache.retrieve(
				self.name, branch, turn, tick
			)
		except KeyError:
			pass
		return list(rules), prio

	def _set_cache(
		self,
		branch: Branch,
		turn: Turn,
		tick: Tick,
		v: tuple[list[RuleName], RulebookPriority],
	) -> None:
		with self.engine._rulebooks_cache.overwriting():
			self.engine._rulebooks_cache.store(
				self.name, branch, turn, tick, v
			)

	def __init__(self, engine: "Engine", name: RulebookName):
		super().__init__()
		self.engine = engine
		self.name = name

	@property
	def priority(self) -> RulebookPriority:
		return self._get_cache(*self.engine._btt())[1]

	@priority.setter
	def priority(self, v: float | RulebookPriority):
		v = RulebookPriority(float(v))
		branch, turn, tick = self.engine._nbtt()
		cache, _ = self._get_cache(branch, turn, tick)
		self._set_cache(branch, turn, tick, (cache, v))
		self.engine.query.set_rulebook(self.name, branch, turn, tick, cache, v)

	def __contains__(self, v: RuleName | Rule) -> bool:
		return getattr(v, "name", v) in self._get_cache(*self.engine._btt())[0]

	def __iter__(self) -> Iterator[Rule]:
		rule_name: RuleName
		for rule_name in self._get_cache(*self.engine._btt())[0]:
			yield self.engine.rule[rule_name]

	def __len__(self):
		try:
			return len(self._get_cache(*self.engine._btt())[0])
		except KeyError:
			return 0

	def __getitem__(self, i: int) -> Rule:
		return self.engine.rule[self._get_cache(*self.engine._btt())[0][i]]

	def _coerce_rule(self, v: Rule | str | RuleName) -> Rule:
		if isinstance(v, Rule):
			return v
		elif isinstance(v, str):
			return self.engine.rule[v]
		else:
			raise TypeError("Invalid rule name", v)

	def __setitem__(self, i: int, v: Rule | str | RuleName):
		v = RuleName(getattr(v, "name", v))
		if v == "truth":
			raise ValueError("Illegal rule name")
		branch, turn, tick = self.engine._nbtt()
		cache, prio = self._get_cache(branch, turn, tick)
		try:
			cache[i] = v
		except IndexError:
			if i != len(cache):
				raise
			if i == 0:
				cache = [v]
			else:
				assert i == len(cache)
				cache.append(v)
			prio = RulebookPriority(0.0)
			self._set_cache(branch, turn, tick, (cache, prio))
		self.engine.query.set_rulebook(
			self.name, branch, turn, tick, cache, prio
		)
		self.engine.rulebook.send(self, i=i, v=v)
		self.send(self, i=i, v=v)

	def insert(self, i: int, v: Rule | str | RuleName) -> None:
		v = RuleName(getattr(v, "name", v))
		if v == "truth":
			raise ValueError("Illegal rule name")
		branch, turn, tick = self.engine._nbtt()
		try:
			cache, prio = self._get_cache(branch, turn, tick)
			cache.insert(i, v)
		except KeyError:
			if i != 0:
				raise IndexError
			cache = [v]
			prio = RulebookPriority(0.0)
		self._set_cache(branch, turn, tick, (cache, prio))
		self.engine.query.set_rulebook(
			self.name, branch, turn, tick, cache, prio
		)
		self.engine.rulebook.send(self, i=i, v=v)
		self.send(self, i=i, v=v)

	def index(
		self, v: Rule | str | RuleName, start: int = 0, stop: int | None = None
	) -> int:
		args = [v, start]
		if stop is not None:
			args.append(stop)
		if isinstance(v, str):
			try:
				return self._get_cache(*self.engine._btt())[0].index(*args)
			except KeyError:
				raise ValueError("Not the name of a rule", v)
		return super().index(*args)

	def __delitem__(self, i: int) -> None:
		branch, turn, tick = self.engine._btt()
		try:
			cache, prio = self._get_cache(branch, turn, tick)
		except KeyError:
			raise IndexError
		del cache[i]
		self.engine.query.set_rulebook(
			self.name, branch, turn, tick, cache, prio
		)
		self._set_cache(branch, turn, tick, (cache, prio))
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

	:param name: If you want the rule's name to be different from the name
		of its first action, supply the name here.
	:param neighborhood: Optional integer; if supplied, the rule will only
		be run when something's changed within this many nodes.
		``neighborhood=0`` means this only runs when something's changed
		*here*, or a place containing this entity.
	:param big: Set to ``True`` if the rule will make many changes to the world,
		so that Lisien can optimize for a big batch of changes.
	:param always: If set to ``True``, the rule will run every turn.

	"""

	rulebook: RuleBook

	def __init__(self, engine: "Engine", rulebook: RuleBook | RulebookName):
		super().__init__()
		self.engine = engine
		self._rule_cache = self.engine.rule._cache
		if isinstance(rulebook, RuleBook):
			self.rulebook = rulebook
		else:
			self.rulebook = self.engine.rulebook[rulebook]

	def __repr__(self):
		return "RuleMapping({})".format([k for k in self])

	def __iter__(self):
		return iter(self.rulebook._get_cache(*self.engine._btt())[0])

	def __len__(self):
		return len(self.rulebook)

	def __contains__(self, k: Rule | RuleName | str) -> bool:
		return k in self.rulebook

	def __getitem__(self, k: RuleName | str) -> Rule:
		if k not in self:
			raise KeyError("Rule '{}' is not in effect".format(k))
		return self._rule_cache[k]

	def __getattr__(self, k: str) -> Rule:
		if k in self:
			return self[k]
		raise AttributeError

	def __setitem__(
		self, k: RuleName | str, v: Rule | RuleFunc | RuleName | str
	):
		if k == "truth":
			raise KeyError("Illegal rule name")
		if isinstance(v, Hashable) and v in self.engine.rule:
			v = self.engine.rule[v]
		elif isinstance(v, str) and hasattr(self.engine.action, v):
			v = getattr(self.engine.action, v)
		if not isinstance(v, Rule) and callable(v):
			# create a new rule, named k, performing action v
			self.engine.rule[k] = v
			v = self.engine.rule[k]
		assert isinstance(v, Rule)
		if isinstance(k, int):
			self.rulebook[k] = v
		else:
			self.rulebook.append(v)

	def __call__(
		self,
		v: Optional[RuleFunc] = None,
		name: Optional[RuleName | str] = None,
		*,
		neighborhood: Optional[int] = ...,
		big: bool = False,
		always: bool = False,
	):
		def wrap(name, v: RuleFunc, **kwargs):
			name = name if name is not None else v.__name__
			if name == "truth":
				raise ValueError("Illegal rule name")
			self[name] = v
			r = self[name]
			if kwargs.get("always"):
				r.always()
			if "neighborhood" in kwargs:
				r.neighborhood = kwargs["neighborhood"]
			if "big" in kwargs:
				r.big = kwargs["big"]
			return r

		kwargs: dict[str, Any] = {"big": big}
		if always:
			kwargs["always"] = True
		if neighborhood is not ...:
			kwargs["neighborhood"] = neighborhood
		if v is None:
			return partial(wrap, name, **kwargs)
		return wrap(name, v, **kwargs)

	def __delitem__(self, k: RuleName | str):
		i = self.rulebook.index(k)
		del self.rulebook[i]
		self.send(self, key=k, val=None)

	@property
	def priority(self) -> RulebookPriority:
		return self.rulebook.priority

	@priority.setter
	def priority(self, v: RulebookPriority | float):
		self.rulebook.priority = RulebookPriority(v)


class RuleFollower(ABC):
	"""Interface for that which has a rulebook associated, which you can
	get a :class:`RuleMapping` into

	"""

	__slots__ = ()
	engine: AbstractEngine
	_rulebook: RuleBook

	@cached_property
	def _rule_mapping(self) -> RuleMapping:
		return self._get_rule_mapping()

	@property
	def rule(
		self, v: RuleFunc | None = None, name: RuleName | str | None = None
	) -> RuleMapping | partial[Rule]:
		if v is not None:
			return self._rule_mapping(v, name)
		return self._rule_mapping

	@property
	def rulebook(self) -> RuleBook:
		if not hasattr(self, "_rulebook"):
			self._upd_rulebook()
		return self._rulebook

	@rulebook.setter
	def rulebook(self, v: RuleBook | RulebookName | str):
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

	def _get_rulebook(self) -> RuleBook:
		return self.engine.rulebook[self._get_rulebook_name()]

	def rules(self) -> Iterator[Rule]:
		if not hasattr(self, "engine"):
			raise AttributeError("Need an engine before I can get rules")
		return self._rule_mapping.values()

	@abstractmethod
	def _get_rule_mapping(self):
		"""Get the :class:`RuleMapping` for my rulebook."""
		raise NotImplementedError("_get_rule_mapping")

	@abstractmethod
	def _get_rulebook_name(self):
		"""Get the name of my rulebook."""
		raise NotImplementedError("_get_rulebook_name")

	@abstractmethod
	def _set_rulebook_name(self, n: RulebookName | str):
		"""Tell the database that this is the name of the rulebook to use for
		me.

		"""
		raise NotImplementedError("_set_rulebook_name")


class AllRuleBooks(MutableMapping, Signal):
	__slots__ = ["engine", "_cache"]

	def __init__(self, engine: "Engine"):
		super().__init__()
		self.engine = engine
		self._cache = {}

	def __iter__(self) -> Iterator[RulebookName]:
		return self.engine._rulebooks_cache.iter_entities(*self.engine._btt())

	def __len__(self):
		return len(list(self))

	def __contains__(self, k: RulebookName | Key):
		return self.engine._rulebooks_cache.contains_entity(
			k, *self.engine._btt()
		)

	def __getitem__(self, k: RulebookName | Key) -> RuleBook:
		if k not in self._cache:
			self._cache[k] = RuleBook(self.engine, k)
		return self._cache[k]

	def __setitem__(
		self, key: RulebookName | Key, value: RuleBook | list[Rule | RuleName]
	):
		if key not in self._cache:
			self._cache[key] = RuleBook(self.engine, key)
		rb = self._cache[key]
		while len(rb) > 0:
			del rb[0]
		rb.extend(value)

	def __delitem__(self, key: RulebookName | Key):
		self.engine._del_rulebook(key)


class AllRules(MutableMapping, Signal):
	"""A mapping of every rule in the game.

	You can use this as a decorator to make a rule and not assign it
	to anything.

	"""

	def __init__(self, engine: "Engine"):
		super().__init__()
		self.engine = engine

	@cached_property
	def _cache(self) -> dict[RuleName, Rule]:
		return self.engine._rules_cache

	def __iter__(self) -> Iterator[RuleName]:
		yield from self._cache

	def __len__(self):
		return len(self._cache)

	def __contains__(self, k: RuleName | str):
		return k in self._cache

	def __getitem__(self, k: RuleName | str) -> Rule:
		return self._cache[k]

	def __setitem__(
		self, k: RuleName | str, v: Rule | RuleFunc | RuleFuncName | str
	):
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
			self._cache[k] = Rule(self.engine, k, actions=[v])
			new = self._cache[k]
		elif isinstance(v, Rule):
			self._cache[k] = v
			new = v
		else:
			raise TypeError(
				"Don't know how to store {} as a rule.".format(type(v))
			)
		self.send(self, key=new, rule=v)

	def __delitem__(self, k: RuleName | Key):
		if k not in self:
			raise KeyError("No such rule")
		for rulebook in self.engine.rulebooks.values():
			try:
				del rulebook[rulebook.index(k)]
			except IndexError:
				pass
		del self._cache[k]
		self.send(self, key=k, rule=None)

	def __call__(
		self,
		v: RuleFunc | RuleFuncName | str | None = None,
		name: RuleName | str | None = None,
		*,
		neighborhood: RuleNeighborhood | int | None = ...,
		always: bool = False,
	):
		def r(name: RuleName | str | None, v: RuleFunc, **kwargs):
			if name is None:
				name = v.__name__
			if name == "truth":
				raise ValueError("Illegal rule name")
			self[name] = v
			ret = self[name]
			if kwargs.get("always"):
				ret.triggers.append("truth")
			if "neighborhood" in kwargs:
				ret.neighborhood = neighborhood
			return ret

		kwargs = {}
		if always:
			kwargs["always"] = True
		if neighborhood is not ...:
			kwargs["neighborhood"] = neighborhood
		if v is None:
			return partial(r, name, **kwargs)
		return r(name, v, **kwargs)

	def new_empty(self, name: RuleName | str) -> Rule:
		"""Make a new rule with no actions or anything, and return it."""
		if name in self:
			raise KeyError("Already have rule {}".format(name))
		new = Rule(self.engine, name)
		self._cache[name] = new
		self.send(self, rule=new, active=True)
		return new
