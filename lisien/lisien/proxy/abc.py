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
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import (
	TYPE_CHECKING,
	Hashable,
	Iterator,
	Literal,
	MutableMapping,
	MutableSequence,
	Optional,
)

from blinker import Signal

from ..types import (
	FuncName,
	Key,
	KeyHint,
	RulebookName,
	RulebookPriority,
	RuleName,
	RuleNeighborhood,
	Value,
	ValueHint,
)
from ..wrap import DictWrapper, ListWrapper, SetWrapper

if TYPE_CHECKING:
	from .character import CharacterProxy, PlaceProxy, PortalProxy, ThingProxy
	from .engine import EngineProxy, FuncStoreProxy, RuleMapProxyDescriptor


class CachingProxy(MutableMapping, Signal):
	"""Abstract class for proxies to lisien entities or mappings thereof"""

	_cache: dict
	rulebook: RuleBookProxy
	engine: EngineProxy

	def _worker_check(self):
		self.engine._worker_check()

	def __init__(self):
		super().__init__()
		self.exists = True

	def __bool__(self):
		return bool(self.exists)

	def __iter__(self):
		yield from self._cache

	def __len__(self):
		return len(self._cache)

	def __contains__(self, k: KeyHint) -> bool:
		return k in self._cache

	def __getitem__(self, k: KeyHint) -> Value:
		if k not in self:
			raise KeyError("No such key: {}".format(k))
		return self._cache_get_munge(k, self._cache[k])

	def setdefault(self, k: KeyHint, default: ValueHint = None) -> Value:
		if k not in self:
			if default is None:
				raise KeyError("No such key", k)
			self[k] = default
			return Value(default)
		return self[k]

	def __setitem__(self, k: KeyHint, v: ValueHint) -> None:
		self._worker_check()
		self._set_item(Key(k), Value(v))
		self._cache[k] = self._cache_set_munge(Key(k), Value(v))
		self.send(self, key=k, value=v)

	def __delitem__(self, k: KeyHint) -> None:
		self._worker_check()
		if k not in self:
			raise KeyError("No such key: {}".format(k))
		self._del_item(Key(k))
		if k in self._cache:
			del self._cache[k]
		self.send(self, key=k, value=None)

	@abstractmethod
	def _apply_delta(self, delta):
		raise NotImplementedError("_apply_delta")

	def _cache_get_munge(self, k: Key, v: Value) -> Value:
		return v

	def _cache_set_munge(self, k: Key, v: Value) -> Value:
		return v

	@abstractmethod
	def _set_item(self, k: Key, v: Value) -> None:
		raise NotImplementedError("Abstract method")

	@abstractmethod
	def _del_item(self, k: Key) -> None:
		raise NotImplementedError("Abstract method")


class CachingEntityProxy(CachingProxy):
	"""Abstract class for proxy objects representing lisien entities"""

	name: Hashable

	def _cache_get_munge(
		self, k: Key, v: Value
	) -> DictWrapper | ListWrapper | SetWrapper | Value:
		if isinstance(v, dict):
			return DictWrapper(lambda: self._cache[k], self, k)
		elif isinstance(v, list):
			return ListWrapper(lambda: self._cache[k], self, k)
		elif isinstance(v, set):
			return SetWrapper(lambda: self._cache[k], self, k)
		return v

	def __repr__(self):
		return "<{}({}) {} at {}>".format(
			self.__class__.__name__, self._cache, self.name, id(self)
		)


class FuncListProxy(MutableSequence, Signal):
	def __init__(
		self,
		rule_proxy: RuleProxy,
		key: Literal["triggers", "prereqs", "actions"],
	):
		super().__init__()
		self.rule = rule_proxy
		self._key = key

	def __iter__(self):
		return iter(self.rule._cache.get(self._key, ()))

	def __len__(self):
		return len(self.rule._cache.get(self._key, ()))

	def __getitem__(self, item: int):
		if self._key not in self.rule._cache:
			raise IndexError(item)
		return self.rule._cache[self._key][item]

	def index(
		self,
		value: str | FuncProxy,
		start: int = 0,
		stop: int | ... = ...,
	):
		if not isinstance(value, FuncProxy):
			value = getattr(self._get_store(), value)
		kwargs = {"start": start}
		if stop is not ...:
			kwargs["stop"] = stop
		return super().index(value, **kwargs)

	def _handle_send(self):
		self.rule.engine.handle(
			f"set_rule_{self._key}",
			**{
				"rule": self.rule.name,
				"branching": True,
				self._key: self.rule._nominate(self.rule._cache[self._key]),
			},
		)

	def _get_store(self) -> FuncStoreProxy:
		return getattr(self.rule.engine, self._key[:-1])

	def __setitem__(self, i: int, value: str | FuncProxy):
		if isinstance(value, str):
			value = getattr(self._get_store(), value)
		if self._key in self.rule._cache:
			self.rule._cache[self._key][i] = value
		elif i == 0:
			self.rule._cache[self._key] = [value]
		else:
			raise IndexError(i)
		self._handle_send()

	def __delitem__(self, key: int):
		if self._key not in self.rule._cache:
			raise IndexError(key)
		del self.rule._cache[self._key][key]
		self._handle_send()

	def insert(self, index: int, value: FuncProxy | str):
		if isinstance(value, str):
			value = getattr(self._get_store(), value)
		elif not isinstance(value, FuncProxy):
			setattr(self._get_store(), value.__name__, value)
			value = getattr(self._get_store(), value.__name__)
		self.rule._cache.setdefault(self._key, []).insert(index, value)
		self._handle_send()


class FuncListProxyDescriptor:
	def __init__(self, key):
		self._key = key

	def __get__(self, instance, owner):
		attname = f"_{self._key}_proxy"
		if not hasattr(instance, attname):
			setattr(instance, attname, FuncListProxy(instance, self._key))
		return getattr(instance, attname)

	def __set__(self, instance, value):
		to_set = []
		for v in value:
			if isinstance(v, FuncProxy):
				to_set.append(v)
			elif not isinstance(v, str):
				raise TypeError(f"Need FuncListProxy or str, got {type(v)}")
			else:
				to_set.append(
					getattr(
						getattr(instance.engine, self._key.removesuffix("s")),
						v,
					)
				)
		instance._cache[self._key] = to_set
		self.__get__(instance, None)._handle_send()


class RuleProxy(Signal):
	triggers = FuncListProxyDescriptor("triggers")
	prereqs = FuncListProxyDescriptor("prereqs")
	actions = FuncListProxyDescriptor("actions")

	@staticmethod
	def _nominate(v):
		ret = []
		for whatever in v:
			if hasattr(whatever, "name"):
				ret.append(whatever.name)
			elif hasattr(whatever, "__name__"):
				ret.append(whatever.__name__)
			else:
				assert isinstance(whatever, str), whatever
				ret.append(whatever)
		return ret

	@property
	def _cache(self):
		return self.engine._rules_cache.setdefault(self.name, {})

	@property
	def neighborhood(self) -> RuleNeighborhood:
		if self.name not in self.engine._neighborhood_cache:
			return self.engine._neighborhood_cache.setdefault(
				self.name,
				self.engine.handle("get_rule_neighborhood", rule=self.name),
			)
		return self.engine._neighborhood_cache[self.name]

	@neighborhood.setter
	def neighborhood(self, v: RuleNeighborhood):
		self.engine._worker_check()
		self.engine.handle(
			"set_rule_neighborhood", rule=self.name, neighborhood=v
		)
		self.engine._neighborhood_cache[self.name] = v

	def trigger(self, trigger: callable | FuncProxy) -> FuncProxy:
		self.triggers.append(trigger)
		if isinstance(trigger, FuncProxy):
			return trigger
		else:
			return getattr(self.engine.trigger, trigger.__name__)

	def prereq(self, prereq: callable | FuncProxy) -> FuncProxy:
		self.prereqs.append(prereq)
		if isinstance(prereq, FuncProxy):
			return prereq
		else:
			return getattr(self.engine.prereq, prereq.__name__)

	def action(self, action: callable | FuncProxy) -> FuncProxy:
		self.actions.append(action)
		if isinstance(action, FuncProxy):
			return action
		else:
			return getattr(self.engine.action, action.__name__)

	def __init__(self, engine: EngineProxy, rulename: RuleName):
		super().__init__()
		self.engine = engine
		self.name = self._name = rulename
		if rulename not in engine._rules_cache:
			engine.handle("new_empty_rule", rule=rulename)
			engine._rules_cache[rulename] = {}

	def __eq__(self, other):
		return hasattr(other, "name") and self.name == other.name


class RuleBookProxy(MutableSequence, Signal):
	@property
	def _cache(
		self,
	) -> list[RuleName]:
		no_rules: list[RuleName] = []
		zero_prio = RulebookPriority(0.0)
		return self.engine._rulebooks_cache.setdefault(
			self.name, (no_rules, zero_prio)
		)[0]

	@property
	def priority(self) -> RulebookPriority:
		no_rules: list[RuleName] = []
		zero_prio = RulebookPriority(0.0)
		return self.engine._rulebooks_cache.setdefault(
			self.name, (no_rules, zero_prio)
		)[1]

	def _worker_check(self):
		self.engine._worker_check()

	def __init__(self, engine: EngineProxy, bookname: RulebookName):
		super().__init__()
		self.engine = engine
		self.name = bookname
		self._proxy_cache = engine._rule_obj_cache

	def __iter__(self) -> Iterator[RulebookName]:
		for k in self._cache:
			if k not in self._proxy_cache:
				self._proxy_cache[k] = RuleProxy(self.engine, k)
			yield self._proxy_cache[k]

	def __len__(self):
		return len(self._cache)

	def __getitem__(self, i):
		k = self._cache[i]
		if k not in self._proxy_cache:
			self._proxy_cache[k] = RuleProxy(self.engine, k)
		return self._proxy_cache[k]

	def __setitem__(self, i, v: RuleProxy | str):
		self._worker_check()
		if isinstance(v, RuleProxy):
			v = v._name
		self._cache[i] = v
		self.engine.handle(
			command="set_rulebook_rule",
			rulebook=self.name,
			i=i,
			rule=v,
			branching=True,
		)
		self.send(self, i=i, val=v)

	def __delitem__(self, i):
		self._worker_check()
		del self._cache[i]
		self.engine.handle(
			command="del_rulebook_rule",
			rulebook=self.name,
			i=i,
			branching=True,
		)
		self.send(self, i=i, val=None)

	def insert(self, i: int, v: RuleProxy | str):
		self._worker_check()
		if isinstance(v, RuleProxy):
			v = v._name
		self._cache.insert(i, v)
		self.engine.handle(
			command="ins_rulebook_rule",
			rulebook=self.name,
			i=i,
			rule=v,
			branching=True,
		)
		for j in range(i, len(self)):
			self.send(self, i=j, val=self[j])


class RuleFollowerProxyDescriptor:
	def __set__(self, inst, val: RuleBookProxy | RuleMapProxy):
		inst.engine._worker_check()
		if isinstance(val, RuleBookProxy):
			rb = val
			val = val.name
		elif isinstance(val, RuleMapProxy):
			if val.name in inst.engine._rulebooks_cache:
				val = val.name
			else:
				inst.engine._rulebooks_cache[val.name] = RuleBookProxy(
					inst.engine, val.name
				)
				val = val.name
		inst._set_rulebook_name(val)
		if hasattr(inst, "send"):
			inst.send(inst, rulebook=val)


class RulebookProxyDescriptor(RuleFollowerProxyDescriptor):
	"""Descriptor that makes the corresponding RuleBookProxy if needed"""

	def __get__(self, inst, cls):
		if inst is None:
			return self
		return inst._get_rulebook_proxy()


class RuleMapProxyDescriptor(RuleFollowerProxyDescriptor):
	def __get__(self, instance, owner):
		if instance is None:
			return self
		if hasattr(instance, "_rule_map_proxy"):
			return instance._rule_map_proxy
		else:
			try:
				name = instance._get_rulebook_name()
			except KeyError:
				name = instance._get_default_rulebook_name()
			proxy = instance._rule_map_proxy = RuleMapProxy(instance, name)
		return proxy


class RuleFollowerProxy(ABC):
	rule = RuleMapProxyDescriptor()
	rulebook = RulebookProxyDescriptor()
	engine: "EngineProxy"

	@abstractmethod
	def _get_default_rulebook_name(self) -> RulebookName:
		pass

	@abstractmethod
	def _get_rulebook_name(self) -> RulebookName:
		pass

	def _worker_check(self):
		self.engine._worker_check()

	def _get_rulebook_proxy(self) -> RuleBookProxy:
		try:
			name = self._get_rulebook_name()
		except KeyError:
			name = self._get_default_rulebook_name()
		if name not in self.engine._rulebook_obj_cache:
			self.engine._rulebook_obj_cache[name] = RuleBookProxy(
				self.engine, name
			)
		return self.engine._rulebook_obj_cache[name]

	@abstractmethod
	def _set_rulebook_name(self, rb: RulebookName) -> None:
		pass


class FuncProxy(object):
	__slots__ = "store", "name"

	def __init__(self, store: FuncStoreProxy, func: FuncName):
		self.store = store
		self.name = func

	def __call__(
		self,
		*args: Value,
		cb: Optional[callable] = None,
		**kwargs: dict[Key, Value],
	):
		return self.store.engine.handle(
			"call_stored_function",
			store=self.store._store,
			func=self.name,
			args=args[1:] if self.store._store == "method" else args,
			kwargs=kwargs,
			cb=partial(self.store.engine._upd_and_cb, cb=cb),
		)[0]

	def __str__(self):
		return self.store._cache[self.name]


class RuleMapProxy(MutableMapping, Signal):
	@property
	def _cache(self) -> list[RuleName]:
		no_rules: list[RuleName] = []
		zero_prio = RulebookPriority(0.0)
		return self.engine._rulebooks_cache.setdefault(
			self.name, (no_rules, zero_prio)
		)[0]

	@property
	def priority(self) -> RulebookPriority:
		no_rules: list[RuleName] = []
		zero_prio = RulebookPriority(0.0)
		return self.engine._rulebooks_cache.setdefault(
			self.name, (no_rules, zero_prio)
		)[1]

	@priority.setter
	def priority(self, v: float):
		self.engine.handle(
			"set_rulebook_priority", rulebook=self.name, priority=v
		)
		if self.name in self.engine._rulebooks_cache:
			rules, _ = self.engine._rulebooks_cache[self.name]
			self.engine._rulebooks_cache[self.name] = (rules, v)
		else:
			self.engine._rulebooks_cache[self.name] = ([], v)

	@property
	def engine(self) -> EngineProxy:
		return self.entity.engine

	def _worker_check(self) -> None:
		self.engine._worker_check()

	def __init__(
		self,
		entity: CharacterProxy | ThingProxy | PlaceProxy | PortalProxy,
		rulebook_name: RulebookName,
	):
		super().__init__()
		self.entity = entity
		self.name = rulebook_name
		self._proxy_cache = entity.engine._rule_obj_cache

	def __iter__(self) -> Iterator[RuleName]:
		return iter(self._cache)

	def __len__(self) -> int:
		return len(self._cache)

	def __call__(
		self,
		action: str | callable | FuncProxy | None = None,
		always: bool = False,
		neighborhood: Optional[RuleNeighborhood] = None,
	) -> callable | RuleProxy:
		if action is None:
			return partial(self, always=always, neighborhood=neighborhood)
		self._worker_check()
		if isinstance(action, FuncProxy):
			rule_name = RuleName(action.name)
			self[rule_name] = [action.name]
		else:
			if callable(action):
				self.engine.action(action)
				action = action.__name__
			self[action] = [action]
		ret = self[action]
		if neighborhood is not None:
			ret.neighborhood = neighborhood
		if always:
			ret.triggers.append(self.engine.trigger.truth)
		return ret

	def __getitem__(self, key: str) -> RuleProxy:
		if key in self._cache:
			if key not in self._proxy_cache:
				self._proxy_cache[key] = RuleProxy(self.engine, key)
			return self._proxy_cache[key]
		raise KeyError("Rule not assigned to rulebook", key, self.name)

	def __setitem__(self, k: str, v: RuleProxy | list[FuncProxy | str]):
		self._worker_check()
		if self.name not in self.engine._rulebooks_cache:
			self.engine.handle("new_empty_rulebook", rulebook=self.name)
			self.engine._rulebooks_cache[self.name] = ([], 0.0)
			self.entity._set_rulebook_name(self.name)
		if isinstance(v, RuleProxy):
			v = v._name
		else:
			if k not in self.engine._rule_obj_cache:
				rp = self.engine._rule_obj_cache[k] = RuleProxy(self.engine, k)
			else:
				rp = self.engine._rule_obj_cache[k]
			rp.actions = v
			v = k
		if k in self._cache:
			return
		i = len(self._cache)
		self._cache.append(k)
		self.engine.handle(
			command="set_rulebook_rule",
			rulebook=self.name,
			i=i,
			rule=v,
			branching=True,
		)
		self.send(self, key=k, val=v)

	def __delitem__(self, key: str):
		self._worker_check()
		i = self._cache.index(RuleName(key))
		if i is None:
			raise KeyError("Rule not set in rulebook", key, self.name)
		del self._cache[i]
		self.engine.handle(
			command="del_rulebook_rule",
			rulebook=self.name,
			i=i,
			branching=True,
		)
		self.send(self, key=key, val=None)
