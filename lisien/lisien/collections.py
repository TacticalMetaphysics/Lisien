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
"""Common classes for collections in lisien

Notably includes wrappers for mutable objects, allowing them to be stored in
the database. These simply store the new value.

Most of these are subclasses of :class:`blinker.Signal`, so you can listen
for changes using the ``connect(..)`` method.

"""

from __future__ import annotations

import ast
import base64
import importlib.util
import json
import os
import sys
from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import MutableMapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from hashlib import blake2b
from inspect import getsource
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Callable, Iterator, TypeVar

import networkx as nx
from blinker import Signal

from .types import (
	AbstractEngine,
	AbstractFunctionStore,
	ActionFunc,
	ActionFuncName,
	CharName,
	KeyHint,
	PrereqFunc,
	PrereqFuncName,
	Stat,
	TriggerFunc,
	TriggerFuncName,
	UniversalKey,
	Value,
	ValueHint,
	sort_set,
)
from .util import dedent_source, getatt
from .wrap import wrapval

if TYPE_CHECKING:
	from .character import Character


# 0x241d is the group separator
# 0x241e is the record separator
# per Unicode 1.1
GROUP_SEP = chr(0x241D).encode()
REC_SEP = chr(0x241E).encode()


class AbstractLanguageDescriptor(Signal, ABC):
	@abstractmethod
	def _get_language(self, inst: StringStore) -> str:
		pass

	@abstractmethod
	def _set_language(self, inst: StringStore, val: str) -> None:
		pass

	def __get__(self, instance: StringStore, owner=None):
		return self._get_language(instance)

	def __set__(self, inst: StringStore, val: str):
		self._set_language(inst, val)
		self.send(inst, language=val)


class LanguageDescriptor(AbstractLanguageDescriptor):
	def _get_language(self, inst: StringStore) -> str:
		return inst._current_language

	def _set_language(self, inst, lang):
		if lang != inst._current_language:
			inst._switch_language(lang)
			if (
				not getattr(inst.engine, "_worker", False)
				and inst.engine.eternal["language"] != lang
			):
				inst.engine.eternal["language"] = lang
			inst._current_language = lang


class TamperEvidentDict[_K, _V](dict[_K, _V]):
	tampered: bool

	def __init__(self, data: list[tuple[_K, _V]] | dict[_K, _V] = ()):
		self.tampered = False
		super().__init__(data)

	def __setitem__(self, key: _K, value: _V) -> None:
		self.tampered = True
		super().__setitem__(key, value)

	def __delitem__(self, key: _K) -> None:
		self.tampered = True
		super().__delitem__(key)


class ChangeTrackingDict[_K, _V](UserDict[_K, _V]):
	def __init__(self, data: list[tuple[_K, _V]] | dict[_K, _V] = ()):
		self.changed = {}
		super().__init__(data)

	def apply_changes(self) -> None:
		self.data.update(self.changed)
		self.changed.clear()

	def copy(self) -> dict[_K, _V]:
		ret = {}
		ret.update(self.data)
		ret.update(self.changed)
		return ret

	def clear(self) -> None:
		self.data.clear()
		self.changed.clear()

	def __contains__(self, item: _K) -> bool:
		return item in self.changed or item in self.data

	def __iter__(self) -> Iterator[_K]:
		yield from self.changed
		yield from self.data

	def __len__(self) -> int:
		return len(self.changed) + len(self.data)

	def __getitem__(self, item: _K) -> _V:
		if item in self.changed:
			return self.changed[item]
		return self.data[item]

	def __setitem__(self, key: _K, value: _V) -> None:
		self.changed[key] = value

	def __delitem__(self, key: _K) -> None:
		if key in self.changed:
			del self.changed[key]
			if key in self.data:
				del self.data[key]
		else:
			del self.data[key]


class StringStore(MutableMapping[str, str], Signal):
	language = LanguageDescriptor()
	_store = "strings"

	def __init__(
		self,
		engine_or_string_dict: AbstractEngine | dict,
		prefix: str | None,
		lang="eng",
	):
		super().__init__()
		if isinstance(engine_or_string_dict, dict):
			self._prefix = None
			self._current_language = lang
			if lang in engine_or_string_dict and isinstance(
				engine_or_string_dict[lang], dict
			):
				self._languages = engine_or_string_dict
			else:
				self._languages = {lang: engine_or_string_dict}
		else:
			self.engine = engine_or_string_dict
			self._languages = {lang: TamperEvidentDict()}
			self._prefix = prefix
			self._current_language = lang
			self._switch_language(lang)

	def _switch_language(self, lang: str) -> None:
		"""Write the current language to disk, and load the new one if available"""
		if self._prefix is None:
			if lang not in self._languages:
				self._languages[lang] = TamperEvidentDict()
			return
		try:
			with open(os.path.join(self._prefix, lang + ".json"), "r") as inf:
				self._languages[lang] = TamperEvidentDict(json.load(inf))
		except FileNotFoundError:
			self._languages[lang] = TamperEvidentDict()
		assert self._current_language in self._languages
		if getattr(self._languages[self._current_language], "tampered", False):
			with open(
				os.path.join(self._prefix, self._current_language + ".json"),
				"w",
			) as outf:
				json.dump(
					self._languages[self._current_language],
					outf,
					indent=4,
					sort_keys=True,
				)
			self._languages[self._current_language].tampered = False

	def __iter__(self) -> Iterator[str]:
		return iter(self._languages[self._current_language])

	def __len__(self) -> int:
		return len(self._languages[self._current_language])

	def __getitem__(self, k: str) -> str:
		return self._languages[self._current_language][k]

	def __setitem__(self, k: str, v: str) -> None:
		"""Set the value of a string for the current language."""
		self._languages[self._current_language][k] = v
		self.send(self, key=k, val=v)

	def __delitem__(self, k: str) -> None:
		"""Delete the string from the current language, and remove it from the
		cache.

		"""
		del self._languages[self._current_language][k]
		self.send(self, key=k, val=None)

	def lang_items(self, lang: str | None = None) -> Iterator[tuple[str, str]]:
		"""Yield pairs of (id, string) for the given language."""
		if (
			self._prefix is not None
			and lang is not None
			and self._current_language != lang
		):
			with open(os.path.join(self._prefix, lang + ".json"), "r") as inf:
				self._languages[lang] = TamperEvidentDict(json.load(inf))
		yield from self._languages[lang or self._current_language].items()

	def save(self, reimport: bool = False) -> None:
		if self._prefix is None:
			return
		if not os.path.exists(self._prefix):
			os.mkdir(self._prefix)
		for lang, d in self._languages.items():
			if not d.tampered:
				continue
			with open(
				os.path.join(self._prefix, lang + ".json"),
				"w",
			) as outf:
				json.dump(
					self._languages[lang],
					outf,
					indent=4,
					sort_keys=True,
				)
			d.tampered = False
		if reimport:
			with open(
				os.path.join(self._prefix, self._current_language + ".json"),
				"r",
			) as inf:
				self._languages[self._current_language] = TamperEvidentDict(
					json.load(inf)
				)

	def blake2b(self) -> bytes:
		the_hash = blake2b()
		for k, v in self.items():
			the_hash.update(k.encode())
			the_hash.update(GROUP_SEP)
			the_hash.update(v.encode())
			the_hash.update(REC_SEP)
		return the_hash.digest()


@dataclass
class CodeHasher(ast.NodeVisitor):
	# What if users want to subclass Place or whatever, and use type
	# annotations to decide what rules to run on which of their subclasses?
	# Well, that's too complicated.

	_indent: int = 0
	_updated: bool = False
	_in_try_star: bool = False

	@cached_property
	def _blake2b(self):
		return blake2b()

	@cached_property
	def pack(self):
		from .facade import EngineFacade

		return EngineFacade(None).pack

	def update(self, data: bytes) -> None:
		self._blake2b.update(data)
		self._updated = True

	def visit(self, node):
		if self._indent > 0:
			self.update(b"\t" * self._indent)
		super().visit(node)

	@contextmanager
	def block(self, extra: bytes = b""):
		self.update(b":")
		if extra:
			self.update(extra)
		self._indent += 1
		yield
		self._indent -= 1

	def traverse(self, node: ast.AST | list[ast.stmt]):
		if isinstance(node, list):
			for item in node:
				self.traverse(item)
		else:
			super().visit(node)

	def visit_Constant(self, node: ast.Constant) -> None:
		v = node.value
		if isinstance(v, tuple):
			with self.delimit(b"(", b")"):
				self.items_view(self._write_constant, v)
		elif v is ...:
			self.update(b"...")
		else:
			self._write_constant(v)

	def maybe_semicolon(self):
		if self._updated:
			self.update(b";")

	def maybe_newline(self):
		if self._updated:
			self.update(b"\n")

	def fill(self, text: bytes = b"", allow_semicolon: bool = True):
		if self._indent == 0 and allow_semicolon:
			self.maybe_semicolon()
			self.update(text)
		else:
			self.maybe_newline()
			self.update(b"\t" * self._indent + text)

	def visit_FunctionDef(self, node):
		for decorator in node.decorator_list:
			self.update(b"@")
			self.hash_expr(decorator)
		self.update(b"def ")
		self.update(node.name.encode())
		with self.delimit(b"(", b")"):
			self.traverse(node.args)
		# we don't care about annotations
		with self.block():
			self.traverse(node.body)

	def visit_AsyncFunctionDef(self, node):
		raise TypeError("Lisien isn't an event loop")

	def visit_comprehension(self, node: ast.comprehension) -> None:
		if node.is_async:
			raise TypeError("Async not supported", node)
		self.update(b" for ")
		self.traverse(node.target)
		for iffy in node.ifs:
			self.update(b" if ")
			self.traverse(iffy)

	def visit_Name(self, node: ast.Name) -> None:
		self.update(node.id.encode())

	boolops = {"And": b"and", "Or": b"or"}

	def visit_BoolOp(self, node):
		with self.parens():
			for c in node.values:
				self.visit(c)
				self.update(self.boolops[node.op.__class__.__name__])

	def visit_NamedExpr(self, node):
		with self.parens():
			self.traverse(node.target)
			self.update(b" := ")
			self.traverse(node.value)

	binop = {
		"Add": b"+",
		"Sub": b"-",
		"Mult": b"*",
		"MatMult": b"@",
		"Div": b"/",
		"Mod": b"%",
		"LShift": b"<<",
		"RShift": b">>",
		"BitOr": b"|",
		"BitXor": b"^",
		"BitAnd": b"&",
		"FloorDiv": b"//",
		"Pow": b"**",
	}

	def visit_BinOp(self, node):
		opstr = self.binop[node.op.__class__.__name__]
		with self.parens():
			self.traverse(node.left)
			self.update(b" " + opstr + b" ")
			self.traverse(node.right)

	cmpops = {
		"Eq": b"==",
		"NotEq": b"!=",
		"Lt": b"<",
		"LtE": b"<=",
		"Gt": b">",
		"GtE": b"b>=",
		"Is": b"is",
		"IsNot": b"is not",
		"In": b"in",
		"NotIn": b"not in",
	}

	@contextmanager
	def parens(self):
		self.update(b"(")
		yield
		self.update(b")")

	def visit_Compare(self, node):
		with self.parens():
			self.traverse(node.left)
			for o, e in zip(node.ops, node.comparators):
				self.update(b" " + self.cmpops[o.__class__.__name__] + b" ")

	unop = {"Invert": b"~", "Not": b"not ", "UAdd": b"+", "USub": b"-"}

	def visit_UnaryOp(self, node):
		opstr = self.unop[node.op.__class__.__name__]
		with self.parens():
			self.update(opstr)
			self.traverse(node.operand)

	def visit_Lambda(self, node):
		self.update(b"lambda ")
		for arg in node.args.posonlyargs:
			self.update(arg.arg.encode())
			self.update(b",")
		self.update(b": ")
		self.traverse(node.body)

	def visit_IfExp(self, node):
		with self.parens():
			self.traverse(node.body)
			self.update(b" if ")
			self.traverse(node.test)
			self.update(b" else ")
			self.traverse(node.orelse)

	def visit_If(self, node):
		self.fill(b"if ", allow_semicolon=False)
		self.traverse(node.test)
		with self.block():
			self.traverse(node.body)
		while (
			node.orelse
			and len(node.orelse) == 1
			and isinstance(node.orelse[0], ast.If)
		):
			node = node.orelse[0]
			self.fill(b"elif ", allow_semicolon=False)
			self.traverse(node.test)
			with self.block():
				self.traverse(node.body)
		if node.orelse:
			self.fill(b"else", allow_semicolon=False)
			with self.block():
				self.traverse(node.orelse)

	@contextmanager
	def delimit(self, left: bytes, right: bytes):
		self.update(left)
		yield
		self.update(right)

	def visit_Dict(self, node):
		with self.delimit(b"{", b"}"):
			for k, v in zip(node.keys, node.values):
				if k is None:
					self.update(b"**")
					self.hash_expr(v)
				else:
					self.hash_expr(k)
					self.update(b": ")
					self.hash_expr(v)
				self.update(b", ")

	def interleave(self, inter, f, seq):
		"""Call f on each item in seq, calling inter() in between."""
		seq = iter(seq)
		try:
			f(next(seq))
		except StopIteration:
			pass
		else:
			for x in seq:
				inter()
				f(x)

	def visit_Import(self, node):
		self.fill(b"import ")
		self.interleave(lambda: self.update(b", "), self.traverse, node.names)

	def visit_ImportFrom(self, node):
		self.fill(b"from ")
		self.update(b"." * (node.level or 0))
		if node.module:
			self.update(node.module.encode())

	def visit_Set(self, node):
		if node.elts:
			with self.delimit(b"{", b"}"):
				self.interleave(
					lambda: self.update(b", "), self.traverse, node.elts
				)
		else:
			self.update(b"{*()}")

	def visit_ListComp(self, node):
		with self.delimit(b"[", b"]"):
			self.traverse(node.elt)
			for gen in node.generators:
				self.traverse(gen)

	def visit_GeneratorExp(self, node):
		with self.delimit(b"(", b")"):
			self.traverse(node.elt)
			for gen in node.generators:
				self.traverse(gen)

	def visit_SetComp(self, node):
		with self.delimit(b"{", b"}"):
			self.traverse(node.elt)
			for gen in node.generators:
				self.traverse(gen)

	def visit_DictComp(self, node):
		with self.delimit(b"{", b"}"):
			self.traverse(node.key)
			self.update(b": ")
			self.traverse(node.value)
			for gen in node.generators:
				self.traverse(gen)

	def visit_Call(self, node):
		self.traverse(node.func)
		with self.delimit(b"(", b")"):
			self.interleave(
				lambda: self.update(b", "), self.traverse, node.args
			)
			self.interleave(
				lambda: self.update(b", "), self.traverse, node.keywords
			)

	def visit_Subscript(self, node):
		self.traverse(node.value)
		with self.delimit(b"[", b"]"):
			if isinstance(node.slice, ast.Tuple) and node.slice.elts:
				self.items_view(self.traverse, node.slice.elts)
			else:
				self.traverse(node.slice)

	def items_view(
		self, traverser: Callable[[ast.AST], None], items: list[ast.AST]
	):
		if len(items) == 1:
			traverser(items[0])
			self.update(b",")
		else:
			self.interleave(lambda: self.update(b", "), traverser, items)

	def visit_Starred(self, node):
		self.update(b"*")
		self.traverse(node.value)

	def visit_Ellipsis(self, node):
		self.update(b"...")

	def visit_Slice(self, node):
		if node.lower:
			self.traverse(node.lower)
		self.update(b":")
		if node.upper:
			self.traverse(node.upper)
		if node.step:
			self.update(b":")
			self.traverse(node.step)

	def visit_Match(self, node):
		self.fill(b"match ", allow_semicolon=False)
		self.traverse(node.subject)
		with self.block():
			for case in node.cases:
				self.traverse(case)

	def visit_arg(self, node):
		self.update(node.arg.encode())

	def visit_arguments(self, node):
		first = True
		all_args = node.posonlyargs + node.args
		defaults = [None] * (
			len(all_args) - len(node.defaults)
		) + node.defaults
		for i, (a, d) in enumerate(zip(all_args, defaults), 1):
			if first:
				first = False
			else:
				self.update(b", ")
			self.traverse(a)
			if d:
				self.update(b"=")
				self.traverse(d)
			if i == len(node.posonlyargs):
				self.update(b", /")
		if node.vararg or node.kwonlyargs:
			if first:
				first = False
			else:
				self.update(b", ")
			self.update(b"*")
			if node.vararg:
				self.update(node.vararg.arg.encode())
		if node.kwonlyargs:
			for a, d in zip(node.kwonlyargs, node.kw_defaults):
				self.update(b", ")
				self.traverse(a)
				if d:
					self.update(b"=")
					self.traverse(d)
		if node.kwarg:
			if first:
				first = False
			else:
				self.update(b", ")
			self.update(b"**" + node.kwarg.arg.encode())

	def visit_keyword(self, node):
		if node.arg is None:
			self.update(b"**")
		else:
			self.update(node.arg.encode())
			self.update(b"=")
		self.traverse(node.value)

	def visit_alias(self, node):
		self.update(node.name.encode())
		if node.asname:
			self.update(b" as " + node.asname.encode())

	def visit_withitem(self, node):
		self.traverse(node.context_expr)
		if node.optional_vars:
			self.update(b" as ")
			self.traverse(node.optional_vars)

	def visit_match_case(self, node):
		self.fill(b"case ", allow_semicolon=False)
		self.traverse(node.pattern)
		if node.guard:
			self.update(b" if ")
			self.traverse(node.guard)
		with self.block():
			self.traverse(node.body)

	def visit_MatchValue(self, node):
		self.traverse(node.value)

	def _write_constant(self, value):
		self.update(repr(value).encode())

	def visit_MatchSingleton(self, node):
		self._write_constant(node.value)

	def visit_MatchSequence(self, node):
		with self.delimit(b"[", b"]"):
			self.interleave(
				lambda: self.update(b", "), self.traverse, node.patterns
			)

	def visit_MatchStar(self, node):
		name = node.name
		if name is None:
			name = b"_"
		self.update(b"*")
		self.update(name.encode())

	def visit_MatchMapping(self, node):
		def upd_key_pattern_pair(pair):
			k, p = pair
			self.traverse(k)
			self.update(b": ")
			self.traverse(p)

		with self.delimit(b"{", b"}"):
			keys = node.keys
			self.interleave(
				lambda: self.update(b", "),
				upd_key_pattern_pair,
				zip(keys, node.patterns, strict=True),
			)
			rest = node.rest
			if rest is not None:
				if keys:
					self.update(b", ")
				self.update(b"**")
				self.update(rest.encode())

	def visit_MatchClass(self, node):
		self.traverse(node.cls)
		with self.delimit(b"(", b")"):
			pats = node.patterns
			self.interleave(lambda: self.update(b", "), self.traverse, pats)
		attrs = node.kwd_attrs
		if attrs:

			def upd_attr_pat(pair):
				attr, pat = pair
				self.update(attr.encode())
				self.update(b"=")
				self.traverse(pat)

			if pats:
				self.update(b", ")
			self.interleave(
				lambda: self.update(b", "),
				upd_attr_pat,
				zip(attrs, node.kwd_patterns, strict=True),
			)

	def visit_MatchAs(self, node):
		name = node.name
		pattern = node.pattern
		if name is None:
			self.update(b"_")
		elif pattern is None:
			self.update(node.name.encode())
		else:
			with self.parens():
				self.traverse(pattern)
				self.update(b" as ")
				self.update(node.name.encode())

	def visit_MatchOr(self, node):
		with self.parens():
			self.interleave(
				lambda: self.update(b" | "), self.traverse, node.patterns
			)

	def visit_FunctionType(self, node):
		"""Since this is a function signature, rather than a real function, ignore it

		The purpose of these hashes is to tell if you have the code needed to load
		a game. Empty signatures don't matter.

		"""
		pass

	def visit_Expr(self, node):
		self.fill()
		self.traverse(node.value)

	def visit_Assign(self, node):
		self.fill()
		for target in node.targets:
			self.traverse(target)
			self.update(b" = ")
		self.traverse(node.value)

	def visit_AugAssign(self, node):
		self.fill()
		self.traverse(node.target)
		self.update(b" ")
		self.update(self.binop[node.op.__class__.__name__])
		self.update(b"= ")
		self.traverse(node.value)

	def visit_AnnAssign(self, node):
		if node.value is None:  # rather than ast.Constant *representing* None
			return
		self.fill()
		with self.delimit(b"(", b")"):
			self.traverse(node.target)
		self.update(b" = ")
		self.traverse(node.value)

	def visit_Return(self, node):
		self.fill(b"return")
		if node.value:
			self.update(b" ")
			self.traverse(node.value)

	def visit_Pass(self, node):
		return

	def visit_Break(self, node):
		self.fill(b"break")

	def visit_Continue(self, node):
		self.fill(b"continue")

	def visit_Delete(self, node):
		self.fill(b"del ")
		self.interleave(
			lambda: self.update(b", "), self.traverse, node.targets
		)

	def visit_Assert(self, node):
		return

	def visit_Global(self, node):
		self.fill(b"global ")
		self.interleave(lambda: self.update(b", "), self.write, node.names)

	def visit_Nonlocal(self, node):
		self.fill(b"nonlocal ")
		self.interleave(lambda: self.update(b", "), self.write, node.names)

	def write(self, s: str) -> None:
		self.update(s.encode())

	def visit_Await(self, node):
		raise TypeError("Lisien isn't an event loop")

	def visit_Yield(self, node):
		with self.parens():
			self.update(b"yield")
			if node.value:
				self.update(b" ")
				self.traverse(node.value)

	def visit_YieldFrom(self, node):
		with self.parens():
			self.update(b"yield from ")
			self.traverse(node.value)

	def visit_Raise(self, node):
		self.fill(b"raise")
		if not node.exc:
			return
		self.update(b" ")
		self.traverse(node.exc)
		if node.cause:
			self.update(b" from ")
			self.traverse(node.cause)

	def _visit_try(self, node):
		self.fill(b"try", allow_semicolon=False)
		with self.block():
			self.traverse(node.body)
		for ex in node.handlers:
			self.traverse(ex)
		if node.orelse:
			self.fill(b"else", allow_semicolon=False)
		if node.finalbody:
			self.fill(b"finally", allow_semicolon=False)
			with self.block():
				self.traverse(node.finalbody)

	def visit_Try(self, node):
		prev_in_try_star = self._in_try_star
		try:
			self._in_try_star = False
			self._visit_try(node)
		finally:
			self._in_try_star = prev_in_try_star

	def visit_TryStar(self, node):
		prev_in_try_star = self._in_try_star
		try:
			self._in_try_star = True
			self._visit_try(node)
		finally:
			self._in_try_star = prev_in_try_star

	def visit_ExceptHandler(self, node):
		self.fill(
			b"except*" if self._in_try_star else b"except",
			allow_semicolon=False,
		)
		if node.type:
			self.update(b" ")
			self.traverse(node.type)
		if node.name:
			self.update(b" as ")
			self.write(node.name)
		with self.block():
			self.traverse(node.body)

	def visit_ClassDef(self, node):
		self.maybe_newline()
		for deco in node.decorator_list:
			self.fill(b"@", allow_semicolon=False)
			self.traverse(deco)
		self.fill(b"class " + node.name.encode(), allow_semicolon=False)
		# we don't care about type parameters
		with self.delimit_if(
			b"(", b")", condition=node.bases or node.keywords
		):
			self.interleave(
				lambda: self.update(b", "), self.traverse, node.bases
			)
			self.interleave(
				lambda: self.update(b", "), self.traverse, node.keywords
			)
		with self.block():
			# we don't care about docstrings
			self.traverse(node.body)

	def visit_For(self, node):
		self.fill(b"for ", allow_semicolon=False)
		self.traverse(node.target)
		self.update(b" in ")
		self.traverse(node.iter)
		with self.block():  # we don't care about type comments
			self.traverse(node.body)
		if node.orelse:
			self.fill(b"else", allow_semicolon=False)
			with self.block():
				self.traverse(node.orelse)

	def visit_AsyncFor(self, node):
		raise TypeError("Lisien isn't an event loop")

	def visit_While(self, node):
		self.fill(b"while ", allow_semicolon=False)
		self.traverse(node.test)
		with self.block():
			self.traverse(node.body)
		if node.orelse:
			self.fill(b"else", allow_semicolon=False)
			with self.block():
				self.traverse(node.orelse)

	def visit_With(self, node):
		self.fill(b"with ", allow_semicolon=False)
		self.interleave(lambda: self.update(b", "), self.traverse, node.items)
		with self.block():  # we don't care about type comments
			self.traverse(node.body)

	def visit_AsyncWith(self, node):
		raise TypeError("Lisien isn't an event loop")

	def _write_ftstring_inner(self, node, is_format_spec=False):
		if isinstance(node, ast.JoinedStr):
			for value in node.values:
				self._write_ftstring_inner(
					value, is_format_spec=is_format_spec
				)
		elif isinstance(node, ast.Constant) and isinstance(node.value, str):
			value = node.value.replace("{", "{{").replace("}", "}}")

			if is_format_spec:
				value = value.replace("\\", "\\\\")
				value = value.replace("'", "\\'")
				value = value.replace('"', '\\"')
				value = value.replace("\n", "\\n")
			self.write(value)
		elif isinstance(node, ast.FormattedValue):
			self.visit_FormattedValue(node)
		elif isinstance(node, ast.Interpolation):
			self.visit_Interpolation(node)
		else:
			raise ValueError("Unexpected node inside JoinedStr", node)

	def _write_ftstring(self, values, prefix):
		self.write(prefix)
		for value in values:
			self._write_ftstring_inner(value)

	def visit_JoinedStr(self, node):
		self._write_ftstring(node.values, "f")

	def visit_TemplateStr(self, node):
		self._write_ftstring(node.values, "t")

	def _write_interpolation(self, node, is_interpolation=False):
		with self.delimit(b"{", b"}"):
			if is_interpolation:
				expr = node.str
			else:
				self.visit(node.value)
				return
			self.write(expr)
			if node.conversion != -1:
				self.update(b"!")
				self.update(node.conversion.to_bytes())
			if node.format_spec:
				self.update(b":")
				self._write_fstring_inner(
					node.format_spec, is_format_spec=True
				)

	def visit_FormattedValue(self, node):
		self._write_interpolation(node)

	def visit_Interpolation(self, node):
		self._write_interpolation(node, is_interpolation=True)

	def visit_List(self, node):
		with self.delimit(b"[", b"]"):
			self.interleave(
				lambda: self.update(b", "), self.traverse, node.elts
			)

	def visit_TypeVar(self, node):
		return

	def visit_TypeVarTuple(self, node):
		return

	def visit_ParamSpec(self, node):
		return

	def visit_TypeAlias(self, node):
		return

	def visit_Tuple(self, node):
		with self.delimit(b"(", b")"):
			self.items_view(self.traverse, node.elts)

	def visit_Attribute(self, node):
		self.traverse(node.value)
		if isinstance(node.value, ast.Constant) and isinstance(
			node.value.value, int
		):
			self.update(b" ")
		self.update(b".")
		self.write(node.attr)

	def digest(self) -> bytes:
		return self._blake2b.digest()

	def hexdigest(self) -> str:
		return self._blake2b.hexdigest()

	def b64digest(self) -> str:
		digest = self.digest()
		encoded = base64.standard_b64encode(digest)
		return encoded.decode()


class FunctionStore[_K: str, _T: FunctionType | MethodType](
	AbstractFunctionStore[_K, _T], Signal
):
	"""A module-like object that lets you alter its code and save your changes.

	Instantiate it with a path to a file that you want to keep the code in.
	Assign functions to its attributes, then call its ``save()`` method,
	and they'll be unparsed and written to the file.

	This is a ``Signal``, so you can pass a function to its ``connect`` method,
	and it will be called when a function is added, changed, or deleted.
	The keyword arguments will be ``attr``, the name of the function, and ``val``,
	the function itself.

	"""

	_filename: str | None

	def __init__(
		self, filename: str | None, initial: dict = None, module: str = None
	):
		if initial is None:
			initial = {}
		super().__init__()
		if filename is None:
			self._filename = None
			self._module = self.__name__ = module
			self._ast = ast.Module(body=[], type_ignores=[])
			self._ast_idx = {}
			self._need_save = False
			self._locl = initial
		else:
			if not filename.endswith(".py"):
				raise ValueError(
					"FunctionStore can only work with pure Python source code"
				)
			self._filename = os.path.abspath(os.path.realpath(filename))
			self._store = os.path.basename(self._filename).removesuffix(".py")
			try:
				self.reimport()
			except (FileNotFoundError, ModuleNotFoundError):
				self._module = module
				self._ast = ast.Module(body=[], type_ignores=[])
				self._ast_idx = {}
				self.save()
			self._need_save = False
			self._locl = {}
			for k, v in initial.items():
				setattr(self, k, v)

	def __dir__(self):
		yield from self._locl
		yield from super().__dir__()

	def __getattr__(self, k):
		if k in self._locl:
			return self._locl[k]
		elif self._need_save:
			self.save()
			return getattr(self._module, k)
		elif hasattr(self._module, k):
			return getattr(self._module, k)
		else:
			raise AttributeError("No attribute ", k)

	def __setattr__(self, k, v):
		if not callable(v):
			super().__setattr__(k, v)
			return
		self._set_source(k, getsource(v), func=v)

	def _set_source(self, k: str, source: str, func: Callable | None = None):
		if func is None:
			holder = {}
			exec(source, holder)
			if k not in holder:
				raise NameError(
					"Function in source has a different name", k, source
				)
			func = holder[k]
		outdented = dedent_source(source)
		expr = ast.parse(outdented)
		expr.body[0].name = k
		if k in self._ast_idx:
			self._ast.body[self._ast_idx[k]] = expr
		else:
			self._ast_idx[k] = len(self._ast.body)
			self._ast.body.append(expr)
		if self._filename is not None:
			self._need_save = True
		if isinstance(self._module, str):
			func.__module__ = self._module
		self._locl[k] = func
		self.send(self, attr=k, val=func)

	def __call__(self, v):
		if isinstance(self._module, str):
			v.__module__ = self._module
		elif hasattr(self._module, "__name__"):
			v.__module__ = self._module.__name__
		setattr(self, v.__name__, v)
		return v

	def __delattr__(self, k):
		del self._locl[k]
		del self._ast.body[self._ast_idx[k]]
		del self._ast_idx[k]
		for name in list(self._ast_idx):
			if name > k:
				self._ast_idx[name] -= 1
		if self._filename is not None:
			self._need_save = True
		self.send(self, attr=k, val=None)

	def save(self, reimport=True):
		if self._filename is None:
			return
		with open(self._filename, "w", encoding="utf-8") as outf:
			outf.write(ast.unparse(self._ast))
		self._need_save = False
		if reimport:
			self.reimport()

	def reimport(self, signal: bool = True):
		if self._filename is None:
			return
		path, filename = os.path.split(self._filename)
		modname = filename[:-3]
		if modname in sys.modules:
			del sys.modules[modname]
		modname = filename[:-3]
		spec = importlib.util.spec_from_file_location(modname, self._filename)
		self._module = importlib.util.module_from_spec(spec)
		sys.modules[modname] = self._module
		spec.loader.exec_module(self._module)
		self._ast = ast.parse(self._module.__loader__.get_data(self._filename))
		self._ast_idx = {}
		for i, node in enumerate(self._ast.body):
			if hasattr(node, "name"):
				self._ast_idx[node.name] = i
			elif hasattr(node, "__name__"):
				self._ast_idx[node.__name__] = i
		if signal:
			self.send(self, attr=None, val=None)

	def iterplain(self):
		for name, idx in self._ast_idx.items():
			yield name, ast.unparse(self._ast.body[idx])

	def store_source(self, v: str, name: str | None = None) -> None:
		self._need_save = True
		outdented = dedent_source(v)
		mod = ast.parse(outdented)
		expr = ast.Expr(mod)
		if len(expr.value.body) != 1:
			raise ValueError("Tried to store more than one function")
		if name is None:
			name = expr.value.body[0].name
		else:
			expr.value.body[0].name = name
		if name in self._ast_idx:
			self._ast.body[self._ast_idx[name]] = expr
		else:
			self._ast_idx[name] = len(self._ast.body)
			self._ast.body.append(expr)
		locl = {}
		exec(compile(mod, self._filename or "", "exec"), {}, locl)
		self._locl.update(locl)
		self.send(self, attr=name, val=locl[name])

	def get_source(self, name: str) -> str:
		return ast.unparse(self._ast.body[self._ast_idx[name]])

	def blake2b(self) -> str:
		"""Return the blake2b hash digest of the code stored here

		Neither formatting nor type annotations are considered significant for
		the purposes of this hash.

		The hash is returned in a base64 string.

		"""
		hasher = CodeHasher()
		todo = dict(self._ast_idx)
		# stripped_ast = deepcopy(self._ast.body)
		# astor.strip_tree(stripped_ast)
		for k in sort_set(todo.keys()):
			funcdef = self._ast.body[todo[k]]
			if not isinstance(funcdef, ast.FunctionDef):
				raise TypeError(
					"Only store function defs in FunctionStore", funcdef
				)
			hasher.visit(funcdef)
		return hasher.b64digest()

	def __getstate__(self):
		return self._locl, self._ast, self._ast_idx

	def __setstate__(self, state):
		self._locl, self._ast, self._ast_idx = state


class TriggerStore(FunctionStore[TriggerFuncName, TriggerFunc]):
	def get_source(self, name: str) -> str:
		if name == "truth":
			return "def truth(*args):\n\treturn True"
		return super().get_source(name)

	@staticmethod
	def truth(*args):
		return True


class PrereqStore(FunctionStore[PrereqFuncName, PrereqFunc]): ...


class ActionStore(FunctionStore[ActionFuncName, ActionFunc]): ...


class UniversalMapping(MutableMapping, Signal):
	"""Mapping for variables that are global but which I keep history for"""

	__slots__ = ["engine"]

	def __init__(self, engine):
		"""Store the engine and initialize my private dictionary of
		listeners.

		"""
		super().__init__()
		self.engine = engine

	def __iter__(self):
		return self.engine._universal_cache.iter_keys(*self.engine.time)

	def __len__(self):
		return self.engine._universal_cache.count_keys(*self.engine.time)

	def __getitem__(self, k: KeyHint | UniversalKey):
		"""Get the current value of this key"""
		return wrapval(
			self,
			k,
			self._get_cache_now(k),
		)

	def _get_cache_now(self, k: UniversalKey):
		return self.engine._universal_cache.retrieve(k, *self.engine.time)

	def __setitem__(self, k: KeyHint | UniversalKey, v: ValueHint | Value):
		"""Set k=v at the current branch and tick"""
		try:
			if v == self._get_cache_now(k):
				return
		except KeyError:
			pass
		branch, turn, tick = self.engine._nbtt()
		self.engine._universal_cache.store(k, branch, turn, tick, v)
		self.engine.db.universal_set(k, branch, turn, tick, v)
		self.send(self, key=k, val=v)

	def _set_cache_now(self, k: UniversalKey, v: Value):
		self.engine._universal_cache.store(k, *self.engine.time, v)

	def __delitem__(self, k: KeyHint | UniversalKey):
		"""Unset this key for the present (branch, tick)"""
		branch, turn, tick = self.engine._nbtt()
		self.engine._universal_cache.store(k, branch, turn, tick, ...)
		self.engine.db.universal_del(k, branch, turn, tick)
		self.send(self, key=k, val=...)


class CharacterMapping(MutableMapping, Signal):
	"""A mapping by which to access :class:`Character` objects.

	If a character already exists, you can always get its name here to
	get the :class:`Character` object. Deleting an item here will
	delete the character from the world, even if there are still
	:class:`Character` objects referring to it; those won't do
	anything useful anymore.

	"""

	engine = getatt("orm")

	def __init__(self, orm):
		self.orm = orm
		Signal.__init__(self)

	def __iter__(self):
		branch, turn, tick = self.engine.time
		return self.engine._graph_cache.iter_keys(branch, turn, tick)

	def __len__(self):
		branch, turn, tick = self.engine.time
		return self.engine._graph_cache.count_keys(branch, turn, tick)

	def __contains__(self, item: KeyHint | CharName) -> bool:
		branch, turn, tick = self.engine.time
		try:
			return (
				self.engine._graph_cache.retrieve(item, branch, turn, tick)
				== "DiGraph"
			)
		except KeyError:
			return False

	def __getitem__(self, name: KeyHint | CharName) -> Character:
		"""Return the named character, if it's been created.

		Try to use the cache if possible.

		"""
		from .character import Character

		name = CharName(name)
		if name not in self:
			raise KeyError("No such character", name)
		cache = self.engine._graph_objs
		if name not in cache:
			cache[name] = Character(
				self.engine, name, init_rulebooks=name not in self
			)
		ret = cache[name]
		if not isinstance(ret, Character):
			raise TypeError(
				"You put something weird in the Character cache", type(ret)
			)
		return ret

	def __setitem__(
		self,
		name: KeyHint | CharName,
		value: dict[KeyHint | Stat, ValueHint | Value] | nx.Graph,
	):
		"""Make a new character by the given name, and initialize its data to
		the given value.

		"""
		self.engine._init_graph(name, "DiGraph", value)
		self.send(self, key=name, val=self.engine.character[name])

	def __delitem__(self, name: KeyHint | CharName):
		self.engine.del_character(name)
		self.send(self, key=name, val=None)


_K = TypeVar("_K")
_V = TypeVar("_V")


class CompositeDict[_K, _V](MutableMapping[_K, _V], Signal):
	"""Combine two dictionaries into one"""

	def __init__(self, d1, d2):
		"""Store dictionaries"""
		super().__init__()
		self.d1 = d1
		self.d2 = d2

	def __iter__(self):
		"""Iterate over both dictionaries' keys"""
		for k in self.d1:
			yield k
		for k in self.d2:
			yield k

	def __len__(self):
		"""Sum the lengths of both dictionaries"""
		return len(self.d1) + len(self.d2)

	def __contains__(self, item):
		return item in self.d1 or item in self.d2

	def __getitem__(self, k):
		"""Get an item from ``d1`` if possible, then ``d2``"""
		try:
			return self.d1[k]
		except KeyError:
			return self.d2[k]

	def __setitem__(self, key, value):
		self.d1[key] = value
		self.send(self, key=key, value=value)

	def __delitem__(self, key):
		deleted = False
		if key in self.d2:
			deleted = True
			del self.d2[key]
		if key in self.d1:
			deleted = True
			del self.d1[key]
		if not deleted:
			raise KeyError("{} is in neither of my wrapped dicts".format(key))
		self.send(self, key=key, value=None)

	def patch(self, d):
		"""Recursive update"""
		for k, v in d.items():
			if k in self:
				self[k].update(v)
			else:
				self[k] = deepcopy(v)
