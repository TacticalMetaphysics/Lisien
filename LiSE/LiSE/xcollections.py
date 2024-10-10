# This file is part of LiSE, a framework for life simulation games.
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
"""Common classes for collections in LiSE

Notably includes wrappers for mutable objects, allowing them to be stored in
the database. These simply store the new value.

Most of these are subclasses of :class:`blinker.Signal`, so you can listen
for changes using the ``connect(..)`` method.

"""

from io import StringIO
from collections.abc import MutableMapping
from copy import deepcopy
from types import MethodType
from inspect import getsource
from ast import parse, Expr, Module
import json
import importlib.util
import sys, os

from blinker import Signal
from astunparse import Unparser

from .util import dedent_source, getatt
from .allegedb.graph import GraphsMapping


class TabUnparser(Unparser):
	def fill(self, text=""):
		__doc__ = Unparser.fill.__doc__
		self.f.write("\n" + "\t" * self._indent + text)


def unparse(tree):
	v = StringIO()
	TabUnparser(tree, file=v)
	return v.getvalue()


class Language(str):
	sigs = {}

	def __new__(cls, sig, v):
		me = str.__new__(cls, v)
		cls.sigs[me] = sig
		return me

	def connect(self, *args, **kwargs):
		self.sigs[self].connect(*args, **kwargs)


class AbstractLanguageDescriptor(Signal):
	def __get__(self, instance, owner=None):
		if not hasattr(self, "lang"):
			self.lang = Language(self, self._get_language(instance))
		return self.lang

	def __set__(self, inst, val):
		self._set_language(inst, val)
		self.lang = Language(self, val)
		self.send(inst, language=val)

	def __str__(self):
		return self.lang


class LanguageDescriptor(AbstractLanguageDescriptor):
	def _get_language(self, inst):
		return inst._language

	def _set_language(self, inst, val):
		inst._load_language(val)
		inst.query.global_set("language", val)


class StringStore(MutableMapping, Signal):
	language = LanguageDescriptor()

	def __init__(self, query, prefix, lang="eng"):
		"""Store the engine, the name of the database table to use, and the
		language code.

		"""
		super().__init__()
		self.query = query
		self._prefix = prefix
		self._language = lang
		self._load_language(lang)

	def _load_language(self, lang):
		if hasattr(self, "_cache"):
			with open(
				os.path.join(self._prefix, self.language + ".json"), "w"
			) as outf:
				json.dump(self._cache, outf)
		try:
			with open(os.path.join(self._prefix, lang + ".json"), "r") as inf:
				self._cache = json.load(inf)
		except FileNotFoundError:
			self._cache = {}
		self._language = lang

	def __iter__(self):
		return iter(self._cache)

	def __len__(self):
		return len(self._cache)

	def __getitem__(self, k):
		return self._cache[k]

	def __setitem__(self, k, v):
		"""Set the value of a string for the current language."""
		self._cache[k] = v
		self.send(self, key=k, val=v)

	def __delitem__(self, k):
		"""Delete the string from the current language, and remove it from the
		cache.

		"""
		del self._cache[k]
		self.send(self, key=k, val=None)

	def lang_items(self, lang=None):
		"""Yield pairs of (id, string) for the given language."""
		if lang is not None and self._language != lang:
			self._load_language(lang)
		yield from self._cache.items()

	def save(self, reimport=False):
		if not os.path.exists(self._prefix):
			os.mkdir(self._prefix)
		with open(
			os.path.join(self._prefix, self._language + ".json"), "w"
		) as outf:
			json.dump(self._cache, outf, indent=4, sort_keys=True)


class FunctionStore(Signal):
	"""A module-like object that lets you alter its code and save your changes.

	Instantiate it with a path to a file that you want to keep the code in.
	Assign functions to its attributes, then call its ``save()`` method,
	and they'll be unparsed and written to the file.

	This is a ``Signal``, so you can pass a function to its ``connect`` method,
	and it will be called when a function is added, changed, or deleted.
	The keyword arguments will be ``attr``, the name of the function, and ``val``,
	the function itself.

	"""

	def __init__(self, filename):
		if not filename.endswith(".py"):
			raise ValueError(
				"FunctionStore can only work with pure Python source code"
			)
		super().__init__()
		self._filename = os.path.abspath(os.path.realpath(filename))
		try:
			self.reimport()
		except (FileNotFoundError, ModuleNotFoundError) as ex:
			self._module = None
			self._ast = Module(body=[])
			self._ast_idx = {}
		self._need_save = False
		self._locl = {}

	def __getattr__(self, k):
		if k in self._locl:
			return self._locl[k]
		elif self._module:
			return getattr(self._module, k)
		elif self._need_save:
			self.save()
			return getattr(self._module, k)
		else:
			raise AttributeError("No attribute " + repr(k))

	def __setattr__(self, k, v):
		if not callable(v):
			super().__setattr__(k, v)
			return
		if self._module is not None:
			setattr(self._module, k, v)
		source = getsource(v)
		outdented = dedent_source(source)
		expr = Expr(parse(outdented))
		expr.value.body[0].name = k
		if k in self._ast_idx:
			self._ast.body[self._ast_idx[k]] = expr
		else:
			self._ast_idx[k] = len(self._ast.body)
			self._ast.body.append(expr)
		self._need_save = True
		self.send(self, attr=k, val=v)

	def __call__(self, v):
		setattr(self, v.__name__, v)
		return v

	def __delattr__(self, k):
		del self._locl[k]
		del self._ast.body[self._ast_idx[k]]
		del self._ast_idx[k]
		for name in list(self._ast_idx):
			if name > k:
				self._ast_idx[name] -= 1
		self._need_save = True
		self.send(self, attr=k, val=None)

	def save(self, reimport=True):
		with open(self._filename, "w", encoding="utf-8") as outf:
			outf.write("# encoding: utf-8")
			Unparser(self._ast, outf)
		self._need_save = False
		if reimport:
			self.reimport()

	def reimport(self):
		importlib.invalidate_caches()
		path, filename = os.path.split(self._filename)
		modname = filename[:-3]
		if modname in sys.modules:
			del sys.modules[modname]
		modname = filename[:-3]
		spec = importlib.util.spec_from_file_location(modname, self._filename)
		self._module = importlib.util.module_from_spec(spec)
		sys.modules[modname] = self._module
		spec.loader.exec_module(self._module)
		self._ast = parse(self._module.__loader__.get_data(self._filename))
		self._ast_idx = {}
		for i, node in enumerate(self._ast.body):
			self._ast_idx[node.name] = i
		self.send(self, attr=None, val=None)

	def iterplain(self):
		for name, idx in self._ast_idx.items():
			yield name, unparse(self._ast.body[idx])

	def store_source(self, v, name=None):
		self._need_save = True
		outdented = dedent_source(v)
		mod = parse(outdented)
		expr = Expr(mod)
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
		exec(compile(mod, self._filename, "exec"), {}, locl)
		self._locl.update(locl)
		self.send(self, attr=name, val=locl[name])

	def get_source(self, name):
		if name == "truth":
			return "def truth(*args):\n\treturn True"
		return unparse(self._ast.body[self._ast_idx[name]])

	@staticmethod
	def truth(*args):
		return True


class MethodStore(FunctionStore):
	def __init__(self, engine):
		self.engine = engine
		super().__init__("method.py")

	def __getattr__(self, item):
		return MethodType(super().__getattr__(item), self.engine)


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
		return self.engine._universal_cache.iter_keys(*self.engine._btt())

	def __len__(self):
		return self.engine._universal_cache.count_keys(*self.engine._btt())

	def __getitem__(self, k):
		"""Get the current value of this key"""
		return self.engine._universal_cache.retrieve(k, *self.engine._btt())

	def __setitem__(self, k, v):
		"""Set k=v at the current branch and tick"""
		branch, turn, tick = self.engine._nbtt()
		self.engine._universal_cache.store(k, branch, turn, tick, v)
		self.engine.query.universal_set(k, branch, turn, tick, v)
		self.send(self, key=k, val=v)

	def __delitem__(self, k):
		"""Unset this key for the present (branch, tick)"""
		branch, turn, tick = self.engine._nbtt()
		self.engine._universal_cache.store(k, branch, turn, tick, None)
		self.engine.query.universal_del(k, branch, turn, tick)
		self.send(self, key=k, val=None)


class CharacterMapping(GraphsMapping, Signal):
	"""A mapping by which to access :class:`Character` objects.

	If a character already exists, you can always get its name here to
	get the :class:`Character` object. Deleting an item here will
	delete the character from the world, even if there are still
	:class:`Character` objects referring to it; those won't do
	anything useful anymore.

	"""

	engine = getatt("orm")

	def __init__(self, orm):
		GraphsMapping.__init__(self, orm)
		Signal.__init__(self)

	def __getitem__(self, name):
		"""Return the named character, if it's been created.

		Try to use the cache if possible.

		"""
		from .character import Character

		if name not in self:
			raise KeyError("No such character")
		cache = self.engine._graph_objs
		if name not in cache:
			cache[name] = Character(
				self.engine, name, init_rulebooks=name not in self
			)
		ret = cache[name]
		if not isinstance(ret, Character):
			raise TypeError("""Tried to get a graph that isn't a Character.
				This should never happen. It probably indicates
				a bug in allegedb.""")
		return ret

	def __setitem__(self, name, value):
		"""Make a new character by the given name, and initialize its data to
		the given value.

		"""
		self.engine._init_graph(name, "DiGraph", value)
		self.send(self, key=name, val=self.engine._graph_objs[name])

	def __delitem__(self, name):
		self.engine.del_graph(name)
		self.send(self, key=name, val=None)


class CompositeDict(MutableMapping, Signal):
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
