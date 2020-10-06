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
"""Common classes for collections in LiSE, of which most can be bound to."""
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from types import MethodType
from inspect import getsource
from ast import parse, Expr, Module
import json
import importlib
import sys, os

from blinker import Signal
from astunparse import unparse, Unparser

from .util import dedent_source


if sys.version_info.minor < 6:
    ModuleNotFoundError = ImportError


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
        if not hasattr(self, 'lang'):
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
        inst._language = val
        inst.query.global_set('language', val)


class StringStore(MutableMapping, Signal):
    """Store strings in database, and format them with one another upon retrieval.

    In any one string, putting the key of another string in curly
    braces will cause the other string to be substituted in.

    """
    language = LanguageDescriptor()

    def __init__(self, query, filename, lang='eng'):
        """Store the engine, the name of the database table to use, and the
        language code.

        """
        super().__init__()
        self.query = query
        self._filename = filename
        self._language = lang
        try:
            with open(filename, 'r') as inf:
                self.cache = json.load(inf)
        except FileNotFoundError:
            self.cache = {lang: {}}

    def __iter__(self):
        return iter(self.cache[self.language])

    def __len__(self):
        return len(self.cache[self.language])

    def __getitem__(self, k):
        return self.cache[self.language][k]

    def __setitem__(self, k, v):
        """Set the value of a string for the current language."""
        self.cache[self.language][k] = v
        self.send(self, key=k, val=v)

    def __delitem__(self, k):
        """Delete the string from the current language, and remove it from the
        cache.

        """
        del self.cache[self.language][k]
        self.send(self, key=k, val=None)

    def lang_items(self, lang=None):
        """Yield pairs of (id, string) for the given language."""
        if lang is None:
            lang = self.language
        yield from self.cache.setdefault(lang, {}).items()

    def save(self, reimport=False):
        with open(self._filename, 'w') as outf:
            json.dump(self.cache, outf, indent=4, sort_keys=True)


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
            raise ValueError("FunctionStore can only work with pure Python source code with .py extension")
        super().__init__()
        self._filename = fullname = os.path.abspath(os.path.realpath(filename))
        path, filename = os.path.split(fullname)
        modname = filename[:-3]
        if sys.path[0] != path:
            if path in sys.path:
                sys.path.remove(path)
            sys.path.insert(0, path)
        try:
            if modname in sys.modules:
                self._module = sys.modules[modname]
            else:
                self._module = importlib.import_module(modname)
            self._ast = parse(self._module.__loader__.get_data(fullname))
            self._ast_idx = {}
            for i, node in enumerate(self._ast.body):
                self._ast_idx[node.name] = i
        except (FileNotFoundError, ModuleNotFoundError):
            self._module = None
            self._ast = Module(body=[])
            self._ast_idx = {}
        self._need_save = False
        self._locl = {}

    def __getattr__(self, k):
        if self._need_save:
            self.save()
        if self._module:
            return getattr(self._module, k)
        elif k in self._locl:
            return self._locl[k]
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
        with open(self._filename, 'w') as outf:
            Unparser(self._ast, outf)
        if reimport:
            importlib.invalidate_caches()
            path, filename = os.path.split(self._filename)
            modname = filename[:-3]
            if modname in sys.modules:
                del sys.modules[modname]
            self._module = importlib.import_module(filename[:-3])
        self._need_save = False

    def iterplain(self):
        for funcdef in self._ast.body:
            yield funcdef.name, unparse(funcdef)

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
        exec(compile(mod, self._filename, 'exec'), {}, locl)
        self._locl.update(locl)
        self.send(self, attr=name, val=locl[name])

    def get_source(self, name):
        return unparse(self._ast.body[self._ast_idx[name]])

    def truth(self, *args):
        return True


class MethodStore(FunctionStore):
    def __init__(self, engine):
        self.engine = engine
        super().__init__('method.py')

    def __getattr__(self, item):
        return MethodType(super().__getattr__(item), self.engine)


class UniversalMapping(MutableMapping, Signal):
    """Mapping for variables that are global but which I keep history for"""
    __slots__ = ['engine']

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
        self.engine.tick = tick
        self.send(self, key=k, val=v)

    def __delitem__(self, k):
        """Unset this key for the present (branch, tick)"""
        branch, turn, tick = self.engine._nbtt()
        self.engine._universal_cache.store(k, branch, turn, tick, None)
        self.engine.query.universal_del(k, branch, turn, tick)
        self.send(self, key=k, val=None)


class CharacterMapping(MutableMapping, Signal):
    """A mapping by which to access :class:`Character` objects.

    If a character already exists, you can always get its name here to
    get the :class:`Character` object. Deleting an item here will
    delete the character from the world, even if there are still
    :class:`Character` objects referring to it; those won't do
    anything useful anymore.

    """
    __slots__ = ['engine', '_cache']

    def __init__(self, engine):
        """Store the engine, initialize caches"""
        super().__init__()
        self.engine = engine
        self._cache = None

    def __iter__(self):
        """Iterate over every character name."""
        return iter(self.engine._graph_objs)

    def __contains__(self, name):
        """Has this character been created?"""
        if self.engine._graph_objs:
            self._cache = None
            return name in self.engine._graph_objs
        # hack to make initial load work
        if self._cache is None:
            self._cache = [ch for ch, typ in self.engine.query.graphs_types() if typ == 'DiGraph']
        return name in self._cache

    def __len__(self):
        """How many characters have been created?"""
        return len(self.engine._graph_objs)

    def __getitem__(self, name):
        """Return the named character, if it's been created.

        Try to use the cache if possible.

        """
        from .character import Character
        if name not in self:
            raise KeyError("No such character")
        cache = self.engine._graph_objs
        if name not in cache:
            cache[name] = Character(self.engine, name)
        ret = cache[name]
        if not isinstance(ret, Character):
            raise TypeError(
                """Tried to get a graph that isn't a Character.
                This should never happen. It probably indicates
                a bug in allegedb."""
            )
        return ret

    def __setitem__(self, name, value):
        """Make a new character by the given name, and initialize its data to
        the given value.

        """
        from .character import Character
        if isinstance(value, Character):
            self.engine._graph_objs[name] = value
            return
        if name in self.engine._graph_objs:
            ch = self.engine._graph_objs[name]
            ch.stat.clear()
            ch.stat.update(value)
        else:
            ch = self.engine._graph_objs[name] = Character(
                self.engine, name, data=value
            )
        self.send(self, key=name, val=ch)

    def __delitem__(self, name):
        """Delete the named character from both the cache and the database."""
        it = self[name]
        it.clear()
        del self.engine._graph_objs[name]
        self.engine.query.del_graph(name)
        self.send(self, key=name, val=None)


class CompositeDict(MutableMapping):
    __slots__ = ['d1', 'd2']

    def __init__(self, d1, d2):
        """Store dictionaries"""
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

    def patch(self, d):
        for k, v in d.items():
            if k in self:
                self[k].update(v)
            else:
                self[k] = deepcopy(v)
