# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Common classes for collections in LiSE, of which most can be bound to."""
from collections import Mapping, MutableMapping
from blinker import Signal
from astunparse import dump as dumpast
from ast import parse, Expr, Name, Store
from inspect import getsource, getmodule
import json


class NotThatMap(Mapping):
    """Wraps another mapping and conceals exactly one of its keys."""
    __slots__ = ['inner', 'k']

    def __init__(self, inner, k):
        """Store the inner mapping and the key to hide."""
        self.inner = inner
        self.k = k

    def __iter__(self):
        """Iterate over every key except the one I'm hiding."""
        for key in self.inner:
            if key != self.k:
                yield key

    def __len__(self):
        """Return the length of my inner mapping minus one, on the assumption
        that at least that one key is present in the inner mapping.

        """
        return len(self.inner) - 1

    def __getitem__(self, key):
        """Raise ``KeyError`` if you're trying to get the hidden key."""
        if key == self.k:
            raise KeyError("masked")
        return self.inner[key]


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
        return self.cache[self.language][k].format_map(NotThatMap(self, k))

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
        yield from self.cache[lang].items()

    def save(self):
        with open(self._filename, 'w') as outf:
            json.dump(self.cache, outf, indent=4, sort_keys=True)


class FunctionStore(Signal):
    def __init__(self, filename):
        self._filename = filename
        with open(filename, 'r') as inf:
            self._ast = parse(inf.read(), filename)
            self._ast_idx = {}
            for i, node in enumerate(self._ast.body):
                if hasattr(node, 'value') and hasattr(node.value, 'func'):
                    self._ast_idx[node.value.func.id] = i
            self._globl = {}
            self._locl = {}
            self._code = exec(compile(self._ast, filename, 'exec'), self._globl, self._locl)

    def __getattr__(self, k):
        return self._locl[k]

    def __setattr__(self, k, v):
        if not callable(v):
            super().__setattr__(k, v)
            return
        self._locl[k] = v
        expr = Expr(value=parse(getsource(v), getmodule(v).__name__, 'exec'))
        expr.value.func.id = k
        if k in self._ast_idx:
            self._ast.body[self._ast_idx[k]] = expr
        else:
            self._ast_idx[k] = len(self._ast.body)
            self._ast.body.append(expr)
        self.send(self, attr=k, val=v)

    def __call__(self, v):
        setattr(self, v.__name__, v)

    def __delattr__(self, k):
        del self._locl[k]
        delattr(self._ast, k)
        self.send(self, attr=k, val=None)

    def save(self):
        with open(self._filename, 'w') as outf:
            outf.write(dumpast(self._ast))


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
        return self.engine._universal_cache.iter_keys(*self.engine.time)

    def __len__(self):
        return self.engine._universal_cache.count_keys(*self.engine.time)

    def __getitem__(self, k):
        """Get the current value of this key"""
        return self.engine._universal_cache.retrieve(k, *self.engine.time)

    def __setitem__(self, k, v):
        """Set k=v at the current branch and tick"""
        (branch, tick) = self.engine.time
        self.engine.query.universal_set(k, branch, tick, v)
        self.engine._universal_cache.store(k, branch, tick, v)
        self.send(self, key=k, val=v)

    def __delitem__(self, k):
        """Unset this key for the present (branch, tick)"""
        branch, tick = self.engine.time
        self.engine.query.universal_del(k, branch, tick)
        self.engine._universal_cache.store(k, branch, tick, None)
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

    def __iter__(self):
        """Iterate over every character name."""
        return iter(self.engine._graph_objs)

    def __contains__(self, name):
        """Has this character been created?"""
        if name in self.engine._graph_objs:
            return True
        return self.engine.query.have_character(name)

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
        if name in self.engine._graph_objs:
            del self.engine._graph_objs[name]
        self.engine.query.del_character(name)
        self.send(self, key=name, val=None)


class CompositeDict(Mapping):
    """Read-only mapping that looks up values in a first dict if
    available, then a second dict if possible.

    Assumes the dicts have no overlap.

    """
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
