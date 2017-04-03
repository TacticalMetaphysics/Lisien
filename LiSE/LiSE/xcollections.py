# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Common classes for collections in LiSE, of which most can be bound to."""
from collections import Mapping, MutableMapping
from blinker import Signal


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
        inst.cache = {}
        self.send(inst, language=val)

    def __str__(self):
        return self.lang


class LanguageDescriptor(AbstractLanguageDescriptor):
    def _get_language(self, inst):
        return inst._language

    def _set_language(self, inst, val):
        inst._language = val


class StringStore(MutableMapping, Signal):
    """Store strings in database, and format them with one another upon retrieval.

    In any one string, putting the key of another string in curly
    braces will cause the other string to be substituted in.

    """
    __slots__ = ['query', 'table', 'cache', 'receivers']

    language = LanguageDescriptor()

    def __init__(self, query, table='strings', lang='eng'):
        """Store the engine, the name of the database table to use, and the
        language code.

        """
        super().__init__()
        self.query = query
        self.query.init_string_table(table)
        self.table = table
        self._language = lang
        self.cache = {}

    def commit(self):
        self.query.commit()

    def __iter__(self):
        """First cache, then iterate over all string IDs for the current
        language.

        """
        for (k, v) in self.query.string_table_lang_items(
                self.table, self.language
        ):
            self.cache[k] = v
        return iter(self.cache.keys())

    def __len__(self):
        """"Count strings in the current language."""
        return self.query.count_all_table(self.table)

    def __getitem__(self, k):
        """Get the string and format it with other strings here."""
        if k not in self.cache:
            v = self.query.string_table_get(
                self.table, self.language, k
            )
            if v is None:
                raise KeyError("No string named {}".format(k))
            self.cache[k] = v
        return self.cache[k].format_map(NotThatMap(self, k))

    def __setitem__(self, k, v):
        """Set the value of a string for the current language."""
        self.cache[k] = v
        self.query.string_table_set(self.table, self.language, k, v)
        self.send(self, key=k, val=v)

    def __delitem__(self, k):
        """Delete the string from the current language, and remove it from the
        cache.

        """
        del self.cache[k]
        self.query.string_table_del(self.table, self.language, k)
        self.send(self, key=k, val=None)

    def lang_items(self, lang=None):
        """Yield pairs of (id, string) for the given language."""
        if lang is None:
            lang = self.language
        yield from self.query.string_table_lang_items(
            self.table, lang
        )


class StoredPartial(object):
    __slots__ = ['_funcname', 'store', 'keywords', 'kwargs', 'name']

    @property
    def engine(self):
        return self.store.engine

    def __init__(self, store, name, **kwargs):
        self._funcname = name
        self.store = store
        self.keywords = list(kwargs.keys())
        self.kwargs = kwargs
        self.name = name + self.engine.json_dump(kwargs)

    def __call__(self, *args):
        return self.store[self._funcname](*args, **self.kwargs)


class FunctionStore(MutableMapping, Signal):
    """Store functions in a SQL database"""
    __slots__ = ['engine', 'query', '_tab', 'cache']

    def __init__(self, engine, query, table):
        """Use ``codedb`` as a connection object. Connect to it, and
        initialize the schema if needed.

        """
        super().__init__()
        self.engine = engine
        self.query = query
        self.query.init_table(table)
        self._tab = table
        self.cache = {}
        self.engine.query.init_func_table(table)

    def __len__(self):
        """Return count of all functions here."""
        return self.query.count_all_table(self._tab)

    def __iter__(self):
        """Iterate over function names in alphabetical order."""
        for row in self.query.func_table_iter(self._tab):
            yield row[0]

    def __contains__(self, name):
        """Check if there's such a function in the database"""
        if not isinstance(name, str):
            return False
        if name in self.cache:
            return True
        return self.query.func_table_contains(self._tab, name)

    def __getitem__(self, name):
        """Reconstruct the named function from its code string stored in the
        code database, and return it.

        """
        if name not in self.cache:
            try:
                self.cache[name] = self.query.func_table_get(self._tab, name)
            except KeyError:
                d = self.query.func_table_get_all(self._tab, name)
                if d['base'] not in self.cache:
                    self.cache[name] = self.query.func_table_get(
                        self._tab, name
                    )
                kwargs = self.engine.json_load(name[len(d['base']):])
                self.cache[name] = StoredPartial(self, d['base'], **kwargs)
        return self.cache[name]

    def __call__(self, fun):
        """Remember the function in the code database. Its key will be its
        ``__name__``.

        """
        if fun in self:
            raise KeyError(
                "Already have a function by that name. "
                "If you want to swap it out for this one, "
                "assign the new function to me like I'm a dictionary."
            )
        self.query.func_table_set(self._tab, fun.__name__, fun)
        self.cache[fun.__name__] = fun
        self.send(self, key=fun.__name__, val=fun)
        return fun

    def __setitem__(self, name, fun):
        """Store the function, marshalled, under the name given."""
        self.query.func_table_set(self._tab, name, fun)
        self.cache[name] = fun
        self.send(self, key=name, val=fun)

    def __delitem__(self, name):
        """Delete the named function from both the cache and the database.

        Listeners to the named function see this as if the function
        were set to ``None``.

        """
        self.query.func_table_del(self._tab, name)
        del self.cache[name]
        self.send(self, key=name, val=None)

    def plain(self, k):
        """Return the plain source code of the function."""
        return self.query.func_table_get_plain(self._tab, k)

    def iterplain(self):
        """Iterate over (name, source) where source is in plaintext, not
        bytecode.

        """
        yield from self.query.func_table_name_plaincode(self._tab)

    def commit(self):
        """Tell my ``QueryEngine`` to commit."""
        self.query.commit()

    def set_source(self, func_name, source):
        """Set the plain, uncompiled source code of ``func_name`` to
        ``source``.

        """
        self.query.func_table_set_source(
            self._tab,
            func_name,
            source
        )

    def partial(self, funcname, **kwargs):
        part = StoredPartial(self, funcname, **kwargs)
        self.cache[funcname + self.engine.json_dump(kwargs)] = part
        return part


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
