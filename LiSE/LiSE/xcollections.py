# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Common classes for collections in LiSE, of which most can be bound to."""
from collections import Mapping, MutableMapping, defaultdict
from .bind import dispatch, listen, listener, unlisten, unlistener


class NotThatMap(Mapping):
    """Wraps another mapping and conceals exactly one of its keys."""
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


class StringStore(MutableMapping):
    """Store strings in database, and format them with one another upon retrieval.

    In any one string, putting the key of another string in curly
    braces will cause the other string to be substituted in.

    """

    def __init__(self, qe, table='strings', lang='eng'):
        """Store the engine, the name of the database table to use, and the
        language code.

        """
        self.db = qe
        self.db.init_string_table(table)
        self.table = table
        self._language = lang
        self._lang_listeners = []
        self.cache = {}
        self._str_listeners = defaultdict(list)

    def _dispatch_lang(self, v):
        """When the language changes, call everything that's listening to
        it.

        """
        for f in self._lang_listeners:
            f(self, v)

    def lang_listener(self, fun):
        """Arrange to call the function when the language changes."""
        listen(self._lang_listeners, fun)
        return fun

    def lang_unlisten(self, fun):
        unlisten(self._lang_listeners, fun)
        return fun

    def _dispatch_str(self, k, v):
        """When some string ``k`` is set to ``v``, notify any listeners of the
        fact.

        That means listeners to ``k`` in particular, and to strings
        generally.

        """
        dispatch(self._str_listeners, k, self, k, v)

    def listener(self, fun=None, string=None):
        """Arrange to call the function when a string is set.

        With optional argument ``string``, only that particular string
        will trigger the listener. Without ``string``, every string
        will.

        """
        return listener(self._str_listeners, fun, string)

    def unlisten(self, fun=None, string=None):
        return unlistener(self._str_listeners, fun, string)

    def commit(self):
        self.db.commit()

    @property
    def language(self):
        """Get the current language."""
        return self._language

    @language.setter
    def language(self, v):
        """Invalidate the cache upon changing the language."""
        self._language = v
        self._dispatch_lang(v)
        self.cache = {}

    def __iter__(self):
        """First cache, then iterate over all string IDs for the current
        language.

        """
        for (k, v) in self.db.string_table_lang_items(
                self.table, self.language
        ):
            self.cache[k] = v
        return iter(self.cache.keys())

    def __len__(self):
        """"Count strings in the current language."""
        return self.db.count_all_table(self.table)

    def __getitem__(self, k):
        """Get the string and format it with other strings here."""
        if k not in self.cache:
            v = self.db.string_table_get(
                self.table, self.language, k
            )
            if v is None:
                raise KeyError("No string named {}".format(k))
            self.cache[k] = v
        return self.cache[k].format_map(NotThatMap(self, k))

    def __setitem__(self, k, v):
        """Set the value of a string for the current language."""
        self.cache[k] = v
        self.db.string_table_set(self.table, self.language, k, v)
        self._dispatch_str(k, v)

    def __delitem__(self, k):
        """Delete the string from the current language, and remove it from the
        cache.

        """
        del self.cache[k]
        self.db.string_table_del(self.table, self.language, k)
        self._dispatch_str(k, None)

    def lang_items(self, lang=None):
        """Yield pairs of (id, string) for the given language."""
        if lang is None:
            lang = self.language
        yield from self.db.string_table_lang_items(
            self.table, lang
        )


class StoredPartial(object):
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


class FunctionStore(MutableMapping):
    """Store functions in a SQL database"""
    def __init__(self, engine, db, table):
        """Use ``codedb`` as a connection object. Connect to it, and
        initialize the schema if needed.

        """
        self.engine = engine
        self.db = db
        self.db.init_table(table)
        self._tab = table
        self._listeners = defaultdict(list)
        self.cache = {}
        self.engine.db.init_func_table(table)

    def _dispatch(self, name, fun):
        """Call listeners to functions generally and to the named function in
        particular when it's set to a new callable.

        """
        dispatch(self._listeners, name, self, name, fun)

    def listener(self, f=None, name=None):
        """Arrange to call a listener function when a stored function changes.

        With optional argument ``name``, the listener will only be
        called when the named function changes. Otherwise it will be
        called when any stored function changes, including when it's
        set the first time.

        """
        return listener(self._listeners, f, name)

    def __len__(self):
        """Return count of all functions here."""
        return self.db.count_all_table(self._tab)

    def __iter__(self):
        """Iterate over function names in alphabetical order."""
        for row in self.db.func_table_iter(self._tab):
            yield row[0]

    def __contains__(self, name):
        """Check if there's such a function in the database"""
        if not isinstance(name, str):
            return False
        if name in self.cache:
            return True
        return self.db.func_table_contains(self._tab, name)

    def __getitem__(self, name):
        """Reconstruct the named function from its code string stored in the
        code database, and return it.

        """
        if name not in self.cache:
            try:
                self.cache[name] = self.db.func_table_get(self._tab, name)
            except KeyError:
                d = self.db.func_table_get_all(self._tab, name)
                if d['base'] not in self.cache:
                    self.cache[name] = self.db.func_table_get(self._tab, name)
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
        self.db.func_table_set(self._tab, fun.__name__, fun)
        self.cache[fun.__name__] = fun
        self._dispatch(fun.__name__, fun)
        return fun

    def __setitem__(self, name, fun):
        """Store the function, marshalled, under the name given."""
        self.db.func_table_set(self._tab, name, fun)
        self.cache[name] = fun
        self._dispatch(name, fun)

    def __delitem__(self, name):
        """Delete the named function from both the cache and the database.

        Listeners to the named function see this as if the function
        were set to ``None``.

        """
        self.db.func_table_del(self._tab, name)
        del self.cache[name]
        self._dispatch(name, None)

    def plain(self, k):
        """Return the plain source code of the function."""
        return self.db.func_table_get_plain(self._tab, k)

    def iterplain(self):
        """Iterate over (name, source) where source is in plaintext, not
        bytecode.

        """
        yield from self.db.func_table_name_plaincode(self._tab)

    def commit(self):
        """Tell my ``QueryEngine`` to commit."""
        self.db.commit()

    def set_source(self, func_name, source):
        """Set the plain, uncompiled source code of ``func_name`` to
        ``source``.

        """
        self.db.func_table_set_source(
            self._tab,
            func_name,
            source
        )

    def partial(self, funcname, **kwargs):
        part = StoredPartial(self, funcname, **kwargs)
        self.cache[funcname + self.engine.json_dump(kwargs)] = part
        return part


class GlobalVarMapping(MutableMapping):
    """Mapping for variables that are global but which I keep history for"""
    def __init__(self, engine):
        """Store the engine and initialize my private dictionary of
        listeners.

        """
        self.engine = engine
        self._listeners = defaultdict(list)

    def _dispatch(self, k, v):
        """Call everyone listening to this key, and everyone who listens to
        all keys.

        """
        (b, t) = self.engine.time
        dispatch(self._listeners, k, b, t, self, k, v)

    def listener(self, fun=None, key=None):
        """Arrange to call this function when a key is set to a new value.

        With optional argument ``key``, only call when that particular
        key changes.

        """
        return listener(self._listeners, fun, key)

    def __iter__(self):
        """Iterate over the global keys whose values aren't null at the moment.

        The values may be None, however.

        """
        for (k, v) in self.engine.db.universal_items(*self.engine.time):
            yield k

    def __len__(self):
        """Just count while iterating"""
        n = 0
        for k in iter(self):
            n += 1
        return n

    def __getitem__(self, k):
        """Get the current value of this key"""
        return self.engine.db.universal_get(k, *self.engine.time)

    def __setitem__(self, k, v):
        """Set k=v at the current branch and tick"""
        (branch, tick) = self.engine.time
        self.engine.db.universal_set(k, branch, tick, v)
        self._dispatch(k, v)

    def __delitem__(self, k):
        """Unset this key for the present (branch, tick)"""
        self.engine.db.universal_del(k)
        self._dispatch(k, None)


class CharacterMapping(MutableMapping):
    """A mapping by which to access :class:`Character` objects.

    If a character already exists, you can always get its name here to
    get the :class:`Character` object. Deleting an item here will
    delete the character from the world, even if there are still
    :class:`Character` objects referring to it; those won't do
    anything useful anymore.

    """
    def __init__(self, engine):
        """Store the engine, initialize caches"""
        self.engine = engine
        self._listeners = defaultdict(list)
        self._cache = {}

    def _dispatch(self, k, v):
        """Call anyone listening for a character named ``k``, and anyone
        listening to all characters

        """
        dispatch(self._listeners, k, self, k, v)

    def listener(self, f=None, char=None):
        """Arrange to call the function when a character is created or
        destroyed.

        With optional argument ``char``, only call when a character by
        that name is created or destroyed.

        """
        return listener(self._listeners, f, char)

    def __iter__(self):
        """Iterate over every character name."""
        return self.engine.db.characters()

    def __contains__(self, name):
        """Has this character been created?"""
        if name in self._cache:
            return True
        return self.engine.db.have_character(name)

    def __len__(self):
        """How many characters have been created?"""
        return self.engine.db.ct_characters()

    def __getitem__(self, name):
        """Return the named character, if it's been created.

        Try to use the cache if possible.

        """
        from .character import Character
        if name not in self:
            raise KeyError("No such character")
        if hasattr(self, '_cache'):
            if name not in self._cache:
                self._cache[name] = Character(self.engine, name)
            return self._cache[name]
        return Character(self.engine, name)

    def __setitem__(self, name, value):
        """Make a new character by the given name, and initialize its data to
        the given value.

        """
        from .character import Character
        if isinstance(value, Character):
            self._cache[name] = value
            return
        self._cache[name] = Character(self.engine, name, data=value)
        self._dispatch(name, self._cache[name])

    def __delitem__(self, name):
        """Delete the named character from both the cache and the database."""
        if hasattr(self, '_cache') and name in self._cache:
            del self._cache[name]
        self.engine.db.del_character(name)
        self._dispatch(name, None)


class CompositeDict(Mapping):
    """Read-only mapping that looks up values in a first dict if
    available, then a second dict if possible.

    Assumes the dicts have no overlap.

    """
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
