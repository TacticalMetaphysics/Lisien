# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
"""The core of LiSE is an object relational mapper with some special
data stores, as well as properties for manipulating the flow of
time.

"""
from random import Random
from collections import (
    defaultdict,
    deque,
    Mapping,
    MutableMapping,
    Callable
)
from sqlite3 import connect

from gorm import ORM as gORM
from .character import Character
from .node import Node
from .portal import Portal
from .rule import AllRuleBooks, AllRules
from .query import QueryEngine
from .util import AbstractEngine, dispatch, listen, listener, unlisten, unlistener, reify


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


class FunctionStoreDB(MutableMapping):
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
            self.cache[name] = self.db.func_table_get(self._tab, name)
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
        self.db.func_table_set(self._tbl, fun.__name__, fun)
        self.cache[fun.__name__] = fun
        self._dispatch(fun.__name__, fun)

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

    If a character already exists, you can always get its name here to get the :class:`Character` object. Deleting an
    item here will delete the character from the world, even if there are still :class:`Character` objects referring
    to it; those won't do anything useful anymore.

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


crhandled_defaultdict = lambda: defaultdict(  # character:
    lambda: defaultdict(
        # rulebook:
        lambda: defaultdict(
            # rule:
            lambda: defaultdict(
                # branch:
                set
                # ticks handled
            )
        )
    )
)


class Engine(AbstractEngine):
    """LiSE, the Life Simulator Engine.

    Each instance of LiSE maintains a connection to a database
    representing the state of a simulated world. Simulation rules
    within this world are described by lists of Python functions, some
    of which make changes to the world.

    The top-level data structure within LiSE is the character. Most
    data within the world model is kept in some character or other;
    these will quite frequently represent people, but can be readily
    adapted to represent any kind of data that can be comfortably
    described as a graph or a JSON object. Every change to a character
    will be written to the database.

    LiSE tracks history as a series of ticks. In each tick, each
    simulation rule is evaluated once for each of the simulated
    entities it's been applied to. World changes in a given tick are
    remembered together, such that the whole world state can be
    rewound: simply set the properties ``branch`` and ``tick`` back to
    what they were just before the change you want to undo.

    """
    char_cls = Character
    node_cls = Node
    portal_cls = Portal

    @reify
    def _rulebooks_cache(self):
        assert(self.caching)
        return defaultdict(list)

    @reify
    def _characters_rulebooks_cache(self):
        assert(self.caching)
        return {}

    @reify
    def _nodes_rulebooks_cache(self):
        assert(self.caching)
        return defaultdict(dict)

    @reify
    def _portals_rulebooks_cache(self):
        assert(self.caching)
        return defaultdict(
            lambda: defaultdict(dict)
        )

    @reify
    def _avatarness_cache(self):
        assert(self.caching)
        return defaultdict(  # character:
            lambda: defaultdict(  # graph:
                lambda: defaultdict(  # node:
                    lambda: defaultdict(  # branch:
                        dict  # tick: is_avatar
                    )
                )
            )
        )

    @reify
    def _active_rules_cache(self):
        assert(self.caching)
        return defaultdict(  # rulebook:
            lambda: defaultdict(  # rule:
                lambda: defaultdict(  # branch:
                    dict  # tick: active
                )
            )
        )

    @reify
    def _node_rules_handled_cache(self):
        assert(self.caching)
        return defaultdict(  # character:
            lambda: defaultdict(  # node:
                lambda: defaultdict(  # rulebook:
                    lambda: defaultdict(  # rule:
                        lambda: defaultdict(  # branch:
                            set  # ticks handled
                        )
                    )
                )
            )
        )

    @reify
    def _portal_rules_handled_cache(self):
        assert(self.caching)
        return defaultdict(  # character:
            lambda: defaultdict(  # nodeA:
                lambda: defaultdict(  # nodeB:
                    lambda: defaultdict(  # rulebook:
                        lambda: defaultdict(  # rule:
                            lambda: defaultdict(  # branch:
                                set  # ticks handled
                            )
                        )
                    )
                )
            )
        )

    @reify
    def _character_rules_handled_cache(self):
        assert(self.caching)
        return crhandled_defaultdict()

    @reify
    def _avatar_rules_handled_cache(self):
        assert(self.caching)
        return crhandled_defaultdict()

    @reify
    def _character_thing_rules_handled_cache(self):
        assert(self.caching)
        return crhandled_defaultdict()

    @reify
    def _character_place_rules_handled_cache(self):
        assert(self.caching)
        return crhandled_defaultdict()

    @reify
    def _character_node_rules_handled_cache(self):
        assert(self.caching)
        return crhandled_defaultdict()

    @reify
    def _character_portal_rules_handled_cache(self):
        assert(self.caching)
        return crhandled_defaultdict()

    @reify
    def _things_cache(self):
        assert(self.caching)
        return defaultdict(  # character:
            lambda: defaultdict(  # thing:
                lambda: defaultdict(  # branch:
                    dict  # tick: (location, next_location)
                )
            )
        )

    @reify
    def db(self):
        return self.gorm.db

    @reify
    def gorm(self):
        return gORM(
            self._worlddb,
            connect_args=self._connect_args,
            alchemy=self._alchemy,
            query_engine_class=QueryEngine,
            json_dump=self.json_dump,
            json_load=self.json_load,
            caching=self.caching
        )

    def __new__(
            cls,
            worlddb,
            codedb,
            connect_args={},
            alchemy=False,
            caching=True,
            commit_modulus=None,
            random_seed=None
    ):
        r = super(cls, Engine).__new__(cls)
        r._worlddb = worlddb
        r._codedb = codedb
        r._connect_args = connect_args
        r._alchemy = alchemy
        r.caching = caching
        r.commit_modulus = commit_modulus
        r.random_seed = random_seed
        return r

    @reify
    def codedb(self):
        return connect(self._codedb)

    @reify
    def worlddb(self):
        if hasattr(self.gorm.db, 'alchemist'):
            return self.gorm.db.alchemist.conn.connection
        else:
            return self.gorm.db.connection

    def __init__(self, *args):
        """Store the connections for the world database and the code database;
        set up listeners; and start a transaction

        """
        self._time_listeners = []
        self._code_qe = QueryEngine(
            self.codedb, self._connect_args, self._alchemy, self.json_dump, self.json_load
        )
        self.eternal = self.db.globl
        self.db.initdb()
        self._code_qe.initdb()
        self._existence = {}
        if self.caching:
            # caches for the rule poller
            for (rulebook, rule) in self.rule.db.rulebooks_rules():
                self._rulebooks_cache[rulebook].append(rule)
            for (
                    character,
                    character_rulebook,
                    avatar_rulebook,
                    character_thing_rulebook,
                    character_place_rulebook,
                    character_node_rulebook,
                    character_portal_rulebook
            ) in self.db.characters_rulebooks():
                self._characters_rulebooks_cache[character] = {
                    'character': character_rulebook,
                    'avatar': avatar_rulebook,
                    'character_thing': character_thing_rulebook,
                    'character_place': character_place_rulebook,
                    'character_node': character_node_rulebook,
                    'character_portal': character_portal_rulebook
                }
            for (character, node, rulebook) in self.db.nodes_rulebooks():
                self._nodes_rulebooks_cache[character][node] = rulebook
            for (
                    character,
                    nodeA,
                    nodeB,
                    rulebook
            ) in self.db.portals_rulebooks():
                self._portals_rulebooks_cache[character][nodeA][nodeB] \
                    = rulebook
            for (
                    rulebook, rule, branch, tick, active
            ) in self.db.dump_active_rules():
                self._active_rules_cache[rulebook][rule][branch][tick] = active
            for (
                    character, node, rulebook, rule, branch, tick
            ) in self.db.dump_node_rules_handled():
                self._node_rules_handled_cache[
                    character
                ][node][rulebook][rule][branch].add(tick)
            for (
                    character, nodeA, nodeB, idx, rulebook, rule, branch, tick
            ) in self.db.dump_portal_rules_handled():
                self._portal_rules_handled_cache[
                    character
                ][nodeA][nodeB][rulebook][rule][branch].add(tick)
            for (dumper, cache) in [
                (
                    self.db.handled_character_rules,
                    self._character_rules_handled_cache
                ),
                (
                    self.db.handled_avatar_rules,
                    self._avatar_rules_handled_cache
                ),
                (
                    self.db.handled_character_thing_rules,
                    self._character_thing_rules_handled_cache
                ),
                (
                    self.db.handled_character_place_rules,
                    self._character_place_rules_handled_cache
                ),
                (
                    self.db.handled_character_node_rules,
                    self._character_node_rules_handled_cache
                ),
                (
                    self.db.handled_character_portal_rules,
                    self._character_portal_rules_handled_cache
                )
            ]:
                for (
                        character, rulebook, rule, branch, tick
                ) in dumper():
                    cache[character][rulebook][rule][branch].add(tick)
            # world caches
            for (
                    character, thing, branch, tick, loc, nextloc
            ) in self.db.things_dump():
                self._things_cache[character][thing][branch][tick] \
                    = (loc, nextloc)
            for (
                    character, graph, avatar, branch, tick, is_avatar
            ) in self.db.avatarness_dump():
                self._avatarness_cache[character][graph][avatar][branch][tick] \
                    = is_avatar
        self._rules_iter = self._follow_rules()
        # set up the randomizer
        self.rando = Random()
        if 'rando_state' in self.universal:
            self.rando.setstate(self.universal['rando_state'])
        else:
            self.rando.seed(self.random_seed)
            self.universal['rando_state'] = self.rando.getstate()
        self.betavariate = self.rando.betavariate
        self.choice = self.rando.choice
        self.expovariate = self.rando.expovariate
        self.gammaraviate = self.rando.gammavariate
        self.gauss = self.rando.gauss
        self.getrandbits = self.rando.getrandbits
        self.lognormvariate = self.rando.lognormvariate
        self.normalvariate = self.rando.normalvariate
        self.paretovariate = self.rando.paretovariate
        self.randint = self.rando.randint
        self.random = self.rando.random
        self.randrange = self.rando.randrange
        self.sample = self.rando.sample
        self.shuffle = self.rando.shuffle
        self.triangular = self.rando.triangular
        self.uniform = self.rando.uniform
        self.vonmisesvariate = self.rando.vonmisesvariate
        self.weibullvariate = self.rando.weibullvariate

    @reify
    def action(self):
        return FunctionStoreDB(self, self._code_qe, 'actions')

    @reify
    def prereq(self):
        return FunctionStoreDB(self, self._code_qe, 'prereqs')

    @reify
    def trigger(self):
        return FunctionStoreDB(self, self._code_qe, 'triggers')

    @reify
    def function(self):
        return FunctionStoreDB(self, self._code_qe, 'functions')

    @property
    def stores(self):
        return (
            self.action,
            self.prereq,
            self.trigger,
            self.function,
            self.string
        )

    @reify
    def rule(self):
        return AllRules(self, self._code_qe)

    @reify
    def rulebook(self):
        return AllRuleBooks(self, self._code_qe)

    @reify
    def string(self):
        return StringStore(self._code_qe)

    @reify
    def universal(self):
        return GlobalVarMapping(self)

    @reify
    def character(self):
        return CharacterMapping(self)

    def _node_exists(self, graph, node):
        """Version of gorm's ``_node_exists`` that caches stuff"""
        if not self.caching:
            return node in self.gorm.get_graph(graph).node
        (branch, rev) = self.time
        if graph not in self._existence:
            self._existence[graph] = {}
        if node not in self._existence[graph]:
            self._existence[graph][node] = {}
        if branch not in self._existence[graph][node]:
            self._existence[graph][node][branch] = {}
        d = self._existence[graph][node][branch]
        if rev not in d:
            try:
                d[rev] = d[max(k for k in d.keys() if k < rev)]
            except ValueError:
                d[rev] = self.db.node_exists(graph, node, branch, rev)
        return self._existence[graph][node][branch][rev]

    def coinflip(self):
        """Return True or False with equal probability."""
        return self.choice((True, False))

    def roll_die(self, d):
        """Roll a die with ``d`` faces. Return the result."""
        return self.randint(1, d)

    def dice(self, n, d):
        """Roll ``n`` dice with ``d`` faces, and yield the results.

        """
        for i in range(0, n):
            yield self.roll_die(d)

    def dice_check(self, n, d, target, comparator=lambda x, y: x <= y):
        """Roll ``n`` dice with ``d`` sides, sum them, and return whether they
        are <= ``target``.

        If ``comparator`` is provided, use it instead of <=.

        """
        return comparator(sum(self.dice(n, d)), target)

    def percent_chance(self, pct):
        """Given a ``pct``% chance of something happening right now, decide at
        random whether it actually happens, and return ``True`` or
        ``False`` as appropriate.

        Values not between 0 and 100 are treated as though they
        were 0 or 100, whichever is nearer.

        """
        if pct <= 0:
            return False
        if pct >= 100:
            return True
        return pct / 100 < self.random()

    def commit(self):
        """Commit to both the world and code databases, and begin a new
        transaction for the world database

        """
        if self.caching:
            self.gorm.branch = self.gorm._obranch
            self.gorm.rev = self.gorm._orev
        for store in self.stores:
            store.commit()
        self.gorm.commit()

    def close(self):
        """Commit changes and close the database."""
        self.commit()
        self.gorm.close()

    def __enter__(self):
        """Return myself. For compatibility with ``with`` semantics."""
        return self

    def __exit__(self, *args):
        """Close on exit."""
        self.close()

    def time_listener(self, v):
        """Arrange to call a function whenever my ``branch`` or ``tick``
        changes.

        The arguments will be the old branch and tick followed by the
        new branch and tick.

        """
        if not isinstance(v, Callable):
            raise TypeError("This is a decorator")
        if v not in self._time_listeners:
            self._time_listeners.append(v)
        return v

    def time_unlisten(self, v):
        if v in self._time_listeners:
            self._time_listeners.remove(v)
        return v

    @property
    def branch(self):
        """Return my gorm's branch"""
        return self.gorm.branch

    @branch.setter
    def branch(self, v):
        """Set my gorm's branch and call listeners"""
        (b, t) = self.time
        if self.caching:
            if v == b:
                return
            if v not in self._branches:
                parent = b
                child = v
                assert(parent in self._branches)
                self._branch_parents[child] = parent
                self._branches[parent][child] = {}
                self._branches[child] = self._branches[parent][child]
                self._branches_start[child] = t
            self.gorm._obranch = v
        else:
            self.gorm.branch = v
        if not hasattr(self, 'locktime'):
            for time_listener in self._time_listeners:
                time_listener(b, t, v, t)

    @property
    def tick(self):
        """Return my gorm's ``rev``"""
        return self.gorm.rev

    @tick.setter
    def tick(self, v):
        """Update gorm's ``rev``, and call listeners"""
        if not isinstance(v, int):
            raise TypeError("tick must be integer")
        (branch_then, tick_then) = self.time
        if self.caching:
            if v == self.tick:
                return
            self.gorm._orev = v
        else:
            self.gorm.rev = v
        if not hasattr(self, 'locktime'):
            for time_listener in self._time_listeners:
                time_listener(branch_then, tick_then, branch_then, v)

    @property
    def time(self):
        """Return tuple of branch and tick"""
        return (self.branch, self.tick)

    @time.setter
    def time(self, v):
        """Set my ``branch`` and ``tick``, and call listeners"""
        (branch_then, tick_then) = self.time
        (branch_now, tick_now) = v
        relock = hasattr(self, 'locktime')
        self.locktime = True
        # setting tick and branch in this order makes it practical to
        # track the timestream genealogy
        self.tick = tick_now
        self.branch = branch_now
        if not relock:
            del self.locktime
        if not hasattr(self, 'locktime'):
            for time_listener in self._time_listeners:
                time_listener(
                    branch_then, tick_then, branch_now, tick_now
                )

    def _active_branches(self, branch=None, tick=None):
        yield from self.gorm._active_branches(branch, tick)

    def _branch_descendants(self, branch=None):
        yield from self.gorm._branch_descendants(branch)

    def _rule_active(self, rulebook, rule):
        cache = self._active_rules_cache[rulebook][rule]
        for (branch, tick) in self._active_branches(*self.time):
            if branch in cache:
                return cache[branch][
                    max(t for t in cache[branch].keys() if t <= tick)
                ]
        return False

    def _poll_char_rules(self):
        if not self.caching:
            return self.db.poll_char_rules(*self.time)

        def handled(rulebook, rule):
            cache = self._character_rules_handled_cache
            return (
                rulebook in cache and
                rule in cache[rulebook] and
                self.branch in cache[rulebook][rule] and
                self.tick in cache[rulebook][rule][self.branch]
            )

        for char in self._characters_rulebooks_cache:
            for (rulemap, rulebook) in self._characters_rulebooks_cache[
                char].items():
                for rule in self._rulebooks_cache[rulebook]:
                    if self._rule_active(rulebook, rule) and not \
                            handled(rulebook, rule):
                        yield (rulemap, char, rulebook, rule)

    def _poll_node_rules(self):
        if not self.caching:
            return self.db.poll_node_rules(*self.time)
        cache = self._node_rules_handled_cache

        def handled(char, node, rulebook, rule):
            return (
                char in cache and
                node in cache[char] and
                rulebook in cache[char][node] and
                rule in cache[char][rule][rulebook] and
                self.branch in cache[char][rule][rulebook] and
                self.tick in cache[char][rule][rulebook][self.branch]
            )

        for char in self._nodes_rulebooks_cache:
            for (node, rulebook) in self._nodes_rulebooks_cache[char].items():
                for rule in self._rulebooks_cache[rulebook]:
                    if (
                                self._rule_active(rulebook, rule) and not
                            handled(char, node, rulebook, rule)
                    ):
                        yield ('node', char, node, rulebook, rule)

    def _poll_portal_rules(self):
        if not self.caching:
            return self.db.poll_portal_rules(*self.time)

        def handled(char, nodeA, nodeB, rulebook, rule):
            cache = self._portal_rules_handled_cache
            return (
                char in cache and
                nodeA in cache[char] and
                nodeB in cache[char][nodeA] and
                rulebook in cache[char][nodeA][nodeB] and
                rule in cache[char][nodeA][nodeB][rulebook] and
                self.branch in cache[char][nodeA][nodeB][rulebook][rule] and
                self.tick in cache[char][nodeA][nodeB][rulebook][rule][
                    self.branch
                ]
            )

        cache = self._portals_rulebooks_cache
        for char in cache:
            for nodeA in cache[char]:
                for (nodeB, rulebook) in cache[char][nodeA].items():
                    for rule in self._rulebooks_cache[rulebook]:
                        if (
                                    self._rule_active(rulebook, rule) and not
                                handled(char, nodeA, nodeB, rulebook, rule)
                        ):
                            yield ('portal', char, nodeA, nodeB, rulebook, rule)

    def _poll_rules(self):
        """Iterate over tuples containing rules yet unresolved in the current tick.

        The tuples are of the form: ``(ruletype, character, entity,
        rulebook, rule)`` where ``ruletype`` is what kind of entity
        the rule is about (character', 'thing', 'place', or
        'portal'), and ``entity`` is the :class:`Place`,
        :class:`Thing`, or :class:`Portal` that the rule is attached
        to. For character-wide rules it is ``None``.

        """
        for (
                rulemap, character, rulebook, rule
        ) in self._poll_char_rules():
            try:
                yield (
                    rulemap,
                    self.character[character],
                    None,
                    rulebook,
                    self.rule[rule]
                )
            except KeyError:
                continue
        for (
                character, node, rulebook, rule
        ) in self._poll_node_rules():
            try:
                c = self.character[character]
                n = c.node[node]
            except KeyError:
                continue
            typ = 'thing' if hasattr(n, 'location') else 'place'
            yield typ, c, n, rulebook, self.rule[rule]
        for (
                character, a, b, i, rulebook, rule
        ) in self._poll_portal_rules():
            try:
                c = self.character[character]
                yield 'portal', c.portal[a][b], rulebook, self.rule[rule]
            except KeyError:
                continue

    def _handled_thing_rule(self, char, thing, rulebook, rule, branch, tick):
        if self.caching:
            cache = self._node_rules_handled_cache
            cache[char][thing][rulebook][rule][branch].add(tick)
        self.db.handled_thing_rule(
            char, thing, rulebook, rule, branch, tick
        )

    def _handled_place_rule(self, char, place, rulebook, rule, branch, tick):
        if self.caching:
            cache = self._node_rules_handled_cache
            cache[char][place][rulebook][rule][branch].add(tick)
        self.db.handled_place_rule(
            char, place, rulebook, rule, branch, tick
        )

    def _handled_portal_rule(
            self, char, nodeA, nodeB, rulebook, rule, branch, tick
    ):
        if self.caching:
            cache = self._portal_rules_handled_cache
            cache[char][nodeA][nodeB][rulebook][rule][branch].add(tick)
        self.db.handled_portal_rule(
            char, nodeA, nodeB, rulebook, rule, branch, tick
        )

    def _handled_character_rule(
            self, typ, char, rulebook, rule, branch, tick
    ):
        if self.caching:
            cache = {
                'character': self._character_rules_handled_cache,
                'avatar': self._avatar_rules_handled_cache,
                'character_thing': self._character_thing_rules_handled_cache,
                'character_place': self._character_place_rules_handled_cache,
                'character_node': self._character_node_rules_handled_cache,
                'character_portal': self._character_portal_rules_handled_cache,
            }[typ]
            cache[char][rulebook][rule][branch].add(tick)
        self.db.handled_character_rule(
            typ, char, rulebook, rule, branch, tick
        )

    def _follow_rules(self):
        """For each rule in play at the present tick, call it and yield a
        tuple describing the results.

        Tuples are of the form: ``(returned, rulename, ruletype,
        rulebook)`` where ``returned`` is whatever the rule itself
        returned upon being called, and ``ruletype`` is what sort of
        entity the rule applies to.

        """
        (branch, tick) = self.time
        for (typ, character, entity, rulebook, rule) in self._poll_rules():
            def follow(*args):
                return (rule(self, *args), rule.name, typ, rulebook)

            if typ in ('thing', 'place', 'portal'):
                yield follow(character, entity)
                if typ == 'thing':
                    self._handled_thing_rule(
                        character.name,
                        entity.name,
                        rulebook,
                        rule.name,
                        branch,
                        tick
                    )
                elif typ == 'place':
                    self._handled_place_rule(
                        character.name,
                        entity.name,
                        rulebook,
                        rule.name,
                        branch,
                        tick
                    )
                else:
                    self._handled_portal_rule(
                        character.name,
                        entity.origin.name,
                        entity.destination.name,
                        rulebook,
                        rule.name,
                        branch,
                        tick
                    )
            else:
                if typ == 'character':
                    yield follow(character)
                elif typ == 'avatar':
                    for avatar in character.avatars():
                        yield follow(character, avatar)
                elif typ == 'character_thing':
                    for thing in character.thing.values():
                        yield follow(character, thing)
                elif typ == 'character_place':
                    for place in character.place.values():
                        yield follow(character, place)
                elif typ == 'character_node':
                    for node in character.node.values():
                        yield follow(character, node)
                elif typ == 'character_portal':
                    for portal in character.portal.values():
                        yield follow(character, portal)
                else:
                    raise ValueError('Unknown type of rule')
                self._handled_character_rule(
                    typ, character.name, rulebook, rule.name, branch, tick
                )

    def advance(self):
        """Follow the next rule, or if there isn't one, advance to the next
        tick.

        """
        try:
            r = next(self._rules_iter)
        except StopIteration:
            self.tick += 1
            self._rules_iter = self._follow_rules()
            self.universal['rando_state'] = self.rando.getstate()
            if self.commit_modulus and self.tick % self.commit_modulus == 0:
                self.gorm.commit()
            r = None
        return r

    def next_tick(self):
        """Call ``advance`` repeatedly, appending its results to a list until
        the tick has ended.  Return the list.

        """
        curtick = self.tick
        r = []
        while self.tick == curtick:
            r.append(self.advance())
        # The last element is always None, but is not a sentinel; any
        # rule may return None.
        return r[:-1]

    def new_character(self, name, **kwargs):
        """Create and return a new :class:`Character`."""
        self.add_character(name, **kwargs)
        return self.character[name]

    def add_character(self, name, data=None, **kwargs):
        """Create the :class:`Character` so it'll show up in my ``character``
        mapping.

        """
        self.gorm.new_digraph(name, data, **kwargs)
        ch = Character(self, name)
        if data is not None:
            for a in data.adj:
                for b in data.adj[a]:
                    assert(
                        a in ch.adj and
                        b in ch.adj[a]
                    )
        if hasattr(self.character, '_cache'):
            self.character._cache[name] = ch

    def del_character(self, name):
        """Remove the Character from the database entirely.

        This also deletes all its history. You'd better be sure.

        """
        self.db.del_character(name)
        self.gorm.del_graph(name)
        del self.character[name]

    def _is_thing(self, character, node):
        """Private utility function to find out if a node is a Thing or not.

        ``character`` argument must be the name of a character, not a
        :class:`Character` object. Likewise ``node`` argument is the
        node's ID.

        """
        return self.db.node_is_thing(character, node, *self.time)
