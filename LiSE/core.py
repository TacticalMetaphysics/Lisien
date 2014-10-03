# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
"""Object relational mapper that serves Characters."""
from random import Random
from collections import MutableMapping, Callable
from sqlite3 import connect
from gorm import ORM as gORM
from .character import Character
from .rule import AllRules
from .query import QueryEngine


alchemyOpError = None
try:
    from sqlalchemy.exc import OperationalError as alchemyOpError
except ImportError:
    pass
from sqlite3 import OperationalError as liteOpError
OperationalError = (alchemyOpError, liteOpError)

alchemyIntegError = None
try:
    from sqlalchemy.exc import IntegrityError as alchemyIntegError
except ImportError:
    pass
from sqlite3 import IntegrityError as liteIntegError
IntegrityError = (alchemyIntegError, liteIntegError)


class FunctionStoreDB(MutableMapping):
    """Store functions in a SQL database"""
    def __init__(self, engine, codedb, table):
        """Use ``codedb`` as a connection object. Connect to it, and
        initialize the schema if needed.

        """
        self.engine = engine
        self.connection = codedb
        self._tab = table
        self.cache = {}
        self.engine.db.init_func_table(table)

    def __len__(self):
        """SELECT COUNT(*) FROM {}""".format(self._tab)
        return self.engine.db.count_all_table(self._tab)

    def __iter__(self):
        """SELECT name FROM {} ORDER BY name""".format(self._tab)
        yield from self.engine.db.func_table_items(self._tab)

    def __contains__(self, name):
        """Check if there's such a function in the database"""
        if name in self.cache:
            return True
        return self.engine.db.func_table_contains(self._tab, name)

    def __getitem__(self, name):
        """Reconstruct the named function from its code string stored in the
        code database, and return it.

        """
        if name not in self.cache:
            self.cache[name] = self.engine.db.func_table_get(self._tab, name)
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
        self.engine.db.func_table_set(self._tbl, fun.__name__, fun.__code__)
        self.cache[fun.__name__] = fun

    def __setitem__(self, name, fun):
        """Store the function, marshalled, under the name given."""
        self.engine.db.func_table_set(self._tab, name, fun.__code__)
        self.cache[name] = fun

    def __delitem__(self, name):
        self.engine.db.func_table_del(self._tab, name)
        del self.cache[name]

    def decompiled(self, name):
        """Use unpyc3 to decompile the function named ``name`` and return the
        resulting unpyc3.DefStatement.

        unpyc3 is imported here, so if you never use this you don't
        need unpyc3.

        """
        from unpyc3 import decompile
        return decompile(self[name])

    def definition(self, name):
        """Return a string showing how the function named ``name`` was
        originally defined.

        It will be decompiled from the bytecode stored in the
        database. Requires unpyc3.

        """
        return str(self.decompiled(name))


class GlobalVarMapping(MutableMapping):
    """Mapping for variables that are global but which I keep history for"""
    def __init__(self, engine):
        """Store the engine"""
        self.engine = engine

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

    def __delitem__(self, k):
        """Unset this key for the present (branch, tick)"""
        self.engine.db.universal_del(k)


class CharacterMapping(MutableMapping):
    def __init__(self, engine):
        self.engine = engine
        if self.engine.caching:
            self._cache = {}

    def __iter__(self):
        yield from self.engine.db.characters()

    def __contains__(self, name):
        return self.engine.db.have_character(name)

    def __len__(self):
        return self.engine.db.ct_characters()

    def __getitem__(self, name):
        if hasattr(self, '_cache'):
            if name not in self._cache:
                if name not in self:
                    raise KeyError("No such character")
                self._cache[name] = Character(self.engine, name)
            return self._cache[name]
        if name not in self:
            raise KeyError("No such character")
        return Character(self.engine, name)

    def __setitem__(self, name, value):
        if isinstance(value, Character):
            self._cache[name] = value
            return
        self._cache[name] = Character(self.engine, name, data=value)

    def __delitem__(self, name):
        if hasattr(self, '_cache') and name in self._cache:
            del self._cache[name]
        self.engine.db.del_character(name)


class Engine(object):
    def __init__(
            self,
            worlddb,
            codedb,
            connect_args={},
            alchemy=False,
            caching=True,
            commit_modulus=None,
            random_seed=None,
            gettext=lambda s: s,
            dicecmp=lambda x, y: x <= y
    ):
        """Store the connections for the world database and the code database;
        set up listeners; and start a transaction

        """
        self.caching = caching
        self.commit_modulus = commit_modulus
        self.gettext = gettext
        self.dicecmp = dicecmp
        self.random_seed = random_seed
        self.codedb = connect(codedb)
        self.gorm = gORM(
            worlddb,
            connect_args=connect_args,
            alchemy=alchemy,
            query_engine_class=QueryEngine
        )
        self.db = self.gorm.db
        self.time_listeners = []
        self.rule = AllRules(self)
        self.eternal = self.db.globl
        self.universal = GlobalVarMapping(self)
        self.character = CharacterMapping(self)
        # start the database
        stores = ('action', 'prereq', 'trigger', 'sense', 'function')
        for store in stores:
            setattr(self, store, FunctionStoreDB(self, self.codedb, store))
        if hasattr(self.gorm.db, 'alchemist'):
            self.worlddb = self.gorm.db.alchemist.conn.connection
        else:
            self.worlddb = self.gorm.db.connection
        self.db.initdb()
        if self.caching:
            self.gorm._obranch = self.gorm.branch
            self.gorm._orev = self.gorm.rev
            self._active_branches_cache = []
        for n in self.db.characters():
            self.character[n] = Character(self, n)
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
        self._existence = {}

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

    def dice(self, n, d):
        """Roll ``n`` dice with ``d`` faces, and return a list of the
        results.

        """
        return [self.randint(1, d) for i in range(0, n)]

    def dice_check(self, n, d, target):
        """Roll ``n`` dice with ``d`` sides, sum them, compare the total to
        ``target``, and return the result.

        The comparison operation defaults to <=. You can specify a
        different one in the ``dicecmp`` argument to my
        constructor. If you need a different comparison for a
        particular roll, call ``sum(self.dice(n, d))`` and do your own
        comparison on the result.

        """
        return self.dicecmp(sum(self.dice(n, d)), target)

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
        self.gorm.commit()

    def close(self):
        if self.caching:
            self.gorm.branch = self.gorm._obranch
            self.gorm.rev = self.gorm._orev
        self.gorm.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def on_time(self, v):
        if not isinstance(v, Callable):
            raise TypeError("This is a decorator")
        self.time_listeners.append(v)

    @property
    def branch(self):
        return self.gorm.branch

    @branch.setter
    def branch(self, v):
        """Set my gorm's branch and call listeners"""
        if self.caching:
            if v == self.branch:
                return
            self.gorm._obranch = v
        else:
            self.gorm.branch = v
        if not hasattr(self, 'locktime'):
            t = self.tick
            for time_listener in self.time_listeners:
                time_listener(self, v, t)

    @property
    def tick(self):
        return self.gorm.rev

    @tick.setter
    def tick(self, v):
        """Update orm's tick, and call listeners"""
        if self.caching:
            if v == self.tick:
                return
            self.gorm._orev = v
        else:
            self.gorm.rev = v
        if not hasattr(self, 'locktime'):
            b = self.branch
            for time_listener in self.time_listeners:
                time_listener(self, b, v)

    @property
    def time(self):
        """Return tuple of branch and tick"""
        return (self.branch, self.tick)

    @time.setter
    def time(self, v):
        """Set my gorm's ``branch`` and ``tick``, and call listeners"""
        self.locktime = True
        (self.branch, self.tick) = v
        (b, t) = v
        for time_listener in self.time_listeners:
            time_listener(self, b, t)
        del self.locktime

    def _active_branches(self):
        yield from self.gorm._active_branches()

    def _poll_rules(self):
        yield from self.db.poll_rules(*self.time)

    def _follow_rules(self):
        (branch, tick) = self.time
        for (ruletyp, charname, rulebook, rulename) in self._poll_rules():
            character = self.character[charname]
            rule = self.rule[rulename]

            def follow(*args):
                return (rule(self, *args), rulename, ruletyp, rulebook)

            if ruletyp == 'character':
                yield follow(character)
            elif ruletyp == 'avatar':
                for avatar in character.avatars():
                    yield follow(character, avatar)
            elif ruletyp == 'thing':
                for thing in character.thing.values():
                    yield follow(character, thing)
            elif ruletyp == 'place':
                for place in character.place.values():
                    yield follow(character, place)
            elif ruletyp == 'portal':
                for portal in character.portals():
                    yield follow(character, portal)
            else:
                raise TypeError("Unknown type of rule")
            self.db.handled_rule(
                ruletyp, charname, rulebook, rulename, branch, tick
            )

    def advance(self):
        """Follow the next rule, or if there isn't one, advance to the next
        tick.

        """
        try:
            r = next(self._rules_iter)
        except StopIteration:
            # if not self._have_rules():
            #     raise ValueError(
            #         "No rules available; can't advance."
            #     )
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
        return r[:-1]

    def new_character(self, name, **kwargs):
        """Create and return a character"""
        self.add_character(name, **kwargs)
        return self.character[name]

    def add_character(self, name, data=None, **kwargs):
        """Create the Character so it'll show up in my `character` dict"""
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
        """Remove the Character from the database entirely"""
        self.db.del_character(name)
        self.gorm.del_graph(name)
        del self.character[name]

    def _is_thing(self, character, node):
        """Private utility function to find out if a node is a Thing or not.

        ``character`` argument must be the name of a character, not a
        Character object. Likewise ``node`` argument is the node's
        ID.

        """
        return self.db.node_is_thing(character, node, *self.time)
