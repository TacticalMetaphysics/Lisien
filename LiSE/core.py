# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
"""Object relational mapper that serves Characters."""
from random import Random
from types import FunctionType, ModuleType
from collections import Mapping, MutableMapping, Callable
from sqlite3 import connect, OperationalError, IntegrityError
from marshal import loads as unmarshalled
from marshal import dumps as marshalled
from gorm import ORM as gORM
from gorm.graph import (
    json_dump,
    json_load
)
from .character import Character
from .rule import AllRules


class CharacterMapping(Mapping):
    def __init__(self, engine):
        self.engine = engine

    def __iter__(self):
        self.engine.cursor.execute(
            "SELECT graph FROM graphs;"
        )
        for row in self.engine.cursor.fetchall():
            yield json_load(row[0])

    def __len__(self):
        return self.engine.cursor.execute(
            "SELECT COUNT(*) FROM graphs;"
        ).fetchone()[0]

    def __contains__(self, name):
        return bool(self.engine.cursor.execute(
            "SELECT COUNT(*) FROM graphs WHERE graph=?;",
            (json_dump(name),)
        ).fetchone()[0])

    def __getitem__(self, name):
        if name not in self:
            raise KeyError("No character named {}, maybe you want to add_character?".format(name))
        return Character(self.engine, name)


class FunctionStore(object):
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


class FunctionStoreDB(FunctionStore, MutableMapping):
    """Store functions in a SQL database"""
    def __init__(self, codedb):
        """Use ``codedb`` as a connection object. Connect to it, and
        initialize the schema if needed.

        """
        self.connection = codedb
        self.cursor = self.connection.cursor()
        self.cache = {}
        try:
            self.cursor.execute("SELECT COUNT(*) FROM function;")
        except OperationalError:
            self.cursor.execute(
                "CREATE TABLE function ("
                "name TEXT NOT NULL PRIMARY KEY, "
                "code TEXT NOT NULL);"
            )

    def __len__(self):
        """SELECT COUNT(*) FROM function"""
        self.cursor.execute(
            "SELECT COUNT(*) FROM function;"
        )
        return self.cursor.fetchone()[0]

    def __iter__(self):
        """SELECT name FROM function ORDER BY name"""
        self.cursor.execute(
            "SELECT name FROM function ORDER BY name;"
        )
        for row in self.cursor.fetchall():
            yield row[0]

    def __contains__(self, name):
        """Check if there's such a function in the database"""
        if name in self.cache:
            return True
        self.cursor.execute(
            "SELECT COUNT(*) FROM function WHERE name=?;",
            (name,)
        )
        return bool(self.cursor.fetchone()[0])

    def __getitem__(self, name):
        """Reconstruct the named function from its code string stored in the
        code database, and return it.

        """
        if name not in self.cache:
            bytecode = self.cursor.execute(
                "SELECT code FROM function WHERE name=?;",
                (name,)
            ).fetchone()
            if bytecode is None:
                raise KeyError("No such function")
            self.cache[name] = FunctionType(unmarshalled(bytecode[0]), globals())
        return self.cache[name]

    def __call__(self, fun):
        """Remember the function in the code database. It will be keyed by its
        ``__name__``.

        """
        try:
            self.cursor.execute(
                "INSERT INTO function (name, code) VALUES (?, ?);",
                (fun.__name__, marshalled(fun.__code__))
            )
        except IntegrityError:
            raise KeyError("Already have a function by that name")
        self.cache[fun.__name__] = fun

    def __setitem__(self, name, fun):
        """Store the function, marshalled, under the name given."""
        mcode = marshalled(fun.__code__)
        try:
            self.cursor.execute(
                "INSERT INTO function (name, code) VALUES (?, ?);",
                (name, mcode)
            )
        except IntegrityError:
            self.cursor.execute(
                "UPDATE function SET code=? WHERE name=?;",
                (mcode, name)
            )
        self.cache[name] = fun

    def __delitem__(self, name):
        """DELETE FROM function WHERE name=?"""
        self.cursor.execute(
            "DELETE FROM function WHERE name=?;",
            (name,)
        )
        del self.cache[name]

    def close(self):
        """Commit the transaction and close the cursor"""
        self.connection.commit()
        self.cursor.close()

    def commit(self):
        """Alias for ``self.connection.commit()``"""
        self.connection.commit()


class FunctionStoreModule(FunctionStore, Mapping):
    """Dict-like wrapper for a module object"""
    def __init__(self, module):
        """Store the module"""
        self._mod = module

    def __iter__(self):
        """Iterate over the module's __all__ attribute"""
        for it in self._mod.__all__:
            yield repr(it)

    def __len__(self):
        """Return the length of the module's __all__ attribute"""
        return len(self._mod.__all__)

    def __getitem__(self, k):
        """Return the ``k`` attribute of the module"""
        return getattr(k, self._mod)

    def __call__(self, k):
        """If ``k`` is in the module's __all__ attribute, return a
        representation of it. If ``k``'s representation is a
        representation of something in the module's __all__ attribute,
        do the same.

        """
        if k in self._mod.__all__:
            return repr(k)
        elif repr(k) in [repr(it) for it in self._mod.__all__]:
            return repr(k)
        else:
            raise KeyError("{} is not a member of {}".format(k, self._mod))


class EternalVarMapping(MutableMapping):
    """Mapping for variables that aren't under revision control"""
    def __init__(self, engine):
        """Store the engine"""
        self.engine = engine

    def __iter__(self):
        """Iterate over the global keys"""
        self.engine.cursor.execute(
            "SELECT key FROM global;"
        )
        for row in self.engine.cursor.fetchall():
            yield json_load(row[0])

    def __len__(self):
        """Count the global keys"""
        self.engine.cursor.execute(
            "SELECT COUNT(*) FROM global;"
        )
        return self.engine.cursor.fetchone()[0]

    def __getitem__(self, k):
        """Get the value for variable ``k``. It will always be a string."""
        self.engine.cursor.execute(
            "SELECT value FROM global WHERE key=?;",
            (json_dump(k),)
        )
        try:
            return json_load(self.engine.cursor.fetchone()[0])
        except TypeError:
            raise KeyError("No value for {}".format(k))

    def __setitem__(self, k, v):
        """Set ``k`` to ``v``, possibly casting ``v`` to a string in the
        process.

        """
        (ks, vs) = (json_dump(k), json_dump(v))
        try:
            self.engine.cursor.execute(
                "INSERT INTO global (key, value) VALUES (?, ?);",
                (ks, vs)
            )
        except IntegrityError:
            self.engine.cursor.execute(
                "UPDATE global SET value=? WHERE key=?;",
                (vs, ks)
            )

    def __delitem__(self, k):
        """Delete ``k``"""
        self.engine.cursor.execute(
            "DELETE FROM global WHERE key=?;",
            (json_dump(k),)
        )


class GlobalVarMapping(MutableMapping):
    """Mapping for variables that are global but which I keep history for"""
    def __init__(self, engine):
        """Store the engine"""
        self.engine = engine

    def __iter__(self):
        """Iterate over the global keys whose values aren't null at the moment.

        The values may be None, however.

        """
        seen = set()
        for (branch, tick) in self.engine._active_branches():
            data = self.engine.cursor.execute(
                "SELECT lise_globals.key, lise_globals.value FROM lise_globals JOIN "
                "(SELECT key, branch, MAX(tick) AS tick "
                "FROM lise_globals "
                "WHERE branch=? "
                "AND tick<=? "
                "GROUP BY key, branch) AS hitick "
                "ON lise_globals.key=hitick.key "
                "AND lise_globals.branch=hitick.branch "
                "AND lise_globals.tick=hitick.tick;",
                (
                    branch,
                    tick
                )
            ).fetchall()
            for (k, v) in data:
                key = json_load(k)
                if v is None:
                    seen.add(key)
                    continue
                if key not in seen:
                    yield key
                seen.add(key)

    def __len__(self):
        """Just count while iterating"""
        n = 0
        for k in iter(self):
            n += 1
        return n

    def __getitem__(self, k):
        """Get the current value of this key"""
        key = json_dump(k)
        (branch, tick) = self.engine.time
        data = self.engine.cursor.execute(
            "SELECT lise_globals.value FROM lise_globals JOIN "
            "(SELECT key, branch, MAX(tick) AS tick "
            "FROM lise_globals "
            "WHERE key=? "
            "AND branch=? "
            "AND tick<=? "
            "GROUP BY key, branch) AS hitick "
            "ON lise_globals.key=hitick.key "
            "AND lise_globals.branch=hitick.branch "
            "AND lise_globals.tick=hitick.tick;",
            (
                key,
                branch,
                tick
            )
        ).fetchall()
        if len(data) == 0:
            raise KeyError("Key not set")
        elif len(data) > 1:
            raise ValueError("Silly data in lise_globals table")
        else:
            v = data[0][0]
            if v is None:  # not decoded yet
                raise KeyError("Key not set right now")
            return json_load(v)

    def __setitem__(self, k, v):
        """Set k=v at the current branch and tick"""
        key = json_dump(k)
        value = json_dump(v)
        (branch, tick) = self.engine.time
        try:
            self.engine.cursor.execute(
                "INSERT INTO lise_globals (key, branch, tick, value) "
                "VALUES (?, ?, ?, ?);",
                (
                    key,
                    branch,
                    tick,
                    value
                )
            )
        except IntegrityError:
            self.engine.cursor.execute(
                "UPDATE lise_globals SET value=? WHERE "
                "key=? AND "
                "branch=? AND "
                "tick=?;",
                (
                    value,
                    key,
                    branch,
                    tick
                )
            )

    def __delitem__(self, k):
        """Unset this key for the present (branch, tick)"""
        key = json_dump(k)
        (branch, tick) = self.engine.time
        try:
            self.engine.cursor.execute(
                "INSERT INTO lise_globals "
                "(key, branch, tick, value) "
                "VALUES (?, ?, ?, ?);",
                (
                    key,
                    branch,
                    tick,
                    None
                )
            )
        except IntegrityError:
            self.engine.cursor.execute(
                "UPDATE lise_globals SET value=? WHERE "
                "key=? AND "
                "branch=? AND "
                "tick=?;",
                (
                    None,
                    key,
                    branch,
                    tick
                )
            )


class Engine(object):
    def __init__(self, worlddb, codedb, commit_modulus=None, random_seed=None, gettext=lambda s: s):
        """Store the connections for the world database and the code database;
        set up listeners; and start a transaction

        """
        self.gettext = gettext
        self.worlddb = connect(worlddb)
        self.gorm = gORM(self.worlddb)
        if isinstance(codedb, ModuleType):
            self.function = FunctionStoreModule(codedb)
        else:
            self.function = FunctionStoreDB(connect(codedb))
        self.cursor = self.worlddb.cursor()
        self.commit_modulus = commit_modulus
        self.cursor.execute("BEGIN;")
        try:
            self.cursor.execute(
                "SELECT * FROM things;"
            ).fetchall()
        except OperationalError:
            self.initdb()
        self.time_listeners = []
        self.rule = AllRules(self)
        self.eternal = EternalVarMapping(self)
        self.globl = GlobalVarMapping(self)
        self.character = CharacterMapping(self)
        self._rules_iter = self._follow_rules()
        # set up the randomizer
        self.rando = Random()
        if 'rando_state' in self.globl:
            self.rando.setstate(self.globl['rando_state'])
        else:
            self.rando.seed(random_seed)
            self.globl['rando_state'] = self.rando.getstate()
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

    def coinflip(self):
        return self.choice((True, False))

    def commit(self):
        """Commit to both the world and code databases, and begin a new
        transaction for the world database

        """
        self.worlddb.commit()
        self.function.commit()
        self.cursor.execute("BEGIN;")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.commit()

    def on_time(self, v):
        if not isinstance(v, Callable):
            raise TypeError("This is a decorator")
        self.time_listeners.append(v)

    @property
    def branch(self):
        """Alias for my gorm's ``branch``"""
        return self.gorm.branch

    @branch.setter
    def branch(self, v):
        """Set my gorm's branch and call listeners"""
        if v == self.gorm.branch:
            return
        self.gorm.branch = v
        if not hasattr(self, 'locktime'):
            t = self.tick
            for time_listener in self.time_listeners:
                time_listener(self, v, t)

    @property
    def tick(self):
        """Alias for my gorm's ``rev``"""
        return self.gorm.rev

    @tick.setter
    def tick(self, v):
        """Update orm's tick, and call listeners"""
        if v == self.gorm.rev:
            return
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
        (self.gorm.branch, self.gorm.rev) = v
        (b, t) = v
        for time_listener in self.time_listeners:
            time_listener(self, b, t)
        del self.locktime

    def _have_rules(self):
        for rulemap in ('character', 'avatar', 'thing', 'place', 'portal'):
            seen = set()
            for (branch, tick) in self._active_branches():
                data = self.cursor.execute(
                    "SELECT {table}_rules.rule, {table}_rules.active FROM {table}_rules JOIN "
                    "(SELECT rulebook, rule, branch, MAX(tick) AS tick "
                    "FROM {table}_rules WHERE "
                    "branch=? AND "
                    "tick<=? GROUP BY rulebook, rule, branch) AS hitick "
                    "ON {table}_rules.rulebook=hitick.rulebook "
                    "AND {table}_rules.rule=hitick.rule "
                    "AND {table}_rules.branch=hitick.branch "
                    "AND {table}_rules.tick=hitick.tick;".format(table=rulemap),
                    (branch, tick)
                ).fetchall()
                for (rule, active) in data:
                    if rule not in seen and active:
                        return True
                    seen.add(rule)
        return False

    def _poll_rules(self):
        for rulemap in ('character', 'avatar', 'thing', 'place', 'portal'):
            seen = set()
            for (branch, tick) in self._active_branches():
                # get all the rules except those already handled
                data = self.cursor.execute(
                    "SELECT "
                    "characters.character, "
                    "characters.{rulemap}_rulebook, "
                    "active_rules.rule, "
                    "active_rules.active, "
                    "handledness.handled "
                    "FROM characters JOIN active_rules ON "
                    "characters.{rulemap}_rulebook=active_rules.rulebook "
                    "JOIN "
                    "(SELECT rulebook, rule, branch, MAX(tick) AS tick "
                    "FROM active_rules WHERE "
                    "branch=? AND "
                    "tick<=? GROUP BY rulebook, rule, branch) AS hitick "
                    "ON active_rules.rulebook=hitick.rulebook "
                    "AND active_rules.rule=hitick.rule "
                    "AND active_rules.branch=hitick.branch "
                    "AND active_rules.tick=hitick.tick "
                    "LEFT OUTER JOIN rulebooks "
                    "ON rulebooks.rulebook=characters.{rulemap}_rulebook "
                    "AND rulebooks.rule=active_rules.rule "
                    "LEFT OUTER JOIN "
                    "(SELECT character, rulebook, rule, branch, tick, 1 as handled "
                    "FROM {rulemap}_rules_handled) AS handledness "
                    "ON handledness.character=characters.character "
                    "AND handledness.rulebook=characters.{rulemap}_rulebook "
                    "AND handledness.rule=active_rules.rule "
                    "AND handledness.branch=? "
                    "AND handledness.tick=?"
                    "ORDER BY rulebooks.idx ASC"
                    ";".format(rulemap=rulemap),
                    (
                        branch,
                        tick,
                        branch,
                        tick
                    )
                ).fetchall()
                for (c, rulebook, rule, active, handled) in data:
                    character = json_load(c)
                    if (character, rulebook, rule) in seen:
                        continue
                    seen.add((character, rulebook, rule))
                    if active and not handled:
                        yield (rulemap, character, rulebook, rule)

    def _follow_rules(self):
        (branch, tick) = self.time
        for (ruletyp, charname, rulebook, rulename) in self._poll_rules():
            character = self.character[charname]
            rule = self.rule[rulename]

            if ruletyp == 'character':
                yield rule(self, character)
            elif ruletyp == 'avatar':
                for avatar in character.iter_avatars():
                    yield rule(self, character, avatar)
            elif ruletyp == 'thing':
                for thing in character.thing.values():
                    yield rule(self, character, thing)
            elif ruletyp == 'place':
                for place in character.place.values():
                    yield rule(self, character, place)
            elif ruletyp == 'portal':
                for portal in character.iter_portals():
                    yield rule(self, character, portal)
            else:
                raise TypeError("Unknown type of rule")
            self.cursor.execute(
                "INSERT INTO {}_rules_handled "
                "(character, rulebook, rule, branch, tick) "
                "VALUES (?, ?, ?, ?, ?);".format(ruletyp),
                (
                    charname,
                    rulebook,
                    rulename,
                    branch,
                    tick
                )
            )

    def advance(self):
        """Follow the next rule, advancing to the next tick if necessary.

        """
        try:
            r = next(self._rules_iter)
        except StopIteration:
            if not self._have_rules():
                raise ValueError(
                    "No rules available; can't advance."
                )
            self.tick += 1
            self._rules_iter = self._follow_rules()
            self.globl['rando_state'] = self.rando.getstate()
            if self.commit_modulus and self.tick % self.commit_modulus == 0:
                self.worlddb.commit()
            r = None
        return r

    def next_tick(self):
        """Call ``advance`` repeatedly, appending its results to a list, until
        it returns ``None``, at which point the tick has ended and
        I'll return the list.

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

    def close(self):
        """Commit database transactions and close cursors"""
        self.worlddb.commit()
        self.cursor.close()
        self.function.close()

    def initdb(self):
        """Set up the database schema, both for gorm and the special
        extensions for LiSE

        """
        self.gorm.initdb()
        statements = [
            "CREATE TABLE lise_globals ("
            "key TEXT NOT NULL, "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick INTEGER NOT NULL DEFAULT 0, "
            "value TEXT, "
            "PRIMARY KEY(key, branch, tick))"
            ";",
            "CREATE INDEX globals_idx ON lise_globals(key)"
            ";",
            "CREATE TABLE rules ("
            "rule TEXT NOT NULL PRIMARY KEY, "
            "actions TEXT NOT NULL DEFAULT '[]', "
            "prereqs TEXT NOT NULL DEFAULT '[]')"
            ";",
            "CREATE TABLE rulebooks ("
            "rulebook TEXT NOT NULL, "
            "idx INTEGER NOT NULL, "
            "rule TEXT NOT NULL, "
            "PRIMARY KEY(rulebook, idx), "
            "FOREIGN KEY(rule) REFERENCES rules(rule))"
            ";",
            "CREATE TABLE active_rules ("
            "rulebook TEXT NOT NULL, "
            "rule TEXT NOT NULL, "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick INTEGER NOT NULL DEFAULT 0, "
            "active BOOLEAN NOT NULL DEFAULT 1, "
            "PRIMARY KEY(rulebook, rule, branch, tick), "
            "FOREIGN KEY(rulebook, rule) REFERENCES rulebooks(rulebook, rule))"
            ";",
            "CREATE INDEX active_rules_idx ON active_rules(rulebook, rule)"
            ";",
            "CREATE TABLE characters ("
            "character TEXT NOT NULL PRIMARY KEY, "
            "character_rulebook TEXT NOT NULL, "
            "avatar_rulebook TEXT NOT NULL, "
            "thing_rulebook TEXT NOT NULL, "
            "place_rulebook TEXT NOT NULL, "
            "portal_rulebook TEXT NOT NULL, "
            "FOREIGN KEY(character) REFERENCES graphs(graph))"
            ";",
            "CREATE TABLE senses ("
            "character TEXT, "  
            # null means every character has this sense
            "sense TEXT NOT NULL, "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick INTEGER NOT NULL DEFAULT 0, "
            "function TEXT NOT NULL, "
            "active BOOLEAN NOT NULL DEFAULT 1, "
            "PRIMARY KEY(character, sense, branch, tick),"
            "FOREIGN KEY(character) REFERENCES graphs(graph))"
            ";",
            "CREATE INDEX senses_idx ON senses(character, sense)"
            ";",
            "CREATE TABLE travel_reqs ("
            "character TEXT NOT NULL DEFAULT '', "
            # empty string means these are required of every character
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick INTEGER NOT NULL DEFAULT 0, "
            "reqs TEXT NOT NULL DEFAULT '[]', "
            "PRIMARY KEY(character, branch, tick), "
            "FOREIGN KEY(character) REFERENCES graphs(graph))"
            ";",
            "CREATE INDEX travel_reqs_idx ON travel_reqs(character)"
            ";",
            "CREATE TABLE things ("
            "character TEXT NOT NULL, "
            "thing TEXT NOT NULL, "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick INTEGER NOT NULL DEFAULT 0, "
            "location TEXT, "  # when null, I'm not a thing; treat me
                               # like any other node
            "next_location TEXT, "  # when set, indicates that I'm en
                                    # route between location and
                                    # next_location
            "PRIMARY KEY(character, thing, branch, tick), "
            "FOREIGN KEY(character, thing) REFERENCES nodes(graph, node), "
            "FOREIGN KEY(character, location) REFERENCES nodes(graph, node))"
            ";",
            "CREATE INDEX things_idx ON things(character, thing)"
            ";",
            "CREATE TABLE avatars ("
            "character_graph TEXT NOT NULL, "
            "avatar_graph TEXT NOT NULL, "
            "avatar_node TEXT NOT NULL, "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick INTEGER NOT NULL DEFAULT 0, "
            "is_avatar BOOLEAN NOT NULL, "
            "PRIMARY KEY(character_graph, avatar_graph, avatar_node, branch, tick), "
            "FOREIGN KEY(character_graph) REFERENCES graphs(graph), "
            "FOREIGN KEY(avatar_graph, avatar_node) REFERENCES nodes(graph, node))"
            ";",
            "CREATE INDEX avatars_idx ON avatars("
            "character_graph, "
            "avatar_graph, "
            "avatar_node)"
            ";",
        ]

        handled = (
            "CREATE TABLE {table}_rules_handled ("
            "character TEXT NOT NULL, "
            "rulebook TEXT NOT NULL, "
            "rule TEXT NOT NULL, "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick INTEGER NOT NULL DEFAULT 0, "
            "PRIMARY KEY(character, rulebook, rule, branch, tick), "
            "FOREIGN KEY(character, rulebook) "
            "REFERENCES characters(character, {table}_rulebook))"
            ";"
        )
        handled_idx = (
            "CREATE INDEX {table}_rules_handled_idx ON "
            "{table}_rules_handled(character, rulebook, rule)"
            ";"
        )
        rulesview = (
            "CREATE VIEW {table}_rules AS "
            "SELECT character, rulebook, rule, branch, tick, active "
            "FROM active_rules JOIN characters ON "
            "active_rules.rulebook=characters.{table}_rulebook"
            ";"
        )
        for tabn in ("character", "avatar", "thing", "place", "portal"):
            statements.extend((
                handled.format(table=tabn),
                handled_idx.format(table=tabn),
                rulesview.format(table=tabn)
            ))

        for stmt in statements:
            self.cursor.execute(stmt)

    def _active_branches(self):
        """Alias for gorm's ``_active_branches``"""
        return self.gorm._active_branches()

    def _iternodes(self, graph):
        """Alias for gorm's ``_iternodes``"""
        return self.gorm._iternodes(graph)

    def add_character(self, name, data=None, **kwargs):
        """Create the Character so it'll show up in my `character` dict"""
        self.gorm.new_digraph(name, data, **kwargs)

    def del_character(self, name):
        """Remove the Character from the database entirely"""
        for stmt in (
                "DELETE FROM things WHERE graph=?;",
                "DELETE FROM avatars WHERE character_graph=?;",
        ):
            self.cursor.execute(stmt, (json_dump(name),))
        self.gorm.del_graph(name)

    def _is_thing(self, character, node):
        """Private utility function to find out if a node is a Thing or not.

        ``character`` argument must be the name of a character, not a
        Character object. Likewise ``node`` argument is the node's
        ID.

        """
        for (branch, rev) in self._active_branches():
            self.cursor.execute(
                "SELECT location FROM things JOIN ("
                "SELECT character, thing, branch, MAX(tick) AS tick "
                "FROM things WHERE "
                "character=? "
                "AND thing=? "
                "AND branch=? "
                "AND tick<=? "
                "GROUP BY character, thing, branch) AS hitick "
                "ON things.character=hitick.character "
                "AND things.thing=hitick.thing "
                "AND things.branch=hitick.branch "
                "AND things.tick=hitick.tick;",
                (
                    json_dump(character),
                    json_dump(node),
                    branch,
                    rev
                )
            )
            data = self.cursor.fetchall()
            if len(data) == 0:
                continue
            elif len(data) > 1:
                raise ValueError("Silly data in things table")
            else:
                return bool(data[0][0])
        return False
