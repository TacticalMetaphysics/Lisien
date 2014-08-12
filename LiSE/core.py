# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
"""Object relational mapper that serves Characters."""
from types import FunctionType, ModuleType
from collections import Mapping, MutableMapping, Callable
from sqlite3 import connect, OperationalError, IntegrityError, DatabaseError
from marshal import loads as unmarshalled
from marshal import dumps as marshalled
from gorm import ORM as gORM
from .character import Character
from .rule import Rule


class CharacterMapping(Mapping):
    def __init__(self, engine):
        self.engine = engine

    def __iter__(self):
        self.engine.cursor.execute(
            "SELECT graph FROM graphs;"
        )
        for row in self.engine.cursor.fetchall():
            yield row[0]

    def __len__(self):
        return self.engine.cursor.execute(
            "SELECT COUNT(*) FROM graphs;"
        ).fetchone()[0]

    def __contains__(self, name):
        return bool(self.engine.cursor.execute(
            "SELECT COUNT(*) FROM graphs WHERE graph=?;",
            (name,)
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
        """SELECT COUNT(*) FROM function WHERE name=?"""
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
        if name in self.cache:
            return self.cache[name]
        bytecode = self.cursor.execute(
            "SELECT code FROM function WHERE name=?;",
            (name,)
        ).fetchone()[0]
        return FunctionType(unmarshalled(bytecode), globals())

    def __call__(self, fun, name=None):
        """Remember the function in the code database. Return the name to use
        for it.

        This raises IntegrityError if you try to store a function on
        top of an existing one. If you really want to do that, assign
        it like I'm a dictionary.

        """
        if isinstance(fun, str):
            if fun not in self:
                raise KeyError("No such function")
            return fun
        if name is None:
            name = fun.__name__
        self.cursor.execute(
            "INSERT INTO function (name, code) VALUES (?, ?);",
            (name, marshalled(fun.__code__))
        )
        self.cache[name] = fun
        return name

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
            yield row[0]

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
            (k,)
        )
        try:
            return self.engine.cursor.fetchone()[0]
        except TypeError:
            raise KeyError("No value for {}".format(k))

    def __setitem__(self, k, v):
        """Set ``k`` to ``v``, possibly casting ``v`` to a string in the
        process.

        """
        del self[k]  # doesn't throw exception when k doesn't exist
        self.engine.cursor.execute(
            "INSERT INTO global (key, value) VALUES (?, ?);",
            (k, v)
        )

    def __delitem__(self, k):
        """Delete ``k``"""
        self.engine.cursor.execute(
            "DELETE FROM global WHERE key=?;",
            (k,)
        )


class Listeners(Mapping):
    """Mapping and decorator for the functions that listen to the time"""
    def __init__(self, engine, tabn):
        """Store the engine and the name of the table I'm about"""
        self.engine = engine
        self.tabn = tabn

    def __call__(self, v, name=None):
        """Store the function and activate it for the present (branch, tick)

        """
        if isinstance(v, Rule):
            self._activate_rule(v)
        elif isinstance(v, Callable):
            vname = self.engine.function(v, name)
            r = Rule(self.engine, vname)
            r.action(vname)
            self._activate_rule(r)
        else:
            vname = self.engine.function(v, name)
            self._activate_rule(Rule(self.engine, vname))

    def _activate_rule(self, rule):
        (branch, tick) = self.engine.time
        try:
            self.engine.cursor.execute(
                "INSERT INTO {tab} (rule, branch, tick, active) VALUES (?, ?, ?, 1);".format(
                    tab=self.tabn
                ),
                (rule.name, branch, tick)
            )
        except IntegrityError:
            self.engine.cursor.execute(
                "UPDATE {tab} SET active=1 WHERE "
                "rule=? AND "
                "branch=? AND "
                "tick=?;".format(tab=self.tabn),
                (rule.name, branch, tick)
            )

    def __iter__(self):
        """Iterate over the names of active rules"""
        r = set()
        seen = set()
        for (branch, tick) in self.engine._active_branches():
            self.engine.cursor.execute(
                "SELECT {tab}.rule, {tab}.active FROM {tab} JOIN ("
                "SELECT rule, branch, MAX(tick) AS tick FROM {tab} "
                "WHERE branch=? "
                "AND tick<=? "
                "GROUP BY rule, branch) AS hitick "
                "ON {tab}.rule=hitick.rule "
                "AND {tab}.branch=hitick.branch "
                "AND {tab}.tick=hitick.tick;".format(
                    tab=self.tabn
                ),
                (
                    branch,
                    tick
                )
            )
            for (rule, active) in self.engine.cursor.fetchall():
                if active and rule not in seen:
                    r.add(rule)
        for a in sorted(r):
            yield a

    def __len__(self):
        """Number of rules active in this table presently"""
        n = 0
        seen = set()
        for (branch, tick) in self.engine._active_branches():
            self.engine.cursor.execute(
                "SELECT {tab}.rule, {tab}.active FROM {tab} JOIN ("
                "SELECT rule, branch, MAX(tick) AS tick "
                "FROM {tab} WHERE "
                "branch=? AND "
                "tick<=? GROUP BY rule, branch) AS hitick "
                "ON {tab}.rule=hitick.rule "
                "AND {tab}.branch=hitick.branch "
                "AND {tab}.tick=hitick.tick;".format(
                    tab=self.tabn
                ),
                (
                    branch,
                    tick
                )
            )
            for (rule, active) in self.engine.cursor.fetchall():
                if active and rule not in seen:
                    n += 1
                seen.add(rule)
        return n

    def __getitem__(self, rulen):
        """Return the rule by the given name if it's active at the
        moment

        """
        for (branch, tick) in self.engine._active_branches():
            self.engine.cursor.execute(
                "SELECT {tab}.active FROM {tab} JOIN ("
                "SELECT rule, branch, MAX(tick) AS tick "
                "FROM {tab} WHERE "
                "rule=? AND "
                "branch=? AND "
                "tick<=? GROUP BY action, branch) AS hitick "
                "ON {tab}.rulehitick.rule "
                "AND {tab}.branch=hitick.branch "
                "AND {tab}.tick=hitick.tick;".format(
                    tab=self.tabn
                ),
                (
                    rulen,
                    branch,
                    tick
                )
            )
            data = self.engine.cursor.fetchall()
            if len(data) == 0:
                continue
            elif len(data) > 1:
                raise ValueError("Silly data in {tab} table".format(self.tabn))
            else:
                active = data[0][0]
                if active:
                    return Rule(self.engine, rulen)
                else:
                    raise KeyError("Listener {} disabled".format(rulen))
        raise KeyError("Listener {} doesn't exist".format(rulen))

    def __delitem__(self, rulen):
        """Deactivate the rule by the given name"""
        (branch, tick) = self.engine.time
        try:
            self.engine.cursor.execute(
                "INSERT INTO {tab} (rule, branch, tick, active) "
                "VALUES (?, ?, ?, ?);".format(tab=self.tabn),
                (
                    rulen,
                    branch,
                    tick,
                    False
                )
            )
        except IntegrityError:
            self.engine.cursor.execute(
                "UPDATE {tab} SET active=? WHERE "
                "rule=? AND "
                "branch=? AND "
                "tick=?;".format(tab=self.tabn),
                (
                    False,
                    rulen,
                    branch,
                    tick
                )
            )


class GeneralRuleMapping(Mapping):
    """Mapping for rules that aren't associated with any particular Character"""
    def __init__(self, engine):
        """Remember the engine"""
        self.engine = engine

    def __iter__(self):
        """Iterate over active rules"""
        seen = set()
        for (branch, tick) in self.engine._active_branches():
            for (rule, active) in self.engine.cursor.execute(
                    "SELECT char_rules.rule, char_rules.active FROM char_rules JOIN "
                    "(SELECT character, rule, branch, MAX(tick) AS tick "
                    "FROM char_rules WHERE "
                    "character IS NULL AND "
                    "branch=? AND "
                    "tick<=? GROUP BY character, rule, branch) AS hitick "
                    "ON char_rules.character=hitick.character "
                    "AND char_rules.rule=hitick.rule "
                    "AND char_rules.branch=hitick.branch "
                    "AND char_rules.tick=hitick.tick;",
                    (branch, tick)
            ).fetchall():
                if active and rule not in seen:
                    yield rule
                seen.add(rule)

    def __len__(self):
        """Count active rules"""
        n = 0
        for rule in iter(self):
            n += 1
        return n

    def __getitem__(self, k):
        """Return rule named thus, if it is active"""
        for (branch, tick) in self.engine._active_branches():
            data = self.engine.cursor.execute(
                "SELECT char_rules.rule, char_rules.active FROM char_rules JOIN "
                "(SELECT character, rule, branch, MAX(tick) AS tick "
                "FROM char_rules WHERE "
                "character IS NULL AND "
                "rule=? AND "
                "branch=? AND "
                "tick<=? GROUP BY character, rule, branch) AS hitick "
                "ON char_rules.character=hitick.character "
                "AND char_rules.rule=hitick.rule "
                "AND char_rules.branch=hitick.branch "
                "AND char_rules.tick=hitick.tick;",
                (k, branch, tick)
            ).fetchone()
            if data is None:
                continue
            (rule, active) = data
            if not active:
                raise KeyError("Rule deactivated")
            return Rule(self.engine, k)

    def __call__(self, v):
        """If passed a Rule, activate it. If passed a string, get the rule by
        that name and activate it. If passed a function (probably
        because I've been used as a decorator), make a rule with the
        same name as the function, with the function itself being the
        first action of the rule, and activate that rule.

        """
        if isinstance(v, Rule):
            self._activate_rule(v)
        elif isinstance(v, Callable):
            vname = self.engine.function(v)
            r = Rule(self.engine, vname)
            r.action(vname)
            self._activate_rule(r)
        else:
            self._activate_rule(Rule(self.engine, v))

    def _activate_rule(self, v):
        """Activate the rule"""
        (branch, tick) = self.engine.time
        try:
            self.engine.cursor.execute(
                "INSERT INTO char_rules "
                "(rule, branch, tick, active) "
                "VALUES (?, ?, ?, ?);",
                (
                    v.name,
                    branch,
                    tick,
                    True
                )
            )
        except IntegrityError:
            self.engine.cursor.executee(
                "UPDATE char_rules SET active=1 WHERE "
                "character IS NULL AND "
                "rule=? AND "
                "branch=? AND "
                "tick=?;",
                (
                    v.name,
                    branch,
                    tick
                )
            )

    def __delitem__(self, rulen):
        """Deactivate the rule"""
        (branch, tick) = self.engine.time
        try:
            self.engine.cursor.execute(
                "INSERT INTO char_rules "
                "(rule, branch, tick, active) "
                "VALUES (?, ?, ?, ?);",
                (
                    rulen,
                    branch,
                    tick,
                    False
                )
            )
        except IntegrityError:
            self.engine.cursor.execute(
                "UPDATE char_rules SET active=? "
                "WHERE character IS NULL AND "
                "rule=? AND "
                "branch=? AND "
                "tick=?;",
                (
                    False,
                    rulen,
                    branch,
                    tick
                )
            )


class Engine(object):
    def __init__(self, worlddb, codedb, gettext=lambda s: s):
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
        self.cursor.execute("BEGIN;")
        try:
            self.cursor.execute(
                "SELECT * FROM things;"
            ).fetchall()
        except OperationalError:
            self.initdb()
        # It wouldn't really make sense to store any sim rules in
        # these things because they also get triggered when you
        # investigate the past or schedule things for the future. So
        # what are they for?
        self.on_branch = Listeners(self, 'branch_listeners')
        self.on_tick = Listeners(self, 'tick_listeners')
        self.on_time = Listeners(self, 'time_listeners')
        self.eternal = EternalVarMapping(self)
        self.character = CharacterMapping(self)
        self.rule = GeneralRuleMapping(self)
        self._rules_iter = self._follow_rules()

    def commit(self):
        """Commit to both the world and code databases, and begin a new
        transaction for the world database

        """
        self.worlddb.commit()
        self.function.commit()
        self.cursor.execute("BEGIN;")

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
        for listener in list(self.on_branch.values()):
            listener(self, v)
        if not hasattr(self, 'locktime'):
            t = self.tick
            for time_listener in list(self.on_time.values()):
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
        for tick_listener in self.on_tick.values():
            tick_listener(self, v)
        if not hasattr(self, 'locktime'):
            b = self.branch
            for time_listener in self.on_tick.values():
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
        for time_listener in self.on_time.values():
            time_listener(self, b, t)
        del self.locktime

    def _follow_rules(self):
        """Execute the rules for the present branch and tick, and record the
        fact of it in the SQL database.

        Yield the rules paired with their results.

        """
        for (character, rule) in self._poll_rules():
            t = self.time
            s = rule(self, character)
            self.time = t
            self._char_rule_handled(character.name, rule.name)
            yield (rule, character, s)

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

    def new_character(self, name):
        """Create and return a character"""
        self.add_character(name)
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
        listener = (
            "CREATE TABLE {} ("
            "rule TEXT NOT NULL, "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick INTEGER NOT NULL DEFAULT 0, "
            "active BOOLEAN NOT NULL DEFAULT 1, "
            "PRIMARY KEY(rule, branch, tick), "
            "FOREIGN KEY(rule) REFERENCES rules(rule))"
            ";"
        )
        listener_idx = (
            "CREATE INDEX {tbl}_idx ON {tbl}(rule)"
            ";"
        )

        self.gorm.initdb()
        statements = [
            "CREATE TABLE rules ("
            "rule TEXT NOT NULL PRIMARY KEY, "
            "actions TEXT NOT NULL DEFAULT '[]', "
            "prereqs TEXT NOT NULL DEFAULT '[]')"
            ";",
            "CREATE TABLE char_rules ("
            "character TEXT, "
            # null here means no particular character
            "rule TEXT NOT NULL, "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick TEXT NOT NULL DEFAULT 0, "
            "active BOOLEAN NOT NULL DEFAULT 1, "
            "PRIMARY KEY(character, rule, branch, tick), "
            "FOREIGN KEY(rule) REFERENCES rules(rule), "
            "FOREIGN KEY(character) REFERENCES graphs(graph))"
            ";",
            "CREATE INDEX char_rules_idx ON char_rules(character, rule)"
            ";",
            "CREATE TABLE rules_handled ("
            "character TEXT, "
            "rule TEXT NOT NULL, "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick INTEGER NOT NULL DEFAULT 0,"
            "PRIMARY KEY(character, rule, branch, tick), "
            "FOREIGN KEY(character) REFERENCES graphs(graph), "
            "FOREIGN KEY(rule) REFERENCES rules(rule))"
            ";",
            "CREATE INDEX rules_handled_idx ON rules_handled(character, rule)"
            ";",
            "CREATE TABLE senses ("
            "character TEXT, "  
            # null means every character has this sense
            "sense TEXT NOT NULL, "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick INTEGER NOT NULL DEFAULT 0, "
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
            listener.format("branch_listeners"),
            listener_idx.format(tbl="branch_listeners"),
            listener.format("tick_listeners"),
            listener_idx.format(tbl="tick_listeners"),
            listener.format("time_listeners"),
            listener_idx.format(tbl="time_listeners")
        ]
        for stmt in statements:
            self.cursor.execute(stmt)

    def _active_branches(self):
        """Alias for gorm's ``_active_branches``"""
        return self.gorm._active_branches()

    def _iternodes(self, graph):
        """Alias for gorm's ``_iternodes``"""
        return self.gorm._iternodes(graph)

    def add_character(self, name, data=None):
        """Create the Character so it'll show up in my `character` dict"""
        self.gorm.new_digraph(name, data)

    def del_character(self, name):
        """Remove the Character from the database entirely"""
        for stmt in (
                "DELETE FROM things WHERE graph=?;",
                "DELETE FROM avatars WHERE character_graph=?;",
                "DELETE FROM char_rules WHERE character=?;"
        ):
            self.cursor.execute(stmt, (name,))
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
                    character,
                    node,
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

    def new_rule(self, name, actions=[], prereqs=[]):
        """Create a new Rule by this name, with these actions and prereqs.

        It will be saved in the SQL database, and the actions and
        prereqs will be saved in the dbm database.

        """
        return Rule(self, name, actions, prereqs)

    def del_rule(self, name):
        self.cursor.execute(
            "DELETE FROM rules WHERE rule=?;",
            (name,)
        )

    def _have_rules(self):
        """Make sure there are rules to follow."""
        for (branch, tick) in self._active_branches():
            self.cursor.execute(
                "SELECT COUNT(*) FROM char_rules JOIN ("
                "SELECT character, rule, branch, MAX(tick) AS tick "
                "FROM char_rules WHERE "
                "branch=? AND "
                "tick<=? GROUP BY character, rule, branch) AS hitick "
                "ON char_rules.character=hitick.character "
                "AND char_rules.rule=hitick.rule "
                "AND char_rules.branch=hitick.branch "
                "AND char_rules.tick=hitick.tick;",
                (
                    branch,
                    tick
                )
            )
            if self.cursor.fetchone()[0] > 0:
                return True
        return False

    def _poll_rules(self):
        """Iterate over Rules that have been activated on Characters and
        haven't already been followed in the present branch and
        tick. Yield each in a pair with its intended arguments.

        The Rules won't come in any particular order. I won't mark
        them as "followed"--that's for the method that actually calls
        them.

        """
        (branch, tick) = self.time
        self.cursor.execute(
            "SELECT character, rule FROM rules_handled "
            "WHERE branch=? "
            "AND tick=?;",
            (branch, tick)
        )
        handled = set(self.cursor.fetchall())
        for (branch, tick) in self._active_branches():
            self.cursor.execute(
                "SELECT "
                "char_rules.character, "
                "char_rules.rule, "
                "char_rules.branch, "
                "char_rules.tick, "
                "char_rules.active "
                "FROM char_rules JOIN ("
                "SELECT character, rule, branch, MAX(tick) AS tick "
                "FROM char_rules WHERE "
                "branch=? AND "
                "tick<=? GROUP BY character, rule, branch) AS hitick "
                "ON char_rules.character=hitick.character "
                "AND char_rules.rule=hitick.rule "
                "AND char_rules.branch=hitick.branch "
                "AND char_rules.tick=hitick.tick;",
                (
                    branch,
                    tick
                )
            )
            for (char, rule, b, t, act) in self.cursor.fetchall():
                if (char, rule) in handled:
                    continue
                if act:
                    yield (self.character[char], Rule(self, rule))
                handled.add((char, rule))

    def _char_rule_handled(self, character, rule):
        """Declare that a rule has been handled for a character at this
        time

        """
        (branch, tick) = self.time
        self.cursor.execute(
            "INSERT INTO rules_handled "
            "(character, rule, branch, tick) "
            "VALUES (?, ?, ?, ?);",
            (
                character,
                rule,
                branch,
                tick
            )
        )
