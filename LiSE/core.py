# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
"""Object relational mapper that serves Characters."""
from types import FunctionType
from collections import Mapping, MutableMapping, Callable
from sqlite3 import connect, OperationalError, IntegrityError
from marshal import loads as unmarshalled
from marshal import dumps as marshalled
from gorm import ORM as gORM
from .character import (
    Character,
    CharRules,
    CharacterSenseMapping
)
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


class FunctionStore(Mapping):
    def __init__(self, codedb):
        self.connection = codedb
        self.cursor = self.connection.cursor()
        try:
            self.cursor.execute("SELECT COUNT(*) FROM function;")
        except OperationalError:
            self.cursor.execute(
                "CREATE TABLE function ("
                "name TEXT NOT NULL PRIMARY KEY, "
                "code TEXT NOT NULL);"
            )
        self.cursor.execute("BEGIN;")
        self.codecache = {}

    def __len__(self):
        self.cursor.execute(
            "SELECT COUNT(*) FROM function;"
        )
        return self.cursor.fetchone()[0]

    def __iter__(self):
        self.cursor.execute(
            "SELECT name FROM function ORDER BY name;"
        )
        for row in self.cursor.fetchall():
            yield row[0]

    def __contains__(self, name):
        self.cursor.execute(
            "SELECT COUNT(*) FROM function WHERE name=?;",
            (name,)
        )
        return bool(self.cursor.fetchone()[0])

    def __getitem__(self, name):
        """Reconstruct the named function from its code string stored in the
        code database, and return it.

        """
        if name not in self.codecache:
            self.cursor.execute(
                "SELECT code FROM function WHERE name=?;",
                (name,)
            )
            code = unmarshalled(self.cursor.fetchone()[0])
            self.codecache[name] = FunctionType(code, globals(), str(name))
        return self.codecache[name]

    def __call__(self, fun, name=None):
        """Remember the function in the code database. Return the name to use
        for it.

        """
        if isinstance(fun, str):
            if fun not in self:
                raise KeyError("No such function")
            return fun
        if name is None:
            name = fun.__name__
        try:
            self.cursor.execute(
                "INSERT INTO function (name, code) VALUES (?, ?);",
                (name, marshalled(fun.__code__))
            )
        except IntegrityError:  # already got a function by that name
            if name not in self:
                raise IntegrityError("Simultaneously have and don't have the function")
            return name
        self.codecache[name] = fun
        return name

    def close(self):
        self.connection.commit()
        self.cursor.close()

    def commit(self):
        self.connection.commit()
        self.cursor.execute("BEGIN;")


class GlobalVarMapping(MutableMapping):
    def __init__(self, engine):
        self.engine = engine

    def __iter__(self):
        self.engine.cursor.execute(
            "SELECT key FROM global;"
        )
        for row in self.engine.cursor.fetchall():
            yield row[0]

    def __len__(self):
        self.engine.cursor.execute(
            "SELECT COUNT(*) FROM global;"
        )
        return self.engine.cursor.fetchone()[0]

    def __getitem__(self, k):
        self.engine.cursor.execute(
            "SELECT value FROM global WHERE key=?;",
            (k,)
        )
        try:
            return self.engine.cursor.fetchone()[0]
        except TypeError:
            raise KeyError("No value for {}".format(k))

    def __setitem__(self, k, v):
        del self[k]  # doesn't throw exception when k doesn't exist
        self.engine.cursor.execute(
            "INSERT INTO global (key, value) VALUES (?, ?);",
            (k, v)
        )

    def __delitem__(self, k):
        self.engine.cursor.execute(
            "DELETE FROM global WHERE key=?;",
            (k,)
        )


class Listeners(Mapping):
    """Mapping and decorator for the functions that listen to the time"""
    def __init__(self, engine, tabn):
        """Store the ORM and the name of the table I'm about"""
        self.engine = engine
        self.tabn = tabn

    def __call__(self, v):
        """Store the function and activate it for the present (branch, tick)

        """
        if isinstance(v, Rule):
            self._activate_rule(v)
        elif isinstance(v, Callable):
            vname = self.engine.function(v)
            self._activate_rule(
                Rule(
                    self.engine,
                    vname,
                    actions=[vname]
                )
            )
        else:
            self._activate_rule(Rule(self.engine, vname))

    def _activate_rule(self, rule):
        (branch, tick) = self.engine.time
        self.engine.cursor.execute(
            "DELETE FROM {tab} WHERE "
            "branch=? AND "
            "tick=? AND "
            "rule=?;".format(
                tab=self.tabn
            ),
            (
                branch,
                tick,
                rule.name
            )
        )
        self.engine.cursor.execute(
            "INSERT INTO {tab} (rule, branch, tick, active) VALUES (?, ?, ?, 1);".format(
                tab=self.tabn
            ),
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
        self.engine.cursor.execute(
            "DELETE FROM {tab} WHERE "
            "rule=? AND "
            "branch=? AND "
            "tick=?;".format(tab=self.tabn),
            (
                rulen,
                branch,
                tick
            )
        )
        self.engine.cursor.execute(
            "INSERT INTO {tab} (rule, branch, tick, active) "
            "VALUES (?, ?, ?, ?);",
            (
                rulen,
                branch,
                tick,
                False
            )
        )


class Engine(object):
    def __init__(self, world_filename, code_filename, gettext=lambda s: s):
        """Store the connections for the world database and the code database;
        set up listeners; and start a transaction

        """
        self.gettext = gettext
        self.worlddb = connect(world_filename)
        self.gorm = gORM(self.worlddb)
        self.function = FunctionStore(connect(code_filename))
        self.cursor = self.worlddb.cursor()
        self.cursor.execute("BEGIN;")
        try:
            self.cursor.execute(
                "SELECT * FROM things;"
            ).fetchall()
        except OperationalError:
            self.initdb()
        self.on_branch = Listeners(self, 'branch_listeners')
        self.on_tick = Listeners(self, 'tick_listeners')
        self.on_time = Listeners(self, 'time_listeners')
        self.eternal = GlobalVarMapping(self)
        self.character = CharacterMapping(self)
        self._rules_iter = self._follow_rules()
        for listname in (
                '_on_del_character',
                '_on_set_character_stat',
                '_on_del_character_stat',
                '_on_add_thing',
                '_on_del_thing',
                '_on_set_thing_stat',
                '_on_del_thing_stat',
                '_on_add_place',
                '_on_del_place',
                '_on_set_place_stat',
                '_on_del_place_stat',
                '_on_add_portal',
                '_on_del_portal',
                '_on_set_portal_stat',
                '_on_del_portal_stat'
        ):
            setattr(self, listname, [])

    def commit(self):
        self.worlddb.commit()
        self.function.commit()
        self.cursor.execute("BEGIN;")

    @property
    def branch(self):
        """Alias for my gorm's branch"""
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
        self.locktime = True
        self.gorm.time = v
        (b, t) = v
        for time_listener in self.on_time.values():
            time_listener(self, b, t)
        del self.locktime

    def _follow_rules(self):
        """Execute the rules for the present branch and tick, and record the
        fact of it in the SQL database.

        Yield the rules paired with their results.

        """
        (branch, tick) = self.time
        for (character, rule) in self._poll_rules():
            s = rule(self, character)
            self._char_rule_handled(character.name, rule.name)
            yield (rule, s)

    def advance(self):
        """Follow the next rule, advancing to the next tick if necessary.

        """
        try:
            return next(self._rules_iter)
        except StopIteration:
            if not self._have_rules():
                raise ValueError(
                    "No rules available; can't advance."
                )
            self.tick += 1
            self._rules_iter = self._follow_rules()
            return self.advance()

    def next_tick(self):
        curtick = self.tick
        r = []
        while self.tick == curtick:
            r.append(self.advance())
        return r

    def new_character(self, name):
        self.add_character(name)
        return self.character[name]

    def close(self):
        self.worlddb.commit()
        self.function.commit()
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

        self.gorm.initdb()
        statements = [
            "CREATE TABLE rules ("
            "rule TEXT NOT NULL PRIMARY KEY, "
            "actions TEXT NOT NULL DEFAULT '[]', "
            "prereqs TEXT NOT NULL DEFAULT '[]')"
            ";",
            "CREATE TABLE char_rules ("
            "character TEXT NOT NULL, "
            "rule TEXT NOT NULL, "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick TEXT NOT NULL DEFAULT 0, "
            "active BOOLEAN NOT NULL DEFAULT 1, "
            "PRIMARY KEY(character, rule, branch, tick), "
            "FOREIGN KEY(rule) REFERENCES rules(rule), "
            "FOREIGN KEY(character) REFERENCES graphs(graph))"
            ";",
            "CREATE TABLE rules_handled ("
            "character TEXT NOT NULL, "
            "rule TEXT NOT NULL, "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick INTEGER NOT NULL DEFAULT 0,"
            "PRIMARY KEY(character, rule, branch, tick), "
            "FOREIGN KEY(character) REFERENCES graphs(graph), "
            "FOREIGN KEY(rule) REFERENCES rules(rule))"
            ";",
            "CREATE TABLE senses ("
            "character TEXT NOT NULL DEFAULT '', "  
            # empty string means every character has this sense
            "sense TEXT NOT NULL, "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick INTEGER NOT NULL DEFAULT 0, "
            "active BOOLEAN NOT NULL DEFAULT 1, "
            "PRIMARY KEY(character, sense, branch, tick),"
            "FOREIGN KEY(character) REFERENCES graphs(graph))"
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
            listener.format("branch_listeners"),
            listener.format("tick_listeners"),
            listener.format("time_listeners"),
        ]
        for stmt in statements:
            self.cursor.execute(stmt)

    def _active_branches(self):
        """Alias for gorm's ``_active_branches``"""
        return self.gorm._active_branches()

    def _iternodes(self, graph):
        """Alias for gorm's ``_iternodes``"""
        return self.gorm._iternodes(graph)

    def add_character(self, name):
        """Create the Character so it'll show up in my `character` dict"""
        self.gorm.new_digraph(name)

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
