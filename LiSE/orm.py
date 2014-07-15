# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
"""Object relational mapper that serves Characters and Rules."""
from collections import Mapping
from sqlite3 import connect
from marshal import dumps as marshalled
from marshal import loads as unmarshalled
from types import FunctionType
from gorm import ORM as gORM
from rule import Rule
from character import (
    Character,
    CharRules
)


class Worldview(object):
    """The part of the ORM that accesses a SQLite database, which holds
    world state

    """
    def __init__(self, arg, obranch=None, otick=None):
        """If instantiated from another Worldview, copy its database to a
        transient, in-memory sqlite3 database of my own. Otherwise
        assume I've been provided with a Connection object.

        """
        if isinstance(arg, Worldview):
            worlddb = connect(":memory:")
            c = worlddb.cursor()
            c.executemany(arg.cursor.iterdump())
            c.close()
        else:
            worlddb = arg
        self.gorm = gORM(worlddb, obranch=obranch, orev=otick)
        self.cursor = self.gorm.cursor
        self.cursor.execute("BEGIN;")

    @property
    def branch(self):
        return self.gorm.branch

    @branch.setter
    def branch(self, v):
        if self.gorm._obranch is not None:
            raise ValueError("Branch has been overridden for this instance")
        self.gorm.branch = v

    @property
    def tick(self):
        return self.gorm.rev

    @tick.setter
    def tick(self, v):
        if self.gorm._orev is not None:
            raise ValueError("Tick has been overridden for this instance")
        self.gorm.rev = v

    @property
    def time(self):
        return (self.branch, self.tick)

    @time.setter
    def time(self, v):
        (self.branch, self.tick) = v

    def __del__(self):
        self.close()

    def close(self):
        self.cursor.connection.commit()
        self.cursor.close()

    def initdb(self):
        """Set up the database schema, both for gorm and the special
        extensions for LiSE

        """
        self.gorm.initdb()
        statements = [
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
            ";"
        ]
        for stmt in statements:
            self.cursor.execute(stmt)

    def _active_branches(self):
        """Alias for gorm's ``_active_branches``"""
        return self.gorm._active_branches()

    def new_character(self, name):
        """Create and return a Character by the given name"""
        self.gorm.new_digraph(name)
        return Character(self, name)

    def get_character(self, name):
        """Return a previously created Character by the given name"""
        try:
            self.gorm.get_graph(name)
        except ValueError:
            raise ValueError("No character named {}".format(name))
        return Character(self, name)

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


class Timeline(Mapping):
    """Access a worldview at an arbitrary tick within a branch"""
    def __init__(self, conn, branch):
        self.connection = conn
        self.branch = branch

    def __iter__(self):
        """Iterate from the start of the branch to the last tick in the branch
        that has any data

        """
        return xrange(self._parrev(), self._maxtick())

    def __len__(self):
        """Number of ticks from start of branch to the last that has data"""
        return self._maxtick() - self._parrev()

    def __getitem__(self, tick):
        return Worldview(self.connection, obranch=self.branch, otick=tick)

    def _parrev(self):
        """Return the parent rev for my branch"""
        c = self.connection.cursor()
        c.execute("SELECT parent_rev FROM branches WHERE branch=?;", (self.branch,))
        r = c.fetchone()[0]
        return r

    def _maxtick(self):
        """Return the last tick that has any data"""
        r = self._parrev()
        c = self.connection.cursor()
        for table in ("nodes", "node_val", "edges", "edge_val", "graph_val"):
            c.execute(
                "SELECT MAX(rev) FROM {tab} WHERE branch=?;".format(tab=table),
                (self.branch,)
            )
            s = c.fetchone()[0]
            r = max((r, s))
        c.close()
        return r


class Branchspace(Mapping):
    def __init__(self, conn):
        self.connection = conn

    def __iter__(self):
        c = self.connection.cursor()
        c.execute(
            "SELECT branch FROM branches;"
        )
        for (b,) in c:
            yield b
        c.close()

    def __len__(self):
        c = self.connection.cursor()
        c.execute(
            "SELECT COUNT(branch) FROM branches;"
        )
        r = c.fetchone()[0]
        c.close()
        return r

    def __getitem__(self, branch):
        return Timeline(self.connection, branch)


class FunctionStore(Mapping):
    """Manage a dbm database of marshalled functions, and a cache of the functions"""
    def __init__(self, codedb):
        self.dbm = codedb
        self.codecache = {}

    def __len__(self):
        return len(self.dbm)

    def __iter__(self):
        return iter(self.dbm)

    def __getitem__(self, name):
        """Reconstruct the named function from its code string stored in the
        code database, and return it.

        """
        if name not in self.codecache:
            code = unmarshalled(self.dbm[str(name)])
            self.codecache[name] = FunctionType(code, globals(), str(name))
        return self.codecache[name]

    def __call__(self, fun):
        """Remember the function in the code database. Return the name to use
        for it.

        """
        if isinstance(fun, str) or isinstance(fun, unicode):
            if fun not in self.dbm:
                raise KeyError("No such function")
            return fun
        self.dbm[fun.__name__] = marshalled(fun.func_code)
        self.codecache[fun.__name__] = fun
        return fun.__name__

    def close(self):
        self.dbm.close()


class ORM(object):
    def __init__(self, worlddb, codedb):
        self.worldview = Worldview(worlddb)
        self.history = Branchspace(worlddb)
        self.function = FunctionStore(codedb)
        self.cursor = self.worldview.cursor

    @property
    def branch(self):
        return self.worldview.branch

    @branch.setter
    def branch(self, v):
        self.worldview.branch = v

    @property
    def branchhist(self):
        return self.history[self.branch]

    @property
    def tick(self):
        return self.worldview.tick

    @tick.setter
    def tick(self, v):
        self.worldview.tick = v

    @property
    def time(self):
        return self.worldview.time

    @time.setter
    def time(self, v):
        self.worldview.time = v

    def initdb(self):
        self.worldview.initdb()
        for stmt in [
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
            "CREATE TABLE branch_listeners ("
            "action TEXT NOT NULL, "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick INTEGER NOT NULL DEFAULT 0, "
            "active BOOLEAN NOT NULL DEFAULT 1, "
            "PRIMARY KEY(action, branch, tick))"
            ";",
            "CREATE TABLE tick_listeners ("
            "action TEXT NOT NULL, "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick INTEGER NOT NULL DEFAULT 0, "
            "active BOOLEAN NOT NULL DEFAULT 1, "
            "PRIMARY KEY(action, branch, tick))"
            ";",
            "CREATE TABLE time_listeners ("
            "action TEXT NOT NULL, "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick INTEGER NOT NULL DEFAULT 0, "
            "active BOOLEAN NOT NULL DEFAULT 1, "
            "PRIMARY KEY(action, branch, tick))"
            ";",
            "CREATE TABLE rules_handled ("
            "character TEXT NOT NULL, "
            "rule TEXT NOT NULL, "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick INTEGER NOT NULL DEFAULT 0,"
            "PRIMARY KEY(character, rule, branch, tick), "
            "FOREIGN KEY(character) REFERENCES graphs(graph), "
            "FOREIGN KEY(rule) REFERENCES rules(rule))"
            ";"
        ]:
            self.cursor.execute(stmt)

    def close(self):
        self.worldview.close()
        self.function.close()

    def new_character(self, name):
        ch = self.worldview.new_character(name)
        ch.rule = CharRules(self, ch)
        return ch

    def get_character(self, name):
        ch = self.worldview.get_character(name)
        ch.rule = CharRules(self, ch)
        return ch

    def del_character(self, name):
        self.worldview.del_character(name)

    def new_rule(self, name, actions=[], prereqs=[]):
        """Create a new Rule by this name, with these actions and prereqs.

        It will be saved in the SQL database, and the actions and
        prereqs will be saved in the dbm database.

        """
        return Rule(self, name, actions, prereqs)

    def get_rule(self, name):
        return Rule(self, name)

    def del_rule(self, name):
        self.cursor.execute(
            "DELETE FROM rules WHERE rule=?;",
            (name,)
        )

    def _active_branches(self):
        return self.worldview._active_branches()

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
                    yield (self.get_character(char), self.get_rule(rule))
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
