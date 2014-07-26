# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
"""Object relational mapper that serves Characters and Rules."""
from collections import Mapping, MutableMapping
from sqlite3 import connect, OperationalError, IntegrityError
from marshal import dumps as marshalled
from marshal import loads as unmarshalled
from types import FunctionType
from gorm import ORM as gORM
from LiSE.rule import Rule
from LiSE.character import (
    Character,
    CharRules
)

class GlobalVarMapping(MutableMapping):
    def __init__(self, worldview):
        self.worldview = worldview

    def __iter__(self):
        self.worldview.cursor.execute(
            "SELECT key FROM global;"
        )
        for row in self.worldview.cursor.fetchall():
            yield row[0]

    def __len__(self):
        self.worldview.cursor.execute(
            "SELECT COUNT(*) FROM global;"
        )
        return self.worldview.cursor.fetchone()[0]

    def __getitem__(self, k):
        self.worldview.cursor.execute(
            "SELECT value FROM global WHERE key=?;",
            (k,)
        )
        try:
            return self.worldview.cursor.fetchone()[0]
        except TypeError:
            raise KeyError("No value for {}".format(k))

    def __setitem__(self, k, v):
        del self[k]  # doesn't throw exception when k doesn't exist
        self.worldview.cursor.execute(
            "INSERT INTO global (key, value) VALUES (?, ?);",
            (k, v)
        )

    def __delitem__(self, k):
        self.worldview.cursor.execute(
            "DELETE FROM global WHERE key=?;",
            (k,)
        )


class CharacterMapping(Mapping):
    def __init__(self, worldview, wrapper=lambda x: x):
        self.worldview = worldview
        self.wrapper = wrapper

    def __iter__(self):
        self.worldview.cursor.execute(
            "SELECT graph FROM graphs;"
        )
        for row in self.worldview.cursor.fetchall():
            yield row[0]

    def __len__(self):
        return self.worldview.cursor.execute(
            "SELECT COUNT(*) FROM graphs;"
        ).fetchone()[0]

    def __contains__(self, name):
        return bool(self.worldview.cursor.execute(
            "SELECT COUNT(*) FROM graphs WHERE graph=?;",
            (name,)
        ).fetchone()[0])

    def __getitem__(self, name):
        if name not in self:
            raise KeyError("No character named {}, maybe you want to add_character?".format(name))
        return self.wrapper(Character(self.worldview, name))


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
        self.eternal = GlobalVarMapping(self)
        self.character = CharacterMapping(self)
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

    def commit(self):
        self.cursor.connection.commit()
        self.cursor.execute("BEGIN;")

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


class Timeline(Mapping):
    """Access a worldview at an arbitrary tick within a branch"""
    def __init__(self, conn, branch):
        self.connection = conn
        self.branch = branch

    def __iter__(self):
        """Iterate from the start of the branch to the last tick in the branch
        that has any data

        """
        return range(self._parrev(), self._maxtick())

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


class ORM(object):
    def __init__(self, worlddb, codedb):
        self.worldview = Worldview(worlddb)
        self.eternal = self.worldview.eternal
        self.character = CharacterMapping(
            self.worldview,
            self._wrap_character
        )
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
        facade = (
            "CREATE TABLE {} ("
            "observer_char TEXT NOT NULL, "
            "observed_char TEXT NOT NULL, "
            "facade TEXT NOT NULL DEFAULT 'perception', "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick INTEGER NOT NULL DEFAULT 0, "
            "mungers TEXT NOT NULL DEFAULT '[]', "
            "PRIMARY KEY(observer_char, observed_char, branch, tick), "
            "FOREIGN KEY(observer_char, observed_char, facade) "
            "REFERENCES facades(observer_char, observed_char, facade))"
            ";"
        )
        listener = (
            "CREATE TABLE {} ("
            "action TEXT NOT NULL, "
            "branch TEXT NOT NULL DEFAULT 'master', "
            "tick INTEGER NOT NULL DEFAULT 0, "
            "active BOOLEAN NOT NULL DEFAULT 1, "
            "PRIMARY KEY(action, branch, tick))"
            ";"
        )

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
                "CREATE TABLE rules_handled ("
                "character TEXT NOT NULL, "
                "rule TEXT NOT NULL, "
                "branch TEXT NOT NULL DEFAULT 'master', "
                "tick INTEGER NOT NULL DEFAULT 0,"
                "PRIMARY KEY(character, rule, branch, tick), "
                "FOREIGN KEY(character) REFERENCES graphs(graph), "
                "FOREIGN KEY(rule) REFERENCES rules(rule))"
                ";",
                "CREATE TABLE mungers ("
                "munger TEXT NOT NULL PRIMARY KEY, "
                "omitter TEXT NOT NULL DEFAULT '[]', "
                "distorter TEXT NOT NULL DEFAULT '[]')"
                ";",
                "CREATE TABLE facades ("
                "observer_char TEXT NOT NULL, "
                "observed_char TEXT NOT NULL, "
                "facade TEXT NOT NULL, "
                "branch TEXT NOT NULL DEFAULT 'master', "
                "tick INTEGER NOT NULL DEFAULT 0, "
                "active BOOLEAN NOT NULL DEFAULT 1, "
                "PRIMARY KEY(observer_char, observed_char, facade, branch, tick), "
                "FOREIGN KEY(observer_char) REFERENCES graphs(graph), "
                "FOREIGN KEY(observed_char) REFERENCES graphs(graph))"
                ";",
                facade.format("facade_things"),
                facade.format("facade_thing_stats"),
                facade.format("facade_places"),
                facade.format("facade_place_stats"),
                facade.format("facade_portals"),
                facade.format("facade_portal_stats"),
                listener.format("branch_listeners"),
                listener.format("tick_listeners"),
                listener.format("time_listeners")
        ]:
            self.cursor.execute(stmt)

    def close(self):
        self.worldview.close()
        self.function.close()

    def add_character(self, name):
        self.worldview.add_character(name)

    def del_character(self, name):
        self.worldview.del_character(name)

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

    def commit(self):
        self.worldview.commit()
        self.function.commit()

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

    def _wrap_character(self, character):
        character.rule = CharRules(self, character)
        return character
