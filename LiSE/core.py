# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
"""Object relational mapper that serves Characters."""
from collections import Mapping, Callable
from sqlite3 import connect, OperationalError
from .orm import ORM
from .rule import Rule

class Listeners(Mapping):
    """Mapping and decorator for the functions that listen to the time"""
    def __init__(self, orm, tabn):
        """Store the ORM and the name of the table I'm about"""
        self.orm = orm
        self.tabn = tabn

    def __call__(self, v):
        """Store the function and activate it for the present (branch, tick)

        """
        if isinstance(v, Rule):
            self._activate_rule(v)
        elif isinstance(v, Callable):
            vname = self.orm.function(v)
            self._activate_rule(
                Rule(
                    self.orm,
                    vname,
                    actions=[vname]
                )
            )
        else:
            self._activate_rule(Rule(self.orm, vname))

    def _activate_rule(self, rule):
        (branch, tick) = self.orm.time
        self.orm.cursor.execute(
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
        self.orm.cursor.execute(
            "INSERT INTO {tab} (rule, branch, tick, active) VALUES (?, ?, ?, 1);".format(
                tab=self.tabn
            ),
            (rule.name, branch, tick)
        )

    def __iter__(self):
        """Iterate over the names of active rules"""
        r = set()
        seen = set()
        for (branch, tick) in self.orm._active_branches():
            self.orm.cursor.execute(
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
            for (rule, active) in self.orm.cursor.fetchall():
                if active and rule not in seen:
                    r.add(rule)
        for a in sorted(r):
            yield a

    def __len__(self):
        """Number of rules active in this table presently"""
        n = 0
        seen = set()
        for (branch, tick) in self.orm._active_branches():
            self.orm.cursor.execute(
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
            for (rule, active) in self.orm.cursor.fetchall():
                if active and rule not in seen:
                    n += 1
                seen.add(rule)
        return n

    def __getitem__(self, rulen):
        """Return the rule by the given name if it's active at the
        moment

        """
        for (branch, tick) in self.orm._active_branches():
            self.orm.cursor.execute(
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
            data = self.orm.cursor.fetchall()
            if len(data) == 0:
                continue
            elif len(data) > 1:
                raise ValueError("Silly data in {tab} table".format(self.tabn))
            else:
                active = data[0][0]
                if active:
                    return Rule(self.orm, rulen)
                else:
                    raise KeyError("Listener {} disabled".format(rulen))
        raise KeyError("Listener {} doesn't exist".format(rulen))

    def __delitem__(self, rulen):
        """Deactivate the rule by the given name"""
        (branch, tick) = self.orm.time
        self.orm.cursor.execute(
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
        self.orm.cursor.execute(
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
        # avoid ndbm due to its lack of a sync method
        self.orm = ORM(
            worlddb=connect(world_filename),
            codedb=connect(code_filename)
        )
        try:
            self.orm.cursor.execute(
                "SELECT * FROM things;"
            ).fetchall()
        except OperationalError:
            self.orm.initdb()
        self.on_branch = Listeners(self.orm, 'branch_listeners')
        self.on_tick = Listeners(self.orm, 'tick_listeners')
        self.on_time = Listeners(self.orm, 'time_listeners')
        self.function = self.orm.function
        self.eternal = self.orm.eternal
        self.character = self.orm.character
        self._rules_iter = self._follow_rules()

    def commit(self):
        """Commit all pending changes to everything"""
        self.orm.commit()

    @property
    def branch(self):
        """Alias for my orm's branch"""
        return self.orm.branch

    @branch.setter
    def branch(self, v):
        """Set my orm's branch and call listeners"""
        if v == self.orm.branch:
            return
        self.orm.branch = v
        for listener in list(self.on_branch.values()):
            listener(self, v)
        if not hasattr(self, 'locktime'):
            t = self.tick
            for time_listener in list(self.on_time.values()):
                time_listener(self, v, t)

    @property
    def tick(self):
        """Alias for my orm's tick"""
        return self.orm.tick

    @tick.setter
    def tick(self, v):
        """Update orm's tick, and call listeners"""
        if v == self.orm.tick:
            return
        self.orm.tick = v
        for tick_listener in list(self.on_tick.values()):
            tick_listener(self, v)
        if not hasattr(self, 'locktime'):
            b = self.branch
            for time_listener in list(self.on_tick.values()):
                time_listener(self, b, v)

    @property
    def time(self):
        """Return tuple of branch and tick"""
        return (self.branch, self.tick)

    @time.setter
    def time(self, v):
        self.locktime = True
        self.orm.time = v
        (b, t) = v
        for time_listener in list(self.on_time.values()):
            time_listener(self, b, t)
        del self.locktime

    def _follow_rules(self):
        """Execute the rules for the present branch and tick, and record the
        fact of it in the SQL database.

        Yield the rules paired with their results.

        """
        (branch, tick) = self.time
        for (character, rule) in self.orm._poll_rules():
            s = rule(self, character)
            self.orm._char_rule_handled(character.name, rule.name)
            yield (rule, s)

    def advance(self):
        """Follow the next rule, advancing to the next tick if necessary.

        """
        try:
            return next(self._rules_iter)
        except StopIteration:
            if not self.orm._have_rules():
                raise ValueError(
                    "No rules available; can't advance."
                )
            self.tick += 1
            self._rules_iter = self._follow_rules()
            return self.advance()

    def add_character(self, name):
        self.orm.add_character(name)

    def new_character(self, name):
        self.orm.add_character(name)
        return self.character[name]

    def del_character(self, name):
        self.orm.del_character(name)

    def close(self):
        self.orm.close()
