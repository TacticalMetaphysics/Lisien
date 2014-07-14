# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
"""Object relational mapper that serves Characters."""
from collections import Mapping
from sqlite3 import connect, OperationalError
import anydbm
from orm import ORM

class Listeners(Mapping):
    """Mapping and decorator for the functions that listen to the time"""
    def __init__(self, orm, tabn):
        """Store the ORM and the name of the table I'm about"""
        self.orm = orm
        self.tabn = tabn

    def __call__(self, fun):
        """Store the function and activate it for the present (branch, tick)

        """
        funn = self.orm.function(fun)
        (branch, tick) = self.orm.time
        self.orm.cursor.execute(
            "DELETE FROM {tab} WHERE "
            "branch=? AND "
            "tick=? AND "
            "action=?;".format(
                tab=self.tabn
            ),
            (
                branch,
                tick,
                funn
            )
        )
        self.orm.cursor.execute(
            "INSERT INTO {tab} (action, branch, tick) "
            "VALUES (?, ?, ?);".format(
                tab=self.tabn
            ),
            (
                funn,
                branch,
                tick
            )
        )

    def __iter__(self):
        """Iterate over the names of active functions"""
        r = set()
        seen = set()
        for (branch, tick) in self.orm._active_branches():
            self.orm.cursor.execute(
                "SELECT {tab}.action, {tab}.active FROM {tab} JOIN ("
                "SELECT action, branch, MAX(tick) AS tick FROM {tab} "
                "WHERE branch=? "
                "AND tick<=? "
                "GROUP BY action, branch) AS hitick "
                "ON {tab}.action=hitick.action "
                "AND {tab}.branch=hitick.branch "
                "AND {tab}.tick=hitick.tick;".format(
                    tab=self.tabn
                ),
                (
                    branch,
                    tick
                )
            )
            for (action, active) in self.orm.cursor.fetchall():
                if active and action not in seen:
                    r.add(action)
        for a in sorted(r):
            yield a

    def __len__(self):
        """Number of functions active in this table presently"""
        n = 0
        seen = set()
        for (branch, tick) in self.orm._active_branches():
            self.orm.cursor.execute(
                "SELECT {tab}.action, {tab}.active FROM {tab} JOIN ("
                "SELECT action, branch, MAX(tick) AS tick "
                "FROM {tab} WHERE "
                "branch=? AND "
                "tick<=? GROUP BY action, branch) AS hitick "
                "ON {tab}.action=hitick.action "
                "AND {tab}.branch=hitick.branch "
                "AND {tab}.tick=hitick.tick;".format(
                    tab=self.tabn
                ),
                (
                    branch,
                    tick
                )
            )
            for (action, active) in self.orm.cursor.fetchall():
                if active and action not in seen:
                    n += 1
                seen.add(action)
        return n

    def __getitem__(self, actn):
        """Return the function by the given name if it's active at the
        moment

        """
        for (branch, tick) in self.orm._active_branches():
            self.orm.cursor.execute(
                "SELECT {tab}.active FROM {tab} JOIN ("
                "SELECT action, branch, MAX(tick) AS tick "
                "FROM {tab} WHERE "
                "action=? AND "
                "branch=? AND "
                "tick<=? GROUP BY action, branch) AS hitick "
                "ON {tab}.action=hitick.action "
                "AND {tab}.branch=hitick.branch "
                "AND {tab}.tick=hitick.tick;".format(
                    tab=self.tabn
                ),
                (
                    actn,
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
                    return self.orm.getfun(actn)
                else:
                    raise KeyError("Listener {} disabled".format(actn))
        raise KeyError("Listener {} doesn't exist".format(actn))

    def __delitem__(self, actn):
        """Deactivate the function by the given name"""
        (branch, tick) = self.orm.time
        self.orm.cursor.execute(
            "DELETE FROM {tab} WHERE "
            "action=? AND "
            "branch=? AND "
            "tick=?;".format(tab=self.tabn),
            (
                actn,
                branch,
                tick
            )
        )
        self.orm.cursor.execute(
            "INSERT INTO {tab} (action, branch, tick, active) "
            "VALUES (?, ?, ?, ?);",
            (
                actn,
                branch,
                tick,
                False
            )
        )

class LiSE(object):
    def __init__(self, world_filename, code_filename, gettext=lambda s: s):
        """Store the connections for the world database and the code database;
        set up listeners; and start a transaction

        """
        self.gettext = gettext
        self.orm = ORM(
            worlddb=connect(world_filename),
            codedb=anydbm.open(code_filename, 'c')
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
        self._rules_iter = self._follow_rules()

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
        for listener in self.on_branch.values():
            listener(self, v)
        if not hasattr(self, 'locktime'):
            t = self.tick
            for time_listener in self.on_time.values():
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
        self.orm.time = v
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
        for (rule, args) in self.orm._poll_rules():
            s = rule(self, *args)
            self.time = (branch, tick)  # in case the rule moved it
            self.orm._char_rule_handled(args[0].name, rule.name)
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

    def new_character(self, name):
        return self.orm.new_character(name)

    def get_character(self, name):
        return self.orm.get_character(name)

    def del_character(self, name):
        return self.orm.del_character(name)

    def close(self):
        self.orm.close()
