# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from .funlist import FunList


class Rule(object):
    """A collection of actions, being functions that enact some change on
    the world, which will be called each tick if and only if all of
    the prereqs return True, they being boolean functions that do not
    change the world.

    """
    def __init__(self, engine, name):
        """Store the ENGINE and my name, make myself a record in the database if
        needed, and instantiate FunList once for my actions and again
        for my prereqs.

        """
        self.engine = engine
        self.name = name
        self.engine.cursor.execute(
            "SELECT COUNT(*) FROM rules WHERE rule=?;",
            (name,)
        )
        (ct,) = self.engine.cursor.fetchone()
        if ct == 0:
            self.engine.cursor.execute(
                "INSERT INTO rules (rule) VALUES (?);",
                (name,)
            )
        self.actions = FunList(self.engine, 'rules', ['rule'], [self.name], 'actions')
        self.prereqs = FunList(self.engine, 'rules', ['rule'], [self.name], 'prereqs')

    @property
    def priority(self):
        return self.engine.cursor.execute(
            "SELECT priority FROM rules WHERE name=?;",
            (self.name,)
        ).fetchone()[0]

    @priority.setter
    def priority(self, v):
        self.engine.cursor.execute(
            "UPDATE rules SET priority=? WHERE name=?;",
            (v, self.name)
        )

    def __cmp__(self, other):
        """Enable sorting by priority"""
        myprio = self.priority
        oprio = other.priority
        if myprio < oprio:
            return -1
        elif myprio == oprio:
            if self.name < other.name:
                return -1
            elif self.name == other.name:
                return 0
            else:
                return 1
        else:
            return 1

    def __call__(self, lise, character):
        """First check the prereqs. If they all pass, execute the actions and
        return a list of all their results.

        After each call to a prereq or action, the sim-time is reset
        to what it was before the rule was called.

        """
        curtime = lise.time
        r = None
        for prereq in self.prereqs:
            # in case one of them moves the time
            if not prereq(lise, character):
                r = []
            lise.time = curtime
            if r is not None:
                break
        if r is not None:
            return r
        r = []
        for action in self.actions:
            r.append(action(lise, character))
            lise.time = curtime
        return r

    def prereq(self, fun):
        """Decorator to append the function to my prereqs list."""
        self.prereqs.append(fun)

    def action(self, fun):
        """Decorator to append the function to my actions list."""
        self.actions.append(fun)
