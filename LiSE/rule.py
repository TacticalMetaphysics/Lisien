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
    def __init__(self, orm, name, actions=[], prereqs=[]):
        """Store the ORM and my name, make myself a record in the database if
        needed, and instantiate FunList once for my actions and again
        for my prereqs.

        """
        self.orm = orm
        self.name = name
        self.orm.cursor.execute(
            "SELECT COUNT(*) FROM rules WHERE rule=?;",
            (name,)
        )
        (ct,) = self.orm.cursor.fetchone()
        if ct == 0:
            self.orm.cursor.execute(
                "INSERT INTO rules (rule) VALUES (?);",
                (name,)
            )
        self.actions = FunList(self.orm, 'rules', ['rule'], [self.name], 'actions', actions)
        self.prereqs = FunList(self.orm, 'rules', ['rule'], [self.name], 'prereqs', prereqs)

    def __call__(self, lise, character):
        """First check the prereqs. If they all pass, execute the actions and
        return a list of all their results.

        After each call to a prereq or action, the sim-time is reset
        to what it was before the rule was called.

        """
        curtime = lise.time
        for prereq in self.prereqs:
            # in case one of them moves the time
            if not prereq(lise, character, self):
                return []
            lise.time = curtime
        r = []
        for action in self.actions:
            r.append(action(lise, character, self))
            lise.time = curtime
        return r

    def prereq(self, fun):
        """Decorator to append the function to my prereqs list."""
        self.prereqs.append(fun)

    def action(self, fun):
        """Decorator to append the function to my actions list."""
        self.actions.append(fun)
