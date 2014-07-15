# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from collections import MutableSequence
from json import dumps as jsonned
from json import loads as unjsonned


class FunList(MutableSequence):
    """Persistent list of functions associated with a rule.

    Stored as JSON lists of strings in the world database. Said
    strings identify functions in the code database.

    """
    def __init__(self, rule, field, data):
        """Store the rule and the name of the field to put the data in, then
        put the data in it.

        """
        self.rule = rule
        self.field = field
        self.orm = rule.orm
        self.name = rule.name
        funns = [self.orm.function(fun) for fun in data]
        if funns:
            self._setlist(funns)

    def _setlist(self, l):
        """Update the rule's record with this new list of strings."""
        self.orm.cursor.execute(
            "UPDATE rules SET {field}=? WHERE rule=?;".format(
                field=self.field
            ), (jsonned(l), self.rule.name)
        )

    def _getlist(self):
        """Return the list, decoded from JSON, but not yet translated to
        actual functions, just their names.

        """
        self.orm.cursor.execute(
            "SELECT {field} FROM rules WHERE rule=?;".format(
                field=self.field
            ), (self.rule.name,)
        )
        return unjsonned(self.orm.cursor.fetchone()[0])

    def __iter__(self):
        """Yield a function from the code database for each item in the
        list

        """
        for funn in self._getlist():
            yield self.orm.function[funn]

    def __len__(self):
        """Return the length of the list (don't translate it to functions)"""
        return len(self._getlist())

    def __getitem__(self, i):
        """Get the function named by the ith item in the list"""
        return self.orm.function[self._getlist()[i]]

    def __setitem__(self, i, v):
        """If ``v`` is a function, store it and get its name, otherwise it's
        the name of a function. Either way, set the ith item to
        the name.

        """
        funn = self.orm.function(v)
        l = self._getlist()
        l[i] = funn
        self._setlist(l)

    def __delitem__(self, i):
        """Delete item i and save the list"""
        l = self._getlist()
        del l[i]
        self._setlist(l)

    def insert(self, i, v):
        """Insert the name of function ``v`` (or just ``v`` if it's a string)
        into position i

        """
        funn = self.orm.function(v)
        l = self._getlist()
        l.insert(i, funn)
        self._setlist(l)


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
        self.actions = FunList(self, 'actions', actions)
        self.prereqs = FunList(self, 'prereqs', prereqs)

    def __call__(self, lise, *args):
        """First check the prereqs. If they all pass, execute the actions and
        return a list of all their results.

        After each call to a prereq or action, the sim-time is reset
        to what it was before the rule was called.

        """
        curtime = lise.time
        for prereq in self.prereqs:
            # in case one of them moves the time
            if not prereq(lise, self, *args):
                return []
            lise.time = curtime
        r = []
        for action in self.actions:
            r.append(action(lise, self, *args))
            lise.time = curtime
        return r

    def prereq(self, fun):
        """Decorator to append the function to my prereqs list."""
        self.prereqs.append(fun)

    def action(self, fun):
        """Decorator to append the function to my actions list."""
        self.actions.append(fun)
