# coding: utf-8
# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
from collections import (
    MutableSequence,
    Callable
)
from json import dumps as jsonned
from json import loads as unjsonned
from util import listen


class FunList(MutableSequence):
    """Persistent list of functions associated with a rule.

    Stored as JSON lists of strings in the world database. Said
    strings identify functions in the code database.

    """
    def __init__(
            self,
            engine,
            funcstore,
            table,
            preset_fields,
            preset_values,
            field
    ):
        """Store the rule and the name of the field to put the data in, then
        put the data in it.

        """
        self.table = table
        self.preset_fields = tuple(preset_fields)
        self.preset_values = tuple(preset_values)
        self.presets = " AND ".join(
            "{}=?".format(f) for f in self.preset_fields
        )
        self.field = field
        self.engine = engine
        self.funcstore = funcstore
        # if I don't have a record yet, make one
        cursor = self.engine.db.connection.cursor()
        if cursor.execute(
            "SELECT COUNT(*) FROM {table} WHERE {presets};".format(
                table=self.table,
                presets=self.presets
            ),
            self.preset_values
        ).fetchone()[0] == 0:
            cursor.execute(
                "INSERT INTO {table} ({fields}, {field}) "
                "VALUES ({qpreset}, '[]');".format(
                    table=self.table,
                    fields=", ".join(self.preset_fields),
                    field=self.field,
                    qpreset=", ".join("?" * len(self.preset_values))
                ),
                self.preset_values
            )
        self._listeners = []

    def _dispatch(self):
        for f in self._listeners:
            f(self)

    def listener(self, f):
        listen(self._listeners, f)

    def _funn(self, v):
        funn = v.__name__ if isinstance(v, Callable) else v
        if funn not in self.funcstore:
            if not isinstance(v, Callable):
                raise KeyError("No such function: " + v)
            self.funcstore[funn] = v
        return funn

    def _setlist(self, l):
        """Update the rule's record with this new list of strings."""
        self.engine.db.connection.cursor().execute(
            "UPDATE {table} SET {field}=? WHERE {presets};".format(
                table=self.table,
                field=self.field,
                presets=self.presets
            ), (jsonned(l),) + self.preset_values
        )

    def _getlist(self):
        """Return the list, decoded from JSON, but not yet translated to
        actual functions, just their names.

        """
        return unjsonned(
            self.engine.db.connection.cursor().execute(
                "SELECT {field} FROM {table} WHERE {presets};".format(
                    field=self.field,
                    table=self.table,
                    presets=self.presets
                ),
                self.preset_values
            ).fetchone()[0]
        )

    def __iter__(self):
        """Yield a function from the code database for each item in the
        list

        """
        for funn in self._getlist():
            yield self.funcstore[funn]

    def __len__(self):
        """Return the length of the list (don't translate it to functions)"""
        return len(self._getlist())

    def __getitem__(self, i):
        """Get the function named by the ith item in the list"""
        return self.funcstore[self._getlist()[i]]

    def __setitem__(self, i, v):
        """If ``v`` is a function, store it and get its name, otherwise it's
        the name of a function. Either way, set the ith item to
        the name.

        """
        l = self._getlist()
        l[i] = self._funn(v)
        self._setlist(l)
        self._dispatch()

    def __delitem__(self, i):
        """Delete item i and save the list"""
        l = self._getlist()
        del l[i]
        self._setlist(l)
        self._dispatch()

    def insert(self, i, v):
        """Insert the name of function ``v`` (or just ``v`` if it's a string)
        into position i

        """
        l = self._getlist()
        l.insert(i, self._funn(v))
        self._setlist(l)
        self._dispatch()
