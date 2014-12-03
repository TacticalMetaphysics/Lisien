# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
"""Classes to listen to changes in the simulation from elsewhere, such
as in the user interface.

"""
from collections import MutableMapping, defaultdict


class AbstractRemoteMapping(MutableMapping):
    def __init__(self, remote):
        self._remote_data = remote
        self._current_data = {}
        self._listeners = defaultdict(list)

    def __iter__(self):
        return iter(self._remote_data)

    def __len__(self):
        return len(self._remote_data)

    def __getitem__(self, k):
        if k not in self._current_data:
            self.fetch(k)
        return self._current_data[k]

    def __setitem__(self, k, v):
        self.push(k, v)
        self.fetch(k)
        return self._current_data[k]

    def __delitem__(self, k):
        self.smite(k)
        del self._current_data[k]

    def listener(self, func=None, key=None):
        if func is None:
            def listen(f):
                self._listeners[key].append(f)
                return f
            return listen
        self._listeners[key].append(func)
        return func

    def unlisten(self, func, key=None):
        if func in self._listeners[key]:
            self._listeners[key].remove(func)

    def fetch(self, k):
        old = self._current_data[k] if k in self._current_data else None
        self._current_data[k] = self._remote_data[k]
        if old != self._current_data[k]:
            for f in self._listeners[k]:
                f(k, self._current_data[k])

    def push(self, k, v):
        self._remote_data[k] = v

    def smite(self, k):
        del self._remote_data[k]
        for f in self._listeners[k]:
            f(k, None)


class CharacterRemoteMapping(AbstractRemoteMapping):
    def __init__(self, character, stat=None):
        super().__init__(character.stat)

        @character.stat.listener(stat=stat)
        def listen(char, k, v):
            self.fetch(k)


class EntityRemoteMapping(AbstractRemoteMapping):
    def __init__(self, ent, stat=None):
        super().__init__(ent)

        @ent.listener(stat=stat)
        def listen(ent, k, v):
            self.fetch(k)
