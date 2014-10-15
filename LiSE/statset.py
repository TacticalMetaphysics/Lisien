from .util import CacheError


class StatSet(object):
    """Mixin class for those that should call listeners when stats are set
    or deleted. Assumes they will have an attribute ``_on_stat_set``
    that is a list of listener functions to call.

    Does not automatically call the listeners. Use methods
    ``_stat_set`` and ``_stat_del`` to do so.

    """

    def on_stat_set(self, fun):
        """Decorator for functions that should be called whenever one of my
        stats is set to a new value or deleted.

        If there's a new value, the arguments will be ``self, key,
        value``. If the stat is getting deleted, the arguments will be
        just ``self, key``.

        Note that the function will NOT be called if a stat appears to
        change value due to time travel.

        This only works when caching's enabled. If it isn't then you
        should implement a trigger in your database instead.

        """
        if not hasattr(self, '_on_stat_set'):
            raise CacheError("Caching disabled")
        self._on_stat_set.append(fun)

    def _stat_set(self, k, v):
        """Notify listeners"""
        if hasattr(self, '_on_stat_set'):
            (branch, tick) = self.engine.time
            for fun in self._on_stat_set:
                fun(self, branch, tick, k, v)

    def _stat_del(self, k):
        """Notify listeners"""
        if hasattr(self, '_on_stat_set'):
            (branch, tick) = self.engine.time
            for fun in self._on_stat_set:
                fun(self, branch, tick, k)
