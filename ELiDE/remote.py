# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
"""Classes to listen to changes in the simulation from elsewhere, such
as in the user interface.

"""
from kivy.event import EventDispatcher
from kivy.properties import (
    DictProperty,
    ListProperty,
    ObjectProperty
)
from kivy.clock import Clock
from kivy.logger import Logger


class MirrorMapping(EventDispatcher):
    """Holds a :class:`DictProperty`, ``mirror``, which always has the
    value of the LiSE entity, ``remote``, that the user should see at
    the moment.

    """
    time = ListProperty(['master', 0])
    remote = ObjectProperty(None, allownone=True)
    mirror = DictProperty()

    def on_time(self, *args):
        if not self.remote:
            Clock.schedule_once(self.on_time, 0)
            return
        self.mirror = dict(self.remote)

    def on_remote(self, *args):
        for (k, v) in self.remote.items():
            if v:
                self.mirror[k] = v

        if not hasattr(self.remote, 'listener'):
            return

        @self.remote.listener
        def when_changed(branch, tick, what, k, v):
            if (
                    branch == self.time[0] and
                    tick == self.time[1] and (
                        k not in self.mirror or
                        self.mirror[k] != v
                    )
            ):
                if v is None:
                    if k in self.mirror:
                        del self.mirror[k]
                else:
                    self.mirror[k] = v
        return True
