# This file is part of LiSE, a framework for life simulation games.
# Copyright (C) 2013-2014 Zachary Spector, ZacharySpector@gmail.com
"""Classes to listen to changes in the simulation from elsewhere, such
as in the user interface.

"""
from kivy.event import EventDispatcher
from kivy.properties import (
    DictProperty,
    ObjectProperty
)
from kivy.clock import Clock
from kivy.logger import Logger


class MirrorMapping(EventDispatcher):
    layout = ObjectProperty()
    remote = ObjectProperty()
    mirror = DictProperty()

    def on_remote(self, *args):
        if not self.layout:
            Clock.schedule_once(self.on_remote, 0)
            return

        self.mirror = dict(self.remote)

        @self.remote.listener
        def when_changed(branch, tick, what, k, v):
            if branch == self.layout.branch and tick == self.layout.tick:
                self.mirror[k] = v
            else:
                Logger.debug(
                    'changed {}, but not mirroring because {} != {}'.format(
                        k,
                        (branch, tick),
                        (self.layout.branch, self.layout.tick)
                    )
                )
        return True
