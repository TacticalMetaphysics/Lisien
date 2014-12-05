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

    def on_layout(self, *args):
        if not self.layout:
            return
        self.layout.bind(time=self.on_time)

    def on_time(self, *args):
        self.mirror = dict(self.remote)

    def on_remote(self, *args):
        if not self.layout:
            Clock.schedule_once(self.on_remote, 0)
            return

        self.mirror = dict(self.remote)

        @self.remote.listener
        def when_changed(branch, tick, what, k, v):
            if (
                    branch == self.layout.branch and
                    tick == self.layout.tick and (
                        k not in self.mirror or
                        self.mirror[k] != v
                    )
            ):
                if k in self.mirror:
                    Logger.debug(
                        "MirrorMapping: changing {} from {} to {}".format(
                            k, self.mirror[k], v
                        )
                    )
                else:
                    Logger.debug(
                        "MirrorMapping: setting {} to {}".format(
                            k, v
                        )
                    )
                self.mirror[k] = v
        return True
