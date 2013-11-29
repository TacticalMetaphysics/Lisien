# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
import effect_cbs


class Effect(object):
    """Respond to an Event fired by Implicator in response to a Cause.

An Effect does its thing in its do() method, which will receive a
Character, along with the current branch and tick. It must be
stateless, with no side effects. Instead, it describes a change to the
world using a tuple. Each Effect may only have one change.

    """

    def __init__(self, imp, do=None):
        self.implicator = imp
        if do is not None:
            self.doer(do)

    def __call__(self, event):
        r = self.do(event.character, event.branch, event.tick)
        if not isinstance(r, tuple):
            raise TypeError(
                "do returned non-tuple")
        return r

    def doer(self, cb):
        if cb.__class__ in (str, unicode):
            self.do = self.imp.closet.get_effect_cb(cb)
        else:
            self.do = cb
