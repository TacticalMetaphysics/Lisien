# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from gamepiece import GamePiece
from kivy.properties import (
    ListProperty,
    ObjectProperty)
from kivy.clock import Clock
from kivy.logger import Logger


"""Widgets to represent places. Pawns move around on top of these."""


class Spot(GamePiece):
    """The icon that represents a Place.

    The Spot is located on the Board that represents the same
    Dimension that the underlying Place is in. Its coordinates are
    relative to its Board, not necessarily the window the Board is in.

    """
    place = ObjectProperty()
    board = ObjectProperty()
    bone = ObjectProperty()
    _touch = ObjectProperty(None, allownone=True)
    pawns_here = ListProperty([])

    def __init__(self, **kwargs):
        if 'board' in kwargs and 'closet' not in kwargs:
            kwargs['closet'] = kwargs['board'].host.closet
        kwargs['bone'] = kwargs['closet'].skeleton[u'spot'][
            unicode(kwargs['board'].facade.observer)][
            unicode(kwargs['place'].character)][
            unicode(kwargs['place'])][
            kwargs['closet'].branch].value_during(
            kwargs['closet'].tick)
        kwargs['graphic_name'] = kwargs['bone'].graphic
        kwargs['imgs'] = kwargs['closet'].get_game_piece(
            kwargs['bone'].graphic).imgs
        self._trigger_move_to_touch = Clock.create_trigger(
            self._move_to_touch)
        super(Spot, self).__init__(**kwargs)
        self.closet.register_time_listener(self.handle_time)
        self.board.spotdict[unicode(self.place)] = self
        self.bind(
            pawns_here=self.upd_pawns_here,
            center=self.upd_pawns_here)

    def __str__(self):
        """Return the name of my :class:`Place`."""
        return str(self.place)

    def __unicode__(self):
        """Return the name of my :class:`Place`."""
        return unicode(self.place)

    def upd_texs(self, *args):
        super(Spot, self).upd_texs(*args)

    def handle_time(self, b, t):
        try:
            self.bone = self.get_bone(b, t)
        except KeyError:
            Logger.debug("Spot: No bone at ({}, {}); delaying".format(
                b, t))
            Clock.schedule_once(lambda dt: self.handle_time(b, t), 0)
            return
        self.graphic_name = self.bone.graphic
        self.repos(b, t)

    def repos(self, b, t):
        if not self.graphic_bone:
            Clock.schedule_once(lambda dt: self.repos(b, t), 0)
            return
        bone = self.get_coord_bone(b, t)
        x = bone.x + self.graphic_bone.offset_x
        y = bone.y + self.graphic_bone.offset_y
        self.pos = (x, y)

    def upd_pawns_here(self, *args):
        for pawn in self.pawns_here:
            pawn.pos = self.center

    def sanetime(self, branch, tick):
        return self.board.facade.sanetime(branch, tick)

    def get_bone(self, branch=None, tick=None):
        (branch, tick) = self.sanetime(branch, tick)
        return self.closet.skeleton[u"spot"][
            unicode(self.board.facade.observer)][
            unicode(self.board.host)][
            unicode(self.place)][branch].value_during(tick)

    def get_coord_bone(self, branch=None, tick=None):
        (branch, tick) = self.sanetime(branch, tick)
        return self.board.facade.closet.skeleton[u"spot_coords"][
            unicode(self.board.facade.observer)][
            unicode(self.board.host)][unicode(self.place)][
            branch].value_during(tick)

    def get_coords(self, branch=None, tick=None, default=None):
        """Return a pair of coordinates for where I should be on my board,
        either now, or at the given point in time.

        """
        try:
            bone = self.get_coord_bone(branch, tick)
            if bone is None:
                raise KeyError
            else:
                return (bone.x, bone.y)
        except KeyError:
            if default is not None:
                self.set_coords(*default)
                return default

    def set_coords(self, x, y, branch=None, tick=None):
        """Set my coordinates on the :class:`Board`.

        Optional arguments may be used to set my coordinates as of
        some time other than "right now".

        """
        (branch, tick) = self.sanetime(branch, tick)
        bone = self.get_coord_bone(branch, tick)
        self.closet.set_bone(bone._replace(
            x=x, y=y,
            branch=branch, tick=tick))

    def set_center_coords(self, x, y, branch=None, tick=None):
        self.set_coords(x - self.width / 2, y - self.height / 2,
                        branch, tick)

    def new_branch(self, parent, branch=None, tick=None):
        """Copy all the stuff from the parent to the child branch as of the
        given tick.

        """
        (branch, tick) = self.sanetime(branch, tick)
        prev = None
        started = False
        for skel in (
                self.board.host.closet.skeleton[u"spot"][
                    unicode(self.board.facade.observer)][
                    unicode(self.place.character)][
                    unicode(self.place)],
                self.board.host.closet.skeleton[u"spot_coords"][
                    unicode(self.board.facade.observer)][
                    unicode(self.place.character)][
                    unicode(self.place)]):
            for bone in skel.iterbones():
                if bone.tick >= tick:
                    yield bone._replace(branch=branch)
                    if (
                            not started and prev is not None and
                            bone.tick > tick and prev.tick < tick):
                        yield prev._replace(
                            branch=branch,
                            tick=tick)
                    started = True
                    prev = bone

    def on_touch_down(self, touch):
        if (
                not self.collide_point(*touch.pos)
                or 'spot' in touch.ud):
            return
        touch.grab(self)
        touch.ud['spot'] = self
        self._touch = touch
        return self

    def on_touch_move(self, touch):
        if 'portaling' in touch.ud or 'pawn' in touch.ud:
            touch.ungrab(self)
            return
        elif 'spot' in touch.ud:
            if touch.ud['spot'] is not self:
                return
            self._touch = touch
            self._trigger_move_to_touch()
            return self

    def _move_to_touch(self, *args):
        if self._touch:
            self.center = self.board.parent.to_local(*self._touch.pos)

    def on_touch_up(self, touch):
        if self._touch:
            self.set_coords(*self.pos)
        self._touch = None
        if self.collide_point(*touch.pos):
            return self

    def __repr__(self):
        return "{}@({},{})".format(self.place.name, self.x, self.y)
