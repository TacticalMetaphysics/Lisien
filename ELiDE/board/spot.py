# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
"""Widgets to represent places. Pawns move around on top of these."""
from kivy.properties import (
    AliasProperty,
    ListProperty,
    ObjectProperty,
    NumericProperty
)
from kivy.clock import Clock
from kivy.logger import Logger
from ELiDE.kivygarden.collider import CollideEllipse
from .pawnspot import PawnSpot


class Spot(PawnSpot):
    """The icon that represents a :class:`Place`.

    Each :class:`Spot` is located on the Board that represents the
    :class:`Character` that the underlying :class:`Place` is in. Its
    coordinates are relative to its :class:`Board`, not necessarily
    the window the :class:`Board` is in.

    """
    offset = NumericProperty(4)
    collider = ObjectProperty()
    place = AliasProperty(
        lambda self: self.remote,
        lambda self, v: self.remote.setter()(v),
        bind=('remote',)
    )
    _touchpos = ListProperty([])

    def __init__(self, **kwargs):
        """Deal with triggers and bindings, and arrange to take care of
        changes in game-time.

        """
        self._trigger_renamed = Clock.create_trigger(self.renamed)
        self._trigger_move_to_touch = Clock.create_trigger(self._move_to_touch)
        self._trigger_upd_pawns_here = Clock.create_trigger(
            self._upd_pawns_here
        )
        self._trigger_upd_from_mirror_pos = Clock.create_trigger(
            self.upd_from_mirror_pos
        )
        self._trigger_upd_to_remote_pos = Clock.create_trigger(
            self.upd_to_remote_pos
        )
        kwargs['size_hint'] = (None, None)
        if 'place' in kwargs:
            kwargs['remote'] = kwargs['place']
            del kwargs['place']
        super().__init__(**kwargs)

    def on_remote(self, *args):
        if not super().on_remote(*args):
            return
        self.bind(
            pos=self._trigger_upd_pawns_here
        )
        self.bind(
            pos=self._trigger_upd_to_remote_pos
        )
        return True

    def on_mirror(self, *args):
        if not super().on_mirror(*args):
            if 'name' in self.remote:
                Logger.debug(
                    'Spot: have remote {} but not its mirror'.format(
                        self.remote['name']
                    )
                )
            return
        if (
                '_x' not in self.mirror or '_y' not in self.mirror
        ):
            Clock.schedule_once(self.on_mirror, 0)
            return
        if self.x != self.mirror['_x'] or self.y != self.mirror['_y']:
            self._trigger_upd_from_mirror_pos()
        return True

    def upd_from_mirror_pos(self, *args):
        if not self.mirror:
            Clock.schedule_once(self.upd_from_mirror_pos, 0)
            return
        if self._touchpos:
            return
        self.unbind(
            pos=self._trigger_upd_to_remote_pos
        )
        self.pos = (
            self.mirror['_x'] * self.board.width,
            self.mirror['_y'] * self.board.height
        )
        (x, y) = self.center
        (w, h) = self.size
        rx = w / 2
        ry = h / 2
        self.collider = CollideEllipse(
            x=x, y=y, rx=rx, ry=ry
        )
        self.bind(
            pos=self._trigger_upd_to_remote_pos
        )

    def upd_to_remote_pos(self, *args):
        self.remote['_x'] = self.x / self.board.width
        self.remote['_y'] = self.y / self.board.height

    def renamed(self, *args):
        if not self.board:
            Clock.schedule_once(self.renamed, 0)
            return
        Logger.debug('Spot: renamed to {}'.format(self.name))
        if hasattr(self, '_oldname'):
            del self.board.spot[self._oldname]
        self.board.spot[self.name] = self
        self._oldname = self.name
        self.mirror = {}
        self.remote = self.board.character.place[self.name]

    def _default_pos(self):
        """Return the position on the board to use when I don't have
        one. Given as a pair of floats between 0 and 1.

        """
        # If one spot is without a position, maybe the rest of them
        # are too, and so maybe the board should do a full layout.
        if not hasattr(self, '_unposd'):
            self.board.spots_unposd += 1
            Logger.debug(
                'Spot: {} unpositioned ({} total)'.format(
                    self.remote['name'],
                    self.board.spots_unposd
                )
            )
            self._unposd = True
        return (0.5, 0.5)

    def _default_image_paths(self):
        """Return a list of paths to use for my graphics by default."""
        return ['orb.png']

    def add_widget(self, wid, i=0, canvas=None):
        """Put the widget's canvas in my ``board``'s ``pawnlayout`` rather
        than my own canvas.

        The idea is that all my child widgets are to be instances of
        :class:`Pawn`, and should therefore be drawn after every
        non-:class:`Pawn` widget, so that pawns are on top of spots
        and arrows.

        """
        super().add_widget(wid, i, canvas)
        if not hasattr(wid, 'group'):
            return
        wid._no_use_canvas = True
        mycanvas = (
            self.canvas.after if canvas == 'after' else
            self.canvas.before if canvas == 'before' else
            self.canvas
        )
        pawncanvas = (
            self.board.pawnlayout.canvas.after if canvas == 'after' else
            self.board.pawnlayout.canvas.before if canvas == 'before' else
            self.board.pawnlayout.canvas
        )
        mycanvas.remove(wid.canvas)
        for child in self.children:
            if hasattr(child, 'group'):
                if child.group in pawncanvas.children:
                    pawncanvas.remove(child.group)
                pawncanvas.add(child.group)
            else:
                pawncanvas.add(child.canvas)
        self.pospawn(wid)

    def pospawn(self, pawn):
        """Given some :class:`Pawn` instance that's to be on top of me, set
        its ``pos`` so that it looks like it's on top of me but
        doesn't cover me so much that you can't select me.

        """
        i = 0
        for child in self.children:
            if child is pawn:
                break
            i += 1
        off = i * self.offset
        (x, y) = self.center
        pawn.pos = (x+off, y+off)

    def _upd_pawns_here(self, *args):
        """Move any :class:`Pawn` atop me so it still *is* on top of me,
        presumably after I've moved.

        """
        for pawn in self.children:
            self.pospawn(pawn)

    def collide_point(self, x, y):
        """Check my collider."""
        if not self.collider:
            return False
        return (x, y) in self.collider

    def on_touch_down(self, touch):
        self.unbind(
            pos=self._trigger_upd_to_remote_pos
        )

    def on_touch_move(self, touch):
        """If I'm being dragged, move to follow the touch."""
        if not self.selected:
            return False
        self._touchpos = touch.pos
        self._trigger_move_to_touch()
        return True

    def _move_to_touch(self, *args):
        """Move so I'm centered at my ``touchpos``, and trigger an update of
        my collider.

        """
        if self._touchpos != [] and self.center != self._touchpos:
            self.center = self._touchpos

    def on_touch_up(self, touch):
        """Unset ``touchpos``"""
        self.bind(
            pos=self._trigger_upd_to_remote_pos
        )
        self._touchpos = []
        self._trigger_upd_to_remote_pos()

    def __repr__(self):
        """Give my place's name and my position."""
        return "{}@({},{})".format(self.place.name, self.x, self.y)
