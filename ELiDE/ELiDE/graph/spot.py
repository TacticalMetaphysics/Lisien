# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Widget to represent :class:`Place`s. :class:`Pawn` moves around on
top of these.

"""
from kivy.clock import Clock

from kivy.properties import (
    AliasProperty,
    ObjectProperty,
    NumericProperty
)
from .pawnspot import GraphPawnSpot
from ..util import trigger


class GraphSpot(GraphPawnSpot):
    """The icon that represents a :class:`Place`.

    Each :class:`Spot` is located on the Board that represents the
    :class:`Character` that the underlying :class:`Place` is in. Its
    coordinates are relative to its :class:`Board`, not necessarily
    the window the :class:`Board` is in.

    """
    default_image_paths = ['atlas://rltiles/floor.atlas/floor-stone']
    default_pos = (0.5, 0.5)

    def __init__(self, **kwargs):
        """Deal with triggers and bindings, and arrange to take care of
        changes in game-time.

        """
        self._pospawn_partials = {}
        self._pospawn_triggers = {}
        kwargs['size_hint'] = (None, None)
        if 'place' in kwargs:
            kwargs['proxy'] = kwargs['place']
            del kwargs['place']
        super().__init__(**kwargs)

    def on_board(self, *args):
        super().on_board(*args)
        self.board.bind(size=self._upd_pos)

    def _upd_pos(self, *args):
        if self.board is None:
            Clock.schedule_once(self._upd_pos, 0)
            return
        self.pos = (
            int(self.proxy.get('_x', self.default_pos[0]) * self.board.width),
            int(self.proxy.get('_y', self.default_pos[1]) * self.board.height)
        )

    def finalize(self, initial=True):
        if initial:
            self._upd_pos()
        super().finalize(initial)

    def push_pos(self, *args):
        """Set my current position, expressed as proportions of the graph's
        width and height, into the ``_x`` and ``_y`` keys of the
        entity in my ``proxy`` property, such that it will be
        recorded in the database.

        """
        self.proxy['_x'] = self.x / self.board.width
        self.proxy['_y'] = self.y / self.board.height
    _trigger_push_pos = trigger(push_pos)

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return False
        self.center = touch.pos
        self._trigger_push_pos()
        touch.ungrab(self)
        self._trigger_push_pos()
        return True

    def __repr__(self):
        """Give my name and position."""
        return "<{}@({},{}) at {}>".format(self.name, self.x, self.y, id(self))
