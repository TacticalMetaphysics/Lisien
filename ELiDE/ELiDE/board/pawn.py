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
"""Widget representing things that move about from place to place."""
from kivy.properties import (
    AliasProperty,
    ObjectProperty,
    NumericProperty
)
from .pawnspot import PawnSpot
from ..util import trigger


class Pawn(PawnSpot):
    """A token to represent a :class:`Thing`.

    :class:`Thing` is the LiSE class to represent items that are
    located in some :class:`Place` or other. Accordingly,
    :class:`Pawn`'s coordinates are never set directly; they are
    instead derived from the location of the :class:`Thing`
    represented. That means a :class:`Pawn` will appear next to the
    :class:`Spot` representing the :class:`Place` that its
    :class:`Thing` is in. The exception is if the :class:`Thing` is
    currently moving from its current :class:`Place` to another one,
    in which case the :class:`Pawn` will appear some distance along
    the :class:`Arrow` that represents the :class:`Portal` it's moving
    through.

    """
    loc_name = ObjectProperty()
    thing = AliasProperty(
        lambda self: self.proxy,
        lambda self, v: self.proxy.setter()(v),
        bind=('proxy',)
    )
    default_image_paths = ['atlas://rltiles/base.atlas/unseen']
    priority = NumericProperty()

    def __init__(self, **kwargs):
        if 'thing' in kwargs:
            kwargs['proxy'] = kwargs['thing']
            del kwargs['thing']
        if 'proxy' in kwargs:
            kwargs['loc_name'] = kwargs['proxy']['location']
        super().__init__(**kwargs)
        self.register_event_type('on_drop')

    def on_parent(self, *args):
        if self.parent:
            self.board = self.parent.board
            self.bind(
                loc_name=self._trigger_relocate
            )
            if self.proxy:
                self._trigger_relocate()

    def relocate(self, *args):
        if not self.proxy.exists:
            return
        location = self.board.spot[self.loc_name]
        if location != self.parent:
            if self.parent:
                self.parent.remove_widget(self)
            location.add_widget(self)
    _trigger_relocate = trigger(relocate)

    def finalize(self, initial=True):
        if initial:
            self.loc_name = self.proxy['location']
            self.priority = self.proxy.get('_priority', 0.0)
        self.bind(
            loc_name=self._trigger_push_location
        )
        super().finalize(initial)

    def unfinalize(self):
        self.unbind(
            loc_name=self._trigger_push_location
        )
        super().unfinalize()

    def pull_from_proxy(self, *args):
        super().pull_from_proxy(*args)
        relocate = False
        if self.loc_name != self.proxy['location']:
            self.loc_name = self.proxy['location']  # aliasing? could be trouble
            relocate = True
        if '_priority' in self.proxy:
            self.priority = self.proxy['_priority']
        if relocate:
            self.relocate()

    def on_priority(self, *args):
        if self.proxy['_priority'] != self.priority:
            self.proxy['_priority'] = self.priority
        self.parent.restack()

    def push_location(self, *args):
        if self.proxy['location'] != self.loc_name:
            self.proxy['location'] = self.loc_name
    _trigger_push_location = trigger(push_location)

    def on_touch_up(self, touch):
        if touch.grab_current is not self:
            return False
        for spot in self.board.spot.values():
            if self.collide_widget(spot) and spot.name != self.loc_name:
                new_spot = spot
                break
        else:
            new_spot = None

        self.dispatch('on_drop', new_spot)
        touch.ungrab(self)
        return True

    def on_drop(self, spot):
        parent = self.parent
        if spot:
            self.loc_name = self.proxy['location'] = spot.name
            parent.remove_widget(self)
            spot.add_widget(self)
        else:
            x, y = parent.positions[self.uid]
            self.pos = parent.x + x, parent.y + y

    def __repr__(self):
        """Give my ``thing``'s name and its location's name."""
        return '<{}-in-{} at {}>'.format(
            self.name,
            self.loc_name,
            id(self)
        )
