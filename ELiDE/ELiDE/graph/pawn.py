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
    NumericProperty,
    ReferenceListProperty
)
from .pawnspot import GraphPawnSpot
from ..pawn import PawnBehavior


class Pawn(GraphPawnSpot, PawnBehavior):
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
    def _get_location_wid(self):
        return self.board.spot[self.loc_name]

    def __repr__(self):
        """Give my ``thing``'s name and its location's name."""
        return '<{}-in-{} at {}>'.format(
            self.name,
            self.loc_name,
            id(self)
        )
