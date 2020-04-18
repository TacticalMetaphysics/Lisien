from kivy.properties import (
    ObjectProperty,
    BooleanProperty,
    ListProperty,
    DictProperty
)

from ..imagestackproxy import ImageStackProxy
from ..pawn import PawnBehavior


class GridPawn(ImageStackProxy, PawnBehavior):
    board = ObjectProperty()

    def __repr__(self):
        """Give my ``thing``'s name and its location's name."""
        return '<{}-in-{} at {}>'.format(
            self.name,
            self.loc_name,
            id(self)
        )
