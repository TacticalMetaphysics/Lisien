from kivy.properties import (
    ObjectProperty,
    BooleanProperty,
    ListProperty,
    DictProperty
)

from ..imagestackproxy import ImageStackProxy


class GridSpot(ImageStackProxy):
    board = ObjectProperty()