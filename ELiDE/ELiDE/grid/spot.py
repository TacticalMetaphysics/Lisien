from kivy.properties import (
    ObjectProperty,
    BooleanProperty,
    ListProperty,
    DictProperty
)

from ..imagestackproxy import ImageStackProxy


class Spot(ImageStackProxy):
    board = ObjectProperty()