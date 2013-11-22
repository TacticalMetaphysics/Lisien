# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from kivy.properties import (
    ListProperty,
    NumericProperty,
    ObjectProperty
)


class ItemLayout(object):
    character = ObjectProperty()
    item_type = NumericProperty()
    keys = ListProperty()
    style = ObjectProperty()
    edbut = ObjectProperty()
