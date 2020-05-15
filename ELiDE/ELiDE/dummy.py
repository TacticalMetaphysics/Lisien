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
from kivy.properties import (
    NumericProperty,
    ObjectProperty,
    ReferenceListProperty,
    StringProperty
)
from kivy.lang import Builder
from kivy.logger import Logger

from .kivygarden.texturestack import ImageStack

from . import menu  # for kv


class Dummy(ImageStack):
    """A widget that looks like the ones on the graph, which, when dragged
    onto the graph, creates one of them.

    """
    _touch = ObjectProperty(None, allownone=True)
    name = StringProperty()
    prefix = StringProperty()
    num = NumericProperty()
    x_start = NumericProperty(0)
    y_start = NumericProperty(0)
    pos_start = ReferenceListProperty(x_start, y_start)
    x_down = NumericProperty(0)
    y_down = NumericProperty(0)
    pos_down = ReferenceListProperty(x_down, y_down)
    x_up = NumericProperty(0)
    y_up = NumericProperty(0)
    pos_up = ReferenceListProperty(x_up, y_up)
    x_center_up = NumericProperty(0)
    y_center_up = NumericProperty(0)
    center_up = ReferenceListProperty(x_center_up, y_center_up)
    right_up = NumericProperty(0)
    top_up = NumericProperty(0)

    def on_paths(self, *args, **kwargs):
        super().on_paths(*args, **kwargs)
        Logger.debug("Dummy: {} got paths {}".format(self.name, self.paths))

    def on_touch_down(self, touch):
        """If hit, record my starting position, that I may return to it in
        ``on_touch_up`` after creating a real :class:`graph.Spot` or
        :class:`graph.Pawn` instance.

        """
        if not self.collide_point(*touch.pos):
            return False
        self.pos_start = self.pos
        self.pos_down = (
            self.x - touch.x,
            self.y - touch.y
        )
        touch.grab(self)
        self._touch = touch
        return True

    def on_touch_move(self, touch):
        """Follow the touch"""
        if touch is not self._touch:
            return False
        self.pos = (
            touch.x + self.x_down,
            touch.y + self.y_down
        )
        return True

    def on_touch_up(self, touch):
        """Return to ``pos_start``, but first, save my current ``pos`` into
        ``pos_up``, so that the layout knows where to put the real
        :class:`graph.Spot` or :class:`graph.Pawn` instance.

        """
        if touch is not self._touch:
            return False
        self.pos_up = self.pos
        self.pos = self.pos_start
        self._touch = None
        return True


kv = """
<Dummy>:
    name: "".join((self.prefix, str(self.num)))
    x_center_up: self.x_up + self.width / 2
    y_center_up: self.y_up + self.height / 2
    right_up: self.x_up + self.width
    top_up: self.y_up + self.height
"""
Builder.load_string(kv)
