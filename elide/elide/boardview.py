# This file is part of Elide, frontend to Lisien, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from kivy.properties import NumericProperty, ObjectProperty
from kivy.uix.stencilview import StencilView


class BoardView(StencilView):
	board = ObjectProperty()
	plane = ObjectProperty()
	scale_min = NumericProperty(allownone=True)
	scale_max = NumericProperty(allownone=True)

	def spot_from_dummy(self, dummy):
		self.plane.spot_from_dummy(dummy)

	def pawn_from_dummy(self, dummy):
		self.plane.pawn_from_dummy(dummy)

	def on_touch_down(self, touch):
		if not self.collide_point(*touch.pos):
			return
		return super().on_touch_down(touch)
