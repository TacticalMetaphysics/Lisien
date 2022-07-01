# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
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
from kivy.clock import Clock
from kivy.properties import NumericProperty, ObjectProperty
from kivy.uix.stencilview import StencilView


class BoardView(StencilView):
	board = ObjectProperty()
	plane = ObjectProperty()
	scale_min = NumericProperty(allownone=True)
	scale_max = NumericProperty(allownone=True)

	def on_pos(self, *args):
		if self.board and self.children:
			self.children[0].pos = self.pos
		else:
			Clock.schedule_once(self.on_pos, 0.001)

	def on_size(self, *args):
		if self.board and self.children:
			self.children[0].size = self.size
		else:
			Clock.schedule_once(self.on_size, 0.001)

	def on_parent(self, *args):
		if self.board and self.children:
			self.children[0].pos = self.pos
			self.children[0].size = self.size
		else:
			Clock.schedule_once(self.on_parent, 0.001)

	def spot_from_dummy(self, dummy):
		self.plane.spot_from_dummy(dummy)

	def pawn_from_dummy(self, dummy):
		self.plane.pawn_from_dummy(dummy)
