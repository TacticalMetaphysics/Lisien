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
from functools import partial

from kivy.properties import NumericProperty, OptionProperty
from kivy.uix.boxlayout import BoxLayout
from networkx import grid_2d_graph

from lisien.character import grid_2d_8graph

from .util import logwrap


class GridGeneratorDialog(BoxLayout):
	xval = NumericProperty()
	yval = NumericProperty()
	directions = OptionProperty(None, options=[None, 4, 8])

	@partial(logwrap, section="GridGeneratorDialog")
	def generate(self, engine):
		x = int(self.xval)
		y = int(self.yval)
		if x < 1 or y < 1:
			return False
		elif self.directions == 4:
			# instead, we're running just after game init, before the view is open on it, and we'll make a character ourselves
			engine.add_character("physical", grid_2d_graph(x, y))
			return True
		elif self.directions == 8:
			engine.add_character("physical", grid_2d_8graph(x, y))
			return True
		else:
			return False

	@logwrap(section="GridGeneratorDialog")
	def validate(self):
		return self.directions and int(self.xval) and int(self.yval)
