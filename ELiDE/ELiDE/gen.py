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
from kivy.properties import NumericProperty, ObjectProperty, OptionProperty
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from .util import trigger
from LiSE.character import grid_2d_8graph
from networkx import grid_2d_graph


class GridGeneratorDialog(BoxLayout):
	xval = NumericProperty()
	yval = NumericProperty()
	directions = OptionProperty(None, options=[None, 4, 8])

	def generate(self, engine):
		x = int(self.xval)
		y = int(self.yval)
		if x < 1 or y < 1:
			return False
		elif self.directions == 4:
			# instead, we're running just after game init, before the view is open on it, and we'll make a character ourselves
			engine.add_character('physical', grid_2d_graph(x, y))
			return True
		elif self.directions == 8:
			engine.add_character('physical', grid_2d_8graph(x, y))
			return True
		else:
			return False

	def validate(self):
		return self.directions and int(self.xval) and int(self.yval)


Builder.load_string("""
<GridGeneratorDialog>:
    directions: 4 if but4.state == 'down' else 8
    orientation: 'vertical'
    BoxLayout:
        orientation: 'horizontal'
        MenuIntInput:
            id: input_x
            hint_text: str(root.xval) if root.xval else 'x'
            set_value: root.setter('xval')
        Label:
            text: 'x'
            size_hint_x: 0.1
        MenuIntInput:
            id: input_y
            hint_text: str(root.yval) if root.yval else 'y'
            set_value: root.setter('yval')
    BoxLayout:
        ToggleButton:
            id: but4
            group: 'dir'
            text: '4-way'
            state: 'down'
        ToggleButton:
            id: but8
            group: 'dir'
            text: '8-way'""")
