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
"""Widget to display the contents of a :class:`kivy.atlas.Atlas` in
one :class:`kivy.uix.togglebutton.ToggleButton` apiece, arranged in a
:class:`kivy.uix.stacklayout.StackLayout`. The user selects graphics
from the :class:`Pallet`, and the :class:`Pallet` updates its
``selection`` list to show what the user selected."""

from kivy.atlas import Atlas
from kivy.clock import Clock, mainthread, triggered
from kivy.graphics import Rectangle
from kivy.logger import Logger
from kivy.properties import (
	DictProperty,
	ListProperty,
	NumericProperty,
	ObjectProperty,
	OptionProperty,
	ReferenceListProperty,
	StringProperty,
)
from kivy.resources import resource_find
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.stacklayout import StackLayout
from kivy.uix.togglebutton import ToggleButton

from .util import load_string_once


def trigger(func):
	return triggered()(func)


class SwatchButton(ToggleButton):
	"""Toggle button containing a texture and its name, which, when
	toggled, will report the fact to the :class:`Pallet` it's in.

	"""

	tex = ObjectProperty()
	"""Texture to display here"""

	def on_state(self, *_):
		if self.state == "down":
			assert self not in self.parent.selection
			if self.parent.selection_mode == "single":
				for wid in self.parent.selection:
					if wid is not self:
						wid.state = "normal"
				self.parent.selection = [self]
			else:
				self.parent.selection.append(self)
		else:
			if self in self.parent.selection:
				self.parent.selection.remove(self)


class Pallet(StackLayout):
	"""Many :class:`SwatchButton`, gathered from an :class:`kivy.atlas.Atlas`."""

	atlas = ObjectProperty()
	""":class:`kivy.atlas.Atlas` object I'll make :class:`SwatchButton` from."""
	filename = StringProperty()
	"""Path to an atlas; will construct :class:`kivy.atlas.Atlas` when set"""
	swatches = DictProperty({})
	""":class:`SwatchButton` widgets here, keyed by name of their graphic"""
	swatch_width = NumericProperty(100)
	"""Width of each and every :class:`SwatchButton` here"""
	swatch_height = NumericProperty(75)
	"""Height of each and every :class:`SwatchButton` here"""
	swatch_size = ReferenceListProperty(swatch_width, swatch_height)
	"""Size of each and every :class:`SwatchButton` here"""
	selection = ListProperty([])
	"""List of :class:`SwatchButton`s that are selected"""
	selection_mode = OptionProperty("single", options=["single", "multiple"])
	"""Whether to allow only a 'single' selected :class:`SwatchButton` (default), or 'multiple'"""

	def __init__(self, **kwargs):
		self._trigger_upd_textures = Clock.create_trigger(self.upd_textures)
		super().__init__(**kwargs)

	def on_selection(self, *_):
		Logger.debug(
			"Pallet: {} got selection {}".format(self.filename, self.selection)
		)

	def on_filename(self, *_):
		if not self.filename:
			return
		resource = resource_find(self.filename)
		if not resource:
			raise ValueError("Couldn't find atlas: {}".format(self.filename))
		self.atlas = Atlas(resource)

	def on_atlas(self, *_):
		if self.atlas is None:
			return
		self.upd_textures()
		self.atlas.bind(textures=self._trigger_upd_textures)

	def upd_textures(self, *_):
		"""Create one :class:`SwatchButton` for each texture"""
		if self.canvas is None:
			Clock.schedule_once(self.upd_textures, 0)
			return
		swatches = self.swatches
		atlas_textures = self.atlas.textures
		remove_widget = self.remove_widget
		add_widget = self.add_widget
		swatch_size = self.swatch_size
		for name, swatch in list(swatches.items()):
			if name not in atlas_textures:
				remove_widget(swatch)
				del swatches[name]
		for name, tex in atlas_textures.items():
			if name in swatches and swatches[name] != tex:
				remove_widget(swatches[name])
			if name not in swatches or swatches[name] != tex:
				swatches[name] = SwatchButton(
					text=name,
					tex=tex,
					size_hint=(None, None),
					size=swatch_size,
				)
				add_widget(swatches[name])


load_string_once("""
<SwatchButton>:
	canvas:
		Rectangle:
			pos:
				(
				root.x + (root.width / 2 - root.tex.size[0] / 2) if root.tex else 0,
				root.y + root.height - root.tex.size[1] if root.tex else 0
				)
			size: root.tex.size
			texture: root.tex
<Pallet>:
	orientation: 'lr-tb'
	padding_y: 100
	size_hint: (None, None)
	height: self.minimum_height
""")


class PalletBox(BoxLayout):
	pallets = ListProperty()
