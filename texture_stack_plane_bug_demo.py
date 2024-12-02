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
"""Code that draws the box around a Pawn or Spot when it's selected"""

from collections import defaultdict
from functools import partial
from operator import itemgetter
from time import monotonic

import numpy as np
from kivy.core.image import Image
from kivy.graphics.fbo import Fbo
from kivy.properties import ObjectProperty, BooleanProperty, ListProperty
from kivy.graphics import (
	InstructionGroup,
	Translate,
	PopMatrix,
	PushMatrix,
	Color,
	Line,
	Rectangle,
)
from kivy.resources import resource_find
from kivy.clock import Clock, mainthread
from kivy.uix.widget import Widget
from kivy.logger import Logger


BACKGROUND = True  # change this to False, and the bug doesn't happen!
DEFAULT_IMAGE_PATH = "examples/canvas/mtexture2.png"
TEST_DATA = [{"name": (x, y), "x": x * 0.04083333333333333, "y": y * 0.04083333333333333, "width": 32, "height": 32, "textures": [DEFAULT_IMAGE_PATH]} for x in range(100) for y in range(100)]


class TextureStackPlane(Widget):
	data = ListProperty()

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._trigger_redraw = Clock.create_trigger(self.redraw)
		self._redraw_bind_uid = self.fbind("data", self._trigger_redraw)

	def on_parent(self, *_):
		if not self.canvas:
			Clock.schedule_once(self.on_parent, 0)
			return
		with self.canvas:
			self._fbo = Fbo(size=self.size)
			self._translate = Translate(x=self.x, y=self.y)
			self._rectangle = Rectangle(
				size=self.size, texture=self._fbo.texture
			)
		self.bind(pos=self._trigger_redraw, size=self._trigger_redraw)
		self._trigger_redraw()

	def on_pos(self, *_):
		if not hasattr(self, "_translate"):
			return
		self._translate.x, self._translate.y = self.pos
		self.canvas.ask_update()

	def on_size(self, *_):
		if not hasattr(self, "_rectangle") or not hasattr(self, "_fbo"):
			return
		self._rectangle.size = self._fbo.size = self.size
		self.redraw()

	@mainthread
	def _add_datum_upd_fbo(self, **datum):
		name = datum["name"]
		texs = datum["textures"]
		x = datum["x"]
		y = datum["y"]
		fbo = self._fbo
		with fbo:
			instructions = self._instructions
			rects = []
			wide = datum.get("width", 0)
			tall = datum.get("height", 0)
			for tex in texs:
				if isinstance(tex, str):
					tex = Image.load(resource_find(tex)).texture
					w, h = tex.size
					if "width" not in datum and w > wide:
						wide = w
					if "height" not in datum and h > tall:
						tall = h
				rects.append(
					Rectangle(texture=tex, pos=(x, y), size=(wide, tall))
				)
			instructions[name] = {
				"rectangles": rects,
				"group": InstructionGroup(),
			}
			grp = instructions[name]["group"]
			for rect in rects:
				grp.add(rect)
			fbo.add(instructions[name]["group"])

	@mainthread
	def _redraw_upd_fbo(self, changed_instructions):
		fbo = self._fbo
		for insts in changed_instructions:
			group = insts["group"]
			group.clear()
			for rect in insts["rectangles"]:
				group.add(rect)
			if "color0" in insts:
				group.add(insts["color0"])
				group.add(insts["line"])
				group.add(insts["color1"])
			if group not in fbo.children:
				fbo.add(group)
		self._rectangle.texture = fbo.texture

	def redraw(self, *_):
		def get_rects(datum):
			width = datum.get("width", 0)
			height = datum.get("height", 0)
			if isinstance(datum["x"], float):
				x = datum["x"] * self_width
			else:
				if not isinstance(datum["x"], int):
					raise TypeError("need int or float for pos")
				x = datum["x"]
			if isinstance(datum["y"], float):
				y = datum["y"] * self_height
			else:
				if not isinstance(datum["y"], int):
					raise TypeError("need int or float for pos")
				y = datum["y"]
			rects = []
			for texture in datum["textures"]:
				if isinstance(texture, str):
					try:
						texture = Image.load(resource_find(texture)).texture
					except Exception:
						texture = Image.load(DEFAULT_IMAGE_PATH).texture
				w, h = texture.size
				if "width" in datum:
					w = width
				elif w > width:
					width = w
				if "height" in datum:
					h = height
				elif h > height:
					height = h
				assert w > 0 and h > 0
				rects.append(
					Rectangle(pos=(x, y), size=(w, h), texture=texture)
				)
			return rects

		if not hasattr(self, "_rectangle"):
			self._trigger_redraw()
			return
		start_ts = monotonic()
		instructions = {}
		self_width = self.width
		self_height = self.height
		todo = []
		for datum in self.data:
			name = datum["name"]
			rects = get_rects(datum)
			grp = InstructionGroup()
			instructions[name] = insts = {
				"rectangles": rects,
				"group": grp,
			}
			todo.append(insts)
		self._fbo.bind()
		self._fbo.clear_buffer()
		self._fbo.release()
		self._redraw_upd_fbo(todo)
		Logger.debug(
			f"TextureStackPlane: redrawn in "
			f"{monotonic() - start_ts:,.2f} seconds"
		)


# The following is a demonstration of a graphical error involving TextureStackPlane
# and its interaction with StencilView.
# When BACKGROUND is True, TextureStackPlane overflows the StencilView it's in,
# even though the background image doesn't.
# When BACKGROUND is False, TextureStackPlane obeys the StencilView.
# The bug doesn't seem to come up when the graphics within TextureStackPlane
# have whole-number coordinates.

if __name__ == "__main__":
	import os
	import json
	from kivy.uix.boxlayout import BoxLayout
	from kivy.uix.floatlayout import FloatLayout
	from kivy.uix.image import Image as ImageWidget
	from kivy.uix.widget import Widget
	from kivy.uix.scatterlayout import ScatterPlaneLayout
	from kivy.uix.stencilview import StencilView
	from kivy.base import runTouchApp

	root = BoxLayout()
	root.add_widget(Widget())
	texstac = TextureStackPlane(data=TEST_DATA, size_hint=(None, None), size=(1024,1024))
	flot = FloatLayout()
	if BACKGROUND:
		texstacbg = ImageWidget(size_hint=(None, None), size=(1024,1024))
		flot.add_widget(texstacbg)
	flot.add_widget(texstac)
	splane = ScatterPlaneLayout()
	splane.add_widget(flot)
	stenc = StencilView()
	stenc.add_widget(splane)
	root.add_widget(stenc)
	root.add_widget(Widget())
	runTouchApp(root)
