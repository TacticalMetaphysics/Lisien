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
from kivy.uix.layout import Layout
from kivy.clock import Clock, mainthread
from kivy.uix.widget import Widget
from kivy.logger import Logger

from ELiDE.util import trigger
from ELiDE.imagestackproxy import ImageStackProxy


class TextureStackPlane(Widget):
	data = ListProperty()
	selected = ObjectProperty(allownone=True)
	color_selected = ListProperty([0.0, 1.0, 1.0, 1.0])
	default_image_path = "atlas://rltiles/base.atlas/unseen"

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._keys = []
		self._left_xs = np.array([])
		self._right_xs = np.array([])
		self._top_ys = np.array([])
		self._bot_ys = np.array([])
		self._instructions = {}
		self._stack_index = {}
		self._trigger_redraw = Clock.create_trigger(self.redraw)
		self._redraw_bind_uid = self.fbind("data", self._trigger_redraw)

	def on_parent(self, *args):
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

	def on_pos(self, *args):
		if not hasattr(self, "_translate"):
			return
		self._translate.x, self._translate.y = self.pos
		self.canvas.ask_update()

	def on_size(self, *args):
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

	def add_datum(self, datum):
		name = datum["name"]
		if "pos" in datum:
			x, y = datum.pop("pos")
		else:
			x = datum["x"]
			y = datum["y"]
		if isinstance(x, float):
			x *= self.width
		if isinstance(y, float):
			y *= self.height
		left_xs = list(self._left_xs)
		right_xs = list(self._right_xs)
		top_ys = list(self._top_ys)
		bot_ys = list(self._bot_ys)
		left_xs.append(x)
		bot_ys.append(y)
		wide = datum.get("width", 0)
		tall = datum.get("height", 0)
		top_ys.append(y + tall)
		right_xs.append(x + wide)
		self._left_xs = np.array(left_xs)
		self._bot_ys = np.array(bot_ys)
		self._top_ys = np.array(top_ys)
		self._right_xs = np.array(right_xs)
		self._stack_index[name] = len(self._keys)
		self._keys.append(name)
		self.unbind_uid("data", self._redraw_bind_uid)
		self.data.append(datum)
		self._redraw_bind_uid = self.fbind("data", self._trigger_redraw)
		self._add_datum_upd_fbo(**datum)

	@mainthread
	def _remove_upd_fbo(self, name):
		if name not in self._instructions:
			return
		grp = self._instructions[name]["group"]
		grp.clear()
		fbo = self._fbo
		fbo.bind()
		fbo.remove(grp)
		fbo.ask_update()
		fbo.clear_buffer()
		fbo.release()
		self._rectangle.texture = fbo.texture
		del self._instructions[name]

	def remove(self, name_or_idx):
		def delarr(arr, i):
			if i == 0:
				return arr[1:]
			elif i == len(arr) - 1:
				return arr[:-1]
			else:
				return np.concatenate((arr[:i], arr[i + 1 :]))

		if name_or_idx in self._keys:
			idx = self._keys.index(name_or_idx)
			name = name_or_idx
		else:
			idx = name_or_idx
			name = self._keys[idx]
		stack_index = self._stack_index
		del stack_index[name]
		del self._keys[idx]
		for key in self._keys[idx:]:
			stack_index[key] -= 1
		self._left_xs = delarr(self._left_xs, idx)
		self._bot_ys = delarr(self._bot_ys, idx)
		self._top_ys = delarr(self._top_ys, idx)
		self._right_xs = delarr(self._right_xs, idx)
		self.unbind_uid("data", self._redraw_bind_uid)
		del self.data[idx]
		self._redraw_bind_uid = self.fbind("data", self._trigger_redraw)
		self._remove_upd_fbo(name)

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

	@mainthread
	def _redraw_remove_fbo(self, removed_instructions):
		fbo = self._fbo
		for insts in removed_instructions:
			fbo.remove(insts["group"])

	def redraw(self, *args):
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
						texture = Image.load(self.default_image_path).texture
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

		def get_lines_and_colors(datum) -> dict:
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
			right = x + width
			top = y + height
			instructions = {}
			colr = Color(rgba=color_selected)
			instructions["color0"] = colr
			line = Line(points=[x, y, right, y, right, top, x, top, x, y])
			instructions["line"] = line
			coler = Color(rgba=[1, 1, 1, 1])
			instructions["color1"] = coler
			return instructions

		if not hasattr(self, "_rectangle"):
			self._trigger_redraw()
			return
		Logger.debug("TextureStackPlane: redrawing")
		start_ts = monotonic()
		instructions = self._instructions
		stack_index = self._stack_index
		keys = list(self._keys)
		left_xs = list(self._left_xs)
		right_xs = list(self._right_xs)
		top_ys = list(self._top_ys)
		bot_ys = list(self._bot_ys)
		self_width = self.width
		self_height = self.height
		selected = self.selected
		color_selected = self.color_selected
		todo = []
		observed = set()
		for datum in self.data:
			name = datum["name"]
			observed.add(name)
			texs = datum["textures"]
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
			if name in stack_index:
				rects = get_rects(datum)
				if name == selected:
					insts = get_lines_and_colors(datum)
				else:
					insts = {}
				insts["rectangles"] = rects
				if name in instructions:
					insts["group"] = instructions[name]["group"]
				else:
					insts["group"] = InstructionGroup()
				todo.append(insts)
				instructions[name] = insts
				width = datum.get("width", 0)
				height = datum.get("height", 0)
				for texture, rect in zip(texs, rects):
					if isinstance(texture, str):
						try:
							texture = Image.load(
								resource_find(texture)
							).texture
						except Exception:
							texture = Image.load(
								self.default_image_path
							).texture
					w, h = texture.size
					if "width" in datum:
						w = width
					elif w > width:
						width = w
					if "height" in datum:
						h = height
					elif h > height:
						height = h
					rect.texture = texture
					assert w > 0 and h > 0
					rect.size = (w, h)
				idx = stack_index[name]
				right = x + width
				left_xs[idx] = x
				right_xs[idx] = right
				bot_ys[idx] = y
				top = y + height
				top_ys[idx] = top
			else:
				width = datum.get("width", 0)
				height = datum.get("height", 0)
				stack_index[name] = len(keys)
				keys.append(name)
				rects = get_rects(datum)
				grp = InstructionGroup()
				if "width" not in datum or "height" not in datum:
					width, height = rects[0].size
				right = x + width
				left_xs.append(x)
				right_xs.append(right)
				bot_ys.append(y)
				top = y + height
				top_ys.append(top)
				instructions[name] = insts = {
					"rectangles": rects,
					"group": grp,
				}
				if name == selected:
					insts.update(get_lines_and_colors())
				todo.append(insts)
		unobserved = instructions.keys() - observed
		get_rid = []
		for gone in unobserved:
			get_rid.append(instructions.pop(gone))
		self._left_xs = np.array(left_xs)
		self._right_xs = np.array(right_xs)
		self._top_ys = np.array(top_ys)
		self._bot_ys = np.array(bot_ys)
		self._keys = keys
		self._fbo.bind()
		self._fbo.clear_buffer()
		self._fbo.release()
		self._redraw_upd_fbo(todo)
		self._redraw_remove_fbo(get_rid)
		Logger.debug(
			f"TextureStackPlane: redrawn in "
			f"{monotonic() - start_ts:,.2f} seconds"
		)

	def iter_collided_keys(self, x, y):
		hits = (
			(self._left_xs <= x)
			& (self._bot_ys <= y)
			& (y <= self._top_ys)
			& (x <= self._right_xs)
		)
		return map(itemgetter(0), filter(itemgetter(1), zip(self._keys, hits)))


class GraphPawnSpot(ImageStackProxy, Layout):
	"""The kind of ImageStack that represents a :class:`Thing` or
	:class:`Place`.

	"""

	board = ObjectProperty()
	engine = ObjectProperty()
	selected = BooleanProperty(False)
	linecolor = ListProperty()
	selected_outline_color = ListProperty([0, 1, 1, 1])
	unselected_outline_color = ListProperty([0, 0, 0, 0])
	use_boardspace = True

	def __init__(self, **kwargs):
		if "proxy" in kwargs:
			kwargs["name"] = kwargs["proxy"].name
		super().__init__(**kwargs)
		self.bind(pos=self._position)

	def on_touch_move(self, touch):
		"""If I'm being dragged, move to follow the touch."""
		if touch.grab_current is not self:
			return False
		self.center = touch.pos
		return True

	def finalize(self, initial=True):
		"""Call this after you've created all the PawnSpot you need and are ready to add them to the board."""
		if getattr(self, "_finalized", False):
			return
		if self.proxy is None or not hasattr(self.proxy, "name"):
			Clock.schedule_once(partial(self.finalize, initial=initial), 0)
			return
		if initial:
			self.name = self.proxy.name
			if "_image_paths" in self.proxy:
				try:
					self.paths = self.proxy["_image_paths"]
				except Exception as ex:
					if not (
						isinstance(ex.args[0], str)
						and ex.args[0].startswith("Unable to load image type")
					):
						raise ex
					self.paths = self.default_image_paths
			else:
				self.paths = self.proxy.setdefault(
					"_image_paths", self.default_image_paths
				)
			zeroes = [0] * len(self.paths)
			self.offxs = self.proxy.setdefault("_offxs", zeroes)
			self.offys = self.proxy.setdefault("_offys", zeroes)
			self.proxy.connect(self._trigger_pull_from_proxy)
			self.finalize_children(initial=True)
		self._push_image_paths_binding = self.fbind(
			"paths", self._trigger_push_image_paths
		)
		self._push_offxs_binding = self.fbind(
			"offxs", self._trigger_push_offxs
		)
		self._push_offys_binding = self.fbind(
			"offys", self._trigger_push_offys
		)
		self._finalized = True

	def unfinalize(self):
		self.unbind_uid("paths", self._push_image_paths_binding)
		self.unbind_uid("offxs", self._push_offxs_binding)
		self.unbind_uid("offys", self._push_offys_binding)
		self._finalized = False

	def pull_from_proxy(self, *args):
		initial = not hasattr(self, "_finalized")
		self.unfinalize()
		for key, att in [
			("_image_paths", "paths"),
			("_offxs", "offxs"),
			("_offys", "offys"),
		]:
			if key in self.proxy and self.proxy[key] != getattr(self, att):
				setattr(self, att, self.proxy[key])
		self.finalize(initial)

	def _trigger_pull_from_proxy(self, *args, **kwargs):
		Clock.unschedule(self.pull_from_proxy)
		Clock.schedule_once(self.pull_from_proxy, 0)

	@trigger
	def _trigger_push_image_paths(self, *args):
		self.proxy["_image_paths"] = list(self.paths)

	@trigger
	def _trigger_push_offxs(self, *args):
		self.proxy["_offxs"] = list(self.offxs)

	@trigger
	def _trigger_push_offys(self, *args):
		self.proxy["_offys"] = list(self.offys)

	def on_linecolor(self, *args):
		"""If I don't yet have the instructions for drawing the selection box
		in my canvas, put them there. In any case, set the
		:class:`Color` instruction to match my current ``linecolor``.

		"""
		if hasattr(self, "color"):
			self.color.rgba = self.linecolor
			return

		def upd_box_translate(*args):
			self.box_translate.xy = self.pos

		def upd_box_points(*args):
			self.box.points = [
				0,
				0,
				self.width,
				0,
				self.width,
				self.height,
				0,
				self.height,
				0,
				0,
			]

		self.boxgrp = boxgrp = InstructionGroup()
		self.color = Color(*self.linecolor)
		self.box_translate = Translate(*self.pos)
		boxgrp.add(PushMatrix())
		boxgrp.add(self.box_translate)
		boxgrp.add(self.color)
		self.box = Line()
		upd_box_points()
		self.bind(size=upd_box_points, pos=upd_box_translate)
		boxgrp.add(self.box)
		boxgrp.add(Color(1.0, 1.0, 1.0))
		boxgrp.add(PopMatrix())

	def on_board(self, *args):
		if not (hasattr(self, "group") and hasattr(self, "boxgrp")):
			Clock.schedule_once(self.on_board, 0)
			return
		self.canvas.add(self.group)
		self.canvas.add(self.boxgrp)

	def add_widget(self, wid, index=None, canvas=None):
		if index is None:
			for index, child in enumerate(self.children, start=1):
				if wid.priority < child.priority:
					index = len(self.children) - index
					break
		super().add_widget(wid, index=index, canvas=canvas)
		self._trigger_layout()

	def do_layout(self, *args):
		# First try to lay out my children inside of me,
		# leaving at least this much space on the sides
		xpad = self.proxy.get("_xpad", self.width / 4)
		ypad = self.proxy.get("_ypad", self.height / 4)
		self.gutter = gutter = self.proxy.get("_gutter", xpad / 2)
		height = self.height - ypad
		content_height = 0
		too_tall = False
		width = self.width - xpad
		content_width = 0
		groups = defaultdict(list)
		for child in self.children:
			group = child.proxy.get("_group", "")
			groups[group].append(child)
			if child.height > height:
				height = child.height
				too_tall = True
		piles = {}
		# Arrange the groups into piles that will fit in me vertically
		for group, members in groups.items():
			members.sort(key=lambda x: x.width * x.height, reverse=True)
			high = 0
			subgroups = []
			subgroup = []
			for member in members:
				high += member.height
				if high > height:
					subgroups.append(subgroup)
					subgroup = [member]
					high = member.height
				else:
					subgroup.append(member)
			subgroups.append(subgroup)
			content_height = max(
				(content_height, sum(wid.height for wid in subgroups[0]))
			)
			content_width += sum(
				max(wid.width for wid in subgrp) for subgrp in subgroups
			)
			piles[group] = subgroups
		self.content_width = content_width + gutter * (len(piles) - 1)
		too_wide = content_width > width
		# If I'm big enough to fit all this stuff, calculate an offset that will ensure
		# it's all centered. Otherwise just offset to my top-right so the user can still
		# reach me underneath all the pawns.
		if too_wide:
			offx = self.width
		else:
			offx = self.width / 2 - content_width / 2
		if too_tall:
			offy = self.height
		else:
			offy = self.height / 2 - content_height / 2
		for pile, subgroups in sorted(piles.items()):
			for subgroup in subgroups:
				subw = subh = 0
				for member in subgroup:
					rel_y = offy + subh
					member.rel_pos = (offx, rel_y)
					x, y = self.pos
					member.pos = x + offx, y + rel_y
					subw = max((subw, member.width))
					subh += member.height
				offx += subw
			offx += gutter

	def _position(self, *args):
		x, y = self.pos
		for child in self.children:
			offx, offy = getattr(child, "rel_pos", (0, 0))
			child.pos = x + offx, y + offy

	def on_selected(self, *args):
		if self.selected:
			self.linecolor = self.selected_outline_color
		else:
			self.linecolor = self.unselected_outline_color


class Stack:
	__slots__ = ["board", "proxy", "__self__"]

	def __init__(self, **kwargs):
		self.board = kwargs["board"]
		self.proxy = kwargs["proxy"]

	@property
	def paths(self):
		name = self.proxy["name"]
		plane = self._stack_plane
		datum = plane.data[plane._stack_index[name]]
		return datum["textures"]

	@paths.setter
	@mainthread
	def paths(self, v):
		name = self.proxy["name"]
		plane = self._stack_plane
		datum = plane.data[plane._stack_index[name]]
		plane.unbind_uid("data", plane._redraw_bind_uid)
		datum["textures"] = v
		insts = plane._instructions[name]
		rects = insts["rectangles"]
		group = insts["group"]
		for rect in rects:
			group.remove(rect)
		rects = insts["rectangles"] = []
		wide = datum.get("width", 0)
		tall = datum.get("height", 0)
		for path in v:
			if not isinstance(path, str):
				raise TypeError("paths must be strings")
			tex = Image.load(path).texture
			w, h = tex.size
			if "width" not in datum and w > wide:
				wide = w
			if "height" not in datum and h > tall:
				tall = h
			rect = Rectangle(texture=tex, pos=self.pos, size=(wide, tall))
			rects.append(rect)
			group.add(rect)
		plane._redraw_bind_uid = plane.fbind("data", plane._trigger_redraw)

	@property
	def selected(self):
		return self._stack_plane.selected == self.proxy["name"]

	@selected.setter
	@mainthread
	def selected(self, v: bool):
		stack_plane: TextureStackPlane = self._stack_plane
		name = self.proxy["name"]
		insts = stack_plane._instructions[name]
		fbo = stack_plane._fbo
		fbo.bind()
		fbo.clear_buffer()
		if v:
			stack_plane.selected = name
			if "color0" in insts:
				insts["color0"].rgba = stack_plane.color_selected
			else:
				idx = stack_plane._stack_index[name]
				left = stack_plane._left_xs[idx]
				bot = stack_plane._bot_ys[idx]
				right = stack_plane._right_xs[idx]
				top = stack_plane._top_ys[idx]
				grp = insts["group"]
				insts["color0"] = Color(rgba=stack_plane.color_selected)
				grp.add(insts["color0"])
				insts["line"] = Line(
					points=[
						left,
						bot,
						right,
						bot,
						right,
						top,
						left,
						top,
						left,
						bot,
					]
				)
				grp.add(insts["line"])
				insts["color1"] = Color(rgba=[1.0, 1.0, 1.0, 1.0])
				grp.add(insts["color1"])
		else:
			if stack_plane.selected == self.proxy["name"]:
				stack_plane.selected = None
			if "color0" in insts:
				insts["color0"].rgba = [0.0, 0.0, 0.0, 0.0]
		fbo.release()

	@property
	def pos(self):
		stack_plane = self._stack_plane
		idx = stack_plane._stack_index[self.proxy["name"]]
		return float(stack_plane._left_xs[idx]), float(
			stack_plane._bot_ys[idx]
		)

	@pos.setter
	@mainthread
	def pos(self, xy):
		x, y = xy
		stack_plane = self._stack_plane
		stack_plane.unbind_uid("data", stack_plane._redraw_bind_uid)
		name = self.proxy["name"]
		insts = stack_plane._instructions[name]
		idx = stack_plane._stack_index[name]
		left = stack_plane._left_xs[idx]
		bot = stack_plane._bot_ys[idx]
		right = stack_plane._right_xs[idx]
		top = stack_plane._top_ys[idx]
		width = right - left
		height = top - bot
		r = x + width
		t = y + height
		stack_plane._left_xs[idx] = x
		stack_plane._bot_ys[idx] = y
		stack_plane._top_ys[idx] = t
		stack_plane._right_xs[idx] = r
		stack_plane._fbo.bind()
		stack_plane._fbo.clear_buffer()
		stack_plane._fbo.release()
		for rect in insts["rectangles"]:
			rect.pos = xy
		if "line" in insts:
			insts["line"].points = [x, y, r, y, r, t, x, t, x, y]
		stack_plane.data[idx]["pos"] = xy
		stack_plane._redraw_bind_uid = stack_plane.fbind(
			"data", stack_plane._trigger_redraw
		)

	@property
	def _stack_plane(self):
		return self.board.stack_plane

	@property
	def x(self):
		stack_plane = self._stack_plane
		idx = stack_plane._stack_index[self.proxy["name"]]
		return float(stack_plane._left_xs[idx])

	@x.setter
	def x(self, x):
		self.pos = x, self.y

	@property
	def y(self):
		stack_plane = self._stack_plane
		idx = stack_plane._stack_index[self.proxy["name"]]
		return float(stack_plane._bot_ys[idx])

	@y.setter
	def y(self, y):
		self.pos = self.x, y

	@property
	def _stack_plane(self):
		return self.board.stack_plane

	@property
	def size(self):
		stack_plane = self._stack_plane
		name = self.proxy["name"]
		idx = stack_plane._stack_index[name]
		left = stack_plane._left_xs[idx]
		bot = stack_plane._bot_ys[idx]
		right = stack_plane._right_xs[idx]
		top = stack_plane._top_ys[idx]
		return float(right - left), float(top - bot)

	@size.setter
	def size(self, wh):
		w, h = wh
		stack_plane = self._stack_plane
		stack_plane.unbind_uid("data", stack_plane._redraw_bind_uid)
		name = self.proxy["name"]
		insts = stack_plane._instructions[name]
		idx = stack_plane._stack_index[name]
		x = stack_plane._left_xs[idx]
		y = stack_plane._bot_ys[idx]
		r = stack_plane._right_xs[idx] = x + w
		t = stack_plane._top_ys[idx] = y + h
		for rect in insts["rectangles"]:
			rect.size = wh
		if "line" in insts:
			insts["line"].points = [x, y, r, y, r, t, x, t, x, y]
		stack_plane.data[idx]["size"] = wh
		stack_plane._redraw_bind_uid = stack_plane.fbind(
			"data", stack_plane._trigger_redraw
		)

	@property
	def width(self):
		stack_plane = self.board.stack_plane
		name = self.proxy["name"]
		idx = stack_plane._stack_index[name]
		left = stack_plane._left_xs[idx]
		right = stack_plane._right_xs[idx]
		return float(right - left)

	@width.setter
	def width(self, w):
		self.size = self.height, w

	@property
	def height(self):
		stack_plane = self.board.stack_plane
		name = self.proxy["name"]
		idx = stack_plane._stack_index[name]
		top = stack_plane._top_ys[idx]
		bot = stack_plane._bot_ys[idx]
		return float(top - bot)

	@height.setter
	def height(self, h):
		self.size = self.width, h

	@property
	def center(self):
		stack_plane = self.board.stack_plane
		name = self.proxy["name"]
		idx = stack_plane._stack_index[name]
		x = stack_plane._left_xs[idx]
		y = stack_plane._bot_ys[idx]
		r = stack_plane._right_xs[idx]
		t = stack_plane._top_ys[idx]
		w = r - x
		h = t - y
		return float(x + w / 2), float(y + h / 2)

	@center.setter
	def center(self, c):
		stack_plane = self.board.stack_plane
		name = self.proxy["name"]
		idx = stack_plane._stack_index[name]
		x = stack_plane._left_xs[idx]
		y = stack_plane._bot_ys[idx]
		r = stack_plane._right_xs[idx]
		t = stack_plane._top_ys[idx]
		w = r - x
		h = t - y
		self.pos = c[0] - w / 2, c[1] - h / 2

	@property
	def center_x(self):
		stack_plane = self.board.stack_plane
		name = self.proxy["name"]
		idx = stack_plane._stack_index[name]
		x = stack_plane._left_xs[idx]
		r = stack_plane._right_xs[idx]
		w = r - x
		return float(x + w / 2)

	@center_x.setter
	def center_x(self, cx):
		stack_plane = self.board.stack_plane
		name = self.proxy["name"]
		idx = stack_plane._stack_index[name]
		x = stack_plane._left_xs[idx]
		r = stack_plane._right_xs[idx]
		w = r - x
		self.pos = cx - w / 2, self.y

	@property
	def center_y(self):
		stack_plane = self.board.stack_plane
		name = self.proxy["name"]
		idx = stack_plane._stack_index[name]
		y = stack_plane._bot_ys[idx]
		t = stack_plane._top_ys[idx]
		h = t - y
		return float(y + h / 2)

	@center_y.setter
	def center_y(self, cy):
		stack_plane = self.board.stack_plane
		name = self.proxy["name"]
		idx = stack_plane._stack_index[name]
		y = stack_plane._bot_ys[idx]
		t = stack_plane._top_ys[idx]
		h = t - y
		self.pos = self.x, cy - h / 2

	@property
	def top(self):
		stack_plane = self.board.stack_plane
		name = self.proxy["name"]
		idx = stack_plane._stack_index[name]
		return float(stack_plane._top_ys[idx])

	@top.setter
	def top(self, t):
		stack_plane = self.board.stack_plane
		name = self.proxy["name"]
		idx = stack_plane._stack_index[name]
		y = stack_plane._bot_ys[idx]
		stack_plane._top_ys[idx] = t
		h = t - y
		stack_plane._bot_ys[idx] = t - h
		self.pos = self.x, t - h

	@property
	def right(self):
		stack_plane = self.board.stack_plane
		name = self.proxy["name"]
		idx = stack_plane._stack_index[name]
		return float(stack_plane._right_xs[idx])

	@right.setter
	def right(self, r):
		stack_plane = self.board.stack_plane
		name = self.proxy["name"]
		idx = stack_plane._stack_index[name]
		x = stack_plane._left_xs[idx]
		stack_plane._right_xs[idx] = r
		w = r - x
		x = stack_plane._left_xs[idx] = r - w
		self.pos = x, self.y

	@property
	def name(self):
		return self.proxy["name"]

	def collide_point(self, x, y):
		pos = self.pos
		if x < pos[0] or y < pos[1]:
			return False
		w, h = self.size
		return x < pos[0] + w and y < pos[1] + h
