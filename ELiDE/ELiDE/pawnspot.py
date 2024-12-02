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
from kivy.clock import Clock, mainthread, triggered
from kivy.uix.widget import Widget
from kivy.logger import Logger

from ELiDE.imagestackproxy import ImageStackProxy


def trigger(func):
	return triggered()(func)


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
		Logger.debug(f"TextureStackPlane: redrawing, with {self.selected} selected")
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
					insts.update(get_lines_and_colors(datum))
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
			"offys",
			self._trigger_push_offys
		)

		def upd_box_translate(*_):
			self.box_translate.xy = self.pos

		def upd_box_points(*_):
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
		self._upd_box_points_binding = self.fbind("size", upd_box_points)
		self._upd_box_translate_binding = self.fbind("pos", upd_box_translate)
		boxgrp.add(self.box)
		boxgrp.add(Color(1.0, 1.0, 1.0))
		boxgrp.add(PopMatrix())
		self._finalized = True

	def unfinalize(self):
		self.unbind_uid("paths", self._push_image_paths_binding)
		self.unbind_uid("offxs", self._push_offxs_binding)
		self.unbind_uid("offys", self._push_offys_binding)
		self._finalized = False

	def pull_from_proxy(self, *_):
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

	def _trigger_pull_from_proxy(self, *_, **__):
		Clock.unschedule(self.pull_from_proxy)
		Clock.schedule_once(self.pull_from_proxy, 0)

	@trigger
	def _trigger_push_image_paths(self, *_):
		self.proxy["_image_paths"] = list(self.paths)

	@trigger
	def _trigger_push_offxs(self, *_):
		self.proxy["_offxs"] = list(self.offxs)

	@trigger
	def _trigger_push_offys(self, *_):
		self.proxy["_offys"] = list(self.offys)

	def on_linecolor(self, *_):
		"""If I don't yet have the instructions for drawing the selection box
		in my canvas, put them there. In any case, set the
		:class:`Color` instruction to match my current ``linecolor``.

		"""
		if hasattr(self, "color"):
			self.color.rgba = self.linecolor

	def on_board(self, *_):
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

	def do_layout(self, *_):
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

	def _position(self, *_):
		x, y = self.pos
		for child in self.children:
			offx, offy = getattr(child, "rel_pos", (0, 0))
			child.pos = x + offx, y + offy

	def on_selected(self, *_):
		if self.selected:
			self.linecolor = self.selected_outline_color
		else:
			self.linecolor = self.unselected_outline_color


class Stack:
	__slots__ = ["board", "proxy", "__self__"]

	default_image_paths = ["atlas://rltiles/floor.atlas/floor-normal"]

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
		if v is None:
			v = self.default_image_paths
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
		Logger.debug(f"Stack: {name} selected")
		insts = stack_plane._instructions[name]
		fbo = stack_plane._fbo
		fbo.bind()
		fbo.clear_buffer()
		if v:
			stack_plane.selected = name
			if "color0" in insts:
				Logger.debug(f"Stack: changing {name}'s color to {stack_plane.color_selected}")
				insts["color0"].rgba = stack_plane.color_selected
			else:
				Logger.debug(f"Stack: creating Color(rgba={stack_plane.color_selected}) for {name}")
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
		for rect in insts["rectangles"]:
			rect: Rectangle
			rect.pos = xy
			rect.flag_update()  # undocumented. sounds right?
		if "line" in insts:
			insts["line"].points = [x, y, r, y, r, t, x, t, x, y]
		stack_plane.data[idx]["pos"] = xy
		stack_plane._redraw_bind_uid = stack_plane.fbind(
			"data", stack_plane._trigger_redraw
		)
		stack_plane._fbo.release()

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


# The following is a demonstration of a graphical error involving TextureStackPlane
# and its interaction with StencilView.
# When BACKGROUND is True, TextureStackPlane overflows the StencilView it's in,
# even though the background image doesn't.
# When BACKGROUND is False, TextureStackPlane obeys the StencilView.

TEST_DATA = [{'x': 0.0, 'y': 0.0, 'width': 32, 'height': 32, 'name': (0, 0), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.0, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (0, 1), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.0, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (0, 2), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.0, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (0, 3), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.0, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (0, 4), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.0, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (0, 5), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.0, 'y': 0.245, 'width': 32, 'height': 32, 'name': (0, 6), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.0, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (0, 7), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.0, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (0, 8), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.0, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (0, 9), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.0, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (0, 10), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.0, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (0, 11), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.0, 'y': 0.49, 'width': 32, 'height': 32, 'name': (0, 12), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.0, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (0, 13), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.0, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (0, 14), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.0, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (0, 15), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.0, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (0, 16), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.0, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (0, 17), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.0, 'y': 0.735, 'width': 32, 'height': 32, 'name': (0, 18), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.0, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (0, 19), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.0, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (0, 20), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.0, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (0, 21), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.0, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (0, 22), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.0, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (0, 23), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.0, 'y': 0.98, 'width': 32, 'height': 32, 'name': (0, 24), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.04083333333333333, 'y': 0.0, 'width': 32, 'height': 32, 'name': (1, 0), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.04083333333333333, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (1, 1), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.04083333333333333, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (1, 2), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.04083333333333333, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (1, 3), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.04083333333333333, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (1, 4), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.04083333333333333, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (1, 5), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.04083333333333333, 'y': 0.245, 'width': 32, 'height': 32, 'name': (1, 6), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.04083333333333333, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (1, 7), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.04083333333333333, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (1, 8), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.04083333333333333, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (1, 9), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.04083333333333333, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (1, 10), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.04083333333333333, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (1, 11), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.04083333333333333, 'y': 0.49, 'width': 32, 'height': 32, 'name': (1, 12), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.04083333333333333, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (1, 13), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.04083333333333333, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (1, 14), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.04083333333333333, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (1, 15), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.04083333333333333, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (1, 16), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.04083333333333333, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (1, 17), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.04083333333333333, 'y': 0.735, 'width': 32, 'height': 32, 'name': (1, 18), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.04083333333333333, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (1, 19), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.04083333333333333, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (1, 20), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.04083333333333333, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (1, 21), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.04083333333333333, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (1, 22), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.04083333333333333, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (1, 23), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.04083333333333333, 'y': 0.98, 'width': 32, 'height': 32, 'name': (1, 24), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.08166666666666667, 'y': 0.0, 'width': 32, 'height': 32, 'name': (2, 0), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.08166666666666667, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (2, 1), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.08166666666666667, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (2, 2), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.08166666666666667, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (2, 3), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.08166666666666667, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (2, 4), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.08166666666666667, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (2, 5), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.08166666666666667, 'y': 0.245, 'width': 32, 'height': 32, 'name': (2, 6), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.08166666666666667, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (2, 7), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.08166666666666667, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (2, 8), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.08166666666666667, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (2, 9), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.08166666666666667, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (2, 10), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.08166666666666667, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (2, 11), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.08166666666666667, 'y': 0.49, 'width': 32, 'height': 32, 'name': (2, 12), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.08166666666666667, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (2, 13), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.08166666666666667, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (2, 14), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.08166666666666667, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (2, 15), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.08166666666666667, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (2, 16), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.08166666666666667, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (2, 17), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.08166666666666667, 'y': 0.735, 'width': 32, 'height': 32, 'name': (2, 18), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.08166666666666667, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (2, 19), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.08166666666666667, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (2, 20), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.08166666666666667, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (2, 21), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.08166666666666667, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (2, 22), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.08166666666666667, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (2, 23), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.08166666666666667, 'y': 0.98, 'width': 32, 'height': 32, 'name': (2, 24), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.1225, 'y': 0.0, 'width': 32, 'height': 32, 'name': (3, 0), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.1225, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (3, 1), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.1225, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (3, 2), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.1225, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (3, 3), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.1225, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (3, 4), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.1025390625, 'y': 0.2861328125, 'width': 32, 'height': 32, 'name': (3, 5), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.1225, 'y': 0.245, 'width': 32, 'height': 32, 'name': (3, 6), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.1225, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (3, 7), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.1225, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (3, 8), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.1225, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (3, 9), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.1225, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (3, 10), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.1225, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (3, 11), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.1225, 'y': 0.49, 'width': 32, 'height': 32, 'name': (3, 12), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.1225, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (3, 13), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.1225, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (3, 14), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.1225, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (3, 15), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.1225, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (3, 16), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.1225, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (3, 17), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.1225, 'y': 0.735, 'width': 32, 'height': 32, 'name': (3, 18), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.1225, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (3, 19), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.1225, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (3, 20), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.1225, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (3, 21), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.1225, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (3, 22), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.1225, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (3, 23), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.1225, 'y': 0.98, 'width': 32, 'height': 32, 'name': (3, 24), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.0, 'width': 32, 'height': 32, 'name': (4, 0), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (4, 1), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (4, 2), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (4, 3), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (4, 4), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (4, 5), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.245, 'width': 32, 'height': 32, 'name': (4, 6), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (4, 7), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.16333333333333333, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (4, 8), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (4, 9), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.16333333333333333, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (4, 10), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (4, 11), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.49, 'width': 32, 'height': 32, 'name': (4, 12), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (4, 13), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (4, 14), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (4, 15), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (4, 16), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.16333333333333333, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (4, 17), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.735, 'width': 32, 'height': 32, 'name': (4, 18), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (4, 19), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (4, 20), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.16333333333333333, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (4, 21), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (4, 22), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.16333333333333333, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (4, 23), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.16333333333333333, 'y': 0.98, 'width': 32, 'height': 32, 'name': (4, 24), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.20416666666666666, 'y': 0.0, 'width': 32, 'height': 32, 'name': (5, 0), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.20416666666666666, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (5, 1), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.20416666666666666, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (5, 2), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.20416666666666666, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (5, 3), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.20416666666666666, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (5, 4), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.20416666666666666, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (5, 5), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.20416666666666666, 'y': 0.245, 'width': 32, 'height': 32, 'name': (5, 6), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.20416666666666666, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (5, 7), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.20416666666666666, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (5, 8), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.20416666666666666, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (5, 9), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.20416666666666666, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (5, 10), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.20416666666666666, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (5, 11), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.20416666666666666, 'y': 0.49, 'width': 32, 'height': 32, 'name': (5, 12), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.20416666666666666, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (5, 13), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.20416666666666666, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (5, 14), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.20416666666666666, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (5, 15), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.20416666666666666, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (5, 16), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.20416666666666666, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (5, 17), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.20416666666666666, 'y': 0.735, 'width': 32, 'height': 32, 'name': (5, 18), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.20416666666666666, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (5, 19), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.20416666666666666, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (5, 20), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.20416666666666666, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (5, 21), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.20416666666666666, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (5, 22), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.20416666666666666, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (5, 23), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.20416666666666666, 'y': 0.98, 'width': 32, 'height': 32, 'name': (5, 24), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.245, 'y': 0.0, 'width': 32, 'height': 32, 'name': (6, 0), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.245, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (6, 1), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.245, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (6, 2), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.245, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (6, 3), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.245, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (6, 4), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.245, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (6, 5), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.245, 'y': 0.245, 'width': 32, 'height': 32, 'name': (6, 6), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.245, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (6, 7), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.245, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (6, 8), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.245, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (6, 9), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.245, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (6, 10), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.245, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (6, 11), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.245, 'y': 0.49, 'width': 32, 'height': 32, 'name': (6, 12), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.245, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (6, 13), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.245, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (6, 14), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.245, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (6, 15), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.245, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (6, 16), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.245, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (6, 17), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.245, 'y': 0.735, 'width': 32, 'height': 32, 'name': (6, 18), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.245, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (6, 19), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.245, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (6, 20), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.245, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (6, 21), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.245, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (6, 22), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.245, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (6, 23), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.245, 'y': 0.98, 'width': 32, 'height': 32, 'name': (6, 24), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.28583333333333333, 'y': 0.0, 'width': 32, 'height': 32, 'name': (7, 0), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.28583333333333333, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (7, 1), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.28583333333333333, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (7, 2), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.28583333333333333, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (7, 3), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.28583333333333333, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (7, 4), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.28583333333333333, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (7, 5), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.28583333333333333, 'y': 0.245, 'width': 32, 'height': 32, 'name': (7, 6), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.28583333333333333, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (7, 7), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.28583333333333333, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (7, 8), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.28583333333333333, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (7, 9), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.28583333333333333, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (7, 10), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.28583333333333333, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (7, 11), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.28583333333333333, 'y': 0.49, 'width': 32, 'height': 32, 'name': (7, 12), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.28583333333333333, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (7, 13), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.28583333333333333, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (7, 14), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.28583333333333333, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (7, 15), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.28583333333333333, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (7, 16), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.28583333333333333, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (7, 17), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.28583333333333333, 'y': 0.735, 'width': 32, 'height': 32, 'name': (7, 18), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.28583333333333333, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (7, 19), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.28583333333333333, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (7, 20), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.28583333333333333, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (7, 21), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.28583333333333333, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (7, 22), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.28583333333333333, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (7, 23), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.28583333333333333, 'y': 0.98, 'width': 32, 'height': 32, 'name': (7, 24), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.32666666666666666, 'y': 0.0, 'width': 32, 'height': 32, 'name': (8, 0), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.32666666666666666, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (8, 1), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.32666666666666666, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (8, 2), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.32666666666666666, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (8, 3), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.32666666666666666, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (8, 4), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.32666666666666666, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (8, 5), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.32666666666666666, 'y': 0.245, 'width': 32, 'height': 32, 'name': (8, 6), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.32666666666666666, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (8, 7), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.32666666666666666, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (8, 8), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.32666666666666666, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (8, 9), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.32666666666666666, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (8, 10), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.32666666666666666, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (8, 11), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.32666666666666666, 'y': 0.49, 'width': 32, 'height': 32, 'name': (8, 12), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.32666666666666666, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (8, 13), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.32666666666666666, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (8, 14), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.32666666666666666, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (8, 15), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.32666666666666666, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (8, 16), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.32666666666666666, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (8, 17), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.32666666666666666, 'y': 0.735, 'width': 32, 'height': 32, 'name': (8, 18), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.32666666666666666, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (8, 19), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.32666666666666666, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (8, 20), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.32666666666666666, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (8, 21), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.32666666666666666, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (8, 22), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.32666666666666666, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (8, 23), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.32666666666666666, 'y': 0.98, 'width': 32, 'height': 32, 'name': (8, 24), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.3675, 'y': 0.0, 'width': 32, 'height': 32, 'name': (9, 0), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.3675, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (9, 1), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.3675, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (9, 2), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.3675, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (9, 3), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.3675, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (9, 4), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.3675, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (9, 5), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.3675, 'y': 0.245, 'width': 32, 'height': 32, 'name': (9, 6), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.3675, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (9, 7), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.36718750000000006, 'y': 0.322265625, 'width': 32, 'height': 32, 'name': (9, 8), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.3675, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (9, 9), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.3675, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (9, 10), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.3675, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (9, 11), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.3675, 'y': 0.49, 'width': 32, 'height': 32, 'name': (9, 12), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.3675, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (9, 13), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.3675, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (9, 14), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.3675, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (9, 15), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.3675, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (9, 16), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.3675, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (9, 17), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.3675, 'y': 0.735, 'width': 32, 'height': 32, 'name': (9, 18), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.3675, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (9, 19), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.3675, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (9, 20), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.3675, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (9, 21), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.3675, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (9, 22), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.3675, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (9, 23), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.3675, 'y': 0.98, 'width': 32, 'height': 32, 'name': (9, 24), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.4083333333333333, 'y': 0.0, 'width': 32, 'height': 32, 'name': (10, 0), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.4083333333333333, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (10, 1), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.4083333333333333, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (10, 2), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.4083333333333333, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (10, 3), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.4083333333333333, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (10, 4), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.4083333333333333, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (10, 5), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.4083333333333333, 'y': 0.245, 'width': 32, 'height': 32, 'name': (10, 6), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.41015625, 'y': 0.2832031250000001, 'width': 32, 'height': 32, 'name': (10, 7), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.4083333333333333, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (10, 8), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.4083333333333333, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (10, 9), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.4083333333333333, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (10, 10), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.4083333333333333, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (10, 11), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.4083333333333333, 'y': 0.49, 'width': 32, 'height': 32, 'name': (10, 12), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.4083333333333333, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (10, 13), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.4083333333333333, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (10, 14), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.4083333333333333, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (10, 15), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.4083333333333333, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (10, 16), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.4083333333333333, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (10, 17), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.4083333333333333, 'y': 0.735, 'width': 32, 'height': 32, 'name': (10, 18), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.4083333333333333, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (10, 19), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.4083333333333333, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (10, 20), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.4083333333333333, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (10, 21), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.4083333333333333, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (10, 22), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.4083333333333333, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (10, 23), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.4083333333333333, 'y': 0.98, 'width': 32, 'height': 32, 'name': (10, 24), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.44916666666666666, 'y': 0.0, 'width': 32, 'height': 32, 'name': (11, 0), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.44916666666666666, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (11, 1), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.44916666666666666, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (11, 2), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.44916666666666666, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (11, 3), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.44916666666666666, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (11, 4), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.44916666666666666, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (11, 5), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.44916666666666666, 'y': 0.245, 'width': 32, 'height': 32, 'name': (11, 6), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.44916666666666666, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (11, 7), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.44916666666666666, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (11, 8), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.44916666666666666, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (11, 9), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.44916666666666666, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (11, 10), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.44916666666666666, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (11, 11), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.44916666666666666, 'y': 0.49, 'width': 32, 'height': 32, 'name': (11, 12), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.44916666666666666, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (11, 13), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.44916666666666666, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (11, 14), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.44916666666666666, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (11, 15), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.44916666666666666, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (11, 16), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.44916666666666666, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (11, 17), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.44916666666666666, 'y': 0.735, 'width': 32, 'height': 32, 'name': (11, 18), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.44916666666666666, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (11, 19), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.44916666666666666, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (11, 20), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.44916666666666666, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (11, 21), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.44916666666666666, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (11, 22), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.44916666666666666, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (11, 23), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.44916666666666666, 'y': 0.98, 'width': 32, 'height': 32, 'name': (11, 24), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.49, 'y': 0.0, 'width': 32, 'height': 32, 'name': (12, 0), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.49, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (12, 1), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.49, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (12, 2), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.49, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (12, 3), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.49, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (12, 4), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.49, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (12, 5), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.49, 'y': 0.245, 'width': 32, 'height': 32, 'name': (12, 6), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.49, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (12, 7), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.49, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (12, 8), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.49, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (12, 9), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.49, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (12, 10), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.49, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (12, 11), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.49, 'y': 0.49, 'width': 32, 'height': 32, 'name': (12, 12), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.49, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (12, 13), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.49, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (12, 14), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.49, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (12, 15), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.49, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (12, 16), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.49, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (12, 17), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.49, 'y': 0.735, 'width': 32, 'height': 32, 'name': (12, 18), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.49, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (12, 19), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.49, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (12, 20), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.49, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (12, 21), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.49, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (12, 22), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.49, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (12, 23), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.49, 'y': 0.98, 'width': 32, 'height': 32, 'name': (12, 24), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5308333333333333, 'y': 0.0, 'width': 32, 'height': 32, 'name': (13, 0), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5308333333333333, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (13, 1), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5308333333333333, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (13, 2), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5308333333333333, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (13, 3), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5308333333333333, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (13, 4), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5308333333333333, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (13, 5), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5308333333333333, 'y': 0.245, 'width': 32, 'height': 32, 'name': (13, 6), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5308333333333333, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (13, 7), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5308333333333333, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (13, 8), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5308333333333333, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (13, 9), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5308333333333333, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (13, 10), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5308333333333333, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (13, 11), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5308333333333333, 'y': 0.49, 'width': 32, 'height': 32, 'name': (13, 12), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5308333333333333, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (13, 13), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5308333333333333, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (13, 14), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5308333333333333, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (13, 15), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5308333333333333, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (13, 16), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5308333333333333, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (13, 17), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5308333333333333, 'y': 0.735, 'width': 32, 'height': 32, 'name': (13, 18), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5308333333333333, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (13, 19), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5308333333333333, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (13, 20), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5308333333333333, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (13, 21), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5308333333333333, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (13, 22), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5308333333333333, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (13, 23), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5308333333333333, 'y': 0.98, 'width': 32, 'height': 32, 'name': (13, 24), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5716666666666667, 'y': 0.0, 'width': 32, 'height': 32, 'name': (14, 0), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5716666666666667, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (14, 1), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5716666666666667, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (14, 2), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5716666666666667, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (14, 3), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5716666666666667, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (14, 4), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5716666666666667, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (14, 5), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5716666666666667, 'y': 0.245, 'width': 32, 'height': 32, 'name': (14, 6), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5716666666666667, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (14, 7), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5716666666666667, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (14, 8), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5716666666666667, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (14, 9), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5716666666666667, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (14, 10), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5716666666666667, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (14, 11), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5716666666666667, 'y': 0.49, 'width': 32, 'height': 32, 'name': (14, 12), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5716666666666667, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (14, 13), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5716666666666667, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (14, 14), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5716666666666667, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (14, 15), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5716666666666667, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (14, 16), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5716666666666667, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (14, 17), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5716666666666667, 'y': 0.735, 'width': 32, 'height': 32, 'name': (14, 18), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5716666666666667, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (14, 19), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5716666666666667, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (14, 20), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5716666666666667, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (14, 21), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5716666666666667, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (14, 22), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.5716666666666667, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (14, 23), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.5716666666666667, 'y': 0.98, 'width': 32, 'height': 32, 'name': (14, 24), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6125, 'y': 0.0, 'width': 32, 'height': 32, 'name': (15, 0), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6125, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (15, 1), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6125, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (15, 2), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6125, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (15, 3), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6125, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (15, 4), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6125, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (15, 5), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6125, 'y': 0.245, 'width': 32, 'height': 32, 'name': (15, 6), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6125, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (15, 7), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6125, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (15, 8), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6125, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (15, 9), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6125, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (15, 10), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6125, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (15, 11), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6125, 'y': 0.49, 'width': 32, 'height': 32, 'name': (15, 12), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6125, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (15, 13), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6125, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (15, 14), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6125, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (15, 15), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6125, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (15, 16), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6125, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (15, 17), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6125, 'y': 0.735, 'width': 32, 'height': 32, 'name': (15, 18), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6125, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (15, 19), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6125, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (15, 20), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6125, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (15, 21), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6125, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (15, 22), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6125, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (15, 23), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6125, 'y': 0.98, 'width': 32, 'height': 32, 'name': (15, 24), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6533333333333333, 'y': 0.0, 'width': 32, 'height': 32, 'name': (16, 0), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6533333333333333, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (16, 1), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6533333333333333, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (16, 2), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6533333333333333, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (16, 3), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6533333333333333, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (16, 4), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6533333333333333, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (16, 5), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6533333333333333, 'y': 0.245, 'width': 32, 'height': 32, 'name': (16, 6), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6533333333333333, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (16, 7), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6533333333333333, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (16, 8), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6533333333333333, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (16, 9), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6533333333333333, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (16, 10), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6533333333333333, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (16, 11), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6533333333333333, 'y': 0.49, 'width': 32, 'height': 32, 'name': (16, 12), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6533333333333333, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (16, 13), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6533333333333333, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (16, 14), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6533333333333333, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (16, 15), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6533333333333333, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (16, 16), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6533333333333333, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (16, 17), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6533333333333333, 'y': 0.735, 'width': 32, 'height': 32, 'name': (16, 18), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6533333333333333, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (16, 19), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6533333333333333, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (16, 20), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6533333333333333, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (16, 21), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6533333333333333, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (16, 22), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6533333333333333, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (16, 23), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6533333333333333, 'y': 0.98, 'width': 32, 'height': 32, 'name': (16, 24), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6941666666666666, 'y': 0.0, 'width': 32, 'height': 32, 'name': (17, 0), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6941666666666666, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (17, 1), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6941666666666666, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (17, 2), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6941666666666666, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (17, 3), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6941666666666666, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (17, 4), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6941666666666666, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (17, 5), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6941666666666666, 'y': 0.245, 'width': 32, 'height': 32, 'name': (17, 6), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6941666666666666, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (17, 7), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6941666666666666, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (17, 8), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6941666666666666, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (17, 9), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6941666666666666, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (17, 10), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6941666666666666, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (17, 11), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6941666666666666, 'y': 0.49, 'width': 32, 'height': 32, 'name': (17, 12), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6941666666666666, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (17, 13), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6941666666666666, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (17, 14), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6941666666666666, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (17, 15), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6941666666666666, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (17, 16), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6941666666666666, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (17, 17), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6941666666666666, 'y': 0.735, 'width': 32, 'height': 32, 'name': (17, 18), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6941666666666666, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (17, 19), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6941666666666666, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (17, 20), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.6941666666666666, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (17, 21), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6941666666666666, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (17, 22), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6941666666666666, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (17, 23), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.6941666666666666, 'y': 0.98, 'width': 32, 'height': 32, 'name': (17, 24), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.735, 'y': 0.0, 'width': 32, 'height': 32, 'name': (18, 0), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.735, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (18, 1), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.735, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (18, 2), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.735, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (18, 3), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.735, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (18, 4), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.735, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (18, 5), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.735, 'y': 0.245, 'width': 32, 'height': 32, 'name': (18, 6), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.735, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (18, 7), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.735, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (18, 8), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.735, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (18, 9), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.735, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (18, 10), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.735, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (18, 11), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.735, 'y': 0.49, 'width': 32, 'height': 32, 'name': (18, 12), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.735, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (18, 13), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.735, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (18, 14), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.735, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (18, 15), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.735, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (18, 16), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.735, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (18, 17), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.735, 'y': 0.735, 'width': 32, 'height': 32, 'name': (18, 18), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.735, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (18, 19), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.735, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (18, 20), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.735, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (18, 21), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.735, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (18, 22), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.735, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (18, 23), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.735, 'y': 0.98, 'width': 32, 'height': 32, 'name': (18, 24), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.7758333333333334, 'y': 0.0, 'width': 32, 'height': 32, 'name': (19, 0), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.7758333333333334, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (19, 1), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.7758333333333334, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (19, 2), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.7758333333333334, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (19, 3), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.7758333333333334, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (19, 4), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.7758333333333334, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (19, 5), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.7758333333333334, 'y': 0.245, 'width': 32, 'height': 32, 'name': (19, 6), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.7758333333333334, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (19, 7), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.7758333333333334, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (19, 8), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.7758333333333334, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (19, 9), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.7758333333333334, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (19, 10), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.7758333333333334, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (19, 11), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.7758333333333334, 'y': 0.49, 'width': 32, 'height': 32, 'name': (19, 12), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.7758333333333334, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (19, 13), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.7758333333333334, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (19, 14), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.7758333333333334, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (19, 15), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.7758333333333334, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (19, 16), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.7758333333333334, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (19, 17), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.7758333333333334, 'y': 0.735, 'width': 32, 'height': 32, 'name': (19, 18), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.7758333333333334, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (19, 19), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.7758333333333334, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (19, 20), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.7758333333333334, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (19, 21), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.7758333333333334, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (19, 22), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.7758333333333334, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (19, 23), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.7758333333333334, 'y': 0.98, 'width': 32, 'height': 32, 'name': (19, 24), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8166666666666667, 'y': 0.0, 'width': 32, 'height': 32, 'name': (20, 0), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8166666666666667, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (20, 1), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8166666666666667, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (20, 2), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8166666666666667, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (20, 3), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8166666666666667, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (20, 4), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8166666666666667, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (20, 5), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8166666666666667, 'y': 0.245, 'width': 32, 'height': 32, 'name': (20, 6), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8166666666666667, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (20, 7), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8166666666666667, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (20, 8), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8166666666666667, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (20, 9), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8166666666666667, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (20, 10), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8166666666666667, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (20, 11), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8166666666666667, 'y': 0.49, 'width': 32, 'height': 32, 'name': (20, 12), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8166666666666667, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (20, 13), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8166666666666667, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (20, 14), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8166666666666667, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (20, 15), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8166666666666667, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (20, 16), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8166666666666667, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (20, 17), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8166666666666667, 'y': 0.735, 'width': 32, 'height': 32, 'name': (20, 18), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8166666666666667, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (20, 19), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8166666666666667, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (20, 20), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8166666666666667, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (20, 21), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8166666666666667, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (20, 22), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8166666666666667, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (20, 23), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8166666666666667, 'y': 0.98, 'width': 32, 'height': 32, 'name': (20, 24), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8574999999999999, 'y': 0.0, 'width': 32, 'height': 32, 'name': (21, 0), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8574999999999999, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (21, 1), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8574999999999999, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (21, 2), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8574999999999999, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (21, 3), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8574999999999999, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (21, 4), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8574999999999999, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (21, 5), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8574999999999999, 'y': 0.245, 'width': 32, 'height': 32, 'name': (21, 6), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8574999999999999, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (21, 7), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8574999999999999, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (21, 8), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8574999999999999, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (21, 9), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8574999999999999, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (21, 10), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8574999999999999, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (21, 11), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8574999999999999, 'y': 0.49, 'width': 32, 'height': 32, 'name': (21, 12), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8574999999999999, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (21, 13), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8574999999999999, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (21, 14), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8574999999999999, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (21, 15), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8574999999999999, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (21, 16), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8574999999999999, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (21, 17), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8574999999999999, 'y': 0.735, 'width': 32, 'height': 32, 'name': (21, 18), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8574999999999999, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (21, 19), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8574999999999999, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (21, 20), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8574999999999999, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (21, 21), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8574999999999999, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (21, 22), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8574999999999999, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (21, 23), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8574999999999999, 'y': 0.98, 'width': 32, 'height': 32, 'name': (21, 24), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8983333333333333, 'y': 0.0, 'width': 32, 'height': 32, 'name': (22, 0), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8983333333333333, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (22, 1), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8983333333333333, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (22, 2), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8983333333333333, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (22, 3), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8983333333333333, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (22, 4), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8983333333333333, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (22, 5), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8983333333333333, 'y': 0.245, 'width': 32, 'height': 32, 'name': (22, 6), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8983333333333333, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (22, 7), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8983333333333333, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (22, 8), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8983333333333333, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (22, 9), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8983333333333333, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (22, 10), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8983333333333333, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (22, 11), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8983333333333333, 'y': 0.49, 'width': 32, 'height': 32, 'name': (22, 12), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8983333333333333, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (22, 13), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8983333333333333, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (22, 14), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8983333333333333, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (22, 15), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8983333333333333, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (22, 16), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8983333333333333, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (22, 17), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8983333333333333, 'y': 0.735, 'width': 32, 'height': 32, 'name': (22, 18), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8983333333333333, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (22, 19), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8983333333333333, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (22, 20), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8983333333333333, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (22, 21), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.8983333333333333, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (22, 22), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8983333333333333, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (22, 23), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.8983333333333333, 'y': 0.98, 'width': 32, 'height': 32, 'name': (22, 24), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.9391666666666667, 'y': 0.0, 'width': 32, 'height': 32, 'name': (23, 0), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.9391666666666667, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (23, 1), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.9391666666666667, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (23, 2), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.9391666666666667, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (23, 3), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.9391666666666667, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (23, 4), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.9391666666666667, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (23, 5), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.9391666666666667, 'y': 0.245, 'width': 32, 'height': 32, 'name': (23, 6), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.9391666666666667, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (23, 7), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.9391666666666667, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (23, 8), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.9391666666666667, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (23, 9), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.9391666666666667, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (23, 10), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.9391666666666667, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (23, 11), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.9391666666666667, 'y': 0.49, 'width': 32, 'height': 32, 'name': (23, 12), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.9391666666666667, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (23, 13), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.9391666666666667, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (23, 14), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.9391666666666667, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (23, 15), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.9391666666666667, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (23, 16), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.9391666666666667, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (23, 17), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.9391666666666667, 'y': 0.735, 'width': 32, 'height': 32, 'name': (23, 18), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.9391666666666667, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (23, 19), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.9391666666666667, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (23, 20), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.9391666666666667, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (23, 21), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.9391666666666667, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (23, 22), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.9391666666666667, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (23, 23), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.9391666666666667, 'y': 0.98, 'width': 32, 'height': 32, 'name': (23, 24), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.98, 'y': 0.0, 'width': 32, 'height': 32, 'name': (24, 0), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.98, 'y': 0.04083333333333333, 'width': 32, 'height': 32, 'name': (24, 1), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.98, 'y': 0.08166666666666667, 'width': 32, 'height': 32, 'name': (24, 2), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.98, 'y': 0.1225, 'width': 32, 'height': 32, 'name': (24, 3), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.98, 'y': 0.16333333333333333, 'width': 32, 'height': 32, 'name': (24, 4), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.98, 'y': 0.20416666666666666, 'width': 32, 'height': 32, 'name': (24, 5), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.98, 'y': 0.245, 'width': 32, 'height': 32, 'name': (24, 6), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.98, 'y': 0.28583333333333333, 'width': 32, 'height': 32, 'name': (24, 7), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.98, 'y': 0.32666666666666666, 'width': 32, 'height': 32, 'name': (24, 8), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.98, 'y': 0.3675, 'width': 32, 'height': 32, 'name': (24, 9), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.98, 'y': 0.4083333333333333, 'width': 32, 'height': 32, 'name': (24, 10), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.98, 'y': 0.44916666666666666, 'width': 32, 'height': 32, 'name': (24, 11), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.98, 'y': 0.49, 'width': 32, 'height': 32, 'name': (24, 12), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.98, 'y': 0.5308333333333333, 'width': 32, 'height': 32, 'name': (24, 13), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.98, 'y': 0.5716666666666667, 'width': 32, 'height': 32, 'name': (24, 14), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.98, 'y': 0.6125, 'width': 32, 'height': 32, 'name': (24, 15), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.98, 'y': 0.6533333333333333, 'width': 32, 'height': 32, 'name': (24, 16), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.98, 'y': 0.6941666666666666, 'width': 32, 'height': 32, 'name': (24, 17), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.98, 'y': 0.735, 'width': 32, 'height': 32, 'name': (24, 18), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.98, 'y': 0.7758333333333334, 'width': 32, 'height': 32, 'name': (24, 19), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.98, 'y': 0.8166666666666667, 'width': 32, 'height': 32, 'name': (24, 20), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.98, 'y': 0.8574999999999999, 'width': 32, 'height': 32, 'name': (24, 21), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.98, 'y': 0.8983333333333333, 'width': 32, 'height': 32, 'name': (24, 22), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.98, 'y': 0.9391666666666667, 'width': 32, 'height': 32, 'name': (24, 23), 'textures': ['atlas://rltiles/floor/floor-moss']}, {'x': 0.98, 'y': 0.98, 'width': 32, 'height': 32, 'name': (24, 24), 'textures': ['atlas://rltiles/floor/floor-normal']}, {'x': 0.1123046875, 'y': 0.2216796875, 'width': 16, 'height': 16, 'name': 'place1', 'textures': ['atlas://kenney1bit.atlas/(1, 14)']}, {'x': 0.1435546875, 'y': 0.22265625, 'width': 16, 'height': 16, 'name': 'place2', 'textures': ['atlas://kenney1bit.atlas/(1, 14)']}, {'x': 0.12109375, 'y': 0.18505859375000003, 'width': 32, 'height': 32, 'name': 'place3', 'textures': ['atlas://floor.atlas/floor-normal']}, {'x': 0.2001953125, 'y': 0.265625, 'width': 32, 'height': 32, 'name': 'place4', 'textures': ['atlas://rltiles/floor.atlas/floor-stone']}, {'x': 0.11328125, 'y': 0.21533203125000006, 'width': 32, 'height': 32, 'name': 'place5', 'textures': ['atlas://rltiles/floor.atlas/floor-stone']}, {'x': 910, 'y': 742, 'width': 32, 'height': 32, 'name': 'wolf0', 'textures': ['atlas://rltiles/dc-mon/war_dog']}, {'x': 450, 'y': 241, 'width': 32, 'height': 32, 'name': 'wolf1', 'textures': ['atlas://rltiles/dc-mon/war_dog']}, {'x': 282, 'y': 742, 'width': 32, 'height': 32, 'name': 'wolf2', 'textures': ['atlas://rltiles/dc-mon/war_dog']}, {'x': 115, 'y': 826, 'width': 32, 'height': 32, 'name': 'wolf3', 'textures': ['atlas://rltiles/dc-mon/war_dog']}, {'x': 701, 'y': 1035, 'width': 32, 'height': 32, 'name': 'wolf4', 'textures': ['atlas://rltiles/dc-mon/war_dog']}, {'x': 491, 'y': 910, 'width': 32, 'height': 32, 'name': 'wolf5', 'textures': ['atlas://rltiles/dc-mon/war_dog']}, {'x': 491, 'y': 617, 'width': 32, 'height': 32, 'name': 'wolf6', 'textures': ['atlas://rltiles/dc-mon/war_dog']}, {'x': 910, 'y': 324, 'width': 32, 'height': 32, 'name': 'wolf7', 'textures': ['atlas://rltiles/dc-mon/war_dog']}, {'x': 241, 'y': 533, 'width': 32, 'height': 32, 'name': 'wolf8', 'textures': ['atlas://rltiles/dc-mon/war_dog']}, {'x': 993, 'y': 533, 'width': 32, 'height': 32, 'name': 'wolf9', 'textures': ['atlas://rltiles/dc-mon/war_dog']}, {'x': 910, 'y': 450, 'width': 32, 'height': 32, 'name': 'sheep0', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 366, 'y': 659, 'width': 32, 'height': 32, 'name': 'sheep1', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 910, 'y': 826, 'width': 32, 'height': 32, 'name': 'sheep2', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 199, 'y': 324, 'width': 32, 'height': 32, 'name': 'sheep3', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 701, 'y': 868, 'width': 32, 'height': 32, 'name': 'sheep4', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 951, 'y': 282, 'width': 32, 'height': 32, 'name': 'sheep5', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 701, 'y': 575, 'width': 32, 'height': 32, 'name': 'sheep6', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 575, 'y': 533, 'width': 32, 'height': 32, 'name': 'sheep7', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 491, 'y': 1035, 'width': 32, 'height': 32, 'name': 'sheep8', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 784, 'y': 115, 'width': 32, 'height': 32, 'name': 'sheep9', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 659, 'y': 241, 'width': 32, 'height': 32, 'name': 'sheep10', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 993, 'y': 282, 'width': 32, 'height': 32, 'name': 'sheep11', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 241, 'y': 701, 'width': 32, 'height': 32, 'name': 'sheep12', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 282, 'y': 450, 'width': 32, 'height': 32, 'name': 'sheep13', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 241, 'y': 73, 'width': 32, 'height': 32, 'name': 'sheep14', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 199, 'y': 993, 'width': 32, 'height': 32, 'name': 'sheep15', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 659, 'y': 659, 'width': 32, 'height': 32, 'name': 'sheep16', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 701, 'y': 450, 'width': 32, 'height': 32, 'name': 'sheep17', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 826, 'y': 199, 'width': 32, 'height': 32, 'name': 'sheep18', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 241, 'y': 742, 'width': 32, 'height': 32, 'name': 'sheep19', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 951, 'y': 993, 'width': 32, 'height': 32, 'name': 'sheep20', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 533, 'y': 282, 'width': 32, 'height': 32, 'name': 'sheep21', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 617, 'y': 533, 'width': 32, 'height': 32, 'name': 'sheep22', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 826, 'y': 408, 'width': 32, 'height': 32, 'name': 'sheep23', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 742, 'y': 32, 'width': 32, 'height': 32, 'name': 'sheep24', 'textures': ['atlas://rltiles/dc-mon/sheep']}, {'x': 241, 'y': 282, 'width': 16, 'height': 16, 'name': 'thing1', 'textures': ['atlas://kenney1bit.atlas/(0, 0)']}, {'x': 366, 'y': 157, 'width': 16, 'height': 16, 'name': 'thing2', 'textures': ['atlas://kenney1bit.atlas/(0, 0)']}]
BACKGROUND = False  # change this to False, and the bug doesn't happen!

if __name__ == "__main__":
	from kivy.uix.boxlayout import BoxLayout
	from kivy.uix.floatlayout import FloatLayout
	from kivy.uix.image import Image as ImageWidget
	from kivy.uix.widget import Widget
	from kivy.uix.scatterlayout import ScatterPlaneLayout
	from kivy.uix.stencilview import StencilView
	from kivy.base import runTouchApp


	class TestTextureStackPlane(TextureStackPlane):
		def on_touch_down(self, touch):
			for collided in self.iter_collided_keys(*touch.pos):
				self._gonna_select = collided

		def on_touch_up(self, touch):
			if hasattr(self, "_gonna_select"):
				name = self._gonna_select
				oldname = self.selected
				insts = self._instructions[name]
				if self._gonna_select in self.iter_collided_keys(*touch.pos):
					Logger.info(f"TestTextureStackPlane: selecting {self._gonna_select}")
					self.selected = name
					if "color0" in insts:
						insts["color0"].rgba = self.color_selected
					else:
						idx = self._stack_index[name]
						left = self._left_xs[idx]
						bot = self._bot_ys[idx]
						right = self._right_xs[idx]
						top = self._top_ys[idx]
						grp = insts["group"]
						insts["color0"] = Color(
							rgba=self.color_selected)
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
					if oldname in self._instructions:
						self._instructions[oldname]["color0"].rgba = [0.0, 0.0, 0.0, 0.0]
				else:
					if "color0" in insts:
						insts["color0"].rgba = [0.0, 0.0, 0.0, 0.0]
				del self._gonna_select

	root = BoxLayout()
	root.add_widget(Widget())
	texstac = TestTextureStackPlane(data=TEST_DATA, size_hint=(None, None), size=(1024,1024))
	flot = FloatLayout()
	if BACKGROUND:
		texstacbg = ImageWidget(source="parchmentBasic.png", size_hint=(None, None), size=(1024,1024))
		flot.add_widget(texstacbg)
	flot.add_widget(texstac)
	splane = ScatterPlaneLayout()
	splane.add_widget(flot)
	stenc = StencilView()
	stenc.add_widget(splane)
	root.add_widget(stenc)
	root.add_widget(Widget())
	runTouchApp(root)
