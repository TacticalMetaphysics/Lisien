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
"""Code that draws the box around a Pawn or Spot when it's selected"""

from collections import defaultdict
from functools import partial

from kivy.clock import Clock, triggered
from kivy.graphics import (
	Color,
	InstructionGroup,
	Line,
	PopMatrix,
	PushMatrix,
	Translate,
)
from kivy.properties import BooleanProperty, ListProperty, ObjectProperty
from kivy.uix.layout import Layout

from elide.imagestackproxy import ImageStackProxy


def trigger(func):
	return triggered()(func)


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
