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
"""Widget to represent :class:`Place`s. :class:`Pawn` moves around on
top of these.

"""

from kivy.clock import Clock, triggered
from kivy_garden.collider import Collide2DPoly

from elide.pawnspot import GraphPawnSpot

from .arrow import get_points, get_quad_vertices


def trigger(func):
	return triggered()(func)


class GraphSpot(GraphPawnSpot):
	"""The icon that represents a :class:`Place`.

	Each :class:`Spot` is located on the Board that represents the
	:class:`Character` that the underlying :class:`Place` is in. Its
	coordinates are relative to its :class:`Board`, not necessarily
	the window the :class:`Board` is in.

	"""

	default_image_paths = ["atlas://rltiles/floor.atlas/floor-stone"]
	default_pos = (0.5, 0.5)

	def __init__(self, **kwargs):
		"""Deal with triggers and bindings, and arrange to take care of
		changes in game-time.

		"""
		self._pospawn_partials = {}
		self._pospawn_triggers = {}
		kwargs["size_hint"] = (None, None)
		if "place" in kwargs:
			kwargs["proxy"] = kwargs["place"]
			del kwargs["place"]
		super().__init__(**kwargs)

	def on_board(self, *args):
		super().on_board(*args)
		self.board.bind(size=self._upd_pos)

	def on_pos(self, *args):
		def upd(orig, dest):
			bgr = r * bg_scale_selected  # change for selectedness pls
			if (orig, dest) not in port_index:
				return
			idx = port_index[orig, dest]
			inst = instructions[orig, dest]
			(ox, oy, dx, dy), (x1, y1, endx, endy, x2, y2) = get_points(
				spot[orig], spot[dest], arrowhead_size
			)
			if ox < dx:
				bot_left_xs[idx] = ox - bgr
				top_right_xs[idx] = dx + bgr
			else:
				bot_left_xs[idx] = dx - bgr
				top_right_xs[idx] = ox + bgr
			if oy < dy:
				bot_left_ys[idx] = oy - bgr
				top_right_ys[idx] = dy + bgr
			else:
				bot_left_ys[idx] = dy - bgr
				top_right_ys[idx] = oy + bgr
			quadverts = get_quad_vertices(
				ox, oy, dx, dy, x1, y1, endx, endy, x2, y2, bgr, r
			)
			inst["shaft_bg"].points = quadverts["shaft_bg"]
			colliders[orig, dest] = Collide2DPoly(points=quadverts["shaft_bg"])
			inst["left_head_bg"].points = quadverts["left_head_bg"]
			inst["right_head_bg"].points = quadverts["right_head_bg"]
			inst["shaft_fg"].points = quadverts["shaft_fg"]
			inst["left_head_fg"].points = quadverts["left_head_fg"]
			inst["right_head_fg"].points = quadverts["right_head_fg"]

		if not self.board:
			return
		arrow_plane = self.board.arrow_plane
		fbo = arrow_plane._fbo
		arrowhead_size = arrow_plane.arrowhead_size
		r = arrow_plane.arrow_width // 2
		bg_scale_selected = arrow_plane.bg_scale_selected
		spot = self.board.spot
		succ = self.board.arrow
		pred = self.board.pred_arrow
		name = self.name
		instructions = arrow_plane._instructions_map
		colliders = arrow_plane._colliders_map
		instructions = arrow_plane._instructions_map
		port_index = arrow_plane._port_index
		bot_left_xs = arrow_plane._bot_left_corner_xs
		bot_left_ys = arrow_plane._bot_left_corner_ys
		top_right_xs = arrow_plane._top_right_corner_xs
		top_right_ys = arrow_plane._top_right_corner_ys
		fbo.bind()
		fbo.clear_buffer()
		if name in succ:
			for dest in succ[name]:
				upd(name, dest)
		if name in pred:
			for orig in pred[name]:
				upd(orig, name)
		fbo.release()
		return super().on_pos(*args)

	def _upd_pos(self, *args):
		if self.board is None:
			Clock.schedule_once(self._upd_pos, 0)
			return
		self.pos = (
			int(self.proxy.get("_x", self.default_pos[0]) * self.board.width),
			int(self.proxy.get("_y", self.default_pos[1]) * self.board.height),
		)

	def finalize(self, initial=True):
		if initial:
			self._upd_pos()
		super().finalize(initial)

	def push_pos(self, *args):
		"""Set my current position, expressed as proportions of the graph's
		width and height, into the ``_x`` and ``_y`` keys of the
		entity in my ``proxy`` property, such that it will be
		recorded in the database.

		"""
		self.proxy["_x"] = self.x / self.board.width
		self.proxy["_y"] = self.y / self.board.height

	_trigger_push_pos = trigger(push_pos)

	def on_touch_up(self, touch):
		if touch.grab_current is not self:
			return False
		self.center = touch.pos
		self._trigger_push_pos()
		touch.ungrab(self)
		self._trigger_push_pos()
		return True

	def __repr__(self):
		"""Give my name and position."""
		return "<{}@({},{}) at {}>".format(self.name, self.x, self.y, id(self))
