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
"""That which displays a one-way connection between two places.

An arrow connects two spots, the origin and the destination, and it
points from the origin to the destination, regardless of where on the
screen they are at the moment.

"""

from collections import defaultdict
from math import atan, cos, pi, sin
from operator import itemgetter
from typing import Iterator, Optional, Tuple

import numpy as np
from kivy.clock import Clock, mainthread
from kivy.core.text import Label
from kivy.graphics import Color, InstructionGroup, Quad, Rectangle, Translate
from kivy.graphics.fbo import Fbo
from kivy.properties import ListProperty, NumericProperty
from kivy.uix.widget import Widget

try:
	from kivy.garden.collider import Collide2DPoly
except (KeyError, ImportError):
	from kivy_garden.collider import Collide2DPoly

fortyfive = pi / 4
cos45 = cos(fortyfive)
sin45 = sin(fortyfive)


def up_and_down(orig, dest, taillen):
	if orig.center_y == dest.center_y:
		raise ValueError("Can't draw an arrow at a point")
	flipped = orig.center_y > dest.center_y
	if flipped:
		orig, dest = dest, orig
	x = int(orig.center_x)
	dy = int(dest.y)
	oy = int(orig.top)
	if flipped:
		oy, dy = dy, oy
	off1 = cos45 * taillen
	off2 = sin45 * taillen
	x0 = x
	y0 = oy
	endx = x
	endy = dy
	x1 = endx - off1
	x2 = endx + off1
	y1 = y2 = endy - off2 if oy < dy else endy + off2
	return [x0, y0, endx, endy], [x1, y1, endx, endy, x2, y2]


def left_and_right(orig, dest, taillen):
	if orig.center_x == dest.center_x:
		raise ValueError("Can't draw an arrow at a point")
	flipped = orig.center_x > dest.center_x
	if flipped:
		orig, dest = dest, orig
	y = int(orig.center_y)
	dx = int(dest.x)
	ox = int(orig.right)
	if flipped:
		ox, dx = dx, ox
	off1 = cos45 * taillen
	off2 = sin45 * taillen
	x0 = ox
	y0 = y
	endx = dx
	endy = y
	y1 = endy - off1
	y2 = endy + off1
	x1 = x2 = endx - off2 if ox < dx else endx + off2
	return [x0, y0, endx, endy], [x1, y1, endx, endy, x2, y2]


def _get_points_first_part(orig, dest, taillen):
	ox, oy = orig.center
	ow, oh = orig.size
	dx, dy = dest.center
	dw, dh = dest.size
	# Flip the endpoints around so that the arrow faces up and right.
	# We'll flip it back at the end as needed.
	xco = 1 if ox < dx else -1
	yco = 1 if oy < dy else -1
	ox *= xco
	dx *= xco
	oy *= yco
	dy *= yco
	# Nudge my endpoints so that I start and end at the edge of
	# a Spot.
	if dy - oy > dx - ox:
		topy = dy - dh / 2
		boty = oy + oh / 2
		leftx = ox
		rightx = dx
	else:
		leftx = ox + ow / 2
		rightx = dx - dw / 2
		topy = dy
		boty = oy
	# Degenerate cases.
	# Also, these need to be handled specially to avoid
	# division by zero later on.
	if rightx == leftx:
		return up_and_down(orig, dest, taillen)
	if topy == boty:
		return left_and_right(orig, dest, taillen)
	return ow, oh, dw, dh, xco, leftx, rightx, yco, topy, boty


def get_points_multi(args):
	"""Return a dictionary mapping (orig, dest) to pairs of point lists for arrows

	Takes an iterable of (orig, dest, taillen) where orig and dest are Spot instances

	taillen is an integer specifying how long the arrowhead should be.

	"""
	ret = {}
	keys = []
	topys = []
	botys = []
	leftxs = []
	rightxs = []
	taillens = []
	xcos = []
	ycos = []
	for orig, dest, taillen in args:
		try:
			p1 = _get_points_first_part(orig, dest, taillen)
		except ValueError:
			p1 = 2, 2, 2, 2, 1, 0, 1, 1, 0, 1
		if len(p1) == 2:
			ret[orig.name, dest.name] = p1
			continue
		ow, oh, dw, dh, xco, leftx, rightx, yco, topy, boty = p1
		keys.append((orig.name, dest.name))
		leftxs.append(leftx)
		rightxs.append(rightx)
		topys.append(topy)
		botys.append(boty)
		taillens.append(taillen)
		xcos.append(xco)
		ycos.append(yco)
	topys = np.array(topys)
	botys = np.array(botys)
	rightxs = np.array(rightxs)
	leftxs = np.array(leftxs)
	rises = np.subtract(topys, botys)
	runs = np.subtract(rightxs, leftxs)
	slopes = np.divide(rises, runs)
	unslopes = np.divide(runs, rises)
	start_thetas = np.arctan(slopes)
	end_thetas = np.arctan(unslopes)
	top_thetas = np.subtract(start_thetas, fortyfive)
	bot_thetas = np.subtract(pi - fortyfive, end_thetas)
	topsins = np.sin(top_thetas)
	topcoss = np.cos(top_thetas)
	botsins = np.sin(bot_thetas)
	botcoss = np.cos(bot_thetas)
	taillens = np.array(taillens)
	xoff1s = np.multiply(topcoss, taillens)
	yoff1s = np.multiply(topsins, taillens)
	xoff2s = np.multiply(botcoss, taillens)
	yoff2s = np.multiply(botsins, taillens)
	xcos = np.array(xcos)
	ycos = np.array(ycos)
	x1s = np.multiply(np.subtract(rightxs, xoff1s), xcos)
	x2s = np.multiply(np.subtract(rightxs, xoff2s), xcos)
	y1s = np.multiply(np.subtract(topys, yoff1s), ycos)
	y2s = np.multiply(np.subtract(topys, yoff2s), ycos)
	startxs = np.multiply(leftxs, xcos)
	startys = np.multiply(botys, ycos)
	endxs = np.multiply(rightxs, xcos)
	endys = np.multiply(topys, ycos)
	for key, startx, starty, endx, endy, x1, y1, endx, endy, x2, y2 in zip(
		keys, startxs, startys, endxs, endys, x1s, y1s, endxs, endys, x2s, y2s
	):
		ret[key] = ([startx, starty, endx, endy], [x1, y1, endx, endy, x2, y2])
	return ret


def get_points(orig, dest, taillen):
	"""Return a pair of lists of points for use making an arrow.

	The first list is the beginning and end point of the trunk of the arrow.

	The second list is the arrowhead.

	"""
	p1 = _get_points_first_part(orig, dest, taillen)
	if len(p1) == 2:
		return p1
	ow, oh, dw, dh, xco, leftx, rightx, yco, topy, boty = p1
	rise = topy - boty
	run = rightx - leftx
	slope = rise / run
	unslope = run / rise
	start_theta = atan(slope)
	end_theta = atan(unslope)

	# make the little wedge at the end so you can tell which way the
	# arrow's pointing, and flip it all back around to the way it was
	top_theta = start_theta - fortyfive
	bot_theta = pi - fortyfive - end_theta
	xoff1 = cos(top_theta) * taillen
	yoff1 = sin(top_theta) * taillen
	xoff2 = cos(bot_theta) * taillen
	yoff2 = sin(bot_theta) * taillen
	x1 = (rightx - xoff1) * xco
	x2 = (rightx - xoff2) * xco
	y1 = (topy - yoff1) * yco
	y2 = (topy - yoff2) * yco
	startx = leftx * xco
	starty = boty * yco
	endx = rightx * xco
	endy = topy * yco
	return [startx, starty, endx, endy], [x1, y1, endx, endy, x2, y2]


def get_quad_vertices(
	ox,
	oy,
	dx,
	dy,
	x1,
	y1,
	endx,
	endy,
	x2,
	y2,
	bgr,
	fgr,
	label_w=100,
	label_h=100,
):
	if dx > ox:
		if dy > oy:
			label_pos = x2 - label_w, dy - label_h
		elif dy < oy:
			label_pos = x1 - label_w, dy
		else:
			label_pos = x2 - label_w, y2 - label_h
	elif dx < ox:
		if dy < oy:
			label_pos = x2, y2
		elif dy > oy:
			label_pos = x1, y1 - label_h
		else:
			label_pos = x2, dy - label_h
	else:
		if dy > oy:
			label_pos = x2 - label_w, dy - label_h
		elif dy < oy:
			label_pos = x2, y2
		else:
			label_pos = dx, dy
	return {
		"shaft_bg": get_thin_rect_vertices(ox, oy, dx, dy, bgr),
		"shaft_fg": get_thin_rect_vertices(ox, oy, dx, dy, fgr),
		"left_head_bg": get_thin_rect_vertices(x1, y1, endx, endy, bgr),
		"right_head_bg": get_thin_rect_vertices(x2, y2, endx, endy, bgr),
		"left_head_fg": get_thin_rect_vertices(x1, y1, endx, endy, fgr),
		"right_head_fg": get_thin_rect_vertices(x2, y2, endx, endy, fgr),
		"label_pos": label_pos,
	}


eight0s = tuple([0] * 8)


class GraphArrow:
	@property
	def slope(self) -> float:
		"""Return a float of the increase in y divided by the increase in x,
		both from left to right.

		Returns ``None`` when vertical.

		"""
		orig = self.origin
		dest = self.destination
		ox = orig.x
		oy = orig.y
		dx = dest.x
		dy = dest.y
		if oy == dy:
			return 0.0
		elif ox == dx:
			return None
		else:
			rise = dy - oy
			run = dx - ox
			return rise / run

	@property
	def y_intercept(self) -> Optional[float]:
		"""Return my Y-intercept.

		I probably don't really hit the left edge of the window, but
		this is where I would, if I were long enough.

		"""
		orig = self.origin
		dest = self.destination
		(ox, oy) = orig.pos
		(dx, dy) = dest.pos
		denominator = dx - ox
		x_numerator = (dy - oy) * ox
		y_numerator = denominator * oy
		return (y_numerator - x_numerator), denominator

	@property
	def reciprocal(self) -> Optional["GraphArrow"]:
		"""The arrow connecting the same spots in the opposite direction"""
		if self.destination.name not in self.board.pred_arrow:
			return
		if (
			self.origin.name
			not in self.board.pred_arrow[self.destination.name]
		):
			return
		return self.board.pred_arrow[self.destination.name][self.origin.name]

	@property
	def selected(self) -> bool:
		return self is self.board.app.selection

	@selected.setter
	def selected(self, b: bool):
		self.board.app.selection = self
		self.repoint(b)

	@property
	def character(self):
		return self.board.character

	def __init__(self, *, board, origin, destination):
		self.board = board
		self.origin = origin
		self.destination = destination

	def collide_point(self, x: float, y: float) -> bool:
		od = (self.origin.name, self.destination.name)
		if od not in self.board.arrow_plane._colliders_map:
			if od not in self.board.arrow_plane._instructions_map:
				shaft_points, head_points = get_points(
					self.origin,
					self.destination,
					self.board.arrow_plane.arrowhead_size,
				)
				r = self.board.arrow_plane.arrow_width / 2
				bg_points = get_quad_vertices(
					*shaft_points,
					*head_points,
					r * self.board.arrow_plane.bg_scale_unselected,
					r,
					0,
					0,
				)["shaft_bg"]
			else:
				bg_points = self.board.arrow_plane._instructions_map[od][
					"shaft_bg"
				].points
			self.board.arrow_plane._colliders_map[od] = Collide2DPoly(
				points=bg_points, cache=True
			)
		return (x, y) in self.board.arrow_plane._colliders_map[
			self.origin.name, self.destination.name
		]

	def pos_along(self, pct: float) -> Tuple[float, float]:
		"""Return coordinates for where a Pawn should be if it has travelled
		along ``pct`` of my length (between 0 and 1).

		Might get complex when I switch over to using beziers for
		arrows, but for now this is quite simple, using distance along
		a line segment.

		"""
		if pct < 0 or pct > 1:
			raise ValueError("Invalid portion")
		(ox, oy) = self.origin.center
		(dx, dy) = self.destination.center
		xdist = (dx - ox) * pct
		ydist = (dy - oy) * pct
		return ox + xdist, oy + ydist

	@mainthread
	def repoint(self, selected: bool = None) -> None:
		arrow_plane = self.board.arrow_plane
		fbo = arrow_plane._fbo
		fbo.bind()
		fbo.clear_buffer()
		shaft_points, head_points = get_points(
			self.origin, self.destination, arrow_plane.arrowhead_size
		)
		r = arrow_plane.arrow_width / 2
		if selected or self.selected:
			bg_scale = arrow_plane.bg_scale_selected
			bg_color = arrow_plane.bg_color_selected
			fg_color = arrow_plane.fg_color_selected
		else:
			bg_scale = arrow_plane.bg_scale_unselected
			bg_color = arrow_plane.bg_color_unselected
			fg_color = arrow_plane.fg_color_unselected
		plane = self.board.arrow_plane
		portal = self.board.character.portal[self.origin.name][
			self.destination.name
		]
		portal_text = str(portal.get(portal.get("_label_stat", None), ""))
		label_kwargs = dict(portal.get("label_kwargs", ()))
		if portal_text is not None:
			label_kwargs["text"] = portal_text
		try:
			label = self.board.arrow_plane._labels[self.origin.name][
				self.destination.name
			]
			label.text = portal_text
		except KeyError:
			label = self.board.arrow_plane.labels[self.origin.name][
				self.destination.name
			] = Label(**DEFAULT_ARROW_LABEL_KWARGS, **label_kwargs)
		if (
			self.origin.name,
			self.destination.name,
		) in plane._instructions_map:
			verts = get_quad_vertices(
				*shaft_points, *head_points, r * bg_scale, r, *label.render()
			)
			insts = self.board.arrow_plane._instructions_map[
				self.origin.name, self.destination.name
			]
			insts["color0"].rgba = bg_color
			insts["color1"].rgba = fg_color
			insts["shaft_bg"].points = verts["shaft_bg"]
			insts["left_head_bg"].points = verts["left_head_bg"]
			insts["right_head_bg"].points = verts["right_head_bg"]
			insts["shaft_fg"].points = verts["shaft_fg"]
			insts["left_head_fg"].points = verts["left_head_fg"]
			insts["right_head_fg"].points = verts["right_head_fg"]
			insts["label_rect"].pos = verts["label_pos"]
			insts["label_rect"].size = label.render()
			label.refresh()
			insts["label_rect"].texture = label.texture
			plane._colliders_map[self.origin.name, self.destination.name] = (
				Collide2DPoly(points=verts["shaft_bg"])
			)
		else:
			plane._instructions_map[
				self.origin.name, self.destination.name
			] = insts = get_instructions(
				*shaft_points,
				*head_points,
				bg_color,
				fg_color,
				*label.render(),
				label,
			)
			plane._colliders_map[
				self.origin.name, self.destination.name
			].points = Collide2DPoly(points=insts["shaft_bg"].points)
		myidx = plane._port_index[self.origin.name, self.destination.name]
		(ox, oy, dx, dy) = shaft_points
		plane._bot_left_corner_xs[myidx] = min((ox, dx))
		plane._bot_left_corner_ys[myidx] = min((oy, dy))
		plane._top_right_corner_xs[myidx] = max((ox, dx))
		plane._top_right_corner_ys[myidx] = max((oy, dy))
		fbo.release()
		fbo.ask_update()
		arrow_plane.canvas.ask_update()


def get_instructions(
	ox: float,
	oy: float,
	dx: float,
	dy: float,
	x1: float,
	y1: float,
	endx: float,
	endy: float,
	x2: float,
	y2: float,
	bgr: float,
	r: float,
	bg_color: Tuple[float, float, float, float],
	fg_color: Tuple[float, float, float, float],
	label_kwargs: dict = None,
	label: Label = None,
) -> dict:
	if label_kwargs is None:
		label_kwargs = {"text": "", **DEFAULT_ARROW_LABEL_KWARGS}
	else:
		label_kwargs = dict(label_kwargs)
		for k, v in DEFAULT_ARROW_LABEL_KWARGS.items():
			if k not in label_kwargs:
				label_kwargs[k] = v
	if label is None:
		label = Label(**label_kwargs)
	else:
		label.text = label_kwargs["text"]
	text_size = label.render()
	quadverts = get_quad_vertices(
		ox, oy, dx, dy, x1, y1, endx, endy, x2, y2, bgr, r, *text_size
	)
	label.refresh()
	return {
		"color0": Color(rgba=bg_color),
		"shaft_bg": Quad(points=quadverts["shaft_bg"]),
		"left_head_bg": Quad(points=quadverts["left_head_bg"]),
		"right_head_bg": Quad(points=quadverts["right_head_bg"]),
		"color1": Color(rgba=fg_color),
		"shaft_fg": Quad(points=quadverts["shaft_fg"]),
		"left_head_fg": Quad(points=quadverts["left_head_fg"]),
		"right_head_fg": Quad(points=quadverts["right_head_fg"]),
		"label_rect": Rectangle(
			pos=quadverts["label_pos"], size=text_size, texture=label.texture
		),
		"label": label,
	}


class GraphArrowWidget(Widget, GraphArrow):
	arrowhead_size = NumericProperty(10)
	arrow_width = NumericProperty(2)
	bg_scale = NumericProperty(5)
	bg_color = ListProperty([0.5, 0.5, 0.5, 0.5])
	fg_color = ListProperty([1.0, 1.0, 1.0, 1.0])

	def __init__(self, **kwargs):
		self._trigger_repoint = Clock.create_trigger(self.repoint)
		super().__init__(**kwargs)

	def on_origin(self, *args):
		if hasattr(self, "_origin_bind_uid"):
			self.unbind_uid(self._origin_bind_uid)
			del self._origin_bind_uid
		if not self.origin:
			return
		self._origin_bind_uid = self.origin.fbind("pos", self._trigger_repoint)

	def on_destination(self, *args):
		if hasattr(self, "_destination_bind_uid"):
			self.unbind_uid(self._destination_bind_uid)
			del self._destination_bind_uid
		if not self.destination:
			return
		self._destination_bind_uid = self.destination.fbind(
			"pos", self._trigger_repoint
		)

	def on_parent(self, *args):
		if not self.origin or not self.destination or not self.canvas:
			Clock.schedule_once(self.on_parent, 0)
			return
		self.canvas.clear()
		if self.parent is None:
			return
		shaft_points, head_points = get_points(
			self.origin, self.destination, self.arrowhead_size
		)
		with self.canvas:
			r = self.arrow_width / 2
			self._instructions = get_instructions(
				*shaft_points,
				*head_points,
				r * self.bg_scale,
				r,
				self.bg_color,
				self.fg_color,
			)

	def repoint(self, *args):
		shaft_points, head_points = get_points(
			self.origin, self.destination, self.arrowhead_size
		)
		r = self.arrow_width / 2
		try:
			portal = self.board.character.portal[self.origin.name][
				self.destination.name
			]
			portal_text = str(portal.get(portal.get("_label_stat", None), ""))
		except (KeyError, AttributeError):
			portal_text = ""
		if hasattr(self, "_label"):
			label = self._label
			label.text = portal_text
		else:
			label = self._label = Label(text=portal_text)
		label_size = label.render()
		verts = get_quad_vertices(
			*shaft_points, *head_points, r * self.bg_scale, r, *label_size
		)
		insts = self._instructions
		insts["color0"].rgba = self.bg_color
		insts["color1"].rgba = self.fg_color
		insts["shaft_bg"].points = verts["shaft_bg"]
		insts["left_head_bg"].points = verts["left_head_bg"]
		insts["right_head_bg"].points = verts["right_head_bg"]
		insts["shaft_fg"].points = verts["shaft_fg"]
		insts["left_head_fg"].points = verts["left_head_fg"]
		insts["right_head_fg"].points = verts["right_head_fg"]
		insts["label_rect"].pos = verts["label_pos"]
		insts["label_rect"].size = label_size
		label.refresh()
		insts["label_rect"].texture = label.texture


class ArrowPlane(Widget):
	data = ListProperty()
	arrowhead_size = NumericProperty(10)
	arrow_width = NumericProperty(2)
	bg_scale_unselected = NumericProperty(4)
	bg_scale_selected = NumericProperty(5)
	bg_color_selected = ListProperty([0.0, 1.0, 1.0, 1.0])
	bg_color_unselected = ListProperty([0.5, 0.5, 0.5, 0.5])
	fg_color_unselected = ListProperty([1.0, 1.0, 1.0, 1.0])
	fg_color_selected = ListProperty([1.0, 1.0, 1.0, 1.0])

	def __init__(self, **kwargs):
		self._labels = defaultdict(lambda: defaultdict(lambda: None))
		self._trigger_redraw = Clock.create_trigger(self.redraw)
		self._redraw_bind_uid = self.fbind("data", self._trigger_redraw)
		self.bind(arrowhead_size=self._trigger_redraw)
		self._colliders_map = {}
		self._instructions_map = {}
		self._port_index = {}
		self._port_l = []
		self._bot_left_corner_ys = np.array([])
		self._bot_left_corner_xs = np.array([])
		self._top_right_corner_ys = np.array([])
		self._top_right_corner_xs = np.array([])
		super().__init__(**kwargs)

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
		self.redraw()

	def have_arrow(self, orig, dest):
		return (orig, dest) in self._port_index

	def redraw(self, *args):
		if not hasattr(self, "_rectangle"):
			self._trigger_redraw()
			return
		fbo = self._fbo
		fbo.bind()
		fbo.clear()
		fbo.clear_buffer()
		add = fbo.add
		r = self.arrow_width // 2
		bg_scale_selected = self.bg_scale_selected
		bg_color_unselected = self.bg_color_unselected
		fg_color_unselected = self.fg_color_unselected
		taillen = self.arrowhead_size
		points_map = get_points_multi(
			(datum["origspot"], datum["destspot"], taillen)
			for datum in self.data
		)
		port_l = []
		bot_left_corner_xs = []
		bot_left_corner_ys = []
		top_right_corner_xs = []
		top_right_corner_ys = []
		port_index = self._port_index
		colliders_map = self._colliders_map
		oxs = []
		oys = []
		dxs = []
		dys = []
		for (ox, oy, dx, dy), _ in points_map.values():
			oxs.append(ox)
			oys.append(oy)
			dxs.append(dx)
			dys.append(dy)
		widths = np.abs(np.array(dxs) - np.array(oxs))
		heights = np.abs(np.array(dys) - np.array(oys))
		lengths = np.sqrt(np.square(widths) + np.square(heights))
		for length, (
			port,
			((ox, oy, dx, dy), (x1, y1, endx, endy, x2, y2)),
		), datum in zip(lengths, points_map.items(), self.data):
			if length < r:
				continue
			bgr = r * bg_scale_selected  # change for selectedness pls
			instructions = get_instructions(
				ox,
				oy,
				dx,
				dy,
				x1,
				y1,
				endx,
				endy,
				x2,
				y2,
				bgr,
				r,
				bg_color_unselected,
				fg_color_unselected,
				datum.get("label_kwargs", {}),
				self._labels[port[0]].get(port[1]),
			)
			instructions["group"] = grp = InstructionGroup()
			grp.add(instructions["color0"])
			grp.add(instructions["shaft_bg"])
			grp.add(instructions["left_head_bg"])
			grp.add(instructions["right_head_bg"])
			grp.add(instructions["color1"])
			grp.add(instructions["shaft_fg"])
			grp.add(instructions["left_head_fg"])
			grp.add(instructions["right_head_fg"])
			grp.add(instructions["label_rect"])
			add(grp)
			self._instructions_map[port] = instructions
			self._labels[port[0]][port[1]] = instructions["label"]
			if ox < dx:
				leftx = ox
				rightx = dx
			else:
				rightx = ox
				leftx = dx
			if oy < dy:
				boty = oy
				topy = dy
			else:
				boty = dy
				topy = oy
			port_index[port] = len(port_l)
			port_l.append(port)
			bot_left_corner_xs.append(leftx - bgr)
			bot_left_corner_ys.append(boty - bgr)
			top_right_corner_xs.append(rightx + bgr)
			top_right_corner_ys.append(topy + bgr)
			colliders_map[port] = Collide2DPoly(
				points=instructions["shaft_bg"].points
			)
		fbo.release()
		self.canvas.ask_update()
		self._port_l = port_l
		self._bot_left_corner_xs = np.array(bot_left_corner_xs)
		self._bot_left_corner_ys = np.array(bot_left_corner_ys)
		self._top_right_corner_xs = np.array(top_right_corner_xs)
		self._top_right_corner_ys = np.array(top_right_corner_ys)

	def add_new_portal(self, datum: dict) -> None:
		orig_spot = datum["origspot"]
		dest_spot = datum["destspot"]
		shaft_points, head_points = get_points(
			orig_spot, dest_spot, self.arrowhead_size
		)
		self.unbind_uid("data", self._redraw_bind_uid)
		self.data.append(datum)
		r = self.arrow_width / 2
		bgr = r * self.bg_scale_unselected
		instructions = self._instructions_map[
			orig_spot.name, dest_spot.name
		] = get_instructions(
			*shaft_points,
			*head_points,
			bgr,
			r,
			self.bg_color_unselected,
			self.fg_color_unselected,
			datum.get("label_kwargs", {}),
			self._labels[orig_spot.name][dest_spot.name],
		)
		instructions["group"] = grp = InstructionGroup()
		grp.add(instructions["color0"])
		grp.add(instructions["shaft_bg"])
		grp.add(instructions["left_head_bg"])
		grp.add(instructions["right_head_bg"])
		grp.add(instructions["color1"])
		grp.add(instructions["shaft_fg"])
		grp.add(instructions["left_head_fg"])
		grp.add(instructions["right_head_fg"])
		# grp.add(instructions["label"])
		self._labels[orig_spot.name][dest_spot.name] = instructions["label"]
		self._fbo.add(grp)
		ox, oy, dx, dy = shaft_points
		if ox < dx:
			left_x = ox
			right_x = dx
		else:
			right_x = ox
			left_x = dx
		if oy < dy:
			bot_y = oy
			top_y = dy
		else:
			bot_y = dy
			top_y = oy
		self._port_index[orig_spot.name, dest_spot.name] = len(self._port_l)
		self._port_l.append((orig_spot.name, dest_spot.name))
		self._bot_left_corner_xs = np.array(
			list(self._bot_left_corner_xs) + [left_x - bgr]
		)
		self._bot_left_corner_ys = np.array(
			list(self._bot_left_corner_ys) + [bot_y - bgr]
		)
		self._top_right_corner_xs = np.array(
			list(self._top_right_corner_xs) + [right_x + bgr]
		)
		self._top_right_corner_ys = np.array(
			list(self._top_right_corner_ys) + [top_y + bgr]
		)
		self._colliders_map[orig_spot.name, dest_spot.name] = Collide2DPoly(
			points=instructions["shaft_bg"].points
		)
		self.canvas.ask_update()
		self._redraw_bind_uid = self.fbind("data", self._trigger_redraw)

	def remove_edge(self, orig, dest):
		self._fbo.bind()
		self._fbo.clear_buffer()
		self._fbo.remove(self._instructions_map[orig, dest]["group"])
		index = self._port_index[orig, dest]
		for port in self._port_l[index:]:
			self._port_index[port] -= 1
		del self._port_index[orig, dest]
		del self._instructions_map[orig, dest]
		del self._colliders_map[orig, dest]
		del self._port_l[index]
		for arr in (
			"_bot_left_corner_ys",
			"_bot_left_corner_xs",
			"_top_right_corner_ys",
			"_top_right_corner_xs",
		):
			dat = list(getattr(self, arr))
			del dat[index]
			setattr(self, arr, np.array(dat))
		self._fbo.release()
		self.canvas.ask_update()

	def update_portal_label(self, orig, dest, text):
		rect = self._instructions_map[orig, dest]["label"]
		label = self._labels[orig][dest]
		label.text = text
		label.refresh()
		rect.texture = label.texture

	def iter_collided_edges(self, x: float, y: float) -> Iterator:
		x, y = map(float, (x, y))
		collider_map = self._colliders_map
		hits = (
			(self._bot_left_corner_xs <= x)
			& (self._bot_left_corner_ys <= y)
			& (x <= self._top_right_corner_xs)
			& (y <= self._top_right_corner_ys)
		)
		for port in map(
			itemgetter(0), filter(itemgetter(1), zip(self._port_l, hits))
		):
			if collider_map[port].collide_point(x, y):
				yield port

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


def get_thin_rect_vertices(ox, oy, dx, dy, r):
	"""Given the starting point, ending point, and width, return a list of
	vertex coordinates at the corners of the line segment
	(really a thin rectangle).

	"""
	if ox < dx:
		leftx = ox
		rightx = dx
		xco = 1
	elif ox > dx:
		leftx = ox * -1
		rightx = dx * -1
		xco = -1
	else:
		return [ox - r, oy, ox + r, oy, ox + r, dy, ox - r, dy]
	if oy < dy:
		boty = oy
		topy = dy
		yco = 1
	elif oy > dy:
		boty = oy * -1
		topy = dy * -1
		yco = -1
	else:
		return [ox, oy - r, dx, oy - r, dx, oy + r, ox, oy + r]

	rise = topy - boty
	run = rightx - leftx
	theta = atan(rise / run)
	theta_prime = ninety - theta
	xoff = cos(theta_prime) * r
	yoff = sin(theta_prime) * r
	x1 = leftx + xoff
	y1 = boty - yoff
	x2 = rightx + xoff
	y2 = topy - yoff
	x3 = rightx - xoff
	y3 = topy + yoff
	x4 = leftx - xoff
	y4 = boty + yoff
	return [
		x1 * xco,
		y1 * yco,
		x2 * xco,
		y2 * yco,
		x3 * xco,
		y3 * yco,
		x4 * xco,
		y4 * yco,
	]


ninety = pi / 2
DEFAULT_ARROW_LABEL_KWARGS = {"font_size": 16, "bold": True}
