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
"""That which displays a one-way connection between two places.

An arrow connects two spots, the origin and the destination, and it
points from the origin to the destination, regardless of where on the
screen they are at the moment.

"""
from math import cos, sin, atan, pi
from operator import itemgetter

import numpy as np
from kivy.uix.widget import Widget
from kivy.graphics.fbo import Fbo
from kivy.graphics import (Translate, Rectangle, Quad, Color, InstructionGroup)
from kivy.properties import (NumericProperty, ListProperty)
from kivy.clock import Clock

try:
	from kivy.garden.collider import Collide2DPoly
except (KeyError, ImportError):
	from ..collide import Collide2DPoly
from ..util import get_thin_rect_vertices, fortyfive

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
	for (orig, dest, taillen) in args:
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
		keys, startxs, startys, endxs, endys, x1s, y1s, endxs, endys, x2s,
		y2s):
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


def get_quad_vertices(ox, oy, dx, dy, x1, y1, endx, endy, x2, y2, bgr, fgr):
	return {
		'shaft_bg': get_thin_rect_vertices(ox, oy, dx, dy, bgr),
		'shaft_fg': get_thin_rect_vertices(ox, oy, dx, dy, fgr),
		'left_head_bg': get_thin_rect_vertices(x1, y1, endx, endy, bgr),
		'right_head_bg': get_thin_rect_vertices(x2, y2, endx, endy, bgr),
		'left_head_fg': get_thin_rect_vertices(x1, y1, endx, endy, fgr),
		'right_head_fg': get_thin_rect_vertices(x2, y2, endx, endy, fgr)
	}


eight0s = tuple([0] * 8)


class GraphArrow:

	@property
	def origin(self):
		return self.board.spot[self.portal['origin']]

	@property
	def destination(self):
		return self.board.spot[self.portal['destination']]

	@property
	def slope(self):
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
			return 0.
		elif ox == dx:
			return None
		else:
			rise = dy - oy
			run = dx - ox
			return rise / run

	@property
	def y_intercept(self):
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
	def reciprocal(self):
		if self.portal['destination'] not in self.board.pred_arrow:
			return
		if self.portal['origin'] not in self.board.pred_arrow[
			self.portal['destination']]:
			return
		return self.board.pred_arrow[self.portal['destination']][
			self.portal['origin']]

	def __init__(self, board, portal):
		self.board = board
		self.portal = portal

	def collide_point(self, x, y):
		return self.board.arrow_plane._colliders_map[
			self.portal['origin'],
			self.portal['destination']].collide_point(x, y)

	def pos_along(self, pct):
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
		self._trigger_redraw = Clock.create_trigger(self.redraw)
		self.bind(data=self._trigger_redraw,
					arrowhead_size=self._trigger_redraw)
		self._colliders_map = {}
		self._instructions_map = {}
		self._port_index = {}
		self._port_l = []
		self._bot_left_corner_ys = []
		self._bot_left_corner_xs = []
		self._top_right_corner_ys = []
		self._top_right_corner_xs = []
		super().__init__(**kwargs)

	def on_parent(self, *args):
		if not self.canvas:
			Clock.schedule_once(self.on_parent, 0)
			return
		with self.canvas:
			self._fbo = Fbo(size=self.size)
			self._translate = Translate(x=self.x, y=self.y)
			self._rectangle = Rectangle(size=self.size,
										texture=self._fbo.texture)
		self._trigger_redraw()

	def redraw(self, *args):
		if not hasattr(self, '_rectangle'):
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
			(datum['origspot'], datum['destspot'], taillen)
			for datum in self.data)
		port_l = []
		bot_left_corner_xs = []
		bot_left_corner_ys = []
		top_right_corner_xs = []
		top_right_corner_ys = []
		port_index = self._port_index
		colliders_map = self._colliders_map
		for port, ((ox, oy, dx, dy), (x1, y1, endx, endy, x2,
										y2)) in points_map.items():
			bgr = r * bg_scale_selected  # change for selectedness pls
			quadverts = get_quad_vertices(ox, oy, dx, dy, x1, y1, endx, endy,
											x2, y2, bgr, r)
			instructions = {
				'color0': Color(rgba=bg_color_unselected),
				'shaft_bg': Quad(points=quadverts['shaft_bg']),
				'left_head_bg': Quad(points=quadverts['left_head_bg']),
				'right_head_bg': Quad(points=quadverts['right_head_bg']),
				'color1': Color(rgba=fg_color_unselected),
				'shaft_fg': Quad(points=quadverts['shaft_fg']),
				'left_head_fg': Quad(points=quadverts['left_head_fg']),
				'right_head_fg': Quad(points=quadverts['right_head_fg']),
			}
			instructions['group'] = grp = InstructionGroup()
			grp.add(instructions['color0'])
			grp.add(instructions['shaft_bg'])
			grp.add(instructions['left_head_bg'])
			grp.add(instructions['right_head_bg'])
			grp.add(instructions['color1'])
			grp.add(instructions['shaft_fg'])
			grp.add(instructions['left_head_fg'])
			grp.add(instructions['right_head_fg'])
			add(grp)
			self._instructions_map[port] = instructions
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
			colliders_map[port] = Collide2DPoly(points=quadverts['shaft_bg'])
		fbo.release()
		self.canvas.ask_update()
		self._port_l = port_l
		self._bot_left_corner_xs = np.array(bot_left_corner_xs)
		self._bot_left_corner_ys = np.array(bot_left_corner_ys)
		self._top_right_corner_xs = np.array(top_right_corner_xs)
		self._top_right_corner_ys = np.array(top_right_corner_ys)

	def remove_edge(self, orig, dest):
		self._fbo.remove(self._instructions_map[orig, dest]['group'])
		index = self._port_index[orig, dest]
		del self._port_index[orig, dest]
		del self._instructions_map[orig, dest]
		del self._colliders_map[orig, dest]
		del self._port_l[index]
		for arr in ('_bot_left_corner_ys', '_bot_left_corner_xs',
					'_top_right_corner_ys', '_top_right_corner_xs'):
			dat = list(getattr(self, arr))
			del dat[index]
			setattr(self, arr, np.array(dat))
		self.canvas.ask_update()

	def iter_collided_edges(self, x, y):
		collider_map = self._colliders_map
		hits = (self._bot_left_corner_xs <= x) & (
			self._bot_left_corner_ys <= y
		) & (x <= self._top_right_corner_xs) & (y <= self._top_right_corner_ys)
		for port in map(itemgetter(0),
						filter(itemgetter(1), zip(self._port_l, hits))):
			if collider_map[port].collide_point(x, y):
				yield port

	def on_pos(self, *args):
		if not hasattr(self, '_translate'):
			return
		self._translate.x, self._translate.y = self.pos
		self.canvas.ask_update()

	def on_size(self, *args):
		if not hasattr(self, '_rectangle') or not hasattr(self, '_fbo'):
			return
		self._rectangle.size = self._fbo.size = self.size
		self.redraw()
