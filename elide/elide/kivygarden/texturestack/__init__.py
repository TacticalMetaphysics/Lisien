# This file is part of the kivy-garden project.
# Copyright (c) Zachary Spector, public@zacharyspector.com
# Available under the terms of the MIT license.
"""Several textures superimposed on one another, and possibly offset
by some amount.

In 2D games where characters can wear different clothes or hold
different equipment, their graphics are often composed of several
graphics layered on one another. This widget simplifies the management
of such compositions.

"""

from operator import itemgetter
from time import monotonic

import numpy as np
from kivy import Logger
from kivy.clock import Clock, mainthread
from kivy.core.image import Image
from kivy.graphics import (
	Color,
	Fbo,
	InstructionGroup,
	Line,
	PopMatrix,
	PushMatrix,
	Rectangle,
	Translate,
)
from kivy.graphics.fbo import Fbo
from kivy.properties import (
	AliasProperty,
	DictProperty,
	ListProperty,
	ObjectProperty,
	StringProperty,
)
from kivy.resources import resource_find
from kivy.uix.widget import Widget


class TextureStack(Widget):
	"""Several textures superimposed on one another, and possibly offset
	by some amount.

	In 2D games where characters can wear different clothes or hold
	different equipment, their graphics are often composed of several
	graphics layered on one another. This widget simplifies the
	management of such compositions.

	"""

	texs = ListProperty()
	"""Texture objects"""
	offxs = ListProperty()
	"""x-offsets. The texture at the same index will be moved to the right
    by the number of pixels in this list.

    """
	offys = ListProperty()
	"""y-offsets. The texture at the same index will be moved upward by
    the number of pixels in this list.

    """
	group = ObjectProperty()
	"""My ``InstructionGroup``, suitable for addition to whatever ``canvas``."""

	def _get_offsets(self):
		return zip(self.offxs, self.offys)

	def _set_offsets(self, offs):
		offxs = []
		offys = []
		for x, y in offs:
			offxs.append(x)
			offys.append(y)
		self.offxs, self.offys = offxs, offys

	offsets = AliasProperty(
		_get_offsets, _set_offsets, bind=("offxs", "offys")
	)
	"""List of (x, y) tuples by which to offset the corresponding texture."""
	_texture_rectangles = DictProperty({})
	"""Private.

    Rectangle instructions for each of the textures, keyed by the
    texture.

    """

	def __init__(self, **kwargs):
		"""Make triggers and bind."""
		kwargs["size_hint"] = (None, None)
		self.translate = Translate(0, 0)
		self.group = InstructionGroup()
		super().__init__(**kwargs)
		self.bind(offxs=self.on_pos, offys=self.on_pos)

	def on_texs(self, *args):
		"""Make rectangles for each of the textures and add them to the canvas."""
		if not self.canvas or not self.texs:
			Clock.schedule_once(self.on_texs, 0)
			return
		texlen = len(self.texs)
		# Ensure each property is the same length as my texs, padding
		# with 0 as needed
		for prop in ("offxs", "offys"):
			proplen = len(getattr(self, prop))
			if proplen > texlen:
				setattr(self, prop, getattr(self, prop)[: proplen - texlen])
			if texlen > proplen:
				propval = list(getattr(self, prop))
				propval += [0] * (texlen - proplen)
				setattr(self, prop, propval)
		self.canvas.remove(self.group)
		self.group = InstructionGroup()
		self._texture_rectangles = {}
		w = h = 0
		(x, y) = self.pos
		self.translate.x = x
		self.translate.y = y
		self.group.add(PushMatrix())
		self.group.add(self.translate)
		for tex, offx, offy in zip(self.texs, self.offxs, self.offys):
			rect = Rectangle(pos=(offx, offy), size=tex.size, texture=tex)
			self._texture_rectangles[tex] = rect
			self.group.add(rect)
			tw = tex.width + offx
			th = tex.height + offy
			if tw > w:
				w = tw
			if th > h:
				h = th
		self.size = (w, h)
		self.group.add(PopMatrix())
		self.canvas.add(self.group)
		self.canvas.ask_update()

	def on_pos(self, *args):
		"""Translate all the rectangles within this widget to reflect the widget's position."""
		(x, y) = self.pos
		self.translate.x = x
		self.translate.y = y

	def clear(self):
		"""Clear my rectangles and ``texs``."""
		self.group.clear()
		self._texture_rectangles = {}
		self.texs = []
		self.size = [1, 1]

	def insert(self, i, tex):
		"""Insert the texture into my ``texs``, waiting for the creation of
		the canvas if necessary.

		"""
		if not self.canvas:
			Clock.schedule_once(lambda dt: self.insert(i, tex), 0)
			return
		self.texs.insert(i, tex)

	def append(self, tex):
		"""``self.insert(len(self.texs), tex)``"""
		self.insert(len(self.texs), tex)

	def __delitem__(self, i):
		"""Remove a texture and its rectangle"""
		tex = self.texs[i]
		try:
			rect = self._texture_rectangles[tex]
			self.canvas.remove(rect)
			del self._texture_rectangles[tex]
		except KeyError:
			pass
		del self.texs[i]

	def __setitem__(self, i, v):
		"""First delete at ``i``, then insert there"""
		if len(self.texs) > 0:
			self._no_upd_texs = True
			self.__delitem__(i)
			self._no_upd_texs = False
		self.insert(i, v)

	def pop(self, i=-1):
		"""Delete the texture at ``i``, and return it."""
		return self.texs.pop(i)


class ImageStack(TextureStack):
	"""Instead of supplying textures themselves, supply paths to where the
	textures may be loaded from.

	"""

	paths = ListProperty()
	"""List of paths to images you want stacked."""
	pathtexs = DictProperty()
	"""Private. Dictionary mapping image paths to textures of the images."""
	pathimgs = DictProperty()
	"""Dictionary mapping image paths to ``kivy.core.Image`` objects."""

	def on_paths(self, *args):
		"""Make textures from the images in ``paths``, and assign them at the
		same index in my ``texs`` as in my ``paths``.

		"""
		for i, path in enumerate(self.paths):
			if path in self.pathtexs:
				if (
					self.pathtexs[path] in self.texs
					and self.texs.index(self.pathtexs[path]) == i
				):
					continue
			else:
				try:
					self.pathimgs[path] = img = Image.load(
						resource_find(path), keep_data=True
					)
				except Exception:
					self.pathimgs[path] = img = Image.load(
						resource_find("atlas://rltiles/misc/floppy"),
						keep_data=True,
					)
				self.pathtexs[path] = img.texture
			if i == len(self.texs):
				self.texs.append(self.pathtexs[path])
			else:
				self.texs[i] = self.pathtexs[path]

	def clear(self):
		"""Clear paths, textures, rectangles"""
		self.paths = []
		super().clear()

	def insert(self, i, v):
		"""Insert a string to my paths"""
		if not isinstance(v, str):
			raise TypeError("Paths only")
		self.paths.insert(i, v)

	def __delitem__(self, i):
		"""Delete texture, rectangle, path"""
		super().__delitem__(i)
		del self.paths[i]

	def pop(self, i=-1):
		"""Delete and return a path"""
		r = self.paths[i]
		del self[i]
		return r


class TextureStackBatchWidget(Widget):
	"""Widget for efficiently drawing many TextureStacks

	Only add TextureStack or ImageStack widgets to this. Avoid adding
	any that are to be changed frequently.

	"""

	critical_props = ["texs", "offxs", "offys", "pos"]
	"""Properties that, when changed on my children, force a redraw."""

	def __init__(self, **kwargs):
		self._trigger_redraw = Clock.create_trigger(self.redraw)
		self._trigger_rebind_children = Clock.create_trigger(
			self.rebind_children
		)
		super(TextureStackBatchWidget, self).__init__(**kwargs)

	def on_parent(self, *args):
		if not self.canvas:
			Clock.schedule_once(self.on_parent, 0)
			return
		if not hasattr(self, "_fbo"):
			with self.canvas:
				self._fbo = Fbo(size=self.size)
				self._fbo.add_reload_observer(self.redraw)
				self._translate = Translate(x=self.x, y=self.y)
				self._rectangle = Rectangle(
					texture=self._fbo.texture, size=self.size
				)
		self.rebind_children()

	def rebind_children(self, *args):
		child_by_uid = {}
		binds = {prop: self._trigger_redraw for prop in self.critical_props}
		for child in self.children:
			child_by_uid[child.uid] = child
			child.bind(**binds)
		if hasattr(self, "_old_children"):
			old_children = self._old_children
			for uid in set(old_children).difference(child_by_uid):
				old_children[uid].unbind(**binds)
		self.redraw()
		self._old_children = child_by_uid

	def redraw(self, *args):
		fbo = self._fbo
		fbo.bind()
		fbo.clear()
		fbo.clear_buffer()
		fbo.release()
		for child in self.children:
			assert child.canvas not in fbo.children
			fbo.add(child.canvas)

	def on_pos(self, *args):
		if not hasattr(self, "_translate"):
			return
		self._translate.x, self._translate.y = self.pos

	def on_size(self, *args):
		if not hasattr(self, "_rectangle"):
			return
		self._rectangle.size = self._fbo.size = self.size
		self.redraw()

	def add_widget(self, widget, index=0, canvas=None):
		if not isinstance(widget, TextureStack):
			raise TypeError("TextureStackBatch is only for TextureStack")
		if index == 0 or len(self.children) == 0:
			self.children.insert(0, widget)
		else:
			children = self.children
			if index >= len(children):
				index = len(children)

			children.insert(index, widget)
		widget.parent = self
		if hasattr(self, "_fbo"):
			self.rebind_children()

	def remove_widget(self, widget):
		if widget not in self.children:
			return
		self.children.remove(widget)
		widget.parent = None
		if hasattr(self, "_fbo"):
			self.rebind_children()


class TextureStackPlane(Widget):
	"""A widget full of Stacks, which are like TextureStack but more performant

	Stacks aren't technically :class:`Widget`s, though they have similar
	properties. This is because, when you have thousands of them, :class:`Widget`
	performs badly.

	"""

	data = ListProperty()
	selected = ObjectProperty(allownone=True)
	color_selected = ListProperty([0.0, 1.0, 1.0, 1.0])
	default_image_path = StringProperty("atlas://rltiles/base.atlas/unseen")

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
		self._redraw_bind_data_uid = self.fbind("data", self._trigger_redraw)
		self._redraw_bind_size_uid = self.fbind("size", self._trigger_redraw)

	def on_parent(self, *_):
		if not self.canvas:
			Clock.schedule_once(self.on_parent, 0)
			return
		with self.canvas:
			self._fbo = Fbo(size=self.size)
			self._push = PushMatrix()
			self._translate = Translate(x=self.x, y=self.y)
			self._rectangle = Rectangle(
				size=self.size, texture=self._fbo.texture
			)
			self._pop = PopMatrix()
		self.bind(pos=self._trigger_redraw, size=self._trigger_redraw)
		self._trigger_redraw()

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
		self.unbind_uid("data", self._redraw_bind_data_uid)
		self.data.append(datum)
		self._redraw_bind_data_uid = self.fbind("data", self._trigger_redraw)
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
		self.unbind_uid("data", self._redraw_bind_data_uid)
		del self.data[idx]
		self._redraw_bind_data_uid = self.fbind("data", self._trigger_redraw)
		self._remove_upd_fbo(name)

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
		if not self.data:
			if hasattr(self, "_fbo"):
				self._fbo.clear()
			return
		Logger.debug(
			f"TextureStackPlane: redrawing, with {self.selected} selected"
		)
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
		self._fbo.size = self.size
		fbo = self._fbo
		for insts in todo:
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
		for insts in get_rid:
			fbo.remove(insts["group"])
		self._rectangle.texture = fbo.texture
		self._translate.x, self._translate.y = self.pos
		self._rectangle.size = self._fbo.size
		Logger.debug(
			f"TextureStackPlane: redrawn in "
			f"{monotonic() - start_ts:,.2f} seconds"
		)

	def iter_collided_keys(self, x, y):
		"""Iterate over the keys of stacks that collide with the given point"""
		hits = (
			(self._left_xs <= x)
			& (self._bot_ys <= y)
			& (y <= self._top_ys)
			& (x <= self._right_xs)
		)
		return map(itemgetter(0), filter(itemgetter(1), zip(self._keys, hits)))


class Stack:
	__slots__ = ["board", "_name", "proxy", "__self__"]

	default_image_paths = ["atlas://rltiles/floor.atlas/floor-normal"]

	def __init__(self, **kwargs):
		self.board = kwargs["board"]
		if "proxy" in kwargs:
			self.proxy = kwargs["proxy"]
			self._name = kwargs["proxy"]["name"]
		elif "name" in kwargs:
			self._name = kwargs["name"]
		else:
			raise TypeError("Stacks need names")

	@property
	def paths(self):
		name = self.name
		plane = self._stack_plane
		datum = plane.data[plane._stack_index[name]]
		return datum["textures"]

	@paths.setter
	@mainthread
	def paths(self, v):
		name = self.name
		plane = self._stack_plane
		datum = plane.data[plane._stack_index[name]]
		plane.unbind_uid("data", plane._redraw_bind_data_uid)
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
		plane._redraw_bind_data_uid = plane.fbind(
			"data", plane._trigger_redraw
		)

	@property
	def selected(self):
		return self._stack_plane.selected == self.name

	@selected.setter
	@mainthread
	def selected(self, v: bool):
		stack_plane: TextureStackPlane = self._stack_plane
		name = self.name
		Logger.debug(f"Stack: {name} selected")
		insts = stack_plane._instructions[name]
		fbo = stack_plane._fbo
		fbo.bind()
		fbo.clear_buffer()
		if v:
			stack_plane.selected = name
			if "color0" in insts:
				Logger.debug(
					f"Stack: changing {name}'s color to {stack_plane.color_selected}"
				)
				insts["color0"].rgba = stack_plane.color_selected
			else:
				Logger.debug(
					f"Stack: creating Color(rgba={stack_plane.color_selected}) for {name}"
				)
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
			if stack_plane.selected == self.name:
				stack_plane.selected = None
			if "color0" in insts:
				insts["color0"].rgba = [0.0, 0.0, 0.0, 0.0]
		fbo.release()

	@property
	def pos(self):
		stack_plane = self._stack_plane
		idx = stack_plane._stack_index[self.name]
		return float(stack_plane._left_xs[idx]), float(
			stack_plane._bot_ys[idx]
		)

	@pos.setter
	@mainthread
	def pos(self, xy):
		x, y = xy
		stack_plane = self._stack_plane
		stack_plane.unbind_uid("data", stack_plane._redraw_bind_data_uid)
		name = self.name
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
		stack_plane._redraw_bind_data_uid = stack_plane.fbind(
			"data", stack_plane._trigger_redraw
		)
		stack_plane._fbo.release()

	@property
	def _stack_plane(self):
		return self.board.stack_plane

	@property
	def x(self):
		stack_plane = self._stack_plane
		idx = stack_plane._stack_index[self.name]
		return float(stack_plane._left_xs[idx])

	@x.setter
	def x(self, x):
		self.pos = x, self.y

	@property
	def y(self):
		stack_plane = self._stack_plane
		idx = stack_plane._stack_index[self.name]
		return float(stack_plane._bot_ys[idx])

	@y.setter
	def y(self, y):
		self.pos = self.x, y

	@property
	def size(self):
		stack_plane = self._stack_plane
		name = self.name
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
		stack_plane.unbind_uid("data", stack_plane._redraw_bind_data_uid)
		name = self.name
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
		stack_plane._redraw_bind_data_uid = stack_plane.fbind(
			"data", stack_plane._trigger_redraw
		)

	@property
	def width(self):
		stack_plane = self.board.stack_plane
		name = self.name
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
		name = self.name
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
		name = self.name
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
		name = self.name
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
		name = self.name
		idx = stack_plane._stack_index[name]
		x = stack_plane._left_xs[idx]
		r = stack_plane._right_xs[idx]
		w = r - x
		return float(x + w / 2)

	@center_x.setter
	def center_x(self, cx):
		stack_plane = self.board.stack_plane
		name = self.name
		idx = stack_plane._stack_index[name]
		x = stack_plane._left_xs[idx]
		r = stack_plane._right_xs[idx]
		w = r - x
		self.pos = cx - w / 2, self.y

	@property
	def center_y(self):
		stack_plane = self.board.stack_plane
		name = self.name
		idx = stack_plane._stack_index[name]
		y = stack_plane._bot_ys[idx]
		t = stack_plane._top_ys[idx]
		h = t - y
		return float(y + h / 2)

	@center_y.setter
	def center_y(self, cy):
		stack_plane = self.board.stack_plane
		name = self.name
		idx = stack_plane._stack_index[name]
		y = stack_plane._bot_ys[idx]
		t = stack_plane._top_ys[idx]
		h = t - y
		self.pos = self.x, cy - h / 2

	@property
	def top(self):
		stack_plane = self.board.stack_plane
		name = self.name
		if name not in stack_plane._stack_index:
			return 100
		idx = stack_plane._stack_index[name]
		return float(stack_plane._top_ys[idx])

	@top.setter
	def top(self, t):
		stack_plane = self.board.stack_plane
		name = self.name
		idx = stack_plane._stack_index[name]
		y = stack_plane._bot_ys[idx]
		stack_plane._top_ys[idx] = t
		h = t - y
		stack_plane._bot_ys[idx] = t - h
		self.pos = self.x, t - h

	@property
	def right(self):
		stack_plane = self.board.stack_plane
		name = self.name
		if name not in stack_plane._stack_index:
			return 100
		idx = stack_plane._stack_index[name]
		return float(stack_plane._right_xs[idx])

	@right.setter
	def right(self, r):
		stack_plane = self.board.stack_plane
		name = self.name
		idx = stack_plane._stack_index[name]
		x = stack_plane._left_xs[idx]
		stack_plane._right_xs[idx] = r
		w = r - x
		x = stack_plane._left_xs[idx] = r - w
		self.pos = x, self.y

	@property
	def name(self):
		return self._name

	def collide_point(self, x, y):
		pos = self.pos
		if x < pos[0] or y < pos[1]:
			return False
		w, h = self.size
		return x < pos[0] + w and y < pos[1] + h
