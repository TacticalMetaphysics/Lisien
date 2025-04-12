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
"""The big widget that shows the graph of the selected Character."""

from functools import partial
from time import monotonic

from kivy.clock import Clock, mainthread, triggered
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.properties import (
	BooleanProperty,
	DictProperty,
	ListProperty,
	NumericProperty,
	ObjectProperty,
	ReferenceListProperty,
	StringProperty,
)
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget
from kivy.vector import Vector

from lisien.util import normalize_layout

from ..boardscatter import BoardScatterPlane
from ..boardview import BoardView
from ..dummy import Dummy
from ..kivygarden.texturestack import Stack, TextureStackPlane
from .arrow import (
	DEFAULT_ARROW_LABEL_KWARGS,
	ArrowPlane,
	GraphArrow,
	GraphArrowWidget,
	get_points_multi,
)
from .pawn import Pawn
from .spot import GraphSpot


def trigger(func):
	return triggered()(func)


class KvLayoutBack(FloatLayout):
	"""What to show behind the graph.

	By default, shows nothing.

	"""


class KvLayoutFront(FloatLayout):
	"""What to show in front of the graph.

	By default, shows nothing.

	"""

	pass


class FinalLayout(FloatLayout):
	def finalize_all(self, *args):
		for child in self.children:
			child.finalize()
		self.bind(children=self._trigger_finalize_all)

	_trigger_finalize_all = trigger(finalize_all)


def get_label_kwargs_from_portal(portal):
	return {
		"text": str(portal.get(portal.get("_label_stat", None), "")),
		**portal.get("_label_kwargs", {}),
		**DEFAULT_ARROW_LABEL_KWARGS,
	}


class GraphBoard(RelativeLayout):
	"""A graphical view onto a :class:`lisien.Character`, resembling a game
	graph.

	"""

	app = ObjectProperty()
	character = ObjectProperty()
	wallpaper_path = StringProperty()
	spot = DictProperty({})
	pawn = DictProperty({})
	arrow = DictProperty({})
	pred_arrow = DictProperty({})
	wallpaper = ObjectProperty()
	kvlayoutback = ObjectProperty()
	arrow_plane = ObjectProperty()
	stack_plane = ObjectProperty()
	kvlayoutfront = ObjectProperty()
	wids = ReferenceListProperty(
		wallpaper, kvlayoutback, arrow_plane, stack_plane, kvlayoutfront
	)
	spots_unposd = ListProperty([])
	layout_tries = NumericProperty(5)
	tracking_vel = BooleanProperty(False)
	selection_candidates = ListProperty([])
	selection = ObjectProperty(allownone=True)
	keep_selection = ObjectProperty(False)
	adding_portal = BooleanProperty(False)
	reciprocal_portal = BooleanProperty(False)
	grabbing = BooleanProperty(True)
	grabbed = ObjectProperty(None, allownone=True)
	spot_cls = ObjectProperty(Stack)
	pawn_cls = ObjectProperty(Stack)
	arrow_cls = GraphArrow
	proto_arrow_cls = ObjectProperty(GraphArrowWidget)
	_scheduled_rm_spot = DictProperty()
	_scheduled_rm_arrow = DictProperty()
	_scheduled_discard_pawn = DictProperty()
	_scheduled_add_pawn = DictProperty()

	@property
	def widkwargs(self):
		return {"size_hint": (None, None), "size": self.size, "pos": (0, 0)}

	def on_touch_down(self, touch):
		"""Check for collisions and select an appropriate entity."""
		if hasattr(self, "_lasttouch") and self._lasttouch == touch:
			return
		touch.push()
		touch.apply_transform_2d(self.to_local)
		if not self.collide_point(*touch.pos):
			touch.pop()
			return
		if self.app.selection:
			if self.app.selection.collide_point(*touch.pos):
				Logger.debug("Board: hit selection")
				if hasattr(self.app.selection, "__self__"):
					touch.grab(self.app.selection)
			else:
				if hasattr(self.app.selection, "selected"):
					self.app.selection.selected = False
				self.app.selection = None
		pawns = list(self.pawns_at(*touch.pos))
		if pawns:
			Logger.debug("Board: hit {} pawns".format(len(pawns)))
			self.selection_candidates = pawns
			if self.app.selection in self.selection_candidates:
				self.selection_candidates.remove(self.app.selection)
			touch.pop()
			return True
		spots = list(self.spots_at(*touch.pos))
		if spots:
			Logger.debug("Board: hit {} spots".format(len(spots)))
			self.selection_candidates = spots
			if self.adding_portal:
				self.origspot = self.selection_candidates.pop(0)
				self.protodest = Dummy(
					name="protodest", pos=touch.pos, size=(0, 0)
				)
				self.add_widget(self.protodest)
				self.protodest.on_touch_down(touch)
				self.protoportal = self.proto_arrow_cls(
					board=self,
					origin=self.origspot,
					destination=self.protodest,
				)
				self.add_widget(self.protoportal)
				if self.reciprocal_portal:
					self.protoportal2 = self.proto_arrow_cls(
						board=self,
						destination=self.origspot,
						origin=self.protodest,
					)
					self.add_widget(self.protoportal2)
			touch.pop()
			return True
		edges = list(self.arrow_plane.iter_collided_edges(*touch.pos))
		if edges:
			Logger.debug("Board: hit {} arrows".format(len(edges)))
			self.selection_candidates = [
				self.arrow[orig][dest] for (orig, dest) in edges
			]
			if self.app.selection in self.selection_candidates:
				self.selection_candidates.remove(self.app.selection)
			if (
				isinstance(self.app.selection, GraphArrow)
				and self.app.selection.reciprocal in self.selection_candidates
			):
				self.selection_candidates.remove(self.app.selection.reciprocal)
			touch.pop()
			return True
		touch.pop()

	def on_touch_move(self, touch):
		"""If an entity is selected, drag it."""
		if hasattr(self, "_lasttouch") and self._lasttouch == touch:
			return
		if self.app.selection in self.selection_candidates:
			self.selection_candidates.remove(self.app.selection)
		if self.app.edit_locked:
			return
		elif self.app.selection:
			if not self.selection_candidates:
				self.keep_selection = True
			ret = super().on_touch_move(touch)
			sel = self.app.selection
			if isinstance(sel, Stack):
				sel.center = touch.pos
				name = sel.proxy.name
				touch.ud["grabbed_pawn_or_spot"] = name
				for dest in self.arrow.get(name, ()):
					self.arrow[name][dest].repoint()
				for orig in self.pred_arrow.get(name, ()):
					self.arrow[orig][name].repoint()
				for thing in sel.proxy.contents():
					pawn = self.pawn[thing["name"]]
					pawn.pos = sel.right, sel.top  # real layout needed
			return ret
		elif self.selection_candidates:
			for cand in self.selection_candidates:
				if cand.collide_point(*touch.pos):
					self.app.selection = cand
					cand.selected = True
					if isinstance(cand, Widget):
						touch.grab(cand)
					ret = super().on_touch_move(touch)
					return ret
		if hasattr(self, "protodest"):
			self.protodest.pos = touch.pos
			self.protoportal._trigger_repoint()
			if hasattr(self, "protoportal2"):
				self.protoportal2._trigger_repoint()

	def portal_touch_up(self, touch):
		"""Try to create a portal between the spots the user chose."""
		try:
			# If the touch ended upon a spot, and there isn't
			# already a portal between the origin and this
			# destination, create one.
			destspot = next(self.spots_at(*touch.pos))
			orig = self.origspot.proxy
			dest = destspot.proxy
			if orig != dest and not (
				orig.name in self.character.portal
				and dest.name in self.character.portal[orig.name]
			):
				symmetrical = hasattr(self, "protoportal2") and not (
					orig.name in self.character.preportal
					and dest.name in self.character.preportal[orig.name]
				)
				port = self.character.new_portal(
					orig.name, dest.name, symmetrical=symmetrical
				)
				self.arrow_plane.add_new_portal(self.make_arrow(port))
				if orig.name not in self.arrow:
					self.arrow[orig.name] = {}
				if dest.name not in self.pred_arrow:
					self.pred_arrow[dest.name] = {}
				self.arrow[orig.name][dest.name] = self.pred_arrow[dest.name][
					orig.name
				] = GraphArrow(
					board=self,
					origin=self.spot[orig.name],
					destination=self.spot[dest.name],
				)
				if symmetrical:
					self.arrow_plane.add_new_portal(
						self.make_arrow(
							self.character.portal[dest.name][orig.name]
						)
					)
					if dest.name not in self.arrow:
						self.arrow[dest.name] = {}
					if orig.name not in self.pred_arrow:
						self.pred_arrow[orig.name] = {}
					self.arrow[dest.name][orig.name] = self.pred_arrow[
						orig.name
					][dest.name] = GraphArrow(
						board=self,
						origin=self.spot[dest.name],
						destination=self.spot[orig.name],
					)
		except StopIteration:
			pass
		self.remove_widget(self.protoportal)
		del self.protoportal
		if hasattr(self, "protoportal2"):
			self.remove_widget(self.protoportal2)
			del self.protoportal2
		del self.protodest

	def on_touch_up(self, touch):
		"""Delegate touch handling if possible, else select something."""

		def unsel_graph_arrow():
			origspot = self.app.selection.origin
			destspot = self.app.selection.destination
			insts = self.arrow_plane._instructions_map[
				origspot.name, destspot.name
			]
			fbo = self.arrow_plane._fbo
			fbo.bind()
			insts["color0"].rgba = self.arrow_plane.bg_color_unselected
			insts["color1"].rgba = self.arrow_plane.fg_color_unselected
			fbo.clear_buffer()
			fbo.release()
			self.arrow_plane.canvas.ask_update()

		if hasattr(self, "_lasttouch") and self._lasttouch == touch:
			return
		self._lasttouch = touch
		touch.push()
		touch.apply_transform_2d(self.to_local)
		if hasattr(self, "protodest"):
			Logger.debug("Board: on_touch_up making a portal")
			touch.ungrab(self)
			ret = self.portal_touch_up(touch)
			touch.pop()
			return ret
		if self.app.selection:
			sel = self.app.selection
			if isinstance(sel, Widget):
				sel.dispatch("on_touch_up", touch)
			elif (
				isinstance(sel, Stack)
				and hasattr(sel, "proxy")
				and hasattr(sel.proxy, "name")
				and "grabbed_pawn_or_spot" in touch.ud
				and sel.proxy.name == touch.ud["grabbed_pawn_or_spot"]
				and not self.app.edit_locked
			):
				if hasattr(sel.proxy, "location"):
					for candidate in self.stack_plane.iter_collided_keys(
						*touch.pos
					):
						if candidate in self.spot:
							newloc = self.character.place[candidate]
							sel.proxy.location = newloc
							newspot = self.spot[candidate]
							sel.pos = newspot.right, newspot.top
							return
					oldloc = sel.proxy.location
					oldspot = self.spot[oldloc.name]
					sel.pos = oldspot.right, oldspot.top
					return
				else:
					prox = sel.proxy
					prox["_x"] = sel.x / self.width
					prox["_y"] = sel.y / self.height
		for candidate in self.selection_candidates:
			if candidate.collide_point(*touch.pos):
				if isinstance(candidate, GraphArrow):
					portal = self.character.portal[candidate.origin.name][
						candidate.destination.name
					]
					insts = self.arrow_plane._instructions_map[
						portal["origin"], portal["destination"]
					]
					fbo = self.arrow_plane._fbo
					fbo.bind()
					insts["color0"].rgba = self.arrow_plane.bg_color_selected
					insts["color1"].rgba = self.arrow_plane.fg_color_selected
					fbo.clear_buffer()
					fbo.release()
					self.arrow_plane.canvas.ask_update()
				if hasattr(candidate, "selected"):
					candidate.selected = True
				if (
					hasattr(self.app.selection, "selected")
					and self.app.selection != candidate
				):
					self.app.selection.selected = False
				if isinstance(self.app.selection, GraphArrow):
					unsel_graph_arrow()
				self.app.selection = candidate
				self.keep_selection = True
				break
		if not self.keep_selection:
			Logger.debug("Board: deselecting " + repr(self.app.selection))
			if hasattr(self.app.selection, "selected"):
				self.app.selection.selected = False
			if isinstance(self.app.selection, GraphArrow):
				unsel_graph_arrow()
			self.app.selection = None
		self.keep_selection = False
		touch.ungrab(self)
		touch.pop()
		return

	def _pull_size(self, *args):
		if self.wallpaper.texture is None:
			Clock.schedule_once(self._pull_size, 0.001)
			return
		self.size = self.wallpaper.size = self.wallpaper.texture.size

	def _pull_image(self, *args):
		self.wallpaper.source = self.wallpaper_path
		self._pull_size()

	def on_parent(self, *args):
		"""Create some subwidgets and trigger the first update."""
		if not self.parent or hasattr(self, "_parented"):
			return
		if not self.wallpaper_path:
			Logger.debug("Board: waiting for wallpaper_path")
			Clock.schedule_once(self.on_parent, 0)
			return
		self._parented = True
		self.wallpaper = Image(source=self.wallpaper_path)
		self.bind(wallpaper_path=self._pull_image)
		self._pull_size()
		self.kvlayoutback = KvLayoutBack(**self.widkwargs)
		self.arrow_plane = ArrowPlane(**self.widkwargs)
		self.stack_plane = TextureStackPlane(**self.widkwargs)
		self.kvlayoutfront = KvLayoutFront(**self.widkwargs)
		for wid in self.wids:
			self.add_widget(wid)
			wid.pos = 0, 0
			wid.size = self.size
			if wid is not self.wallpaper:
				self.bind(size=wid.setter("size"))
		self.update()

	def on_character(self, *args):
		if self.character is None or self.character.engine.closed:
			return
		if self.parent is None:
			Clock.schedule_once(self.on_character, 0)
			return

		wallpaper_path = self.character.stat.setdefault(
			"wallpaper", "parchmentBasic.png"
		)
		self.engine = getattr(self.character, "engine", None)
		if (
			"_control" not in self.character.stat
			or "wallpaper" not in self.character.stat["_control"]
		):
			control = self.character.stat.get("_control", {})
			control["wallpaper"] = "textinput"
		self.wallpaper_path = wallpaper_path
		self.trigger_update()

	def update_from_stat(self, sender, *, k, v):
		if k == "wallpaper":
			self.wallpaper_path = v

	def _trigger_pull_wallpaper(self, *args, **kwargs):
		if kwargs["key"] != "wallpaper":
			return
		if hasattr(self, "_scheduled_pull_wallpaper"):
			Clock.unschedule(self._scheduled_pull_wallpaper)
		self._scheduled_pull_wallpaper = Clock.schedule_once(
			self.pull_wallpaper, 0
		)

	@trigger
	def kv_updated(self, *args):
		self.unbind(wallpaper_path=self.kvlayoutback.setter("wallpaper_path"))
		for wid in self.wids:
			self.remove_widget(wid)
		self.kvlayoutback = KvLayoutBack(
			pos=(0, 0), wallpaper_path=self.wallpaper_path
		)
		self.bind(wallpaper_path=self.kvlayoutback.setter("wallpaper_path"))
		self.kvlayoutfront = KvLayoutFront(**self.widkwargs)
		self.size = self.kvlayoutback.size
		self.kvlayoutback.bind(size=self.setter("size"))
		for wid in self.wids:
			self.add_widget(wid)

	def make_pawn(self, thing):
		"""Make a :class:`Pawn` to represent a :class:`Thing`, store it, and
		return a dict suitable for `StackPlane.add_datum`

		"""
		if thing["name"] in self.pawn:
			raise KeyError("Already have a Pawn for this Thing")
		r = self.pawn_cls(board=self, proxy=thing)
		self.pawn[thing["name"]] = r
		locspot = self.spot[thing["location"]]
		if "_image_paths" in thing:
			texs = list(thing["_image_paths"])
		else:
			texs = list(Pawn.default_image_paths)
		width = height = 0.0
		for tex in texs:
			wide, high = Image(source=tex).texture_size
			if wide > width:
				width = wide
			if high > height:
				height = high
		return {  # need to lay out multiple pawns per spot properly
			"x": int(locspot.right),
			"y": int(locspot.top),
			"width": width,
			"height": height,
			"name": thing["name"],
			"textures": texs,
		}

	def make_spot(self, place):
		"""Make a :class:`Spot` to represent a :class:`Place`, store it, and
		return a dict suitable for `StackPlane.add_datum`

		"""
		if place["name"] in self.spot:
			raise KeyError("Already have a Spot for this Place")
		self.spot[place["name"]] = self.spot_cls(board=self, proxy=place)
		if "_image_paths" in place:
			texs = list(place["_image_paths"])
		else:
			texs = list(GraphSpot.default_image_paths)
		width = height = 0.0
		for tex in texs:
			wide, high = Image(source=tex).texture_size
			if wide > width:
				width = wide
			if high > height:
				height = high
		return {
			"x": place.get("_x", 0.5),
			"y": place.get("_y", 0.5),
			"width": width,
			"height": height,
			"name": place["name"],
			"textures": texs,
		}

	def make_arrow(self, portal):
		if (
			portal["origin"] not in self.spot
			or portal["destination"] not in self.spot
		):
			raise ValueError(
				"An :class:`Arrow` should only be made after "
				"the :class:`Spot`s it connects"
			)
		if (
			portal["origin"] in self.arrow
			and portal["destination"] in self.arrow[portal["origin"]]
		):
			raise KeyError("Already have an Arrow for this Portal")
		return self._core_make_arrow(
			portal,
			self.spot[portal["origin"]],
			self.spot[portal["destination"]],
		)

	def _core_make_arrow(self, portal, origspot, destspot, points=None):
		r = {
			"board": self,
			"portal": portal,
			"origspot": origspot,
			"destspot": destspot,
			"label_kwargs": get_label_kwargs_from_portal(portal),
		}
		if points is not None:
			r["points"] = points
		return r

	def rm_arrows_to_and_from(self, name):
		if name in self.arrow.keys():
			for dest in list(self.arrow[name].keys()):
				self.rm_arrow(name, dest)
		if name in self.pred_arrow.keys():
			for orig in list(self.pred_arrow[name].keys()):
				self.rm_arrow(orig, name)

	def rm_pawn(self, name, *args):
		"""Remove the :class:`Pawn` by the given name."""
		if name not in self.pawn:
			raise KeyError("No Pawn named {}".format(name))
		# Currently there's no way to connect Pawns with Arrows but I
		# think there will be, so, insurance
		self.rm_arrows_to_and_from(name)
		pwn = self.pawn.pop(name)
		if pwn in self.selection_candidates:
			self.selection_candidates.remove(pwn)
		self.stack_plane.remove(name)

	def _trigger_rm_pawn(self, name):
		Clock.schedule_once(partial(self.rm_pawn, name), 0)

	def rm_spot(self, name, *args):
		"""Remove the :class:`Spot` by the given name."""
		if name not in self.spot:
			raise KeyError("No Spot named {}".format(name))
		spot = self.spot.pop(name)
		if spot in self.selection_candidates:
			self.selection_candidates.remove(spot)
		pawns_here = []
		for thing in spot.proxy.contents():
			pawns_here.append(self.pawn[thing.name])
		self.rm_arrows_to_and_from(name)
		self.stack_plane.remove(name)
		for pawn in pawns_here:
			self.rm_pawn(pawn.name)
		if name in self._scheduled_rm_spot:
			del self._scheduled_rm_spot[name]

	def _trigger_rm_spot(self, name):
		part = partial(self.rm_spot, name)
		if name in self._scheduled_rm_spot:
			Clock.unschedule(self._scheduled_rm_spot[name])
		self._scheduled_rm_spot[name] = Clock.schedule_once(part, 0)

	def rm_arrow(self, orig, dest, *args):
		"""Remove the :class:`Arrow` that goes from ``orig`` to ``dest``."""
		if orig not in self.arrow or dest not in self.arrow[orig]:
			raise KeyError("No Arrow from {} to {}".format(orig, dest))
		arr = self.arrow[orig].pop(dest)
		if arr in self.selection_candidates:
			self.selection_candidates.remove(arr)
		self.arrow_plane.remove_edge(orig, dest)
		if (orig, dest) in self._scheduled_rm_arrow:
			del self._scheduled_rm_arrow[orig, dest]

	def _trigger_rm_arrow(self, orig, dest):
		part = partial(self.rm_arrow, orig, dest)
		if (orig, dest) in self._scheduled_rm_arrow:
			Clock.unschedule(self._scheduled_rm_arrow[orig, dest])
		self._scheduled_rm_arrow[orig, dest] = Clock.schedule_once(part, 0)

	def graph_layout(self, graph):
		from networkx.drawing.layout import spring_layout

		return normalize_layout(spring_layout(graph))

	def discard_pawn(self, thingn, *args):
		if thingn in self.pawn:
			self.rm_pawn(thingn)
		if thingn in self._scheduled_discard_pawn:
			del self._scheduled_discard_pawn[thingn]

	def _trigger_discard_pawn(self, thing):
		part = partial(self.discard_pawn, thing)
		if thing in self._scheduled_discard_pawn:
			Clock.unschedule(self._scheduled_discard_pawn[thing])
		self._scheduled_discard_pawn[thing] = Clock.schedule_once(part, 0)

	def _remove_absent_pawns(self, *args):
		for pawn_name in list(self.pawn.keys()):
			if pawn_name not in self.character.thing:
				self.rm_pawn(pawn_name)

	def discard_spot(self, placen, *args):
		if placen in self.spot:
			self.rm_spot(placen)

	def _trigger_discard_spot(self, place):
		Clock.schedule_once(partial(self.discard_spot, place), 0)

	def _remove_absent_spots(self, *args):
		for spot_name in list(self.spot.keys()):
			if spot_name not in self.character.place:
				self.rm_spot(spot_name)

	def discard_arrow(self, orign, destn, *args):
		if orign in self.arrow and destn in self.arrow[orign]:
			self.rm_arrow(orign, destn)

	def _trigger_discard_arrow(self, orig, dest):
		Clock.schedule_once(partial(self.discard_arrow, orig, dest), 0)

	def _remove_absent_arrows(self, *args):
		for arrow_origin in list(self.arrow.keys()):
			for arrow_destination in list(self.arrow[arrow_origin].keys()):
				if (
					arrow_origin not in self.character.portal
					or arrow_destination
					not in self.character.portal[arrow_origin]
				):
					self.rm_arrow(arrow_origin, arrow_destination)

	def add_spot(self, placen, *args):
		if placen in self.character.place and placen not in self.spot:
			spotten = self.make_spot(self.character.place[placen])
			self.stack_plane.add_datum(spotten)
			self.spot[placen].pos = (
				spotten["x"] * self.width,
				spotten["y"] * self.height,
			)

	def _trigger_add_spot(self, placen):
		Clock.schedule_once(partial(self.add_spot, placen), 0)

	def _add_new_spots(self, *args):
		start_ts = monotonic()
		places2add = []
		spots_unposd = []
		nodes_patch = {}
		placemap = self.character.place
		spotmap = self.spot
		default_image_paths = GraphSpot.default_image_paths
		for place_name, place in placemap.items():
			if place_name not in spotmap:
				place = placemap[place_name]
				places2add.append(place)
		make_spot = self.make_spot
		spots_posd = []
		stack_idx = self.stack_plane._stack_index
		for place in places2add:
			spot = make_spot(place)
			if "_x" not in place or "_y" not in place:
				spots_unposd.append(spot)
			elif spot["name"] not in stack_idx:
				spots_posd.append(spot)
		if spots_unposd:
			try:
				nodes_patch_2 = self.grid_layout(spots_unposd)
			except (TypeError, ValueError):
				nodes_patch_2 = self.nx_layout(spots_unposd)
			for k, v in nodes_patch_2.items():
				if k in nodes_patch:
					nodes_patch[k].update(v)
				else:
					nodes_patch[k] = v
		if nodes_patch:
			self.engine.handle(
				command="update_nodes",
				char=self.character.name,
				patch=nodes_patch,
			)
		if spots_posd:
			self.stack_plane.unbind_uid(
				"data", self.stack_plane._redraw_bind_data_uid
			)
			self.stack_plane.data.extend(spots_posd)
			self.stack_plane.redraw()
			self.stack_plane._redraw_bind_data_uid = self.stack_plane.fbind(
				"data", self.stack_plane._trigger_redraw
			)

	def add_arrow(self, orign, destn, *args):
		if not (
			orign in self.character.portal
			and destn in self.character.portal[orign]
		):
			raise ValueError("No portal for arrow {}->{}".format(orign, destn))
		portal = self.character.portal[orign][destn]
		if not (orign in self.arrow and destn in self.arrow[orign]):
			self.arrow_plane.add_new_portal(self.make_arrow(portal))
			the_arrow = GraphArrow(
				board=self,
				origin=self.spot[orign],
				destination=self.spot[destn],
			)
			if orign not in self.arrow:
				self.arrow[orign] = {}
			self.arrow[orign][destn] = the_arrow
			if destn not in self.pred_arrow:
				self.pred_arrow[destn] = {}
			self.pred_arrow[destn][orign] = the_arrow

	def _add_new_arrows(self, *args):
		portmap = self.character.portal
		arrowmap = self.arrow
		pred_arrowmap = self.pred_arrow
		spotmap = self.spot
		append_to_arrow_plane = self.arrow_plane.data.append
		core_make_arrow = self._core_make_arrow
		todo = []
		for arrow_orig, arrow_dests in portmap.items():
			for arrow_dest, portal in arrow_dests.items():
				if (
					arrow_orig not in arrowmap
					or arrow_dest not in arrowmap[arrow_orig]
				):
					todo.append(
						(portal, spotmap[arrow_orig], spotmap[arrow_dest])
					)
					the_arr = GraphArrow(
						board=self,
						origin=spotmap[arrow_orig],
						destination=spotmap[arrow_dest],
					)
					if arrow_orig not in arrowmap:
						arrowmap[arrow_orig] = {}
					if arrow_dest not in arrowmap[arrow_orig]:
						arrowmap[arrow_orig][arrow_dest] = the_arr
					if arrow_dest not in pred_arrowmap:
						pred_arrowmap[arrow_dest] = {}
					if arrow_orig not in pred_arrowmap[arrow_dest]:
						pred_arrowmap[arrow_dest][arrow_orig] = the_arr
		points = get_points_multi(
			(origspot, destspot, 10) for (portal, origspot, destspot) in todo
		)
		for portal, origspot, destspot in todo:
			append_to_arrow_plane(
				core_make_arrow(
					portal,
					origspot,
					destspot,
					points[origspot.name, destspot.name],
				)
			)

	def update_arrow_labels(self, *args):
		portmap = self.character.portal
		arrow_plane = self.arrow_plane
		for datum in arrow_plane.data:
			portal = portmap[datum["origspot"].name][datum["destspot"].name]
			new_kwargs = get_label_kwargs_from_portal(portal)
			if new_kwargs != datum["label_kwargs"]:
				arrow_plane.update_portal(
					datum["origspot"].name, datum["destspot"].name, new_kwargs
				)

	def add_pawn(self, thingn, *args):
		if thingn not in self.character.thing:
			raise KeyError(f"No Thing for pawn: {thingn}")
		if thingn in self.pawn:
			raise KeyError(f"Already have pawn for Thing: {thingn}")
		pwn = self.make_pawn(self.character.thing[thingn])
		stacp = self.stack_plane
		stacp.add_datum(pwn)
		self.pawn[thingn].pos = pwn["x"], pwn["y"]

		if thingn in self._scheduled_add_pawn:
			del self._scheduled_add_pawn[thingn]

	def _trigger_add_pawn(self, thingn):
		part = partial(self.add_pawn, thingn)
		if thingn in self._scheduled_add_pawn:
			Clock.unschedule(self._scheduled_add_pawn[thingn])
		self._scheduled_add_pawn[thingn] = Clock.schedule_once(part, 0)

	def _add_new_pawns(self, *args):
		nodes_patch = {}
		things2add = []
		pawns_added = []
		pawnmap = self.pawn
		for thing_name, thing in self.character.thing.items():
			if thing_name not in pawnmap:
				things2add.append(thing)
		make_pawn = self.make_pawn
		for thing in things2add:
			pwn = make_pawn(thing)
			if "_image_paths" not in thing:
				nodes_patch[thing["name"]] = {
					"_image_paths": list(
						pwn.get("textures", Pawn.default_image_paths)
					)
				}
			pawns_added.append(pwn)
		if nodes_patch:
			self.character.node.patch(nodes_patch)
		self.stack_plane.unbind_uid(
			"data", self.stack_plane._redraw_bind_data_uid
		)
		self.stack_plane.data.extend(pawns_added)
		self.stack_plane.redraw()
		self.stack_plane._redraw_bind_data_uid = self.stack_plane.fbind(
			"data", self.stack_plane._trigger_redraw
		)

	def update(self, *args):
		"""Force an update to match the current state of my character.

		This polls every element of the character, and therefore
		causes me to sync with the lisien core for a long time. Avoid
		when possible.

		"""
		if not hasattr(self, "engine") or getattr(
			self.engine, "closed", False
		):
			Logger.warning(
				"Board: tried to update without a connection to a lisien core"
			)
			return
		if not self.stack_plane or not self.arrow_plane:
			self.trigger_update()
			return
		# remove widgets that don't represent anything anymore
		Logger.debug("GraphBoard: updating")
		start_ts = monotonic()
		self._disconnect_proxy_objects()
		self._remove_absent_pawns()
		self._remove_absent_spots()
		self._remove_absent_arrows()
		# add widgets to represent new stuff
		self._add_new_spots()
		self._update_spot_display()
		if self.arrow_cls:
			self._add_new_arrows()
			self._update_arrow_display()
		self._add_new_pawns()
		self._update_pawn_display()
		self._connect_proxy_objects()
		Logger.debug(
			f"GraphBoard: updated, took {monotonic() - start_ts:,.2f} seconds"
		)

	trigger_update = trigger(update)

	def _disconnect_proxy_objects(self):
		char = self.character
		char.stat.disconnect(self.update_from_character_stat)
		char.node.disconnect(self.update_from_character_node)
		char.portal.disconnect(self.update_from_character_edge)

	def _connect_proxy_objects(self):
		char = self.character
		char.stat.connect(self.update_from_character_stat)
		char.node.connect(self.update_from_character_node)
		char.portal.connect(self.update_from_character_edge)

	def update_from_character_stat(self, character, key, value):
		pass

	@mainthread
	def update_from_character_node(self, node, key, value):
		if hasattr(node, "location"):
			if not node:
				self.rm_pawn(node.name)
			elif node.name not in self.pawn:
				self.add_pawn(node.name)
			elif key == "location":
				loc = self.spot[value]
				thing = self.pawn[node.name]
				thing.pos = loc.right, loc.top
			elif key == "_image_paths":
				self.pawn[node.name].paths = value
		elif node is self.character.node or node is self.character.place:
			if value and key not in self.spot:
				self.add_spot(key)
			elif not value and key in self.spot:
				self.rm_spot(key)
		elif node is self.character.thing:
			if value and key not in self.pawn:
				self.add_pawn(key)
			elif not value and key not in self.pawn:
				self.rm_pawn(key)
		else:
			if not node:
				self.rm_spot(node.name)
			elif node.name not in self.spot:
				self.add_spot(node.name)
			elif key == "_image_paths":
				self.spot[node.name].paths = value

	def _update_spot_display(self):
		"""Change spot graphics to match the state of their place"""

	def _update_pawn_display(self):
		"""Change pawn graphics to match the state of their thing"""

	def update_from_character_edge(self, edge, key, value):
		if edge:
			if not self.arrow_plane.have_arrow(
				edge._origin, edge._destination
			):
				label_kwargs = DEFAULT_ARROW_LABEL_KWARGS.copy()
				if "_label_stat" in edge:
					label_kwargs["text"] = str(
						edge.get(edge["label_stat"], "")
					)
				self.arrow_plane.add_new_portal(
					{
						"origspot": self.spot[edge.origin.name],
						"destspot": self.spot[edge.destination.name],
						"label_kwargs": label_kwargs,
					}
				)
			if key == edge.get("_label_stat"):
				self.arrow_plane.update_portal_label(
					edge._origin, edge._destination, str(value)
				)
		else:
			self.arrow_plane.remove_edge(edge._origin, edge._destination)

	def _update_arrow_display(self):
		"""Change arrow graphics to match the state of their portal"""

	def _apply_node_layout(self, l, spot, *args):
		if self.width == 1 or self.height == 1:
			Clock.schedule_once(
				partial(self._apply_node_layout, l, spot), 0.01
			)
			return
		if not isinstance(spot, dict):
			spot = {spt["name"]: spt for spt in spot}
		node_upd = {}
		newspots = []
		for name, (x, y) in l.items():
			assert 0 <= x <= 0.99, "{} has invalid x: {}".format(name, x)
			assert 0 <= y <= 0.99, "{} has invalid y: {}".format(name, y)
			assert self.stack_plane.width == self.width
			assert self.stack_plane.height == self.height
			node_upd[name] = {"_x": x, "_y": y}
			spot[name]["x"] = x
			spot[name]["y"] = y
			newspots.append(spot[name])
		if newspots:
			self.stack_plane.unbind_uid(
				"data", self.stack_plane._redraw_bind_data_uid
			)
			self.stack_plane.data.extend(newspots)
			self.stack_plane.redraw()
			self.stack_plane._redraw_bind_data_uid = self.stack_plane.fbind(
				"data", self.stack_plane._trigger_redraw
			)
		self.spots_unposd = []
		return node_upd

	def grid_layout(self, spots, *args):
		return self._apply_node_layout(
			normalize_layout({spot["name"]: spot["name"] for spot in spots}),
			spots,
		)

	def nx_layout(self, spots, *args):
		spots_only = self.character.facade()
		for thing in list(spots_only.thing.keys()):
			del spots_only.thing[thing]
		for place in spots_only.place.keys() - set(
			spot["name"] for spot in spots
		):
			del spots_only.place[place]
		return self._apply_node_layout(self.graph_layout(spots_only), spots)

	def arrows(self):
		"""Iterate over all my arrows."""
		for o in self.arrow.values():
			for arro in o.values():
				yield arro

	def pawns_at(self, x, y):
		"""Iterate over pawns that collide the given point."""
		for name in self.pawn.keys() & set(
			self.stack_plane.iter_collided_keys(x, y)
		):
			yield self.pawn[name]

	def spots_at(self, x, y):
		"""Iterate over spots that collide the given point."""
		for name in self.spot.keys() & set(
			self.stack_plane.iter_collided_keys(x, y)
		):
			yield self.spot[name]

	def arrows_at(self, x, y):
		"""Iterate over arrows that collide the given point."""
		for orig, dest in self.arrow_plane.iter_collided_edges(x, y):
			yield self.arrow[orig][dest]


class GraphBoardScatterPlane(BoardScatterPlane):
	selection_candidates = ListProperty([])
	selection = ObjectProperty(allownone=True)
	keep_selection = BooleanProperty(False)
	board = ObjectProperty()
	adding_portal = BooleanProperty(False)
	reciprocal_portal = BooleanProperty()

	def spot_from_dummy(self, dummy):
		"""Make a real place and its spot from a dummy spot.

		Create a new :class:`graph.Spot` instance, along with the
		underlying :class:`lisien.Place` instance, and give it the name,
		position, and imagery of the provided dummy.

		"""
		(x, y) = self.to_local(*dummy.pos_up)
		x /= self.board.width
		y /= self.board.height
		self.board.character.new_place(
			dummy.name, _x=x, _y=y, _image_paths=list(dummy.paths)
		)
		dummy.num += 1

	def pawn_from_dummy(self, dummy):
		"""Make a real thing and its pawn from a dummy pawn.

		Create a new :class:`graph.Pawn` instance, along with the
		underlying :class:`lisien.Thing` instance, and give it the name,
		location, and imagery of the provided dummy.

		"""
		candidates = []
		dummy_center = self.to_local(*dummy.center)
		dummy.pos = self.to_local(*dummy.pos)
		for key in self.board.stack_plane.iter_collided_keys(*dummy_center):
			if key in self.board.spot:
				candidates.append(self.board.spot[key])
		if not candidates:
			return
		whereat = candidates.pop()
		if candidates:
			dist = Vector(*whereat.center).distance(dummy_center)
			while candidates:
				thereat = candidates.pop()
				thereto = Vector(*thereat.center).distance(dummy_center)
				if thereto < dist:
					whereat, dist = thereat, thereto
		self.board.character.new_thing(
			dummy.name, whereat.proxy.name, _image_paths=list(dummy.paths)
		)
		dummy.num += 1

	def on_board(self, *args):
		if hasattr(self, "_oldboard"):
			self.unbind(
				adding_portal=self._oldboard.setter("adding_portal"),
				reciprocal_portal=self._oldboard.setter("reciprocal_portal"),
			)
		self.clear_widgets()
		self.add_widget(self.board)
		self.board.adding_portal = self.adding_portal
		self.board.reciprocal_portal = self.reciprocal_portal
		self.bind(
			adding_portal=self.board.setter("adding_portal"),
			reciprocal_portal=self.board.setter("reciprocal_portal"),
		)
		self._oldboard = self.board


class GraphBoardView(BoardView):
	adding_portal = BooleanProperty(False)
	reciprocal_portal = BooleanProperty(True)
	engine = ObjectProperty()
	character_name = StringProperty()

	def on_character_name(self, *args):
		if (
			not self.engine
			or not self.character_name
			or self.character_name not in self.engine.character
		):
			Clock.schedule_once(self.on_character_name, 0)
			return
		character = self.engine.character[self.character_name]
		self.board = GraphBoard(character=character)


Builder.load_string("""
<GraphBoard>:
	app: app
	size_hint: None, None
<GraphBoardView>:
	plane: boardplane
	GraphBoardScatterPlane:
		id: boardplane
		board: root.board
		adding_portal: root.adding_portal
		reciprocal_portal: root.reciprocal_portal
		scale_min: root.scale_min
		scale_max: root.scale_max
		pos: root.pos
		size: root.size
		size_hint: None, None
""")
