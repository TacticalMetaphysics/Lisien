from collections import defaultdict
from functools import partial
from itertools import chain
from time import monotonic

from kivy.clock import Clock
from kivy.lang.builder import Builder
from kivy.logger import Logger
from kivy.properties import (
	BooleanProperty,
	ListProperty,
	NumericProperty,
	ObjectProperty,
	ReferenceListProperty,
)
from kivy.uix.widget import Widget
from kivy.vector import Vector

from elide.boardscatter import BoardScatterPlane

from ..boardview import BoardView
from ..kivygarden.texturestack import Stack, TextureStackPlane


class GridPawn(Stack):
	default_image_paths = ["atlas://rltiles/base.atlas/unseen"]

	@property
	def _stack_plane(self):
		return self.board.pawn_plane


class GridSpot(Stack):
	default_image_paths = ["atlas://rltiles/floor.atlas/floor-stone"]

	@property
	def _stack_plane(self):
		return self.board.spot_plane


class GridBoard(Widget):
	selection = ObjectProperty()
	selection_candidates = ListProperty()
	character = ObjectProperty()
	tile_width = NumericProperty(32)
	tile_height = NumericProperty(32)
	tile_size = ReferenceListProperty(tile_width, tile_height)
	pawn_cls = GridPawn
	spot_cls = GridSpot

	def __init__(self, **kwargs):
		self.pawn = {}
		self.spot = {}
		self.contained = defaultdict(set)
		super().__init__(**kwargs)

	def update(self):
		make_spot = self.make_spot
		make_pawn = self.make_pawn
		self.spot_plane.data = list(
			map(make_spot, self.character.place.values())
		)
		self.pawn_plane.data = list(
			map(make_pawn, self.character.thing.values())
		)
		self.spot_plane.redraw()
		self.pawn_plane.redraw()

	def add_spot(self, placen, *args):
		if placen not in self.character.place:
			raise KeyError(f"No such place for spot: {placen}")
		if placen in self.spot:
			raise KeyError("Already have a Spot for this Place")
		spt = self.make_spot(self.character.place[placen])
		self.spot_plane.add_datum(spt)
		self.spot[placen] = self.spot_cls(board=self, proxy=spt["proxy"])

	def make_spot(self, place):
		placen = place["name"]
		if not isinstance(placen, tuple) or len(placen) != 2:
			raise TypeError(
				"Can only make spot from places with tuple names of length 2",
				placen,
			)
		if (
			not isinstance(placen, tuple)
			or len(placen) != 2
			or not isinstance(placen[0], int)
			or not isinstance(placen[1], int)
		):
			raise TypeError(
				"GridBoard can only display places named with pairs of ints"
			)
		if "_image_paths" in place:
			textures = list(place["_image_paths"])
		else:
			textures = list(self.spot_cls.default_image_paths)
		r = {
			"name": placen,
			"x": int(placen[0] * self.tile_width),
			"y": int(placen[1] * self.tile_height),
			"width": int(self.tile_width),
			"height": int(self.tile_height),
			"textures": textures,
			"proxy": place,
		}
		return r

	def make_pawn(self, thing) -> dict:
		location = self.spot[thing["location"]]
		r = {
			"name": thing["name"],
			"x": int(location.x),
			"y": int(location.y),
			"width": int(self.tile_width),
			"height": int(self.tile_height),
			"location": location,
			"textures": list(
				thing.get("_image_paths", self.pawn_cls.default_image_paths)
			),
			"proxy": thing,
		}
		return r

	def add_pawn(self, thingn, *args):
		if thingn not in self.character.thing:
			raise KeyError(f"No such thing: {thingn}")
		if thingn in self.pawn:
			raise KeyError(f"Already have a pawn for {thingn}")
		thing = self.character.thing[thingn]
		if thing["location"] not in self.spot:
			# The location is not in the grid. That's fine.
			return
		pwn = self.make_pawn(thing)
		self.pawn[thingn] = self.pawn_cls(board=self, proxy=pwn["proxy"])
		location = pwn["location"]
		self.contained[location].add(thingn)
		self.pawn_plane.add_datum(pwn)

	def _trigger_add_pawn(self, thingn):
		part = partial(self.add_pawn, thingn)
		Clock.unschedule(part)
		Clock.schedule_once(part, 0)

	def on_parent(self, *args):
		if self.character is None:
			Clock.schedule_once(self.on_parent, 0)
			return
		if self.character.engine.closed:
			return
		Logger.debug("GridBoard: on_parent start")
		start_ts = monotonic()
		if not hasattr(self, "pawn_plane"):
			self.pawn_plane = TextureStackPlane(pos=self.pos, size=self.size)
			self.spot_plane = TextureStackPlane(pos=self.pos, size=self.size)
			self.bind(
				pos=self.pawn_plane.setter("pos"),
				size=self.pawn_plane.setter("size"),
			)
			self.bind(
				pos=self.spot_plane.setter("pos"),
				size=self.spot_plane.setter("size"),
			)
			self.add_widget(self.spot_plane)
			self.add_widget(self.pawn_plane)
		spot_data = list(
			map(
				self.make_spot,
				filter(
					lambda spot: isinstance(spot["name"], tuple)
					and len(spot["name"]) == 2,
					self.character.place.values(),
				),
			)
		)
		if not spot_data:
			if hasattr(self.spot_plane, "_redraw_bind_uid"):
				self.spot_plane.unbind_uid(
					"data", self.spot_plane._redraw_bind_data_uid
				)
				del self.spot_plane._redraw_bind_data_uid
			if hasattr(self.pawn_plane, "_redraw_bind_uid"):
				self.pawn_plane.unbind_uid(
					"data", self.pawn_plane._redraw_bind_data_uid
				)
				del self.pawn_plane._redraw_bind_data_uid
			self.spot_plane.data = self.pawn_plane.data = []
			self.spot_plane.redraw()
			self.pawn_plane.redraw()
			self.spot_plane._redraw_bind_data_uid = self.spot_plane.fbind(
				"data", self.spot_plane._trigger_redraw
			)
			self.pawn_plane._redraw_bind_data_uid = self.pawn_plane.fbind(
				"data", self.pawn_plane._trigger_redraw
			)
			self.character.thing.connect(self.update_from_thing)
			self.character.place.connect(self.update_from_place)
			return
		for spt in spot_data:
			self.spot[spt["name"]] = self.spot_cls(
				board=self, proxy=spt["proxy"]
			)
		self.spot_plane.unbind_uid(
			"data", self.spot_plane._redraw_bind_data_uid
		)
		self.spot_plane.data = spot_data
		self.spot_plane.redraw()
		self.spot_plane._redraw_bind_data_uid = self.spot_plane.fbind(
			"data", self.spot_plane._trigger_redraw
		)
		wide = max(datum["x"] for datum in spot_data) + self.tile_width
		high = max(datum["y"] for datum in spot_data) + self.tile_width
		self.size = self.spot_plane.size = self.pawn_plane.size = wide, high
		pawn_data = list(
			map(
				self.make_pawn,
				filter(
					lambda thing: thing["location"] in self.spot,
					self.character.thing.values(),
				),
			)
		)
		for pwn in pawn_data:
			self.pawn[pwn["name"]] = self.pawn_cls(
				board=self, proxy=pwn["proxy"]
			)
		self.pawn_plane.data = pawn_data
		self.character.thing.connect(self.update_from_thing)
		self.character.place.connect(self.update_from_place)
		Logger.debug(
			f"GridBoard: on_parent end, took {monotonic() - start_ts:,.2f}"
			f" seconds"
		)

	def rm_spot(self, name):
		spot = self.spot.pop(name)
		if spot in self.selection_candidates:
			self.selection_candidates.remove(spot)
		for thing in spot.proxy.contents():
			self.rm_pawn(thing.name)
		self.spot_plane.remove(name)

	def rm_pawn(self, name):
		pwn = self.pawn.pop(name)
		if pwn in self.selection_candidates:
			self.selection_candidates.remove(pwn)
		self.pawn_plane.remove(name)

	def update_from_thing(self, thing, key, value):
		if thing and not (key is None and not value):
			if thing.name not in self.pawn:
				self.add_pawn(thing.name)
			elif key == "location":
				if value in self.spot:
					loc = self.spot[value]
					pwn = self.pawn[thing.name]
					pwn.pos = loc.pos
				elif thing.name in self.pawn:
					self.rm_pawn(thing.name)
			elif key == "_image_paths":
				self.pawn[thing.name].paths = value
		else:
			if thing.name in self.pawn:
				self.rm_pawn(thing.name)

	def update_from_place(self, place, key, value):
		if place and not (key is None and not value):
			if key == "_image_paths":
				self.spot[place.name].paths = value
		else:
			if place.name in self.spot:
				self.rm_spot(place.name)

	def on_touch_down(self, touch):
		self.selection_candidates.extend(
			chain(
				map(
					self.pawn.get,
					self.pawn_plane.iter_collided_keys(*touch.pos),
				),
				map(
					self.spot.get,
					self.spot_plane.iter_collided_keys(*touch.pos),
				),
			)
		)
		return super().on_touch_down(touch)

	def on_touch_up(self, touch):
		touched = {
			candidate
			for candidate in self.selection_candidates
			if candidate.collide_point(*touch.pos)
		}
		if not touched:
			self.selection_candidates = []
			return super().on_touch_up(touch)
		if len(touched) == 1:
			self.selection = touched.pop()
			self.selection_candidates = []
			return super().on_touch_up(touch)
		pawns_touched = {
			node for node in touched if isinstance(node, self.pawn_cls)
		}
		if len(pawns_touched) == 1:
			self.selection = pawns_touched.pop()
			self.selection_candidates = []
			return super().on_touch_up(touch)
		elif pawns_touched:
			# TODO: Repeatedly touching a spot with multiple pawns on it
			# 		should cycle through the pawns, and then finally the spot.
			self.selection = pawns_touched.pop()
			self.selection_candidates = []
			return super().on_touch_up(touch)
		spots_touched = touched - pawns_touched
		if len(spots_touched) == 1:
			self.selection = spots_touched.pop()
			self.selection_candidates = []
			return super().on_touch_up(touch)
		assert not spots_touched, (
			"How do you have overlapping spots on a GridBoard??"
		)
		self.selection_candidates = []
		return super().on_touch_up(touch)


class GridBoardScatterPlane(BoardScatterPlane):
	selection_candidates = ListProperty([])
	selection = ObjectProperty(allownone=True)
	keep_selection = BooleanProperty(False)
	board = ObjectProperty()

	def spot_from_dummy(self, dummy):
		raise NotImplementedError("oop")

	def pawn_from_dummy(self, dummy):
		dummy_center = self.to_local(*dummy.center)
		candidates = list(
			self.board.spot_plane.iter_collided_keys(*dummy_center)
		)
		if not candidates:
			return
		whereat_d = self.board.spot[candidates.pop()]
		half_wide = self.board.tile_width / 2
		half_high = self.board.tile_height / 2
		if candidates:
			whereat_center = whereat_d.x + half_wide, whereat_d.y + half_high
			dist = Vector(*whereat_center).distance(dummy_center)
			while candidates:
				thereat_d = self.board.spot[candidates.pop()]
				thereat_center = (
					thereat_d.x + half_wide,
					thereat_d.y + half_high,
				)
				thereto = Vector(*thereat_center).distance(dummy_center)
				if thereto < dist:
					whereat_d, dist = thereat_d, thereto
		self.board.pawn_plane.add_datum(
			self.board.make_pawn(
				self.board.character.new_thing(
					dummy.name, whereat_d.name, _image_paths=list(dummy.paths)
				)
			)
		)
		dummy.num += 1

	def on_board(self, *args):
		self.clear_widgets()
		self.add_widget(self.board)


class GridBoardView(BoardView):
	pass


kv = """
<GridBoard>:
	app: app
	size_hint: None, None
<GridBoardView>:
	plane: boardplane
	GridBoardScatterPlane:
		id: boardplane
		board: root.board
		scale_min: root.scale_min
		scale_max: root.scale_max
		pos: root.pos
		size: root.size
"""
kv_loaded = False
if not kv_loaded:
	Builder.load_string(kv)
	kv_loaded = True
