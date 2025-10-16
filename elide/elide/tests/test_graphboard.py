from math import sqrt

import networkx as nx
import pytest
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.tests.common import UnitTestTouch

from elide.app import ElideApp
from elide.graph.arrow import ArrowPlane
from elide.graph.board import GraphBoard, GraphBoardView
from lisien.collections import FunctionStore
from lisien.facade import CharacterFacade, EngineFacade
from lisien.util import SignalDict

from ..kivygarden.texturestack import TextureStackPlane
from ..util import load_kv
from .util import advance_frames, idle_until


def pos_near(x0, y0, x1, y1):
	return abs(sqrt(x0**2 + y0**2) - sqrt(x1**2 + y1**2)) < 10


def test_layout_grid(elide_app):
	spots_wide = 3
	spots_tall = 3
	graph = nx.grid_2d_graph(spots_wide, spots_tall)
	eng = EngineFacade(None)
	char = eng.new_character("grid", graph)
	app = elide_app
	spotlayout = TextureStackPlane()
	arrowlayout = ArrowPlane()
	board = GraphBoard(
		app=app,
		character=char,
		stack_plane=spotlayout,
		arrow_plane=arrowlayout,
	)
	board.engine = eng
	spotlayout.pos = board.pos
	board.bind(pos=spotlayout.setter("pos"))
	spotlayout.size = board.size
	board.bind(size=spotlayout.setter("size"))
	board.add_widget(spotlayout)
	arrowlayout.pos = board.pos
	board.bind(pos=arrowlayout.setter("pos"))
	arrowlayout.size = board.size
	board.bind(size=arrowlayout.setter("size"))
	board.add_widget(arrowlayout)
	board.update()
	boardview = GraphBoardView(board=board)
	Window.add_widget(boardview)

	@idle_until(timeout=1000, message="Never finished placing spots")
	def all_spots_placed():
		for x in range(spots_wide):
			for y in range(spots_tall):
				if (x, y) not in board.spot:
					return False
		return True

	# Don't get too picky about the exact proportions of the grid; just make sure the
	# spots are positioned logically with respect to one another
	for name, spot in board.spot.items():
		x, y = name
		if x > 0:
			assert spot.x > board.spot[x - 1, y].x
		if y > 0:
			assert spot.y > board.spot[x, y - 1].y
		if x < spots_wide - 1:
			assert spot.x < board.spot[x + 1, y].x
		if y < spots_tall - 1:
			assert spot.y < board.spot[x, y + 1].y


@pytest.fixture
def fake_branches_handle(monkeypatch):
	def handle(self, cmd, **kwargs):
		return {"trunk": (None, 0, 0, 0, 0)}

	monkeypatch.setattr(EngineFacade, "handle", handle, raising=False)


@pytest.fixture
def fake_string_store(monkeypatch):
	monkeypatch.setattr(EngineFacade, "string", SignalDict(), raising=False)


@pytest.fixture
def fake_function_stores(monkeypatch):
	for store in ["trigger", "prereq", "action", "function", "method"]:
		fake = FunctionStore(None)
		fake._cache = {}
		monkeypatch.setattr(EngineFacade, store, fake, raising=False)


@pytest.mark.usefixtures(
	"fake_branches_handle", "fake_string_store", "fake_function_stores", "kivy"
)
def test_select_arrow():
	eng = EngineFacade(None)
	char = eng.new_character("physical")
	char.add_place(0, _x=0.1, _y=0.1)
	char.add_place(1, _x=0.2, _y=0.1)
	char.add_portal(0, 1)
	app = ElideApp()
	app.engine = eng
	board = GraphBoard(app=app, character=char)
	boardview = GraphBoardView(board=board)
	if not board.parent:
		boardview.add_widget(board)
	Window.add_widget(boardview)
	idle_until(
		lambda: board.arrow_plane, 100, "GraphBoard never got arrow_plane"
	)
	idle_until(
		lambda: 0 in board.arrow and 1 in board.arrow[0],
		100,
		"GraphBoard never got arrow",
	)
	idle_until(
		lambda: board.arrow_plane.data,
		100,
		"GraphBoard.arrow_plane.data never populated",
	)
	idle_until(
		lambda: board.arrow_plane._bot_left_corner_xs.shape[0] > 0,
		100,
		"GraphBoard.arrow_plane never got bounding boxes",
	)
	ox, oy = board.spot[0].center
	dx, dy = board.spot[1].center
	motion = UnitTestTouch((ox + ((dx - ox) / 2)), dy)
	motion.touch_down()
	motion.touch_up()
	idle_until(
		lambda: app.selection == board.arrow[0][1],
		100,
		"Arrow not selected",
	)


@pytest.mark.usefixtures("kivy")
def test_select_spot():
	char = CharacterFacade()
	char.add_place(0, _x=0.1, _y=0.1)
	app = ElideApp()
	board = GraphBoard(app=app, character=char)
	boardview = GraphBoardView(board=board)
	Window.add_widget(boardview)
	idle_until(lambda: 0 in board.spot)
	x, y = board.spot[0].center
	motion = UnitTestTouch(x, y)
	motion.touch_down()
	motion.touch_up()
	assert app.selection == board.spot[0]


@pytest.mark.usefixtures("kivy")
def test_select_pawn():
	char = CharacterFacade()
	char.add_place(0, _x=0.1, _y=0.1)
	char.add_thing("that", location=0)
	app = ElideApp()
	board = GraphBoard(app=app, character=char)
	boardview = GraphBoardView(board=board)
	Window.add_widget(boardview)
	idle_until(lambda: 0 in board.spot and "that" in board.pawn, 100)
	motion = UnitTestTouch(*board.pawn["that"].center)
	motion.touch_down()
	motion.touch_up()
	assert app.selection == board.pawn["that"]


@pytest.mark.usefixtures("kivy")
def test_pawn_drag():
	char = CharacterFacade()
	char.add_place(0, _x=0.1, _y=0.1)
	char.add_place(1, _x=0.2, _y=0.1)
	char.add_thing("that", location=0)
	app = ElideApp()
	board = GraphBoard(app=app, character=char)
	boardview = GraphBoardView(board=board)
	Window.add_widget(boardview)
	idle_until(
		lambda: 0 in board.spot and 1 in board.spot and "that" in board.pawn
	)
	that = board.pawn["that"]
	one = board.spot[1]
	touch = UnitTestTouch(*that.center)
	touch.touch_down()
	dist_x = one.center_x - that.center_x
	dist_y = one.center_y - that.center_y
	for i in range(1, 11):
		coef = 1 / i
		x = one.center_x - coef * dist_x
		y = one.center_y - coef * dist_y
		touch.touch_move(x, y)
		advance_frames(1)
	touch.touch_move(*one.center)
	advance_frames(1)
	touch.touch_up(*one.center)
	idle_until(lambda: that.pos != one.center, 100)
	idle_until(lambda: that.proxy["location"] == 1, 100)


def test_spot_and_pawn_from_dummy(elide_app):
	@idle_until(timeout=100)
	def charmenu_present():
		return (
			hasattr(elide_app, "mainscreen")
			and hasattr(elide_app.mainscreen, "charmenu")
			and elide_app.mainscreen.charmenu.charmenu
		)

	charmenu = elide_app.mainscreen.charmenu.charmenu

	@idle_until(timeout=100)
	def charmenu_has_parent():
		return charmenu.parent is not None

	@idle_until(timeout=100)
	def charmenu_has_screen():
		return charmenu.screen is not None

	@idle_until(timeout=100)
	def dummy_place_created():
		return "dummyplace" in charmenu.ids

	dummy_place = charmenu.ids.dummyplace
	x0, y0 = charmenu.to_parent(*dummy_place.center)
	touch = UnitTestTouch(x0, y0)
	touch.touch_down()

	@idle_until(timeout=100)
	def dummy_got_touch():
		return getattr(dummy_place, "_touch") is touch

	x1, y1 = elide_app.mainscreen.mainview.center

	xdist = x1 - x0
	ydist = y1 - y0
	dummy_name = dummy_place.name
	for i in range(15, 0, -1):
		touch.touch_move(x0 + (xdist / i), y0 + (ydist / i))
		advance_frames(1)
	advance_frames(3)
	touch.touch_up()

	boardview = elide_app.mainscreen.boardview
	idle_until(
		lambda: dummy_name in elide_app.character.place,
		100,
		"Didn't create first new place from dummy",
	)
	idle_until(
		lambda: dummy_name in boardview.board.spot,
		100,
		"Didn't create first new spot from dummy",
	)
	first_new_spot = boardview.board.spot[dummy_name]
	assert boardview.plane.to_parent(
		*boardview.board.spot[dummy_name].center
	) == (
		x1,
		y1,
	)

	@idle_until(timeout=100)
	def dummy_returned():
		return charmenu.to_parent(*dummy_place.center) == (x0, y0)

	idle_until(
		lambda: dummy_name != dummy_place.name, 100, "Never renamed dummy"
	)
	dummy_name = dummy_place.name
	x2 = x1 - 50
	y2 = y1 - 50
	xdist = x2 - x0
	ydist = y2 - y0
	touch = UnitTestTouch(x0, y0)
	touch.touch_down()

	idle_until(dummy_got_touch, timeout=100)
	for i in range(15, 0, -1):
		touch.touch_move(x0 + xdist / i, y0 + ydist / i)
		advance_frames(1)
	touch.touch_up()

	idle_until(
		lambda: dummy_name in elide_app.character.place,
		100,
		"Didn't create second new place from dummy",
	)
	assert boardview.plane.to_parent(
		*boardview.board.spot[dummy_name].center
	) == (x2, y2)

	dummy_thing = charmenu.ids.dummything
	dummy_name = dummy_thing.name
	(x0, y0) = charmenu.to_parent(*dummy_thing.center)
	xdist = x1 - x0
	ydist = y1 - y0
	touch = UnitTestTouch(x0, y0)
	touch.touch_down()

	@idle_until(timeout=100)
	def dummy_thing_got_touch():
		return getattr(dummy_thing, "_touch") is touch

	for i in range(15, 0, -1):
		touch.touch_move(x0 + xdist / i, y0 + ydist / i)
		advance_frames(1)
	touch.touch_up()

	idle_until(
		lambda: dummy_name in elide_app.character.thing,
		100,
		"Didn't create a new thing from dummy",
	)
	idle_until(
		lambda: dummy_name in boardview.board.pawn,
		100,
		"Didn't create a new pawn from dummy",
	)
	pawn = boardview.board.pawn[dummy_name]
	dx, dy = getattr(pawn, "rel_pos", (0, 0))

	@idle_until(timeout=100)
	def pawn_positioned_correctly():
		return (
			pawn.x == first_new_spot.right + dx
			and pawn.y == first_new_spot.top + dy
		)


@pytest.mark.usefixtures("kivy")
def test_pawn_add_new_place():
	char = CharacterFacade()
	app = ElideApp()
	board = GraphBoard(app=app, character=char)
	board._connect_proxy_objects()
	boardview = GraphBoardView(board=board)
	Window.add_widget(boardview)
	idle_until(lambda: board.stack_plane)
	char.add_place(1, _x=0.2, _y=0.2)
	board.add_spot(1)
	idle_until(lambda: 1 in board.spot, 100, "Didn't make spot")
	char.add_thing("that", location=1)
	idle_until(lambda: "that" in board.pawn, 100, "Didn't make pawn")
	that = board.pawn["that"]
	one = board.spot[1]
	idle_until(
		lambda: pos_near(*getattr(that, "pos", None), one.right, one.top),
		100,
		f"pawn did not locate within 100 ticks. "
		f"Should be at {one.right, one.top}, is at {that.pos}",
	)


@pytest.mark.usefixtures("line_shaped_graphs")
def test_character_switch_graph(elide_app):
	app = elide_app
	idle_until(
		lambda: hasattr(app, "mainscreen")
		and app.mainscreen.mainview
		and app.mainscreen.statpanel
		and hasattr(app.mainscreen, "gridview")
	)
	idle_until(
		lambda: app.mainscreen.boardview in app.mainscreen.mainview.children
	)
	idle_until(lambda: app.mainscreen.boardview.board.children)
	print(
		f"test_character_switch_graph got app {id(app)}, engine proxy {id(app.engine)}"
	)
	assert len(
		set(
			child.x
			for child in app.mainscreen.boardview.board.stack_plane.children
		)
	) == len(app.mainscreen.boardview.board.stack_plane.children)
	app.character_name = "tall"

	def all_x_same():
		if (
			app.mainscreen.boardview.board is None
			or app.mainscreen.boardview.board.stack_plane is None
			or not app.mainscreen.boardview.board.spot
		):
			return False
		first_x = next(iter(app.mainscreen.boardview.board.spot.values())).x
		return all(
			child.x == first_x
			for child in app.mainscreen.boardview.board.spot.values()
		)

	idle_until(all_x_same, 100, "Never got the new board")
	idle_until(
		lambda: len(
			set(
				child.y
				for child in app.mainscreen.boardview.board.stack_plane.children
			)
		)
		== len(app.mainscreen.boardview.board.stack_plane.children),
		100,
		"New board arranged weird",
	)
