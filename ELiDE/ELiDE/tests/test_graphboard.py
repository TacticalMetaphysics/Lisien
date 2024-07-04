from kivy.tests.common import GraphicUnitTest, UnitTestTouch
import networkx as nx

from LiSE import Engine
from LiSE.character import Facade
from ELiDE.app import ELiDEApp
from ELiDE.graph.board import GraphBoard, GraphBoardView
from ELiDE.pawnspot import TextureStackPlane
from ELiDE.graph.arrow import ArrowPlane
from .util import idle_until, window_with_widget, ELiDEAppTest
from ..dummy import Dummy


class FakeEngineProxy:
	def handle(self, *args, **kwargs):
		pass


class GraphBoardTest(GraphicUnitTest):
	@staticmethod
	def test_layout_grid():
		spots_wide = 3
		spots_tall = 3
		graph = nx.grid_2d_graph(spots_wide, spots_tall)
		char = Facade(graph)
		app = ELiDEApp()
		spotlayout = TextureStackPlane()
		arrowlayout = ArrowPlane()
		board = GraphBoard(
			app=app,
			character=char,
			stack_plane=spotlayout,
			arrow_plane=arrowlayout,
		)
		board.engine = FakeEngineProxy()
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
		win = window_with_widget(boardview)

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

	@staticmethod
	def test_select_arrow():
		char = Facade()
		char.add_place(0, _x=0.1, _y=0.1)
		char.add_place(1, _x=0.2, _y=0.1)
		char.add_portal(0, 1)
		app = ELiDEApp()
		board = GraphBoard(app=app, character=char)
		boardview = GraphBoardView(board=board)
		win = window_with_widget(boardview)
		idle_until(lambda: 0 in board.arrow and 1 in board.arrow[0])
		ox, oy = board.spot[0].center
		dx, dy = board.spot[1].center
		motion = UnitTestTouch((ox + ((dx - ox) / 2)), dy)
		motion.touch_down()
		motion.touch_up()
		assert app.selection == board.arrow[0][1]

	@staticmethod
	def test_select_spot():
		char = Facade()
		char.add_place(0, _x=0.1, _y=0.1)
		app = ELiDEApp()
		board = GraphBoard(app=app, character=char)
		boardview = GraphBoardView(board=board)
		win = window_with_widget(boardview)
		idle_until(lambda: 0 in board.spot)
		x, y = board.spot[0].center
		motion = UnitTestTouch(x, y)
		motion.touch_down()
		motion.touch_up()
		assert app.selection == board.spot[0]

	@staticmethod
	def test_select_pawn():
		char = Facade()
		char.add_place(0, _x=0.1, _y=0.1)
		char.add_thing("that", location=0)
		app = ELiDEApp()
		board = GraphBoard(app=app, character=char)
		boardview = GraphBoardView(board=board)
		win = window_with_widget(boardview)
		idle_until(lambda: 0 in board.spot and "that" in board.pawn, 100)
		motion = UnitTestTouch(*board.pawn["that"].center)
		motion.touch_down()
		motion.touch_up()
		assert app.selection == board.pawn["that"]

	def test_pawn_drag(self):
		char = Facade()
		char.add_place(0, _x=0.1, _y=0.1)
		char.add_place(1, _x=0.2, _y=0.1)
		char.add_thing("that", location=0)
		app = ELiDEApp()
		board = GraphBoard(app=app, character=char)
		boardview = GraphBoardView(board=board)
		win = window_with_widget(boardview)
		idle_until(
			lambda: 0 in board.spot
			and 1 in board.spot
			and "that" in board.pawn
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
			self.advance_frames(1)
		touch.touch_move(*one.center)
		self.advance_frames(1)
		touch.touch_up(*one.center)
		idle_until(lambda: that.pos != one.center, 100)
		idle_until(lambda: that.proxy["location"] == 1, 100)

	@staticmethod
	def test_spot_and_pawn_from_dummy():
		char = Facade()
		app = ELiDEApp()
		board = GraphBoard(app=app, character=char)
		board.connect_proxy_objects()
		view = GraphBoardView(board=board)
		idle_until(
			lambda: view.plane is not None, 100, "Never made BoardScatterPlane"
		)
		idle_until(
			lambda: board.stack_plane is not None, 100, "Never made StackPlane"
		)
		win = window_with_widget(view)
		dummy = Dummy(
			name="hello",
			paths=["atlas://rltiles/base/unseen"],
			size=(32, 32),
			pos=(0, 0),
		)
		board.add_widget(dummy)
		idle_until(
			lambda: dummy in board.children, 100, "Dummy didn't get to board"
		)
		dummy_name = dummy.name
		view.spot_from_dummy(dummy)
		idle_until(
			lambda: dummy_name in char.node, 100, "Dummy didn't add place"
		)
		dummy2 = Dummy(
			name="goodbye",
			paths=["atlas://rltiles/base/unseen"],
			pos=dummy.pos,
			size=(32, 32),
		)
		dummy2_name = dummy2.name = "dummy2"
		board.add_widget(dummy2)
		idle_until(
			lambda: dummy2 in board.children,
			100,
			"Dummy 2 didn't get to board",
		)
		idle_until(
			lambda: board.stack_plane.data,
			100,
			"Dummy 2 didn't get into the board's stack_plane",
		)
		view.pawn_from_dummy(dummy2)
		idle_until(
			lambda: dummy2_name in char.thing, 100, "Dummy 2 didn't add thing"
		)
		idle_until(
			lambda: dummy2_name in board.pawn,
			100,
			"Board didn't add pawn for dummy 2",
		)
		spot = board.spot[dummy_name]
		idle_until(
			lambda: board.pawn[dummy2_name].pos == (spot.right, spot.top),
			100,
			"Dummy 2 didn't get to dummy 1",
		)

	@staticmethod
	def test_pawn_add_new_place():
		char = Facade()
		app = ELiDEApp()
		board = GraphBoard(app=app, character=char)
		board.connect_proxy_objects()
		boardview = GraphBoardView(board=board)
		win = window_with_widget(boardview)
		idle_until(lambda: board.stack_plane)
		char.add_place(1, _x=0.2, _y=0.2)
		board.add_spot(1)
		idle_until(lambda: 1 in board.spot, 100, "Didn't make spot")
		char.add_thing("that", location=1)
		idle_until(lambda: "that" in board.pawn, 100, "Didn't make pawn")
		that = board.pawn["that"]
		one = board.spot[1]
		idle_until(
			lambda: getattr(that, "pos", None) == (one.right, one.top),
			100,
			f"pawn did not locate within 100 ticks. "
			f"Should be at {one.right, one.top}, is at {that.pos}",
		)


class SwitchGraphTest(ELiDEAppTest):
	def test_character_switch_graph(self):
		with Engine(self.prefix) as eng:
			eng.add_character("physical", nx.grid_2d_graph(10, 1))
			eng.add_character("tall", nx.grid_2d_graph(1, 10))
		app = self.app
		window_with_widget(app.build())
		idle_until(
			lambda: hasattr(app, "mainscreen")
			and app.mainscreen.mainview
			and app.mainscreen.statpanel
			and hasattr(app.mainscreen, "gridview")
		)
		idle_until(
			lambda: app.mainscreen.boardview
			in app.mainscreen.mainview.children
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
			first_x = next(
				iter(app.mainscreen.boardview.board.spot.values())
			).x
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
