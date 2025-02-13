import networkx as nx
from kivy.base import EventLoop
from kivy.tests.common import GraphicUnitTest

from elide.grid.board import GridBoard, GridBoardView
from lisien import Engine
from lisien.facade import CharacterFacade

from .util import ELiDEAppTest, all_pawns_placed, all_spots_placed, idle_until


class GridBoardTest(GraphicUnitTest):
	def test_layout_grid(self):
		spots_wide = 3
		spots_tall = 3
		spot_width = 32
		spot_height = 32
		graph = nx.grid_2d_graph(spots_wide, spots_tall)
		char = CharacterFacade(graph)
		char.place[1, 1].add_thing("something")
		otherthing = char.place[2, 2].new_thing("otherthing")
		assert len(char.thing) == 2
		board = GridBoard(character=char)
		self.Window.add_widget(GridBoardView(board=board))
		while not (
			all_spots_placed(board, char) and all_pawns_placed(board, char)
		):
			EventLoop.idle()
		otherthing["location"] = (0, 0)
		board.spot_plane.data = list(map(board.make_spot, char.place.values()))
		board.spot_plane.redraw()
		board.pawn_plane.data = list(map(board.make_pawn, char.thing.values()))

		def arranged():
			for x in range(spots_wide):
				for y in range(spots_tall):
					spot = board.spot[x, y]
					if spot.x != x * spot_width or spot.y != y * spot_height:
						return False
			return True

		idle_until(arranged, 100)
		idle_until(lambda: board.pawn_plane._stack_index)
		this = board.pawn["something"]
		that = board.pawn["otherthing"]
		print(this.pos, board.spot[1, 1].pos)
		idle_until(lambda: this.pos == board.spot[1, 1].pos)
		idle_until(lambda: that.pos == board.spot[0, 0].pos)
		assert this.x == board.spot[1, 1].x
		assert this.y == board.spot[1, 1].y
		assert that.x == board.spot[0, 0].x
		assert that.y == board.spot[0, 0].y


class SwitchGridTest(ELiDEAppTest):
	def test_character_switch_grid(self):
		with Engine(self.prefix) as eng:
			eng.add_character("physical", nx.grid_2d_graph(10, 1))
			eng.add_character("tall", nx.grid_2d_graph(1, 10))
		app = self.app
		self.Window.add_widget(app.build())
		idle_until(
			lambda: hasattr(app, "mainscreen")
			and app.mainscreen.mainview
			and app.mainscreen.statpanel
			and hasattr(app.mainscreen, "gridview")
		)
		app.mainscreen.charmenu.toggle_gridview()
		idle_until(
			lambda: app.mainscreen.gridview in app.mainscreen.mainview.children
		)
		idle_until(lambda: app.mainscreen.gridview.board.children)
		assert len(app.mainscreen.gridview.board.spot) == 10
		assert all(
			spot.y == 0 for spot in app.mainscreen.gridview.board.spot.values()
		)
		idle_until(
			lambda: not all(
				spot.x == 0
				for spot in app.mainscreen.gridview.board.spot.values()
			),
			100,
		)
		app.character_name = "tall"
		idle_until(
			lambda: all(
				spot.x == 0
				for spot in app.mainscreen.gridview.board.spot.values()
			),
			1000,
			"Never got the new board",
		)
		idle_until(
			lambda: not all(
				spot.y == 0
				for spot in app.mainscreen.gridview.board.spot.values()
			),
			1000,
			"New board arranged weird",
		)
