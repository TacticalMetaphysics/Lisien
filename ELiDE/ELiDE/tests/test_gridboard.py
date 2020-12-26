from kivy.base import EventLoop
from kivy.tests.common import GraphicUnitTest
from kivy.logger import Logger
import networkx as nx

from LiSE.character import Facade
from ELiDE.app import ELiDEApp
from ELiDE.grid.board import GridBoard, GridBoardView
from ELiDE.tests.util import MockTouch, all_spots_placed, all_pawns_placed, all_arrows_placed, idle_until, window_with_widget


class GridBoardTest(GraphicUnitTest):
    @staticmethod
    def test_layout_grid():
        spots_wide = 3
        spots_tall = 3
        spot_width = 32
        spot_height = 32
        graph = nx.grid_2d_graph(spots_wide, spots_tall)
        char = Facade(graph)
        something = char.place[1, 1].new_thing('something')
        otherthing = char.place[2, 2].new_thing('otherthing')
        assert len(char.thing) == 2
        board = GridBoard(
            character=char
        )
        win = window_with_widget(GridBoardView(board=board))
        while not (all_spots_placed(board, char) and all_pawns_placed(board, char)):
            EventLoop.idle()
        otherthing['location'] = board.pawn['otherthing'].loc_name = (0, 0)
        zero = board.spot[0, 0]
        that = board.pawn['otherthing']
        idle_until(lambda: that in zero.children, 1000, "pawn 'otherthing' did not relocate within 1000 ticks")
        assert that.parent == zero
        for x in range(spots_wide):
            for y in range(spots_tall):
                spot = board.spot[x, y]
                assert spot.x == x * spot_width
                assert spot.y == y * spot_height
        assert board.pawn['something'].parent == board.spot[1, 1]
        assert board.pawn['something'].pos == board.spot[1, 1].pos
        assert board.pawn['otherthing'].parent == board.spot[0, 0]
        assert board.pawn['otherthing'].pos == board.spot[0, 0].pos