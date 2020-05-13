from kivy.base import EventLoop
from kivy.tests.common import GraphicUnitTest

from LiSE.character import Facade
from ELiDE.app import ELiDEApp
from ELiDE.board.board import Board, BoardView
from .util import MockTouch


class BoardTest(GraphicUnitTest):
    @staticmethod
    def test_select_arrow():
        char = Facade()
        char.add_place(0, _x=0.1, _y=0.1)
        char.add_place(1, _x=0.2, _y=0.1)
        char.add_portal(0, 1)
        app = ELiDEApp()
        board = Board(
            app=app,
            character=char
        )
        boardview = BoardView(board=board)
        EventLoop.ensure_window()
        win = EventLoop.window
        win.add_widget(boardview)
        while 0 not in board.arrow \
                or 1 not in board.arrow[0] \
                or board.arrow[0][1] not in board.arrowlayout.children:
            EventLoop.idle()
        ox, oy = board.spot[0].center
        dx, dy = board.spot[1].center
        motion = MockTouch("unittest", 1, {
            'sx': (ox + ((dx - ox) / 2)) / win.width,
            'sy': dy / win.height})
        EventLoop.post_dispatch_input("begin", motion)
        EventLoop.post_dispatch_input("end", motion)
        assert app.selection == board.arrow[0][1]

    @staticmethod
    def test_select_spot():
        char = Facade()
        char.add_place(0, _x=0.1, _y=0.1)
        app = ELiDEApp()
        board = Board(
            app=app,
            character=char
        )
        boardview = BoardView(board=board)
        EventLoop.ensure_window()
        win = EventLoop.window
        win.add_widget(boardview)
        while 0 not in board.spot \
                or board.spot[0] not in board.spotlayout.children:
            EventLoop.idle()
        x, y = board.spot[0].center
        motion = MockTouch("unittest", 1, {
            'sx': x / win.width,
            'sy': y / win.height})
        EventLoop.post_dispatch_input("begin", motion)
        EventLoop.post_dispatch_input("end", motion)
        assert app.selection == board.spot[0]

    @staticmethod
    def test_select_pawn():
        char = Facade()
        char.add_place(0, _x=0.1, _y=0.1)
        char.add_thing('that', location=0)
        app = ELiDEApp()
        board = Board(
            app=app,
            character=char
        )
        boardview = BoardView(board=board)
        EventLoop.ensure_window()
        win = EventLoop.window
        win.add_widget(boardview)
        while 0 not in board.spot \
                or board.spot[0] not in board.spotlayout.children \
                or board.pawn['that'] not in board.spot[0].children:
            EventLoop.idle()
        x, y = board.pawn['that'].center
        motion = MockTouch("unittest", 1, {
            'sx': x / win.width,
            'sy': y / win.height})
        EventLoop.post_dispatch_input("begin", motion)
        EventLoop.post_dispatch_input("end", motion)
        assert app.selection == board.pawn['that']

    @staticmethod
    def test_pawn_relocate():
        char = Facade()
        char.add_place(0, _x=0.1, _y=0.1)
        char.add_place(1, _x=0.2, _y=0.1)
        char.add_thing('that', location=0)
        app = ELiDEApp()
        board = Board(
            app=app,
            character=char
        )
        boardview = BoardView(board=board)
        EventLoop.ensure_window()
        win = EventLoop.window
        win.add_widget(boardview)
        while 0 not in board.spot \
                or board.spot[0] not in board.spotlayout.children \
                or 1 not in board.spot \
                or board.spot[1] not in board.spotlayout.children \
                or 'that' not in board.pawn \
                or board.pawn['that'] not in board.spot[0].children:
            EventLoop.idle()
        that = board.pawn['that']
        one = board.spot[1]
        # In a real ELiDE session, the following would happen as a
        # result of a Board.update() call
        char.thing['that']['location'] = that.loc_name = 1
        for ticked in range(1000):
            EventLoop.idle()
            if that in one.children:
                return
        else:
            assert False, "pawn did not relocate within 1000 ticks"
