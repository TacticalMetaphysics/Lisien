from types import SimpleNamespace

from blinker import Signal
from kivy.base import EventLoop
from kivy.tests.common import GraphicUnitTest

from LiSE.character import Facade
from ELiDE.app import ELiDEApp
from ELiDE.screen import MainScreen
from ELiDE.spritebuilder import PawnConfigScreen, SpotConfigScreen
from ELiDE.statcfg import StatScreen
from ELiDE.board.board import Board
from .util import TestTouch


class ListenableDict(dict, Signal):
    def __init__(self):
        Signal.__init__(self)


class MockEngine(Signal):
    universal = ListenableDict()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turn = 0
        self._ready = True

    def __setattr__(self, key, value):
        if not hasattr(self, '_ready'):
            super().__setattr__(key, value)
            return
        self.send(self, key=key, value=value)
        super().__setattr__(key, value)

    def next_turn(self, *args, **kwargs):
        self.turn += 1
        kwargs['cb']('next_turn', 'master', self.turn, 0, ([], {}))


class ScreenTest(GraphicUnitTest):
    @staticmethod
    def test_advance_time():
        app = ELiDEApp()
        app.config = {'ELiDE': {'boardchar': 'foo'}}
        app.spotcfg = SpotConfigScreen()
        app.pawncfg = PawnConfigScreen()
        app.statcfg = StatScreen()
        char = Facade()
        char.name = 'foo'
        app.character = char
        app.engine = app.statcfg.engine = MockEngine()
        char.character = SimpleNamespace(engine=app.engine)
        app.selected_proxy = app.proxy = app.statcfg.proxy = ListenableDict()
        screen = MainScreen(app=app, boards={'foo': Board(character=char, app=app)})
        EventLoop.ensure_window()
        win = EventLoop.window
        win.add_widget(screen)
        while 'timepanel' not in screen.ids:
            EventLoop.idle()
        timepanel = screen.ids['timepanel']
        while timepanel.size == [100, 100]:
            EventLoop.idle()
        turnfield = timepanel.ids['turnfield']
        turn_before = int(turnfield.hint_text)
        stepbut = timepanel.ids['stepbut']
        x, y = stepbut.center
        sx = x / win.width
        sy = y / win.height
        motion = TestTouch("unittest", 1, {'sx': sx, 'sy': sy})
        EventLoop.post_dispatch_input("begin", motion)
        EventLoop.post_dispatch_input("end", motion)
        EventLoop.idle()
        assert int(turnfield.hint_text) == turn_before + 1
