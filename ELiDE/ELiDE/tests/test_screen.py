from types import SimpleNamespace

from kivy.base import EventLoop
from kivy.tests.common import GraphicUnitTest

from LiSE.character import Facade
from ELiDE.app import ELiDEApp
from ELiDE.screen import MainScreen
from ELiDE.spritebuilder import PawnConfigScreen, SpotConfigScreen
from ELiDE.statcfg import StatScreen
from ELiDE.graph.board import Board
from .util import MockTouch, ListenableDict, MockEngine


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
        app.engine.character['foo'] = char
        entity = ListenableDict()
        entity.engine = app.engine
        app.selected_proxy = app.proxy = app.statcfg.proxy = entity
        screen = MainScreen(app=app, boards={'foo': Board(
            character=char, app=app)})
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
        motion = MockTouch("unittest", 1, {'sx': sx, 'sy': sy})
        EventLoop.post_dispatch_input("begin", motion)
        EventLoop.post_dispatch_input("end", motion)
        EventLoop.idle()
        assert int(turnfield.hint_text) == turn_before + 1
