from types import SimpleNamespace

from kivy.base import EventLoop
from kivy.config import ConfigParser
from kivy.tests.common import GraphicUnitTest

from LiSE.character import Facade
from ELiDE.app import ELiDEApp
from ELiDE.screen import MainScreen
from ELiDE.spritebuilder import PawnConfigScreen, SpotConfigScreen
from ELiDE.statcfg import StatScreen
from ELiDE.graph.board import GraphBoard
from ELiDE.grid.board import GridBoard
from .util import MockTouch, ListenableDict, MockEngine, idle_until, window_with_widget


class ScreenTest(GraphicUnitTest):
    @staticmethod
    def test_advance_time():
        app = ELiDEApp()
        app.config = ConfigParser(None)
        app.spotcfg = SpotConfigScreen()
        app.pawncfg = PawnConfigScreen()
        app.statcfg = StatScreen()
        char = Facade()
        char.name = 'physical'
        app.character = char
        app.engine = MockEngine()
        char.character = SimpleNamespace(engine=app.engine)
        app.engine.character['physical'] = char
        entity = ListenableDict()
        entity.engine = app.engine
        entity.name = 'name'
        app.selected_proxy = app.proxy = app.statcfg.proxy = entity
        screen = MainScreen(graphboards={'physical': GraphBoard(
            character=char)}, gridboards={
            'physical': GridBoard(character=char)
        })
        win = window_with_widget(screen)
        idle_until(lambda: 'timepanel' in screen.ids)
        timepanel = screen.ids['timepanel']
        idle_until(lambda: timepanel.size != [100, 100])
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
    
    @staticmethod
    def test_play():
        app = ELiDEApp()
        app.config = {'ELiDE': {'boardchar': 'foo'}}
        app.spotcfg = SpotConfigScreen()
        app.pawncfg = PawnConfigScreen()
        app.statcfg = StatScreen()
        char = Facade()
        char.name = 'foo'
        app.character = char
        app.engine = MockEngine()
        char.character = SimpleNamespace(engine=app.engine)
        app.engine.character['foo'] = char
        entity = ListenableDict()
        entity.engine = app.engine
        entity.name = 'name'
        app.selected_proxy = app.proxy = app.statcfg.proxy = entity
        screen = MainScreen(graphboards={'foo': GraphBoard(
            character=char)}, gridboards={
            'foo': GridBoard(character=char)
        }, play_speed=1.0)
        win = window_with_widget(screen)
        idle_until(lambda: 'timepanel' in screen.ids)
        timepanel = screen.ids['timepanel']
        idle_until(lambda: timepanel.size != [100, 100])
        turnfield = timepanel.ids['turnfield']
        turn_before = int(turnfield.hint_text)
        playbut = timepanel.ids['playbut']
        x, y = playbut.center
        sx = x / win.width
        sy = y / win.height
        motion = MockTouch("unittest", 1, {'sx': sx, 'sy': sy})
        EventLoop.post_dispatch_input("begin", motion)
        EventLoop.post_dispatch_input("end", motion)
        idle_until(lambda: int(turnfield.hint_text) == 3, 400, "Time didn't advance fast enough")